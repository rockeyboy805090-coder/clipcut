"""Phase 9: face-TRACKING crop — the 9:16 window follows the speaker.

Free stack only: OpenCV Haar cascade (already in the venv) + FFmpeg.
No paid API, no network, no extra model download.

What it IS:
  track_face() samples the clip once per second, Haar-detects faces
  (largest face wins, centre below y=0.72 ignored — same desk-mug
  false-positive guard as editor.detect_face_centre, Phase 5), and
  returns one centre per second. Gaps (a second with no detection)
  HOLD the last known centre; centres are median-smoothed over 3
  samples so the crop does not jitter frame to frame.
  face_track_crop_expr() turns the track into an FFmpeg crop x
  expression (gte(t,a)*lt(t,b)*X summed over segments), so the crop
  window genuinely MOVES with the face inside one render — it is not
  a single static centre chosen once per clip.

What it is NOT (honest limits, recorded in the render sidecar):
  - Not a per-frame tracker: sampling is 1 Hz, so a face that moves
    fast inside one second is followed with up to ~1s lag, and the
    window moves in 1s steps, not smoothly.
  - Not identity tracking: if two faces are in frame, the LARGEST
    face wins each second — the window can jump if a second person
    becomes larger. No face re-identification is done.
  - Haar only detects roughly frontal faces, ~40px and up, in decent
    light. Profile/back-of-head/sunglasses faces are missed, and a
    missed second just holds the last centre.
  - If NO face is detected in the whole clip, track_face returns
    segments=[] and the caller must fall back to a centre crop and
    SAY SO — tracking is never claimed without a detection.
"""
from __future__ import annotations

from pathlib import Path

import cv2

# Same false-positive guard as editor.detect_face_centre: a seated
# speaker's face centre sits in the upper 2/3 of the frame; detections
# below y=0.72 are usually objects (Phase 5: desk mug at (0.11, 0.84)).
_MAX_CENTRE_Y = 0.72
# Slightly more sensitive than editor.detect_face_centre (minNeighbors
# 5, minSize 60): a 150px reference photo's face is ~60px and only
# fires at minNeighbors 4 — verified live 17 Nov 2026. minSize 40
# still rejects tiny noise blobs.
_MIN_SIZE = (40, 40)
_MIN_NEIGHBORS = 4


def _detect_one(cascade, frame):
    """Largest plausible face in one frame, or None.

    Returns (centre_x, centre_y, rel_width) in 0..1 frame coords.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                     minNeighbors=_MIN_NEIGHBORS,
                                     minSize=_MIN_SIZE)
    if len(faces) == 0:
        return None
    fh, fw = frame.shape[:2]
    cands = []
    for (x, y, w, h) in faces:
        cx, cy = (x + w / 2) / fw, (y + h / 2) / fh
        if cy <= _MAX_CENTRE_Y:
            cands.append((x, y, w, h, cx, cy))
    if not cands:
        return None
    x, y, w, h, cx, cy = max(cands, key=lambda r: r[2] * r[3])
    return (round(float(cx), 3), round(float(cy), 3), round(float(w / fw), 3))


def _median3(values: list[float]) -> list[float]:
    """Median-of-3 smoothing (edges keep their own value)."""
    if len(values) < 3:
        return list(values)
    out = [values[0]]
    for i in range(1, len(values) - 1):
        out.append(sorted(values[i - 1:i + 2])[1])
    out.append(values[-1])
    return out


def track_face(video_path, start: float, end: float,
               step: float = 1.0) -> dict:
    """Sample faces once per `step` seconds over [start, end].

    Times are CLIP-RELATIVE (0 = clip start) — the crop expression
    runs after trim+setpts, so t=0 is the first rendered frame.
    Returns {"segments": [{t_start, t_end, centre_x, centre_y,
    face_rel_width, detected}], "frames_checked", "frames_with_face",
    "face_detected", "centre_x", "centre_y", "note"}.
    Never raises — returns face_detected=False on any failure.
    """
    out = {"segments": [], "frames_checked": 0, "frames_with_face": 0,
           "face_detected": False, "centre_x": 0.5, "centre_y": 0.5,
           "face_rel_width": 0.0, "note": ""}
    try:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.is_file():
            out["note"] = "Haar cascade file missing — face tracking off"
            return out
        cascade = cv2.CascadeClassifier(str(cascade_path))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            out["note"] = "OpenCV could not open video"
            return out
        duration = max(0.0, float(end) - float(start))
        if duration <= 0:
            cap.release()
            out["note"] = "empty clip range"
            return out
        # Sample at clip-relative t = 0, step, 2*step, ... (+ the last
        # instant so the final segment has a value).
        sample_times = []
        t = 0.0
        while t < duration:
            sample_times.append(round(t, 2))
            t += step
        if not sample_times or sample_times[-1] < duration - 0.01:
            sample_times.append(round(duration, 2))
        raw: list[tuple[float, tuple | None]] = []
        for rt in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, (float(start) + rt) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                raw.append((rt, None))
                continue
            out["frames_checked"] += 1
            det = _detect_one(cascade, frame)
            if det is not None:
                out["frames_with_face"] += 1
            raw.append((rt, det))
        cap.release()
        if out["frames_with_face"] == 0:
            out["note"] = ("face_track requested but no face detected in "
                           f"{out['frames_checked']} sampled frames — "
                           "caller must fall back to a centre crop")
            return out
        # Hold last known centre over gaps, then median-smooth x.
        centres_x, centres_y, widths = [], [], []
        last = None
        for _rt, det in raw:
            if det is not None:
                last = det
            centres_x.append(last[0] if last else 0.5)
            centres_y.append(last[1] if last else 0.5)
            widths.append(last[2] if last else 0.0)
        centres_x = _median3(centres_x)
        # First samples before any detection hold 0.5 (centre) — say so.
        first_det = next(i for i, (_rt, d) in enumerate(raw) if d is not None)
        segments = []
        for i, rt in enumerate(sample_times):
            seg_end = (sample_times[i + 1] if i + 1 < len(sample_times)
                       else round(duration, 2))
            if seg_end <= rt:
                continue
            segments.append({
                "t_start": rt,
                "t_end": round(seg_end, 2),
                "centre_x": round(float(centres_x[i]), 3),
                "centre_y": round(float(centres_y[i]), 3),
                "face_rel_width": round(float(widths[i]), 3),
                "detected": raw[i][1] is not None,
            })
        out["segments"] = segments
        out["face_detected"] = True
        mid = len(segments) // 2
        out["centre_x"] = segments[mid]["centre_x"]
        out["centre_y"] = segments[mid]["centre_y"]
        out["face_rel_width"] = segments[mid]["face_rel_width"]
        out["note"] = (
            f"face tracked in {out['frames_with_face']}/{out['frames_checked']} "
            f"sampled frames (1 Hz, hold-last on gaps, median-3 smoothed); "
            f"{first_det} leading sample(s) before the first detection hold centre")
        return out
    except Exception as exc:  # tracking must never break a render
        out["note"] = f"face tracking failed (ignored): {exc}"
        return out


def face_track_crop_expr(track: dict, src_w: int, src_h: int) -> str | None:
    """FFmpeg crop-x expression for a track, or None (no track).

    crop width is the 9:16 width for the source height; each segment
    contributes gte(t,a)*lt(t,b)*X so exactly one term is live at any
    t (boundaries are measure-zero). X is clamped so the crop window
    never leaves the frame. The expression is returned WITHOUT
    surrounding quotes — the caller wraps it in '...' inside the
    filter graph (commas inside must be quoted for the filter parser).
    """
    segments = (track or {}).get("segments") or []
    if not segments:
        return None
    crop_w = int(src_h * 9 / 16)
    crop_w = (crop_w // 2) * 2  # even, for x264
    if crop_w >= src_w:
        return None  # source already narrower than 9:16 — no crop needed
    terms = []
    for seg in segments:
        cx = float(seg["centre_x"])
        x = int(max(0, min(src_w - crop_w, cx * src_w - crop_w / 2)))
        x = (x // 2) * 2
        terms.append(
            f"gte(t,{seg['t_start']})*lt(t,{seg['t_end']})*{x}")
    if not terms:
        return None
    return "+".join(terms)
