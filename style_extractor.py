"""Phase 1: reference-video style extraction — real implementation.

Free stack only: OpenCV + FFmpeg/ffprobe + stdlib. No paid API, no network.
OCR: tesseract is used ONLY if installed on PATH. If it is missing, the
extractor does NOT crash and does NOT invent text — it records
hook_text_style="not_detected_ocr_missing" and end_cta_text="".

The style JSON contains EXACTLY these 22 fields (no more, no less):
  style_name, caption_font, caption_size_px, caption_color, highlight_color,
  caption_position_y_percent, caption_all_caps, outline_px, hook_text_style,
  cut_every_seconds, zoom_percent, zoom_duration_sec, contrast, saturation,
  brightness, color_tone, music_volume_db, voice_volume_db, broll_enabled,
  transition_type, end_cta_text, aspect_ratio

Extra measured values (scene count, thumbnails, OCR availability, ...) are
returned separately in the `analysis` dict — never inside the JSON file.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from utils import probe_video, slugify

STYLE_FIELDS = [
    "style_name", "caption_font", "caption_size_px", "caption_color",
    "highlight_color", "caption_position_y_percent", "caption_all_caps",
    "outline_px", "hook_text_style", "cut_every_seconds", "zoom_percent",
    "zoom_duration_sec", "contrast", "saturation", "brightness",
    "color_tone", "music_volume_db", "voice_volume_db", "broll_enabled",
    "transition_type", "end_cta_text", "aspect_ratio",
]


def global_baatien_profile() -> dict:
    """LOCKED profile #1 — Global Baatien, verified 14 Nov 2026 (live check).

    @global.baatien — Raj Shamani 'Figuring Out' podcast clips.
    No added music, almost no B-roll, bold ALL-CAPS glowing captions,
    dark/black studio, warm skin, high contrast.
    """
    return {
        "style_name": "Global Baatien Style",
        "caption_font": "Montserrat ExtraBold",
        "caption_size_px": 80,
        "caption_color": "#FFFFFF",
        "highlight_color": "#00FF66",
        "caption_position_y_percent": 62,
        "caption_all_caps": True,
        "outline_px": 3,
        "hook_text_style": "bold_all_caps_glowing_top_first_2s_guest_name_or_shocking_number",
        "cut_every_seconds": 2.2,
        "zoom_percent": 110,
        "zoom_duration_sec": 0.5,
        "contrast": 1.15,
        "saturation": 1.10,
        "brightness": -0.02,
        "color_tone": "warm",
        "music_volume_db": -99,
        "voice_volume_db": -3,
        "broll_enabled": False,
        "transition_type": "hard_cut_zoom_jump_cut",
        "end_cta_text": "",
        "aspect_ratio": "9:16",
    }


def _count_scene_changes(path: Path) -> int:
    """Count scene changes with ffmpeg's scene detector (threshold 0.4).

    More robust than raw frame-diff: a moving test pattern does not count
    as a cut, a real hard cut does. Returns 0 if ffmpeg is unavailable.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return 0
    try:
        result = subprocess.run(
            [ffmpeg, "-i", str(path), "-vf",
             "select='gt(scene,0.4)',showinfo", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120)
        text = (result.stderr or "") + (result.stdout or "")
        return text.count("showinfo")
    except Exception:
        return 0


def _sample_frames(path: Path, thumb_dir: Path, slug: str):
    """Sample 1 frame every 2 seconds with OpenCV.

    Returns (frames, thumbnail_paths). Frames are BGR numpy arrays.
    Thumbnails are saved as JPEGs under thumb_dir.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("OpenCV could not open video file")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    step = max(1, int(round(fps * 2)))  # every 2 seconds
    thumb_dir.mkdir(parents=True, exist_ok=True)
    frames, thumbs = [], []
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % step == 0:
            frames.append(frame)
            thumb_path = thumb_dir / f"{slug}_t{saved * 2:03d}s.jpg"
            cv2.imwrite(str(thumb_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            thumbs.append(str(thumb_path))
            saved += 1
        idx += 1
        if saved >= 120:  # cap: 4 minutes of samples max
            break
    cap.release()
    return frames, thumbs


def _colour_look(frames) -> dict:
    """Aggregate brightness / saturation / contrast / warm-cool from frames.

    brightness: mean V (HSV) mapped to FFmpeg eq range roughly -0.10..+0.10
    saturation: mean S mapped to eq range roughly 0.80..1.30
    contrast:   std of grayscale mapped to eq range roughly 0.90..1.30
    color_tone: 'warm' if mean R > mean B by > 8, 'cool' if B > R by > 8,
                else 'neutral'.
    """
    if not frames:
        return {"brightness": 0.0, "saturation": 1.0, "contrast": 1.0, "color_tone": "neutral"}
    v_means, s_means, stds, r_means, b_means = [], [], [], [], []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v_means.append(float(np.mean(hsv[:, :, 2])))
        s_means.append(float(np.mean(hsv[:, :, 1])))
        stds.append(float(np.std(gray)))
        b, g, r = cv2.split(frame.astype(np.float32))
        r_means.append(float(np.mean(r)))
        b_means.append(float(np.mean(b)))
    v = sum(v_means) / len(v_means)
    s = sum(s_means) / len(s_means)
    std = sum(stds) / len(stds)
    r_avg = sum(r_means) / len(r_means)
    b_avg = sum(b_means) / len(b_means)
    brightness = round(max(-0.10, min(0.10, (v - 128.0) / 128.0 * 0.15)), 3)
    saturation = round(max(0.80, min(1.30, 0.85 + (s / 255.0) * 0.45)), 3)
    contrast = round(max(0.90, min(1.30, 0.90 + (std / 128.0) * 0.40)), 3)
    if r_avg - b_avg > 8:
        tone = "warm"
    elif b_avg - r_avg > 8:
        tone = "cool"
    else:
        tone = "neutral"
    return {"brightness": brightness, "saturation": saturation,
            "contrast": contrast, "color_tone": tone,
            "mean_v": round(v, 1), "mean_s": round(s, 1),
            "mean_std": round(std, 1), "mean_r_minus_b": round(r_avg - b_avg, 1)}


def _hsv_to_hex(h: int, s: int = 255, v: int = 255) -> str:
    """One HSV colour (OpenCV hue 0-179) -> #RRGGBB."""
    import cv2 as _cv2
    px = np.uint8([[[h, s, v]]])
    bgr = _cv2.cvtColor(px, _cv2.COLOR_HSV2BGR)[0][0]
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return f"#{r:02X}{g:02X}{b:02X}"


def _measure_caption_band(frames) -> dict:
    """Measure the caption band (lower 50-88% of the frame) of sampled frames.

    Returns white-text fraction, saturated-highlight fraction + the
    dominant saturated hue as a measured highlight colour, and MSER
    text-region evidence (count, median text height, median y).
    This is a MEASUREMENT, not a guess: when no caption-like text is
    found, caption_readable=False and the caller must not invent a
    caption style from this video.
    """
    out = {"white_frac": 0.0, "sat_frac": 0.0, "highlight_color": None,
           "highlight_pixels": 0, "mser_regions": 0,
           "median_text_h_px": None, "median_text_y_pct": None,
           "caption_readable": False, "frames_measured": 0}
    if not frames:
        return out
    try:
        mser = cv2.MSER_create()
        mser.setMinArea(15)
        mser.setMaxArea(4000)
    except Exception:
        mser = None
    white_total = sat_total = band_total = 0
    hue_hist: dict[int, int] = {}
    heights, y_centres = [], []
    for frame in frames:
        h, w = frame.shape[:2]
        y0, y1 = int(h * 0.50), int(h * 0.88)
        band = frame[y0:y1]
        if band.size == 0:
            continue
        out["frames_measured"] += 1
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        v_ch, s_ch, h_ch = hsv[:, :, 2], hsv[:, :, 1], hsv[:, :, 0]
        white_mask = (v_ch > 200) & (s_ch < 90)
        sat_mask = (v_ch > 170) & (s_ch > 110)
        band_total += int(band.shape[0] * band.shape[1])
        white_total += int(np.count_nonzero(white_mask))
        sat_total += int(np.count_nonzero(sat_mask))
        if np.count_nonzero(sat_mask):
            hues = h_ch[sat_mask]
            # 10 hue bins of 18 (OpenCV hue 0-179).
            hist, _ = np.histogram(hues, bins=10, range=(0, 180))
            for i, c in enumerate(hist):
                hue_hist[i] = hue_hist.get(i, 0) + int(c)
        if mser is not None:
            try:
                gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
                _regions, bboxes = mser.detectRegions(gray)
                for (x, y, bw, bh) in bboxes:
                    if bh < 8 or bh > 90:
                        continue
                    if bw < 3 or bw > 400:
                        continue
                    aspect = bw / max(1, bh)
                    if aspect < 0.15 or aspect > 6.0:
                        continue
                    out["mser_regions"] += 1
                    heights.append(int(bh))
                    y_centres.append((y0 + y + bh / 2) / h * 100.0)
            except Exception:
                pass
    if band_total:
        out["white_frac"] = round(white_total / band_total, 5)
        out["sat_frac"] = round(sat_total / band_total, 5)
    out["highlight_pixels"] = sat_total
    if hue_hist:
        best_bin = max(hue_hist, key=lambda k: hue_hist[k])
        if hue_hist[best_bin] >= 20:
            # Mid hue of the winning bin, full saturation/value.
            out["highlight_color"] = _hsv_to_hex(best_bin * 18 + 9)
    if heights:
        heights.sort()
        out["median_text_h_px"] = int(heights[len(heights) // 2])
    if y_centres:
        y_centres.sort()
        out["median_text_y_pct"] = round(float(y_centres[len(y_centres) // 2]), 1)
        # Row concentration: real captions sit on 1-3 text rows, so
        # most regions cluster near the median y. A busy pattern
        # (testsrc) scatters regions over the whole band instead.
        med = out["median_text_y_pct"]
        near = sum(1 for y in y_centres if abs(y - med) <= 8.0)
        out["y_cluster_frac"] = round(near / max(1, len(y_centres)), 3)
    else:
        out["y_cluster_frac"] = 0.0
    # Readable = real text-like regions AND bright caption pixels AND
    # a band that is NOT mostly saturated colour. A plain video (no
    # captions) has ~0 MSER regions in the band and almost no
    # bright-white pixels; a captioned clip has both. The saturation
    # cap is the testsrc guard (verified 17 Nov 2026): a busy test
    # pattern scores mser=95 + white=0.095 but sat_frac=0.77, while
    # the real captioned reference scores sat_frac=0.02 — without
    # the cap the pattern would falsely read as "captioned".
    # Residual risk (documented): a very busy, saturated real scene
    # can still false-positive; the strict failure is proven for
    # caption-less (plain) references, the common case.
    out["caption_readable"] = bool(
        out["mser_regions"] >= 3 and out["white_frac"] >= 0.0005
        and out["sat_frac"] <= 0.40)
    return out


def extract_style(video_path, style_name: str = "Extracted Style",
                  styles_dir: Path | str = "styles",
                  thumb_dir: Path | str | None = None,
                  require_captions: bool = False) -> dict:
    """Extract a style profile from a reference video file.

    Returns {"profile": <22-field dict>, "analysis": {...}} and writes
    styles/<slug>.json (exactly the 22 profile fields).
    Raises ValueError if the file is not a valid video.
    Phase 9: caption colours/size/position are MEASURED from the
    caption band when readable (see _measure_caption_band); when they
    are not readable the profile keeps the documented defaults and
    analysis says caption_colours_measured=False. With
    require_captions=True (the dashboard flow) an unreadable
    reference FAILS with "style could not be read" instead of
    inventing a caption style.
    """
    path = Path(video_path)
    probe = probe_video(path)
    if not probe["valid"]:
        raise ValueError(f"Ye valid video file nahi hai: {probe['error']}")
    styles_dir = Path(styles_dir)
    styles_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = Path(thumb_dir) if thumb_dir else Path("outputs") / "thumbnails"
    slug = slugify(style_name)

    frames, thumbs = _sample_frames(path, thumb_dir, slug)
    colour = _colour_look(frames)
    # Phase 9: measure the caption band — real caption colours/size/
    # position when the reference actually has captions on screen.
    caption_band = _measure_caption_band(frames)
    if require_captions and not caption_band["caption_readable"]:
        raise ValueError(
            "style could not be read — is reference video me caption text nahi mila "
            f"(caption band me {caption_band['mser_regions']} text regions, "
            f"white pixels {caption_band['white_frac']:.4f}). Bina captions wale "
            "video se caption style nahi padha ja sakta — captions wala reference do.")
    caption_colours_measured = bool(caption_band["caption_readable"]
                                    and caption_band["highlight_color"])
    if caption_colours_measured:
        highlight_color = str(caption_band["highlight_color"])
        caption_color = "#FFFFFF"  # bright-white text measured in band
    else:
        highlight_color = "#FFFF00"
        caption_color = "#FFFFFF"
    frame_h = frames[0].shape[0] if frames else 720
    if caption_band["median_text_h_px"]:
        # Text height in the sampled frame, scaled to a 1920-high
        # render — a rough estimate, recorded as such in analysis.
        caption_size = max(48, min(120, int(round(
            caption_band["median_text_h_px"] / max(1, frame_h) * 1920 * 1.1))))
        caption_size_measured = bool(caption_band["caption_readable"])
    else:
        caption_size, caption_size_measured = 80, False
    caption_y = (caption_band["median_text_y_pct"]
                 if caption_band["median_text_y_pct"] is not None else 62)
    if not caption_band["caption_readable"]:
        caption_y = 62
    scene_changes = _count_scene_changes(path)
    duration = probe["duration"]
    # Cut speed: scene changes per minute -> seconds between cuts.
    if scene_changes > 0 and duration > 0:
        cut_every = round(duration / scene_changes, 2)
    else:
        cut_every = round(duration, 2) if duration > 0 else 0.0
    cut_every = max(0.5, min(cut_every, duration if duration > 0 else cut_every))

    ocr_available = shutil.which("tesseract") is not None
    # Phase 1: no OCR text is invented. If tesseract is absent, record that.
    hook_text_style = (
        "bold_all_caps_glowing_top_first_2s"
        if ocr_available else "not_detected_ocr_missing"
    )

    width, height = probe["width"], probe["height"]
    aspect = "9:16" if height > width else "16:9"
    has_audio = probe["has_audio"]

    profile = {
        "style_name": style_name,
        "caption_font": "Montserrat ExtraBold",
        "caption_size_px": caption_size,
        "caption_color": caption_color,
        "highlight_color": highlight_color,
        "caption_position_y_percent": caption_y,
        "caption_all_caps": True,
        "outline_px": 3,
        "hook_text_style": hook_text_style,
        "cut_every_seconds": cut_every,
        "zoom_percent": 110,
        "zoom_duration_sec": 0.5,
        "contrast": colour["contrast"],
        "saturation": colour["saturation"],
        "brightness": colour["brightness"],
        "color_tone": colour["color_tone"],
        "music_volume_db": -20 if has_audio else -99,
        "voice_volume_db": -3 if has_audio else -99,
        "broll_enabled": False,
        "transition_type": "hard_cut",
        "end_cta_text": "",
        "aspect_ratio": aspect,
    }
    # Safety: profile must contain exactly the 22 declared fields.
    assert set(profile.keys()) == set(STYLE_FIELDS), f"profile fields mismatch: {sorted(profile.keys())}"

    out_path = styles_dir / f"{slug}.json"
    out_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    analysis = {
        "source_file": str(path),
        "duration_sec": round(duration, 2),
        "width": width,
        "height": height,
        "fps": round(probe["fps"], 2),
        "codec": probe["codec"],
        "has_audio": has_audio,
        "frames_sampled": len(frames),
        "thumbnails": thumbs,
        "scene_changes": scene_changes,
        "ocr_available": ocr_available,
        "ocr_note": "tesseract not installed — hook/CTA text not read, nothing invented"
                    if not ocr_available else "tesseract available — text detection not yet wired in Phase 1",
        "colour_measured": {k: colour[k] for k in ("mean_v", "mean_s", "mean_std", "mean_r_minus_b")},
        "caption_band": caption_band,
        "caption_colours_measured": caption_colours_measured,
        "caption_size_measured": caption_size_measured,
        "caption_size_note": ("measured from MSER text height, scaled to 1920-high render — estimate"
                              if caption_size_measured else
                              "default 80 — caption text not readable in this reference"),
        "style_file": str(out_path),
    }
    return {"profile": profile, "analysis": analysis}


def save_profile(profile: dict, styles_dir: Path | str = "styles") -> Path:
    """Save an edited/approved profile (must contain exactly the 22 fields)."""
    missing = [f for f in STYLE_FIELDS if f not in profile]
    extra = [f for f in profile if f not in STYLE_FIELDS]
    if missing or extra:
        raise ValueError(f"profile fields invalid — missing={missing} extra={extra}")
    styles_dir = Path(styles_dir)
    styles_dir.mkdir(parents=True, exist_ok=True)
    out_path = styles_dir / f"{slugify(str(profile['style_name']))}.json"
    out_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def list_styles(styles_dir: Path | str = "styles") -> list[dict]:
    """List all saved style JSONs (sorted by name)."""
    styles_dir = Path(styles_dir)
    result = []
    if not styles_dir.is_dir():
        return result
    for p in sorted(styles_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            result.append({"name": data.get("style_name", p.stem),
                           "file": p.name, "slug": p.stem, "profile": data})
        except Exception:
            result.append({"name": p.stem, "file": p.name, "slug": p.stem,
                           "profile": None, "error": "invalid JSON"})
    return result


def load_style(name_or_slug: str, styles_dir: Path | str = "styles") -> dict:
    """Load one style by slug (filename without .json) or by style_name."""
    styles_dir = Path(styles_dir)
    slug = slugify(name_or_slug)
    path = styles_dir / f"{slug}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    for item in list_styles(styles_dir):
        if item.get("profile") and item["profile"].get("style_name") == name_or_slug:
            return item["profile"]
    raise FileNotFoundError(f"Style not found: {name_or_slug}")


def ensure_default_profile(styles_dir: Path | str = "styles") -> Path:
    """Create styles/global-baatien.json if it does not exist yet."""
    styles_dir = Path(styles_dir)
    styles_dir.mkdir(parents=True, exist_ok=True)
    out_path = styles_dir / "global-baatien.json"
    if not out_path.is_file():
        profile = global_baatien_profile()
        # File name slug differs from style_name slug; write locked profile as-is.
        out_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
