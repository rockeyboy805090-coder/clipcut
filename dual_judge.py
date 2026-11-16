"""Phase 3: the Reaction judge — one half of dual-judge scoring.

CreativeCut rule (teardown §1.5): two separate judges must AGREE
before a clip is made.
  * Words judge   — reads ONLY what was said (transcript text).
    Lives in clip_finder.py (Gemini when a key is set, else the
    rule-based scorer).
  * Reaction judge — sees ONLY loudness + chat activity. Lives here.

This module never looks at transcript text. It scores a candidate
window 0-100 from:
  (a) loudness — one ffmpeg pass (`astats`, per-frame RMS/Peak),
      aggregated into 1-second bins. Cheap: one decode, no torch,
      no extra model, safe on 512 MB Render Free.
  (b) chat activity — ONLY when the caller passes it in
      (list of {"time": seconds, "count": n} or
      {"start": s, "end": s, "count": n}). No chat is fetched here:
      Twitch/Kick/YouTube chat wiring belongs to Phase 7 (live
      recording). When no chat data is supplied, the reaction score
      is loudness-only and the reason SAYS SO — chat numbers are
      never invented.

Loudness scoring (documented, deterministic):
  level      0-1   candidate speech-mean RMS vs the whole video's
                   median RMS: -12 dB or quieter -> 0, +6 dB -> 1
  presence   0-30  fraction of 1s bins with RMS > -45 dB, scaled by
                   level (a uniformly quiet window's speech counts
                   less — otherwise quiet clips would score ~50 on
                   presence + dynamics alone and never be dropped)
  relative   0-40  40 * level
  dynamics   0-30  crest (peak - mean) up to 15 + RMS std-dev up to 15
                   (a loud moment that varies beats flat loudness)
  Typical: normal speech ~65-75, loud moment ~85+, quiet (-14 dB)
  window ~20, silence 0.
Chat scoring (only when chat data exists):
  window message rate vs the video's average rate, in log space:
  half the average rate -> 20, average -> 50, double -> 75,
  4x or more -> 100. Combined reaction = 0.7 * loudness + 0.3 * chat.
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path

SILENCE_FLOOR_DB = -45.0   # bins louder than this count as speech
BIN_SEC = 1.0
# dB values at/below this are treated as silence (-inf from ffmpeg).
MIN_DB = -90.0

_FRAME_RE = re.compile(r"frame:\d+\s+pts:\d+\s+pts_time:([0-9.]+)")
_RMS_RE = re.compile(r"RMS_level=(-inf|[0-9.+-]+)")
_PEAK_RE = re.compile(r"Peak_level=(-inf|[0-9.+-]+)")


def _db(value: str) -> float:
    """Parse an ffmpeg dB value; '-inf' becomes MIN_DB (silence)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return MIN_DB
    if v == float("-inf") or v < MIN_DB:
        return MIN_DB
    return v


def extract_loudness_curve(video_path, bin_sec: float = BIN_SEC) -> dict:
    """One ffmpeg pass over the audio track -> per-second loudness bins.

    Returns {"bins": [{"start", "rms_db", "peak_db"}], "duration",
    "median_db", "source": "ffmpeg-astats"}.
    Raises RuntimeError if ffmpeg is missing or the file has no audio.
    Bins with no audio frames get rms_db None (counted as silence).
    """
    video_path = Path(video_path)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg install karo — loudness measure nahi ho sakta")
    # astats with metadata=1 + reset=1 emits one RMS/Peak pair per
    # audio frame (~1024 samples). NOTE: one ametadata filter prints
    # only ONE key (a second key= is ignored) — so two ametadata
    # filters in series, one per key. Each prints its own frame: line
    # + lavfi line; parse line-by-line, keyed by pts_time.
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-y", "-v", "info", "-i", str(video_path),
         "-vn", "-af", "astats=metadata=1:reset=1,"
         "ametadata=print:key=lavfi.astats.Overall.RMS_level,"
         "ametadata=print:key=lavfi.astats.Overall.Peak_level",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        err = (result.stderr or "").strip().splitlines()
        raise RuntimeError(
            "Loudness measure nahi hua — audio track nahi mili ya file "
            f"kharab hai. {err[-1] if err else ''}")
    # Line-by-line state machine: a "frame:" line sets the current
    # time; the following RMS/Peak lines attach to it.
    per_frame: dict[float, dict] = {}
    cur_time: float | None = None
    for line in (result.stderr or "").splitlines():
        fm = _FRAME_RE.search(line)
        if fm:
            cur_time = float(fm.group(1))
            per_frame.setdefault(cur_time, {})
            continue
        if cur_time is None:
            continue
        rm = _RMS_RE.search(line)
        if rm:
            per_frame[cur_time]["rms"] = _db(rm.group(1))
            continue
        pm = _PEAK_RE.search(line)
        if pm:
            per_frame[cur_time]["peak"] = _db(pm.group(1))
    frames: list[tuple[float, float, float]] = []  # (time, rms, peak)
    for t in sorted(per_frame):
        d = per_frame[t]
        if "rms" not in d:
            continue
        frames.append((t, d["rms"], d.get("peak", MIN_DB)))
    if not frames:
        raise RuntimeError(
            "Loudness measure nahi hua — astats se koi audio frame nahi mila")
    duration = max(t for t, _, _ in frames) + 0.05
    n_bins = max(1, int(math.ceil(duration / float(bin_sec))))
    bins: list[dict] = []
    for b in range(n_bins):
        lo, hi = b * bin_sec, (b + 1) * bin_sec
        vals = [(r, p) for t, r, p in frames if lo <= t < hi]
        if not vals:
            bins.append({"start": round(lo, 2), "rms_db": None, "peak_db": None})
        else:
            bins.append({
                "start": round(lo, 2),
                "rms_db": round(sum(r for r, _ in vals) / len(vals), 2),
                "peak_db": round(max(p for _, p in vals), 2),
            })
    heard = [b["rms_db"] for b in bins if b["rms_db"] is not None]
    median_db = None
    if heard:
        s = sorted(heard)
        median_db = round(s[len(s) // 2], 2)
    return {"bins": bins, "duration": round(duration, 2),
            "median_db": median_db, "source": "ffmpeg-astats"}


def _window_bins(curve: dict, start: float, end: float) -> list[dict]:
    return [b for b in curve.get("bins", [])
            if b["rms_db"] is not None
            and float(b["start"]) < end
            and float(b["start"]) + BIN_SEC > start]


def score_loudness(start: float, end: float, curve: dict) -> tuple[int, dict]:
    """Score one window 0-100 from the loudness curve alone.

    Returns (score, breakdown) where breakdown carries the raw
    numbers (mean/peak/median dB, speech fraction) so the reason is
    checkable, not hand-waved.
    """
    bins = _window_bins(curve, float(start), float(end))
    median_db = curve.get("median_db")
    if not bins or median_db is None:
        return 0, {"loudness_score": 0, "mean_db": None, "peak_db": None,
                   "median_db": median_db, "speech_fraction": 0.0,
                   "note": "is window me koi audio bin nahi mila"}
    rms_vals = [b["rms_db"] for b in bins]
    speech_frac = sum(1 for v in rms_vals if v > SILENCE_FLOOR_DB) / len(rms_vals)
    if speech_frac <= 0:
        return 0, {"loudness_score": 0, "mean_db": None, "peak_db": None,
                   "median_db": median_db, "speech_fraction": 0.0,
                   "note": "is window me speech nahi hai (sab silence)"}
    # Mean / std over SPEECH bins only: silent bins (-90 floor) would
    # drag the mean down and inflate crest/std, so dynamics would
    # saturate at 30 for every normal speech window.
    speech_vals = [v for v in rms_vals if v > SILENCE_FLOOR_DB]
    mean_db = sum(speech_vals) / len(speech_vals)
    peak_db = max(b["peak_db"] for b in bins if b["peak_db"] is not None)
    # level factor 0-1: how loud this window is vs the whole video.
    level_factor = max(0.0, min(1.0, (mean_db - median_db + 12.0) / 18.0))
    # presence 0-30, scaled by level (see docstring: quiet speech
    # must not score ~50 on presence + dynamics alone).
    presence = 30.0 * speech_frac * level_factor
    # relative 0-40
    rel = level_factor * 40.0
    # dynamics 0-30: crest (peak-mean) up to 15, RMS std-dev up to 15
    crest = max(0.0, peak_db - mean_db)
    crest_part = min(1.0, crest / 12.0) * 15.0
    mean_sq = sum((v - mean_db) ** 2 for v in speech_vals) / len(speech_vals)
    std_db = math.sqrt(mean_sq)
    var_part = min(1.0, std_db / 10.0) * 15.0
    score = int(round(presence + rel + crest_part + var_part))
    return max(0, min(100, score)), {
        "loudness_score": max(0, min(100, score)),
        "mean_db": round(mean_db, 1), "peak_db": round(peak_db, 1),
        "median_db": median_db, "speech_fraction": round(speech_frac, 2),
        "crest_db": round(crest, 1), "std_db": round(std_db, 1),
        "presence": round(presence, 1), "relative": round(rel, 1),
        "dynamics": round(crest_part + var_part, 1),
        "level_factor": round(level_factor, 2),
    }


def _normalise_chat(chat_activity) -> list[dict] | None:
    """Accept [{"time", "count"}] or [{"start","end","count"}]; else None.

    Returns a sorted list of (start, end, count) buckets, or None when
    the caller passed nothing usable (no chat data — NOT an error).
    """
    if not chat_activity:
        return None
    buckets: list[tuple[float, float, int]] = []
    for item in chat_activity:
        if not isinstance(item, dict):
            continue
        try:
            count = int(item.get("count", 0))
        except (TypeError, ValueError):
            continue
        if count < 0:
            continue
        if "time" in item:
            try:
                t = float(item["time"])
            except (TypeError, ValueError):
                continue
            buckets.append((t, t + 1.0, count))
        elif "start" in item and "end" in item:
            try:
                buckets.append((float(item["start"]), float(item["end"]), count))
            except (TypeError, ValueError):
                continue
    if not buckets:
        return None
    return [{"start": s, "end": e, "count": c}
            for s, e, c in sorted(buckets)]


def score_chat(start: float, end: float, chat_activity,
               video_duration: float | None) -> tuple[int | None, dict]:
    """Score one window 0-100 from chat activity, or (None, {...}).

    Returns None as the score when there is no usable chat data at
    all — the caller must then run on loudness alone and say so.
    """
    buckets = _normalise_chat(chat_activity)
    if buckets is None:
        return None, {"chat_score": None, "chat_messages": 0,
                      "chat_available": False,
                      "note": "chat data nahi mila — sirf loudness se score"}
    window_msgs = 0
    for b in buckets:
        overlap = max(0.0, min(end, b["end"]) - max(start, b["start"]))
        span = max(0.01, b["end"] - b["start"])
        window_msgs += b["count"] * (overlap / span)
    total_msgs = sum(b["count"] for b in buckets)
    dur = float(video_duration) if video_duration else None
    if not dur or dur <= 0:
        dur = max((b["end"] for b in buckets), default=0.0) or None
    if not dur or total_msgs <= 0:
        return None, {"chat_score": None, "chat_messages": 0,
                      "chat_available": False,
                      "note": "chat data khaali hai — sirf loudness se score"}
    avg_rate = total_msgs / dur
    win_len = max(1.0, end - start)
    win_rate = window_msgs / win_len
    ratio = win_rate / avg_rate if avg_rate > 0 else 0.0
    # log-space: 0.5x -> 20, 1x -> 50, 2x -> 75, 4x+ -> 100
    if ratio <= 0:
        score = 0
    else:
        score = int(round(50 + 25 * math.log2(max(ratio, 0.01))))
        score = max(0, min(100, score))
    return score, {"chat_score": score,
                   "chat_messages": round(window_msgs, 1),
                   "chat_available": True,
                   "window_rate_per_min": round(win_rate * 60, 1),
                   "average_rate_per_min": round(avg_rate * 60, 1)}


def reaction_judge(start: float, end: float, curve: dict | None,
                   chat_activity=None,
                   video_duration: float | None = None) -> dict:
    """The Reaction judge: loudness (+ chat when supplied).

    Returns {"score", "reason", "breakdown"}.
    curve None -> neutral 50, reason says loudness could not be
    measured (never a fake score).
    """
    if curve is None:
        return {"score": 50, "reason": (
            "Reaction 50/100 — loudness measure nahi ho saka, neutral score "
            "(video file nahi mili ya audio track nahi hai)"),
            "breakdown": {"loudness_score": None, "chat_score": None,
                          "chat_available": False,
                          "note": "no loudness curve — neutral 50"}}
    loud_score, loud_bd = score_loudness(start, end, curve)
    chat_score, chat_bd = score_chat(start, end, chat_activity, video_duration)
    breakdown = {**loud_bd, **chat_bd}
    if chat_score is None:
        score = loud_score
        reason = (f"Reaction {score}/100 — loudness se (mean "
                  f"{loud_bd.get('mean_db')} dB, video median "
                  f"{loud_bd.get('median_db')} dB, speech "
                  f"{loud_bd.get('speech_fraction', 0) * 100:.0f}%, "
                  f"crest {loud_bd.get('crest_db')} dB); "
                  f"chat data nahi mila, isliye sirf loudness")
    else:
        score = int(round(0.7 * loud_score + 0.3 * chat_score))
        reason = (f"Reaction {score}/100 — loudness {loud_score} (mean "
                  f"{loud_bd.get('mean_db')} dB vs median "
                  f"{loud_bd.get('median_db')} dB) + chat {chat_score} "
                  f"({chat_bd.get('chat_messages')} msgs is window me, "
                  f"avg {chat_bd.get('average_rate_per_min')}/min)")
    return {"score": max(0, min(100, score)), "reason": reason,
            "breakdown": breakdown}
