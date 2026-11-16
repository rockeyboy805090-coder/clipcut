"""Phase 2: clip Options — the exact panel from the CreativeCut teardown.

One module owns every option, its defaults, its validation and its
effect, so the dashboard, /youtube-job, /find-clips and /render-clip
can never drift apart:

  clip count      stepper, default 2 ("The AI scores every moment, and
                  keeps this many.")
  clip length     Shortest 15s, Longest 60s ("The AI can pick any length
                  in between.")
  caption style   Bounce (default), Hype, Clean — real ASS presets in
                  styles/captions/*.json, merged onto the base style at
                  render time (see apply_caption_preset).
  frame layout    Blurred fill (default), Zoomed in, Fill screen
  video quality   720p Fast (default) vs 1080p Sharper (~2x slower)
  video type      Auto-detect (default) + 6 types — recorded on every
                  job/clips file and passed into the Gemini rescore
                  prompt when a key is set (see clip_finder).
  only part       From (minute) / To (minute) — clips only inside it.

Validation rule (spec): bad values are REJECTED with a clear
Hinglish/English message — never silently replaced. The ONLY silent
behaviour is documented defaults when a field is empty/missing.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAPTIONS_DIR = ROOT / "styles" / "captions"

# --- Video types (teardown §1.3, exact labels) ---
VIDEO_TYPES = [
    ("auto", "Auto-detect type"),
    ("gaming", "Gaming and streams"),
    ("podcast", "Podcast or interview"),
    ("education", "Education and how-to"),
    ("business", "Business and money"),
    ("comedy", "Comedy"),
    ("vlog", "Vlog and lifestyle"),
]
VIDEO_TYPE_VALUES = {v for v, _ in VIDEO_TYPES}
VIDEO_TYPE_LABELS = dict(VIDEO_TYPES)

# --- Caption styles (teardown §1.3, exact labels) ---
CAPTION_STYLES = {
    "bounce": {
        "label": "Bounce",
        "description": "Big, bouncy captions with yellow highlights.",
    },
    "hype": {
        "label": "Hype",
        "description": "Bold captions with pink highlights.",
    },
    "clean": {
        "label": "Clean",
        "description": "Quiet, plain captions.",
    },
}

# --- Frame layouts (teardown §1.3, exact labels) ---
LAYOUTS = {
    "blurred_fill": {
        "label": "Blurred fill",
        "description": "Full frame, with the blurred video filling the rest.",
    },
    "zoomed_in": {
        "label": "Zoomed in",
        "description": "Zoomed in, with the blurred video filling the rest.",
    },
    "fill_screen": {
        "label": "Fill screen",
        "description": "Zoomed in, so the frame is filled.",
    },
}

# --- Video quality (teardown §1.3) ---
QUALITIES = {
    "720p": {"label": "720p, Fast", "note": "Quick to make, and looks great.",
             "width": 720, "height": 1280},
    "1080p": {"label": "1080p, Sharper", "note": "About 2x slower to make.",
              "width": 1080, "height": 1920},
}

DEFAULT_OPTIONS = {
    "clip_count": 2,
    "min_sec": 15.0,
    "max_sec": 60.0,
    "caption_style": "bounce",
    "layout": "blurred_fill",
    "quality": "720p",
    "video_type": "auto",
    "from_sec": None,
    "to_sec": None,
    # Phase 3 dual-judge thresholds (CreativeCut rule: clip tabhi jab
    # DONO judge agree karein). Tunable via Options; defaults 50/40.
    "words_threshold": 50,
    "reaction_threshold": 40,
    # Phase 5 tuning (see dual_judge.py + clip_finder._dedupe_rank):
    # dedupe_overlap 0.6 = keep a window unless it overlaps >60% of
    # the shorter clip with an already-picked one (was 0.4 — short
    # videos returned fewer clips than asked).
    # reaction_quiet_db / reaction_quiet_cap = quiet floor: a window
    # whose speech mean is quieter than -38 dB gets its level factor
    # capped at 0.25, so a half-quiet video's quiet half can't pass
    # on a low median alone.
    "dedupe_overlap": 0.6,
    "reaction_quiet_db": -38.0,
    "reaction_quiet_cap": 0.25,
}

# Hard limits from the task spec (validation, not defaults).
MIN_COUNT, MAX_COUNT = 1, 8
MIN_SHORT_SEC = 5.0      # shortest clip must be >= 5s
MAX_LONG_SEC = 120.0     # longest clip must be <= 120s


def _is_empty(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def _as_float(v, field: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{field} number hona chahiye — mila: {v!r}")


def _as_int(v, field: str) -> int:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{field} number hona chahiye — mila: {v!r}")
    if f != int(f):
        raise ValueError(f"{field} poora number hona chahiye — mila: {v!r}")
    return int(f)


def parse_options(raw: dict | None, video_duration: float | None = None) -> dict:
    """Validate raw option fields and return a normalised options dict.

    `raw` may use either the dashboard names (clip_count, min_sec,
    max_sec, from_min, to_min) or the API aliases (top_n, shortest_sec,
    longest_sec, from_sec, to_sec). Empty/missing fields take the
    documented defaults; anything else invalid raises ValueError with
    a Hinglish message. `video_duration` (seconds, when known) bounds
    the From/To window.
    """
    raw = dict(raw or {})

    def pick(*names):
        for n in names:
            if n in raw and not _is_empty(raw[n]):
                return raw[n]
        return None

    opts = dict(DEFAULT_OPTIONS)

    # --- clip count: 1-8 ---
    v = pick("clip_count", "top_n")
    if v is not None:
        n = _as_int(v, "Clip count")
        if not MIN_COUNT <= n <= MAX_COUNT:
            raise ValueError(
                f"Clip count {MIN_COUNT}-{MAX_COUNT} ke beech hona chahiye — mila: {n}")
        opts["clip_count"] = n

    # --- clip length: shortest >= 5s, longest <= 120s, shortest < longest ---
    v = pick("min_sec", "shortest_sec")
    if v is not None:
        opts["min_sec"] = _as_float(v, "Shortest clip length")
    v = pick("max_sec", "longest_sec")
    if v is not None:
        opts["max_sec"] = _as_float(v, "Longest clip length")
    if opts["min_sec"] < MIN_SHORT_SEC:
        raise ValueError(
            f"Shortest clip kam se kam {MIN_SHORT_SEC:.0f} seconds ka hona chahiye — mila: {opts['min_sec']}s")
    if opts["max_sec"] > MAX_LONG_SEC:
        raise ValueError(
            f"Longest clip zyada se zyada {MAX_LONG_SEC:.0f} seconds ka hona chahiye — mila: {opts['max_sec']}s")
    if opts["min_sec"] >= opts["max_sec"]:
        raise ValueError(
            f"Shortest ({opts['min_sec']}s) Longest ({opts['max_sec']}s) se chhota hona chahiye")

    # --- Phase 3 dual-judge thresholds: 0-100 each ---
    v = pick("words_threshold")
    if v is not None:
        n = _as_int(v, "Words threshold")
        if not 0 <= n <= 100:
            raise ValueError(f"Words threshold 0-100 ke beech hona chahiye — mila: {n}")
        opts["words_threshold"] = n
    v = pick("reaction_threshold")
    if v is not None:
        n = _as_int(v, "Reaction threshold")
        if not 0 <= n <= 100:
            raise ValueError(f"Reaction threshold 0-100 ke beech hona chahiye — mila: {n}")
        opts["reaction_threshold"] = n

    # --- Phase 5 tuning: dedupe overlap + reaction quiet floor ---
    v = pick("dedupe_overlap")
    if v is not None:
        f = _as_float(v, "Dedupe overlap")
        if not 0.1 <= f <= 0.9:
            raise ValueError(
                f"Dedupe overlap 0.1-0.9 ke beech hona chahiye — mila: {f}")
        opts["dedupe_overlap"] = f
    v = pick("reaction_quiet_db")
    if v is not None:
        f = _as_float(v, "Reaction quiet dB")
        if not -60.0 <= f <= -20.0:
            raise ValueError(
                f"Reaction quiet dB -60 se -20 ke beech hona chahiye — mila: {f}")
        opts["reaction_quiet_db"] = f
    v = pick("reaction_quiet_cap")
    if v is not None:
        f = _as_float(v, "Reaction quiet cap")
        if not 0.0 <= f <= 1.0:
            raise ValueError(
                f"Reaction quiet cap 0-1 ke beech hona chahiye — mila: {f}")
        opts["reaction_quiet_cap"] = f

    # --- enums: invalid values are errors, never silent defaults ---
    v = pick("caption_style")
    if v is not None:
        s = str(v).strip().lower()
        if s not in CAPTION_STYLES:
            raise ValueError(
                f"Caption style galat hai: {v!r} — Bounce, Hype ya Clean me se chuno")
        opts["caption_style"] = s
    v = pick("layout")
    if v is not None:
        s = str(v).strip().lower()
        if s not in LAYOUTS:
            raise ValueError(
                f"Layout galat hai: {v!r} — Blurred fill, Zoomed in ya Fill screen me se chuno")
        opts["layout"] = s
    v = pick("quality")
    if v is not None:
        s = str(v).strip().lower()
        if s not in QUALITIES:
            raise ValueError(
                f"Quality galat hai: {v!r} — 720p ya 1080p me se chuno")
        opts["quality"] = s
    v = pick("video_type")
    if v is not None:
        s = str(v).strip().lower()
        if s not in VIDEO_TYPE_VALUES:
            raise ValueError(
                f"Video type galat hai: {v!r} — {', '.join(sorted(VIDEO_TYPE_VALUES))} me se chuno")
        opts["video_type"] = s

    # --- From/To window (minutes in the UI, seconds in the API) ---
    from_min = pick("from_min", "from_minute")
    to_min = pick("to_min", "to_minute")
    from_sec = pick("from_sec")
    to_sec = pick("to_sec")
    if from_min is not None:
        opts["from_sec"] = _as_float(from_min, "From (minute)") * 60.0
    elif from_sec is not None:
        opts["from_sec"] = _as_float(from_sec, "From (second)")
    if to_min is not None:
        opts["to_sec"] = _as_float(to_min, "To (minute)") * 60.0
    elif to_sec is not None:
        opts["to_sec"] = _as_float(to_sec, "To (second)")
    if opts["from_sec"] is not None and opts["from_sec"] < 0:
        raise ValueError("From (minute) 0 ya usse zyada hona chahiye")
    if opts["to_sec"] is not None and opts["to_sec"] <= 0:
        raise ValueError("To (minute) 0 se zyada hona chahiye")
    if opts["from_sec"] is not None and opts["to_sec"] is not None:
        if opts["from_sec"] >= opts["to_sec"]:
            raise ValueError(
                f"From ({opts['from_sec'] / 60:g} min) To ({opts['to_sec'] / 60:g} min) se chhota hona chahiye")
    if video_duration is not None:
        dur = float(video_duration)
        if opts["from_sec"] is not None and opts["from_sec"] >= dur:
            raise ValueError(
                f"From ({opts['from_sec'] / 60:g} min) video ki length ({dur / 60:.1f} min) se chhota hona chahiye")
        if opts["to_sec"] is not None and opts["to_sec"] > dur:
            raise ValueError(
                f"To ({opts['to_sec'] / 60:g} min) video ki length ({dur / 60:.1f} min) se zyada nahi ho sakta")

    return opts


def caption_preset(style: str) -> dict:
    """Load one caption preset (styles/captions/<style>.json).

    Returns the preset's style overrides (caption_*, zoom_percent).
    Falls back to the in-code defaults if the file is missing — a
    missing preset file must never break a render.
    """
    name = (style or "bounce").strip().lower()
    if name not in CAPTION_STYLES:
        raise ValueError(f"Caption style galat hai: {style!r}")
    path = CAPTIONS_DIR / f"{name}.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("caption_style") == name:
                return dict(data.get("overrides") or {})
        except Exception:
            pass
    return dict(_BUILTIN_PRESETS[name])


_BUILTIN_PRESETS = {
    # Bounce: big, bouncy captions, yellow highlight.
    "bounce": {"caption_size_px": 88, "caption_color": "#FFFFFF",
               "highlight_color": "#FFD60A", "caption_all_caps": True,
               "outline_px": 4, "caption_position_y_percent": 62,
               "zoom_percent": 110},
    # Hype: bold captions, pink highlight, larger.
    "hype": {"caption_size_px": 96, "caption_color": "#FFFFFF",
             "highlight_color": "#FF2E63", "caption_all_caps": True,
             "outline_px": 4, "caption_position_y_percent": 60,
             "zoom_percent": 115},
    # Clean: quiet, plain captions — highlight == caption colour, so
    # no word stands out; no zoom punch, no all-caps shouting.
    "clean": {"caption_size_px": 72, "caption_color": "#FFFFFF",
              "highlight_color": "#FFFFFF", "caption_all_caps": False,
              "outline_px": 2, "caption_position_y_percent": 68,
              "zoom_percent": 100},
}


def apply_caption_preset(base_style: dict, caption_style: str) -> dict:
    """Return base_style with one caption preset merged on top.

    The base (e.g. Global Baatien) keeps its grade/music/zoom
    defaults; the preset only overrides caption look + zoom punch.
    """
    merged = dict(base_style or {})
    merged.update(caption_preset(caption_style))
    merged["caption_style"] = (caption_style or "bounce").strip().lower()
    return merged


def render_size(quality: str) -> tuple[int, int]:
    """(width, height) for a quality key — 720p = 720x1280, 1080p = 1080x1920."""
    q = QUALITIES.get((quality or "720p").strip().lower())
    if q is None:
        raise ValueError(f"Quality galat hai: {quality!r}")
    return q["width"], q["height"]


def options_summary(opts: dict) -> dict:
    """The small options block recorded on jobs / clips / sidecars."""
    o = dict(opts or {})
    return {
        "clip_count": o.get("clip_count", 2),
        "min_sec": o.get("min_sec", 15.0),
        "max_sec": o.get("max_sec", 60.0),
        "caption_style": o.get("caption_style", "bounce"),
        "layout": o.get("layout", "blurred_fill"),
        "quality": o.get("quality", "720p"),
        "video_type": o.get("video_type", "auto"),
        "from_sec": o.get("from_sec"),
        "to_sec": o.get("to_sec"),
        "words_threshold": o.get("words_threshold", 50),
        "reaction_threshold": o.get("reaction_threshold", 40),
        "dedupe_overlap": o.get("dedupe_overlap", 0.6),
        "reaction_quiet_db": o.get("reaction_quiet_db", -38.0),
        "reaction_quiet_cap": o.get("reaction_quiet_cap", 0.25),
    }
