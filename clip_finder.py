"""Phase 3: viral clip finding with DUAL-JUDGE scoring — real implementation.

Free stack only. Two separate judges must AGREE before a clip is kept
(CreativeCut rule, teardown §1.5):
  * Words judge    — reads ONLY what was said (transcript text).
    Gemini FREE API when GEMINI_API_KEY is set (and the library is
    installed); otherwise the rule-based scorer below. Without a key
    the output honestly records scoring_method="rule_based" — Gemini
    is NEVER claimed when it was not used.
  * Reaction judge — sees ONLY loudness (+ chat activity when the
    caller passes it in). Lives in dual_judge.py; one ffmpeg astats
    pass, no torch, no extra model.
Combine (defaults, tunable via Options): keep a candidate ONLY when
words_score >= 50 AND reaction_score >= 40. Combined score =
round(0.6 * words + 0.4 * reaction); the old single `score` field IS
the combined score, so render/UI code keeps working.

Pipeline:
  transcript (words + segments) -> long segments split into
  sentence-level pieces (Phase 3: coarse Whisper segments yielded
  too few candidates on short videos) -> candidate windows
  (min_sec-max_sec, default 15-60s) -> words score 0-100 each:
    Hook strength, first 5 sec (question/shock/secret/mistake/money/
    number/guest name) = 30
    Emotion / curiosity = 30
    Complete idea in 60s = 20
    Loop potential (end connects to start / ends on a tease) = 20
  -> reaction score 0-100 each (loudness + chat)
  -> BOTH-judges gate -> dedupe overlapping windows -> TOP clips.

Output is saved to outputs/clips/<upload_id>_clips.json.
Words are never invented — hook_line and transcript_text come straight
from the transcript.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

CLIPS_DIR = Path(__file__).resolve().parent / "outputs" / "clips"

MIN_CLIP_SEC = 25.0
TARGET_CLIP_SEC = 45.0
MAX_CLIP_SEC = 65.0

# --- Keyword banks ---
# DEFAULT LANGUAGE IS ENGLISH (user correction, Phase 4).
# English banks are PRIMARY: when language == "en" only the English
# banks score (no Hindi/Urdu words can inflate a score). "hinglish"
# adds the Roman (ASCII) Hindi words; "hi" uses all banks as before.
# The old all-language banks are kept below as fallback for "hi".
HOOK_WORDS_EN = {
    "secret", "mistake", "money", "never", "stop", "free", "warning",
    "exposed", "banned", "reveal", "revealed", "truth", "shocking",
    "shock", "danger", "million", "billion", "lakh", "crore",
    "save", "earn", "rich", "fail", "failure", "win", "first", "last",
    "nobody", "everyone", "wrong", "right", "lie", "lies", "scam",
    "how", "why", "what", "when", "where", "who", "which",
    "how to", "do not", "dont", "don't", "must", "always",
}
EMOTION_WORDS_EN = {
    "fear", "happy", "happiness", "angry", "anger", "cry", "crying",
    "laugh", "laughing", "shock", "shocked", "amazing", "unbelievable",
    "incredible", "crazy", "surprised", "surprise", "pain", "dream",
    "dreams", "win", "winning", "lose", "losing", "success", "fail",
    "failure", "struggle", "hard work", "luck", "emotional", "heart",
    "life", "death", "love", "hate", "happy", "sad", "proud",
}
CURIOSITY_WORDS_EN = {
    "secret", "first time", "never before", "nobody knows", "finally",
    "why", "how", "real", "truth", "hidden", "reason", "because",
    "imagine", "what happens", "what if", "did you know",
}
TEASE_END_WORDS_EN = {"and", "but", "then", "next", "part", "wait",
                      "coming", "tomorrow", "stay tuned", "watch"}
QUESTION_WORDS_EN = {"what", "why", "how", "when", "where", "who", "which"}
QUESTION_WORDS_HINGLISH = {"kya", "kaise", "kyun", "kyon", "kab", "kahan", "kaun"}
QUESTION_WORDS_HINDI = {"क्या", "कैसे", "क्यों", "कब", "कहाँ", "कौन",
                         "کیا", "کیسے", "کیوں", "کب", "کہاں", "کون"}


def _roman_words(bank) -> set:
    """ASCII-only words of a bank (Roman Hinglish subset)."""
    return {w for w in bank if w.isascii()}


def _banks_for(language: str) -> tuple:
    """(hook, emotion, curiosity, tease, question_words) for a language."""
    lang = (language or "en").lower()
    if lang == "en":
        return (HOOK_WORDS_EN, EMOTION_WORDS_EN, CURIOSITY_WORDS_EN,
                TEASE_END_WORDS_EN, QUESTION_WORDS_EN)
    if lang == "hinglish":
        return (HOOK_WORDS_EN | _roman_words(HOOK_WORDS),
                EMOTION_WORDS_EN | _roman_words(EMOTION_WORDS),
                CURIOSITY_WORDS_EN | _roman_words(CURIOSITY_WORDS),
                TEASE_END_WORDS_EN | _roman_words(TEASE_END_WORDS),
                QUESTION_WORDS_EN | QUESTION_WORDS_HINGLISH)
    return (HOOK_WORDS, EMOTION_WORDS, CURIOSITY_WORDS,
            TEASE_END_WORDS,
            QUESTION_WORDS_EN | QUESTION_WORDS_HINGLISH | QUESTION_WORDS_HINDI)


# --- Keyword banks (Roman Hinglish + Devanagari + Urdu script + English) --
# Urdu script is included because Whisper base often renders Hindi in
# Perso-Arabic script (words right, script Urdu) — without these, Hindi
# transcripts score near zero on hook/emotion. Measured Nov 2026.
# These banks are now used ONLY when language == "hi".
HOOK_WORDS = {
    "kya", "kaise", "kyun", "kyon", "kab", "kahan", "kaun", "kitna", "kitne",
    "क्या", "कैसे", "क्यों", "कब", "कहाँ", "कौन", "कितना", "कितने",
    "کیا", "کیسے", "کیوں", "کب", "کہاں", "کون", "کتنا",
    "secret", "raaz", "रहस्य", "राज़", "galti", "गलती", "mistake", "paisa",
    "पैसा", "पैसे", "पैसों", "सीक्रेट", "money", "lakh", "लाख", "crore", "करोड़",
    "ہزار", "हजार", "हज़ार", "free", "muft", "मुफ्त",
    "پیسے", "پیسہ", "سیکریٹ", "غلطی", "لاکھ", "کروڑ", "سچ", "راز",
    "pehli", "पहली", "baar", "never", "kabhi", "shock", "shocking", "sach",
    "सच", "jhooth", "झूठ", "danger", "khatra", "खतरा", "banned", "exposed",
    "reveal", "khulasa", "खुलासा", "stop", "ruko", "रुको", "dekho", "देखो",
    "suno", "सुनो", "warning", "chetawani", "चेतावनी", "viral", "trend",
}
EMOTION_WORDS = {
    "darr", "डर", "khushi", "खुशी", "gussa", "गुस्सा", "rona", "रोना",
    "hasna", "हँसना", "shock", "shocked", "amazing", "unbelievable",
    "incredible", "pagal", "पागल", "hairan", "हैरान", "maza", "मज़ा",
    "dard", "दर्द", "sapna", "सपना", "dream", "jeet", "जीत", "haar", "हार",
    "success", "fail", "failure", "struggle", "mehnat", "मेहनत", "kismat",
    "किस्मत", "emotional", "dil", "दिल", "jaan", "जान", "zindagi", "जिंदगी",
    "life", "death", "maut", "मौत", "love", "pyaar", "प्यार", "hate",
    "ڈر", "خوشی", "غصہ", "رونا", "ہنسنا", "پاگل", "حیران", "مزہ",
    "درد", "سپنا", "جیت", "ہار", "محنت", "قسمت", "دل", "جان", "زندگی",
    "موت", "پیار",
}
CURIOSITY_WORDS = {
    "raaz", "secret", "pehli baar", "kabhi nahi", "koi nahi janta", "aakhir",
    "akhir", "kyun", "kaise", "asli", "real", "truth", "sach", "hidden",
    "chhupa", "छुपा", "wajah", "reason", "kyunki", "agar", "socho", "imagine",
    "kya hoga", "kya hota", "dekhiye", "janiye", "जानिए",
}
TEASE_END_WORDS = {"aur", "phir", "lekin", "magar", "aage", "next", "part",
                   "aage kya", "dekhte", "rukiye", "wait", "aakhir me"}
NUMBER_RE = re.compile(r"\d+")
QUESTION_RE = re.compile(r"[?؟]")


def _norm(text: str) -> str:
    return (text or "").lower()


def _first_seconds_text(words: list[dict], start: float, seconds: float = 5.0) -> str:
    return " ".join(w["word"] for w in words
                    if start <= w["start"] < start + seconds)


def _keyword_hits(text: str, keywords) -> int:
    """Count keywords present as substrings.

    Substring matching is deliberate: (a) Hindi inflects (पैसा/पैसे/पैसों,
    सीक्रेट/secret) and Whisper output varies, so exact-token matching
    misses most real hits; (b) Python's \\w splits Devanagari words at
    vowel signs (matras), so regex tokenisation is broken for Hindi —
    "जीत" tokenises as ["ज", "त"]. Substring matching avoids both.
    False positives on short words are acceptable for a 0-100 heuristic.
    """
    t = _norm(text)
    return sum(1 for kw in keywords if kw in t)


def _split_words(text: str) -> list[str]:
    """Whitespace split with punctuation stripped — works for Hindi and
    English (regex \\w breaks Devanagari at matras, so do not use it)."""
    return [w.strip(".,?!।\"'()[]{}:;—–-") for w in (text or "").split() if w.strip(".,?!।\"'()[]{}:;—–-")]


def _score_hook(first_text: str, language: str = "en") -> int:
    """0-30. Question / shock / secret / mistake / money / number in first 5s."""
    hook_words, _, _, _, question_words = _banks_for(language)
    t = _norm(first_text)
    score = 0
    if QUESTION_RE.search(first_text) or "?" in first_text:
        score += 12
    hits = _keyword_hits(first_text, hook_words)
    score += min(12, hits * 4)
    # Whisper rarely emits "?" — a question-word opening counts too.
    first_tokens = _split_words(first_text)[:4]
    if any(w in question_words for w in first_tokens):
        score += 8
    if NUMBER_RE.search(first_text):
        score += 6
    return min(30, score)


def _score_emotion(text: str, language: str = "en") -> int:
    """0-30. Emotion + curiosity words across the whole segment."""
    _, emotion_words, curiosity_words, _, _ = _banks_for(language)
    t = _norm(text)
    emo = _keyword_hits(text, emotion_words)
    cur = sum(1 for phrase in curiosity_words if phrase in t)
    score = min(18, emo * 3) + min(12, cur * 4)
    if "!" in text:
        score += 2
    return min(30, score)


def _score_complete(duration: float, text: str, min_sec: float = 25.0,
                    max_sec: float = 65.0) -> int:
    """0-20. Complete idea: right length + ends on sentence end.

    Full marks when the clip lands inside the chosen [min_sec,
    max_sec] range (Phase 2: the range comes from the Options panel,
    default 15-60s); near-misses get partial credit.
    """
    score = 0
    if min_sec <= duration <= max_sec:
        score += 12
    elif max(5.0, min_sec - 10) <= duration <= max_sec + 10:
        score += 7
    else:
        score += 3
    stripped = (text or "").strip()
    if stripped and stripped[-1] in ".?!।":
        score += 8
    elif stripped and len(stripped.split()) >= 20:
        score += 4
    return min(20, score)


def _score_loop(text: str, language: str = "en") -> int:
    """0-20. End connects to start (repeated word) or ends on a tease."""
    _, _, _, tease_words, _ = _banks_for(language)
    words = _split_words(_norm(text))
    if len(words) < 10:
        return 5
    first_set = set(words[:8])
    last_set = set(words[-8:])
    stop = {"aur", "hai", "hain", "ka", "ki", "ke", "to", "jo",
            "the", "and", "a", "an", "of", "to", "is", "are", "you"}
    overlap = len(first_set & last_set - stop)
    score = min(12, overlap * 4)
    t = _norm(text)
    if any(w in t[-60:] for w in tease_words):
        score += 8
    return min(20, score)


_SENT_END_RE = re.compile(r"[.!?\u0964\u06d4]+$")
_MAX_SENTENCE_SEC = 25.0  # run-on sentence safety: split long sentences


def _split_segments_into_sentences(transcript: dict) -> list[dict]:
    """Split long Whisper segments into sentence-level pieces.

    Phase 3 fix for the Phase 2 "count vs content" note: Whisper
    merged a 64.5s speech into just 3 segments, so the finder built
    only 2 candidates and dedupe left 1 clip for a count=2 request.
    Sentence pieces give the windower many more start boundaries,
    so short videos yield more candidates. Uses word timestamps
    (real when Whisper gave them, segment-derived otherwise), so
    sub-segment times are honest. A piece ends at a sentence
    terminator (. ? ! । ؟) or at an original segment end; run-on
    sentences are cut at ~25s (preferring a comma).
    Returns [{"start", "end", "text"}].
    """
    words = sorted(transcript.get("words") or [],
                   key=lambda w: float(w.get("start", 0)))
    segments = transcript.get("segments") or []
    if not segments:
        return []
    if not words:
        # No word timings: split each segment's text on sentences,
        # spreading time proportionally by character count.
        out: list[dict] = []
        for seg in segments:
            text = str(seg.get("text") or "").strip()
            start, end = float(seg.get("start", 0)), float(seg.get("end", 0))
            if not text or end <= start:
                out.append({"start": start, "end": end, "text": text})
                continue
            parts = [p.strip() for p in re.split(r"(?<=[.!?\u0964\u06d4])\s+", text) if p.strip()]
            if len(parts) <= 1:
                out.append({"start": start, "end": end, "text": text})
                continue
            total = sum(len(p) for p in parts)
            cur = start
            for p in parts:
                span = (end - start) * (len(p) / total) if total else 0
                out.append({"start": round(cur, 2), "end": round(cur + span, 2),
                            "text": p})
                cur += span
        return out
    out: list[dict] = []
    for seg in segments:
        ss, se = float(seg.get("start", 0)), float(seg.get("end", 0))
        seg_words = [w for w in words
                     if float(w.get("start", 0)) >= ss - 0.05
                     and float(w.get("start", 0)) < se]
        if not seg_words:
            # Segment with no timed words: keep it whole.
            text = str(seg.get("text") or "").strip()
            if text:
                out.append({"start": ss, "end": se, "text": text})
            continue
        cur: list[dict] = []
        last_comma_idx: int | None = None

        def flush(piece: list[dict]) -> None:
            if not piece:
                return
            out.append({"start": round(float(piece[0]["start"]), 2),
                        "end": round(float(piece[-1]["end"]), 2),
                        "text": " ".join(str(w["word"]) for w in piece)})

        for w in seg_words:
            cur.append(w)
            if str(w.get("word", "")).rstrip().endswith(","):
                last_comma_idx = len(cur) - 1
            span = float(w["end"]) - float(cur[0]["start"])
            if _SENT_END_RE.search(str(w.get("word", "")).strip()):
                flush(cur)
                cur, last_comma_idx = [], None
            elif span > _MAX_SENTENCE_SEC:
                # Run-on: cut at the last comma if there is one,
                # otherwise right here.
                if last_comma_idx is not None and last_comma_idx > 0:
                    flush(cur[:last_comma_idx + 1])
                    cur = cur[last_comma_idx + 1:]
                else:
                    flush(cur)
                    cur = []
                last_comma_idx = None
        flush(cur)
    return out


def score_candidate(text: str, first_text: str, duration: float,
                    language: str = "en", min_sec: float = 25.0,
                    max_sec: float = 65.0) -> tuple[int, dict]:
    """Score one candidate; returns (total 0-100, breakdown dict)."""
    hook = _score_hook(first_text, language)
    emo = _score_emotion(text, language)
    comp = _score_complete(duration, text, min_sec, max_sec)
    loop = _score_loop(text, language)
    return hook + emo + comp + loop, {
        "hook": hook, "emotion": emo, "complete": comp, "loop": loop}


def build_candidates(transcript: dict, language: str = "en",
                     min_sec: float = 25.0, max_sec: float = 65.0,
                     from_sec: float | None = None,
                     to_sec: float | None = None,
                     split_sentences: bool = True) -> list[dict]:
    """Sliding windows over transcript segments: min_sec-max_sec each.

    Windows start at every segment boundary and extend until they
    reach the target length (midpoint of the chosen range), closing at
    a segment end, never past max_sec. Phase 2: from_sec/to_sec clips
    the search to one part of the video (Options panel "Only look at
    part of the video"); segments outside the window are ignored and
    edge segments are trimmed to the window bounds. Short videos
    (< min_sec of speech) yield one window covering all speech.
    """
    words = transcript.get("words") or []
    segments = transcript.get("segments") or []
    if from_sec is not None or to_sec is not None:
        lo = float(from_sec) if from_sec is not None else 0.0
        hi = float(to_sec) if to_sec is not None else float("inf")
        words = [w for w in words if float(w.get("end", 0)) > lo
                 and float(w.get("start", 0)) < hi]
        clipped = []
        for s in segments:
            ss, se = float(s.get("start", 0)), float(s.get("end", 0))
            if se <= lo or ss >= hi:
                continue
            s2 = dict(s)
            s2["start"] = max(ss, lo)
            s2["end"] = min(se, hi)
            clipped.append(s2)
        segments = clipped
    if not segments and words:
        # No segments: synthesise one segment per ~10s of words.
        segments = []
        cur = [words[0]]
        for w in words[1:]:
            if w["start"] - cur[0]["start"] >= 10:
                segments.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                                 "text": " ".join(x["word"] for x in cur)})
                cur = [w]
            else:
                cur.append(w)
        if cur:
            segments.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                             "text": " ".join(x["word"] for x in cur)})
    if not segments:
        return []
    # Phase 3: split long segments into sentence pieces BEFORE
    # windowing — many more start boundaries, more candidates on
    # short videos. split_sentences=False keeps the old behaviour
    # (used by the Phase 3 before/after test only).
    if split_sentences:
        segments = _split_segments_into_sentences(
            {"words": words, "segments": segments})
    if not segments:
        return []

    target_sec = (float(min_sec) + float(max_sec)) / 2.0
    total_end = segments[-1]["end"]
    candidates = []
    for i, seg in enumerate(segments):
        start = float(seg["start"])
        text_parts, end = [], start
        for j in range(i, len(segments)):
            s = segments[j]
            if float(s["start"]) - start > float(max_sec):
                break
            text_parts.append(str(s.get("text") or "").strip())
            end = float(s["end"])
            if end - start >= 30 and j > i:
                # Close at a sentence/segment boundary once >=30s.
                if end - start >= target_sec or j == len(segments) - 1:
                    break
        duration = end - start
        if duration < 5:
            continue
        text = " ".join(p for p in text_parts if p)
        if not text:
            continue
        # Keep windows that are long enough, or the single full-speech
        # window when the whole video is short.
        if duration < float(min_sec) and len(segments) > 1:
            continue
        first_text = _first_seconds_text(words, start) if words else text[:80]
        score, breakdown = score_candidate(text, first_text, duration, language,
                                           min_sec, max_sec)
        candidates.append({
            "start": round(start, 2), "end": round(end, 2),
            "duration": round(duration, 2), "score": score,
            "breakdown": breakdown, "text": text,
            "first_text": first_text,
        })
    # If greedy windows produced nothing (very short video), one window.
    if not candidates and segments:
        text = " ".join(str(s.get("text") or "") for s in segments).strip()
        start, end = float(segments[0]["start"]), float(segments[-1]["end"])
        first_text = _first_seconds_text(words, start) if words else text[:80]
        score, breakdown = score_candidate(text, first_text, end - start, language,
                                           min_sec, max_sec)
        candidates.append({"start": round(start, 2), "end": round(end, 2),
                           "duration": round(end - start, 2), "score": score,
                           "breakdown": breakdown, "text": text,
                           "first_text": first_text})
    return candidates


def _hook_line(text: str) -> str:
    """First COMPLETE sentence (or first ~12 words) — the on-screen hook.

    Phase 5 fix: candidate windows can start mid-sentence, so the old
    "first sentence" was often a fragment (e.g. just "help"). Now pick
    the first sentence that actually ends with . ? ! or । AND has at
    least 4 words. If no complete sentence exists, take the first
    12 words (never a 1-word hook unless the whole text is 1 word).
    """
    text = (text or "").strip()
    if not text:
        return ""
    # Complete sentences: run of non-terminator chars + a terminator.
    for m in re.finditer(r"[^.!?\u0964\u06d4]+[.!?\u0964\u06d4]", text):
        sent = m.group(0).strip()
        if len(sent.split()) >= 4:
            words = sent.split()
            if len(words) > 14:
                sent = " ".join(words[:14]) + "…"
            return sent
    words = text.split()
    if len(words) > 14:
        return " ".join(words[:14]) + "…"
    return text


def _reason(breakdown: dict, score: int, language: str = "en") -> str:
    if (language or "en").lower() == "en":
        parts = []
        if breakdown["hook"] >= 18:
            parts.append("strong hook (question/secret/number in first 5 sec)")
        elif breakdown["hook"] >= 10:
            parts.append("decent hook")
        if breakdown["emotion"] >= 18:
            parts.append("high emotion/curiosity")
        elif breakdown["emotion"] >= 10:
            parts.append("some emotion/curiosity")
        if breakdown["complete"] >= 15:
            parts.append("complete idea, right length")
        if breakdown["loop"] >= 12:
            parts.append("loop/tease ending")
        if not parts:
            parts.append("average segment")
        return f"Score {score}/100 — " + ", ".join(parts)
    parts = []
    if breakdown["hook"] >= 18:
        parts.append("strong hook (sawaal/secret/number pehle 5 sec me)")
    elif breakdown["hook"] >= 10:
        parts.append("decent hook")
    if breakdown["emotion"] >= 18:
        parts.append("high emotion/curiosity")
    elif breakdown["emotion"] >= 10:
        parts.append("some emotion/curiosity")
    if breakdown["complete"] >= 15:
        parts.append("complete idea, sahi length")
    if breakdown["loop"] >= 12:
        parts.append("loop/tease ending")
    if not parts:
        parts.append("average segment")
    return f"Score {score}/100 — " + ", ".join(parts)


def _dedupe_rank(candidates: list[dict], top_n: int) -> list[dict]:
    """Rank by score, suppress windows overlapping >40% with a picked one."""
    ranked = sorted(candidates, key=lambda c: (-c["score"], c["start"]))
    picked: list[dict] = []
    for cand in ranked:
        overlap = False
        for p in picked:
            inter = max(0.0, min(cand["end"], p["end"]) - max(cand["start"], p["start"]))
            shorter = min(cand["duration"], p["duration"])
            if shorter > 0 and inter / shorter > 0.4:
                overlap = True
                break
        if not overlap:
            picked.append(cand)
        if len(picked) >= top_n:
            break
    return picked


def _gemini_api_key() -> str:
    """Read GEMINI_API_KEY from env or .env (never log the key)."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _gemini_words_judge(candidates: list[dict], video_type: str = "auto",
                        language: str = "en") -> dict | None:
    """The WORDS judge (Phase 3): Gemini reads ONLY what was said.

    Scores each candidate 0-100 on the transcript text alone — hook
    in the first 5 sec = 30, emotion/curiosity = 30, complete idea
    = 20, loop potential = 20. It is NEVER shown loudness, chat or
    any other signal (that is the Reaction judge's job, dual_judge.py).
    Returns {index: {"words_score", "words_reason", "hook_line"}} or
    None on any failure (no key, no library, API error) — the caller
    then uses the rule-based words score instead.
    Phase 2: the video type (when not auto-detect) is passed in the
    prompt so the model prefers moments that fit that type.
    """
    key = _gemini_api_key()
    if not key:
        return None
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        return None
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        brief = "\n".join(
            f"[{i}] {c['start']:.0f}-{c['end']:.0f}s | {c['text'][:300]}"
            for i, c in enumerate(candidates))
        type_line = ""
        if (video_type or "auto").lower() != "auto":
            type_line = (f"Video type: {video_type}. Prefer moments that "
                         f"fit a {video_type} video. ")
        prompt = (
            "You are the WORDS judge for short video clips. Read ONLY "
            "what was said in each candidate's transcript text — you "
            "cannot hear the audio and you get no loudness or chat data. "
            "Score each candidate 0-100 (hook in first 5 sec = 30, "
            "emotion/curiosity = 30, complete idea = 20, loop "
            "potential = 20). " + type_line +
            "Reply ONLY as a JSON array, "
            "no other text, no markdown fences, each item: "
            '{"id": <index>, "score": <0-100>, "reason": "<short, in Hinglish>", '
            '"hook_line": "<first striking line, copied from the text>"}.\n\n'
            f"Candidates:\n{brief}")
        resp = model.generate_content(prompt)
        raw = (getattr(resp, "text", "") or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        out: dict = {}
        for d in data:
            if not isinstance(d, dict) or "id" not in d:
                continue
            try:
                idx = int(d["id"])
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(candidates):
                out[idx] = {
                    "words_score": max(0, min(100, int(d.get("score", 0)))),
                    "words_reason": str(d.get("reason", ""))[:200],
                    "hook_line": str(d.get("hook_line", ""))[:160],
                }
        return out if out else None
    except Exception:
        return None


def _gemini_rescore(candidates: list[dict],
                    video_type: str = "auto") -> list[dict] | None:
    """Backwards-compatible wrapper (Phase 2 name).

    Applies the Gemini words judge and writes its scores onto the
    candidate dicts (old behaviour). New code should call
    _gemini_words_judge directly.
    """
    judged = _gemini_words_judge(candidates, video_type=video_type)
    if judged is None:
        return None
    out = []
    for i, cand in enumerate(candidates):
        if i in judged:
            cand = dict(cand)
            cand["score"] = judged[i]["words_score"]
            cand["gemini_reason"] = judged[i]["words_reason"]
            cand["gemini_hook"] = judged[i]["hook_line"]
        out.append(cand)
    return out


def find_clips(transcript: dict, upload_id: str, top_n: int = 5,
               style_name: str | None = None, use_gemini: bool = True,
               language: str | None = None, min_sec: float | None = None,
               max_sec: float | None = None, from_sec: float | None = None,
               to_sec: float | None = None, video_type: str = "auto",
               options: dict | None = None, video_path=None,
               chat_activity=None) -> dict:
    """Find top viral clips from a transcript dict — DUAL-JUDGE (Phase 3).

    Returns {"upload_id", "scoring_method", "scoring_language",
    "style_name", "clips": [...], "judges": {...}, "dropped": [...]}
    and saves outputs/clips/<upload_id>_clips.json.
    scoring_method is "gemini" only when the Gemini WORDS judge
    actually scored, otherwise "rule_based".
    Language: "en" by DEFAULT (English keyword banks are primary).
    Falls back to the transcript's language_requested, then "en".

    Phase 2: pass an `options` dict (see options.parse_options) or the
    individual min_sec/max_sec/from_sec/to_sec/video_type params.
    The resolved options are recorded in the output ("options") and on
    every clip ("video_type"), so a later render knows what was
    chosen. Defaults preserve the old behaviour (windows 25-65s).

    Phase 3: `video_path` enables the Reaction judge (loudness curve
    from the actual file). Without it the reaction score is a neutral
    50 and the reason says so. `chat_activity` is an optional list of
    {"time", "count"} / {"start", "end", "count"} buckets — no chat
    is fetched here (Phase 7). Thresholds come from options
    (words_threshold 50, reaction_threshold 40 by default).
    """
    from dual_judge import extract_loudness_curve, reaction_judge

    if options:
        min_sec = options.get("min_sec", min_sec)
        max_sec = options.get("max_sec", max_sec)
        from_sec = options.get("from_sec", from_sec)
        to_sec = options.get("to_sec", to_sec)
        video_type = options.get("video_type", video_type) or "auto"
        top_n = options.get("clip_count", top_n)
    min_sec = float(min_sec) if min_sec is not None else MIN_CLIP_SEC
    max_sec = float(max_sec) if max_sec is not None else MAX_CLIP_SEC
    video_type = (video_type or "auto").strip().lower()
    top_n = max(1, min(8, int(top_n or 5)))
    words_threshold = int((options or {}).get("words_threshold", 50))
    reaction_threshold = int((options or {}).get("reaction_threshold", 40))
    lang = (language or transcript.get("language_requested") or "en").lower()
    if lang not in ("en", "hinglish", "hi"):
        lang = "en"
    candidates = build_candidates(transcript, lang, min_sec=min_sec,
                                  max_sec=max_sec, from_sec=from_sec,
                                  to_sec=to_sec)
    # Segment counts for the Phase 3 report (before/after split).
    segments_count = len(transcript.get("segments") or [])
    try:
        split_segments_count = len(
            _split_segments_into_sentences(transcript))
    except Exception:
        split_segments_count = segments_count

    # --- WORDS judge: Gemini when a key is set, else rule-based. ---
    words_judge = "rule_based"
    judged = None
    if use_gemini and candidates:
        judged = _gemini_words_judge(candidates, video_type=video_type,
                                     language=lang)
        if judged is not None:
            words_judge = "gemini"
    scoring_method = words_judge
    for i, cand in enumerate(candidates):
        rule_score = int(cand["score"])
        cand["rule_score"] = rule_score
        if judged is not None and i in judged:
            cand["words_score"] = int(judged[i]["words_score"])
            cand["words_reason"] = judged[i]["words_reason"]
            cand["gemini_hook"] = judged[i]["hook_line"]
        else:
            cand["words_score"] = rule_score
            cand["words_reason"] = _reason(cand["breakdown"], rule_score, lang)

    # --- REACTION judge: loudness (+ chat when supplied). ---
    curve = None
    reaction_note = None
    if video_path is not None:
        try:
            curve = extract_loudness_curve(video_path)
        except Exception as exc:
            reaction_note = f"loudness measure nahi ho saka: {exc}"
    video_duration = None
    if curve is not None:
        video_duration = curve.get("duration")
    elif transcript.get("words"):
        try:
            video_duration = float(transcript["words"][-1]["end"])
        except Exception:
            video_duration = None
    chat_supplied = bool(chat_activity)
    for cand in candidates:
        rj = reaction_judge(cand["start"], cand["end"], curve,
                            chat_activity=chat_activity,
                            video_duration=video_duration)
        cand["reaction_score"] = int(rj["score"])
        cand["reaction_reason"] = rj["reason"]
        cand["reaction_breakdown"] = rj["breakdown"]
        if reaction_note and curve is None:
            cand["reaction_reason"] = (
                f"Reaction {rj['score']}/100 — {reaction_note}; "
                f"neutral score diya gaya hai")
    if curve is None:
        reaction_judge_name = "neutral (no audio)"
    elif chat_supplied:
        reaction_judge_name = "loudness+chat"
    else:
        reaction_judge_name = "loudness"

    # --- COMBINE (CreativeCut rule): BOTH judges must agree. ---
    kept: list[dict] = []
    dropped: list[dict] = []
    for cand in candidates:
        ws, rs = int(cand["words_score"]), int(cand["reaction_score"])
        cand["combined_score"] = int(round(0.6 * ws + 0.4 * rs))
        cand["score"] = cand["combined_score"]  # old field = combined
        cand["judges_agree"] = ws >= words_threshold and rs >= reaction_threshold
        if cand["judges_agree"]:
            kept.append(cand)
        else:
            why = []
            if ws < words_threshold:
                why.append(f"words judge fail ({ws} < {words_threshold})")
            if rs < reaction_threshold:
                why.append(f"reaction judge fail ({rs} < {reaction_threshold})")
            dropped.append({
                "start": cand["start"], "end": cand["end"],
                "duration": cand["duration"],
                "words_score": ws, "reaction_score": rs,
                "combined_score": cand["combined_score"],
                "reason": "; ".join(why),
            })
    picked = _dedupe_rank(kept, top_n)
    clips = []
    for idx, cand in enumerate(picked, start=1):
        hook = cand.get("gemini_hook") or _hook_line(cand["text"])
        clips.append({
            "clip_id": f"clip_{idx:02d}",
            "start": cand["start"], "end": cand["end"],
            "duration": cand["duration"],
            "score": int(cand["score"]),
            "combined_score": int(cand["combined_score"]),
            "words_score": int(cand["words_score"]),
            "words_reason": cand["words_reason"],
            "words_judge": words_judge,
            "reaction_score": int(cand["reaction_score"]),
            "reaction_reason": cand["reaction_reason"],
            "reaction_breakdown": cand.get("reaction_breakdown"),
            "judges_agree": True,
            "reason": (
                f"Words {cand['words_score']}/100; "
                f"Reaction {cand['reaction_score']}/100; "
                f"Combined {cand['combined_score']}/100 — "
                f"{cand['words_reason']}"),
            "hook_line": hook,
            "transcript_text": cand["text"],
            "scoring_method": scoring_method,
            "scoring_language": lang,
            "video_type": video_type,
            "breakdown": cand.get("breakdown"),
        })
    out = {
        "upload_id": upload_id,
        "scoring_method": scoring_method,
        "scoring_language": lang,
        "style_name": style_name,
        "video_type": video_type,
        "options": {
            "clip_count": top_n, "min_sec": min_sec, "max_sec": max_sec,
            "from_sec": from_sec, "to_sec": to_sec,
            "video_type": video_type,
            # Render-time options travel here too, so /render-clip can
            # default to what the Options panel chose (Phase 2).
            "caption_style": (options or {}).get("caption_style", "bounce"),
            "layout": (options or {}).get("layout", "blurred_fill"),
            "quality": (options or {}).get("quality", "720p"),
            "words_threshold": words_threshold,
            "reaction_threshold": reaction_threshold,
        },
        "judges": {
            "words_judge": words_judge,
            "reaction_judge": reaction_judge_name,
            "words_threshold": words_threshold,
            "reaction_threshold": reaction_threshold,
            "combine": ("clip tabhi jab dono judge agree karein "
                        f"(words >= {words_threshold} AND reaction >= "
                        f"{reaction_threshold}); combined = "
                        "round(0.6*words + 0.4*reaction)"),
            "reaction_note": reaction_note,
        },
        "segments_count": segments_count,
        "split_segments_count": split_segments_count,
        "candidates_count": len(candidates),
        "candidates_after_gate": len(kept),
        "dropped": dropped,
        "clips": clips,
    }
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    (CLIPS_DIR / f"{upload_id}_clips.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_clips(upload_id: str) -> dict:
    """Load saved clips JSON; raises FileNotFoundError if absent.

    Phase 5 backfill: clips files written before Phase 4 have no
    scoring_language — default it to "en" (and per-clip too) instead
    of crashing, and persist the backfill to disk.
    """
    path = CLIPS_DIR / f"{upload_id}_clips.json"
    if not path.is_file():
        raise FileNotFoundError(f"Clips nahi mile: {upload_id} — pehle POST /find-clips karo")
    data = json.loads(path.read_text(encoding="utf-8"))
    lang = data.get("scoring_language") or "en"
    changed = False
    if data.get("scoring_language") != lang:
        data["scoring_language"] = lang
        changed = True
    for clip in data.get("clips", []):
        if clip.get("scoring_language") != lang:
            clip["scoring_language"] = lang
            changed = True
        if "scoring_method" not in clip:
            clip["scoring_method"] = data.get("scoring_method", "rule_based")
            changed = True
        # Phase 3 backfill: clips written before dual-judge have no
        # judge fields — fill from the old single score, honestly
        # marked (reaction unknown, not invented).
        if "words_score" not in clip:
            clip["words_score"] = clip.get("score")
            clip["words_reason"] = clip.get("reason", "")
            clip["words_judge"] = clip.get("scoring_method", "rule_based")
            clip["reaction_score"] = None
            clip["reaction_reason"] = (
                "Reaction judge Phase 3 se pehle ki clip hai — "
                "loudness score recorded nahi hai")
            clip["reaction_breakdown"] = None
            clip["combined_score"] = clip.get("score")
            clip["judges_agree"] = None
            changed = True
    if changed:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data
