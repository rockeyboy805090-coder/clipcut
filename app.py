"""ClipCut — Phase 2 application: upload + style + transcribe + clips.

Phase 0 (verified): FastAPI, FFmpeg check, upload intake, SQLite metadata.
Phase 1 (verified): real style_extractor, ffprobe upload validation, style API,
  Style Card UI, default Global Baatien profile.
Phase 2 (new): local Whisper transcription (/transcribe) + viral clip
  finding (/find-clips, /clips) with Gemini-optional rule-based fallback.
Phase 3 (new): real 9:16 editor (/render-clip, /rendered) — reframe,
  zoom punch, colour grade, ASS karaoke captions, yuv420p export.
Phase 4 (new): workflow / history / batch / test-clip approval —
  project status, /approve-clip + /reject-clip, /render-all with an
  approval guard, /batch (background transcribe + find-clips),
  /projects + DELETE /projects/{id}.
Phase 5 (new): download/share — per-clip + whole-project ZIPs,
  caption JSON for copy, share guide; per-upload render namespace.
Phase 6 (not yet): GitHub + Render deploy (needs user's GitHub username).
V2 (16 Nov 2026): YouTube link -> fully edited clips, automatically —
  POST /youtube-job + GET /youtube-job/{id} (background worker in
  youtube_job.py: yt-dlp download -> transcribe -> find-clips ->
  render test clip -> approve -> render-all -> project ZIP).
"""
from __future__ import annotations

import asyncio
import io
import os
import sqlite3
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from auth import (SESSION_COOKIE, STORAGE_QUOTA_BYTES, create_creator,
                  create_session, create_user, delete_all_sessions,
                  delete_creator, delete_session, delete_user,
                  enforce_storage_quota, get_creator, get_user_by_email,
                  get_user_by_id, get_user_by_token, google_auth_url,
                  google_oauth_configured, init_auth_tables, list_creators,
                  storage_usage_bytes, update_user)

from style_extractor import (ensure_default_profile, extract_style,
                             list_styles, load_style, save_profile)
from transcriber import load_transcript, transcribe
from clip_finder import find_clips, load_clips
from editor import find_render_info, load_render_info, render_clip
from options import (CAPTION_STYLES, LAYOUTS, QUALITIES, VIDEO_TYPES,
                     options_summary, parse_options)
from utils import ffmpeg_status, probe_video, slugify
from workflow import (create_batch, delete_project, get_batch,
                      get_clip_approvals, get_project_status,
                      has_approved_clip, init_workflow_tables,
                      is_clip_rendered_for_upload, list_projects,
                      list_rendered_for_upload, run_batch,
                      set_clip_approval)
from youtube_job import (create_youtube_job, get_youtube_job,
                         init_youtube_jobs_table, run_youtube_job,
                         validate_youtube_url, youtube_runtime_diagnostics)

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
THUMB_DIR = OUTPUT_DIR / "thumbnails"
STYLES_DIR = ROOT / "styles"
STATIC_DIR = ROOT / "static"
DB_PATH = ROOT / "clipcut.sqlite3"
TEMPLATE_PATH = ROOT / "templates" / "index.html"
VIDEO_EXTENSIONS = {".mp4", ".mov"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB per file
CHUNK_SIZE = 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    ensure_default_profile(STYLES_DIR)
    status = ffmpeg_status()
    if status["available"]:
        print(f"[startup] FFmpeg OK: {status['version']}", flush=True)
    else:
        print(f"[startup] WARNING: {status['version']}", flush=True)
    yield


app = FastAPI(title="ClipCut", version="0.9.0-phase3", lifespan=lifespan)


def init_db() -> None:
    for d in (UPLOAD_DIR, OUTPUT_DIR, THUMB_DIR, STYLES_DIR, STATIC_DIR,
              OUTPUT_DIR / "rendered"):
        d.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS uploads (
            id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            original_name TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )""")
        # Phase 1 migration: tag uploads with their owner / creator.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(uploads)")}
        if "user_id" not in cols:
            conn.execute("ALTER TABLE uploads ADD COLUMN user_id TEXT")
        if "creator_id" not in cols:
            conn.execute("ALTER TABLE uploads ADD COLUMN creator_id TEXT")
        conn.commit()
    init_auth_tables(DB_PATH)
    init_workflow_tables()
    init_youtube_jobs_table()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not TEMPLATE_PATH.is_file():
        raise HTTPException(status_code=500, detail="UI template is missing")
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> JSONResponse:
    # youtube diagnostics are cached in youtube_job (one yt-dlp
    # --version + one --list-impersonate-targets at first call) —
    # cheap, and they prove on the LIVE instance whether node and
    # browser impersonation are actually available there.
    return JSONResponse({"status": "ok", "phase": 3, "app": "clipcut-phase3",
                         "ffmpeg": ffmpeg_status(),
                         "youtube": youtube_runtime_diagnostics(),
                         "auth": {"google_configured": google_oauth_configured()}})


def _get_upload_row(upload_id: str) -> dict | None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute(
            "SELECT id, role, original_name, stored_path, size_bytes FROM uploads WHERE id=?",
            (upload_id,)).fetchone()
    if not row:
        return None
    return {"id": row[0], "role": row[1], "original_name": row[2],
            "stored_path": row[3], "size_bytes": row[4]}


async def save_upload(file: UploadFile, role: str,
                      user_id: str | None = None,
                      creator_id: str | None = None) -> dict:
    """Save an upload, then validate it with ffprobe.

    Rejects non-video / 0-duration files with a clear 400 error
    'Ye valid video file nahi hai' and deletes the bad file — Phase 0
    accepted a fake 4-byte MP4, Phase 1 does not.
    """
    original_name = Path(file.filename or "video").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"{role}: only MP4 and MOV files are accepted")

    upload_id = uuid.uuid4().hex
    destination = UPLOAD_DIR / f"{upload_id}{suffix}"
    size = 0
    try:
        with destination.open("wb") as target:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"{role}: file exceeds 8 GiB limit")
                target.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail=f"{role}: uploaded file is empty")
        # --- Phase 1 validation (new) ---
        probe = probe_video(destination)
        if not probe["valid"]:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"Ye valid video file nahi hai ({role}): {probe['error']}")
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.execute(
                "INSERT INTO uploads VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (upload_id, role, original_name, str(destination.relative_to(ROOT)), size,
                 datetime.now(timezone.utc).isoformat(), user_id, creator_id),
            )
            conn.commit()
        return {"id": upload_id, "role": role, "filename": original_name,
                "size_bytes": size, "duration_sec": round(probe["duration"], 2),
                "width": probe["width"], "height": probe["height"],
                "has_audio": probe["has_audio"]}
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.post("/upload")
async def upload_videos(
    request: Request,
    long_video: UploadFile = File(...),
    reference_video: UploadFile | None = File(default=None),
    creator_id: str | None = Form(default=None),
) -> dict:
    user = _current_user(request)
    creator = creator_id.strip() if creator_id else None
    if creator and user:
        row = get_creator(DB_PATH, creator)
        if not row or row["user_id"] != user["id"]:
            raise HTTPException(status_code=400, detail="creator not found")
    primary = await save_upload(long_video, "long video",
                                user_id=user["id"] if user else None,
                                creator_id=creator)
    reference = await save_upload(reference_video, "reference video",
                                  user_id=user["id"] if user else None,
                                  creator_id=creator) if reference_video else None
    quota_deleted: list[dict] = []
    if user:
        # Phase 1 storage rule: 100 MB per user — oldest first.
        quota_deleted = enforce_storage_quota(DB_PATH, ROOT, user["id"])
    return {"message": "Upload saved and validated. Style extraction is available at POST /extract-style.",
            "long_video": primary, "reference_video": reference,
            "storage_quota_deleted": quota_deleted}


@app.post("/extract-style")
async def extract_style_endpoint(
    reference_video: UploadFile | None = File(default=None),
    reference_id: str | None = Form(default=None),
    style_name: str = Form(default="Extracted Style"),
) -> dict:
    """Extract a style profile from a reference video.

    Accepts either a fresh upload (reference_video file) or an already
    uploaded reference (reference_id from /upload). Returns the 22-field
    profile + measured analysis. The profile JSON is saved under styles/.
    """
    if reference_video is not None and reference_video.filename:
        saved = await save_upload(reference_video, "reference video")
        video_path = ROOT / _get_upload_row(saved["id"])["stored_path"]
    elif reference_id:
        row = _get_upload_row(reference_id)
        if not row:
            raise HTTPException(status_code=404, detail=f"reference upload not found: {reference_id}")
        video_path = ROOT / row["stored_path"]
        if not video_path.is_file():
            raise HTTPException(status_code=404, detail="reference file missing on disk")
    else:
        raise HTTPException(
            status_code=400,
            detail="reference video file ya reference_id me se ek do")
    try:
        result = extract_style(video_path, style_name=style_name or "Extracted Style",
                               styles_dir=STYLES_DIR, thumb_dir=THUMB_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"message": "Style extracted and saved.",
            "profile": result["profile"], "analysis": result["analysis"]}


@app.get("/styles")
def list_styles_endpoint() -> dict:
    items = list_styles(STYLES_DIR)
    return {"count": len(items),
            "styles": [{"name": i["name"], "slug": i["slug"], "file": i["file"]} for i in items]}


@app.get("/styles/{name}")
def get_style_endpoint(name: str) -> dict:
    try:
        return {"profile": load_style(name, STYLES_DIR)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Style not found: {name}")


@app.post("/styles")
async def save_style_endpoint(profile: dict) -> dict:
    """Save / approve an edited style profile (Style Card Approve)."""
    try:
        out = save_profile(profile, STYLES_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"message": "Style saved.", "file": out.name,
            "slug": slugify(str(profile.get("style_name", "")))}


@app.post("/transcribe")
async def transcribe_endpoint(
    upload_id: str = Form(...),
    model: str | None = Form(default=None),
    initial_prompt: str | None = Form(default=None),
    language: str = Form(default="en"),
) -> dict:
    """Transcribe an uploaded long video with local Whisper.

    DEFAULT LANGUAGE IS ENGLISH (user correction, Phase 4): language
    "en" forces English decoding, "hinglish" = Roman Hinglish,
    "hi" = Hindi. This is a decoding language, NOT a translator —
    Hindi audio forced to "en" will not become clean English; the
    response records language_requested + language_note honestly.
    """
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    video_path = ROOT / row["stored_path"]
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="uploaded file missing on disk")
    try:
        # CPU work runs in a worker thread (Phase 6 fix, 15 Nov 2026):
        # a sync transcribe inside this async endpoint blocked the
        # event loop for minutes — /health hung and Render's proxy
        # cut the client connection while the server was still
        # working. to_thread keeps the loop (and /health) responsive.
        result = await asyncio.to_thread(
            transcribe, video_path, upload_id, model_name=model or None,
            initial_prompt=initial_prompt or None,
            language=language or "en")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "message": "Transcript ready.",
        "upload_id": upload_id,
        "model": result["model"],
        "language": result["language"],
        "language_requested": result.get("language_requested", "en"),
        "language_note": result.get("language_note", ""),
        "transcript_script": result.get("transcript_script", "latin"),
        "duration_sec": result["duration_sec"],
        "word_count": result["word_count"],
        "word_timestamps_real": result["word_timestamps_real"],
        "text_preview": result["text"][:500],
        "transcript_json": result["transcript_json"],
        "transcript_txt": result["transcript_txt"],
    }


@app.get("/transcript/{upload_id}")
def get_transcript_endpoint(upload_id: str) -> dict:
    try:
        return load_transcript(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/find-clips")
async def find_clips_endpoint(
    upload_id: str = Form(...),
    top_n: int | None = Form(default=None),  # None = Options default (2)
    style_name: str | None = Form(default=None),
    language: str | None = Form(default=None),
    # Phase 2 Options panel (all optional — empty = teardown defaults).
    min_sec: str | None = Form(default=None),
    max_sec: str | None = Form(default=None),
    from_min: str | None = Form(default=None),
    to_min: str | None = Form(default=None),
    caption_style: str | None = Form(default=None),
    layout: str | None = Form(default=None),
    quality: str | None = Form(default=None),
    video_type: str | None = Form(default=None),
    # Phase 3 dual-judge thresholds (defaults 50/40 — see options.py).
    words_threshold: str | None = Form(default=None),
    reaction_threshold: str | None = Form(default=None),
) -> dict:
    """Find top viral clips for an upload.

    Runs transcription first if no transcript exists yet (in English
    by default). Scoring is Gemini if GEMINI_API_KEY is set, else
    rule-based — the response always says which one was used
    (scoring_method) and which language bank scored (scoring_language).
    Phase 2: the Options panel (count / length / From-To / type /
    caption style / layout / quality) is validated here and recorded
    on the clips file, so /render-clip can default to it later.
    """
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    video_path = ROOT / row["stored_path"]
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="uploaded file missing on disk")
    try:
        probe = probe_video(video_path)
        duration = probe["duration"] if probe["valid"] else None
    except Exception:
        duration = None
    try:
        opts = parse_options({
            "clip_count": top_n, "min_sec": min_sec, "max_sec": max_sec,
            "from_min": from_min, "to_min": to_min,
            "caption_style": caption_style, "layout": layout,
            "quality": quality, "video_type": video_type,
            "words_threshold": words_threshold,
            "reaction_threshold": reaction_threshold,
        }, video_duration=duration)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        transcript = load_transcript(upload_id)
    except FileNotFoundError:
        try:
            # Same worker-thread rule as /transcribe (see above).
            transcript = await asyncio.to_thread(
                transcribe, video_path, upload_id,
                language=language or "en")
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    if not transcript.get("words"):
        raise HTTPException(
            status_code=400,
            detail="Transcript khaali hai — video me boli hui baat nahi mili, clips nahi ban sakte")
    # Phase 3: pass the real video file so the Reaction judge can
    # measure loudness (without it the reaction score is neutral 50).
    result = find_clips(transcript, upload_id, top_n=opts["clip_count"],
                        style_name=style_name, options=opts,
                        language=language or transcript.get("language_requested") or "en",
                        video_path=video_path)
    return {"message": "Clips ready.", "options": options_summary(opts), **result}


@app.get("/clips/{upload_id}")
def get_clips_endpoint(upload_id: str) -> dict:
    try:
        return load_clips(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/render-clip")
async def render_clip_endpoint(
    upload_id: str = Form(...),
    clip_id: str = Form(...),
    style_name: str = Form(default="global-baatien"),
    # Phase 2: render-time Options. Empty = whatever /find-clips
    # recorded for this upload (its Options panel), else defaults.
    caption_style: str | None = Form(default=None),
    layout: str | None = Form(default=None),
    quality: str | None = Form(default=None),
) -> dict:
    """Render ONE clip to 9:16 with captions + grading.

    Runs in a worker thread (asyncio.to_thread) — a ~47s clip takes
    1-3 min on CPU. Returns render metadata (paths, resolution,
    caption script, reframe mode, etc.).
    """
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    video_path = ROOT / row["stored_path"]
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="uploaded file missing on disk")
    try:
        clips_data = load_clips(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    clip = next((c for c in clips_data.get("clips", []) if c.get("clip_id") == clip_id), None)
    if not clip:
        raise HTTPException(
            status_code=404,
            detail=f"clip not found: {clip_id} — pehle POST /find-clips karo")
    try:
        transcript = load_transcript(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    try:
        # Render options: explicit fields win; otherwise reuse what
        # /find-clips stored; otherwise the teardown defaults.
        stored = dict(clips_data.get("options") or {})
        opts = parse_options({
            "caption_style": caption_style or stored.get("caption_style"),
            "layout": layout or stored.get("layout"),
            "quality": quality or stored.get("quality"),
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        # Worker thread, same reason as /transcribe (see above) —
        # a ~47s clip takes 1-3 min on CPU.
        meta = await asyncio.to_thread(
            render_clip, video_path, upload_id, clip,
            style_name=style_name or "global-baatien",
            transcript=transcript, options=opts)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"message": "Clip rendered.", **meta}


def _render_info(clip_id: str, upload_id: str | None = None) -> dict:
    """Load a render sidecar, namespaced first, legacy fallback."""
    try:
        return load_render_info(clip_id, upload_id=upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/rendered/{upload_id}/{clip_id}")
def download_rendered_namespaced_endpoint(upload_id: str, clip_id: str):
    """Download a rendered MP4 (Phase 5 namespaced path)."""
    info = _render_info(clip_id, upload_id)
    path = Path(info["output_mp4"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="rendered file missing on disk")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/rendered/{upload_id}/{clip_id}/srt")
def download_srt_namespaced_endpoint(upload_id: str, clip_id: str):
    """Download a rendered clip's SRT captions (namespaced)."""
    info = _render_info(clip_id, upload_id)
    path = Path(info["output_srt"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="SRT file missing on disk")
    return FileResponse(path, media_type="text/plain", filename=path.name)


@app.get("/rendered/{upload_id}/{clip_id}/info")
def render_info_namespaced_endpoint(upload_id: str, clip_id: str) -> dict:
    return _render_info(clip_id, upload_id)


@app.get("/rendered/{clip_id}")
def download_rendered_endpoint(clip_id: str, upload_id: str | None = None):
    """Download a rendered MP4 (legacy flat path — still works).

    Pass ?upload_id=<id> to disambiguate when two uploads both
    rendered clip_01 (Phase 5 namespace); without it the first
    sidecar found is returned.
    """
    info = _render_info(clip_id, upload_id)
    path = Path(info["output_mp4"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="rendered file missing on disk")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/rendered/{clip_id}/srt")
def download_srt_endpoint(clip_id: str, upload_id: str | None = None):
    """Download a rendered clip's SRT captions (legacy path)."""
    info = _render_info(clip_id, upload_id)
    path = Path(info["output_srt"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="SRT file missing on disk")
    return FileResponse(path, media_type="text/plain", filename=path.name)


@app.get("/rendered/{clip_id}/info")
def render_info_endpoint(clip_id: str, upload_id: str | None = None) -> dict:
    return _render_info(clip_id, upload_id)


# ---------------------------------------------------------- Phase 5: download / share

def _clip_files(upload_id: str, clip_id: str) -> dict:
    """All on-disk files for one rendered clip (namespaced or legacy)."""
    info = _render_info(clip_id, upload_id)
    files: dict[str, Path] = {}
    for key, label in (("output_mp4", "mp4"), ("output_srt", "srt"),
                       ("output_ass", "ass"), ("output_txt", "txt"),
                       ("output_caption_txt", "caption_txt"),
                       ("output_json", "json")):
        path = Path(str(info.get(key, "")))
        if path.is_file():
            files[label] = path
    if "mp4" not in files:
        raise HTTPException(status_code=404, detail="rendered file missing on disk")
    return {"info": info, "files": files}


def _zip_bytes(entries: list[tuple[str, Path]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in entries:
            zf.write(path, arcname)
    return buf.getvalue()


# NOTE (Phase 5 fix, 14 Nov 2026): this literal "/all" route MUST stay
# BEFORE "/download/{upload_id}/{clip_id}" below — FastAPI matches
# routes in order, and "{clip_id}" would otherwise swallow "all"
# (returned 404 "Rendered clip nahi mila: all" in the fresh E2E test).
@app.get("/download/{upload_id}/all")
def download_all_zip_endpoint(upload_id: str):
    """ZIP of ALL rendered clips of a project (one folder per clip)."""
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    rendered = list_rendered_for_upload(upload_id)
    if not rendered:
        raise HTTPException(status_code=404, detail="Is project ka koi rendered clip nahi hai — pehle render karo")
    arc = {"mp4": ".mp4", "srt": ".srt", "ass": ".ass", "txt": ".txt",
           "caption_txt": "_caption.txt", "json": ".json"}
    entries: list[tuple[str, Path]] = []
    for meta in sorted(rendered, key=lambda m: str(m.get("clip_id"))):
        cid = str(meta.get("clip_id"))
        for key, ext in (("output_mp4", "mp4"), ("output_srt", "srt"),
                         ("output_ass", "ass"), ("output_txt", "txt"),
                         ("output_caption_txt", "caption_txt"),
                         ("output_json", "json")):
            path = Path(str(meta.get(key, "")))
            if path.is_file():
                entries.append((f"{cid}/{cid}{arc[ext]}", path))
    if not entries:
        raise HTTPException(status_code=404, detail="Rendered files disk par nahi mile")
    return _zip_response(_zip_bytes(entries), f"{upload_id}_all_clips.zip")


@app.get("/download/{upload_id}/{clip_id}")
def download_clip_zip_endpoint(upload_id: str, clip_id: str):
    """ZIP: MP4 + SRT + ASS + transcript .txt + caption .txt + style JSON.

    Style JSON is the style profile used for the render (from
    styles/<slug>.json), falling back to the render sidecar JSON.
    """
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    got = _clip_files(upload_id, clip_id)
    info, files = got["info"], got["files"]
    arc = {"mp4": f"{clip_id}.mp4", "srt": f"{clip_id}.srt",
           "ass": f"{clip_id}.ass", "txt": f"{clip_id}.txt",
           "caption_txt": f"{clip_id}_caption.txt", "json": f"{clip_id}.json"}
    entries = [(arc[ext], path) for ext, path in files.items() if ext in arc]
    slug = str(info.get("style_slug") or "global-baatien")
    style_path = STYLES_DIR / f"{slug}.json"
    if style_path.is_file():
        entries.append((f"{clip_id}_style_{slug}.json", style_path))
    data = _zip_bytes(entries)
    return _zip_response(data, f"{upload_id}_{clip_id}.zip")


def _zip_response(data: bytes, filename: str):
    from fastapi.responses import Response
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/caption/{upload_id}/{clip_id}")
def caption_endpoint(upload_id: str, clip_id: str) -> dict:
    """Title + caption + hashtags as JSON, ready for the Copy button.

    Parsed from the clip's _caption.txt (title = first line, hashtags
    = last #-line). Falls back to the clip hook + transcript text.
    Caption text is the transcript text as Whisper returned it —
    English words when the audio was English.
    """
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    got = _clip_files(upload_id, clip_id)
    info, files = got["info"], got["files"]
    title = str(info.get("hook_line") or "Viral clip")
    caption = ""
    hashtags = ""
    cap_path = files.get("caption_txt")
    if cap_path is not None:
        lines = cap_path.read_text(encoding="utf-8").splitlines()
        lines = [ln for ln in lines if ln.strip() != ""]
        if lines:
            title = lines[0].strip()
            tag_lines = [ln for ln in lines if ln.strip().startswith("#")]
            if tag_lines:
                hashtags = tag_lines[-1].strip()
                body = [ln for ln in lines[1:] if not ln.strip().startswith("#")]
            else:
                body = lines[1:]
            caption = "\n".join(body).strip()
    if not caption:
        try:
            clips_data = load_clips(upload_id)
            clip = next((c for c in clips_data.get("clips", []) if c.get("clip_id") == clip_id), None)
            if clip:
                caption = str(clip.get("transcript_text") or "")[:300]
        except FileNotFoundError:
            caption = ""
    if not hashtags:
        hashtags = "#viralreels #reels #shorts #trendingnow"
    return {"upload_id": upload_id, "clip_id": clip_id, "title": title,
            "caption": caption, "hashtags": hashtags,
            "full_text": f"{title}\n\n{caption}\n\n{hashtags}".strip(),
            "language_note": info.get("language_note", "")}


@app.get("/share-guide", response_class=HTMLResponse)
def share_guide_endpoint() -> HTMLResponse:
    """How to post a rendered clip to Instagram / Facebook from phone.

    No Meta API posting in Phase 5 — no API keys are configured, so
    the honest flow is download -> phone -> post. Said plainly here
    and in the README, not hidden.
    """
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Share Guide — ClipCut</title>
<style>body{background:#09090b;color:#f4f4f5;font-family:sans-serif;max-width:640px;margin:40px auto;padding:0 20px;line-height:1.6}</style>
</head><body>
<h1>Instagram / Facebook par kaise post karein</h1>
<p><b>Note:</b> Is app me abhi direct auto-post NAHI hai — Meta API keys
set nahi hain. Download karke phone se post karo, 2 minute ka kaam hai.</p>
<ol>
<li><b>Download karo:</b> History ya Render section se clip ka ZIP download karo (MP4 + SRT + caption txt andar hai), ya sirf MP4 download karo.</li>
<li><b>Phone par bhejo:</b> ZIP/MP4 ko WhatsApp, Telegram, Google Drive ya USB se apne phone me bhej do. ZIP ko phone par unzip karo.</li>
<li><b>Instagram:</b> Instagram app &rarr; + &rarr; Reel &rarr; video chuno &rarr; caption file se title + caption + hashtags copy-paste karo &rarr; Share.</li>
<li><b>Facebook:</b> Facebook app &rarr; + &rarr; Reel &rarr; video chuno &rarr; wahi caption paste karo &rarr; Share.</li>
<li><b>Best time:</b> Shaam 7–9 PM IST par post karo, aur pehle 1 ghante me aaye comments ka reply do.</li>
</ol>
<p>Export specs: 1080x1920 (9:16), 30fps, H.264, yuv420p — Instagram aur Facebook Reels dono ke liye ready.</p>
<p><a href="/" style="color:#a78bfa">← Wapas ClipCut par jao</a></p>
</body></html>"""
    return HTMLResponse(html)


@app.get("/projects")
def list_projects_endpoint() -> dict:
    """All past projects with status + clips + renders + downloads."""
    projects = list_projects()
    return {"count": len(projects), "projects": projects}


@app.get("/projects/{upload_id}")
def get_project_endpoint(upload_id: str) -> dict:
    try:
        return get_project_status(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/projects/{upload_id}")
def delete_project_endpoint(upload_id: str) -> dict:
    """Delete one project and its files (upload, transcript, clips,
    its own renders, approvals). Rendered files of OTHER uploads are
    never touched — delete only removes files whose sidecar upload_id
    matches."""
    try:
        result = delete_project(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"message": "Project deleted.", **result}


@app.post("/approve-clip")
async def approve_clip_endpoint(
    upload_id: str = Form(...),
    clip_id: str = Form(...),
) -> dict:
    """Approve ONE test clip. This unlocks POST /render-all."""
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    try:
        clips_data = load_clips(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if not any(c.get("clip_id") == clip_id for c in clips_data.get("clips", [])):
        raise HTTPException(status_code=404, detail=f"clip not found: {clip_id}")
    result = set_clip_approval(upload_id, clip_id, "approved")
    return {"message": "Test clip approved — ab POST /render-all se baaki clips render kar sakte ho.",
            **result}


@app.post("/reject-clip")
async def reject_clip_endpoint(
    upload_id: str = Form(...),
    clip_id: str = Form(...),
    reason: str = Form(default=""),
) -> dict:
    """Reject a clip (it stays in the list, marked rejected)."""
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    try:
        clips_data = load_clips(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if not any(c.get("clip_id") == clip_id for c in clips_data.get("clips", [])):
        raise HTTPException(status_code=404, detail=f"clip not found: {clip_id}")
    result = set_clip_approval(upload_id, clip_id, "rejected", reason=reason or "")
    return {"message": "Clip rejected.", **result}


@app.post("/render-all")
async def render_all_endpoint(
    upload_id: str = Form(...),
    style_name: str = Form(default="global-baatien"),
    # Phase 2: same render-time Options as /render-clip.
    caption_style: str | None = Form(default=None),
    layout: str | None = Form(default=None),
    quality: str | None = Form(default=None),
) -> dict:
    """Render ALL clips of an upload, sequentially.

    GUARD: works ONLY after at least one test clip was approved
    (POST /approve-clip). Without approval -> 400
    "Pehle 1 test clip approve karo". Already-rendered clips (sidecar
    upload_id matches) are skipped, not re-rendered. Each clip runs
    in a worker thread — 1-3 min on CPU per clip, so a 4-clip video
    takes ~8 min; /health stays responsive throughout.
    """
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    if not has_approved_clip(upload_id):
        raise HTTPException(
            status_code=400,
            detail="Pehle 1 test clip approve karo — POST /approve-clip se ek clip approve karo, phir /render-all chalega")
    video_path = ROOT / row["stored_path"]
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="uploaded file missing on disk")
    try:
        clips_data = load_clips(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    try:
        transcript = load_transcript(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    try:
        stored = dict(clips_data.get("options") or {})
        opts = parse_options({
            "caption_style": caption_style or stored.get("caption_style"),
            "layout": layout or stored.get("layout"),
            "quality": quality or stored.get("quality"),
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    results: list[dict] = []
    approvals = get_clip_approvals(upload_id)
    for clip in clips_data.get("clips", []):
        cid = str(clip.get("clip_id"))
        if approvals.get(cid, {}).get("status") == "rejected":
            results.append({"clip_id": cid, "status": "skipped_rejected",
                            "reason": approvals[cid].get("reason", "")})
            continue
        existing = is_clip_rendered_for_upload(upload_id, cid)
        if existing is not None:
            results.append({"clip_id": cid, "status": "skipped_already_rendered",
                            "output_mp4": existing.get("output_mp4"),
                            "duration_sec": existing.get("duration_sec")})
            continue
        try:
            # Worker thread, same reason as /transcribe (see above).
            meta = await asyncio.to_thread(
                render_clip, video_path, upload_id, clip,
                style_name=style_name or "global-baatien",
                transcript=transcript, options=opts)
            results.append({"clip_id": cid, "status": "rendered",
                            "output_mp4": meta.get("output_mp4"),
                            "duration_sec": meta.get("duration_sec"),
                            "file_size_bytes": meta.get("file_size_bytes")})
        except (FileNotFoundError, ValueError) as exc:
            results.append({"clip_id": cid, "status": "error", "error": str(exc)})
        except RuntimeError as exc:
            results.append({"clip_id": cid, "status": "error", "error": str(exc)})
    rendered_n = sum(1 for r in results if r["status"] == "rendered")
    skipped_n = sum(1 for r in results if r["status"] == "skipped_already_rendered")
    rejected_n = sum(1 for r in results if r["status"] == "skipped_rejected")
    error_n = sum(1 for r in results if r["status"] == "error")
    return {"message": f"Render-all done: {rendered_n} rendered, {skipped_n} skipped (pehle se rendered), {rejected_n} skipped (rejected), {error_n} error.",
            "upload_id": upload_id, "results": results,
            "project": get_project_status(upload_id)}


@app.post("/batch")
async def create_batch_endpoint(
    background_tasks: BackgroundTasks,
    payload: dict,
) -> dict:
    """Queue transcribe + find-clips for several uploads.

    Body: {"upload_ids": [...], "style_name": "global-baatien",
    "language": "en"}. Language default English — batch transcribes
    every video in that language.
    Returns a batch_id at once; the work runs in the background
    (Whisper is slow on CPU). Poll GET /batch/{batch_id} for progress.
    Rendering is NEVER part of a batch — each video still needs a
    test-clip approval first.
    """
    upload_ids = payload.get("upload_ids") or []
    if not isinstance(upload_ids, list) or not upload_ids:
        raise HTTPException(status_code=400, detail="upload_ids list khaali hai — kam se kam 1 upload id do")
    if len(upload_ids) > 10:
        raise HTTPException(status_code=400, detail="Ek batch me max 10 videos")
    for uid in upload_ids:
        if not _get_upload_row(str(uid)):
            raise HTTPException(status_code=404, detail=f"upload not found: {uid}")
    batch = create_batch([str(u) for u in upload_ids],
                         style_name=str(payload.get("style_name") or ""),
                         language=str(payload.get("language") or "en"))
    background_tasks.add_task(run_batch, batch["batch_id"])
    return {"message": "Batch queued — transcribe + find-clips background me chal raha hai. GET /batch/{batch_id} se progress dekho.",
            "batch_id": batch["batch_id"], "total": len(upload_ids),
            "status": "queued"}


@app.get("/batch/{batch_id}")
def get_batch_endpoint(batch_id: str) -> dict:
    try:
        return get_batch(batch_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------- Phase 1: auth + dashboard + creators

AUTH_TEMPLATES = ROOT / "templates"


def _current_user(request: Request) -> dict | None:
    return get_user_by_token(DB_PATH, request.cookies.get(SESSION_COOKIE))


def _require_user(request: Request) -> dict | None:
    """Return the logged-in user, or None (callers redirect to /login)."""
    return _current_user(request)


def _login_response(user: dict) -> RedirectResponse:
    token = create_session(DB_PATH, user["id"])
    resp = RedirectResponse("/app", status_code=303)
    resp.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax",
                    max_age=30 * 24 * 3600, path="/")
    return resp


def _auth_page(template: str, error: str = "") -> HTMLResponse:
    html = (AUTH_TEMPLATES / template).read_text(encoding="utf-8")
    attr = 'style="display:none"' if not error else ""
    return HTMLResponse(html.replace("__ERROR_ATTR__", attr)
                        .replace("__ERROR__", error))


_APP_CSS = """
body{background:#0d1230;color:#f4f4f5;font-family:sans-serif;margin:0}
a{color:#f9a8d4}.wrap{display:flex;min-height:100vh}
.sidebar{width:230px;background:#12173a;padding:20px 14px;flex-shrink:0}
.sidebar h2{margin:0 0 4px;font-size:20px;letter-spacing:.04em}
.sidebar small{color:#a5b4fc}.sidebar nav a{display:block;padding:9px 10px;border-radius:8px;color:#e5e7eb;text-decoration:none;margin-top:4px}
.sidebar nav a.active,.sidebar nav a:hover{background:#1e2450}
.sidebar .meter{margin-top:18px;font-size:13px;color:#c7d2fe}
.sidebar .meterbar{height:8px;background:#1e2450;border-radius:4px;overflow:hidden;margin-top:6px}
.sidebar .meterbar i{display:block;height:100%;background:#ff2e63}
.main{flex:1;padding:28px;max-width:900px}
.card{background:#171c44;border:1px solid #262c5e;border-radius:14px;padding:18px;margin-top:16px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px}
.tile{background:#171c44;border:1px solid #262c5e;border-radius:14px;padding:16px}
.tile b{font-size:26px;display:block}.tile span{color:#a5b4fc;font-size:13px}
.btn{display:inline-block;background:#ff2e63;color:#fff;border:0;border-radius:999px;padding:10px 20px;font-weight:700;cursor:pointer;text-decoration:none}
.btn.ghost{background:#262c5e}
input,select{width:100%;background:#0d1230;border:1px solid #3b4380;border-radius:10px;color:#fff;padding:10px;margin-top:6px;box-sizing:border-box}
label{display:block;margin-top:12px;font-size:14px;color:#c7d2fe}
.bar-row{display:flex;align-items:flex-end;gap:6px;height:90px;margin-top:10px}
.bar{flex:1;background:#ff2e63;border-radius:4px 4px 0 0;min-height:2px}
.bar.posted{background:#f4f4f5}
.muted{color:#a5b4fc;font-size:14px}
h1{margin:0;font-size:28px;text-transform:uppercase;letter-spacing:.02em}
h3{margin:0 0 8px}
table{width:100%;border-collapse:collapse;font-size:14px}
td,th{padding:8px 4px;border-bottom:1px solid #262c5e;text-align:left}
.optrow{display:flex;gap:10px;flex-wrap:wrap}.optrow>div{flex:1;min-width:140px}
.radio label{display:inline-block;margin:6px 14px 0 0;color:#e5e7eb}
.radio input{width:auto;margin-right:4px}
.err{color:#f87171;font-size:14px}
"""


def _app_layout(user: dict, active: str, body: str) -> HTMLResponse:
    used = storage_usage_bytes(DB_PATH, user["id"])
    used_mb = used / (1024 * 1024)
    pct = min(100, used / STORAGE_QUOTA_BYTES * 100) if STORAGE_QUOTA_BYTES else 0
    creators = list_creators(DB_PATH, user["id"])

    def nav(href: str, label: str, key: str) -> str:
        cls = "active" if active == key else ""
        return f'<a class="{cls}" href="{href}">{label}</a>'

    creator_links = "".join(
        f'<a href="/app/videos?creator={c["id"]}">{c["name"]} ({c["platform"]})</a>'
        for c in creators)
    html = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>ClipCut — {active}</title><style>{_APP_CSS}</style></head><body>"
            "<div class=\"wrap\"><aside class=\"sidebar\">"
            "<h2>ClipCut</h2><small>Clip live streams &amp; long videos</small><nav>"
            + nav("/app", "Home", "Home")
            + nav("/app/videos", "All videos", "All videos")
            + "<div style=\"margin-top:14px;color:#a5b4fc;font-size:12px;"
              "text-transform:uppercase;letter-spacing:.1em\">Creators</div>"
            + creator_links
            + nav("/app", "＋ New creator", "New creator")
            + nav("/app/usage", "AI usage", "AI usage")
            + nav("/app/notifications", "Notifications", "Notifications")
            + nav("/app/settings/profile", "Settings", "Settings")
            + "<a href=\"/logout\">Sign out</a></nav>"
            f"<div class=\"meter\">Storage — {used_mb:.1f} of 100 MB"
            f"<div class=\"meterbar\"><i style=\"width:{pct:.1f}%\"></i></div>"
            "<div class=\"muted\" style=\"margin-top:6px\">When your space is full, "
            "your oldest videos are deleted to make room. Download the clips "
            "you want to keep.</div></div>"
            f"<div class=\"meter\">AI Gemini — 0 today<br><span class=\"muted\">"
            f"{user['name']} · {user['email']}</span></div>"
            "</aside><main class=\"main\">" + body + "</main></div></body></html>")
    return HTMLResponse(html)


def _clips_files_for_user(user_id: str) -> list[dict]:
    """Clip entries (from outputs/clips/*_clips.json) belonging to this
    user's uploads, with each file's mtime for the activity chart."""
    from datetime import date as _date
    import json as _json
    with closing(sqlite3.connect(DB_PATH)) as conn:
        owned = {r[0] for r in conn.execute(
            "SELECT id FROM uploads WHERE user_id=?", (user_id,)).fetchall()}
    out: list[dict] = []
    clips_dir = OUTPUT_DIR / "clips"
    if not clips_dir.is_dir():
        return out
    for path in sorted(clips_dir.glob("*_clips.json")):
        upload_id = path.name[: -len("_clips.json")]
        if upload_id not in owned:
            continue
        try:
            data = _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        mtime = _date.fromtimestamp(path.stat().st_mtime)
        for clip in data.get("clips", []):
            out.append({"upload_id": upload_id, "clip_id": clip.get("clip_id"),
                        "date": mtime})
    return out


def _dashboard_data(user: dict) -> dict:
    from datetime import date as _date, timedelta as _td
    today = _date.today()
    with closing(sqlite3.connect(DB_PATH)) as conn:
        videos_today = conn.execute(
            "SELECT COUNT(*) FROM uploads WHERE user_id=? AND role='long video'"
            " AND substr(created_at, 1, 10)=?",
            (user["id"], today.isoformat())).fetchone()[0]
        videos_total = conn.execute(
            "SELECT COUNT(*) FROM uploads WHERE user_id=? AND role='long video'",
            (user["id"],)).fetchone()[0]
    clips = _clips_files_for_user(user["id"])
    clips_today = sum(1 for c in clips if c["date"] == today)
    activity = []
    for i in range(13, -1, -1):
        day = today - _td(days=i)
        n = sum(1 for c in clips if c["date"] == day)
        activity.append({"day": day.strftime("%b %d"), "clips": n, "posted": 0})
    return {"videos_today": videos_today, "videos_total": videos_total,
            "clips_today": clips_today, "clips_total": len(clips),
            "activity": activity}


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request) -> HTMLResponse:
    if _current_user(request):
        return RedirectResponse("/app", status_code=303)
    err = request.query_params.get("error", "")
    msg = ""
    if err == "google-not-configured":
        msg = ("Google sign-in is not configured yet — set GOOGLE_CLIENT_ID / "
               "GOOGLE_CLIENT_SECRET in the environment first, or sign in by email.")
    elif err == "google-failed":
        msg = "Google sign-in failed — please try again or sign in by email."
    return _auth_page("login.html", msg)


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, email: str = Form(...),
                       password: str = Form(...)):
    user = None
    try:
        from auth import authenticate
        user = authenticate(DB_PATH, email, password)
    except Exception:
        user = None
    if not user:
        return _auth_page("login.html", "Galat email ya password — phir se try karo.")
    return _login_response(user)


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request) -> HTMLResponse:
    if _current_user(request):
        return RedirectResponse("/app", status_code=303)
    return _auth_page("signup.html")


@app.post("/signup", response_class=HTMLResponse)
async def signup_submit(request: Request, name: str = Form(...),
                        email: str = Form(...), password: str = Form(...)):
    try:
        user = create_user(DB_PATH, name, email, password)
    except ValueError as exc:
        return _auth_page("signup.html", str(exc))
    return _login_response(user)


@app.get("/logout")
def logout_get(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        delete_session(DB_PATH, token)
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@app.post("/logout")
def logout_post(request: Request):
    return logout_get(request)


@app.get("/auth/google")
def google_login():
    """Google sign-in entry point. Placeholder until credentials exist:
    without GOOGLE_CLIENT_ID / GOOGLE_REDIRECT_URI configured it sends
    the user back to /login with an explanation instead of failing."""
    url = google_auth_url()
    if not url:
        return RedirectResponse("/login?error=google-not-configured",
                                status_code=303)
    return RedirectResponse(url, status_code=303)


@app.get("/auth/google/callback")
def google_callback(request: Request):
    """Google OAuth callback. UNTESTED — no Google credentials exist
    yet, so this path cannot be exercised until the user supplies
    them (see README). With credentials missing it fails cleanly."""
    if not google_oauth_configured():
        return RedirectResponse("/login?error=google-not-configured",
                                status_code=303)
    import json as _json
    import urllib.parse as _up
    import urllib.request as _ur
    code = request.query_params.get("code", "")
    if not code:
        return RedirectResponse("/login?error=google-failed", status_code=303)
    try:
        data = _up.urlencode({
            "code": code,
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI"),
            "grant_type": "authorization_code"}).encode()
        req = _ur.Request("https://oauth2.googleapis.com/token", data=data)
        token = _json.loads(_ur.urlopen(req, timeout=15).read())
        req = _ur.Request(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {token['access_token']}"})
        info = _json.loads(_ur.urlopen(req, timeout=15).read())
        email = str(info.get("email", "")).lower()
        if not email:
            raise ValueError("no email from Google")
        user = get_user_by_email(DB_PATH, email)
        if not user:
            with closing(sqlite3.connect(DB_PATH)) as conn:
                uid = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO users VALUES (?, ?, ?, NULL, 'google', 1, ?)",
                    (uid, str(info.get("name") or email), email,
                     datetime.now(timezone.utc).isoformat()))
                conn.commit()
            user = get_user_by_id(DB_PATH, uid)
        return _login_response(user)
    except Exception:
        return RedirectResponse("/login?error=google-failed", status_code=303)


@app.get("/app", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    data = _dashboard_data(user)
    creators = list_creators(DB_PATH, user["id"])
    max_clips = max([d["clips"] for d in data["activity"]] + [1])
    bars = "".join(
        f'<div class="bar" style="height:{(d["clips"] / max_clips * 88 + 2):.0f}px"'
        f' title="{d["day"]}: {d["clips"]} clips"></div>'
        for d in data["activity"])
    labels = "".join(f"<span style=\"flex:1;font-size:10px;color:#a5b4fc\">"
                     f"{d['day']}</span>" for d in data["activity"])
    creator_options = "".join(
        f'<option value="{c["id"]}">{c["name"]}</option>' for c in creators)
    if creators:
        creator_rows = "".join(
            f"<tr><td>{c['name']}</td><td>{c['platform']}</td>"
            f"<td>{c['channel_link']}</td>"
            f"<td><a href=\"/app/videos?creator={c['id']}\">Videos</a> · "
            f"<form method=\"post\" action=\"/app/creators/{c['id']}/delete\""
            f" style=\"display:inline\" onsubmit=\"return confirm("
            f"'Delete this creator? Videos filed under it stay, unfiled.');\">"
            f"<button type=\"submit\" style=\"background:none;border:0;color:"
            f"#f87171;cursor:pointer;padding:0\">Delete</button></form></td></tr>"
            for c in creators)
        creators_html = ("<table><tr><th>Name</th><th>Platform</th><th>Channel</th>"
                         "<th></th></tr>" + creator_rows + "</table>")
    else:
        creators_html = ("<p class=\"muted\">No creators yet — Add one to give each "
                         "streamer their own workspace.</p>")
    step1 = "✅" if creators else "1"
    step2 = "✅" if data["videos_total"] else "2"
    type_options = "".join(
        f'<option value="{v}">{label}</option>' for v, label in VIDEO_TYPES)
    caption_radios = "".join(
        f'<label><input type="radio" name="opt-caption" value="{k}"'
        f'{" checked" if k == "bounce" else ""}> {v["label"]}'
        f' <span class="muted">— {v["description"]}</span></label>'
        for k, v in CAPTION_STYLES.items())
    layout_radios = "".join(
        f'<label><input type="radio" name="opt-layout" value="{k}"'
        f'{" checked" if k == "blurred_fill" else ""}> {v["label"]}'
        f' <span class="muted">— {v["description"]}</span></label>'
        for k, v in LAYOUTS.items())
    quality_radios = "".join(
        f'<label><input type="radio" name="opt-quality" value="{k}"'
        f'{" checked" if k == "720p" else ""}> {v["label"]}'
        f' <span class="muted">— {v["note"]}</span></label>'
        for k, v in QUALITIES.items())
    body = f"""
<h1>Home</h1>
<p class="muted">Good afternoon. {len(creators)} creator{'s' if len(creators) != 1 else ''}.</p>
<div class="card"><h3>What do you want to clip today?</h3>
<p class="muted">Paste a link to a video. You get captioned shorts back.</p>
<form id="yt-form"><label>Video link
<input id="yt-url" type="url" placeholder="Paste a YouTube, Twitch or Kick video link" required></label>
<div style="display:flex;gap:10px"><label style="flex:1">File under creator
<select id="yt-creator"><option value="">No creator</option>{creator_options}</select></label>
<label style="flex:1">Type of video
<select id="yt-type">{type_options}</select></label></div>
<details open style="margin-top:14px"><summary style="cursor:pointer;font-weight:700">Options</summary>
<div class="optrow"><div><label>How many clips should the AI make?
<input id="opt-count" type="number" min="1" max="8" step="1" value="2"></label>
<p class="muted">The AI scores every moment, and keeps this many.</p></div>
<div><label>Shortest (seconds)
<input id="opt-min" type="number" min="5" max="120" step="1" value="15"></label>
<label>Longest (seconds)
<input id="opt-max" type="number" min="5" max="120" step="1" value="60"></label>
<p class="muted">The AI can pick any length in between.</p></div></div>
<div class="radio"><label style="display:block">Caption style</label>{caption_radios}</div>
<div class="radio"><label style="display:block">What should the frame look like?</label>{layout_radios}</div>
<div class="radio"><label style="display:block">Video quality</label>{quality_radios}</div>
<div class="optrow"><div><label>Only look at part of the video — From (minute)
<input id="opt-from" type="number" min="0" step="0.5" placeholder="Up to minute"></label></div>
<div><label>To (minute)
<input id="opt-to" type="number" min="0" step="0.5" placeholder="Up to minute"></label></div></div>
</details>
<p id="yt-error" class="err"></p>
<button class="btn" type="submit">Make clips</button></form>
<p id="yt-status" class="muted"></p></div>
<div class="tiles">
<div class="tile"><b>0</b><span>Recording now</span></div>
<div class="tile"><b>{data['clips_today']}</b><span>Clips today</span></div>
<div class="tile"><b>0</b><span>Posted today</span></div>
<div class="tile"><b>0</b><span>Waiting to post</span></div></div>
<div class="card"><h3>Activity — Last 14 days</h3>
<p class="muted">{data['clips_total']} Clips made · 0 Posted · 0 Views · $0.00 Est. earnings</p>
<div class="bar-row">{bars}</div>
<div style="display:flex;gap:6px">{labels}</div>
<p class="muted">Pink = Clips made · White = Posted</p></div>
<div class="card"><h3>AI — Gemini</h3>
<p class="muted">0 AI calls today · 0 Tokens today</p>
<p class="muted">Per-call token ledger Phase 6 me aayega.</p>
<a class="btn ghost" href="/app/usage">View usage</a></div>
<div class="card"><h3>Next steps — Finish setting up ClipCut</h3>
<p>{step1} Add your first creator — Give each streamer their own workspace, settings and folder.</p>
<p>{step2} Make your first clips — Paste a video link in the box above.</p>
<p>3 Connect a posting account — YouTube, Instagram or TikTok, in one click. (Phase 8)</p></div>
<div class="card"><h3>Creators — Each creator's status and results</h3>{creators_html}
<form method="post" action="/app/creators"><h3 style="margin-top:16px">Add a creator</h3>
<p class="muted">ClipCut can record this streamer live and keep the clips organised by them. (Live recording — Phase 7)</p>
<label>Channel link<input name="channel_link" type="url" placeholder="https://www.youtube.com/@name, twitch.tv/name or kick.com/name" required></label>
<label>Name<input name="name" type="text" placeholder="Filled in from the channel, or type one."></label>
<button class="btn" type="submit">Create workspace</button></form></div>
<div class="card"><h3>Post queue</h3>
<p class="muted">Queued clips go out at their scheduled time.</p>
<p class="muted">Nothing queued or posted yet.</p>
<button class="btn ghost" type="button" onclick="alert('Nothing queued — Pause all posting needs a queue first (Phase 8).')">Pause all posting</button></div>
<script>
document.getElementById('yt-form').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const url = document.getElementById('yt-url').value;
  const st = document.getElementById('yt-status');
  const er = document.getElementById('yt-error');
  er.textContent = '';
  const val = (id) => document.getElementById(id).value;
  const radio = (name) => (document.querySelector('input[name="' + name + '"]:checked') || {{}}).value;
  const fromMin = val('opt-from') === '' ? null : Number(val('opt-from'));
  const toMin = val('opt-to') === '' ? null : Number(val('opt-to'));
  if (fromMin !== null && toMin !== null && fromMin >= toMin) {{
    er.textContent = 'From (minute) To (minute) se chhota hona chahiye.';
    return;
  }}
  const options = {{
    clip_count: Number(val('opt-count')) || 2,
    min_sec: Number(val('opt-min')) || 15,
    max_sec: Number(val('opt-max')) || 60,
    caption_style: radio('opt-caption') || 'bounce',
    layout: radio('opt-layout') || 'blurred_fill',
    quality: radio('opt-quality') || '720p',
    video_type: val('yt-type') || 'auto',
    from_min: fromMin, to_min: toMin,
  }};
  st.textContent = 'Job start ho raha hai…';
  try {{
    const r = await fetch('/youtube-job', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{url, language:'en', creator_id: val('yt-creator') || null, options}})}});
    const j = await r.json();
    if (!r.ok) {{ er.textContent = 'Error: ' + (j.detail || 'job start nahi hua'); st.textContent = ''; return; }}
    st.textContent = 'Job queued: ' + j.job_id + ' — status dekhne ke liye /youtube-job/' + j.job_id + ' kholo.';
  }} catch (err) {{ st.textContent = 'Error: ' + err; }}
}});
</script>
"""
    return _app_layout(user, "Home", body)


@app.get("/app/videos", response_class=HTMLResponse)
def all_videos(request: Request, creator: str | None = None) -> HTMLResponse:
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    creators = list_creators(DB_PATH, user["id"])
    projects = list_projects()
    with closing(sqlite3.connect(DB_PATH)) as conn:
        owner = {r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT id, user_id, creator_id FROM uploads").fetchall()}
    rows = []
    for p in projects:
        uid, cid = owner.get(p["id"], (None, None))
        if uid != user["id"]:
            continue
        if creator and cid != creator:
            continue
        rows.append(p)
    if rows:
        body_rows = "".join(
            f"<tr><td>{p.get('id','')[:8]}…</td><td>{p.get('status','')}</td>"
            f"<td>{len(p.get('clips', []))}</td></tr>" for p in rows)
        table = ("<table><tr><th>Video</th><th>Status</th><th>Clips</th></tr>"
                 + body_rows + "</table>")
    else:
        table = "<p class=\"muted\">No videos yet — Add a video above to see it here.</p>"
    creator_name = ""
    if creator:
        c = get_creator(DB_PATH, creator)
        creator_name = f" — {c['name']}" if c else ""
    body = (f"<h1>All videos{creator_name}</h1>"
            f"<p class=\"muted\">Workspace filter: "
            f"<a href=\"/app/videos\">All</a> · " + " · ".join(
                f"<a href=\"/app/videos?creator={c['id']}\">{c['name']}</a>"
                for c in creators) + "</p>"
            f"<div class=\"card\">{table}</div>"
            f"<p><a class=\"btn\" href=\"/app\">＋ Add a video</a> "
            f"<a class=\"btn ghost\" href=\"/\">Open classic studio</a></p>")
    return _app_layout(user, "All videos", body)


@app.get("/app/usage", response_class=HTMLResponse)
def ai_usage(request: Request) -> HTMLResponse:
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    body = """<h1>AI usage</h1>
<p class="muted">Every AI call ClipCut makes, and the tokens it uses. Last 30 days.</p>
<div class="tiles"><div class="tile"><b>0</b><span>AI calls</span></div>
<div class="tile"><b>0</b><span>Tokens</span></div>
<div class="tile"><b>0</b><span>Input tokens</span></div>
<div class="tile"><b>0</b><span>Output tokens</span></div></div>
<div class="card"><h3>Recent AI activity</h3>
<p class="muted">No AI usage in this period yet.</p>
<p class="muted">Per-call ledger (finding clips / words judge / reaction judge /
title+hook, with creator, video and tokens) Phase 6 me aayega.</p></div>"""
    return _app_layout(user, "AI usage", body)


@app.get("/app/notifications", response_class=HTMLResponse)
def notifications(request: Request) -> HTMLResponse:
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    body = """<h1>Notifications</h1><div class="card">
<p class="muted">Nothing yet. Recordings, finished clips and posts that need
a look show up here.</p></div>"""
    return _app_layout(user, "Notifications", body)


@app.get("/app/settings/profile", response_class=HTMLResponse)
def profile_page(request: Request) -> HTMLResponse:
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    body = f"""<h1>Profile</h1><div class="card">
<p class="muted">Account type: Standard · Member since: {user['created_at'][:10]} ·
Sign-in: {user['auth_provider']} · Email: {user['email']} (Confirmed)</p>
<form method="post" action="/app/settings/profile">
<label>Your name<input name="name" value="{user['name']}" required></label>
<label>Email address<input value="{user['email']}" disabled></label>
<p class="muted">Profile picture — taken from Google (Google accounts only).</p>
<button class="btn" type="submit">Save changes</button></form></div>"""
    return _app_layout(user, "Settings", body)


@app.post("/app/settings/profile")
def profile_save(request: Request, name: str = Form(...)):
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    update_user(DB_PATH, user["id"], name=name)
    return RedirectResponse("/app/settings/profile", status_code=303)


@app.get("/app/settings/security", response_class=HTMLResponse)
def security_page(request: Request) -> HTMLResponse:
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    body = """<h1>Security</h1>
<div class="card"><h3>Set a password</h3>
<p class="muted">Add a password so you can sign in by email too.</p>
<form method="post" action="/app/settings/security/password">
<label>New password (min 8 characters)<input name="password" type="password" required></label>
<button class="btn" type="submit">Set password</button></form></div>
<div class="card"><h3>Sign out of other devices</h3>
<form method="post" action="/app/settings/security/signout-all">
<button class="btn ghost" type="submit">Sign out everywhere</button></form></div>
<div class="card"><h3>Delete your account</h3>
<p class="muted">Your account and everything in it is deleted. This cannot be undone.</p>
<form method="post" action="/app/settings/security/delete-account"
onsubmit="return confirm('Pakka? Account + creators + tumhare uploads delete ho jayenge.');">
<button class="btn" type="submit" style="background:#b91c1c">Delete your account</button></form></div>"""
    return _app_layout(user, "Settings", body)


@app.post("/app/settings/security/password")
def set_password(request: Request, password: str = Form(...)):
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    try:
        update_user(DB_PATH, user["id"], password=password)
    except ValueError:
        return RedirectResponse("/app/settings/security", status_code=303)
    return RedirectResponse("/app/settings/security", status_code=303)


@app.post("/app/settings/security/signout-all")
def signout_all(request: Request):
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    delete_all_sessions(DB_PATH, user["id"])
    return _login_response(user)


@app.post("/app/settings/security/delete-account")
def delete_account(request: Request):
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    delete_user(DB_PATH, user["id"], ROOT)
    resp = RedirectResponse("/signup", status_code=303)
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@app.get("/app/settings/apps", response_class=HTMLResponse)
def connected_apps(request: Request) -> HTMLResponse:
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    body = """<h1>Connected apps</h1><div class="card">
<p><b>YouTube</b> — Post Shorts from here. <span class="muted">Connect — Phase 8 me aayega.</span></p>
<p><b>Instagram</b> — Post Reels from here. <span class="muted">Not set up on this server.</span></p>
<p><b>TikTok</b> — Post to TikTok from here. <span class="muted">Not set up on this server.</span></p></div>"""
    return _app_layout(user, "Settings", body)


@app.get("/app/settings/ai", response_class=HTMLResponse)
def ai_settings(request: Request) -> HTMLResponse:
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    body = """<h1>AI settings</h1><div class="card">
<p class="muted">The AI that finds the moments and writes the hooks. Bring your own key. Nothing here is shared.</p>
<p>Status: <b>Ready</b> (Gemini — free key from environment, Phase 2 rescore only)</p>
<p class="muted">Provider choice (Gemini / OpenAI / Grok / Groq / OpenRouter / Claude / Ollama / Custom),
model picker and per-account key storage — Phase 6 me aayega. Keys kabhi chat ya file me mat likhna.</p></div>"""
    return _app_layout(user, "Settings", body)


@app.post("/app/creators")
def create_creator_endpoint(request: Request, name: str = Form(default=""),
                            channel_link: str = Form(...)):
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    try:
        create_creator(DB_PATH, user["id"], name, channel_link)
    except ValueError:
        return RedirectResponse("/app", status_code=303)
    return RedirectResponse("/app", status_code=303)


@app.post("/app/creators/{creator_id}/delete")
def delete_creator_endpoint(request: Request, creator_id: str):
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    delete_creator(DB_PATH, user["id"], creator_id)
    return RedirectResponse("/app", status_code=303)


# ---------------------------------------------------------- V2: YouTube job

@app.post("/youtube-job")
async def create_youtube_job_endpoint(request: Request, payload: dict) -> dict:
    """Start a full YouTube -> clips job. Returns at once; poll status.

    Body: {"url": "<youtube link>", "language": "en", "top_n": 8,
    "creator_id": "<id>", "options": {clip_count, min_sec, max_sec,
    caption_style, layout, quality, video_type, from_sec/to_sec or
    from_min/to_min}}. Language default English; top_n 1-8 (default 8
    when no options are given, else options["clip_count"]). Only
    youtube.com / youtu.be links are accepted — anything else is 400.
    Bad options are rejected with a clear message before any job is
    queued. The worker (daemon thread) downloads with yt-dlp,
    transcribes, finds clips, renders a test clip, approves it,
    renders the rest, and leaves the project ZIP ready. Poll
    GET /youtube-job/{job_id}.
    """
    url = str(payload.get("url") or "")
    try:
        clean_url = validate_youtube_url(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    language = str(payload.get("language") or "en")
    raw_options = payload.get("options") or {}
    if not isinstance(raw_options, dict):
        raise HTTPException(status_code=400, detail="options ek object hona chahiye")
    try:
        opts = parse_options(raw_options)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # top_n: explicit payload value wins over options only when the
    # caller sent one; otherwise the Options panel's count rules.
    top_n = payload.get("top_n")
    if top_n is None:
        top_n = opts["clip_count"]
    try:
        top_n = int(top_n)
    except (TypeError, ValueError):
        top_n = opts["clip_count"]
    creator_id = payload.get("creator_id") or None
    user = _current_user(request)
    if creator_id:
        creator = get_creator(DB_PATH, str(creator_id))
        if not creator or (user and creator["user_id"] != user["id"]):
            raise HTTPException(status_code=400, detail="Creator nahi mila — dobara choose karo")
    job = create_youtube_job(clean_url, language=language, top_n=top_n,
                             options=opts,
                             user_id=user["id"] if user else None,
                             creator_id=str(creator_id) if creator_id else None)
    thread = threading.Thread(target=run_youtube_job, args=(job["job_id"],),
                              daemon=True)
    thread.start()
    return {"job_id": job["job_id"], "status": "queued", "options": opts,
            "message": "YouTube job queued — download se lekar render tak "
                       "background me chalega. GET /youtube-job/{job_id} se "
                       "progress dekho."}


@app.get("/youtube-job/{job_id}")
def get_youtube_job_endpoint(job_id: str) -> dict:
    """Job status + progress + (when done) clips and ZIP link."""
    try:
        return get_youtube_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


def run() -> None:
    host = os.getenv("CLIPCUT_HOST", os.getenv("VCS_HOST", "127.0.0.1"))
    port = int(os.getenv("CLIPCUT_PORT", os.getenv("VCS_PORT", "8000")))
    uvicorn.run("app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
