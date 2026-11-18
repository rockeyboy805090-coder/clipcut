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
Phase 7 (new): live recorder — browser MediaRecorder capture in ~60s
  chunks (POST /live/start, /live/{id}/chunk, /live/{id}/stop), each
  chunk transcribed + dual-judge scored + rendered while live; channel
  live-status check via yt-dlp (POST /live/check); URL capture via
  yt-dlp into kill-tolerant .mkv (POST /live/start-from-url). Single
  worker thread, one chunk at a time — Render Free 512 MB rule.
V2 (16 Nov 2026): YouTube link -> fully edited clips, automatically —
  POST /youtube-job + GET /youtube-job/{id} (background worker in
  youtube_job.py: yt-dlp download -> transcribe -> find-clips ->
  render test clip -> approve -> render-all -> project ZIP).
Phase 9 (17 Nov 2026): upgrades — dashboard file upload (/app/upload,
  link-free, owner-attached, 100 MB quota), Hindi/Hinglish captions
  (caption_language en/hi/hinglish threaded transcribe -> find-clips
  -> render, per-clip re-transcribe), reference-style extractor with
  MEASURED caption colours + strict "style could not be read" failure,
  face-tracking crop layout (face_track.py, 1 Hz Haar track,
  time-varying ffmpeg crop), test-clip approval gate extended
  (render-all login+ownership, gate state in project status +
  video/clip pages).
Phase 10 (18 Nov 2026): polish + launch — public landing page (/,
  honest limits included), classic studio moved to /studio, in-app
  product tour (first-run checklist on the dashboard, live done-states,
  dismissable), real Notifications page (clips needing a decision +
  queue problems), Terms/Privacy/Acceptable Use pages, render.yaml
  verified for Render Free (healthCheckPath /health, CLIPCUT_KEY_SECRET
  via generateValue, optional OAuth keys via sync:false), repo-root
  render.yaml (rootDir: app) so a Blueprint deploy works from the repo.
"""
from __future__ import annotations

import asyncio
import html as _html
import io
import os
import sqlite3
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote as _quote

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from ai_ledger import init_ai_tables, log_ai_call, today_counts, usage_summary
from ai_providers import (PROVIDERS, clear_ai_key, get_ai_settings,
                          get_effective_ai_config, list_models_for_provider,
                          save_ai_settings)
from auth import (SESSION_COOKIE, STORAGE_QUOTA_BYTES, create_creator,
                  create_session, create_user, delete_all_sessions,
                  delete_creator, delete_session, delete_user,
                  enforce_storage_quota, get_creator, get_user_by_email,
                  get_user_by_id, get_user_by_token, google_auth_url,
                  google_oauth_configured, init_auth_tables, list_creators,
                  storage_usage_bytes, update_user)

from style_extractor import (ensure_default_profile, extract_style,
                             list_styles, load_style, save_profile)
from transcriber import (load_transcript, normalize_language,
                         normalize_language_strict, transcribe)
from clip_finder import (create_hand_clip, find_clips, load_clips,
                         update_clip)
from editor import (find_render_info, list_render_versions,
                    load_render_info, render_clip)
from options import (CAPTION_LANGUAGES, CAPTION_STYLES, LAYOUTS, QUALITIES, VIDEO_TYPES,
                     apply_caption_preset, options_summary, parse_options,
                     render_size)
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
from live_recorder import (check_channel_live, count_live_sessions,
                           create_live_session, ensure_live_worker,
                           get_creator_live_status, get_live_session,
                           init_live_tables, list_live_sessions,
                           live_worker_status, record_channel_check,
                           reconcile_live_state, register_chunk,
                           session_public, set_live_db_path,
                           start_url_session, stop_live_session)
from post_queue import (PLATFORMS, cancel_item, delete_youtube_account,
                        exchange_youtube_code, format_ist, format_ist_input,
                        get_queue_item, get_valid_access_token,
                        get_youtube_account, get_youtube_channel,
                        is_paused_all, list_attempts, list_queue,
                        parse_ist_datetime_local, pause_item,
                        post_worker_status, posted_per_day, queue_clip,
                        queue_counts, reconcile_post_state, resume_item,
                        retry_item, save_youtube_account, set_pause_all,
                        set_post_db_path, ensure_post_worker, init_post_tables,
                        youtube_auth_url, youtube_oauth_configured,
                        youtube_oauth_missing, consume_oauth_state)

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
    # Phase 7: live recorder tables + the single chunk worker thread
    # (one chunk at a time — Render Free 512 MB rule).
    set_live_db_path(DB_PATH)
    reconcile_live_state(DB_PATH)
    ensure_live_worker()
    # Phase 8: posting queue tables + the single posting worker thread
    # (one post at a time — same RAM rule as the live recorder).
    init_post_tables(DB_PATH)
    set_post_db_path(DB_PATH)
    reconcile_post_state(DB_PATH)
    ensure_post_worker()
    status = ffmpeg_status()
    if status["available"]:
        print(f"[startup] FFmpeg OK: {status['version']}", flush=True)
    else:
        print(f"[startup] WARNING: {status['version']}", flush=True)
    yield


app = FastAPI(title="ClipCut", version="0.15.0-phase10", lifespan=lifespan)


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
    init_ai_tables(DB_PATH)
    init_workflow_tables()
    init_youtube_jobs_table()
    init_live_tables(DB_PATH)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    # Phase 10: public landing page. The classic studio UI lives at
    # /studio now — same template, same API routes, nothing removed.
    path = ROOT / "templates" / "landing.html"
    if not path.is_file():
        raise HTTPException(status_code=500, detail="Landing page is missing")
    html = path.read_text(encoding="utf-8")
    if _current_user(request):
        # Logged-in visitors go straight to the dashboard, not signup.
        html = html.replace('href="/signup"', 'href="/app"')
        html = html.replace("Create free account", "Open your dashboard")
        html = html.replace("Get clips", "Open your dashboard")
    return HTMLResponse(html)


@app.get("/studio", response_class=HTMLResponse)
def studio() -> HTMLResponse:
    if not TEMPLATE_PATH.is_file():
        raise HTTPException(status_code=500, detail="UI template is missing")
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))


def _legal_page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{title} — ClipCut</title><style>"
        "body{background:#0d1230;color:#f4f4f5;font-family:sans-serif;"
        "max-width:680px;margin:40px auto;padding:0 20px;line-height:1.6}"
        "a{color:#f9a8d4}h1{text-transform:uppercase}</style></head><body>"
        f"<h1>{title}</h1>{body}"
        "<p><a href=\"/\">&larr; Back to ClipCut</a> &middot; "
        "<a href=\"/terms\">Terms</a> &middot; <a href=\"/privacy\">Privacy</a> "
        "&middot; <a href=\"/acceptable-use\">Acceptable Use</a></p>"
        "</body></html>")


@app.get("/terms", response_class=HTMLResponse)
def terms_page() -> HTMLResponse:
    return _legal_page("Terms", """
<p><b>ClipCut</b> is a free, open-source tool for turning long videos and
live streams into captioned short clips. The name "ClipCut" is a working
placeholder, not the final brand.</p>
<ul>
<li>You must be 13 or older. One person per account.</li>
<li>The service is provided as-is, with no uptime or results guarantee.
Free hosting sleeps after 15 minutes idle and its filesystem is
ephemeral — uploads and data can be wiped on restart. Download the
clips you want to keep.</li>
<li>You are responsible for the videos you upload or link, and for
respecting the rights of the people in them. Do not upload content
you do not have the right to use.</li>
<li>Posting to YouTube uses <i>your</i> Google OAuth client and
<i>your</i> account — posts are your responsibility. Nothing is ever
posted without you queueing it first.</li>
<li>AI features use <i>your</i> API key (bring your own key). Key usage
and any provider charges are yours.</li>
<li>We may change or stop the service at any time.</li>
</ul>""")


@app.get("/privacy", response_class=HTMLResponse)
def privacy_page() -> HTMLResponse:
    return _legal_page("Privacy", """
<p>ClipCut collects as little as possible:</p>
<ul>
<li><b>Account:</b> your name, email and a scrypt-hashed password (or
Google sign-in, when configured). No advertising or analytics cookies
— only one essential, httpOnly sign-in cookie.</li>
<li><b>Your videos, transcripts, clips and renders</b> live on the
server you run this on. On the free Render tier that storage is
ephemeral and can be wiped on restart.</li>
<li><b>Your AI keys</b> are encrypted at rest (Fernet, keyed by the
server secret) and are never shown in full — only "Saved, ends in …".
They are never shared with anyone.</li>
<li><b>Posting tokens</b> (YouTube OAuth) are encrypted the same way
and used only to post clips you queued.</li>
<li>Nothing you upload is sold, shared, or used to train models.
Transcripts and clips are sent to your chosen AI provider only when
you set a key and run clip-finding — that provider's own privacy
policy then applies to that text.</li>
<li>Delete your account (Settings → Security) to delete your user,
sessions, creators and owned uploads.</li>
</ul>""")


@app.get("/acceptable-use", response_class=HTMLResponse)
def acceptable_use_page() -> HTMLResponse:
    return _legal_page("Acceptable Use", """
<p>Do not use ClipCut to:</p>
<ul>
<li>Upload or clip content you do not have the right to use, or that
identifies private people without their consent.</li>
<li>Impersonate someone, mislead viewers about who made a clip, or
pass off someone else's work as your own.</li>
<li>Post spam, scams, hate, or anything illegal through the posting
queue — queued posts are your responsibility.</li>
<li>Abuse the service (excessive uploads, attempts to break quotas,
auth, or other users' data) or use it to attack YouTube, Twitch,
Kick, or any AI provider.</li>
<li>Share your account or your API keys with others through the app.</li>
</ul>
<p>Breaking these rules can get your account deleted. If you see abuse,
contact the person running the server.</p>""")


@app.get("/health")
def health() -> JSONResponse:
    # youtube diagnostics are cached in youtube_job (one yt-dlp
    # --version + one --list-impersonate-targets at first call) —
    # cheap, and they prove on the LIVE instance whether node and
    # browser impersonation are actually available there.
    return JSONResponse({"status": "ok", "phase": 10, "app": "clipcut-phase10",
                         "ffmpeg": ffmpeg_status(),
                         "youtube": youtube_runtime_diagnostics(),
                         "live": live_worker_status(),
                         "posting": post_worker_status(),
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


@app.post("/app/upload")
async def app_upload_video(
    request: Request,
    long_video: UploadFile = File(...),
    creator_id: str | None = Form(default=None),
) -> RedirectResponse:
    """Phase 9 (upgrade 1): dashboard file upload — the link-free path.

    The reference app has NO file upload anywhere; this is the
    upgrade. Logged-in users only (401 without login), owner-attached,
    same 100 MB per-user quota as /upload (oldest first), then the
    SAME pipeline: video page -> transcribe -> find-clips -> editor
    -> post queue. A single file over the 100 MB quota is rejected
    (413) instead of silently deleting the user's other videos.
    """
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    creator = creator_id.strip() if creator_id else None
    if creator:
        row = get_creator(DB_PATH, creator)
        if not row or row["user_id"] != user["id"]:
            return RedirectResponse("/app?error=creator+not+found", status_code=303)
    try:
        saved = await save_upload(long_video, "long video",
                                  user_id=user["id"], creator_id=creator)
    except HTTPException as exc:
        return RedirectResponse(f"/app?error={_quote(str(exc.detail))}",
                                status_code=303)
    # 100 MB quota: a file bigger than the whole quota can never fit.
    if saved["size_bytes"] > STORAGE_QUOTA_BYTES:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.execute("DELETE FROM uploads WHERE id=?", (saved["id"],))
            conn.commit()
        suffix = Path(saved["filename"]).suffix.lower()
        (UPLOAD_DIR / f"{saved['id']}{suffix}").unlink(missing_ok=True)
        return RedirectResponse(
            "/app?error=file+100+MB+quota+se+bada+hai", status_code=303)
    enforce_storage_quota(DB_PATH, ROOT, user["id"])
    return RedirectResponse(f"/app/videos/{saved['id']}", status_code=303)


@app.post("/app/extract-style")
async def app_extract_style(
    request: Request,
    reference_video: UploadFile = File(...),
    style_name: str = Form(default="Extracted Style"),
) -> RedirectResponse:
    """Phase 9 (upgrade 3): reference-style extractor, dashboard flow.

    Logged-in users only. STRICT caption check: if the reference has
    no readable caption text, nothing is invented — the user is sent
    back with "style could not be read". On success the saved preset
    appears in the styles list and can be picked in the clip editor.
    """
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not reference_video or not reference_video.filename:
        return RedirectResponse("/app?error=reference+video+file+do",
                                status_code=303)
    try:
        saved = await save_upload(reference_video, "reference video",
                                  user_id=user["id"])
        video_path = ROOT / _get_upload_row(saved["id"])["stored_path"]
        result = extract_style(video_path,
                               style_name=style_name or "Extracted Style",
                               styles_dir=STYLES_DIR, thumb_dir=THUMB_DIR,
                               require_captions=True)
    except HTTPException as exc:
        return RedirectResponse(f"/app?error={_quote(str(exc.detail))}",
                                status_code=303)
    except ValueError as exc:
        return RedirectResponse(f"/app?error={_quote(str(exc))}",
                                status_code=303)
    from utils import slugify as _slugify
    slug = _slugify(str(result["profile"].get("style_name", "")))
    return RedirectResponse(f"/app?extracted={slug}", status_code=303)


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
    # Phase 9 alias: the Options panel sends caption_language.
    caption_language: str | None = Form(default=None),
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
        # Phase 9: unknown caption_language is a 400 here, not a
        # silent English fallback (see normalize_language_strict).
        eff_language = (normalize_language_strict(caption_language)
                        if caption_language else
                        normalize_language(language or "en"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        # CPU work runs in a worker thread (Phase 6 fix, 15 Nov 2026):
        # a sync transcribe inside this async endpoint blocked the
        # event loop for minutes — /health hung and Render's proxy
        # cut the client connection while the server was still
        # working. to_thread keeps the loop (and /health) responsive.
        result = await asyncio.to_thread(
            transcribe, video_path, upload_id, model_name=model or None,
            initial_prompt=initial_prompt or None,
            language=eff_language)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "message": "Transcript ready.",
        "upload_id": upload_id,
        "model": result["model"],
        "model_note": result.get("model_note", ""),
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
    request: Request,
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
    # Phase 9: caption language (en / hi / hinglish) — the Whisper
    # decoding language for this job. caption_language wins over the
    # legacy `language` field when both are sent.
    caption_language: str | None = Form(default=None),
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
    Phase 9: caption_language (en/hi/hinglish) is the transcription
    language — the ASS/SRT carry exactly what Whisper returned in
    that language; nothing is translated.
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
        # Phase 9: unknown caption_language is a 400 here, not a
        # silent English fallback (see normalize_language_strict).
        eff_language = (normalize_language_strict(caption_language)
                        if caption_language else
                        normalize_language(language or "en"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        opts = parse_options({
            "clip_count": top_n, "min_sec": min_sec, "max_sec": max_sec,
            "from_min": from_min, "to_min": to_min,
            "caption_style": caption_style, "layout": layout,
            "quality": quality, "video_type": video_type,
            "caption_language": eff_language,
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
                language=eff_language)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    if not transcript.get("words"):
        raise HTTPException(
            status_code=400,
            detail="Transcript khaali hai — video me boli hui baat nahi mili, clips nahi ban sakte")
    # Phase 9: an existing transcript in ANOTHER language does not
    # satisfy this request — re-transcribe in the chosen language so
    # the captions really follow the choice (Whisper decoding, never
    # a silent translation of the old transcript).
    if transcript.get("language_requested") != eff_language:
        try:
            transcript = await asyncio.to_thread(
                transcribe, video_path, upload_id, language=eff_language)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not transcript.get("words"):
            raise HTTPException(
                status_code=400,
                detail="Transcript khaali hai — video me boli hui baat nahi mili, clips nahi ban sakte")
    # Phase 6: the logged-in user's BYOK config drives the words
    # judge (their provider/model/key); anonymous calls keep the
    # legacy env-key behaviour. Ledger rows are written per user.
    _user = _current_user(request)
    ai_config = None
    ledger_ctx = None
    if _user is not None:
        ai_config = get_effective_ai_config(DB_PATH, _user["id"])
        ledger_ctx = {"db_path": DB_PATH, "user_id": _user["id"],
                      "upload_id": upload_id,
                      "creator_id": row.get("creator_id")}
    # Phase 3: pass the real video file so the Reaction judge can
    # measure loudness (without it the reaction score is neutral 50).
    result = find_clips(transcript, upload_id, top_n=opts["clip_count"],
                        style_name=style_name, options=opts,
                        language=eff_language,
                        video_path=video_path, ai_config=ai_config,
                        ledger_ctx=ledger_ctx)
    return {"message": "Clips ready.", "options": options_summary(opts), **result}


@app.get("/clips/{upload_id}")
def get_clips_endpoint(upload_id: str) -> dict:
    try:
        return load_clips(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


async def _ensure_transcript_language(video_path, upload_id: str,
                                      transcript: dict,
                                      caption_language: str | None) -> tuple[dict, bool]:
    """Phase 9 (per-clip language): if the requested caption language
    differs from the transcript's language, re-transcribe in the
    requested language (Whisper decoding) and return the fresh
    transcript + True. Otherwise return the transcript unchanged +
    False. Never "translates" an old transcript — the words always
    come from Whisper hearing the audio in that language.
    """
    wanted = normalize_language(caption_language) if caption_language else None
    if not wanted or transcript.get("language_requested") == wanted:
        return transcript, False
    fresh = await asyncio.to_thread(transcribe, video_path, upload_id,
                                    language=wanted)
    if not fresh.get("words"):
        raise HTTPException(
            status_code=400,
            detail="Transcript khaali hai — is language me boli hui baat nahi mili")
    return fresh, True


@app.post("/render-clip")
async def render_clip_endpoint(
    request: Request,
    upload_id: str = Form(...),
    clip_id: str = Form(...),
    style_name: str = Form(default="global-baatien"),
    # Phase 2: render-time Options. Empty = whatever /find-clips
    # recorded for this upload (its Options panel), else defaults.
    caption_style: str | None = Form(default=None),
    layout: str | None = Form(default=None),
    quality: str | None = Form(default=None),
    # Phase 9: per-clip caption language — a different language than
    # the stored transcript re-transcribes first (see helper above).
    caption_language: str | None = Form(default=None),
) -> dict:
    """Render ONE clip to 9:16 with captions + grading.

    Runs in a worker thread (asyncio.to_thread) — a ~47s clip takes
    1-3 min on CPU. Returns render metadata (paths, resolution,
    caption script, reframe mode, etc.).

    Phase 6 hardening (Phase 5 flag): login + ownership required.
    Without login -> 401; another user's upload -> 403. Ownerless
    (legacy anonymous) uploads may be rendered by any logged-in user.
    """
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login karo — /login kholo")
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    if not _owns_upload(request, upload_id):
        raise HTTPException(status_code=403, detail="Ye video tumhari nahi hai")
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
            "caption_language": (caption_language
                                 or stored.get("caption_language")
                                 # Phase 9 fallback: a clips file written
                                 # before caption_language existed (or by
                                 # an older call) must not flip the
                                 # project back to English — the stored
                                 # transcript's language is the truth.
                                 or transcript.get("language_requested")),
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Phase 9: per-clip language switch — re-transcribe if needed.
    try:
        transcript, _retrans = await _ensure_transcript_language(
            video_path, upload_id, transcript, opts.get("caption_language"))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
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
    return {"message": "Clip rendered.",
            "retranscribed_for_language": _retrans, **meta}


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
    request: Request,
    upload_id: str = Form(...),
    clip_id: str = Form(...),
) -> dict:
    """Approve ONE test clip. This unlocks POST /render-all.

    Phase 9: login + ownership required (an approval is what unlocks
    bulk rendering, so it must come from the video's owner).
    """
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login karo — /login kholo")
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    if not _owns_upload(request, upload_id):
        raise HTTPException(status_code=403, detail="Ye video tumhari nahi hai")
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
    request: Request,
    upload_id: str = Form(...),
    clip_id: str = Form(...),
    reason: str = Form(default=""),
) -> dict:
    """Reject a clip (it stays in the list, marked rejected).

    Phase 9: login + ownership required, same rule as approve.
    """
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login karo — /login kholo")
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    if not _owns_upload(request, upload_id):
        raise HTTPException(status_code=403, detail="Ye video tumhari nahi hai")
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
    request: Request,
    upload_id: str = Form(...),
    style_name: str = Form(default="global-baatien"),
    # Phase 2: same render-time Options as /render-clip.
    caption_style: str | None = Form(default=None),
    layout: str | None = Form(default=None),
    quality: str | None = Form(default=None),
    # Phase 9: per-job caption language (re-transcribes once if needed).
    caption_language: str | None = Form(default=None),
) -> dict:
    """Render ALL clips of an upload, sequentially.

    GUARD: works ONLY after at least one test clip was approved
    (POST /approve-clip). Without approval -> 400
    "Pehle 1 test clip approve karo". Already-rendered clips (sidecar
    upload_id matches) are skipped, not re-rendered. Each clip runs
    in a worker thread — 1-3 min on CPU per clip, so a 4-clip video
    takes ~8 min; /health stays responsive throughout.
    Phase 9: login + ownership required (the guard used to be
    enforceable by anyone — an unauthenticated call could render a
    whole project once somebody else had approved a clip).
    """
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login karo — /login kholo")
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    if not _owns_upload(request, upload_id):
        raise HTTPException(status_code=403, detail="Ye video tumhari nahi hai")
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
            "caption_language": (caption_language
                                 or stored.get("caption_language")
                                 # Phase 9 fallback: see /render-clip —
                                 # the stored transcript's language wins
                                 # over the English default.
                                 or transcript.get("language_requested")),
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Phase 9: one re-transcribe for the whole job if the language changed.
    try:
        transcript, _retrans = await _ensure_transcript_language(
            video_path, upload_id, transcript, opts.get("caption_language"))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
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
            "upload_id": upload_id,
            "retranscribed_for_language": _retrans,
            "gate": get_project_status(upload_id).get("gate_message", ""),
            "results": results,
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
                         language=normalize_language(
                             str(payload.get("caption_language")
                                 or payload.get("language") or "en")))
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
video{width:100%;max-width:480px;border-radius:12px;background:#000}
textarea{width:100%;background:#0d1230;border:1px solid #3b4380;border-radius:10px;color:#fff;padding:10px;margin-top:6px;box-sizing:border-box;font-family:sans-serif}
.scorebig{font-size:44px;font-weight:800}
.chip{display:inline-block;background:#262c5e;border-radius:999px;padding:4px 12px;font-size:13px;margin:2px 4px 2px 0}
.chip.ok{background:#065f46}.chip.bad{background:#7f1d1d}.chip.warn{background:#78350f}
.scorebar{height:8px;background:#1e2450;border-radius:4px;overflow:hidden;margin:4px 0 10px}
.scorebar i{display:block;height:100%;background:#ff2e63}
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
    # Phase 6: sidebar AI line shows real ledger counts (never breaks
    # the layout if the ledger is unavailable).
    try:
        _tc = today_counts(DB_PATH, user["id"])
        ai_calls, ai_tokens = _tc["calls_today"], _tc["tokens_today"]
        ai_label = PROVIDERS[get_effective_ai_config(
            DB_PATH, user["id"])["provider"]]["label"]
    except Exception:
        ai_calls, ai_tokens, ai_label = 0, 0, "Gemini"

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
            + nav("/app?tour=1", "✦ Product tour", "Tour")
            + nav("/app/videos", "All videos", "All videos")
            + "<div style=\"margin-top:14px;color:#a5b4fc;font-size:12px;"
              "text-transform:uppercase;letter-spacing:.1em\">Creators</div>"
            + creator_links
            + nav("/app", "＋ New creator", "New creator")
            + nav("/app/usage", "AI usage", "AI usage")
            + nav("/app/post-queue", "Post queue", "Post queue")
            + nav("/app/notifications", "Notifications", "Notifications")
            + nav("/app/settings/profile", "Settings", "Settings")
            + "<a href=\"/logout\">Sign out</a></nav>"
            f"<div class=\"meter\">Storage — {used_mb:.1f} of 100 MB"
            f"<div class=\"meterbar\"><i style=\"width:{pct:.1f}%\"></i></div>"
            "<div class=\"muted\" style=\"margin-top:6px\">When your space is full, "
            "your oldest videos are deleted to make room. Download the clips "
            "you want to keep.</div></div>"
            f"<div class=\"meter\">AI {ai_label} — {ai_calls} today · "
            f"{ai_tokens} tokens<br><span class=\"muted\">"
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


def _live_status_label(st: dict) -> str:
    """Human label for a creator's last live check (Phase 7)."""
    s = (st.get("status") or "—")
    if s == "live":
        label = "🔴 LIVE" + (f" — {_html.escape(st['title'])}" if st.get("title") else "")
    elif s == "offline":
        label = "Offline"
    elif s == "unknown":
        label = "Unknown (check failed)"
    else:
        label = "—"
    if st.get("checked_at"):
        label += f" <span class=\"muted\">({str(st['checked_at'])[:16]})</span>"
    return label


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
    # Phase 8: real posting counts (was hardcoded 0).
    try:
        _qc = queue_counts(DB_PATH, user["id"])
        _posted_days = posted_per_day(DB_PATH, user["id"], days=14)
        posted_today = _qc["posted_today"]
        waiting_to_post = _qc["waiting"]
        posted_total = _qc["posted_total"]
    except Exception:
        _posted_days, posted_today = {}, 0
        waiting_to_post, posted_total = 0, 0
    activity = []
    for i in range(13, -1, -1):
        day = today - _td(days=i)
        n = sum(1 for c in clips if c["date"] == day)
        activity.append({"day": day.strftime("%b %d"), "clips": n,
                         "posted": _posted_days.get(day.isoformat(), 0)})
    return {"videos_today": videos_today, "videos_total": videos_total,
            "clips_today": clips_today, "clips_total": len(clips),
            "activity": activity,
            "posted_today": posted_today,
            "waiting_to_post": waiting_to_post,
            "posted_total": posted_total,
            # Phase 7: real live-session counts (was hardcoded 0).
            "recording_now": count_live_sessions(DB_PATH, user["id"]),
            "live_sessions": [session_public(DB_PATH, s) for s in
                              list_live_sessions(DB_PATH, user["id"])
                              if s["status"] in ("live", "stopping")]}


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


TOUR_COOKIE = "clipcut_tour"


def _tour_steps(user: dict, data: dict, ai_has_key: bool) -> list[dict]:
    """Phase 10 product tour: 7 steps, each with a LIVE done-state
    (nothing is ticked unless the user really did it)."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        upload_ids = [r[0] for r in conn.execute(
            "SELECT id FROM uploads WHERE user_id=?",
            (user["id"],)).fetchall()]
    try:
        approved = any(has_approved_clip(uid) for uid in upload_ids)
    except Exception:
        approved = False
    try:
        rendered = any(list_rendered_for_upload(uid) for uid in upload_ids)
    except Exception:
        rendered = False
    try:
        _qc = queue_counts(DB_PATH, user["id"])
        queued = (_qc["waiting"] + _qc["posted_total"]) > 0
    except Exception:
        queued = False
    return [
        {"done": len(list_creators(DB_PATH, user["id"])) > 0,
         "text": "Add your first creator — each streamer gets their own workspace.",
         "href": "/app", "link": "Add a creator ↓"},
        {"done": data["videos_total"] > 0,
         "text": "Paste a link or upload a file — link ki zaroorat nahi, file bhi chalegi.",
         "href": "/app", "link": "Clip box ↑"},
        {"done": data["clips_total"] > 0,
         "text": "Find clips — dual judge (words ≥ 50 AND reaction ≥ 40) picks the moments.",
         "href": "/app/videos", "link": "All videos"},
        {"done": approved,
         "text": "Approve 1 test clip — Render All tabhi unlock hota hai.",
         "href": "/app/videos", "link": "Open a video"},
        {"done": rendered,
         "text": "Render a clip — 9:16, captions, yuv420p. Re-make purana version overwrite nahi karta.",
         "href": "/app/videos", "link": "Clip editor"},
        {"done": queued,
         "text": "Queue a clip for posting — YouTube Shorts, scheduled time par.",
         "href": "/app/post-queue", "link": "Post queue"},
        {"done": ai_has_key,
         "text": "Add your AI key in Settings / AI — bina key ke bhi clips bante hain (rule-based).",
         "href": "/app/settings/ai", "link": "AI settings"},
    ]


@app.post("/app/tour/dismiss")
def tour_dismiss(request: Request):
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    resp = RedirectResponse("/app", status_code=303)
    resp.set_cookie(TOUR_COOKIE, "done", max_age=365 * 24 * 3600,
                    httponly=True, samesite="lax")
    return resp


@app.get("/app", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    data = _dashboard_data(user)
    creators = list_creators(DB_PATH, user["id"])
    # Phase 6: dashboard AI card shows real ledger counts + provider.
    try:
        _tc = today_counts(DB_PATH, user["id"])
        ai_calls_today, ai_tokens_today = _tc["calls_today"], _tc["tokens_today"]
        _cfg = get_effective_ai_config(DB_PATH, user["id"])
        ai_provider_label = PROVIDERS[_cfg["provider"]]["label"]
        ai_model = _cfg["model"]
        ai_has_key = bool(_cfg["has_key"])
    except Exception:
        ai_calls_today, ai_tokens_today = 0, 0
        ai_provider_label, ai_model, ai_has_key = "Gemini", "gemini-2.0-flash", False
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
            f"<td><span data-live-cell=\"{c['id']}\">"
            f"{_live_status_label(get_creator_live_status(DB_PATH, c['id']))}</span> "
            f"<button type=\"button\" data-live-check=\"{c['id']}\" "
            f"style=\"background:none;border:0;color:#93c5fd;cursor:pointer;padding:0\">"
            f"Check</button></td>"
            f"<td><a href=\"/app/videos?creator={c['id']}\">Videos</a> · "
            f"<form method=\"post\" action=\"/app/creators/{c['id']}/delete\""
            f" style=\"display:inline\" onsubmit=\"return confirm("
            f"'Delete this creator? Videos filed under it stay, unfiled.');\">"
            f"<button type=\"submit\" style=\"background:none;border:0;color:"
            f"#f87171;cursor:pointer;padding:0\">Delete</button></form></td></tr>"
            for c in creators)
        creators_html = ("<table><tr><th>Name</th><th>Platform</th><th>Channel</th>"
                         "<th>Live?</th><th></th></tr>" + creator_rows + "</table>")
    else:
        creators_html = ("<p class=\"muted\">No creators yet — Add one to give each "
                         "streamer their own workspace.</p>")
    step1 = "✅" if creators else "1"
    step2 = "✅" if data["videos_total"] else "2"
    # Phase 8: step 3 is done when a YouTube posting account is connected.
    try:
        step3 = "✅" if get_youtube_account(DB_PATH, user["id"]) else "3"
    except Exception:
        step3 = "3"
    # Phase 8: real post-queue card (was a hardcoded empty state).
    try:
        _pq = queue_counts(DB_PATH, user["id"])
        _paused_all = is_paused_all(DB_PATH, user["id"])
        _queued_items = list_queue(DB_PATH, user["id"], status="queued")
        _next_due = _queued_items[0] if _queued_items else None
    except Exception:
        _pq = {"waiting": 0, "posted_total": 0, "failed": 0}
        _paused_all, _next_due = False, None
    if _pq["waiting"] or _pq["posted_total"] or _pq["failed"]:
        _next_line = (f"<p class=\"muted\">Next: <b>{_html.escape(_next_due['title'])}</b> "
                      f"→ {PLATFORMS[_next_due['platform']]['label']} · "
                      f"{format_ist(_next_due['scheduled_at'])}</p>"
                      if _next_due else "")
        _pause_line = ("<p class=\"muted\">⏸ Paused — worker kuch post nahi "
                       "karega jab tak Resume na dabao.</p>" if _paused_all
                       else "<p class=\"muted\">▶ Posting on — due items apne "
                       "time par post honge.</p>")
        pq_card = (f"<div class=\"card\"><h3>Post queue</h3>"
                   f"<p class=\"muted\">{_pq['waiting']} waiting to post · "
                   f"{_pq['posted_total']} posted · {_pq['failed']} failed</p>"
                   f"{_next_line}{_pause_line}"
                   f"<form method=\"post\" action=\"/app/post-queue/pause-all\" "
                   f"style=\"display:inline\">"
                   f"<button class=\"btn ghost\" type=\"submit\">"
                   f"{'Resume all posting' if _paused_all else 'Pause all posting'}"
                   f"</button></form> "
                   f"<a class=\"btn ghost\" href=\"/app/post-queue\">Open post queue</a></div>")
    else:
        pq_card = ("<div class=\"card\"><h3>Post queue</h3>"
                   "<p class=\"muted\">Queued clips go out at their scheduled time.</p>"
                   "<p class=\"muted\">Nothing queued or posted yet.</p>"
                   "<a class=\"btn ghost\" href=\"/app/post-queue\">Open post queue</a></div>")
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
    # Phase 9: caption language radios (en / hi / hinglish) — the
    # Whisper decoding language for this job's captions.
    lang_radios = "".join(
        f'<label><input type="radio" name="opt-lang" value="{k}"'
        f'{" checked" if k == "en" else ""}> {v["label"]}'
        f' <span class="muted">— {v["note"]}</span></label>'
        for k, v in CAPTION_LANGUAGES.items())
    # Phase 9: saved styles for the reference-style card.
    try:
        _styles = list_styles(STYLES_DIR)
    except Exception:
        _styles = []
    if _styles:
        styles_html = ("<p>" + " · ".join(
            f"<b>{_html.escape(s['name'])}</b> "
            f"<span class=\"muted\">({s['slug']})</span>" for s in _styles)
            + "</p><p class=\"muted\">Clip editor me Re-make karte waqt "
              "inme se koi style choose karo.</p>")
    else:
        styles_html = ("<p class=\"muted\">Abhi sirf default Global Baatien "
                       "style hai — upar se reference video do.</p>")
    # Phase 9: banners for upload / style-extract redirects.
    _err = request.query_params.get("error", "")
    _extracted = request.query_params.get("extracted", "")
    if _err:
        _banner = f'<p class="err">Error: {_html.escape(_err)}</p>'
    elif _extracted:
        _banner = (f'<p style="color:#4ade80">Style saved ✓ — '
                   f'{_html.escape(_extracted)} (clip editor me choose karo)</p>')
    else:
        _banner = ""
    # Phase 7: live sessions card (real counts, not hardcoded 0).
    if data["live_sessions"]:
        live_rows = "".join(
            f"<p>🔴 <a href=\"/app/live/{s['session_id']}\">"
            f"{_html.escape(s['title'])}</a> — {s['chunks_done']} chunks done, "
            f"{s['clips_count']} clips · {s['status']} · "
            f"{int(s['total_duration_sec'] // 60)}:{int(s['total_duration_sec'] % 60):02d} min</p>"
            for s in data["live_sessions"])
        live_card = (f"<div class=\"card\"><h3>🔴 Recording now — "
                     f"{data['recording_now']}</h3>{live_rows}"
                     f"<a class=\"btn ghost\" href=\"/app/live\">All live sessions</a></div>")
    else:
        live_card = ("<div class=\"card\"><h3>🔴 Live recording</h3>"
                     "<p class=\"muted\">No live session right now — apni camera ya "
                     "screen record karo, har 60s ke chunk se clips live bante hain.</p>"
                     "<a class=\"btn\" href=\"/live/record\">Record live</a> "
                     "<a class=\"btn ghost\" href=\"/app/live\">All live sessions</a></div>")
    # Phase 10: first-run product tour — live done-states; hidden once
    # dismissed (cookie) unless ?tour=1 forces it back.
    _show_tour = (request.query_params.get("tour") == "1"
                  or request.cookies.get(TOUR_COOKIE) != "done")
    if _show_tour:
        _steps = _tour_steps(user, data, ai_has_key)
        _done_n = sum(1 for s in _steps if s["done"])
        _rows = "".join(
            f"<p>{'✅' if s['done'] else '☐'} {s['text']} "
            f"<a href=\"{s['href']}\">{s['link']}</a></p>" for s in _steps)
        tour_card = (f"<div class=\"card\" id=\"tour\"><h3>✦ Product tour — "
                     f"{_done_n}/7 done</h3><p class=\"muted\">Pehli baar yahan ho "
                     f"to ek-ek karke karo — har step ka link usi jagah le jata hai.</p>"
                     f"{_rows}<form method=\"post\" action=\"/app/tour/dismiss\">"
                     f"<button class=\"btn ghost\" type=\"submit\">Dismiss tour</button></form></div>")
    else:
        tour_card = ""
    body = f"""
<h1>Home</h1>
{_banner}
{tour_card}
<p class="muted">Good afternoon. {len(creators)} creator{'s' if len(creators) != 1 else ''}.</p>
<div class="card"><h3>What do you want to clip today?</h3>
<p class="muted">Paste a link to a video — ya neeche se file upload karo. You get captioned shorts back.</p>
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
<div class="radio"><label style="display:block">Caption language — captions kis bhasha me hon?</label>{lang_radios}</div>
<div class="optrow"><div><label>Only look at part of the video — From (minute)
<input id="opt-from" type="number" min="0" step="0.5" placeholder="Up to minute"></label></div>
<div><label>To (minute)
<input id="opt-to" type="number" min="0" step="0.5" placeholder="Up to minute"></label></div></div>
</details>
<p id="yt-error" class="err"></p>
<button class="btn" type="submit">Make clips</button></form>
<p id="yt-status" class="muted"></p></div>
<div class="card"><h3>Or upload a file — link ki zaroorat nahi</h3>
<p class="muted">Reference app me file upload hai hi nahi — yahan hai. Apni video upload karo, wahi pipeline chalega: transcribe → find-clips → editor → post queue. 100 MB quota (per user, sabse purani video pehle delete hoti hai).</p>
<form method="post" action="/app/upload" enctype="multipart/form-data">
<label>Video file (MP4 / MOV)<input type="file" name="long_video" accept=".mp4,.mov" required></label>
<div style="display:flex;gap:10px"><label style="flex:1">File under creator
<select name="creator_id"><option value="">No creator</option>{creator_options}</select></label></div>
<button class="btn" type="submit">Upload &amp; open video</button></form>
<p class="muted">Upload ke baad video page khulega — wahan transcribe + find-clips karo (caption language wahi choose karo).</p></div>
<div class="card"><h3>Reference style — kisi video ki editing style copy karo</h3>
<p class="muted">Captions wali reference video do — uski caption colours, size aur position measure karke saved style banti hai. Bina captions wale video par "style could not be read" milega — koi invented style nahi banegi.</p>
<form method="post" action="/app/extract-style" enctype="multipart/form-data">
<label>Reference video (MP4 / MOV)<input type="file" name="reference_video" accept=".mp4,.mov" required></label>
<label>Style name<input name="style_name" type="text" placeholder="e.g. My Podcast Style" value="Extracted Style"></label>
<button class="btn" type="submit">Extract style</button></form>
<p class="muted">Saved styles:</p>{styles_html}</div>
<div class="tiles">
<div class="tile"><b>{data['recording_now']}</b><span>Recording now</span></div>
<div class="tile"><b>{data['clips_today']}</b><span>Clips today</span></div>
<div class="tile"><b>{data['posted_today']}</b><span>Posted today</span></div>
<div class="tile"><b>{data['waiting_to_post']}</b><span>Waiting to post</span></div></div>
{live_card}
<div class="card"><h3>Activity — Last 14 days</h3>
<p class="muted">{data['clips_total']} Clips made · {data['posted_total']} Posted · 0 Views · $0.00 Est. earnings</p>
<div class="bar-row">{bars}</div>
<div style="display:flex;gap:6px">{labels}</div>
<p class="muted">Pink = Clips made · White = Posted</p></div>
<div class="card"><h3>AI — {ai_provider_label}</h3>
<p class="muted">{ai_calls_today} AI calls today · {ai_tokens_today} Tokens today
({ai_model}{'' if ai_has_key else ' — no key, rule-based'})</p>
<a class="btn ghost" href="/app/usage">View usage</a>
<a class="btn ghost" href="/app/settings/ai">AI settings</a></div>
<div class="card"><h3>Next steps — Finish setting up ClipCut</h3>
<p>{step1} Add your first creator — Give each streamer their own workspace, settings and folder.</p>
<p>{step2} Make your first clips — Paste a video link in the box above.</p>
<p>{step3} Connect a posting account — YouTube, Instagram or TikTok, in one click. (<a href="/app/settings/apps">Connected apps</a>)</p></div>
<div class="card"><h3>Creators — Each creator's status and results</h3>{creators_html}
<form method="post" action="/app/creators"><h3 style="margin-top:16px">Add a creator</h3>
<p class="muted">ClipCut can record this streamer live and keep the clips organised by them. (Live recording — Phase 7)</p>
<label>Channel link<input name="channel_link" type="url" placeholder="https://www.youtube.com/@name, twitch.tv/name or kick.com/name" required></label>
<label>Name<input name="name" type="text" placeholder="Filled in from the channel, or type one."></label>
<button class="btn" type="submit">Create workspace</button></form></div>
{pq_card}
<script>
document.querySelectorAll('[data-live-check]').forEach((btn) => {{
  btn.addEventListener('click', async () => {{
    const cid = btn.getAttribute('data-live-check');
    const cell = document.querySelector('[data-live-cell="' + cid + '"]');
    btn.textContent = 'Checking…';
    try {{
      const r = await fetch('/live/check', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{creator_id: cid}})}});
      const j = await r.json();
      if (!r.ok) {{ cell.textContent = 'Error: ' + (j.detail || 'check nahi hua'); }}
      else if (j.live === true) {{ cell.textContent = '🔴 LIVE' + (j.title ? ' — ' + j.title : ''); }}
      else if (j.live === false) {{ cell.textContent = 'Offline'; }}
      else {{ cell.textContent = 'Unknown — ' + (j.error || 'pata nahi chala'); }}
    }} catch (err) {{ cell.textContent = 'Error: ' + err; }}
    btn.textContent = 'Check';
  }});
}});
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
    caption_language: radio('opt-lang') || 'en',
    video_type: val('yt-type') || 'auto',
    from_min: fromMin, to_min: toMin,
  }};
  st.textContent = 'Job start ho raha hai…';
  try {{
    const r = await fetch('/youtube-job', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{url, language: radio('opt-lang') || 'en', creator_id: val('yt-creator') || null, options}})}});
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
        # Phase 7 fix (pre-existing bug): get_project_status() returns
        # "upload_id", not "id" — /app/videos 500'd (KeyError) for ANY
        # project, live sessions included. Found by the Phase 7 test.
        pid = p["upload_id"]
        uid, cid = owner.get(pid, (None, None))
        if uid != user["id"]:
            continue
        if creator and cid != creator:
            continue
        rows.append(p)
    if rows:
        body_rows = ""
        for p in rows:
            pid = p["upload_id"]
            # Phase 5: clips with scores, each linking to its editor.
            try:
                clips = load_clips(pid).get("clips", [])
            except FileNotFoundError:
                clips = []
            clip_links = " · ".join(
                f"<a href=\"/app/videos/{pid}/clips/{c['clip_id']}\">"
                f"{c['clip_id']} ({_fmt_score(c.get('combined_score', c.get('score')))})</a>"
                for c in clips) or "<span class=\"muted\">koi clip nahi</span>"
            body_rows += (
                f"<tr><td><a href=\"/app/videos/{pid}\">{pid[:8]}…</a></td>"
                f"<td>{p.get('status','')}</td><td>{clip_links}</td></tr>")
        table = ("<table><tr><th>Video</th><th>Status</th><th>Clips (score)</th></tr>"
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


# ---------------------------------------------------------- Phase 5: clip editor

def _upload_owner(upload_id: str) -> str | None:
    """Owner user_id of an upload, or None (legacy / no owner)."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute("SELECT user_id FROM uploads WHERE id=?",
                           (upload_id,)).fetchone()
    return row[0] if row else None


def _owns_upload(request: Request, upload_id: str) -> bool:
    """True when the caller may open this upload (owner, or no owner)."""
    owner = _upload_owner(upload_id)
    if owner is None:
        return True
    user = _current_user(request)
    return user is not None and user["id"] == owner


def _validate_trim(start, end, duration: float,
                   cur_start: float, cur_end: float) -> tuple[float, float]:
    """Validate trim start/end (seconds) against the source duration.

    None keeps the clip's current value. Rules (Hinglish 400s):
    start >= 0, end <= duration, end > start, end - start >= 5s.
    Returns (start, end) floats.
    """
    s = cur_start if start is None or str(start) == "" else float(start)
    e = cur_end if end is None or str(end) == "" else float(end)
    if s < 0:
        raise ValueError(f"Trim start 0 ya usse zyada hona chahiye — mila: {s}s")
    if e > duration:
        raise ValueError(
            f"Trim end ({e}s) video ki length ({duration:.1f}s) se zyada nahi ho sakta")
    if e <= s:
        raise ValueError(f"Trim end ({e}s) start ({s}s) se bada hona chahiye")
    if e - s < 5:
        raise ValueError(
            f"Trimmed clip kam se kam 5 seconds ka hona chahiye — mila: {e - s:.1f}s")
    return round(s, 2), round(e, 2)


async def _read_body(request: Request) -> dict:
    """JSON body or form fields, as one dict (editor endpoints take both)."""
    ctype = (request.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        try:
            data = await request.json()
            return dict(data) if isinstance(data, dict) else {}
        except Exception:
            return {}
    try:
        form = await request.form()
        return {k: v for k, v in form.multi_items()}
    except Exception:
        return {}


def _clip_or_404(upload_id: str, clip_id: str) -> tuple[dict, dict]:
    """(clips_data, clip) or raises 404."""
    try:
        clips_data = load_clips(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    clip = next((c for c in clips_data.get("clips", [])
                 if str(c.get("clip_id")) == str(clip_id)), None)
    if not clip:
        raise HTTPException(
            status_code=404,
            detail=f"clip not found: {clip_id} — pehle POST /find-clips karo")
    return clips_data, clip


def _fmt_score(v) -> str:
    return "—" if v is None else f"{int(v)}/100"


def _clip_detail_body(user: dict, upload_id: str, clip: dict,
                      versions: list[dict], error: str = "",
                      saved: bool = False, gate_unlocked: bool = False) -> str:
    """HTML body for the clip detail page (score + judges + editor)."""
    esc = _html.escape
    row = _get_upload_row(upload_id)
    duration = 0.0
    if row:
        probe = probe_video(ROOT / row["stored_path"])
        duration = probe["duration"] if probe["valid"] else 0.0
    ws, rs = clip.get("words_score"), clip.get("reaction_score")
    combined = clip.get("combined_score", clip.get("score"))
    agree = clip.get("judges_agree")
    agree_chip = ('<span class="chip ok">✅ Both judges agree</span>' if agree is True
                  else '<span class="chip bad">❌ Gate fail (dropped hota)</span>' if agree is False
                  else '<span class="chip warn">Hand clip — gate nahi laga</span>')
    source = str(clip.get("source") or "ai")
    rb = clip.get("reaction_breakdown") or {}
    latest = versions[-1] if versions else None
    if latest:
        preview = (f'<video controls src="/rendered/{upload_id}/{clip["clip_id"]}">'
                   f'</video><p class="muted">Latest render — v{latest.get("version", 1)} '
                   f'({latest.get("resolution", "?")}, {latest.get("duration_sec", "?")}s)</p>'
                   + (f'<p class="muted">{esc(str(latest.get("language_note", "")))}</p>'
                      if latest.get("language_note") else "")
                   + (f'<p class="muted">Layout: {esc(str(latest.get("layout", "?")))} — '
                      f'{esc(str(latest.get("layout_note", "")))}</p>'
                      if latest.get("layout_note") else ""))
    else:
        preview = (f'<video controls preload="metadata" '
                   f'src="/app/videos/{upload_id}/source#t={clip["start"]},{clip["end"]}">'
                   f'</video><p class="muted">Abhi tak render nahi hua — source ka '
                   f'clip range preview ({clip["start"]}s – {clip["end"]}s).</p>')
    if versions:
        ver_rows = "".join(
            f"<tr><td>v{v.get('version', '?')}</td>"
            f"<td>{esc(str(v.get('resolution', '?')))}</td>"
            f"<td>{esc(str(v.get('caption_style', '?')))}</td>"
            f"<td>{esc(str(v.get('layout', '?')))}</td>"
            f"<td>{v.get('duration_sec', '?')}s</td>"
            f"<td>{(int(v.get('file_size_bytes') or 0) // 1024)} KB</td>"
            f"<td><a href=\"/rendered/{upload_id}/{clip['clip_id']}/v/{v.get('version', 1)}\">MP4</a> · "
            f"<a href=\"/rendered/{upload_id}/{clip['clip_id']}/v/{v.get('version', 1)}/srt\">SRT</a> · "
            f"<a href=\"/rendered/{upload_id}/{clip['clip_id']}/v/{v.get('version', 1)}/caption\">Caption</a></td></tr>"
            for v in versions)
        ver_html = ("<table><tr><th>Version</th><th>Resolution</th><th>Caption</th>"
                    "<th>Layout</th><th>Duration</th><th>Size</th><th>Download</th></tr>"
                    + ver_rows + "</table>"
                    "<p class=\"muted\">Latest version upar preview me hai — "
                    "re-make purana version overwrite nahi karta, naya version banta hai.</p>")
    else:
        ver_html = ("<p class=\"muted\">Abhi koi render nahi hai — neeche "
                    "Re-make dabakar pehla render banao.</p>")
    err_html = f'<p class="err">{esc(error)}</p>' if error else ""
    ok_html = '<p style="color:#4ade80">Saved ✓</p>' if saved else ""
    cap_opts = "".join(
        f'<option value="{k}"{" selected" if k == (latest or {}).get("caption_style", clip.get("caption_style", "bounce")) else ""}>{v["label"]}</option>'
        for k, v in CAPTION_STYLES.items())
    lay_opts = "".join(
        f'<option value="{k}"{" selected" if k == (latest or {}).get("layout", "blurred_fill") else ""}>{v["label"]}</option>'
        for k, v in LAYOUTS.items())
    qual_opts = "".join(
        f'<option value="{k}"{" selected" if k == (latest or {}).get("quality", "720p") else ""}>{v["label"]}</option>'
        for k, v in QUALITIES.items())
    # Phase 9: style preset picker (saved reference styles) + per-clip
    # caption language. Face tracking appears in the layout dropdown
    # automatically (LAYOUTS gained "face_track" in options.py).
    try:
        _saved_styles = list_styles(STYLES_DIR)
    except Exception:
        _saved_styles = []
    style_opts = "".join(
        f'<option value="{s["slug"]}"{" selected" if s["slug"] == (latest or {}).get("style_slug", "global-baatien") else ""}>{esc(s["name"])}</option>'
        for s in _saved_styles) or '<option value="global-baatien">Global Baatien Style</option>'
    lang_opts = "".join(
        f'<option value="{k}"{" selected" if k == (latest or {}).get("language_requested", "en") else ""}>{v["label"]}</option>'
        for k, v in CAPTION_LANGUAGES.items())
    # Phase 8: queue-for-posting card (needs a render first — the
    # worker uploads the actual rendered MP4, never a placeholder).
    from datetime import datetime as _dt8, timedelta as _td8, timezone as _tz8
    _default_sched = format_ist_input(
        (_dt8.now(_tz8.utc) + _td8(hours=1)).isoformat())
    if latest:
        _dur = latest.get("duration_sec")
        _shorts_warn = ""
        try:
            if _dur is not None and float(_dur) > 60:
                _shorts_warn = (f"<p class=\"err\">⚠️ Yeh render {_dur}s ka hai — "
                                "YouTube Shorts 60s se kam ka hota hai. 60s+ wala "
                                "video Shorts nahi, normal video ki tarah post hoga.</p>")
        except (TypeError, ValueError):
            pass
        queue_card = (f"<div class=\"card\"><h3>Queue for posting</h3>"
                      f"{_shorts_warn}"
                      f"<form method=\"post\" action=\"/app/post-queue\">"
                      f"<input type=\"hidden\" name=\"upload_id\" value=\"{upload_id}\">"
                      f"<input type=\"hidden\" name=\"clip_id\" value=\"{clip['clip_id']}\">"
                      f"<input type=\"hidden\" name=\"version\" value=\"{latest.get('version', 1)}\">"
                      f"<label>Platform<select name=\"platform\">"
                      f"<option value=\"youtube\">YouTube Shorts</option>"
                      f"<option value=\"instagram\">Instagram Reels (not set up)</option>"
                      f"<option value=\"tiktok\">TikTok (not set up)</option></select></label>"
                      f"<label>Title<input name=\"title\" maxlength=\"100\" value=\"{esc(str(clip.get('title') or clip.get('hook_line') or ''))}\"></label>"
                      f"<label>Description<textarea name=\"description\" rows=\"3\">{esc(str(clip.get('description') or ''))}</textarea></label>"
                      f"<label>Hashtags<input name=\"hashtags\" value=\"{esc(str(clip.get('hashtags') or ''))}\"></label>"
                      f"<label>Privacy<select name=\"privacy\">"
                      f"<option value=\"unlisted\" selected>Unlisted (safe default)</option>"
                      f"<option value=\"public\">Public</option>"
                      f"<option value=\"private\">Private</option></select></label>"
                      f"<label>Post at (Asia/Kolkata)<input name=\"scheduled_at\" type=\"datetime-local\" value=\"{_default_sched}\" required></label>"
                      f"<label>Repeat<select name=\"recurrence\">"
                      f"<option value=\"once\">Once</option>"
                      f"<option value=\"daily\">Daily</option>"
                      f"<option value=\"weekly\">Weekly</option></select></label>"
                      f"<button class=\"btn\" type=\"submit\">Queue for posting (v{latest.get('version', 1)})</button></form>"
                      f"<p class=\"muted\">YouTube ke liye pehle Settings &gt; Connected apps me account connect karo — "
                      f"bina account ke item \"waiting for account\" me rahega, kabhi fake post nahi hoga.</p></div>")
    else:
        queue_card = ("<div class=\"card\"><h3>Queue for posting</h3>"
                      "<p class=\"muted\">Pehle upar Re-make se render karo — "
                      "queue ke liye rendered MP4 chahiye.</p></div>")
    return f"""
<h1>{esc(str(clip.get('clip_id')))} — {esc(str(clip.get('title') or clip.get('hook_line') or ''))}</h1>
<p class="muted"><a href="/app/videos/{upload_id}">← Back to video</a> · {esc(str(row.get('original_name') if row else ''))} · {clip['start']}s – {clip['end']}s ({clip['duration']}s) · Source: {esc(source)}</p>
{err_html}{ok_html}
<div class="card">{preview}</div>
<div class="card"><h3>Score — Combined <span class="scorebig">{_fmt_score(combined)}</span></h3>
<p>{agree_chip}</p>
<p><b>Words judge:</b> {_fmt_score(ws)}</p>
<div class="scorebar"><i style="width:{int(ws) if ws is not None else 0}%"></i></div>
<p class="muted">{esc(str(clip.get('words_reason') or ''))}</p>
<p><b>Reaction judge:</b> {_fmt_score(rs)}</p>
<div class="scorebar"><i style="width:{int(rs) if rs is not None else 0}%"></i></div>
<p class="muted">{esc(str(clip.get('reaction_reason') or ''))}</p>
<p class="muted">Loudness: mean {rb.get('mean_db', '?')} dB · median {rb.get('median_db', '?')} dB · speech {rb.get('speech_fraction', '?')} · crest {rb.get('crest_db', '?')} dB · level {rb.get('level_factor', '?')}{' · quiet floor cap laga' if rb.get('quiet_cap_applied') else ''}</p>
<p class="muted">Thresholds: words ≥ 50 · reaction ≥ 40 · Combined = 0.6·words + 0.4·reaction</p></div>
<div class="card"><h3>Hook, title, description, hashtags</h3>
<form method="post" action="/app/videos/{upload_id}/clips/{clip['clip_id']}/edit">
<label>Hook (on-screen, first 2s)<input name="hook_line" value="{esc(str(clip.get('hook_line') or ''))}"></label>
<label>Title<input name="title" value="{esc(str(clip.get('title') or ''))}"></label>
<label>Description<textarea name="description" rows="3">{esc(str(clip.get('description') or ''))}</textarea></label>
<label>Hashtags<input name="hashtags" value="{esc(str(clip.get('hashtags') or ''))}"></label>
<button class="btn" type="submit">Save metadata</button></form>
<p class="muted">Save karne se caption file aur agla render yehi text use karenge.</p></div>
<div class="card"><h3>Trim &amp; Re-make</h3>
<p class="muted">Video length: {duration:.1f}s · Trim kam se kam 5s ka hona chahiye.</p>
<form id="remake-form">
<label>Trim start (seconds)<input id="rm-start" type="number" min="0" step="0.1" value="{clip['start']}"></label>
<label>Trim end (seconds)<input id="rm-end" type="number" min="0" step="0.1" value="{clip['end']}"></label>
<div class="optrow"><div><label>Caption style<select id="rm-caption">{cap_opts}</select></label></div>
<div><label>Frame layout<select id="rm-layout">{lay_opts}</select></label></div>
<div><label>Video quality<select id="rm-quality">{qual_opts}</select></label></div></div>
<div class="optrow"><div><label>Style preset (reference style)<select id="rm-style">{style_opts}</select></label></div>
<div><label>Caption language<select id="rm-lang">{lang_opts}</select></label></div></div>
<p class="muted">Caption language badalne par transcript dobara usi language me banega (Whisper) — purana transcript translate nahi hota.</p>
<p id="rm-status" class="muted"></p>
<p id="rm-error" class="err"></p>
<button class="btn" type="submit">Save &amp; Re-make</button>
<button class="btn ghost" type="button" id="rm-save-trim">Save trim only</button></form>
<p class="muted">Re-make 1-3 min leta hai (CPU render). Purana render delete nahi hota — naya version banta hai, neeche Versions me dikhega.</p></div>
<div class="card"><h3>Versions ({len(versions)})</h3>{ver_html}
<p><a class="btn ghost" href="/download/{upload_id}/{clip['clip_id']}">Download latest (ZIP)</a> '
<a class="btn ghost" href="/rendered/{upload_id}/{clip['clip_id']}">Download latest MP4</a></p></div>
{queue_card}
<div class="card"><h3>Approve / Reject</h3>
<form method="post" action="/approve-clip" style="display:inline">
<input type="hidden" name="upload_id" value="{upload_id}">
<input type="hidden" name="clip_id" value="{clip['clip_id']}">
<button class="btn" type="submit">Approve test clip</button></form>
<form method="post" action="/reject-clip" style="display:inline">
<input type="hidden" name="upload_id" value="{upload_id}">
<input type="hidden" name="clip_id" value="{clip['clip_id']}">
<button class="btn ghost" type="submit">Reject</button></form>
<p class="muted">{'✅ Render All unlocked — is video ka ek test clip approved hai.' if gate_unlocked else '🔒 Render All locked — pehle 1 test clip approve karo, phir bulk render (Render All) chalega.'}</p></div>
<script>
const UID = "{upload_id}", CID = "{clip['clip_id']}";
document.getElementById('remake-form').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const st = document.getElementById('rm-status'), er = document.getElementById('rm-error');
  er.textContent = ''; st.textContent = 'Rendering… 1-3 min lag sakte hain, page band mat karo.';
  const body = {{upload_id: UID, clip_id: CID,
    start: Number(document.getElementById('rm-start').value),
    end: Number(document.getElementById('rm-end').value),
    caption_style: document.getElementById('rm-caption').value,
    layout: document.getElementById('rm-layout').value,
    quality: document.getElementById('rm-quality').value,
    caption_language: document.getElementById('rm-lang').value,
    style_name: document.getElementById('rm-style').value}};
  try {{
    const r = await fetch('/remake-clip', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(body)}});
    const j = await r.json();
    if (!r.ok) {{ er.textContent = 'Error: ' + (j.detail || 're-make fail hua'); st.textContent = ''; return; }}
    st.textContent = 'Re-make ho gaya — v' + j.version + ' (' + j.resolution + '). Page reload ho raha hai…';
    setTimeout(() => location.reload(), 800);
  }} catch (err) {{ st.textContent = ''; er.textContent = 'Error: ' + err; }}
}});
document.getElementById('rm-save-trim').addEventListener('click', async () => {{
  const st = document.getElementById('rm-status'), er = document.getElementById('rm-error');
  er.textContent = ''; st.textContent = 'Trim save ho raha hai…';
  const body = {{start: Number(document.getElementById('rm-start').value),
    end: Number(document.getElementById('rm-end').value)}};
  try {{
    const r = await fetch('/app/videos/' + UID + '/clips/' + CID + '/edit', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(body)}});
    const j = await r.json();
    if (!r.ok) {{ er.textContent = 'Error: ' + (j.detail || 'trim save nahi hua'); st.textContent = ''; return; }}
    st.textContent = 'Trim saved ✓ — ab Re-make dabao ya page reload karo.';
  }} catch (err) {{ st.textContent = ''; er.textContent = 'Error: ' + err; }}
}});
</script>
"""


@app.get("/app/videos/{upload_id}", response_class=HTMLResponse)
def video_detail(request: Request, upload_id: str) -> HTMLResponse:
    """One video: all its clips with scores + Hand clip form."""
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    row = _get_upload_row(upload_id)
    if not row:
        return HTMLResponse("<h1>Video not found</h1>", status_code=404)
    if not _owns_upload(request, upload_id):
        return RedirectResponse("/login", status_code=303)
    try:
        clips_data = load_clips(upload_id)
        clips = clips_data.get("clips", [])
    except FileNotFoundError:
        clips_data, clips = {}, []
    probe = probe_video(ROOT / row["stored_path"])
    duration = probe["duration"] if probe["valid"] else 0.0
    # Phase 9: approval-gate card — the same state the server enforces
    # on POST /render-all (see workflow.get_project_status).
    try:
        _proj = get_project_status(upload_id)
        _gate_locked = bool(_proj.get("render_all_locked", True))
        _gate_msg = str(_proj.get("gate_message", ""))
        _approved_list = ", ".join(_proj.get("approved_clips") or [])
    except Exception:
        _gate_locked, _gate_msg, _approved_list = True, "Gate state nahi mila — pehle clips banao.", ""
    _gate_card = (f"<div class=\"card\"><h3>Render All — approval gate</h3>"
                  f"<p>{_html.escape(_gate_msg)}</p>"
                  + (f"<p class=\"muted\">Approved test clip: {_html.escape(_approved_list)}</p>" if _approved_list else "")
                  + (f"<button class=\"btn\" type=\"button\" id=\"render-all-btn\">Render All ({len(clips)} clips)</button>"
                     f"<p id=\"render-all-status\" class=\"muted\"></p><p id=\"render-all-error\" class=\"err\"></p>"
                     if not _gate_locked and clips else
                     "<p class=\"muted\">Render All tabhi chalega jab ek test clip approved ho — clip editor me kisi clip ko Approve dabao.</p>")
                  + "</div>")
    if clips:
        clip_rows = "".join(
            f"<tr><td><a href=\"/app/videos/{upload_id}/clips/{c['clip_id']}\">{c['clip_id']}</a></td>"
            f"<td>{c['start']}s – {c['end']}s</td><td>{c['duration']}s</td>"
            f"<td><b>{_fmt_score(c.get('combined_score', c.get('score')))}</b></td>"
            f"<td>{_fmt_score(c.get('words_score'))}</td>"
            f"<td>{_fmt_score(c.get('reaction_score'))}</td>"
            f"<td>{_html.escape(str(c.get('source') or 'ai'))}</td>"
            f"<td><a href=\"/app/videos/{upload_id}/clips/{c['clip_id']}\">Open editor →</a></td></tr>"
            for c in clips)
        table = ("<table><tr><th>Clip</th><th>Window</th><th>Length</th>"
                 "<th>Combined</th><th>Words</th><th>Reaction</th><th>Source</th><th></th></tr>"
                 + clip_rows + "</table>")
    else:
        table = ("<p class=\"muted\">Abhi koi clip nahi hai — classic studio "
                 "se pehle transcribe + find-clips karo, ya neeche Hand clip banao.</p>")
    body = f"""<h1>{_html.escape(str(row['original_name']))}</h1>
<p class="muted"><a href="/app/videos">← All videos</a> · {duration:.1f}s · {row['size_bytes'] // 1024} KB · {len(clips)} clips</p>
{_gate_card}
<div class="card"><h3>Clips</h3>{table}</div>
<div class="card"><h3>Hand clip — manual range, no AI gate</h3>
<p class="muted">Khud start/end do — AI judge nahi lagta. Scores sirf jaankari ke liye dikhte hain.</p>
<form id="hand-form">
<label>Start (seconds)<input id="hand-start" type="number" min="0" step="0.1" value="0" required></label>
<label>End (seconds)<input id="hand-end" type="number" min="0" step="0.1" value="30" required></label>
<label>Hook (optional)<input id="hand-hook" type="text" placeholder="Pehli line — hook"></label>
<p id="hand-status" class="muted"></p><p id="hand-error" class="err"></p>
<button class="btn" type="submit">Create hand clip</button></form>
<p class="muted">Banane ke baad editor khul jayega — wahan se render karo.</p></div>
<script>
document.getElementById('hand-form').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const st = document.getElementById('hand-status'), er = document.getElementById('hand-error');
  er.textContent = ''; st.textContent = 'Hand clip ban raha hai…';
  const body = {{upload_id: "{upload_id}",
    start: Number(document.getElementById('hand-start').value),
    end: Number(document.getElementById('hand-end').value),
    hook: document.getElementById('hand-hook').value || null}};
  try {{
    const r = await fetch('/hand-clip', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(body)}});
    const j = await r.json();
    if (!r.ok) {{ er.textContent = 'Error: ' + (j.detail || 'hand clip nahi bana'); st.textContent = ''; return; }}
    st.textContent = 'Hand clip ban gaya: ' + j.clip_id + ' — editor khul raha hai…';
    setTimeout(() => location.href = '/app/videos/{upload_id}/clips/' + j.clip_id, 600);
  }} catch (err) {{ st.textContent = ''; er.textContent = 'Error: ' + err; }}
}});
const raBtn = document.getElementById('render-all-btn');
if (raBtn) {{
  raBtn.addEventListener('click', async () => {{
    const st = document.getElementById('render-all-status'), er = document.getElementById('render-all-error');
    er.textContent = ''; st.textContent = 'Render All chal raha hai… har clip 1-3 min leta hai, page band mat karo.';
    raBtn.disabled = true;
    try {{
      const fd = new FormData(); fd.append('upload_id', '{upload_id}');
      const r = await fetch('/render-all', {{method:'POST', body: fd}});
      const j = await r.json();
      if (!r.ok) {{ er.textContent = 'Error: ' + (j.detail || 'render-all fail hua'); st.textContent = ''; raBtn.disabled = false; return; }}
      st.textContent = j.message || 'Render-all done.';
      setTimeout(() => location.reload(), 800);
    }} catch (err) {{ st.textContent = ''; er.textContent = 'Error: ' + err; raBtn.disabled = false; }}
  }});
}}
</script>
"""
    return _app_layout(user, "All videos", body)


@app.get("/app/videos/{upload_id}/source")
def video_source(request: Request, upload_id: str):
    """Stream the source upload (owner only) — clip preview ke liye."""
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    if not _owns_upload(request, upload_id):
        raise HTTPException(status_code=403, detail="Ye video tumhari nahi hai")
    path = ROOT / row["stored_path"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="uploaded file missing on disk")
    return FileResponse(path, media_type="video/mp4", filename=row["original_name"])


@app.get("/app/videos/{upload_id}/clips/{clip_id}", response_class=HTMLResponse)
def clip_detail(request: Request, upload_id: str, clip_id: str) -> HTMLResponse:
    """Clip editor page: preview, scores, metadata, trim, re-make, versions."""
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if not _get_upload_row(upload_id):
        return HTMLResponse("<h1>Video not found</h1>", status_code=404)
    if not _owns_upload(request, upload_id):
        return RedirectResponse("/login", status_code=303)
    _, clip = _clip_or_404(upload_id, clip_id)
    versions = list_render_versions(upload_id, clip_id)
    error = request.query_params.get("error", "")
    saved = request.query_params.get("saved", "") == "1"
    body = _clip_detail_body(user, upload_id, clip, versions,
                             error=error, saved=saved,
                             gate_unlocked=has_approved_clip(upload_id))
    return _app_layout(user, "All videos", body)


@app.get("/app/videos/{upload_id}/clips/{clip_id}/versions")
def clip_versions(request: Request, upload_id: str, clip_id: str) -> dict:
    """All render versions of one clip, as JSON (polling + tests)."""
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login karo — /login kholo")
    if not _owns_upload(request, upload_id):
        raise HTTPException(status_code=403, detail="Ye video tumhari nahi hai")
    _clip_or_404(upload_id, clip_id)
    versions = list_render_versions(upload_id, clip_id)
    return {"upload_id": upload_id, "clip_id": clip_id,
            "count": len(versions), "versions": versions}


@app.post("/app/videos/{upload_id}/clips/{clip_id}/edit")
async def clip_edit(request: Request, upload_id: str, clip_id: str):
    """Save edited metadata and/or trim for one clip (no render).

    Accepts a form POST (browser — redirects back to the detail page)
    or JSON (API — returns the updated clip). Trim is validated
    against the source video duration; bad trim -> 400, nothing saved.
    """
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login karo — /login kholo")
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    if not _owns_upload(request, upload_id):
        raise HTTPException(status_code=403, detail="Ye video tumhari nahi hai")
    _, clip = _clip_or_404(upload_id, clip_id)
    body = await _read_body(request)
    is_json = "application/json" in (request.headers.get("content-type") or "").lower()
    probe = probe_video(ROOT / row["stored_path"])
    duration = probe["duration"] if probe["valid"] else 0.0
    updates: dict = {}
    for field in ("hook_line", "title", "description", "hashtags"):
        if field in body and body[field] is not None and str(body[field]) != "":
            updates[field] = str(body[field])
    try:
        start, end = _validate_trim(body.get("start"), body.get("end"),
                                    duration, float(clip["start"]),
                                    float(clip["end"]))
    except (TypeError, ValueError) as exc:
        detail = str(exc)
        if is_json:
            raise HTTPException(status_code=400, detail=detail)
        return RedirectResponse(
            f"/app/videos/{upload_id}/clips/{clip_id}?error={detail}",
            status_code=303)
    if start != float(clip["start"]) or end != float(clip["end"]):
        updates["start"], updates["end"] = start, end
        updates["duration"] = round(end - start, 2)
    if not updates:
        if is_json:
            return {"message": "Kuch badla nahi — koi field nahi di.", "clip": clip}
        return RedirectResponse(
            f"/app/videos/{upload_id}/clips/{clip_id}?saved=1", status_code=303)
    try:
        updated = update_clip(upload_id, clip_id, updates)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if is_json:
        return {"message": "Clip saved.", "clip": updated}
    return RedirectResponse(
        f"/app/videos/{upload_id}/clips/{clip_id}?saved=1", status_code=303)


@app.post("/remake-clip")
async def remake_clip_endpoint(request: Request):
    """Edit (trim/metadata/options) + re-render in one call.

    Body (JSON or form): upload_id, clip_id, start, end, hook_line,
    title, description, hashtags, caption_style, layout, quality,
    style_name. Trim is validated first (400, nothing saved); then
    the clip is updated and rendered in a worker thread. The new
    render is a NEW version — the old render stays on disk.
    """
    body = await _read_body(request)
    upload_id = str(body.get("upload_id") or "")
    clip_id = str(body.get("clip_id") or "")
    if not upload_id or not clip_id:
        raise HTTPException(status_code=400,
                            detail="upload_id aur clip_id dono chahiye")
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    # Phase 6 hardening (Phase 5 flag): login + ownership required,
    # same rule as /render-clip (401 unauth, 403 other user's video).
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login karo — /login kholo")
    if not _owns_upload(request, upload_id):
        raise HTTPException(status_code=403, detail="Ye video tumhari nahi hai")
    video_path = ROOT / row["stored_path"]
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="uploaded file missing on disk")
    _, clip = _clip_or_404(upload_id, clip_id)
    try:
        transcript = load_transcript(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    probe = probe_video(video_path)
    duration = probe["duration"] if probe["valid"] else 0.0
    try:
        start, end = _validate_trim(body.get("start"), body.get("end"),
                                    duration, float(clip["start"]),
                                    float(clip["end"]))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    updates: dict = {"start": start, "end": end,
                     "duration": round(end - start, 2)}
    for field in ("hook_line", "title", "description", "hashtags"):
        if field in body and body[field] is not None and str(body[field]) != "":
            updates[field] = str(body[field])
    try:
        stored = dict((load_clips(upload_id).get("options") or {}))
        opts = parse_options({
            "caption_style": body.get("caption_style") or stored.get("caption_style"),
            "layout": body.get("layout") or stored.get("layout"),
            "quality": body.get("quality") or stored.get("quality"),
            "caption_language": (body.get("caption_language")
                                 or stored.get("caption_language")
                                 # Phase 9 fallback: see /render-clip —
                                 # the stored transcript's language wins
                                 # over the English default.
                                 or transcript.get("language_requested")),
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Phase 9: per-clip language switch — re-transcribe if needed.
    try:
        transcript, _retrans = await _ensure_transcript_language(
            video_path, upload_id, transcript, opts.get("caption_language"))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    try:
        clip = update_clip(upload_id, clip_id, updates)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    try:
        meta = await asyncio.to_thread(
            render_clip, video_path, upload_id, clip,
            style_name=str(body.get("style_name") or "global-baatien"),
            transcript=transcript, options=opts)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"message": f"Re-make ho gaya — version {meta.get('version')}.",
            "retranscribed_for_language": _retrans, **meta}


@app.post("/hand-clip")
async def hand_clip_endpoint(request: Request):
    """Create a MANUAL clip from [start, end] — no AI gate.

    Body (JSON or form): upload_id, start, end, hook, title,
    description, hashtags. Range is validated against the source
    video (400 on a bad range). Returns the created hand_XX clip —
    render it with POST /render-clip afterwards.
    """
    body = await _read_body(request)
    upload_id = str(body.get("upload_id") or "")
    if not upload_id:
        raise HTTPException(status_code=400, detail="upload_id chahiye")
    row = _get_upload_row(upload_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    video_path = ROOT / row["stored_path"]
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="uploaded file missing on disk")
    try:
        transcript = load_transcript(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    probe = probe_video(video_path)
    duration = probe["duration"] if probe["valid"] else 0.0
    try:
        start = float(body.get("start"))
        end = float(body.get("end"))
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400, detail="Hand clip ke liye start aur end (seconds) number me do")
    if start < 0 or end > duration or end <= start or end - start < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Hand clip range galat hai — start >= 0, end <= {duration:.1f}s, "
                   f"end > start, aur kam se kam 5s ka hona chahiye "
                   f"(mila: {start}s – {end}s)")
    try:
        clip = create_hand_clip(
            transcript, upload_id, video_path, start, end,
            hook=body.get("hook"), title=body.get("title"),
            description=body.get("description"),
            hashtags=body.get("hashtags"),
            language=transcript.get("language_requested") or "en")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Phase 6 ledger: hand clips get a "hand clip" row (0 tokens —
    # the scores are informational rule-based/loudness scores, and
    # the hook/title are the user's own words, not an LLM call).
    # Only logged-in users get a row (anonymous calls stay anonymous).
    try:
        _hand_user = _current_user(request)
        if _hand_user is not None:
            log_ai_call(DB_PATH, _hand_user["id"], "hand clip",
                        provider="—", model="—", upload_id=upload_id,
                        creator_id=row.get("creator_id"),
                        method="rule_based", status="ok",
                        note=(f"hand clip {clip.get('clip_id')} "
                              f"{start}s–{end}s, words "
                              f"{clip.get('words_score')}, reaction "
                              f"{clip.get('reaction_score')}"))
    except Exception:
        pass
    return {"message": "Hand clip ban gaya — AI gate nahi lagaya gaya.",
            **clip}


def _render_info_version(clip_id: str, upload_id: str, version: int) -> dict:
    """Load one exact render version (404 when it does not exist)."""
    try:
        return load_render_info(clip_id, upload_id=upload_id, version=version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/rendered/{upload_id}/{clip_id}/v/{version}")
def download_version_endpoint(upload_id: str, clip_id: str, version: int):
    """Download one exact render version (MP4)."""
    info = _render_info_version(clip_id, upload_id, version)
    path = Path(info["output_mp4"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="rendered file missing on disk")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/rendered/{upload_id}/{clip_id}/v/{version}/srt")
def download_version_srt_endpoint(upload_id: str, clip_id: str, version: int):
    """Download one exact render version's SRT captions."""
    info = _render_info_version(clip_id, upload_id, version)
    path = Path(info["output_srt"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="SRT file missing on disk")
    return FileResponse(path, media_type="text/plain", filename=path.name)


@app.get("/rendered/{upload_id}/{clip_id}/v/{version}/caption")
def download_version_caption_endpoint(upload_id: str, clip_id: str, version: int):
    """Download one exact render version's caption .txt (title + caption + hashtags)."""
    info = _render_info_version(clip_id, upload_id, version)
    path = Path(info["output_caption_txt"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="caption file missing on disk")
    return FileResponse(path, media_type="text/plain", filename=path.name)


@app.get("/app/usage", response_class=HTMLResponse)
def ai_usage(request: Request) -> HTMLResponse:
    """AI usage ledger — every AI call, with tokens. Last 30 days."""
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    s = usage_summary(DB_PATH, user["id"], days=30)
    max_day_tokens = max([d["tokens"] for d in s["per_day"]] + [1])
    day_bars = "".join(
        f'<div class="bar" style="height:{(d["tokens"] / max_day_tokens * 88 + 2):.0f}px"'
        f' title="{d["day"]}: {d["tokens"]} tokens, {d["calls"]} calls"></div>'
        for d in s["per_day"])
    day_labels = "".join(
        f"<span style=\"flex:1;font-size:10px;color:#a5b4fc\">"
        f"{d['day'][5:]}</span>" for d in s["per_day"][::7])
    if s["by_purpose"]:
        purpose_rows = "".join(
            f"<tr><td>{_html.escape(str(p['purpose']))}</td>"
            f"<td>{p['calls']}</td><td>{p['tokens']}</td></tr>"
            for p in s["by_purpose"])
        purpose_html = ("<table><tr><th>Used for</th><th>Calls</th>"
                        "<th>Tokens</th></tr>" + purpose_rows + "</table>")
    else:
        purpose_html = "<p class=\"muted\">No AI usage in this period yet.</p>"
    if s["by_creator"]:
        creator_rows = "".join(
            f"<tr><td>{_html.escape(str(c['name']))}</td>"
            f"<td>{c['calls']}</td><td>{c['tokens']}</td></tr>"
            for c in s["by_creator"])
        creator_html = ("<table><tr><th>Creator</th><th>Calls</th>"
                        "<th>Tokens</th></tr>" + creator_rows + "</table>")
    else:
        creator_html = "<p class=\"muted\">No AI usage in this period yet.</p>"
    if s["recent"]:
        recent_rows = "".join(
            f"<tr><td>{_html.escape(str(r['created_at'][:16]).replace('T', ' '))}</td>"
            f"<td>{_html.escape(str(r['purpose']))}</td>"
            f"<td>{_html.escape(str(r['provider'] or '—'))}"
            f"{(' / ' + _html.escape(str(r['model']))) if r['model'] else ''}</td>"
            f"<td>{_html.escape(str(r['upload_name']))}</td>"
            f"<td>{_html.escape(str(r['creator_name']))}</td>"
            f"<td>{r['total_tokens']}"
            f"{' (est.)' if r['token_source'] == 'estimated' else ''}</td>"
            f"<td>{_html.escape(str(r['method']))}</td>"
            f"<td>{_html.escape(str(r['status']))}</td></tr>"
            for r in s["recent"])
        recent_html = ("<table><tr><th>When</th><th>Used for</th>"
                       "<th>Provider / model</th><th>Video</th><th>Creator</th>"
                       "<th>Tokens</th><th>Method</th><th>Status</th></tr>"
                       + recent_rows + "</table>"
                       + "<p class=\"muted\">\"(est.)\" = provider ne usage "
                         "nahi diya, tokens estimate hain. \"finding clips\" "
                         "rows carry 0 tokens by design — unke tokens "
                         "\"words judge\" row par hain, taaki total "
                         "double-count na ho.</p>")
    else:
        recent_html = "<p class=\"muted\">No AI usage in this period yet.</p>"
    body = f"""<h1>AI usage</h1>
<p class="muted">Every AI call ClipCut makes, and the tokens it uses. Last 30 days.</p>
<div class="tiles"><div class="tile"><b>{s['calls']}</b><span>AI calls</span></div>
<div class="tile"><b>{s['tokens']}</b><span>Tokens</span></div>
<div class="tile"><b>{s['input_tokens']}</b><span>Input tokens</span></div>
<div class="tile"><b>{s['output_tokens']}</b><span>Output tokens</span></div></div>
<div class="card"><h3>Tokens per day — Last 30 days</h3>
<div class="bar-row">{day_bars}</div>
<div style="display:flex;gap:6px">{day_labels}</div></div>
<div class="card"><h3>What it was used for</h3>{purpose_html}</div>
<div class="card"><h3>By creator</h3>{creator_html}</div>
<div class="card"><h3>Recent AI activity — Latest {len(s['recent'])} calls</h3>{recent_html}</div>"""
    return _app_layout(user, "AI usage", body)


@app.get("/app/notifications", response_class=HTMLResponse)
def notifications(request: Request) -> HTMLResponse:
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    # Phase 10: real attention list — clips awaiting a decision, queue
    # problems, live sessions. Nothing is invented; empty = "Nothing yet".
    rows: list[str] = []
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            uploads = conn.execute(
                "SELECT id, original_name FROM uploads WHERE user_id=?",
                (user["id"],)).fetchall()
        for uid, name in uploads:
            try:
                clips = load_clips(uid).get("clips", [])
            except Exception:
                clips = []
            if not clips:
                continue
            approvals = get_clip_approvals(uid)
            pending = [c for c in clips
                       if approvals.get(c.get("clip_id"), {}).get("status")
                       not in ("approved", "rejected")]
            if pending:
                rows.append(
                    f"<p>🎬 <b>{len(pending)} clip(s) need a decision</b> — "
                    f"{_html.escape(name or uid)}: approve a test clip to "
                    f"unlock Render All. "
                    f"<a href=\"/app/videos/{uid}\">Open video</a></p>")
    except Exception:
        pass
    try:
        for status, label in (("failed", "failed to post"),
                              ("waiting_account", "waiting for a YouTube account"),
                              ("not_set_up", "cannot post — platform not set up")):
            items = list_queue(DB_PATH, user["id"], status=status)
            for it in items:
                rows.append(
                    f"<p>📮 <b>{_html.escape(it.get('title') or it.get('clip_id', ''))}</b> "
                    f"{label} ({PLATFORMS.get(it.get('platform'), {}).get('label', it.get('platform'))}). "
                    f"<a href=\"/app/post-queue\">Open post queue</a></p>")
    except Exception:
        pass
    try:
        live = [s for s in list_live_sessions(DB_PATH, user["id"])
                if s and s["status"] in ("live", "stopping")]
        for s in live:
            rows.append(
                f"<p>🔴 <b>Live session running</b> — "
                f"{_html.escape(s.get('title') or s['session_id'])}: "
                f"{s.get('chunks_count', 0)} chunks, {s.get('clips_count', 0)} clips. "
                f"<a href=\"/app/live/{s['session_id']}\">Open session</a></p>")
    except Exception:
        pass
    if rows:
        body = ("<h1>Notifications</h1><div class=\"card\">"
                "<p class=\"muted\">Clips needing attention — newest first "
                "nahi, bas jo abhi action mangta hai.</p>" + "".join(rows) + "</div>")
    else:
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
    err = request.query_params.get("error", "")
    ok = request.query_params.get("connected", "") == "1"
    disc = request.query_params.get("disconnected", "") == "1"
    try:
        yt = get_youtube_account(DB_PATH, user["id"])
    except Exception:
        yt = None
    err_html = f'<p class="err">{_html.escape(err)}</p>' if err else ""
    ok_html = ('<p style="color:#4ade80">YouTube connected ✓ — ab queue '
               'kari hui Shorts post hongi.</p>' if ok else "")
    disc_html = ('<p class="muted">YouTube disconnected — queued Shorts '
                 '"waiting for account" me jayengi.</p>' if disc else "")
    if yt:
        yt_block = (f"<p><b>YouTube</b> — Post Shorts from here.</p>"
                    f"<p>✅ Connected: <b>{_html.escape(yt['channel_name'] or yt['channel_id'])}</b> "
                    f"<span class=\"muted\">({yt['status']})</span></p>"
                    f"<form method=\"post\" action=\"/app/settings/apps/youtube/disconnect\" "
                    f"onsubmit=\"return confirm('YouTube disconnect karo? Queued Shorts ruk jayengi.');\">"
                    f"<button class=\"btn ghost\" type=\"submit\">Disconnect YouTube</button></form>")
    elif youtube_oauth_configured():
        yt_block = ("<p><b>YouTube</b> — Post Shorts from here.</p>"
                    "<p class=\"muted\">Not connected yet.</p>"
                    "<a class=\"btn\" href=\"/app/settings/apps/youtube/connect\">"
                    "Connect YouTube</a>")
    else:
        missing = ", ".join(youtube_oauth_missing())
        yt_block = ("<p><b>YouTube</b> — Post Shorts from here.</p>"
                    "<p class=\"muted\">Connect ke liye Google OAuth client chahiye — "
                    f"yeh values set karo (Google Cloud Console se): <b>{_html.escape(missing)}</b> "
                    "(+ usi project me YouTube Data API v3 enable karo). "
                    "Tab tak queued Shorts \"waiting for account\" me rahengi.</p>")
    body = (f"<h1>Connected apps</h1>{err_html}{ok_html}{disc_html}"
            f"<div class=\"card\">{yt_block}</div>"
            "<div class=\"card\">"
            "<p><b>Instagram</b> — Post Reels from here. "
            "<span class=\"muted\">Not set up on this server.</span></p>"
            "<p><b>TikTok</b> — Post to TikTok from here. "
            "<span class=\"muted\">Not set up on this server.</span></p>"
            "<p class=\"muted\">Instagram — Post Reels from here. "
            "<span class=\"muted\">Not set up on this server.</span></p>"
            "<p><b>TikTok</b> — Post to TikTok from here. "
            "<span class=\"muted\">Not set up on this server.</span></p>"
            "<p class=\"muted\">Dono ke liye official API + approval "
            "chahiye, isliye Phase 8 me sirf YouTube wired hai.</p></div>")
    return _app_layout(user, "Settings", body)


# ---------------------------------------------------------- Phase 8: posting queue

def _queue_item_for_user(request: Request, item_id: str) -> tuple[dict | None, dict | None]:
    """(item, error) — error is {"status": 404|403, "detail": ...}."""
    item = get_queue_item(DB_PATH, item_id)
    if not item:
        return None, {"status": 404, "detail": f"queue item nahi mila: {item_id}"}
    user = _current_user(request)
    if not user or user["id"] != item["user_id"]:
        return None, {"status": 403,
                      "detail": "Ye queue item tumhara nahi hai"}
    return item, None


def _post_queue_body(user: dict, items: list[dict], paused_all: bool,
                     notice: str = "") -> str:
    """HTML body for /app/post-queue."""
    counts = queue_counts(DB_PATH, user["id"])
    notice_html = (f'<p style="color:#4ade80">{_html.escape(notice)}</p>'
                   if notice else "")
    groups = [("Queued — due items apne time par post honge",
               [i for i in items if i["status"] in ("queued", "waiting_account")]),
              ("Not set up — is platform par posting configured nahi hai",
               [i for i in items if i["status"] == "not_set_up"]),
              ("Posted ✓ (sirf asli platform URL ke saath)",
               [i for i in items if i["status"] == "posted"]),
              ("Failed — Retry dabakar dobara try karo",
               [i for i in items if i["status"] == "failed"]),
              ("Cancelled",
               [i for i in items if i["status"] == "cancelled"])]
    sections = ""
    for title, group in groups:
        if not group:
            continue
        rows = ""
        for it in group:
            platform = PLATFORMS[it["platform"]]["label"]
            sched = format_ist(it["scheduled_at"])
            status_chip = it["status"]
            if it["paused"]:
                status_chip += " · ⏸ paused"
            posted = (f"<a href=\"{_html.escape(it['posted_url'])}\">"
                      f"{_html.escape(it['posted_url'])}</a>"
                      if it.get("posted_url") else "—")
            err = (_html.escape(it["last_error"]) if it.get("last_error")
                   else "—")
            actions = ""
            if it["status"] in ("queued", "waiting_account", "not_set_up"):
                if it["paused"]:
                    actions += (f"<form method=\"post\" action=\"/app/post-queue/{it['id']}/resume\" "
                                f"style=\"display:inline\"><button class=\"btn ghost\" type=\"submit\">Resume</button></form> ")
                else:
                    actions += (f"<form method=\"post\" action=\"/app/post-queue/{it['id']}/pause\" "
                                f"style=\"display:inline\"><button class=\"btn ghost\" type=\"submit\">Pause</button></form> ")
                actions += (f"<form method=\"post\" action=\"/app/post-queue/{it['id']}/cancel\" "
                            f"style=\"display:inline\"><button class=\"btn ghost\" type=\"submit\">Cancel</button></form> ")
            if it["status"] in ("failed", "waiting_account", "not_set_up"):
                actions += (f"<form method=\"post\" action=\"/app/post-queue/{it['id']}/retry\" "
                            f"style=\"display:inline\"><button class=\"btn ghost\" type=\"submit\">Retry</button></form> ")
            rows += (f"<tr><td>{_html.escape(it['title'])}</td>"
                     f"<td>{platform}</td><td>{sched}</td>"
                     f"<td>{_html.escape(it['recurrence'])}</td>"
                     f"<td>{_html.escape(status_chip)}</td>"
                     f"<td>{posted}</td><td>{err}</td><td>{actions}</td></tr>")
        sections += (f"<div class=\"card\"><h3>{title} ({len(group)})</h3>"
                     f"<table><tr><th>Title</th><th>Platform</th><th>Post at (IST)</th>"
                     f"<th>Repeat</th><th>Status</th><th>Posted URL</th>"
                     f"<th>Last error</th><th>Actions</th></tr>{rows}</table></div>")
    if not sections:
        sections = ("<div class=\"card\"><p class=\"muted\">Abhi kuch queued nahi hai — "
                    "kisi clip ke editor page se \"Queue for posting\" dabao.</p></div>")
    pause_label = "Resume all posting" if paused_all else "Pause all posting"
    pause_note = ("<p class=\"muted\">⏸ Paused — worker kuch post nahi karega.</p>"
                  if paused_all else
                  "<p class=\"muted\">▶ Posting on — due items apne time par post honge.</p>")
    return (f"<h1>Post queue</h1>{notice_html}"
            f"<p class=\"muted\">{counts['waiting']} waiting · "
            f"{counts['posted_total']} posted · {counts['failed']} failed · "
            f"aaj {counts['posted_today']} posted</p>"
            f"{pause_note}"
            f"<form method=\"post\" action=\"/app/post-queue/pause-all\" style=\"display:inline\">"
            f"<button class=\"btn ghost\" type=\"submit\">{pause_label}</button></form> "
            f"<a class=\"btn ghost\" href=\"/app/settings/apps\">Connected apps</a>"
            f"{sections}")


@app.get("/app/post-queue", response_class=HTMLResponse)
def post_queue_page(request: Request) -> HTMLResponse:
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    notice = ""
    if request.query_params.get("queued") == "1":
        notice = "Queued ✓ — scheduled time par post hoga (ya \"waiting for account\" me rahega jab tak YouTube connect na ho)."
    items = list_queue(DB_PATH, user["id"])
    body = _post_queue_body(user, items, is_paused_all(DB_PATH, user["id"]),
                            notice=notice)
    return _app_layout(user, "Post queue", body)


@app.post("/app/post-queue")
async def post_queue_create(request: Request):
    """Queue a rendered clip version for posting (JSON or form).

    Body: upload_id, clip_id, version (default latest), platform,
    title, description, hashtags, privacy, scheduled_at
    (datetime-local, Asia/Kolkata), recurrence. The render version
    must exist — the worker uploads that exact MP4 file.
    """
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login karo — /login kholo")
    body = await _read_body(request)
    is_json = "application/json" in (request.headers.get("content-type") or "").lower()
    upload_id = str(body.get("upload_id") or "")
    clip_id = str(body.get("clip_id") or "")
    if not upload_id or not clip_id:
        raise HTTPException(status_code=400, detail="upload_id aur clip_id chahiye")
    if not _get_upload_row(upload_id):
        raise HTTPException(status_code=404, detail=f"upload not found: {upload_id}")
    if not _owns_upload(request, upload_id):
        raise HTTPException(status_code=403, detail="Ye video tumhari nahi hai")
    try:
        _, clip = _clip_or_404(upload_id, clip_id)
    except HTTPException as exc:
        raise exc
    try:
        version = int(body.get("version") or 0) or None
        meta = find_render_info(upload_id, clip_id, version=version)
    except Exception:
        meta = None
    if meta is None:
        # find_render_info returns None (not an exception) when no
        # render exists — check explicitly, or this becomes a 500.
        raise HTTPException(
            status_code=404,
            detail=f"Rendered clip nahi mila: {clip_id} — pehle render karo, "
                   "queue ke liye rendered MP4 chahiye")
    version = int(meta.get("version") or 1)
    platform = str(body.get("platform") or "youtube")
    title = str(body.get("title") or clip.get("title")
                or clip.get("hook_line") or "")
    description = str(body.get("description") or clip.get("description") or "")
    hashtags = str(body.get("hashtags") or clip.get("hashtags") or "")
    privacy = str(body.get("privacy") or "unlisted")
    recurrence = str(body.get("recurrence") or "once")
    try:
        scheduled = parse_ist_datetime_local(str(body.get("scheduled_at") or ""))
    except ValueError as exc:
        detail = str(exc)
        if is_json:
            raise HTTPException(status_code=400, detail=detail)
        return RedirectResponse(
            f"/app/videos/{upload_id}/clips/{clip_id}?error={detail}",
            status_code=303)
    try:
        item = queue_clip(DB_PATH, user["id"], upload_id, clip_id, version,
                          platform, title, description=description,
                          hashtags=hashtags, privacy=privacy,
                          scheduled_at=scheduled, recurrence=recurrence)
    except ValueError as exc:
        detail = str(exc)
        if is_json:
            raise HTTPException(status_code=400, detail=detail)
        return RedirectResponse(
            f"/app/videos/{upload_id}/clips/{clip_id}?error={detail}",
            status_code=303)
    if is_json:
        return JSONResponse(status_code=201, content={"message": "Queued.",
                                                      "item": item})
    return RedirectResponse("/app/post-queue?queued=1", status_code=303)


@app.post("/app/post-queue/{item_id}/pause")
async def post_queue_pause(request: Request, item_id: str):
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login karo — /login kholo")
    item, err = _queue_item_for_user(request, item_id)
    if err:
        raise HTTPException(status_code=err["status"], detail=err["detail"])
    return _post_queue_action_response(request, pause_item(DB_PATH, item_id))


@app.post("/app/post-queue/{item_id}/resume")
async def post_queue_resume(request: Request, item_id: str):
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login karo — /login kholo")
    item, err = _queue_item_for_user(request, item_id)
    if err:
        raise HTTPException(status_code=err["status"], detail=err["detail"])
    return _post_queue_action_response(request, resume_item(DB_PATH, item_id))


@app.post("/app/post-queue/{item_id}/cancel")
async def post_queue_cancel(request: Request, item_id: str):
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login karo — /login kholo")
    item, err = _queue_item_for_user(request, item_id)
    if err:
        raise HTTPException(status_code=err["status"], detail=err["detail"])
    return _post_queue_action_response(request, cancel_item(DB_PATH, item_id))


@app.post("/app/post-queue/{item_id}/retry")
async def post_queue_retry(request: Request, item_id: str):
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login karo — /login kholo")
    item, err = _queue_item_for_user(request, item_id)
    if err:
        raise HTTPException(status_code=err["status"], detail=err["detail"])
    return _post_queue_action_response(request, retry_item(DB_PATH, item_id))


def _post_queue_action_response(request: Request, item: dict | None):
    """After a queue action: JSON for API callers, redirect for forms."""
    is_json = "application/json" in (request.headers.get("content-type") or "").lower()
    if is_json:
        if item is None:
            raise HTTPException(status_code=404, detail="queue item nahi mila")
        return {"message": "OK.", "item": item}
    return RedirectResponse("/app/post-queue", status_code=303)


@app.post("/app/post-queue/pause-all")
async def post_queue_pause_all(request: Request):
    """Toggle Pause all posting for this user (form or JSON)."""
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login karo — /login kholo")
    body = await _read_body(request)
    is_json = "application/json" in (request.headers.get("content-type") or "").lower()
    if "paused" in body:
        paused = str(body.get("paused")).lower() in ("1", "true", "yes", "on")
    else:
        paused = not is_paused_all(DB_PATH, user["id"])
    set_pause_all(DB_PATH, user["id"], paused)
    if is_json:
        return {"paused_all": paused,
                "message": ("Posting paused — worker kuch post nahi karega."
                            if paused else "Posting resumed.")}
    return RedirectResponse("/app/post-queue", status_code=303)


@app.get("/app/settings/apps/youtube/connect")
def youtube_connect(request: Request):
    """Start the YouTube OAuth flow (youtube.upload scope).

    Fails cleanly with a message when the OAuth client is not
    configured — no credentials are invented.
    """
    from urllib.parse import quote as _quote
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    try:
        url = youtube_auth_url(user["id"])
    except RuntimeError as exc:
        return RedirectResponse(
            f"/app/settings/apps?error={_quote(str(exc))}", status_code=303)
    return RedirectResponse(url, status_code=303)


@app.get("/auth/youtube/callback")
def youtube_callback(request: Request):
    """Google redirects here after the user approves the
    youtube.upload scope. UNTESTED live — no OAuth client exists
    yet, so this path cannot be exercised until the user supplies
    GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / YOUTUBE_REDIRECT_URI
    (see README). With them missing it fails cleanly."""
    from urllib.parse import quote as _quote
    err = request.query_params.get("error", "")
    if err:
        return RedirectResponse(
            f"/app/settings/apps?error={_quote('Google ne access deny kiya: ' + err)}",
            status_code=303)
    code = request.query_params.get("code", "")
    state = request.query_params.get("state", "")
    user_id = consume_oauth_state(state)
    if not user_id or not code:
        return RedirectResponse(
            "/app/settings/apps?error=" + _quote(
                "YouTube connect fail hua — state/code missing (dobara Connect dabao)."),
            status_code=303)
    try:
        tokens = exchange_youtube_code(code)
        channel = get_youtube_channel(tokens["access_token"])
        save_youtube_account(DB_PATH, user_id, tokens, channel)
    except Exception as exc:
        return RedirectResponse(
            f"/app/settings/apps?error={_quote(str(exc))}", status_code=303)
    return RedirectResponse("/app/settings/apps?connected=1", status_code=303)


@app.post("/app/settings/apps/youtube/disconnect")
def youtube_disconnect(request: Request):
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    delete_youtube_account(DB_PATH, user["id"])
    return RedirectResponse("/app/settings/apps?disconnected=1",
                            status_code=303)


@app.get("/app/settings/ai", response_class=HTMLResponse)
def ai_settings(request: Request) -> HTMLResponse:
    """Settings/AI — BYOK: provider, model, per-account key (masked)."""
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    saved = get_ai_settings(DB_PATH, user["id"])
    cfg = get_effective_ai_config(DB_PATH, user["id"])
    err = request.query_params.get("error", "")
    saved_flag = request.query_params.get("saved", "") == "1"
    provider = saved["provider"] if saved else "gemini"
    model = saved["model"] if saved else PROVIDERS["gemini"]["default_model"]
    custom_url = saved["custom_base_url"] if saved else ""
    if saved and saved["has_key"]:
        key_line = (f"Saved, ends in <b>{_html.escape(saved['key_last4'])}</b> — "
                    "paste a new key to replace it. Full key kabhi dikhayi nahi jati.")
    else:
        key_line = ("No key saved — fresh account par koi key pre-filled "
                    "nahi hoti. Key ke bina rule-based scoring chalega.")
    status = (f"Ready — {PROVIDERS[cfg['provider']]['label']} / "
              f"{_html.escape(cfg['model'])} "
              f"({'key saved' if cfg['has_key'] else 'no key — rule-based'})")
    radios = "".join(
        f'<label><input type="radio" name="provider" value="{pid}"'
        f'{" checked" if pid == provider else ""}> <b>{info["label"]}</b> '
        f'<span class="muted">— {info["note"]}</span></label>'
        for pid, info in PROVIDERS.items())
    err_html = f'<p class="err">{_html.escape(err)}</p>' if err else ""
    saved_html = ('<p style="color:#4ade80">Saved.</p>'
                  if saved_flag else "")
    body = f"""<h1>AI settings</h1><div class="card">
<p class="muted">The AI that finds the moments and writes the hooks. Bring your own key. Nothing here is shared.</p>
<p>Status: <b>{status}</b></p>{err_html}{saved_html}
<form method="post" action="/app/settings/ai">
<label style="display:block;font-weight:700">Provider</label>
<div class="radio">{radios}</div>
<label>Model<input name="model" value="{_html.escape(model)}" required>
<span class="muted">Apne provider ka model id — "Find models" se list bhi le sakte ho (key chahiye, Claude ko chhod kar).</span></label>
<button class="btn ghost" type="button" id="find-models">Find models</button>
<p id="models-msg" class="muted"></p>
<label>Custom base URL (Custom provider ke liye)<input name="custom_base_url" value="{_html.escape(custom_url)}" placeholder="https://your-endpoint/v1"></label>
<label>API key<input name="api_key" type="password" autocomplete="off" placeholder="Paste your key — khaali chhodo to purani key rehti hai">
<span class="muted">{key_line}</span></label>
<button class="btn" type="submit">Save AI settings</button></form>
<form method="post" action="/app/settings/ai/remove-key" style="margin-top:10px"
onsubmit="return confirm('Saved key hata di jaye? Rule-based scoring chalega.');">
<button class="btn ghost" type="submit">Remove saved key</button></form>
<p class="muted">Key server par encrypted save hoti hai (Fernet, server secret se) — database
mein kabhi plaintext nahi rehti. Key kabhi chat ya file me mat likhna.</p></div>
<div class="card"><h3>Provider status</h3>
<p class="muted">Gemini: works. OpenAI / Grok / Groq / OpenRouter / Custom:
OpenAI-compatible adapter implemented, live-untested (koi key nahi hai).
Claude: messages API implemented, live-untested, no list-models API.
Ollama: local only, is machine par Ollama nahi chal raha (untested).</p></div>
<script>
document.getElementById('find-models').addEventListener('click', async () => {{
  const msg = document.getElementById('models-msg');
  const prov = (document.querySelector('input[name="provider"]:checked') || {{}}).value || 'gemini';
  msg.textContent = 'Models dhoondh raha hoon…';
  try {{
    const r = await fetch('/app/settings/ai/models?provider=' + encodeURIComponent(prov));
    const j = await r.json();
    if (!r.ok) {{ msg.textContent = 'Error: ' + (j.detail || 'list nahi mili'); return; }}
    if (j.models && j.models.length) {{
      msg.textContent = j.message + ': ' + j.models.slice(0, 20).join(', ');
    }} else {{ msg.textContent = j.message || 'Koi model nahi mila.'; }}
  }} catch (err) {{ msg.textContent = 'Error: ' + err; }}
}});
</script>"""
    return _app_layout(user, "Settings", body)


@app.post("/app/settings/ai")
def ai_settings_save(request: Request, provider: str = Form(...),
                     model: str = Form(default=""),
                     api_key: str = Form(default=""),
                     custom_base_url: str = Form(default="")):
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    try:
        save_ai_settings(DB_PATH, user["id"], provider, model=model,
                         api_key=api_key or None,
                         custom_base_url=custom_base_url or None)
    except ValueError as exc:
        from urllib.parse import quote as _q
        return RedirectResponse(f"/app/settings/ai?error={_q(str(exc))}",
                                status_code=303)
    return RedirectResponse("/app/settings/ai?saved=1", status_code=303)


@app.post("/app/settings/ai/remove-key")
def ai_settings_remove_key(request: Request):
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    clear_ai_key(DB_PATH, user["id"])
    return RedirectResponse("/app/settings/ai?saved=1", status_code=303)


@app.get("/app/settings/ai/models")
def ai_models_list(request: Request, provider: str = "") -> dict:
    """'Find models' — real provider list, never a fake one.

    Uses the saved key for that provider when one exists (else the
    env Gemini key for gemini, else no key). Claude has no list API
    and Ollama needs a local server — both say so plainly.
    """
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login karo — /login kholo")
    provider = (provider or "gemini").strip().lower()
    if provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    saved = get_ai_settings(DB_PATH, user["id"])
    if saved and saved["provider"] == provider:
        cfg = get_effective_ai_config(DB_PATH, user["id"])
    elif provider == "gemini":
        cfg = get_effective_ai_config(DB_PATH, None)
    else:
        cfg = {"provider": provider,
               "model": PROVIDERS[provider]["default_model"],
               "api_key": None,
               "base_url": PROVIDERS[provider].get("base_url") or "",
               "source": "none", "has_key": False}
    models, message = list_models_for_provider(cfg)
    return {"provider": provider, "models": models, "message": message}


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
    raw_options = payload.get("options") or {}
    if not isinstance(raw_options, dict):
        raise HTTPException(status_code=400, detail="options ek object hona chahiye")
    try:
        opts = parse_options(raw_options)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Phase 9: the Options panel's caption language drives the job's
    # transcription language (payload `language` still wins if sent).
    language = normalize_language(
        str(payload.get("language") or opts.get("caption_language") or "en"))
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


# ---------------------------------------------------------- Phase 7: live recorder

def _live_record_body(creators: list[dict]) -> str:
    """The 'Record live' page. Plain string (not an f-string) so the
    inline JS braces need no escaping; only {CREATOR_OPTIONS} is
    substituted. MediaRecorder segments are made by STOP/START every
    60s (not timeslice blobs) — only that way is each chunk a
    self-contained playable file the server can ffprobe."""
    creator_options = "".join(
        f'<option value="{c["id"]}">{_html.escape(c["name"])}</option>'
        for c in creators)
    return """
<h1>🔴 Record live</h1>
<p class="muted">Apni camera/mic ya screen record karo — har 60 second ka chunk
server par jaata hai, transcribe hota hai, dual-judge se score hota hai aur
captioned clip ban jaata hai — <b>session khatam hone se pehle</b>. Stop dabate
hi saare chunks ek video me jud jaate hain.</p>
<div class="card"><h3>Start a live session</h3>
<label>Title<input id="live-title" type="text" placeholder="e.g. Friday live stream"></label>
<div style="display:flex;gap:10px">
<label style="flex:1">Creator
<select id="live-creator"><option value="">No creator</option>{CREATOR_OPTIONS}</select></label>
<label style="flex:1">Source
<select id="live-source"><option value="camera">Camera + mic</option><option value="screen">Screen (+ tab audio, Chrome)</option></select></label></div>
<div style="display:flex;gap:10px">
<label style="flex:1">How many clips per chunk?
<input id="live-count" type="number" min="1" max="8" value="2"></label>
<label style="flex:1">Caption style
<select id="live-caption"><option value="bounce">Bounce (yellow)</option><option value="hype">Hype (pink)</option><option value="clean">Clean (white)</option></select></label>
<label style="flex:1">Quality
<select id="live-quality"><option value="720p">720p (fast)</option><option value="1080p">1080p (slower)</option></select></label></div>
<p class="muted">Har segment ~60s ka hota hai; segments ke beech ~0.2s ka chhota gap
reh sakta hai (MediaRecorder stop/start). Ek chunk ek baar me process hota hai —
Render Free 512 MB par yahi safe hai, isliye clips thode peeche reh sakte hain.</p>
<p id="live-error" class="err"></p>
<button class="btn" id="live-start" type="button">Start recording</button>
<button class="btn ghost" id="live-stop" type="button" disabled>Stop &amp; finalise</button>
<a class="btn ghost" href="/app/live">All live sessions</a></div>
<div class="card"><h3>Session status</h3>
<p id="live-status" class="muted">Not recording.</p>
<p id="live-clips" class="muted"></p>
<video id="live-preview" autoplay muted playsinline style="width:100%;max-width:480px;border-radius:8px;background:#000"></video></div>
<script>
let liveStream = null, liveRecorder = null, liveSession = null;
let liveStill = false, liveSegTimer = null, livePollTimer = null;
let liveChunksSent = 0, liveStartedAt = 0;
const $ = (id) => document.getElementById(id);
function pickMime() {
  const cands = ['video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm',''];
  for (const m of cands) { if (!m || MediaRecorder.isTypeSupported(m)) return m; }
  return '';
}
async function uploadChunk(blob) {
  const fd = new FormData();
  fd.append('chunk', blob, 'chunk.webm');
  const r = await fetch('/live/' + liveSession + '/chunk', {method:'POST', body: fd});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || 'chunk upload nahi hua');
  liveChunksSent++;
  $('live-status').textContent = 'Recording… ' + liveChunksSent + ' chunk(s) sent · ' +
    Math.floor((Date.now() - liveStartedAt) / 1000) + 's elapsed';
}
function startSegment() {
  if (!liveStill) return;
  liveRecorder = new MediaRecorder(liveStream, {mimeType: pickMime()});
  liveRecorder.ondataavailable = async (e) => {
    if (e.data && e.data.size > 0) {
      try { await uploadChunk(e.data); }
      catch (err) { $('live-error').textContent = 'Chunk error: ' + err; }
    }
  };
  liveRecorder.onstop = () => {
    clearTimeout(liveSegTimer);
    if (liveStill) startSegment();  // next 60s segment
  };
  liveRecorder.start();
  liveSegTimer = setTimeout(() => {
    if (liveRecorder && liveRecorder.state === 'recording') liveRecorder.stop();
  }, 60000);
}
async function pollSession() {
  if (!liveSession) return;
  try {
    const r = await fetch('/live/' + liveSession);
    const j = await r.json();
    if (r.ok) {
      $('live-clips').textContent = j.chunks_done + ' chunk(s) processed · ' +
        j.clips_count + ' clip(s) ready — ' +
        j.clips.map(c => c.clip_id + ' (' + (c.score ?? '?') + ')').join(', ');
    }
  } catch (e) {}
}
$('live-start').addEventListener('click', async () => {
  $('live-error').textContent = '';
  const title = $('live-title').value || 'Live recording';
  const creatorId = $('live-creator').value || null;
  const source = $('live-source').value;
  const options = {
    clip_count: Number($('live-count').value) || 2,
    min_sec: 15, max_sec: 60,
    caption_style: $('live-caption').value || 'bounce',
    layout: 'blurred_fill', quality: $('live-quality').value || '720p',
    video_type: 'auto', from_min: null, to_min: null,
  };
  try {
    const r = await fetch('/live/start', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({title, creator_id: creatorId, options, language:'en'})});
    const j = await r.json();
    if (!r.ok) { $('live-error').textContent = 'Error: ' + (j.detail || 'session start nahi hui'); return; }
    liveSession = j.session_id;
    if (source === 'screen') {
      liveStream = await navigator.mediaDevices.getDisplayMedia({video:true, audio:true});
    } else {
      liveStream = await navigator.mediaDevices.getUserMedia({video:true, audio:true});
    }
    $('live-preview').srcObject = liveStream;
    liveStill = true; liveChunksSent = 0; liveStartedAt = Date.now();
    $('live-start').disabled = true; $('live-stop').disabled = false;
    $('live-status').textContent = 'Recording… session ' + liveSession;
    startSegment();
    livePollTimer = setInterval(pollSession, 10000);
  } catch (err) {
    $('live-error').textContent = 'Error: ' + err;
  }
});
$('live-stop').addEventListener('click', async () => {
  liveStill = false;
  clearTimeout(liveSegTimer); clearInterval(livePollTimer);
  try { if (liveRecorder && liveRecorder.state === 'recording') liveRecorder.stop(); } catch (e) {}
  if (liveStream) liveStream.getTracks().forEach(t => t.stop());
  if (!liveSession) return;
  $('live-status').textContent = 'Finalising… chunks jud rahe hain, thoda time lagega.';
  const r = await fetch('/live/' + liveSession + '/stop', {method:'POST'});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) { $('live-error').textContent = 'Error: ' + (j.detail || 'stop nahi hua'); return; }
  window.location.href = '/app/live/' + liveSession;
});
</script>
""".replace("{CREATOR_OPTIONS}", creator_options)


@app.get("/live/record", response_class=HTMLResponse)
def live_record_page(request: Request) -> HTMLResponse:
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    creators = list_creators(DB_PATH, user["id"])
    return _app_layout(user, "Record live", _live_record_body(creators))


@app.post("/live/start")
async def live_start_endpoint(request: Request):
    """Start a browser-capture live session (login required)."""
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Pehle sign in karo")
    body = await _read_body(request)
    raw_options = body.get("options") or {}
    if not isinstance(raw_options, dict):
        raise HTTPException(status_code=400, detail="options ek object hona chahiye")
    try:
        opts = parse_options(raw_options)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    creator_id = body.get("creator_id") or None
    if creator_id:
        creator = get_creator(DB_PATH, str(creator_id))
        if not creator or creator["user_id"] != user["id"]:
            raise HTTPException(status_code=400, detail="Creator nahi mila — dobara choose karo")
    try:
        session = create_live_session(
            DB_PATH, ROOT, user["id"],
            title=str(body.get("title") or "Live recording"),
            creator_id=str(creator_id) if creator_id else None,
            source="browser", options=opts,
            language=str(body.get("language") or "en"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"session_id": session["session_id"],
            "upload_id": session["upload_id"], "status": "live",
            "message": "Live session shuru — ab browser se 60s ke chunks bhejo"}


@app.post("/live/{session_id}/chunk")
async def live_chunk_endpoint(request: Request, session_id: str,
                              chunk: UploadFile = File(...)):
    """One self-contained chunk file for a live session.

    Login required; only the session owner, while the session is
    live, can upload. The file is streamed to disk (no in-memory
    buffering) and queued for the single worker thread."""
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Pehle sign in karo")
    session = get_live_session(DB_PATH, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Live session nahi mila")
    if session["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Ye session tumhara nahi hai")
    if session["status"] != "live":
        raise HTTPException(status_code=400,
                            detail=f"Session {session['status']} hai — chunk sirf live session me aata hai")
    ext = Path(chunk.filename or "").suffix.lower()
    if ext not in {".webm", ".mp4", ".mov", ".mkv"}:
        raise HTTPException(status_code=400,
                            detail="Chunk .webm, .mp4, .mov ya .mkv hona chahiye")
    session_dir = ROOT / "uploads" / "live" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    raw_path = session_dir / f"incoming_{uuid.uuid4().hex[:12]}{ext}"
    try:
        import shutil as _shutil
        with raw_path.open("wb") as f:
            _shutil.copyfileobj(chunk.file, f, length=1024 * 1024)
    finally:
        try:
            await chunk.close()
        except Exception:
            pass
    try:
        result = register_chunk(DB_PATH, ROOT, session_id, raw_path)
    except (ValueError, FileNotFoundError) as exc:
        try:
            raw_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.post("/live/{session_id}/stop")
def live_stop_endpoint(request: Request, session_id: str) -> dict:
    """Stop a session: no more chunks; finalise in the background."""
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Pehle sign in karo")
    try:
        session = stop_live_session(DB_PATH, ROOT, session_id, user["id"])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return session_public(DB_PATH, session)


@app.get("/live/{session_id}")
def live_status_endpoint(request: Request, session_id: str) -> dict:
    """Live status: chunks received/processing/done + clips so far."""
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Pehle sign in karo")
    session = get_live_session(DB_PATH, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Live session nahi mila")
    if session["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Ye session tumhara nahi hai")
    return session_public(DB_PATH, session)


@app.get("/live")
def live_list_endpoint(request: Request) -> dict:
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Pehle sign in karo")
    sessions = list_live_sessions(DB_PATH, user["id"])
    return {"sessions": [session_public(DB_PATH, s) for s in sessions]}


@app.post("/live/check")
async def live_check_endpoint(request: Request):
    """Is a creator's channel live right now? yt-dlp only, no keys.

    Body: {"creator_id": "..."} or {"channel_link": "..."}.
    Returns {"platform", "live": true|false|null, "title", "error"} —
    live=null means "pata nahi chala", never a guess. The result is
    stored on the creator row (best-effort) when creator_id is given."""
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Pehle sign in karo")
    body = await _read_body(request)
    creator_id = body.get("creator_id") or None
    link = ""
    if creator_id:
        creator = get_creator(DB_PATH, str(creator_id))
        if not creator or creator["user_id"] != user["id"]:
            raise HTTPException(status_code=400, detail="Creator nahi mila — dobara choose karo")
        link = creator["channel_link"]
    else:
        link = str(body.get("channel_link") or "")
    if not link:
        raise HTTPException(status_code=400, detail="channel_link ya creator_id do")
    result = check_channel_live(link)
    if creator_id:
        try:
            record_channel_check(DB_PATH, str(creator_id), result)
        except Exception:
            pass
    return result


@app.post("/live/start-from-url")
async def live_start_from_url_endpoint(request: Request):
    """Record a stream/VOD URL with yt-dlp (best-effort, background).

    Body: {"url", "title"?, "creator_id"?, "options"?, "language"?}.
    The live check is NOT a gate — a URL yt-dlp cannot read fails
    honestly in the session's error field. No API keys are used."""
    user = _require_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Pehle sign in karo")
    body = await _read_body(request)
    url = str(body.get("url") or "")
    raw_options = body.get("options") or {}
    if not isinstance(raw_options, dict):
        raise HTTPException(status_code=400, detail="options ek object hona chahiye")
    try:
        opts = parse_options(raw_options)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    creator_id = body.get("creator_id") or None
    if creator_id:
        creator = get_creator(DB_PATH, str(creator_id))
        if not creator or creator["user_id"] != user["id"]:
            raise HTTPException(status_code=400, detail="Creator nahi mila — dobara choose karo")
    try:
        session = start_url_session(
            DB_PATH, ROOT, user["id"], url,
            title=str(body.get("title") or ""),
            creator_id=str(creator_id) if creator_id else None,
            options=opts, language=str(body.get("language") or "en"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"session_id": session["session_id"],
            "upload_id": session["upload_id"], "status": "live",
            "source": "url",
            "message": "URL capture shuru — yt-dlp stream download kar raha hai; "
                       "Stop dabate hi capture band hoga aur clips banenge"}


@app.get("/app/live", response_class=HTMLResponse)
def live_list_page(request: Request) -> HTMLResponse:
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    sessions = list_live_sessions(DB_PATH, user["id"])
    if sessions:
        rows = "".join(
            f"<tr><td><a href=\"/app/live/{s['session_id']}\">"
            f"{_html.escape(s['title'])}</a></td>"
            f"<td>{s['source']}</td><td>{s['status']}</td>"
            f"<td>{s['chunks_count']}</td><td>{s['clips_count']}</td>"
            f"<td>{int(s['total_duration_sec'] // 60)}:"
            f"{int(s['total_duration_sec'] % 60):02d} min</td>"
            f"<td>{str(s['created_at'])[:16]}</td></tr>" for s in sessions)
        table = ("<table><tr><th>Session</th><th>Source</th><th>Status</th>"
                 "<th>Chunks</th><th>Clips</th><th>Length</th><th>Started</th></tr>"
                 + rows + "</table>")
    else:
        table = "<p class=\"muted\">No live sessions yet.</p>"
    body = (f"<h1>Live sessions</h1><div class=\"card\">{table}</div>"
            f"<p><a class=\"btn\" href=\"/live/record\">🔴 Record live</a> "
            f"<a class=\"btn ghost\" href=\"/app\">Back to Home</a></p>")
    return _app_layout(user, "Live sessions", body)


@app.get("/app/live/{session_id}", response_class=HTMLResponse)
def live_detail_page(request: Request, session_id: str) -> HTMLResponse:
    user = _require_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    session = get_live_session(DB_PATH, session_id)
    if not session or session["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Live session nahi mila")
    pub = session_public(DB_PATH, session)
    chunk_rows = "".join(
        f"<tr><td>#{c['chunk_index']}</td><td>{c['start_sec']}s</td>"
        f"<td>{c['duration_sec']}s</td><td>{c['status']}</td>"
        f"<td>{_html.escape(c.get('note') or c.get('error') or '')}</td></tr>"
        for c in pub["chunks"]) or "<tr><td colspan=\"5\">No chunks yet.</td></tr>"
    clip_links = " · ".join(
        f"<a href=\"/app/videos/{pub['upload_id']}/clips/{c['clip_id']}\">"
        f"{c['clip_id']} ({_fmt_score(c.get('score'))})</a>"
        for c in pub["clips"]) or "<span class=\"muted\">No clips yet — chunks process hote hi yahan aayenge.</span>"
    stop_btn = ""
    if pub["status"] == "live":
        stop_btn = (f"<button class=\"btn\" id=\"live-stop-btn\" type=\"button\">"
                    f"Stop &amp; finalise</button> "
                    f"<script>document.getElementById('live-stop-btn')"
                    f".addEventListener('click', async () => {{"
                    f"const r = await fetch('/live/{session_id}/stop', {{method:'POST'}});"
                    f"if (r.ok) location.reload();"
                    f"else alert('Stop nahi hua: ' + (await r.json()).detail);}});</script>")
    video_link = ""
    if pub["status"] in ("stopped", "failed"):
        video_link = (f"<p><a class=\"btn\" href=\"/app/videos/{pub['upload_id']}\">"
                      f"Open final video</a></p>")
    err = (f"<p class=\"err\">{_html.escape(pub['error'])}</p>"
           if pub.get("error") else "")
    note = (f"<p class=\"muted\">{_html.escape(pub['note'])}</p>"
            if pub.get("note") else "")
    body = (f"<h1>🔴 {_html.escape(pub['title'])}</h1>"
            f"<p class=\"muted\">Source: {pub['source']}"
            f"{' — ' + _html.escape(pub['source_url']) if pub['source_url'] else ''}"
            f" · Status: <b>{pub['status']}</b> · Started {str(pub['created_at'])[:16]}</p>"
            f"<div class=\"card\"><h3>Progress</h3>"
            f"<p>{pub['chunks_done']} chunk(s) done · {pub['chunks_received']} received · "
            f"{pub['chunks_processing']} processing · {pub['chunks_error']} error · "
            f"{pub['clips_count']} clip(s) · {pub['total_duration_sec']}s total</p>"
            f"{err}{note}{stop_btn}{video_link}</div>"
            f"<div class=\"card\"><h3>Clips (live bante hue)</h3><p>{clip_links}</p></div>"
            f"<div class=\"card\"><h3>Chunks</h3><table>"
            f"<tr><th>#</th><th>Start</th><th>Length</th><th>Status</th><th>Note</th></tr>"
            f"{chunk_rows}</table></div>"
            f"<p><a class=\"btn ghost\" href=\"/app/live\">All live sessions</a> "
            f"<a class=\"btn ghost\" href=\"/app\">Back to Home</a></p>")
    return _app_layout(user, "Live session", body)


def run() -> None:
    host = os.getenv("CLIPCUT_HOST", os.getenv("VCS_HOST", "127.0.0.1"))
    port = int(os.getenv("CLIPCUT_PORT", os.getenv("VCS_PORT", "8000")))
    uvicorn.run("app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
