"""Phase 7: live recorder — ClipCut's live-stream clipping.

What this IS (honest scope):
  * Browser capture (the free-tier route that actually works): the
    user's own camera/mic or screen is recorded in the browser with
    MediaRecorder, in ~60s segments. Each segment is uploaded to the
    server, transcribed, dual-judge scored and rendered WHILE the
    session is still live — captioned clips appear before Stop.
    Stop concatenates the segments into one session video.
  * Channel live-status polling (best-effort, no keys): yt-dlp asks
    a channel whether it is live right now. Anything that needs an
    API key / client id is recorded as "needs key — not wired", never
    faked.
  * URL capture (best-effort): a Twitch/Kick/YouTube stream URL is
    recorded with yt-dlp into a kill-tolerant .mkv, then processed
    like a normal upload when the stream ends or the user stops it.
    YouTube downloads can 403 from datacenter IPs (known since
    Phase 6) — the failure is recorded, never hidden.

What this is NOT:
  * Not a 24/7 recorder. Render Free has 512 MB RAM and sleeps
    after 15 min idle; a long live session will OOM or be cut by a
    restart. Chunks are processed ONE AT A TIME by a single worker
    thread (never parallel renders) — slow on purpose, so the
    container survives. A chunk can lag behind real time; the session
    page shows the lag honestly (received vs done chunks).
  * Not real-time chat: the Reaction judge runs on loudness unless
    chat data is supplied by the caller (Phase 3 rule, unchanged).

Chunk contract (important): every uploaded chunk MUST be a
self-contained playable file (ffprobe must read it). MediaRecorder
timeslice blobs are NOT — only the first blob carries the webm
header — so the record page stops/starts the recorder every 60s
instead (each segment is a complete file, ~100-300 ms gap between
segments). The API test uploads ffmpeg-split mp4 chunks, which meet
the same contract.

Safety on Render Free 512 MB:
  * One global worker thread, one chunk at a time. Transcribe frees
    its Whisper model after every chunk (transcriber.py does
    del + gc + malloc_trim); renders run sequentially after it.
  * No video is ever held in memory — chunks live on disk under
    uploads/live/<session_id>/, the final video is an ffmpeg concat.
  * Storage quota (100 MB/user, Phase 1) applies: a session cannot
    start when the user is already over quota, and stop() enforces
    the quota after finalising (oldest uploads go first).
"""
from __future__ import annotations

import json
import queue
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIVE_DIR = ROOT / "uploads" / "live"
UPLOAD_DIR = ROOT / "uploads"
TRANSCRIPTS_DIR = ROOT / "outputs" / "transcripts"
CLIPS_DIR = ROOT / "outputs" / "clips"

# One chunk at a time, globally — the RAM rule. A 60s chunk takes
# ~30-60s to transcribe + ~30-90s per clip render on CPU, so a chunk
# can lag behind real time; the session page shows received vs done.
CHUNK_SECONDS = 60
MAX_CHUNK_BYTES = 200 * 1024 * 1024  # 200 MB per chunk
MAX_FINALISE_WAIT_SEC = 900  # stop() waits at most 15 min for the queue
YOUTUBE_403_MARKERS = ("403", "forbidden", "bot", "sign in to confirm",
                       "are not a robot")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_live_tables(db_path: Path) -> None:
    """Create live tables + creator live-status columns (idempotent)."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS live_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            creator_id TEXT,
            title TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'browser',
            source_url TEXT,
            status TEXT NOT NULL DEFAULT 'live',
            upload_id TEXT NOT NULL,
            options_json TEXT NOT NULL DEFAULT '{}',
            language TEXT NOT NULL DEFAULT 'en',
            created_at TEXT NOT NULL,
            stopped_at TEXT,
            last_chunk_at TEXT,
            total_duration_sec REAL NOT NULL DEFAULT 0,
            chunks_count INTEGER NOT NULL DEFAULT 0,
            clips_count INTEGER NOT NULL DEFAULT 0,
            error TEXT NOT NULL DEFAULT ''
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS live_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            stored_path TEXT NOT NULL,
            start_sec REAL NOT NULL DEFAULT 0,
            duration_sec REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'received',
            error TEXT NOT NULL DEFAULT '',
            note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            UNIQUE(session_id, chunk_index)
        )""")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(creators)")}
        for col, ddl in (("last_live_status", "TEXT"),
                         ("last_live_checked_at", "TEXT"),
                         ("last_live_title", "TEXT")):
            if col not in cols:
                conn.execute(f"ALTER TABLE creators ADD COLUMN {col} {ddl}")
        conn.commit()


# ---------------------------------------------------------------- sessions


def create_live_session(db_path: Path, root: Path, user_id: str,
                        title: str, creator_id: str | None = None,
                        source: str = "browser", source_url: str | None = None,
                        options: dict | None = None,
                        language: str = "en") -> dict:
    """Start a live session + its placeholder upload row.

    The uploads row is created NOW (role 'live session', stored_path
    uploads/<upload_id>.mp4, size 0) so clips, ledger rows and the
    editor all have one stable upload_id from the first chunk. The
    file itself only appears at stop() (ffmpeg concat of the chunks).
    """
    from auth import get_creator, storage_usage_bytes, STORAGE_QUOTA_BYTES

    if creator_id:
        creator = get_creator(db_path, creator_id)
        if not creator or creator["user_id"] != user_id:
            raise ValueError("Creator nahi mila — dobara choose karo")
    if storage_usage_bytes(db_path, user_id) >= STORAGE_QUOTA_BYTES:
        raise ValueError(
            "Storage full hai (100 MB) — pehle purani videos download "
            "karke delete karo, phir live recording shuru karo")
    session_id = uuid.uuid4().hex[:16]
    upload_id = uuid.uuid4().hex
    title = (title or "Live recording").strip() or "Live recording"
    (root / "uploads" / "live" / session_id).mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO uploads VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (upload_id, "live session", title, f"uploads/{upload_id}.mp4",
             0, _now(), user_id, creator_id))
        conn.execute(
            "INSERT INTO live_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, user_id, creator_id, title, source,
             source_url or "", "live", upload_id,
             json.dumps(options or {}), language or "en", _now(), None, None,
             0.0, 0, 0, ""))
        conn.commit()
    return get_live_session(db_path, session_id)


def get_live_session(db_path: Path, session_id: str) -> dict | None:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT session_id, user_id, creator_id, title, source,"
            " source_url, status, upload_id, options_json, language,"
            " created_at, stopped_at, last_chunk_at, total_duration_sec,"
            " chunks_count, clips_count, error FROM live_sessions"
            " WHERE session_id=?", (session_id,)).fetchone()
    if not row:
        return None
    return {"session_id": row[0], "user_id": row[1], "creator_id": row[2],
            "title": row[3], "source": row[4], "source_url": row[5],
            "status": row[6], "upload_id": row[7],
            "options": json.loads(row[8] or "{}"),
            "language": row[9], "created_at": row[10],
            "stopped_at": row[11], "last_chunk_at": row[12],
            "total_duration_sec": row[13], "chunks_count": row[14],
            "clips_count": row[15], "error": row[16]}


def list_live_sessions(db_path: Path, user_id: str,
                       status: str | None = None) -> list[dict]:
    with closing(sqlite3.connect(db_path)) as conn:
        if status:
            rows = conn.execute(
                "SELECT session_id FROM live_sessions WHERE user_id=?"
                " AND status=? ORDER BY created_at DESC",
                (user_id, status)).fetchall()
        else:
            rows = conn.execute(
                "SELECT session_id FROM live_sessions WHERE user_id=?"
                " ORDER BY created_at DESC", (user_id,)).fetchall()
    return [get_live_session(db_path, r[0]) for r in rows if r[0]]


def count_live_sessions(db_path: Path, user_id: str) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM live_sessions WHERE user_id=?"
            " AND status IN ('live', 'stopping')", (user_id,)).fetchone()
    return int(row[0])


def _set_session(db_path: Path, session_id: str, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(f"UPDATE live_sessions SET {sets} WHERE session_id=?",
                     (*fields.values(), session_id))
        conn.commit()


def _list_chunks(db_path: Path, session_id: str) -> list[dict]:
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, session_id, chunk_index, stored_path, start_sec,"
            " duration_sec, status, error, note, created_at FROM live_chunks"
            " WHERE session_id=? ORDER BY chunk_index ASC",
            (session_id,)).fetchall()
    return [{"id": r[0], "session_id": r[1], "chunk_index": r[2],
             "stored_path": r[3], "start_sec": r[4], "duration_sec": r[5],
             "status": r[6], "error": r[7], "note": r[8],
             "created_at": r[9]} for r in rows]


def _set_chunk(db_path: Path, session_id: str, chunk_index: int,
               **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            f"UPDATE live_chunks SET {sets} WHERE session_id=?"
            " AND chunk_index=?", (*fields.values(), session_id, chunk_index))
        conn.commit()


def session_public(db_path: Path, session: dict) -> dict:
    """JSON-safe session view: chunks + live clips, no disk paths."""
    chunks = _list_chunks(db_path, session["session_id"])
    clips = []
    try:
        from clip_finder import load_clips
        clips = load_clips(session["upload_id"]).get("clips", [])
    except Exception:
        clips = []
    return {
        "session_id": session["session_id"],
        "title": session["title"],
        "source": session["source"],
        "source_url": session["source_url"],
        "status": session["status"],
        "upload_id": session["upload_id"],
        "creator_id": session["creator_id"],
        "language": session["language"],
        "created_at": session["created_at"],
        "stopped_at": session["stopped_at"],
        "last_chunk_at": session["last_chunk_at"],
        "total_duration_sec": round(float(session["total_duration_sec"]), 2),
        "chunks_count": session["chunks_count"],
        "chunks_received": sum(1 for c in chunks if c["status"] == "received"),
        "chunks_processing": sum(1 for c in chunks if c["status"] == "processing"),
        "chunks_done": sum(1 for c in chunks if c["status"] == "done"),
        "chunks_error": sum(1 for c in chunks if c["status"] == "error"),
        "clips_count": len(clips),
        "clips": [{"clip_id": c.get("clip_id"), "start": c.get("start"),
                   "end": c.get("end"), "duration": c.get("duration"),
                   "score": c.get("combined_score", c.get("score")),
                   "words_score": c.get("words_score"),
                   "reaction_score": c.get("reaction_score"),
                   "chunk_index": c.get("chunk_index")}
                  for c in clips],
        "chunks": [{"chunk_index": c["chunk_index"],
                    "start_sec": c["start_sec"],
                    "duration_sec": c["duration_sec"],
                    "status": c["status"], "error": c["error"],
                    "note": c["note"]} for c in chunks],
        "error": session["error"],
        "note": ("Re-make editor me tab chalega jab session stop ho jaye — "
                 "us se pehle clips chunk se render hote hain.")
        if session["status"] in ("live", "stopping") else "",
    }


# ---------------------------------------------------------------- chunk intake


def register_chunk(db_path: Path, root: Path, session_id: str,
                   raw_path: Path) -> dict:
    """Queue one uploaded chunk for processing.

    The raw file must already be saved under the session dir. Only
    the owning user, while the session is live, can queue chunks.
    Returns the chunk row (status 'received'). The worker thread
    does the heavy work (transcode -> transcribe -> clips -> render).
    """
    session = get_live_session(db_path, session_id)
    if not session:
        raise FileNotFoundError(f"live session nahi mila: {session_id}")
    if session["status"] != "live":
        raise ValueError(
            f"Session {session['status']} hai — chunk sirf live session me aata hai")
    if not raw_path.is_file() or raw_path.stat().st_size == 0:
        raise ValueError("Chunk file khaali hai ya missing hai")
    if raw_path.stat().st_size > MAX_CHUNK_BYTES:
        raise ValueError(
            f"Chunk bahut bada hai ({raw_path.stat().st_size // (1024 * 1024)} MB) — "
            "max 200 MB per chunk")
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(chunk_index), -1) FROM live_chunks"
            " WHERE session_id=?", (session_id,)).fetchone()
        idx = int(row[0]) + 1
        conn.execute(
            "INSERT INTO live_chunks (session_id, chunk_index, stored_path,"
            " start_sec, duration_sec, status, error, note, created_at)"
            " VALUES (?, ?, ?, 0, 0, 'received', '', '', ?)",
            (session_id, idx, str(raw_path), _now()))
        conn.commit()
    _CHUNK_QUEUE.put((session_id, idx))
    ensure_live_worker()
    return {"session_id": session_id, "chunk_index": idx, "status": "queued"}


# ---------------------------------------------------------------- worker (one chunk at a time)


_CHUNK_QUEUE: "queue.Queue[tuple[str, int]]" = queue.Queue()
_WORKER_THREAD: threading.Thread | None = None
_WORKER_LOCK = threading.Lock()


def ensure_live_worker() -> None:
    """Start the single chunk worker thread (daemon, idempotent)."""
    global _WORKER_THREAD
    with _WORKER_LOCK:
        if _WORKER_THREAD is None or not _WORKER_THREAD.is_alive():
            _WORKER_THREAD = threading.Thread(target=_worker_loop,
                                              daemon=True,
                                              name="clipcut-live-worker")
            _WORKER_THREAD.start()


def live_worker_status() -> dict:
    """Health-view of the chunk worker (for /health)."""
    running = bool(_WORKER_THREAD is not None and _WORKER_THREAD.is_alive())
    return {"worker_running": running,
            "queued_chunks": int(_CHUNK_QUEUE.qsize()),
            "chunk_seconds": CHUNK_SECONDS,
            "max_chunk_bytes_mb": MAX_CHUNK_BYTES // (1024 * 1024)}


def reconcile_live_state(db_path: Path) -> dict:
    """Startup sweep after a server restart (Render restarts often).

    The chunk queue is in-memory, so a restart orphans chunks stuck
    in 'received'/'processing' — they would show as pending forever.
    Mark them 'error' with an honest note (the user can re-send the
    chunk; the session itself stays 'live' and keeps working).
    Live sessions are left alone — a recorder can keep uploading
    chunks into them after the restart. Returns what was fixed.
    """
    with closing(sqlite3.connect(db_path)) as conn:
        stuck = conn.execute(
            "SELECT session_id, chunk_index FROM live_chunks"
            " WHERE status IN ('received', 'processing')").fetchall()
        for session_id, chunk_index in stuck:
            conn.execute(
                "UPDATE live_chunks SET status='error',"
                " error='Server restart hua — chunk process nahi hua, dobara bhejo'"
                " WHERE session_id=? AND chunk_index=?",
                (session_id, chunk_index))
        conn.commit()
    return {"stuck_chunks_marked_error": len(stuck)}


def _worker_loop() -> None:
    while True:
        try:
            session_id, chunk_index = _CHUNK_QUEUE.get()
        except Exception:
            continue
        try:
            _process_chunk(session_id, chunk_index)
        except Exception as exc:  # never kill the worker on one chunk
            try:
                _set_chunk(_DB_PATH, session_id, chunk_index,
                           status="error", error=str(exc)[:500])
            except Exception:
                pass
        finally:
            _CHUNK_QUEUE.task_done()


# The worker needs the app DB path; app.py sets it at startup via
# set_live_db_path() (lifespan). Tests set it directly.
_DB_PATH: Path | None = None


def set_live_db_path(db_path: Path) -> None:
    global _DB_PATH
    _DB_PATH = Path(db_path)


def _db() -> Path:
    if _DB_PATH is None:
        return ROOT / "clipcut.sqlite3"
    return _DB_PATH


def _transcode_chunk(raw_path: Path, out_path: Path) -> None:
    """Normalise a chunk to h264/aac mp4 (same params every chunk, so
    the stop-time concat with -c copy works). Raises RuntimeError."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg install karo — chunk transcode nahi ho sakta")
    result = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", str(raw_path),
         "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart", str(out_path)],
        capture_output=True, text=True, timeout=600)
    if result.returncode != 0 or not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(
            "Chunk transcode nahi hua — file playable nahi hai ya codec "
            f"unsupported hai. {(result.stderr or '').strip()[-300:]}")


def _merge_session_transcript(root: Path, session_upload_id: str,
                              chunk_transcript: dict, chunk_start: float) -> None:
    """Append one chunk's transcript (times offset) to the session's
    merged transcript, so the editor's load_transcript() works while
    the session is still live. Re-make still needs the final video,
    which only exists after stop() — see session_public note."""
    from transcriber import TRANSCRIPTS_DIR  # local: same dir layout
    session_path = TRANSCRIPTS_DIR / f"{session_upload_id}.json"
    if session_path.is_file():
        try:
            merged = json.loads(session_path.read_text(encoding="utf-8"))
        except Exception:
            merged = None
    else:
        merged = None
    if not merged:
        merged = {"upload_id": session_upload_id,
                  "model": chunk_transcript.get("model", "base"),
                  "engine": "faster-whisper",
                  "language": chunk_transcript.get("language", "en"),
                  "language_requested": chunk_transcript.get("language_requested", "en"),
                  "duration_sec": 0.0, "text": "", "word_count": 0,
                  "word_timestamps_real": True, "words": [], "segments": []}
    words = [{"word": w.get("word", ""),
              "start": round(float(w.get("start", 0)) + chunk_start, 2),
              "end": round(float(w.get("end", 0)) + chunk_start, 2)}
             for w in (chunk_transcript.get("words") or [])]
    segments = [{"start": round(float(s.get("start", 0)) + chunk_start, 2),
                 "end": round(float(s.get("end", 0)) + chunk_start, 2),
                 "text": str(s.get("text", "")).strip()}
                for s in (chunk_transcript.get("segments") or [])]
    merged["words"].extend(words)
    merged["segments"].extend(segments)
    merged["text"] = " ".join(
        s["text"] for s in merged["segments"] if s["text"]).strip()
    merged["word_count"] = len(merged["words"])
    merged["duration_sec"] = round(
        max([float(w["end"]) for w in merged["words"]] or [0.0]), 2)
    session_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    (TRANSCRIPTS_DIR / f"{session_upload_id}.txt").write_text(
        merged["text"], encoding="utf-8")


def _process_chunk(session_id: str, chunk_index: int) -> None:
    """Transcribe -> dual-judge clips -> merge -> render, one chunk.

    Runs in the single worker thread. Heavy, sequential, on purpose:
    on 512 MB Render Free a parallel render would OOM the container.
    """
    from clip_finder import find_clips, load_clips, save_clips
    from editor import render_clip
    from transcriber import transcribe_isolated
    from utils import probe_video
    from ai_providers import get_effective_ai_config

    db_path = _db()
    session = get_live_session(db_path, session_id)
    if not session or session["status"] not in ("live", "stopping"):
        return
    chunk = next((c for c in _list_chunks(db_path, session_id)
                  if c["chunk_index"] == chunk_index), None)
    if not chunk or chunk["status"] != "received":
        return
    _set_chunk(db_path, session_id, chunk_index, status="processing")
    root = ROOT
    session_dir = root / "uploads" / "live" / session_id
    raw_path = Path(chunk["stored_path"])
    chunk_mp4 = session_dir / f"chunk_{chunk_index:03d}.mp4"
    chunk_transcript_id = f"live_{session_id}_c{chunk_index:03d}"
    try:
        # 1. Normalise (webm from MediaRecorder, mp4 from the API test).
        _transcode_chunk(raw_path, chunk_mp4)
        try:
            raw_path.unlink(missing_ok=True)
        except OSError:
            pass
        probe = probe_video(chunk_mp4)
        if not probe["valid"]:
            raise ValueError(f"Chunk playable nahi hai: {probe['error']}")
        duration = float(probe["duration"])
        if duration < 5 or not probe["has_audio"]:
            _set_chunk(db_path, session_id, chunk_index, status="done",
                       duration_sec=round(duration, 2),
                       start_sec=float(session["total_duration_sec"]),
                       note="Chunk bahut chhota hai ya audio nahi hai — "
                            "clips nahi ban sakte, session chalta rahega")
            return
        chunk_start = float(session["total_duration_sec"])
        _set_chunk(db_path, session_id, chunk_index,
                   start_sec=chunk_start, duration_sec=round(duration, 2))
        # 2. Transcribe the chunk (frees its Whisper model afterwards).
        transcript = transcribe_isolated(chunk_mp4, chunk_transcript_id,
                                language=session["language"])
        if not transcript.get("words"):
            _set_chunk(db_path, session_id, chunk_index, status="done",
                       note="Transcript khaali hai — is chunk me boli hui baat nahi mili")
            _set_session(db_path, session_id,
                         total_duration_sec=round(chunk_start + duration, 2),
                         chunks_count=int(session["chunks_count"]) + 1,
                         last_chunk_at=_now())
            return
        # 3. Dual-judge clips (owner's BYOK config + ledger on the
        #    session upload, so usage attaches to the session video).
        opts = dict(session["options"] or {})
        ai_config = get_effective_ai_config(db_path, session["user_id"])
        ledger_ctx = {"db_path": db_path, "user_id": session["user_id"],
                      "upload_id": session["upload_id"],
                      "creator_id": session["creator_id"]}
        result = find_clips(transcript, chunk_transcript_id,
                            top_n=int(opts.get("clip_count", 2)),
                            options=opts,
                            language=session["language"],
                            video_path=chunk_mp4, ai_config=ai_config,
                            ledger_ctx=ledger_ctx)
        new_clips = []
        for clip in result.get("clips", []):
            local = dict(clip)
            clip_id = f"live_c{chunk_index:03d}_{clip.get('clip_id')}"
            local.update({
                "clip_id": clip_id,
                "start": round(float(clip["start"]) + chunk_start, 2),
                "end": round(float(clip["end"]) + chunk_start, 2),
                "chunk_index": chunk_index,
                "session_id": session_id,
                "source": "live",
            })
            new_clips.append(local)
        # 4. Merge clips + transcript into the session's files.
        try:
            session_clips = load_clips(session["upload_id"])
        except FileNotFoundError:
            session_clips = {"upload_id": session["upload_id"],
                             "scoring_method": result.get("scoring_method"),
                             "scoring_language": result.get("scoring_language", "en"),
                             "options": opts, "clips": []}
        existing_ids = {str(c.get("clip_id"))
                        for c in session_clips.get("clips", [])}
        session_clips["clips"].extend(
            c for c in new_clips if str(c["clip_id"]) not in existing_ids)
        save_clips(session["upload_id"], session_clips)
        _merge_session_transcript(root, session["upload_id"], transcript,
                                  chunk_start)
        # 5. Render every kept clip NOW, from the chunk file with
        #    LOCAL times (the session video does not exist yet).
        #    Sequential — one render at a time, RAM rule.
        rendered = 0
        for clip in new_clips:
            local_clip = dict(clip)
            local_clip["start"] = round(float(clip["start"]) - chunk_start, 2)
            local_clip["end"] = round(float(clip["end"]) - chunk_start, 2)
            try:
                meta = render_clip(chunk_mp4, session["upload_id"], local_clip,
                                   transcript=transcript, options=opts)
                # Sidecar start/end are chunk-local; patch them to the
                # session timeline so the sidecar matches the clip.
                try:
                    sidecar = Path(str(meta.get("output_json", "")))
                    if sidecar.is_file():
                        data = json.loads(sidecar.read_text(encoding="utf-8"))
                        data["start"] = clip["start"]
                        data["end"] = clip["end"]
                        sidecar.write_text(
                            json.dumps(data, indent=2, ensure_ascii=False),
                            encoding="utf-8")
                except Exception:
                    pass
                rendered += 1
            except Exception as exc:
                # One failed render must not kill the chunk or the
                # session — the clip stays listed, re-make after stop.
                continue
        _set_session(db_path, session_id,
                     total_duration_sec=round(chunk_start + duration, 2),
                     chunks_count=int(session["chunks_count"]) + 1,
                     clips_count=len(session_clips.get("clips", [])),
                     last_chunk_at=_now())
        _set_chunk(db_path, session_id, chunk_index, status="done",
                   note=(f"{len(new_clips)} clips, {rendered} rendered"))
    except Exception as exc:
        _set_chunk(db_path, session_id, chunk_index, status="error",
                   error=str(exc)[:500])
        raise


# ---------------------------------------------------------------- stop / finalise


def stop_live_session(db_path: Path, root: Path, session_id: str,
                      user_id: str) -> dict:
    """Stop a session: no more chunks; finalise in the background.

    Returns at once with status 'stopping'. The finaliser waits for
    the worker queue to drain for this session, concatenates the
    done chunks into uploads/<upload_id>.mp4, enforces the quota,
    and marks the session 'stopped' (or 'failed' with the error).
    """
    session = get_live_session(db_path, session_id)
    if not session:
        raise FileNotFoundError(f"live session nahi mila: {session_id}")
    if session["user_id"] != user_id:
        raise PermissionError("Ye session tumhara nahi hai")
    if session["status"] != "live":
        raise ValueError(f"Session pehle se {session['status']} hai")
    if session["source"] == "url":
        # Order matters (race fix, 16 Nov 2026): status FIRST, kill
        # second. The capture thread writes 'failed'/'stopped' as soon
        # as its yt-dlp process dies — if stop() set 'stopping' AFTER
        # the kill, it overwrote the capture thread's 'failed' and the
        # session stayed 'stopping' until the 15-min finalise timeout.
        _set_session(db_path, session_id, status="stopping")
        _stop_url_capture(session_id)
        # Re-read: the capture may already have finished/failed in
        # the meantime — never overwrite a terminal status.
        current = get_live_session(db_path, session_id)
        if current and current["status"] in ("stopped", "failed"):
            return current
    else:
        _set_session(db_path, session_id, status="stopping")
    thread = threading.Thread(target=_finalise_session,
                              args=(str(db_path), str(root), session_id),
                              daemon=True, name="clipcut-live-finalise")
    thread.start()
    return get_live_session(db_path, session_id)


def _wait_queue_drain(db_path: Path, session_id: str,
                      timeout_sec: int = MAX_FINALISE_WAIT_SEC) -> bool:
    """True when no chunk of this session is received/processing."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        chunks = _list_chunks(db_path, session_id)
        if not any(c["status"] in ("received", "processing") for c in chunks):
            return True
        time.sleep(2)
    return False


def _concat_chunks(root: Path, session_id: str, out_path: Path) -> None:
    """ffmpeg concat of the session's done chunks (same transcode
    params, so -c copy works; re-encode fallback on failure)."""
    from utils import probe_video
    session_dir = root / "uploads" / "live" / session_id
    parts = sorted(session_dir.glob("chunk_*.mp4"))
    if not parts:
        raise ValueError("Koi chunk process nahi hua — video nahi bani")
    if len(parts) == 1:
        shutil.copyfile(parts[0], out_path)
    else:
        list_path = session_dir / "concat.txt"
        list_path.write_text(
            "".join(f"file '{p.resolve()}'\n" for p in parts),
            encoding="utf-8")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg install karo — concat nahi ho sakta")
        result = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(list_path), "-c", "copy", "-movflags", "+faststart",
             str(out_path)], capture_output=True, text=True, timeout=1200)
        if result.returncode != 0 or not out_path.is_file():
            # Fallback: full re-encode (slower, but tolerant of any
            # parameter drift between chunks).
            result = subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                 "-i", str(list_path), "-vf", "format=yuv420p",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                 str(out_path)], capture_output=True, text=True, timeout=1800)
            if result.returncode != 0 or not out_path.is_file():
                raise RuntimeError(
                    "Concat nahi hua — chunks jode nahi ja sake. "
                    f"{(result.stderr or '').strip()[-300:]}")
    probe = probe_video(out_path)
    if not probe["valid"]:
        raise ValueError(f"Final video valid nahi hai: {probe['error']}")


def _finalise_session(db_path: str, root: str, session_id: str) -> None:
    """Background stop work: drain -> concat -> quota -> stopped."""
    from auth import enforce_storage_quota
    db = Path(db_path)
    rt = Path(root)
    session = get_live_session(db, session_id)
    if not session:
        return
    try:
        if session["source"] == "url":
            _finalise_url_session(db, rt, session)
            return
        drained = _wait_queue_drain(db, session_id)
        session = get_live_session(db, session_id)
        chunks = [c for c in _list_chunks(db, session_id)
                  if c["status"] == "done" and float(c.get("duration_sec") or 0) > 0]
        if not chunks:
            # No usable video: remove the placeholder upload row.
            with closing(sqlite3.connect(db)) as conn:
                conn.execute("DELETE FROM uploads WHERE id=?",
                             (session["upload_id"],))
                conn.commit()
            _set_session(db, session_id, status="failed",
                         stopped_at=_now(),
                         error="Koi chunk process nahi hua — video nahi bani")
            return
        out_path = rt / "uploads" / f"{session['upload_id']}.mp4"
        _concat_chunks(rt, session_id, out_path)
        with closing(sqlite3.connect(db)) as conn:
            conn.execute("UPDATE uploads SET size_bytes=? WHERE id=?",
                         (out_path.stat().st_size, session["upload_id"]))
            conn.commit()
        enforce_storage_quota(db, rt, session["user_id"])
        note = "" if drained else ("Kuch chunks queue me reh gaye (timeout) — "
                                   "final video me sirf done chunks hain")
        _set_session(db, session_id, status="stopped", stopped_at=_now(),
                     error=note)
    except Exception as exc:
        _set_session(db, session_id, status="failed", stopped_at=_now(),
                     error=str(exc)[:500])


# ---------------------------------------------------------------- channel live check


def check_channel_live(channel_link: str, timeout: int = 90) -> dict:
    """Is this channel live RIGHT NOW? yt-dlp only, no API keys.

    Returns {"platform", "live": True|False|None, "title", "video_id",
    "checked_at", "method", "error"}. live=None means "pata nahi
    chala" (403 / network / parse fail) — NEVER guessed. YouTube
    live check uses <channel>/live; Twitch/Kick use the channel URL
    itself. A VOD/channel page that is not live returns live=False
    only when yt-dlp says so clearly.
    """
    from auth import detect_platform
    from youtube_job import _base_yt_dlp_args, _yt_dlp_cmd, _yt_dlp_env

    platform = detect_platform(channel_link or "")
    link = (channel_link or "").strip().rstrip("/")
    if not link.startswith("http"):
        return {"platform": platform, "live": None, "title": None,
                "video_id": None, "checked_at": _now(),
                "method": "yt-dlp", "error": "Channel link http(s) se shuru hona chahiye"}
    if platform == "youtube":
        url = link + "/live"
    else:
        url = link
    cmd = (_yt_dlp_cmd() + _base_yt_dlp_args()
           + ["--dump-single-json", "--no-download", "--no-playlist", url])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, env=_yt_dlp_env())
    except subprocess.TimeoutExpired:
        return {"platform": platform, "live": None, "title": None,
                "video_id": None, "checked_at": _now(), "method": "yt-dlp",
                "error": f"Check {timeout}s me complete nahi hua (timeout)"}
    except Exception as exc:
        return {"platform": platform, "live": None, "title": None,
                "video_id": None, "checked_at": _now(), "method": "yt-dlp",
                "error": f"yt-dlp run nahi hua: {exc}"}
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    err_low = err.lower()
    if result.returncode == 0 and out.startswith("{"):
        try:
            data = json.loads(out)
        except Exception:
            data = {}
        if platform == "kick":
            # Kick returns channel info; livestream is null offline.
            livestream = data.get("livestream")
            is_live = data.get("is_live")
            if is_live is True or livestream:
                title = None
                if isinstance(livestream, dict):
                    title = livestream.get("session_title") or livestream.get("title")
                return {"platform": platform, "live": True,
                        "title": title or data.get("title"),
                        "video_id": data.get("id"), "checked_at": _now(),
                        "method": "yt-dlp", "error": ""}
            if is_live is False or livestream is None:
                return {"platform": platform, "live": False,
                        "title": None, "video_id": None, "checked_at": _now(),
                        "method": "yt-dlp", "error": ""}
        is_live = data.get("is_live")
        if isinstance(is_live, bool):
            return {"platform": platform, "live": is_live,
                    "title": data.get("title"),
                    "video_id": data.get("id"), "checked_at": _now(),
                    "method": "yt-dlp", "error": ""}
        # JSON aaya par is_live nahi — a VOD/channel page, not a live check.
        return {"platform": platform, "live": None, "title": data.get("title"),
                "video_id": data.get("id"), "checked_at": _now(),
                "method": "yt-dlp",
                "error": "yt-dlp ne live status nahi bataya (is_live field nahi mila)"}
    # Non-zero exit: offline vs blocked vs other error.
    # YouTube lesson (16 Nov 2026, live test): a 404 on the /live tab
    # API is NOT proof the channel is offline — this datacenter IP
    # gets 404s from the tab API even for existing channels (and
    # "Sign in to confirm you're not a bot" on video pages). Only an
    # explicit "not live" message counts as offline. Conversely, if
    # yt-dlp resolved a video id on the /live URL, the channel IS
    # live — a later "No video formats found" / bot-check is a
    # download problem, not a live-status problem. So the resolved-id
    # check runs BEFORE the 403 check for YouTube.
    if platform == "youtube":
        import re as _re
        m = _re.search(r"\[youtube\] ([A-Za-z0-9_-]{11}):", err)
        if m:
            return {"platform": platform, "live": True, "title": None,
                    "video_id": m.group(1), "checked_at": _now(),
                    "method": "yt-dlp",
                    "error": "Live stream mila, par is IP se uska video "
                             "download/format nahi khul raha (bot-check)"}
        if "404" in err_low or "not found" in err_low:
            return {"platform": platform, "live": None, "title": None,
                    "video_id": None, "checked_at": _now(), "method": "yt-dlp",
                    "error": "YouTube ne live status nahi bataya (tab API 404 — "
                             "is IP par bot-block bhi ho sakta hai)"}
    if any(m in err_low for m in YOUTUBE_403_MARKERS):
        return {"platform": platform, "live": None, "title": None,
                "video_id": None, "checked_at": _now(), "method": "yt-dlp",
                "error": "Platform ne request block kar di (403 / bot-check) — "
                         "live status pata nahi chala"}
    if any(m in err_low for m in ("offline", "not currently live",
                                  "is not live", "no live",
                                  "channel has no", "does not exist")):
        return {"platform": platform, "live": False, "title": None,
                "video_id": None, "checked_at": _now(), "method": "yt-dlp",
                "error": ""}
    return {"platform": platform, "live": None, "title": None,
            "video_id": None, "checked_at": _now(), "method": "yt-dlp",
            "error": f"yt-dlp error (live status pata nahi chala): "
                     f"{err[-300:] or 'unknown error'}"}


def record_channel_check(db_path: Path, creator_id: str,
                         result: dict) -> None:
    """Store the last live-check on the creator row (best-effort)."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "UPDATE creators SET last_live_status=?, last_live_checked_at=?,"
            " last_live_title=? WHERE id=?",
            ("live" if result.get("live") is True else
             "offline" if result.get("live") is False else "unknown",
             result.get("checked_at"), result.get("title") or "",
             creator_id))
        conn.commit()


def get_creator_live_status(db_path: Path, creator_id: str) -> dict:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT last_live_status, last_live_checked_at, last_live_title"
            " FROM creators WHERE id=?", (creator_id,)).fetchone()
    if not row:
        return {"status": "—", "checked_at": None, "title": None}
    return {"status": row[0] or "—", "checked_at": row[1],
            "title": row[2] or None}


# ---------------------------------------------------------------- URL capture (best-effort)


_URL_PROCS: dict[str, subprocess.Popen] = {}
_URL_PROCS_LOCK = threading.Lock()


def _stop_url_capture(session_id: str) -> None:
    """Terminate a running yt-dlp capture (SIGTERM, then SIGKILL)."""
    with _URL_PROCS_LOCK:
        proc = _URL_PROCS.pop(session_id, None)
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def start_url_session(db_path: Path, root: Path, user_id: str, url: str,
                      title: str | None = None, creator_id: str | None = None,
                      options: dict | None = None,
                      language: str = "en") -> dict:
    """Record a stream URL with yt-dlp (best-effort, background).

    Accepts Twitch/Kick/YouTube stream or VOD URLs — and any other
    public video link yt-dlp supports (platform 'other'). The live
    check is NOT a gate: a stream that yt-dlp cannot read fails
    honestly in the session's error field. No API keys are used.
    """
    from auth import detect_platform

    url = (url or "").strip()
    if not url.startswith("http"):
        raise ValueError("URL http(s) se shuru hona chahiye")
    platform = detect_platform(url)
    session = create_live_session(db_path, root, user_id,
                                  title or f"{platform} live capture",
                                  creator_id=creator_id, source="url",
                                  source_url=url, options=options,
                                  language=language)
    thread = threading.Thread(target=_run_url_capture,
                              args=(str(db_path), str(root),
                                    session["session_id"], url),
                              daemon=True, name="clipcut-url-capture")
    thread.start()
    return session


def _run_url_capture(db_path: str, root: str, session_id: str,
                     url: str) -> None:
    """yt-dlp download -> session video -> full pipeline.

    Capture goes to .mkv (kill-tolerant: a stopped live stream still
    leaves a readable file, unlike mp4 whose moov sits at the end).
    On download end (stream over, or stop() killed the process) the
    file is transcoded to mp4 and processed like a normal upload:
    transcribe -> dual-judge find-clips -> render every kept clip,
    sequentially (RAM rule). Any failure marks the session 'failed'
    with the exact error.
    """
    from youtube_job import _base_yt_dlp_args, _yt_dlp_cmd, _yt_dlp_env
    from clip_finder import find_clips
    from editor import render_clip
    from transcriber import transcribe_isolated
    from utils import probe_video
    from ai_providers import get_effective_ai_config

    db = Path(db_path)
    rt = Path(root)
    session = get_live_session(db, session_id)
    if not session:
        return
    session_dir = rt / "uploads" / "live" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    capture_path = session_dir / "capture.mkv"
    cmd = (_yt_dlp_cmd() + _base_yt_dlp_args()
           + ["--no-playlist", "--no-part",
              # Retry limits (16 Nov 2026): these bound FRAGMENT /
              # download retries only. They do NOT make a bot-blocked
              # YouTube URL fail fast — yt-dlp hangs in the EXTRACTION
              # phase (JS challenge / player response wait, ~idle CPU)
              # for many minutes (observed 270s+ and still going).
              # The user-facing mitigation is Stop: it kills yt-dlp
              # at once and the session fails promptly (verified).
              # A hard wall-clock cap is deliberately NOT added here —
              # a real multi-hour live stream legitimately runs for
              # hours. See PHASE7_REPORT.md.
              "--retries", "3", "--fragment-retries", "3",
              "--socket-timeout", "30",
              "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
              "--remux-video", "mkv",
              "-o", str(capture_path), url])
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True,
                                env=_yt_dlp_env())
        with _URL_PROCS_LOCK:
            _URL_PROCS[session_id] = proc
        _, stderr = proc.communicate(timeout=6 * 3600)  # max 6h capture
        with _URL_PROCS_LOCK:
            _URL_PROCS.pop(session_id, None)
        session = get_live_session(db, session_id)
        if not session or session["status"] not in ("live", "stopping"):
            return
        if not capture_path.is_file() or capture_path.stat().st_size == 0:
            tail = (stderr or "").strip().splitlines()[-3:]
            raise RuntimeError(
                "Download nahi hua — " + (" ".join(tail) if tail else
                                          "yt-dlp ne koi file nahi banayi"))
        # Transcode the capture to the session mp4.
        out_path = rt / "uploads" / f"{session['upload_id']}.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg install karo — capture convert nahi ho sakta")
        result = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", str(capture_path),
             "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "23", "-c:a", "aac", "-b:a", "128k",
             "-movflags", "+faststart", str(out_path)],
            capture_output=True, text=True, timeout=1800)
        if result.returncode != 0 or not out_path.is_file():
            raise RuntimeError(
                "Capture convert nahi hua — file kharab ya adhuri hai. "
                f"{(result.stderr or '').strip()[-300:]}")
        probe = probe_video(out_path)
        if not probe["valid"]:
            raise ValueError(f"Captured video valid nahi hai: {probe['error']}")
        with closing(sqlite3.connect(db)) as conn:
            conn.execute("UPDATE uploads SET size_bytes=? WHERE id=?",
                         (out_path.stat().st_size, session["upload_id"]))
            conn.commit()
        # Full pipeline on the captured video.
        transcript = transcribe_isolated(out_path, session["upload_id"],
                                language=session["language"])
        if not transcript.get("words"):
            raise ValueError("Transcript khaali hai — captured video me boli hui baat nahi mili")
        opts = dict(session["options"] or {})
        ai_config = get_effective_ai_config(db, session["user_id"])
        ledger_ctx = {"db_path": db, "user_id": session["user_id"],
                      "upload_id": session["upload_id"],
                      "creator_id": session["creator_id"]}
        result = find_clips(transcript, session["upload_id"],
                            top_n=int(opts.get("clip_count", 2)),
                            options=opts, language=session["language"],
                            video_path=out_path, ai_config=ai_config,
                            ledger_ctx=ledger_ctx)
        rendered = 0
        for clip in result.get("clips", []):
            try:
                render_clip(out_path, session["upload_id"], dict(clip),
                            transcript=transcript, options=opts)
                rendered += 1
            except Exception:
                continue
        from auth import enforce_storage_quota
        enforce_storage_quota(db, rt, session["user_id"])
        _set_session(db, session_id, status="stopped", stopped_at=_now(),
                     total_duration_sec=round(float(probe["duration"]), 2),
                     clips_count=len(result.get("clips", [])),
                     error="" if rendered else
                     "Clips mile par koi render nahi ho paya")
    except Exception as exc:
        with _URL_PROCS_LOCK:
            _URL_PROCS.pop(session_id, None)
        _set_session(db, session_id, status="failed", stopped_at=_now(),
                     error=str(exc)[:500])


def _finalise_url_session(db: Path, rt: Path, session: dict) -> None:
    """stop() for a URL session: the capture thread does the work.

    The capture thread already watches the session status — once it
    sees 'stopping' it finalises after the download ends. Here we
    just wait for it (up to the drain timeout), then mark the result.
    If the thread already finished, there is nothing to do.
    """
    deadline = time.monotonic() + MAX_FINALISE_WAIT_SEC
    while time.monotonic() < deadline:
        current = get_live_session(db, session["session_id"])
        if not current or current["status"] in ("stopped", "failed"):
            return
        time.sleep(3)
    current = get_live_session(db, session["session_id"])
    if current and current["status"] == "stopping":
        _set_session(db, session["session_id"], status="failed",
                     stopped_at=_now(),
                     error="Capture finalise timeout (15 min) — session ko "
                           "dobara check karo, download shayad abhi chal raha hai")
