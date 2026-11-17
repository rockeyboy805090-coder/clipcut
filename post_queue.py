"""Phase 8: posting queue — ClipCut's scheduled posting.

What this IS (honest scope):
  * A real queue: a rendered clip version is queued for a platform
    (YouTube Shorts / Instagram / TikTok) with title/description/
    hashtags, a scheduled time (entered in Asia/Kolkata, stored in
    UTC) and a recurrence (once / daily / weekly). A single worker
    thread picks due items ONE AT A TIME (same RAM rule as the live
    recorder — never parallel uploads) and attempts the post.
  * YouTube Shorts: the REAL upload path is implemented — OAuth 2.0
    (youtube.upload scope) + resumable upload to the YouTube Data
    API v3 (requests, chunked PUT with Content-Range). But the
    Google OAuth client credentials DO NOT exist yet, so Connect
    fails cleanly and due YouTube items sit in `waiting_account`
    with a clear message. A post is NEVER faked: an item is `posted`
    only when the API returns a real video id, and only then is a
    platform URL stored.
  * Instagram / TikTok: the reference app itself shows "Not set up
    on this server" for both — this clone says the same, honestly.
    Items queued for them go to `not_set_up` and the worker never
    pretends to post them.

What this is NOT:
  * Not an auto-poster that posts without the user. Every item is
    created by the user with an explicit schedule, and "Pause all
    posting" stops the worker for that user.
  * Not a YouTube client without credentials. The exact values the
    user must supply are in README / .env.example:
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, YOUTUBE_REDIRECT_URI
    (registered in Google Cloud Console) + YouTube Data API v3
    enabled in the same project.

Safety / Render Free notes:
  * One worker thread, one item at a time; uploads are streamed
    from disk in 8 MB chunks (never the whole file in memory).
  * Startup reconcile: items stuck in `posting` after a restart
    become `failed` with an honest note (Retry works).
  * OAuth tokens are Fernet-encrypted at rest (same server secret
    as the Phase 6 BYOK keys — CLIPCUT_KEY_SECRET env or the
    gitignored .clipcut_key_secret file). The access token is only
    ever decrypted in memory for a single upload call.
"""
from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent

IST = ZoneInfo("Asia/Kolkata")

# Platforms the queue knows. "not set up" is a real status, not a
# missing feature hidden from the user.
PLATFORMS = {
    "youtube": {
        "label": "YouTube Shorts",
        "note": "Resumable upload via YouTube Data API v3 — needs a "
                "connected Google account (OAuth client, user supplies it).",
        "setup": "oauth",
    },
    "instagram": {
        "label": "Instagram Reels",
        "note": "Not set up on this server — CreativeCut me bhi yahi "
                "likha tha. Instagram Graph API + approval chahiye.",
        "setup": "not_set_up",
    },
    "tiktok": {
        "label": "TikTok",
        "note": "Not set up on this server — TikTok Content Posting "
                "API + approval chahiye.",
        "setup": "not_set_up",
    },
}

RECURRENCES = ("once", "daily", "weekly")
PRIVACY_OPTIONS = ("public", "unlisted", "private")
# Statuses: queued (due, will be tried) / waiting_account (YouTube,
# no connected account yet — worker re-checks hourly) / not_set_up
# (IG/TikTok — never tried) / posting (worker is on it now) /
# posted / failed (needs Retry) / cancelled (terminal).
QUEUE_STATUSES = ("queued", "waiting_account", "not_set_up", "posting",
                  "posted", "failed", "cancelled")

# Worker timing.
WORKER_POLL_SEC = 20
WAITING_ACCOUNT_RETRY_MIN = 60  # re-check a missing account hourly

# YouTube Data API v3.
YT_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
YT_TOKEN_URL = "https://oauth2.googleapis.com/token"
YT_UPLOAD_INIT_URL = ("https://www.googleapis.com/upload/youtube/v3/videos"
                      "?uploadType=resumable&part=snippet,status")
YT_API_BASE = "https://www.googleapis.com/youtube/v3"
YT_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YT_UPLOAD_CHUNK = 8 * 1024 * 1024  # 8 MB — multiple of 256 KiB (required)
YT_CATEGORY_ID = "22"  # People & Blogs (categoryId is required by the API)
YT_TITLE_MAX = 100
YT_DESCRIPTION_MAX = 5000

_DB_PATH: Path | None = None
_WORKER_THREAD: threading.Thread | None = None
_WORKER_LOCK = threading.Lock()
# OAuth state tokens: state -> (user_id, created_at). In-memory on
# purpose — a restart just means the user re-clicks Connect.
_OAUTH_STATES: dict[str, tuple[str, float]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def set_post_db_path(db_path: Path) -> None:
    global _DB_PATH
    _DB_PATH = Path(db_path)


def _db() -> Path:
    if _DB_PATH is None:
        return ROOT / "clipcut.sqlite3"
    return _DB_PATH


# ---------------------------------------------------------------- time (IST in, UTC stored)


def parse_ist_datetime_local(value: str) -> datetime:
    """'2026-11-17T19:40' (datetime-local, no zone) -> aware UTC datetime.

    The value is read as Asia/Kolkata wall time, because that is the
    timezone the user lives in. Raises ValueError on a bad value.
    """
    text = (value or "").strip()
    try:
        naive = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"Scheduled time samajh nahi aaya: '{text}' — "
            "YYYY-MM-DDTHH:MM format me do (jaise 2026-11-17T19:40)")
    if naive.tzinfo is not None:
        return naive.astimezone(timezone.utc)
    return naive.replace(tzinfo=IST).astimezone(timezone.utc)


def format_ist(utc_iso: str) -> str:
    """UTC ISO -> '17 Nov 2026, 19:40 IST' for display."""
    try:
        dt = datetime.fromisoformat(utc_iso)
    except Exception:
        return utc_iso
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).strftime("%d %b %Y, %H:%M IST")


def format_ist_input(utc_iso: str) -> str:
    """UTC ISO -> '2026-11-17T19:40' for a datetime-local input."""
    try:
        dt = datetime.fromisoformat(utc_iso)
    except Exception:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).strftime("%Y-%m-%dT%H:%M")


# ---------------------------------------------------------------- tables


def init_post_tables(db_path: Path) -> None:
    """Create queue + accounts + attempts + settings tables (idempotent)."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS post_accounts (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            platform TEXT NOT NULL,
            channel_id TEXT,
            channel_name TEXT,
            status TEXT NOT NULL DEFAULT 'connected',
            access_token_enc TEXT,
            refresh_token_enc TEXT,
            token_expires_at TEXT,
            scopes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(user_id, platform)
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS post_queue (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            upload_id TEXT NOT NULL,
            clip_id TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            platform TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            hashtags TEXT NOT NULL DEFAULT '',
            privacy TEXT NOT NULL DEFAULT 'unlisted',
            scheduled_at TEXT NOT NULL,
            recurrence TEXT NOT NULL DEFAULT 'once',
            status TEXT NOT NULL DEFAULT 'queued',
            paused INTEGER NOT NULL DEFAULT 0,
            posted_url TEXT,
            posted_video_id TEXT,
            posted_at TEXT,
            last_error TEXT NOT NULL DEFAULT '',
            attempts INTEGER NOT NULL DEFAULT 0,
            next_retry_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS post_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            queue_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            platform TEXT NOT NULL,
            attempted_at TEXT NOT NULL,
            result TEXT NOT NULL,
            error TEXT NOT NULL DEFAULT '',
            video_id TEXT,
            video_url TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS post_settings (
            user_id TEXT PRIMARY KEY,
            paused_all INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )""")
        conn.commit()


# ---------------------------------------------------------------- accounts (YouTube OAuth)


def youtube_oauth_configured() -> bool:
    """True only when the user has supplied the OAuth client values."""
    return bool(os.getenv("GOOGLE_CLIENT_ID")
                and os.getenv("GOOGLE_CLIENT_SECRET")
                and os.getenv("YOUTUBE_REDIRECT_URI"))


def youtube_oauth_missing() -> list[str]:
    """Which of the three required values is missing (for the error)."""
    missing = []
    if not os.getenv("GOOGLE_CLIENT_ID"):
        missing.append("GOOGLE_CLIENT_ID")
    if not os.getenv("GOOGLE_CLIENT_SECRET"):
        missing.append("GOOGLE_CLIENT_SECRET")
    if not os.getenv("YOUTUBE_REDIRECT_URI"):
        missing.append("YOUTUBE_REDIRECT_URI")
    return missing


def youtube_auth_url(user_id: str) -> str:
    """Google authorization URL for the youtube.upload scope.

    Raises RuntimeError when the OAuth client is not configured —
    the caller turns that into a clean "Connect needs credentials"
    message, never a crash and never a fake connect.
    """
    missing = youtube_oauth_missing()
    if missing:
        raise RuntimeError(
            "YouTube connect ke liye Google OAuth client chahiye — "
            "yeh values set karo (Google Cloud Console se): "
            + ", ".join(missing)
            + " (+ usi project me YouTube Data API v3 enable karo).")
    import secrets as _secrets
    from urllib.parse import urlencode
    state = _secrets.token_urlsafe(24)
    _OAUTH_STATES[state] = (user_id, time.time())
    params = urlencode({
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "redirect_uri": os.getenv("YOUTUBE_REDIRECT_URI"),
        "response_type": "code",
        "scope": (YT_UPLOAD_SCOPE + " openid email profile"),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    })
    return f"{YT_AUTH_URL}?{params}"


def consume_oauth_state(state: str) -> str | None:
    """Validate a callback state token, return its user_id (once only)."""
    entry = _OAUTH_STATES.pop(state, None)
    if entry is None:
        return None
    user_id, created = entry
    if time.time() - created > 600:  # 10 min
        return None
    return user_id


def exchange_youtube_code(code: str,
                          token_url: str = YT_TOKEN_URL) -> dict:
    """Exchange an auth code for tokens. Raises RuntimeError on failure.

    token_url is a parameter only so tests can point it at a local
    mock server — production always uses Google's URL.
    """
    import requests
    try:
        resp = requests.post(token_url, data={
            "code": code,
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "redirect_uri": os.getenv("YOUTUBE_REDIRECT_URI"),
            "grant_type": "authorization_code",
        }, timeout=30)
    except Exception as exc:
        raise RuntimeError(f"Google se token lene me network error: {exc}")
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(
            f"Google token endpoint ne JSON nahi diya (HTTP {resp.status_code})")
    if resp.status_code != 200 or "access_token" not in data:
        raise RuntimeError(
            "Google token exchange fail hua: "
            + str(data.get("error_description") or data.get("error")
                  or f"HTTP {resp.status_code}"))
    return data


def get_youtube_channel(access_token: str,
                        api_base: str = YT_API_BASE) -> dict:
    """The authenticated user's own channel (id + title)."""
    import requests
    try:
        resp = requests.get(
            f"{api_base}/channels",
            params={"part": "snippet", "mine": "true"},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30)
    except Exception as exc:
        raise RuntimeError(f"YouTube channel lene me network error: {exc}")
    if resp.status_code != 200:
        raise RuntimeError(
            f"YouTube channel API fail hua (HTTP {resp.status_code}) — "
            "kya YouTube Data API v3 us project me enabled hai?")
    items = resp.json().get("items") or []
    if not items:
        raise RuntimeError(
            "Is Google account par koi YouTube channel nahi mila — "
            "pehle youtube.com par channel banao.")
    item = items[0]
    return {"channel_id": item.get("id") or "",
            "channel_name": (item.get("snippet") or {}).get("title") or ""}


def save_youtube_account(db_path: Path, user_id: str, tokens: dict,
                         channel: dict) -> dict:
    """Store (or replace) the user's YouTube account. Tokens are
    Fernet-encrypted at rest — plaintext kabhi DB me nahi jata."""
    from ai_providers import encrypt_key
    expires_at = ""
    if tokens.get("expires_in"):
        try:
            expires_at = (_now_dt()
                          + timedelta(seconds=int(tokens["expires_in"]))
                          ).isoformat()
        except (TypeError, ValueError):
            expires_at = ""
    now = _now()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """INSERT INTO post_accounts
               (id, user_id, platform, channel_id, channel_name, status,
                access_token_enc, refresh_token_enc, token_expires_at,
                scopes, created_at, updated_at)
               VALUES (?, ?, 'youtube', ?, ?, 'connected', ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, platform) DO UPDATE SET
                 channel_id=excluded.channel_id,
                 channel_name=excluded.channel_name,
                 status='connected',
                 access_token_enc=excluded.access_token_enc,
                 refresh_token_enc=excluded.refresh_token_enc,
                 token_expires_at=excluded.token_expires_at,
                 scopes=excluded.scopes,
                 updated_at=excluded.updated_at""",
            (uuid.uuid4().hex, user_id, channel.get("channel_id", ""),
             channel.get("channel_name", ""),
             encrypt_key(tokens["access_token"]) if tokens.get("access_token") else "",
             encrypt_key(tokens["refresh_token"]) if tokens.get("refresh_token") else "",
             expires_at, tokens.get("scope", ""), now, now))
        conn.commit()
    return get_youtube_account(db_path, user_id)


def get_youtube_account(db_path: Path, user_id: str) -> dict | None:
    """The user's YouTube account row (tokens stay encrypted here)."""
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT id, user_id, platform, channel_id, channel_name,"
            " status, access_token_enc, refresh_token_enc,"
            " token_expires_at, scopes, created_at, updated_at"
            " FROM post_accounts WHERE user_id=? AND platform='youtube'",
            (user_id,)).fetchone()
    if not row:
        return None
    keys = ("id", "user_id", "platform", "channel_id", "channel_name",
            "status", "access_token_enc", "refresh_token_enc",
            "token_expires_at", "scopes", "created_at", "updated_at")
    return dict(zip(keys, row))


def delete_youtube_account(db_path: Path, user_id: str) -> bool:
    """Disconnect YouTube (tokens deleted). Queued YouTube items stay
    queued — the worker will mark them waiting_account instead."""
    with closing(sqlite3.connect(db_path)) as conn:
        cur = conn.execute(
            "DELETE FROM post_accounts WHERE user_id=? AND platform='youtube'",
            (user_id,))
        conn.commit()
    return cur.rowcount > 0


def _decrypt_account_token(account: dict, field: str) -> str | None:
    from ai_providers import decrypt_key
    enc = account.get(field) or ""
    if not enc:
        return None
    try:
        return decrypt_key(enc)
    except Exception:
        return None


def refresh_youtube_access_token(db_path: Path, user_id: str,
                                 token_url: str = YT_TOKEN_URL) -> str:
    """Refresh the access token with the stored refresh token.

    Returns a fresh access token (and saves it). Raises RuntimeError
    when there is no refresh token or Google refuses it — in that
    case the user must reconnect (the account row is marked
    'needs_reconnect').
    """
    import requests
    account = get_youtube_account(db_path, user_id)
    if not account:
        raise RuntimeError("YouTube account connected nahi hai.")
    refresh_token = _decrypt_account_token(account, "refresh_token_enc")
    if not refresh_token:
        raise RuntimeError(
            "YouTube account me refresh token nahi hai — dobara Connect karo.")
    try:
        resp = requests.post(token_url, data={
            "refresh_token": refresh_token,
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "grant_type": "refresh_token",
        }, timeout=30)
    except Exception as exc:
        raise RuntimeError(f"Google se token refresh me network error: {exc}")
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(
            f"Google token endpoint ne JSON nahi diya (HTTP {resp.status_code})")
    if resp.status_code != 200 or "access_token" not in data:
        _mark_account_needs_reconnect(db_path, user_id)
        raise RuntimeError(
            "YouTube token refresh fail hua ("
            + str(data.get("error_description") or data.get("error")
                  or f"HTTP {resp.status_code}")
            + ") — YouTube ko dobara Connect karo.")
    from ai_providers import encrypt_key
    expires_at = ""
    if data.get("expires_in"):
        try:
            expires_at = (_now_dt()
                          + timedelta(seconds=int(data["expires_in"]))
                          ).isoformat()
        except (TypeError, ValueError):
            expires_at = ""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "UPDATE post_accounts SET access_token_enc=?,"
            " token_expires_at=?, updated_at=? WHERE user_id=? AND platform='youtube'",
            (encrypt_key(data["access_token"]), expires_at, _now(), user_id))
        conn.commit()
    return str(data["access_token"])


def _mark_account_needs_reconnect(db_path: Path, user_id: str) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "UPDATE post_accounts SET status='needs_reconnect', updated_at=?"
            " WHERE user_id=? AND platform='youtube'", (_now(), user_id))
        conn.commit()


def get_valid_access_token(db_path: Path, user_id: str,
                           token_url: str = YT_TOKEN_URL) -> str:
    """A usable access token: the stored one while it is fresh,
    else a refresh. Raises RuntimeError when no account / no token.

    token_url is a parameter only so tests can point it at a local
    mock server — production always uses Google's URL.
    """
    account = get_youtube_account(db_path, user_id)
    if not account:
        raise RuntimeError("YouTube account connected nahi hai.")
    if account.get("status") == "needs_reconnect":
        raise RuntimeError(
            "YouTube token expire ho gaya — Settings > Connected apps "
            "me dobara Connect karo.")
    expires_at = account.get("token_expires_at") or ""
    fresh = False
    if expires_at:
        try:
            fresh = (datetime.fromisoformat(expires_at)
                     > _now_dt() + timedelta(seconds=60))
        except ValueError:
            fresh = False
    if fresh:
        token = _decrypt_account_token(account, "access_token_enc")
        if token:
            return token
    return refresh_youtube_access_token(db_path, user_id,
                                        token_url=token_url)


# ---------------------------------------------------------------- YouTube upload (real, resumable)


def upload_youtube_short(db_path: Path, user_id: str, video_path: Path,
                         title: str, description: str = "",
                         tags: list[str] | None = None,
                         privacy: str = "unlisted",
                         upload_init_url: str = YT_UPLOAD_INIT_URL,
                         api_base: str = YT_API_BASE,
                         token_url: str = YT_TOKEN_URL) -> dict:
    """Upload one rendered mp4 as a YouTube Short (resumable upload).

    Returns {"video_id": ..., "url": "https://www.youtube.com/shorts/<id>"}.
    Raises RuntimeError with the real API error on failure — never a
    fake success. upload_init_url / api_base are parameters only so
    tests can point them at a local mock server; production always
    uses Google's URLs.
    """
    import requests
    if privacy not in PRIVACY_OPTIONS:
        raise ValueError(f"privacy galat hai: {privacy}")
    if not video_path.is_file():
        raise RuntimeError(f"Video file nahi mili: {video_path}")
    size = video_path.stat().st_size
    if size == 0:
        raise RuntimeError("Video file khaali hai (0 bytes) — upload nahi hoga.")
    title = (title or "").strip()[:YT_TITLE_MAX]
    if not title:
        raise ValueError("YouTube title khaali nahi ho sakta")
    description = (description or "").strip()[:YT_DESCRIPTION_MAX]
    clean_tags = []
    total_len = 0
    for tag in (tags or []):
        tag = str(tag).strip().lstrip("#")
        if not tag or len(tag) > 30:
            continue
        if total_len + len(tag) + 1 > 500:
            break
        clean_tags.append(tag)
        total_len += len(tag) + 1

    access_token = get_valid_access_token(db_path, user_id,
                                          token_url=token_url)
    headers = {"Authorization": f"Bearer {access_token}",
               "Content-Type": "application/json; charset=UTF-8",
               "X-Upload-Content-Type": "video/mp4",
               "X-Upload-Content-Length": str(size)}
    metadata = {"snippet": {"title": title, "description": description,
                            "tags": clean_tags,
                            "categoryId": YT_CATEGORY_ID},
                "status": {"privacyStatus": privacy,
                           "selfDeclaredMadeForKids": False}}
    try:
        init = requests.post(upload_init_url, headers=headers,
                             data=json.dumps(metadata), timeout=60)
    except Exception as exc:
        raise RuntimeError(f"YouTube upload start me network error: {exc}")
    if init.status_code == 401:
        # Token went stale between the check and the call — refresh
        # once and retry the init.
        access_token = refresh_youtube_access_token(db_path, user_id,
                                                    token_url=token_url)
        headers["Authorization"] = f"Bearer {access_token}"
        try:
            init = requests.post(upload_init_url, headers=headers,
                                 data=json.dumps(metadata), timeout=60)
        except Exception as exc:
            raise RuntimeError(f"YouTube upload start me network error: {exc}")
    if init.status_code not in (200, 201):
        raise RuntimeError(
            "YouTube upload session start fail hua (HTTP "
            f"{init.status_code}): {_api_error_text(init)}")
    session_url = init.headers.get("Location") or ""
    if not session_url:
        raise RuntimeError(
            "YouTube ne upload session URL nahi diya (Location header missing).")

    # Chunked PUT with Content-Range; 308 = keep going.
    uploaded = 0
    with open(video_path, "rb") as fh:
        while uploaded < size:
            chunk = fh.read(YT_UPLOAD_CHUNK)
            if not chunk:
                break
            end = uploaded + len(chunk) - 1
            put_headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "video/mp4",
                "Content-Length": str(len(chunk)),
                "Content-Range": f"bytes {uploaded}-{end}/{size}",
            }
            try:
                resp = requests.put(session_url, headers=put_headers,
                                    data=chunk, timeout=120)
            except Exception as exc:
                raise RuntimeError(
                    f"YouTube upload chunk fail hua ({uploaded}-{end}): {exc}")
            if resp.status_code in (200, 201):
                try:
                    video_id = resp.json().get("id") or ""
                except Exception:
                    video_id = ""
                if not video_id:
                    raise RuntimeError(
                        "YouTube ne upload complete kiya par video id nahi diya.")
                return {"video_id": video_id,
                        "url": f"https://www.youtube.com/shorts/{video_id}"}
            if resp.status_code == 308:
                # Resume Incomplete — Range header says how much landed.
                ranged = resp.headers.get("Range") or ""
                if ranged.startswith("bytes=0-"):
                    try:
                        uploaded = int(ranged.split("=", 1)[1].rsplit("-", 1)[1]) + 1
                    except ValueError:
                        uploaded = end + 1
                else:
                    uploaded = end + 1
                fh.seek(uploaded)
                continue
            raise RuntimeError(
                "YouTube upload chunk fail hua (HTTP "
                f"{resp.status_code}): {_api_error_text(resp)}")
    raise RuntimeError(
        "YouTube upload khatam hua par Google ne video id nahi diya — "
        "upload adhoora ho sakta hai, YouTube Studio me check karo.")


def _api_error_text(resp) -> str:
    """Short honest error text from a Google API response."""
    try:
        data = resp.json()
        err = data.get("error") or {}
        msg = err.get("message") or ""
        reason = ""
        details = err.get("errors") or []
        if details:
            reason = str(details[0].get("reason") or "")
        text = " ".join(p for p in (msg, reason) if p)
        return text[:300] if text else f"HTTP {resp.status_code}"
    except Exception:
        return (resp.text or "")[:300] or f"HTTP {resp.status_code}"


# ---------------------------------------------------------------- queue CRUD


def _item_dict(row) -> dict:
    keys = ("id", "user_id", "upload_id", "clip_id", "version", "platform",
            "title", "description", "hashtags", "privacy", "scheduled_at",
            "recurrence", "status", "paused", "posted_url",
            "posted_video_id", "posted_at", "last_error", "attempts",
            "next_retry_at", "created_at", "updated_at")
    d = dict(zip(keys, row))
    d["paused"] = bool(d["paused"])
    return d


def queue_clip(db_path: Path, user_id: str, upload_id: str, clip_id: str,
               version: int, platform: str, title: str,
               description: str = "", hashtags: str = "",
               privacy: str = "unlisted",
               scheduled_at: datetime | None = None,
               recurrence: str = "once") -> dict:
    """Queue one rendered clip version. Validates everything the
    worker will need later (platform, title, privacy, recurrence,
    scheduled time in the future). Raises ValueError on bad input."""
    if platform not in PLATFORMS:
        raise ValueError(
            f"Platform galat hai: {platform} — "
            f"{', '.join(PLATFORMS)} me se ek do")
    title = (title or "").strip()
    if not title:
        raise ValueError("Post ka title khaali nahi ho sakta")
    if len(title) > YT_TITLE_MAX:
        raise ValueError(
            f"Title {YT_TITLE_MAX} characters se zyada nahi ho sakta "
            f"(mila: {len(title)})")
    if len(description or "") > YT_DESCRIPTION_MAX:
        raise ValueError(
            f"Description {YT_DESCRIPTION_MAX} characters se zyada nahi "
            f"ho sakta (mila: {len(description)})")
    if privacy not in PRIVACY_OPTIONS:
        raise ValueError(f"privacy galat hai: {privacy}")
    if recurrence not in RECURRENCES:
        raise ValueError(f"recurrence galat hai: {recurrence}")
    if scheduled_at is None:
        raise ValueError("Scheduled time chahiye")
    if scheduled_at.tzinfo is None:
        scheduled_at = scheduled_at.replace(tzinfo=timezone.utc)
    if scheduled_at <= _now_dt():
        raise ValueError(
            "Scheduled time aane wala hona chahiye — pehle ka time nahi "
            "(Retry ke liye yeh check nahi lagta).")
    item_id = uuid.uuid4().hex
    now = _now()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """INSERT INTO post_queue
               (id, user_id, upload_id, clip_id, version, platform,
                title, description, hashtags, privacy, scheduled_at,
                recurrence, status, paused, last_error, attempts,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0,
                       '', 0, ?, ?)""",
            (item_id, user_id, upload_id, clip_id, int(version), platform,
             title, description or "", hashtags or "", privacy,
             scheduled_at.astimezone(timezone.utc).isoformat(),
             recurrence, now, now))
        conn.commit()
    return get_queue_item(db_path, item_id)


def get_queue_item(db_path: Path, item_id: str) -> dict | None:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT id, user_id, upload_id, clip_id, version, platform,"
            " title, description, hashtags, privacy, scheduled_at,"
            " recurrence, status, paused, posted_url, posted_video_id,"
            " posted_at, last_error, attempts, next_retry_at, created_at,"
            " updated_at FROM post_queue WHERE id=?", (item_id,)).fetchone()
    return _item_dict(row) if row else None


def list_queue(db_path: Path, user_id: str,
               status: str | None = None) -> list[dict]:
    with closing(sqlite3.connect(db_path)) as conn:
        if status:
            rows = conn.execute(
                "SELECT id, user_id, upload_id, clip_id, version, platform,"
                " title, description, hashtags, privacy, scheduled_at,"
                " recurrence, status, paused, posted_url, posted_video_id,"
                " posted_at, last_error, attempts, next_retry_at, created_at,"
                " updated_at FROM post_queue WHERE user_id=? AND status=?"
                " ORDER BY scheduled_at ASC", (user_id, status)).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, user_id, upload_id, clip_id, version, platform,"
                " title, description, hashtags, privacy, scheduled_at,"
                " recurrence, status, paused, posted_url, posted_video_id,"
                " posted_at, last_error, attempts, next_retry_at, created_at,"
                " updated_at FROM post_queue WHERE user_id=?"
                " ORDER BY scheduled_at ASC", (user_id,)).fetchall()
    return [_item_dict(r) for r in rows]


def queue_counts(db_path: Path, user_id: str) -> dict:
    """Real counts for the dashboard tiles + queue page."""
    today = datetime.now(timezone.utc).date().isoformat()
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM post_queue WHERE user_id=?"
            " GROUP BY status", (user_id,)).fetchall()
        posted_today = conn.execute(
            "SELECT COUNT(*) FROM post_queue WHERE user_id=?"
            " AND status='posted' AND substr(posted_at, 1, 10)=?",
            (user_id, today)).fetchone()[0]
    counts = {s: 0 for s in QUEUE_STATUSES}
    for status, n in rows:
        counts[status] = int(n)
    waiting = (counts["queued"] + counts["waiting_account"]
               + counts["not_set_up"])
    return {"by_status": counts, "posted_today": int(posted_today),
            "waiting": waiting,
            "posted_total": counts["posted"],
            "failed": counts["failed"]}


def posted_per_day(db_path: Path, user_id: str, days: int = 14) -> dict:
    """{YYYY-MM-DD: posted count} for the dashboard activity chart."""
    start = (datetime.now(timezone.utc).date()
             - timedelta(days=days - 1)).isoformat()
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT substr(posted_at, 1, 10), COUNT(*) FROM post_queue"
            " WHERE user_id=? AND status='posted' AND posted_at>=?"
            " GROUP BY substr(posted_at, 1, 10)", (user_id, start)).fetchall()
    return {day: int(n) for day, n in rows}


def _set_item(db_path: Path, item_id: str, **fields) -> None:
    fields["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in fields)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(f"UPDATE post_queue SET {sets} WHERE id=?",
                     (*fields.values(), item_id))
        conn.commit()


def pause_item(db_path: Path, item_id: str) -> dict | None:
    item = get_queue_item(db_path, item_id)
    if not item or item["status"] in ("posted", "cancelled"):
        return item
    _set_item(db_path, item_id, paused=1)
    return get_queue_item(db_path, item_id)


def resume_item(db_path: Path, item_id: str) -> dict | None:
    item = get_queue_item(db_path, item_id)
    if not item:
        return item
    _set_item(db_path, item_id, paused=0)
    return get_queue_item(db_path, item_id)


def cancel_item(db_path: Path, item_id: str) -> dict | None:
    item = get_queue_item(db_path, item_id)
    if not item or item["status"] in ("posted", "cancelled"):
        return item
    _set_item(db_path, item_id, status="cancelled")
    return get_queue_item(db_path, item_id)


def retry_item(db_path: Path, item_id: str) -> dict | None:
    """Failed / waiting_account / not_set_up -> queued again.

    The original scheduled time is kept (usually in the past by now),
    so the worker picks it up on its next pass. Posted/cancelled
    items cannot be retried.
    """
    item = get_queue_item(db_path, item_id)
    if not item:
        return item
    if item["status"] not in ("failed", "waiting_account", "not_set_up"):
        return item
    _set_item(db_path, item_id, status="queued", paused=0,
              last_error="", next_retry_at=None)
    return get_queue_item(db_path, item_id)


def set_pause_all(db_path: Path, user_id: str, paused: bool) -> bool:
    now = _now()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """INSERT INTO post_settings (user_id, paused_all, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                 paused_all=excluded.paused_all,
                 updated_at=excluded.updated_at""",
            (user_id, 1 if paused else 0, now))
        conn.commit()
    return bool(paused)


def is_paused_all(db_path: Path, user_id: str) -> bool:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT paused_all FROM post_settings WHERE user_id=?",
            (user_id,)).fetchone()
    return bool(row and row[0])


def log_attempt(db_path: Path, queue_id: str, user_id: str, platform: str,
                result: str, error: str = "", video_id: str = "",
                video_url: str = "") -> None:
    """Every attempt is recorded — including the ones that never post
    (waiting_account / not_set_up / failed). Never raises."""
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                "INSERT INTO post_attempts (queue_id, user_id, platform,"
                " attempted_at, result, error, video_id, video_url)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (queue_id, user_id, platform, _now(), result,
                 (error or "")[:500], video_id, video_url))
            conn.commit()
    except Exception:
        pass


def list_attempts(db_path: Path, queue_id: str) -> list[dict]:
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, queue_id, user_id, platform, attempted_at, result,"
            " error, video_id, video_url FROM post_attempts"
            " WHERE queue_id=? ORDER BY attempted_at ASC",
            (queue_id,)).fetchall()
    keys = ("id", "queue_id", "user_id", "platform", "attempted_at",
            "result", "error", "video_id", "video_url")
    return [dict(zip(keys, r)) for r in rows]


# ---------------------------------------------------------------- worker (one item at a time)


def ensure_post_worker() -> None:
    """Start the single posting worker thread (daemon, idempotent)."""
    global _WORKER_THREAD
    with _WORKER_LOCK:
        if _WORKER_THREAD is None or not _WORKER_THREAD.is_alive():
            _WORKER_THREAD = threading.Thread(target=_worker_loop,
                                              daemon=True,
                                              name="clipcut-post-worker")
            _WORKER_THREAD.start()


def post_worker_status() -> dict:
    running = bool(_WORKER_THREAD is not None and _WORKER_THREAD.is_alive())
    return {"worker_running": running,
            "poll_seconds": WORKER_POLL_SEC,
            "youtube_oauth_configured": youtube_oauth_configured()}


def reconcile_post_state(db_path: Path) -> dict:
    """Startup sweep: items stuck in 'posting' after a restart become
    'failed' with an honest note (the user can Retry them)."""
    with closing(sqlite3.connect(db_path)) as conn:
        stuck = conn.execute(
            "SELECT id FROM post_queue WHERE status='posting'").fetchall()
        for (item_id,) in stuck:
            conn.execute(
                "UPDATE post_queue SET status='failed',"
                " last_error='Server restart hua — post adhoora reh gaya,"
                " Retry dabao', updated_at=? WHERE id=?", (_now(), item_id))
        conn.commit()
    return {"stuck_posting_marked_failed": len(stuck)}


def _due_item(db_path: Path) -> dict | None:
    """One due item: scheduled time passed, not item-paused, user has
    not paused all posting, retry time (if any) passed."""
    now = _now()
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT id, user_id, upload_id, clip_id, version, platform,"
            " title, description, hashtags, privacy, scheduled_at,"
            " recurrence, status, paused, posted_url, posted_video_id,"
            " posted_at, last_error, attempts, next_retry_at, created_at,"
            " updated_at FROM post_queue"
            " WHERE status IN ('queued', 'waiting_account')"
            " AND paused=0 AND scheduled_at<=?"
            " AND (next_retry_at IS NULL OR next_retry_at<=?)"
            " AND NOT EXISTS (SELECT 1 FROM post_settings ps"
            "   WHERE ps.user_id=post_queue.user_id AND ps.paused_all=1)"
            " ORDER BY scheduled_at ASC LIMIT 1", (now, now)).fetchone()
    return _item_dict(row) if row else None


def _worker_loop() -> None:
    while True:
        try:
            item = _due_item(_db())
            if item is not None:
                _process_item(item)
        except Exception:
            # Never kill the worker on one item — next pass retries.
            pass
        time.sleep(WORKER_POLL_SEC)


def _process_item(item: dict) -> None:
    """Attempt one due item. Every outcome is recorded; nothing is
    ever marked posted without a real video id from the platform."""
    db = _db()
    platform = item["platform"]
    item_id = item["id"]
    user_id = item["user_id"]

    if platform in ("instagram", "tiktok"):
        # Reference app: "Not set up on this server" — same here.
        error = (f"Not set up on this server — {PLATFORMS[platform]['label']} "
                 "posting ke liye official API + approval chahiye "
                 "(CreativeCut me bhi yahi likha tha).")
        _set_item(db, item_id, status="not_set_up", last_error=error,
                  next_retry_at=None)
        log_attempt(db, item_id, user_id, platform, "not_set_up", error)
        return

    # YouTube Shorts.
    account = get_youtube_account(db, user_id)
    if not account or account.get("status") == "needs_reconnect":
        if account and account.get("status") == "needs_reconnect":
            error = ("YouTube token expire ho gaya — Settings > Connected "
                     "apps me dobara Connect karo.")
        else:
            error = ("YouTube account connected nahi hai — Settings > "
                     "Connected apps me YouTube Connect karo (Google OAuth "
                     "client chahiye: GOOGLE_CLIENT_ID, "
                     "GOOGLE_CLIENT_SECRET, YOUTUBE_REDIRECT_URI).")
        retry_at = (_now_dt()
                    + timedelta(minutes=WAITING_ACCOUNT_RETRY_MIN)).isoformat()
        _set_item(db, item_id, status="waiting_account", last_error=error,
                  next_retry_at=retry_at)
        log_attempt(db, item_id, user_id, platform, "waiting_account", error)
        return

    # Resolve the rendered file (must exist on disk — the worker
    # uploads the actual mp4, never a placeholder).
    try:
        from editor import find_render_info
        meta = find_render_info(item["upload_id"], item["clip_id"],
                                version=item["version"])
        # find_render_info returns None (not an exception) when no
        # render exists — check explicitly.
        if meta is None:
            raise FileNotFoundError(
                f"render v{item['version']} nahi mila")
        video_path = Path(meta.get("output_mp4") or "")
    except Exception as exc:
        error = f"Rendered clip nahi mila (v{item['version']}): {exc}"
        _set_item(db, item_id, status="failed", last_error=error[:500])
        log_attempt(db, item_id, user_id, platform, "failed", error)
        return
    if not video_path.is_file():
        error = f"Rendered file disk par nahi hai: {video_path}"
        _set_item(db, item_id, status="failed", last_error=error[:500])
        log_attempt(db, item_id, user_id, platform, "failed", error)
        return

    _set_item(db, item_id, status="posting",
              attempts=item["attempts"] + 1)
    tags = [t for t in (item["hashtags"] or "").replace(",", " ").split()
            if t]
    description = item["description"] or ""
    if item["hashtags"]:
        description = (description + "\n" + item["hashtags"]).strip()
    try:
        result = upload_youtube_short(
            db, user_id, video_path, title=item["title"],
            description=description, tags=tags,
            privacy=item["privacy"] or "unlisted")
    except Exception as exc:
        error = str(exc)[:500]
        _set_item(db, item_id, status="failed", last_error=error)
        log_attempt(db, item_id, user_id, platform, "failed", error)
        return

    now = _now()
    _set_item(db, item_id, status="posted",
              posted_url=result["url"], posted_video_id=result["video_id"],
              posted_at=now, last_error="")
    log_attempt(db, item_id, user_id, platform, "posted", "",
                video_id=result["video_id"], video_url=result["url"])
    # Recurring slot: queue the next occurrence (same clip, next time).
    if item["recurrence"] in ("daily", "weekly"):
        try:
            scheduled = datetime.fromisoformat(item["scheduled_at"])
            if scheduled.tzinfo is None:
                scheduled = scheduled.replace(tzinfo=timezone.utc)
            delta = timedelta(days=1 if item["recurrence"] == "daily" else 7)
            queue_clip(db, user_id, item["upload_id"], item["clip_id"],
                       item["version"], platform, item["title"],
                       description=item["description"],
                       hashtags=item["hashtags"],
                       privacy=item["privacy"] or "unlisted",
                       scheduled_at=scheduled + delta,
                       recurrence=item["recurrence"])
        except Exception:
            pass
