"""ClipCut — Phase 1: auth + creators + storage quota.

Email/password auth with scrypt password hashing (same approach the
teardown observed: one-way scrypt hashes, one essential httpOnly
session cookie, no advertising/analytics cookies).

Google OAuth is a PLACEHOLDER: /auth/google only works when the
operator sets GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET /
GOOGLE_REDIRECT_URI in the environment. No client id/secret is
invented or stored here — with them unset the route explains that
Google sign-in is not configured yet. The callback exchange code
exists but is UNTESTED (no credentials exist to test it with).

Storage quota: 100 MB per user (teardown value). Uploads are
counted per user_id; enforce_storage_quota() deletes the OLDEST
uploads (file + DB row + sidecar transcript/clips files) until usage
is back under the limit. It is a plain function so it can be called
from a scheduled job later (Phase 10) — Phase 1 calls it after
each upload is saved.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

SESSION_COOKIE = "clipcut_session"
SESSION_DAYS = 30
STORAGE_QUOTA_BYTES = 100 * 1024 * 1024  # 100 MB, per teardown

# ---------------------------------------------------------------- users


def hash_password(password: str) -> str:
    """scrypt hash: 'scrypt$<salt hex>$<hash hex>' (never reversible)."""
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                        n=16384, r=8, p=1, dklen=64)
    return f"scrypt${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.scrypt(password.encode("utf-8"),
                            salt=bytes.fromhex(salt_hex),
                            n=16384, r=8, p=1, dklen=64)
        return secrets.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def init_auth_tables(db_path: Path) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT,
            auth_provider TEXT NOT NULL DEFAULT 'email',
            email_confirmed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS creators (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            channel_link TEXT NOT NULL,
            platform TEXT NOT NULL DEFAULT 'youtube',
            created_at TEXT NOT NULL
        )""")
        conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_user(db_path: Path, name: str, email: str,
                password: str | None) -> dict:
    """Create an email user. Raises ValueError on duplicate email or
    a password shorter than 8 characters."""
    email = email.strip().lower()
    if len(password or "") < 8:
        raise ValueError("Password kam se kam 8 characters ka hona chahiye")
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute("SELECT id FROM users WHERE email=?",
                           (email,)).fetchone()
        if row:
            raise ValueError("Is email se account pehle se bana hua hai")
        user_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, 'email', 0, ?)",
            (user_id, name.strip(), email,
             hash_password(password) if password else None, _now()))
        conn.commit()
    return get_user_by_id(db_path, user_id)


def get_user_by_email(db_path: Path, email: str) -> dict | None:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT id, name, email, password_hash, auth_provider,"
            " email_confirmed, created_at FROM users WHERE email=?",
            (email.strip().lower(),)).fetchone()
    return _user_dict(row) if row else None


def get_user_by_id(db_path: Path, user_id: str) -> dict | None:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT id, name, email, password_hash, auth_provider,"
            " email_confirmed, created_at FROM users WHERE id=?",
            (user_id,)).fetchone()
    return _user_dict(row) if row else None


def _user_dict(row) -> dict:
    return {"id": row[0], "name": row[1], "email": row[2],
            "password_hash": row[3], "auth_provider": row[4],
            "email_confirmed": bool(row[5]), "created_at": row[6]}


def authenticate(db_path: Path, email: str, password: str) -> dict | None:
    """Return the user on a correct password, else None. Google-only
    accounts (password_hash NULL) can never log in by email."""
    user = get_user_by_email(db_path, email)
    if not user or not user["password_hash"]:
        return None
    return user if verify_password(password, user["password_hash"]) else None


def update_user(db_path: Path, user_id: str, name: str | None = None,
                password: str | None = None) -> dict:
    """Rename and/or set a new password (min 8 chars)."""
    with closing(sqlite3.connect(db_path)) as conn:
        if name is not None:
            conn.execute("UPDATE users SET name=? WHERE id=?",
                         (name.strip(), user_id))
        if password is not None:
            if len(password) < 8:
                raise ValueError("Password kam se kam 8 characters ka hona chahiye")
            conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                         (hash_password(password), user_id))
        conn.commit()
    return get_user_by_id(db_path, user_id)


def delete_user(db_path: Path, user_id: str, root: Path) -> dict:
    """Delete the account and everything owned by it: sessions,
    creators, and uploads filed under this user (files + DB rows +
    transcript/clips sidecars). Uploads with no owner (legacy /
    anonymous rows, user_id NULL) are NOT deleted — they belong to
    nobody, so deleting them would remove another person's work.
    Returns counts so the caller can report what happened."""
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, stored_path FROM uploads WHERE user_id=?",
            (user_id,)).fetchall()
        creator_ids = [r[0] for r in conn.execute(
            "SELECT id FROM creators WHERE user_id=?", (user_id,)).fetchall()]
        conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM creators WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM uploads WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
    deleted_files = 0
    for upload_id, stored_path in rows:
        for p in [root / stored_path,
                  root / "outputs" / "transcripts" / f"{upload_id}.json",
                  root / "outputs" / "transcripts" / f"{upload_id}.txt",
                  root / "outputs" / "clips" / f"{upload_id}_clips.json"]:
            try:
                if p.is_file():
                    p.unlink()
                    deleted_files += 1
            except OSError:
                pass
        rendered = root / "outputs" / "rendered" / upload_id
        if rendered.is_dir():
            for p in rendered.iterdir():
                try:
                    p.unlink()
                    deleted_files += 1
                except OSError:
                    pass
            try:
                rendered.rmdir()
            except OSError:
                pass
    return {"deleted_uploads": len(rows), "deleted_files": deleted_files,
            "deleted_creators": len(creator_ids)}


# ---------------------------------------------------------------- sessions


def create_session(db_path: Path, user_id: str) -> str:
    """Create a session, return the raw token (goes in the cookie).
    Only the SHA-256 of the token is stored."""
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)",
                     (hashlib.sha256(token.encode()).hexdigest(), user_id,
                      _now(), expires.isoformat()))
        conn.commit()
    return token


def get_user_by_token(db_path: Path, token: str | None) -> dict | None:
    if not token:
        return None
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT user_id, expires_at FROM sessions WHERE token_hash=?",
            (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
    if not row:
        return None
    if datetime.fromisoformat(row[1]) < datetime.now(timezone.utc):
        delete_session(db_path, token)
        return None
    return get_user_by_id(db_path, row[0])


def delete_session(db_path: Path, token: str) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash=?",
                     (hashlib.sha256(token.encode()).hexdigest(),))
        conn.commit()


def delete_all_sessions(db_path: Path, user_id: str) -> None:
    """Sign out everywhere (Security page)."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
        conn.commit()


# ---------------------------------------------------------------- creators


def detect_platform(channel_link: str) -> str:
    link = channel_link.lower()
    if "twitch.tv" in link:
        return "twitch"
    if "kick.com" in link:
        return "kick"
    return "youtube"


def create_creator(db_path: Path, user_id: str, name: str,
                   channel_link: str) -> dict:
    channel_link = channel_link.strip()
    if not channel_link.startswith("http"):
        raise ValueError("Channel link http(s) se shuru hona chahiye")
    creator_id = uuid.uuid4().hex
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("INSERT INTO creators VALUES (?, ?, ?, ?, ?, ?)",
                     (creator_id, user_id, name.strip() or channel_link,
                      channel_link, detect_platform(channel_link), _now()))
        conn.commit()
    return get_creator(db_path, creator_id)


def list_creators(db_path: Path, user_id: str) -> list[dict]:
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, user_id, name, channel_link, platform, created_at"
            " FROM creators WHERE user_id=? ORDER BY created_at ASC",
            (user_id,)).fetchall()
    return [{"id": r[0], "user_id": r[1], "name": r[2],
             "channel_link": r[3], "platform": r[4],
             "created_at": r[5]} for r in rows]


def get_creator(db_path: Path, creator_id: str) -> dict | None:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT id, user_id, name, channel_link, platform, created_at"
            " FROM creators WHERE id=?", (creator_id,)).fetchone()
    if not row:
        return None
    return {"id": row[0], "user_id": row[1], "name": row[2],
            "channel_link": row[3], "platform": row[4],
            "created_at": row[5]}


def delete_creator(db_path: Path, user_id: str, creator_id: str) -> bool:
    """Delete a creator of this user. Videos filed under it are kept
    but unfiled (creator_id set to NULL) — deleting a workspace must
    never silently delete the videos inside it."""
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT id FROM creators WHERE id=? AND user_id=?",
            (creator_id, user_id)).fetchone()
        if not row:
            return False
        conn.execute("UPDATE uploads SET creator_id=NULL WHERE creator_id=?",
                     (creator_id,))
        conn.execute("DELETE FROM creators WHERE id=?", (creator_id,))
        conn.commit()
    return True


# ---------------------------------------------------------------- storage quota


def storage_usage_bytes(db_path: Path, user_id: str) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM uploads"
            " WHERE user_id=?", (user_id,)).fetchone()
    return int(row[0])


def enforce_storage_quota(db_path: Path, root: Path, user_id: str,
                          limit_bytes: int = STORAGE_QUOTA_BYTES) -> list[dict]:
    """Delete the user's OLDEST uploads until usage <= limit.
    Returns the deleted uploads (id, name, size). No-op when under
    the limit. Sidecar transcript/clips files of a deleted upload
    are removed too; rendered clips of other uploads are untouched."""
    deleted: list[dict] = []
    while storage_usage_bytes(db_path, user_id) > limit_bytes:
        with closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute(
                "SELECT id, original_name, stored_path, size_bytes"
                " FROM uploads WHERE user_id=? ORDER BY created_at ASC"
                " LIMIT 1", (user_id,)).fetchone()
            if not row:
                break
            conn.execute("DELETE FROM uploads WHERE id=?", (row[0],))
            conn.commit()
        upload_id, name, stored_path, size = row
        for p in [root / stored_path,
                  root / "outputs" / "transcripts" / f"{upload_id}.json",
                  root / "outputs" / "transcripts" / f"{upload_id}.txt",
                  root / "outputs" / "clips" / f"{upload_id}_clips.json"]:
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass
        deleted.append({"id": upload_id, "name": name, "size_bytes": size})
    return deleted


# ---------------------------------------------------------------- Google OAuth (placeholder)


def google_oauth_configured() -> bool:
    return bool(os.getenv("GOOGLE_CLIENT_ID")
                and os.getenv("GOOGLE_CLIENT_SECRET"))


def google_auth_url() -> str | None:
    """Authorization URL, or None when Google sign-in is not
    configured (no credentials invented)."""
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not client_id or not redirect_uri:
        return None
    from urllib.parse import urlencode
    params = urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    })
    return f"https://accounts.google.com/o/oauth2/v2/auth?{params}"
