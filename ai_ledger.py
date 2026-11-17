"""ClipCut — Phase 6: token ledger (AI usage).

Every AI call (and every AI attempt, including the rule-based
fallback) is logged with when, what for, creator, video, provider,
model, input/output/total tokens, method and status. The usage page
(/app/usage) reads only this table — nothing is estimated there
except rows whose token_source is "estimated" (and those say so).

Purposes (teardown §1.8 sample rows map to these):
- "finding clips" — one summary row per find_clips run (0 tokens;
  the real tokens live on the "words judge" row, so totals are
  never double-counted). Note records candidates/kept counts.
- "words judge" — the LLM call, or the rule-based fallback (0 tokens).
- "reaction judge" — loudness (+ chat when supplied); always 0 tokens.
- "hand clip" — manual clip creation (informational scores, 0 tokens).
- "title+hook" — reserved for a future standalone hook/title call;
  today the hook comes back inside the words-judge reply, so no
  separate row is written for it (writing one would double-count).
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date as _date, timedelta as _td
from pathlib import Path

PURPOSES = ("finding clips", "words judge", "reaction judge",
            "title+hook", "hand clip")


def init_ai_tables(db_path: Path) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS ai_settings (
            user_id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            key_enc TEXT NOT NULL DEFAULT '',
            key_last4 TEXT NOT NULL DEFAULT '',
            custom_base_url TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS ai_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            created_at TEXT NOT NULL,
            purpose TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            upload_id TEXT,
            creator_id TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            token_source TEXT NOT NULL DEFAULT 'none',
            method TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'ok',
            note TEXT NOT NULL DEFAULT ''
        )""")
        conn.commit()


def log_ai_call(db_path: Path, user_id: str | None, purpose: str,
                provider: str = "", model: str = "",
                upload_id: str | None = None,
                creator_id: str | None = None,
                input_tokens: int = 0, output_tokens: int = 0,
                token_source: str = "none", method: str = "",
                status: str = "ok", note: str = "") -> int | None:
    """Log one AI call/attempt. Returns the row id, or None on error
    (logging must never break clip finding — callers swallow this)."""
    from datetime import datetime, timezone
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            cur = conn.execute(
                "INSERT INTO ai_usage (user_id, created_at, purpose,"
                " provider, model, upload_id, creator_id, input_tokens,"
                " output_tokens, total_tokens, token_source, method,"
                " status, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, datetime.now(timezone.utc).isoformat(), purpose,
                 provider, model, upload_id, creator_id,
                 int(input_tokens or 0), int(output_tokens or 0),
                 int(input_tokens or 0) + int(output_tokens or 0),
                 token_source, method, status, (note or "")[:500]))
            conn.commit()
            return cur.lastrowid
    except Exception:
        return None


def _rows(conn, sql: str, params: tuple = ()) -> list:
    return conn.execute(sql, params).fetchall()


def today_counts(db_path: Path, user_id: str) -> dict:
    """AI calls + tokens for this user today (UTC date)."""
    today = _date.today().isoformat()
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(total_tokens), 0) FROM ai_usage"
            " WHERE user_id=? AND substr(created_at, 1, 10)=?",
            (user_id, today)).fetchone()
    return {"calls_today": int(row[0]), "tokens_today": int(row[1])}


def usage_summary(db_path: Path, user_id: str, days: int = 30) -> dict:
    """Usage page data for one user (last `days` days, UTC).

    Totals count each logged row once. The "finding clips" summary
    rows carry 0 tokens, so they never inflate the token totals.
    """
    start = (_date.today() - _td(days=days - 1)).isoformat()
    with closing(sqlite3.connect(db_path)) as conn:
        totals = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(total_tokens), 0),"
            " COALESCE(SUM(input_tokens), 0),"
            " COALESCE(SUM(output_tokens), 0) FROM ai_usage"
            " WHERE user_id=? AND substr(created_at, 1, 10)>=?",
            (user_id, start)).fetchone()
        per_day_rows = _rows(
            conn,
            "SELECT substr(created_at, 1, 10) AS day, COUNT(*),"
            " COALESCE(SUM(total_tokens), 0) FROM ai_usage"
            " WHERE user_id=? AND substr(created_at, 1, 10)>=?"
            " GROUP BY day ORDER BY day ASC",
            (user_id, start))
        by_purpose = _rows(
            conn,
            "SELECT purpose, COUNT(*), COALESCE(SUM(total_tokens), 0)"
            " FROM ai_usage WHERE user_id=?"
            " AND substr(created_at, 1, 10)>=?"
            " GROUP BY purpose ORDER BY COUNT(*) DESC",
            (user_id, start))
        # The creators table may not exist on a DB that only ran
        # init_ai_tables (unit tests) — then by_creator is empty
        # instead of crashing the whole usage page.
        try:
            by_creator = _rows(
                conn,
                "SELECT COALESCE(c.name, '(no creator)'),"
                " COALESCE(c.id, ''), COUNT(*), COALESCE(SUM(total_tokens), 0)"
                " FROM ai_usage u LEFT JOIN creators c ON c.id = u.creator_id"
                " WHERE u.user_id=? AND substr(u.created_at, 1, 10)>=?"
                " GROUP BY COALESCE(c.id, '') ORDER BY COUNT(*) DESC",
                (user_id, start))
        except sqlite3.OperationalError:
            by_creator = []
        recent = _rows(
            conn,
            "SELECT id, created_at, purpose, provider, model, upload_id,"
            " creator_id, input_tokens, output_tokens, total_tokens,"
            " token_source, method, status, note FROM ai_usage"
            " WHERE user_id=? AND substr(created_at, 1, 10)>=?"
            " ORDER BY id DESC LIMIT 50",
            (user_id, start))
        upload_names = {}
        if recent:
            ids = {r[5] for r in recent if r[5]}
            if ids:
                try:
                    q = ",".join("?" for _ in ids)
                    for uid, name in conn.execute(
                            f"SELECT id, original_name FROM uploads WHERE id IN ({q})",
                            tuple(ids)):
                        upload_names[uid] = name
                except sqlite3.OperationalError:
                    upload_names = {}
        creator_names = {}
        for r in by_creator:
            creator_names[r[1]] = r[0]
    per_day_map = {r[0]: {"day": r[0], "calls": int(r[1]),
                          "tokens": int(r[2])} for r in per_day_rows}
    per_day = []
    for i in range(days - 1, -1, -1):
        day = (_date.today() - _td(days=i)).isoformat()
        per_day.append(per_day_map.get(
            day, {"day": day, "calls": 0, "tokens": 0}))
    return {
        "calls": int(totals[0]), "tokens": int(totals[1]),
        "input_tokens": int(totals[2]), "output_tokens": int(totals[3]),
        "per_day": per_day,
        "by_purpose": [{"purpose": r[0], "calls": int(r[1]),
                        "tokens": int(r[2])} for r in by_purpose],
        "by_creator": [{"name": r[0], "creator_id": r[1],
                        "calls": int(r[2]), "tokens": int(r[3])}
                       for r in by_creator],
        "recent": [{
            "id": r[0], "created_at": r[1], "purpose": r[2],
            "provider": r[3], "model": r[4], "upload_id": r[5],
            "upload_name": upload_names.get(r[5], r[5] or "—"),
            "creator_id": r[6],
            "creator_name": creator_names.get(r[6] or "", "(no creator)"),
            "input_tokens": int(r[7]), "output_tokens": int(r[8]),
            "total_tokens": int(r[9]), "token_source": r[10],
            "method": r[11], "status": r[12], "note": r[13],
        } for r in recent],
    }
