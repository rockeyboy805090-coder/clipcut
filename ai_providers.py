"""ClipCut — Phase 6: BYOK AI providers + per-user key storage.

Bring-your-own-key: the user picks one of 8 providers, a model, and
pastes their own API key. The key is stored per-user, encrypted at
rest (Fernet, key derived from a server-side secret — see
_server_secret() — never in plaintext, never logged, never shown
back in full; the UI only ever shows "Saved, ends in XXXX").

A fresh account has NO saved key — unlike the reference app, where a
brand-new account mysteriously showed a pre-saved key. With no key
the words judge falls back to the rule-based scorer and the ledger
says so (method rule_based, 0 tokens).

Per-provider status (honest — see README § Phase 6):
- gemini: WORKS (google-generativeai library, same call as Phase 3).
  Live-verified only when the user supplies a key (Phase 6 test ran
  without one, so every test run here is rule_based).
- openai / grok / groq / openrouter / custom: OpenAI-compatible
  /chat/completions adapter IMPLEMENTED, untested live (no key exists
  to test with). Grok is OpenAI-compatible (api.x.ai/v1).
- claude: messages API IMPLEMENTED, untested live (no key). Claude
  has no list-models API — "Find models" says so instead of faking.
- ollama: local /api/chat IMPLEMENTED, untested (no Ollama runs on
  this machine). Needs Ollama on the same machine (Phase 10 deploy:
  Render Free has no Ollama — this provider is local-only).

Token counts: providers that report usage give real counts
(token_source "reported"); otherwise words are estimated at
~1 token per 0.75 words (token_source "estimated", and the ledger
note says so). Rule-based fallback is always 0 tokens, source "none".
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- registry

PROVIDERS: dict[str, dict] = {
    "gemini": {
        "label": "Gemini",
        "kind": "gemini",
        "default_model": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com",
        "key_env": "GEMINI_API_KEY",
        "status": "works",
        "note": "Default. Free key from Google AI Studio.",
    },
    "openai": {
        "label": "OpenAI",
        "kind": "openai_compatible",
        "default_model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "status": "implemented-untested",
        "note": "Chat Completions API. Untested live — no key exists.",
    },
    "grok": {
        "label": "Grok",
        "kind": "openai_compatible",
        "default_model": "grok-3-mini",
        "base_url": "https://api.x.ai/v1",
        "key_env": "GROK_API_KEY",
        "status": "implemented-untested",
        "note": "OpenAI-compatible (api.x.ai/v1). Untested live — no key.",
    },
    "groq": {
        "label": "Groq",
        "kind": "openai_compatible",
        "default_model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "status": "implemented-untested",
        "note": "OpenAI-compatible. Untested live — no key.",
    },
    "openrouter": {
        "label": "OpenRouter",
        "kind": "openai_compatible",
        "default_model": "openai/gpt-4o-mini",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "status": "implemented-untested",
        "note": "OpenAI-compatible. Untested live — no key.",
    },
    "claude": {
        "label": "Claude",
        "kind": "claude",
        "default_model": "claude-sonnet-4-20250514",
        "base_url": "https://api.anthropic.com",
        "key_env": "ANTHROPIC_API_KEY",
        "status": "implemented-untested",
        "note": "Messages API. No list-models API exists — type the model id.",
    },
    "ollama": {
        "label": "Ollama (on this machine)",
        "kind": "ollama",
        "default_model": "llama3.1",
        "base_url": "http://localhost:11434",
        "key_env": "",
        "status": "implemented-untested",
        "note": "Local Ollama only — no key needed, no Ollama on Render Free.",
    },
    "custom": {
        "label": "Custom (OpenAI-compatible)",
        "kind": "openai_compatible",
        "default_model": "",
        "base_url": "",
        "key_env": "",
        "status": "implemented-untested",
        "note": "Your own OpenAI-compatible endpoint — base URL required.",
    },
}

# Legacy env-key default model (Phase 2/3 behaviour, unchanged):
# when NO per-user settings exist, the env GEMINI_API_KEY (if set) is
# used with this model, exactly as before Phase 6.
LEGACY_GEMINI_MODEL = "gemini-1.5-flash"


def valid_provider(provider: str) -> bool:
    return (provider or "").strip().lower() in PROVIDERS


def default_model(provider: str) -> str:
    return PROVIDERS.get((provider or "").lower(), {}).get("default_model", "")


# ---------------------------------------------------------------- key encryption

_SECRET_FILE = ROOT / ".clipcut_key_secret"
_FERNET_PREFIX = "fernet$"


def _server_secret() -> bytes:
    """Server-side secret for key encryption.

    1) CLIPCUT_KEY_SECRET env var (recommended on Render), or
    2) a random secret generated once and stored in
       .clipcut_key_secret (0600, gitignored, never committed).
    Anyone with the server filesystem can decrypt the stored keys —
    this protects database dumps/backups, not a full server breach.
    That limit is documented, not hidden.
    """
    env_secret = os.getenv("CLIPCUT_KEY_SECRET", "").strip()
    if env_secret:
        if len(env_secret) < 16:
            raise ValueError(
                "CLIPCUT_KEY_SECRET kam se kam 16 characters ka hona chahiye")
        return env_secret.encode("utf-8")
    if _SECRET_FILE.is_file():
        raw = _SECRET_FILE.read_text(encoding="utf-8").strip()
        if len(raw) >= 32:
            return raw.encode("utf-8")
    import secrets
    raw = secrets.token_hex(32)
    _SECRET_FILE.write_text(raw, encoding="utf-8")
    try:
        os.chmod(_SECRET_FILE, 0o600)
    except OSError:
        pass
    return raw.encode("utf-8")


def _fernet():
    try:
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ImportError as exc:
        raise RuntimeError(
            "cryptography package install nahi hai — "
            "pip install -r requirements.txt karo") from exc
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                     salt=b"clipcut-ai-key-v1", iterations=200_000)
    key = base64.urlsafe_b64encode(kdf.derive(_server_secret()))
    return Fernet(key)


def encrypt_key(plaintext: str) -> str:
    """Encrypt an API key for storage. Returns 'fernet$<token>'."""
    token = _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")
    return _FERNET_PREFIX + token


def decrypt_key(stored: str) -> str:
    """Decrypt a stored key. Raises ValueError on a bad value."""
    if not stored or not stored.startswith(_FERNET_PREFIX):
        raise ValueError("Stored key ka format galat hai")
    try:
        return _fernet().decrypt(
            stored[len(_FERNET_PREFIX):].encode("ascii")).decode("utf-8")
    except Exception as exc:
        raise ValueError("Stored key decrypt nahi hui") from exc


# ---------------------------------------------------------------- per-user settings

def get_ai_settings(db_path: Path, user_id: str) -> dict | None:
    """Saved AI settings for a user (never includes the plaintext key).

    Returns None when the user never saved anything (fresh account).
    """
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT provider, model, key_enc, key_last4, custom_base_url,"
            " updated_at FROM ai_settings WHERE user_id=?",
            (user_id,)).fetchone()
    if not row:
        return None
    return {"provider": row[0], "model": row[1],
            "has_key": bool(row[2]), "key_last4": row[3] or "",
            "custom_base_url": row[4] or "", "updated_at": row[5]}


def save_ai_settings(db_path: Path, user_id: str, provider: str,
                     model: str | None = None,
                     api_key: str | None = None,
                     custom_base_url: str | None = None) -> dict:
    """Save provider/model/key for a user. Raises ValueError on bad input.

    - provider must be one of the 8 (unknown -> ValueError).
    - model empty -> that provider's default (custom: must be typed).
    - api_key None/"" -> keep the previously saved key (if any).
    - custom provider needs a base URL starting with http(s).
    """
    provider = (provider or "").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider} — chuninda 8 me se ek do")
    model = (model or "").strip() or PROVIDERS[provider]["default_model"]
    if not model:
        raise ValueError(
            "Model naam do — is provider ka koi default model nahi hai")
    if len(model) > 120:
        raise ValueError("Model naam bahut lamba hai (max 120 characters)")
    base_url = (custom_base_url or "").strip()
    if provider == "custom":
        if not (base_url.startswith("http://")
                or base_url.startswith("https://")):
            raise ValueError(
                "Custom provider ke liye base URL chahiye (http(s) se shuru)")
        base_url = base_url.rstrip("/")
    else:
        base_url = PROVIDERS[provider].get("base_url") or ""
    key = (api_key or "").strip()
    if key and len(key) > 500:
        raise ValueError("API key bahut lambi hai (max 500 characters)")
    now = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(db_path)) as conn:
        old = conn.execute(
            "SELECT key_enc, key_last4 FROM ai_settings WHERE user_id=?",
            (user_id,)).fetchone()
        if key:
            key_enc, key_last4 = encrypt_key(key), key[-4:]
        elif old:
            key_enc, key_last4 = old[0], old[1]
        else:
            key_enc, key_last4 = "", ""
        conn.execute(
            "INSERT INTO ai_settings"
            " (user_id, provider, model, key_enc, key_last4,"
            "  custom_base_url, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(user_id) DO UPDATE SET provider=excluded.provider,"
            "  model=excluded.model, key_enc=excluded.key_enc,"
            "  key_last4=excluded.key_last4,"
            "  custom_base_url=excluded.custom_base_url,"
            "  updated_at=excluded.updated_at",
            (user_id, provider, model, key_enc, key_last4, base_url, now))
        conn.commit()
    return get_ai_settings(db_path, user_id)


def clear_ai_key(db_path: Path, user_id: str) -> dict | None:
    """Remove only the saved key (provider/model stay)."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "UPDATE ai_settings SET key_enc='', key_last4='',"
            " updated_at=? WHERE user_id=?",
            (datetime.now(timezone.utc).isoformat(), user_id))
        conn.commit()
    return get_ai_settings(db_path, user_id)


def _decrypt_saved_key(db_path: Path, user_id: str) -> str | None:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT key_enc FROM ai_settings WHERE user_id=?",
            (user_id,)).fetchone()
    if not row or not row[0]:
        return None
    try:
        return decrypt_key(row[0])
    except ValueError:
        return None


def get_effective_ai_config(db_path: Path,
                            user_id: str | None) -> dict:
    """The AI config the words judge should actually use.

    Precedence (documented):
    1) The user's saved settings (any provider) — if they picked a
       provider, the env key is NOT used even for Gemini; their
       explicit choice wins. No saved key -> api_key None (rule-based).
    2) No saved settings at all -> legacy env GEMINI_API_KEY (if set)
       with the legacy model (Phase 2/3 behaviour, unchanged).
    3) Nothing -> api_key None, provider gemini, rule-based fallback.
    Returns {provider, model, api_key, base_url, source, has_key}.
    """
    if user_id:
        saved = get_ai_settings(db_path, user_id)
        if saved is not None:
            provider = saved["provider"]
            info = PROVIDERS[provider]
            base_url = (saved["custom_base_url"]
                        if provider == "custom" else info.get("base_url") or "")
            return {"provider": provider, "model": saved["model"],
                    "api_key": _decrypt_saved_key(db_path, user_id),
                    "base_url": base_url, "source": "user",
                    "has_key": saved["has_key"]}
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return {"provider": "gemini", "model": LEGACY_GEMINI_MODEL,
                "api_key": env_key,
                "base_url": PROVIDERS["gemini"]["base_url"],
                "source": "env", "has_key": True}
    return {"provider": "gemini", "model": PROVIDERS["gemini"]["default_model"],
            "api_key": None, "base_url": PROVIDERS["gemini"]["base_url"],
            "source": "none", "has_key": False}


# ---------------------------------------------------------------- token helpers

def estimate_tokens(text: str) -> int:
    """Rough token estimate (~1 token per 0.75 words). Used ONLY when
    the provider reports no usage — the ledger marks it 'estimated'."""
    words = len(str(text or "").split())
    return max(1, math.ceil(words * 4 / 3))


def _zero_usage(note: str = "") -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "token_source": "none", "note": note}


# ---------------------------------------------------------------- words-judge prompt

def build_words_prompt(candidates: list[dict], video_type: str = "auto") -> str:
    """The WORDS judge prompt (same wording as Phase 3).

    The judge reads ONLY what was said — no loudness, no chat.
    """
    brief = "\n".join(
        f"[{i}] {c['start']:.0f}-{c['end']:.0f}s | {c['text'][:300]}"
        for i, c in enumerate(candidates))
    type_line = ""
    if (video_type or "auto").lower() != "auto":
        type_line = (f"Video type: {video_type}. Prefer moments that "
                     f"fit a {video_type} video. ")
    return (
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


def parse_words_judge_json(raw: str, candidates: list[dict]) -> dict | None:
    """Parse a words-judge JSON reply. None when unparsable/empty."""
    try:
        text = (raw or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
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


# ---------------------------------------------------------------- provider calls

def gemini_words_judge(api_key: str, model: str, candidates: list[dict],
                       video_type: str = "auto",
                       language: str = "en") -> tuple[dict | None, dict]:
    """Gemini words judge. Returns (judged, usage). None on any failure."""
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        return None, _zero_usage("google-generativeai library nahi hai")
    try:
        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(model)
        prompt = build_words_prompt(candidates, video_type)
        resp = gen_model.generate_content(prompt)
        raw = (getattr(resp, "text", "") or "").strip()
        judged = parse_words_judge_json(raw, candidates)
        if judged is None:
            return None, _zero_usage("Gemini se JSON parse nahi hua")
        usage_meta = getattr(resp, "usage_metadata", None)
        if usage_meta is not None:
            try:
                inp = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
                outp = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
            except Exception:
                inp = outp = 0
            if inp or outp:
                return judged, {"input_tokens": inp, "output_tokens": outp,
                                "total_tokens": inp + outp,
                                "token_source": "reported", "note": ""}
        est_in = estimate_tokens(prompt)
        est_out = estimate_tokens(raw)
        return judged, {"input_tokens": est_in, "output_tokens": est_out,
                        "total_tokens": est_in + est_out,
                        "token_source": "estimated",
                        "note": "Gemini ne usage nahi diya — estimate hai"}
    except Exception as exc:
        return None, _zero_usage(f"Gemini call fail: {exc}"[:200])


def _openai_compatible_words_judge(base_url: str, api_key: str, model: str,
                                   candidates: list[dict],
                                   video_type: str = "auto",
                                   extra_headers: dict | None = None
                                   ) -> tuple[dict | None, dict]:
    """Shared /chat/completions caller (OpenAI, Grok, Groq, OpenRouter,
    Custom). Returns (judged, usage). None on any failure."""
    import requests
    prompt = build_words_prompt(candidates, video_type)
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    try:
        resp = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            headers=headers, timeout=120, json={
                "model": model,
                "messages": [
                    {"role": "system",
                     "content": "You are the WORDS judge for short video clips."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            })
    except Exception as exc:
        return None, _zero_usage(f"API call fail: {exc}"[:200])
    if resp.status_code != 200:
        return None, _zero_usage(
            f"API error {resp.status_code}: "
            f"{(resp.text or '')[:150]}")
    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception:
        return None, _zero_usage("API reply ka format samajh nahi aaya")
    judged = parse_words_judge_json(content, candidates)
    if judged is None:
        return None, _zero_usage("API se JSON parse nahi hua")
    usage = data.get("usage") or {}
    try:
        inp = int(usage.get("prompt_tokens") or 0)
        outp = int(usage.get("completion_tokens") or 0)
    except Exception:
        inp = outp = 0
    if inp or outp:
        return judged, {"input_tokens": inp, "output_tokens": outp,
                        "total_tokens": inp + outp,
                        "token_source": "reported", "note": ""}
    est_in = estimate_tokens(prompt)
    est_out = estimate_tokens(content)
    return judged, {"input_tokens": est_in, "output_tokens": est_out,
                    "total_tokens": est_in + est_out,
                    "token_source": "estimated",
                    "note": "Provider ne usage nahi diya — estimate hai"}


def claude_words_judge(api_key: str, model: str, candidates: list[dict],
                       video_type: str = "auto") -> tuple[dict | None, dict]:
    """Claude Messages API. Returns (judged, usage). None on failure."""
    import requests
    prompt = build_words_prompt(candidates, video_type)
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key,
                     "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"},
            timeout=120, json={
                "model": model, "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            })
    except Exception as exc:
        return None, _zero_usage(f"Claude call fail: {exc}"[:200])
    if resp.status_code != 200:
        return None, _zero_usage(
            f"Claude error {resp.status_code}: {(resp.text or '')[:150]}")
    try:
        data = resp.json()
        content = "".join(
            b.get("text", "") for b in data.get("content", [])
            if isinstance(b, dict) and b.get("type") == "text")
    except Exception:
        return None, _zero_usage("Claude reply ka format samajh nahi aaya")
    judged = parse_words_judge_json(content, candidates)
    if judged is None:
        return None, _zero_usage("Claude se JSON parse nahi hua")
    usage = data.get("usage") or {}
    try:
        inp = int(usage.get("input_tokens") or 0)
        outp = int(usage.get("output_tokens") or 0)
    except Exception:
        inp = outp = 0
    if inp or outp:
        return judged, {"input_tokens": inp, "output_tokens": outp,
                        "total_tokens": inp + outp,
                        "token_source": "reported", "note": ""}
    est_in = estimate_tokens(prompt)
    est_out = estimate_tokens(content)
    return judged, {"input_tokens": est_in, "output_tokens": est_out,
                    "total_tokens": est_in + est_out,
                    "token_source": "estimated",
                    "note": "Claude ne usage nahi diya — estimate hai"}


def ollama_words_judge(base_url: str, model: str, candidates: list[dict],
                       video_type: str = "auto") -> tuple[dict | None, dict]:
    """Local Ollama /api/chat. No key needed. Returns (judged, usage)."""
    import requests
    prompt = build_words_prompt(candidates, video_type)
    try:
        resp = requests.post(
            base_url.rstrip("/") + "/api/chat",
            timeout=180, json={
                "model": model, "stream": False, "format": "json",
                "messages": [{"role": "user", "content": prompt}],
            })
    except Exception as exc:
        return None, _zero_usage(f"Ollama se connect nahi ho paya: {exc}"[:200])
    if resp.status_code != 200:
        return None, _zero_usage(
            f"Ollama error {resp.status_code}: {(resp.text or '')[:150]}")
    try:
        data = resp.json()
        content = (data.get("message") or {}).get("content", "")
    except Exception:
        return None, _zero_usage("Ollama reply ka format samajh nahi aaya")
    judged = parse_words_judge_json(content, candidates)
    if judged is None:
        return None, _zero_usage("Ollama se JSON parse nahi hua")
    try:
        inp = int(data.get("prompt_eval_count") or 0)
        outp = int(data.get("eval_count") or 0)
    except Exception:
        inp = outp = 0
    if inp or outp:
        return judged, {"input_tokens": inp, "output_tokens": outp,
                        "total_tokens": inp + outp,
                        "token_source": "reported", "note": ""}
    est_in = estimate_tokens(prompt)
    est_out = estimate_tokens(content)
    return judged, {"input_tokens": est_in, "output_tokens": est_out,
                    "total_tokens": est_in + est_out,
                    "token_source": "estimated",
                    "note": "Ollama ne usage nahi diya — estimate hai"}


def score_words(config: dict, candidates: list[dict],
                video_type: str = "auto",
                language: str = "en") -> tuple[dict | None, dict, str]:
    """Score candidates with the configured provider.

    Returns (judged, usage, method). judged None -> caller must use
    the rule-based words score; method is then "rule_based" and the
    usage note says why (no key / API error). Never raises.
    """
    provider = (config or {}).get("provider", "gemini")
    model = (config or {}).get("model") or default_model(provider)
    api_key = (config or {}).get("api_key") or ""
    base_url = (config or {}).get("base_url") or ""
    kind = PROVIDERS.get(provider, {}).get("kind", "gemini")
    if kind == "ollama":
        judged, usage = ollama_words_judge(base_url, model, candidates,
                                           video_type)
        return judged, usage, ("ollama" if judged is not None
                               else "rule_based")
    if not api_key:
        return None, _zero_usage(
            f"{PROVIDERS.get(provider, {}).get('label', provider)} "
            "ki key save nahi hai — rule-based scoring chalega"), "rule_based"
    try:
        if kind == "gemini":
            judged, usage = gemini_words_judge(api_key, model, candidates,
                                               video_type, language)
        elif kind == "claude":
            judged, usage = claude_words_judge(api_key, model, candidates,
                                               video_type)
        else:  # openai_compatible (openai/grok/groq/openrouter/custom)
            extra = {}
            if provider == "openrouter":
                extra = {"HTTP-Referer": "https://clipcut.local",
                         "X-Title": "ClipCut"}
            judged, usage = _openai_compatible_words_judge(
                base_url, api_key, model, candidates, video_type,
                extra_headers=extra)
    except Exception as exc:  # never break clip finding on an AI error
        return None, _zero_usage(f"Words judge fail: {exc}"[:200]), "rule_based"
    if judged is None:
        return None, usage, "rule_based"
    return judged, usage, provider


# ---------------------------------------------------------------- Find models

def list_models_for_provider(config: dict) -> tuple[list[str], str]:
    """List models for a provider (for the 'Find models' button).

    Returns (models, message). Never invents a list: when the provider
    has no list API (Claude) or no key is available, the message says
    so plainly and models is empty.
    """
    import requests
    provider = (config or {}).get("provider", "gemini")
    api_key = (config or {}).get("api_key") or ""
    base_url = (config or {}).get("base_url") or ""
    kind = PROVIDERS.get(provider, {}).get("kind", "gemini")
    if kind == "claude":
        return [], ("Claude ki koi list-models API nahi hai — "
                    "model id khud type karo (jaise claude-sonnet-4-20250514).")
    if kind == "ollama":
        try:
            resp = requests.get(base_url.rstrip("/") + "/api/tags", timeout=10)
        except Exception as exc:
            return [], (f"Ollama se connect nahi ho paya ({exc}) — "
                       "kya Ollama is machine par chal raha hai?")
        if resp.status_code != 200:
            return [], f"Ollama error {resp.status_code} — models list nahi mili"
        try:
            names = [m.get("name", "") for m in resp.json().get("models", [])]
            return [n for n in names if n], (
                f"{len(names)} model mile" if names
                else "Ollama par koi model download nahi hai — pehle `ollama pull` karo")
        except Exception:
            return [], "Ollama reply ka format samajh nahi aaya"
    if not api_key:
        return [], ("Pehle is provider ki key save karo — "
                    "bina key ke models list nahi mil sakti.")
    try:
        if kind == "gemini":
            resp = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": api_key, "pageSize": 50}, timeout=30)
            if resp.status_code != 200:
                return [], (f"Gemini error {resp.status_code} — "
                            "key galat ho sakti hai ya API band hai")
            names = []
            for m in resp.json().get("models", []):
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    names.append(str(m.get("name", "")).replace("models/", ""))
            return names, (f"{len(names)} model mile" if names
                           else "Koi generateContent model nahi mila")
        resp = requests.get(
            base_url.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        if resp.status_code != 200:
            return [], (f"API error {resp.status_code} — "
                        "key galat ho sakti hai ya endpoint galat hai")
        names = [d.get("id", "") for d in resp.json().get("data", [])]
        return [n for n in names if n], (
            f"{len(names)} model mile" if names
            else "Koi model nahi mila")
    except Exception as exc:
        return [], f"Models list nahi mil payi: {exc}"[:200]
