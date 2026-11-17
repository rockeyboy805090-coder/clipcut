# ClipCut — separate clone repo (working name, NOT final brand)

**Working name "ClipCut" is a placeholder.** Final brand name is still the
user's pick — the reference site's name/logo is deliberately not used
anywhere in this repo, per the original-IP line. Rename in
`app.py` (FastAPI title), `templates/index.html`, and this README when the
final name is chosen.

Source plan: `../CLONE_PLAN.md` (teardown of creativecut.pro, 16 Nov 2026 —
landing + in-app, Google signup). Follow it exactly for scope.

## What this is

Phase 0 scaffold + Phase 1 (16 Nov 2026) of the clone, built as a
**separate repo** from Viral Clip Studio. The original
`~/workspace/viral-clip-app/` was NOT modified — this folder is a copy.

## What's copied vs what's new

Copied intact from Viral Clip Studio (working code, same behaviour):
- `app.py` core routes — upload + ffprobe validation, style extract/save,
  transcribe, find-clips, render-clip, approve/reject, render-all (with
  approval guard), batch, projects + delete, per-clip + project ZIPs,
  caption JSON, share guide
- `transcriber.py` — faster-whisper (CTranslate2 int8, **NO torch**)
- `clip_finder.py` — rule-based scorer + Gemini-optional rescore
- `editor.py` — 9:16 reframe (face centre-crop / blur-fill), zoom punch,
  colour grade, ASS word-by-word captions, yuv420p export
- `youtube_job.py` — YouTube link → clips job (yt-dlp + curl-cffi
  impersonation + Node JS runtime + Invidious fallback)
- `workflow.py`, `style_extractor.py`, `utils.py`, `styles/`,
  `templates/index.html` (classic studio UI, rebranded to ClipCut)
- `requirements.txt`, `render.yaml` (service renamed to `clipcut`)

New in Phase 1 (`auth.py` + Phase 1 routes in `app.py` + `templates/login.html`, `templates/signup.html`):
- Email/password signup/login — scrypt password hashing, httpOnly
  session cookie (`clipcut_session`, 30 days), sign out everywhere
- Google OAuth button — **placeholder only**: works only when
  `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` / `GOOGLE_REDIRECT_URI`
  are set in the environment. No credentials invented or stored.
- Dashboard `/app` — 4 stat tiles, Activity last-14-days, AI panel,
  Next steps (3), Creators list, Post queue empty state; sidebar with
  Home / All videos / Creators / New creator / Storage meter / AI usage /
  Notifications / Sign out. Unauthenticated `/app` redirects to `/login`.
- Creators CRUD — add (channel link + name, platform auto-detected:
  youtube/twitch/kick), list, per-creator workspace filter on All videos,
  delete (videos filed under it are kept, unfiled)
- Storage quota — 100 MB per user, meter in sidebar; oldest uploads
  auto-deleted (file + DB row + transcript/clips sidecars) when over
  quota, enforced after each upload
- Profile / Security pages — rename, set password, sign out everywhere,
  delete account (deletes user + sessions + creators + owned uploads)
- AI usage + Connected apps pages — honest zero/placeholder states;
  real ledger is Phase 6, posting is Phase 8

## Free-tier limits (honest)

- Render Free: 512 MB RAM, sleeps after 15 min idle, ~1 min cold start.
- Transcription: faster-whisper only. **Do NOT re-add openai-whisper /
  torch** — it crashed the 512 MB instance twice in VCS.
- Renders: 1080x1920, H.264, yuv420p (plays on budget Android phones).
- SQLite (`clipcut.sqlite3`) is ephemeral on Render Free — a restart
  wipes uploads/jobs. Neon Postgres migration is a Phase 10 decision.
- YouTube download from a datacenter IP often fails with HTTP 403
  (bot-check). Both the reference site and VCS hit this on 16 Nov 2026.
  File upload (V1) always works; yt-dlp client fallback + Invidious
  fallback are already wired in `youtube_job.py`.

## Run locally

```bash
cp .env.example .env   # fill GEMINI_API_KEY only if you use rescoring
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/signup — create an account, then /app
```

## Phase 2 (16 Nov 2026) — Options panel (TEST-PASS, local real runs)

`options.py` owns every option + validation; report: `../PHASE2_REPORT.md`.
- Dashboard `/app` clip box now has the real Options panel (teardown §1.3):
  clip count stepper (default 2), Shortest 15s / Longest 60s, caption
  styles Bounce (default) / Hype / Clean, layouts Blurred fill
  (default) / Zoomed in / Fill screen, quality 720p Fast (default) /
  1080p Sharper, From/To minutes, video type (7). Classic studio
  (`/`) find-clips + render forms have the same fields.
- Caption presets are real ASS profiles in `styles/captions/*.json`
  (Bounce = yellow #FFD60A highlight, Hype = pink #FF2E63 + larger,
  Clean = plain white, no highlight, no all-caps, no zoom punch).
- Layouts are wired to the editor: Blurred fill forces blur-fill,
  Fill screen forces full-frame centre-crop, Zoomed in = centre-crop
  + 12% extra zoom. Quality is a real switch: 720p = 720x1280,
  1080p = 1080x1920, yuv420p both, ASS PlayRes + caption size scale
  with the output height.
- Options thread end-to-end: dashboard → `/youtube-job` (stored per
  job in `youtube_jobs.options_json`) → worker → `find_clips` (count,
  min/max length, From/To window, video type into the Gemini rescore
  prompt) → `render_clip` (caption/layout/quality). Options are
  recorded on the clips JSON, so `/render-clip` with no options
  reuses what `/find-clips` stored. Video type does NOT change
  rule-based scoring by itself — it is recorded on every clip and
  only steers the Gemini rescore prompt when a key is set.
- Validation rejects bad values with Hinglish messages (400, no
  crash, no silent defaulting except documented defaults): count
  1–8, shortest ≥5s, longest ≤120s, shortest < longest, From < To,
  From/To inside the video duration when known.
- Phase 2 also fixed a Phase 1 bug found while threading: YouTube
  `_register_download` inserted 6 values into the 8-column uploads
  table, so every YouTube job died at registration. It now inserts
  all 8 and tags user_id/creator_id.
## Phase 3 (16 Nov 2026) — Dual-judge scoring (TEST-PASS, local real runs)

Two judges must AGREE before a clip is kept (CreativeCut rule):
- **Words judge** (`clip_finder._gemini_words_judge`): Gemini reads ONLY
  the transcript text when `GEMINI_API_KEY` is set; otherwise the
  rule-based scorer is the words judge and the output says
  `scoring_method: rule_based` (no key is set in this repo).
- **Reaction judge** (`dual_judge.py`, NEW): loudness from one ffmpeg
  `astats` pass (per-second RMS/Peak bins; no torch, no extra model)
  + chat activity ONLY when the caller passes it in — no chat is
  fetched here (Phase 7). No chat data = loudness-only, and the
  reason says so. No video file = neutral 50, reason says so.
- **Combine** (defaults, tunable via Options `words_threshold` 50 /
  `reaction_threshold` 40): keep only when words >= 50 AND reaction
  >= 40; combined = round(0.6*words + 0.4*reaction); the old `score`
  field IS the combined score. Every clip records words_score +
  words_reason, reaction_score + reaction_reason + breakdown,
  combined_score, judges_agree. Dropped candidates are listed in
  the clips JSON (`dropped`, with which judge failed).
- **Sentence splitting** (Phase 2 "count vs content" fix): long
  Whisper segments are split into sentence pieces before windowing.
  Same 128s test video: 5 segments -> 26 pieces, 4 candidates ->
  24 candidates. Same 64.5s video: 3 segments -> 18 pieces,
  2 candidates -> 12.
- Real-run proof (16 Nov 2026): 128s English video, no Gemini key —
  24 candidates, 13 after gate, 3 clips (words 65/56/51, reaction
  68/70/70, combined 66/62/59); words_threshold=100 -> 0 clips;
  invalid threshold -> 400. Quiet-contrast video (volume 0.2 for
  30-60s): loud windows 84-97, quiet window 53 — reaction is
  relative to the video's own median, so a half-quiet video's quiet
  half still passes; the gate-both-ways proof is the controlled
  fabricated-curve test (loud 85, quiet 14; high-words quiet
  candidates dropped with "reaction judge fail").
- Render still works end-to-end with the new scoring: clip_01
  720p/Clean, 720x1280 h264 yuv420p 30fps, 42.67s, render `score`
  field = 66 (combined).
- Still NOT done at the time Phase 3 was written: Phase 5 clip editor,
  Phase 6 BYOK ledger, Phase 7 live recorder, Phase 8 posting queue,
  Phase 10 deploy. (Phases 5 and 6 are done now — see below.)

## Phase 6 (16 Nov 2026) — BYOK + token ledger (TEST-PASS, local real runs)

`ai_providers.py` + `ai_ledger.py` (NEW), report: `../PHASE6_REPORT.md`.
- **Settings/AI page** (`/app/settings/ai`): 8 providers — Gemini
  (default, gemini-2.0-flash), OpenAI, Grok, Groq, OpenRouter, Claude,
  Ollama (local), Custom (OpenAI-compatible, base URL required).
  Model field + "Find models" (real provider list via API; Claude has
  no list API and says so; no key -> clear message, never a fake list).
- **Key storage:** per-user, Fernet-encrypted at rest (PBKDF2 key
  from `CLIPCUT_KEY_SECRET` env, else a generated `.clipcut_key_secret`
  file, 0600, gitignored). UI only ever shows "Saved, ends in XXXX";
  a fresh account shows NO saved key (unlike the reference app).
  Empty key field keeps the old key; "Remove saved key" clears it.
- **Words judge drives the chosen provider:** Gemini via
  google-generativeai (usage from `usage_metadata` when reported);
  OpenAI/Grok/Groq/OpenRouter/Custom via `/chat/completions`; Claude
  via Messages API; Ollama via local `/api/chat`. No key / any API
  failure -> rule-based fallback, ledger says so (method rule_based).
  Precedence: user's saved settings win over the env key; with no
  saved settings at all, the legacy env `GEMINI_API_KEY` is used
  (Phase 2/3 behaviour, unchanged).
- **Token ledger** (`ai_usage` table + `/app/usage`): every AI attempt
  logged — when, purpose (finding clips / words judge / reaction
  judge / hand clip), creator, video, provider, model, input/output/
  total tokens, token source (reported / estimated / none), method,
  status. Usage page: 30-day cards, tokens-per-day chart, "What it
  was used for", "By creator", recent 50 calls. Sidebar "AI … today"
  and the dashboard AI card show real ledger counts. "finding clips"
  rows carry 0 tokens by design (their tokens live on the "words
  judge" row — totals are never double-counted).
- **Auth hardening (Phase 5 flag):** `/render-clip` and `/remake-clip`
  now require login + ownership (401 unauth, 403 other user's video;
  ownerless legacy uploads may be rendered by any logged-in user).
  Note: the anonymous classic studio (`/`) can no longer render
  without login — sign in first, or use the `/app` dashboard.
- **Per-provider status:** Gemini works (live-verification still needs
  the user's own key — no key existed during Phase 6 testing, so all
  test runs are rule_based). OpenAI/Grok/Groq/OpenRouter/Custom:
  adapter implemented, untested live. Claude: implemented, untested,
  no list-models API. Ollama: implemented, untested (no Ollama here;
  local-only — none on Render Free).

## Phase 7 (16 Nov 2026) — Live recorder (TEST-PASS, local real runs)

Full evidence: `../PHASE7_REPORT.md`. Browser capture is the route that
works free: `/live/record` (camera/mic or screen, MediaRecorder,
~60s segments made by stop/start so each chunk is a complete playable
file) → `POST /live/{id}/chunk` → one worker thread, one chunk at a
time (transcribe → dual-judge find-clips → render) so captioned clips
appear BEFORE the session ends → Stop concats the chunks into one
session video in All videos/projects. Dashboard "Recording now" tile
is real now; creators table has a Live? column + Check button.
- **Live check (yt-dlp only, no keys):** `POST /live/check` with a
  creator_id or channel_link. YouTube `<channel>/live`, Twitch/Kick
  channel URL. `live=null` means "pata nahi chala", never a guess.
  Verified 16 Nov 2026: YouTube `@LofiGirl` → live=true (video id
  resolved); Twitch `shroud` → offline (explicit message); YouTube
  `@global.baatien` → unknown (tab API 404 — on a datacenter IP that
  404 is NOT proof of offline); Kick `xqc` → unknown (403).
- **URL capture:** `POST /live/start-from-url` — yt-dlp into
  kill-tolerant .mkv, then the full pipeline when the stream ends or
  Stop kills it. Verified: localhost mp4 → stopped, 2 clips
  (words 65/52, reaction 73/76), both renders 720x1280 yuv420p.
  YouTube is blocked from datacenter IPs (403 / bot-check) and yt-dlp
  hangs in extraction for many minutes on a blocked URL — Stop ends it
  promptly (verified twice). No cookies are used.
- **Safety (Render Free 512 MB):** single worker, sequential renders,
  Whisper freed per chunk, no in-memory video, chunks on disk, 200 MB
  per chunk, 100 MB user quota, startup sweep marks restart-orphaned
  chunks. Observed RSS ~180 MB during a live session. Render sleeps
  after 15 min idle — a live session WILL be cut by a restart/sleep
  there; long recordings belong on a machine that stays awake.
- **Bugs fixed en route:** `/app/videos` 500 for ANY project (pre-existing:
  `p["id"]` vs `get_project_status`'s `"upload_id"`); stop-race that left
  URL sessions stuck in `stopping`; `POST /app/creators` 500 caused by
  the Phase 7 migration (explicit column list now); YouTube 404 wrongly
  read as "offline".
- **Limits:** no live chat (Reaction judge = loudness only); re-make
  needs the stopped session; record page JS code-reviewed only, not
  tested in a real browser yet; Gemini words judge still unverified
  live (no key ever set — all test clips rule_based).

## Not done yet (Phases 8–10, see CLONE_PLAN.md §4)

Phase 8 posting queue · Phase 9 upgrades · Phase 10 polish + landing +
legal pages + deploy.

## Blocked / needs the user

1. Final brand name (placeholder "ClipCut" for now).
2. Google OAuth credentials (Google Cloud Console) — until then the
   Google button explains it is not configured.
3. GitHub repo creation — the user creates it in his browser (agent
   cannot create a user repo); then push this folder and connect Render.
