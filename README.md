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
- Still NOT done: Phase 3 dual-judge scoring (words + loudness/chat),
  Phase 5 clip editor, Phase 6 BYOK ledger, Phase 7 live recorder,
  Phase 8 posting queue, Phase 10 deploy.

## Not done yet (Phases 3–10, see CLONE_PLAN.md §4)

Phase 3 dual-judge scoring · Phase 5 clip editor
(trim/hook/re-make) · Phase 6 BYOK + token ledger · Phase 7 live
recorder · Phase 8 posting queue · Phase 9 upgrades · Phase 10 polish +
landing + legal pages + deploy.

## Blocked / needs the user

1. Final brand name (placeholder "ClipCut" for now).
2. Google OAuth credentials (Google Cloud Console) — until then the
   Google button explains it is not configured.
3. GitHub repo creation — the user creates it in his browser (agent
   cannot create a user repo); then push this folder and connect Render.
