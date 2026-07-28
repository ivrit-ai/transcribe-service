# Architecture

ivrit.ai is a Hebrew-focused audio transcription service built as a non-profit project. It supports multiple languages (Hebrew, Yiddish, English, French, Spanish, German, Chinese) with special optimization for Hebrew via custom Whisper models.

## Tech Stack

### Backend
- **Framework:** FastAPI (async Python)
- **Server:** Uvicorn (ASGI)
- **Speech Recognition:** Custom Whisper models via the `ivrit` package, with CTranslate2 (cloud) and GGML/whisper-cpp (local) backends
- **Speaker Diarization:** PyTorch-based pipeline for identifying speakers
- **Audio Processing:** ffmpeg for transcoding, julius for audio analysis
- **Storage:** Google Drive (cloud mode) or local filesystem (local mode)
- **Authentication:** Google OAuth 2.0 (cloud mode), session-based (local mode)
- **Compute:** RunPod serverless GPU endpoints (cloud mode) or local GPU
- **Analytics:** PostHog for event tracking
- **Models:** Hosted on Hugging Face, downloaded via huggingface-hub

### Frontend
- Vanilla JavaScript single-page application, server-rendered with Jinja2
- Responsive app shell (sticky header, bottom tab bar on phones) with a web app manifest
- CSS variables for dark/light theming
- Lucide icons (SVG)
- Libraries: html-docx-js (DOCX export), Chart.js (statistics), diff (text diffing)
- Full internationalization (i18n) support

## Features

### Transcription
- Multi-language audio transcription with language-specific Whisper models
- Automatic speaker diarization (who said what)
- Timestamped segments
- Files up to 20 hours, 300MB (3GB with custom RunPod)
- Chunked uploads (50MB per chunk)
- Real-time progress tracking with queue position and ETA

### Result Management
- Save, rename, delete transcriptions
- Inline editing of transcription text
- Per-transcript statistics with Chart.js visualizations
- Export formats: plain text, timestamped text, VTT, SRT, DOCX, JSON, speaker-separated text

### Completion Notifications (Web Push)
- Jobs outlive the tab that started them, so completion is pushed rather than polled
- Opt-in is offered when a job is submitted, never on page load
- Notification text is rendered by the service worker, in the language the subscription
  was registered with, and names the file
- Entirely optional: with the VAPID keys unset the feature is invisible and inert
- See [Web Push](#web-push) for the mechanics

### Sharing Files In
- Installed, the app is a share target for audio and video: a file shared from any other app
  arrives in the picker as if it had been chosen there
- Not available on iOS, which does not implement Web Share Target
- See [Web Share Target](#web-share-target) for the mechanics

### User Quota System
- Token-bucket rate limiting with configurable weekly minute credits (default 420 min/week)
- Daily credit replenishment
- Three job queues: short (<=20 min), long (>20 min), private (custom RunPod credentials)
- Users can bring their own RunPod API key to bypass shared quotas

### Deployment Modes
- **Cloud:** Google OAuth + Google Drive storage + RunPod GPU compute
- **Local/On-Premise:** Local filesystem storage + local GPU inference, no OAuth required
- **macOS Installer:** Native .app bundle for Apple Silicon (arm64), installed via shell script
- **PyInstaller Bundles:** Standalone executables for Windows/macOS

## Project Structure

```
app.py                    Main FastAPI application (routes, job queue, quota logic)
run.py                    Uvicorn launcher with CLI argument parsing
config.json               Language and model configuration

db.py                     SQLite/Postgres access layer (quota, stats, push subscriptions)
alembic/                  Database migrations

gdrive_auth.py            Google OAuth token management
gdrive_file_utils.py      Google Drive storage backend
local_file_utils.py       Local filesystem storage backend
file_utils.py             Abstract storage interface

templates/
  index.html              Main single-page app UI
  login.html              Google OAuth login page
  close_window.html       OAuth callback handler
  server-down.html        Error/maintenance page

static/
  i18n.js                 Internationalization string tables
  sw.js                   Service worker (push notifications, share target; served from /sw.js)
  theme.css               Colour tokens shared by index.html and login.html
  manifest.webmanifest    Web app manifest (PWA)
  favicon.png             App icon
  badge.png               Notification badge: the app glyph as an alpha-only
                          silhouette, because Android masks the badge to its
                          alpha channel

installers/osx/
  install-osx.sh          macOS installer script
  launch.sh               App launcher

build_bundle.py           PyInstaller bundling script
```

## API Surface

| Category | Key Endpoints |
|----------|--------------|
| Transcription | `POST /upload`, `POST /upload/precheck`, `POST /upload/youtube`, `GET /download/{job_id}` |
| Audio | `GET /appdata/audio/{id}`, `GET /appdata/audio/stream/{id}` |
| Data | `GET /appdata/toc`, `GET /appdata/results/{id}`, `POST /appdata/edits/{id}` |
| Management | `POST /appdata/rename`, `POST /appdata/delete`, `POST /appdata/donate_data` |
| Auth & Account | `GET /login`, `GET /authorize`, `GET /login/authorized`, `GET /quota`, `GET /balance` |
| Push | `GET /sw.js` (unauthenticated), `GET /push/config`, `POST /push/subscribe` |
| Share target | `POST /share-target` (unauthenticated fallback; the worker normally answers it) |
| System | `GET /languages`, `POST /client_heartbeat`, `GET /stats` |

## Job Queue

Three queue types with configurable parallelism:
- **Short** (<=20 min): max 1 parallel, 20 queued
- **Long** (>20 min): max 1 parallel, 20 queued
- **Private** (custom RunPod): max 1000 parallel, 5000 queued

Jobs go through: upload -> pre-transcoding (ffmpeg to OPUS) -> queue -> RunPod/local inference -> results.

## Configuration

### CLI Arguments (`app.py` / `run.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind to (run.py only) |
| `--port` | `4500` (`4600` in dev) | Server port (run.py only) |
| `--max-minutes-per-week` | `180` (app.py) / `420` (run.py) | Weekly quota credit grant per user |
| `--local` | off | Local mode: use local filesystem + local GPU, no OAuth |
| `--data-dir` | `local_data` | Storage directory (local mode) |
| `--models-dir` | (none) | Directory containing model files (local mode) |
| `--config` | `config.json` | Path to language/model configuration JSON |
| `--dev` | off | Development mode (port 4600, relaxed auth) |
| `--dev-user-email` | `local@example.com` | Override user email in dev mode |
| `--dev-https` | off | Enable HTTPS with self-signed certs in dev mode |
| `--dev-cert-folder` | (none) | Path to folder with cert.pem and key.pem |
| `--staging` | off | Staging mode |
| `--hiatus` | off | Emergency shutdown mode (returns maintenance page) |
| `--verbose` | off | Verbose logging |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_CLIENT_ID` | (empty) | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | (empty) | Google OAuth client secret |
| `GOOGLE_REDIRECT_URI` | `https://transcribe.ivrit.ai/login/authorized` | OAuth callback URL |
| `GOOGLE_ACCESS_TOKEN_EXPIRY_SECONDS` | `3600` | Google access token cache lifetime |
| `GOOGLE_DRIVE_FOLDER_NAME` | `transcribe.ivrit.ai` | App folder name in Google Drive |
| `GOOGLE_ANALYTICS_TAG` | (empty) | Google Analytics tag (optional) |
| `RUNPOD_API_KEY` | (required in cloud) | Default RunPod API key |
| `RUNPOD_ENDPOINT_ID` | (required in cloud) | Default RunPod endpoint ID |
| `RUNPOD_TEMPLATE_ID` | (none) | RunPod template ID for auto-creating user endpoints |
| `BASE_URL` | (required in cloud) | Public base URL of the service |
| `POSTHOG_API_KEY` | (none) | PostHog analytics key (optional, analytics disabled if unset) |
| `TS_HIATUS_MODE` | `0` | Set to `1` to enable hiatus mode via env |
| `TS_USER_EMAIL` | `local@example.com` | User email override (dev/local mode) |
| `VAPID_PUBLIC_KEY` | (none) | Web Push application server key, base64url. Push is off unless all three VAPID vars are set |
| `VAPID_PRIVATE_KEY` | (none) | Web Push signing key, base64url |
| `VAPID_SUBJECT` | (none) | Contact for push services: `mailto:...` or an https URL |
| `TOC_CACHE_MAX_SIZE` | `100` | Max entries in the TOC LRU cache |
| `TOC_VER` | `1.0` | TOC format version |

### `config.json` (Language & Model Configuration)

Defines available languages and their Whisper model variants:

```json
{
  "languages": {
    "<lang_code>": {
      "ct2_model": "<model_name>",       // CTranslate2 model for cloud/server mode
      "ggml_model": "<model_name>",       // GGML model for local mode
      "general_availability": true/false,  // Whether shown to all users
      "enabled": true/false                // Whether the language is active
    }
  },
  "quota_increase_url": "<url>"            // Link shown to users who exhaust quota
}
```

### Hardcoded Constants (`app.py`)

These are compile-time constants in `app.py` that require a code change to tune:

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_PARALLEL_SHORT_JOBS` | 1 | Concurrent short transcription jobs |
| `MAX_PARALLEL_LONG_JOBS` | 1 | Concurrent long transcription jobs |
| `MAX_PARALLEL_PRIVATE_JOBS` | 1000 | Concurrent private (custom RunPod) jobs |
| `MAX_PARALLEL_TRANSCODES` | 4 | Concurrent ffmpeg transcoding operations |
| `MAX_QUEUED_JOBS` | 20 | Max queued jobs per queue (short/long) |
| `MAX_QUEUED_PRIVATE_JOBS` | 5000 | Max queued private jobs |
| `SHORT_JOB_THRESHOLD` | 20 min (1200s) | Audio duration cutoff between short and long queues |
| `SPEEDUP_FACTOR` | 8 (macOS local) / 15 (cloud) | Real-time-to-transcription speed ratio for ETA |
| `TRANSCODING_SPEEDUP` | 100 | Transcoding speed ratio for ETA |
| `SUBMISSION_DELAY` | 15s | Extra seconds added to ETA estimates |
| `MAX_AUDIO_DURATION_IN_HOURS` | 20 | Maximum audio file length |
| `MAX_FILE_SIZE_REGULAR` | 300 MB | Upload limit for regular users |
| `MAX_FILE_SIZE_PRIVATE` | 3 GB | Upload limit for private (custom RunPod) users |
| `UPLOAD_CHUNK_SIZE` | 50 MB | Chunked upload size |
| `DRIVE_FILE_ID_CACHE_SIZE` | 1000 | LRU cache size for Google Drive file ID lookups |
| `QUEUE_SAMPLE_BUCKET_SECONDS` | 15 min (900s) | Width of each queue-depth history bucket |
| `QUEUE_SAMPLE_INTERVAL_SECONDS` | 60s | How often queue depths are sampled into the current bucket |
| `QUEUE_SAMPLE_RETENTION_DAYS` | 30 | How long queue-depth samples are kept before pruning |
| `JOB_EVENT_RETENTION_DAYS` | 90 | How long finished-job events are kept before pruning |
| `HISTORY_PRUNE_INTERVAL_SECONDS` | 3600s | How often expired history rows are deleted |
| `HISTORY_QUEUE_BUCKETS` | 96 | Queue-depth buckets served by `/stats` (24h at 15 min) |
| `HISTORY_HOURLY_BUCKETS` | 48 | Hourly job buckets served by `/stats` (48h) |
| `HISTORY_DAILY_BUCKETS` | 30 | Daily job buckets served by `/stats` (30d) |
| `HISTORY_LANGUAGE_DAYS` | 7 | Window for the per-language breakdown on `/stats` |

## External Services

- **Google OAuth 2.0** - User authentication (scopes: openid, userinfo, drive.file)
- **Google Drive** - Transcript and TOC storage in an app-specific folder
- **RunPod** - Serverless GPU compute via REST + GraphQL APIs
- **Hugging Face** - Model hosting and downloading
- **PostHog** - Analytics and event tracking

## Frontend Internals

### Static Asset Versioning
`theme.css` and `i18n.js` are linked with `?v={{ backend_version }}` — the random id
regenerated at each startup and published as a Jinja **global**
(`templates.env.globals["backend_version"]`, set in `lifespan`). It is a global rather
than a per-response context entry because every template linking either asset needs it,
and one that quietly omits it serves a stale copy — which is exactly how
`close_window.html` was missed once already.

This is not cosmetic. The templates are rendered per request and are always current,
but `/static` is a Starlette `StaticFiles` mount, which sends `ETag` and
`Last-Modified` and **no `Cache-Control`**. With no explicit freshness directive a
browser caches heuristically and reuses the stored copy *without revalidating*, so the
`ETag` never gets consulted. A fresh page then asks a stale `i18n.js` for keys it does
not have, and `I18N.t()` falls back to `|| key` — the user sees key names like
`fileSelected` until a hard refresh. Versioning the URL changes the cache key, so a
restart makes a stale copy unreachable.

Only these two are versioned: they are the assets the page reads as code. Images and
`manifest.webmanifest` are deliberately left alone — a stale copy of those is harmless,
and churning the manifest URL disturbs PWA installs. `server-down.html` links neither
and needs nothing; `close_window.html` links `i18n.js` and translates a live key
(`errorDrivePermissionsRequired`), so it is versioned too.

### App Shell
The page is `header.app-header` → `main.app-main` → `footer.app-footer`, all direct
children of a flex-column `body`. `.app-header` is sticky and holds the app bar
(title, quota pill, language/settings/theme buttons) plus the `nav.tabs` tab strip;
`.app-main` holds the four `.tab-content` panels.

- **Never** apply `transform`, `filter`, `backdrop-filter` or `contain` to `body`,
  `.app-header`, `.app-main` or `.app-footer`. Any of them turns the shell into a
  containing block for the `position: fixed` modals, toasts and overlays nested
  inside it, and they all mis-position.
- z-index: `.app-header` is 120 and forms a stacking context, so the mobile tab bar's
  100 and the language dropdown's 200 only order those two against each other — at
  page level both rank as 120. `#viewer-sticky-header` is 90 (below the app header);
  modals and overlays are 1000+ and are siblings of the shell, so they cover it.
- `--app-header-h` is written from a `ResizeObserver` on `.app-header` and is what
  `#viewer-sticky-header` offsets against; the header's height changes when the
  quota pill wraps and when the tab strip moves to the bottom on phones.
- Colour tokens live in `static/theme.css` (`:root` / `[data-theme="dark"]`), linked
  by both `index.html` and `login.html`: `--bg-color`, `--container-bg`,
  `--text-color`, `--border-color`, `--primary-color`, `--secondary-color`,
  `--input-bg`. JS reads several of these by name, so they are a public API —
  repoint them rather than renaming them. `index.html` adds the layout-only
  `--shell-max` (1100px) and `--app-header-h` inline.
- `login.html` is the PWA launch screen for anonymous visitors (`/` 303s to
  `/login`), so it links the same tokens and manifest and restores `data-theme`
  from localStorage before first paint. `server-down.html` is a standalone
  maintenance interstitial and deliberately keeps its own styling.
- Component primitives: `.panel` (card surface) on the transcribe/files/viewer tabs
  and `.icon-btn` (36px round icon button) in the app bar. The stats tab supplies its
  own dark `.nerd` surface and is `dir="ltr"`.
- Inline `style="display: ..."` is load-bearing (JS toggles it); every other
  declaration belongs in a class.
- One breakpoint, 640px: the tab strip becomes a fixed bottom bar, the quota pill
  wraps to its own row, and paddings tighten.

### PWA
`static/manifest.webmanifest` (standalone display, RTL/Hebrew, 512px icon) is linked
from `<head>` alongside `theme-color`, `apple-touch-icon` and `viewport-fit=cover`.
`updateThemeChrome()` rewrites the `theme-color` meta from `--container-bg` on every
theme change. There is a service worker (see [Web Push](#web-push) and
[Web Share Target](#web-share-target)); it caches no page or asset, so the app cannot be
used offline. Chrome installs it anyway.

`#install-btn` is a labelled pill in the header, hidden until the browser proves the app
can be installed: it is revealed when `beforeinstallprompt` fires (whose default banner is
suppressed in its favour), and clicking it calls that saved event. The event cannot be
prompted with twice, so the button hides the moment it is spent — the browser fires a fresh
one on a later visit if the user declined. That keeps the invariant the click handler
depends on: a visible button with no saved event means iOS and nothing else.

Safari fires no event and exposes no install API, so on iOS the button is revealed by
`'standalone' in navigator` — a WebKit-on-iOS-only property, and therefore a capability
probe rather than a user-agent parse — and opens `#install-modal` describing the
Share → "Add to Home Screen" gesture. The line saying installation is the only route to
notifications on iPhone is hidden unless `/push/config` reported push enabled, so a
keyless deployment does not promise what it cannot send. Nothing is offered at all once
`display-mode: standalone` or `navigator.standalone` says the app is already installed.

### Web Push

Transcription jobs keep running after the tab closes, so completion is delivered by push
rather than polling.

**Service worker.** `static/sw.js` is served from the root route `GET /sw.js` so its scope
covers the whole app; the `/static` mount would scope it to `/static/` and it could never
receive the app's pushes. That route is deliberately **unauthenticated** —
`require_google_login` answers with a 303 to `/login`, and a redirected script response
fails worker registration. The file contains no secrets, and nothing user-specific may ever
be templated into it.

It handles `install` (`skipWaiting`), `activate` (`clients.claim`), `push`,
`notificationclick` and — solely for the share target — `fetch`.

The worker **must never cache a page or an asset**. The app busts its own caches with a
backend version identifier and a caching worker would fight it, so the `fetch` handler is
held to a narrow rule, stated at the top of `sw.js`: respond only to `POST /share-target`,
return without calling `respondWith()` for anything else, and touch no cache other than the
share inbox. Never intercept a `GET`, never call `caches.match`, `cache.add` or
`cache.addAll`.

Registration lives in `serviceWorkerReady`, outside the push module, because the share
target needs the worker in deployments that have no VAPID keys.

**Where the text lives.** All notification copy is a table inside `sw.js`, not in `app.py`
and not in `static/i18n.js` — the latter assigns to `window`, which does not exist in a
worker. The payload therefore carries a `lang`, which is why the subscription row stores
one: the server builds the payload at job completion, when no browser is around to ask.

**`push_subscriptions` table** (Alembic revision `0003`):

| Column | Notes |
|--------|-------|
| `endpoint` | Primary key, so a device re-subscribing upserts instead of duplicating |
| `user_email` | Indexed; scopes both sending and pruning |
| `p256dh`, `auth` | Encryption keys from `PushSubscription.toJSON()` |
| `lang` | UI language at subscribe time; refreshed on load, on language change, and on submit |
| `created_at` | Unix seconds |

**Flow.** `initPushNotifications()` awaits `serviceWorkerReady` and sets `pushReady` only
once `/push/config` reports `enabled`, the browser exposes the API, and a registration
exists. Everything downstream gates on that one flag: a worker now exists wherever service
workers do, so a registration no longer implies push works — an iOS Safari tab has one and
no `Notification` at all. It reads `/push/config` *before* testing `pushSupported()`,
deliberately: that same iOS tab is exactly where the install modal needs to know whether
this deployment sends notifications. `maybeAskForPushPermission()` runs when a job is submitted — never on page load —
and shows the opt-in modal unless the user already answered. Accepting subscribes and
`POST`s to `/push/subscribe`; declining with "don't ask again" writes `never` to the
`push_prompt` localStorage key. Once opted in, `refreshPushSubscription()` re-`POST`s the
current subscription on load, on language change and on submit, which is what actually
keeps rows healthy: it keeps `lang` current,
and if the browser dropped the subscription while the app was closed it re-subscribes
silently (permission is already granted) so the user does not go quietly un-notified. It
also compares the subscription's `applicationServerKey` against the current public key and
tears the subscription down when they differ, which is the only way to recover from a key
rotation — the browser otherwise hands back the same, permanently rejected subscription
forever. The orphaned row is pruned when its endpoint next answers 403 or 410.

There is deliberately no `pushsubscriptionchange` handler: a worker cannot read
localStorage, so it could only guess at `lang`, and support for the event is uneven. The
load-time refresh is the guarantee instead.

`notify_job_finished()` fires from every terminal path of `transcribe_job` — including the
results-upload and TOC-upload failures, where the job is dead from the user's point of view
— and never raises, because a failed notification must not change a job's outcome. Sends
are gathered rather than serialised: the job's queue slot is not released until this
returns, and the queues allow very little concurrency. A 403, 404 or 410 from the push
service deletes the row; anything else is logged and the row kept.

Clicking a notification reuses an open window via `postMessage({type: 'open-results'})`, or
opens `/?results=<id>`; both land in `openResultsById()`.

**Caveats.**
- Rotating the VAPID key pair invalidates every stored subscription: the push service
  answers 403, not 410. Recovery is automatic but not instant — each user is re-subscribed
  by the load-time refresh on their next visit, so jobs finishing before that visit go
  un-notified.
- iOS delivers Web Push only to home-screen-installed PWAs. Feature detection handles this:
  `Notification`/`PushManager` are undefined in a plain Safari tab, so nothing is offered.
- Payloads are encrypted end to end, but the endpoint and the timing of each push are
  visible to Google/Apple, and the filename appears on the device's lock screen.

### Web Share Target

Installed, the app appears in the OS share sheet for audio and video, and a file shared
into it lands exactly where the file picker lands.

`share_target` in the manifest declares `POST` / `multipart/form-data` with a `media` field
accepting `audio/*` and `video/*`. Its `action` is the absolute `/share-target`: the
manifest is served from `/static/`, so a relative action would resolve to
`/static/share-target` — in scope, but swallowed by the static mount rather than routed.

The OS POSTs the files to that URL and the worker's `fetch` handler answers. It cannot hand
them to the page directly — the page may not be running — so it writes each file into the
`ivrit-share-inbox` cache under `/__shared__/<timestamp>-<index>`, keyed by arrival so a
batch keeps its order and a second share does not overwrite the first. A `Response` carries
no filename, so it travels in an `X-Share-Filename` header, percent-encoded because header
values are latin-1 and these filenames are usually Hebrew. The worker then redirects to a
bare `/`.

The page calls `drainSharedFiles()` on every load, not off a redirect parameter: an expired
session sends that redirect through `/login` and back, and a parameter would not survive the
round trip. It empties the cache and passes the files to `handleFiles()`, which owns the
size limits, the batch cap and the toasts — a shared file is treated exactly like a picked
one.

`POST /share-target` also exists as a FastAPI route. It is only reached when no worker
intercepted the POST, in which case the file cannot be recovered and the best available
outcome is landing the user in the app rather than on a 405. It is unauthenticated for the
same reason `/sw.js` is.

The Cache API is used rather than the `ivrit-recordings` IndexedDB store: it has the same
API in worker and window, no version or upgrade path to coordinate, and no `onblocked`
state — `openRecordingDb()` *rejects* on `onblocked`, so sharing a file into a second tab
would break recording persistence.

**Caveat.** WebKit does not implement Web Share Target, so this does nothing on iOS. Push
and installation still work there.

### Key DOM Elements
- `drop-area`, `file-input` — file drag/drop and picker
- `youtube-url-input`, `youtube-paste-btn` — YouTube URL input and paste button
- `transcribe-btn`, `language-select` — transcription controls
- `progress-bar`, `progress-status`, `progress-container` — upload/transcoding progress
- `file-preview`, `file-name` — selected file display
- `transcribe-setup` — wraps everything in the transcribe tab that a live capture replaces;
  new controls in that tab belong **inside** it, or they stay clickable mid-recording
- `recording-interface`, `recorded-audio-preview`, `recovered-recording` — recorder UI, the
  finished take, and the banner offering a recording restored from IndexedDB
- `push-optin-modal`, `push-dont-ask-checkbox` — notification opt-in prompt
- `install-btn`, `install-modal` — the install offer and its iOS-only instructions

### Key JS State Variables
- `selectedFiles` — array of File objects pending upload
- `pendingYoutubeUrl` — URL awaiting rights confirmation
- `currentJobId`, `activeTranscription` — active job tracking
- `transcriptionSegments` — current transcription result segments
- `uploadPhase` — `"idle"` | `"upload"` | `"transcoding"` | `"done"`

### Key JS Functions
- `handleFiles()` — validates and queues files for upload
- `uploadBatch()` — orchestrates sequential file uploads
- `sendStreamingUpload()` — XHR POST to `/upload` with NDJSON streaming
- `uploadYoutubeUrl(url)` — fetch POST to `/upload/youtube` with NDJSON streaming
- `showProgressUI()` / `hideProgressUI()` — toggle progress bar visibility
- `setProgressStatusText(key, vars)` — update progress status with i18n
- `resetUploadState()` — clear all upload-related UI state; the batch-upload path calls it
  only when nothing is left selected
- `resetUploadProgressOnly()` — hide progress UI but keep `selectedFiles`, so a rejected
  batch is one click away from a retry
- `beginRecording()` / `finishRecording()` — shared recorder entry/exit for all capture modes
- `restoreUnsentRecording()` — startup recovery + retention sweep of the IndexedDB store
- `maybeAskForPushPermission()` — called on job submit; runs alongside the upload, never gates it
- `refreshPushSubscription()` — idempotent re-`POST` of the current subscription on load and
  on language change
- `openResultsById(id)` — opens a transcript from a clicked notification, either route
- `serviceWorkerReady` — the single registration of `/sw.js`, awaited by push; the share
  target needs a worker even where push is unconfigured
- `drainSharedFiles()` — empties the share inbox into `handleFiles()` on every load
- `showError()`, `showToast()` — user notifications
- `switchTab(tabName)` — navigate between tabs (wrapped later in the file to lazy-load
  the stats tab, so always call it by name rather than capturing a reference)
- `updateThemeChrome(theme)` — swaps the sun/moon icon and the `theme-color` meta
- `translateServerError(err)` — convert server error objects to i18n strings
- `checkBalance()` — refresh quota display

### Browser Recording Store (IndexedDB)

In-browser recordings survive tab crashes and rejected jobs; their lifetime is deliberately
independent of the transcription job's fate.

- Database `ivrit-recordings`, two stores: `recordings` (keyPath `id`, holds
  `{filename, mimeType, startedAt, lastFlushAt, status}`) and `chunks`
  (compound keyPath `[recordingId, seq]`).
- `chunks` is append-only (`recordings` holds one mutable row per session). Flushes are
  serialized on a promise chain; every `RECORDING_FLUSH_MS` (5s) the pending `MediaRecorder`
  blobs are concatenated into one `chunks` record and `audioChunks` is **cleared**, so the
  heap stays flat and the normal stop path reassembles via `assembleRecordingParts()` — the
  same read the crash-recovery path makes through `assembleRecording()`. `audioChunks`
  therefore holds only the tail a failed flush left behind, and stored parts + that tail
  are the whole recording.
- All three capture modes (mic, screen, mic+screen) share `beginRecording()` /
  `finishRecording()`; they differ only in how they acquire the `MediaStream`.
  Everything a capture replaces lives in `#transcribe-setup`, hidden as one unit, so no
  stale control can act on a live recording.
- **Retention:** an entry is deleted when the upload yields a `transcoding_complete` event,
  when a retake supersedes it, when the user removes the file or discards the banner, or
  when it is swept after `RECORDING_RETENTION_MS` (7 days) on page load. Note
  `sendStreamingUpload()` also resolves
  on a *truncated* stream, which deliberately does **not** delete. Every rejection path —
  including a late transcription-queue rejection, which arrives as a stream `error` — leaves
  the recording intact.
- **Recovery:** `restoreUnsentRecording()` runs at startup and offers the newest entry in the
  `#recovered-recording` banner, skipping `status: 'recording'` entries whose `lastFlushAt` is
  newer than `RECORDING_STALE_MS` (15s). That threshold is what keeps one tab from hijacking
  another tab's live recording; a `'complete'` entry has no recorder behind it and is always
  eligible.
- **Degradation:** if IndexedDB is unavailable (private browsing) or a flush hits
  `QuotaExceededError`, `recordingPersistenceFailed` is set, the recorder falls back to
  accumulating in memory, and the user is toasted once. A recording is never aborted because
  persistence failed.

### Upload Pipeline
- **Client:** precheck → XHR POST `/upload` → NDJSON streaming events
- **Server:** `validate_upload_request_metadata()` → create temp file → queue to `transcoding_queue` → `StreamingResponse`
- **Background:** `submit_next_transcoding_task()` → `handle_transcoding()` → `transcode_to_opus()` → `queue_job()` → `transcribe_job()`

### YouTube Upload Pipeline
- **Client:** validate URL → show rights modal → fetch POST `/upload/youtube` → NDJSON streaming events
- **Server:** validate URL + rights → `download_youtube_audio()` (yt-dlp in executor) → queue to `transcoding_queue` → `StreamingResponse`
- **Background:** same transcoding pipeline as file upload

### Streaming Upload Events (NDJSON)
- `transcoding_waiting` — job queued for transcoding
- `transcoding_started` — transcoding in progress
- `transcoding_progress` — transcoding percent update
- `transcoding_complete` — terminal success event
- `queue_position` — position in transcription queue
- `eta` — estimated time to transcription start
- `youtube_download_started` — YouTube download began
- `youtube_download_progress` — YouTube download percent
- `youtube_download_complete` — YouTube download finished
- `error` — terminal error event

Events are pushed via `upload_event_streams` dict (job_id → asyncio.Queue), emitted with `emit_upload_event()`, and streamed to client via `upload_event_generator()`.

### Modal/Dialog Patterns
- CSS: `.modal` (hidden) + `.modal.show` (visible), fixed position, z-index 1000, dark backdrop
- HTML: `.modal > .modal-content > .modal-header + .form-group + .modal-buttons`
- Existing modals: `settings-modal`, `speaker-rename-modal`, `donate-data-modal`, `youtube-rights-modal`
- Checkbox-gated submit: checkbox change toggles submit button `disabled` state

### i18n Pattern
- HTML: `data-i18n` attribute on elements, `data-i18n-title` for title/aria-label
- JS: `window.I18N.t(key, vars)` for dynamic strings
- String tables in `static/i18n.js` with `he`, `yi`, `en` objects
