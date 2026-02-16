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
  favicon.png             App icon

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

## External Services

- **Google OAuth 2.0** - User authentication (scopes: openid, userinfo, drive.file)
- **Google Drive** - Transcript and TOC storage in an app-specific folder
- **RunPod** - Serverless GPU compute via REST + GraphQL APIs
- **Hugging Face** - Model hosting and downloading
- **PostHog** - Analytics and event tracking

## Frontend Internals

### Key DOM Elements
- `drop-area`, `file-input` — file drag/drop and picker
- `youtube-url-input`, `youtube-paste-btn` — YouTube URL input and paste button
- `transcribe-btn`, `language-select` — transcription controls
- `progress-bar`, `progress-status`, `progress-container` — upload/transcoding progress
- `file-preview`, `file-name` — selected file display

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
- `resetUploadState()` — clear all upload-related UI state
- `showError()`, `showToast()` — user notifications
- `switchTab(tabName)` — navigate between tabs
- `translateServerError(err)` — convert server error objects to i18n strings
- `checkBalance()` — refresh quota display

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
