# CLAUDE.md

## Project Overview

ivrit.ai — a Hebrew-focused audio transcription service. See [ARCHITECTURE.md](ARCHITECTURE.md) for full architecture documentation including tech stack, features, project structure, API surface, and all tunable configuration (CLI flags, environment variables, config.json, and hardcoded constants).

## Quick Reference

- **Main app:** `app.py` (FastAPI, ~3300 lines — routes, job queue, quota, transcription logic)
- **Frontend:** `templates/index.html` (vanilla JS SPA, ~4000 lines)
- **Storage backends:** `gdrive_file_utils.py` (Google Drive), `local_file_utils.py` (local filesystem), `file_utils.py` (abstract interface)
- **Auth:** `gdrive_auth.py` (Google OAuth token management)
- **i18n:** `static/i18n.js`
- **Config:** `config.json` (languages and models)

See [ARCHITECTURE.md](ARCHITECTURE.md) for frontend internals (DOM IDs, JS functions, state variables), upload pipeline flow, streaming event types, modal patterns, and i18n conventions.

## Running

```bash
# Cloud mode (requires Google OAuth + RunPod env vars)
python run.py

# Local mode
python run.py --local --data-dir ./local_data

# Dev mode
python run.py --dev --dev-user-email you@example.com
```

## Code Style

- Python formatting: Black (config in `pyproject.toml`)
- No frontend build step — vanilla JS served directly via Jinja2 templates
