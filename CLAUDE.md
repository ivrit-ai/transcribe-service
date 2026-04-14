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
- Use logging instead of print statements
- When logging an exception, always include the actual exception
- Use non-technical user-friendly messages for the end user (don't expose exceptions)
- Function signatures should have no defaults unless explicitly requested
- No over-engineering: simplest solution that meets requirements wins
- No hacks: localized solutions that should be solved more generally are not acceptable unless explicitly justified

## Agents

Specialized agents live in `.claude/agents/`. They are invoked automatically as subagents during non-trivial work. You can also request a specific agent by name (e.g. "run the reviewer agent on this change").

### Agent Workflow (non-trivial changes)

1. **visionary** — Vision/strategy check (skip for bug fixes and minor changes)
2. **architect** — Design the solution, update `ARCHITECTURE.md` if needed
3. **coder** — Implement the change
4. **reviewer** — Quality gate (no hacks, no over-engineering). No change is complete until approved.
5. **manager** — Completion checklist and summary

### Quick Reference

| Agent | When to use | Invoke with |
|-------|------------|-------------|
| `visionary` | New features, strategic decisions, scope changes | `@visionary` or "check vision alignment" |
| `architect` | New components, interface changes, new dependencies | `@architect` or "design this" |
| `coder` | All implementation work | `@coder` or "implement this" |
| `reviewer` | After implementation, before completion | `@reviewer` or "review this change" |
| `manager` | After reviewer approves | `@manager` or "wrap this up" |

### Policies Enforced by Agents

- **NO HACK**: A hack is a localized solution that should be solved more generally. Push back with a proper solution.
- **NO OVER-ENGINEERING**: No abstractions for single-use cases, no designing for hypothetical futures, no extra layers "just in case".
