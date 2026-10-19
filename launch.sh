#!/bin/sh
# transcribe.ivrit.ai is retired: retired.py sends everyone to app.ivrit.ai.
# The service itself (app.py) is kept in the repo, not run.
set -e
exec uvicorn retired:app --host 0.0.0.0 --port "$PORT" --proxy-headers --forwarded-allow-ips='*'
