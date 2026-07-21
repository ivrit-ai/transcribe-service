#!/bin/sh
set -e
exec uvicorn app:app --host 0.0.0.0 --port "$PORT" --proxy-headers --forwarded-allow-ips='*'
