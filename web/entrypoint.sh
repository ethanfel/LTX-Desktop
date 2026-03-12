#!/usr/bin/env bash
set -euo pipefail

# Ensure all data directories exist.
# LTX_APP_DATA_DIR controls where the backend stores models/outputs/settings.
# FILE_ROOT (/data) is the broader volume where uploads, temp files, etc. live.
mkdir -p /data/LTXDesktop/models /data/LTXDesktop/outputs \
         /data/uploads /data/temp /data/project-assets /data/downloads /data/logs

export LTX_APP_DATA_DIR="/data/LTXDesktop"
export LTX_PORT="${LTX_PORT:-8000}"

echo "============================================"
echo "  LTX Desktop Web UI"
echo "  UI:      http://<host>:6080"
echo "  Backend: 0.0.0.0:${LTX_PORT}"
echo "============================================"

# Start nginx in background (serves frontend + proxies to backend)
nginx

# Start the Python backend in foreground (container lifecycle tied to this)
cd /app/backend
exec uv run --python 3.13 python /app/web/web_launcher.py
