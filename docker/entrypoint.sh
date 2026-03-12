#!/usr/bin/env bash
set -euo pipefail

RESOLUTION="${RESOLUTION:-1920x1080x24}"

# ── Ensure data directories exist and are owned by ltx ───────────────────────
mkdir -p /data/LTXDesktop /data/logs
chown -R ltx:ltx /data

echo "============================================"
echo "  LTX Desktop Docker Container"
echo "  Resolution: ${RESOLUTION%x*}"
echo "  noVNC UI:   http://<host>:6080/vnc.html"
echo "============================================"

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/ltx.conf
