##############################################################################
# LTX Desktop – Docker container with NVIDIA GPU + noVNC browser UI
#
# Runs the full Electron app inside a virtual display (Xvfb) and exposes it
# via noVNC so you can access the UI from any browser.
#
# No modifications to the original LTX Desktop source code are required.
##############################################################################

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Display / VNC
    xvfb x11vnc xdotool fluxbox \
    # noVNC (websocket proxy for VNC)
    novnc websockify \
    # Electron runtime deps
    libgtk-3-0 libnotify4 libnss3 libxss1 libxtst6 xdg-utils \
    libatspi2.0-0 libdrm2 libgbm1 libasound2t64 \
    libx11-xcb1 libxcomposite1 libxcursor1 libxdamage1 \
    libxrandr2 libpango-1.0-0 libcairo2 libcups2 \
    # Build tools
    curl git ca-certificates build-essential pkg-config \
    # Python build deps
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libffi-dev liblzma-dev \
    # Misc
    procps supervisor fonts-liberation dbus-x11 \
    && rm -rf /var/lib/apt/lists/*

# ── Node.js 22 LTS ──────────────────────────────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# ── pnpm ─────────────────────────────────────────────────────────────────────
RUN corepack enable && corepack prepare pnpm@10.30.3 --activate

# ── uv (Python package manager) ─────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /usr/local/bin/uv

# ── Create non-root user (Electron sandbox requires non-root) ────────────────
RUN groupadd -r ltx && useradd -r -g ltx -m -s /bin/bash ltx

# ── App source ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY --chown=ltx:ltx . .

# ── Node dependencies ────────────────────────────────────────────────────────
RUN pnpm install --frozen-lockfile

# ── Python venv with all backend deps ────────────────────────────────────────
USER ltx
RUN cd /app/backend \
    && uv python install 3.13 \
    && uv sync --frozen --extra dev
USER root

# ── Build Vite frontend ─────────────────────────────────────────────────────
RUN pnpm build:frontend

# ── Fix final permissions ────────────────────────────────────────────────────
RUN chown -R ltx:ltx /app/dist /app/dist-electron 2>/dev/null || true

# ── Display / VNC config ────────────────────────────────────────────────────
ENV DISPLAY=:99
ENV RESOLUTION=1920x1080x24

# noVNC port (browser access)
EXPOSE 6080

# ── Persistent data ─────────────────────────────────────────────────────────
# Electron reads XDG_DATA_HOME to find its data dir on Linux.
# Setting XDG_DATA_HOME=/data makes it store everything under /data/LTXDesktop
# (models, settings, outputs, logs).
VOLUME ["/data"]

# ── Supervisord config ──────────────────────────────────────────────────────
COPY docker/supervisord.conf /etc/supervisor/conf.d/ltx.conf
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:6080/ || exit 1

ENTRYPOINT ["/entrypoint.sh"]
