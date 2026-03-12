##############################################################################
# LTX Desktop Web UI — Docker container with NVIDIA GPU
#
# Serves the React frontend directly in the browser (no Electron/VNC).
# Two runtime processes: nginx (static files) + Python backend (FastAPI).
##############################################################################

# ── Stage 1: Build frontend ──────────────────────────────────────────────────
FROM node:22-bookworm AS frontend-builder

WORKDIR /build
COPY package.json pnpm-lock.yaml ./
RUN corepack enable && corepack prepare pnpm@10.30.3 --activate \
    && pnpm install --frozen-lockfile

# Copy only what Vite needs (not backend/, .git/, etc.)
COPY index.html vite.config.ts tsconfig.json tsconfig.node.json ./
COPY frontend/ frontend/
COPY electron/ electron/
COPY public/ public/
RUN pnpm build:frontend

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# System packages (nginx, ffmpeg, build tools for potential JIT compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx ffmpeg \
    curl git ca-certificates build-essential pkg-config \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libffi-dev liblzma-dev \
    procps \
    && rm -rf /var/lib/apt/lists/*

# uv (Python package manager) — pinned version
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /usr/local/bin/uv

# Copy backend source + install Python deps (no dev/test extras)
WORKDIR /app
COPY backend/ backend/
RUN cd backend \
    && uv python install 3.13 \
    && uv sync --frozen --python 3.13

# Copy built frontend from stage 1
COPY --from=frontend-builder /build/dist /srv/frontend

# Copy web layer (shim, launcher, nginx config, entrypoint)
COPY web/electronapi-shim.js /srv/electronapi-shim.js
COPY web/web_launcher.py /app/web/web_launcher.py
COPY web/nginx.conf /etc/nginx/nginx.conf
COPY web/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && chmod -R a+r /srv/

# Settings fallback (default settings.json from repo root)
COPY settings.json /app/settings.json

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

EXPOSE 6080
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:6080/ || exit 1

ENTRYPOINT ["/entrypoint.sh"]
