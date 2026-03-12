# LTX Desktop Web UI — Design Document

**Date:** 2026-03-12
**Goal:** Run LTX Desktop as a web app in Docker (Unraid + NVIDIA GPU) without modifying original repo code.

## Architecture

Replace the Electron + VNC approach with a direct web UI:

```
Browser (:6080)
  ├─ GET /*      → Nginx → Static files (dist/ + electronapi-shim.js)
  ├─ POST /api/* → Nginx → Python backend (:8000)
  └─ WS /api/*   → Nginx → Python backend (:8000) websocket
```

**Three new files, zero original code changes:**

- `web/electronapi-shim.js` — defines `window.electronAPI` with web-native implementations
- `web/web_launcher.py` — wraps the original FastAPI app, adds file management routes
- `web/nginx.conf` — serves frontend, proxies API, injects shim into index.html

**Runtime:** nginx + Python only. No Electron, Node.js, Xvfb, or VNC.

## electronAPI Shim Strategy

The frontend uses `window.electronAPI?.method?.()` (optional chaining), enabling graceful degradation. The shim provides web-native replacements:

### Setup/Bootstrap → Bypass
| Method | Web Implementation |
|--------|-------------------|
| `checkPythonReady` | `{ ready: true }` |
| `startPythonBackend` | no-op |
| `checkFirstRun` | `{ needsSetup: false, needsLicense: false }` |
| `acceptLicense`, `completeSetup` | no-op, return true |

### Backend Connectivity → Direct
| Method | Web Implementation |
|--------|-------------------|
| `getBackend` | `{ url: "", token: "" }` (same origin, no auth) |
| `getBackendHealthStatus` | fetch `/health` |
| `onBackendHealthStatus` | poll `/health` every 5s |

### File Dialogs → HTML5
| Method | Web Implementation |
|--------|-------------------|
| `showOpenFileDialog` | hidden `<input type="file">`, upload via `/api/upload` |
| `showSaveDialog` | generate server-side path under `/data/downloads/` |
| `showOpenDirectoryDialog` | return preset paths |

### File Operations → Backend Endpoints
| Method | Web Implementation |
|--------|-------------------|
| `saveFile`, `saveBinaryFile` | POST `/api/files/write` |
| `readLocalFile` | GET `/api/files/read?path=...` |
| `copyToProjectAssets` | POST `/api/files/copy-to-assets` |
| `checkFilesExist` | POST `/api/files/check-exist` |
| `extractVideoFrame` | POST `/api/files/extract-frame` |

### Export → v1: Download Raw Clips
| Method | Web Implementation |
|--------|-------------------|
| `exportNative` | no-op (returns error message) |
| Generated clips | downloadable via `/api/files/serve/` |

### Nice-to-have → No-op or Trivial
| Method | Web Implementation |
|--------|-------------------|
| `writeLog`, analytics | no-op / console.log |
| `showItemInFolder`, `openLogFolder` | no-op |
| `openLtxApiKeyPage`, `openFalApiKeyPage` | `window.open(url)` |
| `getAppInfo` | static object |
| `platform` | `"linux"` |

## Backend File Routes

New routes mounted by `web_launcher.py` on the existing FastAPI app:

```
POST /api/upload              — multipart upload → /data/uploads/{uuid}_{name}
GET  /api/files/serve/{path}  — serve files under /data/ (videos, images, clips)
POST /api/files/copy-to-assets — copy to /data/project-assets/{projectId}/
POST /api/files/check-exist   — batch file existence check
POST /api/files/extract-frame — ffmpeg frame extraction → /data/temp/
POST /api/files/write         — write file to /data/*
GET  /api/files/read          — read file as base64 + mimeType
```

All routes restricted to `/data/` prefix (path traversal prevention). No authentication.

## Docker Image

Multi-stage build on `nvidia/cuda:12.8.0-devel-ubuntu24.04`:

**Build stage:** Node.js + pnpm to build Vite frontend (`dist/`)
**Runtime stage:** Python 3.13 (uv) + nginx + ffmpeg only

**Processes:** nginx (:6080) + Python backend (:8000)
**Volume:** `/data` (models, outputs, uploads, project assets, temp)
**Port:** 6080

## Scope

### In scope (v1)
- Full generation UI (text-to-video, image-to-video, retake, IC-LoRA)
- Model management (download, load, settings)
- File upload via browser
- Raw clip/image download
- Settings management

### Out of scope (future)
- Timeline video export (ffmpeg compositing)
- Timeline import (FCPXML)
- File relinking workflow

## Constraints
- Zero modifications to original repo code (keeps upstream sync)
- All new files live in `web/` and `docker/`
