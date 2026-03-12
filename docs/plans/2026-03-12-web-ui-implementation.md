# LTX Desktop Web UI — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Electron+VNC Docker setup with a direct web UI using an electronAPI shim, a Python backend wrapper, and nginx — no original code changes.

**Architecture:** Inject a `window.electronAPI` shim into the built frontend at runtime. A Python wrapper (`web_launcher.py`) starts the original backend with file management routes added. Nginx serves static files and proxies API requests. Two processes at runtime: nginx + Python. A DOM-level `file://` URL interceptor rewrites Electron-style file URLs to HTTP serve URLs.

**Tech Stack:** Python/FastAPI (existing backend), nginx, ffmpeg (frame extraction), Vite (build-time only)

**Design doc:** `docs/plans/2026-03-12-web-ui-design.md`

---

### Task 1: Create web_launcher.py — Backend Wrapper

**Files:**
- Create: `web/web_launcher.py`

**Step 1: Write the launcher script**

This script imports the original FastAPI `app` from `ltx2_server` and adds file management routes. It must be run with CWD set to `backend/` so imports resolve.

Key design decisions vs. original plan:
- `FILE_ROOT` is always `/data` (the volume mount), separate from `LTX_APP_DATA_DIR` which the backend uses for models/outputs
- CORS fix: clear `app.user_middleware` before re-adding permissive CORS (original middleware stack otherwise keeps the restrictive one)
- `check_exist` endpoint now validates paths with `_is_safe_path`
- `_ensure_dirs()` called once at startup, not per-request

```python
"""Web launcher — wraps the original LTX backend with file routes for browser UI."""
import os
import sys
import uuid
import shutil
import mimetypes
import subprocess
import base64
import logging
from pathlib import Path

# Add backend to sys.path so we can import the original app
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Set required env vars BEFORE importing the app (which reads them at import time).
# LTX_APP_DATA_DIR is set by the entrypoint to /data/LTXDesktop.
os.environ.setdefault("LTX_APP_DATA_DIR", "/data/LTXDesktop")
os.environ["LTX_AUTH_TOKEN"] = ""      # No auth for web UI
os.environ["LTX_ADMIN_TOKEN"] = ""

from fastapi import UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# --- File route models ---

class CopyToAssetsRequest(BaseModel):
    srcPath: str
    projectId: str

class CheckExistRequest(BaseModel):
    paths: list[str]

class ExtractFrameRequest(BaseModel):
    videoPath: str
    seekTime: float
    width: int | None = None
    quality: int | None = None

class WriteFileRequest(BaseModel):
    path: str
    data: str
    encoding: str | None = None

# --- Path safety ---
# FILE_ROOT is the Docker volume mount — all file ops are restricted to this tree.
# This is separate from LTX_APP_DATA_DIR (which the backend uses for models/outputs).
FILE_ROOT = Path("/data").resolve()

def _is_safe_path(p: str) -> bool:
    """Ensure path is under FILE_ROOT (prevent traversal)."""
    try:
        resolved = Path(p).resolve()
        return resolved == FILE_ROOT or FILE_ROOT in resolved.parents
    except (ValueError, OSError):
        return False

# --- Dirs ---

UPLOADS_DIR = FILE_ROOT / "uploads"
TEMP_DIR = FILE_ROOT / "temp"
ASSETS_DIR = FILE_ROOT / "project-assets"
DOWNLOADS_DIR = FILE_ROOT / "downloads"

def _ensure_dirs() -> None:
    for d in [UPLOADS_DIR, TEMP_DIR, ASSETS_DIR, DOWNLOADS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# --- Build the file router ---

from fastapi import APIRouter

file_router = APIRouter(prefix="/api/files", tags=["files"])

@file_router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> dict[str, str]:
    filename = f"{uuid.uuid4().hex[:8]}_{file.filename or 'upload'}"
    dest = UPLOADS_DIR / filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    return {
        "path": str(dest),
        "url": f"/api/files/serve/{dest.relative_to(FILE_ROOT)}",
    }

@file_router.get("/serve/{file_path:path}")
async def serve_file(file_path: str) -> FileResponse:
    full = FILE_ROOT / file_path
    if not _is_safe_path(str(full)) or not full.is_file():
        return JSONResponse(status_code=404, content={"error": "Not found"})  # type: ignore[return-value]
    media_type = mimetypes.guess_type(str(full))[0] or "application/octet-stream"
    return FileResponse(str(full), media_type=media_type)

@file_router.post("/copy-to-assets")
async def copy_to_assets(req: CopyToAssetsRequest) -> dict[str, object]:
    src = Path(req.srcPath)
    if not _is_safe_path(str(src)) or not src.is_file():
        return {"success": False, "error": "Source file not found"}
    dest_dir = ASSETS_DIR / req.projectId
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(str(src), str(dest))
    url = f"/api/files/serve/{dest.relative_to(FILE_ROOT)}"
    return {"success": True, "path": str(dest), "url": url}

@file_router.post("/check-exist")
async def check_exist(req: CheckExistRequest) -> dict[str, bool]:
    return {p: (_is_safe_path(p) and Path(p).is_file()) for p in req.paths}

@file_router.post("/extract-frame")
async def extract_frame(req: ExtractFrameRequest) -> dict[str, str]:
    src = Path(req.videoPath)
    if not _is_safe_path(str(src)) or not src.is_file():
        return JSONResponse(status_code=404, content={"error": "Video not found"})  # type: ignore[return-value]
    out_name = f"frame_{uuid.uuid4().hex[:8]}.jpg"
    out_path = TEMP_DIR / out_name
    cmd = ["ffmpeg", "-y", "-ss", str(req.seekTime), "-i", str(src), "-frames:v", "1"]
    if req.width:
        cmd += ["-vf", f"scale={req.width}:-2"]
    if req.quality:
        cmd += ["-q:v", str(req.quality)]
    cmd.append(str(out_path))
    subprocess.run(cmd, capture_output=True, timeout=30)
    if not out_path.is_file():
        return JSONResponse(status_code=500, content={"error": "Frame extraction failed"})  # type: ignore[return-value]
    url = f"/api/files/serve/{out_path.relative_to(FILE_ROOT)}"
    return {"path": str(out_path), "url": url}

@file_router.post("/write")
async def write_file(req: WriteFileRequest) -> dict[str, object]:
    if not _is_safe_path(req.path):
        return {"success": False, "error": "Path not allowed"}
    dest = Path(req.path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    encoding = req.encoding or "utf-8"
    if encoding == "base64":
        dest.write_bytes(base64.b64decode(req.data))
    else:
        dest.write_text(req.data, encoding=encoding)
    return {"success": True, "path": str(dest)}

@file_router.get("/read")
async def read_file(path: str) -> dict[str, str]:
    p = Path(path)
    if not _is_safe_path(path) or not p.is_file():
        return JSONResponse(status_code=404, content={"error": "Not found"})  # type: ignore[return-value]
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
    return {"data": data, "mimeType": mime}

# --- Main ---

def main() -> None:
    _ensure_dirs()

    # Change to backend dir so relative imports in ltx2_server work
    os.chdir(str(BACKEND_DIR))

    # Import the original app (this triggers model loading, settings, etc.)
    from ltx2_server import app  # type: ignore[import-untyped]

    # Replace CORS middleware with permissive config for web UI.
    # Must clear user_middleware list then force middleware stack rebuild,
    # otherwise the original restrictive CORS middleware stays in the chain.
    app.user_middleware = [
        m for m in app.user_middleware
        if m.cls is not CORSMiddleware
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware_stack = None  # Force rebuild with new middleware list

    # Mount file routes
    app.include_router(file_router)

    import uvicorn
    port = int(os.environ.get("LTX_PORT", "8000"))
    logger.info(f"Web launcher starting on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    main()
```

**Step 2: Verify syntax**

Run: `cd /media/p5/LTX-Desktop && python -c "import ast; ast.parse(open('web/web_launcher.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add web/web_launcher.py
git commit -m "feat(web): add backend wrapper with file management routes"
```

---

### Task 2: Create electronapi-shim.js

**Files:**
- Create: `web/electronapi-shim.js`

**Step 1: Write the shim**

This script runs before React and defines `window.electronAPI` with web-native implementations. It also patches DOM element `src` setters to intercept `file://` URLs — the frontend creates these in ~25 locations and browsers block them.

Key fixes vs. original plan:
- DOM property override intercepts `file://` URLs on `<img>`, `<video>`, `<audio>`, `<source>` elements
- `saveFile` does NOT auto-trigger browser download (internal saves were causing popups)
- `downloadFile` uses path segment encoding (not `encodeURIComponent` on full path)
- `saveBinaryFile` uses chunked base64 encoding for large files
- Removed duplicate `checkPythonReady`
- `extractVideoFrame` handles `file://` URL input by stripping the prefix

```javascript
/**
 * electronAPI shim for running LTX Desktop as a web app.
 * Injected into index.html at runtime by nginx sub_filter.
 * Replaces Electron IPC with browser-native equivalents.
 */
(function () {
  "use strict";

  // FILE_ROOT matches the Docker volume mount. All server-side paths are under this.
  var FILE_ROOT = "/data";

  // =========================================================================
  // file:// URL interceptor
  //
  // The frontend converts server paths to file:// URLs in ~25 locations for
  // <video>, <img>, and <audio> src attributes. Browsers block file:// from
  // web pages. We patch the DOM property setters to rewrite them to HTTP
  // serve URLs transparently.
  // =========================================================================

  function rewriteFileUrl(url) {
    if (typeof url !== "string" || !url.startsWith("file://")) return url;
    // file:///data/foo/bar.mp4 → /data/foo/bar.mp4
    var fsPath = decodeURIComponent(url.slice(7));
    // Strip leading slash duplication (file:/// → /)
    if (/^\/[A-Za-z]:/.test(fsPath)) fsPath = fsPath.slice(1); // Windows drive
    // Only rewrite paths under FILE_ROOT
    if (fsPath.startsWith(FILE_ROOT + "/")) {
      var relative = fsPath.substring(FILE_ROOT.length + 1);
      // Encode each path segment individually (preserve /)
      var encoded = relative.split("/").map(encodeURIComponent).join("/");
      return "/api/files/serve/" + encoded;
    }
    return url;
  }

  // Patch src property on media elements to intercept file:// URLs
  function patchSrcProperty(prototype) {
    var descriptor = Object.getOwnPropertyDescriptor(prototype, "src");
    if (!descriptor || !descriptor.set) return;
    var originalSet = descriptor.set;
    var originalGet = descriptor.get;
    Object.defineProperty(prototype, "src", {
      get: originalGet,
      set: function (value) {
        originalSet.call(this, rewriteFileUrl(value));
      },
      enumerable: descriptor.enumerable,
      configurable: descriptor.configurable,
    });
  }

  // Patch all relevant element prototypes
  if (typeof HTMLMediaElement !== "undefined") patchSrcProperty(HTMLMediaElement.prototype);
  if (typeof HTMLImageElement !== "undefined") patchSrcProperty(HTMLImageElement.prototype);
  if (typeof HTMLSourceElement !== "undefined") patchSrcProperty(HTMLSourceElement.prototype);

  // Also patch setAttribute for src (React sometimes uses this)
  var origSetAttribute = Element.prototype.setAttribute;
  Element.prototype.setAttribute = function (name, value) {
    if (name === "src" || name === "poster") {
      value = rewriteFileUrl(value);
    }
    return origSetAttribute.call(this, name, value);
  };

  // =========================================================================
  // Helpers
  // =========================================================================

  /** Convert a server filesystem path to a serve URL */
  function pathToServeUrl(serverPath) {
    if (serverPath && serverPath.startsWith(FILE_ROOT + "/")) {
      var relative = serverPath.substring(FILE_ROOT.length + 1);
      return "/api/files/serve/" + relative.split("/").map(encodeURIComponent).join("/");
    }
    return serverPath;
  }

  /** Trigger a hidden file input and return selected files uploaded to server. */
  function openFilePicker(accept, multiple) {
    return new Promise(function (resolve) {
      var input = document.createElement("input");
      input.type = "file";
      if (accept) input.accept = accept;
      if (multiple) input.multiple = true;
      input.style.display = "none";
      document.body.appendChild(input);
      var resolved = false;
      input.addEventListener("change", async function () {
        resolved = true;
        var files = Array.from(input.files || []);
        if (files.length === 0) { cleanup(); resolve(null); return; }
        var paths = [];
        for (var i = 0; i < files.length; i++) {
          var form = new FormData();
          form.append("file", files[i]);
          try {
            var resp = await fetch("/api/files/upload", { method: "POST", body: form });
            var json = await resp.json();
            paths.push(json.path);
          } catch (e) {
            console.error("[LTX Web] Upload failed:", e);
          }
        }
        cleanup();
        resolve(paths.length > 0 ? paths : null);
      });
      function cleanup() { try { document.body.removeChild(input); } catch(e) {} }
      // Detect cancel via focus return
      window.addEventListener("focus", function onFocus() {
        window.removeEventListener("focus", onFocus);
        setTimeout(function () { if (!resolved) { cleanup(); resolve(null); } }, 500);
      });
      input.click();
    });
  }

  /** Build accept string from Electron-style filters. */
  function filtersToAccept(filters) {
    if (!filters || filters.length === 0) return "";
    var exts = [];
    filters.forEach(function (f) {
      (f.extensions || []).forEach(function (ext) { exts.push("." + ext); });
    });
    return exts.join(",");
  }

  /** Trigger browser download for a server-side file. */
  function triggerDownload(serverPath) {
    var url = pathToServeUrl(serverPath);
    if (!url) return;
    var a = document.createElement("a");
    a.href = url;
    a.download = serverPath.split("/").pop() || "download";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  // =========================================================================
  // Health polling
  // =========================================================================

  var healthCallbacks = [];
  var lastHealthStatus = { status: "alive" };

  function pollHealth() {
    fetch("/health")
      .then(function (r) {
        lastHealthStatus = r.ok ? { status: "alive" } : { status: "dead" };
      })
      .catch(function () {
        lastHealthStatus = { status: "dead" };
      })
      .finally(function () {
        healthCallbacks.forEach(function (cb) { try { cb(lastHealthStatus); } catch(e) {} });
      });
  }

  setInterval(pollHealth, 5000);

  // =========================================================================
  // The shim
  // =========================================================================

  window.electronAPI = {
    // Platform
    platform: "linux",

    // Backend connectivity — same origin, no auth
    getBackend: function () {
      return Promise.resolve({ url: "", token: "" });
    },
    getBackendHealthStatus: function () {
      return Promise.resolve(lastHealthStatus);
    },
    onBackendHealthStatus: function (cb) {
      healthCallbacks.push(cb);
      return function () {
        healthCallbacks = healthCallbacks.filter(function (c) { return c !== cb; });
      };
    },

    // Setup/bootstrap — bypass everything (backend already running)
    checkPythonReady: function () { return Promise.resolve({ ready: true }); },
    startPythonBackend: function () { return Promise.resolve(); },
    startPythonSetup: function () { return Promise.resolve(); },
    checkFirstRun: function () { return Promise.resolve({ needsSetup: false, needsLicense: false }); },
    acceptLicense: function () { return Promise.resolve(true); },
    completeSetup: function () { return Promise.resolve(true); },
    fetchLicenseText: function () { return Promise.resolve("Apache-2.0 License"); },
    getNoticesText: function () { return Promise.resolve(""); },
    onPythonSetupProgress: function () {},
    removePythonSetupProgress: function () {},

    // App info
    getAppInfo: function () {
      return Promise.resolve({
        version: "web",
        isPackaged: true,
        modelsPath: FILE_ROOT + "/LTXDesktop/models",
        userDataPath: FILE_ROOT + "/LTXDesktop",
      });
    },
    getModelsPath: function () { return Promise.resolve(FILE_ROOT + "/LTXDesktop/models"); },
    getDownloadsPath: function () { return Promise.resolve(FILE_ROOT + "/downloads"); },
    getResourcePath: function () { return Promise.resolve(null); },
    getProjectAssetsPath: function () { return Promise.resolve(FILE_ROOT + "/project-assets"); },
    getLogPath: function () {
      return Promise.resolve({ logPath: FILE_ROOT + "/logs/ltx.log", logDir: FILE_ROOT + "/logs" });
    },
    checkGpu: function () {
      return fetch("/api/runtime-policy")
        .then(function (r) { return r.json(); })
        .then(function (data) {
          return { available: !data.force_api_generations, name: "NVIDIA GPU", vram: 0 };
        })
        .catch(function () { return { available: true }; });
    },

    // File dialogs → HTML5 file picker + upload to server
    showOpenFileDialog: function (options) {
      var accept = filtersToAccept(options && options.filters);
      var multi = options && options.properties && options.properties.indexOf("multiSelections") >= 0;
      return openFilePicker(accept, multi);
    },
    showSaveDialog: function (options) {
      // Generate a server-side path; user downloads the result after it's written
      var name = (options && options.defaultPath) || "export_" + Date.now();
      var filename = name.split("/").pop() || name;
      return Promise.resolve(FILE_ROOT + "/downloads/" + filename);
    },
    showOpenDirectoryDialog: function () {
      // No directory picker in browsers — return default data dir
      return Promise.resolve(FILE_ROOT + "/LTXDesktop");
    },

    // File operations → backend endpoints
    readLocalFile: function (filePath) {
      return fetch("/api/files/read?path=" + encodeURIComponent(filePath))
        .then(function (r) { return r.json(); });
    },
    saveFile: function (filePath, data, encoding) {
      return fetch("/api/files/write", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: filePath, data: data, encoding: encoding || "utf-8" }),
      }).then(function (r) { return r.json(); });
      // NOTE: No auto-download here. Internal saves (project JSON, settings)
      // should not trigger browser downloads. Use triggerDownload() explicitly
      // for user-facing exports.
    },
    saveBinaryFile: function (filePath, data) {
      // Convert ArrayBuffer to base64 using chunked approach for large files
      var bytes = new Uint8Array(data);
      var chunkSize = 32768;
      var parts = [];
      for (var i = 0; i < bytes.length; i += chunkSize) {
        var chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
        parts.push(String.fromCharCode.apply(null, chunk));
      }
      var b64 = btoa(parts.join(""));
      return fetch("/api/files/write", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: filePath, data: b64, encoding: "base64" }),
      }).then(function (r) { return r.json(); });
    },
    copyToProjectAssets: function (srcPath, projectId) {
      return fetch("/api/files/copy-to-assets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ srcPath: srcPath, projectId: projectId }),
      }).then(function (r) { return r.json(); });
    },
    checkFilesExist: function (filePaths) {
      return fetch("/api/files/check-exist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths: filePaths }),
      }).then(function (r) { return r.json(); });
    },
    searchDirectoryForFiles: function (dir, filenames) {
      var paths = filenames.map(function (f) { return dir + "/" + f; });
      return fetch("/api/files/check-exist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths: paths }),
      })
        .then(function (r) { return r.json(); })
        .then(function (exists) {
          var result = {};
          filenames.forEach(function (f) {
            var full = dir + "/" + f;
            if (exists[full]) result[f] = full;
          });
          return result;
        });
    },

    // Video frame extraction
    extractVideoFrame: function (videoUrl, seekTime, width, quality) {
      // Convert file:// URL or serve URL back to a server filesystem path
      var videoPath = videoUrl;
      if (videoUrl.startsWith("file://")) {
        videoPath = decodeURIComponent(videoUrl.slice(7));
      } else if (videoUrl.startsWith("/api/files/serve/")) {
        videoPath = FILE_ROOT + "/" + decodeURIComponent(videoUrl.replace("/api/files/serve/", ""));
      }
      return fetch("/api/files/extract-frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ videoPath: videoPath, seekTime: seekTime, width: width, quality: quality }),
      }).then(function (r) { return r.json(); });
    },

    // Export — v1: not supported, users download raw clips
    exportNative: function () {
      return Promise.resolve({ success: false, error: "Timeline export not available in web UI. Download clips individually." });
    },
    exportCancel: function () { return Promise.resolve({ ok: true }); },

    // Directory change dialogs — not available in browser
    openModelsDirChangeDialog: function () {
      return Promise.resolve({ success: false, error: "Not available in web UI" });
    },
    openProjectAssetsPathChangeDialog: function () {
      return Promise.resolve({ success: false, error: "Not available in web UI" });
    },

    // Open external links
    openLtxApiKeyPage: function () { window.open("https://ltx.video", "_blank"); return Promise.resolve(true); },
    openFalApiKeyPage: function () { window.open("https://fal.ai", "_blank"); return Promise.resolve(true); },

    // Navigation — no-op in browser
    showItemInFolder: function () { return Promise.resolve(); },
    openParentFolderOfFile: function () { return Promise.resolve(); },
    openLogFolder: function () { return Promise.resolve(true); },

    // Logging
    writeLog: function (level, message) {
      console[level === "ERROR" ? "error" : "log"]("[LTX]", message);
      return Promise.resolve();
    },
    getLogs: function () {
      return Promise.resolve({ logPath: "", lines: ["Logs not available in web UI"], error: undefined });
    },

    // Analytics — no-op
    getAnalyticsState: function () { return Promise.resolve({ analyticsEnabled: false, installationId: "web" }); },
    setAnalyticsEnabled: function () { return Promise.resolve(); },
    sendAnalyticsEvent: function () { return Promise.resolve(); },
  };

  // Expose triggerDownload for explicit user exports
  window.__ltxWebDownload = triggerDownload;

  console.log("[LTX Web] electronAPI shim loaded — file:// URL interception active");
})();
```

**Step 2: Verify syntax**

Run: `node -c web/electronapi-shim.js && echo "OK"`
Expected: `OK`

**Step 3: Commit**

```bash
git add web/electronapi-shim.js
git commit -m "feat(web): add electronAPI shim with file:// URL interception"
```

---

### Task 3: Create nginx.conf

**Files:**
- Create: `web/nginx.conf`

**Step 1: Write nginx config**

Key fixes vs. original plan:
- `sub_filter` moved to `server` block so it applies to SPA fallback routes too
- Added `/ws/` location block for WebSocket proxy (model downloads)
- Health endpoint gets same proxy headers as API

```nginx
worker_processes auto;
error_log /data/logs/nginx-error.log warn;
pid /tmp/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    access_log /data/logs/nginx-access.log;

    sendfile on;
    keepalive_timeout 65;
    client_max_body_size 500M;

    server {
        listen 6080;
        server_name _;

        root /srv/frontend;
        index index.html;

        # Inject the electronAPI shim into every HTML response.
        # This covers both direct /index.html and SPA fallback via try_files.
        sub_filter '</head>' '<script src="/electronapi-shim.js"></script></head>';
        sub_filter_once on;
        sub_filter_types text/html;

        # Serve the shim JS file
        location = /electronapi-shim.js {
            alias /srv/electronapi-shim.js;
        }

        # Proxy API requests to Python backend
        location /api/ {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_read_timeout 600s;
            proxy_send_timeout 600s;

            # WebSocket support (for API endpoints that upgrade)
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Proxy WebSocket connections (model downloads use /ws/ path)
        location /ws/ {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_read_timeout 600s;
            proxy_send_timeout 600s;

            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Health endpoint proxy
        location = /health {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # SPA fallback — serve index.html for all non-file routes
        location / {
            try_files $uri $uri/ /index.html;
        }
    }
}
```

**Step 2: Commit**

```bash
git add web/nginx.conf
git commit -m "feat(web): add nginx config with API/WS proxy and shim injection"
```

---

### Task 4: Create web Dockerfile

**Files:**
- Create: `web/Dockerfile`

**Step 1: Write the multi-stage Dockerfile**

Key fixes vs. original plan:
- `uv sync --frozen` without `--extra dev` (no test deps in production)
- Uses `uv run` instead of hardcoded venv path
- Frontend build stage uses selective COPY for better caching

```dockerfile
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
    && uv sync --frozen

# Copy built frontend from stage 1
COPY --from=frontend-builder /build/dist /srv/frontend

# Copy web layer (shim, launcher, nginx config, entrypoint)
COPY web/electronapi-shim.js /srv/electronapi-shim.js
COPY web/web_launcher.py /app/web/web_launcher.py
COPY web/nginx.conf /etc/nginx/nginx.conf
COPY web/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Settings fallback (default settings.json from repo root)
COPY settings.json /app/settings.json

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

EXPOSE 6080
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:6080/ || exit 1

ENTRYPOINT ["/entrypoint.sh"]
```

**Step 2: Commit**

```bash
git add web/Dockerfile
git commit -m "feat(web): add multi-stage Dockerfile for web UI"
```

---

### Task 5: Create web entrypoint.sh

**Files:**
- Create: `web/entrypoint.sh`

**Step 1: Write the entrypoint**

Key fixes vs. original plan:
- Directory creation matches what web_launcher.py expects (under `/data/`)
- Uses `uv run` for robust Python invocation
- Nginx starts with `daemon on` (default) so it backgrounds itself

```bash
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
exec uv run python /app/web/web_launcher.py
```

**Step 2: Commit**

```bash
git add web/entrypoint.sh
git commit -m "feat(web): add entrypoint script"
```

---

### Task 6: Create docker-compose for web UI

**Files:**
- Create: `web/docker-compose.yml`

**Step 1: Write compose file**

```yaml
##############################################################################
# LTX Desktop Web UI — Docker Compose for Unraid with NVIDIA GPU
#
# Usage:
#   docker compose -f web/docker-compose.yml up -d
#
# Then open http://<your-unraid-ip>:6080 in your browser.
##############################################################################

services:
  ltx-web:
    build:
      context: ..
      dockerfile: web/Dockerfile
    container_name: ltx-web
    restart: unless-stopped

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    ports:
      - "6080:6080"

    volumes:
      - ltx-data:/data

    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

    shm_size: "8gb"

volumes:
  ltx-data:
```

**Step 2: Commit**

```bash
git add web/docker-compose.yml
git commit -m "feat(web): add docker-compose for web UI deployment"
```

---

### Task 7: Update GitHub Actions to build web image

**Files:**
- Modify: `.github/workflows/docker.yml`

**Step 1: Add a second job for the web image**

Rename the existing job to `build-vnc` and add a `build-web` job that uses `web/Dockerfile`. The web image gets tagged as `ltx-desktop-web`.

Key changes:
- Rename `build-and-push` → `build-vnc`
- Add `build-web` job (copy of build-vnc but with `web/Dockerfile` and `IMAGE_NAME` suffix `-web`)

**Step 2: Commit**

```bash
git add .github/workflows/docker.yml
git commit -m "ci: add web UI Docker image build to GitHub Actions"
```

---

### Task 8: Integration test — build and verify

**Step 1: Build the web Docker image**

Run: `docker compose -f web/docker-compose.yml build`

**Step 2: Run the container**

Run: `docker compose -f web/docker-compose.yml up -d`

**Step 3: Verify shim injection**

Run: `curl -s http://localhost:6080/ | grep electronapi-shim`
Expected: `<script src="/electronapi-shim.js"></script>`

**Step 4: Verify shim is served**

Run: `curl -s http://localhost:6080/electronapi-shim.js | head -5`
Expected: Lines starting with `/**`

**Step 5: Verify backend health**

Run: `curl -s http://localhost:6080/health`
Expected: JSON health response

**Step 6: Verify file upload endpoint**

Run: `echo "test" > /tmp/test.txt && curl -s -F "file=@/tmp/test.txt" http://localhost:6080/api/files/upload`
Expected: JSON with `path` and `url` fields

**Step 7: Verify file serving**

Run: Use the `url` from step 6 to fetch the file back:
`curl -s http://localhost:6080/api/files/serve/uploads/<filename>`
Expected: File content

**Step 8: Commit any fixes**

```bash
git add -A
git commit -m "fix(web): integration test fixes"
```
