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
