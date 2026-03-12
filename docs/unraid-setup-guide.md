# LTX Desktop Web UI — Unraid Setup Guide

## Prerequisites

- Unraid 6.12+ with NVIDIA GPU plugin installed
- NVIDIA GPU (tested with RTX 6000 Pro Blackwell)
- Docker image: `ghcr.io/ethanfel/ltx-desktop-web:latest`

---

## Unraid Docker UI Setup

### 1. Add Container

Go to **Docker** tab → **Add Container** and fill in:

| Field | Value |
|-------|-------|
| **Name** | `ltx-desktop` |
| **Repository** | `ghcr.io/ethanfel/ltx-desktop-web:latest` |
| **Network Type** | `bridge` |

### 2. Port Mapping

Click **Add another Path, Port, Variable, Label or Device**:

| Config Type | Container Port | Host Port | Description |
|-------------|---------------|-----------|-------------|
| Port | `6080` | `6080` | Web UI |

### 3. Volume Mapping (Persistent Storage)

One volume holds everything. Click **Add another Path, Port, Variable, Label or Device**:

| Config Type | Container Path | Host Path | Access Mode | Description |
|-------------|---------------|-----------|-------------|-------------|
| Path | `/data` | `/mnt/user/appdata/ltx-desktop` | Read/Write | All app data |

### 4. GPU Passthrough

Under **Extra Parameters**, add:

```
--runtime=nvidia --gpus all --shm-size=8g
```

Or if your Unraid NVIDIA plugin supports it, toggle the GPU assignment in the container settings.

### 5. Environment Variables (Optional)

These are set automatically but can be overridden:

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_VISIBLE_DEVICES` | `all` | Which GPUs to expose |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility,video` | GPU capabilities |
| `LTX_PORT` | `8000` | Internal backend port (no need to change) |

### 6. Apply and Start

Click **Apply**. First start takes 30–60 seconds (backend loads PyTorch + model config).

Open **http://your-unraid-ip:6080** in your browser.

---

## What's Inside `/data`

Everything lives under the single `/data` volume. Here's what each subdirectory contains:

```
/data/
├── LTXDesktop/           ← Backend app data
│   ├── models/           ← Downloaded AI models (~5-20 GB each)
│   ├── outputs/          ← Generated videos and images
│   └── settings.json     ← App settings (resolution, model preferences)
│
├── project-assets/       ← Uploaded media copied into projects
│   └── {projectId}/      ← Per-project asset folders
│
├── uploads/              ← Raw browser uploads (source images/videos)
├── downloads/            ← Exported files (subtitles, clips)
├── temp/                 ← Temporary frame extractions (auto-cleaned hourly)
└── logs/                 ← nginx + backend logs
    ├── nginx-access.log
    ├── nginx-error.log
    └── ltx.log
```

### What matters for backup

| Directory | Size | Backup? | Why |
|-----------|------|---------|-----|
| `LTXDesktop/models/` | 5–50 GB | Optional | Can re-download from UI |
| `LTXDesktop/outputs/` | Varies | **Yes** | Your generated content |
| `LTXDesktop/settings.json` | Tiny | **Yes** | Your preferences |
| `project-assets/` | Varies | **Yes** | Media used in projects |
| `uploads/` | Varies | Optional | Source files you uploaded |
| `downloads/` | Varies | **Yes** | Your exports |
| `temp/` | Small | No | Auto-cleaned every hour |
| `logs/` | Small | No | Diagnostic only |

### Disk space estimate

- **Minimum**: ~15 GB (one model + some outputs)
- **Typical**: ~30-50 GB (2-3 models + active projects)
- **Heavy use**: 100+ GB (all models + lots of generated content)

---

## First Run

1. Open **http://your-unraid-ip:6080**
2. The UI loads immediately — no setup wizard (bypassed for web)
3. Go to **Settings** → download your preferred model (LTX-2 Fast recommended to start)
4. Model download progress shows in the UI via WebSocket
5. Once downloaded, select the model and start generating

---

## Troubleshooting

### UI loads but shows "Backend unavailable"

Backend is still starting. Wait 30-60 seconds and refresh. Check logs:

```bash
docker logs ltx-desktop
```

### 502 Bad Gateway on first load

Same as above — nginx starts before the Python backend finishes initializing. Refresh after ~30 seconds.

### GPU not detected

Verify NVIDIA runtime is working:

```bash
docker exec ltx-desktop nvidia-smi
```

If this fails, check that the Unraid NVIDIA plugin is installed and the GPU is not assigned to a VM.

### Models download is slow

Models are downloaded from HuggingFace. The download runs through the WebSocket proxy. No special configuration needed.

### Out of VRAM

Check GPU memory usage:

```bash
docker exec ltx-desktop nvidia-smi
```

LTX-2 Fast (distilled) uses less VRAM than the full model. Reduce resolution in settings if needed.

---

## Updating

Pull the latest image and recreate the container:

```bash
docker pull ghcr.io/ethanfel/ltx-desktop-web:latest
```

Then restart the container from the Unraid Docker UI. Your `/data` volume persists across updates.
