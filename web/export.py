"""Timeline export for web UI — ports the Electron export pipeline to Python.

Three-pass ffmpeg export:
  1. Video-only pass with filter_complex (trim, speed, scale, concat, letterbox, subtitles)
  2. Audio mixdown (extract PCM per source, accumulate, write WAV)
  3. Final mux (video + audio with target codec)
"""

from __future__ import annotations

import array
import logging
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
NUM_CHANNELS = 2
BYTES_PER_SAMPLE = 2


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class ExportClip:
    url: str
    type: str
    startTime: float
    duration: float
    trimStart: float
    speed: float
    reversed: bool
    flipH: bool
    flipV: bool
    opacity: int
    trackIndex: int
    muted: bool
    volume: float


@dataclass
class FlatSegment:
    filePath: str
    type: str
    startTime: float
    duration: float
    trimStart: float
    speed: float
    reversed: bool
    flipH: bool
    flipV: bool
    opacity: int
    muted: bool
    volume: float


@dataclass
class ExportSubtitle:
    text: str
    startTime: float
    endTime: float
    fontSize: int
    fontFamily: str
    fontWeight: str
    color: str
    backgroundColor: str
    position: str
    italic: bool


@dataclass
class Letterbox:
    ratio: float
    color: str
    opacity: float


# ---------------------------------------------------------------------------
# URL → file path
# ---------------------------------------------------------------------------

def url_to_file_path(url: str) -> str:
    """Convert a file:// URL or HTTP serve URL to a local path."""
    if url.startswith("file://"):
        from urllib.parse import unquote
        return unquote(url[7:])
    if url.startswith("/api/files/serve/"):
        from urllib.parse import unquote
        return "/data/" + unquote(url[len("/api/files/serve/"):])
    return url


# ---------------------------------------------------------------------------
# Timeline flattening
# ---------------------------------------------------------------------------

def flatten_timeline(clips: list[ExportClip]) -> list[FlatSegment]:
    """Flatten multi-track clips into sequential segments (highest track wins)."""
    video_clips = [c for c in clips if c.type in ("video", "image")]
    if not video_clips:
        return []

    boundaries: set[float] = {0.0}
    for c in video_clips:
        boundaries.add(c.startTime)
        boundaries.add(c.startTime + c.duration)
    sorted_b = sorted(boundaries)

    segments: list[FlatSegment] = []
    for i in range(len(sorted_b) - 1):
        t0, t1 = sorted_b[i], sorted_b[i + 1]
        seg_dur = t1 - t0
        if seg_dur < 0.001:
            continue

        mid = (t0 + t1) / 2
        active = [
            c for c in video_clips
            if mid >= c.startTime and mid < c.startTime + c.duration
        ]
        active.sort(key=lambda c: c.trackIndex, reverse=True)

        if active:
            c = active[0]
            offset = t0 - c.startTime
            segments.append(FlatSegment(
                filePath=url_to_file_path(c.url),
                type=c.type,
                startTime=t0,
                duration=seg_dur,
                trimStart=c.trimStart + offset * c.speed,
                speed=c.speed,
                reversed=c.reversed,
                flipH=c.flipH,
                flipV=c.flipV,
                opacity=c.opacity,
                muted=c.muted,
                volume=c.volume,
            ))
        else:
            segments.append(FlatSegment(
                filePath="", type="gap", startTime=t0, duration=seg_dur,
                trimStart=0, speed=1, reversed=False, flipH=False, flipV=False,
                opacity=100, muted=True, volume=0,
            ))

    # Merge adjacent segments from same file with contiguous trim
    merged: list[FlatSegment] = []
    for seg in segments:
        prev = merged[-1] if merged else None
        if (prev and prev.filePath == seg.filePath and prev.filePath != ""
                and prev.speed == seg.speed and prev.reversed == seg.reversed
                and prev.flipH == seg.flipH and prev.flipV == seg.flipV
                and prev.opacity == seg.opacity and prev.muted == seg.muted
                and prev.volume == seg.volume
                and abs((prev.trimStart + prev.duration * prev.speed) - seg.trimStart) < 0.01):
            prev.duration += seg.duration
        else:
            merged.append(FlatSegment(
                filePath=seg.filePath, type=seg.type, startTime=seg.startTime,
                duration=seg.duration, trimStart=seg.trimStart, speed=seg.speed,
                reversed=seg.reversed, flipH=seg.flipH, flipV=seg.flipV,
                opacity=seg.opacity, muted=seg.muted, volume=seg.volume,
            ))
    return merged


# ---------------------------------------------------------------------------
# Video filter graph
# ---------------------------------------------------------------------------

def build_video_filter_graph(
    segments: list[FlatSegment],
    width: int,
    height: int,
    fps: int,
    letterbox: Letterbox | None = None,
    subtitles: list[ExportSubtitle] | None = None,
) -> tuple[list[str], str]:
    """Build ffmpeg -filter_complex inputs and script. Returns (inputs, filter_script)."""
    inputs: list[str] = []
    parts: list[str] = []
    idx = 0

    for i, seg in enumerate(segments):
        if seg.type == "gap":
            inputs += ["-f", "lavfi", "-i",
                       f"color=c=black:s={width}x{height}:r={fps}:d={seg.duration:.6f}"]
            parts.append(f"[{idx}:v]setsar=1[v{i}]")
            idx += 1
        elif seg.type == "image":
            inputs += ["-loop", "1", "-framerate", str(fps),
                       "-t", f"{seg.duration:.6f}", "-i", seg.filePath]
            chain = (f"[{idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                     f"pad={width}:{height}:-1:-1:color=black,setsar=1")
            if seg.flipH:
                chain += ",hflip"
            if seg.flipV:
                chain += ",vflip"
            chain += f"[v{i}]"
            parts.append(chain)
            idx += 1
        else:
            trim_end = seg.trimStart + seg.duration * seg.speed
            inputs += ["-i", seg.filePath]
            chain = (f"[{idx}:v]trim=start={seg.trimStart:.6f}:end={trim_end:.6f},"
                     f"setpts=PTS-STARTPTS")
            if seg.speed != 1:
                chain += f",setpts=PTS/{seg.speed:.6f}"
            if seg.reversed:
                chain += ",reverse"
            chain += (f",scale={width}:{height}:force_original_aspect_ratio=decrease,"
                      f"pad={width}:{height}:-1:-1:color=black,setsar=1")
            if seg.flipH:
                chain += ",hflip"
            if seg.flipV:
                chain += ",vflip"
            chain += f"[v{i}]"
            parts.append(chain)
            idx += 1

    concat_inputs = "".join(f"[v{i}]" for i in range(len(segments)))
    parts.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[concatraw]")
    last_label = "fpsout"
    parts.append(f"[concatraw]fps={fps}[{last_label}]")

    # Letterbox
    if letterbox:
        container_ratio = width / height
        target_ratio = letterbox.ratio
        hex_color = letterbox.color.lstrip("#")
        alpha_hex = format(round(letterbox.opacity * 255), "02x")
        color_str = f"0x{hex_color}{alpha_hex}"
        next_label = "lbout"

        if target_ratio >= container_ratio:
            visible_h = round(width / target_ratio)
            bar_h = round((height - visible_h) / 2)
            if bar_h > 0:
                parts.append(
                    f"[{last_label}]drawbox=x=0:y=0:w=iw:h={bar_h}:c={color_str}:t=fill,"
                    f"drawbox=x=0:y=ih-{bar_h}:w=iw:h={bar_h}:c={color_str}:t=fill[{next_label}]"
                )
                last_label = next_label
        else:
            visible_w = round(height * target_ratio)
            bar_w = round((width - visible_w) / 2)
            if bar_w > 0:
                parts.append(
                    f"[{last_label}]drawbox=x=0:y=0:w={bar_w}:h=ih:c={color_str}:t=fill,"
                    f"drawbox=x=iw-{bar_w}:y=0:w={bar_w}:h=ih:c={color_str}:t=fill[{next_label}]"
                )
                last_label = next_label

    # Subtitles
    if subtitles:
        for si, sub in enumerate(subtitles):
            next_label = f"sub{si}"
            escaped = (sub.text
                       .replace("\\", "\\\\\\\\")
                       .replace("'", "'\\\\\\''")
                       .replace(":", "\\:")
                       .replace("%", "%%")
                       .replace("\n", "\\n"))
            font_size = round(sub.fontSize * (height / 1080))
            font_color = sub.color.replace("#", "0x")

            if sub.position == "top":
                y_expr = "20"
            elif sub.position == "center":
                y_expr = "(h-text_h)/2"
            else:
                y_expr = "h-text_h-30"

            box_part = ""
            if sub.backgroundColor and sub.backgroundColor != "transparent":
                bg_hex = sub.backgroundColor.lstrip("#")
                bg_color = f"0x{bg_hex[:6]}" if len(bg_hex) > 6 else f"0x{bg_hex}"
                bg_alpha = f"{int(bg_hex[6:], 16) / 255:.2f}" if len(bg_hex) > 6 else "0.6"
                box_part = f":box=1:boxcolor={bg_color}@{bg_alpha}:boxborderw=8"

            dt = (f"drawtext=text='{escaped}':fontsize={font_size}:fontcolor={font_color}"
                  f":x=(w-text_w)/2:y={y_expr}{box_part}"
                  f":enable='between(t\\,{sub.startTime:.3f}\\,{sub.endTime:.3f})'")
            parts.append(f"[{last_label}]{dt}[{next_label}]")
            last_label = next_label

    if last_label != "outv":
        parts.append(f"[{last_label}]null[outv]")

    return inputs, ";\n".join(parts)


# ---------------------------------------------------------------------------
# Audio mix
# ---------------------------------------------------------------------------

def _file_has_audio(path: str) -> bool:
    """Check if a file has an audio stream."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=10,
    )
    return bool(r.stdout.strip())


def _extract_pcm(
    file_path: str, trim_start: float, trim_end: float, speed: float, reversed: bool,
) -> bytes:
    """Extract raw s16le PCM from a file via ffmpeg."""
    filters = [
        f"atrim=start={trim_start:.6f}:end={trim_end:.6f}",
        "asetpts=PTS-STARTPTS",
    ]
    if speed != 1:
        remaining = speed
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.6f}")
    if reversed:
        filters.append("areverse")

    r = subprocess.run(
        ["ffmpeg", "-i", file_path, "-af", ",".join(filters),
         "-f", "s16le", "-ac", str(NUM_CHANNELS), "-ar", str(SAMPLE_RATE), "pipe:1"],
        capture_output=True, timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"PCM extraction failed for {file_path}")
    return r.stdout


def mix_audio(clips: list[ExportClip], total_duration: float) -> bytes:
    """Mix all clip audio into a single WAV file returned as bytes."""
    import struct

    audio_probe_cache: dict[str, bool] = {}
    sources: list[dict[str, object]] = []

    for c in clips:
        if c.muted or c.volume <= 0:
            continue
        fp = url_to_file_path(c.url)
        if not fp or not Path(fp).is_file():
            continue
        if c.type == "audio":
            sources.append({
                "filePath": fp,
                "trimStart": c.trimStart,
                "trimEnd": c.trimStart + c.duration * c.speed,
                "timelineStart": c.startTime,
                "speed": c.speed,
                "reversed": c.reversed,
                "volume": c.volume,
            })
        elif c.type == "video":
            if fp not in audio_probe_cache:
                audio_probe_cache[fp] = _file_has_audio(fp)
            if not audio_probe_cache[fp]:
                continue
            sources.append({
                "filePath": fp,
                "trimStart": c.trimStart,
                "trimEnd": c.trimStart + c.duration * c.speed,
                "timelineStart": c.startTime,
                "speed": c.speed,
                "reversed": c.reversed,
                "volume": c.volume,
            })

    total_frames = int(total_duration * SAMPLE_RATE + 0.5)
    total_samples = total_frames * NUM_CHANNELS
    mix_buf = array.array("d", [0.0] * total_samples)

    for src in sources:
        try:
            pcm = _extract_pcm(
                str(src["filePath"]),
                float(src["trimStart"]),  # type: ignore[arg-type]
                float(src["trimEnd"]),  # type: ignore[arg-type]
                float(src["speed"]),  # type: ignore[arg-type]
                bool(src["reversed"]),
            )
            start_frame = round(float(src["timelineStart"]) * SAMPLE_RATE)  # type: ignore[arg-type]
            start_sample = start_frame * NUM_CHANNELS
            vol = float(src["volume"])  # type: ignore[arg-type]
            num_pcm_samples = len(pcm) // BYTES_PER_SAMPLE

            for s in range(num_pcm_samples):
                dest = start_sample + s
                if dest < 0 or dest >= total_samples:
                    continue
                value = struct.unpack_from("<h", pcm, s * BYTES_PER_SAMPLE)[0]
                mix_buf[dest] += value * vol
        except Exception:
            logger.warning("Failed to extract audio from %s", src["filePath"], exc_info=True)

    # Convert to Int16LE WAV
    pcm_data = struct.pack(f"<{total_samples}h", *(
        max(-32768, min(32767, round(v))) for v in mix_buf
    ))

    # Build WAV header
    data_size = len(pcm_data)
    wav = bytearray()
    wav += b"RIFF"
    wav += struct.pack("<I", 36 + data_size)
    wav += b"WAVE"
    wav += b"fmt "
    wav += struct.pack("<I", 16)  # chunk size
    wav += struct.pack("<H", 1)  # PCM format
    wav += struct.pack("<H", NUM_CHANNELS)
    wav += struct.pack("<I", SAMPLE_RATE)
    wav += struct.pack("<I", SAMPLE_RATE * NUM_CHANNELS * BYTES_PER_SAMPLE)  # byte rate
    wav += struct.pack("<H", NUM_CHANNELS * BYTES_PER_SAMPLE)  # block align
    wav += struct.pack("<H", BYTES_PER_SAMPLE * 8)  # bits per sample
    wav += b"data"
    wav += struct.pack("<I", data_size)
    wav += pcm_data
    return bytes(wav)


# ---------------------------------------------------------------------------
# Full export
# ---------------------------------------------------------------------------

def run_export(
    clips: list[ExportClip],
    output_path: str,
    codec: str,
    width: int,
    height: int,
    fps: int,
    quality: int,
    letterbox: Letterbox | None = None,
    subtitles: list[ExportSubtitle] | None = None,
) -> dict[str, object]:
    """Run the full three-pass export. Returns {"success": True} or {"error": "..."}."""
    segments = flatten_timeline(clips)
    if not segments:
        return {"error": "No clips to export"}

    # Verify source files
    for seg in segments:
        if seg.filePath and not Path(seg.filePath).is_file():
            return {"error": f"Source file not found: {Path(seg.filePath).name}"}

    tmp_dir = Path(tempfile.gettempdir())
    ts = uuid.uuid4().hex[:8]
    tmp_video = tmp_dir / f"ltx-export-video-{ts}.mkv"
    tmp_audio = tmp_dir / f"ltx-export-audio-{ts}.wav"

    def cleanup() -> None:
        tmp_video.unlink(missing_ok=True)
        tmp_audio.unlink(missing_ok=True)

    try:
        # STEP 1: Video-only
        logger.info("[Export] Step 1: Video-only (%d segments)", len(segments))
        inputs, filter_script = build_video_filter_graph(
            segments, width, height, fps, letterbox, subtitles,
        )
        filter_file = tmp_dir / f"ltx-filter-{ts}.txt"
        filter_file.write_text(filter_script, encoding="utf-8")

        r = subprocess.run(
            ["ffmpeg", "-y", *inputs, "-filter_complex_script", str(filter_file),
             "-map", "[outv]", "-an",
             "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
             str(tmp_video)],
            capture_output=True, text=True, timeout=600,
        )
        filter_file.unlink(missing_ok=True)
        if r.returncode != 0:
            cleanup()
            return {"error": f"Video export failed: {r.stderr[:500]}"}

        # STEP 2: Audio mixdown
        logger.info("[Export] Step 2: Audio mixdown")
        total_duration = max(
            max((s.startTime + s.duration for s in segments), default=0),
            max((c.startTime + c.duration for c in clips), default=0),
        )
        wav_data = mix_audio(clips, total_duration)
        tmp_audio.write_bytes(wav_data)

        # STEP 3: Final mux
        logger.info("[Export] Step 3: Muxing to %s", codec)
        if codec == "h264":
            v_args = ["-c:v", "copy"]  # video already h264 from step 1
            a_args = ["-c:a", "aac", "-b:a", "192k"]
        elif codec == "prores":
            v_args = ["-c:v", "prores_ks", "-profile:v", str(quality or 3),
                      "-pix_fmt", "yuva444p10le"]
            a_args = ["-c:a", "pcm_s16le"]
        elif codec == "vp9":
            v_args = ["-c:v", "libvpx-vp9", "-b:v", f"{quality or 8}M",
                      "-pix_fmt", "yuv420p"]
            a_args = ["-c:a", "libopus", "-b:a", "128k"]
        else:
            cleanup()
            return {"error": f"Unknown codec: {codec}"}

        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_video), "-i", str(tmp_audio),
             "-map", "0:v", "-map", "1:a", *v_args, *a_args,
             "-shortest", output_path],
            capture_output=True, text=True, timeout=600,
        )
        cleanup()
        if r.returncode != 0:
            return {"error": f"Mux failed: {r.stderr[:500]}"}

        logger.info("[Export] Done: %s", output_path)
        return {"success": True}

    except Exception as exc:
        cleanup()
        return {"error": str(exc)}
