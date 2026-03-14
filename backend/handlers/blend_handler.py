"""AI Blend handler: compose a temporary video from two clips and retake the gap."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from api_types import BlendRequest, BlendResponse
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from services.ltx_pipeline_common import default_guiders
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class BlendHandler(StateHandlerBase):
    def __init__(
        self,
        state: "AppState",
        lock: RLock,
        config: "RuntimeConfig",
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler

    def run(self, req: BlendRequest) -> BlendResponse:
        video_a = Path(req.video_path_a)
        video_b = Path(req.video_path_b)
        if not video_a.exists():
            raise HTTPError(400, f"Video file not found: {req.video_path_a}")
        if not video_b.exists():
            raise HTTPError(400, f"Video file not found: {req.video_path_b}")
        if req.gap_duration < 0.5:
            raise HTTPError(400, "gap_duration must be at least 0.5 seconds")
        if req.context_a < 0 or req.context_b < 0:
            raise HTTPError(400, "context durations must be non-negative")

        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        # Compute frame counts (must satisfy 8k+1 for the retake pipeline)
        fps = req.fps
        context_a_frames = max(1, round(req.context_a * fps))
        gap_frames = max(1, round(req.gap_duration * fps))
        context_b_frames = max(1, round(req.context_b * fps))

        # Snap frame counts to latent temporal boundaries so the
        # TemporalRegionMask aligns exactly with latent positions.
        #
        # The VAE uses 8x temporal compression with a causal first frame,
        # giving pixel boundaries at frames 0, [1-8], [9-16], [17-24], …
        # For the total to satisfy 8k+1:
        #   context_a = 8a+1  (starts at frame 0, causal)
        #   gap       = 8m    (complete latent tiles, no mask bleed)
        #   context_b = 8c    (complete latent tiles)
        #   total     = (8a+1) + 8m + 8c = 8(a+m+c)+1  ✓
        def _snap_to_8k_plus_1(frames: int) -> int:
            """Round frame count up to nearest 8k+1 (1, 9, 17, 25, …)."""
            if frames <= 1:
                return 1
            k = (frames - 2) // 8 + 1  # ceil((frames-1)/8)
            return k * 8 + 1

        def _snap_to_multiple_of_8(frames: int) -> int:
            """Round frame count up to nearest multiple of 8."""
            return max(8, ((frames + 7) // 8) * 8)

        context_a_frames = _snap_to_8k_plus_1(context_a_frames)
        gap_frames = _snap_to_multiple_of_8(gap_frames)
        context_b_frames = _snap_to_multiple_of_8(context_b_frames)
        total_frames = context_a_frames + gap_frames + context_b_frames

        context_a_duration = context_a_frames / fps
        gap_duration = gap_frames / fps
        context_b_duration = context_b_frames / fps

        # Retake mask times: offset by half a frame inward to avoid floating
        # point boundary comparisons.  TemporalRegionMask uses
        # ``(t_end > start) & (t_start < end)`` — if start_time lands on an
        # exact latent boundary the comparison result is undefined due to
        # IEEE-754 rounding differences between Python and PyTorch.
        half_frame = 0.5 / fps
        retake_start = context_a_duration + half_frame
        retake_end = context_a_duration + gap_duration - half_frame

        # Split the gap evenly between clip A's tail and clip B's head.
        # The composite layout becomes:
        #   [context_a: deeper clip A with motion]
        #   [gap_a: clip A approaching keyframe]
        #   [gap_b: clip B departing keyframe]
        #   [context_b: deeper clip B with motion]
        # This gives the retake model real motion in the gap region instead
        # of a frozen keyframe, and context with actual motion for reference.
        gap_a_frames = gap_frames // 2
        gap_b_frames = gap_frames - gap_a_frames
        gap_a_duration = gap_a_frames / fps

        # Seek positions: shift context AWAY from the keyframe
        # Clip A: read (context_a + gap_a) frames ending at the keyframe
        seek_a = max(0, req.seek_end_a - context_a_duration - gap_a_duration)
        # Clip B: read from keyframe forward (gap_b + context_b) frames
        seek_b = req.seek_start_b

        composite_path = None
        retake_output_path: str | None = None
        lossless_a_path: str | None = None
        lossless_b_path: str | None = None
        generation_started = False

        try:
            self._text.prepare_text_encoding(req.prompt, enhance_prompt=False)
        except RuntimeError as exc:
            raise HTTPError(400, str(exc)) from exc

        generation_id = uuid.uuid4().hex[:8]
        output_path = self.config.outputs_dir / f"blend_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{generation_id}.mp4"

        try:
            # If PNG frames exist for the source videos, reconstruct lossless
            # sources to avoid reading from lossy H264.
            effective_a = str(video_a)
            effective_b = str(video_b)
            if self.state.app_settings.save_png_frames:
                effective_a, lossless_a_path = self._maybe_lossless_from_pngs(str(video_a))
                effective_b, lossless_b_path = self._maybe_lossless_from_pngs(str(video_b))

            # Step 1: Compose a temporary video using ffmpeg
            composite_path = self._compose_video(
                video_a=effective_a,
                seek_a=seek_a,
                a_frames=context_a_frames + gap_a_frames,
                video_b=effective_b,
                seek_b=seek_b,
                b_frames=gap_b_frames + context_b_frames,
                total_frames=total_frames,
                fps=fps,
            )

            # Step 2: Validate composite meets retake constraints
            self._validate_composite(composite_path, expected_fps=fps)
            logger.info(
                "Blend timing: context_a=%d frames (%.2fs), "
                "gap=%d frames (%.2fs) [%d from A + %d from B], "
                "context_b=%d frames (%.2fs), total=%d frames, "
                "retake window=[%.3f, %.3f]s",
                context_a_frames, context_a_duration,
                gap_frames, gap_duration, gap_a_frames, gap_b_frames,
                context_b_frames, context_b_duration,
                total_frames,
                retake_start, retake_end,
            )

            # Step 3: Run retake on the composite
            pipeline_state = self._pipelines.load_retake_pipeline(distilled=req.distilled)
            self._generation.start_generation(generation_id)
            generation_started = True
            self._generation.update_progress("loading_model", 5, 0, 1)
            self._generation.update_progress("inference", 15, 0, 1)

            video_guider_params, audio_guider_params = (
                (None, None) if req.distilled else default_guiders()
            )

            retake_output = tempfile.NamedTemporaryFile(suffix=".mkv", delete=False)
            retake_output.close()
            retake_output_path = retake_output.name

            pipeline_state.pipeline.generate_lossless(
                video_path=composite_path,
                prompt=req.prompt,
                start_time=retake_start,
                end_time=retake_end,
                seed=self._resolve_seed(),
                output_path=retake_output_path,
                negative_prompt=self.config.default_negative_prompt,
                num_inference_steps=40,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                regenerate_video=True,
                regenerate_audio=True,
                enhance_prompt=False,
                distilled=req.distilled,
                fps_override=float(fps),
            )

            if self._generation.is_generation_cancelled():
                raise RuntimeError("Generation was cancelled")

            # Verify retake output metadata
            from ltx_pipelines.utils.media_io import get_videostream_metadata
            rt_fps, rt_frames, rt_w, rt_h = get_videostream_metadata(retake_output_path)
            logger.info(
                "Retake output: fps=%.4f, frames=%d, %dx%d (expected %d frames)",
                rt_fps, rt_frames, rt_w, rt_h, total_frames,
            )

            # Diagnostic: save composite + retake for inspection, compare 3 regions
            import shutil
            debug_composite = self.config.outputs_dir / f"_debug_composite_{generation_id}.mkv"
            debug_retake = self.config.outputs_dir / f"_debug_retake_{generation_id}.mkv"
            shutil.copy2(composite_path, str(debug_composite))
            shutil.copy2(retake_output_path, str(debug_retake))
            logger.info(
                "Debug: composite=%s, retake=%s (ctx_a=0-%d, gap=%d-%d, ctx_b=%d-%d)",
                debug_composite, debug_retake,
                context_a_frames - 1,
                context_a_frames, context_a_frames + gap_frames - 1,
                context_a_frames + gap_frames, total_frames - 1,
            )
            # Compare context A (should be identical = preserved), gap (should differ),
            # and context B (should be identical = preserved)
            ctx_a_frame = context_a_frames // 2
            gap_frame = context_a_frames + gap_frames // 2
            ctx_b_frame = context_a_frames + gap_frames + context_b_frames // 2
            self._log_frame_diff(composite_path, retake_output_path, ctx_a_frame)
            self._log_frame_diff(composite_path, retake_output_path, gap_frame)
            self._log_frame_diff(composite_path, retake_output_path, ctx_b_frame)

            # Step 4: Extract just the gap portion from the retake output
            self._extract_gap(
                retake_output_path,
                start_frame=context_a_frames,
                num_frames=gap_frames,
                fps=fps,
                output_path=str(output_path),
            )

            # Diagnostic: verify extracted output
            ext_fps, ext_frames, ext_w, ext_h = get_videostream_metadata(str(output_path))
            logger.info(
                "Extracted gap: fps=%.4f, frames=%d, %dx%d (expected %d frames)",
                ext_fps, ext_frames, ext_w, ext_h, gap_frames,
            )

            from services.ltx_pipeline_common import maybe_extract_pngs

            maybe_extract_pngs(str(output_path), self.state.app_settings.save_png_frames)
            self._generation.update_progress("complete", 100, 1, 1)
            self._generation.complete_generation(str(output_path))
            return BlendResponse(status="complete", video_path=str(output_path))

        except HTTPError:
            if generation_started:
                self._generation.fail_generation("Blend generation failed")
            raise
        except Exception as exc:
            if generation_started:
                self._generation.fail_generation(str(exc))
            if "cancelled" in str(exc).lower():
                return BlendResponse(status="cancelled")
            raise HTTPError(500, f"Blend generation error: {exc}") from exc
        finally:
            self._text.clear_api_embeddings()
            if composite_path:
                Path(composite_path).unlink(missing_ok=True)
            if retake_output_path:
                Path(retake_output_path).unlink(missing_ok=True)
            if lossless_a_path:
                Path(lossless_a_path).unlink(missing_ok=True)
            if lossless_b_path:
                Path(lossless_b_path).unlink(missing_ok=True)

    def _compose_video(
        self,
        *,
        video_a: str,
        seek_a: float,
        a_frames: int,
        video_b: str,
        seek_b: float,
        b_frames: int,
        total_frames: int,
        fps: int,
    ) -> str:
        """Create a composite video from real footage of both clips.

        Layout: [clip A segment] [clip B segment]
        The caller arranges seek positions so that:
          - clip A segment = context (deeper, with motion) + gap tail (approaching keyframe)
          - clip B segment = gap head (departing keyframe) + context (deeper, with motion)
        No frozen frames — the gap region contains actual video from both clips.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".mkv", delete=False)
        tmp.close()
        composite_path = tmp.name

        read_margin = 2.0
        duration_a = a_frames / fps
        duration_b = b_frames / fps

        ss_a = max(0.0, seek_a - read_margin)
        trim_offset_a = seek_a - ss_a
        ss_b = max(0.0, seek_b - read_margin)
        trim_offset_b = seek_b - ss_b

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ss_a), "-i", video_a,
            "-ss", str(ss_b), "-i", video_b,
            "-filter_complex",
            (
                f"[0:v]fps={fps},setpts=PTS-STARTPTS,"
                f"trim=start={trim_offset_a}:end={trim_offset_a + duration_a},setpts=PTS-STARTPTS[va];"
                f"[1:v]fps={fps},setpts=PTS-STARTPTS,"
                f"trim=start={trim_offset_b}:end={trim_offset_b + duration_b},setpts=PTS-STARTPTS[vb];"
                f"[va][vb]concat=n=2:v=1:a=0,"
                f"trim=end_frame={total_frames},setpts=PTS-STARTPTS[vout]"
            ),
            "-map", "[vout]",
            "-c:v", "ffv1", "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-vsync", "cfr",
            "-an",
            composite_path,
        ]

        logger.info("Composing blend video: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error("ffmpeg compose failed: %s", result.stderr)
            raise HTTPError(500, f"Failed to compose blend video: {result.stderr[:500]}")

        return composite_path

    def _extract_gap(
        self,
        video_path: str,
        start_frame: int,
        num_frames: int,
        fps: int,
        output_path: str,
    ) -> None:
        """Extract the gap portion from the retake output using frame-accurate filters.

        Audio timing is derived from exact frame boundaries to avoid sync drift.
        """
        end_frame = start_frame + num_frames
        # Compute audio boundaries from frame indices for exact sync
        audio_start = start_frame / fps
        audio_duration = num_frames / fps

        # Probe whether the input has an audio stream
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            video_path,
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        has_audio = bool(probe_result.stdout.strip())

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS",
        ]
        if has_audio:
            cmd += [
                "-af", f"atrim=start={audio_start}:duration={audio_duration},asetpts=PTS-STARTPTS",
                "-c:a", "aac",
            ]
        else:
            cmd += ["-an"]
        cmd += [
            "-c:v", "libx264", "-preset", "fast", "-crf", "14",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error("ffmpeg extract failed: %s", result.stderr)
            raise HTTPError(500, f"Failed to extract blend gap: {result.stderr[:500]}")

    @staticmethod
    def _maybe_lossless_from_pngs(video_path: str) -> tuple[str, str | None]:
        """If a PNG frames dir exists for this video, reconstruct a lossless MKV.

        Returns ``(effective_path, temp_path)`` where ``temp_path`` is the
        lossless file to clean up (or ``None`` if no reconstruction was done).
        """
        from services.ltx_pipeline_common import png_dir_for_video, video_from_png_frames
        from ltx_pipelines.utils.media_io import get_videostream_metadata

        png_dir = png_dir_for_video(video_path)
        if not Path(png_dir).is_dir() or not any(Path(png_dir).glob("frame_*.png")):
            return video_path, None

        src_fps, _, _, _ = get_videostream_metadata(video_path)
        tmp = tempfile.NamedTemporaryFile(suffix=".mkv", delete=False)
        tmp.close()
        video_from_png_frames(png_dir, src_fps, tmp.name)
        logger.info("Using lossless PNG source for %s: %s", video_path, tmp.name)
        return tmp.name, tmp.name

    def _validate_composite(self, path: str, expected_fps: int) -> None:
        """Validate that the composite video meets retake constraints."""
        from ltx_core.types import SpatioTemporalScaleFactors
        from ltx_pipelines.utils.media_io import get_videostream_metadata

        fps, num_frames, width, height = get_videostream_metadata(path)
        logger.info(
            "Composite metadata: fps=%.4f (expected %d), frames=%d, %dx%d",
            fps, expected_fps, num_frames, width, height,
        )
        if abs(fps - expected_fps) > 0.5:
            raise HTTPError(
                500,
                f"Composite fps mismatch: got {fps:.4f}, expected {expected_fps}. "
                f"The retake mask would target the wrong frames.",
            )
        scale = SpatioTemporalScaleFactors.default()
        if (num_frames - 1) % scale.time != 0:
            snapped = ((num_frames - 1) // scale.time) * scale.time + 1
            raise HTTPError(
                400,
                f"Composite frame count must satisfy 8k+1. Got {num_frames}; expected {snapped}.",
            )
        if width % 32 != 0 or height % 32 != 0:
            raise HTTPError(
                400,
                f"Source video dimensions must be multiples of 32. Got {width}x{height}.",
            )

    @staticmethod
    def _log_frame_diff(video_a: str, video_b: str, frame_idx: int) -> None:
        """Log pixel-level difference at a specific frame between two videos."""
        try:
            def _read_frame(path: str, idx: int) -> bytes:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", path,
                    "-vf", f"select=eq(n\\,{idx})",
                    "-frames:v", "1",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "pipe:1",
                ]
                r = subprocess.run(cmd, capture_output=True, timeout=10)
                return r.stdout

            ba = _read_frame(video_a, frame_idx)
            bb = _read_frame(video_b, frame_idx)
            if ba and bb:
                import numpy as np  # type: ignore[import-untyped]
                a = np.frombuffer(ba, dtype=np.uint8)
                b = np.frombuffer(bb, dtype=np.uint8)
                n = min(len(a), len(b))
                diff = np.abs(a[:n].astype(np.int16) - b[:n].astype(np.int16))
                logger.info(
                    "Frame %d diff (composite vs retake): mean=%.1f, max=%d, "
                    "identical=%s (bytes: %d vs %d)",
                    frame_idx, float(diff.mean()), int(diff.max()),
                    bool(np.array_equal(a[:n], b[:n])), len(a), len(b),
                )
            else:
                logger.warning(
                    "Frame diff: could not read frame %d (a=%d bytes, b=%d bytes)",
                    frame_idx, len(ba), len(bb),
                )
        except Exception as exc:
            logger.warning("Frame diff probe failed: %s", exc)

    def _resolve_seed(self) -> int:
        import time as _time

        settings = self.state.app_settings
        if settings.seed_locked:
            return settings.locked_seed
        return int(_time.time()) % 2147483647
