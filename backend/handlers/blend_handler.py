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
        raw_total = context_a_frames + gap_frames + context_b_frames
        # Snap to 8k+1
        total_frames = ((raw_total - 1 + 7) // 8) * 8 + 1
        # Adjust gap frames to absorb the difference
        gap_frames += total_frames - raw_total

        context_a_duration = context_a_frames / fps
        gap_duration = gap_frames / fps
        context_b_duration = context_b_frames / fps

        # Seek positions in source videos
        seek_a = max(0, req.seek_end_a - context_a_duration)
        seek_b = req.seek_start_b

        composite_path = None
        retake_output_path: str | None = None
        generation_started = False

        try:
            self._text.prepare_text_encoding(req.prompt, enhance_prompt=False)
        except RuntimeError as exc:
            raise HTTPError(400, str(exc)) from exc

        generation_id = uuid.uuid4().hex[:8]
        output_path = self.config.outputs_dir / f"blend_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{generation_id}.mp4"

        try:
            # Step 1: Compose a temporary video using ffmpeg
            composite_path = self._compose_video(
                video_a=str(video_a),
                seek_a=seek_a,
                context_a_frames=context_a_frames,
                video_b=str(video_b),
                seek_b=seek_b,
                context_b_frames=context_b_frames,
                gap_frames=gap_frames,
                total_frames=total_frames,
                fps=fps,
            )

            # Step 2: Validate composite meets retake constraints
            self._validate_composite(composite_path)
            logger.info(
                "Blend timing: context_a=%d frames (%.2fs), gap=%d frames (%.2fs), "
                "context_b=%d frames (%.2fs), total=%d frames, "
                "retake window=[%.3f, %.3f]s",
                context_a_frames, context_a_duration,
                gap_frames, gap_duration,
                context_b_frames, context_b_duration,
                total_frames,
                context_a_duration, context_a_duration + gap_duration,
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

            retake_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            retake_output.close()
            retake_output_path = retake_output.name

            pipeline_state.pipeline.generate(
                video_path=composite_path,
                prompt=req.prompt,
                start_time=context_a_duration,
                end_time=context_a_duration + gap_duration,
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
            )

            if self._generation.is_generation_cancelled():
                raise RuntimeError("Generation was cancelled")

            # Step 4: Extract just the gap portion from the retake output
            self._extract_gap(
                retake_output_path,
                start_frame=context_a_frames,
                num_frames=gap_frames,
                start_time=context_a_duration,
                duration=gap_duration,
                output_path=str(output_path),
            )

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

    def _compose_video(
        self,
        *,
        video_a: str,
        seek_a: float,
        context_a_frames: int,
        video_b: str,
        seek_b: float,
        context_b_frames: int,
        gap_frames: int,
        total_frames: int,
        fps: int,
    ) -> str:
        """Create a composite video: end of clip A + frozen last frame as gap + start of clip B.

        Uses frame-accurate trim filters (not -ss/-t input options) to guarantee
        exact frame counts for the retake pipeline's 8k+1 constraint.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        composite_path = tmp.name

        # Use trim filters in the filtergraph for frame-accurate control.
        # Input -ss is only a coarse seek hint; the trim filter does the real work.
        # We overshoot the input read window to ensure enough frames are available.
        read_margin = 2.0  # extra seconds to read from each source
        duration_a = context_a_frames / fps
        duration_b = context_b_frames / fps

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(max(0, seek_a - read_margin)), "-i", video_a,
            "-ss", str(max(0, seek_b - read_margin)), "-i", video_b,
            "-filter_complex",
            (
                # Trim clip A: seek to the right position, convert fps, take exact frame count
                f"[0:v]fps={fps},setpts=PTS-STARTPTS,"
                f"trim=start={read_margin}:end={read_margin + duration_a},setpts=PTS-STARTPTS[va];"
                # Get last frame of clip A and loop for gap
                f"[0:v]fps={fps},setpts=PTS-STARTPTS,"
                f"trim=start={read_margin + duration_a - 1.0/fps},setpts=PTS-STARTPTS[lastframe];"
                f"[lastframe]loop=loop={gap_frames - 1}:size=1:start=0,setpts=N/{fps}/TB[vgap];"
                # Trim clip B: take exact frame count
                f"[1:v]fps={fps},setpts=PTS-STARTPTS,"
                f"trim=start={read_margin}:end={read_margin + duration_b},setpts=PTS-STARTPTS[vb];"
                # Concatenate and limit to exact total frame count
                f"[va][vgap][vb]concat=n=3:v=1:a=0,"
                f"trim=end_frame={total_frames},setpts=PTS-STARTPTS[vout]"
            ),
            "-map", "[vout]",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "10",
            "-r", str(fps),
            "-vsync", "cfr",
            "-an",  # No audio in composite (retake will generate it)
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
        start_time: float,
        duration: float,
        output_path: str,
    ) -> None:
        """Extract the gap portion from the retake output using frame-accurate filters."""
        end_frame = start_frame + num_frames
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"select='between(n\\,{start_frame}\\,{end_frame - 1})',setpts=PTS-STARTPTS",
            "-af", f"atrim=start={start_time}:duration={duration},asetpts=PTS-STARTPTS",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac",
            "-vsync", "cfr",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error("ffmpeg extract failed: %s", result.stderr)
            raise HTTPError(500, f"Failed to extract blend gap: {result.stderr[:500]}")

    def _validate_composite(self, path: str) -> None:
        """Validate that the composite video meets retake constraints."""
        from ltx_core.types import SpatioTemporalScaleFactors
        from ltx_pipelines.utils.media_io import get_videostream_metadata

        fps, num_frames, width, height = get_videostream_metadata(path)
        del fps
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

    def _resolve_seed(self) -> int:
        import time as _time

        settings = self.state.app_settings
        if settings.seed_locked:
            return settings.locked_seed
        return int(_time.time()) % 2147483647
