"""Shared helpers and primitives for LTX video pipeline wrappers."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from api_types import ImageConditioningInput
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8, sync_device

if TYPE_CHECKING:
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.types import LatentState

logger = logging.getLogger(__name__)


def default_tiling_config() -> TilingConfigType:
    from ltx_core.model.video_vae import TilingConfig

    return TilingConfig.default()


def default_guiders() -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
    """Guider params matching upstream LTX_2_3_PARAMS."""
    from ltx_core.components.guiders import MultiModalGuiderParams

    video = MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[28],
    )
    audio = MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[28],
    )
    return video, audio


def video_chunks_number(num_frames: int, tiling_config: TilingConfigType | None) -> int:
    from ltx_core.model.video_vae import get_video_chunks_number

    return int(get_video_chunks_number(num_frames, tiling_config))


def _encode_video_av(
    video: torch.Tensor | Iterator[torch.Tensor],
    audio: AudioOrNone,
    fps: int,
    output_path: str,
    video_chunks_number_value: int,
    *,
    codec: str,
    pix_fmt: str,
    codec_options: dict[str, str] | None = None,
    audio_codec: str = "aac",
) -> None:
    """Core PyAV encoder used by both lossy and lossless wrappers.

    All ``av`` operations go through ``Any`` to satisfy pyright strict mode
    since the ``av`` library ships without type stubs.
    """
    import av  # type: ignore[import-untyped]
    from tqdm import tqdm  # type: ignore[import-untyped]

    if isinstance(video, torch.Tensor):
        video = iter([video])

    first_chunk = next(video)
    _, height, width, _ = first_chunk.shape

    container: Any = av.open(output_path, mode="w")
    try:
        stream: Any = container.add_stream(codec, rate=int(fps))
        stream.width = width
        stream.height = height
        stream.pix_fmt = pix_fmt
        if codec_options is not None:
            stream.options = codec_options

        a_stream: Any = None
        if audio is not None:
            from ltx_core.types import Audio as _Audio

            audio_obj: _Audio = cast(Any, audio)
            a_stream = container.add_stream(audio_codec, rate=audio_obj.sampling_rate)
            a_stream.codec_context.sample_rate = audio_obj.sampling_rate
            a_stream.codec_context.layout = "stereo"
            a_stream.codec_context.time_base = Fraction(1, audio_obj.sampling_rate)

        def _all_chunks(
            first: torch.Tensor, rest: Iterator[torch.Tensor],
        ) -> Iterator[torch.Tensor]:
            yield first
            yield from rest

        for video_chunk in tqdm(_all_chunks(first_chunk, video), total=video_chunks_number_value):
            video_chunk_cpu = video_chunk.to("cpu").numpy()
            for frame_array in video_chunk_cpu:
                frame: Any = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        if audio is not None and a_stream is not None:
            from ltx_core.types import Audio as _Audio

            audio_obj = cast(Any, audio)
            samples: torch.Tensor = audio_obj.waveform
            if samples.ndim == 1:
                samples = samples[:, None]
            if samples.shape[1] != 2 and samples.shape[0] == 2:
                samples = samples.T
            if samples.shape[1] != 2:
                raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")
            if samples.dtype != torch.int16:
                samples = torch.clip(samples, -1.0, 1.0)
                samples = (samples * 32767.0).to(torch.int16)

            frame_in: Any = av.AudioFrame.from_ndarray(
                samples.contiguous().reshape(1, -1).cpu().numpy(),
                format="s16",
                layout="stereo",
            )
            frame_in.sample_rate = audio_obj.sampling_rate

            cc: Any = a_stream.codec_context
            _AudioResampler: Any = getattr(
                getattr(getattr(av, "audio"), "resampler"), "AudioResampler",
            )
            resampler: Any = _AudioResampler(
                format=cc.format or "fltp",
                layout=cc.layout or "stereo",
                rate=cc.sample_rate or frame_in.sample_rate,
            )
            audio_next_pts: int = 0
            rframe: Any
            for rframe in resampler.resample(frame_in):
                if rframe.pts is None:
                    rframe.pts = audio_next_pts
                audio_next_pts += int(rframe.samples)
                rframe.sample_rate = frame_in.sample_rate
                container.mux(a_stream.encode(rframe))
            for packet in a_stream.encode():
                container.mux(packet)
    finally:
        container.close()


def encode_video_output(
    video: torch.Tensor | Iterator[torch.Tensor],
    audio: AudioOrNone,
    fps: int,
    output_path: str,
    video_chunks_number_value: int,
) -> None:
    """Encode video tensor to H264 MP4 at CRF 14 (near-visually-lossless).

    Reimplements the vendored ``encode_video()`` with higher quality settings:
    CRF 14 instead of libx264 default (23), preset ``fast`` for a good
    speed/quality tradeoff.
    """
    _encode_video_av(
        video, audio, fps, output_path, video_chunks_number_value,
        codec="libx264", pix_fmt="yuv420p",
        codec_options={"crf": "14", "preset": "fast"},
        audio_codec="aac",
    )
    logger.info("Video saved to %s (CRF 14)", output_path)


def encode_video_lossless(
    video: torch.Tensor | Iterator[torch.Tensor],
    audio: AudioOrNone,
    fps: int,
    output_path: str,
    video_chunks_number_value: int,
) -> None:
    """Encode video tensor to FFV1 lossless in MKV container.

    Used only for temporary intermediates (blend pipeline). Files are large
    (~10-20x H264) but mathematically lossless.
    """
    # yuv444p preserves full chroma resolution (no subsampling). There is a
    # minor rgb→yuv colorspace conversion, but for temporary intermediates
    # this is negligible and avoids container compatibility issues.
    _encode_video_av(
        video, audio, fps, output_path, video_chunks_number_value,
        codec="ffv1", pix_fmt="yuv444p",
        audio_codec="pcm_s16le",
    )
    logger.info("Lossless video saved to %s (FFV1/MKV)", output_path)


def extract_frames_as_png(video_path: str, output_dir: str) -> None:
    """Extract all frames from a video file as PNGs using ffmpeg."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vsync", "passthrough",
        str(out / "frame_%05d.png"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.error("PNG extraction failed: %s", result.stderr[:500])
        raise RuntimeError(f"Failed to extract PNG frames: {result.stderr[:500]}")
    logger.info("Extracted PNG frames to %s", output_dir)


def video_from_png_frames(png_dir: str, fps: float, output_path: str) -> None:
    """Create a lossless FFV1/MKV video from a directory of PNG frames.

    Used to reconstruct a lossless source video from archived PNG frames
    for retake/blend input, avoiding the quality loss of decoding from H264.
    """
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(Path(png_dir) / "frame_%05d.png"),
        "-c:v", "ffv1",
        "-pix_fmt", "yuv444p",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.error("PNG-to-video failed: %s", result.stderr[:500])
        raise RuntimeError(f"Failed to create video from PNGs: {result.stderr[:500]}")
    logger.info("Created lossless video from PNGs: %s", output_path)


def png_dir_for_video(video_path: str) -> str:
    """Return the PNG frames subdirectory for a video file.

    E.g. ``/outputs/ltx2_video_20260314_abc.mp4`` → ``/outputs/ltx2_video_20260314_abc_frames/``
    """
    p = Path(video_path)
    return str(p.parent / f"{p.stem}_frames")


def maybe_extract_pngs(video_path: str, save_png_frames: bool) -> None:
    """Extract PNG frames from a video if the setting is enabled."""
    if not save_png_frames:
        return
    output_dir = png_dir_for_video(video_path)
    try:
        extract_frames_as_png(video_path, output_dir)
    except Exception:
        logger.warning("Failed to extract PNG frames for %s", video_path, exc_info=True)


class DistilledNativePipeline:
    """Fast native pipeline implementation moved from ltx2_server.py."""

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        device: torch.device | None = None,
        fp8transformer: bool = False,
    ) -> None:
        from ltx_pipelines.utils import ModelLedger
        from ltx_pipelines.utils.helpers import get_device
        from ltx_pipelines.utils.types import PipelineComponents

        if device is None:
            device = get_device()

        self.device = device
        self.dtype = torch.bfloat16

        from ltx_core.quantization import QuantizationPolicy

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=None,
            quantization=QuantizationPolicy.fp8_cast() if fp8transformer and device_supports_fp8(device) else None,
        )
        self.pipeline_components = PipelineComponents(dtype=self.dtype, device=device)

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType | None = None,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        from ltx_pipelines.utils.helpers import (
            cleanup_memory,
            denoise_audio_video,
            image_conditionings_by_replacing_latent,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.samplers import euler_denoising_loop

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        text_encoder = self.model_ledger.text_encoder()
        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p

        sync_device(self.device)
        del text_encoder
        cleanup_memory()

        video_encoder = self.model_ledger.video_encoder()
        transformer = self.model_ledger.transformer()
        sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        def denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: EulerDiffusionStep,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,
                ),
            )

        output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        conditionings = image_conditionings_by_replacing_latent(
            images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
            height=output_shape.height,
            width=output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        video_state, audio_state = denoise_audio_video(
            output_shape=output_shape,
            conditionings=conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=cast(Any, denoising_loop),
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )

        sync_device(self.device)
        del transformer
        del video_encoder
        cleanup_memory()

        decoded_video = vae_decode_video(video_state.latent, self.model_ledger.video_decoder(), tiling_config)
        decoded_audio = vae_decode_audio(
            audio_state.latent,
            self.model_ledger.audio_decoder(),
            self.model_ledger.vocoder(),
        )
        return decoded_video, decoded_audio
