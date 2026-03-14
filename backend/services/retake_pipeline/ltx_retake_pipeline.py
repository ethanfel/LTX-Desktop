"""LTX retake pipeline wrapper.

Forked orchestration of the retake pipeline flow from ``ltx_pipelines.retake``
with the following fixes applied in-line (no monkey-patching required):

* ``@torch.no_grad()`` instead of ``@torch.inference_mode()`` — the
  transformer checkpoint uses custom autograd functions incompatible with
  inference-mode tensors.
* Tiled video encoding via ``VideoEncoder.tiled_encode`` — the original
  encodes all frames in a single pass which OOMs on most GPUs.
* Tiled video decoding via ``decode_video(..., tiling_config)`` — the
  original omits the tiling argument.
* Lazy ``denoising_loop`` closure (same pattern as ``DistilledA2VPipeline``)
  so that ``del transformer`` actually frees the model before the decoder
  loads.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.utils.media_io import get_videostream_metadata

import logging

from services.retake_pipeline.retake_pipeline import RetakePipeline
from services.services_utils import sync_device

if TYPE_CHECKING:
    from ltx_core.types import LatentState

logger = logging.getLogger(__name__)


class LTXRetakePipeline:
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        device: torch.device,
        *,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        quantization: QuantizationPolicy | None = None,
    ) -> RetakePipeline:
        return LTXRetakePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            device=device,
            loras=loras or [],
            quantization=quantization,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        device: torch.device,
        *,
        loras: list[LoraPathStrengthAndSDOps],
        quantization: QuantizationPolicy | None,
    ) -> None:
        from ltx_pipelines.utils import ModelLedger
        from ltx_pipelines.utils.types import PipelineComponents

        self.device = device
        self.dtype = torch.bfloat16

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=loras,  # type: ignore[arg-type]  # upstream ModelLedger accepts list at runtime
            quantization=quantization,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    @torch.no_grad()
    def _run(  # noqa: PLR0913, PLR0915
        self,
        video_path: str,
        prompt: str,
        start_time: float,
        end_time: float,
        seed: int,
        *,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = False,
        fps_override: float | None = None,
        mask_ramp_latents: int = 0,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.guiders import MultiModalGuider
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.protocols import DiffusionStepProtocol
        from ltx_core.components.schedulers import LTX2Scheduler
        from ltx_core.conditioning import ConditioningItem
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import AudioLatentShape, VideoPixelShape
        from ltx_pipelines.utils.helpers import (
            cleanup_memory,
            multi_modal_guider_denoising_func,
            noise_audio_state,
            noise_video_state,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.media_io import (
            decode_audio_from_file,
            load_video_conditioning,
        )
        from ltx_pipelines.utils.samplers import euler_denoising_loop

        try:
            from ltx_pipelines.retake import TemporalRegionMask
        except ImportError:
            from ltx_pipelines.retake_pipeline import TemporalRegionMask  # type: ignore[no-redef]

        try:
            from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES as _distilled_sigmas
        except ImportError:
            _distilled_sigmas = [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.04, 0.0]

        if start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")

        effective_seed = int(torch.randint(0, 2**31, (1,)).item()) if seed < 0 else seed
        generator = torch.Generator(device=self.device).manual_seed(effective_seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = self.dtype
        tiling = TilingConfig.default()

        # --- Encode source video (tiled) ---
        video_encoder = self.model_ledger.video_encoder()
        metadata_fps, num_pixel_frames, src_width, src_height = get_videostream_metadata(video_path)
        fps = fps_override if fps_override is not None else metadata_fps
        if fps_override is not None and abs(fps_override - metadata_fps) > 0.01:
            logger.warning(
                "Retake fps_override=%.4f differs from metadata fps=%.4f; using override",
                fps_override, metadata_fps,
            )
        logger.info(
            "Retake _run: fps=%.4f, frames=%d, %dx%d, "
            "start_time=%.4f, end_time=%.4f, regenerate_video=%s",
            fps, num_pixel_frames, src_width, src_height,
            start_time, end_time, regenerate_video,
        )
        output_shape = VideoPixelShape(
            batch=1, frames=num_pixel_frames, width=src_width, height=src_height, fps=fps,
        )
        pixel_video = load_video_conditioning(
            video_path=video_path,
            height=output_shape.height,
            width=output_shape.width,
            frame_cap=output_shape.frames,
            dtype=dtype,
            device=self.device,
        )
        initial_video_latent = video_encoder.tiled_encode(pixel_video, tiling)
        del pixel_video

        video_conditionings: list[ConditioningItem] = [
            TemporalRegionMask(
                start_time=start_time if regenerate_video else 0.0,
                end_time=end_time if regenerate_video else 0.0,
                fps=fps,
            )
        ]
        del video_encoder
        cleanup_memory()

        # --- Encode source audio ---
        initial_audio_latent: torch.Tensor | None = None
        audio_conditionings: list[ConditioningItem] = []
        audio_in = decode_audio_from_file(video_path, self.device)
        audio_encoder = self.model_ledger.audio_encoder()

        if audio_in is not None:
            waveform = audio_in.waveform.squeeze(0)
            waveform_sr = audio_in.sampling_rate
        else:
            waveform, waveform_sr = None, None

        if waveform is not None and waveform_sr is not None:
            waveform_batch = waveform.unsqueeze(0) if waveform.dim() == 2 else waveform
            initial_audio_latent = vae_encode_audio(
                Audio(waveform=waveform_batch.to(dtype), sampling_rate=waveform_sr),
                audio_encoder,
                None,
            )
            expected = AudioLatentShape.from_video_pixel_shape(output_shape).frames
            actual = initial_audio_latent.shape[2]
            if actual > expected:
                initial_audio_latent = initial_audio_latent[:, :, :expected, :]
            elif actual < expected:
                pad = torch.zeros(
                    initial_audio_latent.shape[0],
                    initial_audio_latent.shape[1],
                    expected - actual,
                    initial_audio_latent.shape[3],
                    device=initial_audio_latent.device,
                    dtype=initial_audio_latent.dtype,
                )
                initial_audio_latent = torch.cat([initial_audio_latent, pad], dim=2)

            audio_conditionings = [
                TemporalRegionMask(
                    start_time=start_time if regenerate_audio else 0.0,
                    end_time=end_time if regenerate_audio else 0.0,
                    fps=fps,
                )
            ]

        del audio_encoder
        cleanup_memory()

        # --- Text encoding ---
        text_encoder = self.model_ledger.text_encoder()

        v_context_n: torch.Tensor | None = None
        a_context_n: torch.Tensor | None = None

        if distilled:
            context_p = encode_text(text_encoder, prompts=[prompt])[0]
            v_context_p, a_context_p = context_p
        else:
            context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
            v_context_p, a_context_p = context_p
            v_context_n, a_context_n = context_n

        sync_device(self.device)
        del text_encoder
        cleanup_memory()

        # --- Denoising ---
        transformer = self.model_ledger.transformer()

        raw_sigmas: torch.Tensor = (
            torch.tensor(_distilled_sigmas)
            if distilled
            else LTX2Scheduler().execute(steps=num_inference_steps)  # type: ignore[no-untyped-call]
        )
        sigmas = raw_sigmas.to(dtype=torch.float32, device=self.device)

        if distilled:
            def denoising_loop(
                sigmas: torch.Tensor,
                video_state: LatentState,
                audio_state: LatentState,
                stepper: DiffusionStepProtocol,
            ) -> tuple[LatentState, LatentState]:
                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=simple_denoising_func(
                        video_context=v_context_p,
                        audio_context=a_context_p,
                        transformer=transformer,
                    ),
                )
        else:
            assert video_guider_params is not None, "video_guider_params required for non-distilled"
            assert audio_guider_params is not None, "audio_guider_params required for non-distilled"
            assert v_context_n is not None, "v_context_n required for non-distilled"
            assert a_context_n is not None, "a_context_n required for non-distilled"
            video_guider = MultiModalGuider(params=video_guider_params, negative_context=v_context_n)
            audio_guider = MultiModalGuider(params=audio_guider_params, negative_context=a_context_n)

            def denoising_loop(
                sigmas: torch.Tensor,
                video_state: LatentState,
                audio_state: LatentState,
                stepper: DiffusionStepProtocol,
            ) -> tuple[LatentState, LatentState]:
                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=multi_modal_guider_denoising_func(
                        video_guider,
                        audio_guider,
                        v_context=v_context_p,
                        a_context=a_context_p,
                        transformer=transformer,
                    ),
                )

        video_state, video_tools = noise_video_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=video_conditionings,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            initial_latent=initial_video_latent,
        )
        # Diagnostic: log denoise_mask statistics to debug frozen blend output
        mask = video_state.denoise_mask
        mask_sum = mask.sum().item()
        mask_numel = mask.numel()
        logger.info(
            "Retake denoise_mask: shape=%s, sum=%.0f/%.0f (%.1f%% regenerated), "
            "min=%.4f, max=%.4f",
            list(mask.shape), mask_sum, mask_numel,
            100.0 * mask_sum / max(mask_numel, 1),
            mask.min().item(), mask.max().item(),
        )

        # Apply soft ramp at mask boundaries for smoother transitions.
        # Patches are ordered temporal-outer, spatial-inner:
        #   [t0_s0, t0_s1, …, t0_sK, t1_s0, …]
        # We find the boundary between mask=0 and mask=1 and ramp over
        # `mask_ramp_latents` temporal positions on each side.
        if mask_ramp_latents > 0 and mask.shape[1] > 0:
            from ltx_core.types import VideoLatentShape
            lat_shape = VideoLatentShape.from_pixel_shape(
                shape=output_shape,
                latent_channels=self.pipeline_components.video_latent_channels,
                scale_factors=self.pipeline_components.video_scale_factors,
            )
            n_temporal = lat_shape.frames
            n_spatial = mask.shape[1] // n_temporal if n_temporal > 0 else 0
            if n_spatial > 0:
                # Reshape to [B, T, S, 1] for easy temporal indexing
                mask_4d = mask.view(1, n_temporal, n_spatial, 1)
                # Find first and last temporal positions with mask=1
                temporal_mask = mask_4d[0, :, 0, 0]  # [T] — same for all spatial
                ones = (temporal_mask > 0.5).nonzero(as_tuple=True)[0]
                if len(ones) > 0:
                    first_one = int(ones[0].item())
                    last_one = int(ones[-1].item())
                    # Ramp up: positions [first_one - ramp, first_one) go from 0→1
                    for i in range(mask_ramp_latents):
                        t = first_one - mask_ramp_latents + i
                        if 0 <= t < n_temporal:
                            val = (i + 1) / (mask_ramp_latents + 1)
                            mask_4d[0, t, :, 0] = val
                    # Ramp down: positions (last_one, last_one + ramp] go from 1→0
                    for i in range(mask_ramp_latents):
                        t = last_one + 1 + i
                        if 0 <= t < n_temporal:
                            val = 1.0 - (i + 1) / (mask_ramp_latents + 1)
                            mask_4d[0, t, :, 0] = val
                    # Write back
                    mask.copy_(mask_4d.view_as(mask))
                    logger.info(
                        "Applied mask ramp (%d latent positions): min=%.4f, max=%.4f",
                        mask_ramp_latents, mask.min().item(), mask.max().item(),
                    )
        audio_state, audio_tools = noise_audio_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=audio_conditionings,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            initial_latent=initial_audio_latent,
        )

        video_state, audio_state = denoising_loop(sigmas, video_state, audio_state, stepper)

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        sync_device(self.device)
        del transformer, denoising_loop
        cleanup_memory()

        # --- Decode audio first (eager, small) then free those models ---
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.model_ledger.audio_decoder(), self.model_ledger.vocoder(),
        )
        del audio_state
        cleanup_memory()

        # --- Decode video (lazy generator, tiled) ---
        decoded_video = vae_decode_video(
            video_state.latent, self.model_ledger.video_decoder(), tiling, generator,
        )

        return decoded_video, decoded_audio

    @torch.no_grad()
    def generate(
        self,
        *,
        video_path: str,
        prompt: str,
        start_time: float,
        end_time: float,
        seed: int,
        output_path: str,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = True,
    ) -> None:
        fps, num_frames, _, _ = get_videostream_metadata(video_path)
        video_iter, audio = self._run(
            video_path=video_path,
            prompt=prompt,
            start_time=start_time,
            end_time=end_time,
            seed=seed,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            regenerate_video=regenerate_video,
            regenerate_audio=regenerate_audio,
            enhance_prompt=enhance_prompt,
            distilled=distilled,
        )
        audio_out: Audio | None = audio
        tiling_config = TilingConfig.default()
        video_chunks = get_video_chunks_number(num_frames, tiling_config)
        from services.ltx_pipeline_common import encode_video_output

        encode_video_output(
            video=video_iter,
            audio=audio_out,
            fps=int(fps),
            output_path=output_path,
            video_chunks_number_value=int(video_chunks),
        )

    @torch.no_grad()
    def generate_lossless(
        self,
        *,
        video_path: str,
        prompt: str,
        start_time: float,
        end_time: float,
        seed: int,
        output_path: str,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = True,
        fps_override: float | None = None,
        mask_ramp_latents: int = 0,
    ) -> None:
        metadata_fps, num_frames, _, _ = get_videostream_metadata(video_path)
        effective_fps = fps_override if fps_override is not None else metadata_fps
        video_iter, audio = self._run(
            video_path=video_path,
            prompt=prompt,
            start_time=start_time,
            end_time=end_time,
            seed=seed,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            regenerate_video=regenerate_video,
            regenerate_audio=regenerate_audio,
            enhance_prompt=enhance_prompt,
            distilled=distilled,
            fps_override=fps_override,
            mask_ramp_latents=mask_ramp_latents,
        )
        audio_out: Audio | None = audio
        tiling_config = TilingConfig.default()
        video_chunks = get_video_chunks_number(num_frames, tiling_config)
        from services.ltx_pipeline_common import encode_video_lossless

        encode_video_lossless(
            video=video_iter,
            audio=audio_out,
            fps=int(effective_fps),
            output_path=output_path,
            video_chunks_number_value=int(video_chunks),
        )
