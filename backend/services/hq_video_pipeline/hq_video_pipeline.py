"""HQ video pipeline protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

from api_types import ImageConditioningInput

if TYPE_CHECKING:
    import torch


class HQVideoPipeline(Protocol):
    pipeline_kind: ClassVar[Literal["hq"]]

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        distilled_lora_path: str,
        device: torch.device,
        loras: list[object] | None = None,
    ) -> "HQVideoPipeline":
        ...

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        ...

    def warmup(self, output_path: str) -> None:
        ...

    def compile_transformer(self) -> None:
        ...
