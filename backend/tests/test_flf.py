"""Tests for first-and-last-frame (FLF) image-to-video generation."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from api_types import ImageConditioningInput
from state.app_state_types import GpuSlot, VideoPipelineState, VideoPipelineWarmth
from tests.fakes.services import FakeFastVideoPipeline


def _enable_local_text_encoding(test_state) -> None:
    test_state.state.app_settings.use_local_text_encoder = True


def _make_test_image(tmp_path: Path, name: str = "test.png", color: str = "red") -> str:
    from PIL import Image

    img = Image.new("RGB", (64, 64), color)
    path = tmp_path / name
    img.save(str(path))
    return str(path)


_T2V_JSON = {
    "prompt": "test",
    "resolution": "540p",
    "model": "fast",
    "duration": "2",
    "fps": "24",
}


class TestFLFGeneration:
    def test_flf_both_frames(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        """FLF with both first and last frame images produces two ImageConditioningInput entries."""
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        first_image = _make_test_image(tmp_path, "first.png", "red")
        last_image = _make_test_image(tmp_path, "last.png", "blue")

        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "imagePath": first_image,
                "lastFrameImagePath": last_image,
                "lastFrameStrength": 0.8,
            },
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"

        pipeline = fake_services.fast_video_pipeline
        assert len(pipeline.generate_calls) == 1
        call = pipeline.generate_calls[0]
        images = call["images"]
        assert len(images) == 2

        first = images[0]
        assert first.frame_idx == 0
        assert first.strength == 1.0

        last = images[1]
        num_frames = call["num_frames"]
        assert last.frame_idx == (num_frames - 1) // 8
        assert last.strength == 0.8

    def test_flf_last_frame_only(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        """Users can provide just a last frame without a first frame."""
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        last_image = _make_test_image(tmp_path, "last.png", "green")

        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "lastFrameImagePath": last_image,
            },
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"

        pipeline = fake_services.fast_video_pipeline
        call = pipeline.generate_calls[0]
        images = call["images"]
        assert len(images) == 1

        last = images[0]
        num_frames = call["num_frames"]
        assert last.frame_idx == (num_frames - 1) // 8
        assert last.strength == 1.0  # default strength

    def test_flf_default_strength(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        """Last frame strength defaults to 1.0 when not specified."""
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        last_image = _make_test_image(tmp_path, "last.png")

        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "lastFrameImagePath": last_image,
            },
        )

        assert r.status_code == 200
        pipeline = fake_services.fast_video_pipeline
        call = pipeline.generate_calls[0]
        assert call["images"][0].strength == 1.0

    def test_flf_invalid_last_frame_image(self, client, test_state, create_fake_model_files, tmp_path):
        """Invalid last frame image path returns 400."""
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={
                **_T2V_JSON,
                "lastFrameImagePath": "/no/such/file.png",
            },
        )
        assert r.status_code == 400

    def test_no_flf_t2v(self, client, test_state, fake_services, create_fake_model_files):
        """Standard T2V without FLF has empty images list."""
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post("/api/generate", json=_T2V_JSON)
        assert r.status_code == 200

        pipeline = fake_services.fast_video_pipeline
        call = pipeline.generate_calls[0]
        assert call["images"] == []
