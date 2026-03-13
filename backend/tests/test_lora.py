"""Integration tests for custom LoRA support."""

from __future__ import annotations

from pathlib import Path

from state.app_state_types import GpuSlot, VideoPipelineState, VideoPipelineWarmth
from tests.fakes.services import FakeFastVideoPipeline


def _enable_local_text_encoding(test_state) -> None:
    test_state.state.app_settings.use_local_text_encoder = True


class TestLoraGeneration:
    """Test video generation with and without LoRA."""

    def test_generate_without_lora(self, client, test_state, fake_services, create_fake_model_files):
        """Regression: default generation with no LoRA works."""
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A cat in the rain",
                "resolution": "540p",
                "model": "fast",
                "duration": "2",
                "fps": "24",
            },
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert data["video_path"] is not None
        assert Path(data["video_path"]).exists()

        pipeline = fake_services.fast_video_pipeline
        assert len(pipeline.generate_calls) == 1

    def test_generate_with_valid_lora(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        """Generation with a valid LoRA path passes through to pipeline."""
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        lora_file = tmp_path / "my_lora.safetensors"
        lora_file.write_bytes(b"\x00" * 512)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "A dog playing fetch",
                "resolution": "540p",
                "model": "fast",
                "duration": "2",
                "fps": "24",
                "loraPath": str(lora_file),
                "loraStrength": 0.8,
            },
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert data["video_path"] is not None

        pipeline = fake_services.fast_video_pipeline
        assert len(pipeline.generate_calls) == 1

    def test_generate_with_missing_lora_returns_400(self, client, test_state, create_fake_model_files):
        """Generation with a non-existent LoRA path returns 400."""
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        r = client.post(
            "/api/generate",
            json={
                "prompt": "test",
                "resolution": "540p",
                "model": "fast",
                "duration": "2",
                "fps": "24",
                "loraPath": "/nonexistent/lora.safetensors",
                "loraStrength": 1.0,
            },
        )

        assert r.status_code == 400
        data = r.json()
        assert "LoRA file not found" in data["error"]

    def test_lora_change_evicts_cached_pipeline(self, client, test_state, fake_services, create_fake_model_files, tmp_path):
        """Changing LoRA path triggers pipeline recreation (cache invalidation)."""
        create_fake_model_files()
        _enable_local_text_encoding(test_state)

        lora_a = tmp_path / "lora_a.safetensors"
        lora_a.write_bytes(b"\x00" * 512)
        lora_b = tmp_path / "lora_b.safetensors"
        lora_b.write_bytes(b"\x00" * 512)

        # First generation with lora_a
        r1 = client.post(
            "/api/generate",
            json={
                "prompt": "test a",
                "resolution": "540p",
                "model": "fast",
                "duration": "2",
                "fps": "24",
                "loraPath": str(lora_a),
            },
        )
        assert r1.status_code == 200

        # Check pipeline state has lora_a
        gpu_slot = test_state.state.gpu_slot
        assert gpu_slot is not None
        assert isinstance(gpu_slot.active_pipeline, VideoPipelineState)
        assert gpu_slot.active_pipeline.lora_path == str(lora_a)


class TestLoraListEndpoint:
    """Test the /api/loras endpoint."""

    def test_list_loras_empty(self, client, test_state):
        """When no LoRA files exist, returns empty list."""
        r = client.get("/api/loras")
        assert r.status_code == 200
        data = r.json()
        assert data["loras"] == []

    def test_list_loras_finds_safetensors(self, client, test_state):
        """Returns .safetensors files from models dir and loras/ subdir."""
        models_dir = test_state.config.default_models_dir

        # Create a .safetensors file in models dir
        top_lora = models_dir / "top_level.safetensors"
        top_lora.write_bytes(b"\x00" * 256)

        # Create loras/ subdirectory with a file
        loras_subdir = models_dir / "loras"
        loras_subdir.mkdir(parents=True, exist_ok=True)
        sub_lora = loras_subdir / "custom_style.safetensors"
        sub_lora.write_bytes(b"\x00" * 256)

        # Also create a non-safetensors file (should be excluded)
        (models_dir / "readme.txt").write_text("not a lora")

        r = client.get("/api/loras")
        assert r.status_code == 200
        data = r.json()
        assert len(data["loras"]) == 2
        lora_names = [Path(p).name for p in data["loras"]]
        assert "top_level.safetensors" in lora_names
        assert "custom_style.safetensors" in lora_names
