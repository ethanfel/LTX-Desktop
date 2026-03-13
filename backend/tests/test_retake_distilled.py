"""Integration tests for the retake distilled/full quality toggle."""

from __future__ import annotations

import uuid

import numpy as np
import imageio.v2 as imageio

from api_types import RetakeRequest


class TestRetakeRequestDistilledField:
    def test_defaults_to_true(self):
        req = RetakeRequest(video_path="/tmp/v.mp4", start_time=0, duration=3)
        assert req.distilled is True

    def test_accepts_false(self):
        req = RetakeRequest(video_path="/tmp/v.mp4", start_time=0, duration=3, distilled=False)
        assert req.distilled is False

    def test_accepts_true_explicitly(self):
        req = RetakeRequest(video_path="/tmp/v.mp4", start_time=0, duration=3, distilled=True)
        assert req.distilled is True


class TestRetakeDistilledPipeline:
    def _make_valid_video(self, test_state, *, frames: int = 9, width: int = 64, height: int = 64, fps: int = 24) -> str:
        video_file = test_state.config.outputs_dir / f"retake_valid_{uuid.uuid4().hex[:6]}.mp4"
        writer = imageio.get_writer(str(video_file), fps=fps, codec="libx264", macro_block_size=None)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(frames):
            writer.append_data(frame)
        writer.close()
        return str(video_file)

    def _setup_local(self, test_state, create_fake_model_files):
        create_fake_model_files(include_zit=False)
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.config.force_api_generations = False

    def test_distilled_true_default(self, client, test_state, create_fake_model_files, fake_services):
        self._setup_local(test_state, create_fake_model_files)
        video_path = self._make_valid_video(test_state)

        r = client.post("/api/retake", json={
            "video_path": video_path,
            "start_time": 0,
            "duration": 3,
            "prompt": "test prompt",
        })
        assert r.status_code == 200
        assert r.json()["status"] == "complete"

        call = fake_services.retake_pipeline.generate_calls[-1]
        assert call["distilled"] is True
        assert call["video_guider_params"] is None
        assert call["audio_guider_params"] is None

    def test_distilled_false_passes_guiders(self, client, test_state, create_fake_model_files, fake_services):
        self._setup_local(test_state, create_fake_model_files)
        video_path = self._make_valid_video(test_state)

        r = client.post("/api/retake", json={
            "video_path": video_path,
            "start_time": 0,
            "duration": 3,
            "prompt": "test prompt",
            "distilled": False,
        })
        assert r.status_code == 200
        assert r.json()["status"] == "complete"

        call = fake_services.retake_pipeline.generate_calls[-1]
        assert call["distilled"] is False
        assert call["video_guider_params"] is not None
        assert call["audio_guider_params"] is not None

    def test_distilled_true_explicit(self, client, test_state, create_fake_model_files, fake_services):
        self._setup_local(test_state, create_fake_model_files)
        video_path = self._make_valid_video(test_state)

        r = client.post("/api/retake", json={
            "video_path": video_path,
            "start_time": 0,
            "duration": 3,
            "prompt": "test prompt",
            "distilled": True,
        })
        assert r.status_code == 200
        assert r.json()["status"] == "complete"

        call = fake_services.retake_pipeline.generate_calls[-1]
        assert call["distilled"] is True
