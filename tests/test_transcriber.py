from __future__ import annotations

from pathlib import Path

import numpy as np

import ptarmigan_flow.transcriber as transcriber_module
from ptarmigan_flow.transcriber import MoonshineTranscriber


def test_preflight_uses_bundled_tiny_english_model_without_download(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundled_model = tmp_path / "tiny-en"
    bundled_model.mkdir()

    class FakeModelArch:
        TINY = object()

    class FakeTranscript:
        lines: list[object] = []

    class FakeTranscriber:
        def __init__(self, *, model_path: Path, model_arch: object) -> None:
            self.model_path = model_path
            self.model_arch = model_arch

        def transcribe_without_streaming(self, _audio: list[float], *, sample_rate: int) -> object:
            assert sample_rate == 16000
            return FakeTranscript()

    def fail_download(*_args: object, **_kwargs: object) -> tuple[str, object]:
        raise AssertionError("bundled tiny English model should not be downloaded")

    monkeypatch.setattr(
        transcriber_module,
        "_moonshine_model_arch",
        lambda _name: FakeModelArch.TINY,
    )
    monkeypatch.setattr(
        transcriber_module,
        "_bundled_moonshine_model_path",
        lambda _language, _model_size: bundled_model,
    )
    monkeypatch.setattr(
        transcriber_module,
        "_download_moonshine_model",
        fail_download,
    )
    monkeypatch.setattr(transcriber_module, "_moonshine_transcriber_class", lambda: FakeTranscriber)

    backend = MoonshineTranscriber(
        model_size="tiny",
        language="en",
        device="mps",
    )

    assert backend.preflight_model() == "moonshine-voice"
    assert backend._resolved_model_path == str(bundled_model)
    assert backend._resolved_model_arch == "tiny"
    assert isinstance(backend._transcriber, FakeTranscriber)


def test_normalize_audio_converts_stereo_to_mono() -> None:
    audio = np.array([[0.5, -0.5], [1.0, 0.0]], dtype=np.float32)

    normalized = MoonshineTranscriber._normalize_audio(audio)

    np.testing.assert_allclose(normalized, np.array([0.0, 0.5], dtype=np.float32))
