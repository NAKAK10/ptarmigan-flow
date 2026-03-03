from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from moonshine_flow.transcriber import MoonshineTranscriber
from moonshine_flow.transcription_corrections import TranscriptionCorrections


def _install_fake_moonshine(monkeypatch, get_model_for_language, fake_transcriber_cls) -> None:
    moonshine_voice_mod = ModuleType("moonshine_voice")
    moonshine_voice_mod.get_model_for_language = get_model_for_language

    moonshine_voice_transcriber_mod = ModuleType("moonshine_voice.transcriber")
    moonshine_voice_transcriber_mod.Transcriber = fake_transcriber_cls

    monkeypatch.setitem(sys.modules, "moonshine_voice", moonshine_voice_mod)
    monkeypatch.setitem(sys.modules, "moonshine_voice.transcriber", moonshine_voice_transcriber_mod)


def test_normalize_audio_downmixes_stereo() -> None:
    stereo = np.array([[0.5, -0.5], [1.5, -1.5]], dtype=np.float32)
    mono = MoonshineTranscriber._normalize_audio(stereo)

    assert mono.shape == (2,)
    assert np.all(mono <= 1.0)
    assert np.all(mono >= -1.0)


def test_stringify_transcript_handles_lines() -> None:
    transcript = SimpleNamespace(
        lines=[SimpleNamespace(text=" hi "), SimpleNamespace(text="there")]
    )
    assert MoonshineTranscriber._stringify_transcript(transcript) == "hi there"


def test_stringify_transcript_removes_spaces_between_japanese_chars() -> None:
    transcript = SimpleNamespace(
        lines=[SimpleNamespace(text=" そ う い う よ う "), SimpleNamespace(text=" 僕 ら は? ")]
    )
    assert MoonshineTranscriber._stringify_transcript(transcript) == "そういうよう僕らは?"


def test_stringify_transcript_keeps_spaces_for_non_japanese_text() -> None:
    transcript = SimpleNamespace(
        lines=[
            SimpleNamespace(text=" version 2 "),
            SimpleNamespace(text="A B"),
            SimpleNamespace(text="今日は 2026 年"),
        ]
    )
    assert MoonshineTranscriber._stringify_transcript(transcript) == "version 2 A B 今日は 2026 年"


def test_preflight_model_downloads_and_initializes(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeBackendTranscriber:
        def __init__(self, model_path: str, model_arch: object) -> None:
            calls["model_path"] = model_path
            calls["model_arch"] = model_arch

        def transcribe_without_streaming(self, audio_data, sample_rate: int = 16000):
            calls["sample_rate"] = sample_rate
            calls["audio_len"] = len(audio_data)
            return SimpleNamespace(lines=[])

    fake_arch = SimpleNamespace(name="BASE")

    def fake_get_model_for_language(wanted_language: str, wanted_model_arch: object):
        calls["wanted_language"] = wanted_language
        calls["wanted_model_arch"] = wanted_model_arch
        return "/tmp/fake-model", fake_arch

    _install_fake_moonshine(monkeypatch, fake_get_model_for_language, FakeBackendTranscriber)

    transcriber = MoonshineTranscriber(model_size="base", language="ja", device="mps")
    monkeypatch.setattr(transcriber, "_resolve_model_arch", lambda _size: "ARCH")

    backend = transcriber.preflight_model()

    assert backend == "moonshine-voice"
    assert calls["wanted_language"] == "ja"
    assert calls["wanted_model_arch"] == "ARCH"
    assert calls["model_path"] == "/tmp/fake-model"
    assert calls["model_arch"] == fake_arch
    assert calls["sample_rate"] == 16000
    assert calls["audio_len"] == 3200


def test_transcribe_uses_initialized_backend(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeBackendTranscriber:
        def __init__(self, model_path: str, model_arch: object) -> None:
            del model_path, model_arch

        def transcribe_without_streaming(self, audio_data, sample_rate: int = 16000):
            calls["sample_rate"] = sample_rate
            calls["audio_len"] = len(audio_data)
            return SimpleNamespace(lines=[SimpleNamespace(text=" こんにちは ")])

        def close(self) -> None:
            calls["closed"] = True

    def fake_get_model_for_language(wanted_language: str, wanted_model_arch: object):
        del wanted_language, wanted_model_arch
        return "/tmp/fake-model", SimpleNamespace(name="TINY")

    _install_fake_moonshine(monkeypatch, fake_get_model_for_language, FakeBackendTranscriber)

    transcriber = MoonshineTranscriber(model_size="tiny", language="ja", device="mps")
    monkeypatch.setattr(transcriber, "_resolve_model_arch", lambda _size: "ARCH")
    transcriber.preflight_model()

    text = transcriber.transcribe(np.array([0.1, 0.2], dtype=np.float32), sample_rate=16000)

    assert text == "こんにちは"
    assert calls["sample_rate"] == 16000
    assert calls["audio_len"] == 16002

    transcriber.close()
    assert calls["closed"] is True


def test_language_is_used_as_is() -> None:
    transcriber = MoonshineTranscriber(model_size="base", language="ja", device="cpu")
    assert transcriber._resolved_language == "ja"


def test_transcribe_applies_text_corrections(monkeypatch) -> None:
    class FakeBackendTranscriber:
        def __init__(self, model_path: str, model_arch: object) -> None:
            del model_path, model_arch

        def transcribe_without_streaming(self, audio_data, sample_rate: int = 16000):
            del audio_data, sample_rate
            return SimpleNamespace(lines=[SimpleNamespace(text="むーんしゃいんふ")])

    def fake_get_model_for_language(wanted_language: str, wanted_model_arch: object):
        del wanted_language, wanted_model_arch
        return "/tmp/fake-model", SimpleNamespace(name="BASE")

    _install_fake_moonshine(monkeypatch, fake_get_model_for_language, FakeBackendTranscriber)

    corrections = TranscriptionCorrections(
        exact_lookup={"むーんしゃいんふ": "Moonshine Flow"},
        regex_rules=[],
    )
    transcriber = MoonshineTranscriber(
        model_size="base",
        language="ja",
        device="mps",
        post_processor=corrections,
    )
    monkeypatch.setattr(transcriber, "_resolve_model_arch", lambda _size: "ARCH")
    transcriber.preflight_model()

    text = transcriber.transcribe(np.array([0.1, 0.2], dtype=np.float32), sample_rate=16000)
    assert text == "Moonshine Flow"


def test_transcribe_respects_configured_trailing_silence(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeBackendTranscriber:
        def __init__(self, model_path: str, model_arch: object) -> None:
            del model_path, model_arch

        def transcribe_without_streaming(self, audio_data, sample_rate: int = 16000):
            calls["sample_rate"] = sample_rate
            calls["audio_len"] = len(audio_data)
            return SimpleNamespace(lines=[SimpleNamespace(text=" ok ")])

    def fake_get_model_for_language(wanted_language: str, wanted_model_arch: object):
        del wanted_language, wanted_model_arch
        return "/tmp/fake-model", SimpleNamespace(name="BASE")

    _install_fake_moonshine(monkeypatch, fake_get_model_for_language, FakeBackendTranscriber)

    transcriber = MoonshineTranscriber(
        model_size="base",
        language="ja",
        device="mps",
        trailing_silence_seconds=0.25,
    )
    monkeypatch.setattr(transcriber, "_resolve_model_arch", lambda _size: "ARCH")
    transcriber.preflight_model()

    text = transcriber.transcribe(np.array([0.1, 0.2], dtype=np.float32), sample_rate=16000)
    assert text == "ok"
    assert calls["sample_rate"] == 16000
    assert calls["audio_len"] == 4002
