from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ptarmigan_flow.stt import haas, haas_runtime
from ptarmigan_flow.stt.haas import HaasBackendSettings, HaasSTTBackend


class CountingProcessor:
    def __init__(self):
        self.inputs = []

    def apply(self, text):
        self.inputs.append(text)
        return f"post:{text}"


class FakeSpeech:
    """Only synthetic bytes and WAV inspection; never executes an installed runtime."""

    def __init__(self):
        self.calls = []
        self.paths = []
        self.wavs = []
        self.text = " 日 本 語 hello "
        self.reply = None
        self.version = "speech.cpp 0.9.4, C API 3.7\n"
        self.code = 0
        self.timeout = False
        self.kills = 0

    def popen(self, argv, *, stdout, stderr, **kwargs):
        self.calls.append(argv)
        if argv[1] == "--version":
            output = self.version.encode()
        else:
            path = Path(argv[-1])
            self.paths.append(path)
            assert path.is_file()
            with wave.open(str(path), "rb") as wav:
                self.wavs.append((wav.getparams(), wav.readframes(wav.getnframes())))
            reply = self.reply
            if reply is None:
                reply = json.dumps({"file": str(path), "text": self.text, "stop": "complete"})
            output = reply.encode()
        stdout.write(output)
        stderr.write(b"loading model; timing information")
        speech = self

        class Process:
            returncode = None

            def wait(self, *, timeout):
                if self.returncode is None:
                    if speech.timeout:
                        raise subprocess.TimeoutExpired(argv, timeout)
                    self.returncode = speech.code
                return self.returncode

            def poll(self):
                return self.returncode

            def kill(self):
                speech.kills += 1
                self.returncode = -9

        return Process()


@pytest.fixture
def fixture_model(tmp_path, monkeypatch):
    data = b"small synthetic fixture, not a real GGUF"
    artifact = replace(
        haas.HAAS_Q4_K_ARTIFACT,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    path = tmp_path / "model path;$(literal).gguf"
    path.write_bytes(data)
    monkeypatch.setattr(haas, "HAAS_Q4_K_ARTIFACT", artifact)
    return path


@pytest.fixture
def speech(monkeypatch):
    speech = FakeSpeech()
    monkeypatch.setattr(haas_runtime, "find_speech_executable", lambda: Path("/fake/speech"))
    monkeypatch.setattr(haas_runtime.subprocess, "Popen", speech.popen)
    return speech


def _backend(path, *, processor=None, **kwargs):
    return HaasSTTBackend(
        HaasBackendSettings(trailing_silence_seconds=0, **kwargs),
        model_path=path,
        post_processor=processor,
    )


def _assert_temps_removed(speech):
    assert speech.paths
    assert all(not path.exists() and not path.parent.exists() for path in speech.paths)


def test_preflight_checks_runtime_artifact_and_inference_without_postprocessing(
    fixture_model,
    speech,
) -> None:
    processor = CountingProcessor()
    backend = _backend(fixture_model, processor=processor)
    assert not backend.warm_state().ready
    assert backend.preflight_model() == "speech.cpp-haas"
    assert len(speech.calls) == 2
    backend.preflight_model()
    assert len(speech.calls) == 2
    state = backend.warm_state()
    assert state.ready
    assert not state.warmed
    assert not state.warmup_running
    assert not state.supports_keydown_warmup
    assert state.resource_mode == "subprocess"
    assert state.last_activity_at_monotonic is None
    assert processor.inputs == []
    params, data = speech.wavs[0]
    assert params.framerate == 16000 and params.nframes == 3200
    assert not np.frombuffer(data, dtype="<i2").any()
    _assert_temps_removed(speech)


def test_transcribe_safe_argv_pcm16_normalization_and_single_postprocessing(
    fixture_model,
    speech,
    monkeypatch,
) -> None:
    processor = CountingProcessor()
    backend = _backend(fixture_model, processor=processor)
    monkeypatch.setattr(haas.time, "monotonic", lambda: 123.0)
    audio = np.array([[2, 1], [-2, -1], [0.5, -0.5], [2, -1]], dtype=np.float32)
    assert backend.transcribe(audio, 48000) == "post:日本語 hello"
    assert processor.inputs == ["日本語 hello"]
    assert speech.calls[-1] == [
        "/fake/speech",
        "asr",
        str(fixture_model),
        "--format",
        "json",
        "--device",
        "auto",
        "--language",
        "ja",
        str(speech.paths[-1]),
    ]
    params, data = speech.wavs[-1]
    assert params.nchannels == 1 and params.sampwidth == 2 and params.framerate == 48000
    np.testing.assert_array_equal(np.frombuffer(data, dtype="<i2"), [32767, -32767, 0, 16383])
    assert backend.warm_state().last_activity_at_monotonic == 123.0
    assert not backend.warm_state().warmed
    assert "quantization=Q4_K" in backend.backend_summary()
    assert "warmed=no" in backend.runtime_status()
    assert not backend.supports_realtime_input()
    assert backend.warmup_for_low_latency() is None
    assert backend.maybe_release_idle_resources() is None
    _assert_temps_removed(speech)


@pytest.mark.parametrize(
    "requested, resolved",
    [
        ("cpu", "cpu"),
        ("auto", "auto"),
        ("mps", "auto"),
        ("gpu", "gpu"),
    ],
)
def test_device_reaches_actual_preflight_and_transcription_argv(
    requested,
    resolved,
    fixture_model,
    speech,
) -> None:
    backend = _backend(fixture_model, device=requested)
    backend.transcribe(np.zeros(16), 16000)
    asr_calls = [call for call in speech.calls if call[1] == "asr"]
    assert len(asr_calls) == 2
    for call in asr_calls:
        assert call[call.index("--device") + 1] == resolved
    assert f"device={resolved}" in backend.backend_summary()
    _assert_temps_removed(speech)


@pytest.mark.parametrize("device", ["", "cuda", "metal", "MTL0", "cpu --threads 4"])
def test_unsupported_device_rejected_before_runtime_work(device, fixture_model, speech):
    with pytest.raises(ValueError, match="Haas model.device must be"):
        _backend(fixture_model, device=device)
    assert speech.calls == []


def test_trailing_silence_and_auto_language(fixture_model, speech) -> None:
    backend = HaasSTTBackend(
        HaasBackendSettings(language=" AUTO ", trailing_silence_seconds=0.25),
        model_path=fixture_model,
    )
    backend.transcribe(np.ones(160, dtype=np.float32), 16000)
    params, data = speech.wavs[-1]
    assert params.nframes == 160 + 4000
    assert speech.calls[-1][-2] == "auto"
    np.testing.assert_array_equal(np.frombuffer(data, dtype="<i2")[160:], np.zeros(4000))


@pytest.mark.parametrize("language", ["en", "zh", "", "japanese"])
def test_japanese_only_validation(language, tmp_path) -> None:
    with pytest.raises(ValueError, match="Japanese-only"):
        _backend(tmp_path / "absent.gguf", language=language)


def test_unsupported_model_and_nonfinite_padding_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="supports only"):
        _backend(tmp_path / "x.gguf", model_id="other/model")
    for value in [float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="silence must be finite"):
            HaasSTTBackend(HaasBackendSettings(trailing_silence_seconds=value))


@pytest.mark.parametrize("shape", [(0,), (0, 1)])
def test_no_audio_does_no_runtime_or_model_work(shape, tmp_path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("No-audio requests must not start runtime/model work")

    monkeypatch.setattr(haas_runtime, "find_speech_executable", forbidden)
    backend = _backend(tmp_path / "not-downloaded.gguf", processor=CountingProcessor())
    audio = np.zeros(shape, dtype=np.float32)
    assert backend.transcribe(audio, 16000) == ""
    assert list(backend.transcribe_stream(audio, 16000)) == []
    assert not backend.warm_state().ready


def test_empty_recognized_text_is_valid_and_skips_postprocessing(fixture_model, speech) -> None:
    processor = CountingProcessor()
    backend = _backend(fixture_model, processor=processor)
    speech.text = " \n "
    assert list(backend.transcribe_stream(np.zeros(16), 16000)) == []
    assert processor.inputs == []
    assert backend.warm_state().ready


def test_stream_yields_one_final_update(fixture_model, speech) -> None:
    processor = CountingProcessor()
    backend = _backend(fixture_model, processor=processor)
    assert list(backend.transcribe_stream(np.zeros(16), 16000)) == ["post:日本語 hello"]
    assert processor.inputs == ["日本語 hello"]


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "not JSON",
        "{}\n{}",
        "[]",
        "null",
        '{"text":null,"stop":"complete"}',
        '{"text":"partial","stop":"model_limit"}',
    ],
)
def test_invalid_or_partial_json_fails_and_cleans_temps(reply, fixture_model, speech) -> None:
    backend = _backend(fixture_model)
    speech.reply = reply
    with pytest.raises(RuntimeError, match="invalid|incomplete"):
        backend.transcribe(np.zeros(16), 16000)
    assert not backend.warm_state().ready
    _assert_temps_removed(speech)


@pytest.mark.parametrize(
    "mutation",
    [
        {"text": None},
        {"file": "another.wav"},
        {"stop": "cancelled"},
        {"stop": "model_limit"},
        {"error": "failed"},
    ],
)
def test_valid_json_object_with_invalid_contract_rejected(
    mutation,
    fixture_model,
    speech,
    monkeypatch,
) -> None:
    backend = _backend(fixture_model)
    backend.preflight_model()

    def bad_output(args, **kwargs):
        return json.dumps({"file": args[-1], "text": "partial", "stop": "complete", **mutation})

    monkeypatch.setattr(backend._runtime, "run", bad_output)
    processor = CountingProcessor()
    backend._post_processor = processor
    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        backend.transcribe(np.zeros(16), 16000)
    assert processor.inputs == []


@pytest.mark.parametrize("code", [1, 2, 3])
def test_process_failure_removes_temps_and_resets_ready(code, fixture_model, speech) -> None:
    backend = _backend(fixture_model)
    backend.preflight_model()
    speech.code = code
    with pytest.raises(RuntimeError, match=f"status {code}"):
        backend.transcribe(np.zeros(16), 16000)
    assert not backend.warm_state().ready
    _assert_temps_removed(speech)


def test_inference_timeout_reaps_and_cleans_temp_wav(fixture_model, speech) -> None:
    backend = _backend(fixture_model)
    backend.preflight_model()
    speech.timeout = True
    with pytest.raises(RuntimeError, match="timed out"):
        backend.transcribe(np.zeros(16), 16000)
    assert speech.kills == 1
    assert not backend.warm_state().ready
    _assert_temps_removed(speech)
    speech.timeout = False
    assert backend.transcribe(np.zeros(16), 16000) == "日本語 hello"


def test_old_version_and_corrupt_model_fail_before_probe(fixture_model, speech) -> None:
    backend = _backend(fixture_model)
    speech.version = "speech.cpp 0.9.3, C API 3.7"
    with pytest.raises(RuntimeError, match="too old"):
        backend.preflight_model()
    assert len(speech.calls) == 1 and not speech.paths
    speech.version = "speech.cpp 0.9.4, C API 3.7"
    fixture_model.write_bytes(b"bad model")
    with pytest.raises(RuntimeError, match="wrong size"):
        backend.preflight_model()
    assert len(speech.calls) == 2 and not speech.paths
    assert not backend.warm_state().ready


def test_default_path_uses_exact_offline_cache(
    fixture_model, speech, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-cache"))
    cached = haas.HAAS_Q4_K_ARTIFACT.cached_path()
    cached.parent.mkdir(parents=True)
    cached.symlink_to(fixture_model)
    backend = HaasSTTBackend(HaasBackendSettings())
    backend.preflight_model()
    assert speech.calls[-1][2] == str(cached)


@pytest.mark.parametrize(
    "audio, rate",
    [
        (np.zeros((2, 2, 2)), 16000),
        (np.array([float("nan")]), 16000),
        (np.array([float("inf")]), 16000),
        (np.array([1j]), 16000),
        (np.array(["x"]), 16000),
        (np.ones(16), 0),
        (np.ones(16), -1),
        (np.ones(16), 16000.0),
        (np.ones(16), True),
    ],
)
def test_invalid_audio_does_not_start_runtime(audio, rate, fixture_model, speech) -> None:
    with pytest.raises(ValueError):
        _backend(fixture_model).transcribe(audio, rate)
    assert speech.calls == []


def test_temp_directory_removed_when_wav_write_fails(fixture_model, speech, monkeypatch) -> None:
    backend = _backend(fixture_model)
    paths = []

    def broken_open(path, mode):
        paths.append(Path(path))
        Path(path).write_bytes(b"partial wave")
        raise OSError("disk full")

    monkeypatch.setattr(haas.wave, "open", broken_open)
    with pytest.raises(OSError, match="disk full"):
        backend.preflight_model()
    assert all(not path.parent.exists() for path in paths)
    assert not backend.warm_state().ready


def test_cancel_between_preflight_steps_prevents_probe_and_can_retry(
    fixture_model,
    speech,
    monkeypatch,
) -> None:
    backend = _backend(fixture_model)
    original = backend._runtime.check_version

    def cancel_after_version(**kwargs):
        original(**kwargs)
        backend.cancel_pending_transcription()

    monkeypatch.setattr(backend._runtime, "check_version", cancel_after_version)
    with pytest.raises(haas_runtime.SpeechCppCancelledError):
        backend.preflight_model()
    assert not speech.paths
    monkeypatch.setattr(backend._runtime, "check_version", original)
    assert backend.preflight_model() == "speech.cpp-haas"


def test_close_cancels_inference_and_prevents_future_requests(
    fixture_model,
    speech,
    monkeypatch,
) -> None:
    backend = _backend(fixture_model)
    backend.preflight_model()
    started = threading.Event()
    killed = threading.Event()
    original_popen = speech.popen

    def blocking_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)

        def wait(*, timeout):
            started.set()
            if not killed.wait(timeout=timeout):
                raise subprocess.TimeoutExpired(args[0], timeout)
            return process.returncode

        def kill():
            process.returncode = -9
            killed.set()

        process.wait = wait
        process.kill = kill
        return process

    monkeypatch.setattr(haas_runtime.subprocess, "Popen", blocking_popen)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(backend.transcribe, np.zeros(16), 16000)
        assert started.wait(timeout=1)
        backend.close()
        with pytest.raises(haas_runtime.SpeechCppCancelledError):
            result.result(timeout=1)
    backend.close()
    assert not backend.warm_state().ready
    assert backend.warm_state().last_activity_at_monotonic is None
    with pytest.raises(haas_runtime.SpeechCppCancelledError):
        backend.transcribe(np.zeros(16), 16000)
    _assert_temps_removed(speech)


def test_concurrent_requests_are_serialized_and_use_distinct_wavs(
    fixture_model,
    speech,
    monkeypatch,
) -> None:
    backend = _backend(fixture_model)
    backend.preflight_model()
    first_started = threading.Event()
    second_prepared = threading.Event()
    release_first = threading.Event()
    original_popen = speech.popen
    original_prepare = backend._prepare_audio

    def blocking_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        if len(speech.calls) == 3:
            original_wait = process.wait

            def wait(*, timeout):
                first_started.set()
                if not release_first.wait(timeout=timeout):
                    raise subprocess.TimeoutExpired(args[0], timeout)
                return original_wait(timeout=timeout)

            process.wait = wait
        return process

    def prepare_second(audio, rate):
        prepared = original_prepare(audio, rate)
        second_prepared.set()
        return prepared

    monkeypatch.setattr(haas_runtime.subprocess, "Popen", blocking_popen)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(backend.transcribe, np.zeros(16), 16000)
        assert first_started.wait(timeout=1)
        monkeypatch.setattr(backend, "_prepare_audio", prepare_second)
        second = pool.submit(backend.transcribe, np.zeros(32), 16000)
        try:
            assert second_prepared.wait(timeout=1)
            assert len(speech.calls) == 3  # Second request is waiting on the backend lock.
        finally:
            release_first.set()
        assert first.result(timeout=1) == "日本語 hello"
        assert second.result(timeout=1) == "日本語 hello"
    assert len(speech.calls) == 4  # One version, one probe, two utterances.
    assert len(set(speech.paths)) == 3
    _assert_temps_removed(speech)


@pytest.mark.parametrize("seconds, expected_timeout", [(0.1, 20), (10, 50), (100, 60)])
def test_inference_timeout_scales_but_remains_bounded(
    seconds,
    expected_timeout,
    fixture_model,
    speech,
    monkeypatch,
) -> None:
    backend = _backend(fixture_model)
    backend.preflight_model()
    seen = []
    original = backend._runtime.run

    def record_timeout(args, *, timeout_seconds, generation):
        seen.append(timeout_seconds)
        return original(args, timeout_seconds=timeout_seconds, generation=generation)

    monkeypatch.setattr(backend._runtime, "run", record_timeout)
    backend.transcribe(np.zeros(int(seconds * 16000)), 16000)
    assert seen == [expected_timeout]
