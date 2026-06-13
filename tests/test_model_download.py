from __future__ import annotations

from pathlib import Path

import ptarmigan_flow.stt.model_download as model_download
from ptarmigan_flow.stt.model_families import (
    GRANITE_HF_MODEL_ID,
    GRANITE_MLX_MODEL_ID,
    VOXTRAL_HF_MODEL_ID,
    VOXTRAL_MLX_MODEL_ID,
    WHISPER_HF_MODEL_ID,
    WHISPER_MLX_MODEL_ID,
)


def test_resolve_download_repo_id_resolves_runtime_hf_repos_on_macos_arm64(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_download, "_is_macos_arm64", lambda: True)

    assert (
        model_download.resolve_download_repo_id(f"granite:{GRANITE_HF_MODEL_ID}")
        == GRANITE_MLX_MODEL_ID
    )
    assert (
        model_download.resolve_download_repo_id(f"mlx:{WHISPER_HF_MODEL_ID}")
        == WHISPER_MLX_MODEL_ID
    )
    assert (
        model_download.resolve_download_repo_id(f"voxtral:{VOXTRAL_HF_MODEL_ID}")
        == VOXTRAL_MLX_MODEL_ID
    )


def test_resolve_download_repo_id_returns_none_for_non_snapshot_models() -> None:
    assert model_download.resolve_download_repo_id("moonshine:tiny") is None
    assert model_download.resolve_download_repo_id("vllm:vendor/model") is None


def test_is_model_downloaded_treats_non_snapshot_models_as_ready() -> None:
    assert model_download.is_model_downloaded("moonshine:tiny") is True


def test_is_model_downloaded_checks_huggingface_snapshot_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_download, "_is_macos_arm64", lambda: False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    token = "mlx:vendor/example-model"

    assert model_download.is_model_downloaded(token) is False

    snapshot = (
        tmp_path
        / "hf-home"
        / "hub"
        / "models--vendor--example-model"
        / "snapshots"
        / "abc123"
    )
    snapshot.mkdir(parents=True)
    assert model_download.is_model_downloaded(token) is False

    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    assert model_download.is_model_downloaded(token) is True


def test_download_model_reports_ready_without_snapshot_for_non_hf_model() -> None:
    events: list[tuple[float | None, str]] = []

    def fake_snapshot_download(**_kwargs: object) -> None:
        raise AssertionError("snapshot_download should not be called")

    model_download.download_model(
        "moonshine:tiny",
        progress_callback=lambda fraction, message: events.append((fraction, message)),
        snapshot_download=fake_snapshot_download,
    )

    assert events == [(1.0, "ready")]


def test_download_model_passes_progress_from_injected_snapshot_download(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_download, "_is_macos_arm64", lambda: False)
    events: list[tuple[float | None, str]] = []
    seen: dict[str, object] = {}

    def fake_snapshot_download(repo_id: str, *, tqdm_class: type[object]) -> None:
        seen["repo_id"] = repo_id
        progress = tqdm_class(total=100, desc="Downloading weights")
        progress.update(25)
        progress.update(75)
        progress.close()

    model_download.download_model(
        "mlx:vendor/example-model",
        progress_callback=lambda fraction, message: events.append((fraction, message)),
        snapshot_download=fake_snapshot_download,
    )

    assert seen == {"repo_id": "vendor/example-model"}
    assert (None, "preparing") in events
    assert (0.25, "Downloading weights") in events
    assert (1.0, "Downloading weights") in events
    assert events[-1] == (1.0, "ready")


def test_download_model_reports_indeterminate_when_tqdm_total_is_unknown(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_download, "_is_macos_arm64", lambda: False)
    events: list[tuple[float | None, str]] = []

    def fake_snapshot_download(repo_id: str, *, tqdm_class: type[object]) -> None:
        progress = tqdm_class(total=None, desc=f"Downloading {repo_id}")
        progress.update(1)
        progress.close()

    model_download.download_model(
        "granite:vendor/unknown-size-model",
        progress_callback=lambda fraction, message: events.append((fraction, message)),
        snapshot_download=fake_snapshot_download,
    )

    assert (None, "Downloading vendor/unknown-size-model") in events
    assert events[-1] == (1.0, "ready")
