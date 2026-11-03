from __future__ import annotations

import builtins
import hashlib
import runpy
from dataclasses import replace

import pytest

from ptarmigan_flow.stt import haas_artifact, haas_runtime
from ptarmigan_flow.stt.haas_artifact import (
    HAAS_MODEL_ID,
    HAAS_MODEL_TOKEN,
    HAAS_Q4_K_ARTIFACT,
    huggingface_hub_cache,
)


def test_pinned_artifact_identity() -> None:
    assert HAAS_MODEL_ID == "sakasegawa/Haas-ja-v1.0-GGUF"
    assert HAAS_MODEL_TOKEN == "haas:sakasegawa/Haas-ja-v1.0-GGUF"
    assert HAAS_Q4_K_ARTIFACT.repo_id == HAAS_MODEL_ID
    assert HAAS_Q4_K_ARTIFACT.revision == "25c400baaa9c002df8feffc140804e84228f069b"
    assert HAAS_Q4_K_ARTIFACT.filename == "Haas-616M-ja-v1.0-Q4_K.gguf"
    assert HAAS_Q4_K_ARTIFACT.size_bytes == 351901216
    assert HAAS_Q4_K_ARTIFACT.sha256 == (
        "d24d53f29b75214a933ff482b0de1217e0a800f4118cca39ad6cf03609a89d6f"
    )


@pytest.fixture
def artifact():
    return replace(HAAS_Q4_K_ARTIFACT, size_bytes=4, sha256=hashlib.sha256(b"gguf").hexdigest())


def test_cache_requires_exact_revision_filename_and_size(tmp_path, artifact) -> None:
    expected = artifact.cached_path(tmp_path)
    assert expected == (
        tmp_path
        / "models--sakasegawa--Haas-ja-v1.0-GGUF"
        / "snapshots"
        / artifact.revision
        / artifact.filename
    )
    assert not artifact.is_cached(tmp_path)
    expected.parent.mkdir(parents=True)
    (expected.parent / "README.md").write_text("ready?")
    (expected.parent / "Haas-616M-ja-v1.0-Q8_0.gguf").write_bytes(b"gguf")
    old = expected.parent.parent / "old-revision" / artifact.filename
    old.parent.mkdir()
    old.write_bytes(b"gguf")
    assert not artifact.is_cached(tmp_path)
    expected.write_bytes(b"ggu")
    assert not artifact.is_cached(tmp_path)
    expected.write_bytes(b"gguf")
    assert artifact.is_cached(tmp_path)


def test_hf_snapshot_symlink_keeps_gguf_filename(tmp_path, artifact) -> None:
    blob = tmp_path / "blob-without-extension"
    blob.write_bytes(b"gguf")
    snapshot = artifact.cached_path(tmp_path)
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)
    assert artifact.is_cached(tmp_path)
    assert artifact.verify_file(snapshot) == snapshot
    blob.unlink()
    assert not artifact.is_cached(tmp_path)
    with pytest.raises(RuntimeError, match="Cannot read Haas model"):
        artifact.verify_file(snapshot)


def test_integrity_verification_rejects_missing_truncated_and_corrupt_files(
    tmp_path,
    artifact,
) -> None:
    path = tmp_path / "model.gguf"
    with pytest.raises(RuntimeError, match="Cannot read"):
        artifact.verify_file(path)
    path.write_bytes(b"gg")
    with pytest.raises(RuntimeError, match="wrong size"):
        artifact.verify_file(path)
    path.write_bytes(b"oops")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        artifact.verify_file(path)
    path.write_bytes(b"gguf")
    assert artifact.verify_file(path) == path
    with pytest.raises(ValueError, match="end in .gguf"):
        artifact.verify_file(tmp_path / "hash-named-blob")


def test_cache_read_errors_count_as_missing(tmp_path, artifact, monkeypatch) -> None:
    def denied(_path):
        raise PermissionError("unreadable")

    monkeypatch.setattr(type(tmp_path), "is_file", denied)
    assert not artifact.is_cached(tmp_path)


def test_cache_environment_precedence(tmp_path, monkeypatch) -> None:
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(haas_artifact.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert huggingface_hub_cache() == tmp_path / ".cache/huggingface/hub"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert huggingface_hub_cache() == tmp_path / "xdg/huggingface/hub"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert huggingface_hub_cache() == tmp_path / "hf/hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "legacy"))
    assert huggingface_hub_cache() == tmp_path / "legacy"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit"))
    assert huggingface_hub_cache() == tmp_path / "explicit"


def test_shared_helpers_import_without_model_or_download_dependencies(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "torch", "mlx", "huggingface_hub", "tqdm"}:
            raise AssertionError(f"Heavy dependency imported: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    runpy.run_path(haas_artifact.__file__)
    runpy.run_path(haas_runtime.__file__)
