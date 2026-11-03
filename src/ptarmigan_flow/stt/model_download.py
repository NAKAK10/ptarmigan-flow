"""On-demand STT model snapshot download helpers."""

from __future__ import annotations

import io
import os
import platform
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from ptarmigan_flow.stt.factory import parse_stt_model
from ptarmigan_flow.stt.haas_artifact import HAAS_Q4_K_ARTIFACT
from ptarmigan_flow.stt.model_families import resolve_runtime_model_id

ProgressCallback = Callable[[float | None, str], None]
SnapshotDownload = Callable[..., object]

_SNAPSHOT_PREFIXES = {"granite", "mlx", "voxtral"}


def _is_macos_arm64() -> bool:
    system = platform.system().strip().lower()
    machine = platform.machine().strip().lower()
    return system == "darwin" and machine in {"arm64", "aarch64"}


def resolve_download_repo_id(token: str) -> str | None:
    """Return the Hugging Face repo id to snapshot-download for an STT token."""
    prefix, model_id = parse_stt_model(token)
    if prefix == "haas":
        return HAAS_Q4_K_ARTIFACT.repo_id
    if prefix not in _SNAPSHOT_PREFIXES:
        return None
    return resolve_runtime_model_id(
        prefix=prefix,
        model_id=model_id,
        macos_arm64=_is_macos_arm64(),
    )


def huggingface_cache_hub_dir() -> Path:
    """Return Hugging Face Hub's local cache directory."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def is_huggingface_repo_downloaded(repo_id: str) -> bool:
    """Return True when the Hugging Face snapshot cache has any real snapshot content."""
    snapshots_dir = (
        huggingface_cache_hub_dir() / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    )
    if not snapshots_dir.is_dir():
        return False
    try:
        for snapshot in snapshots_dir.iterdir():
            if not snapshot.is_dir():
                continue
            try:
                next(snapshot.iterdir())
                return True
            except StopIteration:
                continue
    except OSError:
        return False
    return False


def is_model_downloaded(token: str) -> bool:
    """Return True when a model token is ready without running a snapshot download."""
    prefix, _model_id = parse_stt_model(token)
    if prefix == "haas":
        return HAAS_Q4_K_ARTIFACT.is_cached()
    repo_id = resolve_download_repo_id(token)
    if repo_id is None:
        return True
    return is_huggingface_repo_downloaded(repo_id)


def _progress_tqdm_factory(progress_callback: ProgressCallback) -> type[tqdm]:
    class SnapshotProgress(tqdm):
        def __init__(
            self,
            iterable: Iterable[Any] | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            kwargs.setdefault("file", io.StringIO())
            kwargs.setdefault("leave", False)
            super().__init__(iterable, *args, **kwargs)
            self._progress_message = str(kwargs.get("desc") or "downloading")
            self._emit_progress()

        def __iter__(self) -> Iterator[Any]:
            yield from super().__iter__()

        def update(self, n: int | float = 1) -> bool | None:
            result = super().update(n)
            self._emit_progress()
            return result

        def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
            self._progress_message = str(desc or "downloading")
            super().set_description(desc=desc, refresh=refresh)
            self._emit_progress()

        def _emit_progress(self) -> None:
            total = self.total
            fraction: float | None = None
            if total:
                fraction = max(0.0, min(float(self.n) / float(total), 1.0))
            progress_callback(fraction, self._progress_message)

    return SnapshotProgress


def download_model(
    token: str,
    *,
    progress_callback: ProgressCallback,
    snapshot_download: SnapshotDownload | None = None,
    hf_hub_download: Callable[..., str] | None = None,
) -> None:
    """Download the HF snapshot for an STT token, reporting structured progress."""
    prefix, _model_id = parse_stt_model(token)
    if prefix == "haas":
        _download_haas(progress_callback, hf_hub_download=hf_hub_download)
        return
    repo_id = resolve_download_repo_id(token)
    if repo_id is None:
        progress_callback(1.0, "ready")
        return

    if snapshot_download is None:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        snapshot_download = hf_snapshot_download

    progress_callback(None, "preparing")
    snapshot_download(
        repo_id,
        tqdm_class=_progress_tqdm_factory(progress_callback),
    )
    progress_callback(1.0, "ready")


def _download_haas(
    progress_callback: ProgressCallback,
    *,
    hf_hub_download: Callable[..., str] | None,
) -> None:
    artifact = HAAS_Q4_K_ARTIFACT
    cached = artifact.cached_path()
    force_download = cached.exists()
    if artifact.is_cached():
        progress_callback(None, "verifying")
        try:
            artifact.verify_file(cached)
        except RuntimeError:
            pass  # Re-fetch corrupt cached bytes even if HF considers them present.
        else:
            progress_callback(1.0, "ready")
            return
    if hf_hub_download is None:
        from huggingface_hub import hf_hub_download

    progress_callback(None, "preparing")
    path = hf_hub_download(
        repo_id=artifact.repo_id,
        filename=artifact.filename,
        revision=artifact.revision,
        force_download=force_download,
        tqdm_class=_progress_tqdm_factory(progress_callback),
    )
    progress_callback(None, "verifying")
    artifact.verify_file(path)
    progress_callback(1.0, "ready")
