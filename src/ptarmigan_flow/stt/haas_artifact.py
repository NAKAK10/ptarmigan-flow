"""Pinned Haas artifact identity and offline cache checks (standard library only)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

HAAS_MODEL_ID = "sakasegawa/Haas-ja-v1.0-GGUF"
HAAS_MODEL_TOKEN = f"haas:{HAAS_MODEL_ID}"


def huggingface_hub_cache() -> Path:
    """Resolve HF's cache overrides without importing its download dependencies."""
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return cache_home / "huggingface" / "hub"


@dataclass(frozen=True, slots=True)
class HaasModelArtifact:
    """One exact file; neither a repository snapshot nor another quantization suffices."""

    repo_id: str
    revision: str
    filename: str
    size_bytes: int
    sha256: str

    def cached_path(self, cache_dir: Path | None = None) -> Path:
        root = huggingface_hub_cache() if cache_dir is None else Path(cache_dir).expanduser()
        # Preserve the .gguf snapshot filename: resolving its HF symlink yields a
        # hash-named blob, which speech's command line does not recognize as a path.
        return (
            root
            / f"models--{self.repo_id.replace('/', '--')}"
            / "snapshots"
            / self.revision
            / self.filename
        )

    def is_cached(self, cache_dir: Path | None = None) -> bool:
        """Cheap, offline completeness check; preflight also verifies the digest."""
        path = self.cached_path(cache_dir)
        try:
            return path.is_file() and path.stat().st_size == self.size_bytes
        except OSError:
            return False

    def verify_file(self, path: str | Path) -> Path:
        """Verify the pinned bytes before passing a local .gguf path to speech."""
        path = Path(path).expanduser().absolute()
        if path.suffix.lower() != ".gguf":
            raise ValueError("Haas model path must end in .gguf")
        try:
            with path.open("rb") as model:
                if os.fstat(model.fileno()).st_size != self.size_bytes:
                    raise RuntimeError(f"Haas model has the wrong size: {path}; download it again")
                digest = hashlib.file_digest(model, "sha256").hexdigest()
        except OSError as exc:
            raise RuntimeError(f"Cannot read Haas model: {path}; download {self.filename}") from exc
        if digest != self.sha256:
            raise RuntimeError(f"Haas model SHA-256 mismatch: {path}; download it again")
        return path


HAAS_Q4_K_ARTIFACT = HaasModelArtifact(
    repo_id=HAAS_MODEL_ID,
    revision="25c400baaa9c002df8feffc140804e84228f069b",
    filename="Haas-616M-ja-v1.0-Q4_K.gguf",
    size_bytes=351901216,
    sha256="d24d53f29b75214a933ff482b0de1217e0a800f4118cca39ad6cf03609a89d6f",
)
