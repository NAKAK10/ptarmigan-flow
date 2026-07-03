"""Application service for transcript correction dictionary loading."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ptarmigan_flow.text_processing.corrections import merge_rulesets
from ptarmigan_flow.text_processing.default_corrections import build_default_ruleset
from ptarmigan_flow.text_processing.repository import (
    CorrectionDictionaryLoadResult,
    TomlCorrectionRepository,
)


@dataclass(slots=True)
class CorrectionService:
    """Resolve dictionary paths and load correction rules for runtime."""

    repository: TomlCorrectionRepository

    @classmethod
    def create_default(cls) -> CorrectionService:
        return cls(repository=TomlCorrectionRepository())

    def load_for_config(
        self,
        *,
        config: object,
        config_path: Path,
    ) -> CorrectionDictionaryLoadResult:
        path_value = self._dictionary_path_from_config(config)
        dictionary_path, explicit = self.repository.resolve_dictionary_path(
            path_value,
            config_path=config_path,
        )
        result = self.repository.load(
            dictionary_path,
            explicitly_configured=explicit,
        )
        merged_rules = merge_rulesets(build_default_ruleset(), result.rules)
        return replace(result, rules=merged_rules)

    @staticmethod
    def _dictionary_path_from_config(config: object) -> str | None:
        text_cfg = getattr(config, "text", None)
        path_value = getattr(text_cfg, "dictionary_path", None)
        if not isinstance(path_value, str):
            return None
        stripped = path_value.strip()
        return stripped or None
