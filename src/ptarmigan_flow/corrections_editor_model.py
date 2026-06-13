"""Editable correction dictionary model for the macOS app."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import tomli_w

from ptarmigan_flow.text_processing.repository import (
    CorrectionDictionaryError,
    TomlCorrectionRepository,
)


@dataclass(frozen=True, slots=True)
class CorrectionValidationError:
    """Validation error tied to one editable dictionary row."""

    section: str
    key: str
    index: int | None
    pattern: str
    message: str


@dataclass(slots=True)
class CorrectionsEditorModel:
    """PyObjC-independent editor state for transcription corrections."""

    exact: dict[str, list[str]] = field(default_factory=dict)
    regex: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> CorrectionsEditorModel:
        repository = TomlCorrectionRepository()
        repository.load(path, explicitly_configured=False)
        if not path.exists():
            return cls()

        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        exact = cls._copy_table(payload.get("exact", {}), section="exact")
        regex = cls._copy_table(payload.get("regex", {}), section="regex")
        return cls(exact=exact, regex=regex)

    def add_exact(self, key: str, candidates: list[str]) -> None:
        self.exact[key] = list(candidates)

    def add_regex(self, key: str, patterns: list[str]) -> None:
        self.regex[key] = list(patterns)

    def update_exact(self, old_key: str, key: str, candidates: list[str]) -> None:
        self._replace_entry(self.exact, old_key, key, candidates)

    def update_regex(self, old_key: str, key: str, patterns: list[str]) -> None:
        self._replace_entry(self.regex, old_key, key, patterns)

    def remove_exact(self, key: str) -> None:
        self.exact.pop(key, None)

    def remove_regex(self, key: str) -> None:
        self.regex.pop(key, None)

    def validate(self) -> list[CorrectionValidationError]:
        errors: list[CorrectionValidationError] = []
        errors.extend(self._validate_table("exact", self.exact))
        errors.extend(self._validate_table("regex", self.regex))
        for key, patterns in self.regex.items():
            for index, pattern in enumerate(patterns):
                try:
                    re.compile(pattern)
                except re.error as exc:
                    errors.append(
                        CorrectionValidationError(
                            section="regex",
                            key=key,
                            index=index,
                            pattern=pattern,
                            message=f"invalid regex: {exc}",
                        )
                    )
        return errors

    def save(self, path: Path) -> None:
        errors = self.validate()
        if errors:
            joined = "; ".join(error.message for error in errors)
            raise CorrectionDictionaryError(f"Cannot save correction dictionary: {joined}")

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"exact": self._copy_table(self.exact, section="exact")}
        payload["regex"] = self._copy_table(self.regex, section="regex")
        path.write_text(tomli_w.dumps(payload), encoding="utf-8")

    @staticmethod
    def _replace_entry(
        table: dict[str, list[str]],
        old_key: str,
        key: str,
        values: list[str],
    ) -> None:
        if old_key != key:
            table.pop(old_key, None)
        table[key] = list(values)

    @staticmethod
    def _copy_table(table: object, *, section: str) -> dict[str, list[str]]:
        if not isinstance(table, dict):
            raise CorrectionDictionaryError(f"[{section}] must be a table of arrays")
        copied: dict[str, list[str]] = {}
        for key, values in table.items():
            if not isinstance(key, str):
                raise CorrectionDictionaryError(f"[{section}] keys must be strings")
            if not isinstance(values, list):
                raise CorrectionDictionaryError(f"[{section}].{key} must be an array")
            copied[key] = list(values)
        return copied

    @staticmethod
    def _validate_table(
        section: str,
        table: dict[str, list[str]],
    ) -> list[CorrectionValidationError]:
        errors: list[CorrectionValidationError] = []
        for key, values in table.items():
            if not key.strip():
                errors.append(
                    CorrectionValidationError(
                        section=section,
                        key=key,
                        index=None,
                        pattern="",
                        message=f"[{section}] key cannot be empty",
                    )
                )
            if not values:
                errors.append(
                    CorrectionValidationError(
                        section=section,
                        key=key,
                        index=None,
                        pattern="",
                        message=f"[{section}].{key} must have at least one value",
                    )
                )
            for index, value in enumerate(values):
                if not isinstance(value, str) or not value.strip():
                    errors.append(
                        CorrectionValidationError(
                            section=section,
                            key=key,
                            index=index,
                            pattern=str(value),
                            message=f"[{section}].{key}[{index}] cannot be empty",
                        )
                    )
        return errors
