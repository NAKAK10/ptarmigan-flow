"""Correction dictionary repository."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ptarmigan_flow.text_processing.corrections import CompiledRegexRule, CorrectionRuleSet
from ptarmigan_flow.text_processing.normalizer import normalize_transcript_text


class CorrectionDictionaryError(ValueError):
    """Raised when dictionary TOML is syntactically or structurally invalid."""


@dataclass(slots=True)
class CorrectionLoadWarning:
    """Non-fatal warning emitted while loading a dictionary."""

    message: str


@dataclass(slots=True)
class CorrectionDictionaryLoadResult:
    path: Path
    loaded: bool
    rules: CorrectionRuleSet
    warnings: list[CorrectionLoadWarning]
    disabled_regex_count: int


class TomlCorrectionRepository:
    """Load and validate correction dictionaries from TOML."""

    @staticmethod
    def default_dictionary_path() -> Path:
        return Path("~/.config/ptarmigan-flow/transcription_corrections.toml").expanduser()

    @classmethod
    def resolve_dictionary_path(
        cls,
        dictionary_path: str | None,
        *,
        config_path: Path | None = None,
    ) -> tuple[Path, bool]:
        if dictionary_path and dictionary_path.strip():
            raw = Path(dictionary_path).expanduser()
            if raw.is_absolute() or config_path is None:
                return raw, True
            return (config_path.parent / raw).resolve(), True
        return cls.default_dictionary_path(), False

    def load(
        self,
        path: Path,
        *,
        explicitly_configured: bool,
    ) -> CorrectionDictionaryLoadResult:
        warnings: list[CorrectionLoadWarning] = []

        if not path.exists():
            if explicitly_configured:
                warnings.append(
                    CorrectionLoadWarning(
                        message=(
                            "Correction dictionary not found at configured path: "
                            f"{path}; continuing without rules"
                        )
                    )
                )
            return CorrectionDictionaryLoadResult(
                path=path,
                loaded=False,
                rules=CorrectionRuleSet.empty(),
                warnings=warnings,
                disabled_regex_count=0,
            )

        payload = self._load_toml(path)
        exact_table, regex_table = self._validate_top_level(path, payload)
        rules, build_warnings, disabled_regex_count = self._build_ruleset(
            path, exact_table, regex_table
        )
        warnings.extend(build_warnings)

        return CorrectionDictionaryLoadResult(
            path=path,
            loaded=True,
            rules=rules,
            warnings=warnings,
            disabled_regex_count=disabled_regex_count,
        )

    def build_ruleset_from_tables(
        self,
        exact_table: dict[object, object],
        regex_table: dict[object, object],
        *,
        source_label: str,
    ) -> CorrectionRuleSet:
        """Build a ruleset from in-memory tables (e.g. built-in defaults).

        Reuses the same validation and compilation logic as :meth:`load`.
        Any warnings/disabled-regex counts are discarded; callers supplying
        built-in tables are expected to provide only valid patterns.
        """
        rules, _warnings, _disabled = self._build_ruleset(source_label, exact_table, regex_table)
        return rules

    def _build_ruleset(
        self,
        source: str | Path,
        exact_table: dict[object, object],
        regex_table: dict[object, object],
    ) -> tuple[CorrectionRuleSet, list[CorrectionLoadWarning], int]:
        exact_lookup = self._build_exact_lookup(source, exact_table)
        regex_rules, regex_warnings, disabled_regex_count = self._build_regex_rules(
            source, regex_table
        )
        rules = CorrectionRuleSet(exact_lookup=exact_lookup, regex_rules=regex_rules)
        return rules, regex_warnings, disabled_regex_count

    @staticmethod
    def _load_toml(path: Path) -> dict[object, object]:
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            line = getattr(exc, "lineno", None)
            col = getattr(exc, "colno", None)
            location = ""
            if isinstance(line, int) and isinstance(col, int):
                location = f": line {line}, column {col}"
            raise CorrectionDictionaryError(f"{path}{location}: {exc}") from exc
        except OSError as exc:
            raise CorrectionDictionaryError(f"{path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise CorrectionDictionaryError(f"{path}: top-level TOML must be a table")
        return payload

    @staticmethod
    def _validate_top_level(
        path: Path,
        payload: dict[object, object],
    ) -> tuple[dict[object, object], dict[object, object]]:
        unknown_keys = sorted(str(key) for key in payload if key not in {"exact", "regex"})
        if unknown_keys:
            joined = ", ".join(unknown_keys)
            raise CorrectionDictionaryError(
                f"{path}: unsupported top-level table(s): {joined}; allowed: exact, regex"
            )

        exact_table = payload.get("exact", {})
        regex_table = payload.get("regex", {})
        if not isinstance(exact_table, dict):
            raise CorrectionDictionaryError(f"{path}: [exact] must be a table of arrays")
        if not isinstance(regex_table, dict):
            raise CorrectionDictionaryError(f"{path}: [regex] must be a table of arrays")
        return exact_table, regex_table

    def _build_exact_lookup(
        self,
        source: str | Path,
        exact_table: dict[object, object],
    ) -> dict[str, str]:
        exact_lookup: dict[str, str] = {}
        for canonical, variants in exact_table.items():
            self._validate_key_and_values(source, "exact", canonical, variants)
            canonical_text = normalize_transcript_text(str(canonical))
            assert isinstance(variants, list)
            for index, variant in enumerate(variants):
                assert isinstance(variant, str)
                normalized_variant = normalize_transcript_text(variant)
                if not normalized_variant:
                    raise CorrectionDictionaryError(
                        f"{source}: [exact].{canonical}[{index}] is empty after normalization"
                    )
                prev = exact_lookup.get(normalized_variant)
                if prev is not None and prev != canonical_text:
                    raise CorrectionDictionaryError(
                        f"{source}: exact variant {variant!r} maps to both "
                        f"{prev!r} and {canonical_text!r}"
                    )
                exact_lookup[normalized_variant] = canonical_text
        return exact_lookup

    def _build_regex_rules(
        self,
        source: str | Path,
        regex_table: dict[object, object],
    ) -> tuple[list[CompiledRegexRule], list[CorrectionLoadWarning], int]:
        regex_rules: list[CompiledRegexRule] = []
        warnings: list[CorrectionLoadWarning] = []
        disabled_regex_count = 0
        order = 0

        for canonical, patterns in regex_table.items():
            self._validate_key_and_values(source, "regex", canonical, patterns)
            canonical_text = normalize_transcript_text(str(canonical))
            assert isinstance(patterns, list)
            for index, pattern in enumerate(patterns):
                assert isinstance(pattern, str)
                try:
                    compiled = re.compile(pattern)
                except re.error as exc:
                    disabled_regex_count += 1
                    warnings.append(
                        CorrectionLoadWarning(
                            message=(
                                "Disabled invalid regex rule: "
                                f'canonical="{canonical_text}" pattern_index={index} '
                                f'pattern={pattern!r} error="{exc}"'
                            )
                        )
                    )
                    continue

                if compiled.match("") is not None:
                    disabled_regex_count += 1
                    warnings.append(
                        CorrectionLoadWarning(
                            message=(
                                "Disabled zero-length regex rule: "
                                f'canonical="{canonical_text}" pattern_index={index} '
                                f"pattern={pattern!r}"
                            )
                        )
                    )
                    continue

                regex_rules.append(
                    CompiledRegexRule(
                        canonical=canonical_text,
                        pattern=pattern,
                        compiled=compiled,
                        order=order,
                    )
                )
                order += 1

        return regex_rules, warnings, disabled_regex_count

    @staticmethod
    def _validate_key_and_values(
        source: str | Path,
        table_name: str,
        canonical: object,
        values: object,
    ) -> None:
        if not isinstance(canonical, str):
            raise CorrectionDictionaryError(f"{source}: [{table_name}] keys must be strings")
        if not canonical.strip():
            raise CorrectionDictionaryError(f"{source}: [{table_name}] keys cannot be empty")
        if not isinstance(values, list):
            raise CorrectionDictionaryError(
                f"{source}: [{table_name}].{canonical} must be an array of strings"
            )
        if not values:
            raise CorrectionDictionaryError(f"{source}: [{table_name}].{canonical} cannot be empty")
        for index, item in enumerate(values):
            if not isinstance(item, str):
                raise CorrectionDictionaryError(
                    f"{source}: [{table_name}].{canonical}[{index}] must be a string"
                )
            if not item.strip():
                raise CorrectionDictionaryError(
                    f"{source}: [{table_name}].{canonical}[{index}] cannot be empty"
                )
