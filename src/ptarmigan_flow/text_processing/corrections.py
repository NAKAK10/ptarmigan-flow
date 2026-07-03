"""Correction rules and application logic."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from ptarmigan_flow.text_processing.interfaces import TextPostProcessor
from ptarmigan_flow.text_processing.normalizer import normalize_transcript_text


@dataclass(slots=True)
class CompiledRegexRule:
    canonical: str
    pattern: str
    compiled: re.Pattern[str]
    order: int


@dataclass(slots=True)
class CorrectionRuleSet(TextPostProcessor):
    """Immutable correction rules for transcript text."""

    exact_lookup: dict[str, str]
    regex_rules: list[CompiledRegexRule]

    @classmethod
    def empty(cls) -> CorrectionRuleSet:
        return cls(exact_lookup={}, regex_rules=[])

    @property
    def exact_count(self) -> int:
        return len(self.exact_lookup)

    @property
    def regex_count(self) -> int:
        return len(self.regex_rules)

    def apply(self, text: str) -> str:
        normalized = normalize_transcript_text(text)
        if not normalized:
            return ""

        exact_hit = self.exact_lookup.get(normalized)
        if exact_hit is not None:
            return exact_hit

        if not self.regex_rules:
            return normalized

        candidates: list[tuple[int, int, int, str]] = []
        for rule in self.regex_rules:
            for match in rule.compiled.finditer(normalized):
                if match.start() == match.end():
                    continue
                candidates.append((match.start(), match.end(), rule.order, rule.canonical))

        if not candidates:
            return normalized

        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))

        selected: list[tuple[int, int, str]] = []
        cursor = 0
        for start, end, _order, canonical in candidates:
            if start < cursor:
                continue
            selected.append((start, end, canonical))
            cursor = end

        if not selected:
            return normalized

        parts: list[str] = []
        cursor = 0
        for start, end, canonical in selected:
            if cursor < start:
                parts.append(normalized[cursor:start])
            parts.append(canonical)
            cursor = end
        if cursor < len(normalized):
            parts.append(normalized[cursor:])
        return "".join(parts)


def merge_rulesets(
    base: CorrectionRuleSet,
    override: CorrectionRuleSet,
) -> CorrectionRuleSet:
    """Merge two rulesets, with ``override`` taking precedence over ``base``.

    Exact lookups from ``override`` replace matching keys in ``base``. Regex
    rules from ``override`` are placed before those from ``base`` and renumbered
    so ``override`` rules receive smaller ``order`` values and therefore win
    ties during application.
    """
    exact_lookup = {**base.exact_lookup, **override.exact_lookup}
    merged_regex = [
        replace(rule, order=index)
        for index, rule in enumerate([*override.regex_rules, *base.regex_rules])
    ]
    return CorrectionRuleSet(exact_lookup=exact_lookup, regex_rules=merged_regex)
