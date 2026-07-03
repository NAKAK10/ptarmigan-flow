"""Built-in default correction rules.

These entries are always merged into the runtime-loaded ruleset (with the
user's dictionary taking precedence). They are never written to the user's
dictionary file and never appear in the WebUI editor.
"""

from __future__ import annotations

from ptarmigan_flow.text_processing.corrections import CorrectionRuleSet
from ptarmigan_flow.text_processing.repository import TomlCorrectionRepository

_DEFAULT_SOURCE_LABEL = "<built-in defaults>"

# Whole-utterance replacements. Kept empty: AI-tool names commonly appear
# mid-sentence, so corrections are expressed as regex substitutions below.
DEFAULT_EXACT_TABLE: dict[str, list[str]] = {}

# Substring replacements for common AI-tool names dictated in Japanese.
DEFAULT_REGEX_TABLE: dict[str, list[str]] = {
    "Claude": ["クロード"],
    "ChatGPT": ["チャット\\s*GPT", "チャットジーピーティー"],
    "GPT": ["ジーピーティー"],
    "OpenAI": ["オープン\\s*エーアイ", "オープン\\s*AI"],
    "Gemini": ["ジェミニ", "ジェミナイ"],
    "Copilot": ["コパイロット", "コーパイロット"],
    "Anthropic": ["アンソロピック", "アントロピック"],
}


def build_default_ruleset() -> CorrectionRuleSet:
    """Build the built-in default ruleset.

    Never raises for the built-in tables above; they are curated to be valid.
    """
    return TomlCorrectionRepository().build_ruleset_from_tables(
        DEFAULT_EXACT_TABLE,
        DEFAULT_REGEX_TABLE,
        source_label=_DEFAULT_SOURCE_LABEL,
    )
