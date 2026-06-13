from __future__ import annotations

from pathlib import Path

from ptarmigan_flow.corrections_editor_model import CorrectionsEditorModel


def test_load_missing_dictionary_starts_empty(tmp_path: Path) -> None:
    path = tmp_path / "transcription_corrections.toml"

    model = CorrectionsEditorModel.load(path)

    assert model.exact == {}
    assert model.regex == {}


def test_edit_and_remove_rules_then_round_trip_save(tmp_path: Path) -> None:
    path = tmp_path / "transcription_corrections.toml"
    model = CorrectionsEditorModel()

    model.add_exact("Ptarmigan Flow", ["ぷたーみがんふろー", "ぷたーみがんふ"])
    model.add_regex("GPT", ["(?i)じーぴーてぃー"])
    model.update_exact("Ptarmigan Flow", "PtarmiganFlow", ["ぷたーみがんふ"])
    model.update_regex("GPT", "GPT-4", ["(?i)gpt[ -]?4"])
    model.remove_exact("missing")
    model.remove_regex("GPT-4")

    model.save(path)
    loaded = CorrectionsEditorModel.load(path)

    assert loaded.exact == {"PtarmiganFlow": ["ぷたーみがんふ"]}
    assert loaded.regex == {}
    assert "[exact]" in path.read_text(encoding="utf-8")
    assert "[regex]" in path.read_text(encoding="utf-8")


def test_validate_reports_invalid_regex_with_row_context(tmp_path: Path) -> None:
    path = tmp_path / "transcription_corrections.toml"
    model = CorrectionsEditorModel()
    model.add_regex("Ptarmigan Flow", ["ぷたーみがんふ(ろー)?", "(invalid"])

    errors = model.validate()

    assert len(errors) == 1
    assert errors[0].section == "regex"
    assert errors[0].key == "Ptarmigan Flow"
    assert errors[0].index == 1
    assert errors[0].pattern == "(invalid"
    assert "invalid regex" in errors[0].message
    assert not path.exists()


def test_load_preserves_editable_exact_and_regex_tables(tmp_path: Path) -> None:
    path = tmp_path / "transcription_corrections.toml"
    path.write_text(
        """
[exact]
"Ptarmigan Flow" = ["ぷたーみがんふろー", "ぷたーみがんふ"]

[regex]
"GPT" = ["(?i)じーぴーてぃー", "(?i)gpt[ -]?4"]
""".strip(),
        encoding="utf-8",
    )

    model = CorrectionsEditorModel.load(path)

    assert model.exact == {"Ptarmigan Flow": ["ぷたーみがんふろー", "ぷたーみがんふ"]}
    assert model.regex == {"GPT": ["(?i)じーぴーてぃー", "(?i)gpt[ -]?4"]}
