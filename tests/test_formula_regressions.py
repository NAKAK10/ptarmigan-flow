from __future__ import annotations

from pathlib import Path


def test_formula_does_not_reinstall_moved_project_files() -> None:
    formula_path = Path(__file__).resolve().parents[1] / "Formula" / "ptarmigan-flow.rb"
    content = formula_path.read_text(encoding="utf-8")

    assert 'libexec.install buildpath.children' in content

    # Homebrew Pathname#install moves files, so these must not be reinstalled.
    assert 'libexec.install buildpath/"README.md"' not in content
    assert 'libexec.install buildpath/"pyproject.toml"' not in content
    assert 'libexec.install buildpath/"uv.lock"' not in content


def test_formula_sets_setuptools_scm_version_for_stable_builds() -> None:
    formula_path = Path(__file__).resolve().parents[1] / "Formula" / "ptarmigan-flow.rb"
    content = formula_path.read_text(encoding="utf-8")

    assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PTARMIGAN_FLOW" in content


def test_formula_installs_pflow_alias_wrapper() -> None:
    formula_path = Path(__file__).resolve().parents[1] / "Formula" / "ptarmigan-flow.rb"
    content = formula_path.read_text(encoding="utf-8")

    assert "%w[ptarmigan-flow pflow].each do |command_name|" in content
    assert 'shell_output("#{bin}/pflow --help")' in content


def test_formula_has_python_health_probe_and_fallback() -> None:
    formula_path = Path(__file__).resolve().parents[1] / "Formula" / "ptarmigan-flow.rb"
    content = formula_path.read_text(encoding="utf-8")

    assert "def python_is_healthy?(python)" in content
    assert "/opt/homebrew/opt/python@3.11/bin/python3.11" in content
    assert "brew reinstall python@3.11" in content
