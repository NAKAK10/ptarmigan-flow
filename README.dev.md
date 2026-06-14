# Developer Guide

Local setup, testing, debugging, and build instructions for contributors and maintainers.

## Prerequisites

- Python 3.11 (`>=3.11,<3.12`)
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- macOS arm64 (required for MLX backends and full integration testing)

## Setup

```bash
uv sync --extra dev
```

Installs runtime dependencies plus dev extras: `pytest`, `pyinstaller`, `ruff`.

## Running Tests

```bash
pytest                                         # default — excludes memory_heavy tests
pytest -m memory_heavy                         # only high-memory STT backend tests
pytest tests/test_cli_config_commands.py       # single file
pytest -p no:cacheprovider                     # disable pytest cache
```

Pytest config lives in `pyproject.toml` under `[tool.pytest.ini_options]`:
- `pythonpath = ["src"]`
- Default addopts: `-q -m 'not memory_heavy'`

### Test patterns

**Mocking CLI internals:**
```python
import ptarmigan_flow.presentation.cli.commands as commands
monkeypatch.setattr(commands, "some_function", lambda: ...)
```

**Simulating interactive input:**
```python
def _install_input_sequence(monkeypatch, responses):
    iterator = iter(responses)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(iterator))
    monkeypatch.setattr(commands, "_is_interactive_session", lambda: True)
```

## Linting & Formatting

```bash
uv run ruff check .           # lint (rules: E/F/I/UP/B, line-length 100, py311)
uv run ruff check --fix .     # auto-fix safe issues
uv run ruff format .          # format
```

Config in `pyproject.toml` → `[tool.ruff]`.

> There are ~51 pre-existing errors (E501/I001/UP035) in untouched files. Only fix errors in files you modify.

## Running the CLI Locally

```bash
uv run python -m ptarmigan_flow --help
uv run python -m ptarmigan_flow run                    # start daemon
uv run python -m ptarmigan_flow doctor                 # health check
uv run python -m ptarmigan_flow check-permissions --request
uv run python -m ptarmigan_flow config show
uv run python -m ptarmigan_flow list model
uv run python -m ptarmigan_flow list model --hub-search whisper --backend mlx --limit 5
```

Or install in editable mode and use the short aliases:
```bash
uv pip install -e .
ptarmigan-flow --help
pflow run
```

Config is auto-created at `~/.config/ptarmigan-flow/config.toml` on first run. See `config.example.toml` for all available options.

## Debugging

### Log level

Set in `~/.config/ptarmigan-flow/config.toml`:
```toml
[runtime]
log_level = "DEBUG"   # INFO (default), DEBUG, WARNING
```

Log format: `%(asctime)s %(levelname)s [%(name)s] %(message)s`
Implemented in `src/ptarmigan_flow/logging_setup.py` → `configure_logging(level)`.

### Environment variables

| Variable | Effect |
|---|---|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Raises Hugging Face Hub rate limits for `list model --hub-search` |
| `NO_COLOR` | Disables ANSI color in log output |
| `APP_VERSION` | Overrides version string when building with PyInstaller |

### Daemon timing

Constants in `src/ptarmigan_flow/daemon.py` that are useful to tune when debugging recording or hotkey behaviour:

```python
_HOTKEY_COOLDOWN_SECONDS = 0.25
_RECORDING_STALE_GRACE_SECONDS = 0.5
_MAIN_LOOP_IDLE_SLEEP_SECONDS = 0.2
```

### Frozen app dispatch

In the PyInstaller build, `sys.executable` is the app binary — it cannot accept `-m module`. The function `_dispatch_cli_args(argv)` in `src/ptarmigan_flow/macos_app.py` intercepts subprocess invocations and routes them in-process:

- `-m ptarmigan_flow.cli …` → `ptarmigan_flow.cli.main()`
- `-m ptarmigan_flow.activity_overlay …` → `ptarmigan_flow.activity_overlay.main(argv)`

Detect frozen context with `getattr(sys, "frozen", False)`.

## Building the macOS App Locally (Unsigned)

```bash
export APP_VERSION="0.0.0-dev"
uv run pyinstaller --clean --noconfirm packaging/macos/PtarmiganFlow.spec
open dist/PtarmiganFlow.app
```

The spec at `packaging/macos/PtarmiganFlow.spec` collects native assets (Metal shaders, dylibs, tokenizer data) for `mlx`, `mlx_whisper`, `mlx_audio`, `voxmlx`, `moonshine_voice`, and `mistral_common` via `collect_all()`.

This local build is **ad-hoc signed**, so its `cdhash` changes on every rebuild and macOS stops
honoring previously-granted Accessibility / Input Monitoring permissions (System Settings keeps
showing "ON" but `AXIsProcessTrusted()` returns `False`, so onboarding correctly refuses to
advance). Re-grant for the new build after each rebuild:

```bash
tccutil reset Accessibility com.ptarmiganflow.app
tccutil reset ListenEvent com.ptarmiganflow.app
# then relaunch and re-allow in System Settings (remove the entry with "−" and re-add it)
```

A stable signing identity (see release signing below) avoids this by giving the app a fixed
designated requirement so the grant survives rebuilds.

For a signed + notarized release build, see `docs/release-prep.md` and trigger the `release-macos-app.yml` workflow manually with an existing tag.

### Required GitHub Secrets for release signing

| Secret | Description |
|---|---|
| `APPLE_CERTIFICATE_BASE64` | base64-encoded Developer ID Application `.p12` |
| `APPLE_CERTIFICATE_PASSWORD` | `.p12` password |
| `APPLE_TEAM_ID` | Apple Developer Team ID |
| `APPLE_ID` | Apple ID email for notarization |
| `APPLE_APP_SPECIFIC_PASSWORD` | App-specific password for notarization |

## GitHub Workflows

| File | Trigger | Purpose |
|---|---|---|
| `release-macos-app.yml` | Manual dispatch (tag input) | Build signed/notarized `.app`, create draft release |
| `pages.yml` | Push to `main` (`site/**`) / manual | Transcode video, deploy `site/` to GitHub Pages |
| `update-homebrew-formula-on-release.yml` | Release published / manual | Update SHA256 in Homebrew formula |

There is no CI test or lint workflow. Run `pytest` and `ruff check .` before pushing.

## Project Structure

```
src/ptarmigan_flow/
  presentation/cli/
    commands.py        # all command implementations
    parser.py          # argparse setup
    entrypoint.py      # main() dispatcher
  stt/
    model_catalog.py   # verified presets + HF Hub search
    factory.py         # parse_stt_model(), backend dispatch
    model_families.py  # model ID constants
  daemon.py            # main recording/transcription loop
  config.py            # Pydantic config models, load/write
  macos_app.py         # PyInstaller entry point + onboarding window
  logging_setup.py     # configure_logging()

packaging/macos/
  PtarmiganFlow.spec   # PyInstaller build spec
  entitlements.plist   # hardened-runtime entitlements (audio-input, JIT, etc.)

tests/                 # 30+ test files
docs/
  release-prep.md      # Apple signing setup guide
```
