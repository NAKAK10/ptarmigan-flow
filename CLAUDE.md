# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role Assignment

- **Claude**: Management role — requirements, design, planning, review.
- **Codex**: Exploration and implementation — code investigation, changes, and tests.

When a task involves writing code or exploring files, delegate to Codex rather than acting directly.

## Commands

### Setup
```bash
uv sync --extra dev   # installs runtime + dev deps (pytest, pyinstaller, ruff)
```

### Tests
```bash
pytest                                       # default run (excludes memory_heavy)
pytest -m memory_heavy                       # high-memory STT backend tests only
pytest tests/test_cli_config_commands.py     # single file
pytest -p no:cacheprovider                   # disable pytest cache
```

### Lint & Format
```bash
uv run ruff check .         # lint
uv run ruff check --fix .   # auto-fix safe issues
uv run ruff format .        # format
```
~51 pre-existing lint errors (E501/I001/UP035) exist in unmodified files — only fix errors in files you touch.

### Run locally
```bash
uv run python -m ptarmigan_flow run
uv run python -m ptarmigan_flow doctor
uv run python -m ptarmigan_flow --help
```

Or install editable and use the short alias:
```bash
uv pip install -e .
pflow run
```

### Build the macOS app (unsigned)

Use the dev build script — it kills any running instance, resets TCC permissions, builds, and opens the app:
```bash
bash scripts/build-dev.sh
```

Or manually:
```bash
pkill -x "PtarmiganFlow" || true
tccutil reset Accessibility com.ptarmiganflow.app
tccutil reset ListenEvent com.ptarmiganflow.app
export APP_VERSION="0.0.0-dev"
uv run pyinstaller --clean --noconfirm packaging/macos/PtarmiganFlow.spec
open dist/PtarmiganFlow.app
```

After each rebuild, re-grant permissions in System Settings when prompted (ad-hoc signing invalidates the previous TCC grant).

## Architecture

### Two execution modes

**CLI daemon** (`pflow run` / `ptarmigan-flow run`): invokes `ptarmigan_flow.presentation.cli.commands` via the `ptarmigan_flow.cli` alias (which redirects `sys.modules`).

**macOS App** (`PtarmiganFlow.app`): a PyInstaller-frozen binary whose entry point is `macos_app.py`. It spawns a `WKWebView` window (PyObjC / WebKit) and serves the `webui/` frontend inside it. JS → Python messages are dispatched through `web_bridge.py` (`WebBridgeDispatcher`). Because the frozen binary stands in for the Python interpreter, `_dispatch_cli_args()` in `macos_app.py` routes `[sys.executable, "-m", "ptarmigan_flow.cli", ...]` calls in-process.

Detect frozen context with `getattr(sys, "frozen", False)`.

### Package layout

```
src/ptarmigan_flow/
  presentation/cli/      # argparse CLI (commands.py, parser.py, entrypoint.py)
  stt/                   # STT backend abstraction
    factory.py           # parse_stt_model(), backend dispatch
    model_catalog.py     # verified presets + HF Hub search
    runtime_backend.py   # wraps backends with recovery logic
    # backends: moonshine, mlx_whisper, granite_mlx/transformers, voxtral_mlx/transformers, vllm_realtime
  domain/                # pure domain models (transcription_session.py)
  ports/                 # abstract port interfaces (runtime.py)
  application/use_cases/ # application-layer use cases
  webui/                 # HTML/CSS/JS frontend for the macOS WKWebView window
  daemon.py              # hold-to-record / release-to-transcribe main loop
  macos_app.py           # PyInstaller entry point + onboarding window (PyObjC)
  web_bridge.py          # PyObjC-free dispatcher for WKWebView JS→Python calls
  config.py              # Pydantic config models, load/write (~/.config/ptarmigan-flow/config.toml)
  permissions.py         # macOS TCC permission checks and requests
  onboarding_flow.py     # first-launch setup state machine
  hotkey_monitor.py      # global hotkey via pynput
  audio_recorder.py      # sounddevice-based audio capture
  output_injector.py     # pastes transcription text into the active app
  activity_overlay.py    # visual recording indicator (runs as a subprocess)
```

### STT backend selection

Backends are selected via `stt.model` in config (e.g. `"mlx:openai/whisper-large-v3"`, `"moonshine:moonshine/base"`). `parse_stt_model()` in `stt/factory.py` parses the string and dispatches to the correct backend class. `stt/model_catalog.py` holds verified presets; `pflow list model --hub-search` queries Hugging Face.

### macOS-specific constraints

- Python 3.11 only (`>=3.11,<3.12`).
- MLX backends (`mlx_whisper`, `mlx_audio`, `voxmlx`) are Apple Silicon only (arm64 guard in `pyproject.toml`).
- TCC permissions required: Microphone, Accessibility, Input Monitoring.

## CI / Workflows

No automated test or lint CI. Run `pytest` and `ruff check .` before pushing.

| Workflow | Trigger | Purpose |
|---|---|---|
| `release-macos-app.yml` | Manual dispatch | Build signed/notarized `.app`, create draft release |
| `pages.yml` | Push to `main` (`site/**`) | Deploy `site/` to GitHub Pages |
| `update-homebrew-formula-on-release.yml` | Release published | Update SHA256 in Homebrew formula |
