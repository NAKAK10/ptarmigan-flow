# README for LLMs and Agents

If you are an LLM/agent working with this repository, read this file first.
It is intentionally detailed so you can answer most operational questions without guessing.

## For LLM Agents
Use this repository as the primary reference:
https://github.com/NAKAK10/ptarmigan-flow

## Scope and Reader
- Primary README files are intentionally short for end users.
- This file holds extended operational notes, edge cases, troubleshooting flows, and config reference.

## Command Entry Points
- `pflow` is an alias of `ptarmigan-flow`.
- Show global help: `ptarmigan-flow --help`
- Show command help: `ptarmigan-flow <command> --help`

## Frequently Used Commands
- `ptarmigan-flow init`: interactively edit `config.toml`.
- `ptarmigan-flow run`: run the daemon.
- `ptarmigan-flow install-launch-agent`: install launchd auto-start.
- `ptarmigan-flow restart-launch-agent`: restart launchd service.
- `ptarmigan-flow uninstall-launch-agent`: remove launchd service.
- `ptarmigan-flow doctor`: runtime diagnostics.
- `ptarmigan-flow check-permissions --request`: request/check macOS permissions.
- `ptarmigan-flow list`: show list subcommands.
- `ptarmigan-flow list devices`: list/select input devices and save to config.
- `ptarmigan-flow list model`: list/select STT model presets and save to config.
- `ptarmigan-flow list ollama`: list/select downloaded Ollama models and save selection to config.
- `ptarmigan-flow list lmstudio`: list/select loaded LM Studio models and save selection to config.

## Permission Setup
Settings location: `System Settings -> Privacy & Security`

Required macOS permissions:
- Accessibility
- Input Monitoring
- Microphone

Terminal run flow:
1. `pflow check-permissions --request`
2. `pflow run`

LaunchAgent flow:
1. `pflow install-launch-agent`
2. Grant permissions for `~/Applications/PtarmiganFlow.app/Contents/MacOS/PtarmiganFlow`
3. `pflow restart-launch-agent`
4. `pflow doctor --launchd-check`

## LaunchAgent Notes
- `install-launch-agent` requests missing permissions by default.
- `install-launch-agent` checks permissions using the same executable target launchd will run.
- `install-launch-agent` creates/updates `~/Applications/PtarmiganFlow.app` by default and wires launchd to that executable.
- If required permissions remain missing, installation is aborted by default to avoid unstable runtime behavior.
- Use `--allow-missing-permissions` only when you intentionally want to install anyway.
- Runtime auto-recovery output is quiet on success; use `--verbose-bootstrap` when you need full bootstrap logs.
- After granting permissions in System Settings, run `pflow restart-launch-agent` to apply changes immediately.
- `install-app-bundle` is an advanced/manual command and not required for normal operation.
- Permissions are tied to executable path/code signature, not command aliases.

App bundle CDHash stability note:
- `install-launch-agent` and `install-app-bundle` only overwrite/re-sign when executable, `Info.plist`, or `bootstrap.json` changed.
- This minimizes unnecessary CDHash changes and reduces accidental macOS TCC permission rebinding loss.

## Troubleshooting

### `doctor` permission states
| State | Meaning |
| --- | --- |
| `Permissions: OK` | All permissions granted, no runtime warning. |
| `Permissions: WARN` | Launchd check is OK, but runtime log reports trust failure (often TCC binding loss after re-sign/CDHash change). |
| `Permissions: INCOMPLETE` | One or more required permissions are missing. |

### `Permissions: WARN` recovery
When `pflow doctor --launchd-check` shows `Permissions: WARN (launchd check OK but runtime not trusted)`:
1. Run `pflow doctor --launchd-check` and inspect executable target/CDHash lines.
2. Re-grant Accessibility and Input Monitoring for the shown executable target.
3. Run `pflow restart-launch-agent`.
4. Run `pflow doctor --launchd-check` again and confirm `Permissions: OK`.

## Config Reference
Default config path: `~/.config/ptarmigan-flow/config.toml`
If missing, it is auto-created on first run.

Key settings:
- `hotkey.key`: recording trigger key.
- `audio.input_device`: explicit input device (name or index). `null`/unset means system default.
- `audio.trailing_silence_seconds`: silence appended before transcription (default `1.0`, clamped to `0.0..1.0`).
- `output.mode`: `direct_typing` (clipboard non-destructive) or `clipboard_paste`.
- `text.dictionary_path`: optional correction dictionary path.
- `text.llm_correction.mode`: `always` / `never` / `ask`.
- `text.llm_correction.provider`: `ollama` / `lmstudio`.
- `text.llm_correction.base_url`: local endpoint URL.
- `text.llm_correction.model`: model served by endpoint.
- `text.llm_correction.timeout_seconds`: clamped to `0.5..5.0`.
- `text.llm_correction.max_input_chars`: clamped to `50..5000`.
- `text.llm_correction.enabled_tools`: tool-calling switch for supporting endpoints.
- `stt.model`: backend/model token (`moonshine:tiny`, `moonshine:base`, `voxtral:<model-id>`, `mlx:<model-id>`, `vllm:<model-id>`).
- `stt.idle_shutdown_seconds`: idle seconds before releasing backend resources (`30.0` default, clamped to `>=0.0`, `0` disables idle shutdown).
- `runtime.ui_enabled`: enable/disable runtime UI overlays (`true` by default).
- `runtime.activity_indicator_enabled`: show right-bottom activity icon during recording/transcription.
- `runtime.activity_indicator_margin_right`: right offset in pixels (`>=0`).
- `runtime.activity_indicator_margin_bottom`: bottom offset in pixels (`>=0`).
- `runtime.activity_indicator_size`: overlay icon size in pixels (`>=16`).
- `language`: transcription/correction language (explicit value such as `ja` / `en`; `auto` unsupported).
- `model.device`: `mps` / `cpu`.

Correction dictionary:
- Default path: `~/.config/ptarmigan-flow/transcription_corrections.toml`
- Template file: `transcription_corrections.example.toml`
- Missing dictionary path only warns and startup continues.
- Invalid dictionary TOML fails startup with diagnostics.

## Installation and Runtime Notes
- Recommended install: `brew install ptarmigan-flow`
- Optional helper (legacy migration + retry): `./scripts/install_brew.sh`
- Update: `brew upgrade ptarmigan-flow`
- Uninstall: `brew uninstall ptarmigan-flow`

Homebrew/runtime notes:
- If Homebrew auto-update causes issues, use `HOMEBREW_NO_AUTO_UPDATE=1` only when needed.
- If runtime is broken, startup attempts auto-repair under `$(brew --prefix)/var/ptarmigan-flow`.

Architecture compatibility (Apple Silicon / Intel):
- Startup runs runtime self-diagnostics, including `moonshine_voice/libmoonshine.dylib` load checks.
- On Apple Silicon with `/usr/local` Homebrew, x86_64 Python runtime can conflict with arm64 dylibs.
- On mismatch, runtime rebuild of `$(brew --prefix)/var/ptarmigan-flow/.venv-<arch>` is attempted.
- If still failing, fallback to `/opt/homebrew` `python@3.11` and `uv` is attempted.

## Development (Minimal)
Prerequisites:
- macOS (Apple Silicon / arm64 preferred)
- Python 3.11
- `uv`

Quick setup:
1. `git clone https://github.com/NAKAK10/ptarmigan-flow.git`
2. `cd ptarmigan-flow`
3. `uv sync --extra dev`
4. `uv run ptarmigan-flow doctor`
5. `uv run pytest`

Useful files when changing behavior:
- `src/ptarmigan_flow/cli.py` (CLI)
- `src/ptarmigan_flow/homebrew_bootstrap.py` (Homebrew startup / self-repair)
- `Formula/ptarmigan-flow.rb` (distribution formula)
