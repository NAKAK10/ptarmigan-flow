<div align="center">
  <img src="https://github.com/user-attachments/assets/2d14cfd0-ec79-4b31-b634-e7c995528726" alt="ptarmigan flow banner" width="800">
</div>

# ptarmigan-flow

A push-to-talk transcription daemon for macOS.
It records only while a global hotkey is held, and when released it transcribes with Moonshine and pastes into the active app.
It uses Moonshine from the official repository: https://github.com/moonshine-ai/moonshine

[日本語](./README.ja.md)

For detailed operational notes, edge cases, troubleshooting, and full command/config reference, see [README.llm.md](./README.llm.md).

## For LLM Agents
Fetch the installation guide and follow it:

```bash
curl -s https://raw.githubusercontent.com/NAKAK10/ptarmigan-flow/main/README.llm.md
```

## Quickstart
```bash
brew install ptarmigan-flow
ptarmigan-flow doctor
ptarmigan-flow check-permissions --request
ptarmigan-flow run
```
`pflow` is a shorthand alias, so `pflow doctor` / `pflow run` also work.

If playback audio quality drops while recording (for example with Bluetooth headsets),
set `audio.input_device_policy = "playback_friendly"` and keep `audio.input_device` unset.

## Usage Sample Video

https://github.com/user-attachments/assets/f763be1b-54af-4342-886d-016837be7884

## Command Reference
### Core Commands
| Command | Description |
| --- | --- |
| `ptarmigan-flow init` | Interactively edit `config.toml` with current values as defaults. |
| `ptarmigan-flow run` | Run the background daemon. |
| `ptarmigan-flow install-launch-agent` | One-time setup: install the launchd agent for auto-start at login. |
| `ptarmigan-flow restart-launch-agent` | Restart the launchd agent to apply newly granted macOS permissions. |

For full command list and options:
- `ptarmigan-flow --help`
- `ptarmigan-flow <command> --help`

## Installation (Homebrew)
### Fast path (recommended)
```bash
brew install ptarmigan-flow
```

### Optional helper (legacy migration + retry)
```bash
./scripts/install_brew.sh
```

### Troubleshooting install failures
If `brew install ptarmigan-flow` fails with:
`Broken Python installation, platform.mac_ver() returned an empty value`

The formula now validates `/usr/local/opt/python@3.11` and automatically
falls back to `/opt/homebrew/opt/python@3.11` when available.
If both runtimes are unhealthy, repair Python and retry:

```bash
brew reinstall python@3.11
./scripts/install_brew.sh
```

Update / uninstall:
```bash
brew upgrade ptarmigan-flow
brew uninstall ptarmigan-flow
```
