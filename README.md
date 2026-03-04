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

<video src="./assets/usage-sample.mov" controls muted playsinline>
  Your browser does not support the video tag.
</video>

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

Update / uninstall:
```bash
brew upgrade ptarmigan-flow
brew uninstall ptarmigan-flow
```
