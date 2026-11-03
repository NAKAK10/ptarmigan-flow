# Haas Japanese Q4_K

Haas is an optional Japanese recognizer. Your current model, language and other
settings stay unchanged until you explicitly select it. This integration supports
only `haas:sakasegawa/Haas-ja-v1.0-GGUF`, using its pinned Q4_K file. It does not
make arbitrary Hugging Face GGUF models compatible with speech.cpp, llama.cpp,
Ollama or LM Studio.

## Install speech.cpp from the Mac app

On Apple Silicon, open Settings and use **Install speech.cpp** on the Haas card
(or **Retry** after a failure). This also works when the app runs under Rosetta
on Apple Silicon. The app downloads only the official pinned speech.cpp **0.10.1
macOS arm64 Metal ZIP**, verifies its hardcoded SHA256, and installs the standalone
`speech` executable at:

```text
~/Library/Application Support/ptarmigan-flow/runtimes/speech.cpp/0.10.1/macos-arm64-metal/speech
```

No sudo, Homebrew or shell changes are needed. Existing external executables and
configuration are left untouched. Explicitly installing through the app activates
this managed runtime unless `PTARMIGAN_FLOW_SPEECH_CPP_PATH` overrides it. Failed
installs preserve the previous managed copy. Installing speech.cpp does not
download the GGUF model; use the separate model Download action below.

Intel Macs have no official upstream macOS binary and cannot use this app-managed
installer. Use the manual/source installation instructions below. The installer
is a Settings action; there is no corresponding `pflow` installation command.

## Manual runtime installation (CLI or source build)

Install **nyosegawa/speech.cpp 0.9.4 or later** using its
[installation documentation](https://github.com/nyosegawa/speech.cpp/blob/v0.9.4/docs/install.md)
and [release archives](https://github.com/nyosegawa/speech.cpp/releases). Choose the
archive for your platform and verify its release SHA256SUMS before installing.
The macOS release is for Apple Silicon; Intel Macs need a compatible source build.
The executable is not bundled with Ptarmigan Flow; manual installation remains
available for CLI users and compatible source builds.

Confirm the installation yourself with `speech --version`; it must identify itself
as `speech.cpp` and report at least `0.9.4`.

Executable discovery searches, in order:

1. `PTARMIGAN_FLOW_SPEECH_CPP_PATH`: one executable path, including paths with spaces;
   arguments or shell commands are not accepted. An invalid override is an error.
2. The managed runtime at the Application Support path above.
3. `PATH`.
4. `~/.local/bin/speech` (the official installer's command location).
5. `/opt/homebrew/bin/speech`, then `/usr/local/bin/speech`.

For example, a CLI session can use
`export PTARMIGAN_FLOW_SPEECH_CPP_PATH="/absolute/path/to/speech"`.
A Finder-launched app may not inherit shell exports or Homebrew's PATH. Installing
through the Haas card or at a searched location avoids depending on those exports.
Restart the app after changing its runtime environment. Availability discovery checks paths only;
background preflight checks the version, checksum and an actual short recognition.

## Choose the model and language

In Settings, choose **Japanese** in Language and **Haas Japanese v1.0 (Q4_K)** in
the existing model selector, then save. A conflicting English or Chinese selection
is rejected with guidance; selecting Haas does not silently change your language.
The model card remains visible before speech.cpp is installed and explains the
runtime requirement. **Automatic** is also available for Haas, but the model still
recognizes Japanese only. Other backends' language behavior is unchanged.
When leaving Haas with `auto` selected, explicitly choose a language supported by
the next backend. In the CLI, choose `ja` first with `pflow config language`, then
change the model; the GUI lets you change both fields before saving.

For the CLI:

```sh
pflow config language       # choose ja before selecting Haas
pflow list model            # choose haas:sakasegawa/Haas-ja-v1.0-GGUF
pflow download-model --model haas:sakasegawa/Haas-ja-v1.0-GGUF
pflow doctor
pflow run
```

`pflow config stt` also includes the preset. Or explicitly set both values in
`~/.config/ptarmigan-flow/config.toml`:

```toml
language = "ja" # "auto" is also accepted for Haas

[stt]
model = "haas:sakasegawa/Haas-ja-v1.0-GGUF"
```

The separate GUI model Download action and CLI download command fetch only
`Haas-616M-ja-v1.0-Q4_K.gguf` (351,901,216 bytes), from revision
`25c400baaa9c002df8feffc140804e84228f069b`. Its SHA-256 is
`d24d53f29b75214a933ff482b0de1217e0a800f4118cca39ad6cf03609a89d6f`.
These values come from the
[speech.cpp 0.9.4 catalog](https://github.com/nyosegawa/speech.cpp/blob/v0.9.4/tools/catalog/catalog.json).
Other quantizations, old revisions and README-only snapshots do not count as
downloaded. Cache status checks the exact filename and size; downloading and
preflight verify the checksum. To repair corrupt cached bytes, run the explicit
`download-model --model ...` command again, or explicitly retry the GUI download;
both verify the checksum and re-fetch if needed. Ordinary GUI status stays a
cheap filename/size check.

The cache respects `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME` and
`XDG_CACHE_HOME`, falling back to `~/.cache/huggingface/hub`. A complete pinned
file works offline. CLI startup downloads a missing file after permissions are
checked; the GUI uses its existing download/progress flow before startup.

## Latency and recording limits

Each utterance starts a fresh process and reloads the weights. Loading and
initialization add overhead on every utterance, but this is not necessarily the
full initial cold-start penalty: OS file caches and runtime/driver caches can make
later invocations substantially faster. The adapter retains no model in its own
process and does not claim to keep the model warm. It returns one final transcript
after release;
streaming input, a persistent worker and custom vocabulary are not implemented.
Preflight also runs a separate short probe whose text is discarded.

In a limited local smoke on an M3 with 16 GB RAM, speech.cpp 0.10.1 and the
pinned Q4_K model recognized the same synthesized Japanese utterance exactly in
two `ja` adapter calls and one `auto` call. Initial preflight took 5.63 seconds;
the following adapter calls took 0.26–0.32 seconds, and repeated raw CLI calls
took 0.22–0.30 seconds. These are observations from one short input and machine,
not a robust benchmark, guaranteed latency or a comparison with Whisper accuracy.

The existing `[model] device` setting is honored: `cpu` passes `--device cpu`,
`gpu` explicitly requests a GPU, and `auto` lets speech.cpp choose the first GPU
or the CPU when none is available. The app's existing `mps` default maps to
`auto`, allowing Metal on supported Macs without passing an unsupported literal
`mps`. Other values, including runtime-specific device names, are rejected by
this adapter with guidance to choose one of these settings.

Use short push-to-talk utterances. The model can drop sentences in long,
multi-sentence recordings, and its attention memory grows with input length.
Speech.cpp recommends VAD segmentation for long recordings; this integration
does not enable VAD or download a second model. The default 30-second recording
limit is unchanged, and increasing it does not remove this limitation.

Requests have a 20–60 second inference timeout. Nonzero exits, model-limited or
incomplete results, and malformed output are errors; partial text is not pasted.
Stopping voice input cancels the owned process and cleans up its temporary WAV.
Actual end-to-end latency and recognition quality depend on the machine and audio;
no benchmark result is promised by this integration.
