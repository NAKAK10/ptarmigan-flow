from pathlib import Path

import pytest

from moonshine_flow.config import AppConfig, load_config, write_example_config


def test_write_example_and_load_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    write_example_config(cfg_path)

    loaded = load_config(cfg_path)
    assert isinstance(loaded, AppConfig)
    assert loaded.hotkey.key == "right_cmd"
    assert loaded.language == "en"
    assert loaded.stt.model == "moonshine:base"
    assert loaded.stt.idle_shutdown_seconds == 30.0
    assert loaded.audio.release_tail_seconds == 0.25
    assert loaded.audio.hotkey_release_reconcile_seconds == 0.25
    assert loaded.audio.hotkey_idle_reconcile_seconds == 1.0
    assert loaded.audio.trailing_silence_seconds == 1.0
    assert loaded.audio.input_device_policy.value == "playback_friendly"
    assert loaded.output.mode.value == "direct_typing"
    assert loaded.runtime.activity_indicator_enabled is True
    assert loaded.runtime.activity_indicator_margin_right == 24
    assert loaded.runtime.activity_indicator_margin_bottom == 24
    assert loaded.runtime.activity_indicator_size == 42
    assert loaded.text.dictionary_path is None
    assert loaded.text.llm_correction.mode.value == "never"
    assert loaded.text.llm_correction.provider == "ollama"
    assert loaded.text.llm_correction.timeout_seconds == 5.0
    assert loaded.text.llm_correction.max_input_chars == 500
    assert loaded.text.llm_correction.enabled_tools is False


def test_load_config_creates_missing_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "new.toml"
    loaded = load_config(cfg_path)

    assert cfg_path.exists()
    assert loaded.audio.sample_rate == 16000
    assert loaded.audio.hotkey_release_reconcile_seconds == 0.25
    assert loaded.audio.hotkey_idle_reconcile_seconds == 1.0
    assert loaded.audio.trailing_silence_seconds == 1.0
    assert loaded.audio.input_device_policy.value == "playback_friendly"
    assert loaded.stt.idle_shutdown_seconds == 30.0
    assert loaded.language == "en"
    assert loaded.output.mode.value == "direct_typing"
    assert loaded.runtime.activity_indicator_enabled is True
    assert loaded.runtime.activity_indicator_margin_right == 24
    assert loaded.runtime.activity_indicator_margin_bottom == 24
    assert loaded.runtime.activity_indicator_size == 42


def test_load_config_accepts_input_device_policy(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
language = "ja"

[hotkey]
key = "right_cmd"

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
max_record_seconds = 30
input_device = 3
input_device_policy = "external_preferred"

[stt]
model = "moonshine:base"

[model]
device = "mps"

[output]
mode = "clipboard_paste"
paste_shortcut = "cmd+v"

[runtime]
log_level = "INFO"
notify_on_error = true
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(cfg_path)
    assert loaded.audio.input_device == 3
    assert loaded.audio.input_device_policy.value == "external_preferred"


def test_load_config_clamps_audio_tail_durations_over_limit(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
language = "ja"

[hotkey]
key = "right_cmd"

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
max_record_seconds = 30
release_tail_seconds = 2.5
trailing_silence_seconds = 3.0

[stt]
model = "moonshine:base"

[model]
device = "mps"

[output]
mode = "clipboard_paste"
paste_shortcut = "cmd+v"

[runtime]
log_level = "INFO"
notify_on_error = true
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(cfg_path)
    assert loaded.audio.release_tail_seconds == 1.0
    assert loaded.audio.trailing_silence_seconds == 1.0


def test_load_config_clamps_audio_tail_durations_under_zero(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
language = "ja"

[hotkey]
key = "right_cmd"

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
max_record_seconds = 30
release_tail_seconds = -0.5
trailing_silence_seconds = -2.0

[stt]
model = "moonshine:base"

[model]
device = "mps"

[output]
mode = "clipboard_paste"
paste_shortcut = "cmd+v"

[runtime]
log_level = "INFO"
notify_on_error = true
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(cfg_path)
    assert loaded.audio.release_tail_seconds == 0.0
    assert loaded.audio.trailing_silence_seconds == 0.0


def test_load_config_clamps_hotkey_reconcile_seconds_under_zero(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
language = "ja"

[hotkey]
key = "right_cmd"

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
max_record_seconds = 30
hotkey_release_reconcile_seconds = -0.5
hotkey_idle_reconcile_seconds = -1.0

[stt]
model = "moonshine:base"
idle_shutdown_seconds = -5.0

[model]
device = "mps"

[output]
mode = "clipboard_paste"
paste_shortcut = "cmd+v"

[runtime]
log_level = "INFO"
notify_on_error = true
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(cfg_path)
    assert loaded.audio.hotkey_release_reconcile_seconds == 0.0
    assert loaded.audio.hotkey_idle_reconcile_seconds == 0.0
    assert loaded.stt.idle_shutdown_seconds == 0.0


def test_load_config_clamps_activity_indicator_bounds(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
language = "ja"

[hotkey]
key = "right_cmd"

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
max_record_seconds = 30

[stt]
model = "moonshine:base"

[model]
device = "mps"

[output]
mode = "clipboard_paste"
paste_shortcut = "cmd+v"

[runtime]
log_level = "INFO"
notify_on_error = true
activity_indicator_enabled = true
activity_indicator_margin_right = -8
activity_indicator_margin_bottom = -9
activity_indicator_size = 10
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(cfg_path)
    assert loaded.runtime.activity_indicator_margin_right == 0
    assert loaded.runtime.activity_indicator_margin_bottom == 0
    assert loaded.runtime.activity_indicator_size == 16


def test_load_config_clamps_llm_timeout_and_input_chars(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
language = "ja"

[hotkey]
key = "right_cmd"

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
max_record_seconds = 30

[stt]
model = "moonshine:base"

[model]
device = "mps"

[output]
mode = "clipboard_paste"
paste_shortcut = "cmd+v"

[runtime]
log_level = "INFO"
notify_on_error = true

[text.llm_correction]
mode = "always"
provider = "ollama"
base_url = "http://localhost:11434"
model = "qwen2.5:7b-instruct"
timeout_seconds = 10.0
max_input_chars = 10
enabled_tools = false
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(cfg_path)
    assert loaded.text.llm_correction.timeout_seconds == 5.0
    assert loaded.text.llm_correction.max_input_chars == 50


def test_load_config_maps_legacy_disable_tools_to_enabled_tools(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
language = "ja"

[hotkey]
key = "right_cmd"

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
max_record_seconds = 30

[stt]
model = "moonshine:base"

[model]
device = "mps"

[output]
mode = "clipboard_paste"
paste_shortcut = "cmd+v"

[runtime]
log_level = "INFO"
notify_on_error = true

[text.llm_correction]
mode = "always"
provider = "ollama"
base_url = "http://localhost:11434"
model = "qwen2.5:7b-instruct"
disable_tools = false
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(cfg_path)
    assert loaded.text.llm_correction.enabled_tools is True


def test_load_config_maps_legacy_enabled_to_mode(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
language = "ja"

[hotkey]
key = "right_cmd"

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
max_record_seconds = 30

[stt]
model = "moonshine:base"

[model]
device = "mps"

[output]
mode = "clipboard_paste"
paste_shortcut = "cmd+v"

[runtime]
log_level = "INFO"
notify_on_error = true

[text.llm_correction]
enabled = true
provider = "ollama"
base_url = "http://localhost:11434"
model = "qwen2.5:7b-instruct"
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(cfg_path)
    assert loaded.text.llm_correction.mode.value == "always"


def test_load_config_accepts_custom_llm_provider(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
language = "ja"

[hotkey]
key = "right_cmd"

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
max_record_seconds = 30

[stt]
model = "moonshine:base"

[model]
device = "mps"

[output]
mode = "clipboard_paste"
paste_shortcut = "cmd+v"

[runtime]
log_level = "INFO"
notify_on_error = true

[text.llm_correction]
mode = "always"
provider = "custom-provider"
base_url = "http://localhost:8080"
model = "my-model"
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(cfg_path)
    assert loaded.text.llm_correction.provider == "custom-provider"


def test_load_config_rejects_auto_language(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
language = "auto"

[hotkey]
key = "right_cmd"

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
max_record_seconds = 30

[stt]
model = "moonshine:base"

[model]
device = "mps"

[output]
mode = "clipboard_paste"
paste_shortcut = "cmd+v"

[runtime]
log_level = "INFO"
notify_on_error = true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="auto"):
        load_config(cfg_path)


def test_load_config_rejects_legacy_model_language(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[hotkey]
key = "right_cmd"

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
max_record_seconds = 30

[stt]
model = "moonshine:base"

[model]
language = "ja"
device = "mps"

[output]
mode = "clipboard_paste"
paste_shortcut = "cmd+v"

[runtime]
log_level = "INFO"
notify_on_error = true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model\\.language"):
        load_config(cfg_path)


def test_load_config_rejects_legacy_model_size(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[hotkey]
key = "right_cmd"

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
max_record_seconds = 30

[model]
size = "base"
device = "mps"

[output]
mode = "clipboard_paste"
paste_shortcut = "cmd+v"

[runtime]
log_level = "INFO"
notify_on_error = true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model\\.size"):
        load_config(cfg_path)


def test_load_config_migrates_legacy_model_size_when_opted_in(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[hotkey]
key = "right_cmd"

[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
max_record_seconds = 30

[model]
size = "tiny"
device = "mps"

[output]
mode = "clipboard_paste"
paste_shortcut = "cmd+v"

[runtime]
log_level = "INFO"
notify_on_error = true
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(cfg_path, allow_legacy_model_size=True)
    assert loaded.stt.model == "moonshine:tiny"
