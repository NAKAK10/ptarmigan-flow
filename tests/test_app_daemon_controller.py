from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import ptarmigan_flow.app_daemon_controller as controller_module
from ptarmigan_flow.app_daemon_controller import (
    DaemonController,
    build_daemon_from_config,
    daemon_run_command,
)
from ptarmigan_flow.config import AppConfig
from ptarmigan_flow.presentation.cli import commands as cli_commands


class _FakeDaemon:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.run_calls = 0
        self.stop_calls = 0
        self.transcriber = SimpleNamespace(
            preflight_model=lambda: "fake-backend",
            runtime_status=lambda: "fake runtime ready",
        )

    def run_forever(self) -> None:
        self.run_calls += 1
        self.started.set()
        self.stopped.wait(timeout=1.0)

    def stop(self) -> None:
        self.stop_calls += 1
        self.stopped.set()


class _FakeProcess:
    def __init__(self, *, returncode: int | None = None, wait_times_out: bool = False) -> None:
        self.returncode = returncode
        self.wait_times_out = wait_times_out
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self.wait_times_out:
            self.returncode = 0

    def wait(self, timeout: float | None = None) -> int | None:
        self.wait_timeouts.append(timeout)
        if self.wait_times_out and self.kill_calls == 0:
            raise subprocess.TimeoutExpired(cmd="ptarmigan-flow", timeout=timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


def test_daemon_run_command_routes_cli_run_through_current_executable(
    monkeypatch, tmp_path
) -> None:
    executable = tmp_path / "PtarmiganFlow"
    monkeypatch.setattr(controller_module.sys, "executable", str(executable))

    assert daemon_run_command("~/ptarmigan/config.toml") == [
        str(executable),
        "-m",
        "ptarmigan_flow.cli",
        "run",
        "--config",
        str(Path("~/ptarmigan/config.toml").expanduser()),
    ]


def test_daemon_controller_starts_subprocess_with_built_command() -> None:
    process = _FakeProcess()
    commands: list[list[str]] = []

    def runner(command: list[str]) -> _FakeProcess:
        commands.append(command)
        return process

    controller = DaemonController(lambda: ["pflow", "run"], runner=runner)
    controller.start()

    assert controller.is_running is True
    assert controller.last_error is None
    assert commands == [["pflow", "run"]]


def test_daemon_controller_ignores_second_start_while_running() -> None:
    process = _FakeProcess()
    commands: list[list[str]] = []

    def runner(command: list[str]) -> _FakeProcess:
        commands.append(command)
        return process

    controller = DaemonController(lambda: ["pflow", "run"], runner=runner)
    controller.start()
    controller.start()

    assert commands == [["pflow", "run"]]


def test_daemon_controller_stop_terminates_subprocess_and_clears_reference() -> None:
    process = _FakeProcess()
    controller = DaemonController(lambda: ["pflow", "run"], runner=lambda _command: process)

    controller.start()
    controller.stop()

    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.wait_timeouts == [5.0]
    assert controller.is_running is False


def test_daemon_controller_stop_kills_subprocess_after_timeout() -> None:
    process = _FakeProcess(wait_times_out=True)
    controller = DaemonController(
        lambda: ["pflow", "run"],
        runner=lambda _command: process,
        join_timeout=0.25,
    )

    controller.start()
    controller.stop()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_timeouts == [0.25, 0.25]
    assert controller.is_running is False


def test_daemon_controller_records_runner_error_without_raising() -> None:
    expected = RuntimeError("runner failed")

    def runner(_command: list[str]) -> _FakeProcess:
        raise expected

    controller = DaemonController(lambda: ["pflow", "run"], runner=runner)

    controller.start()

    assert controller.last_error is expected
    assert controller.is_running is False


def test_daemon_controller_records_immediate_nonzero_exit() -> None:
    process = _FakeProcess(returncode=2)
    controller = DaemonController(lambda: ["pflow", "run"], runner=lambda _command: process)

    controller.start()

    assert isinstance(controller.last_error, RuntimeError)
    assert "return code 2" in str(controller.last_error)
    assert controller.is_running is False


def test_build_daemon_from_config_ensures_loads_and_builds_daemon(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config = AppConfig()
    daemon = _FakeDaemon()
    calls: list[tuple[str, Path] | tuple[str, AppConfig]] = []

    def fake_ensure(path: Path) -> None:
        calls.append(("ensure", path))

    def fake_load(path: Path) -> AppConfig:
        calls.append(("load", path))
        return config

    def fake_build(loaded_config: AppConfig, **kwargs) -> _FakeDaemon:
        assert kwargs == {}
        calls.append(("build", loaded_config))
        return daemon

    monkeypatch.setattr(controller_module, "ensure_config_exists", fake_ensure)
    monkeypatch.setattr(controller_module, "load_config", fake_load)
    monkeypatch.setattr(controller_module, "build_daemon", fake_build)

    result = build_daemon_from_config(config_path)

    assert result is daemon
    assert calls == [
        ("ensure", config_path),
        ("load", config_path),
        ("build", config),
    ]


def test_cli_run_uses_shared_daemon_builder(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config = AppConfig()
    rules = object()
    post_processor = object()
    daemon = _FakeDaemon()
    build_calls: list[dict[str, object]] = []

    def fake_build_daemon(loaded_config: AppConfig, **kwargs) -> _FakeDaemon:
        build_calls.append({"config": loaded_config, **kwargs})
        return daemon

    monkeypatch.setattr(cli_commands, "_remove_stale_pyc_modules", lambda _names: None)
    monkeypatch.setattr(cli_commands, "_resolve_config_path", lambda _value: config_path)
    monkeypatch.setattr(cli_commands, "load_config", lambda path: config)
    monkeypatch.setattr(cli_commands, "configure_logging", lambda _level: None)
    monkeypatch.setattr(
        cli_commands,
        "_load_corrections_with_diagnostics",
        lambda _config, *, config_path: (
            SimpleNamespace(
                warnings=[],
                loaded=False,
                rules=rules,
                path=config_path,
                disabled_regex_count=0,
            ),
            None,
        ),
    )
    for name in (
        "_is_moonshine_stt_model",
        "_is_vllm_stt_model",
        "_is_voxtral_stt_model",
        "_is_granite_stt_model",
        "_is_mlx_stt_model",
    ):
        monkeypatch.setattr(cli_commands, name, lambda _config: False)
    monkeypatch.setattr(
        cli_commands,
        "check_all_permissions",
        lambda: SimpleNamespace(all_granted=True),
    )
    monkeypatch.setattr(cli_commands, "_llm_enabled_for_this_run", lambda _config: False)
    monkeypatch.setattr(
        cli_commands,
        "_build_runtime_post_processor",
        lambda _config, *, base_processor, llm_enabled_override: post_processor,
    )
    monkeypatch.setattr(cli_commands, "_streaming_supported_by_output_mode", lambda _config: True)
    monkeypatch.setattr(cli_commands, "_log_stt_startup_download_if_needed", lambda _model: None)
    monkeypatch.setattr(cli_commands, "_stt_model_from_config", lambda _config: config.stt.model)
    monkeypatch.setattr(cli_commands, "build_daemon", fake_build_daemon)

    result = cli_commands.cmd_run(SimpleNamespace(config=str(config_path)))

    assert result == 0
    assert build_calls == [
        {
            "config": config,
            "post_processor": post_processor,
            "enable_streaming": True,
        }
    ]
    assert daemon.run_calls == 1
    assert daemon.stop_calls == 1
