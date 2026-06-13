from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import ptarmigan_flow.app_daemon_controller as controller_module
from ptarmigan_flow.app_daemon_controller import DaemonController, build_daemon_from_config
from ptarmigan_flow.config import AppConfig
from ptarmigan_flow.presentation.cli import commands as cli_commands


def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for condition")


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


def test_daemon_controller_starts_and_stops_daemon_on_background_thread() -> None:
    daemon = _FakeDaemon()
    controller = DaemonController(lambda: daemon)

    controller.start()

    _wait_until(daemon.started.is_set)
    assert controller.is_running is True
    assert daemon.run_calls == 1

    controller.stop()

    assert daemon.stop_calls == 1
    assert controller.is_running is False


def test_daemon_controller_ignores_second_start_while_running() -> None:
    daemons: list[_FakeDaemon] = []

    def factory() -> _FakeDaemon:
        daemon = _FakeDaemon()
        daemons.append(daemon)
        return daemon

    controller = DaemonController(factory)
    controller.start()
    _wait_until(lambda: len(daemons) == 1 and daemons[0].started.is_set())

    controller.start()

    assert len(daemons) == 1
    controller.stop()


def test_daemon_controller_records_factory_error_without_raising() -> None:
    expected = RuntimeError("factory failed")
    controller = DaemonController(lambda: (_ for _ in ()).throw(expected))

    controller.start()

    _wait_until(lambda: controller.last_error is expected)
    assert controller.is_running is False


def test_daemon_controller_records_run_forever_error_without_raising() -> None:
    expected = RuntimeError("run failed")

    class FailingDaemon(_FakeDaemon):
        def run_forever(self) -> None:
            self.started.set()
            raise expected

    daemon = FailingDaemon()
    controller = DaemonController(lambda: daemon)

    controller.start()

    _wait_until(lambda: controller.last_error is expected)
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
