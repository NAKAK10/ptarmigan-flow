"""CLI parser composition."""

from __future__ import annotations

import argparse

from ptarmigan_flow.presentation.cli import commands
from ptarmigan_flow.stt.model_catalog import HUB_SEARCH_BACKENDS


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command parser."""
    parser = argparse.ArgumentParser(description="ptarmigan-flow")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {commands._resolve_app_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run background daemon")
    run_parser.add_argument("--config", default=None, help="Path to config TOML")
    run_parser.set_defaults(func=commands.cmd_run)

    init_parser = subparsers.add_parser("init", help="Interactively edit config")
    init_parser.add_argument("--config", default=None, help="Path to config TOML")
    init_parser.set_defaults(func=commands.cmd_init)

    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument("--config", default=None, help="Path to config TOML")
    config_parser = subparsers.add_parser(
        "config",
        parents=[config_parent],
        help="Interactively edit config by section",
    )
    config_parser.set_defaults(func=commands.cmd_config, config_target=None)
    config_subparsers = config_parser.add_subparsers(dest="config_target")
    for section_name, section_help in commands._config_section_help_items():
        section_parser = config_subparsers.add_parser(
            section_name,
            parents=[config_parent],
            help=section_help,
        )
        section_parser.set_defaults(func=commands.cmd_config, config_target=section_name)

    list_parser = subparsers.add_parser(
        "list",
        help="List available resources",
    )
    list_parser.set_defaults(func=commands.cmd_list)
    list_subparsers = list_parser.add_subparsers(dest="list_target")
    list_devices_parser = list_subparsers.add_parser(
        "devices",
        help="List audio input devices and save selected device to config",
    )
    list_devices_parser.add_argument("--config", default=None, help="Path to config TOML")
    list_devices_parser.set_defaults(func=commands.cmd_list_devices)

    list_model_parser = list_subparsers.add_parser(
        "model",
        help="List STT model presets and save selected model to config",
    )
    list_model_parser.add_argument("--config", default=None, help="Path to config TOML")
    list_model_parser.add_argument(
        "--hub-search",
        metavar="QUERY",
        default=None,
        help="Search Hugging Face Hub for ASR models (results are unverified)",
    )
    list_model_parser.add_argument(
        "--backend",
        choices=list(HUB_SEARCH_BACKENDS),
        default=None,
        help="Backend prefix to combine with Hub search results",
    )
    list_model_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of Hub search results (default: 20)",
    )
    list_model_parser.set_defaults(func=commands.cmd_list_model)

    list_typing_parser = list_subparsers.add_parser(
        "typing",
        help="List output typing modes and save selected mode to config",
    )
    list_typing_parser.add_argument("--config", default=None, help="Path to config TOML")
    list_typing_parser.set_defaults(func=commands.cmd_list_typing)

    list_ollama_parser = list_subparsers.add_parser(
        "ollama",
        help="List downloaded Ollama models",
    )
    list_ollama_parser.add_argument("--config", default=None, help="Path to config TOML")
    list_ollama_parser.set_defaults(func=commands.cmd_list_ollama)

    list_lmstudio_parser = list_subparsers.add_parser(
        "lmstudio",
        help="List loaded LM Studio models",
    )
    list_lmstudio_parser.add_argument("--config", default=None, help="Path to config TOML")
    list_lmstudio_parser.set_defaults(func=commands.cmd_list_lmstudio)

    download_model_parser = subparsers.add_parser(
        "download-model",
        help="Download the configured STT model snapshot",
    )
    download_model_parser.add_argument("--config", default=None, help="Path to config TOML")
    download_model_parser.add_argument(
        "--model",
        default=None,
        help="STT model token to download (default: stt.model from config)",
    )
    download_model_parser.set_defaults(func=commands.cmd_download_model)

    check_parser = subparsers.add_parser("check-permissions", help="Check macOS permissions")
    check_parser.add_argument(
        "--request",
        action="store_true",
        help="Request missing macOS permissions (shows system prompts when possible)",
    )
    check_parser.set_defaults(func=commands.cmd_check_permissions)

    doctor_parser = subparsers.add_parser("doctor", help="Show runtime diagnostics")
    doctor_parser.add_argument("--config", default=None, help="Path to config TOML")
    doctor_parser.add_argument(
        "--launchd-check",
        action="store_true",
        help="Compare permission status in launchd context via launchctl asuser",
    )
    doctor_parser.set_defaults(func=commands.cmd_doctor)

    install_parser = subparsers.add_parser("install-launch-agent", help="Install launchd agent")
    install_parser.add_argument("--config", default=None, help="Path to config TOML")
    install_parser.add_argument(
        "--request-permissions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request missing macOS permissions before installing launchd agent",
    )
    install_parser.add_argument(
        "--allow-missing-permissions",
        action="store_true",
        help="Install launchd agent even when required macOS permissions are missing",
    )
    install_parser.add_argument(
        "--verbose-bootstrap",
        action="store_true",
        help="Show detailed runtime bootstrap logs when recovery runs",
    )
    install_parser.add_argument(
        "--install-app-bundle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create/update ~/Applications/PtarmiganFlow.app before installing launchd agent",
    )
    install_parser.set_defaults(func=commands.cmd_install_launch_agent)

    uninstall_parser = subparsers.add_parser("uninstall-launch-agent", help="Remove launchd agent")
    uninstall_parser.set_defaults(func=commands.cmd_uninstall_launch_agent)

    restart_parser = subparsers.add_parser("restart-launch-agent", help="Restart launchd agent")
    restart_parser.set_defaults(func=commands.cmd_restart_launch_agent)

    app_bundle_parser = subparsers.add_parser(
        "install-app-bundle",
        help="Create or update ~/Applications/PtarmiganFlow.app from current runtime",
    )
    app_bundle_parser.add_argument(
        "--path",
        default=None,
        help="Custom .app destination path (default: ~/Applications/PtarmiganFlow.app)",
    )
    app_bundle_parser.set_defaults(func=commands.cmd_install_app_bundle)

    update_parser = subparsers.add_parser(
        "update",
        help="Update the Homebrew installation and refresh installed launch agent",
    )
    update_parser.set_defaults(func=commands.cmd_update)

    refresh_launch_agent_after_update_parser = subparsers.add_parser(
        "_refresh-launch-agent-after-update",
        help=argparse.SUPPRESS,
    )
    refresh_launch_agent_after_update_parser.set_defaults(
        func=commands.cmd_refresh_launch_agent_after_update
    )

    return parser
