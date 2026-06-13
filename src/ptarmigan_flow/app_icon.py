"""Helpers for loading the packaged macOS app icon."""

from __future__ import annotations

from importlib import resources

APP_ICON_FILE = "PtarmiganFlow.icns"
APP_ICON_RESOURCE_PACKAGE = "ptarmigan_flow.resources"


def app_icon_resource() -> resources.abc.Traversable:
    return resources.files(APP_ICON_RESOURCE_PACKAGE).joinpath(APP_ICON_FILE)


def app_icon_bytes() -> bytes:
    return app_icon_resource().read_bytes()
