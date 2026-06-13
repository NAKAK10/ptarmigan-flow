"""Login item registration via SMAppService."""

from __future__ import annotations

import sys
from typing import Any


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _service_management_module() -> Any:
    import ServiceManagement

    return ServiceManagement


def _main_app_service() -> Any:
    return _service_management_module().SMAppService.mainAppService()


def _enabled_status() -> Any:
    return _service_management_module().SMAppServiceStatusEnabled


def _coerce_objc_bool(result: Any) -> bool:
    """Normalize a PyObjC method result to a plain bool.

    PyObjC bridges ``- (BOOL)...AndReturnError:(NSError **)error`` to a
    ``(success, error)`` tuple, so a non-empty tuple would otherwise always be
    truthy. Unwrap the leading success flag when present.
    """
    if isinstance(result, tuple):
        return bool(result[0]) if result else False
    return bool(result)


def register() -> bool:
    """Enable launch at login for the current app bundle."""
    if not _is_macos():
        return False
    try:
        return _coerce_objc_bool(_main_app_service().registerAndReturnError_(None))
    except Exception:
        return False


def unregister() -> bool:
    """Disable launch at login for the current app bundle."""
    if not _is_macos():
        return False
    try:
        return _coerce_objc_bool(_main_app_service().unregisterAndReturnError_(None))
    except Exception:
        return False


def is_enabled() -> bool:
    """Return whether the current app bundle is registered as a login item."""
    if not _is_macos():
        return False
    try:
        return _main_app_service().status() == _enabled_status()
    except Exception:
        return False
