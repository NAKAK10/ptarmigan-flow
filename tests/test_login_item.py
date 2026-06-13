from __future__ import annotations

from ptarmigan_flow import login_item


class FakeLoginService:
    def __init__(
        self,
        *,
        register_result: bool = True,
        unregister_result: bool = True,
        status_value: object = "disabled",
    ) -> None:
        self.register_result = register_result
        self.unregister_result = unregister_result
        self.status_value = status_value
        self.register_calls = 0
        self.unregister_calls = 0

    def registerAndReturnError_(self, _error) -> bool:  # noqa: N802
        self.register_calls += 1
        return self.register_result

    def unregisterAndReturnError_(self, _error) -> bool:  # noqa: N802
        self.unregister_calls += 1
        return self.unregister_result

    def status(self) -> object:
        return self.status_value


def _patch_macos_service(monkeypatch, service: FakeLoginService) -> None:
    monkeypatch.setattr(login_item, "_is_macos", lambda: True)
    monkeypatch.setattr(login_item, "_main_app_service", lambda: service)
    monkeypatch.setattr(login_item, "_enabled_status", lambda: "enabled")


def test_non_macos_login_item_functions_are_noops(monkeypatch) -> None:
    def fail_if_called():
        raise AssertionError("ServiceManagement should not be loaded outside macOS")

    monkeypatch.setattr(login_item, "_is_macos", lambda: False)
    monkeypatch.setattr(login_item, "_main_app_service", fail_if_called)

    assert login_item.register() is False
    assert login_item.unregister() is False
    assert login_item.is_enabled() is False


def test_register_uses_main_app_service(monkeypatch) -> None:
    service = FakeLoginService(register_result=True)
    _patch_macos_service(monkeypatch, service)

    assert login_item.register() is True
    assert service.register_calls == 1


def test_register_returns_false_when_service_registration_fails(monkeypatch) -> None:
    service = FakeLoginService(register_result=False)
    _patch_macos_service(monkeypatch, service)

    assert login_item.register() is False
    assert service.register_calls == 1


def test_unregister_uses_main_app_service(monkeypatch) -> None:
    service = FakeLoginService(unregister_result=True)
    _patch_macos_service(monkeypatch, service)

    assert login_item.unregister() is True
    assert service.unregister_calls == 1


def test_is_enabled_reflects_smappservice_status(monkeypatch) -> None:
    service = FakeLoginService(status_value="enabled")
    _patch_macos_service(monkeypatch, service)

    assert login_item.is_enabled() is True

    service.status_value = "disabled"
    assert login_item.is_enabled() is False


class TupleReturningLoginService(FakeLoginService):
    """Mimic PyObjC bridging of `(BOOL)...AndReturnError:` to a (success, error) tuple."""

    def registerAndReturnError_(self, _error):  # noqa: N802
        self.register_calls += 1
        return (self.register_result, None if self.register_result else "boom")

    def unregisterAndReturnError_(self, _error):  # noqa: N802
        self.unregister_calls += 1
        return (self.unregister_result, None if self.unregister_result else "boom")


def test_register_unwraps_pyobjc_success_tuple(monkeypatch) -> None:
    service = TupleReturningLoginService(register_result=True)
    _patch_macos_service(monkeypatch, service)

    assert login_item.register() is True


def test_register_unwraps_pyobjc_failure_tuple(monkeypatch) -> None:
    service = TupleReturningLoginService(register_result=False)
    _patch_macos_service(monkeypatch, service)

    assert login_item.register() is False


def test_unregister_unwraps_pyobjc_failure_tuple(monkeypatch) -> None:
    service = TupleReturningLoginService(unregister_result=False)
    _patch_macos_service(monkeypatch, service)

    assert login_item.unregister() is False
