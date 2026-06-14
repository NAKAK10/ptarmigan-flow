"""WKWebView host for the PtarmiganFlow macOS UI."""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import Any

from ptarmigan_flow.web_bridge import WebBridgeDispatcher

try:  # pragma: no cover - exercised only on macOS with PyObjC installed.
    import objc
    from Foundation import NSArray, NSDictionary, NSObject
except ImportError:  # pragma: no cover - lets source-level tests import this module.
    _NS_ARRAY_TYPES: tuple[type[Any], ...] = ()
    _NS_DICT_TYPES: tuple[type[Any], ...] = ()

    class _ObjCShim:
        @staticmethod
        def python_method(func):
            return func

        @staticmethod
        def super(cls, instance):
            return super(cls, instance)

    class NSObject:  # type: ignore[no-redef]
        pass

    objc = _ObjCShim()  # type: ignore[assignment]
else:
    _NS_ARRAY_TYPES = (NSArray,)
    _NS_DICT_TYPES = (NSDictionary,)


def _ns_to_py(obj: Any) -> Any:
    if isinstance(obj, _NS_DICT_TYPES):
        return {_ns_to_py(key): _ns_to_py(value) for key, value in obj.items()}
    if isinstance(obj, _NS_ARRAY_TYPES):
        return [_ns_to_py(item) for item in obj]
    return obj


def webui_resource_dir() -> Path:
    """Return the bundled webui resource directory."""
    return Path(importlib.resources.files("ptarmigan_flow").joinpath("webui"))


class WebUIController(NSObject):
    """Owns the WKWebView window and bridges JS messages to Python."""

    def initWithBridge_title_(self, bridge: WebBridgeDispatcher, title: str):  # noqa: N802
        self = objc.super(WebUIController, self).init()
        if self is None:
            return None

        from AppKit import (  # noqa: PLC0415
            NSBackingStoreBuffered,
            NSWindow,
            NSWindowStyleMaskClosable,
            NSWindowStyleMaskMiniaturizable,
            NSWindowStyleMaskTitled,
        )
        from Foundation import NSURL, NSMakeRect  # noqa: PLC0415
        from WebKit import (  # noqa: PLC0415
            WKUserContentController,
            WKWebView,
            WKWebViewConfiguration,
        )

        self.bridge = bridge
        self.NSURL = NSURL
        self.route = "onboarding"
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 860, 640),
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable,
            NSBackingStoreBuffered,
            False,
        )
        self.window.setTitle_(title)
        self.window.setReleasedWhenClosed_(False)

        content_controller = WKUserContentController.alloc().init()
        content_controller.addScriptMessageHandler_name_(self, "bridge")
        config = WKWebViewConfiguration.alloc().init()
        config.setUserContentController_(content_controller)

        self.web_view = WKWebView.alloc().initWithFrame_configuration_(
            self.window.contentView().bounds(),
            config,
        )
        self.web_view.setAutoresizingMask_(18)
        self.window.contentView().addSubview_(self.web_view)
        self.window.center()
        self._load_webui()
        return self

    @objc.python_method
    def _load_webui(self) -> None:
        resource_dir = webui_resource_dir()
        index_path = resource_dir / "index.html"
        index_url = self.NSURL.fileURLWithPath_(str(index_path))
        read_access_url = self.NSURL.fileURLWithPath_(str(resource_dir))
        self.web_view.loadFileURL_allowingReadAccessToURL_(index_url, read_access_url)

    def userContentController_didReceiveScriptMessage_(self, _controller, message):  # noqa: N802
        body = _ns_to_py(message.body())
        if not isinstance(body, dict):
            self.dispatch_to_javascript(
                {"id": None, "ok": False, "error": "Bridge message body must be an object."}
            )
            return

        message_id = body.get("id")
        action = str(body.get("action", ""))
        payload = body.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        try:
            result = self.bridge.handle_action(action, payload)
        except Exception as exc:
            self.dispatch_to_javascript({"id": message_id, "ok": False, "error": str(exc)})
            return
        self.dispatch_to_javascript({"id": message_id, "ok": True, "result": result})

    @objc.python_method
    def dispatch_to_javascript(self, message: dict[str, Any]) -> None:
        script = "window.app.dispatch(" + json.dumps(message) + ")"
        self.web_view.evaluateJavaScript_completionHandler_(script, None)

    @objc.python_method
    def push_event(self, event: str, payload: dict[str, Any]) -> None:
        self.dispatch_to_javascript({"event": event, "payload": payload})

    @objc.python_method
    def show(self, *, route: str = "onboarding") -> None:
        self.route = route
        self.window.makeKeyAndOrderFront_(None)
        self.window.orderFrontRegardless()
        self.push_event("routeChanged", {"route": route})
