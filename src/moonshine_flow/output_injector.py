"""Text output injection into focused macOS app."""

from __future__ import annotations

import logging
import shlex
import subprocess
import time

import pyperclip

LOGGER = logging.getLogger(__name__)


APPLE_MODIFIERS = {
    "cmd": "command down",
    "command": "command down",
    "ctrl": "control down",
    "control": "control down",
    "alt": "option down",
    "option": "option down",
    "shift": "shift down",
}


class OutputInjector:
    """Inject transcription text into currently focused app."""

    def __init__(self, mode: str, paste_shortcut: str) -> None:
        self.mode = mode
        self.paste_shortcut = paste_shortcut

    @staticmethod
    def _parse_shortcut(shortcut: str) -> tuple[str, list[str]]:
        parts = [token.strip().lower() for token in shortcut.split("+") if token.strip()]
        if not parts:
            raise ValueError("Shortcut cannot be empty")

        key = parts[-1]
        modifiers: list[str] = []
        for token in parts[:-1]:
            if token not in APPLE_MODIFIERS:
                raise ValueError(f"Unsupported shortcut modifier: {token}")
            modifiers.append(APPLE_MODIFIERS[token])

        if len(key) != 1:
            raise ValueError("Paste shortcut key must be a single character")
        return key, modifiers

    def _send_shortcut(self) -> None:
        key, modifiers = self._parse_shortcut(self.paste_shortcut)
        modifiers_script = ", ".join(modifiers)

        if modifiers_script:
            script = (
                f'tell application "System Events" '
                f'to keystroke "{key}" using {{{modifiers_script}}}'
            )
        else:
            script = f'tell application "System Events" to keystroke "{key}"'

        LOGGER.debug("Executing AppleScript: %s", shlex.quote(script))
        subprocess.run(["osascript", "-e", script], check=True)

    @staticmethod
    def _escape_applescript_text(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _send_text_via_quartz(text: str) -> bool:
        try:
            from Quartz import (
                CGEventCreateKeyboardEvent,
                CGEventKeyboardSetUnicodeString,
                CGEventPost,
                kCGHIDEventTap,
            )
        except Exception:
            return False

        try:
            chunk_size = 32
            for offset in range(0, len(text), chunk_size):
                chunk = text[offset : offset + chunk_size]
                if not chunk:
                    continue
                key_down = CGEventCreateKeyboardEvent(None, 0, True)
                key_up = CGEventCreateKeyboardEvent(None, 0, False)
                CGEventKeyboardSetUnicodeString(key_down, len(chunk), chunk)
                CGEventKeyboardSetUnicodeString(key_up, len(chunk), chunk)
                CGEventPost(kCGHIDEventTap, key_down)
                CGEventPost(kCGHIDEventTap, key_up)
                time.sleep(0.001)
            return True
        except Exception:
            LOGGER.debug("Quartz direct typing failed; fallback to AppleScript", exc_info=True)
            return False

    def _send_text_direct(self, text: str) -> None:
        if self._send_text_via_quartz(text):
            return

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        scripts: list[str] = []
        for index, line in enumerate(lines):
            if line:
                escaped = self._escape_applescript_text(line)
                scripts.append(f'tell application "System Events" to keystroke "{escaped}"')
            if index < len(lines) - 1:
                scripts.append('tell application "System Events" to key code 36')

        if not scripts:
            return
        command = ["osascript"]
        for script in scripts:
            LOGGER.debug("Executing AppleScript: %s", shlex.quote(script))
            command.extend(["-e", script])
        subprocess.run(command, check=True)

    def inject(self, text: str) -> bool:
        """Inject text into active app via configured output mode."""
        if not text.strip():
            LOGGER.debug("Skipping empty transcription output")
            return False

        if self.mode == "clipboard_paste":
            pyperclip.copy(text)
            self._send_shortcut()
            LOGGER.info("Transcription pasted into active app")
            return True
        if self.mode == "direct_typing":
            self._send_text_direct(text)
            LOGGER.info("Transcription typed into active app")
            return True
        raise ValueError(f"Unsupported output mode: {self.mode}")
