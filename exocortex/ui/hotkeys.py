"""Global hotkeys via Win32 ``RegisterHotKey`` + a Qt native event filter.

Works while a fullscreen meeting app has focus, unlike Qt's own (window-scoped)
shortcut mechanism. ``parse_hotkey`` turns a text binding like ``"ctrl+alt+space"``
into ``(modifiers, vk)``; :class:`GlobalHotkeys` registers a dict of such bindings and
dispatches ``WM_HOTKEY`` messages back to their callbacks on the Qt main thread.

Pure ``ctypes`` (no pywin32); the real ``user32`` calls are isolated behind a small
``user32`` object (``RegisterHotKey``/``UnregisterHotKey``) so tests can inject a fake
one and exercise ``install()``/``uninstall()`` offline, without a real desktop.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from typing import Any, Callable

from PySide6.QtCore import QAbstractNativeEventFilter, QCoreApplication, QObject, Signal

log = logging.getLogger(__name__)

# ---- RegisterHotKey modifiers ----------------------------------------------------------
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

WM_HOTKEY = 0x0312

#: First id handed to ``RegisterHotKey``; subsequent bindings get consecutive ids.
_FIRST_HOTKEY_ID = 0xB00

_MODIFIER_ALIASES: dict[str, int] = {
    "ctrl": MOD_CONTROL,
    "control": MOD_CONTROL,
    "alt": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN,
    "meta": MOD_WIN,
    "super": MOD_WIN,
}

_NAMED_VKS: dict[str, int] = {
    "space": 0x20,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "enter": 0x0D,
    "esc": 0x1B,
    "tab": 0x09,
    "backspace": 0x08,
    "insert": 0x2D,
    "delete": 0x2E,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "grave": 0xC0,
    "-": 0xBD,
    "=": 0xBB,
    "[": 0xDB,
    "]": 0xDD,
    ";": 0xBA,
    "'": 0xDE,
    ",": 0xBC,
    ".": 0xBE,
    "/": 0xBF,
    "backslash": 0xDC,
}
for _n in range(1, 13):
    _NAMED_VKS[f"f{_n}"] = 0x70 + (_n - 1)
del _n


def parse_hotkey(text: str) -> tuple[int, int]:
    """Parse a ``"+"``-separated binding (e.g. ``"ctrl+alt+space"``) into ``(modifiers, vk)``.

    Case-insensitive and order-independent. Accepts ``"ctrl"``/``"control"``, ``"alt"``,
    ``"shift"``, ``"win"``/``"meta"``/``"super"`` as modifiers, a single letter or digit,
    or one of the named keys (arrows, function keys, punctuation, ...). Raises
    :class:`ValueError` if the text is empty, malformed, names no key, or names more
    than one non-modifier key.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"empty hotkey: {text!r}")
    parts = [p.strip().lower() for p in text.split("+")]
    if any(p == "" for p in parts):
        raise ValueError(f"malformed hotkey: {text!r}")
    modifiers = 0
    vk: int | None = None
    for part in parts:
        if part in _MODIFIER_ALIASES:
            modifiers |= _MODIFIER_ALIASES[part]
            continue
        if vk is not None:
            raise ValueError(f"hotkey names more than one key: {text!r}")
        if part in _NAMED_VKS:
            vk = _NAMED_VKS[part]
        elif len(part) == 1 and (part.isalpha() or part.isdigit()):
            vk = ord(part.upper())
        else:
            raise ValueError(f"unknown key {part!r} in hotkey {text!r}")
    if vk is None:
        raise ValueError(f"hotkey has no key, only modifiers: {text!r}")
    return modifiers, vk


class _NullUser32:
    """Fallback ``user32`` used off Windows: every call fails harmlessly."""

    def RegisterHotKey(self, hwnd: Any, id: int, modifiers: int, vk: int) -> bool:  # noqa: N802, A002
        return False

    def UnregisterHotKey(self, hwnd: Any, id: int) -> bool:  # noqa: N802, A002
        return False


def _default_user32() -> Any:
    """The real ``user32`` on Windows (with argtypes set), else a harmless no-op stand-in."""
    if sys.platform != "win32":
        return _NullUser32()
    dll = ctypes.WinDLL("user32", use_last_error=True)
    dll.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
    dll.RegisterHotKey.restype = wintypes.BOOL
    dll.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    dll.UnregisterHotKey.restype = wintypes.BOOL
    return dll


class GlobalHotkeys(QObject, QAbstractNativeEventFilter):
    """``RegisterHotKey``-based global hotkeys that work while another app has focus.

    ``install(app)`` registers every binding (ids starting at ``0xB00``) and installs
    ``self`` as a native event filter on ``app``; ``uninstall()`` reverses both.
    A binding that fails to register (already taken by another app, or unparseable) is
    recorded in :attr:`failed` instead of raising. On ``WM_HOTKEY``, the matching
    callback runs on the Qt main thread (wrapped in ``try``/``except``) and
    :attr:`triggered` is emitted with the binding text.
    """

    triggered = Signal(str)

    def __init__(
        self,
        bindings: dict[str, Callable[[], None]],
        *,
        user32: Any | None = None,
        parent: QObject | None = None,
    ) -> None:
        QObject.__init__(self, parent)
        QAbstractNativeEventFilter.__init__(self)
        self._bindings = dict(bindings)
        self._user32: Any = user32 if user32 is not None else _default_user32()
        self._app: QCoreApplication | None = None
        self._ids: dict[int, str] = {}
        self._callbacks: dict[int, Callable[[], None]] = {}
        self.failed: list[str] = []

    def install(self, app: QCoreApplication) -> None:
        """Register every binding against ``app``'s native event loop.

        Safe to call only once per instance; a second call while still installed logs
        a warning and does nothing (call :meth:`uninstall` first to re-register).
        """
        if self._app is not None:
            log.warning("GlobalHotkeys.install() called while already installed; ignoring")
            return
        self.failed = []
        self._ids.clear()
        self._callbacks.clear()
        hotkey_id = _FIRST_HOTKEY_ID
        for text, callback in self._bindings.items():
            try:
                modifiers, vk = parse_hotkey(text)
            except ValueError:
                log.error("cannot register hotkey %r: unparseable binding", text)
                self.failed.append(text)
                hotkey_id += 1
                continue
            ok = False
            try:
                ok = bool(self._user32.RegisterHotKey(None, hotkey_id, modifiers | MOD_NOREPEAT, vk))
            except OSError:
                log.exception("RegisterHotKey raised for %r", text)
            if ok:
                self._ids[hotkey_id] = text
                self._callbacks[hotkey_id] = callback
            else:
                log.warning("RegisterHotKey failed for %r (already bound by another app?)", text)
                self.failed.append(text)
            hotkey_id += 1
        app.installNativeEventFilter(self)
        self._app = app

    def uninstall(self) -> None:
        """Unregister every successfully-registered binding and stop filtering events."""
        for hotkey_id in list(self._ids):
            try:
                self._user32.UnregisterHotKey(None, hotkey_id)
            except OSError:
                log.exception("UnregisterHotKey raised for id %s", hotkey_id)
        self._ids.clear()
        self._callbacks.clear()
        if self._app is not None:
            self._app.removeNativeEventFilter(self)
            self._app = None

    def nativeEventFilter(self, eventType: Any, message: Any) -> tuple[bool, int]:
        """Handle ``WM_HOTKEY``; ignore everything else, including the Qt dispatcher event."""
        # QByteArray compares equal to bytes. "windows_dispatcher_MSG" is a second event Qt
        # emits for the same message; reacting to it too would fire callbacks twice.
        if eventType != b"windows_generic_MSG":
            return False, 0
        try:
            msg = wintypes.MSG.from_address(int(message))
        except (TypeError, ValueError, OSError):
            return False, 0
        if msg.message == WM_HOTKEY:
            self._dispatch(int(msg.wParam))
        return False, 0

    def _dispatch(self, hotkey_id: int) -> None:
        text = self._ids.get(hotkey_id)
        callback = self._callbacks.get(hotkey_id)
        if text is None or callback is None:
            return
        try:
            callback()
        except Exception:
            log.exception("hotkey callback for %r raised", text)
        self.triggered.emit(text)
