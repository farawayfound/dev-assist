"""Tests for :mod:`exocortex.ui.win32` and :mod:`exocortex.ui.hotkeys`.

Non-live tests run fully offline: no real hotkey ever reaches the OS (a fake ``user32``
stands in for :class:`~exocortex.ui.hotkeys.GlobalHotkeys`), and the win32 ctypes helpers
are exercised with ``hwnd=0`` and with ``win32.IS_WINDOWS`` monkeypatched to ``False``, so
they never depend on a real window. They use a shared ``QCoreApplication`` (no GUI platform
plugin needed, since ``installNativeEventFilter`` is a ``QCoreApplication`` feature).

Live tests (``@pytest.mark.live``, skipped by default per ``pytest.ini``) open a small real
frameless window and register a real global hotkey; run them explicitly on a real desktop
with ``QT_QPA_PLATFORM`` unset::

    pytest tests/test_win32_hotkeys.py -m live -q
"""
from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from typing import Any, Callable

import pytest

from exocortex.ui import win32
from exocortex.ui.hotkeys import (
    MOD_ALT,
    MOD_CONTROL,
    MOD_NOREPEAT,
    MOD_SHIFT,
    MOD_WIN,
    WM_HOTKEY,
    GlobalHotkeys,
    parse_hotkey,
)

# ---------------------------------------------------------------------------------------
# Qt application: QCoreApplication is enough for these tests (native event filters and
# QObject signals are QtCore features, so no GUI platform plugin is ever loaded here).
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qcoreapp() -> Any:
    """One ``QCoreApplication`` reused by every non-live test in this session."""
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


class FakeUser32:
    """Injectable stand-in for ``user32`` so :class:`GlobalHotkeys` can be tested offline.

    Records every ``RegisterHotKey``/``UnregisterHotKey`` call and can simulate a specific
    binding already being taken by another app (real Win32 error 1409) by listing its
    ``(modifiers, vk)`` in ``fail_keys``.
    """

    def __init__(self, fail_keys: set[tuple[int, int]] | None = None) -> None:
        self.registered: list[tuple[int, int, int]] = []  # (id, modifiers, vk)
        self.unregistered: list[int] = []
        self._fail_keys = fail_keys or set()

    def RegisterHotKey(self, hwnd: Any, id: int, modifiers: int, vk: int) -> bool:  # noqa: N802, A002
        self.registered.append((id, modifiers, vk))
        return (modifiers & ~MOD_NOREPEAT, vk) not in self._fail_keys

    def UnregisterHotKey(self, hwnd: Any, id: int) -> bool:  # noqa: N802, A002
        self.unregistered.append(id)
        return True


def _msg(message: int, wparam: int) -> wintypes.MSG:
    return wintypes.MSG(None, message, wparam, 0, 0, wintypes.POINT(0, 0))


# ---------------------------------------------------------------------------------------
# parse_hotkey
# ---------------------------------------------------------------------------------------

PARSE_CASES: list[tuple[str, tuple[int, int]]] = [
    ("ctrl+alt+space", (MOD_CONTROL | MOD_ALT, 0x20)),
    ("Ctrl+Alt+Space", (MOD_CONTROL | MOD_ALT, 0x20)),  # case-insensitive
    ("alt+f9", (MOD_ALT, 0x78)),
    ("shift+a", (MOD_SHIFT, ord("A"))),
    ("win+1", (MOD_WIN, ord("1"))),
    ("meta+q", (MOD_WIN, ord("Q"))),
    ("super+q", (MOD_WIN, ord("Q"))),
    ("control+alt+left", (MOD_CONTROL | MOD_ALT, 0x25)),
    ("alt+shift+up", (MOD_ALT | MOD_SHIFT, 0x26)),
    ("shift+alt+up", (MOD_ALT | MOD_SHIFT, 0x26)),  # order-independent
    ("ctrl+-", (MOD_CONTROL, 0xBD)),
    ("ctrl+=", (MOD_CONTROL, 0xBB)),
    ("ctrl+,", (MOD_CONTROL, 0xBC)),
    ("ctrl+alt+x", (MOD_CONTROL | MOD_ALT, ord("X"))),
    ("f12", (0, 0x7B)),
    ("delete", (0, 0x2E)),
    ("ctrl+alt+down", (MOD_CONTROL | MOD_ALT, 0x28)),
]


@pytest.mark.parametrize("text,expected", PARSE_CASES)
def test_parse_hotkey_valid(text: str, expected: tuple[int, int]) -> None:
    assert parse_hotkey(text) == expected


PARSE_ERROR_CASES = [
    "",
    "   ",
    "ctrl+alt",  # only modifiers, no key
    "ctrl+unknownkey",  # unknown key name
    "ctrl+a+b",  # two keys named
    "ctrl++",  # empty part
    "++space",  # leading empty part
]


@pytest.mark.parametrize("text", PARSE_ERROR_CASES)
def test_parse_hotkey_errors(text: str) -> None:
    with pytest.raises(ValueError):
        parse_hotkey(text)


# ---------------------------------------------------------------------------------------
# win32 ctypes helpers: graceful behaviour with hwnd=0 and off "Windows"
# ---------------------------------------------------------------------------------------


def test_set_click_through_hwnd_zero_does_not_raise() -> None:
    win32.set_click_through(0, True)
    win32.set_click_through(0, False)


def test_set_no_activate_hwnd_zero_does_not_raise() -> None:
    win32.set_no_activate(0, True)
    win32.set_no_activate(0, False)


def test_set_topmost_hwnd_zero_does_not_raise() -> None:
    win32.set_topmost(0)


def test_last_error_message_zero_code() -> None:
    assert win32.last_error_message(0) == "error 0"


def test_last_error_message_known_code_is_human_readable() -> None:
    if not win32.IS_WINDOWS:
        pytest.skip("Windows-only")
    text = win32.last_error_message(1400)  # ERROR_INVALID_WINDOW_HANDLE
    assert "1400" in text
    assert text != "error 1400"  # FormatMessageW actually resolved a message


def test_off_windows_helpers_never_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate running off Windows: every helper must degrade gracefully, never raise."""
    monkeypatch.setattr(win32, "IS_WINDOWS", False)
    win32.set_click_through(12345, True)
    win32.set_no_activate(12345, True)
    win32.set_topmost(12345)
    assert win32.last_error_message(1400) == "error 1400"


# ---------------------------------------------------------------------------------------
# GlobalHotkeys.install() / uninstall() against a fake user32
# ---------------------------------------------------------------------------------------


def test_install_registers_each_binding_with_sequential_ids_and_noreapeat(qcoreapp: Any) -> None:
    calls: list[str] = []
    fake = FakeUser32()
    bindings: dict[str, Callable[[], None]] = {
        "ctrl+alt+space": lambda: calls.append("toggle"),
        "ctrl+alt+a": lambda: calls.append("answer"),
    }
    gh = GlobalHotkeys(bindings, user32=fake)
    gh.install(qcoreapp)
    try:
        assert gh.failed == []
        assert len(fake.registered) == 2
        ids = [rid for rid, _, _ in fake.registered]
        assert ids == [0xB00, 0xB01]  # sequential, starting at 0xB00
        for _, modifiers, _ in fake.registered:
            assert modifiers & MOD_NOREPEAT
    finally:
        gh.uninstall()
    assert sorted(fake.unregistered) == [0xB00, 0xB01]


def test_install_records_failed_binding_without_raising(qcoreapp: Any) -> None:
    mods, vk = parse_hotkey("ctrl+alt+n")
    fake = FakeUser32(fail_keys={(mods, vk)})
    bindings: dict[str, Callable[[], None]] = {"ctrl+alt+space": lambda: None, "ctrl+alt+n": lambda: None}
    gh = GlobalHotkeys(bindings, user32=fake)
    gh.install(qcoreapp)
    try:
        assert gh.failed == ["ctrl+alt+n"]
        assert len(fake.registered) == 2  # both attempted
    finally:
        gh.uninstall()
    # only the succeeding one was ever unregistered
    assert len(fake.unregistered) == 1


def test_install_records_unparseable_binding_as_failed(qcoreapp: Any) -> None:
    fake = FakeUser32()
    gh = GlobalHotkeys({"ctrl+bogus": lambda: None}, user32=fake)
    gh.install(qcoreapp)
    try:
        assert gh.failed == ["ctrl+bogus"]
        assert fake.registered == []  # never even reached RegisterHotKey
    finally:
        gh.uninstall()


def test_double_install_is_a_noop(qcoreapp: Any) -> None:
    fake = FakeUser32()
    gh = GlobalHotkeys({"ctrl+alt+space": lambda: None}, user32=fake)
    gh.install(qcoreapp)
    try:
        gh.install(qcoreapp)
        assert len(fake.registered) == 1
    finally:
        gh.uninstall()


def test_uninstall_without_install_does_not_raise(qcoreapp: Any) -> None:
    gh = GlobalHotkeys({"ctrl+alt+space": lambda: None}, user32=FakeUser32())
    gh.uninstall()  # must be a no-op, not an error


def test_uninstall_twice_after_install_is_safe(qcoreapp: Any) -> None:
    fake = FakeUser32()
    gh = GlobalHotkeys({"ctrl+alt+space": lambda: None}, user32=fake)
    gh.install(qcoreapp)
    gh.uninstall()
    gh.uninstall()  # second call must also be a no-op, not an error
    assert sorted(fake.unregistered) == [0xB00]


def test_reinstall_after_uninstall_reregisters_and_resets_failed(qcoreapp: Any) -> None:
    """uninstall() then install() again must fully re-register (per the class docstring)."""
    fake = FakeUser32()
    gh = GlobalHotkeys({"ctrl+alt+space": lambda: None, "ctrl+alt+a": lambda: None}, user32=fake)
    gh.install(qcoreapp)
    assert gh.failed == []
    assert len(fake.registered) == 2
    gh.uninstall()
    assert sorted(fake.unregistered) == [0xB00, 0xB01]

    fake2 = FakeUser32()
    gh._user32 = fake2  # swap in a fresh fake to prove install() truly runs RegisterHotKey again
    gh.install(qcoreapp)
    try:
        assert gh.failed == []
        assert len(fake2.registered) == 2
        ids = [rid for rid, _, _ in fake2.registered]
        assert ids == [0xB00, 0xB01]  # ids restart from the same base on re-install
    finally:
        gh.uninstall()
    assert sorted(fake2.unregistered) == [0xB00, 0xB01]


def test_install_with_empty_bindings_is_safe(qcoreapp: Any) -> None:
    fake = FakeUser32()
    gh = GlobalHotkeys({}, user32=fake)
    gh.install(qcoreapp)
    try:
        assert gh.failed == []
        assert fake.registered == []
    finally:
        gh.uninstall()  # must not raise with nothing registered
    assert fake.unregistered == []


def test_global_hotkeys_accepts_parent_kwarg(qcoreapp: Any) -> None:
    from PySide6.QtCore import QObject

    parent = QObject()
    fake = FakeUser32()
    gh = GlobalHotkeys({"ctrl+alt+space": lambda: None}, user32=fake, parent=parent)
    assert gh.parent() is parent
    gh.install(qcoreapp)
    gh.uninstall()


def test_gap_in_hotkey_ids_when_a_binding_is_unparseable(qcoreapp: Any) -> None:
    """An unparseable binding still consumes an id slot; the next binding's id has a gap."""
    fake = FakeUser32()
    bindings: dict[str, Callable[[], None]] = {
        "ctrl+bogus": lambda: None,
        "ctrl+alt+space": lambda: None,
    }
    gh = GlobalHotkeys(bindings, user32=fake)
    gh.install(qcoreapp)
    try:
        assert gh.failed == ["ctrl+bogus"]
        assert len(fake.registered) == 1
        assert fake.registered[0][0] == 0xB01  # id 0xB00 was consumed by the failed parse
    finally:
        gh.uninstall()


def test_default_user32_is_null_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    from exocortex.ui import hotkeys as hk

    monkeypatch.setattr(hk.sys, "platform", "linux")
    user32 = hk._default_user32()
    assert isinstance(user32, hk._NullUser32)
    assert user32.RegisterHotKey(None, 1, 0, 0) is False
    assert user32.UnregisterHotKey(None, 1) is False


def test_default_user32_is_real_dll_with_argtypes_on_windows() -> None:
    from exocortex.ui import hotkeys as hk

    if sys.platform != "win32":
        pytest.skip("Windows-only")
    user32 = hk._default_user32()
    assert user32.RegisterHotKey.argtypes is not None
    assert user32.UnregisterHotKey.argtypes is not None


# ---------------------------------------------------------------------------------------
# nativeEventFilter dispatch
# ---------------------------------------------------------------------------------------


def test_native_event_filter_dispatches_matching_hotkey(qcoreapp: Any) -> None:
    calls: list[str] = []
    triggered: list[str] = []
    fake = FakeUser32()
    gh = GlobalHotkeys({"ctrl+alt+f9": lambda: calls.append("fired")}, user32=fake)
    gh.triggered.connect(triggered.append)
    gh.install(qcoreapp)
    try:
        hotkey_id = fake.registered[0][0]
        msg = _msg(WM_HOTKEY, hotkey_id)
        handled, result = gh.nativeEventFilter(b"windows_generic_MSG", ctypes.addressof(msg))
        assert handled is False
        assert result == 0
        assert calls == ["fired"]
        assert triggered == ["ctrl+alt+f9"]
    finally:
        gh.uninstall()


def test_native_event_filter_ignores_dispatcher_msg(qcoreapp: Any) -> None:
    calls: list[str] = []
    fake = FakeUser32()
    gh = GlobalHotkeys({"ctrl+alt+f9": lambda: calls.append("fired")}, user32=fake)
    gh.install(qcoreapp)
    try:
        hotkey_id = fake.registered[0][0]
        msg = _msg(WM_HOTKEY, hotkey_id)
        gh.nativeEventFilter(b"windows_dispatcher_MSG", ctypes.addressof(msg))
        assert calls == []
    finally:
        gh.uninstall()


def test_native_event_filter_ignores_non_hotkey_messages(qcoreapp: Any) -> None:
    calls: list[str] = []
    fake = FakeUser32()
    gh = GlobalHotkeys({"ctrl+alt+f9": lambda: calls.append("fired")}, user32=fake)
    gh.install(qcoreapp)
    try:
        hotkey_id = fake.registered[0][0]
        msg = _msg(0x0010, hotkey_id)  # WM_CLOSE, not WM_HOTKEY
        gh.nativeEventFilter(b"windows_generic_MSG", ctypes.addressof(msg))
        assert calls == []
    finally:
        gh.uninstall()


def test_native_event_filter_ignores_unknown_hotkey_id(qcoreapp: Any) -> None:
    calls: list[str] = []
    fake = FakeUser32()
    gh = GlobalHotkeys({"ctrl+alt+f9": lambda: calls.append("fired")}, user32=fake)
    gh.install(qcoreapp)
    try:
        msg = _msg(WM_HOTKEY, 0xFFFF)  # id nobody registered
        gh.nativeEventFilter(b"windows_generic_MSG", ctypes.addressof(msg))
        assert calls == []
    finally:
        gh.uninstall()


def test_native_event_filter_callback_exception_is_caught_and_logged(
    qcoreapp: Any, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeUser32()

    def boom() -> None:
        raise RuntimeError("boom")

    gh = GlobalHotkeys({"ctrl+alt+f9": boom}, user32=fake)
    gh.install(qcoreapp)
    try:
        hotkey_id = fake.registered[0][0]
        msg = _msg(WM_HOTKEY, hotkey_id)
        with caplog.at_level(logging.ERROR):
            handled, result = gh.nativeEventFilter(b"windows_generic_MSG", ctypes.addressof(msg))
        assert (handled, result) == (False, 0)  # never raised out of nativeEventFilter
        assert any("boom" in r.message or "raised" in r.message for r in caplog.records)
    finally:
        gh.uninstall()


# ---------------------------------------------------------------------------------------
# Live tests: real desktop window + real global hotkey (run with `-m live`)
# ---------------------------------------------------------------------------------------


def _read_exstyle(hwnd: int) -> int:
    """Independent (not exocortex.ui.win32) read of GWL_EXSTYLE, for black-box assertions."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongPtrW.restype = ctypes.c_longlong
    return int(user32.GetWindowLongPtrW(wintypes.HWND(hwnd), win32.GWL_EXSTYLE))


@pytest.mark.live
def test_live_click_through_on_real_window() -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication([])
    widget = QWidget()
    widget.setWindowFlags(
        Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
    )
    widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    widget.resize(120, 80)
    widget.show()
    app.processEvents()
    try:
        hwnd = int(widget.winId())
        win32.set_click_through(hwnd, True)
        assert _read_exstyle(hwnd) & win32.WS_EX_TRANSPARENT

        win32.set_click_through(hwnd, False)
        assert not (_read_exstyle(hwnd) & win32.WS_EX_TRANSPARENT)
    finally:
        widget.close()
        app.processEvents()


@pytest.mark.live
def test_live_register_real_hotkey_reports_no_failure() -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    gh = GlobalHotkeys({"ctrl+alt+f9": lambda: None})
    gh.install(app)
    try:
        assert "ctrl+alt+f9" not in gh.failed
    finally:
        gh.uninstall()
