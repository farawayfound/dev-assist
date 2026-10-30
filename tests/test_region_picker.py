"""Offscreen tests for the screen-region picker."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Any  # noqa: E402

from PySide6.QtCore import QPoint, QRect, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import exocortex.ui.region_picker as region_picker_mod  # noqa: E402
from exocortex.ui.region_picker import RegionPicker, logical_to_physical, virtual_desktop_rect  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_virtual_desktop_rect_is_non_empty(app: QApplication) -> None:
    r = virtual_desktop_rect()
    assert r.width() > 0 and r.height() > 0


def test_logical_to_physical_identity_at_dpr_1(app: QApplication) -> None:
    x, y, w, h = logical_to_physical(QRect(10, 20, 300, 200))
    # offscreen platform reports dpr 1.0
    assert (w, h) == (300, 200) and x == 10 and y == 20


def test_drag_emits_selection(app: QApplication) -> None:
    picker = RegionPicker()
    got: list[tuple[int, int, int, int]] = []
    picker.selected.connect(lambda x, y, w, h: got.append((x, y, w, h)))
    picker.begin()
    QTest.qWait(20)
    QTest.mousePress(picker, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(50, 60))
    QTest.mouseMove(picker, QPoint(250, 210))
    QTest.mouseRelease(picker, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(250, 210))
    QTest.qWait(20)
    assert len(got) == 1
    x, y, w, h = got[0]
    assert (w, h) == (200, 150)
    assert (x, y) == (picker.x() + 50, picker.y() + 60)
    assert not picker.isVisible()


def test_escape_cancels_and_tiny_selection_cancels(app: QApplication) -> None:
    picker = RegionPicker()
    cancelled: list[int] = []
    picker.cancelled.connect(lambda: cancelled.append(1))
    picker.begin()
    QTest.qWait(20)
    QTest.keyClick(picker, Qt.Key.Key_Escape)
    assert cancelled == [1] and not picker.isVisible()

    picker.begin()
    QTest.mousePress(picker, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(10, 10))
    QTest.mouseRelease(picker, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(12, 12))
    QTest.qWait(20)
    assert cancelled == [1, 1]


def test_picker_shows_when_begun(app: QApplication) -> None:
    """dev-assist is a visible copilot: the picker just appears (no capture gating)."""
    picker = RegionPicker()
    picker.begin()
    QTest.qWait(10)
    assert picker.isVisible()
    picker._cancel()


# =====================================================================================
# focus (2026-10-29): the picker leaves the keyboard with the window the user was in
# =====================================================================================
class _StubWin32:
    """Stand-in for ``exocortex.ui.win32``: the picker calls ``foreground_window`` /
    ``activate_window`` for the focus hand-back."""

    def __init__(self, foreground: int = 0) -> None:
        self.foreground = foreground            # what GetForegroundWindow would report
        self.activated: list[int] = []

    def foreground_window(self) -> int:
        return self.foreground

    def activate_window(self, hwnd: int) -> bool:
        self.activated.append(hwnd)
        self.foreground = hwnd
        return True


class _SpyKeyHook:
    """Stand-in for ``llhooks.BareKeyHook``: records its bindings and install / uninstall, hooks
    nothing (no test may touch the real keyboard)."""

    instances: list["_SpyKeyHook"] = []

    def __init__(self, bindings: Any, on_tap: Any = None, **kw: Any) -> None:
        self.bindings = dict(bindings) if not isinstance(bindings, str) else {bindings: on_tap}
        self.kw = kw
        self.installed = False
        self.failed = False
        self.installs = 0
        self.uninstalls = 0
        _SpyKeyHook.instances.append(self)

    def install(self) -> bool:
        self.installed = True
        self.installs += 1
        return True

    def uninstall(self) -> None:
        self.installed = False
        self.uninstalls += 1


class _FakeGuiApp:
    """The real ``QGuiApplication`` with ``platformName()`` reporting the ``windows`` platform -
    the one thing the picker gates its key hook and focus hand-back on, since these tests run
    under ``offscreen``."""

    @staticmethod
    def platformName() -> str:
        return "windows"

    @staticmethod
    def screens():
        return QGuiApplication.screens()

    @staticmethod
    def screenAt(point):
        return QGuiApplication.screenAt(point)

    @staticmethod
    def primaryScreen():
        return QGuiApplication.primaryScreen()


def _windows_picker(
    monkeypatch: pytest.MonkeyPatch, *, key_hook: Any = None, foreground: int = 0
) -> tuple[RegionPicker, _StubWin32]:
    """A picker that believes it runs on a real Windows desktop with a stub ``win32``. The Esc /
    Enter key hook is ``key_hook`` -- a spy class, or None for the take-the-focus fallback -- never
    the real one."""
    stub = _StubWin32(foreground)
    monkeypatch.setattr(region_picker_mod, "_win32", stub)
    monkeypatch.setattr(region_picker_mod, "QGuiApplication", _FakeGuiApp())
    monkeypatch.setattr(region_picker_mod, "_KeyHook", key_hook)
    return RegionPicker(), stub


def test_on_windows_the_picker_never_takes_the_focus_and_reads_esc_enter_through_the_hook(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SpyKeyHook.instances = []
    picker, stub = _windows_picker(monkeypatch, key_hook=_SpyKeyHook, foreground=0xBEEF)
    activations: list[int] = []
    picker.activateWindow = lambda: activations.append(1)   # type: ignore[method-assign]
    cancelled: list[int] = []
    picker.cancelled.connect(lambda: cancelled.append(1))
    assert picker.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assert picker.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

    picker.begin()
    QTest.qWait(10)
    assert picker.isVisible()
    assert activations == []
    (hook,) = _SpyKeyHook.instances
    assert hook.installed and set(hook.bindings) == {"esc", "enter"}
    hook.bindings["esc"]()                 # the hook reports Esc: cancel, deferred off the hook
    QTest.qWait(30)
    assert cancelled == [1] and not picker.isVisible()
    assert not hook.installed and hook.uninstalls == 1
    assert stub.activated == []            # the keyboard never moved, so nothing is given back

    got: list[tuple[int, int]] = []
    picker.selected.connect(lambda x, y, w, h: got.append((w, h)))
    picker.begin()
    QTest.qWait(10)
    assert hook.installed and hook.installs == 2 and len(_SpyKeyHook.instances) == 1   # reused
    QTest.mousePress(picker, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(50, 60))
    QTest.mouseMove(picker, QPoint(250, 210))
    hook.bindings["enter"]()               # Enter accepts the rectangle dragged so far
    QTest.qWait(30)
    assert got == [(200, 150)] and not picker.isVisible() and not hook.installed
    assert activations == [] and stub.activated == []


def test_without_the_hook_the_picker_takes_the_focus_and_gives_it_back(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback (no hook): the picker activates itself as before, and on close hands the
    keyboard back to the window that had it -- hiding an active owned window makes Windows activate
    its owner, the overlay, which is where the focus used to end up after every pick."""
    picker, stub = _windows_picker(monkeypatch, key_hook=None, foreground=0xBEEF)
    activations: list[int] = []
    picker.activateWindow = lambda: activations.append(1)   # type: ignore[method-assign]
    assert not (picker.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus)
    picker.begin()
    QTest.qWait(10)
    assert activations == [1]
    stub.foreground = 0x1234               # the owner got activated when the picker went down
    picker._cancel()
    assert stub.activated == [0xBEEF]
    # a foreground that did not move is left alone
    picker.begin()
    QTest.qWait(10)
    picker._cancel()
    assert stub.activated == [0xBEEF]


def test_a_hook_that_fails_to_install_leaves_the_mouse_gestures_and_never_activates(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailingHook(_SpyKeyHook):
        def install(self) -> bool:
            self.failed = True
            return False

    _SpyKeyHook.instances = []
    picker, stub = _windows_picker(monkeypatch, key_hook=_FailingHook, foreground=0xBEEF)
    activations: list[int] = []
    picker.activateWindow = lambda: activations.append(1)   # type: ignore[method-assign]
    cancelled: list[int] = []
    picker.cancelled.connect(lambda: cancelled.append(1))
    picker.begin()
    QTest.qWait(10)
    assert picker.isVisible() and activations == []       # a no-activate window: never activated
    QTest.mouseClick(picker, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, QPoint(30, 30))
    assert cancelled == [1] and not picker.isVisible()
    assert stub.activated == []
