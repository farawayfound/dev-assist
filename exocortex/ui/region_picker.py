"""Full-desktop rubber-band picker that returns a screen region in physical pixels.

Usage (from the overlay, on the Qt thread)::

    picker = RegionPicker()
    picker.selected.connect(lambda x, y, w, h: actions.set_screen_region(x, y, w, h))
    picker.cancelled.connect(...)
    picker.begin()

The window covers the union of all screens, is translucent and always on top. Drag to select,
Enter/double-click to accept, Esc to cancel. Coordinates are converted from Qt's logical pixels
to physical screen pixels using the device pixel ratio of the screen under the selection, which is
what `mss` expects.

Focus (2026-10-29): on the real Windows platform the picker never takes the keyboard. It is a
``WindowDoesNotAcceptFocus`` window (clicks and drags reach it without activating it) and reads its
own Esc / Enter through a low-level keyboard hook (:class:`~exocortex.ui.llhooks.BareKeyHook`,
swallowing the two keys while it is up), so the window the user was working in stays focused
throughout and nothing has to be given back when the picker closes. Where the hook is unavailable
(another platform, the offscreen tests) the picker takes the focus as any window would and, on
Windows, hands it back to the window that had it (``win32.activate_window``) when it closes --
without that step Windows activates the picker's owner, the overlay.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QKeyEvent, QMouseEvent, QPainter, QPen, QResizeEvent, QScreen
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)

try:  # the foreground-window helpers are optional (missing on non-Windows / in tests)
    from exocortex.ui import win32 as _win32
except Exception:  # noqa: BLE001
    _win32 = None  # type: ignore[assignment]

try:  # the Esc / Enter key hook (Windows-only; without it the picker takes the focus instead)
    from exocortex.ui.llhooks import BareKeyHook as _KeyHook
except Exception:  # noqa: BLE001
    _KeyHook = None  # type: ignore[assignment]

#: How long Esc / Enter may be held and still count (the hook reports a key on its release).
_KEY_HOLD_S = 3.0


def _real_windows() -> bool:
    """True only on real Windows with the native "windows" Qt platform (never under the offscreen
    QPA the tests use): the one case the picker may hook the keyboard or touch the foreground."""
    return sys.platform == "win32" and QGuiApplication.platformName() == "windows"


def virtual_desktop_rect() -> QRect:
    """Union of all screens' geometry in logical pixels."""
    rect = QRect()
    for s in QGuiApplication.screens():
        rect = rect.united(s.geometry())
    return rect


def logical_to_physical(rect: QRect) -> tuple[int, int, int, int]:
    """Convert a logical-pixel rect to physical screen pixels (x, y, w, h)."""
    screen: QScreen | None = QGuiApplication.screenAt(rect.center()) or QGuiApplication.primaryScreen()
    dpr = float(screen.devicePixelRatio()) if screen else 1.0
    if screen is not None and dpr != 1.0:
        # Physical origin of the screen = logical origin * dpr (Qt places logical geometries so that
        # each screen's logical origin corresponds to its physical origin divided by its own dpr).
        g = screen.geometry()
        px = int(round(g.x() * dpr + (rect.x() - g.x()) * dpr))
        py = int(round(g.y() * dpr + (rect.y() - g.y()) * dpr))
        return px, py, int(round(rect.width() * dpr)), int(round(rect.height() * dpr))
    return rect.x(), rect.y(), rect.width(), rect.height()


class RegionPicker(QWidget):
    """Translucent full-desktop overlay for dragging out a rectangle."""

    selected = Signal(int, int, int, int)   # x, y, w, h in physical screen pixels
    cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # 2026-10-29: with the key hook available the picker never becomes the active window (the
        # flag maps to WS_EX_NOACTIVATE; clicks and drags still arrive) -- see the module docstring.
        self._keys_hooked = _KeyHook is not None and _real_windows()
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        if self._keys_hooked:
            flags |= Qt.WindowType.WindowDoesNotAcceptFocus
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self._done = False
        self._key_hook: Any = None          # the BareKeyHook while the picker is up (hooked path)
        self._prev_foreground = 0           # the window that had the keyboard when begin() ran

    # ---- public --------------------------------------------------------------------------
    def begin(self) -> None:
        """Show the picker over the whole desktop.

        The keyboard stays with the window the user was in: Esc / Enter arrive through the key
        hook (see the module docstring). Only where that hook does not exist does the picker
        take the focus, and then it gives it back when it closes.
        """
        self._origin = self._current = None
        self._done = False
        self._prev_foreground = 0
        self.setGeometry(virtual_desktop_rect())
        self._prev_foreground = self._foreground_now()
        self.show()
        self.raise_()
        if self._keys_hooked:
            if not self._install_key_hook():
                log.warning("region picker: the Esc / Enter key hook is unavailable; right-click cancels, "
                            "double-click accepts")
            return   # never activated on this path (Qt refuses it for the flag, and nothing needs it)
        self.activateWindow()                                    # keyboard focus for Esc / Enter
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def selection_rect(self) -> QRect:
        """Rectangle between the press point and the current point (exclusive size, may be empty)."""
        if self._origin is None or self._current is None:
            return QRect()
        x0, x1 = sorted((self._origin.x(), self._current.x()))
        y0, y1 = sorted((self._origin.y(), self._current.y()))
        return QRect(x0, y0, x1 - x0, y1 - y0)

    # ---- internals -----------------------------------------------------------------------
    def _finish(self) -> None:
        if self._done:
            return
        rect = self.selection_rect()
        if rect.width() < 8 or rect.height() < 8:
            self._cancel()
            return
        self._done = True
        # translate widget-local coordinates to global logical coordinates
        global_rect = QRect(self.mapToGlobal(rect.topLeft()), rect.size())
        x, y, w, h = logical_to_physical(global_rect)
        self._end()
        self.selected.emit(x, y, w, h)

    def _cancel(self) -> None:
        if self._done:
            return
        self._done = True
        self._end()
        self.cancelled.emit()

    # ---- focus (2026-10-29) --------------------------------------------------------------
    def _end(self) -> None:
        """Take the picker down: hide it, release the key hook and, when the picker had to take the
        focus (the fallback path), give it back to the window that had it."""
        self.hide()
        self._uninstall_key_hook()
        self._restore_foreground()

    def _install_key_hook(self) -> bool:
        """Hook Esc (cancel) and Enter (accept) for as long as the picker is up. Returns whether the
        hook is in place; a failure is logged, and the mouse (right-click, double-click) remains."""
        if self._key_hook is None:
            try:
                self._key_hook = _KeyHook(
                    {"esc": lambda: QTimer.singleShot(0, self, self._cancel),
                     "enter": lambda: QTimer.singleShot(0, self, self._finish)},
                    max_tap_s=_KEY_HOLD_S,
                )
            except Exception:  # noqa: BLE001
                log.exception("region picker: the key hook could not be created")
                return False
        if self._key_hook.installed:
            return True
        if self._key_hook.failed:
            return False
        try:
            return bool(self._key_hook.install())
        except Exception:  # noqa: BLE001
            log.exception("region picker: the key hook could not be installed")
            return False

    def _uninstall_key_hook(self) -> None:
        hook = self._key_hook
        if hook is None or not hook.installed:
            return
        try:
            hook.uninstall()
        except Exception:  # noqa: BLE001
            log.exception("region picker: the key hook could not be removed")

    def _foreground_now(self) -> int:
        """The window that has the keyboard, as a handle (0 when unknown or not on real Windows)."""
        if _win32 is None or not _real_windows():
            return 0
        try:
            return int(_win32.foreground_window())
        except Exception:  # noqa: BLE001
            log.debug("region picker: foreground_window() failed", exc_info=True)
            return 0

    def _restore_foreground(self) -> None:
        """Hand the keyboard back to the window that had it when the picker appeared -- only if it
        moved (the hooked path never takes it, so this is a no-op there)."""
        prev, self._prev_foreground = self._prev_foreground, 0
        if not prev or _win32 is None or not _real_windows():
            return
        try:
            if int(_win32.foreground_window()) == prev:
                return
            if not _win32.activate_window(prev):
                log.debug("region picker: could not give the keyboard back to hwnd=%s", prev)
        except Exception:  # noqa: BLE001
            log.debug("region picker: restoring the foreground window failed", exc_info=True)

    # ---- events --------------------------------------------------------------------------
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.update()

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._origin = e.position().toPoint()
            self._current = self._origin
            self.update()
        elif e.button() == Qt.MouseButton.RightButton:
            self._cancel()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._origin is not None:
            self._current = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton and self._origin is not None:
            self._current = e.position().toPoint()
            self.update()
            self._finish()

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        self._finish()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self._cancel()
        elif e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish()
        else:
            super().keyPressEvent(e)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 90))
        sel = self.selection_rect()
        if not sel.isNull():
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            p.fillRect(sel, Qt.GlobalColor.transparent)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            pen = QPen(QColor(125, 211, 252), 2)
            p.setPen(pen)
            p.drawRect(sel.adjusted(0, 0, -1, -1))
            label = f"{sel.width()} × {sel.height()}   (Enter = accept, Esc = cancel)"
            p.setPen(QColor(232, 234, 240))
            p.drawText(sel.left() + 6, max(14, sel.top() - 6), label)
        else:
            p.setPen(QColor(232, 234, 240))
            hint = "Drag to select the screen region to read  •  Esc to cancel"
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, hint)
        p.end()
