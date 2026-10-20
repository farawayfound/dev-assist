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
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QKeyEvent, QMouseEvent, QPainter, QPen, QResizeEvent, QScreen
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)


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
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self._done = False

    # ---- public --------------------------------------------------------------------------
    def begin(self) -> None:
        """Show the picker over the whole desktop and take keyboard focus."""
        self._origin = self._current = None
        self._done = False
        self.setGeometry(virtual_desktop_rect())
        self.show()
        self.raise_()
        self.activateWindow()
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
        self.hide()
        self.selected.emit(x, y, w, h)

    def _cancel(self) -> None:
        if self._done:
            return
        self._done = True
        self.hide()
        self.cancelled.emit()

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
