"""GhostText: the inline completion overlay drawn at the caret of another application.

A tiny frameless, transparent, always-on-top window that never takes focus and lets every mouse
event pass through to the app underneath (``WindowTransparentForInput`` + ``WS_EX_TRANSPARENT``
+ ``WS_EX_NOACTIVATE``). It is a visible overlay like the rest of the app: nothing hides it from
screen capture.

It is positioned in PHYSICAL screen pixels straight from the caret coordinates the accessibility
APIs report (``win32.set_window_rect_physical``), so it lines up with the caret on any monitor at
any DPI without going through Qt's logical screen mapping. The font is sized to the target's
font (``CompletionEvent.font_px``, physical px, or the caret height when unknown) so that what
the user types over the ghost text lands where the ghost text was.

Continuation lines are aligned with the start of the caret's line: the width of the text already
on that line (``current_line``) is measured in the matched font and subtracted from the caret x.
"""
from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QGuiApplication, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from exocortex.bus import CompletionEvent
from exocortex.config import Config

log = logging.getLogger(__name__)

try:
    from exocortex.ui import win32
except Exception:  # pragma: no cover - win32.py missing
    win32 = None  # type: ignore[assignment]

_PAD_X = 2
_PAD_Y = 1
_MIN_FONT_PX = 8.0
_MIN_LINE_H = 6
_MAX_FONT_PX = 160.0
#: Typical ratio of a text line's height to its font pixel size, used when only the caret
#: height is known.
_LINE_TO_FONT = 1.3
_MONO_FAMILIES = ["Cascadia Mono", "Consolas", "Courier New", "monospace"]
_UI_FAMILIES = ["Segoe UI", "Segoe UI Variable", "sans-serif"]


def _is_windows_platform() -> bool:
    return sys.platform == "win32" and QGuiApplication.platformName() == "windows"


class GhostText(QWidget):
    """Draws the current completion at the caret. Drive it with ``show_completion``/``hide_completion``."""

    def __init__(self, cfg: Config, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._lines: list[str] = []
        self._offsets: list[float] = []      # logical x offset of each line relative to the caret
        self._font = QFont()
        self._line_h = 0.0                    # logical px
        self._ascent = 0.0
        self._origin_x = 0.0                  # logical x of the caret inside this widget
        self._last_key: tuple = ()
        self._primed = False
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    # ---- win32 helpers ------------------------------------------------------------------------
    def _prime(self) -> None:
        """Apply click-through / no-activate on the native window once (so the ghost never grabs
        the mouse or the keyboard from the app the user is typing in)."""
        if self._primed:
            return
        self._primed = True
        if not (_is_windows_platform() and win32 is not None):
            return
        try:
            hwnd = int(self.winId())
            win32.set_click_through(hwnd, True)
            win32.set_no_activate(hwnd, True)
        except Exception:
            log.exception("ghost: win32 priming failed")

    # ---- public API -------------------------------------------------------------------------
    def show_completion(self, ev: CompletionEvent) -> None:
        if not ev.visible or not ev.text.strip():
            self.hide_completion()
            return
        self._prime()
        dpr = self._dpr_at(ev.x, ev.y)
        line_h = ev.line_h if ev.line_h >= _MIN_LINE_H else 0   # a 1-px hidden-input caret is not a line
        if ev.font_px > 0:
            font_px_phys = ev.font_px
        elif line_h > 0:
            font_px_phys = line_h / _LINE_TO_FONT
        else:
            font_px_phys = self._cfg.completion.default_font_pt * dpr * 96.0 / 72.0
        if line_h > 0:
            font_px_phys = min(font_px_phys, line_h * 0.95)
        font_px_phys = max(_MIN_FONT_PX, min(_MAX_FONT_PX, font_px_phys))
        font_px = font_px_phys / dpr

        self._font = QFont()
        self._font.setFamilies(self._families(ev))
        self._font.setPixelSize(max(1, round(font_px)))
        fm = QFontMetricsF(self._font)
        self._ascent = fm.ascent()
        natural_h = fm.height()
        self._line_h = max(natural_h, line_h / dpr) if line_h > 0 else natural_h

        self._lines = ev.text.split("\n")
        cur_w = fm.horizontalAdvance(ev.current_line) if ev.current_line else 0.0
        self._offsets = [0.0] + [-cur_w] * (len(self._lines) - 1)
        widths = [fm.horizontalAdvance(line) for line in self._lines]
        left = min(self._offsets)                     # <= 0
        right = max(off + w for off, w in zip(self._offsets, widths))
        self._origin_x = -left + _PAD_X
        width_l = (right - left) + 2 * _PAD_X + 2
        height_l = self._line_h * len(self._lines) + 2 * _PAD_Y

        # Physical placement: the caret's top-left is the origin of the first line.
        x_phys = ev.x + round(left * dpr) - round(_PAD_X * dpr)
        y_phys = ev.y - round(_PAD_Y * dpr)
        w_phys = max(1, round(width_l * dpr))
        h_phys = max(1, round(height_l * dpr))
        key = (x_phys, y_phys, w_phys, h_phys, ev.text, round(font_px, 1))
        if key == self._last_key and self.isVisible():
            return
        self._last_key = key
        self._place(x_phys, y_phys, w_phys, h_phys, dpr)
        if not self.isVisible():
            self.show()
        self.update()

    def hide_completion(self) -> None:
        self._last_key = ()
        if self.isVisible():
            self.hide()

    # ---- helpers --------------------------------------------------------------------------------
    def _families(self, ev: CompletionEvent) -> list[str]:
        fams: list[str] = []
        if ev.font_name:
            fams.append(ev.font_name)
        fams += _MONO_FAMILIES if ev.mono else _UI_FAMILIES
        return fams

    @staticmethod
    def _dpr_at(x: int, y: int) -> float:
        if _is_windows_platform() and win32 is not None:
            try:
                return max(0.5, win32.dpi_for_point(x, y) / 96.0)
            except Exception:  # noqa: BLE001
                pass
        screen = QGuiApplication.primaryScreen()
        return float(screen.devicePixelRatio()) if screen is not None else 1.0

    def _place(self, x: int, y: int, w: int, h: int, dpr: float) -> None:
        """Position in physical pixels (SetWindowPos); Qt fallback in logical pixels elsewhere."""
        self.resize(max(1, round(w / dpr)), max(1, round(h / dpr)))
        if _is_windows_platform() and win32 is not None:
            try:
                if win32.set_window_rect_physical(int(self.winId()), x, y, w, h):
                    return
            except Exception:  # noqa: BLE001
                log.debug("ghost: physical placement failed", exc_info=True)
        self.move(round(x / dpr), round(y / dpr))

    # ---- painting ------------------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        if not self._lines:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setFont(self._font)
        color = QColor(self._cfg.completion.ghost_color)
        color.setAlphaF(max(0.1, min(1.0, self._cfg.completion.ghost_opacity)))
        shadow = QColor(0, 0, 0, 90)
        top_pad = (self._line_h - QFontMetricsF(self._font).height()) / 2.0
        for i, line in enumerate(self._lines):
            if not line:
                continue
            x = self._origin_x + self._offsets[i]
            baseline = _PAD_Y + i * self._line_h + top_pad + self._ascent
            painter.setPen(shadow)
            painter.drawText(QPointF(x + 1, baseline + 1), line)
            painter.setPen(color)
            painter.drawText(QPointF(x, baseline), line)
        painter.end()

    def layout_rect(self) -> QRect:
        """The widget's current logical rect (for tests)."""
        return self.geometry()
