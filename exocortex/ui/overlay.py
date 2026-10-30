"""OverlayWindow: the always-on-top, screen-capture-invisible "glass" panel.

Composition root for the overlay chrome (title bar, tabs, status bar, ask box) and
the event pump that turns bus events into widget updates. Talks to the engine only
through the ``EngineActions`` protocol and never blocks the Qt thread: every method
that reaches into ``exocortex.ui.win32`` / ``exocortex.ui.hotkeys`` is wrapped so a
missing or misbehaving optional module degrades to a log warning instead of a crash,
and every call is skipped on a non-Windows / non-"windows" QPA platform (tests run
under ``QT_QPA_PLATFORM=offscreen``).
"""
from __future__ import annotations

import dataclasses
import logging
import sys
import time

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, QTimer, Qt, Signal
from PySide6.QtGui import (
    QActionGroup,
    QCloseEvent,
    QContextMenuEvent,
    QCursor,
    QGuiApplication,
    QMouseEvent,
    QResizeEvent,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QAbstractScrollArea,
)

from exocortex.actions import EngineActions
from exocortex.bus import (
    AssistDeltaEvent,
    AssistDoneEvent,
    AssistErrorEvent,
    AssistStartedEvent,
    AudioLevelEvent,
    AutoAssistChangedEvent,
    CheckpointEvent,
    CompletionEvent,
    CompletionModeEvent,
    ComponentStatusEvent,
    ContextLibraryEvent,
    ContextUsageEvent,
    DashboardModeEvent,
    DashboardUpdatedEvent,
    Event,
    EventBus,
    FinalTranscriptEvent,
    HandsFreeEvent,
    InsightsUpdatedEvent,
    ManualAdvanceEvent,
    ModelInfoEvent,
    ModelLibraryEvent,
    ModelPullEvent,
    ModelSearchEvent,
    NotesErrorEvent,
    NotesUpdatedEvent,
    PartialTranscriptEvent,
    ScreenStatusEvent,
    ScreenTextEvent,
    SessionEvent,
    SpeakerRenamedEvent,
    drain,
)
from exocortex.config import Config
from exocortex.llm.prompts import normalize_role
from exocortex.plan import split_plan
from exocortex.ui.markdown_lite import md_to_html
from exocortex.ui.resources import ResourceSampler
from exocortex.ui.theme import COLORS, build_qss
from exocortex.ui.widgets import (
    AskBox,
    AssistView,
    MonitorView,
    NotesView,
    ScreenView,
    StatusBar,
    ToggleSwitch,
    TranscriptView,
)

log = logging.getLogger(__name__)

try:
    from exocortex.ui import win32
except Exception:  # pragma: no cover - win32.py may be missing/mid-implementation
    win32 = None  # type: ignore[assignment]
    log.warning("exocortex.ui.win32 unavailable; capture exclusion / click-through disabled")

try:
    from exocortex.ui.hotkeys import GlobalHotkeys
except Exception:  # pragma: no cover - hotkeys.py may be missing/mid-implementation
    GlobalHotkeys = None  # type: ignore[assignment]
    log.warning("exocortex.ui.hotkeys unavailable; global hotkeys disabled")

try:
    from exocortex.ui.llhooks import BareKeyHook, MouseWheelHook, PressHoldMouseHook, is_bare_key
except Exception:  # pragma: no cover - llhooks.py may be missing/mid-implementation
    BareKeyHook = MouseWheelHook = PressHoldMouseHook = None  # type: ignore[assignment]
    is_bare_key = None  # type: ignore[assignment]
    log.warning("exocortex.ui.llhooks unavailable; hands-free wheel scrolling, the arrow keys, "
                "the pause key and the manual-advance gesture disabled")

try:
    from exocortex.ui.region_picker import RegionPicker
except Exception:  # pragma: no cover - region_picker.py may be missing/mid-implementation
    RegionPicker = None  # type: ignore[assignment]
    log.warning("exocortex.ui.region_picker unavailable; region selection disabled")

try:
    from exocortex.ui.ghost import GhostText
except Exception:  # pragma: no cover - ghost.py may be missing/mid-implementation
    GhostText = None  # type: ignore[assignment]
    log.warning("exocortex.ui.ghost unavailable; ghost-text completions disabled")

try:
    from exocortex.ui.settings import SettingsDialog
except Exception:  # pragma: no cover - settings.py may be missing/mid-implementation
    SettingsDialog = None  # type: ignore[assignment]
    log.warning("exocortex.ui.settings unavailable; the settings dialog is disabled")

try:
    from exocortex.ui.models import ModelsDialog
except Exception:  # pragma: no cover - models.py may be missing/mid-implementation
    ModelsDialog = None  # type: ignore[assignment]
    log.warning("exocortex.ui.models unavailable; the Models window is disabled")

try:
    from exocortex.ui import dashboard as dashboard_mod
    from exocortex.ui.dashboard import DashboardView
except Exception:  # pragma: no cover - dashboard.py may be missing/mid-implementation
    dashboard_mod = None  # type: ignore[assignment]
    DashboardView = None  # type: ignore[assignment]
    log.warning("exocortex.ui.dashboard unavailable; dashboard mode is disabled")

_SLIM_MIN_WIDTH = 300
# The width the slim bar is allowed to grow to when the full-mode window is wider than it. With
# Checkpoint + Code in the bar its own minimum, measured under the real Windows platform plugin
# (Segoe UI Variable 9 pt), is 456 px with Explain hidden and 530 px with it shown -- 474 / 548 px
# of window once the root margins are added -- so 560 covers the worst case; whatever the cap,
# `_set_completion_mode` never lets the window end up narrower than the bar's real minimum. (The
# offscreen QPA's fallback font is ~60 % wider, so tests must not assert these numbers.)
_SLIM_MAX_WIDTH = 560
_PANEL_HEIGHT = 230   # the slim bar's Walkthrough / Explanation drop-down
_QWIDGETSIZE_MAX = 16777215
_PANEL_PLACEHOLDERS = {
    "walkthrough": "*(no walkthrough yet — one is written before each solution while explanations are on)*",
    "explanation": "*(no explanation yet — it follows once the solution has been typed out; "
                   "`Explain ▾ › Explain now` asks for it right away)*",
}

_POLL_MS = 30
#: The tab the overlay opens on, every launch: Assist (Live / Assist / Notes / Screen = 0..3).
#: Hands-free mode starts there too (its view is the Assist tab's output), and leaving the slim
#: bar of completion mode lands on whatever tab is current -- Assist unless the user moved.
_START_TAB = 1
_TOPMOST_MS = 1500
_TOAST_MS = 2500
#: The Monitor section samples CPU / RAM / VRAM and re-reads the context snapshot every Nth poll
#: tick (`_POLL_MS` each), i.e. about once a second -- cheap readings that never need 30 ms cadence.
_MONITOR_SAMPLE_TICKS = 33
_EDGE_MARGIN = 24
_NUDGE_PX = 40
_MIN_WIDTH = 340
_MIN_HEIGHT = 200
_GRIP_PX = 6          # resize band; inside the root layout's 8 px margin, so it covers no control
_GRIP_CORNER_PX = 18  # this close to the end of an edge band, the drag resizes the corner
_TITLE_STRIP_PX = 40  # top strip (title bar: drag handle + menu) that must stay on a screen
_KEEP_VISIBLE_PX = 120


#: The Assist menu's Role submenu, in display order (keys: exocortex.llm.prompts.ROLES).
_ROLE_LABELS: tuple[tuple[str, str], ...] = (
    ("driver", "Driver (I'm at the keyboard)"),
    ("navigator", "Navigator (I'm reviewing)"),
    ("participant", "Participant (discussion / design review)"),
)
#: Whom suggested questions go to, per role (the toast after a role switch).
_ROLE_AUDIENCE: dict[str, str] = {"driver": "the navigator / team", "navigator": "the driver", "participant": "the group"}
#: Index of the Notes tab (Live / Assist / Notes / Screen = 0..3): the Insights pane lives there.
_NOTES_TAB = 2
#: The full overlay's tabs by index, and every view the View menu can switch to (2026-10-21).
_TAB_KEYS: tuple[str, ...] = ("live", "assist", "notes", "screen")
_VIEW_LABELS: tuple[tuple[str, str], ...] = (
    ("live", "Live"), ("assist", "Assist"), ("notes", "Notes"), ("screen", "Screen"), ("dashboard", "Dashboard"),
)
_SCREEN_SOURCE_LABELS: tuple[tuple[str, str], ...] = (("off", "Off"), ("auto", "Active window"))


def _hotkey_hint(binding: str) -> str:
    """A configured global hotkey as a menu's shortcut column shows it ("\tCtrl+Alt+W"), or ""
    when that hotkey is off. Display text only: nothing is registered with Qt."""
    parts = [p.strip() for p in str(binding or "").split("+") if p.strip()]
    if not parts:
        return ""
    return "\t" + "+".join(p.upper() if len(p) == 1 else p.capitalize() for p in parts)


def _assist_footer(ev: AssistDoneEvent) -> str:
    """Footer text for a finished suggestion job, matching AssistView.on_done's style."""
    if ev.cancelled:
        return "cancelled"
    return f"ttft {ev.ttft_s:.2f} s - {ev.tok_per_s:.0f} tok/s"


def _is_windows_platform() -> bool:
    """True only on real Windows with the native "windows" Qt platform plugin.

    False under the "offscreen" QPA used by tests (even on a Windows CI box) and on
    any non-Windows OS, so every ``win32.*`` call in this module is guarded by it.
    """
    return sys.platform == "win32" and QGuiApplication.platformName() == "windows"


# --------------------------------------------------------------- small helpers -----
class _ElidingLabel(QLabel):
    """A one-line QLabel that elides (never wraps or forces the layout wider).

    ``minimumSizeHint`` is just an ellipsis, so this label never stops the window
    shrinking (down to the 340 px floor the overlay must support); the full text
    reappears as the window widens, and always as the tooltip.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = text
        self._apply_elided()

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override
        self._full_text = text
        self.setToolTip(text)
        self._apply_elided()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._apply_elided()

    def _apply_elided(self) -> None:
        fm = self.fontMetrics()
        elided = fm.elidedText(self._full_text, Qt.TextElideMode.ElideRight, max(self.width(), 1))
        super().setText(elided)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        fm = self.fontMetrics()
        return QSize(fm.horizontalAdvance("…"), fm.height())


class _DotIndicator(QLabel):
    """A small filled circle used as the overall readiness indicator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self.set_color(COLORS["warn"])

    def set_color(self, color: str) -> None:
        self.setStyleSheet(f"background: {color}; border-radius: 5px;")


def _on_screen_pos(window: QWidget, pos: QPoint) -> QPoint:
    """The position nearest ``pos`` that keeps ``window``'s title strip reachable on a screen.

    The window is frameless, so its title bar is the only drag handle and the only way to the
    menu: if that strip leaves every screen, the panel can no longer be moved back by mouse.
    """
    strip = QRect(pos.x(), pos.y(), window.width(), _TITLE_STRIP_PX)
    keep = min(_KEEP_VISIBLE_PX, window.width())
    for screen in QGuiApplication.screens():
        geo = screen.availableGeometry()
        if geo.top() <= strip.top() and strip.bottom() <= geo.bottom() and geo.intersected(strip).width() >= keep:
            return pos
    screen = QGuiApplication.screenAt(strip.center()) or window.screen() or QGuiApplication.primaryScreen()
    if screen is None:  # pragma: no cover - no screen at all
        return pos
    geo = screen.availableGeometry()
    x = max(geo.left() - window.width() + keep, min(pos.x(), geo.right() - keep + 1))
    y = max(geo.top(), min(pos.y(), geo.bottom() - _TITLE_STRIP_PX + 1))
    return QPoint(x, y)


class _EdgeGrip(QWidget):
    """Invisible resize band along one edge of the frameless window.

    Dragging it resizes that edge; within ``_GRIP_CORNER_PX`` of either end it resizes the
    corner instead, so the window can be resized from all four edges and all four corners.
    """

    def __init__(self, edge: Qt.Edge, parent: QWidget) -> None:
        super().__init__(parent)
        self._edge = edge
        self._edges = edge
        self._press: QPoint | None = None
        self._start = QRect()
        self.setMouseTracking(True)
        self.setCursor(self._cursor_for(edge))

    def place(self, size: QSize) -> None:
        """Lay the band along its edge of a window of ``size``."""
        if self._edge == Qt.Edge.TopEdge:
            self.setGeometry(0, 0, size.width(), _GRIP_PX)
        elif self._edge == Qt.Edge.BottomEdge:
            self.setGeometry(0, size.height() - _GRIP_PX, size.width(), _GRIP_PX)
        elif self._edge == Qt.Edge.LeftEdge:
            self.setGeometry(0, 0, _GRIP_PX, size.height())
        else:
            self.setGeometry(size.width() - _GRIP_PX, 0, _GRIP_PX, size.height())
        self.raise_()

    def edges_at(self, pos: QPoint) -> Qt.Edge:
        """The edge(s) a drag starting at ``pos`` (band coordinates) resizes."""
        edges = self._edge
        if self._edge in (Qt.Edge.TopEdge, Qt.Edge.BottomEdge):
            if pos.x() < _GRIP_CORNER_PX:
                edges |= Qt.Edge.LeftEdge
            elif pos.x() >= self.width() - _GRIP_CORNER_PX:
                edges |= Qt.Edge.RightEdge
        elif pos.y() < _GRIP_CORNER_PX:
            edges |= Qt.Edge.TopEdge
        elif pos.y() >= self.height() - _GRIP_CORNER_PX:
            edges |= Qt.Edge.BottomEdge
        return edges

    @staticmethod
    def _cursor_for(edges: Qt.Edge) -> Qt.CursorShape:
        if edges in (Qt.Edge.TopEdge | Qt.Edge.LeftEdge, Qt.Edge.BottomEdge | Qt.Edge.RightEdge):
            return Qt.CursorShape.SizeFDiagCursor
        if edges in (Qt.Edge.TopEdge | Qt.Edge.RightEdge, Qt.Edge.BottomEdge | Qt.Edge.LeftEdge):
            return Qt.CursorShape.SizeBDiagCursor
        if edges in (Qt.Edge.LeftEdge, Qt.Edge.RightEdge):
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.SizeVerCursor

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._edges = self.edges_at(event.position().toPoint())
            self._press = event.globalPosition().toPoint()
            self._start = self.window().geometry()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._press is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            self.setCursor(self._cursor_for(self.edges_at(event.position().toPoint())))
            super().mouseMoveEvent(event)
            return
        window = self.window()
        delta = event.globalPosition().toPoint() - self._press
        geo = QRect(self._start)
        # The moving edge stops where the window would drop below its (layout) minimum size.
        if self._edges & Qt.Edge.LeftEdge:
            geo.setLeft(min(geo.left() + delta.x(), geo.right() - window.minimumWidth() + 1))
        if self._edges & Qt.Edge.RightEdge:
            geo.setRight(max(geo.right() + delta.x(), geo.left() + window.minimumWidth() - 1))
        if self._edges & Qt.Edge.TopEdge:
            geo.setTop(min(geo.top() + delta.y(), geo.bottom() - window.minimumHeight() + 1))
        if self._edges & Qt.Edge.BottomEdge:
            geo.setBottom(max(geo.bottom() + delta.y(), geo.top() + window.minimumHeight() - 1))
        window.setGeometry(geo)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        self._press = None
        super().mouseReleaseEvent(event)


class _TitleBar(QFrame):
    """Draggable title bar: left mouse button press-drag moves the top-level window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("titlebar")
        self._drag_offset: QPoint | None = None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            window = self.window()
            self._drag_offset = event.globalPosition().toPoint() - window.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._drag_offset is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            window = self.window()
            window.move(_on_screen_pos(window, event.globalPosition().toPoint() - self._drag_offset))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        self._drag_offset = None
        super().mouseReleaseEvent(event)


class _ExplainPanel(QFrame):
    """Completion mode's drop-down under the slim bar: the plan's **walkthrough** (the thought
    process written before the solution) and the **explanation** of the finished solution, as
    read-only rendered markdown, switchable and closable. Text is set from the bus events as they
    stream in and painted lazily (``flush``) by the overlay's poll timer while the panel is shown.
    """

    closed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("explainpanel")
        self.setStyleSheet("QPushButton#tab { padding: 3px 7px; font-size: 9pt; }")
        self._texts: dict[str, str] = {"walkthrough": "", "explanation": ""}
        self._section = "walkthrough"
        self._dirty = True
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 2, 4, 2)
        v.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(2)
        self.walkthrough_btn = QPushButton("Walkthrough", self)
        self.explanation_btn = QPushButton("Explanation", self)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for btn, section in ((self.walkthrough_btn, "walkthrough"), (self.explanation_btn, "explanation")):
            btn.setObjectName("tab")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked=False, s=section: self.show_section(s))
            self._group.addButton(btn)
            row.addWidget(btn)
        self.walkthrough_btn.setChecked(True)
        row.addStretch(1)
        self.close_btn = QPushButton("×", self)
        self.close_btn.setObjectName("icon")
        self.close_btn.setToolTip("Close")
        self.close_btn.clicked.connect(self.close_panel)
        row.addWidget(self.close_btn)
        v.addLayout(row)
        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(False)
        v.addWidget(self.browser, 1)
        self.setFixedHeight(_PANEL_HEIGHT)

    @property
    def section(self) -> str:
        return self._section

    def has(self, section: str) -> bool:
        return bool(self._texts.get(section, "").strip())

    def set_text(self, section: str, text: str) -> None:
        if self._texts.get(section) == text:
            return
        self._texts[section] = text
        if section == self._section:
            self._dirty = True

    def show_section(self, section: str) -> None:
        if section not in self._texts:
            return
        self._section = section
        (self.walkthrough_btn if section == "walkthrough" else self.explanation_btn).setChecked(True)
        self._dirty = True
        self.flush()

    def flush(self) -> None:
        """Repaint the current section if its text changed since the last paint."""
        if not self._dirty:
            return
        self._dirty = False
        text = self._texts[self._section] or _PANEL_PLACEHOLDERS[self._section]
        self.browser.setHtml(md_to_html(text))
        bar = self.browser.verticalScrollBar()
        bar.setValue(bar.maximum())

    def clear(self) -> None:
        for key in self._texts:
            self._texts[key] = ""
        self._dirty = True
        self.flush()

    def close_panel(self) -> None:
        self.hide()
        self.closed.emit()


class _PopupTopmostFilter(QObject):
    """QApplication-wide filter for every newly shown top-level window (QMenu popups,
    QInputDialog, the settings dialog, tooltips, ...): it is pushed ABOVE the overlay in the
    z-order as it appears.

    Each such window is a separate HWND, and a plain popup is not topmost, so without the
    ``HWND_TOPMOST`` push it opens *behind* the always-on-top overlay (or a fullscreen app) and
    the user cannot see or use it. This is pure window-management hygiene -- dev-assist is a
    visible copilot, so nothing here hides any window from screen capture.
    """

    def __init__(self, cfg: Config, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        if win32 is not None and event.type() == QEvent.Type.Show and isinstance(obj, QWidget) and obj.isWindow():
            try:
                hwnd = int(obj.winId())
            except Exception:
                log.exception("winId() failed for a new top-level window")
                return False
            try:
                win32.set_topmost(hwnd)
            except Exception:
                log.exception("failed to raise popup window above the overlay")
        return False


# --------------------------------------------------------------- overlay window ----
class OverlayWindow(QWidget):
    """Frameless, always-on-top, translucent 'glass' panel.

    Layout (top -> bottom): title bar (drag handle, status dot, title/subtitle,
    auto-answer/mic/click-through toggles, opacity slider, hide button, "..." menu) /
    tab strip [Live | Assist | Notes | Screen], opening on Assist (``_START_TAB``) / stacked
    content (the Assist tab's own
    layout is a resizable ``QSplitter``, see ``AssistView``) / AskBox / control row
    (``_build_control_row``, one row: Read, Revert, Reset, Pause, Checkpoint, then --
    right-aligned, accent colour -- Text and Code) / StatusBar + size grip. Consumes
    bus events via a 30 ms QTimer (``bus.drain``) and never blocks the UI thread. Public slots
    used by hotkeys/engine: ``toggle_visible``, ``focus_ask``, ``nudge``, ``set_click_through``,
    ``set_opacity``, ``show_toast``.

    Frameless, so it supplies its own window management: title-bar drag, resize from every edge
    and corner (``_EdgeGrip``), a right-click menu with Quit everywhere, and ``_on_screen_pos`` /
    ``_keep_on_screen`` so the title bar - the only drag handle - can never be lost off-screen.

    dev-assist is a VISIBLE copilot: the panel is meant to be screen-shared and to show up in the
    taskbar / Alt-Tab like any ordinary window. The only Win32 touches are usability ones -- a
    periodic ``HWND_TOPMOST`` backstop so a meeting app cannot bury the always-on-top panel, an
    app-level filter (``_PopupTopmostFilter``) that raises pop-ups/dialogs above it, and optional
    click-through (``WS_EX_TRANSPARENT``). Nothing conceals the window from capture.
    """

    def __init__(self, cfg: Config, bus: EventBus, actions: EngineActions) -> None:
        super().__init__(None)
        self._cfg = cfg
        self._bus = bus
        self._actions = actions
        self._closing_programmatically = False
        self._component_states: dict[str, str] = {}
        # Monitor section (View -> Monitor): a resource sampler and a tick counter so the poll
        # loop refreshes it about once a second rather than every 30 ms.
        self._resource_sampler = ResourceSampler()
        self._monitor_tick = 0
        self._click_through = cfg.ui.click_through
        self._hotkeys = None
        self._screen_mode = cfg.screen.mode
        self._region_picker: RegionPicker | None = None
        self._assist_job_kinds: dict[int, str] = {}
        self._completion_mode = False
        self._normal_geometry: QRect | None = None
        self._ghost: "GhostText | None" = None
        self._completion_state = ""
        # explanations: the streaming plan (walkthrough) and explanation texts for the slim panel
        self._plan_text = ""
        self._plan_job: int | None = None
        self._explain_text = ""
        self._explain_job: int | None = None
        self._settings: "SettingsDialog | None" = None
        self._models_dialog: "ModelsDialog | None" = None
        self._models_prompted = False
        # hands-free mode: the output-only, click-through view (see _apply_layout); the engine's
        # HandsFreeEvents keep `_hands_free` in sync, `_hf_click_through` is the WS_EX_TRANSPARENT
        # state the cursor-tracking poll last applied (None = not managed by hands-free).
        self._hands_free = False
        self._hf_click_through: bool | None = None
        self._hf_state = ""
        # Hands-free view, while the cursor hovers an output pane (the window is click-through
        # there): a WH_MOUSE_LL hook scrolls the pane with the wheel, and over the expanded
        # history a WH_KEYBOARD_LL hook snaps through its entries with Up / Down taps -- both
        # installed only during the hover (without the hook module, the panes are made clickable
        # on hover instead). A third hook (a tap of the right Shift key by default,
        # `[hotkeys].toggle_pause`) toggles Pause / Resume at all times.
        self._wheel_hook: "MouseWheelHook | None" = None
        self._nav_key_hook: "BareKeyHook | None" = None
        self._pause_key_hook: "BareKeyHook | None" = None
        # Manual advance ([handsfree].manual): the Manual switch next to Hands-free. While on, a
        # fourth hook watches for the press-and-hold gesture anywhere on screen (left button held
        # `manual_hold_s` = Text, middle = Code); installed only then, and only on the real
        # Windows platform (the offscreen QPA tests must never hook the developer's real mouse).
        self._manual = bool(cfg.handsfree.manual)
        self._manual_hook: "PressHoldMouseHook | None" = None
        # Dashboard mode (2026-10-20): the board (exocortex.ui.dashboard) replaces the tab strip
        # and the stacked tabs; the window widens to [dashboard].width and the normal geometry is
        # remembered and restored on exit, the way completion mode's slim bar does it.
        self._dashboard_mode = False
        self._project_detail = ""        # the engine's last word on the repository / toolkit (the View menu shows it)
        self._dashboard_normal_geometry: QRect | None = None

        # Window flags must be set once, before the first show(). A visible copilot: a normal
        # top-level window (it appears in the taskbar / Alt-Tab), frameless and always-on-top.
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet(build_qss(cfg.ui.font_size))
        self.setWindowOpacity(cfg.ui.opacity)

        self._build_ui()
        self._apply_geometry()

        self._q = bus.subscribe()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(_POLL_MS)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start()

        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(lambda: self.toast.setVisible(False))

        self._topmost_timer = QTimer(self)
        self._topmost_timer.setInterval(_TOPMOST_MS)
        self._topmost_timer.timeout.connect(self._reassert_topmost)

        self._popup_filter: _PopupTopmostFilter | None = None
        if _is_windows_platform() and win32 is not None:
            self._popup_filter = _PopupTopmostFilter(self._cfg, self)
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self._popup_filter)

        if cfg.completion.enabled:
            # The engine enters completion mode itself on start(); mirror it in the UI without
            # calling back into the engine (the button handler would).
            self._completion_mode = False
            # `self` as the receiver context: Qt drops the queued callback if the window
            # is destroyed before it runs, so a fast close can never dereference a dead
            # completion_btn here.
            QTimer.singleShot(0, self, lambda: self.completion_btn.setChecked(True))
        if cfg.dashboard.enabled and self.dashboard_view is not None and not cfg.completion.enabled:
            # Open on the board (the engine publishes DashboardModeEvent too; the second one is a
            # no-op). Completion mode wins when both are set: the slim bar has no room for a board.
            QTimer.singleShot(0, self, lambda: self.dashboard_btn.setChecked(True))

        self._install_hotkeys()
        if self._manual:
            self._install_manual_hook()
        if self._hotkeys is not None and self._hotkeys.failed:
            failed = list(self._hotkeys.failed)
            # Receiver context as above: no orphaned callback into a destroyed window.
            QTimer.singleShot(0, self, lambda: self.show_toast("Hotkey conflict: " + ", ".join(failed)))

    # ---------------------------------------------------------------- UI building --
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.root = QFrame(self)
        self.root.setObjectName("root")
        outer.addWidget(self.root)

        root_layout = QVBoxLayout(self.root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        self.slim = self._build_slim_bar()
        self.slim.hide()
        root_layout.addWidget(self.slim)
        self.slim_panel = _ExplainPanel(self.root)
        self.slim_panel.hide()
        self.slim_panel.closed.connect(self._apply_slim_size)
        root_layout.addWidget(self.slim_panel)

        self.titlebar = self._build_titlebar()
        root_layout.addWidget(self.titlebar)
        self.tabstrip = self._build_tabstrip()
        root_layout.addWidget(self.tabstrip)

        self.stack = QStackedWidget(self.root)
        self.live_view = TranscriptView(self._actions, self.stack)
        self.assist_view = AssistView(self._actions, self.stack)
        self.notes_view = NotesView(self._actions, self.stack)
        self.screen_view = ScreenView(self._cfg, self._actions, self.stack)
        self.screen_view.mode_requested.connect(self._on_screen_mode_requested)
        self.screen_view.region_requested.connect(self._open_region_picker)
        self.stack.addWidget(self.live_view)
        self.stack.addWidget(self.assist_view)
        self.stack.addWidget(self.notes_view)
        self.stack.addWidget(self.screen_view)
        root_layout.addWidget(self.stack, 1)
        # Dashboard mode's board: hidden until the mode is on (it then stands in for the tab
        # strip + stack; the ask box, control row and status bar stay).
        self.dashboard_view = DashboardView(self._cfg, self.root) if DashboardView is not None else None
        if self.dashboard_view is not None:
            self.dashboard_view.hide()
            self.dashboard_view.item_action.connect(self._on_dashboard_item)
            self.dashboard_view.item_added.connect(self._on_dashboard_add)
            self.dashboard_view.refresh_requested.connect(self._on_dashboard_refresh)
            self.dashboard_view.notes_requested.connect(lambda: self._actions.generate_notes())
            root_layout.addWidget(self.dashboard_view, 1)
        self.screen_view.set_mode(self._screen_mode)
        self._show_tab(_START_TAB)   # also enables the hands-free switch (usable from Assist)

        self.askbox = AskBox(self._actions, self.root)
        root_layout.addWidget(self.askbox)

        self.control_row = self._build_control_row()
        root_layout.addWidget(self.control_row)

        # The Monitor section (View -> Monitor): performance / context / resources, at the bottom
        # just above the status strip. Hidden by default ([ui].show_monitor); _apply_view() below
        # applies the saved View preferences.
        self.monitor = MonitorView(self._cfg, self.root)
        self.monitor.set_model(self._cfg.llm.model)
        root_layout.addWidget(self.monitor)

        bottom = QHBoxLayout()
        self.status = StatusBar(self.root)
        bottom.addWidget(self.status, 1)
        self.size_grip = QSizeGrip(self.root)
        bottom.addWidget(self.size_grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        root_layout.addLayout(bottom)

        self.toast = QLabel("", self)
        self.toast.setObjectName("toast")
        self.toast.hide()

        self._edge_grips = [
            _EdgeGrip(edge, self)
            for edge in (Qt.Edge.LeftEdge, Qt.Edge.RightEdge, Qt.Edge.TopEdge, Qt.Edge.BottomEdge)
        ]

        # Text widgets consume right-clicks for their own Copy/Select-all menu, and they cover
        # most of the panel: extend that menu (see eventFilter) so Hide/Quit stay one right-click
        # away everywhere. Scroll-area based editors deliver the event to their viewport.
        self._text_menu_owners: dict[QObject, QWidget] = {}
        for editor in self.findChildren(QTextEdit) + self.findChildren(QPlainTextEdit):
            self._text_menu_owners[editor.viewport()] = editor
        for line_edit in self.findChildren(QLineEdit):
            self._text_menu_owners[line_edit] = line_edit
        for target in self._text_menu_owners:
            target.installEventFilter(self)

        # Apply the saved View preferences (which strips are shown). Safe here: completion and
        # hands-free are both off at construction, so this just hides the Monitor by default.
        self._apply_view()

    def _build_titlebar(self) -> QFrame:
        bar = _TitleBar(self.root)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)   # 9 icon buttons + slider must fit the 340 px floor (each has its own padding)
        # The dashboard button (2026-10-20) made it nine: a tighter padding for THIS bar's icon
        # buttons keeps the row inside the floor (the slim bar and the tabs have their own sheets).
        bar.setStyleSheet("QPushButton#icon { padding: 2px 4px; }")

        self.status_dot = _DotIndicator(bar)
        layout.addWidget(self.status_dot)

        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        self.title_label = _ElidingLabel("ExoCortex", bar)
        self.title_label.setObjectName("title")
        self.subtitle_label = _ElidingLabel(self._cfg.llm.model, bar)
        self.subtitle_label.setObjectName("subtitle")
        title_col.addWidget(self.title_label)
        title_col.addWidget(self.subtitle_label)
        layout.addLayout(title_col)
        layout.addStretch(1)

        self.dashboard_btn = QPushButton("▦", bar)
        self.dashboard_btn.setObjectName("icon")
        self.dashboard_btn.setCheckable(True)
        self.dashboard_btn.setChecked(False)
        self.dashboard_btn.setToolTip(
            "Dashboard mode: session notes, questions to ask, blind spots & follow-ups, next steps & "
            "prompts, side by side (Ctrl+Alt+D)"
        )
        self.dashboard_btn.setEnabled(DashboardView is not None)
        self.dashboard_btn.toggled.connect(self._on_dashboard_toggled)
        layout.addWidget(self.dashboard_btn)

        self.explain_btn = QPushButton("\U0001f4a1", bar)
        self.explain_btn.setObjectName("icon")
        self.explain_btn.setCheckable(True)
        self.explain_btn.setChecked(self._cfg.explain.enabled)
        self.explain_btn.setToolTip(
            "Explanations: a walkthrough before every solution, minimal comments in the code, and an "
            "explanation once you have typed it out (Ctrl+Alt+E)"
        )
        self.explain_btn.toggled.connect(self._on_explain_toggled)
        layout.addWidget(self.explain_btn)

        self.completion_btn = QPushButton("⌨", bar)
        self.completion_btn.setObjectName("icon")
        self.completion_btn.setCheckable(True)
        self.completion_btn.setChecked(False)
        self.completion_btn.setToolTip("Completion mode: slim bar + inline completions where you type (Ctrl+Alt+K)")
        self.completion_btn.toggled.connect(self._on_completion_toggled)
        layout.addWidget(self.completion_btn)

        self.auto_btn = QPushButton("⚡", bar)
        self.auto_btn.setObjectName("icon")
        self.auto_btn.setCheckable(True)
        self.auto_btn.setChecked(self._cfg.assist.auto_answer_questions)
        self.auto_btn.setToolTip("Auto-answer questions (Ctrl+Alt+T)")
        self.auto_btn.toggled.connect(self._on_auto_toggled)
        layout.addWidget(self.auto_btn)

        self.mic_btn = QPushButton("\U0001f3a4", bar)
        self.mic_btn.setObjectName("icon")
        self.mic_btn.setCheckable(True)
        self.mic_btn.setChecked(self._cfg.audio.mic_start_muted)
        self.mic_btn.setToolTip("Mute microphone (Ctrl+Alt+M)")
        self.mic_btn.toggled.connect(self._on_mic_toggled)
        layout.addWidget(self.mic_btn)

        self.clickthrough_btn = QPushButton("◈", bar)
        self.clickthrough_btn.setObjectName("icon")
        self.clickthrough_btn.setCheckable(True)
        self.clickthrough_btn.setChecked(self._click_through)
        self.clickthrough_btn.setToolTip("Click-through (Ctrl+Alt+C)")
        self.clickthrough_btn.toggled.connect(self.set_click_through)
        layout.addWidget(self.clickthrough_btn)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal, bar)
        self.opacity_slider.setRange(40, 100)
        self.opacity_slider.setValue(round(self._cfg.ui.opacity * 100))
        self.opacity_slider.setFixedWidth(42)
        self.opacity_slider.setToolTip("Opacity")
        self.opacity_slider.valueChanged.connect(lambda v: self.set_opacity(v / 100.0))
        layout.addWidget(self.opacity_slider)

        self.hide_btn = QPushButton("−", bar)
        self.hide_btn.setObjectName("icon")
        self.hide_btn.setToolTip("Hide (Ctrl+Alt+Space)")
        self.hide_btn.clicked.connect(self.toggle_visible)
        layout.addWidget(self.hide_btn)

        self.menu_btn = QPushButton("⋯", bar)
        self.menu_btn.setObjectName("icon")
        self._menu = QMenu(self)
        self._menu.aboutToShow.connect(self._rebuild_menu)
        self.menu_btn.setMenu(self._menu)
        layout.addWidget(self.menu_btn)
        self.slim_menu_btn.setMenu(self._menu)

        return bar

    def _build_slim_bar(self) -> QWidget:
        """Completion mode's one-row UI, left -> right: back to full view, context, explain
        (hidden unless explanations are on), reset, pause/resume, checkpoint, code, status dot +
        label, hide, menu. Doubles as the drag handle (it is a _TitleBar). The full overlay's
        Text button is deliberately absent here (and from the Context menu): a walkthrough is
        prose to read in the Assist tab, which the slim bar hides."""
        bar = _TitleBar(self.root)
        bar.setObjectName("slim")
        bar.setStyleSheet("QPushButton { padding: 3px 7px; font-size: 9pt; } QLabel { font-size: 9pt; }")
        h = QHBoxLayout(bar)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(4)

        # Constructed in the same left -> right order they are laid out below (also the order
        # findChildren()/tab order follow), so the two stay in sync without a second place to edit.
        self.slim_full_btn = QPushButton("Full", bar)
        self.slim_full_btn.setToolTip("Back to the full overlay (Ctrl+Alt+K)")
        self.slim_full_btn.clicked.connect(lambda: self.completion_btn.setChecked(False))

        self.slim_context_btn = QPushButton("Context", bar)
        self.slim_context_btn.setToolTip("What feeds the completions, and a standing hint")
        self._context_menu = QMenu(self)
        self._context_menu.aboutToShow.connect(self._rebuild_context_menu)
        self.slim_context_btn.setMenu(self._context_menu)

        self.slim_explain_btn = QPushButton("Explain", bar)
        self.slim_explain_btn.setToolTip("Walkthrough and explanation of the current solution")
        self._explain_menu = QMenu(self)
        self._explain_menu.aboutToShow.connect(self._rebuild_explain_menu)
        self.slim_explain_btn.setMenu(self._explain_menu)
        self.slim_explain_btn.setVisible(self._cfg.explain.enabled)

        self.slim_reset_btn = QPushButton("Reset", bar)
        self.slim_reset_btn.setToolTip("Clear the session context (transcript, answers, notes)")
        self.slim_reset_btn.clicked.connect(self._on_reset)

        self.slim_pause_btn = QPushButton("Pause", bar)
        self.slim_pause_btn.setCheckable(True)
        self.slim_pause_btn.setToolTip("Pause / resume completions and all automatic activity")
        self.slim_pause_btn.toggled.connect(self._on_slim_pause_toggled)

        self.slim_checkpoint_btn = QPushButton("Checkpoint", bar)
        self.slim_checkpoint_btn.setToolTip(
            "Soft reset of the coding context: solve, Code and completions start fresh (Ctrl+Alt+B)"
        )
        self.slim_checkpoint_btn.clicked.connect(self._on_checkpoint)

        self.slim_code_btn = QPushButton("Code", bar)
        self.slim_code_btn.setObjectName("primary")
        self.slim_code_btn.setToolTip("Write the code for the task discussed / shown, right now (Ctrl+Alt+G)")
        self.slim_code_btn.clicked.connect(self._on_code)

        self.slim_dot = _DotIndicator(bar)
        self.slim_status = _ElidingLabel("completion", bar)
        self.slim_status.setObjectName("subtitle")

        self.slim_hide_btn = QPushButton("\u2212", bar)
        self.slim_hide_btn.setObjectName("icon")
        self.slim_hide_btn.setToolTip("Hide (Ctrl+Alt+Space)")
        self.slim_hide_btn.clicked.connect(self.toggle_visible)

        self.slim_menu_btn = QPushButton("\u22ef", bar)
        self.slim_menu_btn.setObjectName("icon")

        for w in (self.slim_full_btn, self.slim_context_btn, self.slim_explain_btn, self.slim_reset_btn,
                  self.slim_pause_btn, self.slim_checkpoint_btn, self.slim_code_btn):
            h.addWidget(w)
        h.addSpacing(6)
        h.addWidget(self.slim_dot)
        h.addWidget(self.slim_status, 1)
        h.addWidget(self.slim_hide_btn)
        h.addWidget(self.slim_menu_btn)
        return bar

    def _rebuild_context_menu(self) -> None:
        cc = self._cfg.completion
        menu = self._context_menu
        menu.clear()
        for key, label, on in (
            ("transcript", "Transcript (what was said)", cc.use_transcript),
            ("answers", "Previous answers", cc.use_answers),
            ("notes", "Notes", cc.use_notes),
            ("screen", "Screen text (OCR)", cc.use_screen),
        ):
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(on)
            act.toggled.connect(lambda checked, k=key: self._actions.set_completion_context({k: checked}))
        menu.addSeparator()
        # 2026-10-20: the on-demand lookups (exocortex.project) that answers and the dashboard get.
        ctx = self._cfg.project
        repo_action = menu.addAction("Repository lookup (answers, dashboard)")
        repo_action.setCheckable(True)
        repo_action.setChecked(bool(ctx.repo_search))
        repo_action.toggled.connect(lambda checked: self._actions.set_completion_context({"repo": checked}))
        toolkit_action = menu.addAction("Toolkit catalog (answers, dashboard)")
        toolkit_action.setCheckable(True)
        toolkit_action.setChecked(bool(ctx.toolkit_in_answers))
        toolkit_action.toggled.connect(lambda checked: self._actions.set_completion_context({"toolkit": checked}))
        menu.addSeparator()
        hint_text = f"Hint: {cc.hint}" if cc.hint else "Hint\u2026 (none set)"
        hint_action = menu.addAction(hint_text)
        hint_action.triggered.connect(self._prompt_completion_hint)
        if cc.hint:
            clear_action = menu.addAction("Clear hint")
            clear_action.triggered.connect(lambda: self._actions.set_completion_context({}, ""))
        menu.addSeparator()
        explain_action = menu.addAction("Explanations (walkthrough \u00b7 comments \u00b7 explanation)")
        explain_action.setCheckable(True)
        explain_action.setChecked(self.explain_btn.isChecked())
        explain_action.toggled.connect(self.explain_btn.setChecked)
        handsfree_action = menu.addAction("Hands-free (read \u00b7 checkpoint \u00b7 text \u00b7 code by itself)")
        handsfree_action.setCheckable(True)
        handsfree_action.setChecked(self._hands_free)
        handsfree_action.toggled.connect(lambda checked: self._set_hands_free(checked))
        manual_action = menu.addAction("Manual advance (generate on press-and-hold only)")
        manual_action.setCheckable(True)
        manual_action.setChecked(self._manual)
        manual_action.toggled.connect(lambda checked: self._set_manual(checked))
        checkpoint_action = menu.addAction("Checkpoint (new coding task)")
        checkpoint_action.triggered.connect(self._on_checkpoint)
        compact_action = menu.addAction("Compact context now")
        compact_action.triggered.connect(self._on_compact)
        code_action = menu.addAction("Code now")
        code_action.triggered.connect(self._on_code)
        menu.addSeparator()
        menu.addAction("Settings…").triggered.connect(self.open_settings)
        menu.addAction("Models…").triggered.connect(lambda: self.open_models())
        # 2026-10-06: the reference-document library lives on the Models window's Context tab.
        menu.addAction("Context documents…").triggered.connect(lambda: self.open_models("Context"))

    # ---------------------------------------------------------------- settings ---------
    def open_settings(self, tab: str | None = None) -> None:
        """Show the (non-modal) settings dialog, reusing it if it is already open; ``tab`` jumps
        straight to one of its tabs by name (the Assist menu opens "Assist"). A menu's
        ``triggered(bool)`` may land in ``tab``: only a non-empty string counts."""
        if SettingsDialog is None:
            self.show_toast("Settings unavailable")
            return
        if self._settings is None:
            self._settings = SettingsDialog(self._cfg, self._actions.list_models, self)
            self._settings.applied.connect(self._on_settings_applied)
        else:
            self._settings.load()
        if isinstance(tab, str) and tab and hasattr(self._settings, "show_tab"):
            self._settings.show_tab(tab)
        self._settings.show()
        self._settings.raise_()
        self._settings.activateWindow()

    def open_models(self, tab: str | None = None) -> None:
        """Show the (non-modal) Models window, reusing it if it is already open. ``tab`` jumps
        straight to one of its tabs (Library / Discover / Tuning / Server / Context / Hardware)."""
        if ModelsDialog is None:
            self.show_toast("Models window unavailable")
            return
        if self._models_dialog is None:
            self._models_dialog = ModelsDialog(self._cfg, self._actions, self)
            self._models_dialog.applied.connect(self._on_settings_applied)
        else:
            self._models_dialog.load()
        if tab:
            self._models_dialog.show_tab(tab)
        self._models_dialog.show()
        self._models_dialog.raise_()
        self._models_dialog.activateWindow()

    def _on_settings_applied(self, changed: dict) -> None:
        """Mirror changed values in the overlay's own controls, then let the engine react."""
        cfg = self._cfg
        if "ui.opacity" in changed:
            self.set_opacity(cfg.ui.opacity)
        if "explain.enabled" in changed:
            self.explain_btn.blockSignals(True)
            self.explain_btn.setChecked(cfg.explain.enabled)
            self.explain_btn.blockSignals(False)
            self.slim_explain_btn.setVisible(cfg.explain.enabled)
            if not cfg.explain.enabled and not self.slim_panel.isHidden():
                self.slim_panel.hide()
                self._apply_slim_size()
        if "assist.auto_answer_questions" in changed:
            self.auto_btn.blockSignals(True)
            self.auto_btn.setChecked(cfg.assist.auto_answer_questions)
            self.auto_btn.blockSignals(False)
        if "audio.mic_start_muted" in changed:
            self.mic_btn.blockSignals(True)
            self.mic_btn.setChecked(cfg.audio.mic_start_muted)
            self.mic_btn.blockSignals(False)
        if any(str(key).startswith("screen.") for key in changed):
            self.screen_view.sync_toggles()
        if "dashboard.enabled" in changed:
            self._set_dashboard_mode(bool(cfg.dashboard.enabled), notify_engine=False)
        if any(str(key).startswith("dashboard.") for key in changed) and self.dashboard_view is not None:
            self.dashboard_view.apply_config()
        if any(str(k).startswith("ui.show_") for k in changed):
            self._apply_view()
        if any(str(k).startswith("monitor.") for k in changed):
            self.monitor.apply_config()
        try:
            self._actions.apply_settings(dict(changed))
        except Exception:
            log.exception("actions.apply_settings() failed")
        self.show_toast("Settings applied")

    def _rebuild_explain_menu(self) -> None:
        menu = self._explain_menu
        menu.clear()
        panel = self.slim_panel
        for section, label in (("walkthrough", "Walkthrough"), ("explanation", "Explanation")):
            act = menu.addAction(label if panel.has(section) else f"{label} (none yet)")
            act.triggered.connect(lambda _checked=False, s=section: self._open_panel(s))
        menu.addSeparator()
        menu.addAction("Explain now").triggered.connect(self._on_explain_now)
        menu.addAction("Rethink (solve again)").triggered.connect(lambda: self._actions.solve(""))
        if not panel.isHidden():
            menu.addSeparator()
            menu.addAction("Close panel").triggered.connect(panel.close_panel)

    # ---------------------------------------------------------------- explanations -----
    def _on_explain_toggled(self, checked: bool) -> None:
        self._cfg.explain.enabled = checked          # the engine sets it too; keep the UI's view current
        self.slim_explain_btn.setVisible(checked)
        if not checked and not self.slim_panel.isHidden():
            self.slim_panel.hide()
            self._apply_slim_size()
        try:
            self._actions.set_explanations(checked)
        except Exception:
            log.exception("actions.set_explanations(%r) failed", checked)
        self.show_toast("Explanations on" if checked else "Explanations off")

    def _on_explain_now(self) -> None:
        try:
            self._actions.explain_now()
        except Exception:
            log.exception("actions.explain_now() failed")

    def _open_panel(self, section: str) -> None:
        """Show the slim bar's drop-down on `section` (only meaningful in completion mode)."""
        self.slim_panel.show_section(section)
        if not self._completion_mode:
            return
        if self.slim_panel.isHidden():
            self.slim_panel.show()
        self._apply_slim_size()

    def _slim_height(self) -> int:
        h = self.slim.sizeHint().height() + 16
        if not self.slim_panel.isHidden():
            h += self.slim_panel.height() + 6
        return h

    def _apply_slim_size(self) -> None:
        """Re-fit the window to the slim bar (plus the drop-down when it is open).

        The layout only recomputes its minimum size on the next event-loop pass; forcing it now
        lets the window shrink immediately after the panel is hidden (the minimum would otherwise
        still include the panel and win over the new maximum)."""
        if not self._completion_mode:
            return
        self.root.layout().activate()
        self.layout().activate()
        h = max(self._slim_height(), self.layout().minimumSize().height())
        self.setMaximumHeight(h)
        self.resize(self.width(), h)
        self.move(_on_screen_pos(self, self.pos()))

    def _set_slim_activity(self, text: str) -> None:
        """The slim bar's status label doubles as the explanations activity readout."""
        if self._cfg.explain.enabled:
            self.slim_status.setText(text)

    def _on_plan_started(self, ev: AssistStartedEvent) -> None:
        if not self.explain_btn.isChecked():
            return
        if ev.kind in ("solve", "code"):
            self._plan_job = ev.job_id
            self._plan_text = ""
            self.slim_panel.set_text("walkthrough", "")
            self.slim_panel.set_text("explanation", "")
            self._set_slim_activity("thinking\u2026")
            if self._completion_mode:
                self._open_panel("walkthrough")
        elif ev.kind == "explain":
            self._explain_job = ev.job_id
            self._explain_text = ""
            self.slim_panel.set_text("explanation", "")
            self._set_slim_activity("explaining\u2026")

    def _on_plan_delta(self, ev: AssistDeltaEvent, kind: str | None) -> None:
        if not self.explain_btn.isChecked():
            return
        if kind in ("solve", "code") and ev.job_id == self._plan_job:
            self._plan_text += ev.delta
            self.slim_panel.set_text("walkthrough", split_plan(self._plan_text).walkthrough)
        elif kind == "explain" and ev.job_id == self._explain_job:
            self._explain_text += ev.delta
            self.slim_panel.set_text("explanation", self._explain_text)

    def _on_plan_done(self, ev: AssistDoneEvent, kind: str | None) -> None:
        if not self.explain_btn.isChecked():
            self._plan_job = self._explain_job = None   # switched off mid-stream: never reopen the panel
            return
        if kind in ("solve", "code") and ev.job_id == self._plan_job:
            self._plan_job = None
            if ev.cancelled:
                self._set_slim_activity("cancelled")
                return
            self._plan_text = ev.text or self._plan_text
            self.slim_panel.set_text("walkthrough", split_plan(self._plan_text).walkthrough)
            self._set_slim_activity("solution ready \u2014 type it out")
        elif kind == "explain" and ev.job_id == self._explain_job:
            self._explain_job = None
            if ev.cancelled:
                self._set_slim_activity("cancelled")
                return
            self._explain_text = ev.text or self._explain_text
            self.slim_panel.set_text("explanation", self._explain_text)
            self._set_slim_activity("explanation ready")
            if self._completion_mode:
                self._open_panel("explanation")

    def _prompt_completion_hint(self) -> None:
        text, ok = QInputDialog.getText(
            self, "Completion hint", "Standing instruction for completions\n(e.g. \"Python 3, terse\", \"reply in French\"):",
            text=self._cfg.completion.hint,
        )
        if ok:
            self._actions.set_completion_context({}, text.strip())
            self.show_toast("Hint set" if text.strip() else "Hint cleared")

    # ---------------------------------------------------------------- completion mode ---
    def _on_completion_toggled(self, checked: bool) -> None:
        if checked == self._completion_mode:
            return
        self._set_completion_mode(checked)

    def _on_slim_pause_toggled(self, checked: bool) -> None:
        # The slim bar's Pause mirrors the main Pause control; that one drives the engine.
        if self.pause_btn.isChecked() != checked:
            self.pause_btn.setChecked(checked)
        self.slim_pause_btn.setText("Resume" if checked else "Pause")

    def _set_completion_mode(self, enabled: bool) -> None:
        """Collapse to the slim bar (and start completions) or restore the full overlay."""
        if enabled and self._dashboard_mode:
            self._set_dashboard_mode(False)   # the slim bar and the board cannot share the window
        self._completion_mode = enabled
        self.completion_btn.blockSignals(True)
        self.completion_btn.setChecked(enabled)
        self.completion_btn.blockSignals(False)
        full_widgets = (self.titlebar, self.tabstrip, self.stack, self.askbox, self.control_row,
                        self.monitor, self.status, self.size_grip)
        if enabled:
            self._normal_geometry = QRect(self.geometry())
            self._restore_click_through()   # the slim bar must be clickable everywhere
            for w in full_widgets:
                w.hide()
            self.slim.show()
            self.setMinimumSize(_SLIM_MIN_WIDTH, 0)
            height = self._slim_height()
            self.setMaximumHeight(height)
            # The full-mode window's width is only a preference (grown up to _SLIM_MAX_WIDTH when
            # it is wide enough); the slim bar's OWN layout minimum (which varies with whether
            # Explain is showing) is a hard floor -- otherwise a narrow full-mode window (e.g. the
            # 460px default) would squeeze the row below what its buttons need to render, clipping
            # or overlapping them (see slim.minimumSizeHint()).
            width = max(_SLIM_MIN_WIDTH, min(self._normal_geometry.width(), _SLIM_MAX_WIDTH))
            geo = self._normal_geometry
            self.setGeometry(geo.x() + geo.width() - width, geo.y(), width, height)
            # `self.slim` sits inside `self.root`'s own layout, behind that layout's margins AND
            # `self.root`'s stylesheet border -- both invisible to `min()` above -- so measure the
            # actual shortfall after a real layout pass (forced with `activate()`, since a resize
            # this way doesn't relayout children synchronously) and widen the window by exactly
            # that, rather than guessing at the chrome width analytically.
            self.root.layout().activate()
            shortfall = self.slim.minimumSizeHint().width() - self.slim.width()
            if shortfall > 0:
                width += shortfall
                self.setGeometry(geo.x() + geo.width() - width, geo.y(), width, height)
            self.move(_on_screen_pos(self, self.pos()))
        else:
            self.slim.hide()
            self.slim_panel.hide()
            for w in full_widgets:
                w.show()
            self.setMaximumHeight(_QWIDGETSIZE_MAX)
            self.setMinimumSize(0, 0)
            if self._ghost is not None:
                self._ghost.hide_completion()
            # Settle which strips show (the View prefs / hands-free view) BEFORE restoring the
            # geometry: `full_widgets` includes the Monitor, so showing it unconditionally would
            # raise the layout's minimum height and clamp the restored height upward. `_apply_layout`
            # hides it again when its pref is off; activate() makes that take effect first.
            self._apply_layout()   # back to the hands-free view if that is on; applies the View prefs
            if self._normal_geometry is not None:
                self.root.layout().activate()
                self.setGeometry(self._normal_geometry)
        try:
            self._actions.set_completion_mode(enabled)
        except Exception:
            log.exception("actions.set_completion_mode(%r) failed", enabled)
        self.show_toast("Completion mode" if enabled else "Full overlay")

    def _on_completion_event(self, ev: CompletionEvent) -> None:
        if GhostText is None:
            return
        if not ev.visible or not self._completion_mode:
            if self._ghost is not None:
                self._ghost.hide_completion()
            return
        if self._ghost is None:
            self._ghost = GhostText(self._cfg)
        self._ghost.show_completion(ev)

    def _set_completion_status(self, state: str, detail: str) -> None:
        self._completion_state = state
        color = {"ready": COLORS["ok"], "error": COLORS["err"]}.get(state, COLORS["warn"])
        self.slim_dot.set_color(color)
        self.slim_status.setText(detail or state)

    # ---------------------------------------------------------------- dashboard mode ---
    @property
    def dashboard_mode(self) -> bool:
        return self._dashboard_mode

    @property
    def _hands_free_view_active(self) -> bool:
        """The hands-free view (output only, click-through content) is what is on screen:
        hands-free is on and neither the slim bar nor the board has taken over the window."""
        return self._hands_free and not self._completion_mode and not self._dashboard_mode

    def _on_dashboard_toggled(self, checked: bool) -> None:
        if checked != self._dashboard_mode:
            self._set_dashboard_mode(checked)

    def _sync_dashboard_button(self) -> None:
        if self.dashboard_btn.isChecked() != self._dashboard_mode:
            self.dashboard_btn.blockSignals(True)
            self.dashboard_btn.setChecked(self._dashboard_mode)
            self.dashboard_btn.blockSignals(False)

    def _set_dashboard_mode(self, enabled: bool, *, notify_engine: bool = True) -> None:
        """Enter / leave dashboard mode and (unless mirroring a bus event) tell the engine.

        The board stands in for the tab strip and the stacked tabs; the title bar, the ask box,
        the control row and the status bar stay, so asking, pausing, Text / Code all keep
        working. The window widens to ``[dashboard].width`` (growing leftwards from its right
        edge, clamped to the screen) and the previous geometry is restored on exit. Completion
        mode's slim bar is left first when it is on; the hands-free view resumes on exit.
        Screen capture / OCR are untouched either way (their cadence is the Screen tab's).
        """
        if self.dashboard_view is None:
            if enabled:
                self.show_toast("Dashboard unavailable")
            return
        if enabled == self._dashboard_mode:
            self._sync_dashboard_button()
            return
        if enabled and self._completion_mode:
            self._set_completion_mode(False)
        self._dashboard_mode = enabled
        self._sync_dashboard_button()
        if enabled:
            # Remember the size BEFORE the board is shown: its four columns raise the layout's
            # minimum width, and Qt grows the window to it as soon as the view is visible.
            self._dashboard_normal_geometry = QRect(self.geometry())
            self._restore_click_through()
            self._release_hover_hooks()
            for w in (self.askbox, self.size_grip, self.clickthrough_btn):
                w.setVisible(True)   # the hands-free view may have hidden them
            self.dashboard_view.apply_config()
        self.tabstrip.setVisible(not enabled)
        self.stack.setVisible(not enabled)
        self.dashboard_view.setVisible(enabled)
        if enabled:
            self._apply_view()   # the control row / status / Monitor as the View menu has them
        # The layout recomputes its minimum size only on the next event-loop pass; forcing it now
        # lets the window take the board's width at once, and shrink back at once on exit (the
        # stale, board-sized minimum would otherwise clamp the restored geometry -- the same
        # trick as `_apply_slim_size`).
        self.root.layout().activate()
        self.layout().activate()
        if enabled:
            self._apply_dashboard_geometry()
        else:
            geo = self._dashboard_normal_geometry
            self._dashboard_normal_geometry = None
            if geo is not None:
                self.setGeometry(geo)
                self.move(_on_screen_pos(self, geo.topLeft()))
            self._apply_layout()   # back to the hands-free view when that is on
        if notify_engine:
            try:
                self._actions.set_dashboard_mode(enabled)
            except Exception:
                log.exception("actions.set_dashboard_mode(%r) failed", enabled)
        self.show_toast("Dashboard mode" if enabled else "Full overlay")

    def _apply_dashboard_geometry(self) -> None:
        geo = self._dashboard_normal_geometry or self.geometry()
        screen = QGuiApplication.screenAt(geo.center()) or self.screen() or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        try:
            wanted = int(self._cfg.dashboard.width)
        except (TypeError, ValueError):
            wanted = geo.width()
        width = max(_MIN_WIDTH, wanted, geo.width())
        height = geo.height()
        if avail is not None:
            width = min(width, avail.width())
            height = min(max(height, _MIN_HEIGHT), avail.height())
        x = geo.right() - width + 1   # grow leftwards from the panel's right edge
        y = geo.y()
        if avail is not None:
            x = max(avail.left(), min(x, avail.right() - width + 1))
            y = max(avail.top(), min(y, avail.bottom() - height + 1))
        self.setGeometry(x, y, width, height)
        self.move(_on_screen_pos(self, QPoint(x, y)))

    def _on_dashboard_refresh(self, sections: tuple = ()) -> None:
        """A column's ↻ (one section) or "Refresh all" (everything: the board and the notes)."""
        secs = tuple(str(s) for s in (sections or ()) if str(s).strip())
        try:
            self._actions.dashboard_refresh(secs)
            if not secs:
                self._actions.generate_notes()
        except Exception:
            log.exception("actions.dashboard_refresh(%r) failed", secs)

    def _on_dashboard_item(self, item_id: int, action: str, text: str) -> None:
        try:
            self._actions.dashboard_item(int(item_id), str(action), str(text or ""))
        except Exception:
            log.exception("actions.dashboard_item(%r, %r) failed", item_id, action)

    def _on_dashboard_add(self, kind: str, text: str, parent_id: int = -1) -> None:
        """An item added by hand on the board; `parent_id` (-1 = none) is the action item a
        sub-task, goal or prompt belongs to."""
        try:
            self._actions.dashboard_add(str(kind), str(text), int(parent_id) if parent_id >= 0 else None)
        except Exception:
            log.exception("actions.dashboard_add(%r) failed", kind)

    def _on_dashboard_updated(self, ev: DashboardUpdatedEvent) -> None:
        if self.dashboard_view is not None:
            self.dashboard_view.set_snapshot(ev.items, ev.notes, ev.notes_at, ev.generated_at, ev.detail)

    def _build_control_row(self) -> QWidget:
        """Always-visible bottom controls, one row, left -> right: read the screen now, revert to
        the previous answer, reset the session, pause/resume all automatic activity, checkpoint
        the coding context, compact the context window, then -- right-aligned, accent colour,
        Text left of Code -- Text (a plain-English walkthrough of the task discussed/shown now)
        and Code (write the code for it now). Text and Compact live only here: the slim bar
        (completion mode) hides this row and does not carry them (see `_build_slim_bar`). The row
        can also be hidden entirely from the View menu ([ui].show_controls).

        The row holds eight labelled buttons now (Compact was added 2026-10-07, after Checkpoint).
        Its horizontal size policy is *Ignored* so it can never raise the window's own minimum
        width: at the 340 px floor the right-hand buttons clip a little (there is ~80 px of slack
        at the 460 px default width), and the offscreen QPA's ~60 %-wider fallback font must not
        push the test-time floor out. A machine without Segoe UI is better served by a slightly
        clipped right-hand button than by a wider minimum window."""
        row = QWidget(self.root)
        row.setStyleSheet("QPushButton { padding: 3px 6px; font-size: 9pt; }")
        row.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)

        self.read_btn = QPushButton("Read", row)
        self.read_btn.setToolTip("Read the screen right now (capture + OCR), without waiting for the next pass")
        self.read_btn.clicked.connect(self._on_read)

        self.revert_btn = QPushButton("Revert", row)
        self.revert_btn.setToolTip("Show the previous answer")
        self.revert_btn.clicked.connect(self._on_revert)

        self.reset_btn = QPushButton("Reset", row)
        self.reset_btn.setToolTip("Clear the transcript, answers and notes and start a fresh session")
        self.reset_btn.clicked.connect(self._on_reset)

        self.pause_btn = QPushButton("Pause", row)
        self.pause_btn.setCheckable(True)
        self.pause_btn.setToolTip("Pause all automatic activity (screen reading, auto-answer, auto-solve)")
        self.pause_btn.toggled.connect(self._on_pause_toggled)

        self.checkpoint_btn = QPushButton("Checkpoint", row)
        self.checkpoint_btn.setToolTip(
            "Soft reset of the coding context: solve, Code and completions start fresh from here "
            "(the transcript, notes and answer history keep the whole meeting) (Ctrl+Alt+B)"
        )
        self.checkpoint_btn.clicked.connect(self._on_checkpoint)

        self.compact_btn = QPushButton("Compact", row)
        self.compact_btn.setToolTip(
            "Compact the context window now: fold everything but the newest lines of the current "
            "context into a short running summary (frees room without losing the thread), like the "
            "/compact command. The summary is written in the background."
        )
        self.compact_btn.clicked.connect(self._on_compact)

        self.text_btn = QPushButton("Text", row)
        self.text_btn.setObjectName("primary")
        self.text_btn.setToolTip(
            "Walk through the task discussed / shown in plain English, right now: what is asked, the "
            "key insight, the approach step by step, edge cases and complexity"
        )
        self.text_btn.clicked.connect(self._on_text)

        self.code_btn = QPushButton("Code", row)
        self.code_btn.setObjectName("primary")
        self.code_btn.setToolTip(
            "Write the code for the task discussed / shown, right now (Ctrl+Alt+G)"
        )
        self.code_btn.clicked.connect(self._on_code)

        for btn in (self.read_btn, self.revert_btn, self.reset_btn, self.pause_btn,
                    self.checkpoint_btn, self.compact_btn):
            h.addWidget(btn)
        h.addStretch(1)
        h.addWidget(self.text_btn)
        h.addWidget(self.code_btn)
        return row

    def _on_read(self) -> None:
        self._actions.capture_screen_now()

    def _on_revert(self) -> None:
        self._show_tab(1)
        self.assist_view.revert()

    def _on_checkpoint(self) -> None:
        """Ask the engine to checkpoint the coding context. Calls ``actions.checkpoint()`` ONLY
        -- every panel update (the history divider, the toast, dropping the plan/explanation
        state) is driven by the ``CheckpointEvent`` the engine publishes back on the bus (see
        ``_on_checkpoint_event``), so it happens once, whether triggered by this button, the
        slim bar's button, a menu entry or the hotkey."""
        try:
            self._actions.checkpoint()
        except Exception:
            log.exception("actions.checkpoint() failed")

    def _on_compact(self) -> None:
        """Compact the context window now (the Controls row's Compact button, the ⋯/Context menu
        entry): fold all but the newest lines of the current context into a running summary. The
        engine returns how many characters were folded; the summary itself is written in the
        background, so the Monitor's context readout catches up on the next answer."""
        try:
            dropped = self._actions.compact()
        except Exception:
            log.exception("actions.compact() failed")
            return
        if dropped:
            self.show_toast("Compacting context…")
        elif self._cfg.context.compaction != "summary":
            self.show_toast("Compaction is off (Models › Context)")
        else:
            self.show_toast("Nothing to compact yet")

    def _on_code(self) -> None:
        """Ask the engine to write the code for the task discussed/shown, right now."""
        if not self._completion_mode:
            self._show_tab(1)
        try:
            self._actions.code()
        except Exception:
            log.exception("actions.code() failed")

    def _on_text(self) -> None:
        """Ask the engine for a plain-English walkthrough of the task discussed/shown, right now
        (the Text button; full overlay only, so the Assist tab it lands in is always shown)."""
        if not self._completion_mode:
            self._show_tab(1)
        try:
            self._actions.text()
        except Exception:
            log.exception("actions.text() failed")

    def _on_pause_toggled(self, checked: bool) -> None:
        self.pause_btn.setText("Resume" if checked else "Pause")
        if self.slim_pause_btn.isChecked() != checked:
            self.slim_pause_btn.blockSignals(True)
            self.slim_pause_btn.setChecked(checked)
            self.slim_pause_btn.blockSignals(False)
            self.slim_pause_btn.setText("Resume" if checked else "Pause")
        self._actions.set_paused(checked)
        self.screen_view.set_paused(checked)
        self.show_toast("Paused" if checked else "Resumed")

    def _on_reset(self) -> None:
        self._actions.new_session()
        self.assist_view.clear()
        self.live_view.clear()
        self.notes_view.clear()
        self._set_notes_badge(False)
        if self.dashboard_view is not None:
            self.dashboard_view.clear()
        self.screen_view.finish_suggestion("", cancelled=True, footer="")
        self.monitor.clear()
        self._plan_text = self._explain_text = ""
        self._plan_job = self._explain_job = None
        self.slim_panel.clear()
        if not self.slim_panel.isHidden():
            self.slim_panel.hide()
            self._apply_slim_size()
        self.show_toast("Session reset")

    def _build_tabstrip(self) -> QWidget:
        strip = QWidget(self.root)
        # A 4th tab ("Screen") no longer fits the theme's default #tab padding/font at the
        # overlay's 340 px floor width; a smaller, tighter style just for this strip (a
        # widget's own style sheet wins over the inherited one) keeps all four on one row.
        strip.setStyleSheet("QPushButton#tab { padding: 3px 5px; font-size: 9pt; }")
        # Like the control row: the strip's own minimum (four tabs + the hands-free label and
        # switch, wider still under the offscreen QPA's fallback font) must never raise the
        # window's 340 px floor; the eliding label gives way first when the strip is squeezed.
        strip.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(strip)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._tab_group = QButtonGroup(strip)
        self._tab_group.setExclusive(True)
        self._tab_buttons: list[QPushButton] = []
        for i, name in enumerate(("Live", "Assist", "Notes", "Screen")):
            btn = QPushButton(name, strip)
            btn.setObjectName("tab")
            btn.setCheckable(True)
            btn.setChecked(i == _START_TAB)
            self._tab_group.addButton(btn, i)
            layout.addWidget(btn)
            self._tab_buttons.append(btn)
        layout.addStretch(1)
        # Manual / Auto advance switch, left of the hands-free pair. ON = manual: nothing is
        # generated until a mouse button is pressed and held (left: Text, middle: Code); the
        # transcript, screen reading and auto-checkpoints keep running. Usable from every tab,
        # with hands-free on or off, and it stays clickable in the hands-free view.
        self.manual_label = _ElidingLabel("Manual", strip)
        self.manual_label.setObjectName("subtitle")
        self.manual_switch = ToggleSwitch(strip)
        manual_tip = (
            "Manual advance: generate only on the mouse gesture, anywhere on screen -- hold the left "
            "button for Text; click (or hold) the middle button / three-finger-tap the touchpad for "
            "Code. Which action needs the hold is set on the Hands-free settings tab. Transcript, "
            "screen reading and checkpoints keep running; the overlay never comes to the front. "
            "Off = Auto: everything automatic works as usual (Ctrl+Alt+J)."
        )
        self.manual_label.setToolTip(manual_tip)
        self.manual_switch.setToolTip(manual_tip)
        self.manual_switch.blockSignals(True)
        self.manual_switch.setChecked(self._manual)   # [handsfree].manual; the hook installs after _install_hotkeys
        self.manual_switch.blockSignals(False)
        self.manual_switch.toggled.connect(self._on_manual_toggled)
        layout.addWidget(self.manual_label)
        layout.addWidget(self.manual_switch)
        # Hands-free switch, right-aligned. Enabled only while the Assist tab is current (its
        # output lands there); once on, the tabs disappear (_apply_layout) and only this switch
        # and the title bar stay clickable, so it is the way back out.
        self.handsfree_label = _ElidingLabel("Hands-free", strip)
        self.handsfree_label.setObjectName("subtitle")
        self.handsfree_switch = ToggleSwitch(strip)
        tip = (
            "Hands-free: sample the screen and, on a new question or a big scroll, automatically read "
            "it, checkpoint, write the walkthrough and the code. The overlay becomes a click-through "
            "output pane; only the title bar and these switches stay clickable (Ctrl+Alt+H). Available "
            "from the Assist tab."
        )
        self.handsfree_label.setToolTip(tip)
        self.handsfree_switch.setToolTip(tip)
        self.handsfree_switch.setEnabled(False)   # set from the current tab by _show_tab(_START_TAB) in _build_ui
        self.handsfree_switch.toggled.connect(self._on_handsfree_toggled)
        layout.addWidget(self.handsfree_label)
        layout.addWidget(self.handsfree_switch)
        self._tab_group.idClicked.connect(self._show_tab)
        return strip

    # ---------------------------------------------------------------- hands-free --
    def _on_handsfree_toggled(self, checked: bool) -> None:
        """The switch was flipped by the user (or the hotkey / the slim bar's menu entry)."""
        self._set_hands_free(checked)

    def _set_hands_free(self, enabled: bool, *, notify_engine: bool = True) -> None:
        """Enter / leave the hands-free view and (unless mirroring a bus event) tell the engine.

        The view (``_apply_layout``): the Assist tab's output only -- tabs, ask box, control row,
        status bar, size grip, question label, footer, choice buttons, Stop / Answer-last and
        the history toggle all go; the expanded history stays; the title bar (drag, ⋯ menu,
        settings) and the switch remain, and the content underneath becomes click-through
        (``_update_hands_free_click_through``). The title bar's subtitle shows the controller's
        state instead of the model name.
        """
        changed = enabled != self._hands_free
        self._hands_free = enabled
        if self.handsfree_switch.isChecked() != enabled:
            self.handsfree_switch.blockSignals(True)
            self.handsfree_switch.setChecked(enabled)
            self.handsfree_switch.blockSignals(False)
        if enabled and not self._completion_mode:
            self._show_tab(1)
        self._apply_layout()
        if notify_engine:
            try:
                self._actions.set_hands_free(enabled)
            except Exception:
                log.exception("actions.set_hands_free(%r) failed", enabled)
        if changed:
            self.show_toast("Hands-free ON" if enabled else "Hands-free OFF")

    def _apply_layout(self) -> None:
        """Show / hide the full overlay's parts for the hands-free view (a no-op in completion
        mode, whose slim bar owns the layout, and in dashboard mode, whose board does;
        ``_set_completion_mode`` / ``_set_dashboard_mode`` re-apply this on exit).

        Hands-free wins over the View preferences: it shows only the Assist output, the title bar
        and the Manual / Hands-free switches (so there is always a way back out), hiding the
        control row, status strip, Monitor and the tab *buttons* while keeping the tab *strip*
        (the switches live there). When hands-free is off, which strips show is left to
        ``_apply_view`` (the View menu)."""
        if self._completion_mode or self._dashboard_mode:
            return
        hf = self._hands_free
        self.assist_view.set_hands_free_view(hf)
        if hf:
            self.askbox.setVisible(False)
            self.clickthrough_btn.setVisible(False)
            self.control_row.setVisible(False)
            self.status.setVisible(False)
            self.size_grip.setVisible(False)
            self.monitor.setVisible(False)
            self.tabstrip.setVisible(True)              # keep the strip -- its switches are the way out
            for btn in self._tab_buttons:
                btn.setVisible(False)                   # but not the tab buttons
            self.stack.setCurrentIndex(1)
        else:
            self.askbox.setVisible(True)
            self.clickthrough_btn.setVisible(True)
            self.size_grip.setVisible(True)
            self._restore_click_through()
            self._release_hover_hooks()
            self._apply_view()                          # control row / status / tabs / monitor per [ui].show_*
        self._update_handsfree_switch_enabled()
        self._apply_hands_free_subtitle()

    def _apply_view(self) -> None:
        """Apply the View preferences ([ui].show_tabs / show_controls / show_status / show_monitor)
        to the full overlay. A no-op in completion mode (the slim bar owns the layout) and while
        hands-free is on (``_apply_layout`` forces its minimal view there). Each strip hides
        independently; the title bar is never touched, so the panel is always reachable, and when
        the tab strip is hidden the content stack keeps whatever tab was last selected. In
        dashboard mode the board stands in for the tab strip (left hidden here) and the hands-free
        view waits for the board to close, so only the control row / status / Monitor apply."""
        if self._completion_mode or (self._hands_free and not self._dashboard_mode):
            return
        ui = self._cfg.ui
        if not self._dashboard_mode:
            self.tabstrip.setVisible(bool(ui.show_tabs))
            for btn in self._tab_buttons:
                btn.setVisible(bool(ui.show_tabs))
        self.control_row.setVisible(bool(ui.show_controls))
        self.status.setVisible(bool(ui.show_status))
        self.monitor.setVisible(bool(ui.show_monitor))

    def _set_view_option(self, key: str, shown: bool) -> None:
        """A View-menu toggle changed: mirror it on ``[ui]``, re-apply the layout, and persist it
        through the engine (``apply_settings`` writes the config file; the key is UI-only, so the
        engine simply saves it)."""
        attr = f"show_{key}"
        if not hasattr(self._cfg.ui, attr):
            return
        setattr(self._cfg.ui, attr, bool(shown))
        self._apply_view()
        try:
            self._actions.apply_settings({f"ui.{attr}": bool(shown)})
        except Exception:
            log.exception("actions.apply_settings() failed for ui.%s", attr)

    def _update_handsfree_switch_enabled(self) -> None:
        """The switch is usable from the Assist tab, and always while hands-free is on (it is
        the way out)."""
        self.handsfree_switch.setEnabled(self._hands_free or self.stack.currentIndex() == 1)

    def _apply_hands_free_subtitle(self) -> None:
        if self._hands_free:
            self.subtitle_label.setText(f"hands-free · {self._hf_state or 'on'}")
        else:
            self.subtitle_label.setText(self._cfg.llm.model)

    def _on_hands_free_event(self, ev: HandsFreeEvent) -> None:
        """The engine's controller changed state (or was switched by settings / the console)."""
        self._hf_state = ev.state + (f" · {ev.detail}" if ev.detail else "")
        if ev.enabled != self._hands_free:
            self._set_hands_free(ev.enabled, notify_engine=False)
        self._apply_hands_free_subtitle()

    # ---------------------------------------------------------------- manual advance --
    def _on_manual_toggled(self, checked: bool) -> None:
        """The Manual switch was flipped by the user (or the hotkey / the slim bar's menu entry)."""
        self._set_manual(checked)

    def _set_manual(self, enabled: bool, *, notify_engine: bool = True) -> None:
        """Manual / Auto advance. Mirrors the switch, installs / removes the press-and-hold mouse
        hook, and (unless mirroring a bus event) tells the engine, which flips
        ``[handsfree].manual`` -- the flag the assistant's automatic lanes and the hands-free
        controller read live -- and publishes the `ManualAdvanceEvent` every other surface follows."""
        changed = enabled != self._manual
        self._manual = enabled
        if self.manual_switch.isChecked() != enabled:
            self.manual_switch.blockSignals(True)
            self.manual_switch.setChecked(enabled)
            self.manual_switch.blockSignals(False)
        if enabled:
            self._install_manual_hook()
        else:
            self._uninstall_manual_hook()
        if notify_engine:
            try:
                self._actions.set_manual_advance(enabled)
            except Exception:
                log.exception("actions.set_manual_advance(%r) failed", enabled)
        if changed:
            self.show_toast("Manual advance ON — hold left = Text, click middle = Code" if enabled
                            else "Auto advance")

    def _install_manual_hook(self) -> None:
        """The press-and-hold gesture hook (``llhooks.PressHoldMouseHook``): installed only while
        manual advance is on (its per-move callback cost is paid only then), only on the real
        Windows platform. The hold time and the quick-middle-click option are read live from the
        config at every press, so a settings change needs nothing here."""
        if PressHoldMouseHook is None or not _is_windows_platform():
            return
        if self._manual_hook is None:
            cfg = self._cfg
            self._manual_hook = PressHoldMouseHook(
                {
                    "left": lambda: QTimer.singleShot(0, self, lambda: self._manual_fire("text")),
                    "middle": lambda: QTimer.singleShot(0, self, lambda: self._manual_fire("code")),
                },
                hold_s=lambda: cfg.handsfree.manual_hold_s,
                require_hold=lambda button: (cfg.handsfree.manual_hold_text if button == "left"
                                             else cfg.handsfree.manual_hold_code),
                schedule=lambda delay_s, check: QTimer.singleShot(int(delay_s * 1000), self, check),
            )
        if not self._manual_hook.installed and not self._manual_hook.failed:
            if not self._manual_hook.install():
                self.show_toast("Manual advance: the mouse hook could not be installed")

    def _uninstall_manual_hook(self) -> None:
        if self._manual_hook is not None and self._manual_hook.installed:
            try:
                self._manual_hook.uninstall()
            except Exception:
                log.exception("failed to uninstall the manual-advance mouse hook")

    def _manual_fire(self, kind: str) -> None:
        """A press-and-hold fired (deferred off the hook callback): generate now. Never touches
        the window -- no show / raise / focus -- so the overlay keeps its translucent, background
        state; the output streams into the Assist view like any other answer."""
        if not self._manual:
            return
        if self._gesture_blocked_at(QCursor.pos()):
            return   # a long press on the app's own UI (a scrollbar, the title bar) is not the gesture
        self.show_toast("Manual · Code" if kind == "code" else "Manual · Text")
        try:
            self._actions.manual_advance(kind)
        except Exception:
            log.exception("actions.manual_advance(%r) failed", kind)

    def _gesture_blocked_at(self, pos: QPoint) -> bool:
        """Whether a press-and-hold at a global position landed on the app's own UI rather than
        on "the screen". The settings dialog and the region picker always block. The overlay
        itself blocks only where it actually takes the click: in the hands-free view just the
        title bar / the switches (`_hands_free_clickable_at` -- the content is click-through, so
        a hold over the answer text is a hold on the app underneath and must fire); with the
        user's click-through mode on, nowhere at all; otherwise its whole frame."""
        for w in (self._settings, self._region_picker):
            try:
                if w is not None and w.isVisible() and w.frameGeometry().contains(pos):
                    return True
            except Exception:  # noqa: BLE001 - a widget mid-destruction must not kill the gesture
                continue
        try:
            if not self.isVisible() or not self.frameGeometry().contains(pos):
                return False
            if self._hands_free_view_active:
                return self._hands_free_clickable_at(pos)
            if self._click_through:
                return False
            return True
        except Exception:  # noqa: BLE001
            log.exception("manual advance: own-window check failed")
            return True    # fail closed: better a dropped gesture than one fired off our own UI

    def _hands_free_clickable_at(self, pos: QPoint) -> bool:
        """Whether a global cursor position is over a part of the hands-free view that must stay
        clickable: the title bar (drag, menu, settings, hide) or the switch and its label -- and,
        only when the wheel hook is not available, an output pane (so the native wheel reaches
        it; see ``_hands_free_scroll_area_at``)."""
        for w in (self.titlebar, self.handsfree_switch, self.handsfree_label,
                  self.manual_switch, self.manual_label):
            if not w.isVisible():
                continue
            if QRect(w.mapToGlobal(QPoint(0, 0)), w.size()).contains(pos):
                return True
        if not self._wheel_hook_usable and self._hands_free_scroll_area_at(pos) is not None:
            return True
        return False

    @property
    def _wheel_hook_usable(self) -> bool:
        """Whether wheel scrolling over a click-through pane can go through the low-level hook
        (``llhooks.MouseWheelHook``); when not (no module, or the hook failed to install) the
        panes are made clickable while hovered instead."""
        return MouseWheelHook is not None and not (self._wheel_hook is not None and self._wheel_hook.failed)

    def _hands_free_scroll_area_at(self, pos: QPoint) -> "QAbstractScrollArea | None":
        """The output pane under a global cursor position in the hands-free view -- the answer
        browser, or the history list when it is expanded (any ``QAbstractScrollArea`` of the
        view) -- else None."""
        if not (self._hands_free_view_active and self.isVisible()):
            return None
        if not self.frameGeometry().contains(pos):
            return None
        w = self.childAt(self.mapFromGlobal(pos))
        while w is not None and w is not self:
            if isinstance(w, QAbstractScrollArea) and w.isVisible():
                return w
            w = w.parentWidget()
        return None

    def _on_hands_free_wheel(self, delta: int, horizontal: bool) -> bool:
        """``MouseWheelHook`` callback (main thread, inside the hook): scroll the output pane
        under the cursor by three lines per notch and swallow the notch, so the window underneath
        does not scroll too. False (pass the notch on) when the cursor is not over a pane."""
        area = self._hands_free_scroll_area_at(QCursor.pos())
        if area is None:
            return False
        bar = area.horizontalScrollBar() if horizontal else area.verticalScrollBar()
        step = max(1, bar.singleStep()) * 3
        bar.setValue(int(round(bar.value() - (delta / 120.0) * step)))
        return True

    def _history_snap(self, direction: int) -> None:
        """An Up (-1) / Down (+1) tap while the history is hovered: snap to the previous / next
        entry (``AssistView.history_snap``)."""
        try:
            self.assist_view.history_snap(direction)
        except Exception:
            log.exception("history snap failed")

    def _update_hands_free_hover_hooks(self, pos: QPoint) -> None:
        """Keep the wheel hook installed exactly while the cursor is over an output pane of the
        hands-free view, and the Up / Down hook exactly while it is over the expanded history
        (so their per-message cost is paid only then, and the arrow keys keep working everywhere
        else)."""
        area = self._hands_free_scroll_area_at(pos) if MouseWheelHook is not None else None
        if area is not None:
            if self._wheel_hook is None:
                self._wheel_hook = MouseWheelHook(self._on_hands_free_wheel)
            if not self._wheel_hook.installed and not self._wheel_hook.failed:
                self._wheel_hook.install()
        elif self._wheel_hook is not None and self._wheel_hook.installed:
            self._wheel_hook.uninstall()
        over_history = area is not None and area is self.assist_view.history_scroll and BareKeyHook is not None
        if over_history:
            if self._nav_key_hook is None:
                self._nav_key_hook = BareKeyHook({
                    "up": lambda: QTimer.singleShot(0, lambda: self._history_snap(-1)),
                    "down": lambda: QTimer.singleShot(0, lambda: self._history_snap(1)),
                })
            if not self._nav_key_hook.installed and not self._nav_key_hook.failed:
                self._nav_key_hook.install()
        elif self._nav_key_hook is not None and self._nav_key_hook.installed:
            self._nav_key_hook.uninstall()

    def _release_hover_hooks(self) -> None:
        for hook in (self._wheel_hook, self._nav_key_hook):
            if hook is not None and hook.installed:
                try:
                    hook.uninstall()
                except Exception:
                    log.exception("failed to uninstall a hover hook")

    def _update_hands_free_click_through(self) -> None:
        """Cursor-tracking partial click-through for the hands-free view (every poll tick).

        ``WS_EX_TRANSPARENT`` is a whole-window flag, so the window is made click-through while
        the cursor is over the content and clickable again while it is over the title bar or the
        switch (``QCursor.pos()`` still works while the flag is set: the OS routes clicks, not
        the cursor). The flag is only touched when the answer changes; the user's own
        click-through setting is restored when the view ends (``_restore_click_through``). The
        hover hooks (``_update_hands_free_hover_hooks``) follow the same poll: over an output
        pane the wheel scrolls that pane although the window is click-through there, and over
        the history Up / Down taps snap through its entries.
        """
        if not (self._hands_free_view_active and self.isVisible()):
            self._release_hover_hooks()
            return
        if not (_is_windows_platform() and win32 is not None):
            return
        pos = QCursor.pos()
        self._update_hands_free_hover_hooks(pos)
        want = not self._hands_free_clickable_at(pos)
        if want == self._hf_click_through:
            return
        self._hf_click_through = want
        try:
            win32.set_click_through(int(self.winId()), want or self._click_through)
        except Exception:
            log.exception("hands-free click-through update failed")

    def _restore_click_through(self) -> None:
        """Leave the hands-free view's cursor-tracking click-through: back to the user's setting."""
        if self._hf_click_through is None:
            return
        self._hf_click_through = None
        self.set_click_through(self._click_through)

    def _apply_geometry(self) -> None:
        width = max(_MIN_WIDTH, self._cfg.ui.width)
        height = max(_MIN_HEIGHT, self._cfg.ui.height)
        x, y = self._cfg.ui.x, self._cfg.ui.y
        placed = x >= 0 and y >= 0
        screen = (QGuiApplication.screenAt(QPoint(x, y)) if placed else None) or QGuiApplication.primaryScreen()
        geo = screen.availableGeometry() if screen is not None else None
        if geo is not None:
            # A saved size can be larger than the screen (a smaller monitor, or a window that
            # was grown past it before exit); never restore one that cannot fit.
            width = min(width, geo.width())
            height = min(height, geo.height())
        if not placed:
            if geo is not None:
                x = geo.right() - width - _EDGE_MARGIN
                y = geo.top() + _EDGE_MARGIN
            else:  # pragma: no cover - no screen at all
                x, y = 100, 100
        self.setGeometry(x, y, width, height)
        self.move(_on_screen_pos(self, QPoint(x, y)))

    # ---------------------------------------------------------------- menu --------
    def _rebuild_menu(self) -> None:
        # The submenus from the previous opening (Assist, Role, Rename speakers) are children of
        # the menu that clear() leaves alive: drop them, or every opening leaks a few.
        for submenu in self._menu.findChildren(QMenu, options=Qt.FindChildOption.FindDirectChildrenOnly):
            submenu.deleteLater()
        self._menu.clear()
        self._populate_assist_menu(self._menu.addMenu("Assist"))
        self._populate_view_menu(self._menu.addMenu("View"))
        dashboard_action = self._menu.addAction(f"Dashboard mode{_hotkey_hint(self._cfg.hotkeys.toggle_dashboard)}")
        dashboard_action.setCheckable(True)
        dashboard_action.setChecked(self._dashboard_mode)
        dashboard_action.setEnabled(self.dashboard_view is not None)
        dashboard_action.toggled.connect(lambda checked: self._set_dashboard_mode(checked))
        self._menu.addSeparator()
        export_action = self._menu.addAction("Export session")
        export_action.triggered.connect(self._on_export)
        new_action = self._menu.addAction("New session")
        new_action.triggered.connect(self._actions.new_session)
        settings_action = self._menu.addAction("Settings…")
        settings_action.triggered.connect(self.open_settings)
        models_action = self._menu.addAction("Models…")
        models_action.triggered.connect(lambda: self.open_models())
        context_action = self._menu.addAction("Context documents…")   # 2026-10-06: the Context tab
        context_action.triggered.connect(lambda: self.open_models("Context"))

        self._menu.addSeparator()
        compact_action = self._menu.addAction("Compact context now")
        compact_action.triggered.connect(self._on_compact)

        self._menu.addSeparator()

        rename_menu = self._menu.addMenu("Rename speakers...")
        speakers = self.live_view.known_speakers()
        if not speakers:
            placeholder = rename_menu.addAction("(no speakers yet)")
            placeholder.setEnabled(False)
        for speaker_id, label in speakers.items():
            act = rename_menu.addAction(f"{label} ({speaker_id})")
            act.triggered.connect(
                lambda checked=False, sid=speaker_id, cur=label: self._prompt_rename(sid, cur)
            )

        self._menu.addSeparator()
        quit_action = self._menu.addAction("Quit")
        quit_action.triggered.connect(self.close)

    def _populate_assist_menu(self, menu: QMenu) -> None:
        """The "⋯" menu's Assist submenu, rebuilt each time the menu opens so every check mark
        reads the live config: the co-pilot's on-demand actions, the switches for what it does
        by itself, and the pair-programming settings that change mid-session -- your role (pairs
        swap often) and the session context -- one click away instead of in the settings dialog."""
        a, h = self._cfg.assist, self._cfg.hotkeys
        menu.addAction(f"Suggest questions now{_hotkey_hint(h.insights)}").triggered.connect(
            lambda: self._on_insights_now(("questions",)))
        menu.addAction("Blind spots && follow-ups now").triggered.connect(
            lambda: self._on_insights_now(("blindspots",)))
        menu.addAction(f"Recap the session{_hotkey_hint(h.recap)}").triggered.connect(lambda: self._on_recap())
        menu.addAction(f"Regenerate notes{_hotkey_hint(h.notes)}").triggered.connect(
            lambda: self._actions.generate_notes())
        menu.addAction("Refresh the dashboard now").triggered.connect(lambda: self._on_dashboard_refresh(()))
        menu.addSeparator()
        switches = (
            ("Suggest questions automatically", a.suggest_questions,
             lambda checked: self._set_insight_option(questions=checked)),
            ("Flag blind spots && follow-ups automatically", a.flag_blind_spots,
             lambda checked: self._set_insight_option(blind_spots=checked)),
            (f"Auto-answer spoken questions{_hotkey_hint(h.toggle_auto)}", self.auto_btn.isChecked(),
             self.auto_btn.setChecked),
            ("Solve / suggest when the screen changes", self._cfg.screen.auto_suggest,
             lambda checked: self._set_screen_option("auto_suggest", checked)),
            ("Answer questions that appear on screen", self._cfg.screen.answer_screen_questions,
             lambda checked: self._set_screen_option("answer_screen_questions", checked)),
        )
        for label, on, apply in switches:
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(bool(on))
            act.toggled.connect(apply)
        menu.addSeparator()
        role = normalize_role(a.role)
        role_menu = menu.addMenu(f"Role: {role.capitalize()}")
        group = QActionGroup(role_menu)
        group.setExclusive(True)
        for key, label in _ROLE_LABELS:
            act = role_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(key == role)
            group.addAction(act)
            act.triggered.connect(lambda _checked=False, r=key: self._set_role(r))
        context = " ".join((a.session_context or "").split())
        shown = (context[:40] + "...") if len(context) > 43 else context
        menu.addAction(
            f"Session context: {shown.replace('&', '&&')}" if context else "Session context... (none set)"
        ).triggered.connect(lambda: self._prompt_session_context())
        menu.addSeparator()
        menu.addAction("Assist settings…").triggered.connect(lambda: self.open_settings("Assist"))

    # ---------------------------------------------------------------- View menu ---
    def _current_view(self) -> str:
        """What is on screen: "dashboard", "completion" (the slim bar) or one of the full
        overlay's tabs ("live" / "assist" / "notes" / "screen")."""
        if self._dashboard_mode:
            return "dashboard"
        if self._completion_mode:
            return "completion"
        index = self.stack.currentIndex()
        return _TAB_KEYS[index] if 0 <= index < len(_TAB_KEYS) else "assist"

    def _populate_view_menu(self, menu: QMenu) -> None:
        """The "⋯" menu's View submenu (2026-10-21), rebuilt each time the menu opens so every
        check mark reads the live state: which view is on screen (switchable), that view's own
        display options first -- the dashboard's columns, each one hideable; the Screen tab's
        source and OCR readout; the Assist tab's history pane; the Live tab's partials -- then
        the settings every view shares: the screen source (active window, a monitor, a region,
        off; pause) and the repository the answers and the dashboard look things up in. Between
        the two, which strips of the full overlay show (2026-10-07, `_build_view_submenu`)."""
        cfg = self._cfg
        view = self._current_view()
        group = QActionGroup(menu)
        group.setExclusive(True)
        for key, label in _VIEW_LABELS:
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(key == view)
            act.setEnabled(key != "dashboard" or self.dashboard_view is not None)
            group.addAction(act)
            act.triggered.connect(lambda _checked=False, k=key: self._switch_view(k))
        if view == "completion":
            here = menu.addAction("Completion mode (the slim bar)")
            here.setCheckable(True)
            here.setChecked(True)
            here.setEnabled(False)
        menu.addSeparator()
        if view == "dashboard" and self.dashboard_view is not None:
            head = menu.addAction("Dashboard columns")
            head.setEnabled(False)
            for key, title in dashboard_mod.COLUMNS:
                flag = dashboard_mod.COLUMN_FLAGS[key]
                act = menu.addAction(title)
                act.setCheckable(True)
                act.setChecked(bool(getattr(cfg.dashboard, flag, True)))
                act.toggled.connect(lambda checked, f=flag: self._set_dashboard_option(f, checked))
            prompts_action = menu.addAction("Suggested prompts under the action items")
            prompts_action.setCheckable(True)
            prompts_action.setChecked(bool(cfg.dashboard.show_prompts))
            prompts_action.toggled.connect(lambda checked: self._set_dashboard_option("show_prompts", checked))
            resolved_action = menu.addAction("Answered / done items (collapsed under the open ones)")
            resolved_action.setCheckable(True)
            resolved_action.setChecked(bool(cfg.dashboard.show_resolved))
            resolved_action.toggled.connect(lambda checked: self._set_dashboard_option("show_resolved", checked))
            menu.addSeparator()
        elif view == "live":
            partials = menu.addAction("In-progress speech (partial transcripts)")
            partials.setCheckable(True)
            partials.setChecked(bool(cfg.asr.partials_enabled))
            partials.toggled.connect(lambda checked: setattr(cfg.asr, "partials_enabled", bool(checked)))
            menu.addSeparator()
        elif view == "assist":
            history = menu.addAction("Answer history pane")
            history.setCheckable(True)
            history.setChecked(self.assist_view.history_toggle.isChecked())
            history.toggled.connect(self.assist_view.history_toggle.setChecked)
            explain = menu.addAction(f"Explanations (walkthrough · comments · explanation){_hotkey_hint(cfg.hotkeys.toggle_explanations)}")
            explain.setCheckable(True)
            explain.setChecked(self.explain_btn.isChecked())
            explain.toggled.connect(self.explain_btn.setChecked)
            menu.addSeparator()
        elif view == "screen":
            readout = menu.addAction("OCR readout (the raw screen text)")
            readout.setCheckable(True)
            readout.setChecked(self.screen_view.screen_text_toggle.isChecked())
            readout.toggled.connect(self.screen_view.screen_text_toggle.setChecked)
            menu.addSeparator()
        head = menu.addAction("Overlay strips")
        head.setEnabled(False)
        self._build_view_submenu(menu)
        menu.addSeparator()
        self._populate_screen_source_menu(menu.addMenu("Screen source"))
        pause_action = menu.addAction("Screen: pause/resume")
        pause_action.triggered.connect(self._on_toggle_screen)
        self._populate_repository_menu(menu.addMenu("Repository"))

    def _populate_screen_source_menu(self, menu: QMenu) -> None:
        """Off / the active window / one of the monitors / a region: the Screen tab's mode row
        and ``[screen].monitor``, as a radio list. Monitors are numbered the way the capture
        numbers them -- the primary one first, then the others -- with their size and position."""
        cfg = self._cfg
        group = QActionGroup(menu)
        group.setExclusive(True)
        for mode, label in _SCREEN_SOURCE_LABELS:
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(self._screen_mode == mode)
            group.addAction(act)
            act.triggered.connect(lambda _checked=False, m=mode: self._on_screen_mode_requested(m))
        screens = list(QGuiApplication.screens())
        primary = QGuiApplication.primaryScreen()
        ordered = ([primary] if primary in screens else []) + [s for s in screens if s is not primary]
        for index, screen in enumerate(ordered):
            geo = screen.geometry()
            label = f"Monitor {index}" + (" (primary)" if screen is primary else "") + f" · {geo.width()}×{geo.height()}"
            if screen is not primary:
                label += f" at {geo.x()},{geo.y()}"
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(self._screen_mode == "monitor" and int(cfg.screen.monitor) == index)
            group.addAction(act)
            act.triggered.connect(lambda _checked=False, i=index: self._set_screen_monitor(i))
        region = menu.addAction("Region…" if self._screen_mode != "region" else "Region… (change it)")
        region.setCheckable(True)
        region.setChecked(self._screen_mode == "region")
        group.addAction(region)
        region.triggered.connect(self._open_region_picker)

    def _populate_repository_menu(self, menu: QMenu) -> None:
        """Which repository the lookups and the toolkit discovery use: auto-detected from Claude
        Code's own session logs (the default), or a folder chosen here / in the settings."""
        c = self._cfg.project
        current = menu.addAction(self._project_detail or "Project: not known yet")
        current.setEnabled(False)
        menu.addSeparator()
        auto = menu.addAction("Auto-detect from Claude Code (the project of its latest session)")
        auto.setCheckable(True)
        auto.setChecked(bool(c.repo_auto))
        auto.setEnabled(not (c.repo_root or "").strip())
        auto.toggled.connect(lambda checked: self._apply_project_option("repo_auto", bool(checked)))
        root = (c.repo_root or "").strip()
        choose = menu.addAction(f"Folder: {root}" if root else "Choose a folder…")
        choose.triggered.connect(self._choose_repo_root)
        if root:
            clear = menu.addAction("Clear the folder (back to auto-detect)")
            clear.triggered.connect(lambda: self._apply_project_option("repo_root", ""))
        menu.addSeparator()
        menu.addAction("Project settings…").triggered.connect(lambda: self.open_settings("Project"))

    def _switch_view(self, key: str) -> None:
        if key == "dashboard":
            self._set_dashboard_mode(True)
            return
        if self._dashboard_mode:
            self._set_dashboard_mode(False)
        if self._completion_mode:
            self.completion_btn.setChecked(False)
        if key in _TAB_KEYS:
            self._show_tab(_TAB_KEYS.index(key))

    def _set_dashboard_option(self, flag: str, value: bool) -> None:
        """A View-menu switch for the board: the config, the view, and the engine (which saves)."""
        setattr(self._cfg.dashboard, flag, bool(value))
        if self.dashboard_view is not None:
            self.dashboard_view.apply_config()
        try:
            self._actions.apply_settings({f"dashboard.{flag}": bool(value)})
        except Exception:
            log.exception("actions.apply_settings(dashboard.%s) failed", flag)

    def _set_screen_monitor(self, index: int) -> None:
        """Read monitor `index` (the capture's numbering, primary = 0) from now on."""
        self._cfg.screen.monitor = int(index)
        try:
            self._actions.apply_settings({"screen.monitor": int(index)})
        except Exception:
            log.exception("actions.apply_settings(screen.monitor) failed")
        self._on_screen_mode_requested("monitor")

    def _apply_project_option(self, key: str, value: object) -> None:
        setattr(self._cfg.project, key, value)
        try:
            self._actions.apply_settings({f"project.{key}": value})   # the engine re-detects and says what it found
        except Exception:
            log.exception("actions.apply_settings(project.%s) failed", key)

    def _choose_repo_root(self) -> None:
        from pathlib import Path

        from PySide6.QtWidgets import QFileDialog

        start = (self._cfg.project.repo_root or "").strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Repository root (the project being worked on)", start)
        if path:
            self._apply_project_option("repo_root", path)

    def _on_insights_now(self, sections: tuple[str, ...] = ()) -> None:
        """Refresh the co-pilot insights now (an Assist-menu entry or the hotkey; the Notes tab's
        own buttons call the engine directly) and show the Notes tab, where they land."""
        if not self._completion_mode:
            self._show_tab(_NOTES_TAB)
        try:
            self._actions.insights_now(tuple(sections))
        except Exception:
            log.exception("actions.insights_now() failed")

    def _on_recap(self) -> None:
        """Recap the session now; the answer streams into the Assist tab."""
        if not self._completion_mode:
            self._show_tab(1)
        try:
            self._actions.recap()
        except Exception:
            log.exception("actions.recap() failed")

    def _set_insight_option(self, *, questions: bool | None = None, blind_spots: bool | None = None) -> None:
        try:
            self._actions.set_insight_options(questions=questions, blind_spots=blind_spots)
        except Exception:
            log.exception("actions.set_insight_options() failed")
            return
        # The engine sets these too; keep the menu's view current whatever the actions object is.
        if questions is not None:
            self._cfg.assist.suggest_questions = bool(questions)
        if blind_spots is not None:
            self._cfg.assist.flag_blind_spots = bool(blind_spots)

    def _set_role(self, role: str) -> None:
        role = normalize_role(role)
        try:
            self._actions.set_role(role)
        except Exception:
            log.exception("actions.set_role(%r) failed", role)
            return
        self._cfg.assist.role = role
        self.show_toast(f"Role: {role} -- suggested questions now go to {_ROLE_AUDIENCE[role]}")

    def _prompt_session_context(self) -> None:
        text, ok = QInputDialog.getText(
            self, "Session context",
            "What is this session about? Project, stack, today's goal -- every answer, insight and recap uses it:",
            text=self._cfg.assist.session_context,
        )
        if not ok:
            return
        try:
            self._actions.set_session_context(text)
        except Exception:
            log.exception("actions.set_session_context() failed")
            return
        self._cfg.assist.session_context = " ".join(text.split())
        self.show_toast("Session context set" if text.strip() else "Session context cleared")

    def _set_screen_option(self, key: str, value: bool) -> None:
        """A screen switch flipped in the Assist menu: config, engine, and the Screen tab's checkbox."""
        if key == "auto_suggest":
            try:
                self._actions.set_auto_suggest(bool(value))
            except Exception:
                log.exception("actions.set_auto_suggest() failed")
        setattr(self._cfg.screen, key, bool(value))
        self.screen_view.sync_toggles()

    def _on_insights_updated(self, ev: InsightsUpdatedEvent) -> None:
        """New co-pilot insights: into the Notes tab's pane -- and, while another tab is in front,
        a dot on the Notes tab instead of switching to it (a background pass must never pull the
        driver's eyes off their work)."""
        self.notes_view.set_insights(ev.questions, ev.blindspots, ev.generated_at, ev.auto)
        if self.stack.currentIndex() != _NOTES_TAB and (ev.questions or ev.blindspots):
            self._set_notes_badge(True)

    def _set_notes_badge(self, on: bool) -> None:
        if len(self._tab_buttons) > _NOTES_TAB:
            btn = self._tab_buttons[_NOTES_TAB]
            btn.setText("Notes •" if on else "Notes")
            btn.setToolTip("New co-pilot insights: questions to ask, blind spots & follow-ups" if on else "")

    def _build_view_submenu(self, menu: QMenu) -> None:
        """Populate a "View" menu with one checkable entry per overlay strip, reflecting the
        current ``[ui].show_*`` preferences and toggling them through ``_set_view_option``. Shared
        by the ⋯ menu and the completion-mode context menu."""
        menu.setToolTipsVisible(True)
        for key, label, tip in (
            ("tabs", "Tabs", "The top strip: the Live / Assist / Notes / Screen tabs and the "
                              "Manual / Hands-free switches (the current tab's content stays)"),
            ("controls", "Controls", "The bottom button row: Read, Revert, Reset, Pause, "
                                     "Checkpoint, Compact, Text, Code"),
            ("status", "Status", "The bottom status strip: audio levels, ASR / LLM state, capture "
                                 "and screen status"),
            ("monitor", "Monitor", "The performance monitor: LLM speed, the context window and its "
                                   "contents, and CPU / RAM / VRAM use"),
        ):
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(bool(getattr(self._cfg.ui, f"show_{key}", True)))
            act.setToolTip(tip)
            act.toggled.connect(lambda checked, k=key: self._set_view_option(k, checked))

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802 - Qt override
        """Right-click anywhere on the panel's chrome opens the "..." menu (which has Quit), so
        the panel can always be closed even when its title bar is out of reach."""
        self._menu.popup(event.globalPos())
        event.accept()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        owner = self._text_menu_owners.get(obj) if event.type() == QEvent.Type.ContextMenu else None
        if owner is None:
            return super().eventFilter(obj, event)
        menu = owner.createStandardContextMenu()
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        menu.addSeparator()
        menu.addAction("Hide overlay").triggered.connect(self.toggle_visible)
        menu.addAction("Quit ExoCortex").triggered.connect(self.close)
        menu.popup(event.globalPos())
        return True

    def _on_export(self) -> None:
        try:
            path = self._actions.export_session()
            self.show_toast(f"Exported to {path}")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as a toast
            log.exception("export_session() failed")
            self.show_toast(f"Export failed: {exc}")

    def _prompt_rename(self, speaker_id: str, current: str) -> None:
        text, ok = QInputDialog.getText(self, "Rename speaker", "Name:", text=current)
        if ok and text.strip():
            self._actions.rename_speaker(speaker_id, text.strip())

    # ---------------------------------------------------------------- toggles -----
    def _on_auto_toggled(self, checked: bool) -> None:
        self._actions.set_auto(checked)

    def _on_mic_toggled(self, checked: bool) -> None:
        self._actions.set_mic_muted(checked)

    def _show_tab(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        if 0 <= index < len(self._tab_buttons):
            self._tab_buttons[index].setChecked(True)
        if index == _NOTES_TAB:
            self._set_notes_badge(False)
        self._update_handsfree_switch_enabled()

    # ---------------------------------------------------------------- screen reading -
    def _on_screen_mode_requested(self, mode: str) -> None:
        self._screen_mode = mode
        try:
            self._actions.set_screen_mode(mode)
        except Exception:
            log.exception("actions.set_screen_mode(%r) failed", mode)
        self.screen_view.set_mode(mode)

    def _open_region_picker(self) -> None:
        """Lazily create the (GUI-thread-only) RegionPicker and show it, topmost over the overlay."""
        if RegionPicker is None:
            self.show_toast("Region selection unavailable")
            return
        if self._region_picker is None:
            self._region_picker = RegionPicker(self)
            self._region_picker.selected.connect(self._on_region_selected)
            self._region_picker.cancelled.connect(self._on_region_cancelled)
        self._region_picker.begin()

    def _on_region_selected(self, x: int, y: int, w: int, h: int) -> None:
        self._screen_mode = "region"
        try:
            self._actions.set_screen_region(x, y, w, h)
        except Exception:
            log.exception("actions.set_screen_region() failed")
        self.screen_view.set_mode("region")
        self.show_toast(f"Region set {w}x{h}")

    def _on_region_cancelled(self) -> None:
        self.screen_view.set_mode(self._screen_mode)
        self.show_toast("Region selection cancelled")

    def _on_toggle_screen(self) -> None:
        try:
            paused = self._actions.toggle_screen_paused()
        except Exception:
            log.exception("actions.toggle_screen_paused() failed")
            return
        self.screen_view.set_paused(paused)
        self.show_toast("Screen reading paused" if paused else "Screen reading resumed")

    # ---------------------------------------------------------------- hotkeys -----
    def _install_hotkeys(self) -> None:
        if GlobalHotkeys is None or not _is_windows_platform():
            # RegisterHotKey is a real, machine-wide Win32 call: only ever make it on the
            # real "windows" Qt platform, never under the "offscreen" plugin tests use (even
            # on a Windows dev box), so pytest never grabs the user's actual global hotkeys.
            return
        h = self._cfg.hotkeys
        bindings = {
            h.toggle_visible: self.toggle_visible,
            h.answer_last: self._actions.answer_last,
            h.ask_focus: self.focus_ask,
            h.notes: self._actions.generate_notes,
            h.toggle_auto: lambda: self.auto_btn.setChecked(not self.auto_btn.isChecked()),
            h.toggle_mic: lambda: self.mic_btn.setChecked(not self.mic_btn.isChecked()),
            h.click_through: lambda: self.set_click_through(not self._click_through),
            h.move_left: lambda: self.nudge(-_NUDGE_PX, 0),
            h.move_right: lambda: self.nudge(_NUDGE_PX, 0),
            h.move_up: lambda: self.nudge(0, -_NUDGE_PX),
            h.move_down: lambda: self.nudge(0, _NUDGE_PX),
            h.select_region: self._open_region_picker,
            h.suggest_now: lambda: self._actions.suggest_now(),
            h.toggle_screen: self._on_toggle_screen,
            h.toggle_completion: lambda: self.completion_btn.setChecked(not self.completion_btn.isChecked()),
            h.toggle_explanations: lambda: self.explain_btn.setChecked(not self.explain_btn.isChecked()),
            h.checkpoint: self._on_checkpoint,
            h.code: self._on_code,
            h.toggle_handsfree: lambda: self._set_hands_free(not self._hands_free),
            h.toggle_manual: lambda: self._set_manual(not self._manual),
            h.insights: lambda: self._on_insights_now(()),
            h.recap: self._on_recap,
            h.toggle_dashboard: lambda: self._set_dashboard_mode(not self._dashboard_mode),
            h.quit: self.close,
        }
        # A hotkey left empty is off (see HotkeysConfig): never hand "" to RegisterHotKey, where it
        # would only come back as a bogus "Hotkey conflict" toast.
        bindings = {key: callback for key, callback in bindings.items() if str(key or "").strip()}
        pause_key = str(getattr(h, "toggle_pause", "") or "").strip()
        if pause_key and is_bare_key is not None and is_bare_key(pause_key):
            self._install_pause_key(pause_key)
        elif pause_key:
            bindings[pause_key] = self._toggle_pause_hotkey
        try:
            self._hotkeys = GlobalHotkeys(bindings)
        except Exception:
            log.exception("failed to construct GlobalHotkeys")
            self._hotkeys = None
            return
        app = QApplication.instance()
        if app is None:
            log.warning("no QApplication instance; hotkeys not installed")
            return
        try:
            self._hotkeys.install(app)
        except Exception:
            log.exception("failed to install global hotkeys")

    def _install_pause_key(self, key: str) -> None:
        """``[hotkeys].toggle_pause`` as a bare key (``llhooks.BareKeyHook``): a TAP of that key
        -- pressed and released with no other key in between -- toggles Pause / Resume. The right
        Shift key by default: Shift held for a capital letter is never a tap, and a modifier key
        is never swallowed. Only on the real Windows platform (like the hotkeys)."""
        if BareKeyHook is None:
            return
        try:
            hook = BareKeyHook(key, lambda: QTimer.singleShot(0, self._toggle_pause_hotkey))
        except ValueError:
            log.warning("cannot use %r as the pause / resume key", key)
            return
        if hook.install():
            self._pause_key_hook = hook
            log.info("pause / resume: a tap of the %s key", key)
        else:
            log.warning("the pause / resume key %r could not be hooked", key)

    def _toggle_pause_hotkey(self) -> None:
        """The pause / resume hotkey: flips the Pause button (which pauses everything automatic,
        hands-free watching included) and says so in a toast, since the button itself is hidden
        in the hands-free view and the slim bar."""
        paused = not self.pause_btn.isChecked()
        self.pause_btn.setChecked(paused)
        self.show_toast("Paused" if paused else "Resumed")

    # ---------------------------------------------------------------- window mgmt -
    def _keep_on_screen(self) -> None:
        """Pull the panel back when its title strip has ended up off every screen (a monitor was
        unplugged, the resolution changed, the window was resized past a screen edge...)."""
        if QApplication.mouseButtons() != Qt.MouseButton.NoButton:
            return  # mid-drag / mid-resize: do not fight the user
        pos = _on_screen_pos(self, self.pos())
        if pos != self.pos():
            log.info("overlay title bar was off-screen at %s; moving it to %s", self.pos(), pos)
            self.move(pos)

    def _reassert_topmost(self) -> None:
        """Topmost-backstop tick: nothing is restacked while a popup is open.

        A menu or a combo box dropdown is its own top-level window and owns the mouse grab.
        Pushing the overlay - and then the settings dialog - back to the top of the topmost
        band drops that popup *behind* them: the list is still open and still grabbing, but
        invisible, so it reads as the dropdown closing by itself a beat after it was opened
        and the next click only dismisses it. A popup lives for a few seconds; the backstop
        resumes on the first tick after it closes.
        """
        popup_open = QApplication.activePopupWidget() is not None
        if not popup_open:
            self._keep_on_screen()
        if not (_is_windows_platform() and win32 is not None):
            return
        if not popup_open:
            try:
                win32.set_topmost(int(self.winId()))
            except Exception:
                log.exception("set_topmost() failed")
            self._raise_transients()

    def _raise_transients(self) -> None:
        """The settings dialog (and any tool tip over it) must end up above the overlay again
        after the topmost backstop pushed the overlay to the top of the topmost band.

        Raised bottom-up: ``set_topmost`` puts its window at the *top* of the topmost band, so
        the window raised last ends up in front. The dialog goes over the overlay and a tool tip
        over the dialog - the other order hides whatever the user is currently reading. Menus and
        dropdowns are absent on purpose: :meth:`_reassert_topmost` skips the tick entirely while
        one of those is open rather than restacking around it.
        """
        transients: list[QWidget] = []
        if self._settings is not None and self._settings.isVisible():
            transients.append(self._settings)
        transients.extend(
            w for w in QApplication.topLevelWidgets()
            if w.isVisible() and w.windowType() == Qt.WindowType.ToolTip
        )
        for w in transients:
            try:
                win32.set_topmost(int(w.winId()))
            except Exception:
                log.exception("set_topmost() failed for a transient window")

    # ---------------------------------------------------------------- events ------
    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if not self._topmost_timer.isActive():
            self._topmost_timer.start()
        self.set_click_through(self._click_through)
        self._hf_click_through = None   # hands-free's cursor-tracking state re-applies on the next poll

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        for grip in self._edge_grips:
            grip.place(self.size())

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        self._save_geometry()
        self._stop_timers()
        self._teardown_popup_filter()
        if not self._closing_programmatically:
            try:
                self._actions.quit()
            except Exception:
                log.exception("actions.quit() failed")
        if self._hotkeys is not None:
            try:
                self._hotkeys.uninstall()
            except Exception:
                log.exception("failed to uninstall hotkeys")
        self._release_hover_hooks()
        if self._pause_key_hook is not None:
            try:
                self._pause_key_hook.uninstall()
            except Exception:
                log.exception("failed to uninstall the pause key hook")
        self._uninstall_manual_hook()
        self._bus.unsubscribe(self._q)
        super().closeEvent(event)

    def _stop_timers(self) -> None:
        """Stop every recurring QTimer owned (directly or via a child widget) by this
        window. ``close()`` only hides a QWidget (Qt does not delete it), so an
        un-stopped ``QTimer`` keeps firing - and keeps this whole widget tree alive -
        for the rest of the process; this made every offline test leak a live
        30 ms/1.5 s/60 ms timer trio for the remainder of the pytest session."""
        self._poll_timer.stop()
        self._topmost_timer.stop()
        self._toast_timer.stop()
        self.assist_view.shutdown()
        self.notes_view.shutdown()
        if self.dashboard_view is not None:
            self.dashboard_view.shutdown()
        if self._settings is not None:
            self._settings.close()
        if self._ghost is not None:
            self._ghost.hide_completion()
            self._ghost.close()

    def _teardown_popup_filter(self) -> None:
        """Remove the app-wide popup-topmost filter installed in ``__init__``.

        The filter must not outlive the overlay it serves: after this window closes it would
        keep pushing *any* window shown later in this process to the top of the z-order, which
        is wrong, and it leaks state across anything that shares one QApplication (unit tests
        included). The filter QObject itself is a child of this window, so it dies with it;
        removing it from the application here makes the removal immediate and explicit instead
        of destructor-order dependent.
        """
        filter_obj = self._popup_filter
        if filter_obj is None:
            return
        self._popup_filter = None
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(filter_obj)

    def close_silently(self) -> None:
        """Close without invoking ``actions.quit()`` (used when the engine itself
        is already shutting down and has its own quit sequence in flight)."""
        self._closing_programmatically = True
        self.close()

    def _save_geometry(self) -> None:
        if self._dashboard_mode and self._dashboard_normal_geometry is not None:
            # The persisted window size is the NORMAL one; the board's own width is its setting.
            geo = self._dashboard_normal_geometry
            self._cfg.dashboard.width = max(_MIN_WIDTH, self.width())
        elif self._completion_mode and self._normal_geometry is not None:
            geo = self._normal_geometry
        else:
            geo = self.geometry()
        self._cfg.ui.x = geo.x()
        self._cfg.ui.y = geo.y()
        self._cfg.ui.width = geo.width()
        self._cfg.ui.height = geo.height()
        self._cfg.ui.opacity = self.windowOpacity()
        self._cfg.ui.click_through = self._click_through

    # ---------------------------------------------------------------- bus pump ----
    def _poll(self) -> None:
        for ev in drain(self._q):
            self._handle_event(ev)
        if self._completion_mode and not self.slim_panel.isHidden():
            self.slim_panel.flush()
        self._refresh_monitor()
        self._update_hands_free_click_through()

    def _refresh_monitor(self) -> None:
        """Sample resources and re-read the context snapshot for the Monitor, about once a second
        while it is on screen. A no-op while the Monitor's View
        pref is off, or the whole overlay is hidden (``isVisibleTo`` ignores the window's own hidden
        state, so the window is checked explicitly) -- no sampling runs behind a hidden panel."""
        if not self.isVisible() or not self.monitor.isVisibleTo(self):
            self._monitor_tick = 0
            # Drop the CPU baseline so the first reading after the Monitor is shown again is a
            # fresh interval, not an average over the hidden period.
            self._resource_sampler.reset()
            return
        if self._monitor_tick % _MONITOR_SAMPLE_TICKS == 0:
            try:
                self.monitor.set_resources(self._resource_sampler.sample())
            except Exception:  # noqa: BLE001 - the sampler already fails soft; never break the poll
                log.debug("monitor resource sample failed", exc_info=True)
            try:
                snap = self._actions.context_usage()
            except Exception:  # noqa: BLE001
                log.debug("context_usage() failed", exc_info=True)
                snap = None
            if snap:
                self.monitor.set_context(snap)
        self._monitor_tick += 1

    def _handle_event(self, ev: Event) -> None:
        if isinstance(ev, PartialTranscriptEvent):
            self.live_view.add_partial(ev.source, ev.text, ev.t_start)
        elif isinstance(ev, FinalTranscriptEvent):
            self.live_view.add_final(ev.segment)
        elif isinstance(ev, SpeakerRenamedEvent):
            self.live_view.rename_speaker(ev.speaker_id, ev.new_label)
        elif isinstance(ev, AudioLevelEvent):
            self.status.set_level(ev.source, ev.rms)
        elif isinstance(ev, AssistStartedEvent):
            self._on_assist_started(ev)
        elif isinstance(ev, AssistDeltaEvent):
            kind = self._assist_job_kinds.get(ev.job_id)
            if kind in ("complete", "dashboard"):
                pass   # inline completions: the ghost window; a dashboard refresh: applied to the board on Done
            elif kind == "suggest":
                self.screen_view.append_suggestion(ev.delta)
            elif kind == "insights":
                self.notes_view.append_insights(ev.job_id, ev.delta)
            else:
                self.assist_view.on_delta(ev.job_id, ev.delta)
                self._on_plan_delta(ev, kind)
        elif isinstance(ev, AssistDoneEvent):
            kind = self._assist_job_kinds.pop(ev.job_id, None)
            if kind != "complete":
                # Every real answer (including inline suggestions) carries fresh timing for the
                # Monitor's performance readout; inline completions never report through here.
                self.monitor.on_assist_done(ev.ttft_s, ev.tokens, ev.tok_per_s, ev.cancelled)
            if kind == "complete":
                pass
            elif kind == "dashboard":
                if self.dashboard_view is not None:
                    self.dashboard_view.finish_update(ev.job_id, ev.cancelled and not ev.skipped)
            elif kind == "suggest":
                self.screen_view.finish_suggestion(ev.text, ev.cancelled, _assist_footer(ev))
            elif kind == "insights":
                self.notes_view.finish_insights(ev.job_id, ev.cancelled and not ev.skipped)
            else:
                self.assist_view.on_done(ev.job_id, ev.text, ev.ttft_s, ev.tokens, ev.tok_per_s, ev.cancelled)
                self._on_plan_done(ev, kind)
        elif isinstance(ev, AssistErrorEvent):
            kind = self._assist_job_kinds.pop(ev.job_id, None)
            if kind == "complete":
                self._set_completion_status("error", ev.message)
            elif kind == "dashboard":
                if self.dashboard_view is not None:
                    self.dashboard_view.update_error(ev.job_id, ev.message)
            elif kind == "suggest":
                self.screen_view.show_error(ev.message)
            elif kind == "insights":
                self.notes_view.insights_error(ev.job_id, ev.message)
            else:
                self.assist_view.on_error(ev.job_id, ev.message)
        elif isinstance(ev, CompletionEvent):
            self._on_completion_event(ev)
        elif isinstance(ev, CompletionModeEvent):
            if ev.enabled != self._completion_mode:
                self.completion_btn.setChecked(ev.enabled)
        elif isinstance(ev, ScreenTextEvent):
            self.screen_view.set_screen_text(ev.text, len(ev.lines))
            detail = f"{ev.mode} · {ev.source_name} · {ev.ocr_ms:.0f} ms"
            self.screen_view.set_status(detail, level="active")
            # ScreenStatusEvent only fires on mode/source/pause changes and errors (see
            # screen/reader.py), not on every capture, so the status bar's per-capture
            # latency/source readout (SPEC: "Screen: auto · Zoom Meeting · 140 ms") has to be
            # driven from here too, or it would go stale between foreground-window changes.
            self.status.set_screen_status("active", detail)
        elif isinstance(ev, ScreenStatusEvent):
            self.screen_view.set_status(ev.detail or ev.state, level=ev.state)
            self.status.set_screen_status(ev.state, ev.detail)
        elif isinstance(ev, NotesUpdatedEvent):
            self.notes_view.set_notes(ev.markdown, ev.generated_at)
            if self.dashboard_view is not None:
                self.dashboard_view.set_notes(ev.markdown, ev.generated_at)
        elif isinstance(ev, InsightsUpdatedEvent):
            self._on_insights_updated(ev)
        elif isinstance(ev, DashboardUpdatedEvent):
            self._on_dashboard_updated(ev)
        elif isinstance(ev, DashboardModeEvent):
            if ev.enabled != self._dashboard_mode:
                self._set_dashboard_mode(ev.enabled, notify_engine=False)
        elif isinstance(ev, NotesErrorEvent):
            self.notes_view.set_error(ev.message)
            self.show_toast(f"Notes error: {ev.message}")
        elif isinstance(ev, ComponentStatusEvent):
            self._on_component_status(ev)
        elif isinstance(ev, SessionEvent):
            self.show_toast(self._session_toast_text(ev))
        elif isinstance(ev, AutoAssistChangedEvent):
            self.auto_btn.setChecked(ev.enabled)
        elif isinstance(ev, CheckpointEvent):
            self._on_checkpoint_event(ev)
        elif isinstance(ev, HandsFreeEvent):
            self._on_hands_free_event(ev)
        elif isinstance(ev, ManualAdvanceEvent):
            if ev.enabled != self._manual:
                self._set_manual(ev.enabled, notify_engine=False)
        elif isinstance(ev, ModelLibraryEvent):
            self._on_model_library_event(ev)
        elif isinstance(ev, ContextUsageEvent):
            # The Monitor shows the live context breakdown for every assist job; the per-poll
            # `context_usage()` snapshot fills in the running summary the event does not carry
            # (set_context leaves the summary alone when the key is absent, and normalises the
            # document count so the event's contributing-count and the snapshot agree).
            self.monitor.set_context(dataclasses.asdict(ev))
            self._forward_to_models_dialog(ev)
        elif isinstance(ev, (ModelPullEvent, ModelSearchEvent, ModelInfoEvent, ContextLibraryEvent)):
            # 2026-10-06: the Context tab re-reads the library on ContextLibraryEvent and renders
            # the usage panel from ContextUsageEvent; like the model events, they only matter
            # while the Models window exists.
            self._forward_to_models_dialog(ev)
        else:
            log.debug("overlay: unhandled event type %s", type(ev).__name__)

    def _forward_to_models_dialog(self, ev: object) -> None:
        """Model events (and ``ComponentStatusEvent("llm", ...)``) only matter to the Models
        window; dropped on the floor while it has never been opened."""
        if self._models_dialog is not None:
            self._models_dialog.on_event(ev)

    def _on_model_library_event(self, ev: ModelLibraryEvent) -> None:
        self._forward_to_models_dialog(ev)
        if ev.reason == "empty-library" and not self._models_prompted:
            self._models_prompted = True
            self.open_models("Discover")
            self.show_toast("No model yet -- pull one in Models")

    def _on_checkpoint_event(self, ev: CheckpointEvent) -> None:
        """The engine confirmed the checkpoint: mark it in the history and drop the UI-side
        plan/explanation state (the coding context it belonged to no longer exists)."""
        self.assist_view.mark_checkpoint(time.strftime("%H:%M:%S"))
        self._plan_text = self._explain_text = ""
        self._plan_job = self._explain_job = None
        self.slim_panel.clear()
        if not self.slim_panel.isHidden():
            self.slim_panel.hide()
            self._apply_slim_size()
        self.show_toast("Checkpoint — coding context starts fresh")

    def _on_assist_started(self, ev: AssistStartedEvent) -> None:
        self._assist_job_kinds[ev.job_id] = ev.kind
        if ev.kind == "complete":
            return
        if ev.kind == "dashboard":
            if self.dashboard_view is not None:
                self.dashboard_view.begin_update(ev.job_id, ev.question)   # never switches tabs or views
            return
        if ev.kind == "insights":
            self.notes_view.begin_insights(ev.job_id, ev.question)   # never switches tabs
            return
        if ev.kind == "suggest":
            self.screen_view.begin_suggestion(ev.question)
            if not self._hands_free:
                self._show_tab(3)
        else:
            self.assist_view.on_started(ev.job_id, ev.kind, ev.question)
            self._show_tab(1)
            self._on_plan_started(ev)

    def _on_component_status(self, ev: ComponentStatusEvent) -> None:
        self._component_states[ev.component] = ev.state
        if ev.component == "asr":
            self.status.set_asr_status(ev.state, ev.detail)
        elif ev.component == "llm":
            self.status.set_llm_status(ev.state, ev.detail)
            self.monitor.set_llm_state(ev.state, ev.detail)
            self._forward_to_models_dialog(ev)
        elif ev.component == "completion":
            self._set_completion_status(ev.state, ev.detail)
            if ev.state == "error":
                self.show_toast(f"Completion: {ev.detail or 'error'}")
            return
        elif ev.component in ("project", "dashboard"):
            # A repository / toolkit found or changed mid-session, a dashboard problem: say so, never a dot.
            if ev.component == "project" and ev.detail:
                self._project_detail = ev.detail      # the View menu's Repository entry shows the latest
            self.show_toast(ev.detail or ev.component)
            return
        elif ev.state == "error":
            self.show_toast(f"{ev.component}: {ev.detail or 'error'}")
        self._update_dot()

    def _update_dot(self) -> None:
        watched = (self._component_states.get("asr"), self._component_states.get("llm"))
        if "error" in watched:
            self.status_dot.set_color(COLORS["err"])
        elif all(s == "ready" for s in watched):
            self.status_dot.set_color(COLORS["ok"])
        else:
            self.status_dot.set_color(COLORS["warn"])

    @staticmethod
    def _session_toast_text(ev: SessionEvent) -> str:
        labels = {"new": "New session started", "saved": "Session saved", "exported": "Session exported"}
        text = labels.get(ev.kind, ev.kind)
        return f"{text}: {ev.path}" if ev.path else text

    # ---------------------------------------------------------------- public slots-
    def toggle_visible(self) -> None:
        """Show the overlay if hidden, hide it if shown."""
        self.setVisible(not self.isVisible())

    def focus_ask(self) -> None:
        """Ensure the overlay is visible and give the ask box keyboard focus."""
        if not self.isVisible():
            self.show()
        self.askbox.focus()

    def nudge(self, dx: int, dy: int) -> None:
        """Move the window by ``(dx, dy)`` logical pixels, keeping its title bar on a screen."""
        self.move(_on_screen_pos(self, QPoint(self.x() + dx, self.y() + dy)))

    def set_click_through(self, enabled: bool) -> None:
        """Toggle WS_EX_TRANSPARENT so clicks pass through to the app underneath."""
        changed = enabled != self._click_through
        self._click_through = enabled
        self.clickthrough_btn.blockSignals(True)
        self.clickthrough_btn.setChecked(enabled)
        self.clickthrough_btn.blockSignals(False)
        if _is_windows_platform() and win32 is not None:
            try:
                hwnd = int(self.winId())
                win32.set_click_through(hwnd, enabled)
                win32.set_no_activate(hwnd, enabled)
            except Exception:
                log.exception("failed to toggle click-through")
        if changed:
            self.show_toast(
                "Click-through ON (Ctrl+Alt+C to disable)" if enabled else "Click-through OFF"
            )

    def set_opacity(self, value: float) -> None:
        """Set window opacity (0.4..1.0) and sync the title bar slider."""
        value = max(0.4, min(1.0, value))
        self.setWindowOpacity(value)
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(round(value * 100))
        self.opacity_slider.blockSignals(False)

    def show_toast(self, text: str) -> None:
        """Show ``text`` centred at the top of the panel for ~2.5 s."""
        self.toast.setText(text)
        self.toast.adjustSize()
        x = (self.width() - self.toast.width()) // 2
        self.toast.move(max(0, x), 10)
        self.toast.show()
        self.toast.raise_()
        self._toast_timer.start(_TOAST_MS)
