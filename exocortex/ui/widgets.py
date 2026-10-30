"""The overlay's content widgets: transcript, assist, notes, status bar and ask box.

Every widget here is Qt-only and talks to the engine exclusively through an
``EngineActions`` implementation handed in at construction time; it never imports
``exocortex.app``. Widgets expose small, explicit update methods (``add_final``,
``on_delta``, ``set_notes``, ...) that ``OverlayWindow`` calls when it drains events
off the bus — no widget here subscribes to the bus itself.
"""
from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from exocortex.bus import Source, ToolCallEvent, TranscriptSegment
from exocortex.config import Config
from exocortex.llm.prompts import (
    INSIGHT_SECTIONS,
    insight_sections_in,
    parse_insights,
    render_insights,
    split_choices,
)
from exocortex.ui.markdown_lite import md_to_html
from exocortex.ui.theme import COLORS, speaker_color

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, actions.py has no Qt deps
    from exocortex.actions import EngineActions

log = logging.getLogger(__name__)

_MAX_TRANSCRIPT_ROWS = 400
_JUMP_THRESHOLD_PX = 40
_ASSIST_REFRESH_MS = 60
_MAX_HISTORY = 20
_MAX_SUGGESTION_HISTORY = 10
_HISTORY_MAX_HEIGHT = 200   # Screen tab suggestion history: still a hard cap (see _history_scroll)
_HISTORY_MIN_HEIGHT = 48    # Assist tab history: the QSplitter's floor for the bottom pane
_QWIDGETSIZE_MAX = 16777215   # Qt's QWIDGETSIZE_MAX -- "no maximum" for setMaximumHeight/Width

#: Assist-panel question-label prefix per job kind (default "Q: ").
_QUESTION_PREFIXES: dict[str, str] = {
    "screen": "On screen: ", "solve": "Solving: ", "explain": "", "code": "Code: ", "text": "Text: ",
    "recap": "",
}

#: The Notes tab's Insights pane: its title, and what it shows before the first insights arrive.
_INSIGHTS_TITLE = "Co-pilot insights"
_INSIGHTS_PLACEHOLDER = (
    "*No questions or blind spots yet. They refresh in the background as the session goes on "
    "(Assist menu), or press Questions / Blind spots above.*"
)

#: (mode, button label, QButtonGroup id) for the Screen tab's mode row, in display order.
_SCREEN_MODES: list[tuple[str, str]] = [
    ("off", "Off"),
    ("auto", "Active window"),
    ("monitor", "Monitor"),
    ("region", "Region…"),
]
_SCREEN_MODE_IDS: dict[str, int] = {mode: i for i, (mode, _label) in enumerate(_SCREEN_MODES)}


def _screen_state_color(state: str) -> str:
    """Colour for a screen-reading state: ok=active, warn=paused/off, err=error, dim=unknown."""
    if state == "active":
        return COLORS["ok"]
    if state in ("paused", "off"):
        return COLORS["warn"]
    if state == "error":
        return COLORS["err"]
    return COLORS["text_dim"]


def _history_scroll(container: QWidget, parent: QWidget, *, max_height: int | None = _HISTORY_MAX_HEIGHT) -> QScrollArea:
    """Wrap a collapsible history list in a scroll area, initially hidden.

    The entries are word-wrapped labels; laid out directly in a tab, their combined height
    becomes the tab's - and so the top-level window's - *minimum* height, and Qt grows the
    window to fit (20 long answers measured > 16 000 px, pushing the title bar far off-screen).
    Inside a scroll area they cost at most ``max_height`` - the Screen tab's suggestion history
    keeps the default hard cap (``_HISTORY_MAX_HEIGHT``); the Assist tab's history passes
    ``max_height=None`` since its own ``QSplitter`` (``AssistView.history_splitter``), not a
    fixed height, now bounds how tall it can grow.
    """
    scroll = QScrollArea(parent)
    scroll.setWidgetResizable(True)
    if max_height is not None:
        scroll.setMaximumHeight(max_height)
    scroll.setWidget(container)
    scroll.hide()
    return scroll


def fmt_mmss(seconds: float) -> str:
    """Format a non-negative second count as ``mm:ss`` (clamped at 0)."""
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


# ------------------------------------------------------------------ toggle switch ----
_SWITCH_W = 36
_SWITCH_H = 18


class ToggleSwitch(QAbstractButton):
    """A pill-shaped on/off switch (the hands-free toggle at the right of the tab strip).

    A checkable ``QAbstractButton`` painted by hand: an accent-coloured track with the knob on
    the right when checked, a dim track with the knob on the left when not, everything faded
    while disabled. Uses the theme's colours only, so it reads on the glass panel; ``toggled``
    is the signal to connect to, exactly like a check box.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(_SWITCH_W, _SWITCH_H)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return self.sizeHint()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        track_h = min(h, _SWITCH_H)
        track = QRectF(0, (h - track_h) / 2, w, track_h)
        on = self.isChecked()
        track_color = QColor(COLORS["accent"]) if on else QColor(255, 255, 255, 40)
        knob_color = QColor(COLORS["bg_solid"]) if on else QColor(COLORS["text_dim"])
        if not self.isEnabled():
            track_color.setAlpha(60 if on else 25)
            knob_color.setAlpha(120)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(track, track_h / 2, track_h / 2)
        margin = 2.0
        d = track_h - 2 * margin
        x = track.right() - margin - d if on else track.left() + margin
        painter.setBrush(knob_color)
        painter.drawEllipse(QRectF(x, track.top() + margin, d, d))
        painter.end()


def level_to_percent(rms: float) -> int:
    """Map a linear 0..1 RMS level to a 0..100 meter value with a log-ish curve."""
    rms = max(0.0, min(1.0, rms))
    return max(0, min(100, round(100.0 * math.log10(1.0 + 9.0 * rms))))


# ------------------------------------------------------------------ speaker chip ----
class _SpeakerChip(QLabel):
    """A bold, coloured speaker-name label; double-click requests a rename."""

    doubleClicked = Signal(str)  # speaker_id

    def __init__(self, speaker_id: str, label: str, parent: QWidget | None = None) -> None:
        super().__init__(label, parent)
        self.speaker_id = speaker_id
        self.setStyleSheet(f"color: {speaker_color(speaker_id)}; font-weight: 600;")
        self.setToolTip("Double-click to rename")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        self.doubleClicked.emit(self.speaker_id)
        super().mouseDoubleClickEvent(event)


# ------------------------------------------------------------------ transcript row --
class _SegmentRow(QFrame):
    """One transcript line: timestamp, speaker chip, text, optional Answer button."""

    def __init__(
        self,
        segment: TranscriptSegment,
        actions: "EngineActions",
        *,
        partial: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.speaker_id = segment.speaker_id
        self._actions = actions
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(6)

        time_label = QLabel(fmt_mmss(segment.t_start), self)
        time_label.setStyleSheet(f"color: {COLORS['text_faint']};")
        time_label.setFixedWidth(38)
        layout.addWidget(time_label)

        self.chip = _SpeakerChip(segment.speaker_id, segment.speaker_label, self)
        self.chip.doubleClicked.connect(self._on_rename_requested)
        layout.addWidget(self.chip)

        self.text_label = QLabel(segment.text, self)
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if partial:
            self.text_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-style: italic;")
        layout.addWidget(self.text_label, 1)

        if segment.is_question and not partial:
            self.setStyleSheet(f"background: {COLORS['question']}; border-radius: 6px;")
            answer_btn = QPushButton("Answer", self)
            answer_btn.setObjectName("icon")
            answer_btn.clicked.connect(self._actions.answer_last)
            layout.addWidget(answer_btn)

    def _on_rename_requested(self, speaker_id: str) -> None:
        current = self.chip.text()
        text, ok = QInputDialog.getText(self, "Rename speaker", "Name:", text=current)
        if ok and text.strip():
            self._actions.rename_speaker(speaker_id, text.strip())


class TranscriptView(QScrollArea):
    """Scrolling list of transcript rows for the Live tab.

    Finals accumulate top-to-bottom (capped at ``_MAX_TRANSCRIPT_ROWS``, oldest
    dropped first); each source's in-progress partial is shown as a trailing
    italic row that is replaced on every update and removed on the final.
    """

    def __init__(self, actions: "EngineActions", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._actions = actions
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QWidget(self)
        self._vbox = QVBoxLayout(self._container)
        self._vbox.setContentsMargins(2, 2, 2, 2)
        self._vbox.setSpacing(2)
        self._vbox.addStretch(1)
        self.setWidget(self._container)

        self._final_rows: list[_SegmentRow] = []
        self._partial_rows: dict[Source, _SegmentRow] = {}
        self._known_speakers: dict[str, str] = {}
        self._user_scrolled_up = False

        self.jump_pill = QPushButton("↓ jump to latest", self)
        self.jump_pill.setObjectName("icon")
        self.jump_pill.hide()
        self.jump_pill.clicked.connect(self._scroll_to_bottom)

        bar = self.verticalScrollBar()
        bar.valueChanged.connect(self._on_scrolled)

    # -- helpers ------------------------------------------------------------------
    def _insert_row(self, row: QWidget) -> None:
        self._vbox.insertWidget(self._vbox.count() - 1, row)

    def _on_scrolled(self, _value: int) -> None:
        bar = self.verticalScrollBar()
        distance = bar.maximum() - bar.value()
        was_up = self._user_scrolled_up
        self._user_scrolled_up = distance > _JUMP_THRESHOLD_PX
        if self._user_scrolled_up != was_up:
            self.jump_pill.setVisible(self._user_scrolled_up)

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
        self._user_scrolled_up = False
        self.jump_pill.hide()

    def _maybe_autoscroll(self) -> None:
        if not self._user_scrolled_up:
            QTimer.singleShot(0, self._scroll_to_bottom)

    # -- public API -----------------------------------------------------------------
    def add_partial(self, source: Source, text: str, t_start: float) -> None:
        """Show/replace the trailing in-progress row for ``source``."""
        old = self._partial_rows.pop(source, None)
        if old is not None:
            self._vbox.removeWidget(old)
            old.deleteLater()
        if not text:
            self._maybe_autoscroll()
            return
        label = "Me" if source == Source.MIC else self._known_speakers.get("spk1", "Speaker")
        seg = TranscriptSegment(
            id=-1,
            source=source,
            speaker_id="me" if source == Source.MIC else "spk1",
            speaker_label=label,
            text=text,
            t_start=t_start,
            t_end=t_start,
            confidence=1.0,
        )
        row = _SegmentRow(seg, self._actions, partial=True, parent=self._container)
        self._insert_row(row)
        self._partial_rows[source] = row
        self._maybe_autoscroll()

    def add_final(self, segment: TranscriptSegment) -> None:
        """Append a finished segment, removing that source's partial row."""
        old = self._partial_rows.pop(segment.source, None)
        if old is not None:
            self._vbox.removeWidget(old)
            old.deleteLater()

        self._known_speakers[segment.speaker_id] = segment.speaker_label
        row = _SegmentRow(segment, self._actions, parent=self._container)
        self._insert_row(row)
        self._final_rows.append(row)

        while len(self._final_rows) > _MAX_TRANSCRIPT_ROWS:
            oldest = self._final_rows.pop(0)
            self._vbox.removeWidget(oldest)
            oldest.deleteLater()

        self._maybe_autoscroll()

    def rename_speaker(self, speaker_id: str, new_label: str) -> None:
        """Relabel every row (final and partial) currently showing ``speaker_id``."""
        self._known_speakers[speaker_id] = new_label
        for row in (*self._final_rows, *self._partial_rows.values()):
            if row.speaker_id == speaker_id:
                row.chip.setText(new_label)

    def clear(self) -> None:
        """Remove every final and partial row (used by the Reset control)."""
        for row in (*self._final_rows, *self._partial_rows.values()):
            self._vbox.removeWidget(row)
            row.deleteLater()
        self._final_rows.clear()
        self._partial_rows.clear()
        self._known_speakers.clear()
        self._scroll_to_bottom()

    def row_count(self) -> int:
        """Number of final rows currently displayed (for tests/diagnostics)."""
        return len(self._final_rows)

    def known_speakers(self) -> dict[str, str]:
        """Copy of every ``speaker_id -> label`` seen so far (for the rename menu)."""
        return dict(self._known_speakers)


# ------------------------------------------------------------------ assist view -----
class AssistView(QWidget):
    """The Assist tab: current question, streaming answer, footer stats, resizable history.

    A vertical ``QSplitter`` (``history_splitter``) holds two non-collapsible panes: the top one
    (question label, answer browser, choice buttons, footer, the Stop / Answer-last row) and the
    bottom one (``history_toggle`` + ``history_scroll``). Collapsed, the bottom pane is pinned to
    the toggle's own height and the handle is disabled (nothing to drag); expanded, the handle is
    enabled and the pane is restored to the last height the user dragged it to (remembered in
    ``_history_height``, starting at ``_HISTORY_MAX_HEIGHT``). The history list itself no longer
    has a hard maximum height -- the splitter is what bounds it now -- so a long history never
    inflates this tab's (and so the overlay window's) minimum size; see ``_history_scroll``.
    """

    def __init__(self, actions: "EngineActions", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._actions = actions
        self._job_id: int | None = None
        self._kind = ""
        self._buffer = ""
        self._dirty = False
        # (question, answer); a checkpoint divider is stored as (None, "— checkpoint ... —").
        self._history: list[tuple[str | None, str]] = []
        self._view_index = 0   # 0 = newest/live answer; >0 = reverted N answers back
        self._history_height = _HISTORY_MAX_HEIGHT   # remembered across collapse/expand
        self._hands_free_view = False   # output-only presentation (see set_hands_free_view)
        self._streaming = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.history_splitter = QSplitter(Qt.Orientation.Vertical, self)
        layout.addWidget(self.history_splitter, 1)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        self.question_label = QLabel("Q: –", top)
        self.question_label.setObjectName("subtitle")
        self.question_label.setWordWrap(True)
        self.question_label.setTextFormat(Qt.TextFormat.PlainText)   # the question is the user's / the transcript's text
        top_layout.addWidget(self.question_label)

        self.answer_browser = QTextBrowser(top)
        self.answer_browser.setOpenExternalLinks(False)
        top_layout.addWidget(self.answer_browser, 1)

        # Choice buttons: shown only when an autonomous "solve" answer forks and the model asks
        # the user to pick a direction. Clicking one re-runs the solve with that focus, so the
        # user never has to type.
        self.choices_container = QWidget(top)
        self.choices_layout = QVBoxLayout(self.choices_container)
        self.choices_layout.setContentsMargins(0, 0, 0, 0)
        self.choices_layout.setSpacing(4)
        self.choices_container.hide()
        top_layout.addWidget(self.choices_container)

        self.footer_label = QLabel("", top)
        self.footer_label.setObjectName("subtitle")
        # Plain text: on_tool puts text the MODEL chose (a search query, a command, a url, an error
        # reason) in here, and an AutoText label renders "<b>", "<div>" or "&lt;" as HTML -- the
        # footer would stop showing what was actually asked.
        self.footer_label.setTextFormat(Qt.TextFormat.PlainText)
        # A tool call's line (on_tool) can run to ~150 characters; clipped, it must never push the
        # panel's minimum width up (an explicit minimum overrides the label-wide minimumSizeHint).
        self.footer_label.setMinimumWidth(1)
        top_layout.addWidget(self.footer_label)

        btn_row = QHBoxLayout()
        self.stop_btn = QPushButton("Stop", top)
        self.stop_btn.clicked.connect(self._actions.cancel_assist)
        self.stop_btn.hide()
        self.answer_last_btn = QPushButton("Answer last (Ctrl+Alt+A)", top)
        self.answer_last_btn.setObjectName("primary")
        # Natural width, but free to shrink (clipping its label) rather than raise the window's
        # 340 px floor. It used to be `Ignored`: next to the row's stretch that laid it out at
        # zero width, so the button was never visible on the Assist tab.
        self.answer_last_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.answer_last_btn.setMinimumWidth(1)   # an explicit minimum overrides the label-wide minimumSizeHint
        self.answer_last_btn.clicked.connect(self._actions.answer_last)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.answer_last_btn)
        btn_row.addStretch(1)
        top_layout.addLayout(btn_row)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        self.history_toggle = QToolButton(bottom)
        self.history_toggle.setText("History")
        self.history_toggle.setCheckable(True)
        self.history_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.history_toggle.toggled.connect(self._on_history_toggled)
        bottom_layout.addWidget(self.history_toggle)

        self.history_container = QWidget(bottom)
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setContentsMargins(8, 0, 0, 0)
        # No fixed cap here (unlike the Screen tab's suggestion history): the splitter above is
        # what bounds this pane now, so it can be dragged taller; it keeps a small minimum so it
        # never disappears under the handle.
        self.history_scroll = _history_scroll(self.history_container, bottom, max_height=None)
        self.history_scroll.setMinimumHeight(_HISTORY_MIN_HEIGHT)
        bottom_layout.addWidget(self.history_scroll)

        self.history_splitter.addWidget(top)
        self.history_splitter.addWidget(bottom)
        self.history_splitter.setCollapsible(0, False)
        self.history_splitter.setCollapsible(1, False)
        self.history_splitter.setStretchFactor(0, 1)
        self.history_splitter.setStretchFactor(1, 0)
        self.history_splitter.splitterMoved.connect(self._on_splitter_moved)
        self._on_history_toggled(False)   # establish the collapsed state (no toggled signal yet)

        self._timer = QTimer(self)
        self._timer.setInterval(_ASSIST_REFRESH_MS)
        self._timer.timeout.connect(self._flush)
        self._timer.start()

    def shutdown(self) -> None:
        """Stop the periodic redraw timer (call when the owning window is closing).

        Without this the timer keeps firing every ``_ASSIST_REFRESH_MS`` for the rest
        of the process even after the overlay window is closed, since Qt does not
        destroy a top-level widget on ``close()``.
        """
        self._timer.stop()

    def _on_history_toggled(self, checked: bool) -> None:
        self.history_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self.history_scroll.setVisible(checked)
        bottom = self.history_splitter.widget(1)
        self.history_splitter.handle(1).setEnabled(checked)
        # Always resolve the actual pixel sizes here (never just the maximumHeight constraint):
        # a constraint change alone only takes effect on the splitter's next resize event, which
        # can leave the old (wrong) sizes on screen until something else happens to resize it.
        total = self.history_splitter.height() or sum(self.history_splitter.sizes())
        if checked:
            bottom.setMaximumHeight(_QWIDGETSIZE_MAX)
            bottom_h = min(self._history_height, max(_HISTORY_MIN_HEIGHT, total)) if total else self._history_height
        else:
            # Collapsed: pin the bottom pane to exactly the toggle's own height (nothing else is
            # visible there) so dragging is pointless and the handle is disabled to match.
            bottom_h = self.history_toggle.sizeHint().height()
            bottom.setMaximumHeight(bottom_h)
        self.history_splitter.setSizes([max(0, total - bottom_h), bottom_h])

    def _on_splitter_moved(self, pos: int, index: int) -> None:
        """Remember the dragged height so the next expand restores it (only while actually
        expanded -- a drag can't happen while collapsed, the handle is disabled)."""
        if not self.history_toggle.isChecked():
            return
        sizes = self.history_splitter.sizes()
        if len(sizes) == 2 and sizes[1] > 0:
            self._history_height = sizes[1]

    def _flush(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        self.answer_browser.setHtml(md_to_html(self._buffer))
        bar = self.answer_browser.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _clear_history_widgets(self) -> None:
        while self.history_layout.count():
            item = self.history_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _render_history(self) -> None:
        """Rebuild the history list's widgets from ``self._history`` (answers and dividers)."""
        self._clear_history_widgets()
        for q, a in self._history:
            entry = QLabel(self.history_container)
            entry.setWordWrap(True)
            if q is None:   # a checkpoint divider
                entry.setTextFormat(Qt.TextFormat.PlainText)
                entry.setText(a)
                entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
                entry.setStyleSheet(f"color: {COLORS['text_faint']};")
            else:
                entry.setTextFormat(Qt.TextFormat.RichText)
                entry.setText(f"<b>Q:</b> {md_to_html(q)}<br><b>A:</b> {md_to_html(a)}")
            self.history_layout.addWidget(entry)

    def _push_history(self, question: str, answer: str) -> None:
        self._history.insert(0, (question, answer))
        del self._history[_MAX_HISTORY:]
        self._render_history()

    def mark_checkpoint(self, when: str) -> None:
        """Insert a divider ("— checkpoint HH:MM:SS —", dim, centred) into the history so the
        log shows where the coding context was cut. No-op when the history is empty or its
        newest entry already is a divider (called from ``OverlayWindow._on_checkpoint_event``,
        driven by the bus's ``CheckpointEvent`` -- never directly by a click handler).

        Also clears the choice buttons (they belonged to the old task, now dropped) and sets the
        footer accordingly; the current answer text is left alone -- the history entry already
        has it, and it stays true (it was produced before the cut).
        """
        if not self._history or self._history[0][0] is None:
            return
        self._history.insert(0, (None, f"— checkpoint {when} —"))
        del self._history[_MAX_HISTORY:]
        # Every existing entry just shifted down by one slot; `_view_index` is a position in
        # `_history`, not an identity, so it must shift with them to keep pointing at the same
        # entry (the one currently on screen) -- otherwise the very next `revert()` would step
        # onto the divider's old neighbour, i.e. the answer already displayed, instead of a
        # genuinely earlier one (every other `_history[0]`-inserting path -- `_push_history`'s
        # callers, `clear()` -- resets `_view_index` to 0 for the same reason: it must never be
        # left stale after `_history`'s indices move).
        self._view_index += 1
        self._render_history()
        self._clear_choices()
        self.footer_label.setText("checkpoint — the next Code / solve starts fresh")

    # -- hands-free view -------------------------------------------------------------
    def set_hands_free_view(self, on: bool) -> None:
        """Output-only presentation for hands-free mode: only the answer text stays, plus the
        history list when it was expanded. The question label, footer, choice buttons, the
        Stop / Answer-last row and the history toggle are hidden (and stay hidden while ``on``,
        whatever the streaming state) and the splitter handle is disabled; ``False`` restores
        everything, including a Stop button for an answer still streaming."""
        self._hands_free_view = on
        for w in (self.question_label, self.footer_label, self.answer_last_btn, self.history_toggle):
            w.setVisible(not on)
        self.stop_btn.setVisible(not on and self._streaming)
        if on:
            self.choices_container.hide()
        elif self.choices_layout.count():
            self.choices_container.show()
        expanded = self.history_toggle.isChecked()
        self.history_scroll.setVisible(expanded)
        self.history_splitter.handle(1).setEnabled(expanded and not on)

    @property
    def hands_free_view(self) -> bool:
        return self._hands_free_view

    def history_entries(self) -> list[QWidget]:
        """The history list's answer entries, newest first (checkpoint dividers left out)."""
        out: list[QWidget] = []
        for i in range(self.history_layout.count()):
            w = self.history_layout.itemAt(i).widget()
            if w is None or i >= len(self._history):
                continue
            if self._history[i][0] is None:      # a divider
                continue
            out.append(w)
        return out

    def history_snap(self, direction: int) -> bool:
        """Scroll the (expanded) history list so that the next entry down (``direction > 0``) or
        the previous entry up (``< 0``) from the current scroll position sits at the top of the
        list, and mark it; False when there is no such entry (the list is hidden, empty, or
        already at that end). The Up / Down arrow taps of the hands-free view."""
        if not self.history_scroll.isVisible():
            return False
        entries = self.history_entries()
        if not entries:
            return False
        bar = self.history_scroll.verticalScrollBar()
        current = bar.value()
        tops = [(w.y(), w) for w in entries]
        if direction > 0:
            ahead = [(y, w) for y, w in tops if y > current + 2]
            target = min(ahead, key=lambda t: t[0]) if ahead else None
        else:
            behind = [(y, w) for y, w in tops if y < current - 2]
            target = max(behind, key=lambda t: t[0]) if behind else None
        if target is None:
            return False
        y, widget = target
        if direction > 0 and y > bar.maximum() and current >= bar.maximum():
            return False
        bar.setValue(min(y, bar.maximum()))
        self._mark_history_entry(widget)
        return True

    def _mark_history_entry(self, widget: QWidget | None) -> None:
        """A thin accent bar on the history entry the arrow keys last snapped to (none: clear)."""
        accent = COLORS.get("accent", "#4c8bf5")
        for w in self.history_entries():
            w.setStyleSheet(f"border-left: 2px solid {accent}; padding-left: 6px;" if w is widget else "")

    # -- public API, driven by AssistStarted/Delta/Done/Error events ----------------
    def on_started(self, job_id: int, kind: str, question: str) -> None:
        self._job_id = job_id
        self._kind = kind
        self._buffer = ""
        self._dirty = True
        self._view_index = 0
        self._streaming = True
        self._clear_choices()
        prefix = _QUESTION_PREFIXES.get(kind, "Q: ")
        self.question_label.setText(f"{prefix}{question}")
        self.footer_label.setText(f"{kind} - streaming...")
        if not self._hands_free_view:
            self.stop_btn.show()
        self._flush()

    def on_delta(self, job_id: int, delta: str) -> None:
        if job_id != self._job_id:
            return
        self._buffer += delta
        self._dirty = True

    def on_tool(self, ev: ToolCallEvent) -> None:
        """A tool call of the current answer (2026-10-07, capabilities): the footer reads
        ``<kind> - <tool> <what it was asked>...`` while the call runs, then
        ``<kind> - <tool> <what> (<size hint or the reason>)`` once it is done / failed / refused,
        until ``on_done`` (or ``on_error``) replaces it. A call of any other job -- or one that
        arrives after this answer has finished -- is ignored."""
        if ev.job_id != self._job_id or not self._streaming:
            return
        head = f"{self._kind} - {ev.name}" + (f" {ev.summary}" if ev.summary else "")
        if ev.state == "running":
            self.footer_label.setText(f"{head}…")
        else:
            self.footer_label.setText(f"{head} ({ev.detail})" if ev.detail else head)

    def on_done(
        self,
        job_id: int,
        text: str,
        ttft_s: float,
        tokens: int,
        tok_per_s: float,
        cancelled: bool = False,
    ) -> None:
        if job_id != self._job_id:
            return
        self._streaming = False
        body, choices = split_choices(text)
        self._buffer = body if (choices and body) else (body or text)
        self._dirty = True
        self._flush()
        self.stop_btn.hide()
        self._show_choices(choices if not cancelled else [])
        stats = "cancelled" if cancelled else f"ttft {ttft_s:.2f} s - {tok_per_s:.0f} tok/s"
        if choices and not cancelled:
            stats = "choose an option" if not body else stats
        self.footer_label.setText(f"{self._kind} - {stats}" if self._kind else stats)
        question = self.question_label.text()
        for p in (*_QUESTION_PREFIXES.values(), "Q: "):
            question = question.removeprefix(p)
        if body:
            self._push_history(question, body)
            self._view_index = 0

    def _clear_choices(self) -> None:
        while self.choices_layout.count():
            item = self.choices_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.choices_container.hide()

    def _show_choices(self, choices: list[str]) -> None:
        """Render up to a handful of click-to-choose option buttons under the answer (kept
        hidden in the hands-free view, which shows no interactive element)."""
        self._clear_choices()
        if not choices:
            return
        for label in choices[:4]:
            btn = QPushButton(label, self.choices_container)
            btn.setObjectName("primary")
            btn.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _checked=False, opt=label: self._actions.solve(opt))
            self.choices_layout.addWidget(btn)
        if not self._hands_free_view:
            self.choices_container.show()

    def revert(self) -> None:
        """Show the previous answer (step one further back through history each call, skipping
        over checkpoint dividers -- there is no answer on one to revert to)."""
        idx = self._view_index
        while True:
            idx += 1
            if idx >= len(self._history):
                self.footer_label.setText("no earlier answer")
                return
            if self._history[idx][0] is not None:
                break
        self._view_index = idx
        question, answer = self._history[idx]
        self._clear_choices()
        self.question_label.setText(f"Q: {question}")
        self._buffer = answer
        self._dirty = True
        self._flush()
        self.footer_label.setText(f"reverted ({self._view_index} back)")

    def clear(self) -> None:
        """Reset the panel to empty (used by the Reset control)."""
        self._job_id = None
        self._buffer = ""
        self._history.clear()
        self._view_index = 0
        self._clear_choices()
        self.question_label.setText("Q: –")
        self.footer_label.setText("")
        self.answer_browser.setHtml("")
        self._clear_history_widgets()

    def on_error(self, job_id: int, message: str) -> None:
        if job_id != self._job_id:
            return
        self._streaming = False
        self.stop_btn.hide()
        self.answer_browser.setHtml(
            f'<p style="color:{COLORS["err"]}">{md_to_html(message)}</p>'
        )
        self.footer_label.setText("error")


# ------------------------------------------------------------------ screen view -----
class ScreenView(QWidget):
    """The Screen tab: mode/read/suggest controls, latest OCR text and a suggestions panel.

    Reads and writes ``cfg.screen.use_in_answers`` / ``cfg.screen.answer_screen_questions``
    directly (there is no ``EngineActions`` method for either); every other control goes
    through ``actions`` (``suggest_now``, ``set_auto_suggest``; the Read button that forces a
    capture sits in the overlay's bottom control row).
    Renders correctly at the overlay's 340 px floor width: every row wraps or stacks
    instead of forcing the window wider, and nothing scrolls horizontally.
    """

    mode_requested = Signal(str)   # "off" | "auto" | "monitor" (not "region" - see region_requested)
    region_requested = Signal()    # "Region..." button / select_region hotkey: open the picker

    def __init__(self, cfg: Config, actions: "EngineActions", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._actions = actions
        self._paused = False
        self._suggestion_buffer = ""
        self._suggestion_history: list[tuple[str, str]] = []

        # This tab packs more controls than the overlay's 340 px floor comfortably fits at the
        # overlay's normal font size; a ScreenView-local style sheet takes precedence over the
        # OverlayWindow-level one (a widget's own style sheet wins over an inherited one for the
        # same property), so this alone - not setFont(), which the ancestor sheet would still
        # override - shrinks every button/label in this tab without touching the rest of the
        # overlay. (The toggle checkboxes get their own word-wrapping label - see _toggle_row -
        # since QCheckBox's own text never wraps regardless of font size.)
        self.setStyleSheet("QPushButton, QLabel#subtitle, QLabel#status { font-size: 9pt; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        layout.addWidget(self._build_mode_row())
        layout.addLayout(self._build_action_col())
        layout.addLayout(self._build_toggle_col())

        self.status_label = QLabel("", self)
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # The raw OCR readout is hidden by default (it is rarely worth looking at); a collapsible
        # toggle reveals it. `_lines_count` is remembered so the toggle's label stays accurate.
        self._lines_count = 0
        self.screen_text_toggle = QToolButton(self)
        self.screen_text_toggle.setCheckable(True)
        self.screen_text_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.screen_text_toggle.toggled.connect(self._on_screen_text_toggled)
        layout.addWidget(self.screen_text_toggle)

        self.screen_text_header = QLabel("", self)
        self.screen_text_header.setObjectName("subtitle")
        layout.addWidget(self.screen_text_header)

        self.screen_text_edit = QPlainTextEdit(self)
        self.screen_text_edit.setReadOnly(True)
        self.screen_text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.screen_text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.screen_text_edit.setFixedHeight(110)
        self.screen_text_edit.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-family: Consolas, 'Cascadia Mono', 'Courier New', monospace;"
        )
        layout.addWidget(self.screen_text_edit)
        self.screen_text_toggle.setChecked(bool(self._cfg.screen.show_text))
        self._on_screen_text_toggled(self.screen_text_toggle.isChecked())

        self.suggestions_header = QLabel("Suggestions", self)
        self.suggestions_header.setObjectName("subtitle")
        layout.addWidget(self.suggestions_header)

        self.suggestion_browser = QTextBrowser(self)
        self.suggestion_browser.setOpenExternalLinks(False)
        layout.addWidget(self.suggestion_browser, 1)

        self.suggestion_footer = QLabel("", self)
        self.suggestion_footer.setObjectName("subtitle")
        layout.addWidget(self.suggestion_footer)

        self.suggestion_history_toggle = QToolButton(self)
        self.suggestion_history_toggle.setText("Suggestion history")
        self.suggestion_history_toggle.setCheckable(True)
        self.suggestion_history_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.suggestion_history_toggle.toggled.connect(self._on_history_toggled)
        layout.addWidget(self.suggestion_history_toggle)

        self.suggestion_history_container = QWidget(self)
        self.suggestion_history_layout = QVBoxLayout(self.suggestion_history_container)
        self.suggestion_history_layout.setContentsMargins(8, 0, 0, 0)
        self.suggestion_history_scroll = _history_scroll(self.suggestion_history_container, self)
        layout.addWidget(self.suggestion_history_scroll)

    # -- building --------------------------------------------------------------------
    def _build_mode_row(self) -> QWidget:
        """2x2 grid of exclusive mode buttons - stays inside 340 px where a single row wouldn't."""
        container = QWidget(self)
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        self._mode_group = QButtonGroup(container)
        self._mode_group.setExclusive(True)
        self._mode_buttons: dict[str, QPushButton] = {}
        for i, (mode, label) in enumerate(_SCREEN_MODES):
            btn = QPushButton(label, container)
            btn.setObjectName("tab")
            btn.setCheckable(True)
            self._mode_group.addButton(btn, i)
            grid.addWidget(btn, i // 2, i % 2)
            self._mode_buttons[mode] = btn
        self._mode_group.idClicked.connect(self._on_mode_clicked)
        return container

    def _build_action_col(self) -> QVBoxLayout:
        # "Read" (force a capture + OCR now) lives in the overlay's always-visible bottom control
        # row (OverlayWindow._build_control_row), not here.
        col = QVBoxLayout()
        self.suggest_now_btn = QPushButton("Suggest now (Ctrl+Alt+S)", self)
        self.suggest_now_btn.clicked.connect(lambda: self._actions.suggest_now())
        col.addWidget(self.suggest_now_btn)
        return col

    def _build_toggle_col(self) -> QVBoxLayout:
        col = QVBoxLayout()
        self.use_in_answers_cb = QCheckBox(self)
        self.use_in_answers_cb.setChecked(self._cfg.screen.use_in_answers)
        self.use_in_answers_cb.toggled.connect(self._on_use_in_answers_toggled)
        self.auto_suggest_cb = QCheckBox(self)
        self.auto_suggest_cb.setChecked(self._cfg.screen.auto_suggest)
        self.auto_suggest_cb.toggled.connect(self._on_auto_suggest_toggled)
        self.answer_screen_questions_cb = QCheckBox(self)
        self.answer_screen_questions_cb.setChecked(self._cfg.screen.answer_screen_questions)
        self.answer_screen_questions_cb.toggled.connect(self._on_answer_screen_questions_toggled)
        col.addWidget(self._toggle_row(self.use_in_answers_cb, "Use in answers"))
        col.addWidget(self._toggle_row(self.auto_suggest_cb, "Auto-suggest"))
        col.addWidget(self._toggle_row(self.answer_screen_questions_cb, "Answer on-screen questions"))
        return col

    def sync_toggles(self) -> None:
        """Re-read the three screen toggles from the config after it was changed elsewhere (the
        Assist menu, the settings dialog), without re-running their handlers."""
        for checkbox, value in (
            (self.use_in_answers_cb, self._cfg.screen.use_in_answers),
            (self.auto_suggest_cb, self._cfg.screen.auto_suggest),
            (self.answer_screen_questions_cb, self._cfg.screen.answer_screen_questions),
        ):
            checkbox.blockSignals(True)
            checkbox.setChecked(bool(value))
            checkbox.blockSignals(False)

    def _toggle_row(self, checkbox: QCheckBox, text: str) -> QWidget:
        """A checkbox paired with a word-wrapping label (QCheckBox's own text never wraps,
        and "Answer on-screen questions" does not fit one line at the 340 px floor)."""
        row = QWidget(self)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        label = QLabel(text, row)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {COLORS['text_dim']};")
        h.addWidget(checkbox, 0)
        h.addWidget(label, 1)
        return row

    # -- internal handlers -------------------------------------------------------------
    def _on_mode_clicked(self, button_id: int) -> None:
        mode = next((m for m, i in _SCREEN_MODE_IDS.items() if i == button_id), None)
        if mode is None:
            return
        if mode == "region":
            self.region_requested.emit()
            return
        self.mode_requested.emit(mode)

    def _on_use_in_answers_toggled(self, checked: bool) -> None:
        self._cfg.screen.use_in_answers = checked

    def _on_auto_suggest_toggled(self, checked: bool) -> None:
        self._actions.set_auto_suggest(checked)

    def _on_answer_screen_questions_toggled(self, checked: bool) -> None:
        self._cfg.screen.answer_screen_questions = checked

    def _on_history_toggled(self, checked: bool) -> None:
        self.suggestion_history_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self.suggestion_history_scroll.setVisible(checked)

    def _push_suggestion_history(self, text: str) -> None:
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), text.strip())
        self._suggestion_history.insert(0, (time.strftime("%H:%M:%S"), first_line))
        del self._suggestion_history[_MAX_SUGGESTION_HISTORY:]
        while self.suggestion_history_layout.count():
            item = self.suggestion_history_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for ts, line in self._suggestion_history:
            entry = QLabel(f"{ts} — {line}", self.suggestion_history_container)
            entry.setObjectName("subtitle")
            entry.setWordWrap(True)
            self.suggestion_history_layout.addWidget(entry)

    # -- public API ---------------------------------------------------------------------
    def set_status(self, text: str, level: str = "") -> None:
        """Update the status line; ``level`` (active/paused/off/error) selects its colour."""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {_screen_state_color(level)};")

    def _on_screen_text_toggled(self, checked: bool) -> None:
        self.screen_text_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self.screen_text_toggle.setText(f"Screen text ({self._lines_count} lines)")
        self.screen_text_header.setVisible(False)  # header folded into the toggle label
        self.screen_text_edit.setVisible(checked)
        self._cfg.screen.show_text = checked

    def set_screen_text(self, text: str, lines_count: int) -> None:
        """Store the latest OCR text; only render it while the readout is expanded."""
        self._lines_count = lines_count
        self.screen_text_toggle.setText(f"Screen text ({lines_count} lines)")
        if self.screen_text_toggle.isChecked():
            self.screen_text_edit.setPlainText(text)

    def set_mode(self, mode: str) -> None:
        """Check the button for ``mode`` without emitting ``mode_requested``/``region_requested``."""
        btn = self._mode_buttons.get(mode)
        if btn is None:
            return
        self._mode_group.blockSignals(True)
        btn.setChecked(True)
        self._mode_group.blockSignals(False)

    def set_paused(self, paused: bool) -> None:
        """Reflect a pause/resume (e.g. from the toggle_screen hotkey) ahead of the next event.

        Covers both directions: without this, resuming left the status line showing the
        stale "paused" text in the warn colour until the next ScreenTextEvent/ScreenStatusEvent
        arrived - which, in mode "off" (the reader idles and never captures), never happens,
        so the tab would claim "paused" forever after being resumed.
        """
        self._paused = paused
        if paused:
            self.set_status("paused", level="paused")
        else:
            self.set_status("resumed", level="")

    def begin_suggestion(self, question: str) -> None:
        """Start a new streaming suggestion (mirrors ``AssistView.on_started``)."""
        self._suggestion_buffer = ""
        self.suggestion_browser.setHtml("")
        self.suggestion_footer.setText(f"{question} - streaming..." if question else "streaming...")

    def append_suggestion(self, delta: str) -> None:
        """Append streamed text to the current suggestion."""
        self._suggestion_buffer += delta
        self.suggestion_browser.setHtml(md_to_html(self._suggestion_buffer))
        bar = self.suggestion_browser.verticalScrollBar()
        bar.setValue(bar.maximum())

    def finish_suggestion(self, text: str, cancelled: bool, footer: str = "") -> None:
        """Finalise the current suggestion and, unless cancelled/empty, push it to history."""
        self._suggestion_buffer = text
        self.suggestion_browser.setHtml(md_to_html(text))
        self.suggestion_footer.setText(footer)
        if text and not cancelled:
            self._push_suggestion_history(text)

    def show_error(self, text: str) -> None:
        """Show a suggestion-job error in the suggestions panel."""
        self.suggestion_browser.setHtml(f'<p style="color:{COLORS["err"]}">{md_to_html(text)}</p>')
        self.suggestion_footer.setText("error")


# ------------------------------------------------------------------ notes view ------
class NotesView(QWidget):
    """The Notes tab: the co-pilot's Insights pane -- questions to ask, blind spots & follow-ups,
    refreshed in the background and on demand -- above the running notes, in a resizable vertical
    splitter (``splitter``); a Questions / Blind spots row over the insights, Regenerate / Recap
    and the notes timestamp under the notes.

    The insights are a living document: :meth:`set_insights` (driven by ``InsightsUpdatedEvent``)
    replaces what is shown. While an insights job streams (:meth:`begin_insights` /
    :meth:`append_insights` / :meth:`finish_insights`, driven by the Assist events of kind
    "insights"), a section whose heading has already streamed shows the new text and the others
    stay as they were. Nothing here switches tabs or touches the Assist tab: a background pass
    must never pull the driver's eyes off their code.
    """

    def __init__(self, actions: "EngineActions", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._actions = actions
        self._insights: dict[str, str] = {}
        self._insight_job: int | None = None
        self._insight_buffer = ""
        self._insight_dirty = False
        self._insights_updated_at: float | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.splitter = QSplitter(Qt.Orientation.Vertical, self)
        layout.addWidget(self.splitter, 1)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(4)
        self.insights_label = QLabel(_INSIGHTS_TITLE, top)
        self.insights_label.setObjectName("subtitle")
        # Free to clip rather than raise the overlay's 340 px floor ("... updating ..." is long).
        self.insights_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        head.addWidget(self.insights_label, 1)
        self.questions_btn = QPushButton("Questions", top)
        # Natural width, free to shrink (clipping its label) rather than raise the overlay's
        # 340 px floor -- the same treatment as AssistView.answer_last_btn.
        self.questions_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.questions_btn.setMinimumWidth(1)
        self.questions_btn.setToolTip("Suggest questions worth asking right now")
        self.questions_btn.clicked.connect(lambda: self._actions.insights_now(("questions",)))
        head.addWidget(self.questions_btn)
        self.blindspots_btn = QPushButton("Blind spots", top)
        self.blindspots_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.blindspots_btn.setMinimumWidth(1)
        self.blindspots_btn.setToolTip(
            "Flag blind spots and follow-ups: unhandled edge cases, missing tests, risky "
            "assumptions, things to follow up on"
        )
        self.blindspots_btn.clicked.connect(lambda: self._actions.insights_now(("blindspots",)))
        head.addWidget(self.blindspots_btn)
        top_layout.addLayout(head)
        self.insights_browser = QTextBrowser(top)
        self.insights_browser.setOpenExternalLinks(False)
        top_layout.addWidget(self.insights_browser, 1)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)
        self.browser = QTextBrowser(bottom)
        bottom_layout.addWidget(self.browser, 1)

        self.splitter.addWidget(top)
        self.splitter.addWidget(bottom)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 3)

        footer = QHBoxLayout()
        self.regenerate_btn = QPushButton("Regenerate (Ctrl+Alt+N)", self)
        self.regenerate_btn.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.regenerate_btn.clicked.connect(self._actions.generate_notes)
        footer.addWidget(self.regenerate_btn)
        self.recap_btn = QPushButton("Recap", self)
        self.recap_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.recap_btn.setMinimumWidth(1)
        self.recap_btn.setToolTip("Recap the session so far -- goal, done, now, next -- in the Assist tab")
        self.recap_btn.clicked.connect(lambda: self._actions.recap())
        footer.addWidget(self.recap_btn)
        self.updated_label = QLabel("", self)
        self.updated_label.setObjectName("subtitle")
        footer.addWidget(self.updated_label)
        footer.addStretch(1)
        layout.addLayout(footer)

        self._timer = QTimer(self)
        self._timer.setInterval(_ASSIST_REFRESH_MS)
        self._timer.timeout.connect(self._flush_insights)
        self._timer.start()
        self._render_insights()

    def shutdown(self) -> None:
        """Stop the streaming-preview timer (the owning window is closing; see AssistView.shutdown)."""
        self._timer.stop()

    # -- running notes -----------------------------------------------------------------
    def set_notes(self, markdown: str, generated_at: float) -> None:
        self.browser.setHtml(md_to_html(markdown))
        self.updated_label.setText(f"last updated {fmt_mmss(generated_at)}")

    def clear(self) -> None:
        """Empty the notes, the insights and both timestamps (used by the Reset control)."""
        self.browser.setHtml("")
        self.updated_label.setText("")
        self._insights = {}
        self._insight_job = None
        self._insight_buffer = ""
        self._insight_dirty = False
        self._insights_updated_at = None
        self.insights_label.setText(_INSIGHTS_TITLE)
        self._render_insights()

    def set_error(self, message: str) -> None:
        self.updated_label.setText(f"error: {message}")

    # -- co-pilot insights ---------------------------------------------------------------
    @property
    def insights(self) -> dict[str, str]:
        """What the pane holds: section -> markdown bullets (the last ``set_insights``)."""
        return dict(self._insights)

    @property
    def streaming_insights(self) -> bool:
        return self._insight_job is not None

    def begin_insights(self, job_id: int, label: str = "") -> None:
        self._insight_job = job_id
        self._insight_buffer = ""
        self._insight_dirty = False
        self.insights_label.setText(f"{_INSIGHTS_TITLE} - updating {(label or 'insights').lower()}...")
        self._render_insights()

    def append_insights(self, job_id: int, delta: str) -> None:
        if job_id != self._insight_job:
            return
        self._insight_buffer += delta
        self._insight_dirty = True

    def finish_insights(self, job_id: int, cancelled: bool = False) -> None:
        if job_id != self._insight_job:
            return
        self._insight_job = None
        self._insight_buffer = ""
        self._insight_dirty = False
        self._render_insights()
        self._set_insights_status("update cancelled" if cancelled else "")

    def insights_error(self, job_id: int, message: str) -> None:
        if job_id != self._insight_job:
            return
        self._insight_job = None
        self._insight_buffer = ""
        self._insight_dirty = False
        self._render_insights()
        self._set_insights_status(f"error: {message}")

    def set_insights(self, questions: str, blindspots: str, generated_at: float, auto: bool = False) -> None:
        """Replace the insights shown (an ``InsightsUpdatedEvent``). While a job streams, the new
        document is kept and shown once that job finishes."""
        self._insights = {"questions": questions or "", "blindspots": blindspots or ""}
        self._insights_updated_at = generated_at
        if self._insight_job is None:
            self._render_insights()
            self._set_insights_status("")

    def _set_insights_status(self, extra: str) -> None:
        parts = [_INSIGHTS_TITLE]
        if self._insights_updated_at is not None:
            parts.append(f"updated {fmt_mmss(self._insights_updated_at)}")
        if extra:
            parts.append(extra)
        self.insights_label.setText(" - ".join(parts))

    def _flush_insights(self) -> None:
        if not self._insight_dirty:
            return
        self._insight_dirty = False
        self._render_insights()

    def _render_insights(self) -> None:
        sections = dict(self._insights)
        if self._insight_job is not None and self._insight_buffer.strip():
            streamed = insight_sections_in(self._insight_buffer)
            if streamed:
                parsed = parse_insights(self._insight_buffer, INSIGHT_SECTIONS)
                for section in streamed:
                    sections[section] = parsed.get(section, "")
        markdown = render_insights(sections)
        if not markdown:
            markdown = "*Thinking...*" if self._insight_job is not None else _INSIGHTS_PLACEHOLDER
        self.insights_browser.setHtml(md_to_html(markdown))


# ------------------------------------------------------------------ status bar ------
class _BadgeLabel(QLabel):
    """Single-line status label that elides instead of forcing the window wider.

    Its minimum width is tiny, so the status bar can always shrink to the overlay's 340 px
    floor; when squeezed, the text is elided with an ellipsis (the tooltip keeps the detail).
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._full = text
        self.setWordWrap(False)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override
        self._full = text
        self._relayout()

    def text(self) -> str:  # noqa: D401 - Qt override returning the full (un-elided) text
        return self._full

    def minimumSizeHint(self):  # noqa: N802,ANN201 - Qt override
        from PySide6.QtCore import QSize

        return QSize(16, super().minimumSizeHint().height())

    def sizeHint(self):  # noqa: N802,ANN201 - Qt override
        # Based on the FULL text, not the currently elided one (otherwise the layout shrinks the
        # label a little more on every relayout until only an ellipsis is left).
        from PySide6.QtCore import QSize

        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 6, super().sizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802,ANN001 - Qt override
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        fm = self.fontMetrics()
        avail = max(self.width() - 2, 16)
        super().setText(fm.elidedText(self._full, Qt.TextElideMode.ElideRight, avail) if fm.horizontalAdvance(self._full) > avail else self._full)


class StatusBar(QFrame):
    """Bottom status strip: audio levels, ASR/LLM state, screen state, auto badge."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusbar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self.mic_level = QProgressBar(self)
        self.mic_level.setRange(0, 100)
        self.mic_level.setTextVisible(False)
        self.mic_level.setFixedWidth(28)
        self.mic_level.setToolTip("Mic level")

        self.system_level = QProgressBar(self)
        self.system_level.setRange(0, 100)
        self.system_level.setTextVisible(False)
        self.system_level.setFixedWidth(28)
        self.system_level.setToolTip("System audio level")

        # Status labels are single-line badges; the full detail lives in their tooltips so the
        # bar never wraps into a tall block (verified on the real window).
        self.asr_label = _BadgeLabel("ASR –", self)
        self.asr_label.setObjectName("status")
        self.asr_label.setWordWrap(False)
        self.llm_label = _BadgeLabel("LLM –", self)
        self.llm_label.setObjectName("status")
        self.llm_label.setWordWrap(False)
        self.screen_label = _BadgeLabel("", self)
        self.screen_label.setObjectName("status")
        self.screen_label.setWordWrap(False)
        # Hidden until the first ScreenTextEvent/ScreenStatusEvent (like auto_badge below):
        # an always-visible-but-empty label still reserves its width + inter-item spacing in
        # the layout, which alone pushes the overlay's computed minimum width a few pixels past
        # its documented 340 px floor (_MIN_WIDTH in overlay.py) even though the label is idle.
        self.screen_label.hide()
        self.auto_badge = QLabel("auto", self)
        self.auto_badge.setObjectName("status")
        self.auto_badge.setStyleSheet(f"color: {COLORS['accent']}; font-weight: 600;")
        self.auto_badge.hide()

        for w in (
            self.mic_level,
            self.system_level,
            self.asr_label,
            self.llm_label,
            self.screen_label,
        ):
            layout.addWidget(w)
        layout.addStretch(1)
        layout.addWidget(self.auto_badge)

    def set_level(self, source: Source, rms: float) -> None:
        pct = level_to_percent(rms)
        (self.mic_level if source == Source.MIC else self.system_level).setValue(pct)

    # Badges are plain ASCII on purpose: symbol/emoji glyphs come from fallback fonts whose
    # metrics inflate the labels' size hints and squeeze the whole bar. Colour carries the state.
    @staticmethod
    def _badge(prefix: str, state: str) -> str:
        suffix = {"loading": "…", "busy": "…", "error": " !"}.get(state, "")
        return f"{prefix}{suffix}"

    def set_asr_status(self, state: str, detail: str = "") -> None:
        self.asr_label.setText(self._badge("ASR", state))
        self.asr_label.setToolTip(f"Speech recognition: {state}" + (f" - {detail}" if detail else ""))
        self.asr_label.setStyleSheet(f"color: {_state_color(state)};")

    def set_llm_status(self, state: str, detail: str = "") -> None:
        short = _shorten(detail, 16) if state == "ready" and detail else ""
        self.llm_label.setText(self._badge("LLM", state) + (f" {short}" if short else ""))
        self.llm_label.setToolTip(f"Language model: {state}" + (f" - {detail}" if detail else ""))
        self.llm_label.setStyleSheet(f"color: {_state_color(state)};")

    def set_screen_status(self, state: str, detail: str = "") -> None:
        """Update the compact "Screen ..." badge; reveals it on first use (see __init__).

        active -> "Screen 140 ms" (timing taken from the detail when present); paused ->
        "Screen paused"; off -> "Screen off"; error -> "Screen !". The full detail (mode,
        window title, timing or error text) is in the tooltip; colour shows the state.
        """
        self.screen_label.show()
        if state == "active":
            ms = ""
            if detail.rstrip(")").endswith("ms"):
                ms = detail.rstrip(")").rsplit("(", 1)[-1].strip() if "(" in detail else detail.rsplit("·", 1)[-1].strip()
            text = "Screen" + (f" {ms}" if ms else "")
        elif state == "paused":
            text = "Screen paused"
        elif state == "off":
            text = "Screen off"
        else:
            text = "Screen !"
        self.screen_label.setText(text)
        self.screen_label.setToolTip(f"Screen reading: {state}" + (f" - {detail}" if detail else ""))
        self.screen_label.setStyleSheet(f"color: {_screen_state_color(state)};")

    def set_auto_badge(self, enabled: bool) -> None:
        self.auto_badge.setVisible(enabled)


def _state_color(state: str) -> str:
    if state == "ready":
        return COLORS["ok"]
    if state == "error":
        return COLORS["err"]
    return COLORS["warn"]


def _shorten(text: str, n: int) -> str:
    """Elide `text` to at most `n` characters with an ellipsis (for status badges)."""
    text = " ".join(text.split())
    return text if len(text) <= n else text[: max(1, n - 1)] + "…"


# ------------------------------------------------------------------ monitor ---------
#: Colour per context-window source, in the order they stack in the segmented bar (oldest /
#: most-stable first). The names match ContextUsageEvent's ``*_tokens`` fields.
_CONTEXT_SEGMENTS: list[tuple[str, str, str]] = [
    # (snapshot key, short label, colour)
    ("system_tokens", "system", "#a5b4fc"),       # indigo-300
    ("earlier_tokens", "earlier", "#c4b5fd"),      # violet-300
    ("reference_tokens", "docs", "#6ee7b7"),       # emerald-300
    ("lookup_tokens", "lookup", "#99f6e4"),        # teal-200
    ("screen_tokens", "screen", "#fdba74"),        # orange-300
    ("transcript_tokens", "transcript", None),     # None -> the accent colour (the live speech)
]


class _Dot(QLabel):
    """A small filled circle used as a readiness indicator (a local twin of the overlay's
    ``_DotIndicator`` so this module keeps no cross-import)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self.set_color(COLORS["warn"])

    def set_color(self, color: str) -> None:
        self.setStyleSheet(f"background: {color}; border-radius: 5px;")


class _SegBar(QWidget):
    """A thin, multi-colour horizontal bar: the context window filled by source, against the
    server's per-request slot. Each segment is proportional to its token count; the remainder is
    the free budget (faint). Painting works under the offscreen test platform."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._segments: list[tuple[str, int, str]] = []    # (label, tokens, colour)
        self._total = 0                                    # slot tokens (the full width)
        self.setFixedHeight(12)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_segments(self, segments: list[tuple[str, int, str]], total: int) -> None:
        self._segments = [(lbl, max(0, int(tok)), col) for lbl, tok, col in segments]
        self._total = max(0, int(total))
        parts = ", ".join(f"{lbl} {tok}" for lbl, tok, _c in self._segments if tok > 0)
        used = sum(tok for _l, tok, _c in self._segments)
        self.setToolTip(
            (f"Context window: {used} / {self._total or '?'} tokens" + (f"  ({parts})" if parts else ""))
        )
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()
        radius = h / 2
        # Track (the free budget).
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 22))
        painter.drawRoundedRect(QRectF(0, 0, w, h), radius, radius)
        # Width the segments are scaled against: the slot if known, else the sum of the segments.
        used = sum(tok for _l, tok, _c in self._segments)
        scale_total = self._total if self._total > 0 else used
        if scale_total <= 0 or w <= 0:
            painter.end()
            return
        painter.setClipRect(0, 0, w, h)
        x = 0.0
        for _label, tokens, colour in self._segments:
            if tokens <= 0:
                continue
            seg_w = w * (tokens / scale_total)
            if seg_w <= 0:
                continue
            painter.setBrush(QColor(colour))
            painter.drawRoundedRect(QRectF(x, 0, seg_w + radius, h), radius, radius)
            x += seg_w
        painter.end()


class MonitorView(QFrame):
    """The "View -> Monitor" section (2026-10-07): a compact, always-glanceable readout of the
    local model's **performance** (last answer's time-to-first-token and tokens/second), the
    **context window and its contents** (how the per-request budget is filled, by source, plus
    the running summary of the earlier meeting and the document count), and **resource
    utilisation** (CPU, system RAM, the dedicated GPU's use and VRAM, the integrated GPU, and the
    NPU).

    Which of those appear is driven by ``[monitor]`` (the Settings "Monitor" tab); the overlay
    calls :meth:`apply_config` when it changes. The hardware-utilisation rows (GPU / iGPU / NPU)
    additionally hide themselves when the machine reports no such device.

    Driven entirely by ``OverlayWindow`` -- it feeds performance from ``AssistDoneEvent``, the
    model state from ``ComponentStatusEvent("llm")``, the context from ``ContextUsageEvent`` /
    ``EngineActions.context_usage()``, and resources from ``exocortex.ui.resources`` on a slow
    timer. It never subscribes to the bus itself (like every widget here).
    """

    def __init__(self, cfg: Config, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self.setObjectName("monitor")
        self.setStyleSheet(
            "QFrame#monitor { border: 1px solid %s; border-radius: 10px; }"
            " QLabel { font-size: 8pt; }"
            " QLabel#monhead { color: %s; font-weight: 600; letter-spacing: 0.4px; }"
            " QLabel#monval { color: %s; }"
            % (COLORS["border"], COLORS["text_dim"], COLORS["text"])
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(5)

        # ---- performance (its own box so the whole section can be hidden) ------------
        self._perf_box = QWidget(self)
        perf_row = QHBoxLayout(self._perf_box)
        perf_row.setContentsMargins(0, 0, 0, 0)
        perf_row.setSpacing(6)
        self._perf_dot = _Dot(self._perf_box)
        head = QLabel("LLM", self._perf_box)
        head.setObjectName("monhead")
        self._model_label = _BadgeLabel("", self._perf_box)
        self._model_label.setObjectName("monval")
        self._perf_label = QLabel("—", self._perf_box)
        self._perf_label.setObjectName("monval")
        self._perf_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        perf_row.addWidget(self._perf_dot)
        perf_row.addWidget(head)
        perf_row.addWidget(self._model_label, 1)
        perf_row.addWidget(self._perf_label)
        outer.addWidget(self._perf_box)

        # ---- context window + contents ----------------------------------------------
        self._ctx_box = QWidget(self)
        ctx_v = QVBoxLayout(self._ctx_box)
        ctx_v.setContentsMargins(0, 0, 0, 0)
        ctx_v.setSpacing(4)
        ctx_head = QHBoxLayout()
        ctx_head.setSpacing(6)
        ch = QLabel("Context", self._ctx_box)
        ch.setObjectName("monhead")
        self._ctx_label = QLabel("—", self._ctx_box)
        self._ctx_label.setObjectName("monval")
        self._ctx_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        ctx_head.addWidget(ch)
        ctx_head.addStretch(1)
        ctx_head.addWidget(self._ctx_label)
        ctx_v.addLayout(ctx_head)
        self._seg_bar = _SegBar(self._ctx_box)
        ctx_v.addWidget(self._seg_bar)
        self._legend_label = _BadgeLabel("", self._ctx_box)
        self._legend_label.setObjectName("monval")
        ctx_v.addWidget(self._legend_label)
        # The running summary of the transcript that fell out of the window ("contents").
        self._summary = QPlainTextEdit(self._ctx_box)
        self._summary.setReadOnly(True)
        self._summary.setFrameShape(QFrame.Shape.NoFrame)
        self._summary.setMaximumHeight(48)
        self._summary.setPlaceholderText("no earlier summary yet")
        self._summary.setStyleSheet("font-size: 8pt;")
        ctx_v.addWidget(self._summary)
        outer.addWidget(self._ctx_box)

        # ---- resource utilisation ---------------------------------------------------
        self._res_box = QWidget(self)
        res_v = QVBoxLayout(self._res_box)
        res_v.setContentsMargins(0, 0, 0, 0)
        res_v.setSpacing(3)
        self._res_head = QLabel("Resources", self._res_box)
        self._res_head.setObjectName("monhead")
        res_v.addWidget(self._res_head)
        # One bar per device, keyed so apply_config()/set_resources() can show or hide each.
        self._bars: dict[str, QProgressBar] = {}
        for key, label in (("cpu", "CPU"), ("ram", "RAM"), ("gpu", "GPU"),
                           ("igpu", "iGPU"), ("npu", "NPU")):
            bar = QProgressBar(self._res_box)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFixedHeight(14)
            bar.setFormat(f"{label} —")
            res_v.addWidget(bar)
            self._bars[key] = bar
        outer.addWidget(self._res_box)

        self._summary_text = ""
        #: Whether each hardware device has ever reported a reading; a device that never does
        #: (absent on this machine) keeps its row hidden even when its [monitor] toggle is on.
        self._available: dict[str, bool] = {"cpu": True, "ram": True, "gpu": False,
                                            "igpu": False, "npu": False}
        self.apply_config()

    # ---- configuration ----------------------------------------------------------------
    def apply_config(self) -> None:
        """Show or hide each section / row per ``[monitor]`` (the Settings "Monitor" tab). Called
        at build time and whenever the settings change."""
        m = self._cfg.monitor
        self._perf_box.setVisible(bool(m.show_performance))
        self._ctx_box.setVisible(bool(m.show_context))
        wanted = {"cpu": m.show_cpu, "ram": m.show_ram, "gpu": m.show_gpu,
                  "igpu": m.show_igpu, "npu": m.show_npu}
        any_res = False
        for key, bar in self._bars.items():
            # Core rows (cpu/ram) show whenever enabled; optional hardware rows also need the
            # device to exist on this machine (self._available), so an absent iGPU/NPU never
            # leaves a dead "—" row behind.
            show = bool(wanted[key]) and (self._available[key] or key in ("cpu", "ram"))
            bar.setVisible(show)
            any_res = any_res or show
        self._res_head.setVisible(any_res)
        self._res_box.setVisible(any_res)

    # ---- performance ------------------------------------------------------------------
    def set_llm_state(self, state: str, detail: str = "") -> None:
        self._perf_dot.set_color(_state_color(state))
        if state == "ready" and detail:
            self._model_label.setText(_shorten(detail, 40))
        elif state in ("loading", "busy"):
            self._model_label.setText(_shorten(detail, 40) if detail else state + "…")
        elif state == "error":
            self._model_label.setText("error" + (f": {_shorten(detail, 30)}" if detail else ""))

    def set_model(self, name: str) -> None:
        if name:
            self._model_label.setText(_shorten(name, 40))

    def on_assist_done(self, ttft_s: float, tokens: int, tok_per_s: float, cancelled: bool) -> None:
        if cancelled:
            return
        self._perf_label.setText(f"ttft {ttft_s:.2f}s · {tok_per_s:.0f} tok/s · {tokens} tok")
        self._perf_label.setToolTip(
            f"Last answer: {ttft_s:.2f}s to the first token, {tok_per_s:.0f} tokens/second, "
            f"{tokens} tokens generated"
        )

    # ---- context ----------------------------------------------------------------------
    def set_context(self, snap: dict) -> None:
        """Render ``EngineActions.context_usage()`` (or a ContextUsageEvent turned into a dict)."""
        if not snap:
            return
        slot = int(snap.get("slot_tokens", 0) or 0)
        segments: list[tuple[str, int, str]] = []
        used = 0
        legend_parts: list[str] = []
        for key, label, colour in _CONTEXT_SEGMENTS:
            tok = int(snap.get(key, 0) or 0)
            used += tok
            segments.append((label, tok, colour or COLORS["accent"]))
            if tok > 0:
                legend_parts.append(f"{label} {tok}")
        self._seg_bar.set_segments(segments, slot)
        pct = f" ({100 * used / slot:.0f}%)" if slot > 0 else ""
        prompt_tokens = snap.get("prompt_tokens")
        server = f" · server {int(prompt_tokens)}" if prompt_tokens else ""
        trimmed = " · trimmed" if snap.get("trimmed") else ""
        # Both feeds are normalised to the CONTRIBUTING count: a ContextUsageEvent carries it as
        # `documents`, the poll snapshot as `documents_used` (its `documents` is the ENABLED count).
        docs = int(snap.get("documents_used", snap.get("documents", 0)) or 0)
        doc_txt = f" · {docs} doc{'s' if docs != 1 else ''} used" if docs else ""
        self._ctx_label.setText(f"{used} / {slot or '?'}{pct}{server}")
        self._ctx_label.setToolTip(
            "Context window used / per-request slot (the app's token estimate; 'server' is the "
            "model's own prompt-token count for the last request)"
        )
        self._legend_label.setText("  ".join(legend_parts) + trimmed + doc_txt)
        # Only the full poll snapshot carries the running summary; a raw ContextUsageEvent does
        # not, so leave the existing text alone when the key is absent (never blank it).
        if "earlier_summary" in snap:
            self._summary_text = str(snap.get("earlier_summary", "") or "")
            self._render_summary()

    def _render_summary(self) -> None:
        """Paint the earlier-meeting summary."""
        self._summary.setPlaceholderText("no earlier summary yet")
        if self._summary.toPlainText() != self._summary_text:
            self._summary.setPlainText(self._summary_text)

    # ---- resources --------------------------------------------------------------------
    @staticmethod
    def _set_pct(bar: QProgressBar, label: str, pct: float | None, suffix: str = "", tip: str = "") -> None:
        if pct is None:
            bar.setValue(0)
            bar.setFormat(f"{label} —{suffix}")
        else:
            bar.setValue(int(round(max(0.0, min(100.0, pct)))))
            bar.setFormat(f"{label} {pct:.0f}%{suffix}")
        if tip:
            bar.setToolTip(tip)

    def set_resources(self, sample: "ResourceSample") -> None:
        """Update every resource bar from one :class:`~exocortex.ui.resources.ResourceSample`. A
        hardware device that reports a value (even 0.0) is marked available, so its row appears;
        one that is always ``None`` (absent) keeps its row hidden."""
        cpu = getattr(sample, "cpu_percent", None)
        self._set_pct(self._bars["cpu"], "CPU", cpu, tip="Whole-machine CPU utilisation")

        ram_used = getattr(sample, "ram_used_mb", None)
        ram_total = getattr(sample, "ram_total_mb", None)
        if ram_used is not None and ram_total:
            pct = 100 * ram_used / ram_total
            self._bars["ram"].setValue(int(round(pct)))
            self._bars["ram"].setFormat(f"RAM {ram_used / 1024:.1f} / {ram_total / 1024:.1f} GB")
            self._bars["ram"].setToolTip("System memory in use / total")
        else:
            self._bars["ram"].setValue(0)
            self._bars["ram"].setFormat("RAM —")

        # Dedicated GPU: utilisation plus VRAM in use, on one bar (bar value tracks VRAM).
        gpu = getattr(sample, "gpu_percent", None)
        vused = getattr(sample, "vram_used_mb", None)
        vtotal = getattr(sample, "vram_total_mb", None)
        source = getattr(sample, "vram_source", "none")
        gpu_bar = self._bars["gpu"]
        util_txt = f"{gpu:.0f}%" if gpu is not None else "—"
        if vused is not None and vtotal:
            gpu_bar.setValue(int(round(100 * vused / vtotal)))
            gpu_bar.setFormat(f"GPU {util_txt} · {vused / 1024:.1f}/{vtotal / 1024:.1f} GB")
            gpu_bar.setToolTip(f"Dedicated GPU utilisation; VRAM in use / total (source: {source})")
        else:
            gpu_bar.setValue(int(round(gpu)) if gpu is not None else 0)
            gpu_bar.setFormat(f"GPU {util_txt}")
            gpu_bar.setToolTip("Dedicated GPU utilisation")
        gpu_available = gpu is not None or (vused is not None and bool(vtotal))

        igpu = getattr(sample, "igpu_percent", None)
        self._set_pct(self._bars["igpu"], "iGPU", igpu, tip="Integrated GPU utilisation")
        npu = getattr(sample, "npu_percent", None)
        self._set_pct(self._bars["npu"], "NPU", npu, tip="NPU / AI-accelerator utilisation")

        # Learn which optional devices exist, then re-apply the show/hide rules.
        avail = dict(self._available)
        avail["gpu"] = avail["gpu"] or gpu_available
        avail["igpu"] = avail["igpu"] or (igpu is not None)
        avail["npu"] = avail["npu"] or (npu is not None)
        if avail != self._available:
            self._available = avail
            self.apply_config()

    def clear(self) -> None:
        """Forget the per-session readouts (a Reset / New session): the context and performance
        belong to the session; the resource meters keep sampling."""
        self._perf_label.setText("—")
        self._ctx_label.setText("—")
        self._legend_label.setText("")
        self._seg_bar.set_segments([], 0)
        self._summary_text = ""
        self._render_summary()


# ------------------------------------------------------------------ ask box ---------
class _AskLineEdit(QLineEdit):
    """QLineEdit that clears itself (instead of propagating) on Escape."""

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override
        if event.key() == Qt.Key.Key_Escape:
            self.clear()
            event.accept()
            return
        super().keyPressEvent(event)


class AskBox(QWidget):
    """Bottom input row: free-form question box + Send button."""

    def __init__(self, actions: "EngineActions", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._actions = actions
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.edit = _AskLineEdit(self)
        self.edit.setPlaceholderText("Ask anything about the meeting... (Ctrl+Alt+Q)")
        self.edit.returnPressed.connect(self._send)
        layout.addWidget(self.edit, 1)

        self.send_btn = QPushButton("Send", self)
        self.send_btn.setObjectName("primary")
        self.send_btn.clicked.connect(self._send)
        layout.addWidget(self.send_btn)

    def _send(self) -> None:
        text = self.edit.text().strip()
        if not text:
            return
        self._actions.ask(text)
        self.edit.clear()

    def focus(self) -> None:
        self.edit.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.edit.selectAll()
