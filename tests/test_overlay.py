"""Offline tests for exocortex.ui.markdown_lite / widgets / overlay.

Everything here runs under QT_QPA_PLATFORM=offscreen (set before PySide6 is imported,
so it takes effect for the one QApplication this whole file shares): no real audio
devices, no LM Studio, no real desktop windows. Win32 calls made by OverlayWindow are
expected to be skipped gracefully on this platform (``QGuiApplication.platformName()
!= "windows"``) and must never raise.

A short @pytest.mark.live test at the bottom shows a real window and checks
SetWindowDisplayAffinity actually took effect; it is skipped by default.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from typing import Any

import pytest

from PySide6.QtCore import QEvent, QPoint, QRect, Qt
from PySide6.QtGui import QTextDocument
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from exocortex.bus import (
    AssistDeltaEvent,
    AssistDoneEvent,
    AssistErrorEvent,
    AssistStartedEvent,
    AudioLevelEvent,
    AutoAssistChangedEvent,
    ComponentStatusEvent,
    EventBus,
    FinalTranscriptEvent,
    NotesErrorEvent,
    NotesUpdatedEvent,
    PartialTranscriptEvent,
    SessionEvent,
    Source,
    SpeakerRenamedEvent,
    TranscriptSegment,
)
from exocortex.config import Config
from exocortex.ui import overlay as overlay_mod
from exocortex.ui import widgets as widgets_mod
from exocortex.ui.markdown_lite import md_to_html


def _blocks(doc: "QTextDocument") -> list:
    """Every QTextBlock of `doc`, in order."""
    out = []
    block = doc.begin()
    while block.isValid():
        out.append(block)
        block = block.next()
    return out
from exocortex.ui.overlay import OverlayWindow
from exocortex.ui.widgets import AskBox, StatusBar, TranscriptView, fmt_mmss, level_to_percent

_WAIT_MS = 80


def wait_until(predicate, timeout_ms: int = 5000, step_ms: int = 20) -> bool:
    """Poll ``predicate`` while pumping the Qt event loop, for slower/loaded CI machines."""
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return True
        QTest.qWait(step_ms)
        elapsed += step_ms
    return predicate()


# --------------------------------------------------------------------- fixtures ----
@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication(["exocortex-tests"])
    return app


class FakeActions:
    """Records every call made through the EngineActions protocol."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def answer_last(self) -> None:
        self.calls.append(("answer_last", ()))

    def ask(self, text: str) -> None:
        self.calls.append(("ask", (text,)))

    def generate_notes(self) -> None:
        self.calls.append(("generate_notes", ()))

    def cancel_assist(self) -> None:
        self.calls.append(("cancel_assist", ()))

    def set_auto(self, enabled: bool) -> None:
        self.calls.append(("set_auto", (enabled,)))

    def set_mic_muted(self, muted: bool) -> None:
        self.calls.append(("set_mic_muted", (muted,)))

    def rename_speaker(self, speaker_id: str, label: str) -> None:
        self.calls.append(("rename_speaker", (speaker_id, label)))

    def export_session(self) -> Path:
        self.calls.append(("export_session", ()))
        return Path(".")

    def new_session(self) -> None:
        self.calls.append(("new_session", ()))

    def set_screen_mode(self, mode: str) -> None:
        self.calls.append(("set_screen_mode", (mode,)))

    def set_screen_region(self, x: int, y: int, w: int, h: int) -> None:
        self.calls.append(("set_screen_region", (x, y, w, h)))

    def capture_screen_now(self) -> None:
        self.calls.append(("capture_screen_now", ()))

    def suggest_now(self) -> None:
        self.calls.append(("suggest_now", ()))

    def solve(self, focus: str = "") -> None:
        self.calls.append(("solve", (focus,)))

    def checkpoint(self) -> None:
        self.calls.append(("checkpoint", ()))

    def code(self, focus: str = "") -> None:
        self.calls.append(("code", (focus,)))

    def set_auto_suggest(self, enabled: bool) -> None:
        self.calls.append(("set_auto_suggest", (enabled,)))

    def toggle_screen_paused(self) -> bool:
        self.calls.append(("toggle_screen_paused", ()))
        return False

    def set_paused(self, paused: bool) -> None:
        self.calls.append(("set_paused", (paused,)))

    def set_completion_mode(self, enabled: bool) -> None:
        self.calls.append(("set_completion_mode", (enabled,)))

    def set_completion_context(self, sources: dict, hint: str | None = None) -> None:
        self.calls.append(("set_completion_context", (dict(sources), hint)))

    def set_model(self, model_id: str) -> None:
        self.calls.append(("set_model", (model_id,)))

    def list_models(self) -> list[str]:
        self.calls.append(("list_models", ()))
        return ["maple-preview"]

    def quit(self) -> None:
        self.calls.append(("quit", ()))


@pytest.fixture()
def actions() -> FakeActions:
    return FakeActions()


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def window(qapp: QApplication, bus: EventBus, actions: FakeActions):
    cfg = Config()
    win = OverlayWindow(cfg, bus, actions)
    win.show()
    qapp.processEvents()
    QTest.qWait(30)
    yield win
    win.close_silently()
    qapp.processEvents()


def make_segment(**kw: Any) -> TranscriptSegment:
    defaults = dict(
        id=1,
        source=Source.SYSTEM,
        speaker_id="spk1",
        speaker_label="Speaker 1",
        text="hello there",
        t_start=1.0,
        t_end=2.0,
        confidence=0.9,
        is_question=False,
    )
    defaults.update(kw)
    return TranscriptSegment(**defaults)


# =====================================================================================
# markdown_lite
# =====================================================================================
class TestMarkdownLite:
    def test_empty_string(self) -> None:
        assert md_to_html("") == ""

    def test_plain_paragraph(self) -> None:
        out = md_to_html("just some text")
        assert out == "<p>just some text</p>"

    def test_two_paragraphs(self) -> None:
        out = md_to_html("first\n\nsecond")
        assert "<p>first</p>" in out
        assert "<p>second</p>" in out

    def test_line_break_within_paragraph(self) -> None:
        out = md_to_html("line one\nline two")
        assert "<p>line one<br>\nline two</p>" == out

    def test_bold(self) -> None:
        assert "<strong>bold</strong>" in md_to_html("this is **bold** text")

    def test_italic(self) -> None:
        assert "<em>italic</em>" in md_to_html("this is *italic* text")

    def test_inline_code(self) -> None:
        assert "<code>x = 1</code>" in md_to_html("run `x = 1` now")

    def test_fenced_code_block(self) -> None:
        out = md_to_html("Here:\n```python\ndef f(x):\n    return x * 2\n```\nDone.")
        assert "<pre" in out
        assert "def f(x):" in out
        assert "    return x * 2" in out   # indentation preserved
        assert "```" not in out            # fences consumed
        assert "<p>Here:</p>" in out and "<p>Done.</p>" in out

    def test_fenced_code_escapes_html(self) -> None:
        out = md_to_html("```\n<script>x && y</script>\n```")
        assert "<pre" in out
        assert "&lt;script&gt;" in out and "&amp;&amp;" in out

    def test_unterminated_fence_still_renders(self) -> None:
        out = md_to_html("```python\nx = 1")
        assert "<pre" in out and "x = 1" in out

    def test_bold_italic_code_combo(self) -> None:
        out = md_to_html("**bold** and *italic* and `code`")
        assert "<strong>bold</strong>" in out
        assert "<em>italic</em>" in out
        assert "<code>code</code>" in out

    def test_unordered_list_dash(self) -> None:
        out = md_to_html("- one\n- two")
        assert "<ul><li>one</li><li>two</li></ul>" == out

    def test_unordered_list_star(self) -> None:
        out = md_to_html("* one\n* two")
        assert "<ul>" in out and "<li>one</li>" in out and "<li>two</li>" in out

    def test_ordered_list(self) -> None:
        out = md_to_html("1. first\n2. second")
        assert "<ol><li>first</li><li>second</li></ol>" == out

    def test_ordered_list_out_of_sequence_line_keeps_its_number(self) -> None:
        # A lone "3. 16" (a multiple-choice option named by its label) must keep its number:
        # a plain <ol> would renumber it from 1 and lose the label.
        out = md_to_html("The correct option is:\n3. 16")
        assert out == '<p>The correct option is:</p>\n<ol start="3"><li>16</li></ol>'
        doc = QTextDocument()
        doc.setHtml(out)
        starts = [b.textList().format().start() for b in _blocks(doc) if b.textList() is not None]
        assert starts == [3]

    def test_ordered_list_restarts_and_resumes_with_the_right_numbers(self) -> None:
        out = md_to_html("1. a\n2. b\n1. c")
        assert out == "<ol><li>a</li><li>b</li></ol>\n<ol><li>c</li></ol>"
        out = md_to_html("1. a\n3. c")
        assert out == '<ol><li>a</li></ol>\n<ol start="3"><li>c</li></ol>'
        # items separated by blank lines stay list items with their own numbers
        out = md_to_html("1. First\n\n2. Second\n\n3. Third")
        assert out == '<ol><li>First</li></ol>\n<ol start="2"><li>Second</li></ol>\n<ol start="3"><li>Third</li></ol>'

    def test_checkbox_unchecked(self) -> None:
        out = md_to_html("- [ ] todo item")
        assert "\u2610" in out
        assert "todo item" in out
        assert "\u2611" not in out

    def test_checkbox_checked(self) -> None:
        out = md_to_html("- [x] done item")
        assert "\u2611" in out
        assert "done item" in out

    def test_checkbox_checked_uppercase_x(self) -> None:
        out = md_to_html("- [X] done item")
        assert "\u2611" in out

    def test_heading_levels(self) -> None:
        assert md_to_html("# Title") == "<h1>Title</h1>"
        assert md_to_html("## Summary") == "<h2>Summary</h2>"
        assert md_to_html("### Sub") == "<h3>Sub</h3>"

    def test_html_is_escaped(self) -> None:
        out = md_to_html("<script>alert(1)</script> & co")
        assert "<script>" not in out
        assert "&lt;script&gt;" in out
        assert "&amp; co" in out

    def test_notes_style_mixed_document(self) -> None:
        text = "## Summary\nWe discussed **Kafka** vs *RabbitMQ*.\n\n## Action items\n- [ ] write doc\n- [x] send invite"
        out = md_to_html(text)
        assert out.index("<h2>Summary</h2>") < out.index("<p>")
        assert "<strong>Kafka</strong>" in out
        assert "<em>RabbitMQ</em>" in out
        assert "<h2>Action items</h2>" in out
        assert "\u2610" in out and "\u2611" in out

    def test_asterisk_edge_case_does_not_crash(self) -> None:
        # a lone/unmatched '*' must not raise and must still escape safely
        out = md_to_html("weird * text")
        assert "weird" in out


# =====================================================================================
# widgets helpers
# =====================================================================================
def test_level_to_percent_bounds() -> None:
    assert level_to_percent(0.0) == 0
    assert level_to_percent(1.0) == 100
    assert level_to_percent(-5.0) == 0
    assert level_to_percent(5.0) == 100
    assert 0 < level_to_percent(0.1) < 100


def test_fmt_mmss() -> None:
    assert fmt_mmss(0) == "00:00"
    assert fmt_mmss(65) == "01:05"
    assert fmt_mmss(-5) == "00:00"


# =====================================================================================
# OverlayWindow construction / win32 safety
# =====================================================================================
def test_construct_and_show_no_exception(window: OverlayWindow, qapp: QApplication) -> None:
    assert qapp.platformName() != "windows"
    assert window.isVisible()


def test_status_bar_defaults(window: OverlayWindow) -> None:
    assert window.status.asr_label.text().startswith("ASR")
    assert window.status.llm_label.text().startswith("LLM")


# =====================================================================================
# Live tab / transcript
# =====================================================================================
def test_partial_transcript_shown_then_replaced(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(PartialTranscriptEvent(source=Source.SYSTEM, text="hello wor", t_start=1.0))
    QTest.qWait(_WAIT_MS)
    assert Source.SYSTEM in window.live_view._partial_rows
    assert window.live_view._partial_rows[Source.SYSTEM].text_label.text() == "hello wor"

    bus.publish(PartialTranscriptEvent(source=Source.SYSTEM, text="hello world", t_start=1.0))
    QTest.qWait(_WAIT_MS)
    assert len(window.live_view._partial_rows) == 1
    assert window.live_view._partial_rows[Source.SYSTEM].text_label.text() == "hello world"
    assert window.live_view.row_count() == 0


def test_final_transcript_removes_partial_and_adds_row(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(PartialTranscriptEvent(source=Source.SYSTEM, text="hello wor", t_start=1.0))
    QTest.qWait(_WAIT_MS)

    seg = make_segment(text="hello world")
    bus.publish(FinalTranscriptEvent(segment=seg))
    QTest.qWait(_WAIT_MS)

    assert Source.SYSTEM not in window.live_view._partial_rows
    assert window.live_view.row_count() == 1
    assert window.live_view._final_rows[0].text_label.text() == "hello world"


def test_question_row_has_answer_button_calling_actions(
    window: OverlayWindow, bus: EventBus, actions: FakeActions
) -> None:
    seg = make_segment(text="what is the trade-off?", is_question=True)
    bus.publish(FinalTranscriptEvent(segment=seg))
    QTest.qWait(_WAIT_MS)

    row = window.live_view._final_rows[0]
    buttons = [b for b in row.findChildren(QPushButton) if b.text() == "Answer"]
    assert len(buttons) == 1
    QTest.mouseClick(buttons[0], Qt.MouseButton.LeftButton)
    assert ("answer_last", ()) in actions.calls


def test_non_question_row_has_no_answer_button(window: OverlayWindow, bus: EventBus) -> None:
    seg = make_segment(text="just a statement.", is_question=False)
    bus.publish(FinalTranscriptEvent(segment=seg))
    QTest.qWait(_WAIT_MS)
    row = window.live_view._final_rows[0]
    buttons = [b for b in row.findChildren(QPushButton) if b.text() == "Answer"]
    assert buttons == []


def test_speaker_renamed_event_relabels_existing_rows(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(FinalTranscriptEvent(segment=make_segment(speaker_id="spk1", speaker_label="Speaker 1")))
    QTest.qWait(_WAIT_MS)
    assert window.live_view._final_rows[0].chip.text() == "Speaker 1"

    bus.publish(SpeakerRenamedEvent(speaker_id="spk1", new_label="Alice"))
    QTest.qWait(_WAIT_MS)
    assert window.live_view._final_rows[0].chip.text() == "Alice"


def test_double_click_speaker_chip_renames(
    window: OverlayWindow, bus: EventBus, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        widgets_mod.QInputDialog, "getText", staticmethod(lambda *a, **kw: ("Bob", True))
    )
    bus.publish(FinalTranscriptEvent(segment=make_segment(speaker_id="spk2", speaker_label="Speaker 2")))
    QTest.qWait(_WAIT_MS)
    chip = window.live_view._final_rows[0].chip
    QTest.mouseDClick(chip, Qt.MouseButton.LeftButton)
    assert ("rename_speaker", ("spk2", "Bob")) in actions.calls


def test_transcript_row_cap(window: OverlayWindow, bus: EventBus) -> None:
    total = widgets_mod._MAX_TRANSCRIPT_ROWS + 10
    for i in range(total):
        bus.publish(FinalTranscriptEvent(segment=make_segment(id=i, text=f"line {i}", t_start=float(i))))

    def _done() -> bool:
        rows = window.live_view._final_rows
        return bool(rows) and rows[-1].text_label.text() == f"line {total - 1}"

    assert wait_until(_done), "final rows did not settle after the row-cap flood"
    assert window.live_view.row_count() == widgets_mod._MAX_TRANSCRIPT_ROWS
    # oldest rows were dropped, most recent kept
    assert window.live_view._final_rows[-1].text_label.text() == f"line {total - 1}"


def test_audio_level_event_updates_progress_bars(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(AudioLevelEvent(source=Source.MIC, rms=1.0, peak=1.0))
    bus.publish(AudioLevelEvent(source=Source.SYSTEM, rms=0.0, peak=0.0))
    QTest.qWait(_WAIT_MS)
    assert window.status.mic_level.value() == 100
    assert window.status.system_level.value() == 0


# =====================================================================================
# Assist tab
# =====================================================================================
def test_assist_started_switches_to_assist_tab(window: OverlayWindow, bus: EventBus) -> None:
    window._show_tab(0)   # start from Live: the overlay itself opens on Assist (test_overlay_start.py)
    assert window.stack.currentIndex() == 0
    bus.publish(AssistStartedEvent(job_id=1, kind="manual", question="what's the plan?"))
    QTest.qWait(_WAIT_MS)
    assert window.stack.currentIndex() == 1
    assert "what's the plan?" in window.assist_view.question_label.text()
    assert window.assist_view.stop_btn.isVisible()


def test_assist_deltas_stream_into_answer_and_done_sets_footer(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(AssistStartedEvent(job_id=7, kind="auto", question="q?"))
    bus.publish(AssistDeltaEvent(job_id=7, delta="Hello "))
    bus.publish(AssistDeltaEvent(job_id=7, delta="**world**"))
    assert wait_until(lambda: "world" in window.assist_view.answer_browser.toPlainText())
    text = window.assist_view.answer_browser.toPlainText()
    assert "Hello" in text and "world" in text

    bus.publish(AssistDoneEvent(job_id=7, text="Hello world", ttft_s=0.32, tokens=41, tok_per_s=41.0))
    QTest.qWait(_WAIT_MS)
    assert "auto" in window.assist_view.footer_label.text()
    assert "0.32" in window.assist_view.footer_label.text()
    assert not window.assist_view.stop_btn.isVisible()


def test_assist_stop_button_calls_cancel_assist(window: OverlayWindow, bus: EventBus, actions: FakeActions) -> None:
    bus.publish(AssistStartedEvent(job_id=2, kind="manual", question="q?"))
    QTest.qWait(_WAIT_MS)
    QTest.mouseClick(window.assist_view.stop_btn, Qt.MouseButton.LeftButton)
    assert ("cancel_assist", ()) in actions.calls


def test_assist_answer_last_button_calls_actions(window: OverlayWindow, actions: FakeActions) -> None:
    QTest.mouseClick(window.assist_view.answer_last_btn, Qt.MouseButton.LeftButton)
    assert ("answer_last", ()) in actions.calls


def test_assist_error_event_shown_in_view(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(AssistStartedEvent(job_id=3, kind="ask", question="q?"))
    QTest.qWait(_WAIT_MS)
    bus.publish(AssistErrorEvent(job_id=3, message="LM Studio server not reachable"))
    QTest.qWait(_WAIT_MS)
    assert "LM Studio server not reachable" in window.assist_view.answer_browser.toPlainText()
    assert not window.assist_view.stop_btn.isVisible()


def test_assist_done_pushes_history(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(AssistStartedEvent(job_id=9, kind="manual", question="history q"))
    bus.publish(AssistDoneEvent(job_id=9, text="history answer", ttft_s=0.1, tokens=5, tok_per_s=10.0))
    QTest.qWait(_WAIT_MS)
    assert window.assist_view.history_layout.count() == 1


# =====================================================================================
# Notes tab
# =====================================================================================
def test_notes_updated_event_renders_markdown(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(NotesUpdatedEvent(markdown="## Summary\n- point one", generated_at=125.0))
    QTest.qWait(_WAIT_MS)
    assert "Summary" in window.notes_view.browser.toPlainText()
    assert "point one" in window.notes_view.browser.toPlainText()
    assert window.notes_view.updated_label.text() == "last updated 02:05"


def test_notes_regenerate_button_calls_actions(window: OverlayWindow, actions: FakeActions) -> None:
    QTest.mouseClick(window.notes_view.regenerate_btn, Qt.MouseButton.LeftButton)
    assert ("generate_notes", ()) in actions.calls


def test_notes_error_event_shows_toast(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(NotesErrorEvent(message="notes failed"))
    QTest.qWait(_WAIT_MS)
    assert window.toast.isVisible()
    assert "notes failed" in window.toast.text()
    assert "notes failed" in window.notes_view.updated_label.text()


# =====================================================================================
# Status / component events
# =====================================================================================
def test_component_status_updates_labels_and_dot(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(ComponentStatusEvent(component="asr", state="loading", detail="cuda float16"))
    QTest.qWait(_WAIT_MS)
    assert "loading" in window.status.asr_label.toolTip()

    bus.publish(ComponentStatusEvent(component="asr", state="ready", detail="cuda float16"))
    bus.publish(ComponentStatusEvent(component="llm", state="ready", detail="maple-preview"))
    QTest.qWait(_WAIT_MS)
    assert window.status.asr_label.text() == "ASR" and "cuda float16" in window.status.asr_label.toolTip()
    assert window.status.llm_label.text() == "LLM maple-preview"
    from exocortex.ui.theme import COLORS

    assert COLORS["ok"] in window.status_dot.styleSheet()


def test_component_status_error_turns_dot_red_and_toasts(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(ComponentStatusEvent(component="asr", state="error", detail="cuda oom"))
    QTest.qWait(_WAIT_MS)
    from exocortex.ui.theme import COLORS

    assert COLORS["err"] in window.status_dot.styleSheet()
    assert "cuda oom" in window.status.asr_label.toolTip()


def test_session_event_shows_toast(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(SessionEvent(kind="saved", path="C:/sessions/x"))
    QTest.qWait(_WAIT_MS)
    assert window.toast.isVisible()
    assert "saved" in window.toast.text().lower()


def test_auto_assist_changed_event_syncs_toggle(window: OverlayWindow, bus: EventBus) -> None:
    window.auto_btn.setChecked(True)
    bus.publish(AutoAssistChangedEvent(enabled=False))
    QTest.qWait(_WAIT_MS)
    assert window.auto_btn.isChecked() is False

    bus.publish(AutoAssistChangedEvent(enabled=True))
    QTest.qWait(_WAIT_MS)
    assert window.auto_btn.isChecked() is True


def test_auto_toggle_button_calls_set_auto(window: OverlayWindow, actions: FakeActions) -> None:
    window.auto_btn.setChecked(not window.auto_btn.isChecked())
    assert any(c[0] == "set_auto" for c in actions.calls)


def test_mic_toggle_button_calls_set_mic_muted(window: OverlayWindow, actions: FakeActions) -> None:
    window.mic_btn.setChecked(not window.mic_btn.isChecked())
    assert any(c[0] == "set_mic_muted" for c in actions.calls)


# =====================================================================================
# AskBox
# =====================================================================================
def test_askbox_enter_sends_and_clears(window: OverlayWindow, actions: FakeActions) -> None:
    window.askbox.edit.setText("what time is it")
    QTest.keyClick(window.askbox.edit, Qt.Key.Key_Return)
    assert ("ask", ("what time is it",)) in actions.calls
    assert window.askbox.edit.text() == ""


def test_askbox_escape_clears_without_sending(window: OverlayWindow, actions: FakeActions) -> None:
    window.askbox.edit.setText("never sent")
    QTest.keyClick(window.askbox.edit, Qt.Key.Key_Escape)
    assert window.askbox.edit.text() == ""
    assert not any(c[0] == "ask" for c in actions.calls)


def test_askbox_send_button(window: OverlayWindow, actions: FakeActions) -> None:
    window.askbox.edit.setText("via button")
    QTest.mouseClick(window.askbox.send_btn, Qt.MouseButton.LeftButton)
    assert ("ask", ("via button",)) in actions.calls


def test_askbox_empty_text_does_not_call_ask(window: OverlayWindow, actions: FakeActions) -> None:
    window.askbox.edit.setText("   ")
    QTest.keyClick(window.askbox.edit, Qt.Key.Key_Return)
    assert not any(c[0] == "ask" for c in actions.calls)


def test_focus_ask_gives_focus(window: OverlayWindow) -> None:
    window.focus_ask()
    QTest.qWait(_WAIT_MS)
    assert window.isVisible()
    assert window.askbox.edit.hasFocus()


# =====================================================================================
# Public slots
# =====================================================================================
def test_toggle_visible(window: OverlayWindow) -> None:
    assert window.isVisible()
    window.toggle_visible()
    assert not window.isVisible()
    window.toggle_visible()
    assert window.isVisible()


def test_nudge_moves_window(window: OverlayWindow) -> None:
    before = window.pos()
    window.nudge(40, 0)
    assert window.pos().x() == before.x() + 40


def test_set_opacity_clamped_and_syncs_slider(window: OverlayWindow) -> None:
    window.set_opacity(0.55)
    # Qt may quantize window opacity (8-bit alpha internally on some platforms).
    assert abs(window.windowOpacity() - 0.55) < 0.01
    assert window.opacity_slider.value() == 55

    window.set_opacity(2.0)
    assert window.windowOpacity() == 1.0
    window.set_opacity(-1.0)
    assert window.windowOpacity() == 0.4


def test_show_toast_sets_text_and_hides_later(window: OverlayWindow) -> None:
    window.show_toast("hello toast")
    assert window.toast.isVisible()
    assert window.toast.text() == "hello toast"


def test_set_click_through_toggles_state_without_exception(window: OverlayWindow) -> None:
    window.set_click_through(True)
    assert window._click_through is True
    assert window.clickthrough_btn.isChecked() is True
    window.set_click_through(False)
    assert window._click_through is False


# =====================================================================================
# Close / geometry persistence
# =====================================================================================
def test_close_event_calls_quit_and_saves_geometry(qapp: QApplication, bus: EventBus, actions: FakeActions) -> None:
    cfg = Config()
    win = OverlayWindow(cfg, bus, actions)
    win.show()
    qapp.processEvents()
    win.resize(500, 700)
    win.close()
    qapp.processEvents()
    assert ("quit", ()) in actions.calls
    assert cfg.ui.width == 500
    assert cfg.ui.height == 700


def test_close_silently_does_not_call_quit(qapp: QApplication, bus: EventBus, actions: FakeActions) -> None:
    cfg = Config()
    win = OverlayWindow(cfg, bus, actions)
    win.show()
    qapp.processEvents()
    win.close_silently()
    qapp.processEvents()
    assert not any(c[0] == "quit" for c in actions.calls)


def test_close_silently_stops_all_recurring_timers(
    qapp: QApplication, bus: EventBus, actions: FakeActions
) -> None:
    """Regression: closing must stop every QTimer, or it keeps firing (and keeps the
    whole widget tree alive) for the rest of the process, since close() only hides a
    QWidget rather than destroying it."""
    cfg = Config()
    win = OverlayWindow(cfg, bus, actions)
    win.show()
    qapp.processEvents()
    QTest.qWait(30)
    assert win._poll_timer.isActive()
    assert win._topmost_timer.isActive()
    assert win.assist_view._timer.isActive()

    win.close_silently()
    qapp.processEvents()

    assert not win._poll_timer.isActive()
    assert not win._topmost_timer.isActive()
    assert not win._toast_timer.isActive()
    assert not win.assist_view._timer.isActive()


def test_close_event_also_stops_all_recurring_timers(
    qapp: QApplication, bus: EventBus, actions: FakeActions
) -> None:
    """Same as above but through a real (non-silent) close()."""
    cfg = Config()
    win = OverlayWindow(cfg, bus, actions)
    win.show()
    qapp.processEvents()
    win.close()
    qapp.processEvents()
    assert not win._poll_timer.isActive()
    assert not win._topmost_timer.isActive()
    assert not win.assist_view._timer.isActive()


def test_events_published_after_close_are_not_processed(
    qapp: QApplication, bus: EventBus, actions: FakeActions
) -> None:
    """The poll timer is stopped and the queue unsubscribed on close, so an event
    published afterwards (e.g. a slow-to-shut-down engine thread) must not touch any
    widget on the now-closed window."""
    cfg = Config()
    win = OverlayWindow(cfg, bus, actions)
    win.show()
    qapp.processEvents()
    win.close_silently()
    qapp.processEvents()

    bus.publish(FinalTranscriptEvent(segment=make_segment(text="after close")))
    QTest.qWait(_WAIT_MS)
    qapp.processEvents()
    assert win.live_view.row_count() == 0


# =====================================================================================
# Hotkeys must never touch the real OS outside the native "windows" Qt platform
# =====================================================================================
class _SpyHotkeys:
    """Stand-in for GlobalHotkeys that records construction/install/uninstall instead
    of ever touching RegisterHotKey."""

    instances: list["_SpyHotkeys"] = []

    def __init__(self, bindings: dict) -> None:
        self.bindings = bindings
        self.installed = False
        self.uninstalled = False
        self.failed: list[str] = []
        _SpyHotkeys.instances.append(self)

    def install(self, app: Any) -> None:
        self.installed = True

    def uninstall(self) -> None:
        self.uninstalled = True


def test_hotkeys_not_installed_under_offscreen_platform(
    qapp: QApplication, bus: EventBus, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: GlobalHotkeys.install() makes a REAL, machine-wide RegisterHotKey
    Win32 call on Windows (it only checks sys.platform, not the Qt platform plugin).
    The overlay must not install real hotkeys while running under the "offscreen" QPA
    platform tests use, even on a real Windows dev machine, or every offline test run
    would transiently grab the user's actual global hotkeys."""
    _SpyHotkeys.instances = []
    monkeypatch.setattr(overlay_mod, "GlobalHotkeys", _SpyHotkeys)
    cfg = Config()
    win = OverlayWindow(cfg, bus, actions)
    win.show()
    qapp.processEvents()
    assert win._hotkeys is None
    assert _SpyHotkeys.instances == []
    win.close_silently()
    qapp.processEvents()


# =====================================================================================
# window management: the stub win32 the topmost-backstop tests drive
# =====================================================================================
class _StubWin32:
    """Stand-in for exocortex.ui.win32 that records the usability calls the overlay makes.

    dev-assist is a visible copilot, so there is no capture/exposure surface to simulate -- the
    only win32 the overlay touches now is the topmost backstop, click-through and no-activate.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))

    def set_topmost(self, hwnd: int) -> None:
        self._record("set_topmost", hwnd)

    def set_click_through(self, hwnd: int, enabled: bool) -> None:
        self._record("set_click_through", hwnd, enabled)

    def set_no_activate(self, hwnd: int, enabled: bool) -> None:
        self._record("set_no_activate", hwnd, enabled)

    def names(self) -> list[str]:
        return [name for name, _args in self.calls]


@pytest.fixture()
def fake_windows(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """Make the overlay believe it runs on a real Windows desktop with a stub ``win32``."""

    def build(stub: _StubWin32, **ui_overrides: Any) -> Any:
        monkeypatch.setattr(overlay_mod, "win32", stub)
        monkeypatch.setattr(overlay_mod, "_is_windows_platform", lambda: True)
        monkeypatch.setattr(overlay_mod, "GlobalHotkeys", _SpyHotkeys)
        _SpyHotkeys.instances = []
        cfg = Config()
        for key, value in ui_overrides.items():
            setattr(cfg.ui, key, value)
        return cfg

    return build


# --------------------------------------------------------- bottom controls + choices ----
def test_control_row_order_is_read_revert_reset_pause_checkpoint_text_code(window: OverlayWindow) -> None:
    """One row: Read, Revert, Reset, Pause, Checkpoint, a stretch, then Text and Code on the right."""
    from PySide6.QtWidgets import QSizePolicy as _QSP

    row = window.read_btn.parentWidget()
    assert row is window.control_row
    layout = row.layout()
    items = [layout.itemAt(i) for i in range(layout.count())]
    widgets = [it.widget() for it in items]
    assert [w.text() for w in widgets[:5]] == ["Read", "Revert", "Reset", "Pause", "Checkpoint"]
    assert widgets[5] is None and items[5].spacerItem() is not None   # the stretch that right-aligns Text + Code
    assert widgets[6] is window.text_btn and widgets[6].text() == "Text"
    assert widgets[7] is window.code_btn and widgets[7].text() == "Code" and len(items) == 8
    assert window.text_btn.objectName() == "primary" == window.code_btn.objectName()
    # The row never raises the window's minimum width (the offscreen font is far wider than the
    # real one; see _build_control_row).
    assert row.sizePolicy().horizontalPolicy() == _QSP.Policy.Ignored
    # The Screen tab no longer carries its own copy of the read button.
    assert not hasattr(window.screen_view, "read_now_btn")


def test_control_row_read_forces_a_capture(window: OverlayWindow, actions: FakeActions) -> None:
    window.read_btn.click()
    assert actions.calls == [("capture_screen_now", ())]


def test_control_row_revert_pause_reset(window: OverlayWindow, actions: FakeActions) -> None:
    # Two answers so there is something to revert to.
    window.assist_view.on_started(1, "solve", "Solving")
    window.assist_view.on_done(1, "first answer", 0.1, 3, 60.0)
    window.assist_view.on_started(2, "solve", "Solving")
    window.assist_view.on_done(2, "second answer", 0.1, 3, 60.0)

    window.revert_btn.click()
    assert "first answer" in window.assist_view.answer_browser.toPlainText()

    window.pause_btn.setChecked(True)
    assert ("set_paused", (True,)) in actions.calls
    assert window.pause_btn.text() == "Resume"
    window.pause_btn.setChecked(False)
    assert ("set_paused", (False,)) in actions.calls

    window.reset_btn.click()
    assert ("new_session", ()) in actions.calls
    assert window.assist_view.answer_browser.toPlainText().strip() == ""


def test_solve_answer_renders_choice_buttons(window: OverlayWindow, actions: FakeActions) -> None:
    from PySide6.QtWidgets import QPushButton as _QPB

    window.assist_view.on_started(5, "solve", "Solving")
    text = "[[CHOICE]] Solve in Python\n[[CHOICE]] Solve in C++"
    window.assist_view.on_done(5, text, 0.1, 4, 60.0)

    assert window.assist_view.choices_container.isVisibleTo(window.assist_view)
    btns = window.assist_view.choices_container.findChildren(_QPB)
    assert [b.text() for b in btns] == ["Solve in Python", "Solve in C++"]
    btns[1].click()
    assert ("solve", ("Solve in C++",)) in actions.calls


def test_multiple_choice_verdict_renders_as_its_own_bold_line(window: OverlayWindow) -> None:
    text = "7 has no divisors besides 1 and itself.\n\n**Answer: C — 7**"
    html = md_to_html(text)
    assert html.endswith("<p><strong>Answer: C — 7</strong></p>")
    window.assist_view.on_started(7, "solve", "Solving")
    window.assist_view.on_done(7, text, 0.1, 4, 60.0)
    shown = window.assist_view.answer_browser.toPlainText()
    assert shown.rstrip().endswith("Answer: C — 7")
    assert not window.assist_view.choices_container.isVisibleTo(window.assist_view)


def test_solve_answer_without_choices_shows_no_buttons(window: OverlayWindow) -> None:
    from PySide6.QtWidgets import QPushButton as _QPB

    window.assist_view.on_started(6, "solve", "Solving")
    window.assist_view.on_done(6, "just the solution, no forks", 0.1, 4, 60.0)
    assert not window.assist_view.choices_container.findChildren(_QPB)


# ------------------------------------------- window stays reachable / resize / right-click ----
def _long_answer() -> str:
    code = "\n".join(f"line_{j} = {j}" for j in range(25))
    return "\n\n".join(f"Step {i}: " + "word " * 60 + f"\n\n```python\n{code}\n```" for i in range(3))


def test_expanding_history_does_not_grow_the_window(window: OverlayWindow) -> None:
    """Regression: 20 long answers laid out outside a scroll area forced the window's minimum
    height past 16 000 px the moment "History" was expanded, shoving the title bar off-screen."""
    for i in range(1, 12):
        window.assist_view.on_started(i, "solve", f"q{i}")
        window.assist_view.on_done(i, _long_answer(), 0.1, 700, 40.0)
    before = window.height()
    window.assist_view.history_toggle.setChecked(True)
    QTest.qWait(_WAIT_MS)
    assert window.assist_view.history_scroll.isVisibleTo(window.assist_view)
    assert window.height() == before

    window.screen_view.finish_suggestion("a suggestion\nwith detail", cancelled=False)
    window.screen_view.suggestion_history_toggle.setChecked(True)
    QTest.qWait(_WAIT_MS)
    assert window.height() == before


def test_saved_geometry_is_clamped_to_the_screen(qapp: QApplication, bus: EventBus, actions: FakeActions) -> None:
    cfg = Config()
    cfg.ui.x, cfg.ui.y, cfg.ui.height = 100, 100_000, 8054
    win = OverlayWindow(cfg, bus, actions)
    try:
        geo = win.screen().availableGeometry()
        assert win.height() <= geo.height()
        assert geo.top() <= win.y() <= geo.bottom() - overlay_mod._TITLE_STRIP_PX + 1
    finally:
        win.close_silently()
        qapp.processEvents()


def test_nudge_keeps_the_title_bar_on_screen(window: OverlayWindow) -> None:
    geo = window.screen().availableGeometry()
    window.nudge(0, -100_000)
    assert window.y() == geo.top()
    window.nudge(0, 100_000)
    assert window.y() == geo.bottom() - overlay_mod._TITLE_STRIP_PX + 1
    window.nudge(-100_000, 0)
    assert window.geometry().right() >= geo.left() + overlay_mod._KEEP_VISIBLE_PX - 1


def _grip_event(kind: Any, grip: Any, local: Any, global_pos: Any, buttons: Any) -> Any:
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent

    return QMouseEvent(
        kind, QPointF(local), QPointF(global_pos), Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier
    )


def test_edge_grips_resize_from_every_edge_and_corner(window: OverlayWindow) -> None:
    from PySide6.QtCore import QEvent, QPoint

    window.setGeometry(200, 60, 400, 600)
    QTest.qWait(_WAIT_MS)
    grips = {g._edge: g for g in window._edge_grips}
    assert set(grips) == {Qt.Edge.LeftEdge, Qt.Edge.RightEdge, Qt.Edge.TopEdge, Qt.Edge.BottomEdge}

    top = grips[Qt.Edge.TopEdge]
    assert top.geometry().width() == window.width()
    assert top.edges_at(QPoint(2, 2)) == Qt.Edge.TopEdge | Qt.Edge.LeftEdge
    assert top.edges_at(QPoint(window.width() // 2, 2)) == Qt.Edge.TopEdge
    assert top.edges_at(QPoint(window.width() - 2, 2)) == Qt.Edge.TopEdge | Qt.Edge.RightEdge

    # Drag the top-left corner 30 px left / 20 px up: that corner moves, the opposite one stays.
    start = window.geometry()
    press = start.topLeft() + QPoint(2, 2)
    left = Qt.MouseButton.LeftButton
    top.mousePressEvent(_grip_event(QEvent.Type.MouseButtonPress, top, QPoint(2, 2), press, left))
    top.mouseMoveEvent(_grip_event(QEvent.Type.MouseMove, top, QPoint(2, 2), press + QPoint(-30, -20), left))
    top.mouseReleaseEvent(
        _grip_event(QEvent.Type.MouseButtonRelease, top, QPoint(2, 2), press, Qt.MouseButton.NoButton)
    )
    geo = window.geometry()
    assert (geo.left(), geo.top()) == (start.left() - 30, start.top() - 20)
    assert geo.bottomRight() == start.bottomRight()

    # Dragging the left edge far to the right stops at the minimum width; the right edge stays.
    lgrip = grips[Qt.Edge.LeftEdge]
    start = window.geometry()
    mid = QPoint(2, start.height() // 2)
    press = start.topLeft() + mid
    lgrip.mousePressEvent(_grip_event(QEvent.Type.MouseButtonPress, lgrip, mid, press, left))
    lgrip.mouseMoveEvent(_grip_event(QEvent.Type.MouseMove, lgrip, mid, press + QPoint(5000, 0), left))
    lgrip.mouseReleaseEvent(_grip_event(QEvent.Type.MouseButtonRelease, lgrip, mid, press, Qt.MouseButton.NoButton))
    assert window.width() == window.minimumWidth()
    assert window.geometry().right() == start.right()


def test_right_click_opens_menu_with_quit(window: OverlayWindow, qapp: QApplication) -> None:
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtWidgets import QMenu

    # On the chrome: the "..." menu itself.
    pos = QPoint(5, 5)
    window.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Reason.Mouse, pos, window.mapToGlobal(pos)))
    assert window._menu.isVisible()
    assert "Quit" in [a.text() for a in window._menu.actions()]
    window._menu.hide()

    # On a text widget: its own Copy/Select-all menu, extended with Hide/Quit.
    for owner, target in (
        (window.assist_view.answer_browser, window.assist_view.answer_browser.viewport()),
        (window.askbox.findChild(widgets_mod.QLineEdit), window.askbox.findChild(widgets_mod.QLineEdit)),
    ):
        ev = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, pos, target.mapToGlobal(pos))
        qapp.sendEvent(target, ev)
        menus = [m for m in owner.findChildren(QMenu) if m.isVisible()]
        assert menus, owner
        texts = [a.text() for a in menus[-1].actions()]
        assert "Hide overlay" in texts and "Quit ExoCortex" in texts
        menus[-1].close()


# =====================================================================================
# an open menu / dropdown owns the z-order: the topmost backstop must not restack under it
# =====================================================================================
def test_topmost_tick_does_not_restack_while_a_popup_is_open(
    qapp: QApplication, bus: EventBus, actions: FakeActions, fake_windows: Any
) -> None:
    """Measured on build 26200: each ``set_topmost`` puts its window at the top of the topmost
    band, so pushing the overlay (and then the settings dialog) back up while a combo box
    dropdown is open drops the list *behind* them. It stays open and keeps the mouse grab, so
    it reads as the dropdown vanishing a beat after it was opened and the next click only
    dismisses it. The tick skips the restack entirely while a popup is up."""
    from PySide6.QtWidgets import QMenu

    stub = _StubWin32()
    cfg = fake_windows(stub)
    win = OverlayWindow(cfg, bus, actions)
    win.show()
    qapp.processEvents()

    menu = QMenu(win)
    menu.addAction("something")
    menu.popup(win.mapToGlobal(QPoint(5, 5)))
    qapp.processEvents()
    assert QApplication.activePopupWidget() is menu

    stub.calls.clear()
    win._reassert_topmost()
    qapp.processEvents()
    assert "set_topmost" not in stub.names()          # nothing was restacked under the menu

    menu.close()
    qapp.processEvents()
    stub.calls.clear()
    win._reassert_topmost()
    assert "set_topmost" in stub.names()              # and it resumes once the menu is gone

    win.close_silently()
    qapp.processEvents()


def test_transients_are_raised_bottom_up_so_a_tool_tip_stays_readable(
    qapp: QApplication, bus: EventBus, actions: FakeActions, fake_windows: Any
) -> None:
    """The window raised last ends up in front, so the order is overlay, then the settings
    dialog, then any tool tip over it - raising the dialog last would hide the tip."""
    from PySide6.QtCore import QPoint as _QPoint
    from PySide6.QtWidgets import QToolTip

    stub = _StubWin32()
    cfg = fake_windows(stub)
    win = OverlayWindow(cfg, bus, actions)
    win.show()
    win.open_settings()
    qapp.processEvents()
    assert win._settings is not None

    QToolTip.showText(win.mapToGlobal(_QPoint(20, 20)), "the full model id", win)
    qapp.processEvents()
    tip = next((w for w in QApplication.topLevelWidgets()
                if w.isVisible() and w.metaObject().className() == "QTipLabel"), None)

    stub.calls.clear()
    win._reassert_topmost()
    raised = [args[0] for name, args in stub.calls if name == "set_topmost"]
    assert raised[0] == int(win.winId())
    assert raised[1] == int(win._settings.winId())
    if tip is not None:                               # the platform plugin may not show one
        assert raised[-1] == int(tip.winId())

    QToolTip.hideText()
    win.close_silently()
    qapp.processEvents()
