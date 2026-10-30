"""Offscreen tests for the overlay's capabilities wiring (2026-10-07): the "Capabilities..." action
in both menus (it opens the Models window on its Capabilities tab), the forwarding of
CapabilityMapEvent / McpToolsEvent to the open Models dialog with a toast for an indexed map and
for an error, and ToolCallEvent -> the Assist tab's footer. Mirrors tests/test_overlay_context.py."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from exocortex.bus import (
    AssistDoneEvent,
    AssistStartedEvent,
    CapabilityMapEvent,
    ContextLibraryEvent,
    EventBus,
    McpToolsEvent,
    ToolCallEvent,
)
from exocortex.config import Config
from exocortex.ui.overlay import OverlayWindow
from tests.test_overlay_completion import FakeActions

_WAIT_MS = 80
_ACTION = "Capabilities…"


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-tests"])


@pytest.fixture()
def actions() -> FakeActions:
    return FakeActions()


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def window(qapp: QApplication, bus: EventBus, actions: FakeActions):
    win = OverlayWindow(Config(), bus, actions)
    win.show()
    qapp.processEvents()
    QTest.qWait(30)
    yield win
    win.close_silently()
    qapp.processEvents()


class _FakeModelsDialog:
    """Stand-in for ``ModelsDialog``: records ``on_event`` calls and the tab it was told to show."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.show_tab_calls: list[str] = []
        self.load_calls = 0
        self.shown = 0

    def on_event(self, ev: object) -> None:
        self.events.append(ev)

    def load(self) -> None:
        self.load_calls += 1

    def show_tab(self, name: str) -> None:
        self.show_tab_calls.append(name)

    def show(self) -> None:
        self.shown += 1

    def raise_(self) -> None:
        pass

    def activateWindow(self) -> None:
        pass


# =====================================================================================
# the menus
# =====================================================================================

def test_both_menus_have_a_capabilities_action_after_context_documents(window: OverlayWindow) -> None:
    window._rebuild_menu()
    texts = [a.text() for a in window._menu.actions()]
    assert texts.count(_ACTION) == 1
    assert texts[texts.index("Context documents…") + 1] == _ACTION

    window._rebuild_context_menu()
    ctx_texts = [a.text() for a in window._context_menu.actions()]
    assert ctx_texts.count(_ACTION) == 1
    assert ctx_texts[ctx_texts.index("Context documents…") + 1] == _ACTION


def test_main_menu_action_opens_the_models_window_on_capabilities(
    window: OverlayWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str | None] = []
    monkeypatch.setattr(window, "open_models", lambda tab=None: opened.append(tab))
    window._rebuild_menu()
    next(a for a in window._menu.actions() if a.text() == _ACTION).trigger()
    assert opened == ["Capabilities"]


def test_slim_bar_menu_action_opens_the_models_window_on_capabilities(
    window: OverlayWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str | None] = []
    monkeypatch.setattr(window, "open_models", lambda tab=None: opened.append(tab))
    window._rebuild_context_menu()
    next(a for a in window._context_menu.actions() if a.text() == _ACTION).trigger()
    assert opened == ["Capabilities"]


def test_the_action_shows_the_tab_on_a_reused_dialog(window: OverlayWindow) -> None:
    window._rebuild_menu()
    fake = _FakeModelsDialog()
    window._models_dialog = fake
    next(a for a in window._menu.actions() if a.text() == _ACTION).trigger()
    assert fake.load_calls == 1 and fake.shown == 1
    assert fake.show_tab_calls == ["Capabilities"]


def test_the_action_builds_the_real_dialog_on_the_capabilities_tab(window: OverlayWindow) -> None:
    window._rebuild_menu()
    assert window._models_dialog is None
    next(a for a in window._menu.actions() if a.text() == _ACTION).trigger()
    dlg = window._models_dialog
    assert dlg is not None and dlg.isVisible()
    assert dlg.tabs.tabText(dlg.tabs.currentIndex()) == "Capabilities"
    dlg.close()
    QTest.qWait(30)


def test_the_existing_context_documents_action_still_opens_the_context_tab(window: OverlayWindow) -> None:
    window._rebuild_menu()
    fake = _FakeModelsDialog()
    window._models_dialog = fake
    next(a for a in window._menu.actions() if a.text() == "Context documents…").trigger()
    assert fake.show_tab_calls == ["Context"]


# =====================================================================================
# event forwarding and toasts
# =====================================================================================

def test_capability_events_are_forwarded_to_the_open_dialog(window: OverlayWindow, bus: EventBus) -> None:
    fake = _FakeModelsDialog()
    window._models_dialog = fake
    maps = CapabilityMapEvent(reason="indexing", name="proj")
    tools = McpToolsEvent(request_id=2, server="fs", tools=({"name": "read", "description": ""},))
    bus.publish(maps)
    bus.publish(tools)
    QTest.qWait(_WAIT_MS)
    assert fake.events == [maps, tools]
    assert fake.show_tab_calls == []                 # forwarding never changes the visible tab


def test_an_indexed_map_toasts_its_file_count(window: OverlayWindow, bus: EventBus) -> None:
    fake = _FakeModelsDialog()
    window._models_dialog = fake
    ev = CapabilityMapEvent(reason="indexed", name="proj", files=212, dirs=31)
    bus.publish(ev)
    QTest.qWait(_WAIT_MS)
    assert fake.events == [ev]
    assert window.toast.isVisibleTo(window)
    assert window.toast.text() == "Capabilities map 'proj' indexed: 212 files"


def test_the_launch_refresh_of_an_unchanged_map_is_forwarded_but_does_not_toast(
        window: OverlayWindow, bus: EventBus) -> None:
    # 2026-10-29: "Map 'BloomingHealth': 28 files" at every launch read as 28 documents loaded
    fake = _FakeModelsDialog()
    window._models_dialog = fake
    window.show_toast("sentinel")
    ev = CapabilityMapEvent(reason="indexed", name="proj", files=28, dirs=2, automatic=True, changed=False)
    bus.publish(ev)
    QTest.qWait(_WAIT_MS)
    assert fake.events == [ev]                       # the tab still re-reads its list
    assert window.toast.text() == "sentinel"


def test_the_launch_refresh_of_a_changed_map_toasts_as_a_refresh(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(CapabilityMapEvent(reason="indexed", name="proj", files=29, dirs=2, automatic=True, changed=True))
    QTest.qWait(_WAIT_MS)
    assert window.toast.text() == "Capabilities map 'proj' refreshed: 29 files"


def test_a_manual_index_of_an_unchanged_map_still_toasts(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(CapabilityMapEvent(reason="indexed", name="proj", files=5, dirs=1, changed=False))
    QTest.qWait(_WAIT_MS)
    assert window.toast.text() == "Capabilities map 'proj' indexed: 5 files"


def test_a_map_error_toasts_the_reason(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(CapabilityMapEvent(reason="error: folder not found: C:/gone"))
    QTest.qWait(_WAIT_MS)
    assert window.toast.text() == "Capabilities map error: folder not found: C:/gone"


@pytest.mark.parametrize("reason", ["indexing", "loaded", "deleted"])
def test_the_other_map_events_do_not_toast(window: OverlayWindow, bus: EventBus, reason: str) -> None:
    window.show_toast("sentinel")
    bus.publish(CapabilityMapEvent(reason=reason, name="proj"))
    QTest.qWait(_WAIT_MS)
    assert window.toast.text() == "sentinel"


def test_an_mcp_tools_event_does_not_toast(window: OverlayWindow, bus: EventBus) -> None:
    window.show_toast("sentinel")
    bus.publish(McpToolsEvent(request_id=1, server="fs", error="refused"))
    QTest.qWait(_WAIT_MS)
    assert window.toast.text() == "sentinel"


def test_capability_events_before_the_dialog_exists_do_not_open_it_or_crash(
    window: OverlayWindow, bus: EventBus
) -> None:
    assert window._models_dialog is None
    bus.publish(CapabilityMapEvent(reason="indexed", name="x", files=1, dirs=1))
    bus.publish(McpToolsEvent(request_id=1, server="fs"))
    QTest.qWait(_WAIT_MS)
    assert window._models_dialog is None
    assert window.toast.text() == "Capabilities map 'x' indexed: 1 files"      # the toast does not need the window


def test_capability_events_reach_a_real_dialog_and_do_not_disturb_other_forwarding(
    window: OverlayWindow, bus: EventBus
) -> None:
    window.open_models("Capabilities")
    dlg = window._models_dialog
    assert dlg is not None
    bus.publish(CapabilityMapEvent(reason="indexing", name="proj"))
    bus.publish(ContextLibraryEvent(reason="added", doc_id="d"))
    QTest.qWait(_WAIT_MS)
    assert dlg.cap_map_status.text() == "Indexing proj\u2026"
    dlg.close()
    QTest.qWait(30)


# =====================================================================================
# ToolCallEvent -> the Assist footer
# =====================================================================================

def _start(bus: EventBus, job_id: int = 7, kind: str = "ask") -> None:
    bus.publish(AssistStartedEvent(job_id=job_id, kind=kind, question="what does a.py do?"))
    QTest.qWait(_WAIT_MS)


def test_a_running_tool_call_shows_in_the_footer(window: OverlayWindow, bus: EventBus) -> None:
    _start(bus)
    bus.publish(ToolCallEvent(job_id=7, index=1, name="read_file", summary="src/a.py", state="running"))
    QTest.qWait(_WAIT_MS)
    assert window.assist_view.footer_label.text() == "ask - read_file src/a.py\u2026"


def test_a_finished_tool_call_shows_its_detail(window: OverlayWindow, bus: EventBus) -> None:
    _start(bus)
    bus.publish(ToolCallEvent(job_id=7, index=1, name="read_file", summary="src/a.py", state="running"))
    bus.publish(ToolCallEvent(job_id=7, index=1, name="read_file", summary="src/a.py", state="done",
                              detail="3.2 KB", elapsed_s=0.02))
    QTest.qWait(_WAIT_MS)
    assert window.assist_view.footer_label.text() == "ask - read_file src/a.py (3.2 KB)"


@pytest.mark.parametrize("state, detail", [("error", "file not found"), ("denied", "outside the readable folders")])
def test_a_failed_or_denied_tool_call_shows_the_reason(
    window: OverlayWindow, bus: EventBus, state: str, detail: str
) -> None:
    _start(bus)
    bus.publish(ToolCallEvent(job_id=7, index=2, name="run_command", summary="dir C:/x", state=state, detail=detail))
    QTest.qWait(_WAIT_MS)
    assert window.assist_view.footer_label.text() == f"ask - run_command dir C:/x ({detail})"


def test_a_tool_call_without_detail_has_no_empty_parentheses(window: OverlayWindow, bus: EventBus) -> None:
    _start(bus)
    bus.publish(ToolCallEvent(job_id=7, index=1, name="list_dir", summary="src", state="done"))
    QTest.qWait(_WAIT_MS)
    assert window.assist_view.footer_label.text() == "ask - list_dir src"


def test_another_jobs_tool_call_is_ignored(window: OverlayWindow, bus: EventBus) -> None:
    _start(bus, job_id=7)
    before = window.assist_view.footer_label.text()
    bus.publish(ToolCallEvent(job_id=99, index=1, name="read_file", summary="x", state="running"))
    QTest.qWait(_WAIT_MS)
    assert window.assist_view.footer_label.text() == before == "ask - streaming..."


def test_the_done_event_replaces_the_tool_line(window: OverlayWindow, bus: EventBus) -> None:
    _start(bus)
    bus.publish(ToolCallEvent(job_id=7, index=1, name="read_file", summary="src/a.py", state="done", detail="1 KB"))
    bus.publish(AssistDoneEvent(job_id=7, text="It parses.", ttft_s=0.32, tokens=12, tok_per_s=40.0))
    QTest.qWait(_WAIT_MS)
    text = window.assist_view.footer_label.text()
    assert "read_file" not in text and text.startswith("ask - ttft 0.32")


def test_a_late_tool_call_after_the_answer_finished_is_ignored(window: OverlayWindow, bus: EventBus) -> None:
    _start(bus)
    bus.publish(AssistDoneEvent(job_id=7, text="Done.", ttft_s=0.1, tokens=3, tok_per_s=30.0))
    QTest.qWait(_WAIT_MS)
    footer = window.assist_view.footer_label.text()
    bus.publish(ToolCallEvent(job_id=7, index=1, name="read_file", summary="late", state="running"))
    QTest.qWait(_WAIT_MS)
    assert window.assist_view.footer_label.text() == footer


def test_a_tool_call_of_a_suggest_job_never_touches_the_assist_footer(window: OverlayWindow, bus: EventBus) -> None:
    _start(bus, job_id=7, kind="ask")
    bus.publish(AssistStartedEvent(job_id=8, kind="suggest", question="screen"))
    QTest.qWait(_WAIT_MS)
    before = window.assist_view.footer_label.text()
    bus.publish(ToolCallEvent(job_id=8, index=1, name="read_file", summary="x", state="running"))
    QTest.qWait(_WAIT_MS)
    assert window.assist_view.footer_label.text() == before


def test_a_long_tool_line_does_not_raise_the_assist_views_minimum_width(
    window: OverlayWindow, bus: EventBus
) -> None:
    _start(bus)
    narrow = window.assist_view.minimumSizeHint().width()
    bus.publish(ToolCallEvent(job_id=7, index=1, name="read_file", summary="x" * 120, state="running"))
    QTest.qWait(_WAIT_MS)
    assert window.assist_view.minimumSizeHint().width() == narrow


# =====================================================================================
# text the model chose is shown as typed
# =====================================================================================

_MARKUP = [
    ("<div>", ""),                 # a tag at the start of the summary: AutoText would render it as HTML
    ("echo <b>x</b>", ""),
    ("a &lt; b", ""),
    ("a < b && c > d", ""),
    ("<img src=x>", ""),
    ("x", "<b>boom</b>"),          # the failure reason is model-influenced too
]


def test_the_footer_is_plain_text(window: OverlayWindow) -> None:
    assert window.assist_view.footer_label.textFormat() == Qt.TextFormat.PlainText
    assert window.assist_view.question_label.textFormat() == Qt.TextFormat.PlainText


@pytest.mark.parametrize("summary, detail", _MARKUP)
def test_a_tool_call_with_markup_is_shown_as_typed(
    window: OverlayWindow, bus: EventBus, summary: str, detail: str
) -> None:
    _start(bus)
    footer = window.assist_view.footer_label
    bus.publish(ToolCallEvent(job_id=7, index=1, name="read_file", summary="src/a.py", state="running"))
    QTest.qWait(_WAIT_MS)
    one_line = footer.sizeHint().height()

    state = "error" if detail else "running"
    bus.publish(ToolCallEvent(job_id=7, index=2, name="search_text", summary=summary, state=state, detail=detail))
    QTest.qWait(_WAIT_MS)

    expected = f"ask - search_text {summary}" + (f" ({detail})" if detail else "…")
    assert footer.text() == expected
    assert footer.textFormat() == Qt.TextFormat.PlainText      # text() round-trips either way: this is the real check
    assert footer.sizeHint().height() == one_line              # no block element, image or bold line height


def test_a_question_with_markup_is_shown_as_typed(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(AssistStartedEvent(job_id=9, kind="ask", question="what does <b>a</b> do &amp; <img src=x>?"))
    QTest.qWait(_WAIT_MS)
    label = window.assist_view.question_label
    assert label.text() == "Q: what does <b>a</b> do &amp; <img src=x>?"
    assert label.textFormat() == Qt.TextFormat.PlainText


def test_the_toast_shows_a_map_name_as_typed(window: OverlayWindow, bus: EventBus) -> None:
    assert window.toast.textFormat() == Qt.TextFormat.PlainText
    bus.publish(CapabilityMapEvent(reason="indexed", name="<b>proj</b>", files=3, dirs=1))
    QTest.qWait(_WAIT_MS)
    assert window.toast.text() == "Capabilities map '<b>proj</b>' indexed: 3 files"
