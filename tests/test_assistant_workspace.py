"""Offline tests for the Assistant side of the 2026-10-29 multi-file work: a Code / Text job with
`read_now` has the screen read afresh before its prompt is built (the refresh hook runs before the
provider is read; `[screen].read_on_shortcut` and `read_now=False` skip it; no other kind refreshes);
the task's workspace is fed from the screen events and the job's prompt carries the OTHER views as
<workspace> while the current one is the <screen>; a checkpoint and a reset clear it; the hands-free
context provider is handed the fresh screen; and a Code answer is normalised to code only
(`[assist].code_only`). Reuses tests/test_assistant.py's FakeClient and helpers."""
from __future__ import annotations

import time
from typing import Any

import pytest

from exocortex.bus import AssistDoneEvent, AssistStartedEvent, EventBus, ScreenTextEvent
from exocortex.config import Config
from exocortex.llm.assistant import Assistant
from exocortex.transcript import Transcript
from tests.test_assistant import FakeClient, drain_until

REPO_PY = "\n".join([
    "EXPLORER", "src", "repo.py", "models.py",
    "class UserRepository:",
    "    def __init__(self, session: Session) -> None:",
    "        self.session = session",
    "    def get_user(self, user_id: int) -> User:",
    "        return self.session.query(User).get(user_id)",
])
MODELS_PY = "\n".join([
    "EXPLORER", "src", "repo.py", "models.py",
    "from dataclasses import dataclass",
    "@dataclass",
    "class User:",
    "    user_id: int",
    "    email_address: str",
])


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def transcript() -> Transcript:
    return Transcript()


def wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def sent_text(client: FakeClient) -> str:
    """Everything the last request carried (the rendered prompt, or the chat messages' text)."""
    call = client.calls[-1]
    if "prompt" in call:
        return str(call["prompt"])
    return "\n".join(str(m.get("content", "")) for m in call["messages"])


def screen_event(text: str, *, title: str = "", process: str = "", changed: bool = True) -> ScreenTextEvent:
    lines = tuple(text.splitlines())
    return ScreenTextEvent(mode="auto", source_name="Fake Window", rect=(0, 0, 800, 600), text=text, lines=lines,
                           changed=changed, ocr_ms=5.0, captured_at=0.0, engine="fake", accurate=False,
                           window_title=title, window_process=process)


def make(bus: EventBus, transcript: Transcript, cfg: Config | None = None, batches: list[list[Any]] | None = None):
    cfg = cfg or Config()
    # the screen events these tests publish must not start automatic solves / screen answers
    cfg.screen.auto_suggest = False
    cfg.screen.answer_screen_questions = False
    client = FakeClient(token_batches=batches or [["x = 1"], ["x = 2"], ["x = 3"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    return cfg, client, a


def run_job(bus: EventBus, a: Assistant, start) -> AssistDoneEvent:
    """Start a job and return ITS Done event (matched by the id `start()` returns; any Done when it
    returns None, as `suggest_now` does)."""
    sub = bus.subscribe()
    job_id = start()
    events = drain_until(
        sub, lambda e: isinstance(e, AssistDoneEvent) and (job_id is None or e.job_id == job_id), timeout=5.0,
    )
    return next(e for e in reversed(events) if isinstance(e, AssistDoneEvent))


def workspace_body(sent: str) -> str:
    """The body of the user turn's <workspace> block (the system prompt mentions the tag too)."""
    return sent.rsplit("<workspace>\n", 1)[1].split("\n</workspace>")[0]


# =====================================================================================
# read-on-shortcut
# =====================================================================================
def test_a_code_job_reads_the_screen_afresh_before_reading_the_provider(bus: EventBus, transcript: Transcript) -> None:
    cfg, client, a = make(bus, transcript)
    order: list[str] = []
    screen = {"text": "the old pass"}

    def refresh() -> bool:
        order.append("refresh")
        screen["text"] = "the file the user is on now"
        return True

    a.set_screen_provider(lambda: (order.append("provider"), screen["text"])[1])
    a.set_screen_refresh(refresh)
    a.start()
    try:
        run_job(bus, a, lambda: a.code(""))
        assert order[:2] == ["refresh", "provider"]
        assert a.last_screen_text == "the file the user is on now"
        assert "the file the user is on now" in sent_text(client)
        run_job(bus, a, lambda: a.text(""))
        assert order.count("refresh") == 2
    finally:
        a.stop()


def test_read_now_false_the_setting_off_or_a_failed_read_leave_the_latest_text_in_use(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg, client, a = make(bus, transcript)
    refreshes: list[int] = []
    a.set_screen_provider(lambda: "latest text")
    a.set_screen_refresh(lambda: (refreshes.append(1), False)[1])
    a.start()
    try:
        run_job(bus, a, lambda: a.code("", read_now=False))
        assert refreshes == []
        done = run_job(bus, a, lambda: a.code(""))          # a read that reports failure: logged, text kept
        assert refreshes == [1] and not done.cancelled
        assert a.last_screen_text == "latest text"
        cfg.screen.read_on_shortcut = False
        run_job(bus, a, lambda: a.code(""))
        assert refreshes == [1]
    finally:
        a.stop()


def test_a_refresh_that_raises_never_breaks_the_job(bus: EventBus, transcript: Transcript) -> None:
    cfg, client, a = make(bus, transcript)
    a.set_screen_provider(lambda: "latest text")

    def boom() -> bool:
        raise RuntimeError("reader gone")

    a.set_screen_refresh(boom)
    a.start()
    try:
        done = run_job(bus, a, lambda: a.code(""))
        assert not done.cancelled and done.text.strip()
        assert a.last_screen_text == "latest text"
    finally:
        a.stop()


def test_other_kinds_never_refresh(bus: EventBus, transcript: Transcript) -> None:
    cfg, client, a = make(bus, transcript)
    refreshes: list[int] = []
    a.set_screen_provider(lambda: "some screen")
    a.set_screen_refresh(lambda: (refreshes.append(1), True)[1])
    a.start()
    try:
        run_job(bus, a, lambda: a.suggest_now())
        run_job(bus, a, lambda: a.ask("what is this?"))
        assert refreshes == []
    finally:
        a.stop()


# =====================================================================================
# the workspace
# =====================================================================================
def test_the_code_prompt_carries_the_other_views_as_workspace_and_the_current_as_screen(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg, client, a = make(bus, transcript)
    current = {"text": REPO_PY}
    a.set_screen_provider(lambda: current["text"])
    a.start()
    try:
        bus.publish(screen_event(REPO_PY, title="repo.py - myproj - Visual Studio Code", process="Code.exe"))
        assert wait_until(lambda: len(a.workspace.views()) == 1)
        bus.publish(screen_event(MODELS_PY, title="models.py - myproj - Visual Studio Code", process="Code.exe"))
        assert wait_until(lambda: len(a.workspace.views()) == 2)
        current["text"] = MODELS_PY                                 # the user is in models.py now
        run_job(bus, a, lambda: a.code(""))
        sent = sent_text(client)
        assert "<workspace>\n[repo.py]\n" in sent
        body = workspace_body(sent)
        assert body.startswith("[repo.py]\n") and "class UserRepository:" in body
        assert "[models.py]" not in sent                             # the current view is the <screen>
        user_turn = sent.rsplit("<|im_start|>user", 1)[1]            # the system rule mentions both tags
        assert user_turn.index("\n</workspace>") < user_turn.index("<screen>")
        assert "email_address: str" in user_turn.split("<screen>")[1]
        assert "Write the code for the file on screen, consistent with them." in sent
        assert "<workspace> (when present) holds the OTHER files" in sent  # the system rule (workspace on)
        assert a.workspace.current is not None and a.workspace.current.file == "models.py"
        # Text gets the same block with its own ask
        run_job(bus, a, lambda: a.text(""))
        sent = sent_text(client)
        assert "<workspace>\n[repo.py]\n" in sent and "Walk through what the file on screen needs" in sent
    finally:
        a.stop()


def test_a_single_view_renders_no_workspace_block(bus: EventBus, transcript: Transcript) -> None:
    cfg, client, a = make(bus, transcript)
    a.set_screen_provider(lambda: REPO_PY)
    a.start()
    try:
        run_job(bus, a, lambda: a.code(""))
        sent = sent_text(client)
        assert "<workspace>\n" not in sent and "<screen>" in sent    # the tag appears in the rule only
        assert len(a.workspace.views()) == 1                          # the job's own screen was sorted in
    finally:
        a.stop()


def test_the_workspace_setting_off_feeds_nothing_and_renders_nothing(bus: EventBus, transcript: Transcript) -> None:
    cfg, client, a = make(bus, transcript)
    cfg.screen.workspace = False
    a.set_screen_provider(lambda: MODELS_PY)
    a.start()
    try:
        bus.publish(screen_event(REPO_PY, title="repo.py - myproj - Visual Studio Code"))
        time.sleep(0.2)
        assert a.workspace.views() == []
        run_job(bus, a, lambda: a.code(""))
        sent = sent_text(client)
        assert "<workspace>" not in sent                               # neither the block nor the rule
    finally:
        a.stop()


def test_checkpoint_and_reset_clear_the_workspace(bus: EventBus, transcript: Transcript) -> None:
    cfg, client, a = make(bus, transcript)
    a.start()
    try:
        bus.publish(screen_event(REPO_PY, title="repo.py - myproj - Visual Studio Code"))
        bus.publish(screen_event(MODELS_PY, title="models.py - myproj - Visual Studio Code"))
        assert wait_until(lambda: len(a.workspace.views()) == 2)
        a.checkpoint()
        assert a.workspace.views() == [] and a.workspace.current is None
        bus.publish(screen_event(REPO_PY, title="repo.py - myproj - Visual Studio Code"))
        assert wait_until(lambda: len(a.workspace.views()) == 1)
        a.reset()
        assert a.workspace.views() == []
    finally:
        a.stop()


def test_unchanged_events_teach_nothing_but_keep_the_title(bus: EventBus, transcript: Transcript) -> None:
    cfg, client, a = make(bus, transcript)
    a.start()
    try:
        bus.publish(screen_event(REPO_PY, title="repo.py - myproj - Visual Studio Code", changed=False))
        time.sleep(0.2)
        assert a.workspace.views() == []
        assert a._last_screen_title == "repo.py - myproj - Visual Studio Code"
    finally:
        a.stop()


# =====================================================================================
# hands-free: the context provider gets the fresh screen
# =====================================================================================
def test_the_hands_free_context_provider_is_handed_the_fresh_screen(bus: EventBus, transcript: Transcript) -> None:
    cfg, client, a = make(bus, transcript)
    seen: list[str] = []

    def provider(current: str = "") -> str:
        seen.append(current)
        return f"[earlier]\nold lines\n\n[now]\n{current}"

    a.set_screen_provider(lambda: "the fresh screen")
    a.set_hands_free(True, provider)
    a.start()
    try:
        run_job(bus, a, lambda: a.code(""))
        assert seen == ["the fresh screen"]
        assert a.last_screen_text.endswith("[now]\nthe fresh screen")
        # an older zero-argument provider keeps working
        a.set_hands_free(True, lambda: "plain cumulative context")
        run_job(bus, a, lambda: a.code(""))
        assert a.last_screen_text == "plain cumulative context"
    finally:
        a.stop()


# =====================================================================================
# code-only answers
# =====================================================================================
def test_a_code_answer_is_normalised_to_code_only(bus: EventBus, transcript: Transcript) -> None:
    messy = "Here is the solution:\n```python\nx = 1\n```\n**Time:** O(1)"
    cfg, client, a = make(bus, transcript, batches=[[messy], [messy]])
    a.set_screen_provider(lambda: "task")
    a.start()
    try:
        done = run_job(bus, a, lambda: a.code(""))
        assert done.text == "```python\nx = 1\n\n# Here is the solution:\n# Time: O(1)\n```"
        assert a.recent_code() == done.text                       # what later jobs build on is the clean code
        cfg.assist.code_only = False
        done = run_job(bus, a, lambda: a.code(""))
        assert done.text == messy
    finally:
        a.stop()


def test_a_text_answer_is_never_reformatted(bus: EventBus, transcript: Transcript) -> None:
    prose = "- restate the task\n- key insight: a dict\n- edge cases"
    cfg, client, a = make(bus, transcript, batches=[[prose]])
    a.set_screen_provider(lambda: "task")
    a.start()
    try:
        done = run_job(bus, a, lambda: a.text(""))
        assert done.text == prose
    finally:
        a.stop()


def test_started_events_still_carry_the_kind_and_question(bus: EventBus, transcript: Transcript) -> None:
    cfg, client, a = make(bus, transcript)
    a.start()
    try:
        sub = bus.subscribe()
        a.code("", read_now=False)
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        st = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert st.kind == "code" and st.question == "the task being discussed"
    finally:
        a.stop()
