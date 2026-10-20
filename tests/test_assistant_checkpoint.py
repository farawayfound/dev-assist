"""Offline tests for checkpoints (`Assistant.checkpoint`) and the "code" job (the Code button):
the checkpoint-scoped transcript/answers window, lane/cancel semantics, the plan being dropped,
"code" building on the previous code answer since the checkpoint, and `reset()` clearing it all.

See tests/test_assistant.py and tests/test_assistant_explain.py for the fake client / scripted
streaming pattern reused here (`FakeClient`/`drain_until`/`make_segment` are imported from the
former; `ScriptedClient`/`chunks` from the latter for the explanations-mode plan tests).
"""
from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from exocortex.bus import (
    AssistDoneEvent,
    AssistStartedEvent,
    CheckpointEvent,
    EventBus,
    drain,
)
from exocortex.config import Config
from exocortex.llm import assistant as assistant_mod
from exocortex.llm.assistant import Assistant, _AssistJob
from exocortex.transcript import Transcript
from tests.test_assistant import FakeClient, drain_until, make_segment


def started(events: list[Any], kind: str) -> list[AssistStartedEvent]:
    return [e for e in events if isinstance(e, AssistStartedEvent) and e.kind == kind]


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def transcript() -> Transcript:
    return Transcript()


# --------------------------------------------------------------------------------------------
# _CANCELS / _detect_choices: the static facts checkpoints/code rely on
# --------------------------------------------------------------------------------------------


def test_cancels_dict_covers_code_both_ways() -> None:
    assert assistant_mod._CANCELS["code"] == ("suggest", "solve", "code", "explain")
    assert "code" in assistant_mod._CANCELS["solve"]
    assert "code" in assistant_mod._CANCELS["suggest"]


def test_detect_choices_never_fires_for_code() -> None:
    job = _AssistJob(id=1, kind="code", question="the task being discussed", cancel=threading.Event())
    assert Assistant._detect_choices(job, "1. 4\n2. 6\n3. 7") == (None, False)


# --------------------------------------------------------------------------------------------
# checkpoint(): the event
# --------------------------------------------------------------------------------------------


def test_checkpoint_publishes_event_with_the_last_segment_id(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient()
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    a.start()
    try:
        transcript.add(make_segment(1, text="hello", t_end=1.0))
        transcript.add(make_segment(2, text="world", t_end=2.0))
        a.checkpoint()
        events = drain_until(sub, lambda e: isinstance(e, CheckpointEvent))
        ev = next(e for e in events if isinstance(e, CheckpointEvent))
        assert ev.segment_id == 2
    finally:
        a.stop()


def test_checkpoint_on_empty_transcript_has_none_segment_id(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient()
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    a.start()
    try:
        a.checkpoint()
        events = drain_until(sub, lambda e: isinstance(e, CheckpointEvent))
        ev = next(e for e in events if isinstance(e, CheckpointEvent))
        assert ev.segment_id is None
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# checkpoint(): lane/cancel semantics -- cancels solve/code/explain/complete, never the main lane
# --------------------------------------------------------------------------------------------


def test_checkpoint_cancels_a_running_solve_and_drops_a_queued_explain(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[list("abcdefgh")], chunk_delay=0.05)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some task")
    sub = bus.subscribe()
    a.start()
    try:
        a.solve("")
        started_events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "solve")
        solve_id = next(e for e in started_events if isinstance(e, AssistStartedEvent)).job_id
        time.sleep(0.06)                # let a couple of chunks stream
        a.explain_now()                 # queued in the explain lane, behind the running solve
        a.checkpoint()                  # must cancel the solve AND drop the queued explain

        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == solve_id)
        solve_done = next(e for e in events if isinstance(e, AssistDoneEvent) and e.job_id == solve_id)
        assert solve_done.cancelled is True

        time.sleep(0.2)                 # give the worker time; the explain job must never start
        assert not any(isinstance(e, AssistStartedEvent) and e.kind == "explain" for e in drain(sub))
    finally:
        a.stop()


def test_checkpoint_never_touches_a_running_main_lane_job(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[list("abcdefgh")], chunk_delay=0.05)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    a.start()
    try:
        a.ask("slow question")
        started_events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "ask")
        ask_id = next(e for e in started_events if isinstance(e, AssistStartedEvent)).job_id
        time.sleep(0.06)
        a.checkpoint()

        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == ask_id)
        done = next(e for e in events if isinstance(e, AssistDoneEvent) and e.job_id == ask_id)
        assert done.cancelled is False
        assert done.text == "abcdefgh"
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# checkpoint(): the transcript/answers window
# --------------------------------------------------------------------------------------------


def test_solve_after_checkpoint_sees_only_segments_since_it(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["ok"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some task")
    sub = bus.subscribe()
    a.start()
    try:
        transcript.add(make_segment(1, text="OLD before checkpoint", t_end=1.0))
        a.checkpoint()
        drain_until(sub, lambda e: isinstance(e, CheckpointEvent))
        transcript.add(make_segment(2, text="NEW after checkpoint", t_end=2.0))

        a.solve("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        prompt = client.calls[-1]["prompt"]
        assert "NEW after checkpoint" in prompt
        assert "OLD before checkpoint" not in prompt
    finally:
        a.stop()


def test_ask_after_checkpoint_still_sees_the_whole_transcript(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["ok"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    a.start()
    try:
        transcript.add(make_segment(1, text="OLD before checkpoint", t_end=1.0))
        a.checkpoint()
        drain_until(sub, lambda e: isinstance(e, CheckpointEvent))
        transcript.add(make_segment(2, text="NEW after checkpoint", t_end=2.0))

        a.ask("what happened?")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        prompt = client.calls[-1]["prompt"]
        assert "OLD before checkpoint" in prompt and "NEW after checkpoint" in prompt
    finally:
        a.stop()


def test_complete_after_checkpoint_excludes_older_answers_and_transcript(
    bus: EventBus, transcript: Transcript
) -> None:
    from tests.test_assistant_explain import ScriptedClient

    cfg = Config()
    # A "complete" job always passes `extra_body` (see Assistant._stream_text): unlike the plain
    # FakeClient above, ScriptedClient's stream_completion (test_assistant_complete.FakeClient)
    # accepts it, and scripts a different token list per call, in order.
    client = ScriptedClient([["old answer"], ["new answer"], ["ld"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    a.start()
    try:
        transcript.add(make_segment(1, text="OLD spoken line", t_end=1.0))
        a.ask("old question")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        a.checkpoint()
        drain_until(sub, lambda e: isinstance(e, CheckpointEvent))
        transcript.add(make_segment(2, text="NEW spoken line", t_end=2.0))
        a.ask("new question")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))

        a.complete("hello wor", "", "field")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        prompt = client.calls[-1]["prompt"]
        assert "new question" in prompt and "new answer" in prompt
        assert "old question" not in prompt and "old answer" not in prompt
        assert "NEW spoken line" in prompt and "OLD spoken line" not in prompt
    finally:
        a.stop()


def test_checkpoint_racing_a_job_already_past_its_cancel_check_does_not_resurrect_the_plan(
    bus: EventBus, transcript: Transcript, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `_run_assist_job` checks `job.cancel.is_set()` once and, if False, calls
    `_begin_plan` a few statements later with no re-check. A `checkpoint()` landing in that exact
    window (job already past the check; not yet at `_begin_plan`) used to have its plan-clear
    silently overwritten by the already-in-flight job's `_begin_plan`, resurrecting a plan keyed
    to the job the checkpoint just cancelled. Forces that interleaving deterministically by
    pausing inside `_begin_plan` (after the caller's own cancel check has already passed) while
    `checkpoint()` runs on another thread."""
    from tests.test_assistant_explain import ScriptedClient, chunks
    from tests.test_plan import ANSWER

    cfg = Config()
    cfg.explain.enabled = True
    client = ScriptedClient([chunks(ANSWER)])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some task")
    sub = bus.subscribe()

    reached_begin_plan = threading.Event()
    checkpoint_done = threading.Event()
    real_begin_plan = Assistant._begin_plan

    def paused_begin_plan(self: Assistant, job: _AssistJob, screen_text: str) -> None:
        reached_begin_plan.set()
        checkpoint_done.wait(timeout=5.0)
        real_begin_plan(self, job, screen_text)

    monkeypatch.setattr(Assistant, "_begin_plan", paused_begin_plan)
    a.start()
    try:
        a.solve("")
        started_events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        solve_id = next(e for e in started_events if isinstance(e, AssistStartedEvent)).job_id
        assert reached_begin_plan.wait(timeout=5.0)

        a.checkpoint()                  # races in between the cancel check and _begin_plan
        checkpoint_done.set()

        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == solve_id)
        assert a.plan() is None         # must stay dropped -- never resurrected by the raced job
    finally:
        a.stop()


def test_checkpoint_racing_a_just_finished_code_job_does_not_repopulate_code_answers(
    bus: EventBus, transcript: Transcript, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `_run_assist_job` captures `cancelled = job.cancel.is_set()` once, right after
    the LLM stream finishes, and reuses that stale value several statements later to decide
    whether to call `_append_code_answer`. A `checkpoint()` landing in that window (the job's
    stream already finished; the answer not yet appended) used to re-populate the very
    `_code_answers` list checkpoint() just cleared, handing the next "code" job a stale, unrelated
    <previous> block. Forces the interleaving deterministically by pausing inside
    `_append_code_answer` while `checkpoint()` runs on another thread."""
    cfg = Config()
    client = FakeClient(token_batches=[["first code answer"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()

    reached_append = threading.Event()
    checkpoint_done = threading.Event()
    append_returned = threading.Event()
    real_append = Assistant._append_code_answer

    def paused_append(self: Assistant, job: _AssistJob, entry: Any) -> None:
        reached_append.set()
        checkpoint_done.wait(timeout=5.0)
        try:
            real_append(self, job, entry)
        finally:
            append_returned.set()   # only THIS call's own pause; the 2nd job's call sets it again

    monkeypatch.setattr(Assistant, "_append_code_answer", paused_append)
    a.start()
    try:
        a.code("")
        started_events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        code_id = next(e for e in started_events if isinstance(e, AssistStartedEvent)).job_id
        assert reached_append.wait(timeout=5.0)

        a.checkpoint()                  # races in right as the code job finished streaming
        checkpoint_done.set()
        assert append_returned.wait(timeout=5.0)   # let the raced job's own append attempt finish

        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == code_id)
        assert a._code_answers == []    # must stay empty -- never repopulated by the raced job

        # The next "code" job in this (new) checkpoint must not see the stale answer either.
        append_returned.clear()
        a.code("")
        assert append_returned.wait(timeout=5.0)
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id != code_id)
        # `system_code`'s own instructions always mention "<previous>" (describing what it is WHEN
        # present), so check for the stale answer's actual text rather than that bare substring.
        assert "first code answer" not in client.calls[-1]["prompt"]
    finally:
        a.stop()


def test_checkpoint_drops_the_plan(bus: EventBus, transcript: Transcript) -> None:
    from tests.test_assistant_explain import ScriptedClient, chunks
    from tests.test_plan import ANSWER

    cfg = Config()
    cfg.explain.enabled = True
    client = ScriptedClient([chunks(ANSWER)])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some task")
    sub = bus.subscribe()
    a.start()
    try:
        a.solve("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text)
        assert a.plan() is not None

        a.checkpoint()
        drain_until(sub, lambda e: isinstance(e, CheckpointEvent))
        assert a.plan() is None
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# code(): side lane, manual-while-paused, stored as an answer, builds on the previous code
# --------------------------------------------------------------------------------------------


def test_code_is_side_lane_and_does_not_cancel_a_running_ask(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[list("a" * 20), ["code answer"]], chunk_delay=0.01)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.ask("slow question")
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "ask")
        a.code("")                      # queued behind the running ask (side lane): must not cancel it
        events += drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == "code answer", timeout=6.0)
        kinds = [e.kind for e in events if isinstance(e, AssistStartedEvent)]
        assert kinds == ["ask", "code"]
        ask_done = next(e for e in events if isinstance(e, AssistDoneEvent) and e.text != "code answer")
        assert ask_done.cancelled is False
    finally:
        a.stop()


def test_code_not_refused_while_paused_unlike_solve(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["ok"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.set_paused(True)
        a.code("")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert started(events, "code")

        a.solve("")                     # contrast: solve() IS refused while paused
        time.sleep(0.2)
        assert not started(drain(sub), "solve")
    finally:
        a.stop()


def test_code_stored_as_answer_with_kind_code_in_events(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["- printed code"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.code("")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert started(events, "code")
        assert a.answers and a.answers[-1][2] == "- printed code"
    finally:
        a.stop()


def test_code_prompt_includes_the_previous_code_answer(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["```python\ndef f(): pass\n```"], ["ok"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.code("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        a.code("now handle duplicates")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        prompt = client.calls[-1]["prompt"]
        assert "<previous>" in prompt and "def f(): pass" in prompt
        assert "now handle duplicates" in prompt
    finally:
        a.stop()


def test_code_is_plan_when_explanations_are_on(bus: EventBus, transcript: Transcript) -> None:
    from tests.test_assistant_explain import ScriptedClient, chunks
    from tests.test_plan import ANSWER, SOLUTION

    cfg = Config()
    cfg.explain.enabled = True
    client = ScriptedClient([chunks(ANSWER)])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.code("")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text)
        assert started(events, "code")
        view = a.plan()
        assert view is not None and view.done and view.solution == SOLUTION
    finally:
        a.stop()


def test_solve_cancels_a_running_code(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[list("a" * 20), ["solved"]], chunk_delay=0.01)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.code("")
        drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "code")
        time.sleep(0.03)
        a.solve("")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == "solved", timeout=6.0)
        assert any(isinstance(e, AssistDoneEvent) and e.cancelled for e in events)
    finally:
        a.stop()


def test_suggest_cancels_a_running_code(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[list("a" * 20), ["suggested"]], chunk_delay=0.01)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.code("")
        drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "code")
        time.sleep(0.03)
        a.suggest_now()
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == "suggested", timeout=6.0)
        assert any(isinstance(e, AssistDoneEvent) and e.cancelled for e in events)
    finally:
        a.stop()


def test_code_updates_the_auto_suggest_throttle_state(bus: EventBus, transcript: Transcript) -> None:
    """`_record_suggest_start` must run for "code" too, so auto-solve does not immediately
    re-solve the same screen right after a Code press."""
    cfg = Config()
    client = FakeClient(token_batches=[["ok"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "line one\nline two")
    sub = bus.subscribe()
    a.start()
    try:
        a.code("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert a._last_suggest_time > float("-inf")
        assert a._last_suggested_lines == {"line one", "line two"}
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# reset() clears the checkpoint state
# --------------------------------------------------------------------------------------------


def test_reset_clears_checkpoint_state(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["ok"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some task")
    sub = bus.subscribe()
    a.start()
    try:
        transcript.add(make_segment(1, text="OLD", t_end=1.0))
        a.checkpoint()
        drain_until(sub, lambda e: isinstance(e, CheckpointEvent))
        a.reset()
        transcript.add(make_segment(2, text="NEW", t_end=2.0))

        a.solve("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        prompt = client.calls[-1]["prompt"]
        assert "OLD" in prompt and "NEW" in prompt
    finally:
        a.stop()
