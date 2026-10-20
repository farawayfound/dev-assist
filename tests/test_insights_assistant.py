"""The co-pilot insights + recap engine (exocortex.llm.assistant, 2026-10-20): on-demand and
background insight passes, their lane and preemption rules, the living insights document and its
InsightsUpdatedEvent, the recap kind, the dev-session notes template and the session context.

Offline: tests/test_assistant.py's FakeClient. The background-pass rules are tested by calling
`_maybe_auto_insights()` directly with a hand-moved clock while the insight-timer thread is parked
(`INSIGHT_TICK_S` patched to an hour); one test lets the real timer thread run.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from exocortex.bus import (
    AssistDoneEvent,
    AssistStartedEvent,
    EventBus,
    InsightsUpdatedEvent,
    NotesUpdatedEvent,
    Source,
)
from exocortex.config import Config
from exocortex.llm import assistant as assistant_mod
from exocortex.llm import prompts
from exocortex.llm.assistant import Assistant, _JobSlot
from exocortex.transcript import Transcript
from tests.test_assistant import FakeClient, drain_until, make_segment

ANSWER = (
    "## Questions to ask\n- Should retry() back off on 429s?\n"
    "## Blind spots & follow-ups\n- No test for an empty payload"
)


class ManualClock:
    """A session clock the test moves by hand."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture(autouse=True)
def _park_the_insight_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The insight-timer thread must not start passes on its own while a test drives
    `_maybe_auto_insights()` by hand (a test that wants it patches the tick back down)."""
    monkeypatch.setattr(assistant_mod, "INSIGHT_TICK_S", 3600.0)


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def transcript() -> Transcript:
    return Transcript()


def _say(transcript: Transcript, seg_id: int, text: str, *, me: bool = False) -> None:
    transcript.add(make_segment(
        seg_id, text=text, t_start=float(seg_id), t_end=seg_id + 1.0,
        source=Source.MIC if me else Source.SYSTEM,
        speaker_id="me" if me else "spk1", speaker_label="Me" if me else "Sam",
    ))


def _start(cfg: Config, bus: EventBus, transcript: Transcript, client: FakeClient,
           clock: Any = None, screen: Any = "") -> Assistant:
    a = Assistant(cfg, bus, client, transcript, clock=clock or time.monotonic)
    a.set_screen_provider(screen if callable(screen) else (lambda: screen))
    a.start()
    return a


def _done_of(job_id: int):
    return lambda e: isinstance(e, AssistDoneEvent) and e.job_id == job_id


# --------------------------------------------------------------------------------------------
# on demand
# --------------------------------------------------------------------------------------------


def test_insights_now_publishes_the_lifecycle_and_updates_the_living_document(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg = Config()
    cfg.assist.session_context = "billing service"
    _say(transcript, 1, "We need per-tenant rate limiting on the export endpoint.")
    client = FakeClient(token_batches=[[ANSWER]])
    sub = bus.subscribe()
    a = _start(cfg, bus, transcript, client, screen="def export(tenant):\n    ...")
    try:
        job_id = a.insights_now()
        events = drain_until(sub, _done_of(job_id))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        updated = next(e for e in events if isinstance(e, InsightsUpdatedEvent))
        done = events[-1]
        assert started.kind == "insights" and started.job_id == job_id
        assert started.question == "Questions & blind spots"
        assert events.index(updated) < events.index(done)   # the pane holds the lists before the stream ends
        assert updated.questions == "- Should retry() back off on 429s?"
        assert updated.blindspots == "- No test for an empty payload"
        assert updated.auto is False and not done.cancelled
        assert a.insights == {"questions": updated.questions, "blindspots": updated.blindspots}
        assert a.answers == [] and a.suggestions == []       # a living document, never an answer
        call = client.calls[0]
        assert "The user is the DRIVER" in call["prompt"]
        assert "Session context (what the team is working on): billing service" in call["prompt"]
        assert "per-tenant rate limiting" in call["prompt"] and "def export(tenant):" in call["prompt"]
        assert "</previous>" not in call["prompt"]           # no <previous> block: nothing to keep yet
        assert call["max_tokens"] == cfg.llm.insights_max_tokens
        assert a.insights_markdown.startswith("## Questions to ask\n- Should retry()")
    finally:
        a.stop()


def test_a_refresh_sees_the_current_list_and_a_one_section_refresh_keeps_the_other(
    bus: EventBus, transcript: Transcript
) -> None:
    _say(transcript, 1, "The limiter resets every minute.")
    client = FakeClient(token_batches=[[ANSWER], ["## Blind spots & follow-ups\n- Retry storm when the limiter resets"]])
    sub = bus.subscribe()
    a = _start(Config(), bus, transcript, client)
    try:
        drain_until(sub, _done_of(a.insights_now()))
        job_id = a.insights_now(("blindspots",))
        events = drain_until(sub, _done_of(job_id))
        started = next(e for e in events if isinstance(e, AssistStartedEvent) and e.job_id == job_id)
        updated = next(e for e in events if isinstance(e, InsightsUpdatedEvent))
        assert started.question == "Blind spots & follow-ups"
        assert updated.questions == "- Should retry() back off on 429s?"      # untouched
        assert updated.blindspots == "- Retry storm when the limiter resets"
        second = client.calls[1]["prompt"]
        assert "<previous>\n## Blind spots & follow-ups\n- No test for an empty payload\n</previous>" in second
        assert "## Questions to ask" not in second.split("<previous>", 1)[1]   # only the refreshed section is sent
        assert "Update the blind spots & follow-ups now" in second
    finally:
        a.stop()


def test_nothing_to_add_keeps_the_document_and_publishes_no_update(bus: EventBus, transcript: Transcript) -> None:
    _say(transcript, 1, "Let's keep going.")
    client = FakeClient(token_batches=[[ANSWER], [prompts.SKIP_MARKER]])
    sub = bus.subscribe()
    a = _start(Config(), bus, transcript, client)
    try:
        drain_until(sub, _done_of(a.insights_now()))
        job_id = a.insights_now()
        events = drain_until(sub, _done_of(job_id))
        assert not any(isinstance(e, InsightsUpdatedEvent) for e in events)
        assert events[-1].skipped
        assert a.insights["questions"] == "- Should retry() back off on 429s?"
    finally:
        a.stop()


def test_none_placeholders_clear_a_section_whose_items_were_all_answered(bus: EventBus, transcript: Transcript) -> None:
    _say(transcript, 1, "Yes, retry backs off. Done.")
    client = FakeClient(token_batches=[[ANSWER], ["## Questions to ask\n- (none)\n## Blind spots & follow-ups\n- (none)"]])
    sub = bus.subscribe()
    a = _start(Config(), bus, transcript, client)
    try:
        drain_until(sub, _done_of(a.insights_now()))
        events = drain_until(sub, _done_of(a.insights_now()))
        updated = next(e for e in events if isinstance(e, InsightsUpdatedEvent))
        assert updated.questions == "" and updated.blindspots == ""
        assert a.insights_markdown == "## Questions to ask\n- (none)\n\n## Blind spots & follow-ups\n- (none)"
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# background passes
# --------------------------------------------------------------------------------------------


def test_a_background_pass_waits_for_the_interval_and_enough_new_speech(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.assist.insight_interval_s = 100
    cfg.assist.insight_min_new_words = 5
    clock = ManualClock()
    client = FakeClient(token_batches=[[ANSWER]])
    sub = bus.subscribe()
    a = _start(cfg, bus, transcript, client, clock=clock)
    try:
        _say(transcript, 1, "one two three four five six seven")
        assert a._maybe_auto_insights() is False            # start(): one full interval first
        clock.advance(101)
        assert a._maybe_auto_insights() is True
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert next(e for e in events if isinstance(e, InsightsUpdatedEvent)).auto is True
        clock.advance(101)
        assert a._maybe_auto_insights() is False            # nothing new since that pass
        _say(transcript, 2, "too few")
        assert a._maybe_auto_insights() is False            # 2 new words < 5
        _say(transcript, 3, "and now there are plenty more")
        assert a._maybe_auto_insights() is True
    finally:
        a.stop()


def test_a_background_pass_runs_on_a_substantial_screen_change_alone(bus: EventBus, transcript: Transcript) -> None:
    """The driver writing code in silence still gets blind spots on it."""
    cfg = Config()
    cfg.assist.insight_interval_s = 50
    cfg.assist.insight_min_new_words = 50
    clock = ManualClock()
    screen = {"text": "\n".join(f"line {i} of the old code" for i in range(8))}
    client = FakeClient(token_batches=[[ANSWER]])
    sub = bus.subscribe()
    a = _start(cfg, bus, transcript, client, clock=clock, screen=lambda: screen["text"])
    try:
        clock.advance(51)
        assert a._maybe_auto_insights() is True             # nothing seen yet, 8 lines on screen
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        clock.advance(51)
        assert a._maybe_auto_insights() is False            # same screen, no speech
        screen["text"] = "\n".join(f"brand new function {i} here" for i in range(8))
        assert a._maybe_auto_insights() is True
    finally:
        a.stop()


def test_background_passes_are_held_back_while_paused_manual_completion_or_switched_off(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg = Config()
    cfg.assist.insight_interval_s = 10
    cfg.assist.insight_min_new_words = 1
    clock = ManualClock()
    client = FakeClient(token_batches=[[ANSWER]])
    sub = bus.subscribe()
    a = _start(cfg, bus, transcript, client, clock=clock)
    try:
        _say(transcript, 1, "We should cache the tenant lookup.")
        clock.advance(11)
        a.set_paused(True)
        assert a._maybe_auto_insights() is False
        a.set_paused(False)
        cfg.handsfree.manual = True
        assert a._maybe_auto_insights() is False
        cfg.handsfree.manual = False
        a.set_auto_suppressed(True)                          # completion mode
        assert a._maybe_auto_insights() is False
        a.set_auto_suppressed(False)
        cfg.assist.suggest_questions = cfg.assist.flag_blind_spots = False
        assert a._maybe_auto_insights() is False
        cfg.assist.suggest_questions = True
        cfg.assist.insight_interval_s = 0                    # on demand only
        assert a._maybe_auto_insights() is False
        cfg.assist.insight_interval_s = 10
        assert a._maybe_auto_insights() is True
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        assert events[-1].question == "Questions to ask"     # only the enabled section
    finally:
        a.stop()


def test_a_background_pass_yields_to_any_other_job_and_is_retried_sooner(
    bus: EventBus, transcript: Transcript, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(assistant_mod, "INSIGHT_RETRY_S", 5.0)
    cfg = Config()
    cfg.assist.insight_interval_s = 100
    cfg.assist.insight_min_new_words = 1
    clock = ManualClock()
    client = FakeClient(token_batches=[list("x" * 300), ["code out"]], chunk_delay=0.01)
    sub = bus.subscribe()
    a = _start(cfg, bus, transcript, client, clock=clock)
    try:
        _say(transcript, 1, "Write the limiter.")
        clock.advance(101)
        assert a._maybe_auto_insights() is True
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "insights")
        insight_id = events[-1].job_id
        code_id = a.code("write it")
        events = drain_until(sub, _done_of(code_id))
        insight_done = next(e for e in events if isinstance(e, AssistDoneEvent) and e.job_id == insight_id)
        assert insight_done.cancelled
        assert not any(isinstance(e, InsightsUpdatedEvent) for e in events)
        # Retried INSIGHT_RETRY_S after it started, not a whole interval later.
        clock.advance(4)
        assert a._maybe_auto_insights() is False
        clock.advance(2)
        assert a._maybe_auto_insights() is True
    finally:
        a.stop()


def test_a_background_pass_never_replaces_a_queued_on_demand_request(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.assist.insight_interval_s = 100
    cfg.assist.insight_min_new_words = 1
    clock = ManualClock()
    client = FakeClient(token_batches=[list("y" * 60), [ANSWER]], chunk_delay=0.01)
    sub = bus.subscribe()
    a = _start(cfg, bus, transcript, client, clock=clock)
    try:
        a.text("walk me through it")                         # keeps the worker busy
        drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "text")
        manual_id = a.insights_now(("questions",))           # queued behind the walkthrough
        _say(transcript, 1, "Plenty of new words to look at here.")
        clock.advance(101)
        assert a._maybe_auto_insights() is False             # the slot holds an on-demand request
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "insights")
        assert events[-1].job_id == manual_id and events[-1].question == "Questions to ask"
    finally:
        a.stop()


def test_an_on_demand_request_supersedes_a_running_background_pass(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.assist.insight_interval_s = 100
    cfg.assist.insight_min_new_words = 1
    clock = ManualClock()
    client = FakeClient(token_batches=[list("z" * 300), [ANSWER]], chunk_delay=0.01)
    sub = bus.subscribe()
    a = _start(cfg, bus, transcript, client, clock=clock)
    try:
        _say(transcript, 1, "We changed the schema.")
        clock.advance(101)
        assert a._maybe_auto_insights() is True
        auto_id = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))[-1].job_id
        manual_id = a.insights_now()
        events = drain_until(sub, _done_of(manual_id))
        assert next(e for e in events if isinstance(e, AssistDoneEvent) and e.job_id == auto_id).cancelled
        assert next(e for e in events if isinstance(e, InsightsUpdatedEvent)).auto is False
    finally:
        a.stop()


def test_the_insight_timer_thread_starts_background_passes_by_itself(
    bus: EventBus, transcript: Transcript, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(assistant_mod, "INSIGHT_TICK_S", 0.05)
    cfg = Config()
    cfg.assist.insight_interval_s = 1
    cfg.assist.insight_min_new_words = 1
    clock = ManualClock()
    client = FakeClient(token_batches=[[ANSWER]])
    sub = bus.subscribe()
    a = _start(cfg, bus, transcript, client, clock=clock)
    try:
        _say(transcript, 1, "The cache key must include the tenant.")
        clock.advance(2)
        events = drain_until(sub, lambda e: isinstance(e, InsightsUpdatedEvent), timeout=5.0)
        assert events[-1].auto is True and events[-1].questions
    finally:
        a.stop()


def test_job_slot_put_if_only_replaces_what_the_predicate_allows() -> None:
    slot: _JobSlot[str] = _JobSlot()
    assert slot.put_if("a", lambda queued: False)            # empty slot: stored
    assert not slot.put_if("b", lambda queued: queued == "x")
    assert slot.put_if("c", lambda queued: queued == "a")
    assert slot.try_get() == "c"
    slot.close()
    assert not slot.put_if("d", lambda queued: True)


# --------------------------------------------------------------------------------------------
# recap, reset, notes template, session context
# --------------------------------------------------------------------------------------------


def test_recap_is_a_manual_answer_built_from_notes_transcript_and_screen(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["## Summary\nParser rewrite."], ["**Goal:** ship the parser"]])
    sub = bus.subscribe()
    a = _start(cfg, bus, transcript, client, screen="parser.py")
    try:
        _say(transcript, 1, "We rewrote the parser today and it handles nested quotes now.")
        a.generate_notes(force=True)
        drain_until(sub, lambda e: isinstance(e, NotesUpdatedEvent))
        a.set_paused(True)                                   # manual: works while paused
        job_id = a.recap()
        events = drain_until(sub, _done_of(job_id))
        started = next(e for e in events if isinstance(e, AssistStartedEvent) and e.job_id == job_id)
        assert started.kind == "recap" and started.question == "Recap of the session so far"
        call = client.calls[-1]
        assert "<notes>\n## Summary\nParser rewrite.\n</notes>" in call["prompt"]
        assert "nested quotes" in call["prompt"] and "parser.py" in call["prompt"] and "**Goal:**" in call["prompt"]
        assert call["max_tokens"] == cfg.llm.recap_max_tokens
        assert a.answers[-1][1:] == ("Recap of the session so far", "**Goal:** ship the parser")
    finally:
        a.stop()


def test_reset_clears_the_insights(bus: EventBus, transcript: Transcript) -> None:
    _say(transcript, 1, "Ship it.")
    client = FakeClient(token_batches=[[ANSWER]])
    sub = bus.subscribe()
    a = _start(Config(), bus, transcript, client)
    try:
        drain_until(sub, _done_of(a.insights_now()))
        assert a.insights
        a.reset()
        assert a.insights == {} and a.insights_markdown == ""
    finally:
        a.stop()


def test_notes_use_the_dev_template_by_default(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    _say(transcript, 1, "We fixed the race in the cache warmer.")
    client = FakeClient(token_batches=[["## Summary\nok"], ["## Summary\nok"]])
    sub = bus.subscribe()
    a = _start(cfg, bus, transcript, client)
    try:
        a.generate_notes(force=True)
        drain_until(sub, lambda e: isinstance(e, NotesUpdatedEvent))
        assert "working session of software engineers" in client.calls[0]["prompt"]
        cfg.notes.template = "meeting"
        a.generate_notes(force=True)
        drain_until(sub, lambda e: isinstance(e, NotesUpdatedEvent))
        assert "working session of software engineers" not in client.calls[1]["prompt"]
    finally:
        a.stop()


def test_the_session_context_reaches_answers_too(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.assist.session_context = "payments API"
    client = FakeClient(token_batches=[["30 seconds."]])
    sub = bus.subscribe()
    a = _start(cfg, bus, transcript, client)
    try:
        a.ask("what's our timeout?")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert "Session context (what the team is working on): payments API" in client.calls[0]["prompt"]
    finally:
        a.stop()
