"""Offline tests for the Assistant's kind "text" job (the Text button): a plain-English
walkthrough of the task discussed/shown since the last checkpoint -- the prose twin of "code".

Covers the static facts (`_CANCELS`, `_detect_choices`), the started/question/prompt shape,
checkpoint scoping and the <previous> code block, that it never feeds `_code_answers` or the
plan, pause/lane/cancel semantics (latest-wins among "text" jobs, immune to solve/code, cancelled
by checkpoint), the skip marker, and the `cfg.assist.comments` plumbing shared with solve/code/
complete (`prompts.COMMENT_LEVELS`, `prompts._comment_rule`/`_comment_rule_complete`,
`assistant_mod.COMMENT_TOKEN_HEADROOM`).

See tests/test_assistant.py and tests/test_assistant_checkpoint.py for the fake client / scripted
streaming pattern reused here (`FakeClient`/`drain_until`/`make_segment` are imported from the
former).
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
from exocortex.llm import prompts
from exocortex.llm.assistant import Assistant, _AssistJob
from exocortex.transcript import Transcript
from tests.test_assistant import FakeClient, drain_until, make_segment


def started(events: list[Any], kind: str) -> list[AssistStartedEvent]:
    return [e for e in events if isinstance(e, AssistStartedEvent) and e.kind == kind]


def dones(events: list[Any]) -> list[AssistDoneEvent]:
    return [e for e in events if isinstance(e, AssistDoneEvent)]


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def transcript() -> Transcript:
    return Transcript()


# --------------------------------------------------------------------------------------------
# _CANCELS / _detect_choices: the static facts "text" relies on
# --------------------------------------------------------------------------------------------


def test_cancels_text_is_the_odd_one_out() -> None:
    assert assistant_mod._CANCELS["text"] == ("text",)
    for kind, cancels in assistant_mod._CANCELS.items():
        if kind == "text":
            continue
        assert "text" not in cancels, f"{kind!r} must never cancel a running text job"


def test_detect_choices_never_fires_for_text() -> None:
    job = _AssistJob(id=1, kind="text", question="the task being discussed", cancel=threading.Event())
    assert Assistant._detect_choices(job, "1. 4\n2. 6\n3. 7") == (None, False)


# --------------------------------------------------------------------------------------------
# text(): started event, default/focus question, the rendered prompt, max_tokens
# --------------------------------------------------------------------------------------------


def test_text_default_question_and_with_focus(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["a walkthrough"], ["another one"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.text("")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        st = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert st.kind == "text"
        assert st.question == "the task being discussed"

        a.text("explain the DP")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        st2 = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert st2.question == "explain the DP"
        # the user turn (inside the rendered ChatML prompt) ends with the focus text -- user_text()
        # takes `focus` verbatim as the ask when one is given -- followed only by the one-sentence
        # given-code reminder ([assist].consider_given_code, on by default; 2026-10-29), right
        # before the closing <|im_end|> tag.
        prompt = client.calls[-1]["prompt"]
        assert f"explain the DP{prompts._given_code_ask('text', True)}{prompts.IM_END}" in prompt
    finally:
        a.stop()


def test_text_prompt_shape_and_max_tokens(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["ok"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.text("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        call = client.calls[-1]
        assert 'pressed "Text"' in call["prompt"]
        assert "Output the walkthrough now." in call["prompt"]
        assert call["max_tokens"] == cfg.llm.text_max_tokens
    finally:
        a.stop()


def test_text_max_tokens_follows_custom_config_value(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.llm.text_max_tokens = 555
    client = FakeClient(token_batches=[["ok"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.text("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert client.calls[-1]["max_tokens"] == 555
    finally:
        a.stop()


def test_text_prompt_includes_screen_text_even_when_use_in_answers_is_off(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg = Config()
    cfg.screen.use_in_answers = False    # gates "auto"/"manual"/"ask" only, never "text"
    client = FakeClient(token_batches=[["ok"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some very particular on-screen content xyz")
    sub = bus.subscribe()
    a.start()
    try:
        a.text("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert "some very particular on-screen content xyz" in client.calls[-1]["prompt"]
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# checkpoint scoping: "text" sees only the transcript since the checkpoint, like solve/code
# --------------------------------------------------------------------------------------------


def test_text_after_checkpoint_sees_only_segments_since_it(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["ok"], ["walkthrough"], ["asked"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some task")
    sub = bus.subscribe()
    a.start()
    try:
        transcript.add(make_segment(1, text="OLD before checkpoint", t_end=1.0))
        a.checkpoint()
        drain_until(sub, lambda e: isinstance(e, CheckpointEvent))
        transcript.add(make_segment(2, text="NEW after checkpoint", t_end=2.0))

        a.text("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        prompt = client.calls[-1]["prompt"]
        assert "NEW after checkpoint" in prompt
        assert "OLD before checkpoint" not in prompt

        # ask() is meeting-wide: it still sees the whole transcript, checkpoint or not.
        a.ask("what happened?")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        prompt = client.calls[-1]["prompt"]
        assert "OLD before checkpoint" in prompt and "NEW after checkpoint" in prompt
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# <previous>: text picks up a prior code answer (within the same checkpoint) but never feeds one
# --------------------------------------------------------------------------------------------


def test_text_prompt_carries_previous_code_answer_until_a_checkpoint(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["```python\ndef f(): pass\n```"], ["a walkthrough"], ["ok"], ["another walkthrough"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.code("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))

        a.text("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        prompt = client.calls[-1]["prompt"]
        assert "<previous>" in prompt and "def f(): pass" in prompt

        a.checkpoint()
        drain_until(sub, lambda e: isinstance(e, CheckpointEvent))

        a.text("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        prompt = client.calls[-1]["prompt"]
        # `system_text`'s own instructions always mention "<previous>" (describing what it is WHEN
        # present); check the actual stale answer's text rather than that bare substring.
        assert "def f(): pass" not in prompt
    finally:
        a.stop()


def test_text_answer_never_feeds_code_answers(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["a spoken-style walkthrough"], ["ok"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.text("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert a.answers and a.answers[-1][2] == "a spoken-style walkthrough"

        a.code("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        # `system_code`'s own instructions always mention "<previous>" (describing what it is
        # WHEN present); check the actual text never rode along instead of the bare substring.
        assert "a spoken-style walkthrough" not in client.calls[-1]["prompt"]
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# the plan: a "text" job never creates one, and never replaces/cancels one a "solve" built
# --------------------------------------------------------------------------------------------


def test_text_leaves_the_plan_alone_with_explanations_enabled(bus: EventBus, transcript: Transcript) -> None:
    from tests.test_assistant_explain import ScriptedClient, chunks
    from tests.test_plan import ANSWER, SOLUTION

    cfg = Config()
    cfg.explain.enabled = True
    client = ScriptedClient([["a walkthrough"], chunks(ANSWER), ["another walkthrough"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        # a lone "text" job must not create a plan, even with explanations on.
        a.text("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert a.plan() is None

        # a "solve" DOES build the plan.
        a.solve("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text)
        view = a.plan()
        assert view is not None and view.done and view.solution == SOLUTION

        # a following "text" must not replace or cancel that plan.
        a.text("walk me through it")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        view2 = a.plan()
        assert view2 is not None and view2.solution == SOLUTION
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# manual while paused
# --------------------------------------------------------------------------------------------


def test_text_works_while_paused(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["a walkthrough while paused"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.set_paused(True)
        a.text("")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert started(events, "text")
        done = next(e for e in events if isinstance(e, AssistDoneEvent))
        assert done.cancelled is False
        assert done.text == "a walkthrough while paused"
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# lane/cancel semantics: latest text wins, immune to solve/code, killed by checkpoint
# --------------------------------------------------------------------------------------------


def test_second_text_cancels_the_first(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[list("a" * 20), ["second text"]], chunk_delay=0.01)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.text("first")
        drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "text")
        time.sleep(0.03)
        a.text("second")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == "second text", timeout=6.0)
        done_events = dones(events)
        first_done = next(e for e in done_events if e.text != "second text")
        second_done = next(e for e in done_events if e.text == "second text")
        assert first_done.cancelled is True
        assert second_done.cancelled is False
    finally:
        a.stop()


def test_text_does_not_cancel_a_running_solve(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[list("a" * 20), ["a walkthrough"]], chunk_delay=0.01)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.solve("")
        first_batch = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "solve")
        time.sleep(0.03)
        a.text("")
        second_batch = drain_until(
            sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == "a walkthrough", timeout=6.0
        )
        events = first_batch + second_batch
        kinds = [e.kind for e in events if isinstance(e, AssistStartedEvent)]
        assert kinds == ["solve", "text"]
        solve_done = next(e for e in dones(events) if e.text != "a walkthrough")
        assert solve_done.cancelled is False
        assert solve_done.text == "a" * 20
    finally:
        a.stop()


@pytest.mark.parametrize("second_kind", ["solve", "code"])
def test_running_text_is_not_cancelled_by_solve_or_code(
    bus: EventBus, transcript: Transcript, second_kind: str
) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[list("a" * 20), ["ok"]], chunk_delay=0.01)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.text("")
        drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "text")
        time.sleep(0.03)
        getattr(a, second_kind)("")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == "ok", timeout=6.0)
        text_done = next(e for e in dones(events) if e.text != "ok")
        assert text_done.cancelled is False
        assert text_done.text == "a" * 20
    finally:
        a.stop()


def test_checkpoint_cancels_a_running_text(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[list("abcdefgh")], chunk_delay=0.05)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.text("")
        drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "text")
        time.sleep(0.06)
        a.checkpoint()
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        done = next(e for e in dones(events))
        assert done.cancelled is True
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# skip marker
# --------------------------------------------------------------------------------------------


def test_text_skip_marker_is_dropped(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[[prompts.SKIP_MARKER]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "nothing technical here")
    sub = bus.subscribe()
    a.start()
    try:
        a.text("")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        done = next(e for e in dones(events))
        assert done.cancelled is True
        assert done.text == ""
        assert a.answers == []
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# cfg.assist.comments plumbing: solve() / code() prompt + budget for every level, and "bogus"
# --------------------------------------------------------------------------------------------

_SOLVE_CODE_PHRASE = {
    "none": "no comments at all",
    "minimal": "minimal but important comments",
    "normal": "well commented",
    "detailed": "pseudo-code",
}

_COMPLETE_PHRASE = {
    "none": "never add comments",
    "minimal": "genuinely important step",
    "normal": "each logical step",
    "detailed": "most lines of code",
}


@pytest.mark.parametrize("level", prompts.COMMENT_LEVELS)
def test_comment_level_shapes_solve_and_code_prompt_and_budget(level: str) -> None:
    cfg = Config()
    cfg.assist.comments = level
    bus = EventBus()
    transcript = Transcript()
    client = FakeClient(token_batches=[["ok"], ["ok2"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        expected_tokens = cfg.llm.solve_max_tokens + assistant_mod.COMMENT_TOKEN_HEADROOM[level]

        a.solve("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        call = client.calls[-1]
        assert _SOLVE_CODE_PHRASE[level] in call["prompt"]
        assert call["max_tokens"] == expected_tokens
        if level == "detailed":
            assert "requirements" in call["prompt"] and "comment block" in call["prompt"]

        a.code("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        call = client.calls[-1]
        assert _SOLVE_CODE_PHRASE[level] in call["prompt"]
        assert call["max_tokens"] == expected_tokens
    finally:
        a.stop()


def test_unknown_comment_level_behaves_as_normal(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.assist.comments = "bogus"
    client = FakeClient(token_batches=[["ok"], ["ok2"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        expected_tokens = cfg.llm.solve_max_tokens + assistant_mod.COMMENT_TOKEN_HEADROOM["normal"]

        a.solve("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert _SOLVE_CODE_PHRASE["normal"] in client.calls[-1]["prompt"]
        assert client.calls[-1]["max_tokens"] == expected_tokens

        a.code("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert _SOLVE_CODE_PHRASE["normal"] in client.calls[-1]["prompt"]
        assert client.calls[-1]["max_tokens"] == expected_tokens
    finally:
        a.stop()


@pytest.mark.parametrize("level", prompts.COMMENT_LEVELS)
def test_comment_level_budget_adds_walkthrough_tokens_when_explaining(level: str) -> None:
    """With explanations on, a "solve"/"code" answer is a plan: the budget also gets
    `walkthrough_max_tokens` on top of the same `COMMENT_TOKEN_HEADROOM` (see `_code_budget`)."""
    cfg = Config()
    cfg.assist.comments = level
    cfg.explain.enabled = True
    bus = EventBus()
    transcript = Transcript()
    client = FakeClient(token_batches=[["ok"]])
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        expected_tokens = (
            cfg.llm.solve_max_tokens
            + cfg.explain.walkthrough_max_tokens
            + assistant_mod.COMMENT_TOKEN_HEADROOM[level]
        )
        a.solve("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert client.calls[-1]["max_tokens"] == expected_tokens
    finally:
        a.stop()


@pytest.mark.parametrize("level", prompts.COMMENT_LEVELS)
def test_complete_prompt_carries_comment_rule_for_each_level(level: str) -> None:
    from tests.test_assistant_complete import FakeClient as CompleteFakeClient

    cfg = Config()
    cfg.assist.comments = level
    bus = EventBus()
    transcript = Transcript()
    client = CompleteFakeClient()
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    a.start()
    try:
        a.complete("some prefix", "", "field")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert _COMPLETE_PHRASE[level] in client.calls[-1]["prompt"]
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# the text lane: a queued walkthrough is never overwritten by (or overwrites) a side-lane job
# --------------------------------------------------------------------------------------------


def test_text_has_its_own_lane_dispatched_after_the_side_lane_and_before_explain() -> None:
    """Review finding: every side kind shares one single-item mailbox, so a queued "text" used to
    be silently overwritten by the next auto-solve (no Started/Done ever) -- and a queued "code"
    by a Text press. "text" therefore gets its own slot, between the side lane and explain.
    (2026-10-20: the recap lane sits right after it and the insight lane after explain -- both
    their own slots for the same reason; see tests/test_insights_assistant.py. The dashboard
    lane -- a refresh of the tracked board, exocortex.dashboard -- follows the insight lane,
    still above the completion lane; see tests/test_dashboard_service.py.)"""
    a = Assistant(Config(), EventBus(), FakeClient(), Transcript(), clock=time.monotonic)
    slots = a._assist_slots()
    assert slots == (a._assist_slot, a._suggest_slot, a._text_slot, a._recap_slot, a._explain_slot,
                     a._insight_slot, a._dashboard_slot, a._complete_slot)
    assert slots.index(a._suggest_slot) < slots.index(a._text_slot) < slots.index(a._explain_slot)
    assert slots.index(a._insight_slot) < slots.index(a._dashboard_slot) < slots.index(a._complete_slot)
    assert len({id(s) for s in slots}) == 8


def test_queued_text_survives_a_later_code_submission(bus: EventBus, transcript: Transcript) -> None:
    """A slow solve is running, the user presses Text (queued: text never cancels a solve), then
    Code arrives (cancels the running solve, queues on the side lane). The walkthrough must still
    run -- after the code, since the side lane dispatches first -- instead of vanishing."""
    cfg = Config()
    client = FakeClient(token_batches=[list("a" * 20), ["code out"], ["walkthrough out"]], chunk_delay=0.01)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.solve("")
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "solve")
        time.sleep(0.03)
        a.text("")
        a.code("")
        events += drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == "walkthrough out", timeout=6.0)
        kinds = [e.kind for e in events if isinstance(e, AssistStartedEvent)]
        assert kinds == ["solve", "code", "text"]
        by_text = {e.text: e for e in dones(events)}
        assert by_text["code out"].cancelled is False
        assert by_text["walkthrough out"].cancelled is False
        assert [q for _t, q, _a in a.answers] == ["the task being discussed", "the task being discussed"]
    finally:
        a.stop()


def test_queued_code_is_not_evicted_by_a_text_press(bus: EventBus, transcript: Transcript) -> None:
    """The mirror image: a slow walkthrough is running, Code is queued behind it (code never
    cancels a running text), then a second Text press cancels the running walkthrough. The queued
    Code must run (first: side lane before text lane), then the second walkthrough."""
    cfg = Config()
    client = FakeClient(token_batches=[list("a" * 20), ["code out"], ["second walkthrough"]], chunk_delay=0.01)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.text("first")
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "text")
        time.sleep(0.03)
        a.code("")
        a.text("second")
        events += drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == "second walkthrough", timeout=6.0)
        kinds = [e.kind for e in events if isinstance(e, AssistStartedEvent)]
        assert kinds == ["text", "code", "text"]
        by_text = {e.text: e for e in dones(events)}
        assert by_text["code out"].cancelled is False
        assert by_text["second walkthrough"].cancelled is False
        assert next(e for e in dones(events) if e.text not in by_text or e.text.startswith("a")).cancelled is True
    finally:
        a.stop()


def test_checkpoint_drops_a_queued_text(bus: EventBus, transcript: Transcript) -> None:
    """Not just the running one: a walkthrough queued behind a running solve belongs to the old
    coding task too, so `checkpoint()` must clear the text lane's slot (review coverage gap)."""
    cfg = Config()
    client = FakeClient(token_batches=[list("abcdefgh")], chunk_delay=0.05)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.solve("")
        solve_id = next(e for e in drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
                        if isinstance(e, AssistStartedEvent)).job_id
        time.sleep(0.06)
        a.text("")                      # queued in the text lane, never started
        a.checkpoint()
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == solve_id)
        assert next(e for e in dones(events) if e.job_id == solve_id).cancelled is True
        time.sleep(0.2)
        assert not started(drain(sub), "text")
    finally:
        a.stop()


def test_reset_drops_a_queued_text_and_records_nothing(bus: EventBus, transcript: Transcript) -> None:
    """Review finding: `reset()` (New session) cancelled only the running job, so a walkthrough
    queued behind it ran against the fresh, empty session and became its first answer."""
    cfg = Config()
    client = FakeClient(token_batches=[list("abcdefgh"), ["OLD SESSION WALKTHROUGH"]], chunk_delay=0.05)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        a.code("")
        code_id = next(e for e in drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
                       if isinstance(e, AssistStartedEvent)).job_id
        time.sleep(0.06)
        a.text("")
        a.reset()
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == code_id)
        assert next(e for e in dones(events) if e.job_id == code_id).cancelled is True
        time.sleep(0.2)
        assert not started(drain(sub), "text")
        assert a.answers == []
    finally:
        a.stop()


def test_text_does_not_touch_the_auto_solve_throttle(bus: EventBus, transcript: Transcript) -> None:
    """A walkthrough solves nothing: recording it in `_last_suggest_time` would make the next
    screen's automatic solve wait out `suggest_min_interval_s` (a manual prose request must never
    delay it), and the same screen needs no guard since auto-solve only fires on a changed
    ScreenTextEvent. "code" keeps recording (pre-existing, documented behaviour)."""
    cfg = Config()
    client = FakeClient(token_batches=[["walkthrough"], ["code"]], chunk_delay=0.0)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "some coding task")
    sub = bus.subscribe()
    a.start()
    try:
        assert a._last_suggest_time == float("-inf")
        a.text("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == "walkthrough")
        assert a._last_suggest_time == float("-inf") and a._last_suggested_lines == set()
        a.code("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == "code")
        assert a._last_suggest_time > float("-inf")
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# the headroom numbers themselves (the budget tests above read the same dict the code does)
# --------------------------------------------------------------------------------------------


def test_comment_token_headroom_literal_values() -> None:
    """Pins the per-level numbers independently of the plumbing tests, which would move along
    with an accidental edit of the dict (review finding: tautology)."""
    assert assistant_mod.COMMENT_TOKEN_HEADROOM == {"none": 0, "minimal": 0, "normal": 100, "detailed": 350}
    assert set(assistant_mod.COMMENT_TOKEN_HEADROOM) == set(prompts.COMMENT_LEVELS)
