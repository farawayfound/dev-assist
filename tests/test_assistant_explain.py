"""Offline tests for explanations mode in the Assistant: the plan built from a "solve", the
"explain" job (triggered by the completer or by the code appearing on screen), the lane rules
that keep a queued solve safe from completions, and the same-task guard."""
from __future__ import annotations

import queue
import time
from typing import Any

import pytest

from exocortex.bus import (
    AssistDoneEvent,
    AssistStartedEvent,
    EventBus,
    ScreenTextEvent,
    Source,
    TranscriptSegment,
    drain,
)
from exocortex.config import Config
from exocortex.llm import assistant as assistant_mod
from exocortex.llm.assistant import Assistant
from exocortex.transcript import Transcript
from tests.test_assistant_complete import FakeClient, drain_until
from tests.test_plan import ANSWER, SOLUTION

PROBLEM = ("Two Sum", "Given an array of integers nums and an integer target,",
           "return indices of the two numbers such that they add up to target.", "class Solution:",
           "def twoSum(self, nums, target):")


class ScriptedClient(FakeClient):
    """Streams a different token list per call (`scripts`, consumed in order; then `tokens`)."""

    def __init__(self, scripts: list[list[str]], chunk_delay: float = 0.002) -> None:
        super().__init__(tokens=["- fallback"], chunk_delay=chunk_delay)
        self.scripts = list(scripts)

    def stream_completion(self, prompt: str, **kw: Any):
        if self.scripts:
            self.tokens = self.scripts.pop(0)
        return super().stream_completion(prompt, **kw)


def chunks(text: str, size: int = 12) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


def screen_event(lines: tuple[str, ...]) -> ScreenTextEvent:
    return ScreenTextEvent(mode="auto", source_name="chrome", rect=(0, 0, 1, 1), text="\n".join(lines),
                           lines=lines, changed=True, ocr_ms=1.0, captured_at=0.0)


def started(events: list[Any], kind: str) -> list[AssistStartedEvent]:
    return [e for e in events if isinstance(e, AssistStartedEvent) and e.kind == kind]


@pytest.fixture()
def cfg() -> Config:
    cfg = Config()
    cfg.explain.enabled = True
    cfg.screen.auto_suggest = True
    cfg.screen.suggest_min_interval_s = 0
    return cfg


def make(cfg: Config, client: FakeClient, screen: str = "\n".join(PROBLEM)) -> tuple[Assistant, EventBus, "queue.Queue[Any]"]:
    bus = EventBus()
    tr = Transcript()
    tr.add(TranscriptSegment(id=1, source=Source.SYSTEM, speaker_id="spk1", speaker_label="Priya",
                             text="Let's do two sum.", t_start=0.0, t_end=2.0, confidence=0.9))
    a = Assistant(cfg, bus, client, tr, clock=time.monotonic)
    a.set_screen_provider(lambda: screen)
    sub = bus.subscribe()
    a.start()
    return a, bus, sub


def test_solve_with_explanations_builds_the_plan(cfg: Config) -> None:
    client = ScriptedClient([chunks(ANSWER)])
    a, bus, sub = make(cfg, client)
    try:
        a.solve("")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text)
        assert started(events, "solve")
        prompt = client.calls[-1]["prompt"]
        assert "Walkthrough:" in prompt and "well commented" in prompt   # default comment level: "normal"
        assert client.calls[-1]["max_tokens"] == (
            cfg.llm.solve_max_tokens + cfg.explain.walkthrough_max_tokens
            + assistant_mod.COMMENT_TOKEN_HEADROOM["normal"]      # the default comment level's headroom
        )
        view = a.plan()
        assert view is not None and view.done and view.is_code and view.solution == SOLUTION
        assert a.plan_walkthrough.startswith("- We need two indices")
        assert a.plan_explanation == ""
        assert a.answers and a.answers[-1][2] == ANSWER
    finally:
        a.stop()


def test_plan_typed_explains_once_with_the_plan_in_the_prompt(cfg: Config) -> None:
    client = ScriptedClient([chunks(ANSWER), ["- It maps ", "values to indices."]])
    a, bus, sub = make(cfg, client)
    try:
        a.solve("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text)
        plan_id = a.plan().id
        a.plan_typed(plan_id + 99)                                  # stale id: ignored
        a.plan_typed(plan_id)
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text.startswith("- It maps"))
        assert started(events, "explain") and started(events, "explain")[0].question == "Explanation"
        prompt = client.calls[-1]["prompt"]
        assert "<walkthrough>" in prompt and "- We need two indices" in prompt
        assert "<solution>" in prompt and "seen = {}" in prompt and "<screen>" in prompt
        assert client.calls[-1]["max_tokens"] == cfg.explain.explain_max_tokens
        assert a.plan_explanation == "- It maps values to indices."
        assert a.answers[-1][1] == "Explanation"
        a.plan_typed(plan_id)                                       # already explained: no new job
        time.sleep(0.15)
        assert not started(drain(sub), "explain")
    finally:
        a.stop()


def test_code_visible_on_screen_triggers_the_explanation(cfg: Config) -> None:
    client = ScriptedClient([chunks(ANSWER), ["- explained"]])
    a, bus, sub = make(cfg, client)
    try:
        bus.publish(screen_event(PROBLEM))
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == ANSWER)
        # half the solution on screen: nothing yet
        code_lines = tuple(ln.strip() for ln in SOLUTION.split("\n") if not ln.strip().startswith("#"))
        bus.publish(screen_event(PROBLEM + code_lines[:4]))
        time.sleep(0.2)
        assert not started(drain(sub), "explain")
        bus.publish(screen_event(PROBLEM + code_lines))
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == "- explained")
        assert started(events, "explain")
        assert not started(events, "solve")                          # the same task was not re-solved
    finally:
        a.stop()


def test_a_new_task_replaces_the_plan_but_typing_the_solution_does_not(cfg: Config) -> None:
    client = ScriptedClient([chunks(ANSWER), chunks(ANSWER)])
    a, bus, sub = make(cfg, client)
    try:
        bus.publish(screen_event(PROBLEM))
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == ANSWER)
        first = a.plan().id
        bus.publish(screen_event(PROBLEM + ("seen = {}", "for i, n in enumerate(nums):", "x", "y", "z", "w")))
        time.sleep(0.2)
        assert not started(drain(sub), "solve") and a.plan().id == first
        other = ("Valid Parentheses", "Given a string s containing just the characters", "determine if the input is valid",
                 "class Solution:", "def isValid(self, s):")
        bus.publish(screen_event(other))
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == ANSWER)
        assert started(events, "solve") and a.plan().id != first
    finally:
        a.stop()


def test_completion_mode_still_solves_when_explanations_are_on(cfg: Config) -> None:
    client = ScriptedClient([chunks(ANSWER)])
    a, bus, sub = make(cfg, client)
    try:
        a.set_auto_suppressed(True)
        cfg.explain.enabled = False
        bus.publish(screen_event(PROBLEM))
        time.sleep(0.2)
        assert not started(drain(sub), "solve")
        cfg.explain.enabled = True
        bus.publish(screen_event(PROBLEM))
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        assert started(events, "solve")
        # ensure_plan(): a plan for the same screen is kept, a different screen gets a new solve
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        a.ensure_plan()
        time.sleep(0.15)
        assert not started(drain(sub), "solve")
        a.set_screen_provider(lambda: "Something else entirely\nline two\nline three")
        a.ensure_plan()
        assert started(drain_until(sub, lambda e: isinstance(e, AssistStartedEvent)), "solve")
    finally:
        a.stop()


def test_completion_never_displaces_a_queued_solve(cfg: Config) -> None:
    client = ScriptedClient([["a"] * 30, chunks(ANSWER), ["ld"]], chunk_delay=0.01)
    a, bus, sub = make(cfg, client)
    try:
        a.ask("slow question")                                          # main lane, ~0.3 s
        drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "ask")
        a.solve("")                                                     # queued behind it (side lane)
        job = a.complete("hello wor", "", "f")                          # completion lane: must not replace the solve
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == job, timeout=6.0)
        kinds = [e.kind for e in events if isinstance(e, AssistStartedEvent)]
        assert kinds == ["solve", "complete"] or kinds == ["ask", "solve", "complete"]
        assert a.plan() is not None and a.plan().done
    finally:
        a.stop()


def test_new_solve_cancels_a_running_explanation_and_reset_drops_the_plan(cfg: Config) -> None:
    client = ScriptedClient([chunks(ANSWER), ["- slow"] * 40, chunks(ANSWER)], chunk_delay=0.01)
    a, bus, sub = make(cfg, client)
    try:
        a.solve("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == ANSWER)
        a.explain_now()
        drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "explain")
        time.sleep(0.05)
        a.solve("")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == ANSWER, timeout=6.0)
        explain_done = [e for e in events if isinstance(e, AssistDoneEvent) and e.cancelled]
        assert explain_done, "the running explanation was cancelled by the new solve"
        assert a.plan() is not None and a.plan_explanation == ""
        a.reset()
        assert a.plan() is None and a.plan_walkthrough == ""
    finally:
        a.stop()


def test_containment_guard_constant_is_sane() -> None:
    assert 0.0 < assistant_mod.PLAN_KEEP_FRACTION < 1.0
    assert assistant_mod._CANCELS["complete"] == ("complete",)
    assert "solve" not in assistant_mod._CANCELS["explain"]



def test_explanations_toggled_off_mid_solve_still_finishes_the_plan(cfg: Config) -> None:
    client = ScriptedClient([chunks(ANSWER)], chunk_delay=0.01)
    a, bus, sub = make(cfg, client)
    try:
        a.solve("")
        drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "solve")
        time.sleep(0.05)
        cfg.explain.enabled = False                       # user toggles it off while the solve streams
        assert a.plan() is None
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text == ANSWER, timeout=6.0)
        cfg.explain.enabled = True                        # ... and back on: the plan is finished, not stuck
        view = a.plan()
        assert view is not None and view.done and view.solution == SOLUTION
    finally:
        a.stop()
