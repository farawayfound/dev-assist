"""Tests for manual advance mode ([handsfree].manual): the press-and-hold mouse gesture that
replaces every automatic generation while the toggle is on.

  (a) exocortex.ui.llhooks.PressHoldMouseHook -- driven through the same fake ``user32`` as
      tests/test_llhooks.py (the captured hook procedure is called directly with fake
      MSLLHOOKSTRUCTs), an injectable clock and a hand-run scheduler: a left hold fires once at
      the hold time (or on release without a scheduler), a short click / a drag / another button /
      a wheel notch / injected input never fire, a quick middle-click fires exactly when
      ``middle_tap`` is on, and ``hold_s`` is re-read at every press.
  (b) exocortex.handsfree.HandsFreeController -- with ``cfg.handsfree.manual`` on (read live), a
      new task still checkpoints and resets the cumulative context but ARMS instead of running
      (no text/code hook call, state "armed"), deviation never regenerates code, and
      ``trigger_step()`` runs exactly one step -- with `HANDS_FREE_CODE_FOCUS` for a code task --
      then returns to "armed". Reuses tests/test_handsfree.py's fakes and helpers.
  (c) exocortex.llm.assistant.Assistant -- the manual flag (read live from the shared Config)
      gates the spoken-question auto-answer, the on-screen question answer and the automatic
      solve/suggest, while text()/code() (the gesture's targets) keep working. Reuses
      tests/test_assistant.py's FakeClient/drain_until/make_segment.
  (d) exocortex.app.Engine + exocortex.config -- set_manual_advance() flips the config flag and
      publishes ManualAdvanceEvent, manual_advance() routes to the controller's trigger_step()
      exactly when hands-free is on and not paused (else a plain text()/code()),
      apply_settings({"handsfree.manual": ...}) wires the same path, and the new [handsfree] /
      [hotkeys] fields round-trip through to_toml()/load_config().
  (e) exocortex.ui.settings -- the three new fields exist on the Hands-free tab with the right
      kinds and bounds.

Fully offline; no real hook, thread pool or model anywhere.
"""
from __future__ import annotations

import ctypes
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from exocortex.bus import (
    AssistStartedEvent,
    EventBus,
    FinalTranscriptEvent,
    HandsFreeEvent,
    ManualAdvanceEvent,
    ScreenTextEvent,
    Source,
    drain,
)
from exocortex.config import Config, HandsFreeConfig, HotkeysConfig, load_config, to_toml
from exocortex.handsfree import ARMED_DETAIL, HANDS_FREE_CODE_FOCUS, HANDS_FREE_CODE_LABEL
from exocortex.transcript import Transcript
from exocortex.ui import llhooks
from exocortex import app as app_mod
from tests.test_assistant import FakeClient, drain_until, make_segment
from tests.test_handsfree import (  # noqa: F401 -- bus/hooks/clock/cfg/make_controller are fixtures
    BASELINE_LINES,
    TRIGGER_LINES,
    FakeClock,
    FakeHooks,
    bus,
    cfg,
    clock,
    complete_step,
    expect_checkpoint,
    hooks,
    make_controller,
    publish_screen,
    settle,
    start_and_settle,
    wait_until,
)
from tests.test_llhooks import FakeUser32


# =====================================================================================
# (a) PressHoldMouseHook
# =====================================================================================


class ManualScheduler:
    """Records (delay, check) pairs; a test runs a scheduled check by hand."""

    def __init__(self) -> None:
        self.scheduled: list[tuple[float, Callable[[], None]]] = []

    def __call__(self, delay_s: float, check: Callable[[], None]) -> None:
        self.scheduled.append((float(delay_s), check))

    def run_last(self) -> None:
        self.scheduled[-1][1]()


class HookClock:
    def __init__(self, start: float = 100.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


def _mouse_struct(x: int = 0, y: int = 0, flags: int = 0) -> "llhooks.MSLLHOOKSTRUCT":
    return llhooks.MSLLHOOKSTRUCT(pt=llhooks.POINT(x, y), mouseData=0, flags=flags, time=0, dwExtraInfo=0)


def _send(fake: FakeUser32, msg: int, x: int = 0, y: int = 0, flags: int = 0) -> int:
    struct = _mouse_struct(x, y, flags)
    return int(fake.proc(0, msg, ctypes.addressof(struct)))


def _make_hook(**kw: Any) -> tuple["llhooks.PressHoldMouseHook", FakeUser32, HookClock, ManualScheduler, dict[str, int]]:
    fired = {"text": 0, "code": 0}
    fake = FakeUser32()
    hook_clock = HookClock()
    scheduler = ManualScheduler()
    bindings = kw.pop("bindings", None) or {
        "mouse_left": lambda: fired.__setitem__("text", fired["text"] + 1),
        "mouse_middle": lambda: fired.__setitem__("code", fired["code"] + 1),
    }
    hook = llhooks.PressHoldMouseHook(
        bindings, user32=fake, clock=hook_clock,
        schedule=kw.pop("schedule", scheduler), **kw,
    )
    assert hook.install() is True
    return hook, fake, hook_clock, scheduler, fired


def test_press_hold_left_fires_once_at_threshold_while_held() -> None:
    hook, fake, hook_clock, scheduler, fired = _make_hook(hold_s=0.8)
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    assert scheduler.scheduled and scheduler.scheduled[-1][0] == pytest.approx(0.8)
    hook_clock.t += 0.8
    scheduler.run_last()
    assert fired == {"text": 1, "code": 0}
    hook_clock.t += 0.4
    _send(fake, llhooks.WM_LBUTTONUP, 10, 10)     # release after the fire: no second fire
    assert fired == {"text": 1, "code": 0}
    assert hook.fires == 1


def test_short_left_click_never_fires_and_stale_check_is_ignored() -> None:
    hook, fake, hook_clock, scheduler, fired = _make_hook(hold_s=0.8)
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    hook_clock.t += 0.1
    _send(fake, llhooks.WM_LBUTTONUP, 10, 10)
    hook_clock.t += 1.0
    scheduler.run_last()                          # the timer for the released press: stale
    assert fired == {"text": 0, "code": 0}
    assert hook.fires == 0


def test_left_hold_without_scheduler_fires_on_release() -> None:
    hook, fake, hook_clock, _scheduler, fired = _make_hook(hold_s=0.8, schedule=None)
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    hook_clock.t += 1.0
    _send(fake, llhooks.WM_LBUTTONUP, 10, 10)
    assert fired["text"] == 1


def test_moving_the_pointer_cancels_the_hold() -> None:
    hook, fake, hook_clock, scheduler, fired = _make_hook(hold_s=0.8, move_px=12)
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    _send(fake, llhooks.WM_MOUSEMOVE, 40, 10)     # a drag / selection
    hook_clock.t += 1.0
    scheduler.run_last()
    _send(fake, llhooks.WM_LBUTTONUP, 40, 10)
    assert fired == {"text": 0, "code": 0}


def test_small_pointer_jitter_does_not_cancel() -> None:
    hook, fake, hook_clock, scheduler, fired = _make_hook(hold_s=0.8, move_px=12)
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    _send(fake, llhooks.WM_MOUSEMOVE, 15, 6)      # within the tolerance
    hook_clock.t += 0.8
    scheduler.run_last()
    assert fired["text"] == 1


@pytest.mark.parametrize("cancel_msg", [llhooks.WM_RBUTTONDOWN, llhooks.WM_XBUTTONDOWN,
                                        llhooks.WM_MOUSEWHEEL, llhooks.WM_MOUSEHWHEEL])
def test_other_input_mid_hold_cancels(cancel_msg: int) -> None:
    hook, fake, hook_clock, scheduler, fired = _make_hook(hold_s=0.8)
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    _send(fake, cancel_msg, 10, 10)
    hook_clock.t += 1.0
    scheduler.run_last()
    _send(fake, llhooks.WM_LBUTTONUP, 10, 10)
    assert fired == {"text": 0, "code": 0}


def test_second_bound_button_mid_hold_cancels_and_does_not_arm() -> None:
    hook, fake, hook_clock, scheduler, fired = _make_hook(hold_s=0.8)
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    _send(fake, llhooks.WM_MBUTTONDOWN, 10, 10)   # a chord: cancels, arms nothing
    hook_clock.t += 1.0
    scheduler.run_last()
    _send(fake, llhooks.WM_LBUTTONUP, 10, 10)
    _send(fake, llhooks.WM_MBUTTONUP, 10, 10)     # the middle's up finds nothing armed
    assert fired == {"text": 0, "code": 0}


def test_middle_hold_fires_code_when_its_hold_is_required() -> None:
    hook, fake, hook_clock, scheduler, fired = _make_hook(hold_s=0.8, require_hold=True)
    _send(fake, llhooks.WM_MBUTTONDOWN, 10, 10)
    hook_clock.t += 0.8
    scheduler.run_last()
    assert fired == {"text": 0, "code": 1}


def test_quick_middle_click_fires_code_only_without_its_hold_requirement() -> None:
    for hold_code, expected in ((False, 1), (True, 0)):
        hook, fake, hook_clock, scheduler, fired = _make_hook(
            hold_s=0.8, require_hold={"mouse_left": True, "mouse_middle": hold_code})
        _send(fake, llhooks.WM_MBUTTONDOWN, 10, 10)
        hook_clock.t += 0.1
        _send(fake, llhooks.WM_MBUTTONUP, 10, 10)
        assert fired["code"] == expected, f"manual_hold_code={hold_code}"


def test_quick_left_click_never_fires_while_its_hold_is_required() -> None:
    hook, fake, hook_clock, scheduler, fired = _make_hook(
        hold_s=0.8, require_hold={"mouse_left": True, "mouse_middle": False})   # the shipped defaults
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    hook_clock.t += 0.1
    _send(fake, llhooks.WM_LBUTTONUP, 10, 10)
    assert fired == {"text": 0, "code": 0}


def test_quick_left_click_fires_when_its_hold_is_not_required() -> None:
    hook, fake, hook_clock, scheduler, fired = _make_hook(
        hold_s=0.8, require_hold={"mouse_left": False, "mouse_middle": False})
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    hook_clock.t += 0.1
    _send(fake, llhooks.WM_LBUTTONUP, 10, 10)
    assert fired == {"text": 1, "code": 0}


def test_injected_middle_click_is_processed_like_real_input() -> None:
    """A touchpad three-finger tap mapped to a middle click arrives INJECTED (Windows sends it
    through SendInput); it must fire like a real quick middle-click."""
    hook, fake, hook_clock, scheduler, fired = _make_hook(
        hold_s=0.8, require_hold={"mouse_left": True, "mouse_middle": False})
    _send(fake, llhooks.WM_MBUTTONDOWN, 10, 10, flags=llhooks._LLMHF_INJECTED)
    hook_clock.t += 0.05
    _send(fake, llhooks.WM_MBUTTONUP, 10, 10, flags=llhooks._LLMHF_INJECTED)
    assert fired == {"text": 0, "code": 1}


def test_hold_seconds_and_require_hold_are_read_live() -> None:
    knobs = {"hold": 2.0, "hold_code": True}
    hook, fake, hook_clock, scheduler, fired = _make_hook(
        hold_s=lambda: knobs["hold"],
        require_hold=lambda button: True if button == "mouse_left" else knobs["hold_code"])
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    assert scheduler.scheduled[-1][0] == pytest.approx(2.0)
    hook_clock.t += 1.0
    _send(fake, llhooks.WM_LBUTTONUP, 10, 10)     # held 1.0 < 2.0: no fire
    assert fired["text"] == 0
    knobs["hold"] = 0.5                            # settings changed; no re-install
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    assert scheduler.scheduled[-1][0] == pytest.approx(0.5)
    hook_clock.t += 0.6
    _send(fake, llhooks.WM_LBUTTONUP, 10, 10)
    assert fired["text"] == 1
    _send(fake, llhooks.WM_MBUTTONDOWN, 10, 10)   # hold_code still True: a tap does nothing
    hook_clock.t += 0.05
    _send(fake, llhooks.WM_MBUTTONUP, 10, 10)
    assert fired["code"] == 0
    knobs["hold_code"] = False                     # unchecked in settings, read at release
    _send(fake, llhooks.WM_MBUTTONDOWN, 10, 10)
    hook_clock.t += 0.05
    _send(fake, llhooks.WM_MBUTTONUP, 10, 10)
    assert fired["code"] == 1


def test_messages_are_never_swallowed() -> None:
    fired: list[str] = []
    fake = FakeUser32(call_next=42)
    hook_clock = HookClock()
    hook = llhooks.PressHoldMouseHook({"mouse_left": lambda: fired.append("x")},
                                      hold_s=0.2, schedule=None, user32=fake, clock=hook_clock)
    hook.install()
    assert _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10) == 42
    hook_clock.t += 0.3
    assert _send(fake, llhooks.WM_LBUTTONUP, 10, 10) == 42   # fires AND passes the message on
    assert fired == ["x"]


def test_unbound_button_is_not_armed() -> None:
    hook, fake, hook_clock, scheduler, fired = _make_hook(
        bindings={"mouse_left": lambda: None})
    _send(fake, llhooks.WM_MBUTTONDOWN, 10, 10)
    assert not scheduler.scheduled


@pytest.mark.parametrize("bad", [{}, {"right": lambda: None}])
def test_bad_bindings_raise(bad: dict) -> None:
    with pytest.raises(ValueError):
        llhooks.PressHoldMouseHook(bad)


def test_uninstall_mid_press_resets_state_for_the_next_install() -> None:
    """Manual toggled off while a button is held: the release is never seen. Re-installing the
    same hook object must start clean, or every later press reads as a chord and never fires."""
    hook, fake, hook_clock, scheduler, fired = _make_hook(hold_s=0.8)
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    hook.uninstall()                              # mid-press; the WM_LBUTTONUP is never delivered
    hook_clock.t += 5.0
    assert hook.install() is True
    _send(fake, llhooks.WM_MBUTTONDOWN, 20, 20)   # a fresh hold of the OTHER button
    hook_clock.t += 0.8
    scheduler.run_last()
    assert fired == {"text": 0, "code": 1}
    scheduler.scheduled.clear()
    _send(fake, llhooks.WM_MBUTTONUP, 20, 20)
    _send(fake, llhooks.WM_LBUTTONDOWN, 30, 30)   # and of the original button
    hook_clock.t += 0.8
    scheduler.run_last()
    assert fired == {"text": 1, "code": 1}


def test_uninstall_invalidates_a_pending_scheduled_check() -> None:
    hook, fake, hook_clock, scheduler, fired = _make_hook(hold_s=0.8)
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    hook.uninstall()
    hook_clock.t += 1.0
    scheduler.run_last()                          # the old press's timer: stale generation
    assert fired == {"text": 0, "code": 0}


# =====================================================================================
# (b) HandsFreeController in manual mode
# =====================================================================================


def test_manual_new_task_checkpoints_and_arms_without_generating(
    bus: EventBus, hooks: FakeHooks, clock: FakeClock, cfg: Config, make_controller
) -> None:
    cfg.handsfree.manual = True
    c = make_controller(cfg, bus, hooks, clock)
    sub = bus.subscribe()
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)              # the auto-checkpoint still happens
    assert wait_until(lambda: c.state == "armed")
    time.sleep(0.2)
    assert hooks.text_calls == 0 and hooks.code_calls == 0
    assert c.context_text()                       # the cumulative context was still reset/merged
    events = drain(sub)
    armed = [e for e in events if isinstance(e, HandsFreeEvent) and e.state == "armed"]
    assert armed and armed[-1].detail == ARMED_DETAIL


def test_trigger_step_runs_one_step_and_returns_to_armed(
    bus: EventBus, hooks: FakeHooks, clock: FakeClock, cfg: Config, make_controller
) -> None:
    cfg.handsfree.manual = True
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: c.state == "armed")
    c.trigger_step("text")
    complete_step(bus, hooks, "text", 1)
    assert wait_until(lambda: c.state == "armed")   # the task stays armed for further gestures
    assert hooks.checkpoint_calls == 1              # a gesture never re-checkpoints
    c.trigger_step("code")
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "armed")
    assert hooks.text_calls == 1 and hooks.code_calls == 1


def test_trigger_step_code_task_gets_the_hands_free_code_focus(
    bus: EventBus, hooks: FakeHooks, clock: FakeClock, make_controller
) -> None:
    cfg = Config()
    cfg.handsfree.settle_s = 0.05
    cfg.handsfree.accurate_ocr = False
    cfg.handsfree.manual = True                    # default on_code="code": a code task
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)   # TRIGGER_LINES read as a code task
    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: c.state == "armed")
    assert c.last_task_kind == "code"
    c.trigger_step("code")
    assert wait_until(lambda: hooks.code_calls == 1)
    assert hooks.code_focus_labels[-1] == (HANDS_FREE_CODE_FOCUS, HANDS_FREE_CODE_LABEL)
    c.trigger_step("text")                         # text for the same code task: the plain ask
    assert wait_until(lambda: hooks.text_calls == 1)
    assert hooks.text_focus_labels[-1] == ("", "")


def test_manual_mode_never_regenerates_code_on_deviation(
    bus: EventBus, hooks: FakeHooks, clock: FakeClock, cfg: Config, make_controller
) -> None:
    # A completed cycle in AUTO first (so the app owns some code), then manual goes on live.
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, BASELINE_LINES)
    expect_checkpoint(bus, hooks, 1)
    complete_step(bus, hooks, "text", 1)
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")
    cfg.handsfree.manual = True                    # flipped live, like the settings dialog does
    hooks.recent_code_text = "def two_sum(nums, target):\n    return []"
    deviated = ["seen = {}", "for i, v in enumerate(nums):", "    seen[v] = i", "x = target - v"]
    settle(bus, clock, c, BASELINE_LINES[:1] + deviated)
    time.sleep(0.3)
    assert c.regenerations == 0
    assert hooks.code_calls == 1                   # only the baseline cycle's


def test_trigger_step_is_ignored_while_paused(
    bus: EventBus, hooks: FakeHooks, clock: FakeClock, cfg: Config, make_controller
) -> None:
    cfg.handsfree.manual = True
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: c.state == "armed")
    c.set_paused(True)
    assert wait_until(lambda: c.state == "paused")
    c.trigger_step("text")
    time.sleep(0.3)
    assert hooks.text_calls == 0


def test_manual_flipped_on_mid_cycle_arms_the_remaining_steps(
    bus: EventBus, hooks: FakeHooks, clock: FakeClock, cfg: Config, make_controller
) -> None:
    """A cycle whose step list was built under Auto must not keep auto-firing its remaining
    steps once manual goes on: the queued steps arm instead."""
    c = make_controller(cfg, bus, hooks, clock)                 # Auto: steps = [text, code]
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: hooks.text_calls == 1)            # the text step is streaming
    cfg.handsfree.manual = True                                 # flipped mid-cycle
    complete_step(bus, hooks, "text", 1)
    assert wait_until(lambda: c.state == "armed")
    time.sleep(0.2)
    assert hooks.code_calls == 0                                # the queued code step never fired
    c.trigger_step("code")                                      # ...but the gesture still runs it
    assert wait_until(lambda: hooks.code_calls == 1)


def test_resume_auto_runs_the_armed_task_without_a_second_checkpoint(
    bus: EventBus, hooks: FakeHooks, clock: FakeClock, cfg: Config, make_controller
) -> None:
    cfg.handsfree.manual = True
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: c.state == "armed")
    cfg.handsfree.manual = False                                # the engine flips the flag first...
    c.resume_auto()                                             # ...then pokes the controller
    complete_step(bus, hooks, "text", 1)                        # cfg fixture: on_code="ignore"
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")
    assert hooks.checkpoint_calls == 1                          # arming already checkpointed


def test_resume_auto_is_a_noop_with_nothing_armed(
    bus: EventBus, hooks: FakeHooks, clock: FakeClock, cfg: Config, make_controller
) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    c.start()
    assert wait_until(lambda: hooks.capture_calls == [False])
    c.resume_auto()
    time.sleep(0.2)
    assert hooks.text_calls == 0 and hooks.code_calls == 0 and hooks.checkpoint_calls == 0


def test_manual_off_mid_session_restores_auto_cycles(
    bus: EventBus, hooks: FakeHooks, clock: FakeClock, cfg: Config, make_controller
) -> None:
    cfg.handsfree.manual = True
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: c.state == "armed")
    cfg.handsfree.manual = False                   # back to Auto, read live
    settle(bus, clock, c, LUNCH_FREE_LINES)
    expect_checkpoint(bus, hooks, 2)
    complete_step(bus, hooks, "text", 1)
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")


LUNCH_FREE_LINES = [
    "Design a rate limiter for a public HTTP API.",
    "Requests above the limit must receive HTTP 429.",
    "Explain the sliding window versus token bucket trade-off.",
]


# =====================================================================================
# (c) Assistant: the manual flag gates every automatic lane
# =====================================================================================


def _make_assistant(cfg: Config, bus: EventBus, transcript: Transcript, client: FakeClient):
    from exocortex.llm.assistant import Assistant

    return Assistant(cfg, bus, client, transcript, clock=time.monotonic)


def _screen_event(lines: tuple[str, ...]) -> ScreenTextEvent:
    return ScreenTextEvent(mode="auto", source_name="Fake", rect=(0, 0, 100, 100),
                           text="\n".join(lines), lines=lines, changed=True, ocr_ms=1.0, captured_at=0.0)


def test_manual_suppresses_auto_solve_and_screen_questions() -> None:
    cfg = Config()
    cfg.handsfree.manual = True
    cfg.screen.auto_suggest = True
    cfg.screen.answer_screen_questions = True
    cfg.screen.suggest_min_interval_s = 0
    bus = EventBus()
    assistant = _make_assistant(cfg, bus, Transcript(), FakeClient(token_batches=[["x"]], chunk_delay=0.0))
    sub = bus.subscribe()
    assistant.start()
    try:
        bus.publish(_screen_event(("How would you implement a thread-safe LRU cache here?",)))
        time.sleep(0.3)
        assert not any(isinstance(e, AssistStartedEvent) for e in drain(sub))
    finally:
        assistant.stop()


def test_manual_suppresses_spoken_auto_answer_but_not_text_or_code() -> None:
    cfg = Config()
    cfg.handsfree.manual = True
    cfg.assist.coalesce_ms = 30
    cfg.screen.auto_suggest = False
    cfg.screen.answer_screen_questions = False
    bus = EventBus()
    transcript = Transcript()
    assistant = _make_assistant(cfg, bus, transcript,
                                FakeClient(token_batches=[["a"], ["b"]], chunk_delay=0.0))
    sub = bus.subscribe()
    assistant.start()
    try:
        seg = make_segment(1, text="How would you design a rate limiter for the public API?",
                           is_question=True)
        transcript.add(seg)
        bus.publish(FinalTranscriptEvent(segment=seg))
        time.sleep(0.4)                            # past the coalesce window
        assert not any(isinstance(e, AssistStartedEvent) for e in drain(sub))
        # ...while the gesture's targets still work:
        job_id = assistant.text()
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert started.kind == "text" and started.job_id == job_id
    finally:
        assistant.stop()


def test_manual_off_restores_auto_solve() -> None:
    cfg = Config()
    cfg.handsfree.manual = False
    cfg.screen.auto_suggest = True
    cfg.screen.answer_screen_questions = False
    cfg.screen.suggest_min_interval_s = 0
    bus = EventBus()
    assistant = _make_assistant(cfg, bus, Transcript(),
                                FakeClient(token_batches=[["an answer"]], chunk_delay=0.0))
    sub = bus.subscribe()
    assistant.start()
    try:
        bus.publish(_screen_event(("Some entirely new screen content appears now.",)))
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        assert next(e for e in events if isinstance(e, AssistStartedEvent)).kind == "solve"
    finally:
        assistant.stop()


# =====================================================================================
# (d) Engine wiring + config round-trip
# =====================================================================================


class FakeHandsFree:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.steps: list[str] = []
        self.resumes = 0

    def trigger_step(self, kind: str) -> None:
        self.steps.append(kind)

    def resume_auto(self) -> None:
        self.resumes += 1


class RecordingAssistant:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def text(self, focus: str = "", label: str = "", read_now: bool = True) -> int:
        self.calls.append(("text", focus))
        return 1

    def code(self, focus: str = "", label: str = "", fresh: bool = False, read_now: bool = True) -> int:
        # 2026-10-29: `read_now` (read the screen afresh) is accepted and ignored here -- the
        # routing is what these tests check; tests/test_handsfree_readnow.py covers the flag.
        self.calls.append(("code", focus) if not fresh else ("code_fresh", focus))
        return 2


def _bare_engine() -> app_mod.Engine:
    eng = app_mod.Engine(Config(), EventBus())
    eng._llm_ready.set()                          # never defer text()/code() in these tests
    eng.assistant = RecordingAssistant()
    return eng


def test_set_manual_advance_flips_config_and_publishes_event() -> None:
    eng = _bare_engine()
    sub = eng.bus.subscribe()
    eng.set_manual_advance(True)
    assert eng.cfg.handsfree.manual is True
    events = drain(sub)
    assert any(isinstance(e, ManualAdvanceEvent) and e.enabled for e in events)
    eng.set_manual_advance(False)
    assert eng.cfg.handsfree.manual is False
    assert any(isinstance(e, ManualAdvanceEvent) and not e.enabled for e in drain(sub))


def test_set_manual_advance_off_resumes_an_enabled_controller() -> None:
    eng = _bare_engine()
    eng.handsfree = FakeHandsFree(enabled=True)
    eng.set_manual_advance(True)
    assert eng.handsfree.resumes == 0
    eng.set_manual_advance(False)                  # back to Auto: run whatever is armed
    assert eng.handsfree.resumes == 1

    eng2 = _bare_engine()
    eng2.handsfree = FakeHandsFree(enabled=False)  # controller off: nothing to resume
    eng2.set_manual_advance(False)
    assert eng2.handsfree.resumes == 0


def test_manual_advance_routes_to_the_controller_when_hands_free_is_on() -> None:
    eng = _bare_engine()
    eng.handsfree = FakeHandsFree(enabled=True)
    eng.manual_advance("code")
    eng.manual_advance("text")
    eng.manual_advance("bogus")                    # anything not "code" is a text
    assert eng.handsfree.steps == ["code", "text", "text"]
    assert eng.assistant.calls == []


def test_manual_advance_falls_back_to_plain_text_code() -> None:
    eng = _bare_engine()                           # no controller at all
    eng.manual_advance("code")
    eng.manual_advance("text")
    assert eng.assistant.calls == [("code", ""), ("text", "")]

    eng2 = _bare_engine()
    eng2.handsfree = FakeHandsFree(enabled=False)  # controller off: plain calls
    eng2.manual_advance("code")
    assert eng2.handsfree.steps == []
    assert eng2.assistant.calls == [("code", "")]

    eng3 = _bare_engine()
    eng3.handsfree = FakeHandsFree(enabled=True)
    eng3._paused = True                            # paused: manual still works, bypassing the controller
    eng3.manual_advance("text")
    assert eng3.handsfree.steps == []
    assert eng3.assistant.calls == [("text", "")]


def test_apply_settings_manual_publishes_the_sync_event(tmp_path: Path) -> None:
    eng = _bare_engine()
    eng.cfg.path = tmp_path / "config.toml"
    sub = eng.bus.subscribe()
    eng.cfg.handsfree.manual = True                # the dialog wrote it already
    eng.apply_settings({"handsfree.manual": True})
    assert any(isinstance(e, ManualAdvanceEvent) and e.enabled for e in drain(sub))
    assert eng.cfg.path.is_file()                  # apply_settings saved the config


def test_manual_config_defaults_and_round_trip(tmp_path: Path) -> None:
    hf = Config().handsfree
    assert hf.manual is False
    assert hf.manual_hold_s == 0.8
    assert hf.manual_hold_text is True     # left clicks happen constantly: the hold is required
    assert hf.manual_hold_code is False    # a quick middle press / three-finger tap triggers
    assert hf.manual_hold_code_fresh is False   # 2026-10-29: the third gesture is a click too
    assert Config().hotkeys.toggle_manual == "ctrl+alt+j"
    # 2026-10-29: the gestures themselves are [hotkeys] bindings (Settings > Shortcuts)
    assert (Config().hotkeys.manual_text, Config().hotkeys.manual_code, Config().hotkeys.manual_code_fresh) == (
        "mouse_left", "mouse_middle", "mouse4")
    assert Config().hotkeys.swallow_thumb_buttons is True

    cfg = Config()
    cfg.handsfree.manual = True
    cfg.handsfree.manual_hold_s = 1.5
    cfg.handsfree.manual_hold_text = False
    cfg.handsfree.manual_hold_code = True
    cfg.handsfree.manual_hold_code_fresh = True
    cfg.hotkeys.manual_code_fresh = "ctrl+mouse5"
    cfg.hotkeys.swallow_thumb_buttons = False
    p = tmp_path / "config.toml"
    p.write_text(to_toml(cfg), encoding="utf-8")
    loaded = load_config(p)
    assert loaded.handsfree == cfg.handsfree
    assert loaded.hotkeys == cfg.hotkeys


def test_old_config_without_manual_fields_loads_with_defaults(tmp_path: Path) -> None:
    # incl. a config saved by the first manual-advance build, whose manual_middle_tap key is
    # gone now (unknown keys warn and are ignored)
    p = tmp_path / "config.toml"
    p.write_text("[handsfree]\nenabled = true\nsettle_s = 2.0\nmanual_middle_tap = true\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.handsfree.enabled is True
    assert cfg.handsfree.manual is False
    assert cfg.handsfree.manual_hold_s == 0.8
    assert cfg.handsfree.manual_hold_text is True
    assert cfg.handsfree.manual_hold_code is False
    assert cfg.handsfree.manual_hold_code_fresh is False
    assert cfg.hotkeys.manual_code_fresh == "mouse4"


def test_manual_advance_code_fresh_routes_to_the_controller_or_a_fresh_code() -> None:
    """2026-10-29: the third gesture. With hands-free on it is the controller's "code_fresh" step;
    otherwise a plain code(fresh=True) -- never a text."""
    eng = _bare_engine()
    eng.handsfree = FakeHandsFree(enabled=True)
    eng.manual_advance("code_fresh")
    assert eng.handsfree.steps == ["code_fresh"]
    assert eng.assistant.calls == []

    eng2 = _bare_engine()
    eng2.manual_advance("code_fresh")
    assert eng2.assistant.calls == [("code_fresh", "")]

    eng3 = _bare_engine()
    eng3.handsfree = FakeHandsFree(enabled=True)
    eng3._paused = True
    eng3.manual_advance("code_fresh")
    assert eng3.handsfree.steps == [] and eng3.assistant.calls == [("code_fresh", "")]


def test_engine_code_fresh_is_deferred_and_replayed_with_its_flag() -> None:
    """A fresh code request made before the model is ready keeps its flag through the replay."""
    eng = app_mod.Engine(Config(), EventBus())
    eng.assistant = RecordingAssistant()
    assert eng.code(fresh=True) is None                # deferred: the LLM is not ready
    assert eng.assistant.calls == []
    eng._llm_ready.set()
    eng._replay_pending()
    assert eng.assistant.calls == [("code_fresh", "")]


def test_trigger_step_code_fresh_runs_a_fresh_code_step(
    bus: EventBus, hooks: FakeHooks, clock: FakeClock, cfg: Config, make_controller
) -> None:
    """The controller's "code_fresh" step is a "code" step whose hook call carries fresh=True (the
    phase published is "code"), and the task stays armed afterwards like any gesture."""
    cfg.handsfree.manual = True
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: c.state == "armed")
    sub = bus.subscribe()
    c.trigger_step("code_fresh")
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "armed")
    assert hooks.code_calls == 1 and hooks.text_calls == 0
    # TRIGGER_LINES is a code task: the fresh step keeps the code-task ask, plus the flag
    assert [call for call in hooks.calls if call[0] == "code"] == [
        ("code", HANDS_FREE_CODE_FOCUS, HANDS_FREE_CODE_LABEL, "fresh")]
    assert any(isinstance(e, HandsFreeEvent) and e.state == "code" for e in drain(sub))


# =====================================================================================
# (e) settings fields
# =====================================================================================


def test_manual_settings_fields_exist_on_the_hands_free_tab() -> None:
    from exocortex.ui.settings import FIELDS, all_fields

    by_path = {f.path: f for f in all_fields()}
    manual = by_path["handsfree.manual"]
    assert manual.kind == "bool"
    assert manual in FIELDS["Hands-free"]
    hold = by_path["handsfree.manual_hold_s"]
    assert hold.kind == "float"
    assert hold.lo == pytest.approx(0.2) and hold.hi == pytest.approx(3.0)
    assert hold in FIELDS["Hands-free"]
    for key in ("handsfree.manual_hold_text", "handsfree.manual_hold_code", "handsfree.manual_hold_code_fresh"):
        f = by_path[key]
        assert f.kind == "bool"
        assert f in FIELDS["Hands-free"]
    assert "handsfree.manual_middle_tap" not in by_path   # replaced by the per-action holds
