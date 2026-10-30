"""The hands-free side of read-on-shortcut (2026-10-29): a manual gesture's step asks the engine for a
fresh screen read (`read_now=True`) while the cycle's own steps, which follow the controller's accurate
read, pass False; `ScreenContext.text(current=...)` / `HandsFreeController.context_text(current)` render
a fresh screen as the "on screen now" part; and the engine's `_HandsFreeHooks` forward `fresh` and
`read_now` (the missing `fresh` used to make a hands-free Code-from-scratch step raise). Reuses the
fixtures and fakes of tests/test_handsfree.py and tests/test_handsfree_wiring.py."""
from __future__ import annotations

from exocortex import app as app_mod
from exocortex.bus import EventBus
from exocortex.config import Config
from exocortex.handsfree import CONTEXT_EARLIER_HEADER, CONTEXT_NOW_HEADER, ScreenContext
from tests.test_handsfree import (  # noqa: F401 - fixtures
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
    start_and_settle,
    wait_until,
)
from tests.test_handsfree_wiring import FakeAssistant


def test_a_manual_gesture_step_reads_afresh_while_the_cycles_own_steps_do_not(
    bus: EventBus, hooks: FakeHooks, clock: FakeClock, cfg: Config, make_controller
) -> None:
    c = make_controller(cfg, bus, hooks, clock)                 # Auto: steps = [text, code]
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    complete_step(bus, hooks, "text", 1)
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")
    assert hooks.read_now_flags == [("text", False), ("code", False)]
    cfg.handsfree.manual = True
    c.trigger_step("code")
    complete_step(bus, hooks, "code", 2)
    c.trigger_step("text")
    complete_step(bus, hooks, "text", 2)
    c.trigger_step("code_fresh")
    complete_step(bus, hooks, "code", 3)
    assert wait_until(lambda: len(hooks.read_now_flags) == 5)
    assert hooks.read_now_flags[2:] == [("code", True), ("text", True), ("code", True)]


def test_screen_context_renders_a_fresh_screen_as_the_now_part_without_recording_it() -> None:
    ctx = ScreenContext()
    ctx.merge(["def a(): pass", "def b(): pass", "def c(): pass"], "def a(): pass\ndef b(): pass\ndef c(): pass")
    fresh = "def c(): pass\ndef d(): pass"
    out = ctx.text(1000, current=fresh)
    assert out.endswith(f"{CONTEXT_NOW_HEADER}\n{fresh}")
    earlier = out.split(CONTEXT_NOW_HEADER)[0]
    assert CONTEXT_EARLIER_HEADER in earlier and "def a(): pass" in earlier and "def b(): pass" in earlier
    assert "def c(): pass" not in earlier                         # still on screen: not "earlier"
    assert ctx.text(1000) == "def a(): pass\ndef b(): pass\ndef c(): pass"   # nothing recorded
    assert ctx.text(1000, current="   ") == ctx.text(1000)        # blank = the recorded screen


def test_controller_context_text_takes_the_fresh_screen(
    bus: EventBus, hooks: FakeHooks, clock: FakeClock, cfg: Config, make_controller
) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    assert c.context_text("fresh") == ""                          # off: nothing
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: not c.context.empty)
    out = c.context_text("pass\n    return []")                # like the recorded text, stripped
    assert out.endswith(f"{CONTEXT_NOW_HEADER}\npass\n    return []")
    assert "Two Sum" in out                                       # the trigger lines scrolled away


def test_engine_hooks_forward_fresh_and_read_now() -> None:
    eng = app_mod.Engine(Config(), EventBus())
    fake = FakeAssistant()
    eng.assistant = fake
    eng._llm_ready.set()
    hooks = app_mod._HandsFreeHooks(eng)
    hooks.code("f", "l")
    hooks.code("f", "l", fresh=True, read_now=True)
    hooks.text("f", "l", read_now=True)
    hooks.text()
    assert fake.fresh_flags == [False, True]
    assert fake.read_now_flags == [("code", False), ("code", True), ("text", True), ("text", False)]


def test_engine_code_and_text_default_to_a_fresh_read() -> None:
    eng = app_mod.Engine(Config(), EventBus())
    fake = FakeAssistant()
    eng.assistant = fake
    eng._llm_ready.set()
    eng.code("fix it")
    eng.text()
    eng.code("", read_now=False)
    assert fake.read_now_flags == [("code", True), ("text", True), ("code", False)]


def test_engine_refresh_screen_uses_the_reader_and_the_configured_timeout() -> None:
    cfg = Config()
    cfg.screen.read_timeout_s = 2.5
    eng = app_mod.Engine(cfg, EventBus())
    assert eng._refresh_screen() is False                         # no reader

    class _Reader:
        def __init__(self) -> None:
            self.calls: list[float] = []

        def capture_and_wait(self, timeout_s: float) -> bool:
            self.calls.append(timeout_s)
            return True

    reader = _Reader()
    eng.screen = reader
    assert eng._refresh_screen() is True and reader.calls == [2.5]
    eng.screen = object()                                         # an older reader without the method
    assert eng._refresh_screen() is False

    class _Broken:
        def capture_and_wait(self, timeout_s: float) -> bool:
            raise RuntimeError("boom")

    eng.screen = _Broken()
    assert eng._refresh_screen() is False
