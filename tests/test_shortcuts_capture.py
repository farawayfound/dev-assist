"""2026-10-29: the mouse hook's new reach -- every mouse-bound shortcut: the thumb (X) buttons, the
right button, modifiers read at the press, the per-binding swallow rule, rebinding in place -- and
the Shortcuts tab's capture core, `ChordRecorder`, with its `CaptureKeyHook` / `CaptureMouseHook`
adapters. Everything is driven through tests/test_llhooks.py's fake user32 (the captured hook
procedure is called with fake MSLLHOOKSTRUCT / KBDLLHOOKSTRUCTs), an injectable clock and a
hand-run scheduler. No Qt, no real hook anywhere."""
from __future__ import annotations

import ctypes
from typing import Any

import pytest

from exocortex.ui import llhooks
from tests.test_llhooks import FakeUser32
from tests.test_manual_advance import HookClock, ManualScheduler, _send


def _xsend(fake: FakeUser32, down: bool, which: int, x: int = 0, y: int = 0) -> int:
    struct = llhooks.MSLLHOOKSTRUCT(pt=llhooks.POINT(x, y), mouseData=(which & 0xFFFF) << 16,
                                    flags=0, time=0, dwExtraInfo=0)
    msg = llhooks.WM_XBUTTONDOWN if down else llhooks.WM_XBUTTONUP
    return int(fake.proc(0, msg, ctypes.addressof(struct)))


def _ksend(fake: FakeUser32, down: bool, vk: int) -> int:
    struct = llhooks.KBDLLHOOKSTRUCT(vkCode=vk, scanCode=0, flags=0, time=0, dwExtraInfo=0)
    return int(fake.proc(0, llhooks.WM_KEYDOWN if down else llhooks.WM_KEYUP, ctypes.addressof(struct)))


def _hook(bindings: dict[str, Any], **kw: Any):
    fake = kw.pop("fake", None) or FakeUser32(call_next=7)
    clock = HookClock()
    sched = ManualScheduler()
    hook = llhooks.PressHoldMouseHook(bindings, user32=fake, clock=clock, schedule=kw.pop("schedule", sched), **kw)
    assert hook.install() is True
    return hook, fake, clock, sched


# =====================================================================================
# PressHoldMouseHook: thumb buttons, right button, modifiers, swallow, rebind
# =====================================================================================


def test_mouse4_click_fires_and_is_swallowed_when_asked() -> None:
    fired: list[str] = []
    hook, fake, clock, _sched = _hook({"mouse4": lambda: fired.append("fresh")}, require_hold=False,
                                      swallow=lambda text: text == "mouse4")
    assert _xsend(fake, True, llhooks.XBUTTON1) == 1       # swallowed: the browser never sees the Back press
    clock.t += 0.05
    assert _xsend(fake, False, llhooks.XBUTTON1) == 1      # ... nor its release
    assert fired == ["fresh"] and hook.fires == 1
    assert hook.bindings == ("mouse4",)


def test_mouse5_fires_and_unbound_x_buttons_pass_through() -> None:
    fired: list[int] = []
    _hook_, fake, clock, _sched = _hook({"mouse5": lambda: fired.append(5)}, require_hold=False)
    assert _xsend(fake, True, llhooks.XBUTTON2) == 7       # not swallowed by default
    clock.t += 0.05
    assert _xsend(fake, False, llhooks.XBUTTON2) == 7
    assert fired == [5]
    assert _xsend(fake, True, llhooks.XBUTTON1) == 7       # mouse4 is not bound: passes, arms nothing
    assert _xsend(fake, False, llhooks.XBUTTON1) == 7
    assert _xsend(fake, True, 0) == 7                      # an X message naming no button
    assert fired == [5]


def test_an_unknown_x_button_still_cancels_a_press_in_flight() -> None:
    fired: list[str] = []
    _hook_, fake, clock, _sched = _hook({"mouse_left": lambda: fired.append("text")}, require_hold=False)
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    _xsend(fake, True, 0)                                  # some other button mid-press: a chord
    clock.t += 0.05
    _send(fake, llhooks.WM_LBUTTONUP, 10, 10)
    assert fired == []


def test_right_button_can_be_bound_and_passes_through_by_default() -> None:
    fired: list[str] = []
    _hook_, fake, clock, _sched = _hook({"mouse_right": lambda: fired.append("r")}, require_hold=False)
    assert _send(fake, llhooks.WM_RBUTTONDOWN, 1, 1) == 7
    clock.t += 0.05
    assert _send(fake, llhooks.WM_RBUTTONUP, 1, 1) == 7
    assert fired == ["r"]


def test_modifiers_must_match_exactly() -> None:
    fired: list[str] = []
    fake = FakeUser32(call_next=7)
    _hook_, fake, clock, _sched = _hook(
        {"ctrl+mouse_left": lambda: fired.append("ctrl"), "mouse_left": lambda: fired.append("plain")},
        fake=fake, require_hold=False)

    def click() -> None:
        _send(fake, llhooks.WM_LBUTTONDOWN, 1, 1)
        clock.t += 0.05
        _send(fake, llhooks.WM_LBUTTONUP, 1, 1)

    click()
    assert fired == ["plain"]
    fake._async_state[llhooks.VK_CONTROL] = 0x8000
    click()
    assert fired == ["plain", "ctrl"]
    fake._async_state[llhooks.VK_SHIFT] = 0x8000           # ctrl+shift+click: no such binding
    click()
    assert fired == ["plain", "ctrl"]
    fake._async_state.clear()
    fake._async_state[llhooks.VK_LWIN] = 0x8000            # win+click: none either
    click()
    assert fired == ["plain", "ctrl"]


def test_swallow_is_decided_per_binding_at_the_press() -> None:
    fired: list[str] = []
    fake = FakeUser32(call_next=7)
    _hook_, fake, clock, _sched = _hook(
        {"ctrl+mouse_left": lambda: fired.append("ctrl"), "mouse_left": lambda: fired.append("plain")},
        fake=fake, require_hold=False, swallow=lambda text: "+" in text)
    assert _send(fake, llhooks.WM_LBUTTONDOWN, 1, 1) == 7  # a bare left press always reaches the app
    assert _send(fake, llhooks.WM_LBUTTONUP, 1, 1) == 7
    fake._async_state[llhooks.VK_CONTROL] = 0x8000
    assert _send(fake, llhooks.WM_LBUTTONDOWN, 1, 1) == 1  # a modified click is a hotkey: consumed
    fake._async_state.clear()                              # ctrl released before the button: the up is
    assert _send(fake, llhooks.WM_LBUTTONUP, 1, 1) == 1    # ... still swallowed, as decided at the press
    assert fired == ["plain", "ctrl"]


def test_a_cancelled_swallowed_press_swallows_its_release_too() -> None:
    _hook_, fake, clock, _sched = _hook({"mouse4": lambda: None}, require_hold=True, swallow=True)
    assert _xsend(fake, True, llhooks.XBUTTON1, 10, 10) == 1
    _send(fake, llhooks.WM_MOUSEMOVE, 60, 60)              # a drag: cancelled
    assert _xsend(fake, False, llhooks.XBUTTON1, 60, 60) == 1   # the app never sees half a click


def test_rebind_replaces_the_bindings_in_place_and_forgets_a_press() -> None:
    fired: list[str] = []
    hook, fake, clock, _sched = _hook({"mouse_middle": lambda: fired.append("m")}, require_hold=False)
    _send(fake, llhooks.WM_MBUTTONDOWN, 1, 1)              # armed...
    hook.rebind({"x1": lambda: fired.append("fresh")})     # ... the table changes under it
    assert hook.bindings == ("mouse4",)                    # aliases are stored canonically
    clock.t += 0.05
    _send(fake, llhooks.WM_MBUTTONUP, 1, 1)
    assert fired == []                                     # the forgotten press never fires
    _xsend(fake, True, llhooks.XBUTTON1)
    clock.t += 0.05
    _xsend(fake, False, llhooks.XBUTTON1)
    assert fired == ["fresh"]
    assert hook.installed and fake.unhooked == []          # never re-installed


@pytest.mark.parametrize("bad", [
    {}, {"right": lambda: None}, {"f9": lambda: None}, {"ctrl+alt+g": lambda: None},
    {"mouse4": lambda: None, "x1": lambda: None}, {"nope": lambda: None},
])
def test_non_mouse_or_duplicate_bindings_are_rejected(bad: dict) -> None:
    with pytest.raises(ValueError):
        llhooks.PressHoldMouseHook(bad)
    hook = llhooks.PressHoldMouseHook({"mouse4": lambda: None}, user32=FakeUser32())
    with pytest.raises(ValueError):
        hook.rebind(bad)
    assert hook.bindings == ("mouse4",)                    # a rejected rebind leaves the table alone


def test_require_hold_and_swallow_callables_receive_the_canonical_text() -> None:
    seen: list[tuple[str, str]] = []
    hook, fake, clock, _sched = _hook(
        {"button4": lambda: None},
        require_hold=lambda text: seen.append(("hold", text)) or False,
        swallow=lambda text: seen.append(("swallow", text)) or False)
    _xsend(fake, True, llhooks.XBUTTON1)
    clock.t += 0.05
    _xsend(fake, False, llhooks.XBUTTON1)
    assert seen == [("swallow", "mouse4"), ("hold", "mouse4")]
    assert hook.fires == 1


def test_a_required_hold_still_fires_once_at_the_threshold_with_the_new_tokens() -> None:
    fired: list[str] = []
    _hook_, fake, clock, sched = _hook({"mouse_left": lambda: fired.append("text")}, hold_s=0.8)
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    clock.t += 0.8
    sched.run_last()
    assert fired == ["text"]
    _send(fake, llhooks.WM_LBUTTONUP, 10, 10)
    assert fired == ["text"]                               # not twice


# =====================================================================================
# ChordRecorder
# =====================================================================================


def _recorder(**kw: Any):
    done: list[str | None] = []
    clock = HookClock()
    sched = ManualScheduler()
    rec = llhooks.ChordRecorder(done.append, quiet_s=0.5, schedule=kw.pop("schedule", sched), clock=clock, **kw)
    rec.start()
    return rec, done, clock, sched


def test_a_combination_is_taken_half_a_second_after_the_last_key_once_released() -> None:
    rec, done, clock, sched = _recorder()
    assert rec.listening
    assert rec.press("lctrl") and rec.press("lalt") and rec.press("g")
    assert rec.release("g") and rec.release("lalt") and rec.release("lctrl")
    assert done == []                                      # not yet: the quiet time
    clock.t += 0.2
    sched.run_last()
    assert done == []                                      # ran early: re-scheduled
    clock.t += 0.4
    sched.run_last()
    assert done == ["ctrl+alt+g"] and rec.result == "ctrl+alt+g" and not rec.listening
    assert rec.press("x") is False                         # finished: nothing more is taken


def test_a_modifier_still_held_delays_the_decision_until_its_release() -> None:
    rec, done, clock, sched = _recorder()
    rec.press("lctrl")
    clock.t += 0.6
    sched.run_last()                                       # quiet, but Ctrl is still down: wait
    assert done == [] and rec.listening
    rec.press("g")
    rec.release("g")
    rec.release("lctrl")
    assert done == []
    clock.t += 0.5
    sched.run_last()
    assert done == ["ctrl+g"]


def test_a_modifier_on_its_own_becomes_a_tap() -> None:
    rec, done, clock, sched = _recorder()
    rec.press("rshift")
    rec.release("rshift")
    clock.t += 0.5
    sched.run_last()
    assert done == ["rshift"]


def test_a_single_key_becomes_a_tap_and_a_mouse_button_a_mouse_binding() -> None:
    rec, done, clock, sched = _recorder()
    rec.press("f9")
    rec.release("f9")
    clock.t += 0.5
    sched.run_last()
    assert done == ["f9"]
    rec2, done2, clock2, sched2 = _recorder()
    rec2.press("mouse5")
    rec2.release("mouse5")
    clock2.t += 0.5
    sched2.run_last()
    assert done2 == ["mouse5"]


def test_a_mouse_button_with_a_held_modifier() -> None:
    rec, done, clock, sched = _recorder()
    rec.press("lctrl")
    rec.press("mouse4")
    rec.release("mouse4")
    rec.release("lctrl")
    clock.t += 0.5
    sched.run_last()
    assert done == ["ctrl+mouse4"]


def test_the_release_of_the_last_held_key_after_the_quiet_time_finishes_at_once() -> None:
    rec, done, clock, _sched = _recorder()
    rec.press("lctrl")
    rec.press("g")
    rec.release("g")
    clock.t += 0.7                                         # Ctrl kept down past the quiet time
    rec.release("lctrl")
    assert done == ["ctrl+g"]


def test_esc_cancels_and_reports_none() -> None:
    rec, done, _clock, _sched = _recorder()
    rec.press("lctrl")
    assert rec.press("esc") is True
    assert done == [None] and rec.result is None and not rec.listening
    assert rec.press("g") is False and rec.release("g") is False


def test_cancel_then_start_again() -> None:
    rec, done, clock, sched = _recorder()
    rec.cancel()
    assert done == [None]
    rec.cancel()                                           # idempotent
    assert done == [None]
    rec.start()
    rec.press("f9")
    rec.release("f9")
    clock.t += 0.5
    sched.run_last()
    assert done == [None, "f9"]


def test_finish_now_without_a_scheduler() -> None:
    done: list[str | None] = []
    rec = llhooks.ChordRecorder(done.append, schedule=None)
    rec.start()
    rec.press("lshift")
    rec.press("a")
    rec.finish_now()
    assert done == ["shift+a"]
    rec.finish_now()                                       # a no-op once finished
    assert done == ["shift+a"]


def test_a_stale_scheduled_check_is_ignored() -> None:
    rec, done, clock, sched = _recorder()
    rec.press("a")
    stale = sched.scheduled[-1][1]
    rec.press("b")                                         # a later press: the first check is stale
    clock.t += 0.6
    stale()
    assert done == []                                      # (it would have finished with "a")
    sched.run_last()
    assert done == ["b"]


# =====================================================================================
# the capture hooks
# =====================================================================================


def test_capture_key_hook_feeds_and_swallows_only_while_listening() -> None:
    done: list[str | None] = []
    clock, sched = HookClock(), ManualScheduler()
    rec = llhooks.ChordRecorder(done.append, schedule=sched, clock=clock)
    fake = FakeUser32(call_next=7)
    hook = llhooks.CaptureKeyHook(rec, user32=fake)
    assert hook.install()
    assert _ksend(fake, True, 0xA2) == 7                   # not listening: passes through
    rec.start()
    assert _ksend(fake, True, 0xA2) == 1                   # VK_LCONTROL
    assert _ksend(fake, True, ord("G")) == 1
    assert _ksend(fake, False, ord("G")) == 1
    assert _ksend(fake, False, 0xA2) == 1
    assert _ksend(fake, True, 0x90) == 7                   # Num Lock: no name, passes through
    clock.t += 0.5
    sched.run_last()
    assert done == ["ctrl+g"]
    assert _ksend(fake, True, ord("G")) == 7               # finished: passes again
    hook.uninstall()
    assert fake.unhooked == [1]


def test_capture_key_hook_esc_cancels() -> None:
    done: list[str | None] = []
    rec = llhooks.ChordRecorder(done.append, schedule=ManualScheduler())
    fake = FakeUser32(call_next=7)
    assert llhooks.CaptureKeyHook(rec, user32=fake).install()
    rec.start()
    assert _ksend(fake, True, 0x1B) == 1                   # Esc: swallowed, cancels
    assert done == [None]


def test_capture_mouse_hook_ignores_clicks_over_the_dialog_and_swallows_the_rest() -> None:
    done: list[str | None] = []
    clock, sched = HookClock(), ManualScheduler()
    rec = llhooks.ChordRecorder(done.append, schedule=sched, clock=clock)
    fake = FakeUser32(call_next=7)
    hook = llhooks.CaptureMouseHook(rec, ignore_at=lambda x, y: x < 100, user32=fake)
    assert hook.install()
    rec.start()
    assert _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10) == 7   # over the dialog: pass, not recorded
    assert _send(fake, llhooks.WM_LBUTTONUP, 10, 10) == 7
    assert rec.listening and done == []
    assert _xsend(fake, True, llhooks.XBUTTON1, 500, 500) == 1   # elsewhere: taken and swallowed
    assert _xsend(fake, False, llhooks.XBUTTON1, 500, 500) == 1
    clock.t += 0.5
    sched.run_last()
    assert done == ["mouse4"]
    assert _xsend(fake, True, llhooks.XBUTTON1, 500, 500) == 7   # finished: passes
    assert _send(fake, llhooks.WM_MOUSEMOVE, 500, 500) == 7      # never anything but buttons


def test_capture_hooks_install_both_and_uninstall_both() -> None:
    rec = llhooks.ChordRecorder(lambda _t: None, schedule=ManualScheduler())
    fake = FakeUser32()
    hooks = llhooks.CaptureHooks(rec, user32=fake)
    assert hooks.install() is True and hooks.installed
    assert hooks.keys.installed and hooks.mouse.installed
    hooks.uninstall()
    assert not hooks.installed and fake.unhooked == [1, 1]


def test_capture_hooks_report_failure_when_the_keyboard_hook_cannot_install() -> None:
    rec = llhooks.ChordRecorder(lambda _t: None, schedule=ManualScheduler())
    hooks = llhooks.CaptureHooks(rec, user32=FakeUser32(handle=0))
    assert hooks.install() is False and not hooks.installed
