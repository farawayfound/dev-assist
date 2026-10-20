"""One focused case split out of tests/test_manual_advance.py: a scheduler that runs early (Qt's
coarse timers can fire up to ~5 % ahead of time) must re-schedule the remainder of the hold, not
fire short and not give up."""
from __future__ import annotations

from tests.test_manual_advance import _make_hook, _send
from exocortex.ui import llhooks


def test_early_scheduler_reschedules_the_remainder_instead_of_firing_short() -> None:
    hook, fake, hook_clock, scheduler, fired = _make_hook(hold_s=0.8)
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    hook_clock.t += 0.76                          # the coarse timer fired ~5 % early
    scheduler.run_last()
    assert fired["text"] == 0                     # not fired short...
    assert len(scheduler.scheduled) == 2          # ...but re-armed for the remainder
    delay = scheduler.scheduled[-1][0]
    assert 0.02 <= delay <= 0.05
    hook_clock.t += delay
    scheduler.run_last()
    assert fired["text"] == 1                     # fires at the true threshold, still held


def test_early_scheduler_reschedule_is_still_cancelled_by_release() -> None:
    hook, fake, hook_clock, scheduler, fired = _make_hook(hold_s=0.8)
    _send(fake, llhooks.WM_LBUTTONDOWN, 10, 10)
    hook_clock.t += 0.5
    scheduler.run_last()                          # very early: re-schedules
    _send(fake, llhooks.WM_LBUTTONUP, 10, 10)     # released before the threshold
    hook_clock.t += 1.0
    scheduler.run_last()                          # the re-scheduled check: stale generation
    assert fired["text"] == 0
