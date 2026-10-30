"""`ScreenReader.capture_and_wait` (2026-10-29, read-on-shortcut): a forced pass the caller can block
on -- true once the worker ran the pass that saw the request (text published, or nothing to publish
while paused), false at once when the worker is not running, false on timeout, and a `stop()` wakes a
waiter. Reuses tests/test_screen_reader.py's fakes and fixture (no real screen, no real OCR)."""
from __future__ import annotations

import threading
import time

from exocortex.bus import EventBus, ScreenTextEvent
from tests.test_screen_reader import (  # noqa: F401 - make_reader is a fixture
    _BLACK,
    _WHITE,
    ScriptedEngine,
    ScriptedGrabber,
    _cfg,
    _drain_until,
    make_reader,
)


class SlowGrabber(ScriptedGrabber):
    """A grabber whose every grab takes ``delay_s`` (a stuck or slow capture)."""

    def __init__(self, image, *, delay_s: float) -> None:
        super().__init__(image)
        self.delay_s = delay_s

    def grab(self, mode, *, region=None, monitor=0):
        time.sleep(self.delay_s)
        return super().grab(mode, region=region, monitor=monitor)


def _first_event(q) -> None:
    _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))


def test_capture_and_wait_returns_once_the_forced_pass_published_the_fresh_text(make_reader) -> None:
    cfg = _cfg(mode="auto", interval_s=10.0)          # long interval: only the forced pass ticks
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    engine = ScriptedEngine(("first screen",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()
    _first_event(q)
    assert reader.latest_text == "first screen"
    engine.texts = ("the file the user is on now",)
    grabber.image = _WHITE.copy()
    t0 = time.monotonic()
    assert reader.capture_and_wait(3.0) is True
    assert time.monotonic() - t0 < 2.0
    assert reader.latest_text == "the file the user is on now"
    events = [e for e in _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
              if isinstance(e, ScreenTextEvent)]
    assert events and events[-1].text == "the file the user is on now" and events[-1].changed is True


def test_capture_and_wait_on_an_unchanged_screen_still_completes(make_reader) -> None:
    cfg = _cfg(mode="auto", interval_s=10.0)
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    engine = ScriptedEngine(("steady",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()
    _first_event(q)
    assert reader.capture_and_wait(3.0) is True        # forced: OCR ran, changed=False published
    assert reader.capture_and_wait(3.0) is True        # and again
    assert reader.latest_text == "steady"


def test_capture_and_wait_is_false_at_once_when_the_worker_is_not_running(make_reader) -> None:
    reader = make_reader(_cfg(mode="auto"), EventBus())
    t0 = time.monotonic()
    assert reader.capture_and_wait(5.0) is False
    assert time.monotonic() - t0 < 0.5


def test_capture_and_wait_returns_promptly_while_paused_or_off(make_reader) -> None:
    cfg = _cfg(mode="auto", interval_s=10.0)
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    engine = ScriptedEngine(("text",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()
    _first_event(q)
    reader.pause()
    t0 = time.monotonic()
    assert reader.capture_and_wait(3.0) is True        # the pass ran (and published "paused"), nothing new
    assert time.monotonic() - t0 < 2.0
    assert reader.latest_text == "text"
    reader.resume()
    reader.set_mode("off")
    assert reader.capture_and_wait(3.0) is True


def test_capture_and_wait_times_out_on_a_slow_pass_and_a_later_call_catches_up(make_reader) -> None:
    cfg = _cfg(mode="auto", interval_s=10.0)
    bus = EventBus()
    q = bus.subscribe()
    grabber = SlowGrabber(_BLACK.copy(), delay_s=0.6)
    engine = ScriptedEngine(("slow",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()
    _first_event(q)
    assert reader.capture_and_wait(0.1) is False       # the pass is still grabbing
    assert reader.capture_and_wait(3.0) is True        # a later request is served by a later pass


def test_stop_wakes_a_waiting_caller(make_reader) -> None:
    cfg = _cfg(mode="auto", interval_s=10.0)
    bus = EventBus()
    q = bus.subscribe()
    grabber = SlowGrabber(_BLACK.copy(), delay_s=0.8)
    engine = ScriptedEngine(("slow",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()
    _first_event(q)
    results: list[bool] = []
    waiter = threading.Thread(target=lambda: results.append(reader.capture_and_wait(10.0)), daemon=True)
    waiter.start()
    time.sleep(0.1)
    t0 = time.monotonic()
    reader.stop()
    waiter.join(timeout=5.0)
    assert not waiter.is_alive() and time.monotonic() - t0 < 4.0
    assert results and results[0] in (True, False)      # released by the stop (or by the pass finishing)
