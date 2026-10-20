"""Unit tests for exocortex.screen.reader.ScreenReader.

Offline: a fake grabber returns scripted ``Frame``s and a fake OCR engine returns scripted
``OcrResult``s (injected via the ``grabber_factory``/``engine_factory`` constructor keywords),
against a real ``EventBus`` and a fake clock. No real screen or real OCR engine is touched.
``exocortex.screen.capture`` and ``exocortex.screen.ocr`` are imported directly (not faked)
for their dataclasses/exception: neither module has import-time side effects, and by the time
these tests run both are real, finished modules (see SPEC.md).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np
import pytest

import exocortex.screen.reader as reader_module
from exocortex.bus import EventBus, ScreenStatusEvent, ScreenTextEvent, drain
from exocortex.config import ScreenConfig
from exocortex.screen.capture import Frame
from exocortex.screen.ocr import OcrLine, OcrResult, OcrUnavailable
from exocortex.screen.reader import ScreenReader

# --------------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------------- #


class ScriptedGrabber:
    """Fake ScreenGrabber. ``grab()`` builds a Frame from the mutable ``.image`` /
    ``.source_name`` / ``.rect`` attributes each call (so a test can change ``.image``
    between waits to simulate a screen change), or returns ``None`` while ``.fail`` is set
    (simulating "no capturable window/region")."""

    def __init__(self, image: np.ndarray, *, source_name: str = "Fake Window", rect: tuple[int, int, int, int] = (0, 0, 64, 64)) -> None:
        self.image = image
        self.source_name = source_name
        self.rect = rect
        self.fail = False
        self.calls: list[tuple[str, Any, int]] = []
        self.closed = 0

    def grab(self, mode: str, *, region: tuple[int, int, int, int] | None = None, monitor: int = 0) -> Frame | None:
        self.calls.append((mode, region, monitor))
        if self.fail:
            return None
        return Frame(image=self.image, rect=self.rect, source_name=self.source_name, mode=mode, ts=time.monotonic())

    def close(self) -> None:
        self.closed += 1


class ModeAwareGrabber:
    """Fake grabber whose captured image is byte-identical for every mode/monitor/region, but
    whose ``source_name`` encodes exactly what it was asked for. Used to test that a
    set_mode/set_monitor/set_region switch is reflected promptly even when the old and new
    sources happen to render pixel-identically (so the cheap signature diff alone would not
    notice anything changed)."""

    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.calls: list[tuple[str, Any, int]] = []

    def grab(self, mode: str, *, region: tuple[int, int, int, int] | None = None, monitor: int = 0) -> Frame | None:
        self.calls.append((mode, region, monitor))
        name = f"region:{region}" if mode == "region" else f"{mode}:{monitor}"
        return Frame(image=self.image, rect=(0, 0, 64, 64), source_name=name, mode=mode, ts=time.monotonic())


class ScriptedEngine:
    """Fake OcrEngine. ``recognize()`` returns an OcrResult built from the mutable ``.texts``
    tuple, raising ``RuntimeError`` for the first ``fail_times`` calls (a flaky/erroring OCR
    pass that a later call recovers from)."""

    name = "fake"

    def __init__(self, texts: tuple[str, ...] = ("",), *, fail_times: int = 0, elapsed_ms: float = 5.0) -> None:
        self.texts = texts
        self.fail_times = fail_times
        self.elapsed_ms = elapsed_ms
        self.calls = 0
        self.closed = 0

    def recognize(self, image_bgr: np.ndarray) -> OcrResult:
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("scripted OCR failure")
        lines = tuple(OcrLine(text=t, box=(0, i * 20, 100, 18), confidence=1.0) for i, t in enumerate(self.texts))
        return OcrResult(lines=lines, text="\n".join(self.texts), elapsed_ms=self.elapsed_ms, engine=self.name)

    def close(self) -> None:
        self.closed += 1


class FakeClock:
    """Monotonically increasing fake session clock."""

    def __init__(self) -> None:
        self._t = 0.0

    def __call__(self) -> float:
        self._t += 0.01
        return self._t


def _image(fill: int) -> np.ndarray:
    """A 64x64 solid-colour BGR image (big enough for capture.signature's 32x18 grid)."""
    return np.full((64, 64, 3), fill, dtype=np.uint8)


_BLACK = _image(0)
_WHITE = _image(255)


def _cfg(
    *,
    mode: str = "off",
    interval_s: float = 0.05,
    ocr_engine: str = "auto",
    max_text_chars: int = 6000,
    save_text: bool = False,
) -> ScreenConfig:
    cfg = ScreenConfig()
    cfg.mode = mode
    cfg.interval_s = interval_s
    cfg.ocr_engine = ocr_engine
    cfg.max_text_chars = max_text_chars
    cfg.save_text = save_text
    return cfg


def _wait_until(predicate: Callable[[], bool], timeout: float = 3.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _drain_until(q: Any, want: Callable[[list], bool], timeout: float = 3.0, interval: float = 0.02) -> list:
    """Repeatedly drain `q` into a list until `want(events)` or timeout; returns the list."""
    events: list = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events.extend(drain(q, max_items=10000))
        if want(events):
            return events
        time.sleep(interval)
    events.extend(drain(q, max_items=10000))
    return events


@pytest.fixture
def make_reader():
    """Builds ScreenReader instances and stop()s every one of them at teardown."""
    created: list[ScreenReader] = []

    def _make(cfg: ScreenConfig, bus: EventBus, clock: Callable[[], float] | None = None, **kw: Any) -> ScreenReader:
        r = ScreenReader(cfg, bus, clock or FakeClock(), **kw)
        created.append(r)
        return r

    yield _make
    for r in created:
        try:
            r.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------------- #


def test_start_mode_off_publishes_status_once(make_reader):
    cfg = _cfg(mode="off", interval_s=0.03)
    bus = EventBus()
    q = bus.subscribe()
    reader = make_reader(
        cfg, bus,
        grabber_factory=lambda: ScriptedGrabber(_BLACK.copy()),
        engine_factory=lambda name: ScriptedEngine(("x",)),
    )
    reader.start()
    time.sleep(0.3)  # several intervals worth of ticks while off
    events = drain(q)
    off_events = [e for e in events if isinstance(e, ScreenStatusEvent) and e.state == "off"]
    assert len(off_events) == 1
    assert not any(isinstance(e, ScreenTextEvent) for e in events)


def test_switch_to_auto_publishes_text_and_active_status(make_reader):
    cfg = _cfg(mode="off", interval_s=0.03)
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy(), source_name="Zoom Meeting")
    engine = ScriptedEngine(("hello world",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()
    _drain_until(q, lambda evs: any(isinstance(e, ScreenStatusEvent) and e.state == "off" for e in evs))

    reader.set_mode("auto")
    events = _drain_until(
        q,
        lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs)
        and any(isinstance(e, ScreenStatusEvent) and e.state == "active" for e in evs),
    )

    text_events = [e for e in events if isinstance(e, ScreenTextEvent)]
    assert text_events
    assert text_events[0].text == "hello world"
    assert text_events[0].changed is True
    assert text_events[0].mode == "auto"
    assert text_events[0].source_name == "Zoom Meeting"

    active = [e for e in events if isinstance(e, ScreenStatusEvent) and e.state == "active"]
    assert active
    assert "auto" in active[0].detail and "Zoom Meeting" in active[0].detail


def test_unchanged_frame_produces_no_new_event(make_reader):
    cfg = _cfg(mode="auto", interval_s=0.03)
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    engine = ScriptedEngine(("same text",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()

    events = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    assert len([e for e in events if isinstance(e, ScreenTextEvent)]) == 1

    time.sleep(0.3)  # several more intervals with a bit-identical frame
    more = drain(q)
    assert not any(isinstance(e, ScreenTextEvent) for e in more)


def test_capture_now_on_unchanged_frame_has_changed_false(make_reader):
    cfg = _cfg(mode="auto", interval_s=1.0)  # long interval: only capture_now() should tick
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    engine = ScriptedEngine(("steady text",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()

    first = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    first_text = [e for e in first if isinstance(e, ScreenTextEvent)]
    assert first_text[0].changed is True

    reader.capture_now()
    second = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    second_text = [e for e in second if isinstance(e, ScreenTextEvent)]
    assert second_text
    assert second_text[-1].changed is False
    assert second_text[-1].text == "steady text"


def test_text_truncated_to_max_chars(make_reader):
    long_text = "abcdefghijklmnopqrstuvwxyz"
    cfg = _cfg(mode="auto", interval_s=0.03, max_text_chars=10)
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    engine = ScriptedEngine((long_text,))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()

    events = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    text_events = [e for e in events if isinstance(e, ScreenTextEvent)]
    assert text_events[0].text == long_text[:10]
    assert len(text_events[0].text) == 10


def test_save_text_writes_txt_never_image(make_reader, tmp_path):
    cfg = _cfg(mode="auto", interval_s=0.03, save_text=True)
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    engine = ScriptedEngine(("snapshot me",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.set_snapshot_dir(tmp_path)
    reader.start()

    _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) and e.changed for e in evs))
    screen_dir = tmp_path / "screen"
    assert _wait_until(lambda: screen_dir.is_dir() and any(screen_dir.iterdir()))

    files = list(screen_dir.iterdir())
    assert files
    assert all(f.suffix == ".txt" for f in files)
    assert files[0].read_text(encoding="utf-8") == "snapshot me"

    all_files = list(tmp_path.rglob("*"))
    assert not any(f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp") for f in all_files)


def test_ocr_unavailable_falls_back_to_other_engine_name(make_reader):
    cfg = _cfg(mode="auto", interval_s=0.03, ocr_engine="windows")
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    fallback_engine = ScriptedEngine(("fallback text",))
    requested: list[str] = []

    def factory(name: str) -> ScriptedEngine:
        requested.append(name)
        if name == "windows":
            raise OcrUnavailable("no Windows OCR language pack")
        return fallback_engine

    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=factory)
    reader.start()

    events = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    text_events = [e for e in events if isinstance(e, ScreenTextEvent)]
    assert text_events and text_events[0].text == "fallback text"
    assert requested[:2] == ["windows", "rapidocr"]


def test_engine_exception_reports_error_and_later_tick_recovers(monkeypatch, make_reader):
    monkeypatch.setattr(reader_module, "_ERROR_BACKOFF_S", 0.05)
    cfg = _cfg(mode="auto", interval_s=0.03)
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    engine = ScriptedEngine(("recovered",), fail_times=2)
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()

    error_events = _drain_until(q, lambda evs: any(isinstance(e, ScreenStatusEvent) and e.state == "error" for e in evs))
    assert any(isinstance(e, ScreenStatusEvent) and e.state == "error" for e in error_events)
    assert any(t.name == "screen-reader" and t.is_alive() for t in threading.enumerate())

    later = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs), timeout=5.0)
    text_events = [e for e in later if isinstance(e, ScreenTextEvent)]
    assert text_events and text_events[-1].text == "recovered"


def test_pause_resume_semantics(make_reader):
    cfg = _cfg(mode="auto", interval_s=0.03)
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    engine = ScriptedEngine(("initial",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()

    _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))

    reader.pause()
    assert _wait_until(lambda: reader.paused)
    paused_events = _drain_until(q, lambda evs: any(isinstance(e, ScreenStatusEvent) and e.state == "paused" for e in evs))
    assert any(isinstance(e, ScreenStatusEvent) and e.state == "paused" for e in paused_events)

    grabber.image = _WHITE.copy()  # screen "changes" while paused
    time.sleep(0.2)
    stale = drain(q)
    assert not any(isinstance(e, ScreenTextEvent) for e in stale)

    reader.resume()
    events = _drain_until(
        q,
        lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs)
        and any(isinstance(e, ScreenStatusEvent) and e.state == "active" for e in evs),
    )
    assert any(isinstance(e, ScreenTextEvent) for e in events)
    assert any(isinstance(e, ScreenStatusEvent) and e.state == "active" for e in events)


def test_set_region_switches_mode_and_updates_cfg_region(make_reader):
    cfg = _cfg(mode="auto")
    bus = EventBus()
    reader = make_reader(cfg, bus)

    reader.set_region(10, 20, 30, 40)

    assert reader.mode == "region"
    assert cfg.region == [10, 20, 30, 40]


@pytest.mark.parametrize("w, h", [(0, 10), (10, 0), (-5, 10)])
def test_set_region_rejects_non_positive_size(make_reader, w, h):
    cfg = _cfg()
    bus = EventBus()
    reader = make_reader(cfg, bus)
    with pytest.raises(ValueError):
        reader.set_region(0, 0, w, h)


def test_set_mode_rejects_unknown_value(make_reader):
    cfg = _cfg()
    bus = EventBus()
    reader = make_reader(cfg, bus)
    with pytest.raises(ValueError):
        reader.set_mode("bogus")


def test_grab_returning_none_reports_error_once_and_recovers(make_reader):
    cfg = _cfg(mode="auto", interval_s=0.03)
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    grabber.fail = True
    engine = ScriptedEngine(("recovered text",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()

    events = _drain_until(q, lambda evs: any(isinstance(e, ScreenStatusEvent) and e.state == "error" for e in evs))
    error_events = [e for e in events if isinstance(e, ScreenStatusEvent) and e.state == "error"]
    assert len(error_events) == 1
    assert "no capturable window" in error_events[0].detail

    time.sleep(0.15)  # further failed ticks must not repeat the status
    more = drain(q)
    assert not any(isinstance(e, ScreenStatusEvent) and e.state == "error" for e in more)

    grabber.fail = False
    recovered = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    assert any(isinstance(e, ScreenTextEvent) and e.text == "recovered text" for e in recovered)


def test_stop_leaves_no_screen_reader_thread_alive(make_reader):
    cfg = _cfg(mode="auto", interval_s=0.03)
    bus = EventBus()
    grabber = ScriptedGrabber(_BLACK.copy())
    engine = ScriptedEngine(("x",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()

    assert _wait_until(lambda: any(t.name == "screen-reader" and t.is_alive() for t in threading.enumerate()))

    reader.stop()

    assert not any(t.name == "screen-reader" and t.is_alive() for t in threading.enumerate())


def test_stop_closes_grabber_and_engine(make_reader):
    """Regression test: the worker used to exit without ever calling ScreenGrabber.close() /
    OcrEngine.close(), leaking the underlying mss/WindowsOcr resource on every stop() (and thus
    on every future restart of the same process)."""
    cfg = _cfg(mode="auto", interval_s=0.03)
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    engine = ScriptedEngine(("hello",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()

    _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    assert engine.closed == 0
    assert grabber.closed == 0

    reader.stop()

    assert grabber.closed == 1
    assert engine.closed == 1


def test_stop_close_failure_does_not_raise(make_reader):
    """A grabber/engine whose close() itself raises must not make stop() raise."""
    cfg = _cfg(mode="auto", interval_s=0.03)
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    grabber.close = lambda: (_ for _ in ()).throw(RuntimeError("mss close boom"))  # type: ignore[method-assign]
    engine = ScriptedEngine(("hello",))
    engine.close = lambda: (_ for _ in ()).throw(RuntimeError("engine close boom"))  # type: ignore[method-assign]
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()

    _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    reader.stop()  # must not raise


@pytest.mark.parametrize("switch", ["mode", "monitor", "region"])
def test_source_switch_refreshes_even_with_pixel_identical_image(make_reader, switch):
    """Regression test: set_mode()/set_monitor()/set_region() used to only wake the worker, not
    force it -- so if the newly grabbed frame happened to look pixel-identical to the previous
    one (same signature), the cheap "skip if unchanged" check would swallow the switch entirely:
    no new ScreenTextEvent, no new "active" ScreenStatusEvent, and latest_event/mode's
    source_name would stay stale. SPEC.md requires a ScreenStatusEvent "on mode/source ...
    changes", which must not depend on the two sources happening to render differently."""
    cfg = _cfg(mode="monitor", interval_s=1.0)  # long interval: only a forced switch should tick
    bus = EventBus()
    q = bus.subscribe()
    grabber = ModeAwareGrabber(_BLACK.copy())
    engine = ScriptedEngine(("same text",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()

    first = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    first_sources = {e.source_name for e in first if isinstance(e, ScreenTextEvent)}
    assert first_sources == {"monitor:0"}

    if switch == "mode":
        reader.set_mode("auto")
        want_source = "auto:0"
    elif switch == "monitor":
        reader.set_monitor(1)
        want_source = "monitor:1"
    else:
        reader.set_region(1, 2, 3, 4)
        want_source = "region:(1, 2, 3, 4)"

    events = _drain_until(
        q,
        lambda evs: any(isinstance(e, ScreenTextEvent) and e.source_name == want_source for e in evs)
        and any(isinstance(e, ScreenStatusEvent) and e.state == "active" and want_source in e.detail for e in evs),
        timeout=2.0,
    )
    text_events = [e for e in events if isinstance(e, ScreenTextEvent)]
    assert text_events and text_events[-1].source_name == want_source

    active = [e for e in events if isinstance(e, ScreenStatusEvent) and e.state == "active"]
    assert any(want_source in e.detail for e in active)
    assert reader.latest_event is not None and reader.latest_event.source_name == want_source


@pytest.mark.live
def test_live_screen_reader_end_to_end_capture(make_reader):
    """Real mss + real OCR engine (no factories injected): one monitor capture, one OCR pass,
    then a clean stop() -- the smoke test SPEC.md's "screen-share-invisible meeting co-pilot"
    promise ultimately rests on."""
    cfg = _cfg(mode="monitor", interval_s=0.5)
    bus = EventBus()
    q = bus.subscribe()
    reader = make_reader(cfg, bus)
    reader.start()

    events = _drain_until(
        q,
        lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs)
        or any(isinstance(e, ScreenStatusEvent) and e.state == "error" for e in evs),
        timeout=15.0,
    )
    assert events, "screen-reader produced no event at all within 15s"
    errors = [e for e in events if isinstance(e, ScreenStatusEvent) and e.state == "error"]
    assert not errors, f"screen-reader reported error(s): {[e.detail for e in errors]}"

    reader.stop()
    assert not any(t.name == "screen-reader" and t.is_alive() for t in threading.enumerate())
