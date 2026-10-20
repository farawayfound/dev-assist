"""Tests for the `window_title` / `window_process` fields added to `ScreenTextEvent`
(`ScreenReader._foreground_info`, called once per tick and spliced into the published event).

Offline, same fakes as tests/test_screen_reader.py (a scripted grabber + a scripted OCR engine
against a real EventBus and a fake clock); `ScriptedGrabber`/`ScriptedEngine`/`FakeClock`/
`make_reader`/`_drain_until` are copied from that file rather than imported, so this file does
not depend on another test module's private internals staying stable (see
tests/test_assistant_screen.py's module docstring for the same reasoning).

`ScreenReader._foreground_info()` never raises to its caller -- it wraps the one call it makes
(`exocortex.screen.foreground.foreground_window`) in its own `try/except` and already returns
``("", "")`` for `None` or an error (see its docstring: "never a reason to skip the OCR"). So the
"None-ish / raises" half of the contract is exercised at its real source -- by monkeypatching
`foreground_window` itself -- rather than by replacing `_foreground_info` (which `_tick` calls
with no `try/except` of its own: stubbing `_foreground_info` to raise would abort that tick
before OCR ever ran, which is not what "must never fail the OCR tick" is about here). The
"returns a real window" half is exercised the same way, for one consistent code path across all
three cases.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pytest

import exocortex.screen.foreground as foreground_module
import exocortex.screen.reader as reader_module
from exocortex.bus import EventBus, ScreenTextEvent, drain
from exocortex.config import ScreenConfig
from exocortex.screen.capture import Frame
from exocortex.screen.foreground import WindowInfo
from exocortex.screen.ocr import OcrLine, OcrResult
from exocortex.screen.reader import ScreenReader

# --------------------------------------------------------------------------------- #
# fakes (copied from tests/test_screen_reader.py)
# --------------------------------------------------------------------------------- #


class ScriptedGrabber:
    def __init__(self, image: np.ndarray, *, source_name: str = "Fake Window",
                 rect: tuple[int, int, int, int] = (0, 0, 64, 64)) -> None:
        self.image = image
        self.source_name = source_name
        self.rect = rect

    def grab(self, mode: str, *, region: tuple[int, int, int, int] | None = None, monitor: int = 0) -> Frame | None:
        return Frame(image=self.image, rect=self.rect, source_name=self.source_name, mode=mode, ts=time.monotonic())

    def close(self) -> None:
        pass


class ScriptedEngine:
    name = "fake"

    def __init__(self, texts: tuple[str, ...] = ("",), *, elapsed_ms: float = 5.0) -> None:
        self.texts = texts
        self.elapsed_ms = elapsed_ms

    def recognize(self, image_bgr: np.ndarray) -> OcrResult:
        lines = tuple(OcrLine(text=t, box=(0, i * 20, 100, 18), confidence=1.0) for i, t in enumerate(self.texts))
        return OcrResult(lines=lines, text="\n".join(self.texts), elapsed_ms=self.elapsed_ms, engine=self.name)

    def close(self) -> None:
        pass


class FakeClock:
    def __init__(self) -> None:
        self._t = 0.0

    def __call__(self) -> float:
        self._t += 0.01
        return self._t


def _image(fill: int) -> np.ndarray:
    return np.full((64, 64, 3), fill, dtype=np.uint8)


def _cfg(*, mode: str = "auto", interval_s: float = 0.03) -> ScreenConfig:
    cfg = ScreenConfig()
    cfg.mode = mode
    cfg.interval_s = interval_s
    return cfg


def _drain_until(q: Any, want: Callable[[list], bool], timeout: float = 3.0, interval: float = 0.02) -> list:
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


def _first_text_event(events: list) -> ScreenTextEvent:
    return next(e for e in events if isinstance(e, ScreenTextEvent))


# --------------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------------- #


def test_screen_text_event_carries_the_foreground_window(make_reader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        foreground_module, "foreground_window",
        lambda: WindowInfo(hwnd=1, title="main.py - Visual Studio Code", process_name="Code.exe", rect=(0, 0, 800, 600)),
    )
    bus = EventBus()
    q = bus.subscribe()
    reader = make_reader(
        _cfg(), bus,
        grabber_factory=lambda: ScriptedGrabber(_image(0)),
        engine_factory=lambda name: ScriptedEngine(("def solve():",)),
    )
    reader.start()
    events = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    event = _first_text_event(events)
    assert event.window_title == "main.py - Visual Studio Code"
    assert event.window_process == "Code.exe"


def test_screen_text_event_window_blank_when_foreground_window_returns_none(
    make_reader, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(foreground_module, "foreground_window", lambda: None)
    bus = EventBus()
    q = bus.subscribe()
    reader = make_reader(
        _cfg(), bus,
        grabber_factory=lambda: ScriptedGrabber(_image(0)),
        engine_factory=lambda name: ScriptedEngine(("hello",)),
    )
    reader.start()
    events = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    event = _first_text_event(events)
    assert event.window_title == ""
    assert event.window_process == ""


def test_screen_text_event_window_blank_when_foreground_window_raises(
    make_reader, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_foreground_info`'s own try/except must swallow this -- the OCR tick still runs and the
    event still gets published, just with blank window fields (never a reason to skip the OCR)."""

    def _boom() -> WindowInfo | None:
        raise RuntimeError("no foreground window handle available")

    monkeypatch.setattr(foreground_module, "foreground_window", _boom)
    bus = EventBus()
    q = bus.subscribe()
    reader = make_reader(
        _cfg(), bus,
        grabber_factory=lambda: ScriptedGrabber(_image(0)),
        engine_factory=lambda name: ScriptedEngine(("hello",)),
    )
    reader.start()
    events = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    event = _first_text_event(events)
    assert event.window_title == ""
    assert event.window_process == ""


def test_foreground_info_itself_returns_blank_tuple_on_none_or_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct unit check of the staticmethod (no reader/OCR machinery involved)."""
    monkeypatch.setattr(foreground_module, "foreground_window", lambda: None)
    assert reader_module.ScreenReader._foreground_info() == ("", "")

    def _boom() -> WindowInfo | None:
        raise OSError("boom")

    monkeypatch.setattr(foreground_module, "foreground_window", _boom)
    assert reader_module.ScreenReader._foreground_info() == ("", "")

    monkeypatch.setattr(
        foreground_module, "foreground_window",
        lambda: WindowInfo(hwnd=2, title="t", process_name="p.exe", rect=(0, 0, 1, 1)),
    )
    assert reader_module.ScreenReader._foreground_info() == ("t", "p.exe")
