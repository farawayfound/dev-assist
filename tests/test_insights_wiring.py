"""Wiring around the pair-programming co-pilot (2026-10-20): the screen reader blanking this app's
own (now capturable) windows before OCR, the session folder's insights.md, and the Engine's
insights / recap actions (deferred while the LLM starts) and setters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from exocortex.bus import EventBus, ScreenTextEvent, drain
from exocortex.config import Config, ScreenConfig
from exocortex.screen import foreground
from exocortex.screen.capture import Frame, mask_rects
from exocortex.screen.ocr import OcrLine, OcrResult
from exocortex.screen.reader import ScreenReader
from exocortex.session import SessionRecorder
from tests.test_session import FakeAssistant, FakeTranscript


# --------------------------------------------------------------------------------------------
# the own-window mask
# --------------------------------------------------------------------------------------------


def _frame() -> Frame:
    """A white 200x100 frame captured at screen position (1000, 500)."""
    return Frame(image=np.full((100, 200, 3), 255, dtype=np.uint8), rect=(1000, 500, 200, 100),
                 source_name="editor", mode="auto", ts=0.0)


def test_mask_rects_clips_shifts_and_ignores_rects_off_the_frame() -> None:
    frame = _frame()
    assert mask_rects(frame, [(1150, 450, 100, 100), (0, 0, 10, 10), (1000, 590, 5, 50)]) == 2
    black = (frame.image == 0).all(axis=2)
    assert black[0:50, 150:200].all() and black[90:100, 0:5].all()
    assert int(black.sum()) == 50 * 50 + 10 * 5


def test_own_window_rects_skips_minimized_and_rectless_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(foreground, "_own_windows_raw", lambda: [1, 2, 3])
    monkeypatch.setattr(foreground, "_is_minimized", lambda hwnd: hwnd == 2)
    monkeypatch.setattr(foreground, "_window_rect", lambda hwnd: None if hwnd == 3 else (10, 20, 30, 40))
    assert foreground.own_window_rects() == [(10, 20, 30, 40)]


class _Grabber:
    def __init__(self, frame: Frame) -> None:
        self.frame = frame

    def grab(self, mode: str, *, region: Any = None, monitor: int = 0) -> Frame:
        return self.frame


class _Engine:
    name = "fake"

    def __init__(self) -> None:
        self.images: list[np.ndarray] = []

    def recognize(self, image: np.ndarray) -> OcrResult:
        self.images.append(image.copy())
        line = OcrLine(text="def export(tenant):", box=(0, 0, 10, 10), confidence=0.9)
        return OcrResult(lines=(line,), text=line.text, elapsed_ms=1.0, engine="fake")


def _tick(cfg: ScreenConfig, engine: _Engine) -> list[Any]:
    bus = EventBus()
    sub = bus.subscribe()
    reader = ScreenReader(cfg, bus, lambda: 1.0)
    reader._tick(_Grabber(_frame()), lambda name: engine, lambda img: float(img.mean()), lambda a, b: True,
                 lambda lines: "\n".join(line.text for line in lines), RuntimeError)
    return drain(sub)


def test_the_reader_blanks_this_apps_windows_before_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(foreground, "own_window_rects", lambda: [(1100, 500, 100, 100)])   # the overlay
    cfg = ScreenConfig()
    cfg.mode = "auto"
    engine = _Engine()
    events = _tick(cfg, engine)
    assert engine.images, "OCR never ran"
    for image in engine.images:
        assert (image[:, 100:200] == 0).all() and (image[:, :100] == 255).all()
    assert any(isinstance(e, ScreenTextEvent) for e in events)


def test_the_reader_reads_the_frame_as_captured_with_the_exclusion_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(foreground, "own_window_rects", lambda: [(1100, 500, 100, 100)])
    cfg = ScreenConfig()
    cfg.mode = "auto"
    cfg.exclude_own_windows = False
    engine = _Engine()
    _tick(cfg, engine)
    assert engine.images and all((image == 255).all() for image in engine.images)


def test_a_failing_mask_never_breaks_screen_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> list:
        raise OSError("EnumWindows went away")

    monkeypatch.setattr(foreground, "own_window_rects", boom)
    cfg = ScreenConfig()
    cfg.mode = "auto"
    engine = _Engine()
    assert any(isinstance(e, ScreenTextEvent) for e in _tick(cfg, engine))


# --------------------------------------------------------------------------------------------
# the session folder
# --------------------------------------------------------------------------------------------


def _cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.paths.sessions_dir = str(tmp_path)
    return cfg


def test_the_session_saves_insights_md_when_the_assistant_has_insights(tmp_path: Path) -> None:
    assistant = FakeAssistant()
    assistant.insights_markdown = "## Questions to ask\n- q"
    recorder = SessionRecorder(_cfg(tmp_path), EventBus(), FakeTranscript(), assistant)
    folder = recorder.start()
    recorder.stop()
    assert (folder / "insights.md").read_text(encoding="utf-8") == "## Questions to ask\n- q"


def test_the_session_writes_no_insights_md_without_insights(tmp_path: Path) -> None:
    recorder = SessionRecorder(_cfg(tmp_path), EventBus(), FakeTranscript(), FakeAssistant())
    folder = recorder.start()
    recorder.stop()
    assert not (folder / "insights.md").exists()


# --------------------------------------------------------------------------------------------
# the Engine
# --------------------------------------------------------------------------------------------


class _RecordingAssistant:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def insights_now(self, sections: Any = None) -> int:
        self.calls.append(("insights_now", sections))
        return 1

    def recap(self, focus: str = "") -> int:
        self.calls.append(("recap",))
        return 2


def test_insights_and_recap_are_deferred_while_the_llm_starts_then_replayed() -> None:
    from exocortex.app import Engine

    eng = Engine(Config(), EventBus())
    fake = _RecordingAssistant()
    eng.assistant = fake
    eng.insights_now(("blindspots",))
    eng.insights_now(("questions", "blindspots"))   # latest insights request wins
    eng.recap()
    assert fake.calls == []
    assert [p[0] for p in eng._pending] == ["insights", "recap"]
    eng._llm_ready.set()
    eng._replay_pending()
    assert fake.calls == [("insights_now", ("questions", "blindspots")), ("recap",)]
    eng.insights_now()                              # ready now: straight through; None = both sections
    assert fake.calls[-1] == ("insights_now", None)


def test_engine_setters_normalise_into_the_config() -> None:
    from exocortex.app import Engine

    cfg = Config()
    eng = Engine(cfg, EventBus())
    eng.set_role("NAVIGATOR")
    assert cfg.assist.role == "navigator"
    eng.set_role("pilot")
    assert cfg.assist.role == "driver"
    eng.set_session_context("  billing\n service ")
    assert cfg.assist.session_context == "billing service"
    eng.set_insight_options(questions=False)
    assert cfg.assist.suggest_questions is False and cfg.assist.flag_blind_spots is True
    eng.set_insight_options(blind_spots=False, questions=True)
    assert cfg.assist.suggest_questions is True and cfg.assist.flag_blind_spots is False
