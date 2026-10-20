"""Extra defect-hunting tests for exocortex.screen (foreground.py, capture.py, ocr.py).

Written by an independent verifier pass over the implementer's tests/test_screen_capture_ocr.py.
These focus specifically on paths that file did not exercise: RapidOCR's downscale/box-scaling
round trip (the real screen is well above ``_RAPIDOCR_MAX_SIDE``, but the real-engine subprocess
test's synthetic image is not, so that path was never actually hit), the WindowsOcr union-box
math and its private event loop being reused (not recreated) across calls, a few
``ScreenGrabber``/``_clip_rect`` boundary cases, and a couple of ``lines_to_text`` row-bucketing
shapes the original suite did not cover (a single line, many items in one row, embedded
tabs/newlines inside one OCR line's own text).

All offline: no real screen, no real OCR engine, no winrt/mss import beyond what ``ocr.py`` and
``capture.py`` already do lazily (``_downscale_if_needed`` uses PIL, which is a plain, already
required, always-importable dependency here -- not one of the lazily-guarded engine imports).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from exocortex.screen import capture, ocr

# --------------------------------------------------------------------------------- #
# capture._clip_rect boundary cases (via ScreenGrabber.grab, its only caller)
# --------------------------------------------------------------------------------- #

MONITORS = [
    {"left": -2560, "top": 0, "width": 7680, "height": 2448},
    {"left": -2560, "top": 848, "width": 2560, "height": 1600, "is_primary": False},
    {"left": 0, "top": 0, "width": 5120, "height": 1440, "is_primary": True},
]


class _FakeShot:
    def __init__(self, region: dict[str, int]) -> None:
        self.width = region["width"]
        self.height = region["height"]
        bgra = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.bgra = bgra.tobytes()


class _FakeMSS:
    def __init__(self, monitors: list[dict[str, Any]]) -> None:
        self.monitors = monitors

    def grab(self, region: dict[str, int]) -> _FakeShot:
        return _FakeShot(region)

    def close(self) -> None:
        pass


def _grabber() -> capture.ScreenGrabber:
    return capture.ScreenGrabber(sct_factory=lambda: _FakeMSS(MONITORS))


def test_grab_monitor_negative_index_returns_none() -> None:
    assert _grabber().grab("monitor", monitor=-1) is None


def test_grab_region_zero_width_returns_none() -> None:
    assert _grabber().grab("region", region=(0, 0, 0, 100)) is None


def test_grab_region_negative_width_returns_none() -> None:
    assert _grabber().grab("region", region=(0, 0, -5, 100)) is None


def test_grab_region_touching_edge_exactly_is_empty() -> None:
    # Virtual screen is (-2560, 0, 7680, 2448); a region starting exactly at its right edge
    # has zero overlap and must return None, not a zero-width Frame.
    vx, _vy, vw, _vh = (-2560, 0, 7680, 2448)
    region = (vx + vw, 0, 100, 100)
    assert _grabber().grab("region", region=region) is None


def test_grab_region_one_pixel_overlap_is_not_empty() -> None:
    vx, _vy, vw, _vh = (-2560, 0, 7680, 2448)
    region = (vx + vw - 1, 0, 100, 100)
    frame = _grabber().grab("region", region=region)
    assert frame is not None
    assert frame.rect == (vx + vw - 1, 0, 1, 100)


# --------------------------------------------------------------------------------- #
# ocr._scale_box / ocr._union_boxes (pure helpers, no engine needed)
# --------------------------------------------------------------------------------- #


def test_scale_box_identity_at_scale_one() -> None:
    assert ocr._scale_box((10.0, 20.0, 30.0, 40.0), 1.0) == (10, 20, 30, 40)


def test_scale_box_scales_up_for_downscaled_coords() -> None:
    # A box measured on an image that was shrunk to 50% must be doubled to map back.
    assert ocr._scale_box((10.0, 20.0, 30.0, 40.0), 0.5) == (20, 40, 60, 80)


def test_union_boxes_single_box_is_itself() -> None:
    assert ocr._union_boxes([(5.0, 5.0, 10.0, 20.0)]) == (5.0, 5.0, 10.0, 20.0)


def test_union_boxes_spans_all_inputs() -> None:
    boxes = [(0.0, 0.0, 10.0, 10.0), (20.0, 5.0, 10.0, 10.0), (5.0, 15.0, 5.0, 5.0)]
    # x: min(0)=0, y: min(0)=0; x2: max(20+10)=30, y2: max(15+5)=20
    assert ocr._union_boxes(boxes) == (0.0, 0.0, 30.0, 20.0)


# --------------------------------------------------------------------------------- #
# ocr._downscale_if_needed
# --------------------------------------------------------------------------------- #


def test_downscale_if_needed_no_op_when_within_limit() -> None:
    image = np.zeros((300, 900, 3), dtype=np.uint8)
    out, scale = ocr._downscale_if_needed(image, 2200)
    assert scale == 1.0
    assert out is image  # untouched, not even copied


def test_downscale_if_needed_shrinks_large_image() -> None:
    image = np.zeros((1000, 3000, 3), dtype=np.uint8)  # longest side 3000 > 2200
    out, scale = ocr._downscale_if_needed(image, 2200)
    assert 0.0 < scale < 1.0
    assert max(out.shape[:2]) <= 2200
    # aspect ratio preserved (within integer rounding)
    assert abs((out.shape[1] / out.shape[0]) - (image.shape[1] / image.shape[0])) < 0.01


def test_downscale_if_needed_zero_dimension_is_a_no_op() -> None:
    image = np.zeros((0, 3000, 3), dtype=np.uint8)
    out, scale = ocr._downscale_if_needed(image, 2200)
    assert scale == 1.0
    assert out is image


# --------------------------------------------------------------------------------- #
# RapidOcr.recognize(): full downscale -> recognize -> scale-back-up round trip, with a
# fake underlying engine (the real screen is 5120px wide, well above _RAPIDOCR_MAX_SIDE, so
# this path is live in production but the implementer's real-engine test image (900x300)
# never triggers it).
# --------------------------------------------------------------------------------- #


def _make_rapid_ocr(fake_engine: Any) -> ocr.RapidOcr:
    """Build a RapidOcr instance without running its real __init__ (no rapidocr import)."""
    engine = object.__new__(ocr.RapidOcr)
    engine._engine = fake_engine
    return engine


def test_rapidocr_recognize_scales_boxes_back_up_after_downscale() -> None:
    h, w = 1000, 3000  # longest side 3000 > _RAPIDOCR_MAX_SIDE (2200)
    image = np.zeros((h, w, 3), dtype=np.uint8)
    _, expected_scale = ocr._downscale_if_needed(image, ocr._RAPIDOCR_MAX_SIDE)
    assert expected_scale < 1.0

    # A box in ORIGINAL image coordinates; the fake engine reports it in DOWNSCALED
    # coordinates, as the real rapidocr engine would since it only ever sees the shrunk image.
    orig_box = (300.0, 100.0, 200.0, 40.0)  # x, y, w, h
    ox, oy, ow, oh = orig_box
    down_pts = np.array(
        [
            [ox * expected_scale, oy * expected_scale],
            [(ox + ow) * expected_scale, oy * expected_scale],
            [(ox + ow) * expected_scale, (oy + oh) * expected_scale],
            [ox * expected_scale, (oy + oh) * expected_scale],
        ],
        dtype=np.float32,
    )

    seen_shapes: list[tuple[int, int]] = []

    def fake_engine(scaled_image: np.ndarray) -> SimpleNamespace:
        seen_shapes.append(scaled_image.shape[:2])
        return SimpleNamespace(txts=("Hello",), boxes=[down_pts], scores=(0.98,))

    rapid = _make_rapid_ocr(fake_engine)
    result = rapid.recognize(image)

    # The engine was actually invoked on the downscaled image, not the original.
    assert seen_shapes and max(seen_shapes[0]) <= ocr._RAPIDOCR_MAX_SIDE

    assert result.engine == "rapidocr"
    assert len(result.lines) == 1
    line = result.lines[0]
    assert line.text == "Hello"
    assert line.confidence == pytest.approx(0.98)
    # Scaled back up to within 2px of the original box (rounding twice: once shrinking the
    # image, once scaling the box back).
    for got, want in zip(line.box, orig_box):
        assert abs(got - want) <= 2, f"box {line.box} strayed too far from original {orig_box}"


def test_rapidocr_recognize_skips_none_text_and_none_box() -> None:
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)

    def fake_engine(scaled_image: np.ndarray) -> SimpleNamespace:
        return SimpleNamespace(txts=("Hello", None, "  ", "World"), boxes=[pts, pts, pts, None], scores=(0.9, 0.9, 0.9, 0.9))

    rapid = _make_rapid_ocr(fake_engine)
    result = rapid.recognize(np.zeros((100, 100, 3), dtype=np.uint8))
    assert [ln.text for ln in result.lines] == ["Hello"]


def test_rapidocr_recognize_handles_empty_result() -> None:
    def fake_engine(scaled_image: np.ndarray) -> SimpleNamespace:
        return SimpleNamespace(txts=(), boxes=[], scores=())

    rapid = _make_rapid_ocr(fake_engine)
    result = rapid.recognize(np.zeros((50, 50, 3), dtype=np.uint8))
    assert result.lines == ()
    assert result.text == ""


# --------------------------------------------------------------------------------- #
# WindowsOcr.recognize(): union-box math and event-loop reuse across calls, with a fake
# winrt-shaped engine (no real winrt.windows.media.ocr.OcrEngine construction, so this is
# safe to run in the same process as the RapidOcr tests above).
# --------------------------------------------------------------------------------- #


class _FakeWord:
    def __init__(self, x: float, y: float, w: float, h: float) -> None:
        self.bounding_rect = SimpleNamespace(x=x, y=y, width=w, height=h)


class _FakeOcrLine:
    def __init__(self, text: str, words: list[_FakeWord]) -> None:
        self.text = text
        self.words = words


class _FakeOcrResult:
    def __init__(self, lines: list[_FakeOcrLine]) -> None:
        self.lines = lines


class _FakeWinrtEngine:
    """Stands in for winrt's OcrEngine: an async ``recognize_async`` returning fake lines."""

    def __init__(self, lines_per_call: list[list[_FakeOcrLine]]) -> None:
        self._calls = iter(lines_per_call)
        self.bitmaps_seen: list[Any] = []

    async def recognize_async(self, bitmap: Any) -> _FakeOcrResult:
        self.bitmaps_seen.append(bitmap)
        return _FakeOcrResult(next(self._calls))


def _make_windows_ocr(fake_winrt_engine: Any, monkeypatch: pytest.MonkeyPatch) -> ocr.WindowsOcr:
    """Build a WindowsOcr without running its real __init__ (no winrt OcrEngine construction).

    ``_bgr_to_software_bitmap`` is monkeypatched to a no-op sentinel so this never touches
    winrt.windows.graphics.imaging/storage.streams either -- only our own recognize()/
    union-box/loop-reuse logic is under test here.
    """
    import asyncio

    monkeypatch.setattr(ocr, "_bgr_to_software_bitmap", lambda image: "fake-bitmap")
    engine = object.__new__(ocr.WindowsOcr)
    engine._engine = fake_winrt_engine
    engine._max_dim = 10000
    engine._loop = asyncio.new_event_loop()
    return engine


def test_windows_ocr_recognize_unions_word_boxes_into_a_line_box(monkeypatch: pytest.MonkeyPatch) -> None:
    words = [_FakeWord(10, 10, 20, 15), _FakeWord(35, 12, 25, 15)]
    fake = _FakeWinrtEngine([[_FakeOcrLine("Hello World", words)]])
    engine = _make_windows_ocr(fake, monkeypatch)

    result = engine.recognize(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result.engine == "windows"
    assert len(result.lines) == 1
    line = result.lines[0]
    assert line.text == "Hello World"
    # union of (10,10,20,15) and (35,12,25,15): x=[10,60), y=[10,27)
    assert line.box == (10, 10, 50, 17)
    assert line.confidence == 1.0
    engine.close()


def test_windows_ocr_recognize_drops_lines_with_no_words(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeWinrtEngine([[_FakeOcrLine("", []), _FakeOcrLine("Real", [_FakeWord(0, 0, 5, 5)])]])
    engine = _make_windows_ocr(fake, monkeypatch)
    result = engine.recognize(np.zeros((10, 10, 3), dtype=np.uint8))
    assert [ln.text for ln in result.lines] == ["Real"]
    engine.close()


def test_windows_ocr_reuses_the_same_event_loop_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    words = [_FakeWord(0, 0, 5, 5)]
    fake = _FakeWinrtEngine([[_FakeOcrLine("One", words)], [_FakeOcrLine("Two", words)]])
    engine = _make_windows_ocr(fake, monkeypatch)
    loop_before = engine._loop

    r1 = engine.recognize(np.zeros((10, 10, 3), dtype=np.uint8))
    assert not engine._loop.is_closed()
    assert engine._loop is loop_before  # not recreated between calls

    r2 = engine.recognize(np.zeros((10, 10, 3), dtype=np.uint8))
    assert engine._loop is loop_before
    assert not engine._loop.is_closed()

    assert (r1.text, r2.text) == ("One", "Two")
    engine.close()
    assert engine._loop.is_closed()
    engine.close()  # idempotent, must not raise


# --------------------------------------------------------------------------------- #
# lines_to_text(): row-bucketing shapes not covered by the original suite
# --------------------------------------------------------------------------------- #


def _line(text: str, x: int, y: int, w: int = 50, h: int = 20, conf: float = 1.0) -> ocr.OcrLine:
    return ocr.OcrLine(text=text, box=(x, y, w, h), confidence=conf)


def test_lines_to_text_single_line_no_trailing_newline() -> None:
    assert ocr.lines_to_text([_line("Only line", x=0, y=0)]) == "Only line"


def test_lines_to_text_many_items_one_row_sorted_left_to_right() -> None:
    lines = [_line("D", x=300, y=0), _line("A", x=0, y=2), _line("C", x=200, y=1), _line("B", x=100, y=0)]
    assert ocr.lines_to_text(lines) == "A  B  C  D"


def test_lines_to_text_collapses_embedded_tabs_and_newlines_within_one_item() -> None:
    lines = [_line("multiple\t\ttabs\nand\nnewlines", x=0, y=0)]
    assert ocr.lines_to_text(lines) == "multiple tabs and newlines"


def test_new_lines_is_case_and_whitespace_normalised_both_ways() -> None:
    # A previous line that itself has irregular whitespace/case must still suppress a
    # current line that only differs in whitespace/case.
    previous = ["  Foo\tBar  "]
    current = ["foo bar", "Foo Bar!"]
    assert ocr.new_lines(previous, current) == ["Foo Bar!"]
