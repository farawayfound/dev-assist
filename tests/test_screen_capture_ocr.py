"""Tests for exocortex.screen: foreground.py, capture.py, ocr.py.

Offline tests never touch a real screen, a real OCR engine, or a real window: ``mss`` is
replaced with a small fake object (``sct_factory`` / monkeypatched ``mss.MSS``), and
:func:`~exocortex.screen.foreground.foreground_window`'s single raw Win32 seam
(``_foreground_raw``) is monkeypatched directly.

The "real engine" tests render a small image with PIL and run the actual installed Windows
OCR / RapidOCR engines on it — but each in its own subprocess. That is deliberate, not just
style: constructing ``winrt.windows.media.ocr.OcrEngine`` and then constructing
``rapidocr.RapidOCR`` *in the same process* reliably segfaults the interpreter on this
machine (verified: RapidOCR-then-WindowsOCR is fine; WindowsOCR-then-RapidOCR crashes,
apparently a COM apartment conflict between WinRT activation and RapidOCR's native deps).
Running each engine in its own short-lived subprocess makes the two tests independent and
keeps a real-engine crash from taking down the whole pytest session.

Tests that grab the real screen or read the real foreground window are marked
``@pytest.mark.live`` (skipped by default per pytest.ini)::

    "D:\\repos\\ai-infrastructure\\exo-cortex\\.venv\\Scripts\\python.exe" -m pytest tests/test_screen_capture_ocr.py -m live -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import mss as mss_module
import numpy as np
import pytest

from exocortex.screen import capture, foreground, ocr

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------------- #
# Fake mss: a minimal stand-in for mss.MSS (monitors list + grab()) used by both
# ScreenGrabber (via sct_factory) and, monkeypatched onto the real `mss` module, by
# foreground.monitor_rects()/virtual_screen_rect().
# --------------------------------------------------------------------------------- #


class FakeShot:
    """Stands in for mss.screenshot.ScreenShot: .width/.height/.size/.bgra.

    Pixel values are derived from the grabbed region so tests can assert on exactly what
    was captured: B = left % 256, G = top % 256, R = 255, A = 255 for every pixel.
    """

    def __init__(self, region: dict[str, int]) -> None:
        self.width = region["width"]
        self.height = region["height"]
        self.size = (self.width, self.height)
        bgra = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        bgra[:, :, 0] = region["left"] % 256
        bgra[:, :, 1] = region["top"] % 256
        bgra[:, :, 2] = 255
        bgra[:, :, 3] = 255
        self.bgra = bgra.tobytes()


class FakeMSS:
    """Stands in for mss.MSS: a fixed `.monitors` list, `.grab()`, and context-manager support."""

    def __init__(self, monitors: list[dict[str, Any]]) -> None:
        self.monitors = monitors
        self.grabbed: list[dict[str, int]] = []
        self.closed = False

    def grab(self, region: dict[str, int]) -> FakeShot:
        self.grabbed.append(dict(region))
        return FakeShot(region)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "FakeMSS":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# Matches SPEC.md's measured machine: laptop 2560x1600 at (-2560, 848) not primary,
# Odyssey 5120x1440 at (0, 0) primary, union spans both.
MONITORS = [
    {"left": -2560, "top": 0, "width": 7680, "height": 2448},  # mss's "all in one" union
    {"left": -2560, "top": 848, "width": 2560, "height": 1600, "is_primary": False},  # laptop
    {"left": 0, "top": 0, "width": 5120, "height": 1440, "is_primary": True},  # Odyssey (primary)
]
VIRTUAL_RECT = (-2560, 0, 7680, 2448)
PRIMARY_RECT = (0, 0, 5120, 1440)
LAPTOP_RECT = (-2560, 848, 2560, 1600)


def _grabber() -> capture.ScreenGrabber:
    return capture.ScreenGrabber(sct_factory=lambda: FakeMSS(MONITORS))


# --------------------------------------------------------------------------------- #
# monitor ordering (primary first)
# --------------------------------------------------------------------------------- #


def test_screen_grabber_monitors_primary_first() -> None:
    assert _grabber().monitors() == [PRIMARY_RECT, LAPTOP_RECT]


def test_foreground_monitor_rects_primary_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mss_module, "MSS", lambda **kwargs: FakeMSS(MONITORS))
    assert foreground.monitor_rects() == [PRIMARY_RECT, LAPTOP_RECT]


def test_foreground_virtual_screen_rect_is_the_union_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mss_module, "MSS", lambda **kwargs: FakeMSS(MONITORS))
    assert foreground.virtual_screen_rect() == VIRTUAL_RECT


def test_order_primary_first_falls_back_to_first_entry_when_none_marked() -> None:
    mons = [{"left": 0, "top": 0, "width": 100, "height": 100}, {"left": 100, "top": 0, "width": 50, "height": 50}]
    assert foreground.order_primary_first(mons) == mons


def test_order_primary_first_empty_list() -> None:
    assert foreground.order_primary_first([]) == []


# --------------------------------------------------------------------------------- #
# ScreenGrabber.grab(): fake mss, region clipping, monitor/auto modes
# --------------------------------------------------------------------------------- #


def test_grab_monitor_mode_returns_expected_frame() -> None:
    g = _grabber()
    frame = g.grab("monitor", monitor=0)
    assert frame is not None
    assert frame.rect == PRIMARY_RECT
    assert frame.mode == "monitor"
    assert frame.source_name == "monitor 0"
    x, y, w, h = frame.rect
    assert frame.image.shape == (h, w, 3)
    assert frame.image.dtype == np.uint8
    assert frame.image.flags["C_CONTIGUOUS"]
    # FakeShot encodes left/top into B/G; R is always 255 -> confirms BGR channel order.
    assert (frame.image[:, :, 0] == PRIMARY_RECT[0] % 256).all()
    assert (frame.image[:, :, 1] == PRIMARY_RECT[1] % 256).all()
    assert (frame.image[:, :, 2] == 255).all()


def test_grab_monitor_mode_second_monitor() -> None:
    frame = _grabber().grab("monitor", monitor=1)
    assert frame is not None
    assert frame.rect == LAPTOP_RECT
    assert frame.source_name == "monitor 1"


def test_grab_monitor_out_of_range_returns_none() -> None:
    assert _grabber().grab("monitor", monitor=5) is None


def test_grab_region_mode_fully_inside_is_unclipped() -> None:
    region = (100, 100, 200, 150)
    frame = _grabber().grab("region", region=region)
    assert frame is not None
    assert frame.rect == region
    assert frame.mode == "region"
    assert frame.source_name == "region"


def test_grab_region_mode_without_region_returns_none() -> None:
    assert _grabber().grab("region") is None


def test_grab_region_clipped_to_virtual_screen() -> None:
    # Straddles the left edge of the virtual screen (-2560, 0, 7680, 2448).
    region = (-3000, -100, 500, 500)
    frame = _grabber().grab("region", region=region)
    assert frame is not None
    assert frame.rect == (-2560, 0, 60, 400)


def test_grab_region_entirely_off_screen_returns_none() -> None:
    assert _grabber().grab("region", region=(100_000, 100_000, 100, 100)) is None


def test_grab_unknown_mode_returns_none() -> None:
    assert _grabber().grab("bogus") is None


def test_grab_auto_uses_foreground_window(monkeypatch: pytest.MonkeyPatch) -> None:
    win = foreground.WindowInfo(hwnd=42, title="Zoom Meeting", process_name="zoom.exe", rect=(10, 10, 300, 200))
    monkeypatch.setattr(foreground, "foreground_window", lambda: win)
    frame = _grabber().grab("auto")
    assert frame is not None
    assert frame.rect == (10, 10, 300, 200)
    assert frame.source_name == "Zoom Meeting"
    assert frame.mode == "auto"


def test_grab_auto_falls_back_to_primary_monitor_when_no_foreground_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(foreground, "foreground_window", lambda: None)
    frame = _grabber().grab("auto")
    assert frame is not None
    assert frame.rect == PRIMARY_RECT
    assert frame.source_name == "monitor 0"


def test_grab_auto_clips_foreground_window_rect_to_virtual_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    # A window rect that pokes off the right/bottom edge of the virtual screen.
    win = foreground.WindowInfo(hwnd=1, title="Off-screen-ish", process_name="x.exe", rect=(5000, 2000, 5000, 5000))
    monkeypatch.setattr(foreground, "foreground_window", lambda: win)
    frame = _grabber().grab("auto")
    assert frame is not None
    vx, vy, vw, vh = VIRTUAL_RECT
    x, y, w, h = frame.rect
    assert x + w <= vx + vw
    assert y + h <= vy + vh


def test_grabber_lazily_creates_and_reuses_one_sct_instance() -> None:
    made: list[FakeMSS] = []

    def factory() -> FakeMSS:
        fake = FakeMSS(MONITORS)
        made.append(fake)
        return fake

    g = capture.ScreenGrabber(sct_factory=factory)
    assert made == []  # not created at construction time
    g.grab("monitor", monitor=0)
    g.grab("monitor", monitor=1)
    assert len(made) == 1  # same instance reused
    g.close()
    assert made[0].closed is True


# --------------------------------------------------------------------------------- #
# signature() / is_changed()
# --------------------------------------------------------------------------------- #


def _solid(h: int, w: int, value: int) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_signature_shape_and_dtype() -> None:
    sig = capture.signature(_solid(360, 640, 100))
    assert sig.shape == (18, 32)
    assert sig.dtype == np.float32


def test_signature_handles_sizes_not_divisible_by_grid() -> None:
    # 700x1000 is not a multiple of 18/32; must crop instead of raising.
    sig = capture.signature(_solid(700, 1000, 77))
    assert sig.shape == (18, 32)
    assert np.allclose(sig, 77.0)


def test_signature_smaller_than_one_block_is_all_zero() -> None:
    sig = capture.signature(_solid(5, 5, 200))
    assert sig.shape == (18, 32)
    assert np.count_nonzero(sig) == 0


def test_is_changed_identical_images_is_false() -> None:
    img = _solid(360, 640, 60)
    img[0:60, 0:60] = 255
    sig_a = capture.signature(img)
    sig_b = capture.signature(img.copy())
    assert capture.is_changed(sig_a, sig_b) is False


def test_is_changed_moved_white_box_is_true() -> None:
    base = _solid(360, 640, 50)
    a = base.copy()
    a[0:60, 0:60] = 255
    b = base.copy()
    b[300:360, 580:640] = 255
    assert capture.is_changed(capture.signature(a), capture.signature(b)) is True


def test_is_changed_none_previous_is_true() -> None:
    assert capture.is_changed(None, capture.signature(_solid(360, 640, 10))) is True


def test_is_changed_respects_threshold() -> None:
    base = _solid(360, 640, 50)
    slightly = base.copy()
    slightly[:, :] = 51  # tiny, uniform difference
    sig_a, sig_b = capture.signature(base), capture.signature(slightly)
    assert capture.is_changed(sig_a, sig_b, threshold=4.0) is False
    assert capture.is_changed(sig_a, sig_b, threshold=0.5) is True


# --------------------------------------------------------------------------------- #
# foreground_window()
# --------------------------------------------------------------------------------- #


def test_foreground_window_none_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(foreground, "IS_WINDOWS", False)
    assert foreground.foreground_window() is None


def test_foreground_window_none_for_our_own_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(foreground, "_foreground_raw", lambda: (123, os.getpid(), "SomeAppWindowClass", "Our App"))
    assert foreground.foreground_window() is None


def test_foreground_window_none_for_desktop_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    other_pid = os.getpid() + 1
    for class_name in ("Progman", "WorkerW", "Shell_TrayWnd"):
        monkeypatch.setattr(foreground, "_foreground_raw", lambda cn=class_name: (7, other_pid, cn, "Desktop"))
        assert foreground.foreground_window() is None


def test_foreground_window_none_when_raw_lookup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(foreground, "_foreground_raw", lambda: None)
    assert foreground.foreground_window() is None


def test_foreground_window_none_when_minimized(monkeypatch: pytest.MonkeyPatch) -> None:
    other_pid = os.getpid() + 1
    monkeypatch.setattr(foreground, "_foreground_raw", lambda: (9, other_pid, "Notepad", "untitled - Notepad"))
    monkeypatch.setattr(foreground, "_is_minimized", lambda hwnd: True)
    assert foreground.foreground_window() is None


def test_foreground_window_none_when_rect_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    other_pid = os.getpid() + 1
    monkeypatch.setattr(foreground, "_foreground_raw", lambda: (9, other_pid, "Notepad", "untitled - Notepad"))
    monkeypatch.setattr(foreground, "_is_minimized", lambda hwnd: False)
    monkeypatch.setattr(foreground, "_window_rect", lambda hwnd: None)
    assert foreground.foreground_window() is None


def test_foreground_window_returns_window_info_for_a_normal_window(monkeypatch: pytest.MonkeyPatch) -> None:
    other_pid = os.getpid() + 1
    monkeypatch.setattr(foreground, "_foreground_raw", lambda: (9, other_pid, "Notepad", "untitled - Notepad"))
    monkeypatch.setattr(foreground, "_is_minimized", lambda hwnd: False)
    monkeypatch.setattr(foreground, "_window_rect", lambda hwnd: (10, 20, 300, 400))
    monkeypatch.setattr(foreground, "_process_name", lambda pid: "notepad.exe")
    info = foreground.foreground_window()
    assert info == foreground.WindowInfo(hwnd=9, title="untitled - Notepad", process_name="notepad.exe", rect=(10, 20, 300, 400))


# --------------------------------------------------------------------------------- #
# ocr.lines_to_text() / new_lines()
# --------------------------------------------------------------------------------- #


def _line(text: str, x: int, y: int, w: int = 50, h: int = 20, conf: float = 1.0) -> ocr.OcrLine:
    return ocr.OcrLine(text=text, box=(x, y, w, h), confidence=conf)


def test_lines_to_text_orders_rows_top_to_bottom_and_items_left_to_right() -> None:
    # Two rows, each with two items, fed in shuffled order.
    lines = [
        _line("Bar", x=80, y=98, h=20),
        _line("Hello", x=10, y=10, h=20),
        _line("World", x=100, y=12, h=20),
        _line("Foo", x=10, y=100, h=20),
    ]
    assert ocr.lines_to_text(lines) == "Hello  World\nFoo  Bar"


def test_lines_to_text_collapses_internal_whitespace() -> None:
    lines = [_line("  multiple   spaces  ", x=0, y=0)]
    assert ocr.lines_to_text(lines) == "multiple spaces"


def test_lines_to_text_drops_empty_lines() -> None:
    lines = [_line("", x=0, y=0), _line("   ", x=0, y=20), _line("Real text", x=0, y=40)]
    assert ocr.lines_to_text(lines) == "Real text"


def test_lines_to_text_empty_input() -> None:
    assert ocr.lines_to_text([]) == ""


def test_lines_to_text_uses_row_band_from_median_height() -> None:
    # Two items whose y-centres differ by a bit more than the small-height band -> two rows.
    lines = [_line("Top", x=0, y=0, h=10), _line("Bottom", x=0, y=20, h=10)]
    text = ocr.lines_to_text(lines)
    assert text == "Top\nBottom"


def test_new_lines_normalises_case_and_whitespace() -> None:
    previous = ["Hello world", "foo"]
    current = ["HELLO   WORLD", "bar", "  foo  "]
    assert ocr.new_lines(previous, current) == ["bar"]


def test_new_lines_empty_previous_returns_all_current() -> None:
    assert ocr.new_lines([], ["a", "b"]) == ["a", "b"]


def test_new_lines_no_new_lines() -> None:
    assert ocr.new_lines(["a", "b"], ["a", "b"]) == []


# --------------------------------------------------------------------------------- #
# create_engine() dispatch (offline, fake engine classes)
# --------------------------------------------------------------------------------- #


class _AlwaysUnavailable:
    def __init__(self) -> None:
        raise ocr.OcrUnavailable("fake: never available")


class _FakeEngine:
    def __init__(self, name: str) -> None:
        self.name = name

    def recognize(self, image_bgr: np.ndarray) -> ocr.OcrResult:  # pragma: no cover - not exercised
        raise NotImplementedError


def test_create_engine_windows_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "WindowsOcr", lambda: _FakeEngine("windows"))
    assert ocr.create_engine("windows").name == "windows"


def test_create_engine_rapidocr_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "RapidOcr", lambda: _FakeEngine("rapidocr"))
    assert ocr.create_engine("rapidocr").name == "rapidocr"


def test_create_engine_auto_prefers_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "WindowsOcr", lambda: _FakeEngine("windows"))
    monkeypatch.setattr(ocr, "RapidOcr", lambda: (_ for _ in ()).throw(AssertionError("should not be called")))
    assert ocr.create_engine("auto").name == "windows"


def test_create_engine_auto_falls_back_to_rapidocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "WindowsOcr", _AlwaysUnavailable)
    monkeypatch.setattr(ocr, "RapidOcr", lambda: _FakeEngine("rapidocr"))
    assert ocr.create_engine("auto").name == "rapidocr"


def test_create_engine_auto_raises_when_both_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr, "WindowsOcr", _AlwaysUnavailable)
    monkeypatch.setattr(ocr, "RapidOcr", _AlwaysUnavailable)
    with pytest.raises(ocr.OcrUnavailable):
        ocr.create_engine("auto")


def test_create_engine_unknown_name_raises() -> None:
    with pytest.raises(ocr.OcrUnavailable):
        ocr.create_engine("bogus")


# --------------------------------------------------------------------------------- #
# Real-engine tests: each runs in its own subprocess (see module docstring for why).
# --------------------------------------------------------------------------------- #

_REAL_OCR_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from exocortex.screen import ocr as ocr_mod


    def _render() -> np.ndarray:
        img = Image.new("RGB", (900, 300), "white")
        draw = ImageDraw.Draw(img)
        text = "ExoCortex meeting notes\\nBudget: 1.2M\\nWhat is the timeline?"
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except OSError:
            try:
                font = ImageFont.load_default(size=28)
            except TypeError:
                font = ImageFont.load_default()
        draw.text((10, 10), text, fill="black", font=font)
        rgb = np.array(img)
        return rgb[:, :, ::-1].copy()  # RGB -> BGR


    def main() -> None:
        engine_name = sys.argv[1]
        try:
            engine = ocr_mod.create_engine(engine_name)
        except ocr_mod.OcrUnavailable as exc:
            print(json.dumps({"unavailable": str(exc)}))
            return
        result = engine.recognize(_render())
        print(json.dumps({"text": result.text, "elapsed_ms": result.elapsed_ms, "engine": result.engine}))


    if __name__ == "__main__":
        main()
    """
)


def _run_real_ocr_subprocess(engine_name: str) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-c", _REAL_OCR_SUBPROCESS_SCRIPT, engine_name],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"{engine_name} OCR subprocess exited {proc.returncode}\n"
            f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
        )
    stdout = proc.stdout.strip()
    assert stdout, f"{engine_name} OCR subprocess printed nothing; stderr: {proc.stderr[-2000:]}"
    data: dict[str, Any] = json.loads(stdout.splitlines()[-1])
    if "unavailable" in data:
        pytest.skip(f"{engine_name} OCR unavailable: {data['unavailable']}")
    return data


@pytest.mark.parametrize("engine_name", ["rapidocr", "windows"])
def test_real_ocr_engine_reads_rendered_text(engine_name: str) -> None:
    """Renders the SPEC.md sample text and runs the real, installed OCR engine on it.

    Order matters: rapidocr is parametrized before windows so that, even though each engine
    actually runs in its own subprocess (and is therefore already isolated from the other),
    a future refactor that accidentally merges them back into one process fails safe on the
    known-safe ordering rather than the known-crashing one.
    """
    data = _run_real_ocr_subprocess(engine_name)
    text_lower = data["text"].lower()
    for word in ("meeting", "budget", "timeline"):
        assert word in text_lower, f"{engine_name}: {word!r} missing from OCR text: {data['text']!r}"
    assert data["elapsed_ms"] < 3000


# --------------------------------------------------------------------------------- #
# Live tests: real screen / real foreground window (run with `-m live`)
# --------------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_grab_primary_monitor() -> None:
    frame = capture.ScreenGrabber().grab("monitor", monitor=0)
    assert frame is not None
    x, y, w, h = frame.rect
    assert frame.image.shape == (h, w, 3)
    assert frame.image.dtype == np.uint8
    assert w > 0 and h > 0


@pytest.mark.live
def test_live_grab_foreground_window() -> None:
    frame = capture.ScreenGrabber().grab("auto")
    assert frame is not None  # falls back to the primary monitor even with nothing focused
    x, y, w, h = frame.rect
    assert frame.image.shape == (h, w, 3)


@pytest.mark.live
def test_live_foreground_window_smoke() -> None:
    info = foreground.foreground_window()
    if info is not None:
        assert info.rect[2] > 0 and info.rect[3] > 0
        assert info.hwnd != 0
