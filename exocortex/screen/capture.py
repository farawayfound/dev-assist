"""Screen capture via ``mss``, plus the cheap "did the screen change" signature.

``ScreenGrabber`` owns exactly one ``mss.MSS`` instance, created lazily on first use: mss
objects are not thread-safe across threads, so :mod:`exocortex.screen.reader`'s worker
thread is meant to own a single ``ScreenGrabber`` for its whole lifetime. ``mss`` is imported
lazily (inside :meth:`ScreenGrabber._mss`) so this module has no import-time side effects and
stays importable on any platform; tests inject a fake via the ``sct_factory`` constructor
keyword instead of touching the real screen.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from exocortex.screen import foreground

log = logging.getLogger(__name__)

#: 32x18 (16:9) thumbnail used as a cheap "did the screen change" signature.
_SIG_COLS = 32
_SIG_ROWS = 18


@dataclass(frozen=True)
class Frame:
    """One captured screen image."""

    image: np.ndarray  # H×W×3 uint8 BGR, C-contiguous, owned (not a view into mss's buffer)
    rect: tuple[int, int, int, int]  # x, y, w, h in screen pixels, clipped to the virtual screen
    source_name: str  # window title (auto), "monitor N" (monitor), "region" (region)
    mode: str
    ts: float  # time.monotonic() at capture


def _clip_rect(
    rect: tuple[int, int, int, int], bounds: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """Intersect ``rect`` with ``bounds``; ``None`` if the result is empty or invalid."""
    rx, ry, rw, rh = rect
    if rw <= 0 or rh <= 0:
        return None
    bx, by, bw, bh = bounds
    x1, y1 = max(rx, bx), max(ry, by)
    x2, y2 = min(rx + rw, bx + bw), min(ry + rh, by + bh)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def mask_rects(frame: Frame, rects: "Iterable[tuple[int, int, int, int]]", fill: int = 0) -> int:
    """Blank (``fill``) every part of ``frame.image`` that one of ``rects`` covers, in place;
    returns how many of the rects touched the frame.

    ``rects`` are in screen pixels, like ``frame.rect``, so each is clipped to the frame and
    shifted into image coordinates first. Used to keep this app's own windows out of the OCR
    (``[screen].exclude_own_windows``, :func:`exocortex.screen.foreground.own_window_rects`): a
    flat area reads as no text, and stays the same from frame to frame, so the overlay's own
    streaming answer never counts as a screen change either."""
    fx, fy, fw, fh = frame.rect
    height, width = frame.image.shape[:2]
    touched = 0
    for rect in rects:
        clipped = _clip_rect(tuple(rect), (fx, fy, fw, fh))
        if clipped is None:
            continue
        x, y, w, h = clipped
        x0, y0 = x - fx, y - fy
        x1, y1 = min(width, x0 + w), min(height, y0 + h)
        if x1 > x0 and y1 > y0:
            frame.image[y0:y1, x0:x1] = fill
            touched += 1
    return touched


class ScreenGrabber:
    """Grabs BGR frames of the active window / a monitor / an arbitrary region via ``mss``.

    Owns one ``mss.MSS`` instance, created lazily so constructing a ``ScreenGrabber`` (e.g.
    at import/config time) never touches the real screen. Pass ``sct_factory`` to inject a
    fake ``mss``-shaped object for tests.
    """

    def __init__(self, *, sct_factory: Callable[[], Any] | None = None) -> None:
        self._sct_factory = sct_factory
        self._sct: Any | None = None

    def _mss(self) -> Any:
        if self._sct is None:
            if self._sct_factory is not None:
                self._sct = self._sct_factory()
            else:
                import mss

                self._sct = mss.MSS()
        return self._sct

    def close(self) -> None:
        """Release the underlying ``mss`` instance, if one was created."""
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                log.debug("mss close() failed", exc_info=True)
            self._sct = None

    def monitors(self) -> list[tuple[int, int, int, int]]:
        """Every physical monitor as ``(x, y, w, h)``, primary first (see foreground.py)."""
        mons = list(self._mss().monitors)
        real = mons[1:] if len(mons) > 1 else mons
        return [foreground.monitor_dict_to_rect(m) for m in foreground.order_primary_first(real)]

    def _virtual_rect(self) -> tuple[int, int, int, int]:
        mons = list(self._mss().monitors)
        return foreground.monitor_dict_to_rect(mons[0])

    def grab(
        self, mode: str, *, region: tuple[int, int, int, int] | None = None, monitor: int = 0
    ) -> Frame | None:
        """Capture ``mode`` ("auto" | "monitor" | "region") and return a :class:`Frame`.

        - ``auto``: the foreground window's rect; falls back to the primary monitor when
          there is no usable foreground window.
        - ``monitor``: ``self.monitors()[monitor]`` (0 = primary).
        - ``region``: the caller-supplied ``region`` rect.

        The chosen rect is always clipped to the virtual screen; an empty/invalid rect (e.g.
        an out-of-range monitor index, or a region entirely off-screen) returns ``None``.
        """
        vrect = self._virtual_rect()

        if mode == "region":
            if region is None:
                log.warning("grab(mode='region') called without a region")
                return None
            rect = _clip_rect(region, vrect)
            source_name = "region"
        elif mode == "monitor":
            mons = self.monitors()
            if not (0 <= monitor < len(mons)):
                log.warning("grab: monitor index %s out of range (%d monitor(s))", monitor, len(mons))
                return None
            rect = _clip_rect(mons[monitor], vrect)
            source_name = f"monitor {monitor}"
        elif mode == "auto":
            win = foreground.foreground_window()
            if win is not None:
                rect = _clip_rect(win.rect, vrect)
                source_name = win.title or win.process_name or "window"
            else:
                mons = self.monitors()
                if not mons:
                    return None
                rect = _clip_rect(mons[0], vrect)
                source_name = "monitor 0"
        else:
            log.warning("grab: unknown mode %r", mode)
            return None

        if rect is None:
            return None
        return self._grab_rect(rect, source_name=source_name, mode=mode)

    def _grab_rect(self, rect: tuple[int, int, int, int], *, source_name: str, mode: str) -> Frame | None:
        x, y, w, h = rect
        try:
            shot = self._mss().grab({"left": x, "top": y, "width": w, "height": h})
        except Exception:
            log.exception("mss grab failed for rect=%s", rect)
            return None
        bgra = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        image = np.ascontiguousarray(bgra[:, :, :3])  # BGR, copied out of mss's reused buffer
        return Frame(image=image, rect=rect, source_name=source_name, mode=mode, ts=time.monotonic())


def signature(image: np.ndarray) -> np.ndarray:
    """Area-averaged ``_SIG_ROWS``x``_SIG_COLS`` (18x32) float32 grayscale thumbnail of ``image``.

    Vectorised via a numpy reshape/mean; when ``image``'s dimensions are not exact multiples
    of 32/18 it is first cropped (top-left) to the nearest multiple. An image smaller than one
    block in either axis yields an all-zero thumbnail (there is nothing to average).
    """
    h, w = image.shape[:2]
    gray = image[:, :, :3].astype(np.float32).mean(axis=2)

    gh = (h // _SIG_ROWS) * _SIG_ROWS
    gw = (w // _SIG_COLS) * _SIG_COLS
    if gh == 0 or gw == 0:
        return np.zeros((_SIG_ROWS, _SIG_COLS), dtype=np.float32)

    gray = gray[:gh, :gw]
    block_h, block_w = gh // _SIG_ROWS, gw // _SIG_COLS
    thumb = gray.reshape(_SIG_ROWS, block_h, _SIG_COLS, block_w).mean(axis=(1, 3))
    return thumb.astype(np.float32)


def is_changed(a: np.ndarray | None, b: np.ndarray, threshold: float = 4.0) -> bool:
    """``True`` when ``a`` is ``None`` or the mean absolute difference from ``b`` exceeds ``threshold``.

    ``threshold`` is on the 0..255 grayscale scale that :func:`signature` produces. A shape
    mismatch (e.g. comparing signatures of differently-sized captures) also counts as changed.
    """
    if a is None:
        return True
    if a.shape != b.shape:
        return True
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()) > threshold
