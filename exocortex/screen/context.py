"""Code/text context around the caret, reconstructed from screen OCR.

Some editors expose nothing through accessibility: Monaco (VS Code, LeetCode, CodeSandbox, many
web IDEs) keeps its text in a hidden 1-px textarea whose only accessible content is a notice, unless
the user turns on screen-reader mode. The text *is* on screen, though, and the screen reader OCRs
the active window every couple of seconds with a bounding box per line. Given the caret position
(from MSAA/UIA, which still works) this module rebuilds:

* the lines **above** the caret that belong to the same pane (same left margin, allowing for
  indentation), with their indentation restored from the box x-offsets in units of the estimated
  character width -- the description pane of a LeetCode page, further left, is excluded;
* the caret's own line cut at the caret (``current_left``), or its indentation when the line is
  still empty;
* a few lines **below** the caret.

Gutter line numbers (a separate column, or merged into the line by OCR) are stripped. Everything is
a best effort over OCR output: characters may be misread and spacing is approximate, which is why
the completer compares blind-mode text fuzzily.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from exocortex.screen.ocr import OcrLine

#: Ratio of character width to font pixel size (Consolas/Cascadia/Courier ~0.55-0.6).
CHAR_W_PER_FONT_PX = 0.6
#: Ratio of character width to line height when the font size is unknown.
CHAR_W_PER_LINE_H = 0.46
#: Lines further left than this many characters from the caret are another pane.
_PANE_MAX_CHARS_LEFT = 60
#: Lines further than this many line heights from the caret are ignored altogether.
_PANE_MAX_LINES_AWAY = 80

_LINE_NUMBER_RE = re.compile(r"^(\d{1,6})(?:\s+(.*))?$")


@dataclass(frozen=True)
class ScreenSnapshot:
    """The latest OCR result with geometry: boxes are relative to ``rect`` (screen px)."""

    lines: tuple[OcrLine, ...]
    rect: tuple[int, int, int, int]      # x, y, w, h of the captured frame on screen
    captured_at: float                   # seconds since session start (engine clock)
    source_name: str = ""


@dataclass(frozen=True)
class CaretContext:
    """Text around the caret as rebuilt from OCR."""

    above: str            # lines above the caret in the caret's pane, indentation restored
    current_left: str     # the caret's line up to the caret (indentation included)
    current_full: str     # the caret's line as OCR read it (no indentation)
    below: str            # a few lines below the caret
    pane_left: float      # screen x of the pane's text margin
    char_w: float         # estimated character width (screen px)
    has_line: bool        # whether an OCR line was found under the caret

    @property
    def prefix(self) -> str:
        return f"{self.above}\n{self.current_left}" if self.above else self.current_left

    @property
    def suffix(self) -> str:
        rest = self.current_full[len(self.current_left.lstrip(" ")):] if self.has_line else ""
        return f"{rest}\n{self.below}".strip("\n") if self.below else rest


@dataclass
class _Item:
    text: str
    x: float          # screen x of the text start (after any gutter number)
    y: float
    w: float
    h: float
    merged: bool = False   # x is an estimate: OCR merged a gutter line number into this line

    @property
    def yc(self) -> float:
        return self.y + self.h / 2.0


def _items(snapshot: ScreenSnapshot) -> list[_Item]:
    fx, fy = snapshot.rect[0], snapshot.rect[1]
    out: list[_Item] = []
    for ln in snapshot.lines:
        text = " ".join(ln.text.split())
        if not text:
            continue
        x, y, w, h = ln.box
        out.append(_Item(text=text, x=fx + x, y=fy + y, w=float(w), h=float(h)))
    return out


def _strip_gutter(items: list[_Item], char_w: float) -> list[_Item]:
    """Drop standalone line-number boxes and strip merged leading line numbers.

    Only when the numbers look like a gutter: at least four items carry one and they are mostly
    increasing top to bottom.
    """
    numbered: list[tuple[_Item, re.Match[str]]] = []
    for it in items:
        m = _LINE_NUMBER_RE.match(it.text)
        if m:
            numbered.append((it, m))
    if len(numbered) < 4:
        return items
    numbered.sort(key=lambda pair: pair[0].y)
    values = [int(m.group(1)) for _it, m in numbered]
    increasing = sum(1 for a, b in zip(values, values[1:]) if b > a)
    if increasing < 0.6 * (len(values) - 1):
        return items
    out: list[_Item] = []
    for it in items:
        m = _LINE_NUMBER_RE.match(it.text)
        if not m:
            out.append(it)
            continue
        rest = (m.group(2) or "").strip()
        if not rest:
            continue  # a bare gutter number
        shift = (len(m.group(1)) + 1) * char_w
        out.append(_Item(text=rest, x=it.x + shift, y=it.y, w=max(0.0, it.w - shift), h=it.h, merged=True))
    return out


def estimate_char_width(font_px: float, line_h: float, items: Sequence[_Item] | None = None) -> float:
    if font_px > 0:
        return max(3.0, font_px * CHAR_W_PER_FONT_PX)
    if line_h > 0:
        return max(3.0, line_h * CHAR_W_PER_LINE_H)
    if items:
        hs = sorted(it.h for it in items)
        return max(3.0, hs[len(hs) // 2] * CHAR_W_PER_LINE_H)
    return 8.0


def context_at(
    snapshot: ScreenSnapshot | None,
    x: int,
    y: int,
    line_h: float,
    font_px: float,
    *,
    max_above_lines: int = 40,
    max_above_chars: int = 1500,
    max_below_lines: int = 8,
) -> CaretContext | None:
    """Rebuild the text around the caret at screen position ``(x, y)`` (top-left of the caret)."""
    if snapshot is None:
        return None
    fx, fy, fw, fh = snapshot.rect
    if not (fx - 4 <= x <= fx + fw + 4 and fy - 4 <= y <= fy + fh + 4):
        return None
    items = _items(snapshot)
    if not items:
        return None
    char_w = estimate_char_width(font_px, line_h, items)
    items = _strip_gutter(items, char_w)
    if not items:
        return None
    lh = line_h if line_h > 0 else sorted(it.h for it in items)[len(items) // 2]
    cy = y + lh / 2.0

    # The caret's own line: vertically overlapping, starting left of (or at) the caret.
    own: _Item | None = None
    best = float("inf")
    for it in items:
        if not (it.y - 0.3 * it.h <= cy <= it.y + 1.3 * it.h):
            continue
        if it.x > x + 2 * char_w:
            continue
        d = x - it.x
        if d < best:
            best, own = d, it

    # Pane: nearby lines whose left edge is not far left of the caret / own line.
    ref_x = own.x if own is not None else float(x)
    near = [it for it in items if abs(it.yc - cy) <= _PANE_MAX_LINES_AWAY * lh
            and ref_x - _PANE_MAX_CHARS_LEFT * char_w <= it.x <= ref_x + 2 * char_w]
    # The pane's text margin: prefer lines whose x is exact (no merged gutter number).
    exact = [it.x for it in near if not it.merged]
    pane_left = min(exact) if len(exact) >= 2 else min((it.x for it in near), default=ref_x)
    pane = [it for it in items if abs(it.yc - cy) <= _PANE_MAX_LINES_AWAY * lh
            and pane_left - 1.5 * char_w <= it.x <= pane_left + _PANE_MAX_CHARS_LEFT * char_w
            or (it.merged and abs(it.yc - cy) <= _PANE_MAX_LINES_AWAY * lh
                and pane_left - 8 * char_w <= it.x <= pane_left + _PANE_MAX_CHARS_LEFT * char_w)]

    def indent(it_x: float) -> str:
        return " " * max(0, round((it_x - pane_left) / char_w))

    above_items = sorted((it for it in pane if it.yc < cy - 0.3 * lh), key=lambda it: it.y)
    above_lines = [indent(it.x) + it.text for it in above_items][-max_above_lines:]
    above = "\n".join(above_lines)
    if len(above) > max_above_chars:
        above = above[-max_above_chars:]
        cut = above.find("\n")
        if 0 <= cut < len(above) - 1:
            above = above[cut + 1:]

    below_items = sorted((it for it in pane if it.yc > cy + 0.3 * lh), key=lambda it: it.y)
    below = "\n".join(indent(it.x) + it.text for it in below_items[:max_below_lines])

    if own is not None:
        n = max(0, min(len(own.text), round((x - own.x) / char_w)))
        current_left = indent(own.x) + own.text[:n]
        current_full = own.text
    else:
        current_left = indent(float(x))
        current_full = ""
    return CaretContext(
        above=above, current_left=current_left, current_full=current_full, below=below,
        pane_left=pane_left, char_w=char_w, has_line=own is not None,
    )
