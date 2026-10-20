"""Caret tracking: where the user is typing right now, and what they have typed.

Everything comes from Windows accessibility interfaces -- the same ones screen readers use --
never from a keyboard hook:

* **UI Automation** (``IUIAutomation`` via comtypes) gives the focused element, whether it is an
  editable text control, the text before/after the caret (``TextPattern`` /
  ``TextPattern2.GetCaretRange``), its label/placeholder, and the font size/name at the caret.
* **MSAA** ``OBJID_CARET`` gives the caret's exact screen position. It works where the classic
  Win32 caret does not exist: Chromium (VS Code, Chrome, Edge, Electron apps) exposes a system
  caret for accessibility tools this way (verified on this machine: VS Code's message input
  reports the caret at the right pixel while ``GetGUIThreadInfo`` sees no caret at all).
* ``GetGUIThreadInfo`` is the fallback for classic Win32 edit controls, and UIA character
  bounding rectangles the fallback after that.

Two hard rules: a password field (``IsPassword``) is never read, and nothing is read while a
window of *this* process (the overlay, the ask box, the slim bar) has the focus.

Chromium note: the first UIA query after Chromium gains focus returns only its top-level pane;
that query switches Chromium's accessibility on, and the next query returns the real focused
element. Callers poll, so this resolves itself within one tick.

All coordinates are PHYSICAL screen pixels. ``CaretTracker`` must be attached and probed on one
thread (COM apartment); the Completer runs it on its own thread.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

# UIA control types / pattern ids / attribute ids (UIAutomationClient.h)
UIA_EDIT = 50004
UIA_DOCUMENT = 50030
UIA_PANE = 50033
UIA_TEXT_PATTERN = 10014
UIA_TEXT_PATTERN2 = 10024
UIA_VALUE_PATTERN = 10002
UIA_FONT_NAME_ATTR = 40005
UIA_FONT_SIZE_ATTR = 40006
TEXTUNIT_CHARACTER = 0
ENDPOINT_START = 0
ENDPOINT_END = 1

OBJID_CARET = 0xFFFFFFF8
GUI_CARETBLINKING = 0x1

#: Elements whose control type is one of these are treated as text targets when they expose a
#: text or value pattern. (Chromium reports <textarea>/<input> as Edit and contenteditable /
#: Monaco as Edit or Document.)
_TEXT_TYPES = {UIA_EDIT, UIA_DOCUMENT}

_LOG_EVERY_S = 30.0

#: Accessible names/help texts that mean "this editor exposes no text" (Monaco: VS Code, LeetCode,
#: web IDEs). Their text must come from the screen instead (see `screen/context.py`).
BLIND_MARKERS = ("not accessible", "screen reader", "accessibility options")
#: A caret shorter than this (physical px) is a hidden 1-px input, not a real text line.
MIN_CARET_H = 6


class _GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", wintypes.RECT),
    ]


@dataclass(frozen=True)
class CaretInfo:
    """One observation of the focused text target."""

    hwnd: int                 # focus window (control or top-level), 0 if unknown
    pid: int
    app: str                  # process image name, e.g. "Code.exe"
    field_name: str           # accessible name of the field ("Message input", "Search", ...)
    field_hint: str           # placeholder text when the field is empty, else ""
    control_type: int
    x: int                    # caret top-left, physical screen px
    y: int
    h: int                    # caret height (~line height), physical px; 0 = unknown
    dpi: int
    font_pt: float            # 0.0 = unknown
    font_name: str
    text_before: str          # text before the caret, clamped to the field (last N chars)
    text_after: str           # text after the caret (first M chars)
    source: str               # which tiers produced this ("uia+msaa", "uia+gui", "uia", ...)
    blind: bool = False       # the editor exposes no text (Monaco etc.): read it from the screen

    @property
    def key(self) -> tuple[int, str, int]:
        """Identity of the field the caret is in (a change means a different field)."""
        return (self.hwnd, self.field_name, self.control_type)

    @property
    def font_px(self) -> float:
        """Font size in physical pixels (0.0 when the font size is unknown)."""
        return self.font_pt * self.dpi / 72.0 if self.font_pt > 0 else 0.0

    @property
    def current_line(self) -> str:
        """Text before the caret on the caret's own line."""
        return self.text_before.rsplit("\n", 1)[-1]


def detect_placeholder(
    text_before: str, text_after: str, name: str, help_text: str, caret_gap_px: float = 0.0, font_px: float = 0.0
) -> bool:
    """True when an empty field is showing its placeholder as document text.

    Frameworks expose a placeholder as the field's text in one of two ways: with the caret at
    offset 0 (text follows the caret), or -- Chromium -- with the UIA selection at the END of the
    placeholder while the *visual* caret (MSAA) sits at the field's left edge. The second case is
    recognised by ``caret_gap_px``: how far the UIA end-of-text rectangle lies to the right of the
    visual caret (more than two characters' worth means they disagree). A match with the field's
    accessible name/help text is a certain signal in either case.
    """
    before, after = text_before.strip(), text_after.strip()
    doc = (text_before + text_after).strip()
    if not doc:
        return False
    norm_doc = " ".join(doc.split()).casefold().rstrip(".…")
    for candidate in (name, help_text):
        cand = " ".join((candidate or "").split()).casefold().rstrip(".…")
        if cand and cand == norm_doc:
            return True
    short_single_line = "\n" not in doc and len(doc) <= 120
    if not before and after:
        return short_single_line                       # caret at offset 0, text follows
    if before and not after and short_single_line:
        return caret_gap_px > 2.0 * max(font_px, 8.0)  # Chromium: caret drawn far left of the text end
    return False


class CaretTracker:
    """Polls the focused text control. Create, then ``attach()`` on the polling thread."""

    def __init__(self, prefix_chars: int = 1500, suffix_chars: int = 200) -> None:
        self.prefix_chars = max(0, int(prefix_chars))
        self.suffix_chars = max(0, int(suffix_chars))
        self.available = False
        self.detail = "not attached"
        self._uia: Any = None
        self._C: Any = None            # comtypes.gen.UIAutomationClient
        self._A: Any = None            # comtypes.gen.Accessibility (MSAA)
        self._user32: Any = None
        self._kernel32: Any = None
        self._oleacc: Any = None
        self._own_pid = os.getpid()
        self._last_log = 0.0
        self._names: dict[int, str] = {}
        self._com_initialized = False
        self.last_reject = ""

    # ---- lifecycle ----------------------------------------------------------------------
    def attach(self) -> bool:
        """Initialise COM + UIA on the calling thread. Returns availability."""
        if not IS_WINDOWS:
            self.detail = "not Windows"
            return False
        try:
            import comtypes
            import comtypes.client
        except Exception as exc:  # noqa: BLE001
            self.detail = f"comtypes unavailable: {exc}"
            log.warning("caret tracker: %s", self.detail)
            return False
        try:
            try:
                comtypes.CoInitialize()
                self._com_initialized = True
            except OSError:
                pass  # already initialised on this thread
            comtypes.client.GetModule("UIAutomationCore.dll")
            from comtypes.gen import UIAutomationClient as C  # type: ignore[import-not-found]

            self._C = C
            self._uia = comtypes.client.CreateObject(C.CUIAutomation8, interface=C.IUIAutomation)
        except Exception as exc:  # noqa: BLE001
            self.detail = f"UI Automation unavailable: {exc}"
            log.warning("caret tracker: %s", self.detail)
            return False
        try:
            comtypes.client.GetModule("oleacc.dll")
            from comtypes.gen import Accessibility as A  # type: ignore[import-not-found]

            self._A = A
            self._oleacc = ctypes.oledll.oleacc
        except Exception as exc:  # noqa: BLE001
            log.info("caret tracker: MSAA caret unavailable (%s); using GUI/UIA caret only", exc)
            self._A = None
        self._user32 = ctypes.windll.user32
        self._kernel32 = ctypes.windll.kernel32
        self.available = True
        self.detail = "UI Automation" + (" + MSAA caret" if self._A is not None else "")
        return True

    def detach(self) -> None:
        self._uia = None
        self.available = False
        if self._com_initialized:
            try:
                import comtypes

                comtypes.CoUninitialize()
            except Exception:  # noqa: BLE001
                pass
            self._com_initialized = False

    # ---- probing ----------------------------------------------------------------------------
    def probe(self) -> CaretInfo | None:
        """Observe the focused text target; ``None`` when there is nothing to complete."""
        if not self.available:
            return None
        try:
            return self._probe()
        except Exception as exc:  # noqa: BLE001 - COM errors are routine (app closing, etc.)
            now = time.monotonic()
            if now - self._last_log > _LOG_EVERY_S:
                self._last_log = now
                log.debug("caret probe failed: %s: %s", type(exc).__name__, exc)
            return None

    def _probe(self) -> CaretInfo | None:
        u = self._user32
        fg = u.GetForegroundWindow()
        if not fg:
            return None
        pid = wintypes.DWORD()
        tid = u.GetWindowThreadProcessId(wintypes.HWND(fg), ctypes.byref(pid))
        if pid.value == self._own_pid:
            return None  # never read our own overlay / ask box

        gti = _GUITHREADINFO()
        gti.cbSize = ctypes.sizeof(_GUITHREADINFO)
        have_gti = bool(u.GetGUIThreadInfo(tid, ctypes.byref(gti)))
        hwnd_focus = int(gti.hwndFocus or 0) if have_gti else 0

        el = self._uia.GetFocusedElement()
        if el is None:
            return self._reject("no focused element")
        if el.CurrentIsPassword:
            return self._reject("password field")
        ctype = int(el.CurrentControlType)
        name = str(el.CurrentName or "")
        try:
            help_text = str(el.CurrentHelpText or "")
        except Exception:  # noqa: BLE001
            help_text = ""
        el_hwnd = int(el.CurrentNativeWindowHandle or 0)
        hwnd = hwnd_focus or el_hwnd or int(fg)

        text_pat, text_pat2 = self._text_patterns(el)
        value_pat = self._value_pattern(el)
        if text_pat is None and value_pat is None:
            return self._reject(f"no text/value pattern (type {ctype}, {name!r})")
        if ctype not in _TEXT_TYPES and text_pat is None:
            return self._reject(f"not a text control (type {ctype}, {name!r})")
        if value_pat is not None:
            try:
                if bool(value_pat.CurrentIsReadOnly):
                    return self._reject("read-only")
            except Exception:  # noqa: BLE001
                pass

        text_before, text_after, font_pt, font_name, caret_rng, doc_rng = "", "", 0.0, "", None, None
        source = "uia"
        if text_pat is not None:
            text_before, text_after, font_pt, font_name, caret_rng, doc_rng = self._read_text(text_pat, text_pat2)
        elif value_pat is not None:
            try:
                value = str(value_pat.CurrentValue or "")
            except Exception:  # noqa: BLE001
                value = ""
            text_before = value[-self.prefix_chars:] if self.prefix_chars else value
            source = "uia-value"

        pos = self._caret_position(hwnd, el_hwnd, int(fg), gti if have_gti else None, caret_rng, doc_rng, el)
        if pos is None:
            return self._reject("no caret position")
        x, y, h, how, uia_x = pos
        source = f"{source}+{how}"

        try:
            dpi = int(u.GetDpiForWindow(wintypes.HWND(hwnd or fg)) or 96)
        except Exception:  # noqa: BLE001
            dpi = 96

        if 0 < h < MIN_CARET_H:
            h = 0  # a hidden 1-px input's caret says nothing about the line height

        field_hint = ""
        blind = False
        lowered = (name + " " + help_text).casefold()
        if any(marker in lowered for marker in BLIND_MARKERS):
            # Monaco-style editor: the accessible text is only a notice. Read the screen instead.
            blind, text_before, text_after = True, "", ""
        else:
            gap = float(uia_x - x) if (uia_x is not None and how != "uia-rect") else 0.0
            font_px = font_pt * dpi / 72.0 if font_pt > 0 else (float(h) / 1.3 if h else 0.0)
            if detect_placeholder(text_before, text_after, name, help_text, gap, font_px):
                field_hint = " ".join((text_before + text_after).split())
                text_before, text_after = "", ""

        return CaretInfo(
            hwnd=hwnd, pid=int(pid.value), app=self._process_name(int(pid.value)),
            field_name=name, field_hint=field_hint, control_type=ctype,
            x=x, y=y, h=h, dpi=dpi, font_pt=font_pt, font_name=font_name,
            text_before=text_before, text_after=text_after, source=source, blind=blind,
        )

    def _reject(self, why: str) -> None:
        """Record why the last probe found nothing (see `last_reject`), rate-limited to the log."""
        self.last_reject = why
        now = time.monotonic()
        if now - self._last_log > _LOG_EVERY_S:
            self._last_log = now
            log.debug("caret probe: %s", why)
        return None

    # ---- UIA helpers ------------------------------------------------------------------------
    def _pattern(self, el: Any, pattern_id: int, iface: Any) -> Any:
        try:
            raw = el.GetCurrentPattern(pattern_id)
        except Exception:  # noqa: BLE001
            return None
        if not raw:
            return None
        try:
            return raw.QueryInterface(iface)
        except Exception:  # noqa: BLE001
            return None

    def _text_patterns(self, el: Any) -> tuple[Any, Any]:
        C = self._C
        tp2 = self._pattern(el, UIA_TEXT_PATTERN2, C.IUIAutomationTextPattern2)
        if tp2 is not None:
            return tp2, tp2
        return self._pattern(el, UIA_TEXT_PATTERN, C.IUIAutomationTextPattern), None

    def _value_pattern(self, el: Any) -> Any:
        return self._pattern(el, UIA_VALUE_PATTERN, self._C.IUIAutomationValuePattern)

    def _read_text(self, tp: Any, tp2: Any) -> tuple[str, str, float, str, Any, Any]:
        """(text_before, text_after, font_pt, font_name, caret_range, document_range)."""
        doc = tp.DocumentRange
        caret = None
        if tp2 is not None:
            try:
                res = tp2.GetCaretRange()
                active, rng = self._unpack_caret(res)
                if rng is not None:
                    caret = rng
            except Exception:  # noqa: BLE001
                caret = None
        if caret is None:
            sel = tp.GetSelection()
            if sel is not None and sel.Length > 0:
                caret = sel.GetElement(0)
                # A real selection: the caret is at its active (end) side.
                caret.MoveEndpointByRange(ENDPOINT_START, caret, ENDPOINT_END)
        if caret is None:
            return "", "", 0.0, "", None, doc

        # Document start -> caret, and caret -> document end. Built from the field's own
        # DocumentRange rather than by moving endpoints character-wise: Chromium's
        # MoveEndpointByUnit wanders across the whole page (or moves a single character),
        # while range-to-range endpoint moves stay inside the field.
        before = doc.Clone()
        before.MoveEndpointByRange(ENDPOINT_END, caret, ENDPOINT_START)
        text_before = str(before.GetText(-1) or "")
        if self.prefix_chars and len(text_before) > self.prefix_chars:
            text_before = text_before[-self.prefix_chars:]

        after = doc.Clone()
        after.MoveEndpointByRange(ENDPOINT_START, caret, ENDPOINT_END)
        text_after = str(after.GetText(self.suffix_chars or -1) or "")

        font_pt, font_name = self._font(before if text_before else doc)
        return text_before.replace("\r\n", "\n").replace("\r", "\n"), text_after.replace("\r\n", "\n"), font_pt, font_name, caret, doc

    @staticmethod
    def _unpack_caret(res: Any) -> tuple[bool, Any]:
        """comtypes returns GetCaretRange's two out-params as a tuple in declaration order."""
        if isinstance(res, (tuple, list)):
            active, rng = True, None
            for item in res:
                if hasattr(item, "GetText"):
                    rng = item
                elif isinstance(item, (bool, int)):
                    active = bool(item)
            return active, rng
        if hasattr(res, "GetText"):
            return True, res
        return False, None

    def _font(self, rng: Any) -> tuple[float, str]:
        size, name = 0.0, ""
        try:
            v = rng.GetAttributeValue(UIA_FONT_SIZE_ATTR)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 1.0 < float(v) < 200.0:
                size = float(v)
        except Exception:  # noqa: BLE001
            pass
        try:
            v = rng.GetAttributeValue(UIA_FONT_NAME_ATTR)
            if isinstance(v, str):
                name = v
        except Exception:  # noqa: BLE001
            pass
        return size, name

    # ---- caret position tiers ----------------------------------------------------------------
    def _caret_position(
        self, hwnd: int, el_hwnd: int, fg: int, gti: Any, caret_rng: Any, doc_rng: Any, el: Any
    ) -> tuple[int, int, int, str, int | None] | None:
        """(x, y, h, tier, uia_x): the caret position from the best available tier, plus the x
        of the UIA end-of-text rectangle when known (used for placeholder detection)."""
        bounds = self._element_rect(el)
        uia_pos = self._uia_caret_rect(caret_rng) if caret_rng is not None else None
        uia_x = uia_pos[0] if uia_pos is not None else None
        # 1. MSAA OBJID_CARET (Chromium, WinUI, classic edits)
        for target in dict.fromkeys(h for h in (hwnd, el_hwnd, fg) if h):
            pos = self._msaa_caret(target)
            if pos is not None and self._plausible(pos, bounds):
                return (*pos, "msaa", uia_x)
        # 2. classic Win32 caret
        if gti is not None and gti.hwndCaret:
            r = gti.rcCaret
            if r.bottom > r.top:
                pt = wintypes.POINT(r.left, r.top)
                self._user32.ClientToScreen(gti.hwndCaret, ctypes.byref(pt))
                pos = (int(pt.x), int(pt.y), int(r.bottom - r.top))
                if self._plausible(pos, bounds):
                    return (*pos, "gui", uia_x)
        # 3. UIA character rectangles around the caret
        if uia_pos is not None and self._plausible(uia_pos, bounds):
            return (*uia_pos, "uia-rect", uia_x)
        return None

    def _msaa_caret(self, hwnd: int) -> tuple[int, int, int] | None:
        if self._A is None or not hwnd:
            return None
        A = self._A
        pacc = ctypes.POINTER(A.IAccessible)()
        try:
            self._oleacc.AccessibleObjectFromWindow(
                wintypes.HWND(hwnd), wintypes.DWORD(OBJID_CARET), ctypes.byref(A.IAccessible._iid_), ctypes.byref(pacc)
            )
            left, top, width, height = pacc.accLocation(0)
        except Exception:  # noqa: BLE001
            return None
        if height <= 0 or height > 400:
            return None
        return int(left), int(top), int(height)

    def _uia_caret_rect(self, caret: Any) -> tuple[int, int, int] | None:
        try:
            nxt = caret.Clone()
            nxt.ExpandToEnclosingUnit(TEXTUNIT_CHARACTER)
            rects = list(nxt.GetBoundingRectangles() or [])
            if len(rects) >= 4 and rects[3] > 0:
                return int(rects[0]), int(rects[1]), int(rects[3])
            prev = caret.Clone()
            prev.MoveEndpointByUnit(ENDPOINT_START, TEXTUNIT_CHARACTER, -1)
            rects = list(prev.GetBoundingRectangles() or [])
            if len(rects) >= 4 and rects[3] > 0:
                return int(rects[0] + rects[2]), int(rects[1]), int(rects[3])
        except Exception:  # noqa: BLE001
            return None
        return None

    @staticmethod
    def _element_rect(el: Any) -> tuple[int, int, int, int] | None:
        try:
            rc = el.CurrentBoundingRectangle
            return int(rc.left), int(rc.top), int(rc.right), int(rc.bottom)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _plausible(pos: tuple[int, int, int], bounds: tuple[int, int, int, int] | None, slack: int = 8) -> bool:
        """A caret must lie inside (or just outside) the focused element's rectangle."""
        if bounds is None:
            return True
        left, top, right, bottom = bounds
        if right - left <= 0 or bottom - top <= 0:
            return True  # Chromium's top-level pane / unknown bounds: cannot judge
        x, y, h = pos
        return (left - slack) <= x <= (right + slack) and (top - slack) <= y and (y + h) <= (bottom + slack)

    # ---- process name ---------------------------------------------------------------------
    def _process_name(self, pid: int) -> str:
        name = self._names.get(pid)
        if name is not None:
            return name
        name = f"pid {pid}"
        try:
            handle = self._kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                buf = ctypes.create_unicode_buffer(1024)
                size = wintypes.DWORD(1024)
                if self._kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    name = buf.value.rsplit("\\", 1)[-1]
                self._kernel32.CloseHandle(handle)
        except Exception:  # noqa: BLE001
            pass
        if len(self._names) > 64:
            self._names.clear()
        self._names[pid] = name
        return name
