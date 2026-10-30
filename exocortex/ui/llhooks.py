"""Low-level Win32 input hooks (``SetWindowsHookEx`` ``WH_MOUSE_LL`` / ``WH_KEYBOARD_LL``) for two
things ``RegisterHotKey`` cannot do:

* :class:`MouseWheelHook` -- see the wheel while the overlay is click-through. In hands-free mode
  the window carries ``WS_EX_TRANSPARENT`` over its content, so Windows routes every mouse
  message, wheel included, to whatever is underneath; the hook sees the wheel first, and when
  the overlay decides the cursor is over one of its output panes, it scrolls that pane itself and
  swallows the notch (the app underneath does not scroll too). Installed only while it is
  needed (the cursor over such a pane), so the per-move cost of a Python callback is paid only
  then.
* :class:`BareKeyHook` -- a single key as a toggle: a TAP of the right Shift key (pressed and
  released with no other key in between, within ``max_tap_s``) pauses / resumes the app.
  ``RegisterHotKey`` cannot bind a modifier key on its own, and a hook lets typing keep working:
  Shift held for a capital letter is not a tap (another key came in between), so nothing fires,
  and a modifier key is never swallowed. A plain key ("right", the arrow) is swallowed while it
  acts as the toggle (no modifier held); with Shift / Ctrl / Alt / Win held it passes through.
* :class:`PressHoldMouseHook` -- every shortcut bound to a MOUSE button: the manual-advance
  gestures (a button pressed and held in place, or clicked, anywhere on screen -- the overlay is
  usually click-through / in the background then, so its own widgets could never see the press)
  and, since 2026-10-29, any other ``[hotkeys]`` entry bound to a mouse button
  (:mod:`exocortex.bindings`). ``RegisterHotKey`` cannot bind mouse buttons at all. Installed only
  while something is bound to a mouse button. A bare left / middle / right press is never
  swallowed (it still reaches the app underneath); a thumb button or a modified click is, the way
  a keyboard hotkey is consumed.
* :class:`ChordRecorder` + :class:`CaptureHooks` -- the Shortcuts tab's **Set** button
  (2026-10-29): the next key combination or mouse button pressed anywhere, swallowed while
  listening, becomes the binding half a second after the last press.

Both run their callback on the thread that installed the hook (the Qt main thread, whose
message loop Windows uses to deliver the hook), so a callback may touch widgets -- but must be
quick: Windows silently drops a low-level hook whose callback exceeds its timeout (~300 ms), so
the callbacks here do almost nothing and hand the real work to the overlay through the
``on_wheel`` / ``on_tap`` callables (which the overlay defers with ``QTimer.singleShot(0)``).
``user32`` is injectable (tests drive the captured hook procedure with fake messages); off
Windows every ``install()`` is a harmless no-op that reports ``installed == False``.
"""
from __future__ import annotations

import ctypes
import logging
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

from exocortex.bindings import (
    MODIFIER_OF_KEY,
    MODIFIERS,
    MOUSE_BUTTONS,
    key_token_for_vk,
    normalize_binding,
    parse_binding,
)
from exocortex.bindings import NAMED_VKS as _NAMED_VKS

log = logging.getLogger(__name__)

WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_MBUTTONDOWN = 0x0207
WM_MBUTTONUP = 0x0208
WM_MOUSEWHEEL = 0x020A
WM_XBUTTONDOWN = 0x020B
WM_XBUTTONUP = 0x020C
WM_MOUSEHWHEEL = 0x020E
XBUTTON1 = 0x0001                    # HIWORD(mouseData) of a WM_XBUTTON* message: the "Back" thumb button
XBUTTON2 = 0x0002                    # ... the "Forward" thumb button
WHEEL_DELTA = 120
_LLKHF_INJECTED = 0x10
_LLMHF_INJECTED = 0x01

VK_LSHIFT, VK_RSHIFT = 0xA0, 0xA1
VK_LCONTROL, VK_RCONTROL = 0xA2, 0xA3
VK_LMENU, VK_RMENU = 0xA4, 0xA5
VK_LWIN, VK_RWIN = 0x5B, 0x5C
VK_SHIFT, VK_CONTROL, VK_MENU = 0x10, 0x11, 0x12

#: Bare modifier keys by name -> the virtual keys that count as that key.
MODIFIER_KEYS: dict[str, tuple[int, ...]] = {
    "rshift": (VK_RSHIFT,), "lshift": (VK_LSHIFT,), "shift": (VK_LSHIFT, VK_RSHIFT),
    "rctrl": (VK_RCONTROL,), "lctrl": (VK_LCONTROL,), "ctrl": (VK_LCONTROL, VK_RCONTROL),
    "rcontrol": (VK_RCONTROL,), "lcontrol": (VK_LCONTROL,), "control": (VK_LCONTROL, VK_RCONTROL),
    "ralt": (VK_RMENU,), "lalt": (VK_LMENU,), "alt": (VK_LMENU, VK_RMENU),
    "rwin": (VK_RWIN,), "lwin": (VK_LWIN,), "win": (VK_LWIN, VK_RWIN),
}
#: Every modifier virtual key (a bare-key hook never swallows these, and a plain key pressed
#: while one of them is held is a combination, not the toggle).
_MODIFIER_VKS = frozenset(vk for vks in MODIFIER_KEYS.values() for vk in vks)
_HELD_CHECK_VKS = (VK_SHIFT, VK_CONTROL, VK_MENU, VK_LWIN, VK_RWIN)

_ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32
_HOOKPROC = (ctypes.WINFUNCTYPE if hasattr(ctypes, "WINFUNCTYPE") else ctypes.CFUNCTYPE)(
    ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_ssize_t
)


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("pt", POINT), ("mouseData", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("time", ctypes.c_uint32), ("dwExtraInfo", _ULONG_PTR)]


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", ctypes.c_uint32), ("scanCode", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("time", ctypes.c_uint32), ("dwExtraInfo", _ULONG_PTR)]


def is_bare_key(text: str) -> bool:
    """Whether a ``[hotkeys]`` binding names a single key for a :class:`BareKeyHook` (no ``+``:
    a modifier such as ``"rshift"``, a named key such as ``"right"``, a letter or a digit)
    rather than a ``RegisterHotKey`` combination."""
    try:
        parse_bare_key(text)
    except ValueError:
        return False
    return True


def parse_bare_key(text: str) -> tuple[int, ...]:
    """The virtual keys a bare-key binding names; ``ValueError`` for a combination or an
    unknown name."""
    name = (text or "").strip().lower()
    if not name or "+" in name:
        raise ValueError(f"not a single key: {text!r}")
    if name in MODIFIER_KEYS:
        return MODIFIER_KEYS[name]
    if name in _NAMED_VKS:
        return (_NAMED_VKS[name],)
    if len(name) == 1 and (name.isalpha() or name.isdigit()):
        return (ord(name.upper()),)
    raise ValueError(f"unknown key: {text!r}")


class _NullUser32:
    """Off Windows: every call fails harmlessly."""

    def SetWindowsHookExW(self, *_: Any) -> int:  # noqa: N802
        return 0

    def UnhookWindowsHookEx(self, *_: Any) -> bool:  # noqa: N802
        return False

    def CallNextHookEx(self, *_: Any) -> int:  # noqa: N802
        return 0

    def GetAsyncKeyState(self, *_: Any) -> int:  # noqa: N802
        return 0


def _default_user32() -> Any:
    if sys.platform != "win32":
        return _NullUser32()
    dll = ctypes.WinDLL("user32", use_last_error=True)
    dll.SetWindowsHookExW.argtypes = [ctypes.c_int, _HOOKPROC, ctypes.c_void_p, ctypes.c_uint32]
    dll.SetWindowsHookExW.restype = ctypes.c_void_p
    dll.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
    dll.UnhookWindowsHookEx.restype = ctypes.c_int
    dll.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_ssize_t]
    dll.CallNextHookEx.restype = ctypes.c_ssize_t
    dll.GetAsyncKeyState.argtypes = [ctypes.c_int]
    dll.GetAsyncKeyState.restype = ctypes.c_short
    return dll


def _module_handle() -> int:
    if sys.platform != "win32":
        return 0
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        return int(kernel32.GetModuleHandleW(None) or 0)
    except Exception:  # noqa: BLE001
        return 0


class _LowLevelHook:
    """One ``SetWindowsHookEx`` low-level hook. Subclasses implement :meth:`_on_message` and
    return True to swallow the message."""

    hook_id = 0

    def __init__(self, *, user32: Any | None = None) -> None:
        self._user32: Any = user32 if user32 is not None else _default_user32()
        self._handle: int = 0
        self._proc: Any = None          # keeps the ctypes callback alive while installed
        self.failed = False

    @property
    def installed(self) -> bool:
        return bool(self._handle)

    def install(self) -> bool:
        """Install the hook on the calling thread (it needs this thread's message loop). Returns
        whether it is installed; a failure is logged once and remembered in :attr:`failed`."""
        if self._handle:
            return True
        proc = _HOOKPROC(self._callback)
        try:
            handle = self._user32.SetWindowsHookExW(self.hook_id, proc, _module_handle(), 0)
        except Exception:  # noqa: BLE001
            log.exception("%s: SetWindowsHookEx raised", type(self).__name__)
            handle = 0
        if not handle:
            if not self.failed:
                log.warning("%s: SetWindowsHookEx failed (error %s)", type(self).__name__,
                            ctypes.get_last_error() if hasattr(ctypes, "get_last_error") else "?")
            self.failed = True
            return False
        self._handle = int(handle)
        self._proc = proc
        return True

    def uninstall(self) -> None:
        if not self._handle:
            return
        try:
            self._user32.UnhookWindowsHookEx(self._handle)
        except Exception:  # noqa: BLE001
            log.exception("%s: UnhookWindowsHookEx raised", type(self).__name__)
        self._handle = 0
        self._proc = None

    def _callback(self, n_code: int, w_param: int, l_param: int) -> int:
        try:
            if n_code >= 0 and self._on_message(int(w_param), int(l_param)):
                return 1
        except Exception:  # noqa: BLE001 - never let an exception escape into the hook chain
            log.exception("%s: hook callback failed", type(self).__name__)
        return int(self._user32.CallNextHookEx(self._handle, n_code, w_param, l_param) or 0)

    def _on_message(self, w_param: int, l_param: int) -> bool:  # pragma: no cover - abstract
        return False


class MouseWheelHook(_LowLevelHook):
    """``WH_MOUSE_LL`` hook that hands every wheel notch to ``on_wheel(delta, horizontal)``
    (``delta`` in wheel units, +120 per notch away from the user) and swallows it when that
    returns True. Everything else passes straight through."""

    hook_id = WH_MOUSE_LL

    def __init__(self, on_wheel: Callable[[int, bool], bool], *, user32: Any | None = None) -> None:
        super().__init__(user32=user32)
        self._on_wheel = on_wheel

    def _on_message(self, w_param: int, l_param: int) -> bool:
        if w_param not in (WM_MOUSEWHEEL, WM_MOUSEHWHEEL) or not l_param:
            return False
        data = MSLLHOOKSTRUCT.from_address(l_param)
        delta = (int(data.mouseData) >> 16) & 0xFFFF
        if delta >= 0x8000:
            delta -= 0x10000
        return bool(self._on_wheel(delta, w_param == WM_MOUSEHWHEEL))


class BareKeyHook(_LowLevelHook):
    """``WH_KEYBOARD_LL`` hook that calls a binding's callback when its key
    (:func:`parse_bare_key`) is tapped: pressed and released with no other key pressed in
    between, within ``max_tap_s``. ``bindings`` maps key names to callbacks (``{"rshift": pause}``,
    ``{"up": prev, "down": next}``); a single ``key`` with ``on_tap`` is the one-binding form. A
    modifier key is never swallowed; a plain key is swallowed (down and up) while no modifier is
    held, since it then IS the toggle."""

    hook_id = WH_KEYBOARD_LL

    def __init__(
        self, bindings: "Mapping[str, Callable[[], None]] | str", on_tap: Callable[[], None] | None = None,
        *, max_tap_s: float = 0.6, user32: Any | None = None, clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(user32=user32)
        if isinstance(bindings, str):
            if on_tap is None:
                raise ValueError("on_tap is required with a single key")
            bindings = {bindings: on_tap}
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._vk_to_key: dict[int, str] = {}
        for name, callback in bindings.items():
            for vk in parse_bare_key(name):
                self._vk_to_key[vk] = name
            self._callbacks[name] = callback
        if not self._callbacks:
            raise ValueError("no key bindings")
        self.keys = tuple(self._callbacks)
        self._max_tap_s = float(max_tap_s)
        self._clock = clock
        self._down_vk: int | None = None
        self._down_at: float | None = None
        self._other_key = False
        self._swallowing = False
        self.taps = 0

    def _modifier_held(self) -> bool:
        try:
            return any(int(self._user32.GetAsyncKeyState(vk)) & 0x8000 for vk in _HELD_CHECK_VKS)
        except Exception:  # noqa: BLE001
            return False

    def _on_message(self, w_param: int, l_param: int) -> bool:
        if not l_param:
            return False
        vk = int(KBDLLHOOKSTRUCT.from_address(l_param).vkCode)
        down = w_param in (WM_KEYDOWN, WM_SYSKEYDOWN)
        up = w_param in (WM_KEYUP, WM_SYSKEYUP)
        armed = self._down_vk is not None
        if vk not in self._vk_to_key or (armed and vk != self._down_vk):
            if down and armed:
                self._other_key = True      # a combination (or a second bound key), not a tap
            return False
        if down:
            if not armed:                   # auto-repeat sends further downs: keep the first
                plain = vk not in _MODIFIER_VKS
                if plain and self._modifier_held():
                    return False            # Ctrl+Right etc.: a combination for the app underneath, not a tap
                self._down_vk = vk
                self._down_at = self._clock()
                self._other_key = False
                self._swallowing = plain
            return self._swallowing
        if up:
            started, self._down_at, self._down_vk = self._down_at, None, None
            swallowed, self._swallowing = self._swallowing, False
            if started is not None and not self._other_key and self._clock() - started <= self._max_tap_s:
                self.taps += 1
                try:
                    self._callbacks[self._vk_to_key[vk]]()
                except Exception:  # noqa: BLE001
                    log.exception("BareKeyHook: callback for %r failed", self._vk_to_key.get(vk))
            return swallowed
        return False

_BUTTON_DOWN = {WM_LBUTTONDOWN: "mouse_left", WM_MBUTTONDOWN: "mouse_middle", WM_RBUTTONDOWN: "mouse_right"}
_BUTTON_UP = {WM_LBUTTONUP: "mouse_left", WM_MBUTTONUP: "mouse_middle", WM_RBUTTONUP: "mouse_right"}
_XBUTTONS = {XBUTTON1: "mouse4", XBUTTON2: "mouse5"}
#: What `mouse_button_of` reports for a WM_XBUTTON* message naming no known button: a press of
#: SOME unbound button (it still cancels a gesture in flight, like any other button).
UNKNOWN_BUTTON = "mouse_x?"


def mouse_button_of(w_param: int, data: "MSLLHOOKSTRUCT") -> tuple[str | None, bool]:
    """``(button, is_down)`` for a mouse-button message: a canonical button name
    (:data:`exocortex.bindings.MOUSE_BUTTONS`), :data:`UNKNOWN_BUTTON` for an X-button message
    the grammar has no name for, or ``None`` for a message that is not a button at all."""
    if w_param in _BUTTON_DOWN:
        return _BUTTON_DOWN[w_param], True
    if w_param in _BUTTON_UP:
        return _BUTTON_UP[w_param], False
    if w_param in (WM_XBUTTONDOWN, WM_XBUTTONUP):
        which = (int(data.mouseData) >> 16) & 0xFFFF
        return _XBUTTONS.get(which, UNKNOWN_BUTTON), w_param == WM_XBUTTONDOWN
    return None, False


class PressHoldMouseHook(_LowLevelHook):
    """``WH_MOUSE_LL`` hook behind every shortcut bound to a MOUSE button
    (:func:`exocortex.bindings.parse_binding`: ``"mouse_left"``, ``"mouse_middle"``,
    ``"mouse_right"``, ``"mouse4"`` / ``"mouse5"`` -- the thumb buttons -- with optional ``ctrl`` /
    ``alt`` / ``shift`` / ``win`` modifiers, as in ``"ctrl+mouse4"``). One hook serves the
    manual-advance gestures (the left button held = Text, middle = Code, mouse 4 = Code from
    scratch by default) and any other ``[hotkeys]`` entry bound to a mouse button. ``bindings``
    maps binding texts to callbacks; :meth:`rebind` replaces them in place (the Shortcuts tab, the
    Manual switch) without re-installing the hook.

    A press arms the binding whose button AND modifiers (read with ``GetAsyncKeyState`` at the
    press) match; a press with no matching binding passes straight through. ``require_hold``
    says, per binding text, whether the press must be HELD for ``hold_s`` (True, the default -- a
    bool for all, a mapping, or a callable taking the binding text, read at each release so a
    settings change needs no re-install). A required hold fires ONCE, the moment ``hold_s``
    elapses with the button still down -- through ``schedule(delay_s, check)``, a delayed-call
    primitive the caller provides (the overlay hands in ``QTimer.singleShot``; this module stays
    Qt-free) -- or, without a scheduler, on release. A binding whose hold is NOT required fires on
    release however short the press: a quick click counts, which is what lets a touchpad's
    three-finger tap (an instantaneous middle click, once Windows' touchpad settings map it so)
    trigger Code. Either way, moving the pointer more than ``move_px`` (a drag, a text selection,
    an autoscroll), pressing any other button, or a wheel notch cancels the press, so
    clicking-and-dragging never fires. ``hold_s`` may be a callable too.

    ``swallow`` says, per binding text (read at the press; the same forms as ``require_hold``,
    default False), whether the press is kept from the app underneath -- its down AND the
    matching up -- the way a keyboard hotkey is consumed. The overlay swallows the thumb buttons
    (a browser or an editor would otherwise go Back / Forward on the very press that asks for
    code) and modified clicks, never a bare left / middle / right press: those must still reach
    the app. INJECTED input is processed like real input on purpose: Windows delivers the
    three-finger-tap middle click through ``SendInput`` (it arrives with ``LLMHF_INJECTED`` set),
    so filtering it would break the touchpad path."""

    hook_id = WH_MOUSE_LL

    _CANCEL_MESSAGES = (WM_MOUSEWHEEL, WM_MOUSEHWHEEL)

    def __init__(
        self, bindings: "Mapping[str, Callable[[], None]]", *,
        hold_s: "float | Callable[[], float]" = 0.8,
        require_hold: "bool | Mapping[str, bool] | Callable[[str], bool]" = True,
        swallow: "bool | Mapping[str, bool] | Callable[[str], bool]" = False,
        schedule: "Callable[[float, Callable[[], None]], None] | None" = None,
        move_px: int = 12, user32: Any | None = None, clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(user32=user32)
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._by_button: dict[str, dict[frozenset[str], str]] = {}
        self._hold_s = hold_s
        self._require_hold = require_hold
        self._swallow = swallow
        self._schedule = schedule
        self._move_px = int(move_px)
        self._clock = clock
        self._armed: str | None = None       # the binding text of the press being tracked (one at a time)
        self._armed_button: str | None = None
        self._down_at = 0.0
        self._down_pos = (0, 0)
        self._hold_wanted = 0.8              # hold_s as read at the press
        self._cancelled = False
        self._fired = False
        self._swallowing = False             # the armed press is kept from the app underneath
        self._gen = 0                        # bumped per press; a scheduled check that no longer matches is stale
        self.fires = 0                       # holds / taps that fired (tests / status)
        self.rebind(bindings)

    @property
    def bindings(self) -> tuple[str, ...]:
        """The canonical binding texts currently bound."""
        return tuple(self._callbacks)

    def rebind(self, bindings: "Mapping[str, Callable[[], None]]") -> None:
        """Replace the bindings in place (a press in flight is forgotten). ``ValueError`` for an
        empty mapping, a text that is not a mouse binding, or two texts naming the same gesture."""
        callbacks: dict[str, Callable[[], None]] = {}
        by_button: dict[str, dict[frozenset[str], str]] = {}
        for text, callback in bindings.items():
            try:
                binding = parse_binding(text)
            except ValueError as e:
                raise ValueError(f"unknown mouse button: {text!r}") from e
            if binding.kind != "mouse":
                raise ValueError(f"not a mouse binding: {text!r}")
            if binding.text in callbacks:
                raise ValueError(f"duplicate mouse binding: {text!r}")
            callbacks[binding.text] = callback
            by_button.setdefault(binding.key, {})[frozenset(binding.modifiers)] = binding.text
        if not callbacks:
            raise ValueError("no button bindings")
        self._callbacks = callbacks
        self._by_button = by_button
        self._forget_press()

    def uninstall(self) -> None:
        """Uninstall AND forget any press in flight: with the hook gone the release is never
        seen, and stale press state would make the next install read every fresh press as a
        chord (`_armed is not None`) -- the gesture would be dead until a matching release."""
        super().uninstall()
        self._forget_press()

    def _forget_press(self) -> None:
        self._armed = None
        self._armed_button = None
        self._cancelled = False
        self._fired = False
        self._swallowing = False
        self._gen += 1                       # invalidates any pending scheduled _check_hold

    def _hold_seconds(self) -> float:
        value = self._hold_s() if callable(self._hold_s) else self._hold_s
        try:
            return max(0.05, float(value))
        except (TypeError, ValueError):
            return 0.8

    @staticmethod
    def _option(value: Any, text: str, default: bool) -> bool:
        try:
            if callable(value):
                return bool(value(text))
            if isinstance(value, Mapping):
                return bool(value.get(text, default))
            return bool(value)
        except Exception:  # noqa: BLE001
            return default

    def _hold_required(self, text: str) -> bool:
        return self._option(self._require_hold, text, True)

    def _swallow_wanted(self, text: str) -> bool:
        return self._option(self._swallow, text, False)

    def _held_modifiers(self) -> frozenset[str]:
        held: set[str] = set()
        try:
            state = self._user32.GetAsyncKeyState
            for vk, name in ((VK_CONTROL, "ctrl"), (VK_MENU, "alt"), (VK_SHIFT, "shift"),
                             (VK_LWIN, "win"), (VK_RWIN, "win")):
                if int(state(vk)) & 0x8000:
                    held.add(name)
        except Exception:  # noqa: BLE001
            return frozenset()
        return frozenset(held)

    def _match(self, button: str) -> str | None:
        """The binding text a press of ``button`` with the modifiers held right now arms, if any."""
        options = self._by_button.get(button)
        if not options:
            return None
        return options.get(self._held_modifiers())

    def _fire(self, text: str) -> None:
        self.fires += 1
        try:
            self._callbacks[text]()
        except Exception:  # noqa: BLE001
            log.exception("PressHoldMouseHook: callback for %r failed", text)

    def _check_hold(self, gen: int) -> None:
        """The scheduled hold check (``hold_s`` after the press): fire if that same press is
        still armed and still in place. A scheduler that ran early (Qt's coarse timers may fire
        up to ~5 % ahead) re-schedules the remainder instead of firing short."""
        if gen != self._gen or self._armed is None or self._cancelled or self._fired:
            return
        remaining = self._hold_wanted - (self._clock() - self._down_at)
        if remaining > 0.005:
            if self._schedule is not None:
                try:
                    self._schedule(remaining, lambda: self._check_hold(gen))
                except Exception:  # noqa: BLE001
                    log.exception("PressHoldMouseHook: schedule failed")
            return
        self._fired = True
        self._fire(self._armed)

    def _on_message(self, w_param: int, l_param: int) -> bool:
        if not l_param:
            return False
        data = MSLLHOOKSTRUCT.from_address(l_param)
        # No LLMHF_INJECTED filtering here -- the touchpad's three-finger-tap middle click IS
        # injected (Windows sends it through SendInput); see the class docstring.
        if w_param == WM_MOUSEMOVE:
            if self._armed is not None and not self._cancelled and not self._fired:
                if (abs(int(data.pt.x) - self._down_pos[0]) > self._move_px
                        or abs(int(data.pt.y) - self._down_pos[1]) > self._move_px):
                    self._cancelled = True   # a drag / selection, not a hold
            return False
        if w_param in self._CANCEL_MESSAGES:
            if self._armed is not None:
                self._cancelled = True
            return False
        button, down = mouse_button_of(w_param, data)
        if button is None:
            return False
        if down:
            if self._armed is not None:
                self._cancelled = True       # a second button mid-press: a chord, not the gesture
                return False
            text = self._match(button)
            if text is None:
                return False
            self._gen += 1
            self._armed, self._armed_button = text, button
            self._down_at = self._clock()
            self._down_pos = (int(data.pt.x), int(data.pt.y))
            self._hold_wanted = self._hold_seconds()
            self._cancelled = False
            self._fired = False
            self._swallowing = self._swallow_wanted(text)
            if self._schedule is not None:
                gen = self._gen
                try:
                    self._schedule(self._hold_wanted, lambda: self._check_hold(gen))
                except Exception:  # noqa: BLE001
                    log.exception("PressHoldMouseHook: schedule failed")
            return self._swallowing
        if button != self._armed_button:
            return False
        held = self._clock() - self._down_at
        text, swallowed = self._armed, self._swallowing
        fire = (text is not None and not self._fired and not self._cancelled
                and (held >= self._hold_wanted or not self._hold_required(text)))
        self._forget_press()                 # also invalidates the pending scheduled check
        if fire:
            self._fire(text)
        return swallowed


class ChordRecorder:
    """Qt-free core of the Shortcuts tab's **Set** button (2026-10-29): collects the keys and
    mouse buttons pressed after :meth:`start` into ONE binding text and reports it through
    ``on_done(text)`` once ``quiet_s`` (0.5 s) have passed since the last press with no key held
    any more -- so a combination is taken as a whole ("ctrl+alt+g"), a single key or a modifier on
    its own becomes a tap ("f9", "rshift"), and a mouse button a mouse binding ("mouse4", with the
    modifiers held at the press). ``"esc"`` cancels (``on_done(None)``); so does :meth:`cancel`.

    The adapters feed it: :class:`CaptureHooks` (the low-level hooks: anywhere on screen, the
    input swallowed while listening) or the settings dialog's Qt event filter when the hooks are
    unavailable. Timing goes through ``schedule(delay_s, check)`` like :class:`PressHoldMouseHook`
    (the dialog hands in ``QTimer.singleShot``); without a scheduler :meth:`finish_now` ends the
    recording. ``press`` / ``release`` take the token names of :mod:`exocortex.bindings`
    (``"lctrl"``, ``"g"``, ``"f9"``, ``"mouse4"``) and return True while the recorder is listening
    -- the caller's cue to swallow the event."""

    def __init__(
        self, on_done: Callable[[str | None], None], *, quiet_s: float = 0.5,
        schedule: "Callable[[float, Callable[[], None]], None] | None" = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_done = on_done
        self._quiet_s = max(0.05, float(quiet_s))
        self._schedule = schedule
        self._clock = clock
        self._listening = False
        self._held: set[str] = set()         # modifier keys currently down
        self._mods: set[str] = set()         # the modifiers held when the key was pressed
        self._key: str | None = None         # the last non-modifier token pressed
        self._last_modifier: str | None = None
        self._last_press = 0.0
        self._gen = 0
        self.result: str | None = None       # what the last recording reported

    @property
    def listening(self) -> bool:
        return self._listening

    def start(self) -> None:
        self._listening = True
        self._held.clear()
        self._mods.clear()
        self._key = None
        self._last_modifier = None
        self._gen += 1
        self.result = None

    def cancel(self) -> None:
        """Stop listening and report ``None`` (a no-op when not listening)."""
        if not self._listening:
            return
        self._listening = False
        self._gen += 1
        self._report(None)

    def press(self, token: str) -> bool:
        """A key or mouse button went down. Returns True while listening (swallow the event)."""
        if not self._listening:
            return False
        token = token.strip().lower()
        if token == "esc":
            self.cancel()
            return True
        if token in MODIFIER_OF_KEY:
            self._held.add(token)
            self._last_modifier = token
        else:
            self._key = token
            self._mods = {MODIFIER_OF_KEY[t] for t in self._held}
        self._last_press = self._clock()
        self._arm(self._quiet_s)
        return True

    def release(self, token: str) -> bool:
        """A key or mouse button went up. Returns True while listening (swallow the event)."""
        if not self._listening:
            return False
        self._held.discard(token.strip().lower())
        if not self._held and (self._key is not None or self._last_modifier is not None):
            self._arm(self._quiet_s - (self._clock() - self._last_press))
        return True

    def finish_now(self) -> None:
        """End the recording with what was pressed so far (tests; a caller with no scheduler)."""
        if self._listening:
            self._finish()

    def _arm(self, delay_s: float) -> None:
        self._gen += 1
        gen = self._gen
        if delay_s <= 0.005:
            self._finish()
            return
        if self._schedule is not None:
            try:
                self._schedule(delay_s, lambda: self._check(gen))
            except Exception:  # noqa: BLE001
                log.exception("ChordRecorder: schedule failed")

    def _check(self, gen: int) -> None:
        if gen != self._gen or not self._listening:
            return
        if self._held:
            return                           # a modifier is still down: its release re-arms
        remaining = self._quiet_s - (self._clock() - self._last_press)
        if remaining > 0.005:
            if self._schedule is not None:
                try:
                    self._schedule(remaining, lambda: self._check(gen))
                except Exception:  # noqa: BLE001
                    log.exception("ChordRecorder: schedule failed")
            return
        self._finish()

    def _finish(self) -> None:
        self._listening = False
        self._gen += 1
        if self._key is not None:
            parts = [m for m in MODIFIERS if m in self._mods] + [self._key]
        elif self._last_modifier is not None:
            parts = [self._last_modifier]
        else:
            self._report(None)
            return
        try:
            text: str | None = normalize_binding("+".join(parts))
        except ValueError:
            text = None
        self._report(text)

    def _report(self, text: str | None) -> None:
        self.result = text
        try:
            self._on_done(text)
        except Exception:  # noqa: BLE001
            log.exception("ChordRecorder: on_done failed")


class CaptureKeyHook(_LowLevelHook):
    """``WH_KEYBOARD_LL`` adapter of :class:`ChordRecorder`: every key the grammar has a name for
    is fed to the recorder and swallowed while it listens (so a hotkey registered by this app or
    another never fires, and nothing is typed into the app underneath)."""

    hook_id = WH_KEYBOARD_LL

    def __init__(self, recorder: ChordRecorder, *, user32: Any | None = None) -> None:
        super().__init__(user32=user32)
        self._recorder = recorder

    def _on_message(self, w_param: int, l_param: int) -> bool:
        if not l_param or not self._recorder.listening:
            return False
        token = key_token_for_vk(int(KBDLLHOOKSTRUCT.from_address(l_param).vkCode))
        if token is None:
            return False
        if w_param in (WM_KEYDOWN, WM_SYSKEYDOWN):
            return self._recorder.press(token)
        if w_param in (WM_KEYUP, WM_SYSKEYUP):
            return self._recorder.release(token)
        return False


class CaptureMouseHook(_LowLevelHook):
    """``WH_MOUSE_LL`` adapter of :class:`ChordRecorder`: a button pressed anywhere but over the
    caller's own window (``ignore_at(x, y)``: the settings dialog, whose Set / Cancel clicks must
    keep working) is fed to the recorder and swallowed, its release too."""

    hook_id = WH_MOUSE_LL

    def __init__(self, recorder: ChordRecorder, *, ignore_at: Callable[[int, int], bool] | None = None,
                 user32: Any | None = None) -> None:
        super().__init__(user32=user32)
        self._recorder = recorder
        self._ignore_at = ignore_at
        self._swallowed: set[str] = set()

    def _on_message(self, w_param: int, l_param: int) -> bool:
        if not l_param:
            return False
        data = MSLLHOOKSTRUCT.from_address(l_param)
        button, down = mouse_button_of(w_param, data)
        if button is None or button not in MOUSE_BUTTONS:
            return False
        if down:
            if not self._recorder.listening:
                return False
            if self._ignore_at is not None:
                try:
                    if self._ignore_at(int(data.pt.x), int(data.pt.y)):
                        return False
                except Exception:  # noqa: BLE001
                    log.exception("CaptureMouseHook: ignore_at failed")
            taken = self._recorder.press(button)
            if taken:
                self._swallowed.add(button)
            return taken
        if button in self._swallowed:
            self._swallowed.discard(button)
            self._recorder.release(button)
            return True
        return False


class CaptureHooks:
    """Both capture hooks as one object. ``install()`` needs the keyboard hook; the mouse hook is
    best-effort (keys only when it fails)."""

    def __init__(self, recorder: ChordRecorder, *, ignore_mouse_at: Callable[[int, int], bool] | None = None,
                 user32: Any | None = None) -> None:
        self.keys = CaptureKeyHook(recorder, user32=user32)
        self.mouse = CaptureMouseHook(recorder, ignore_at=ignore_mouse_at, user32=user32)

    @property
    def installed(self) -> bool:
        return self.keys.installed

    def install(self) -> bool:
        if not self.keys.install():
            return False
        if not self.mouse.install():
            log.warning("CaptureHooks: the mouse hook could not be installed; keys only")
        return True

    def uninstall(self) -> None:
        self.mouse.uninstall()
        self.keys.uninstall()
