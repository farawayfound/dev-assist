"""One grammar for every shortcut in ``[hotkeys]`` (2026-10-29, Settings > Shortcuts).

A binding is a lower-case, ``+``-separated text naming ONE trigger plus any of the modifiers
``ctrl``, ``alt``, ``shift`` and ``win``. The trigger decides how the overlay installs it:

* a keyboard key with at least one modifier (``"ctrl+alt+g"``) is a ``RegisterHotKey`` hotkey
  (:class:`exocortex.ui.hotkeys.GlobalHotkeys`) -- what every hotkey was before this module;
* a single key with no modifier (``"rshift"``, ``"f9"``, ``"right"``) is a TAP of that key:
  pressed and released with no other key in between (:class:`exocortex.ui.llhooks.BareKeyHook`).
  Before 2026-10-29 only ``toggle_pause`` could be one; now any shortcut can;
* a mouse button (``"mouse4"``, ``"mouse_middle"``, ``"ctrl+mouse_left"``) is a press of that
  button anywhere on screen with exactly those modifiers held
  (:class:`exocortex.ui.llhooks.PressHoldMouseHook`): a click, or -- for the manual-advance
  gestures whose ``[handsfree].manual_hold_*`` asks for it -- a press-and-hold.

``""`` means unbound. :func:`parse_binding` is the one validator the Shortcuts tab, the overlay
and the hands-free status line agree through; :func:`describe_binding` renders a binding the way
the UI shows it ("Ctrl+Alt+G", "Right Shift (tap)", "Mouse 4 (thumb, back)").

Qt-free and ctypes-free on purpose: the engine side (:func:`manual_gesture_summary`, the
hands-free "armed" detail) imports it too. :data:`NAMED_VKS` lived in :mod:`exocortex.ui.hotkeys`
before and is still importable there under its old name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from exocortex.config import Config

#: Named keyboard keys -> Windows virtual-key codes. Letters and digits are ``ord(upper)``.
NAMED_VKS: dict[str, int] = {
    "space": 0x20,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "enter": 0x0D,
    "esc": 0x1B,
    "tab": 0x09,
    "backspace": 0x08,
    "insert": 0x2D,
    "delete": 0x2E,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "grave": 0xC0,
    "-": 0xBD,
    "=": 0xBB,
    "[": 0xDB,
    "]": 0xDD,
    ";": 0xBA,
    "'": 0xDE,
    ",": 0xBC,
    ".": 0xBE,
    "/": 0xBF,
    "backslash": 0xDC,
}
for _n in range(1, 13):
    NAMED_VKS[f"f{_n}"] = 0x70 + (_n - 1)
del _n
_KEY_NAME_BY_VK: dict[int, str] = {vk: name for name, vk in NAMED_VKS.items()}

#: Canonical modifiers, in the order a binding text spells them.
MODIFIERS: tuple[str, ...] = ("ctrl", "alt", "shift", "win")
MODIFIER_ALIASES: dict[str, str] = {
    "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "shift": "shift",
    "win": "win", "meta": "win", "super": "win",
}
#: Which canonical modifier a modifier KEY name belongs to -- one side ("rshift", "lctrl") or
#: either side ("shift"). These are the names a bare-key tap of a modifier is written with.
MODIFIER_OF_KEY: dict[str, str] = {
    "rshift": "shift", "lshift": "shift", "shift": "shift",
    "rctrl": "ctrl", "lctrl": "ctrl", "ctrl": "ctrl",
    "rcontrol": "ctrl", "lcontrol": "ctrl", "control": "ctrl",
    "ralt": "alt", "lalt": "alt", "alt": "alt",
    "rwin": "win", "lwin": "win", "win": "win",
}
MODIFIER_KEYS: tuple[str, ...] = tuple(MODIFIER_OF_KEY)
#: Side-specific modifier virtual keys (what a low-level keyboard hook reports) -> the key name;
#: the three generic codes are what Qt / ``GetAsyncKeyState`` callers may hand in instead.
MODIFIER_KEY_BY_VK: dict[int, str] = {
    0xA0: "lshift", 0xA1: "rshift", 0xA2: "lctrl", 0xA3: "rctrl", 0xA4: "lalt", 0xA5: "ralt",
    0x5B: "lwin", 0x5C: "rwin", 0x10: "shift", 0x11: "ctrl", 0x12: "alt",
}

#: Mouse buttons, canonical names. "mouse4" / "mouse5" are the thumb buttons: XBUTTON1 ("Back")
#: and XBUTTON2 ("Forward") to Windows, buttons 4 and 5 in the usual gaming convention.
MOUSE_BUTTONS: tuple[str, ...] = ("mouse_left", "mouse_middle", "mouse_right", "mouse4", "mouse5")
MOUSE_ALIASES: dict[str, str] = {
    "mouse_left": "mouse_left", "lmb": "mouse_left", "lbutton": "mouse_left", "left_button": "mouse_left",
    "mouse_middle": "mouse_middle", "mmb": "mouse_middle", "mbutton": "mouse_middle",
    "middle_button": "mouse_middle",
    "mouse_right": "mouse_right", "rmb": "mouse_right", "rbutton": "mouse_right", "right_button": "mouse_right",
    "mouse4": "mouse4", "mouse_4": "mouse4", "x1": "mouse4", "xbutton1": "mouse4", "button4": "mouse4",
    "mouse_back": "mouse4", "back": "mouse4",
    "mouse5": "mouse5", "mouse_5": "mouse5", "x2": "mouse5", "xbutton2": "mouse5", "button5": "mouse5",
    "mouse_forward": "mouse5", "forward": "mouse5",
}
#: Long and short labels of each mouse button (the Shortcuts tab; the status line).
MOUSE_LABELS: dict[str, tuple[str, str]] = {
    "mouse_left": ("Left button", "left"),
    "mouse_middle": ("Middle button", "middle"),
    "mouse_right": ("Right button", "right"),
    "mouse4": ("Mouse 4 (thumb, back)", "mouse 4"),
    "mouse5": ("Mouse 5 (thumb, forward)", "mouse 5"),
}
_MODIFIER_LABELS = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win"}
_KEY_LABELS = {
    "space": "Space", "left": "Left", "up": "Up", "right": "Right", "down": "Down", "enter": "Enter",
    "esc": "Esc", "tab": "Tab", "backspace": "Backspace", "insert": "Insert", "delete": "Delete",
    "home": "Home", "end": "End", "pageup": "Page Up", "pagedown": "Page Down", "grave": "`",
    "backslash": "\\",
}


@dataclass(frozen=True)
class Binding:
    """A parsed binding. ``kind``: ``"hotkey"`` (a key with modifiers, RegisterHotKey), ``"tap"``
    (a single key, a low-level keyboard hook) or ``"mouse"`` (a mouse button, the mouse hook)."""

    text: str                    # canonical text: modifiers in MODIFIERS order, then the key
    modifiers: tuple[str, ...]
    key: str                     # a key name, a modifier key name (tap) or a MOUSE_BUTTONS token
    kind: str

    @property
    def is_mouse(self) -> bool:
        return self.kind == "mouse"


def parse_binding(text: str) -> Binding:
    """Parse one binding text; ``ValueError`` for anything that is not exactly one trigger with
    optional modifiers (a blank text included -- callers treat ``""`` as unbound before calling)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"empty binding: {text!r}")
    parts = [p.strip().lower() for p in text.split("+")]
    if any(not p for p in parts):
        raise ValueError(f"malformed binding: {text!r}")
    mods: list[str] = []
    key: str | None = None
    for part in parts:
        if part in MODIFIER_ALIASES and len(parts) > 1:
            mod = MODIFIER_ALIASES[part]
            if mod not in mods:
                mods.append(mod)
            continue
        if key is not None:
            raise ValueError(f"binding names more than one key: {text!r}")
        if part in MOUSE_ALIASES:
            key = MOUSE_ALIASES[part]
        elif part in MODIFIER_OF_KEY:
            key = part
        elif part in NAMED_VKS or (len(part) == 1 and (part.isalpha() or part.isdigit())):
            key = part
        else:
            raise ValueError(f"unknown key {part!r} in binding {text!r}")
    if key is None:
        raise ValueError(f"binding has no key, only modifiers: {text!r}")
    if key in MODIFIER_OF_KEY and mods:
        raise ValueError(f"a modifier key can only be bound on its own, as a tap: {text!r}")
    ordered = tuple(m for m in MODIFIERS if m in mods)
    if key in MOUSE_BUTTONS:
        kind = "mouse"
    elif ordered:
        kind = "hotkey"
    else:
        kind = "tap"
    return Binding("+".join((*ordered, key)), ordered, key, kind)


def normalize_binding(text: str) -> str:
    """The canonical text of ``text``; ``""`` for a blank one; ``ValueError`` when invalid."""
    if not isinstance(text, str) or not text.strip():
        return ""
    return parse_binding(text).text


def is_valid_binding(text: str) -> bool:
    """Whether ``text`` is a binding or blank (unbound)."""
    try:
        normalize_binding(text)
    except ValueError:
        return False
    return True


def binding_kind(text: str) -> str:
    """``"hotkey"`` | ``"tap"`` | ``"mouse"`` | ``""`` (blank); ``ValueError`` when invalid."""
    if not isinstance(text, str) or not text.strip():
        return ""
    return parse_binding(text).kind


def _key_label(key: str) -> str:
    if key in _KEY_LABELS:
        return _KEY_LABELS[key]
    if len(key) > 1 and key[0] == "f" and key[1:].isdigit():
        return key.upper()
    return key.upper() if len(key) == 1 else key


def describe_binding(text: str, *, short: bool = False) -> str:
    """The binding in words: "Ctrl+Alt+G", "Right Shift (tap)", "F9 (tap)", "Mouse 4 (thumb,
    back)", "Ctrl+Mouse 4". ``short`` is the status-line form ("left", "mouse 4", "Right Shift",
    "Ctrl+Alt+G"). A blank text is "(none)" ("" when short); an invalid one is returned as typed."""
    raw = (text or "").strip()
    if not raw:
        return "" if short else "(none)"
    try:
        binding = parse_binding(raw)
    except ValueError:
        return raw
    mods = [_MODIFIER_LABELS[m] for m in binding.modifiers]
    if binding.kind == "mouse":
        long_label, short_label = MOUSE_LABELS[binding.key]
        if short:
            name = short_label
        else:
            name = long_label.split(" (")[0] if mods else long_label
        return "+".join((*mods, name))
    if binding.kind == "tap":
        key = binding.key
        if key in MODIFIER_OF_KEY:
            if key in ("shift", "ctrl", "control", "alt", "win"):
                side = ""
            else:
                side = "Right " if key[0] == "r" else "Left "
            label = side + _MODIFIER_LABELS[MODIFIER_OF_KEY[key]]
        else:
            label = _key_label(key)
        return label if short else f"{label} (tap)"
    return "+".join((*mods, _key_label(binding.key)))


def key_token_for_vk(vk: int) -> str | None:
    """The key name a virtual-key code is written as in a binding -- a side-specific modifier
    ("lctrl"), a letter / digit, or a named key -- or ``None`` for a key the grammar has no name
    for (a media key, Num Lock...)."""
    if vk in MODIFIER_KEY_BY_VK:
        return MODIFIER_KEY_BY_VK[vk]
    if 0x41 <= vk <= 0x5A or 0x30 <= vk <= 0x39:
        return chr(vk).lower()
    return _KEY_NAME_BY_VK.get(vk)


def manual_gesture_summary(cfg: "Config") -> str:
    """The manual-advance gestures as configured, for the hands-free "armed" detail and the Manual
    toast: "hold left = Text, click middle = Code, click mouse 4 = Code from scratch" with the
    defaults. Follows ``[hotkeys].manual_*`` and the ``[handsfree].manual_hold_*`` switches; an
    unbound or invalid gesture is left out."""
    h, hf = cfg.hotkeys, cfg.handsfree
    parts: list[str] = []
    for text, hold, name in (
        (h.manual_text, hf.manual_hold_text, "Text"),
        (h.manual_code, hf.manual_hold_code, "Code"),
        (h.manual_code_fresh, hf.manual_hold_code_fresh, "Code from scratch"),
    ):
        try:
            binding = parse_binding(text)
        except ValueError:
            continue
        if binding.kind == "mouse":
            verb = "hold" if hold else "click"
        elif binding.kind == "tap":
            verb = "tap"
        else:
            verb = "press"
        parts.append(f"{verb} {describe_binding(binding.text, short=True)} = {name}")
    return ", ".join(parts)
