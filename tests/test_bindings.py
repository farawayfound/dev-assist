"""exocortex.bindings (2026-10-29): the one grammar for every `[hotkeys]` entry -- key combinations
(RegisterHotKey hotkeys), single keys (taps) and mouse buttons -- its canonical forms and aliases,
what it rejects, the human descriptions, and the manual-gesture summary the hands-free status line
and the Manual toast show. Fully offline, no Qt."""
from __future__ import annotations

import dataclasses

import pytest

from exocortex import bindings as b
from exocortex.config import Config, HotkeysConfig
from exocortex.ui import hotkeys as hk
from exocortex.ui import llhooks


@pytest.mark.parametrize(
    "text,canonical,kind,key,mods",
    [
        ("ctrl+alt+g", "ctrl+alt+g", "hotkey", "g", ("ctrl", "alt")),
        ("Alt+Ctrl+G", "ctrl+alt+g", "hotkey", "g", ("ctrl", "alt")),          # order and case do not matter
        ("control+shift+f5", "ctrl+shift+f5", "hotkey", "f5", ("ctrl", "shift")),
        ("win+space", "win+space", "hotkey", "space", ("win",)),
        ("meta+=", "win+=", "hotkey", "=", ("win",)),
        (" ctrl + alt + x ", "ctrl+alt+x", "hotkey", "x", ("ctrl", "alt")),
        ("rshift", "rshift", "tap", "rshift", ()),
        ("f9", "f9", "tap", "f9", ()),
        ("right", "right", "tap", "right", ()),
        ("a", "a", "tap", "a", ()),
        ("shift", "shift", "tap", "shift", ()),
        ("mouse4", "mouse4", "mouse", "mouse4", ()),
        ("x1", "mouse4", "mouse", "mouse4", ()),
        ("button4", "mouse4", "mouse", "mouse4", ()),
        ("button5", "mouse5", "mouse", "mouse5", ()),
        ("xbutton2", "mouse5", "mouse", "mouse5", ()),
        ("lmb", "mouse_left", "mouse", "mouse_left", ()),
        ("MOUSE_MIDDLE", "mouse_middle", "mouse", "mouse_middle", ()),
        ("rbutton", "mouse_right", "mouse", "mouse_right", ()),
        ("ctrl+mouse_left", "ctrl+mouse_left", "mouse", "mouse_left", ("ctrl",)),
        ("shift+alt+mouse5", "alt+shift+mouse5", "mouse", "mouse5", ("alt", "shift")),
    ],
)
def test_parse_binding_canonical_forms(text: str, canonical: str, kind: str, key: str, mods: tuple[str, ...]) -> None:
    parsed = b.parse_binding(text)
    assert parsed.text == canonical
    assert parsed.kind == kind
    assert parsed.key == key
    assert parsed.modifiers == mods
    assert parsed.is_mouse == (kind == "mouse")
    assert b.normalize_binding(text) == canonical
    assert b.binding_kind(text) == kind
    assert b.is_valid_binding(text)


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "ctrl+", "+a", "ctrl+alt", "ctrl+a+b", "ctrl+rshift", "foo", "left+mouse4", "mouse6",
     "ctrl+alt+mouse4+a", "ctrl++a", "button"],
)
def test_parse_binding_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        b.parse_binding(bad)


def test_blank_is_unbound_not_invalid() -> None:
    assert b.normalize_binding("") == ""
    assert b.normalize_binding("   ") == ""
    assert b.is_valid_binding("") and b.is_valid_binding("  ")
    assert b.binding_kind("") == ""
    assert not b.is_valid_binding("ctrl+")


def test_the_arrow_key_left_is_a_key_and_never_a_mouse_button() -> None:
    """The old manual-advance code keyed its mouse bindings "left" / "middle"; in the unified
    grammar "left" is the arrow key, the mouse buttons are "mouse_left" etc."""
    assert b.parse_binding("left").kind == "tap"
    assert b.parse_binding("ctrl+alt+left") == b.Binding("ctrl+alt+left", ("ctrl", "alt"), "left", "hotkey")
    assert b.parse_binding("mouse_left").kind == "mouse"
    assert "left" not in b.MOUSE_ALIASES and "middle" not in b.MOUSE_ALIASES


@pytest.mark.parametrize(
    "text,long,short",
    [
        ("ctrl+alt+g", "Ctrl+Alt+G", "Ctrl+Alt+G"),
        ("ctrl+alt+left", "Ctrl+Alt+Left", "Ctrl+Alt+Left"),
        ("win+space", "Win+Space", "Win+Space"),
        ("rshift", "Right Shift (tap)", "Right Shift"),
        ("lctrl", "Left Ctrl (tap)", "Left Ctrl"),
        ("rcontrol", "Right Ctrl (tap)", "Right Ctrl"),
        ("shift", "Shift (tap)", "Shift"),
        ("f9", "F9 (tap)", "F9"),
        ("right", "Right (tap)", "Right"),
        ("pageup", "Page Up (tap)", "Page Up"),
        ("grave", "` (tap)", "`"),
        ("mouse_left", "Left button", "left"),
        ("mouse_middle", "Middle button", "middle"),
        ("mouse4", "Mouse 4 (thumb, back)", "mouse 4"),
        ("mouse5", "Mouse 5 (thumb, forward)", "mouse 5"),
        ("ctrl+mouse4", "Ctrl+Mouse 4", "Ctrl+mouse 4"),
        ("ctrl+mouse_left", "Ctrl+Left button", "Ctrl+left"),
        ("", "(none)", ""),
        ("ctrl+", "ctrl+", "ctrl+"),          # invalid: returned as typed
    ],
)
def test_describe_binding(text: str, long: str, short: str) -> None:
    assert b.describe_binding(text) == long
    assert b.describe_binding(text, short=True) == short


def test_every_default_hotkey_is_valid_and_agrees_with_the_older_parsers() -> None:
    """Every string in HotkeysConfig parses, and a hotkey / tap still parses with the parser that
    actually registers it (RegisterHotKey's `parse_hotkey`, the bare-key hook's `parse_bare_key`)."""
    defaults = HotkeysConfig()
    seen: set[str] = set()
    for f in dataclasses.fields(HotkeysConfig):
        value = getattr(defaults, f.name)
        if not isinstance(value, str) or not value:
            continue                                           # unbound: dev-assist ships the move_* nudges off
        parsed = b.parse_binding(value)
        assert parsed.text == value, f.name                    # the defaults are written canonically
        assert parsed.text not in seen, f"{f.name} shares {value!r}"
        seen.add(parsed.text)
        if parsed.kind == "hotkey":
            hk.parse_hotkey(value)
        elif parsed.kind == "tap":
            llhooks.parse_bare_key(value)
    assert {b.parse_binding(getattr(defaults, k)).kind for k in ("manual_text", "manual_code", "manual_code_fresh")} == {"mouse"}
    assert defaults.manual_code_fresh == "mouse4"


def test_key_token_for_vk() -> None:
    assert b.key_token_for_vk(0xA1) == "rshift"
    assert b.key_token_for_vk(0xA2) == "lctrl"
    assert b.key_token_for_vk(0x5C) == "rwin"
    assert b.key_token_for_vk(0x10) == "shift"             # the generic code Qt may report
    assert b.key_token_for_vk(0x41) == "a"
    assert b.key_token_for_vk(0x30) == "0"
    assert b.key_token_for_vk(0x70) == "f1"
    assert b.key_token_for_vk(0x7B) == "f12"
    assert b.key_token_for_vk(0x20) == "space"
    assert b.key_token_for_vk(0x27) == "right"
    assert b.key_token_for_vk(0x90) is None                # Num Lock: no name in the grammar
    # round trip through the table
    for name, vk in b.NAMED_VKS.items():
        assert b.key_token_for_vk(vk) == name


def test_named_vks_still_importable_from_hotkeys_under_the_old_name() -> None:
    assert hk._NAMED_VKS is b.NAMED_VKS
    assert llhooks._NAMED_VKS is b.NAMED_VKS
    assert hk.parse_hotkey("ctrl+alt+space") == (hk.MOD_CONTROL | hk.MOD_ALT, 0x20)


def test_manual_gesture_summary_defaults_and_follows_the_config() -> None:
    cfg = Config()
    assert b.manual_gesture_summary(cfg) == "hold left = Text, click middle = Code, click mouse 4 = Code from scratch"
    cfg.handsfree.manual_hold_code = True
    cfg.handsfree.manual_hold_text = False
    assert b.manual_gesture_summary(cfg).startswith("click left = Text, hold middle = Code")
    cfg.hotkeys.manual_text = ""                           # unbound: left out
    cfg.hotkeys.manual_code_fresh = "f9"                   # a key tap
    cfg.hotkeys.manual_code = "ctrl+alt+k"                 # a hotkey
    assert b.manual_gesture_summary(cfg) == "press Ctrl+Alt+K = Code, tap F9 = Code from scratch"
    cfg.hotkeys.manual_code = "bogus+"                     # invalid: left out, never raises
    assert b.manual_gesture_summary(cfg) == "tap F9 = Code from scratch"
    cfg.hotkeys.manual_code_fresh = ""
    assert b.manual_gesture_summary(cfg) == ""
