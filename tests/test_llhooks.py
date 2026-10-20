"""Offline tests for exocortex.ui.llhooks: parse_bare_key / is_bare_key, BareKeyHook and
MouseWheelHook (both driven through a fake ``user32`` -- no real SetWindowsHookEx call is ever
made), and the shared _LowLevelHook install()/uninstall() bookkeeping.

No Qt is needed here: llhooks.py is plain ctypes, independent of PySide6.
"""
from __future__ import annotations

import ctypes
import logging
from typing import Any

import pytest

from exocortex.ui import llhooks


# =====================================================================================
# fake user32: records SetWindowsHookExW / UnhookWindowsHookEx / CallNextHookEx and captures
# the installed hook procedure so a test can drive it directly with fake messages.
# =====================================================================================
class FakeUser32:
    def __init__(
        self, *, handle: int = 1, async_state: dict[int, int] | None = None, call_next: int = 0,
    ) -> None:
        self._handle = handle
        self.proc: Any = None
        self.hook_id: int | None = None
        self.unhooked: list[int] = []
        self.call_next_calls: list[tuple[int, int, int, int]] = []
        self._async_state = dict(async_state or {})
        self._call_next = call_next

    def SetWindowsHookExW(self, hook_id: int, proc: Any, hmod: int, thread_id: int) -> int:  # noqa: N802
        self.hook_id = hook_id
        self.proc = proc
        return self._handle

    def UnhookWindowsHookEx(self, handle: int) -> int:  # noqa: N802
        self.unhooked.append(handle)
        return 1

    def CallNextHookEx(self, handle: int, n_code: int, w_param: int, l_param: int) -> int:  # noqa: N802
        self.call_next_calls.append((handle, n_code, w_param, l_param))
        return self._call_next

    def GetAsyncKeyState(self, vk: int) -> int:  # noqa: N802
        return self._async_state.get(vk, 0)


def _kbd(vk: int) -> "llhooks.KBDLLHOOKSTRUCT":
    return llhooks.KBDLLHOOKSTRUCT(vkCode=vk, scanCode=0, flags=0, time=0, dwExtraInfo=0)


def _send_key(fake: FakeUser32, down: bool, vk: int) -> int:
    msg = llhooks.WM_KEYDOWN if down else llhooks.WM_KEYUP
    struct = _kbd(vk)
    return int(fake.proc(0, msg, ctypes.addressof(struct)))


def _wheel_struct(high_word: int) -> "llhooks.MSLLHOOKSTRUCT":
    mouse_data = (high_word & 0xFFFF) << 16
    return llhooks.MSLLHOOKSTRUCT(
        pt=llhooks.POINT(0, 0), mouseData=mouse_data, flags=0, time=0, dwExtraInfo=0
    )


# =====================================================================================
# parse_bare_key / is_bare_key
# =====================================================================================
@pytest.mark.parametrize(
    "text,expected",
    [
        ("rshift", (0xA1,)),
        ("shift", (0xA0, 0xA1)),
        ("right", (0x27,)),
        ("a", (ord("A"),)),
    ],
)
def test_parse_bare_key_valid(text: str, expected: tuple[int, ...]) -> None:
    assert llhooks.parse_bare_key(text) == expected
    assert llhooks.is_bare_key(text) is True


@pytest.mark.parametrize("text", ["ctrl+alt+p", "", "foo"])
def test_parse_bare_key_invalid_raises_and_is_bare_key_false(text: str) -> None:
    with pytest.raises(ValueError):
        llhooks.parse_bare_key(text)
    assert llhooks.is_bare_key(text) is False


# =====================================================================================
# BareKeyHook: taps, cancellation, auto-repeat, timing, modifiers, multi-key routing
# =====================================================================================
def test_barekeyhook_tap_fires_once_and_counts() -> None:
    calls: list[int] = []
    fake = FakeUser32()
    hook = llhooks.BareKeyHook("rshift", lambda: calls.append(1), user32=fake)
    assert hook.install() is True
    vk = llhooks.VK_RSHIFT
    _send_key(fake, True, vk)
    _send_key(fake, False, vk)
    assert calls == [1]
    assert hook.taps == 1


def test_barekeyhook_modifier_key_never_swallowed_returns_call_next_value() -> None:
    fake = FakeUser32(call_next=42)
    hook = llhooks.BareKeyHook("rshift", lambda: None, user32=fake)
    hook.install()
    vk = llhooks.VK_RSHIFT
    assert _send_key(fake, True, vk) == 42
    assert _send_key(fake, False, vk) == 42


def test_barekeyhook_plain_key_swallowed_when_no_modifier_held() -> None:
    fake = FakeUser32()
    hook = llhooks.BareKeyHook("right", lambda: None, user32=fake)
    hook.install()
    vk = llhooks.parse_bare_key("right")[0]
    assert _send_key(fake, True, vk) == 1
    assert _send_key(fake, False, vk) == 1


def test_barekeyhook_plain_key_passes_through_when_modifier_held() -> None:
    fake = FakeUser32(async_state={llhooks.VK_SHIFT: 0x8000}, call_next=7)
    hook = llhooks.BareKeyHook("right", lambda: None, user32=fake)
    hook.install()
    vk = llhooks.parse_bare_key("right")[0]
    assert _send_key(fake, True, vk) == 7


def test_barekeyhook_other_key_between_down_and_up_cancels_tap() -> None:
    """A Shift-held capital letter: rshift down, "A" down/up while rshift is still held, then
    rshift up -- no tap (a different key came in between)."""
    calls: list[int] = []
    fake = FakeUser32()
    hook = llhooks.BareKeyHook("rshift", lambda: calls.append(1), user32=fake)
    hook.install()
    vk = llhooks.VK_RSHIFT
    _send_key(fake, True, vk)
    _send_key(fake, True, ord("A"))
    _send_key(fake, False, ord("A"))
    _send_key(fake, False, vk)
    assert calls == []
    assert hook.taps == 0


def test_barekeyhook_autorepeat_keeps_first_timestamp() -> None:
    """Several auto-repeat downs between the first down and the up must not touch the clock (the
    fake's 2-value iterator would raise if they did) and must not reset the tap's start time."""
    calls: list[int] = []
    times = iter([0.0, 0.5])
    fake = FakeUser32()
    hook = llhooks.BareKeyHook(
        "rshift", lambda: calls.append(1), user32=fake, max_tap_s=0.6, clock=lambda: next(times)
    )
    hook.install()
    vk = llhooks.VK_RSHIFT
    _send_key(fake, True, vk)
    for _ in range(5):
        _send_key(fake, True, vk)   # auto-repeat: armed already, no further clock() calls
    _send_key(fake, False, vk)
    assert calls == [1]
    assert hook.taps == 1


def test_barekeyhook_hold_longer_than_max_tap_s_does_not_fire() -> None:
    calls: list[int] = []
    times = iter([0.0, 1.0])
    fake = FakeUser32()
    hook = llhooks.BareKeyHook(
        "rshift", lambda: calls.append(1), user32=fake, max_tap_s=0.6, clock=lambda: next(times)
    )
    hook.install()
    vk = llhooks.VK_RSHIFT
    _send_key(fake, True, vk)
    _send_key(fake, False, vk)
    assert calls == []
    assert hook.taps == 0


def test_barekeyhook_multikey_routes_to_the_right_callback() -> None:
    up_calls: list[int] = []
    down_calls: list[int] = []
    fake = FakeUser32()
    hook = llhooks.BareKeyHook(
        {"up": lambda: up_calls.append(1), "down": lambda: down_calls.append(1)}, user32=fake
    )
    hook.install()
    up_vk = llhooks.parse_bare_key("up")[0]
    down_vk = llhooks.parse_bare_key("down")[0]

    _send_key(fake, True, up_vk)
    _send_key(fake, False, up_vk)
    assert up_calls == [1]
    assert down_calls == []

    _send_key(fake, True, down_vk)
    _send_key(fake, False, down_vk)
    assert down_calls == [1]
    assert up_calls == [1]   # unchanged


def test_barekeyhook_second_bound_key_pressed_while_first_is_down_cancels() -> None:
    up_calls: list[int] = []
    down_calls: list[int] = []
    fake = FakeUser32()
    hook = llhooks.BareKeyHook(
        {"up": lambda: up_calls.append(1), "down": lambda: down_calls.append(1)}, user32=fake
    )
    hook.install()
    up_vk = llhooks.parse_bare_key("up")[0]
    down_vk = llhooks.parse_bare_key("down")[0]

    _send_key(fake, True, up_vk)
    _send_key(fake, True, down_vk)   # a second bound key while the first is held: cancels
    _send_key(fake, False, down_vk)
    _send_key(fake, False, up_vk)
    assert up_calls == []
    assert down_calls == []


def test_barekeyhook_negative_ncode_passes_through() -> None:
    fake = FakeUser32(call_next=99)
    hook = llhooks.BareKeyHook("rshift", lambda: None, user32=fake)
    hook.install()
    struct = _kbd(llhooks.VK_RSHIFT)
    result = int(fake.proc(-1, llhooks.WM_KEYDOWN, ctypes.addressof(struct)))
    assert result == 99
    assert fake.call_next_calls  # forwarded to CallNextHookEx


def test_barekeyhook_callback_exception_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    def boom() -> None:
        raise RuntimeError("boom")

    fake = FakeUser32()
    hook = llhooks.BareKeyHook("rshift", boom, user32=fake)
    hook.install()
    vk = llhooks.VK_RSHIFT
    with caplog.at_level(logging.ERROR):
        _send_key(fake, True, vk)
        _send_key(fake, False, vk)   # must not raise out of the hook procedure
    assert any("boom" in r.getMessage() or "callback" in r.getMessage().lower() for r in caplog.records)


# =====================================================================================
# MouseWheelHook
# =====================================================================================
def test_mousewheelhook_positive_delta_and_vertical() -> None:
    seen: list[tuple[int, bool]] = []
    fake = FakeUser32()
    hook = llhooks.MouseWheelHook(lambda d, h: seen.append((d, h)) or False, user32=fake)
    hook.install()
    struct = _wheel_struct(120)
    fake.proc(0, llhooks.WM_MOUSEWHEEL, ctypes.addressof(struct))
    assert seen == [(120, False)]


def test_mousewheelhook_negative_delta() -> None:
    seen: list[tuple[int, bool]] = []
    fake = FakeUser32()
    hook = llhooks.MouseWheelHook(lambda d, h: seen.append((d, h)) or False, user32=fake)
    hook.install()
    struct = _wheel_struct(0xFF88)
    fake.proc(0, llhooks.WM_MOUSEWHEEL, ctypes.addressof(struct))
    assert seen == [(-120, False)]


def test_mousewheelhook_horizontal_message_sets_horizontal_true() -> None:
    seen: list[tuple[int, bool]] = []
    fake = FakeUser32()
    hook = llhooks.MouseWheelHook(lambda d, h: seen.append((d, h)) or False, user32=fake)
    hook.install()
    struct = _wheel_struct(120)
    fake.proc(0, llhooks.WM_MOUSEHWHEEL, ctypes.addressof(struct))
    assert seen == [(120, True)]


def test_mousewheelhook_swallowed_only_when_callback_returns_true() -> None:
    struct = _wheel_struct(120)

    fake_false = FakeUser32(call_next=5)
    hook_false = llhooks.MouseWheelHook(lambda d, h: False, user32=fake_false)
    hook_false.install()
    assert int(fake_false.proc(0, llhooks.WM_MOUSEWHEEL, ctypes.addressof(struct))) == 5

    fake_true = FakeUser32(call_next=5)
    hook_true = llhooks.MouseWheelHook(lambda d, h: True, user32=fake_true)
    hook_true.install()
    assert int(fake_true.proc(0, llhooks.WM_MOUSEWHEEL, ctypes.addressof(struct))) == 1


def test_mousewheelhook_other_messages_pass_through_and_callback_not_called() -> None:
    calls: list[tuple[int, bool]] = []
    fake = FakeUser32(call_next=3)
    hook = llhooks.MouseWheelHook(lambda d, h: calls.append((d, h)) or True, user32=fake)
    hook.install()
    struct = _wheel_struct(120)
    result = int(fake.proc(0, llhooks.WM_KEYDOWN, ctypes.addressof(struct)))
    assert result == 3
    assert calls == []


# =====================================================================================
# _LowLevelHook: install()/uninstall() bookkeeping, failure, off-platform
# =====================================================================================
def test_lowlevelhook_install_success_sets_installed_and_uninstall_clears_it() -> None:
    fake = FakeUser32(handle=77)
    hook = llhooks.MouseWheelHook(lambda d, h: False, user32=fake)
    assert hook.install() is True
    assert hook.installed is True
    assert hook.failed is False
    hook.uninstall()
    assert hook.installed is False
    assert fake.unhooked == [77]


def test_lowlevelhook_install_failure_sets_failed_and_second_attempt_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeUser32(handle=0)
    hook = llhooks.MouseWheelHook(lambda d, h: False, user32=fake)
    with caplog.at_level(logging.WARNING):
        assert hook.install() is False
    assert hook.installed is False
    assert hook.failed is True
    n_records = len(caplog.records)
    assert hook.install() is False
    assert len(caplog.records) == n_records   # no repeat warning


def test_lowlevelhook_null_user32_off_platform_install_is_false() -> None:
    hook = llhooks.MouseWheelHook(lambda d, h: False, user32=llhooks._NullUser32())
    assert hook.install() is False
    assert hook.installed is False
    assert hook.failed is True


def test_lowlevelhook_double_install_is_a_noop() -> None:
    fake = FakeUser32(handle=1)
    hook = llhooks.MouseWheelHook(lambda d, h: False, user32=fake)
    assert hook.install() is True
    first_proc = fake.proc
    assert hook.install() is True
    assert fake.proc is first_proc   # SetWindowsHookExW not called again
