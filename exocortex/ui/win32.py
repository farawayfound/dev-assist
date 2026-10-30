"""ctypes-only Win32 helpers for the overlay window.

Covers the handful of window tweaks a visible, always-on-top HUD needs: click-through
(``WS_EX_TRANSPARENT``), never-steal-focus (``WS_EX_NOACTIVATE``), a periodic topmost
backstop (``SetWindowPos HWND_TOPMOST``), physical-pixel placement for the ghost-text window,
and Windows error-message formatting. Pure ctypes, no pywin32, and no Qt.

dev-assist is a VISIBLE copilot: it is meant to be screen-shared and to appear normally in the
taskbar / Alt-Tab and in any process listing. Nothing here conceals the window from screen
capture, from remote-desktop / mirror / virtual-display pipelines, or from the taskbar.

Every public function is safe to call on any platform and with any ``hwnd`` (including ``0``):
off Windows, or with a null handle, it logs at most and returns a "did nothing" value
(``False``/``None``) instead of raising. On Windows a failing Win32 call is logged (with
``GetLastError``) and reported through the return value, never as an exception.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes

log = logging.getLogger(__name__)

#: True only when running on Windows; every function below checks this before touching
#: the real Win32 API so the module imports and runs harmlessly elsewhere.
IS_WINDOWS = sys.platform == "win32"

# ---- extended window style bits (GWL_EXSTYLE) -----------------------------------------
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000        # Qt sets this on a translucent frameless window; kept when toggling transparency
WS_EX_TRANSPARENT = 0x00000020    # click-through: mouse events fall through to the app underneath
WS_EX_NOACTIVATE = 0x08000000     # the window never steals keyboard focus on show

# ---- SetWindowPos ---------------------------------------------------------------------
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010

_FORMAT_MESSAGE_FROM_SYSTEM = 0x00001000
_FORMAT_MESSAGE_IGNORE_INSERTS = 0x00000200

#: ``MonitorFromPoint`` flag: nearest monitor to a point.
_MONITOR_DEFAULTTONEAREST = 2
_MDT_EFFECTIVE_DPI = 0


def _load_dlls() -> tuple[ctypes.WinDLL, ctypes.WinDLL] | tuple[None, None]:
    """Load user32/kernel32 (per-thread last-error tracking). ``(None, None)`` off Windows."""
    if not IS_WINDOWS:
        return None, None
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongPtrW.restype = ctypes.c_longlong
    user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_longlong]
    user32.SetWindowLongPtrW.restype = ctypes.c_longlong
    user32.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL

    kernel32.FormatMessageW.argtypes = [
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    kernel32.FormatMessageW.restype = wintypes.DWORD
    return user32, kernel32


_user32, _kernel32 = _load_dlls()


def last_error_message(code: int) -> str:
    """Human-readable text for a Win32 ``GetLastError`` code via ``FormatMessageW``."""
    if not IS_WINDOWS or not code:
        return f"error {code}"
    buf = ctypes.create_unicode_buffer(512)
    try:
        n = _kernel32.FormatMessageW(  # type: ignore[union-attr]
            _FORMAT_MESSAGE_FROM_SYSTEM | _FORMAT_MESSAGE_IGNORE_INSERTS,
            None,
            code,
            0,
            buf,
            len(buf),
            None,
        )
    except OSError:
        log.exception("FormatMessageW raised for code=%s", code)
        return f"error {code}"
    if not n:
        return f"error {code}"
    return f"{buf.value.strip()} (error {code})"


def _get_exstyle(hwnd: int) -> int | None:
    """Current ``GWL_EXSTYLE`` bits for ``hwnd``, or ``None`` if unavailable."""
    if not IS_WINDOWS or not hwnd:
        return None
    ctypes.set_last_error(0)
    try:
        value = _user32.GetWindowLongPtrW(wintypes.HWND(hwnd), GWL_EXSTYLE)  # type: ignore[union-attr]
    except OSError:
        log.exception("GetWindowLongPtrW raised for hwnd=%s", hwnd)
        return None
    if value == 0:
        err = ctypes.get_last_error()
        if err != 0:
            log.warning("GetWindowLongPtrW failed for hwnd=%s: %s", hwnd, last_error_message(err))
            return None
    return int(value)


def _set_exstyle(hwnd: int, value: int) -> bool:
    """Set ``GWL_EXSTYLE`` bits for ``hwnd``. 0 is a legal return, so ``SetLastError(0)`` first."""
    if not IS_WINDOWS or not hwnd:
        return False
    ctypes.set_last_error(0)
    try:
        result = _user32.SetWindowLongPtrW(wintypes.HWND(hwnd), GWL_EXSTYLE, ctypes.c_longlong(value))  # type: ignore[union-attr]
    except OSError:
        log.exception("SetWindowLongPtrW raised for hwnd=%s", hwnd)
        return False
    if result == 0:
        err = ctypes.get_last_error()
        if err != 0:
            log.warning("SetWindowLongPtrW failed for hwnd=%s: %s", hwnd, last_error_message(err))
            return False
    return True


def set_click_through(hwnd: int, enabled: bool) -> None:
    """Toggle ``WS_EX_TRANSPARENT`` (keeping ``WS_EX_LAYERED``) via raw ``SetWindowLongPtrW``.

    Deliberately does not use Qt's ``WindowTransparentForInput`` flag, which can recreate the
    native window.
    """
    cur = _get_exstyle(hwnd)
    if cur is None:
        return
    new = cur | WS_EX_LAYERED
    new = (new | WS_EX_TRANSPARENT) if enabled else (new & ~WS_EX_TRANSPARENT)
    _set_exstyle(hwnd, new)


def set_no_activate(hwnd: int, enabled: bool) -> None:
    """Toggle ``WS_EX_NOACTIVATE`` (the window never steals keyboard focus on show)."""
    cur = _get_exstyle(hwnd)
    if cur is None:
        return
    new = (cur | WS_EX_NOACTIVATE) if enabled else (cur & ~WS_EX_NOACTIVATE)
    _set_exstyle(hwnd, new)


def set_topmost(hwnd: int) -> None:
    """Force ``hwnd`` back to the top of the z-order (``SetWindowPos HWND_TOPMOST``).

    Intended as a periodic backstop (e.g. a ``QTimer`` every ~1.5 s) against meeting apps
    stealing topmost z-order back from ``WindowStaysOnTopHint`` alone.
    """
    if not IS_WINDOWS or not hwnd:
        return
    ctypes.set_last_error(0)
    try:
        ok = _user32.SetWindowPos(  # type: ignore[union-attr]
            wintypes.HWND(hwnd),
            wintypes.HWND(HWND_TOPMOST),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        )
    except OSError:
        log.exception("SetWindowPos raised for hwnd=%s", hwnd)
        return
    if not ok:
        err = ctypes.get_last_error()
        log.debug("SetWindowPos(HWND_TOPMOST) failed for hwnd=%s: %s", hwnd, last_error_message(err))


def foreground_window() -> int:
    """Handle of the window that has the keyboard right now (``GetForegroundWindow``); 0 when
    there is none or off Windows."""
    if not IS_WINDOWS or _user32 is None:
        return 0
    try:
        return int(_user32.GetForegroundWindow() or 0)
    except OSError:
        log.exception("GetForegroundWindow raised")
        return 0


def activate_window(hwnd: int) -> bool:
    """Give the keyboard back to ``hwnd`` (``SetForegroundWindow``).

    2026-10-29: used to return the focus to the window the user was working in after one of this
    app's windows had to take it (the region picker when its key hook is unavailable). Windows
    lets a process set the foreground window while it is itself the foreground process, which is
    the case at that moment. A handle that no longer names a window is ignored. True when the
    window was activated.
    """
    if not IS_WINDOWS or _user32 is None or not hwnd:
        return False
    try:
        if not _user32.IsWindow(wintypes.HWND(hwnd)):
            return False
        return bool(_user32.SetForegroundWindow(wintypes.HWND(hwnd)))
    except OSError:
        log.exception("SetForegroundWindow raised for hwnd=%s", hwnd)
        return False


# ---------------------------------------------------------------------------------------------
# Physical-pixel helpers for windows that must line up with what other apps draw (ghost text).
# ---------------------------------------------------------------------------------------------
def dpi_for_point(x: int, y: int) -> int:
    """Effective DPI of the monitor containing the PHYSICAL point ``(x, y)`` (96 if unknown)."""
    if not IS_WINDOWS or _user32 is None:
        return 96
    try:
        monitor = _user32.MonitorFromPoint(wintypes.POINT(int(x), int(y)), _MONITOR_DEFAULTTONEAREST)
        if not monitor:
            return 96
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        dpi_x, dpi_y = wintypes.UINT(), wintypes.UINT()
        if shcore.GetDpiForMonitor(monitor, _MDT_EFFECTIVE_DPI, ctypes.byref(dpi_x), ctypes.byref(dpi_y)) == 0:
            return int(dpi_x.value) or 96
    except Exception:  # noqa: BLE001
        log.debug("dpi_for_point failed", exc_info=True)
    return 96


def set_window_rect_physical(hwnd: int, x: int, y: int, w: int, h: int) -> bool:
    """Move/size ``hwnd`` in PHYSICAL screen pixels, topmost, without activating it."""
    if not IS_WINDOWS or not hwnd or _user32 is None:
        return False
    ctypes.set_last_error(0)
    try:
        ok = _user32.SetWindowPos(
            wintypes.HWND(hwnd), wintypes.HWND(HWND_TOPMOST), int(x), int(y), max(1, int(w)), max(1, int(h)),
            SWP_NOACTIVATE,
        )
    except OSError:
        log.exception("SetWindowPos raised for hwnd=%s", hwnd)
        return False
    if not ok:
        log.debug("SetWindowPos(physical) failed for hwnd=%s: %s", hwnd, last_error_message(ctypes.get_last_error()))
    return bool(ok)
