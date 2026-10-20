"""Foreground window + monitor geometry, ctypes only (no pywin32).

Used by :mod:`exocortex.screen.capture` to resolve what "the active window" and "the
monitors" mean for screen reading. Every Win32 helper here is guarded by :data:`IS_WINDOWS`
and never raises: a bad handle, a missing DLL, or a failing API call is logged and reported
through the return value (``None``), exactly like :mod:`exocortex.ui.win32`.

``mss`` is imported lazily inside :func:`monitor_rects` / :func:`virtual_screen_rect` so this
module has no import-time side effects and stays importable (and its ctypes helpers testable)
on any platform.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

#: True only on Windows; every Win32-touching function checks this first.
IS_WINDOWS = sys.platform == "win32"

# Desktop/taskbar window classes that are never a meaningful "foreground window" for
# screen reading, even though GetForegroundWindow() can legitimately return them.
_IGNORED_CLASSES = frozenset({"Progman", "WorkerW", "Shell_TrayWnd"})

# DWMWA_EXTENDED_FRAME_BOUNDS: the visible window bounds (excludes the invisible resize
# border GetWindowRect includes on Windows 10/11).
_DWMWA_EXTENDED_FRAME_BOUNDS = 9
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

#: EnumWindows callback type (Windows only).
_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM) if IS_WINDOWS else None


@dataclass(frozen=True)
class WindowInfo:
    """The current foreground window, as far as screen reading cares about it."""

    hwnd: int
    title: str
    process_name: str
    rect: tuple[int, int, int, int]  # x, y, w, h in screen pixels


def _load_dlls() -> tuple[Any, Any, Any]:
    """Load user32/dwmapi/kernel32 with argtypes set. ``(None, None, None)`` off Windows."""
    if not IS_WINDOWS:
        return None, None, None
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL

    dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long  # HRESULT

    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    return user32, dwmapi, kernel32


_user32, _dwmapi, _kernel32 = _load_dlls()


# --------------------------------------------------------------------------------- #
# Raw Win32 access, factored out so tests can monkeypatch a single seam.
# --------------------------------------------------------------------------------- #


def _foreground_raw() -> tuple[int, int, str, str] | None:
    """``(hwnd, pid, class_name, title)`` for the current foreground window, or ``None``.

    The one function :func:`foreground_window` depends on for "what window is this"; tests
    monkeypatch it directly to simulate our own process, the desktop, etc. without a real
    window.
    """
    if not IS_WINDOWS or _user32 is None:
        return None
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return None
    pid = wintypes.DWORD(0)
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    cls_buf = ctypes.create_unicode_buffer(256)
    _user32.GetClassNameW(hwnd, cls_buf, len(cls_buf))
    title_buf = ctypes.create_unicode_buffer(1024)
    _user32.GetWindowTextW(hwnd, title_buf, len(title_buf))
    return int(hwnd), int(pid.value), cls_buf.value, title_buf.value


def _is_minimized(hwnd: int) -> bool:
    if not IS_WINDOWS or _user32 is None:
        return False
    try:
        return bool(_user32.IsIconic(wintypes.HWND(hwnd)))
    except OSError:
        log.exception("IsIconic raised for hwnd=%s", hwnd)
        return False


def _window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """Visible window bounds via DWMWA_EXTENDED_FRAME_BOUNDS, falling back to GetWindowRect."""
    if not IS_WINDOWS or _dwmapi is None or _user32 is None:
        return None
    rect = wintypes.RECT()
    try:
        hr = _dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd), _DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)
        )
    except OSError:
        log.exception("DwmGetWindowAttribute raised for hwnd=%s", hwnd)
        hr = -1
    if hr != 0:
        try:
            if not _user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
                return None
        except OSError:
            log.exception("GetWindowRect raised for hwnd=%s", hwnd)
            return None
    x, y = int(rect.left), int(rect.top)
    w, h = int(rect.right - rect.left), int(rect.bottom - rect.top)
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _process_name(pid: int) -> str:
    """Base file name (e.g. ``"chrome.exe"``) of the process owning ``pid``, or ``""``."""
    if not IS_WINDOWS or _kernel32 is None or not pid:
        return ""
    try:
        handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    except OSError:
        log.exception("OpenProcess raised for pid=%s", pid)
        return ""
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(260)
        size = wintypes.DWORD(len(buf))
        ok = _kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        if not ok:
            return ""
        return os.path.basename(buf.value)
    except OSError:
        log.exception("QueryFullProcessImageNameW raised for pid=%s", pid)
        return ""
    finally:
        _kernel32.CloseHandle(handle)


def foreground_window() -> WindowInfo | None:
    """The current foreground window, or ``None`` when there is nothing useful to read.

    Returns ``None``:
    - on any non-Windows platform,
    - for a window that belongs to our own process (`GetWindowThreadProcessId` -> `os.getpid()`),
    - for the desktop/taskbar (class ``Progman``/``WorkerW``/``Shell_TrayWnd``),
    - for a minimized window (`IsIconic`),
    - for a window with an empty/invalid rect.
    """
    raw = _foreground_raw()
    if raw is None:
        return None
    hwnd, pid, class_name, title = raw
    if pid == os.getpid():
        return None
    if class_name in _IGNORED_CLASSES:
        return None
    if _is_minimized(hwnd):
        return None
    rect = _window_rect(hwnd)
    if rect is None:
        return None
    return WindowInfo(hwnd=hwnd, title=title, process_name=_process_name(pid), rect=rect)


# --------------------------------------------------------------------------------- #
# Monitor geometry (via mss)
# --------------------------------------------------------------------------------- #


def _own_windows_raw() -> list[int]:
    """Handles of this process's visible top-level windows (the overlay, the ghost text, an open
    menu, tooltip or dialog), via ``EnumWindows``; ``[]`` off Windows or when the call fails.

    The one seam :func:`own_window_rects` depends on, so tests can monkeypatch it."""
    if not IS_WINDOWS or _user32 is None or _WNDENUMPROC is None:
        return []
    pid = os.getpid()
    found: list[int] = []

    def _visit(hwnd: Any, _lparam: Any) -> bool:
        try:
            if hwnd and _user32.IsWindowVisible(hwnd):
                owner = wintypes.DWORD(0)
                _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
                if owner.value == pid:
                    found.append(int(hwnd))
        except Exception:  # noqa: BLE001 - one odd window must not end the enumeration
            pass
        return True

    try:
        _user32.EnumWindows(_WNDENUMPROC(_visit), 0)
    except OSError:
        log.exception("EnumWindows raised")
        return []
    return found


def own_window_rects() -> list[tuple[int, int, int, int]]:
    """Screen rects ``(x, y, w, h)`` of this process's visible, non-minimized top-level windows,
    in the same coordinates as :attr:`WindowInfo.rect` (and so as a captured frame's ``rect``).

    The overlay is an ordinary window that screen capture sees, so the screen reader blanks
    these out of every frame before OCR (``[screen].exclude_own_windows``): otherwise the
    co-pilot would read its own answers and suggested questions back as screen content."""
    rects: list[tuple[int, int, int, int]] = []
    for hwnd in _own_windows_raw():
        if _is_minimized(hwnd):
            continue
        rect = _window_rect(hwnd)
        if rect is not None:
            rects.append(rect)
    return rects


def monitor_dict_to_rect(monitor: dict[str, Any]) -> tuple[int, int, int, int]:
    """Convert one raw ``mss`` monitor dict to a plain ``(x, y, w, h)`` tuple.

    Not underscore-prefixed: :mod:`exocortex.screen.capture` reuses it (and
    :func:`order_primary_first` below) so ``ScreenGrabber.monitors()`` orders monitors
    identically to the free functions here, without opening a second ``mss.MSS``.
    """
    return (int(monitor["left"]), int(monitor["top"]), int(monitor["width"]), int(monitor["height"]))


def order_primary_first(monitors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reorder ``monitors`` (mss dicts, no union entry) so the primary one comes first.

    Falls back to the first entry when none is marked ``is_primary`` (defensive; mss always
    reports it on Windows, but nothing here should crash if it didn't).
    """
    if not monitors:
        return []
    primary_idx = next((i for i, m in enumerate(monitors) if m.get("is_primary")), 0)
    primary = monitors[primary_idx]
    rest = [m for i, m in enumerate(monitors) if i != primary_idx]
    return [primary, *rest]


def _mss_monitors() -> list[dict[str, Any]]:
    import mss

    with mss.MSS() as sct:
        return list(sct.monitors)


def monitor_rects() -> list[tuple[int, int, int, int]]:
    """Every physical monitor as ``(x, y, w, h)``, primary first.

    ``mss``'s ``monitors[0]`` is the virtual-screen union of all displays, not a real
    monitor, so it is excluded here (see :func:`virtual_screen_rect`).
    """
    mons = _mss_monitors()
    real = mons[1:] if len(mons) > 1 else mons
    return [monitor_dict_to_rect(m) for m in order_primary_first(real)]


def virtual_screen_rect() -> tuple[int, int, int, int]:
    """The bounding rect of the whole virtual screen (union of every monitor)."""
    mons = _mss_monitors()
    return monitor_dict_to_rect(mons[0])
