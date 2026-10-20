"""One session at a time: find and force-close every earlier ExoCortex (``--replace``).

``run.bat`` and ``run.vbs`` start the app with ``--replace``. Before the new instance loads anything
heavy, every other running ExoCortex -- any ``python -m exocortex`` interpreter, whatever started
it -- is terminated together with the helpers it spawned (its llama-server first of all), and the
new process waits until they have exited. The wait matters: the old llama-server holds ~17 GB of
the 5090's 24 GB, and a second copy of the model loading next to it spills into system RAM and
stalls the whole desktop, since that GPU also drives the display (the 2026-09-24 incident).

What is closed -- and nothing else:

* **roots**: Python interpreters (``python*.exe`` / ``pythonw*.exe`` / the ``py.exe`` launcher)
  whose command line runs the ``exocortex`` package (``-m exocortex``, or its ``__main__.py``);
* their **descendants**, only of the kinds ExoCortex itself starts: llama-server, Python (the
  venv's launcher stub runs the real interpreter as its child; the doctor's probe) and the console
  host -- a program the old session happened to open is never killed with it;
* a **stray model server**: a llama-server whose ``--log-file`` lies in the sessions folder (only
  ExoCortex puts one there), for the rare case its ExoCortex parent is already gone.

llama-swap's servers, LM Studio and every other process are left alone.

**Newest wins**: only processes created *before* this one are closed, and never this process's
own ancestors (the venv's ``pythonw.exe`` stub that launched us runs ``-m exocortex`` too), so two
launches in quick succession cannot take each other down.

"Force close" is literal -- ``TerminateProcess``, nothing is saved. The closed session keeps what
its recorder last autosaved (every ``[paths].autosave_interval_s``, 30 s by default); its window
position is not written back.

:func:`select_targets` is pure and unit-tested on synthetic process tables; :func:`snapshot` and
:func:`terminate` are the ctypes Win32 layer (Toolhelp32 for the process list,
``NtQueryInformationProcess(ProcessCommandLineInformation)`` for command lines, no psutil). Off
Windows :func:`replace_previous` does nothing.

:func:`free_llama_swap` is the model-server counterpart of the process takeover: this laptop also
runs a llama-swap service (a fleet node) that can hold its own copy of the 27B; two copies
oversubscribe the display GPU and page the desktop into the ground (the 2026-09-24/25 mid-meeting
freezes). With the owner's sign-off, a ``--replace`` launch asks it to unload before our model
loads, and a normal quit asks again on the way out, so a session starts with the GPU free and
leaves it free. Everything is best-effort over its public HTTP endpoints; a machine without
llama-swap pays one failed local GET.
"""
from __future__ import annotations

import json
import logging
import ntpath
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: Python interpreters and the ``py`` launcher, by executable file name.
_INTERPRETER_RE = re.compile(r"^(?:python|pythonw|py|pyw)(?:\d+(?:\.\d+)*)?(?:\.exe)?$", re.IGNORECASE)
_LLAMA_SERVER_RE = re.compile(r"^llama-server(?:\.exe)?$", re.IGNORECASE)
#: Console hosts: a console-mode ExoCortex (``python.exe``) may own one.
_CONSOLE_HOST_RE = re.compile(r"^(?:conhost|openconsole)(?:\.exe)?$", re.IGNORECASE)
#: ``py.exe`` launcher version selectors: ``-3.12``, ``-3-64``, ``-V:3.12``.
_PY_LAUNCHER_OPT_RE = re.compile(r"^-(?:\d+(?:\.\d+)*(?:-(?:32|64))?|V:\S+)$")
_MODULES = frozenset({"exocortex", "exocortex.__main__"})


@dataclass(frozen=True)
class ProcInfo:
    """One row of the process table."""

    pid: int
    ppid: int
    name: str                                  # executable file name, e.g. "pythonw.exe"
    created: int | None = None                 # creation time (FILETIME, 100 ns ticks); None if unreadable
    argv: tuple[str, ...] | None = None        # parsed command line; None when not read or unreadable


@dataclass
class ReplaceResult:
    """What :func:`replace_previous` did."""

    closed: list[ProcInfo] = field(default_factory=list)
    failed: list[tuple[ProcInfo, str]] = field(default_factory=list)

    @property
    def sessions_closed(self) -> int:
        """Top-level ExoCortex processes closed: a venv stub and the interpreter it ran count once."""
        pids = {p.pid for p in self.closed if is_exocortex_argv(p.argv)}
        return sum(1 for p in self.closed if p.pid in pids and p.ppid not in pids)

    @property
    def notice(self) -> str:
        """One line for the user ("" when there was nothing to close)."""
        if self.failed:
            p, why = self.failed[0]
            return f"Could not close an earlier ExoCortex ({p.name}, pid {p.pid}): {why}"
        n = self.sessions_closed
        if n == 1:
            return "Closed the previous ExoCortex session"
        if n > 1:
            return f"Closed {n} previous ExoCortex sessions"
        if self.closed:
            return "Closed a stray ExoCortex model server"
        return ""


# ---------------------------------------------------------------------------- shared model server ---
def _default_http_get(url: str, timeout: float) -> tuple[int, str]:
    """(status, body) with every transport failure mapped to (0, "") -- callers branch on status."""
    import httpx

    try:
        r = httpx.get(url, timeout=timeout)
        return r.status_code, r.text
    except Exception:  # noqa: BLE001 - no llama-swap here is the normal case on other machines
        log.debug("swap: GET %s failed", url, exc_info=True)
        return 0, ""


def _swap_running_count(get: "Callable[[str, float], tuple[int, str]]", base: str) -> int | None:
    """Resident models on the llama-swap at ``base``; None when it cannot be asked or understood.

    The ``http_get`` contract maps transport failures to ``(0, "")``, but "never raises" must not
    depend on an injectable keeping its promises, so a raising one is caught here too."""
    try:
        status, body = get(base + "/running", 5.0)
    except Exception:  # noqa: BLE001
        log.debug("swap: /running raised", exc_info=True)
        return None
    if status != 200:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    rows = data.get("running") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return None
    return len(rows)


def free_llama_swap(
    url: str,
    *,
    timeout_s: float = 45.0,
    http_get: "Callable[[str, float], tuple[int, str]] | None" = None,
    clock: "Callable[[], float]" = time.monotonic,
    sleep: "Callable[[float], None]" = time.sleep,
) -> str:
    """Ask the llama-swap at ``url`` to unload every resident model and wait until they are gone.

    Returns a one-line notice for the toast ("" when there was nothing to do or no server).
    ``GET /unload`` is llama-swap's own public "unload everything" endpoint; polling ``/running``
    afterwards matters because the point is the GPU being FREE before our model starts loading --
    an exited server releases its memory, an "unloading" one has not yet. Never raises: a launch
    or a quit must not fail because the shared server misbehaved. Interferes with whatever that
    server was doing on purpose -- the foreground session owns the GPU (owner's call, 2026-09-25).
    """
    base = (url or "").strip().rstrip("/")
    if not base:
        return ""
    get = http_get or _default_http_get
    count = _swap_running_count(get, base)
    if not count:
        return ""
    log.info("swap: %d resident model(s) at %s; asking it to unload", count, base)
    try:
        get(base + "/unload", timeout_s)   # llama-swap answers when the unload is done; poll anyway
    except Exception:  # noqa: BLE001 - the poll below is the arbiter either way
        log.debug("swap: /unload raised", exc_info=True)
    deadline = clock() + timeout_s
    while clock() < deadline:
        if _swap_running_count(get, base) == 0:
            log.info("swap: shared model server freed")
            return f"Freed the shared model server ({count} model(s))"
        sleep(1.0)
    log.warning("swap: model(s) still resident after %.0f s", timeout_s)
    return "The shared model server did not release its model in time"


# --------------------------------------------------------------------------------- selection ---
def is_exocortex_argv(argv: Sequence[str] | None) -> bool:
    """Whether ``argv`` (a parsed command line) is a Python interpreter running ExoCortex:
    ``python -m exocortex [...]`` -- interpreter options (``-u``, ``-X utf8``, ``-Xutf8``) and
    ``py.exe`` version selectors may come first -- or ``python <path>\\exocortex\\__main__.py``.
    Another module or script is not, ``python -m pytest -m exocortex`` included, and neither is
    ``llama-server -m <model>`` (not an interpreter). The app runs under its real command line
    (nothing rewrites the process's PEB), so the plain interpreter reading is all that is matched."""
    if not argv:
        return False
    if not _INTERPRETER_RE.match(ntpath.basename(argv[0])):
        return False
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            return i + 1 < len(argv) and _is_main_script(argv[i + 1])
        if not arg.startswith("-") or arg == "-":
            return _is_main_script(arg)           # the first positional argument is the script
        if arg.startswith("--"):
            i += 2 if arg == "--check-hash-based-pycs" else 1
            continue
        if _PY_LAUNCHER_OPT_RE.match(arg):
            i += 1
            continue
        # A cluster of one-letter options ("-uB"); -c / -m / -W / -X take a value, either attached
        # ("-mexocortex", "-Xutf8") or as the next argument. -c and -m end the options.
        step = 1
        for k, ch in enumerate(arg[1:], start=1):
            if ch in "cmWX":
                attached = arg[k + 1:]
                value = attached if attached else (argv[i + 1] if i + 1 < len(argv) else "")
                if ch == "m":
                    return value in _MODULES
                if ch == "c":
                    return False
                step = 1 if attached else 2
                break
        i += step
    return False


def _is_main_script(path: str) -> bool:
    """``<...>\\exocortex\\__main__.py``, or the package directory itself (``python exocortex``)."""
    parts = [p for p in re.split(r"[\\/]+", path.strip().casefold()) if p and p != "."]
    if parts and parts[-1] == "exocortex":
        return True
    return parts[-2:] == ["exocortex", "__main__.py"]


def _is_helper(p: ProcInfo) -> bool:
    """A descendant kind ExoCortex itself starts (closed with its root): llama-server, Python,
    the console host."""
    return bool(_LLAMA_SERVER_RE.match(p.name) or _INTERPRETER_RE.match(p.name)
                or _CONSOLE_HOST_RE.match(p.name))


def _log_file_arg(argv: Sequence[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg == "--log-file" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--log-file="):
            return arg.split("=", 1)[1]
    return None


def _inside(path: str, folder: str) -> bool:
    """Whether Windows ``path`` lies strictly inside ``folder`` (case- and separator-insensitive)."""
    p = ntpath.normcase(ntpath.normpath(path))
    f = ntpath.normcase(ntpath.normpath(folder)).rstrip("\\")
    return bool(f) and p.startswith(f + "\\")


def _created_before(p: ProcInfo, me: ProcInfo) -> bool:
    """``p`` started before ``me`` (unknown times do not protect: the table was taken just now)."""
    return p.created is None or me.created is None or p.created < me.created


def _born_after(child: ProcInfo, parent: ProcInfo) -> bool:
    """``parent`` can really be ``child``'s parent: Windows reuses PIDs, and a process whose parent
    exited keeps the old PID as its parent id, so a "parent" created after the child is a stranger."""
    return child.created is None or parent.created is None or child.created >= parent.created


def _ancestor_pids(me: ProcInfo, by_pid: dict[int, ProcInfo]) -> set[int]:
    seen: set[int] = set()
    cur = me
    while True:
        parent = by_pid.get(cur.ppid)
        if parent is None or parent.pid == cur.pid or parent.pid in seen or not _born_after(cur, parent):
            return seen
        seen.add(parent.pid)
        cur = parent


def select_targets(
    procs: Iterable[ProcInfo], self_pid: int, sessions_dir: str | os.PathLike[str] | None = None,
) -> list[ProcInfo]:
    """The processes :func:`replace_previous` closes (see the module docstring): earlier ExoCortex
    interpreters, their llama-server / Python / console-host descendants, and stray llama-servers
    logging into ``sessions_dir``. Never ``self_pid`` or its ancestors, never anything created
    after ``self_pid``. Empty when ``self_pid`` is not in ``procs`` (no ancestors to protect)."""
    table = list(procs)
    by_pid = {p.pid: p for p in table}
    me = by_pid.get(self_pid)
    if me is None:
        log.warning("replace: this process (pid %d) is not in the process list; closing nothing", self_pid)
        return []
    protected = {me.pid} | _ancestor_pids(me, by_pid)
    children: dict[int, list[ProcInfo]] = defaultdict(list)
    for p in table:
        if p.pid != p.ppid:
            children[p.ppid].append(p)

    chosen: dict[int, ProcInfo] = {}
    pending = [p for p in table
               if p.pid not in protected and is_exocortex_argv(p.argv) and _created_before(p, me)]
    for root in pending:
        chosen[root.pid] = root
    while pending:
        parent = pending.pop()
        for child in children.get(parent.pid, ()):
            if (child.pid in chosen or child.pid in protected or not _is_helper(child)
                    or not _born_after(child, parent)):
                continue
            chosen[child.pid] = child
            pending.append(child)

    if sessions_dir is not None:
        folder = os.fspath(sessions_dir)
        for p in table:
            if p.pid in chosen or p.pid in protected or not _LLAMA_SERVER_RE.match(p.name):
                continue
            log_file = _log_file_arg(p.argv or ())
            if log_file and _inside(log_file, folder) and _created_before(p, me):
                chosen[p.pid] = p
    return list(chosen.values())


# ---------------------------------------------------------------------------- orchestration ---
KillFn = Callable[[Sequence[ProcInfo], float], "tuple[list[ProcInfo], list[tuple[ProcInfo, str]]]"]


def replace_previous(
    sessions_dir: str | os.PathLike[str] | None = None,
    *,
    timeout_s: float = 10.0,
    list_processes: Callable[[], list[ProcInfo]] | None = None,
    kill: KillFn | None = None,
    self_pid: int | None = None,
) -> ReplaceResult:
    """Force-close every earlier ExoCortex (and its helpers, and any stray ExoCortex llama-server
    logging into ``sessions_dir``), waiting up to ``timeout_s`` for them to exit. Never raises: a
    failure is logged and the new session starts anyway. ``list_processes`` / ``kill`` /
    ``self_pid`` are injectable for tests (default: :func:`snapshot`, :func:`terminate`,
    ``os.getpid()``)."""
    if list_processes is None:
        if sys.platform != "win32":
            return ReplaceResult()
        list_processes = snapshot
    kill = kill or terminate
    started = time.monotonic()
    try:
        targets = select_targets(list_processes(), os.getpid() if self_pid is None else self_pid,
                                 sessions_dir)
    except Exception:  # noqa: BLE001 - never let the takeover keep the app from starting
        log.exception("replace: could not list processes; starting alongside whatever is running")
        return ReplaceResult()
    if not targets:
        log.info("replace: no other ExoCortex is running")
        return ReplaceResult()
    for p in targets:
        log.info("replace: closing %s pid %d: %s", p.name, p.pid, " ".join(p.argv or ())[:240])
    try:
        closed, failed = kill(targets, timeout_s)
    except Exception as exc:  # noqa: BLE001
        log.exception("replace: closing the earlier session failed")
        return ReplaceResult(failed=[(p, str(exc)) for p in targets])
    result = ReplaceResult(list(closed), list(failed))
    for p, why in result.failed:
        log.warning("replace: could not close %s pid %d: %s", p.name, p.pid, why)
    log.info("replace: closed %d process(es) in %.2f s", len(result.closed), time.monotonic() - started)
    return result


# ------------------------------------------------------------------------------- Win32 layer ---
PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
_TH32CS_SNAPPROCESS = 0x00000002
_PROCESS_COMMAND_LINE_INFORMATION = 60          # PROCESSINFOCLASS, Windows 8.1+
_STATUS_INFO_LENGTH_MISMATCH = 0xC0000004
_WAIT_OBJECT_0 = 0
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87                    # OpenProcess on a PID that no longer exists

_API: Any = None


def _api() -> Any:
    global _API
    if _API is None:
        _API = _Win32()
    return _API


class _Win32:
    """The kernel32 / ntdll / shell32 calls this module needs, bound once with explicit types."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        self._wt = wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        nt = ctypes.WinDLL("ntdll")
        sh = ctypes.WinDLL("shell32", use_last_error=True)

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                        ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]

        class UNICODE_STRING(ctypes.Structure):
            _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT),
                        ("Buffer", ctypes.c_void_p)]

        self._entry_t = PROCESSENTRY32W
        self._ustr_t = UNICODE_STRING
        k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        for fn in (k32.Process32FirstW, k32.Process32NextW):
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
            fn.restype = wintypes.BOOL
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        k32.GetProcessTimes.restype = wintypes.BOOL
        k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.TerminateProcess.restype = wintypes.BOOL
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.WaitForSingleObject.restype = wintypes.DWORD
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        k32.LocalFree.argtypes = [ctypes.c_void_p]
        k32.LocalFree.restype = ctypes.c_void_p
        nt.NtQueryInformationProcess.argtypes = [wintypes.HANDLE, ctypes.c_ulong, ctypes.c_void_p,
                                                 ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]
        nt.NtQueryInformationProcess.restype = ctypes.c_long
        sh.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
        sh.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
        self._k32, self._nt, self._sh = k32, nt, sh
        self._invalid_handle = wintypes.HANDLE(-1).value

    def last_error(self) -> int:
        return int(self._ct.get_last_error())

    def processes(self) -> list[tuple[int, int, str]]:
        """(pid, parent pid, exe name) of every process (Toolhelp32)."""
        ct = self._ct
        snap = self._k32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
        if not snap or snap == self._invalid_handle:
            raise OSError(self.last_error(), "CreateToolhelp32Snapshot failed")
        try:
            entry = self._entry_t()
            entry.dwSize = ct.sizeof(entry)
            rows: list[tuple[int, int, str]] = []
            ok = self._k32.Process32FirstW(snap, ct.byref(entry))
            while ok:
                rows.append((int(entry.th32ProcessID), int(entry.th32ParentProcessID), entry.szExeFile))
                ok = self._k32.Process32NextW(snap, ct.byref(entry))
            return rows
        finally:
            self._k32.CloseHandle(snap)

    def open(self, pid: int, access: int) -> int | None:
        return self._k32.OpenProcess(access, False, pid) or None

    def close(self, handle: int) -> None:
        self._k32.CloseHandle(handle)

    def creation_time(self, handle: int) -> int | None:
        ct, ft = self._ct, self._wt.FILETIME
        created, exited, kernel, user = ft(), ft(), ft(), ft()
        if not self._k32.GetProcessTimes(handle, ct.byref(created), ct.byref(exited), ct.byref(kernel),
                                         ct.byref(user)):
            return None
        return (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)

    def command_line(self, handle: int) -> str | None:
        """The process's command line (needs only PROCESS_QUERY_LIMITED_INFORMATION)."""
        ct = self._ct
        size = ct.c_ulong(0)
        status = self._nt.NtQueryInformationProcess(handle, _PROCESS_COMMAND_LINE_INFORMATION, None, 0,
                                                    ct.byref(size)) & 0xFFFFFFFF
        if status not in (0, _STATUS_INFO_LENGTH_MISMATCH) or size.value < ct.sizeof(self._ustr_t):
            return None
        buf = ct.create_string_buffer(size.value)
        status = self._nt.NtQueryInformationProcess(handle, _PROCESS_COMMAND_LINE_INFORMATION, buf, size.value,
                                                    ct.byref(size)) & 0xFFFFFFFF
        if status != 0:
            return None
        text = self._ustr_t.from_buffer(buf)
        if not text.Buffer or not text.Length:
            return ""
        return ct.wstring_at(text.Buffer, text.Length // 2)

    def split(self, command_line: str) -> list[str]:
        """``CommandLineToArgvW``: the same quoting rules the C runtime (and Python) apply."""
        if not command_line.strip():
            return []           # the API would answer with this process's own path for ""
        ct = self._ct
        argc = ct.c_int(0)
        argv = self._sh.CommandLineToArgvW(command_line, ct.byref(argc))
        if not argv:
            return []
        try:
            return [argv[i] for i in range(argc.value)]
        finally:
            self._k32.LocalFree(ct.cast(argv, ct.c_void_p))

    def terminate(self, handle: int) -> bool:
        return bool(self._k32.TerminateProcess(handle, 1))

    def wait(self, handle: int, timeout_ms: int) -> bool:
        """True once the process has exited (the handle is signalled)."""
        return self._k32.WaitForSingleObject(handle, max(0, int(timeout_ms))) == _WAIT_OBJECT_0


def snapshot() -> list[ProcInfo]:
    """Every process on the machine: pid, parent pid, exe name, creation time where it can be read,
    and the parsed command line of the interpreters and llama-servers (the only kinds the selection
    reads). A process that cannot be opened (another user, a protected process) just has no times
    and no command line -- it can then never be selected. Off Windows: empty."""
    if sys.platform != "win32":
        return []
    api = _api()
    out: list[ProcInfo] = []
    for pid, ppid, name in api.processes():
        created: int | None = None
        argv: tuple[str, ...] | None = None
        handle = api.open(pid, PROCESS_QUERY_LIMITED_INFORMATION) if pid else None
        if handle:
            try:
                created = api.creation_time(handle)
                if _INTERPRETER_RE.match(name) or _LLAMA_SERVER_RE.match(name):
                    line = api.command_line(handle)
                    argv = tuple(api.split(line)) if line else None
            finally:
                api.close(handle)
        out.append(ProcInfo(pid=pid, ppid=ppid, name=name, created=created, argv=argv))
    return out


def _describe_error(code: int) -> str:
    if code == _ERROR_ACCESS_DENIED:
        return "access denied (is it running as administrator?)"
    return f"error {code}"


def terminate(targets: Sequence[ProcInfo], timeout_s: float = 10.0) -> tuple[list[ProcInfo], list[tuple[ProcInfo, str]]]:
    """``TerminateProcess`` every target, then wait -- up to ``timeout_s`` in total -- until each has
    exited (an exited process has released its GPU memory). Returns (closed, failed). A target that
    is already gone counts as closed; one whose PID now belongs to another process (its creation
    time changed) is left alone and reported in neither list."""
    if sys.platform != "win32":
        return [], [(p, "not Windows") for p in targets]
    api = _api()
    closed: list[ProcInfo] = []
    failed: list[tuple[ProcInfo, str]] = []
    opened: list[tuple[ProcInfo, int]] = []
    for p in targets:
        handle = api.open(p.pid, PROCESS_TERMINATE | SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION)
        if not handle:
            code = api.last_error()
            if code == _ERROR_INVALID_PARAMETER:
                closed.append(p)            # already exited (e.g. its job object took it with its parent)
            else:
                failed.append((p, _describe_error(code)))
            continue
        if p.created is not None and api.creation_time(handle) != p.created:
            api.close(handle)               # the PID was reused meanwhile: not ours to touch
            continue
        if not api.terminate(handle):
            # An exiting process refuses with ERROR_ACCESS_DENIED too; the wait below decides.
            log.debug("replace: TerminateProcess(%d) failed: error %d", p.pid, api.last_error())
        opened.append((p, handle))
    deadline = time.monotonic() + max(0.0, timeout_s)
    for p, handle in opened:
        try:
            if api.wait(handle, int((deadline - time.monotonic()) * 1000)):
                closed.append(p)
            else:
                failed.append((p, "still running"))
        finally:
            api.close(handle)
    return closed, failed
