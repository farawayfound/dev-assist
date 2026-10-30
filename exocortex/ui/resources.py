"""A tiny, dependency-free sampler of this machine's live resource use, for the overlay's
performance Monitor (the "View -> Monitor" section, 2026-10-07).

Three numbers, each read without adding a dependency and each failing soft (``None``) when it
cannot be read, exactly like :mod:`exocortex.gpu`:

* **CPU** -- the whole-machine busy percentage between two :func:`ResourceSampler.sample` calls,
  from ``kernel32!GetSystemTimes`` (idle / kernel / user tick totals). The first sample has no
  previous reading to difference against, so it returns ``None`` for CPU; every later one is the
  average over the interval since the previous call.
* **RAM** -- this process's working set (resident) in MiB, from ``psapi!GetProcessMemoryInfo``.
  This is the whole app, which on Windows includes the child ``llama-server`` only if it shares
  the process (it does not): it is the Python/Qt side, a useful "are we leaking" readout.
* **VRAM** -- used / total MiB and the source, straight from :mod:`exocortex.gpu`
  (``vram_total_mb`` / ``vram_free_mb`` / ``vram_source``): NVML on NVIDIA, the PDH performance
  counters otherwise, and nothing on a machine with neither.

Everything here is Windows-only in practice; off Windows (and under the offscreen test platform)
every field is ``None`` and nothing raises. The sampler keeps only the previous CPU tick totals
as state, so it is cheap to call on a slow timer (the Monitor samples about once a second).
"""
from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

_IS_WIN = sys.platform == "win32"


@dataclass(frozen=True)
class ResourceSample:
    """One reading. Every field is ``None`` when it could not be read on this machine.

    A utilisation field that is ``None`` means "no such device / counter here" (the Monitor hides
    that row); ``0.0`` means the device exists and is idle (the Monitor shows it)."""
    cpu_percent: float | None = None       # whole-machine CPU busy %, averaged over the interval
    proc_ram_mb: int | None = None         # this process's working set (resident), MiB
    ram_used_mb: int | None = None         # system memory in use, MiB
    ram_total_mb: int | None = None        # system memory total, MiB
    vram_used_mb: int | None = None        # dedicated-GPU memory in use, MiB
    vram_total_mb: int | None = None       # dedicated-GPU memory total, MiB
    vram_source: str = "none"              # "nvml" | "pdh" | "none" (see exocortex.gpu.vram_source)
    gpu_percent: float | None = None       # dedicated-GPU busy % (PDH GPU Engine, max over engines)
    igpu_percent: float | None = None      # integrated-GPU busy %
    npu_percent: float | None = None       # NPU / AI-accelerator busy % (PDH NPU Engine)


class _FileTime(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


def _ft_to_int(ft: _FileTime) -> int:
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


# ---- PDH engine-utilisation counters (GPU Engine / NPU Engine) ---------------------------------
_PDH_FMT_DOUBLE = 0x00000200
_PDH_MORE_DATA = 0x800007D2
_PDH_CSTATUS_OK = frozenset({0x00000000, 0x00000001})


class _PdhFmtDouble(ctypes.Structure):
    _fields_ = [("CStatus", ctypes.c_ulong), ("doubleValue", ctypes.c_double)]


class _PdhItemDouble(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _PdhFmtDouble)]


class _PdhEngineUtil:
    """Reads a Windows ``\\<set> Engine(*)\\Utilization Percentage`` PDH counter (GPU Engine or NPU
    Engine). Opened lazily, primed with two throwaway collections (the counter reports 0 until the
    second), and disabled forever on any failure. :meth:`read` returns ``[(instance_name, value)]``
    for every valid item, so the caller can group by adapter LUID; ``None`` when the counter set
    does not exist on this OS (e.g. no NPU, or a Windows build without the counter).

    The utilisation of an adapter is taken as the MAX over its engine instances (3D, Compute, Copy,
    Video, ...), which tracks Task Manager's single GPU/NPU figure far better than summing them."""

    def __init__(self, counter_path: str) -> None:
        self._path = counter_path
        self._ok: bool | None = None      # None = not yet opened, False = disabled, True = open
        self._pdh: Any = None
        self._query: Any = None
        self._counter: Any = None

    def _open(self) -> bool:
        self._ok = False
        if not _IS_WIN:
            return False
        try:
            pdh = ctypes.WinDLL("pdh")
            pdh.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            pdh.PdhOpenQueryW.restype = ctypes.c_long
            pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p,
                                                  ctypes.POINTER(ctypes.c_void_p)]
            pdh.PdhAddEnglishCounterW.restype = ctypes.c_long
            pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            pdh.PdhCollectQueryData.restype = ctypes.c_long
            pdh.PdhGetFormattedCounterArrayW.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                                         ctypes.POINTER(ctypes.c_ulong),
                                                         ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
            pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_long
            pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
            pdh.PdhCloseQuery.restype = ctypes.c_long
            query = ctypes.c_void_p()
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
                return False
            counter = ctypes.c_void_p()
            if pdh.PdhAddEnglishCounterW(query, self._path, 0, ctypes.byref(counter)) != 0:
                pdh.PdhCloseQuery(query)
                return False
            pdh.PdhCollectQueryData(query)
            pdh.PdhCollectQueryData(query)
            self._pdh, self._query, self._counter, self._ok = pdh, query, counter, True
            return True
        except Exception:  # noqa: BLE001 - no such counter on this OS / locale: stays off
            log.debug("resources: PDH counter %r unavailable", self._path, exc_info=True)
            return False

    def read(self) -> list[tuple[str, float]] | None:
        if self._ok is False:
            return None
        if self._ok is None and not self._open():
            return None
        try:
            pdh = self._pdh
            if (pdh.PdhCollectQueryData(self._query) & 0xFFFFFFFF) != 0:
                self._ok = False
                return None
            buf_size, item_count = ctypes.c_ulong(0), ctypes.c_ulong(0)
            rc = pdh.PdhGetFormattedCounterArrayW(self._counter, _PDH_FMT_DOUBLE,
                                                  ctypes.byref(buf_size), ctypes.byref(item_count), None)
            if (rc & 0xFFFFFFFF) not in (0, _PDH_MORE_DATA) or buf_size.value == 0:
                return []
            buf = (ctypes.c_char * buf_size.value)()
            rc = pdh.PdhGetFormattedCounterArrayW(self._counter, _PDH_FMT_DOUBLE,
                                                  ctypes.byref(buf_size), ctypes.byref(item_count), buf)
            if (rc & 0xFFFFFFFF) != 0:
                return []
            items = ctypes.cast(buf, ctypes.POINTER(_PdhItemDouble * item_count.value)).contents
            out: list[tuple[str, float]] = []
            for item in items:
                if item.FmtValue.CStatus in _PDH_CSTATUS_OK and item.szName:
                    out.append((item.szName, float(item.FmtValue.doubleValue)))
            return out
        except Exception:  # noqa: BLE001 - never raise into the sampler over a counter hiccup
            log.debug("resources: PDH read of %r failed; disabling", self._path, exc_info=True)
            self._ok = False
            return None


class ResourceSampler:
    """Stateful sampler: hold the previous CPU tick totals so each :meth:`sample` can report the
    busy percentage over the interval since the last call. All reads fail soft."""

    def __init__(self) -> None:
        self._prev_idle: int | None = None
        self._prev_busy: int | None = None
        self._k32: Any = None
        self._psapi: Any = None
        # PDH engine-utilisation counters (opened lazily on first use, disabled if absent).
        self._gpu_engine = _PdhEngineUtil("\\GPU Engine(*)\\Utilization Percentage")
        self._npu_engine = _PdhEngineUtil("\\NPU Engine(*)\\Utilization Percentage")
        # Adapter LUIDs (from exocortex.gpu, read once): which GPU Engine instances are the
        # dedicated GPU vs the integrated one. "" when that adapter is absent.
        self._dgpu_luid = ""
        self._igpu_luid = ""
        self._identify_adapters()
        if _IS_WIN:
            try:
                self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                self._k32.GetSystemTimes.restype = wintypes.BOOL
                self._k32.GetSystemTimes.argtypes = [
                    ctypes.POINTER(_FileTime), ctypes.POINTER(_FileTime), ctypes.POINTER(_FileTime)
                ]
                # GetCurrentProcess returns a HANDLE (a pseudo-handle, -1); without the restype
                # ctypes hands back a truncated Python int and GetProcessMemoryInfo then fails with
                # ERROR_INVALID_HANDLE, so the RAM readout silently stays empty on 64-bit Python.
                self._k32.GetCurrentProcess.restype = wintypes.HANDLE
                self._k32.GetCurrentProcess.argtypes = []
                self._k32.GlobalMemoryStatusEx.restype = wintypes.BOOL
                self._k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
            except Exception:  # pragma: no cover - defensive; a 32-bit or stripped kernel32
                log.debug("GetSystemTimes unavailable", exc_info=True)
                self._k32 = None
            try:
                self._psapi = ctypes.WinDLL("psapi", use_last_error=True)
                self._psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
                self._psapi.GetProcessMemoryInfo.argtypes = [
                    wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wintypes.DWORD
                ]
            except Exception:  # pragma: no cover
                log.debug("psapi unavailable", exc_info=True)
                self._psapi = None

    # ---- CPU -----------------------------------------------------------------------------
    def _cpu_percent(self) -> float | None:
        if self._k32 is None:
            return None
        idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
        try:
            if not self._k32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                return None
        except Exception:  # pragma: no cover - defensive
            return None
        idle_t = _ft_to_int(idle)
        # kernel time INCLUDES idle time; busy = (kernel - idle) + user.
        busy_t = (_ft_to_int(kernel) - idle_t) + _ft_to_int(user)
        prev_idle, prev_busy = self._prev_idle, self._prev_busy
        self._prev_idle, self._prev_busy = idle_t, busy_t
        if prev_idle is None or prev_busy is None:
            return None                         # first reading: no interval to difference against
        d_idle = idle_t - prev_idle
        d_busy = busy_t - prev_busy
        total = d_idle + d_busy
        if total <= 0:
            return None
        pct = 100.0 * d_busy / total
        return max(0.0, min(100.0, pct))

    # ---- RAM -----------------------------------------------------------------------------
    def _proc_ram_mb(self) -> int | None:
        if self._psapi is None or self._k32 is None:
            return None
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        try:
            handle = self._k32.GetCurrentProcess()
            ok = self._psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
            )
        except Exception:  # pragma: no cover - defensive
            return None
        if not ok:
            return None
        return int(counters.WorkingSetSize // (1024 * 1024))

    def _system_ram(self) -> tuple[int | None, int | None]:
        """(used_mb, total_mb) of physical system memory, from ``GlobalMemoryStatusEx``."""
        if self._k32 is None:
            return None, None
        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        try:
            if not self._k32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return None, None
        except Exception:  # pragma: no cover - defensive
            return None, None
        total = int(stat.ullTotalPhys // (1024 * 1024))
        used = int((stat.ullTotalPhys - stat.ullAvailPhys) // (1024 * 1024))
        return used, total

    # ---- GPU / iGPU / NPU utilisation ----------------------------------------------------
    def _engine_util(self) -> tuple[float | None, float | None, float | None]:
        """(dedicated-GPU %, integrated-GPU %, NPU %) from the PDH engine counters. Each is the
        MAX utilisation over that device's engine instances, or ``None`` when the device / counter
        is absent. GPU instances are split by adapter LUID; with no LUIDs known the whole GPU-Engine
        reading is attributed to the dedicated GPU."""
        gpu_items = self._gpu_engine.read()
        dgpu = igpu = None
        if gpu_items is not None:
            dgpu_vals: list[float] = []
            igpu_vals: list[float] = []
            unsplit: list[float] = []
            for name, val in gpu_items:
                low = name.lower()
                if self._dgpu_luid and f"luid_{self._dgpu_luid}" in low:
                    dgpu_vals.append(val)
                elif self._igpu_luid and f"luid_{self._igpu_luid}" in low:
                    igpu_vals.append(val)
                else:
                    unsplit.append(val)
            if self._dgpu_luid or self._igpu_luid:
                if self._dgpu_luid:
                    dgpu = max(dgpu_vals) if dgpu_vals else 0.0
                if self._igpu_luid:
                    igpu = max(igpu_vals) if igpu_vals else 0.0
            elif unsplit:
                # No adapter identity: attribute the whole GPU-Engine reading to "the GPU".
                dgpu = max(unsplit)
        npu_items = self._npu_engine.read()
        npu = None
        if npu_items is not None:
            npu = max((v for _n, v in npu_items), default=0.0)
        return dgpu, igpu, npu

    # ---- VRAM ----------------------------------------------------------------------------
    @staticmethod
    def _vram() -> tuple[int | None, int | None, str]:
        try:
            from exocortex import gpu
        except Exception:  # pragma: no cover - gpu module optional
            return None, None, "none"
        try:
            total = gpu.vram_total_mb()
            free = gpu.vram_free_mb()
            source = gpu.vram_source()
        except Exception:  # pragma: no cover - every gpu reader already fails soft, belt and braces
            return None, None, "none"
        used = None
        if total is not None and free is not None:
            used = max(0, int(total) - int(free))
        return used, (int(total) if total is not None else None), str(source or "none")

    def reset(self) -> None:
        """Forget the previous CPU tick totals, so the next :meth:`sample` reports ``None`` for CPU
        and starts a fresh interval. Called when the Monitor is hidden, so the first reading after
        it is shown again is not an average over the whole hidden period."""
        self._prev_idle = None
        self._prev_busy = None

    # ---- adapter identity ----------------------------------------------------------------
    def _identify_adapters(self) -> None:
        """Find the dedicated- and integrated-GPU LUIDs (from :mod:`exocortex.gpu`, read only), so
        GPU Engine utilisation can be split between them. Fails soft to no split."""
        try:
            from exocortex import gpu
            gpus = gpu.detect_gpus()
        except Exception:  # pragma: no cover - gpu module optional / probe failure
            return
        for g in gpus:
            luid = (getattr(g, "luid", "") or "").lower()
            if not luid:
                continue
            if getattr(g, "integrated", False):
                if not self._igpu_luid:
                    self._igpu_luid = luid
            elif not self._dgpu_luid:
                self._dgpu_luid = luid

    def sample(self) -> ResourceSample:
        """One reading now. Never raises; unreadable fields come back ``None``."""
        cpu = proc_ram = None
        ram_used = ram_total = None
        vram_used = vram_total = None
        source = "none"
        gpu = igpu = npu = None
        try:
            cpu = self._cpu_percent()
        except Exception:  # pragma: no cover - defensive, the readers already guard themselves
            log.debug("cpu sample failed", exc_info=True)
        try:
            proc_ram = self._proc_ram_mb()
        except Exception:  # pragma: no cover
            log.debug("proc ram sample failed", exc_info=True)
        try:
            ram_used, ram_total = self._system_ram()
        except Exception:  # pragma: no cover
            log.debug("system ram sample failed", exc_info=True)
        try:
            vram_used, vram_total, source = self._vram()
        except Exception:  # pragma: no cover
            log.debug("vram sample failed", exc_info=True)
        try:
            gpu, igpu, npu = self._engine_util()
        except Exception:  # pragma: no cover
            log.debug("engine utilisation sample failed", exc_info=True)
        return ResourceSample(
            cpu_percent=cpu, proc_ram_mb=proc_ram, ram_used_mb=ram_used, ram_total_mb=ram_total,
            vram_used_mb=vram_used, vram_total_mb=vram_total, vram_source=source,
            gpu_percent=gpu, igpu_percent=igpu, npu_percent=npu,
        )
