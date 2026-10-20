"""GPU discovery (vendor-neutral) and the GPU memory watch behind the "gpu-guard" thread.

Two halves:

* **Discovery** -- :func:`detect_gpus` enumerates the machine's display adapters through DXGI
  (``dxgi.dll``, ctypes, no dependency): vendor, name, dedicated / shared memory and -- through a
  throw-away D3D12 device -- whether the adapter is UMA (an integrated GPU sharing system RAM: the
  Radeon 8060S / 780M, Intel Arc iGPUs) or discrete. :func:`primary_gpu` picks the one that runs
  the model (a discrete GPU with the most memory, else the biggest integrated one) and
  :func:`gpu_vendor` names its vendor: this is what routes llama.cpp (CUDA / ROCm / Vulkan / CPU,
  ``exocortex.llm.server``), Whisper (CUDA or CPU threads), OCR (OpenVINO / DirectML) and the
  run script's optional pip extras (:func:`extras_for_vendor`). ``python -m exocortex.gpu`` prints
  :func:`describe` for scripts; ``--vendor`` prints one word, ``--json`` the adapters as JSON,
  ``--extras`` the requirement files run.bat should additionally install (see :func:`main`).
* **Memory watch** -- :func:`vram_free_mb` and :class:`VramGuard` (below).

Verified on 2026-09-25 on a Ryzen AI MAX+ 395 (Radeon 8060S, 32 GiB UMA carve-out): DXGI reports
vendor 0x1002, DedicatedVideoMemory 32590 MiB, SharedSystemMemory 16204 MiB, D3D12 ``UMA = 1``;
``llama-server --list-devices`` (Vulkan) sees the same adapter with 48971 MiB. On the RTX 5080 /
5090 laptops the discrete NVIDIA card is the primary and everything below behaves as before.

2026-09-26 (intel-integration): :func:`driver_version` reads the display driver string for an
adapter out of the registry (``winreg``, stdlib, injectable via ``reg_reader`` for tests) --
verified live on this AMD box: ``"32.0.31032.1003"``, matching Win32_VideoController. It is
Intel-only in :func:`describe`'s lines and the ``--json`` output (gated on ``vendor == "intel"``)
so NVIDIA and AMD output stays byte-identical to before this branch; :func:`intel_arc_notes` turns
that driver string plus a discrete Intel ``GpuInfo`` into advisory lines for Arc Alchemist/Battlemage/Pro
cards (Vulkan-over-SYCL, the Xe2 cooperative-matrix driver threshold, Alchemist's
``GGML_VK_DISABLE_F16`` folklore): no Intel Arc GPU exists on either dev box this branch was built
on, so this path is unit-tested only (`tests/test_gpu_detect.py`) -- see its docstring for sources
and caveats.

GPU memory watch: NVML readings and the pure alert logic behind the "gpu-guard" thread.

Why this exists: the RTX 5090 that runs the model also drives the display. When a SECOND large
model lands on it (this machine runs a separate llama-swap service that other machines send work
to), the card oversubscribes, WDDM pages GPU memory into system RAM over PCIe, and everything on
the desktop -- the compositor included -- stalls for seconds at a time. It happened mid-meeting on
2026-09-24 and again on 2026-09-25; from inside the app it looked like "the model got slow", and
from the desk it looked like the laptop dying. The guard makes that state *visible the moment it
starts* (status event + toast) instead of discoverable in logs after the meeting is lost.

Split for testability:

* :func:`vram_free_mb` -- NVML first (``nvml.dll`` ships with the NVIDIA driver and is loaded
  lazily via ctypes, no pynvml dependency; any failure -- other OS, no NVIDIA driver, the driver
  resetting mid-session -- parks it at ``None`` forever after, exactly as before). When NVML has
  nothing to say and the primary adapter is not NVIDIA, the reading comes from Windows' own
  ``GPU Adapter Memory`` performance counters (PDH, ``pdh.dll``, :func:`_pdh_dedicated_used_mb`):
  free = the adapter's dedicated memory (DXGI) - its "Dedicated Usage" counter, matched by LUID,
  clamped at 0 -- vendor-neutral, adapter-wide (what Task Manager shows), and on a UMA iGPU
  "dedicated" is the firmware carve-out. Same ladder, exposed for callers that want the raw
  numbers: :func:`vram_total_mb` (NVML total, else the primary's DXGI dedicated memory) and
  :func:`vram_source` (``"nvml"`` | ``"pdh"`` | ``"none"``, whichever :func:`vram_free_mb` would
  use right now). The guard treats ``None`` as "no reading", never as pressure.
* :class:`VramGuard` -- pure decision logic (injectable clock and reading function, unit-tested):
  a dip must persist for ``consecutive`` samples to alert (a game launching or a browser tab
  spiking briefly must not), alerts repeat at most every ``cooldown_s`` while the pressure lasts,
  and one recovery message follows when free memory climbs clearly back out (threshold + 512 MiB,
  so a reading jittering around the threshold cannot flap alert/recovered/alert).

The Engine owns the thread (``Engine._gpu_guard_loop``): a sample every ~5 s costs one NVML call
(microseconds). Alerts are published as ``ComponentStatusEvent("app", "error", ...)`` -- the
overlay toasts those without touching the readiness dot -- and recoveries as ``"info"``.
"""
from __future__ import annotations

import ctypes
import logging
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

#: Extra MiB above the threshold required before a recovery is announced (anti-flap hysteresis).
RECOVERY_MARGIN_MB = 512

# =============================================================================================
# GPU discovery (DXGI + D3D12, ctypes)
# =============================================================================================

#: PCI vendor ids -> vendor names used throughout the app (config comments, doctor, run.bat).
VENDOR_IDS: dict[int, str] = {0x10DE: "nvidia", 0x1002: "amd", 0x1022: "amd", 0x8086: "intel",
                              0x5143: "qualcomm", 0x1414: "microsoft"}

#: Vendors whose "GPU" is a software renderer (Microsoft Basic Render Driver, WARP): never a target.
_SOFTWARE_VENDORS = frozenset({"microsoft"})


@dataclass(frozen=True)
class GpuInfo:
    """One hardware display adapter as DXGI enumerates it (``index`` = DXGI / DirectML order)."""

    index: int
    name: str
    vendor: str            # "nvidia" | "amd" | "intel" | "qualcomm" | "other"
    vendor_id: int
    device_id: int
    dedicated_mb: int      # DXGI DedicatedVideoMemory (on a UMA iGPU: the firmware carve-out)
    shared_mb: int         # DXGI SharedSystemMemory (system RAM the adapter may also use)
    uma: bool              # D3D12 architecture UMA flag: integrated GPU sharing system RAM
    luid: str              # "0x00000000_0x00016a0e" -- PDH instance "luid_<hi>_<lo>_phys_0"
    software: bool = False

    @property
    def integrated(self) -> bool:
        return self.uma

    @property
    def budget_mb(self) -> int:
        """Memory llama.cpp can reasonably put a model in: dedicated only on a discrete card (the
        shared half would page over PCIe); dedicated + shared on a UMA iGPU, where both are the
        same DRAM (the Vulkan device on the 8060S reports 48971 MiB = 32590 + 16204 roughly)."""
        return self.dedicated_mb + self.shared_mb if self.uma else self.dedicated_mb


_gpus_cache: list[GpuInfo] | None = None


def detect_gpus(*, refresh: bool = False) -> list[GpuInfo]:
    """Hardware display adapters in DXGI order (software adapters dropped). Cached after the
    first call (``refresh=True`` re-enumerates). ``[]`` off Windows or when DXGI cannot be used;
    never raises."""
    global _gpus_cache
    if _gpus_cache is not None and not refresh:
        return list(_gpus_cache)
    gpus: list[GpuInfo] = []
    if sys.platform == "win32":
        try:
            gpus = [g for g in _enumerate_dxgi() if not g.software]
        except Exception:  # noqa: BLE001 - discovery is advisory; every consumer has a CPU path
            log.debug("gpu: DXGI enumeration failed", exc_info=True)
            gpus = []
    _gpus_cache = gpus
    return list(gpus)


def primary_gpu(gpus: list[GpuInfo] | None = None) -> GpuInfo | None:
    """The adapter the model should run on: a discrete GPU with the most dedicated memory beats
    any integrated one (an RTX next to an Intel/AMD iGPU stays the target, as before); with only
    integrated GPUs, the one with the most dedicated memory. ``None`` when nothing was found."""
    gpus = detect_gpus() if gpus is None else gpus
    hw = [g for g in gpus if not g.software]
    if not hw:
        return None
    discrete = [g for g in hw if not g.uma]
    pool = discrete or hw
    return max(pool, key=lambda g: (g.dedicated_mb, g.shared_mb, -g.index))


def gpu_vendor(gpus: list[GpuInfo] | None = None) -> str:
    """``"nvidia"`` / ``"amd"`` / ``"intel"`` / ... of :func:`primary_gpu`, or ``"none"``.

    When DXGI enumerated nothing (``gpus`` empty and none passed in), NVML gets a say: an NVIDIA
    driver that initialises and hands out a device handle is proof enough of an NVIDIA GPU, and
    keeps a detection hiccup from demoting such a machine to the CPU / Vulkan paths.
    """
    g = primary_gpu(gpus)
    if g is not None:
        return g.vendor
    if gpus is None and _init():
        return "nvidia"
    return "none"


def primary_budget_mb(gpus: list[GpuInfo] | None = None) -> int | None:
    """``primary_gpu().budget_mb``, or -- when DXGI enumerated nothing but NVML answers (an NVIDIA
    driver hiccup in DXGI, the same case :func:`gpu_vendor` vouches for) -- the NVIDIA card's total
    memory per NVML; ``None`` when neither knows. This is what sizes the automatic model choice."""
    g = primary_gpu(gpus)
    if g is not None:
        return g.budget_mb
    if gpus is None:
        total = vram_total_mb()
        if total:
            return int(total)
    return None


def describe(gpus: list[GpuInfo] | None = None) -> str:
    """One line per adapter for logs and the doctor, e.g.
    ``Intel Arc(TM) B580 Graphics [intel, discrete, 12288 MiB dedicated + 0 MiB shared,
    driver 32.0.101.6795]`` -- the driver clause is Intel-only (gated on ``g.vendor == "intel"``)
    so NVIDIA and AMD adapters' lines stay byte-identical to before :func:`driver_version` existed;
    it is also omitted (not appended) when :func:`driver_version` cannot find one, so this never
    regresses the line on a machine/OS where the registry lookup finds nothing (e.g. in tests that
    construct a synthetic ``GpuInfo`` with no matching subkey)."""
    gpus = detect_gpus() if gpus is None else gpus
    if not gpus:
        return "no GPU adapter detected (DXGI unavailable or no hardware adapter)"
    primary = primary_gpu(gpus)
    lines = []
    #: Two identical-model Intel Arc cards share (vendor_id, device_id); this ranks each one among
    #: its same-model siblings in `gpus` (0, 1, 2, ...) so driver_version() can tell their registry
    #: subkeys apart instead of handing both the same (first-matching) driver string -- see
    #: driver_version's `rank` parameter. A no-op (always rank 0) whenever there is at most one
    #: adapter of a given model, which is every machine this has actually run on.
    model_rank: dict[tuple[int, int], int] = {}
    for g in gpus:
        kind = "integrated/UMA" if g.uma else "discrete"
        tag = "  <- primary" if primary is not None and g.index == primary.index else ""
        driver = None
        if g.vendor == "intel":
            model_key = (g.vendor_id, g.device_id)
            rank = model_rank.get(model_key, 0)
            model_rank[model_key] = rank + 1
            driver = driver_version(g, rank=rank)
        driver_txt = f", driver {driver}" if driver else ""
        lines.append(f"{g.name} [{g.vendor}, {kind}, {g.dedicated_mb} MiB dedicated + {g.shared_mb} MiB shared"
                      f"{driver_txt}]{tag}")
    return "\n".join(lines)


# ---- display driver version (winreg) --------------------------------------------------------
#: HKLM\SYSTEM\CurrentControlSet\Control\Class\{Display class GUID}\NNNN -- one subkey per
#: display adapter instance, each carrying MatchingDeviceID/DeviceInstanceID (identifies the
#: hardware) and DriverVersion (what we want). Enumerated instead of using SetupAPI so this stays
#: stdlib-only (winreg), matching the rest of this module's ctypes-only, no-new-dependency style.
_DISPLAY_CLASS_KEY = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"

#: (vendor_id, device_id, rank) -> DriverVersion string, or None (looked up and not found) --
#: populated only by the DEFAULT registry reader (see driver_version's `injected` branch below),
#: so an injected `reg_reader` used in tests never pollutes this cache across tests. `rank`
#: disambiguates two physically distinct adapters that share the same (vendor_id, device_id) --
#: see driver_version's docstring -- so they never share a cached answer.
_driver_cache: dict[tuple[int, int, int], str | None] = {}


def _registry_subkeys() -> Iterator[tuple[str, dict[str, str]]]:
    """Default ``reg_reader``: yields ``(subkey_name, {value_name: value})`` for every ``NNNN``
    subkey under :data:`_DISPLAY_CLASS_KEY`, string values only (``DriverVersion``,
    ``MatchingDeviceID`` etc. are all REG_SZ). Windows only (``winreg`` does not exist elsewhere);
    any registry error yields nothing rather than raising -- the caller treats "nothing yielded"
    the same as "no match found"."""
    import winreg  # stdlib, Windows-only: imported lazily so this module still imports anywhere

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS_KEY) as class_key:
        index = 0
        while True:
            try:
                sub_name = winreg.EnumKey(class_key, index)
            except OSError:
                break
            index += 1
            try:
                with winreg.OpenKey(class_key, sub_name) as sub_key:
                    values: dict[str, str] = {}
                    j = 0
                    while True:
                        try:
                            name, value, _vtype = winreg.EnumValue(sub_key, j)
                        except OSError:
                            break
                        j += 1
                        if isinstance(value, str):
                            values[name] = value
                    yield sub_name, values
            except OSError:  # noqa: PERF203 - a locked/vanished subkey: skip it, keep enumerating
                continue


def driver_version(
    g: GpuInfo, *, reg_reader: Callable[[], Iterable[tuple[str, dict[str, str]]]] | None = None,
    rank: int = 0,
) -> str | None:
    """The display driver version for adapter `g` (e.g. ``"32.0.101.6795"`` for an Intel Arc card,
    ``"32.0.31032.1003"`` measured here on the Radeon 8060S), read from the registry: every ``NNNN``
    subkey of :data:`_DISPLAY_CLASS_KEY` is scanned for ones whose ``MatchingDeviceID`` /
    ``DeviceInstanceID`` / any ``HardwareInformation.*`` value contains ``VEN_<vendor_id:04x>&
    DEV_<device_id:04x>`` (case-insensitive); the ``rank``-th such subkey's (0 = the first found,
    in registry enumeration order) ``DriverVersion`` is returned.

    `rank` disambiguates two adapters that share the same (vendor_id, device_id) -- e.g. two
    identical discrete cards -- which a PCI vendor/device id alone cannot tell apart (each gets its
    own registry subkey with a distinct ``DeviceInstanceID`` bus path, but that path is not parsed
    here). Every caller that only ever looks at one adapter per call (this function's default,
    doctor.py's Intel Arc advisory on ``primary`` alone) gets `rank`'s default, 0 -- today's
    original "first match wins" behaviour, unchanged. Callers that walk every detected adapter
    (:func:`describe`, `main`'s ``--json`` path) pass each same-model adapter's position among its
    siblings (0, 1, 2, ...) so a second identical adapter is not silently served the first one's
    driver string. `rank` beyond the number of matching subkeys clamps to the last one found.

    ``None`` off Windows, when no subkey matches, or on any registry error -- never raises.
    Cached per ``(vendor_id, device_id, rank)`` (registry access is a handful of syscalls per
    adapter, and this is called once per adapter by every :func:`describe`/doctor pass); the cache
    is skipped entirely when `reg_reader` is passed explicitly, so injected fakes (tests) never see
    a stale answer from a previous test or from the real machine.

    `reg_reader` is the injection point for tests: a zero-arg callable yielding
    ``(subkey_name, {value_name: value})`` pairs, exactly what :func:`_registry_subkeys` (the
    default) yields from the real registry.
    """
    key = (g.vendor_id, g.device_id, rank)
    injected = reg_reader is not None
    if not injected:
        if key in _driver_cache:
            return _driver_cache[key]
        if sys.platform != "win32":
            _driver_cache[key] = None
            return None
    reader = reg_reader or _registry_subkeys
    needle = f"ven_{g.vendor_id:04x}&dev_{g.device_id:04x}"
    matches: list[str] = []
    try:
        for _sub_name, values in reader():
            low = {str(name).lower(): value for name, value in values.items()}
            haystack = " ".join(
                str(value) for name, value in low.items()
                if name in ("matchingdeviceid", "deviceinstanceid") or name.startswith("hardwareinformation.")
            ).lower()
            if needle in haystack:
                driver = low.get("driverversion")
                if driver:
                    matches.append(str(driver))
    except Exception:  # noqa: BLE001 - registry access is advisory; never take a caller down over it
        log.debug("gpu: driver_version registry read failed", exc_info=True)
        matches = []
    result = matches[min(rank, len(matches) - 1)] if matches else None
    if not injected:
        _driver_cache[key] = result
    return result


# ---- Intel Arc discrete-GPU advisories ------------------------------------------------------
#: llama.cpp community measurement (2026): the Intel driver version from which Xe2 (Battlemage
#: B570/B580/B770/Pro B60/B70, and the Arc 140V iGPU) exposes VK_KHR_cooperative_matrix, which
#: llama.cpp's Vulkan backend uses for a large prompt-processing speedup (+329% pp512 on
#: gemma-2-2b, one report). No Intel Arc hardware exists on either dev box this branch was built
#: on, so this threshold is unverified here -- see intel_arc_notes()'s final note.
_XE2_COOPMAT_DRIVER = (32, 0, 101, 6795)

#: Best-effort PCI device-id ranges for identifying Alchemist vs Battlemage when the adapter name
#: carries no useful token (a generic OEM string, a hypervisor-passthrough name, ...). Sourced
#: from public Intel/PCI-ID-database listings, 2026-09-26; NOT verified against real hardware on
#: either dev box -- treat as a fallback behind the name-token check below, and widen/correct
#: these ranges if a real Arc card reports an id outside them.
#: Alchemist (DG2): consumer A-series (A310..A770) plus the workstation/mobile A-series.
_ALCHEMIST_DEVICE_IDS: tuple[range, ...] = (range(0x56A0, 0x56C0), range(0x5690, 0x56A0))
#: Battlemage (BMG): Arc B570/B580/B770 and Pro B60/B70.
_BATTLEMAGE_DEVICE_IDS: tuple[range, ...] = (range(0xE20B, 0xE213),)


def _intel_generation(name: str, device_id: int) -> str:
    """``"battlemage"`` | ``"alchemist"`` | ``""`` (unrecognised): name tokens first (LM Studio /
    Windows report the marketing name fairly consistently -- "Arc(TM) A770 Graphics", "Arc(TM)
    B580 Graphics", "Arc(TM) Pro B70 Graphics"), the best-effort device-id ranges above as a
    fallback for a generic/OEM name string."""
    low = name.lower()
    if "arc(tm) b" in low or "pro b" in low or re.search(r"\bb\d{3}\b", low):
        return "battlemage"
    if "arc(tm) a" in low or re.search(r"\ba\d{3}\b", low):
        return "alchemist"
    if any(device_id in r for r in _BATTLEMAGE_DEVICE_IDS):
        return "battlemage"
    if any(device_id in r for r in _ALCHEMIST_DEVICE_IDS):
        return "alchemist"
    return ""


def _parse_driver_tuple(driver: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(part) for part in driver.strip().split("."))
    except (ValueError, AttributeError):
        return None


def intel_arc_notes(g: GpuInfo, driver: str | None) -> list[str]:
    """Advisory strings for an Intel DISCRETE GPU (Arc A/B-series, Arc Pro B60/B70) -- ``[]`` for
    any other vendor or an integrated adapter (the Arc 140V/140T iGPUs already get ordinary
    ``vendor == "intel"`` Vulkan routing from :mod:`exocortex.llm.server` with no special-casing
    needed).

    Sourced from public llama.cpp-community reporting, 2026-09-26 (see the module-level research
    notes; NO Intel Arc GPU exists on either dev box this branch was built on -- everything below
    is unverified locally):

    * Vulkan, not SYCL, is llama.cpp's current recommendation for Arc (SYCL lags); LM Studio's
      runtime catalog offers no SYCL/OpenVINO pack at all, so this app's existing vendor ladder
      (``exocortex.llm.server.backend_ladder``, ``intel`` -> ``vulkan, cpu``) already lands there
      with zero extra code -- this note just makes that visible in the doctor.
    * Xe2 (Battlemage / the 140V iGPU) needs Intel driver >= 32.0.101.6795 for cooperative-matrix
      acceleration in llama.cpp's Vulkan backend; an older driver on a Battlemage card gets a
      warning naming the exact threshold.
    * Alchemist (A-series) had fp16/garbage-output reports in some Vulkan builds; ggml's Vulkan
      backend has ``GGML_VK_DISABLE_F16`` and other environment switches for this (see
      :mod:`exocortex.llm.runtimes`, once it lands, for the verified list scanned from
      ``ggml-vulkan.dll`` -- this note is unverified community folklore, not a measurement).
    * The default 75% GPU-memory budget (:attr:`GpuInfo.budget_mb`) sized against `g.dedicated_mb`.
    """
    if g.vendor != "intel" or g.uma:
        return []
    notes = [
        "Vulkan is the recommended llama.cpp backend for Intel Arc (SYCL lags as of 2026; LM "
        "Studio's runtime catalog has no SYCL/OpenVINO pack, so this app's normal vendor ladder "
        "already lands on Vulkan here)"
    ]
    generation = _intel_generation(g.name, g.device_id)
    if generation == "battlemage":
        parsed = _parse_driver_tuple(driver) if driver else None
        if parsed is not None and parsed < _XE2_COOPMAT_DRIVER:
            notes.append(
                f"driver {driver} is older than 32.0.101.6795 -- update the Intel Arc driver to get "
                "Xe2 cooperative-matrix acceleration in llama.cpp's Vulkan backend (llama.cpp "
                "community measurement, 2026)"
            )
        elif parsed is not None:
            notes.append(
                "driver is 32.0.101.6795 or newer: Xe2 cooperative-matrix acceleration should be "
                "active in llama.cpp's Vulkan backend (llama.cpp community measurement, 2026)"
            )
        else:
            notes.append(
                "could not read the driver version to confirm Xe2 cooperative-matrix support "
                "(needs Intel driver 32.0.101.6795 or newer; llama.cpp community measurement, 2026)"
            )
    elif generation == "alchemist":
        notes.append(
            "Alchemist (A-series): if llama.cpp's Vulkan backend produces garbled output, try "
            "GGML_VK_DISABLE_F16=1 in [llm].llama_env (llama.cpp community reports; unverified on "
            "this branch's hardware)"
        )
    if g.dedicated_mb > 0:
        budget_gb = g.dedicated_mb / 1024
        fits_gb = 0.75 * budget_gb
        notes.append(f"{budget_gb:.0f} GB card: the default 75% GPU-memory budget fits GGUFs up to "
                     f"~{fits_gb:.0f} GB")
    notes.append("no Intel GPU was available to test this branch -- run --doctor and one live answer, then report")
    return notes


# ---- DXGI / D3D12 plumbing -------------------------------------------------------------------
def _guid(text: str) -> ctypes.Structure:
    class GUID(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16),
                    ("Data4", ctypes.c_ubyte * 8)]

    parts = text.strip("{}").split("-")
    g = GUID()
    g.Data1, g.Data2, g.Data3 = int(parts[0], 16), int(parts[1], 16), int(parts[2], 16)
    for i, b in enumerate(bytes.fromhex(parts[3] + parts[4])):
        g.Data4[i] = b
    return g


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint32), ("DeviceId", ctypes.c_uint32),
                ("SubSysId", ctypes.c_uint32), ("Revision", ctypes.c_uint32),
                ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t), ("AdapterLuid", _LUID), ("Flags", ctypes.c_uint32)]


class _D3D12_FEATURE_DATA_ARCHITECTURE(ctypes.Structure):
    _fields_ = [("NodeIndex", ctypes.c_uint32), ("TileBasedRenderer", ctypes.c_int), ("UMA", ctypes.c_int),
                ("CacheCoherentUMA", ctypes.c_int)]


_DXGI_ADAPTER_FLAG_SOFTWARE = 2
_D3D_FEATURE_LEVEL_11_0 = 0xB000
_D3D12_FEATURE_ARCHITECTURE = 1


def _com_method(obj: ctypes.c_void_p, index: int, restype: Any, *argtypes: Any) -> Any:
    """Bind vtable slot ``index`` of the COM interface pointer ``obj``."""
    vtable = ctypes.cast(ctypes.c_void_p.from_address(obj.value), ctypes.POINTER(ctypes.c_void_p))
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return proto(vtable[index])


def _com_release(obj: ctypes.c_void_p) -> None:
    if obj and obj.value:
        try:
            _com_method(obj, 2, ctypes.c_ulong)(obj)   # IUnknown::Release
        except Exception:  # noqa: BLE001
            pass


def _d3d12_is_uma(d3d12: Any, adapter: ctypes.c_void_p) -> bool | None:
    """``D3D12_FEATURE_DATA_ARCHITECTURE.UMA`` for ``adapter`` via a throw-away device; ``None`` if
    D3D12 is unavailable or the device cannot be created (very old GPUs)."""
    if d3d12 is None:
        return None
    dev = ctypes.c_void_p()
    hr = d3d12.D3D12CreateDevice(adapter, _D3D_FEATURE_LEVEL_11_0,
                                 ctypes.byref(_guid("{189819f1-1db6-4b57-be54-1821339b85f7}")), ctypes.byref(dev))
    if hr != 0 or not dev.value:
        return None
    try:
        arch = _D3D12_FEATURE_DATA_ARCHITECTURE()
        check = _com_method(dev, 13, ctypes.c_long, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint)  # CheckFeatureSupport
        if check(dev, _D3D12_FEATURE_ARCHITECTURE, ctypes.byref(arch), ctypes.sizeof(arch)) != 0:
            return None
        return bool(arch.UMA)
    finally:
        _com_release(dev)


def _enumerate_dxgi() -> list[GpuInfo]:
    dxgi = ctypes.WinDLL("dxgi")
    dxgi.CreateDXGIFactory1.restype = ctypes.c_long
    factory = ctypes.c_void_p()
    hr = dxgi.CreateDXGIFactory1(ctypes.byref(_guid("{770aae78-f26f-4dba-a829-253c83d1b387}")), ctypes.byref(factory))
    if hr != 0 or not factory.value:
        raise OSError(f"CreateDXGIFactory1 failed: 0x{hr & 0xFFFFFFFF:08x}")
    try:
        d3d12 = ctypes.WinDLL("d3d12")
        d3d12.D3D12CreateDevice.restype = ctypes.c_long
    except OSError:
        d3d12 = None
    out: list[GpuInfo] = []
    try:
        enum_adapters1 = _com_method(factory, 12, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        index = 0
        while index < 32:
            adapter = ctypes.c_void_p()
            if enum_adapters1(factory, index, ctypes.byref(adapter)) != 0 or not adapter.value:
                break
            try:
                desc = _DXGI_ADAPTER_DESC1()
                get_desc1 = _com_method(adapter, 10, ctypes.c_long, ctypes.POINTER(_DXGI_ADAPTER_DESC1))
                if get_desc1(adapter, ctypes.byref(desc)) != 0:
                    continue
                software = bool(desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE)
                vendor = VENDOR_IDS.get(int(desc.VendorId), "other")
                if vendor in _SOFTWARE_VENDORS:
                    software = True
                uma = None if software else _d3d12_is_uma(d3d12, adapter)
                if uma is None:
                    # No D3D12 verdict: an adapter with (almost) no dedicated memory is integrated.
                    uma = desc.DedicatedVideoMemory < 1024 * 1024 * 1024
                out.append(GpuInfo(
                    index=index, name=str(desc.Description).strip(), vendor=vendor,
                    vendor_id=int(desc.VendorId), device_id=int(desc.DeviceId),
                    dedicated_mb=int(desc.DedicatedVideoMemory // (1024 * 1024)),
                    shared_mb=int(desc.SharedSystemMemory // (1024 * 1024)),
                    uma=bool(uma),
                    luid=f"0x{desc.AdapterLuid.HighPart & 0xFFFFFFFF:08x}_0x{desc.AdapterLuid.LowPart:08x}",
                    software=software,
                ))
            finally:
                _com_release(adapter)
                index += 1
    finally:
        _com_release(factory)
    return out


# =============================================================================================
# GPU memory watch
# =============================================================================================

_NVML_SUCCESS = 0

#: Lazy NVML state: ``ok`` is None until the first call, then True/False forever. A failure at any
#: point (load, init, handle, a later reading after a driver reset) parks it at False so every
#: subsequent call is a cheap early return -- the guard quietly stops rather than spamming errors.
_state: dict = {"ok": None, "lib": None, "handle": None}


class _NvmlMemory(ctypes.Structure):
    """``nvmlMemory_t`` (v1): three unsigned 64-bit fields, bytes."""

    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _init() -> bool:
    if _state["ok"] is not None:
        return bool(_state["ok"])
    _state["ok"] = False
    if sys.platform != "win32":
        return False
    try:
        lib = ctypes.WinDLL("nvml.dll")          # installed with the NVIDIA driver (System32)
        if lib.nvmlInit_v2() != _NVML_SUCCESS:
            return False
        handle = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != _NVML_SUCCESS:
            return False
        _state["lib"], _state["handle"], _state["ok"] = lib, handle, True
        return True
    except Exception:  # noqa: BLE001 - no NVIDIA driver / odd DLL: the guard just stays off
        log.debug("gpu: NVML unavailable", exc_info=True)
        return False


def _nvml_read_memory() -> _NvmlMemory | None:
    """One ``nvmlDeviceGetMemoryInfo`` call for GPU 0; ``None`` (NVML parked off) on any failure.
    Shared by :func:`vram_free_mb` and :func:`vram_total_mb` -- same call, same parking."""
    if not _init():
        return None
    try:
        mem = _NvmlMemory()
        rc = _state["lib"].nvmlDeviceGetMemoryInfo(_state["handle"], ctypes.byref(mem))
        if rc != _NVML_SUCCESS:
            log.warning("gpu: nvmlDeviceGetMemoryInfo failed (%s); disabling VRAM readings", rc)
            _state["ok"] = False
            return None
        return mem
    except Exception:  # noqa: BLE001 - a driver reset mid-call must never take the guard thread down
        log.warning("gpu: VRAM reading failed; disabling", exc_info=True)
        _state["ok"] = False
        return None


def vram_free_mb() -> int | None:
    """Free memory of the primary GPU in MiB, or ``None`` when it cannot be read.

    NVML first (GPU 0 via the NVIDIA driver) -- byte-for-byte the original behaviour and parking
    (``_state["ok"]`` parks ``False`` forever after any failure). Only when NVML has nothing to
    say does this fall through: if the primary adapter (:func:`primary_gpu`) exists and is not
    NVIDIA, the reading is its DXGI dedicated memory minus :func:`_pdh_dedicated_used_mb`,
    clamped at 0. An NVIDIA primary with NVML unavailable (e.g. a driver reset) never falls
    through to PDH -- NVML is authoritative for NVIDIA, so that case is "no reading", not a
    vendor-neutral guess. Never raises."""
    mem = _nvml_read_memory()
    if mem is not None:
        return int(mem.free // (1024 * 1024))
    primary = primary_gpu()
    if primary is None or primary.vendor == "nvidia":
        return None
    used = _pdh_dedicated_used_mb(primary.luid)
    if used is None:
        return None
    return max(0, primary.dedicated_mb - used)


def vram_total_mb() -> int | None:
    """Total memory of the primary GPU in MiB: NVML total on NVIDIA, else the primary adapter's
    DXGI dedicated memory (on a UMA iGPU, the firmware carve-out). ``None`` when neither is
    available."""
    mem = _nvml_read_memory()
    if mem is not None:
        return int(mem.total // (1024 * 1024))
    primary = primary_gpu()
    return primary.dedicated_mb if primary is not None else None


def vram_source() -> str:
    """Which reading :func:`vram_free_mb` uses right now: ``"nvml"`` (NVIDIA driver), ``"pdh"``
    (Windows' own GPU Adapter Memory counters, any other vendor), or ``"none"`` (neither)."""
    if _init():
        return "nvml"
    primary = primary_gpu()
    if primary is not None and primary.vendor != "nvidia" and _pdh_dedicated_used_mb(primary.luid) is not None:
        return "pdh"
    return "none"


# ---- PDH fallback (non-NVIDIA adapters): "GPU Adapter Memory" performance counters ------------
#: Lazy PDH state, one query kept open for the process lifetime -- same parking discipline as
#: NVML's ``_state``: ``ok`` is None until first use, then True/False forever (a failure never
#: retries; the guard thread just stops getting readings).
_pdh_state: dict = {"ok": None, "lib": None, "query": None, "counter": None, "luid": None}

_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2
#: PDH_CSTATUS_VALID_DATA / _NEW_DATA: the only two per-item statuses worth summing.
_PDH_CSTATUS_OK = frozenset({0x00000000, 0x00000001})


class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
    """``PDH_FMT_COUNTERVALUE``: a status DWORD followed by a union of value types. Only the
    ``largeValue`` (LONGLONG) member is read, at its correct union offset -- valid for any of the
    union's members since they all start there."""

    _fields_ = [("CStatus", ctypes.c_ulong), ("largeValue", ctypes.c_longlong)]


class _PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _PDH_FMT_COUNTERVALUE)]


def _pdh_open(luid: str) -> bool:
    """Open a PDH query with one wildcard-instance counter for ``luid`` and prime it with two
    throwaway collections (some PDH counters return no data until the second collection; doing
    that once here means every later call only needs one). Parks ``_pdh_state["ok"]`` at False
    forever on any failure."""
    _pdh_state["ok"] = False
    if sys.platform != "win32":
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
        pdh.PdhGetFormattedCounterArrayW.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                                                      ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
        pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_long
        pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        pdh.PdhCloseQuery.restype = ctypes.c_long

        query = ctypes.c_void_p()
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
            return False
        path = f"\\GPU Adapter Memory(luid_{luid}_phys_*)\\Dedicated Usage"
        counter = ctypes.c_void_p()
        if pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(counter)) != 0:
            pdh.PdhCloseQuery(query)               # the one handle this function opened: do not leak it
            return False
        pdh.PdhCollectQueryData(query)
        pdh.PdhCollectQueryData(query)
        _pdh_state.update(lib=pdh, query=query, counter=counter, luid=luid, ok=True)
        return True
    except Exception:  # noqa: BLE001 - no such counter on this OS / locale: PDH just stays off
        log.debug("gpu: PDH 'GPU Adapter Memory' counter unavailable", exc_info=True)
        return False


def _pdh_sum_dedicated_usage_mb(pdh: Any, counter: ctypes.c_void_p) -> int | None:
    """Sum the ``largeValue`` (bytes) of every valid item behind a wildcard-instance counter,
    converted to MiB. ``None`` when the array cannot be read or has no valid item."""
    buf_size, item_count = ctypes.c_ulong(0), ctypes.c_ulong(0)
    rc = pdh.PdhGetFormattedCounterArrayW(counter, _PDH_FMT_LARGE, ctypes.byref(buf_size), ctypes.byref(item_count), None)
    if (rc & 0xFFFFFFFF) not in (0, _PDH_MORE_DATA) or buf_size.value == 0:
        return None
    buf = (ctypes.c_char * buf_size.value)()
    rc = pdh.PdhGetFormattedCounterArrayW(counter, _PDH_FMT_LARGE, ctypes.byref(buf_size), ctypes.byref(item_count), buf)
    if (rc & 0xFFFFFFFF) != 0:
        return None
    items = ctypes.cast(buf, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM_W * item_count.value)).contents
    total_bytes, matched = 0, 0
    for item in items:
        if item.FmtValue.CStatus in _PDH_CSTATUS_OK:
            total_bytes += int(item.FmtValue.largeValue)
            matched += 1
    return total_bytes // (1024 * 1024) if matched else None


def _pdh_dedicated_used_mb(luid: str) -> int | None:
    """The adapter's "Dedicated Usage" PDH counter (summed across its ``phys_*`` nodes) in MiB,
    or ``None`` when it cannot be read -- then and forever after, exactly like NVML. ``luid`` is
    ``GpuInfo.luid`` ("0x00000000_0x00016a0e"); the PDH instance is "luid_<luid>_phys_0" (a
    wildcard picks up every ``phys_N`` a multi-die adapter may expose)."""
    if _pdh_state["ok"] is False:
        return None
    if _pdh_state["ok"] is None or _pdh_state["luid"] != luid:
        if not _pdh_open(luid):
            return None
    try:
        pdh = _pdh_state["lib"]
        if (pdh.PdhCollectQueryData(_pdh_state["query"]) & 0xFFFFFFFF) != 0:
            _pdh_state["ok"] = False
            return None
        used = _pdh_sum_dedicated_usage_mb(pdh, _pdh_state["counter"])
        if used is None:
            _pdh_state["ok"] = False
            return None
        return used
    except Exception:  # noqa: BLE001 - never take the guard thread down over a counter hiccup
        log.debug("gpu: PDH reading failed; disabling", exc_info=True)
        _pdh_state["ok"] = False
        return None


# ---- run.bat / doctor pip extras ---------------------------------------------------------------
#: requirements-*.txt files run.bat should additionally install for a given primary-GPU vendor.
#: A pure mapping -- retuned freely as the pip packages this needs evolve.
_EXTRAS_BY_VENDOR: dict[str, list[str]] = {
    "nvidia": ["requirements-nvidia.txt"],
    "amd": ["requirements-directml.txt"],
}


def extras_for_vendor(vendor: str) -> list[str]:
    """Extra requirement files for ``vendor`` ("nvidia" | "amd" | "intel" | "qualcomm" | "other" |
    "none"), or ``[]`` when nothing extra is needed."""
    return list(_EXTRAS_BY_VENDOR.get(vendor, []))


# =============================================================================================
# Portable-install hardware helpers (exocortex.hardware's probes)
# =============================================================================================
# 2026-09-27 (portable-install): three small, independently-injectable helpers so
# exocortex.hardware.collect() can build a HardwareReport without any new third-party dependency
# (ctypes/winreg only, matching this module's existing style) -- system RAM, the CPU model string,
# and (NVIDIA only) the actual driver version, as opposed to the display-driver string
# driver_version() already reads for Intel.


def _nvml_driver_version() -> str | None:
    """``nvmlSystemGetDriverVersion`` via the existing lazy NVML loader (:func:`_init`); the same
    e.g. ``"592.14"`` string ``nvidia-smi`` reports. ``None`` when NVML is unavailable or the call
    fails -- never raises."""
    if not _init():
        return None
    try:
        buf = ctypes.create_string_buffer(96)
        rc = _state["lib"].nvmlSystemGetDriverVersion(buf, ctypes.c_uint(96))
        if rc != _NVML_SUCCESS:
            return None
        text = buf.value.decode("utf-8", errors="replace").strip()
        return text or None
    except Exception:  # noqa: BLE001 - advisory only
        log.debug("gpu: nvmlSystemGetDriverVersion failed", exc_info=True)
        return None


def _nvidia_version_from_registry_string(driver: str | None) -> str | None:
    """Re-derives NVIDIA's own ``"NNN.NN"`` driver numbering (what ``nvidia-smi`` / the NVIDIA
    Control Panel show) from the Windows registry ``DriverVersion`` string :func:`driver_version`
    reads (e.g. ``"32.0.15.9200"``): the last two dot-separated groups are concatenated
    (``"15"`` + ``"9200"`` -> ``"159200"``), a single leading digit is dropped (``"59200"``), and
    the remainder is re-split into major (all but the last two digits) and minor (the last two,
    always 2 digits) -- ``"592.00"``. ``None`` for ``None``/empty input or a string that does not
    have at least four dot-separated groups with two digits left after dropping the leading one."""
    if not driver:
        return None
    parts = driver.strip().split(".")
    if len(parts) < 4:
        return None
    combined = parts[-2] + parts[-1]
    if len(combined) < 2:
        return None
    trimmed = combined[1:]
    if len(trimmed) < 3:
        return None
    return f"{trimmed[:-2]}.{trimmed[-2:]}"


def nvidia_driver_version(gpus: list[GpuInfo] | None = None) -> str | None:
    """The primary GPU's actual NVIDIA driver version (e.g. ``"592.14"``), or ``None`` when the
    primary adapter (:func:`primary_gpu`/:func:`gpu_vendor` of `gpus`) is not NVIDIA.

    NVML's ``nvmlSystemGetDriverVersion`` first (:func:`_nvml_driver_version`); when NVML has
    nothing to say (no NVIDIA driver loaded, or DXGI enumerated nothing but NVML still proves an
    NVIDIA GPU per :func:`gpu_vendor`'s own fallback -- then there is no `GpuInfo` to read a
    registry string from either, so the answer is ``None``), the registry ``DriverVersion`` string
    for the primary adapter (:func:`driver_version`) is re-derived via
    :func:`_nvidia_version_from_registry_string`. Never raises."""
    if gpu_vendor(gpus) != "nvidia":
        return None
    nvml = _nvml_driver_version()
    if nvml:
        return nvml
    primary = primary_gpu(gpus)
    if primary is None:
        return None
    try:
        registry = driver_version(primary)
    except Exception:  # noqa: BLE001 - advisory only
        registry = None
    return _nvidia_version_from_registry_string(registry)


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def system_memory_mb() -> tuple[int | None, int | None]:
    """``(total_mb, avail_mb)`` of physical RAM via ``GlobalMemoryStatusEx`` (ctypes, stdlib-only).
    ``(None, None)`` off Windows or on any failure -- never raises."""
    if sys.platform != "win32":
        return None, None
    try:
        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
        k32.GlobalMemoryStatusEx.restype = ctypes.c_int
        if not k32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None, None
        total_mb = int(stat.ullTotalPhys // (1024 * 1024))
        avail_mb = int(stat.ullAvailPhys // (1024 * 1024))
        return total_mb, avail_mb
    except Exception:  # noqa: BLE001 - advisory only
        log.debug("gpu: GlobalMemoryStatusEx failed", exc_info=True)
        return None, None


def cpu_name() -> str:
    """The CPU model string, e.g. ``"AMD Ryzen AI MAX+ 395"`` -- ``winreg``
    ``HKLM\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0``'s ``ProcessorNameString`` (what
    Windows itself shows in System Information), falling back to ``platform.processor()``; ``""``
    when neither has an answer. Never raises."""
    if sys.platform == "win32":
        try:
            import winreg

            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
                value, _vtype = winreg.QueryValueEx(k, "ProcessorNameString")
                text = str(value).strip()
                if text:
                    return text
        except Exception:  # noqa: BLE001 - advisory only; fall through to the platform fallback
            log.debug("gpu: cpu_name registry read failed", exc_info=True)
    try:
        import platform

        return platform.processor() or ""
    except Exception:  # noqa: BLE001
        return ""


class VramGuard:
    """Turns a stream of free-MiB readings into at most an occasional alert / recovery message.

    ``sample()`` is called on a fixed cadence and returns ``None`` almost always; a string is
    either the low-memory alert (repeated at most every ``cooldown_s`` while pressure persists)
    or the single recovery message. ``None`` readings reset the dip streak: no data is never
    treated as pressure, and a driver hiccup cannot alert.
    """

    def __init__(
        self,
        free_mb_fn: Callable[[], int | None],
        threshold_mb: int,
        *,
        consecutive: int = 2,
        cooldown_s: float = 180.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._free = free_mb_fn
        self._threshold = int(threshold_mb)
        self._consecutive = max(1, int(consecutive))
        self._cooldown = float(cooldown_s)
        self._clock = clock
        self._streak = 0
        self._last_alert: float | None = None
        self._alerted = False

    def sample(self) -> str | None:
        try:
            free = self._free()
        except Exception:  # noqa: BLE001 - a broken reader must never break the guard loop
            log.debug("gpu: free_mb_fn raised", exc_info=True)
            free = None
        if free is None:
            self._streak = 0
            return None
        if free < self._threshold:
            self._streak += 1
            now = self._clock()
            if self._streak >= self._consecutive and (
                self._last_alert is None or now - self._last_alert >= self._cooldown
            ):
                self._last_alert = now
                self._alerted = True
                return (
                    f"GPU memory low: {free} MiB free -- another model may be loading; "
                    "answers and the desktop may stall"
                )
            return None
        self._streak = 0
        if self._alerted and free >= self._threshold + RECOVERY_MARGIN_MB:
            self._alerted = False
            return f"GPU memory recovered ({free} MiB free)"
        return None


# =============================================================================================
# CLI -- ``python -m exocortex.gpu`` (run.bat and the doctor shell out to this)
# =============================================================================================
def main(argv: list[str] | None = None) -> int:
    """``describe()`` by default; ``--vendor`` prints exactly one word, ``--json`` the adapters
    (``dataclasses.asdict``) plus ``"primary"`` (index) and ``"vendor"``, ``--extras`` the
    requirement files run.bat should additionally install, space-separated on one line. Always
    exits 0 -- this is parsed by a batch file, never a reason to fail the launch."""
    import argparse
    import dataclasses
    import json

    parser = argparse.ArgumentParser(prog="python -m exocortex.gpu", add_help=True,
                                      description="GPU discovery for ExoCortex (run.bat / doctor).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--vendor", action="store_true", help="print only the primary GPU's vendor")
    group.add_argument("--json", action="store_true", help="print every adapter as JSON")
    group.add_argument("--extras", action="store_true", help="print this vendor's extra pip requirement files")
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 0   # --help or a bad flag: never fail the caller (run.bat just gets no output)

    gpus = detect_gpus()
    vendor = gpu_vendor(gpus)
    if args.vendor:
        print(vendor)
    elif args.json:
        primary = primary_gpu(gpus)
        adapters = []
        model_rank: dict[tuple[int, int], int] = {}
        for g in gpus:
            d = dataclasses.asdict(g)
            if g.vendor == "intel":
                model_key = (g.vendor_id, g.device_id)
                rank = model_rank.get(model_key, 0)
                model_rank[model_key] = rank + 1
                d["driver"] = driver_version(g, rank=rank)
            adapters.append(d)
        payload = {
            "adapters": adapters,
            "primary": primary.index if primary is not None else None,
            "vendor": vendor,
        }
        print(json.dumps(payload))
    elif args.extras:
        print(" ".join(extras_for_vendor(vendor)))
    else:
        print(describe(gpus))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
