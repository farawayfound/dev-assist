"""Windows Core Audio endpoint state: whether a capture device is muted, and its input volume.

A WASAPI shared-mode stream on a muted (or 0 %) capture endpoint opens without any error and then
delivers digital silence, so ``AudioCapture`` reports "started" while hearing nothing -- the one
microphone failure PortAudio cannot see. This module reads (and, on request, clears) that mute
through ``IAudioEndpointVolume``, matching endpoints by the friendly name PortAudio reports for
WASAPI devices ("Microphone Array (Realtek(R) Audio)").

``comtypes`` (already a requirement, for the caret tracker) is imported lazily and the handful of
Core Audio interfaces needed are declared by hand, so this module imports cleanly anywhere; every
public function returns ``None`` / ``False`` off Windows or when Core Audio is unavailable. Calls
are cheap (a few ms) and meant for the Qt main thread, where COM is already initialised.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

_E_CAPTURE = 1
_E_CONSOLE = 0
_DEVICE_STATE_ACTIVE = 0x1
_STGM_READ = 0
_VT_LPWSTR = 31
_PKEY_FRIENDLY_NAME = ("{A45C254E-DF1C-4EFD-8020-67D146A850E0}", 14)
_CLSID_MM_DEVICE_ENUMERATOR = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"


@dataclass(frozen=True)
class EndpointState:
    """One capture endpoint as Windows sees it."""

    name: str
    muted: bool
    volume: float  # master volume scalar, 0..1


_interfaces: dict[str, Any] | None = None


def _declare() -> dict[str, Any]:
    """Declare the Core Audio interfaces once (needs comtypes, hence not at import time)."""
    global _interfaces
    if _interfaces is not None:
        return _interfaces
    from ctypes import HRESULT, POINTER, c_float, c_uint, c_ushort, c_void_p, c_wchar_p
    from ctypes.wintypes import BOOL, DWORD, LPWSTR

    from comtypes import COMMETHOD, GUID, IUnknown

    class PROPERTYKEY(ctypes.Structure):
        _fields_ = [("fmtid", GUID), ("pid", DWORD)]

    class PROPVARIANT(ctypes.Structure):
        _fields_ = [("vt", c_ushort), ("r1", c_ushort), ("r2", c_ushort), ("r3", c_ushort),
                    ("pwszVal", c_wchar_p), ("pad", c_void_p)]

    class IAudioEndpointVolume(IUnknown):
        _iid_ = GUID("{5CDF2C82-841E-4546-9722-0CF74078229A}")
        _methods_ = [
            COMMETHOD([], HRESULT, "RegisterControlChangeNotify", (["in"], c_void_p)),
            COMMETHOD([], HRESULT, "UnregisterControlChangeNotify", (["in"], c_void_p)),
            COMMETHOD([], HRESULT, "GetChannelCount", (["out"], POINTER(c_uint))),
            COMMETHOD([], HRESULT, "SetMasterVolumeLevel", (["in"], c_float), (["in"], POINTER(GUID))),
            COMMETHOD([], HRESULT, "SetMasterVolumeLevelScalar", (["in"], c_float), (["in"], POINTER(GUID))),
            COMMETHOD([], HRESULT, "GetMasterVolumeLevel", (["out"], POINTER(c_float))),
            COMMETHOD([], HRESULT, "GetMasterVolumeLevelScalar", (["out"], POINTER(c_float))),
            COMMETHOD([], HRESULT, "SetChannelVolumeLevel", (["in"], c_uint), (["in"], c_float), (["in"], POINTER(GUID))),
            COMMETHOD([], HRESULT, "SetChannelVolumeLevelScalar", (["in"], c_uint), (["in"], c_float), (["in"], POINTER(GUID))),
            COMMETHOD([], HRESULT, "GetChannelVolumeLevel", (["in"], c_uint), (["out"], POINTER(c_float))),
            COMMETHOD([], HRESULT, "GetChannelVolumeLevelScalar", (["in"], c_uint), (["out"], POINTER(c_float))),
            COMMETHOD([], HRESULT, "SetMute", (["in"], BOOL), (["in"], POINTER(GUID))),
            COMMETHOD([], HRESULT, "GetMute", (["out"], POINTER(BOOL))),
        ]

    class IPropertyStore(IUnknown):
        _iid_ = GUID("{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}")
        _methods_ = [
            COMMETHOD([], HRESULT, "GetCount", (["out"], POINTER(DWORD))),
            COMMETHOD([], HRESULT, "GetAt", (["in"], DWORD), (["out"], POINTER(PROPERTYKEY))),
            COMMETHOD([], HRESULT, "GetValue", (["in"], POINTER(PROPERTYKEY)), (["out"], POINTER(PROPVARIANT))),
        ]

    class IMMDevice(IUnknown):
        _iid_ = GUID("{D666063F-1587-4E43-81F1-B948E807363F}")
        _methods_ = [
            COMMETHOD([], HRESULT, "Activate", (["in"], POINTER(GUID)), (["in"], DWORD), (["in"], c_void_p),
                      (["out"], POINTER(c_void_p))),
            COMMETHOD([], HRESULT, "OpenPropertyStore", (["in"], DWORD), (["out"], POINTER(POINTER(IPropertyStore)))),
            COMMETHOD([], HRESULT, "GetId", (["out"], POINTER(LPWSTR))),
            COMMETHOD([], HRESULT, "GetState", (["out"], POINTER(DWORD))),
        ]

    class IMMDeviceCollection(IUnknown):
        _iid_ = GUID("{0BD7A1BE-7A1A-44DB-8397-CC5392387B5E}")
        _methods_ = [
            COMMETHOD([], HRESULT, "GetCount", (["out"], POINTER(c_uint))),
            COMMETHOD([], HRESULT, "Item", (["in"], c_uint), (["out"], POINTER(POINTER(IMMDevice)))),
        ]

    class IMMDeviceEnumerator(IUnknown):
        _iid_ = GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
        _methods_ = [
            COMMETHOD([], HRESULT, "EnumAudioEndpoints", (["in"], ctypes.c_int), (["in"], DWORD),
                      (["out"], POINTER(POINTER(IMMDeviceCollection)))),
            COMMETHOD([], HRESULT, "GetDefaultAudioEndpoint", (["in"], ctypes.c_int), (["in"], ctypes.c_int),
                      (["out"], POINTER(POINTER(IMMDevice)))),
        ]

    _interfaces = {
        "GUID": GUID, "PROPERTYKEY": PROPERTYKEY, "IAudioEndpointVolume": IAudioEndpointVolume,
        "IMMDeviceEnumerator": IMMDeviceEnumerator,
    }
    return _interfaces


def _with_com(fn: Callable[[], T], fallback: T) -> T:
    """Run ``fn`` with COM initialised on this thread; any failure is logged and gives ``fallback``."""
    if sys.platform != "win32":
        return fallback
    try:
        import comtypes
    except Exception:  # noqa: BLE001
        log.debug("comtypes unavailable; no endpoint state", exc_info=True)
        return fallback
    initialized = False
    try:
        comtypes.CoInitialize()
        initialized = True
    except OSError:
        pass  # already initialised on this thread in another mode
    try:
        return fn()
    except Exception:  # noqa: BLE001
        log.debug("Core Audio endpoint query failed", exc_info=True)
        return fallback
    finally:
        if initialized:
            comtypes.CoUninitialize()


def _friendly_name(device: Any, i: dict[str, Any]) -> str:
    store = device.OpenPropertyStore(_STGM_READ)
    key = i["PROPERTYKEY"](i["GUID"](_PKEY_FRIENDLY_NAME[0]), _PKEY_FRIENDLY_NAME[1])
    value = store.GetValue(ctypes.byref(key))
    try:
        return str(value.pwszVal or "") if value.vt == _VT_LPWSTR else ""
    finally:
        ctypes.oledll.ole32.PropVariantClear(ctypes.byref(value))


def _find_capture_endpoint(name: str | None) -> tuple[Any, str] | None:
    """The active capture endpoint called ``name`` (exact, then case-insensitive containment
    either way), or the default capture endpoint when ``name`` is empty / "default"."""
    import comtypes

    i = _declare()
    enum = comtypes.CoCreateInstance(i["GUID"](_CLSID_MM_DEVICE_ENUMERATOR), i["IMMDeviceEnumerator"],
                                     comtypes.CLSCTX_ALL)
    wanted = (name or "").strip()
    if not wanted or wanted.lower() == "default":
        device = enum.GetDefaultAudioEndpoint(_E_CAPTURE, _E_CONSOLE)
        return device, _friendly_name(device, i)
    collection = enum.EnumAudioEndpoints(_E_CAPTURE, _DEVICE_STATE_ACTIVE)
    devices = [collection.Item(k) for k in range(collection.GetCount())]
    named = [(d, _friendly_name(d, i)) for d in devices]
    for device, friendly in named:
        if friendly == wanted:
            return device, friendly
    low = wanted.lower()
    for device, friendly in named:
        if friendly and (low in friendly.lower() or friendly.lower() in low):
            return device, friendly
    return None


def _volume(device: Any) -> Any:
    import comtypes

    i = _declare()
    iface = i["IAudioEndpointVolume"]
    ptr = device.Activate(ctypes.byref(iface._iid_), comtypes.CLSCTX_ALL, None)
    return ctypes.cast(ptr, ctypes.POINTER(iface))


def capture_endpoint_state(name: str | None = None) -> EndpointState | None:
    """Mute + volume of the capture endpoint called ``name`` (a PortAudio WASAPI device name), or
    of the default one. ``None`` when it cannot be found or Core Audio is unavailable."""

    def _query() -> EndpointState | None:
        found = _find_capture_endpoint(name)
        if found is None:
            return None
        device, friendly = found
        vol = _volume(device)
        return EndpointState(name=friendly, muted=bool(vol.GetMute()), volume=float(vol.GetMasterVolumeLevelScalar()))

    return _with_com(_query, None)


def set_capture_muted(name: str | None, muted: bool) -> bool:
    """Mute or unmute the capture endpoint called ``name`` (or the default one) in Windows itself,
    exactly like the Sound settings' toggle. True when it was applied."""

    def _apply() -> bool:
        found = _find_capture_endpoint(name)
        if found is None:
            return False
        device, friendly = found
        _volume(device).SetMute(bool(muted), None)
        log.info("capture endpoint %r %s in Windows", friendly, "muted" if muted else "unmuted")
        return True

    return _with_com(_apply, False)
