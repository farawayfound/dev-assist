"""Local OCR over a captured screen frame: Windows OCR (PyWinRT) or RapidOCR, plus the
reading-order text assembly shared by both engines.

Both real engines (``winrt.windows.media.ocr`` and ``rapidocr``) are imported lazily inside
the classes that use them, so this module has no import-time side effects and stays
importable on any platform; on a platform/venv without the engine, construction raises
:class:`OcrUnavailable` instead of an import error escaping.

RapidOCR can run its PP-OCR models through an accelerator instead of onnxruntime on the CPU
(``device``, see :func:`resolve_ocr_device`): OpenVINO on an Intel integrated GPU or NPU, or --
on a non-Intel GPU -- onnxruntime's DirectML execution provider (``onnxruntime-directml``).
Measured on the ORIGINAL target laptop (Core Ultra 9 285H / Arc 140T / AI Boost, a dense
2560x1600 page): 3.7 s and 37 s of CPU time per pass on onnxruntime, 0.73 s / 0.74 s CPU on the
iGPU, 1.45 s / 0.6 s CPU on the NPU, 1.9 s / 9.9 s CPU on OpenVINO's CPU plugin; Windows OCR
takes 0.11 s. OCR never touches the discrete NVIDIA GPU (that one belongs to Whisper and
llama-server).

AMD iGPU (measured 2026-09-25, Ryzen AI MAX+ 395 / Radeon 8060S, OpenVINO 2026.4.0,
``core.available_devices`` = ``['CPU', 'GPU']``, GPU ``FULL_DEVICE_NAME`` "gfx1151 (iGPU)"):
OpenVINO's GPU plugin is NOT usable here -- ``RapidOcr("igpu", ...)`` crashes the interpreter
(``STATUS_STACK_BUFFER_OVERRUN``, 0xC0000409) inside the plugin's kernel compiler, which emits
Intel GEN/vISA inline assembly (literal ``.decl`` / ``.implicit_PSEUDO_INPUT`` directives) that
the AMD OpenCL ICD cannot parse -- reproducible cold and with a warm on-disk ``CACHE_DIR`` (see
``scripts/bench_ocr.py``'s docstring for the exact table and error). :func:`resolve_ocr_device`
therefore never routes a non-Intel OpenVINO ``GPU`` device through OpenVINO; instead it offers
onnxruntime's DirectML execution provider (:func:`directml_available`, ``device_spec`` "dml:<id>",
run through :class:`_OrtSession`) to any non-NVIDIA hardware GPU. Measured on this same iGPU
with ``onnxruntime-directml`` on a real desktop capture: DirectML worked and returned text
matching the CPU engine (see ``scripts/bench_ocr.py``'s docstring for the numbers) -- it is this
machine's ``"auto"`` default for "igpu" today (a human can flip it back once OpenVINO fixes the
AMD GPU plugin).

Intel Arc DISCRETE GPU (added 2026-09-26, no Intel hardware to verify live -- see
:func:`resolve_ocr_device`'s docstring, rule (a2)): OpenVINO names an Arc card e.g. "Intel(R)
Arc(TM) B580 Graphics (dGPU)" -- the same "(dGPU)" suffix rule (a) uses to keep OCR off an NVIDIA
laptop's discrete GPU (which holds Whisper and llama-server there). On an Intel-vendor discrete
GPU the exclusion is wrong: the emitted kernels are genuine Intel GEN/vISA (the crash above is an
AMD-OpenCL-can't-parse-Intel-assembly problem, not an Intel-plugin problem), and on a machine
where Arc is the only real GPU it is not shadowing anything -- there is no separate NVIDIA card
for it to steal cycles from. Rule (a2) therefore accepts an Intel ``(dGPU)`` OpenVINO name too,
but only once :func:`exocortex.gpu.detect_gpus` (never OpenVINO's own device list, which cannot
name a vendor beyond "Intel"/"NVIDIA"/a raw string) confirms no NVIDIA adapter shares the machine.

Process-order rule (measured 2026-09-23): a :class:`RapidOcr` created AFTER a :class:`WindowsOcr`
crashes the interpreter (access violation, with onnxruntime or OpenVINO on any device), while
RapidOCR first and Windows OCR second coexist for the life of the process. :class:`RapidOcr`
refuses (:class:`OcrUnavailable`) instead of crashing, and the screen reader builds its accurate
RapidOCR engine before the Windows OCR sampling engine.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from exocortex.gpu import GpuInfo

log = logging.getLogger(__name__)

#: RapidOCR is noticeably slower on very large captures; downscale to this before recognizing
#: and scale the returned boxes back up (SPEC.md: "half-resolution does not make it faster"
#: below this, but a 5120px-wide capture is well above it).
_RAPIDOCR_MAX_SIDE = 2200

#: ``[screen].ocr_device`` values. "auto" = igpu if present, else npu, else cpu.
OCR_DEVICES: tuple[str, ...] = ("auto", "igpu", "npu", "cpu")

#: Rec-model input widths the iGPU / NPU are compiled for (static shapes): a batch of text crops
#: is padded to the next bucket. PP-OCR rec inputs are 48 px high with width = 48 x the widest
#: crop's aspect ratio in the batch, from 320 up; each bucket is one compile (NPU ~8-14 s, iGPU
#: ~2-6 s the first time, ~0.1-0.8 s from the on-disk cache after that). Static shapes matter on
#: the iGPU too: with dynamic shapes OpenVINO's GPU plugin JIT-compiles kernels for every new
#: crop width it meets, and a fresh screen has dozens (measured: a 37 s accurate read).
_NPU_REC_WIDTH_BUCKETS: tuple[int, ...] = (320, 480, 640, 800, 960, 1280, 1600, 1920, 2560, 3200, 4096)
#: The buckets compiled eagerly at warm-up (the rest compile on first use and are cached).
_WARM_REC_WIDTH_BUCKETS: tuple[int, ...] = (320, 480, 640, 800, 960, 1280)
_NPU_BATCH = 6   # RapidOCR's rec_batch_num / cls_batch_num
#: Det-model inputs are padded (bottom/right) to a multiple of this, so a foreground window of any
#: size maps onto a handful of shapes: each distinct shape is a compile on the NPU (static shapes)
#: and a first-inference cost on the iGPU. The prediction map is cropped back afterwards.
_DET_PAD = 256

#: Set once a :class:`WindowsOcr` exists in this process (see the module docstring's order rule).
_windows_ocr_created = False

#: Cached result of :func:`directml_available` (``None`` = not probed yet).
_dml_available: bool | None = None


@dataclass(frozen=True)
class OcrLine:
    """One recognised line of text."""

    text: str
    box: tuple[int, int, int, int]  # x, y, w, h, relative to the frame that was recognized
    confidence: float


@dataclass(frozen=True)
class OcrResult:
    """The full result of one :meth:`OcrEngine.recognize` call."""

    lines: tuple[OcrLine, ...]
    text: str
    elapsed_ms: float
    engine: str


class OcrUnavailable(RuntimeError):
    """Raised when an OCR engine cannot be created (missing language pack, missing package, ...)."""


class OcrEngine(Protocol):
    """What :mod:`exocortex.screen.reader` needs from an OCR engine."""

    name: str

    def recognize(self, image_bgr: np.ndarray) -> OcrResult: ...


# --------------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------------- #


def _downscale_if_needed(image_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Shrink ``image_bgr`` so its longer side is at most ``max_side``. Returns ``(image, scale)``.

    ``scale`` is the factor applied to the original (``scale <= 1.0``; ``1.0`` = untouched),
    so a box found in the returned image maps back to the original via ``coord / scale``.
    """
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side or h == 0 or w == 0:
        return image_bgr, 1.0
    scale = max_side / float(longest)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    from PIL import Image

    resized = np.array(Image.fromarray(image_bgr, mode="RGB").resize((new_w, new_h), Image.BILINEAR))
    return resized, scale


def _scale_box(box: tuple[float, float, float, float], scale: float) -> tuple[int, int, int, int]:
    x, y, w, h = box
    if scale != 1.0:
        x, y, w, h = x / scale, y / scale, w / scale, h / scale
    return (round(x), round(y), round(w), round(h))


def lines_to_text(lines: Sequence[OcrLine]) -> str:
    """Assemble ``lines`` into reading-order text.

    Lines are bucketed into rows by y-centre (within ``0.6 * median line height`` of the
    row's running mean y-centre), rows are ordered top -> bottom and, within a row, items
    left -> right. Each item's own text has internal whitespace collapsed first; items in a
    row are joined with two spaces, rows with ``"\\n"``, and empty items/rows are dropped.
    """
    items: list[tuple[OcrLine, str]] = []
    for ln in lines:
        text = " ".join(ln.text.split())
        if text:
            items.append((ln, text))
    if not items:
        return ""

    def y_centre(ln: OcrLine) -> float:
        return ln.box[1] + ln.box[3] / 2.0

    heights = [ln.box[3] for ln, _ in items if ln.box[3] > 0]
    median_h = float(np.median(heights)) if heights else 1.0
    band = max(median_h * 0.6, 1e-6)

    rows: list[list[tuple[OcrLine, str]]] = []
    for item in sorted(items, key=lambda pair: y_centre(pair[0])):
        yc = y_centre(item[0])
        for row in rows:
            row_yc = sum(y_centre(p[0]) for p in row) / len(row)
            if abs(yc - row_yc) <= band:
                row.append(item)
                break
        else:
            rows.append([item])
    rows.sort(key=lambda row: sum(y_centre(p[0]) for p in row) / len(row))

    row_texts = []
    for row in rows:
        row_sorted = sorted(row, key=lambda pair: pair[0].box[0])
        row_text = "  ".join(text for _, text in row_sorted)
        if row_text:
            row_texts.append(row_text)
    return "\n".join(row_texts)


def _normalize_line(line: str) -> str:
    return " ".join(line.split()).casefold()


def new_lines(previous: Sequence[str], current: Sequence[str]) -> list[str]:
    """Lines in ``current`` not present (case/space-normalised) in ``previous``."""
    seen = {_normalize_line(p) for p in previous}
    return [line for line in current if _normalize_line(line) not in seen]


# --------------------------------------------------------------------------------- #
# Windows OCR (winrt.windows.media.ocr)
# --------------------------------------------------------------------------------- #


class WindowsOcr:
    """Windows' built-in OCR engine via PyWinRT (``Windows.Media.Ocr``).

    Creates the engine and a private ``asyncio`` event loop once, in ``__init__``; each
    :meth:`recognize` call runs the async recognition to completion on that same loop
    (``run_until_complete``), so this instance is meant to be used from one thread.
    """

    name = "windows"
    device = "cpu"

    def __init__(self) -> None:
        global _windows_ocr_created
        from winrt.windows.media.ocr import OcrEngine as _WinOcrEngine

        engine = _WinOcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            raise OcrUnavailable("Windows OCR: no OCR engine for the current user profile languages")
        _windows_ocr_created = True
        self._engine = engine
        self._max_dim = int(_WinOcrEngine.max_image_dimension)
        self._loop = asyncio.new_event_loop()

    def close(self) -> None:
        """Close the private event loop. Safe to call more than once."""
        if not self._loop.is_closed():
            self._loop.close()

    def recognize(self, image_bgr: np.ndarray) -> OcrResult:
        t0 = time.monotonic()
        scaled, scale = _downscale_if_needed(image_bgr, self._max_dim)
        bitmap = _bgr_to_software_bitmap(scaled)
        result = self._loop.run_until_complete(self._engine.recognize_async(bitmap))

        lines: list[OcrLine] = []
        for line in result.lines:
            words = list(line.words)
            if not words:
                continue
            boxes = [(w.bounding_rect.x, w.bounding_rect.y, w.bounding_rect.width, w.bounding_rect.height) for w in words]
            box = _scale_box(_union_boxes(boxes), scale)
            lines.append(OcrLine(text=line.text, box=box, confidence=1.0))

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return OcrResult(lines=tuple(lines), text=lines_to_text(lines), elapsed_ms=elapsed_ms, engine=self.name)


def _union_boxes(boxes: Sequence[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return (x1, y1, x2 - x1, y2 - y1)


def _bgr_to_software_bitmap(image_bgr: np.ndarray) -> Any:
    from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
    from winrt.windows.storage.streams import DataWriter

    h, w = image_bgr.shape[:2]
    bgra = np.empty((h, w, 4), dtype=np.uint8)
    bgra[:, :, :3] = image_bgr[:, :, :3]
    bgra[:, :, 3] = 255

    writer = DataWriter()
    writer.write_bytes(np.ascontiguousarray(bgra).tobytes())
    buffer = writer.detach_buffer()
    return SoftwareBitmap.create_copy_from_buffer(buffer, BitmapPixelFormat.BGRA8, w, h)


# --------------------------------------------------------------------------------- #
# RapidOCR
# --------------------------------------------------------------------------------- #


def _silence_rapidocr_logging() -> None:
    """Quiet RapidOCR's colour stdout logger (SPEC.md: "logs to stdout in colour")."""
    for name, candidate in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(candidate, logging.Logger) and name.lower().startswith("rapidocr"):
            candidate.setLevel(logging.WARNING)
            candidate.propagate = False
    rapidocr_logger = logging.getLogger("RapidOCR")
    rapidocr_logger.setLevel(logging.WARNING)
    rapidocr_logger.propagate = False


def openvino_devices() -> dict[str, str]:
    """``{device: full name}`` as OpenVINO reports them (``{}`` when it is not installed or
    cannot enumerate). E.g. ``{"CPU": "Intel(R) Core(TM) Ultra 9 285H", "GPU.0": "Intel(R)
    Arc(TM) 140T GPU (32GB) (iGPU)", "GPU.1": "NVIDIA GeForce RTX 5090 Laptop GPU (dGPU)",
    "NPU": "Intel(R) AI Boost"}``."""
    try:
        import openvino as ov
    except Exception:  # noqa: BLE001 - optional dependency
        return {}
    try:
        core = ov.Core()
        out: dict[str, str] = {}
        for dev in core.available_devices:
            try:
                out[str(dev)] = str(core.get_property(dev, "FULL_DEVICE_NAME"))
            except Exception:  # noqa: BLE001
                out[str(dev)] = str(dev)
        return out
    except Exception:  # noqa: BLE001
        log.debug("openvino device enumeration failed", exc_info=True)
        return {}


def directml_available() -> bool:
    """Whether onnxruntime's DirectML execution provider is usable in this interpreter --
    ``"DmlExecutionProvider" in onnxruntime.get_available_providers()``. ``False`` on any failure
    (onnxruntime not installed, the CPU-only wheel installed instead of ``onnxruntime-directml``,
    an odd DLL) -- never raises. Cached after the first call (module global ``_dml_available``)."""
    global _dml_available
    if _dml_available is not None:
        return _dml_available
    try:
        import onnxruntime as ort

        _dml_available = "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001 - optional dependency; absence must never be an error
        log.debug("onnxruntime DirectML probe failed", exc_info=True)
        _dml_available = False
    return _dml_available


def _pick_dml_adapter(gpus: "list[GpuInfo]") -> tuple[int, bool] | None:
    """The DirectML ``(device_id, is_uma)`` to use for ``gpus`` (hardware adapters, DXGI order),
    or ``None`` when DirectML must not be used: no adapters, or any of them is NVIDIA (that GPU
    holds Whisper and llama-server, never OCR). One adapter -> id 0. Several -> ``len - 1`` if one
    of them is UMA (onnxruntime's DML provider orders adapters by
    ``DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE``, so the integrated one is expected last -- a
    heuristic, not verified against a real multi-GPU-plus-AMD-iGPU machine) else id 0."""
    if not gpus or any(g.vendor == "nvidia" for g in gpus):
        return None
    if len(gpus) == 1:
        return 0, bool(gpus[0].uma)
    if any(g.uma for g in gpus):
        return len(gpus) - 1, True
    return 0, False


def _no_suffix_intel_gpu_is_confirmed_discrete(gpus: "list[GpuInfo] | None") -> bool:
    """Whether a hardware enumeration positively shows this machine's Intel GPU adapter(s) are ALL
    discrete (non-UMA) -- used to keep rule (a)'s pre-existing no-suffix OpenVINO-name fallback
    (:func:`resolve_ocr_device`'s docstring) from silently treating a genuinely discrete, but
    unsuffixed, Intel dGPU as the safe integrated case (which would bypass rule (a2)'s NVIDIA gate
    entirely).

    ``False`` -- the old, parity-preserving default -- whenever :func:`exocortex.gpu.detect_gpus`
    (or the injected ``gpus``) has no Intel entry at all: an empty/failed enumeration, or one that
    simply did not see an Intel adapter, proves nothing about the OpenVINO device it is being
    cross-checked against, so it never withholds rule (a)'s fallback from a real Iris-Xe-style
    laptop whose DXGI enumeration happened to fail or whose adapter this cannot correlate to --
    that case stays byte-identical to main. Only when every Intel adapter found is discrete does
    this return ``True``."""
    from exocortex.gpu import detect_gpus

    adapters = detect_gpus() if gpus is None else gpus
    intel_adapters = [g for g in adapters if g.vendor == "intel"]
    return bool(intel_adapters) and all(not g.uma for g in intel_adapters)


def resolve_ocr_device(
    requested: str = "auto",
    devices: dict[str, str] | None = None,
    *,
    gpus: "list[GpuInfo] | None" = None,
    dml: bool | None = None,
) -> tuple[str, str]:
    """Map a ``[screen].ocr_device`` value to ``(kind, device_spec)``.

    ``kind`` is ``"igpu"``, ``"gpu"``, ``"npu"`` or ``"cpu"``. ``device_spec`` is one of: an
    OpenVINO device to compile for (``"GPU.0"``, ``"NPU"``, ``"CPU"``), ``"dml:<id>"`` for
    onnxruntime's DirectML execution provider on adapter ``<id>``, or ``""`` for plain
    onnxruntime on the CPU.

    For ``"igpu"`` (and the igpu step of ``"auto"``), in order:

    (a) an OpenVINO ``GPU`` device whose name says Intel, and either says ``(iGPU)`` or does not
        say ``(dGPU)`` -- UNCHANGED for Intel hardware. A non-Intel OpenVINO ``GPU`` name is
        deliberately excluded here: measured 2026-09-25 on a Ryzen AI MAX+ 395 / Radeon 8060S
        (OpenVINO 2026.4.0 sees ``GPU`` "gfx1151 (iGPU)"), compiling through it crashes the
        interpreter (see the module docstring) -- so it falls through to (b) instead.
    (a2) else, an OpenVINO ``GPU`` device whose name says Intel AND ``(dGPU)`` (added 2026-09-26
        for Intel Arc discrete GPUs, e.g. "Intel(R) Arc(TM) B580 Graphics (dGPU)" -- no Intel
        hardware to verify live), used with ``kind`` ``"gpu"`` IF ``gpus`` (default
        :func:`exocortex.gpu.detect_gpus`, overridable here too) contains no NVIDIA adapter. Rule
        (a) already excludes an Intel dGPU (that suffix means "this is the laptop's non-OCR GPU"
        on the NVIDIA machines it was written for); (a2) only reopens it once ``gpus`` proves that
        rule's premise false -- no NVIDIA adapter shares this machine, so the Intel card IS the
        machine's GPU (an Intel-only desktop, or an Arc dGPU next to a non-Intel iGPU) and is not
        shadowing anything. It never fires when an Intel iGPU was already found by (a): a laptop
        with both (e.g. Arc 140V iGPU + an Arc Pro dGPU) keeps using its iGPU, exactly as before
        this rule existed. An NVIDIA adapter anywhere in ``gpus`` -- even alongside the Arc card --
        blocks (a2) exactly as it blocks the DirectML step (b) below: OCR still never touches a
        machine's NVIDIA GPU. An EMPTY or failed hardware enumeration (:func:`exocortex.gpu.detect_gpus`
        returns ``[]`` on any DXGI error -- advisory, never raises) is inconclusive, not proof there
        is no NVIDIA adapter, so it blocks (a2) too, exactly like :func:`_pick_dml_adapter`'s own
        ``not gpus`` guard for the DirectML step.
    (a, no-suffix cross-check) rule (a)'s no-suffix branch (an OpenVINO Intel ``GPU`` name with
        neither ``(iGPU)`` nor ``(dGPU)`` -- older OpenVINO / some single-GPU systems) predates (a2)
        and, on its own, would keep bypassing (a2)'s NVIDIA gate for a genuinely discrete Intel card
        whose name happens to omit the ``(dGPU)`` tag. So that branch is cross-checked against
        ``gpus``: when every Intel adapter :func:`exocortex.gpu.detect_gpus` reports is discrete
        (non-UMA), the no-suffix name is treated as NOT proven safe and is left for (a2) / (b) to
        route (and gate) instead. An empty adapter list, or one with no Intel entry at all, proves
        nothing either way, so the no-suffix branch keeps its original, ungated behaviour then --
        this never changes a real Iris-Xe-style laptop's result, on an NVIDIA machine or otherwise.
    (b) else DirectML when ``dml`` (default :func:`directml_available`) is True and ``gpus``
        (default :func:`exocortex.gpu.detect_gpus`) yields an adapter per :func:`_pick_dml_adapter`
        (at least one hardware adapter, none of them NVIDIA). ``kind`` is ``"igpu"`` when the
        chosen adapter is UMA (integrated), else ``"gpu"``.
    (c) else cpu, as today.

    ``"npu"`` and ``"cpu"`` are unchanged. ``devices`` overrides :func:`openvino_devices`,
    ``gpus`` overrides :func:`exocortex.gpu.detect_gpus` and ``dml`` overrides
    :func:`directml_available` (all three for tests). An explicit device that ends up unavailable
    falls back to cpu with a warning.
    """
    requested = (requested or "auto").strip().lower()
    if requested not in OCR_DEVICES:
        log.warning("unknown [screen].ocr_device %r; using auto", requested)
        requested = "auto"
    devs = openvino_devices() if devices is None else devices
    igpu = next((d for d, name in devs.items()
                 if d.upper().startswith("GPU") and "intel" in name.lower() and "(iGPU)" in name), None)
    if igpu is None:
        no_suffix_candidate = next((d for d, name in devs.items()
                                     if d.upper().startswith("GPU") and "intel" in name.lower()
                                     and "(dGPU)" not in name), None)
        if no_suffix_candidate is not None and not _no_suffix_intel_gpu_is_confirmed_discrete(gpus):
            igpu = no_suffix_candidate
    npu = next((d for d in devs if d.upper().startswith("NPU")), None)
    cpu = "CPU" if any(d.upper() == "CPU" for d in devs) else ""

    def intel_dgpu_step() -> str | None:
        """Rule (a2): an Intel ``(dGPU)`` OpenVINO name, only when (a) found no Intel iGPU and no
        NVIDIA adapter shares the machine (see the docstring above)."""
        if igpu is not None:
            return None
        candidate = next((d for d, name in devs.items()
                           if d.upper().startswith("GPU") and "intel" in name.lower() and "(dGPU)" in name), None)
        if candidate is None:
            return None
        from exocortex.gpu import detect_gpus

        adapters = detect_gpus() if gpus is None else gpus
        if not adapters or any(g.vendor == "nvidia" for g in adapters):
            return None
        return candidate

    def dml_step() -> tuple[str, str] | None:
        use_dml = directml_available() if dml is None else dml
        if not use_dml:
            return None
        from exocortex.gpu import detect_gpus

        adapters = detect_gpus() if gpus is None else gpus
        picked = _pick_dml_adapter(adapters)
        if picked is None:
            return None
        device_id, uma = picked
        return ("igpu" if uma else "gpu"), f"dml:{device_id}"

    if requested in ("auto", "igpu"):
        if igpu:
            return "igpu", igpu
        intel_dgpu = intel_dgpu_step()
        if intel_dgpu:
            return "gpu", intel_dgpu
        result = dml_step()
        if result is not None:
            return result
        if requested == "igpu":
            log.warning(
                "OCR device igpu requested but no Intel integrated GPU is visible to OpenVINO and "
                "DirectML is unavailable or unsuitable (%s); using cpu",
                ", ".join(devs) or "openvino not installed",
            )
            return "cpu", cpu
        if npu:  # requested == "auto": fall further down the ladder
            return "npu", npu
        return "cpu", cpu
    if requested == "npu":
        if npu:
            return "npu", npu
        log.warning("OCR device npu requested but no NPU is visible to OpenVINO (%s); using cpu",
                    ", ".join(devs) or "openvino not installed")
        return "cpu", cpu
    return "cpu", cpu


def describe_ocr_device(kind: str, spec: str) -> str:
    """Human-readable accelerator description for the doctor / logs, e.g. ``"igpu via OpenVINO
    GPU"``, ``"gpu via OpenVINO GPU.0"`` (an Intel Arc discrete GPU, rule (a2) in
    :func:`resolve_ocr_device`), ``"igpu via DirectML device 0"``, ``"cpu via OpenVINO CPU"``,
    ``"cpu via onnxruntime"``."""
    if spec.startswith("dml:"):
        return f"{kind} via DirectML device {spec.split(':', 1)[1]}"
    if spec:
        return f"{kind} via OpenVINO {spec}"
    return f"{kind} via onnxruntime"


def npu_width_bucket(width: int) -> int:
    """The static rec-input width the NPU is compiled for, for a batch ``width`` px wide."""
    for bucket in _NPU_REC_WIDTH_BUCKETS:
        if width <= bucket:
            return bucket
    return ((width + 511) // 512) * 512


def pad_det_input(x: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Pad a det-model input ``[1, 3, H, W]`` bottom/right to multiples of :data:`_DET_PAD`
    (zeros = mid-grey after PP-OCR's normalisation). Returns ``(padded, H, W)`` so the caller can
    crop the prediction map back to the real size."""
    h, w = int(x.shape[2]), int(x.shape[3])
    hp = ((h + _DET_PAD - 1) // _DET_PAD) * _DET_PAD
    wp = ((w + _DET_PAD - 1) // _DET_PAD) * _DET_PAD
    if hp == h and wp == w:
        return x, h, w
    padded = np.zeros((x.shape[0], x.shape[1], hp, wp), dtype=np.float32)
    padded[:, :, :h, :w] = x
    return padded, h, w


def pad_static_batch(x: np.ndarray, kind: str) -> tuple[np.ndarray, int]:
    """Pad a rec/cls input batch to the NPU's static shape: batch to a multiple of
    :data:`_NPU_BATCH`, and (rec only) width to :func:`npu_width_bucket`. Returns the padded
    array and the real batch size (the output is sliced back to it). Padding is zeros = the
    mid-grey PP-OCR already pads short crops with, so it only adds trailing CTC blanks."""
    n = x.shape[0]
    nb = _NPU_BATCH if n <= _NPU_BATCH else ((n + _NPU_BATCH - 1) // _NPU_BATCH) * _NPU_BATCH
    if kind == "rec":
        wb = npu_width_bucket(int(x.shape[3]))
        padded = np.zeros((nb, x.shape[1], x.shape[2], wb), dtype=np.float32)
        padded[:n, :, :, : x.shape[3]] = x
    else:
        padded = np.zeros((nb,) + tuple(x.shape[1:]), dtype=np.float32)
        padded[:n] = x
    return padded, n


class _OpenVinoSession:
    """Drop-in for RapidOCR's inference session (``__call__(input) -> first output``), running
    one of its ONNX models through OpenVINO on ``device``.

    Two independent knobs. ``bucket`` (iGPU and NPU): inputs are padded to a small set of shapes
    -- det to multiples of :data:`_DET_PAD`, rec / cls with :func:`pad_static_batch` -- because
    both accelerators pay per distinct input shape (the NPU a compile, the GPU plugin a kernel
    JIT for every new crop width: measured 37 s for one fresh screen without bucketing).
    ``static`` (NPU only, which needs static shapes): the model is reshaped and compiled per
    shape on first use; the GPU keeps the dynamic-shape model, because a reshaped model loaded
    back from OpenVINO's on-disk cache on the GPU came back wrong (measured: 0-1 lines instead
    of 95), while the dynamic model caches correctly. All compiles / JIT kernels land in the
    cache (``cache_dir``), so they are paid once per machine. Calls are serialised with a lock
    (the warm-up runs on its own thread).
    """

    def __init__(self, core: Any, model_path: Path, device: str, *, static: bool, kind: str,
                 bucket: bool | None = None) -> None:
        self._core = core
        self._model = core.read_model(str(model_path))
        self.device = device
        self.static = static
        self.bucket = static if bucket is None else bucket
        self.kind = kind
        self._requests: dict[Any, Any] = {}
        self._lock = threading.Lock()
        self.compile_ms: list[float] = []
        if not static:
            self._requests[None] = self._compile(self._model)

    def _compile(self, model: Any) -> Any:
        import os

        config: dict[str, str] = {"PERFORMANCE_HINT": "LATENCY"}
        if self.device.upper().startswith("CPU"):
            # Leave half the cores to Whisper / diarization / llama-server's CPU side.
            config["INFERENCE_NUM_THREADS"] = str(max(2, (os.cpu_count() or 4) // 2))
        t0 = time.perf_counter()
        compiled = self._core.compile_model(model, self.device, config)
        self.compile_ms.append((time.perf_counter() - t0) * 1000.0)
        return compiled.create_infer_request()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        n = int(x.shape[0])
        det_h, det_w = int(x.shape[2]) if x.ndim == 4 else 0, int(x.shape[3]) if x.ndim == 4 else 0
        key: Any = None
        if self.bucket:
            if self.kind == "det":
                x, det_h, det_w = pad_det_input(x)
            else:
                x, n = pad_static_batch(x, self.kind)
        if self.static:
            key = tuple(int(d) for d in x.shape)
        with self._lock:
            request = self._requests.get(key)
            if request is None:
                model = self._model.clone()
                model.reshape({0: list(key)})
                request = self._compile(model)
                self._requests[key] = request
                log.info("OpenVINO %s: compiled %s for %s in %.0f ms", self.device, self.kind, key, self.compile_ms[-1])
            request.infer({0: x})
            out = request.get_output_tensor(0).data
            if self.kind == "det" and self.bucket:
                return np.array(out[:, :, :det_h, :det_w], copy=True)
            if self.bucket:
                return np.array(out[:n], copy=True)
            return np.array(out, copy=True)

    def warm_rec_buckets(self, buckets: Sequence[int] = _WARM_REC_WIDTH_BUCKETS) -> None:
        """Bucketed rec sessions only: compile / JIT (or load from the cache) the given width
        buckets now, with an all-zero batch, so the first real screen does not pay for them one
        by one."""
        if not self.bucket or self.kind != "rec":
            return
        for width in buckets:
            self(np.zeros((_NPU_BATCH, 3, 48, int(width)), dtype=np.float32))


class _OrtSession:
    """Drop-in for RapidOCR's inference session (``__call__(input) -> first output``), mirroring
    :class:`_OpenVinoSession` but running the same ONNX file through onnxruntime's DirectML
    execution provider on ``device_id`` instead of OpenVINO.

    DML has no on-disk kernel cache like OpenVINO's ``CACHE_DIR``: it JIT-compiles (recompiles)
    its graph for every new input shape it meets, in-process, for the life of the session -- so
    the SAME bucketing as :class:`_OpenVinoSession` (:func:`pad_det_input` for det,
    :func:`pad_static_batch` for rec/cls) matters here too, to keep the number of distinct shapes
    -- and recompiles -- small. ``enable_mem_pattern=False`` and ``ORT_SEQUENTIAL`` execution mode
    are required by the DML execution provider (onnxruntime's own docs); graph optimisation stays
    at ``ORT_ENABLE_ALL``. Calls are serialised with a lock, like :class:`_OpenVinoSession`, and
    the compile-free cost of each new shape's first call is kept in ``first_call_ms`` for logging.
    """

    def __init__(self, ort_module: Any, model_path: Path, device_id: int, *, kind: str) -> None:
        self._ort = ort_module
        self.device_id = int(device_id)
        self.kind = kind
        opts = ort_module.SessionOptions()
        opts.enable_mem_pattern = False
        opts.execution_mode = ort_module.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [("DmlExecutionProvider", {"device_id": self.device_id}), "CPUExecutionProvider"]
        self._session = ort_module.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._lock = threading.Lock()
        self.first_call_ms: dict[Any, float] = {}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        n = int(x.shape[0])
        det_h, det_w = (int(x.shape[2]), int(x.shape[3])) if x.ndim == 4 else (0, 0)
        if self.kind == "det":
            x, det_h, det_w = pad_det_input(x)
        else:
            x, n = pad_static_batch(x, self.kind)
        key = tuple(int(d) for d in x.shape)
        with self._lock:
            t0 = time.perf_counter() if key not in self.first_call_ms else None
            out = self._session.run([self._output_name], {self._input_name: x})[0]
            if t0 is not None:
                self.first_call_ms[key] = (time.perf_counter() - t0) * 1000.0
                log.info("DirectML device %d: first call for %s shape %s took %.0f ms",
                          self.device_id, self.kind, key, self.first_call_ms[key])
            if self.kind == "det":
                return np.array(out[:, :, :det_h, :det_w], copy=True)
            return np.array(out[:n], copy=True)

    def warm_rec_buckets(self, buckets: Sequence[int] = _WARM_REC_WIDTH_BUCKETS) -> None:
        """Rec sessions only: run the given width buckets now (an all-zero batch), so the first
        real screen does not pay for DML's per-new-shape recompile one bucket at a time."""
        if self.kind != "rec":
            return
        for width in buckets:
            self(np.zeros((_NPU_BATCH, 3, 48, int(width)), dtype=np.float32))


def _rapidocr_model_file(stage: Any, default_name: str) -> Path:
    """The ONNX file RapidOCR loaded for ``stage`` (its onnxruntime session remembers the path);
    falls back to the model bundled with the package."""
    session = getattr(getattr(stage, "session", None), "session", None)
    path = getattr(session, "_model_path", None)
    if path:
        return Path(path)
    import rapidocr

    return Path(rapidocr.__file__).resolve().parent / "models" / default_name


class RapidOcr:
    """RapidOCR (PP-OCRv6 small det + rec) on the CPU via onnxruntime, or -- through OpenVINO or
    onnxruntime's DirectML execution provider -- on an iGPU / NPU / dGPU (``device``, see
    :func:`resolve_ocr_device`).

    The 180-degree text-orientation classifier is off: a screen is never upside down and it cost
    ~150 ms per pass. With OpenVINO, the det and rec sessions RapidOCR built are swapped for
    :class:`_OpenVinoSession`s compiled from the same ONNX files -- static shape buckets on the
    iGPU and the NPU, dynamic shapes on the CPU (``cache_dir`` holds the compiled kernels). With
    DirectML (a non-Intel GPU, see the module docstring), they are swapped for
    :class:`_OrtSession`s instead, on the same ONNX files. Either accelerator's failure falls back
    to onnxruntime on the CPU with a warning, never an exception -- whether it fails while being
    attached (construction: ``__init__`` catches ``_attach_directml``/``_attach_openvino``) or
    later, mid-session (a single ``recognize()``/``warm()`` call demotes ``self`` to the CPU from
    then on: see ``_recognize``). ``device`` defaults to the CPU:
    the accelerators are an explicit choice (``[screen].ocr_device``, which the app always passes)
    because their first use compiles kernels for seconds. Must be created BEFORE any
    :class:`WindowsOcr` in the process (module docstring).
    """

    name = "rapidocr"
    device = "cpu"
    ov_device = ""
    #: Class-level fallback so an instance built without ``__init__`` (tests swap in a fake
    #: ``_engine``) still serialises ``recognize``; ``__init__`` gives each instance its own.
    _lock: threading.Lock = threading.Lock()

    def __init__(self, device: str = "cpu", cache_dir: Path | str | None = None) -> None:
        if _windows_ocr_created:
            raise OcrUnavailable(
                "RapidOCR cannot be created after Windows OCR in the same process (it crashes the "
                "interpreter); create the RapidOCR engine first"
            )
        _silence_rapidocr_logging()
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise OcrUnavailable(f"RapidOCR is not installed: {exc}") from exc
        try:
            # cfg.Global.log_level is applied at construction time (RapidOCR.__init__ does
            # logger.setLevel(cfg.Global.log_level.upper()) before doing anything else), so
            # this beats the default INFO level even for the first call.
            self._engine = RapidOCR(params={"Global.log_level": "warning", "Global.use_cls": False})
        except Exception as exc:  # pragma: no cover - depends on local model files
            raise OcrUnavailable(f"RapidOCR failed to initialise: {exc}") from exc
        _silence_rapidocr_logging()
        self.device, self.ov_device = resolve_ocr_device(device)
        self.compile_ms: list[float] = []
        self._lock = threading.Lock()   # recognize() and warm() (own thread) never overlap
        if self.ov_device.startswith("dml:"):
            try:
                self._attach_directml(int(self.ov_device.split(":", 1)[1]))
            except Exception as exc:  # noqa: BLE001 - never lose OCR over an accelerator problem
                log.warning("RapidOCR: DirectML on %s failed (%s); using onnxruntime on the CPU", self.ov_device, exc)
                self.device, self.ov_device = "cpu", ""
        elif self.ov_device:
            try:
                self._attach_openvino(self.ov_device, Path(cache_dir) if cache_dir else None)
            except Exception as exc:  # noqa: BLE001 - never lose OCR over an accelerator problem
                log.warning("RapidOCR: OpenVINO on %s failed (%s); using onnxruntime on the CPU", self.ov_device, exc)
                self.device, self.ov_device = "cpu", ""
        log.info("RapidOCR ready: %s", describe_ocr_device(self.device, self.ov_device))

    @property
    def accelerator(self) -> str:
        """Read-only alias for :attr:`ov_device` (an OpenVINO device name, ``"dml:<id>"``, or
        ``""``) -- the name reflects the field's OpenVINO-only origin; this one does not."""
        return self.ov_device

    def _attach_directml(self, device_id: int) -> None:
        import onnxruntime as ort

        det = _OrtSession(ort, _rapidocr_model_file(self._engine.text_det, "PP-OCRv6_det_small.onnx"),
                          device_id, kind="det")
        rec = _OrtSession(ort, _rapidocr_model_file(self._engine.text_rec, "PP-OCRv6_rec_small.onnx"),
                          device_id, kind="rec")
        # Saved so a later inference-time accelerator failure (_recognize) can swap back to the
        # plain onnxruntime-on-CPU sessions RapidOCR itself built, instead of only guarding
        # construction (this method raising) as the class docstring's "never an exception"
        # promise otherwise only half-keeps.
        self._cpu_sessions = (self._engine.text_det.session, self._engine.text_rec.session)
        self._engine.text_det.session = det
        self._engine.text_rec.session = rec
        self._sessions = (det, rec)

    def _attach_openvino(self, ov_device: str, cache_dir: Path | None) -> None:
        import openvino as ov

        core = ov.Core()
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(cache_dir)})
        static = ov_device.upper().startswith("NPU")
        bucket = ov_device.upper().startswith(("NPU", "GPU"))
        det = _OpenVinoSession(core, _rapidocr_model_file(self._engine.text_det, "PP-OCRv6_det_small.onnx"),
                               ov_device, static=static, bucket=bucket, kind="det")
        rec = _OpenVinoSession(core, _rapidocr_model_file(self._engine.text_rec, "PP-OCRv6_rec_small.onnx"),
                               ov_device, static=static, bucket=bucket, kind="rec")
        # See the matching comment in _attach_directml: kept so _recognize can fall back to the
        # CPU at inference time too, not just when attaching the accelerator raises.
        self._cpu_sessions = (self._engine.text_det.session, self._engine.text_rec.session)
        self._engine.text_det.session = det
        self._engine.text_rec.session = rec
        self._sessions = (det, rec)
        self.compile_ms = [ms for s in self._sessions for ms in s.compile_ms]

    def warm(self, width: int, height: int) -> float:
        """Compile the accelerator's kernels now, on the caller's thread, instead of on the first
        real read: the rec model for the common crop-width buckets (:data:`_WARM_REC_WIDTH_BUCKETS`)
        and, through one recognition of a synthetic ``width``x``height`` page of text, the det model
        for this screen size. Returns the elapsed ms. A no-op-ish 50 ms on onnxruntime."""
        from PIL import Image, ImageDraw

        t_start = time.perf_counter()
        for session in getattr(self, "_sessions", ()):
            try:
                session.warm_rec_buckets()
            except Exception:  # noqa: BLE001 - warm-up is an optimisation only
                log.debug("RapidOCR: rec bucket warm-up failed", exc_info=True)
        w, h = max(64, int(width)), max(64, int(height))
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)
        try:
            from PIL import ImageFont

            font = ImageFont.truetype("arial.ttf", max(12, h // 40))
        except Exception:  # noqa: BLE001
            font = None
        samples = ("def solve(nums):", "for i in range(n): total += nums[i]",
                   "Given an array of integers, return the indices of the two numbers that add up to the target value.",
                   "return max(best, cur)", "Constraints: 2 <= nums.length <= 10^4, -10^9 <= nums[i] <= 10^9, one valid answer",
                   "x = 1")
        step = max(16, h // (len(samples) + 2))
        for i, text in enumerate(samples):
            draw.text((20, 20 + i * step), text, fill="black", font=font)
        bgr = np.ascontiguousarray(np.array(img)[:, :, ::-1])
        try:
            self.recognize(bgr)
        except Exception:  # noqa: BLE001 - warm-up is an optimisation only, never fatal
            log.debug("RapidOCR: det warm-up recognize() failed", exc_info=True)
        elapsed = (time.perf_counter() - t_start) * 1000.0
        log.info("RapidOCR warm-up on %s: %.0f ms (%dx%d)", self.device, elapsed, w, h)
        return elapsed

    def recognize(self, image_bgr: np.ndarray) -> OcrResult:
        with self._lock:
            return self._recognize(image_bgr)

    def _recognize(self, image_bgr: np.ndarray) -> OcrResult:
        t0 = time.monotonic()
        scaled, scale = _downscale_if_needed(image_bgr, _RAPIDOCR_MAX_SIDE)
        try:
            out = self._engine(scaled)
        except Exception as exc:  # noqa: BLE001 - never lose OCR over an accelerator problem
            cpu_sessions = getattr(self, "_cpu_sessions", None)
            if self.device == "cpu" or not cpu_sessions:
                raise
            log.warning(
                "RapidOCR: accelerator inference failed (%s); falling back to onnxruntime on the CPU",
                exc,
            )
            self._engine.text_det.session, self._engine.text_rec.session = cpu_sessions
            self.device, self.ov_device = "cpu", ""
            out = self._engine(scaled)

        lines: list[OcrLine] = []
        texts = out.txts or ()
        boxes = out.boxes if out.boxes is not None else [None] * len(texts)
        scores = out.scores if out.scores is not None else [1.0] * len(texts)
        for text, box_pts, score in zip(texts, boxes, scores):
            if not text or not text.strip() or box_pts is None:
                continue
            pts = np.asarray(box_pts, dtype=np.float32)
            x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
            box = _scale_box((x1, y1, x2 - x1, y2 - y1), scale)
            lines.append(OcrLine(text=text, box=box, confidence=float(score)))

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return OcrResult(lines=tuple(lines), text=lines_to_text(lines), elapsed_ms=elapsed_ms, engine=self.name)


# --------------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------------- #


def engine_label(engine: Any) -> str:
    """``"windows"`` / ``"rapidocr/igpu"`` -- the engine's name plus its device when it has one
    other than the CPU (for status lines and :attr:`~exocortex.bus.ScreenTextEvent.engine`)."""
    name = str(getattr(engine, "name", "") or "ocr")
    device = str(getattr(engine, "device", "") or "")
    return f"{name}/{device}" if device and device != "cpu" else name


def create_engine(name: str = "auto", device: str = "cpu", cache_dir: Path | str | None = None) -> OcrEngine:
    """Build an :class:`OcrEngine`. ``"auto"`` tries :class:`WindowsOcr` then :class:`RapidOcr`.
    ``device`` / ``cache_dir`` only concern RapidOCR (:func:`resolve_ocr_device`; the app passes
    ``[screen].ocr_device`` and ``<models_dir>/ov_cache``; the defaults build the CPU engine
    exactly as before, so ``RapidOcr()`` stays a valid construction).

    Raises :class:`OcrUnavailable` if the requested engine (or, for ``"auto"``, neither
    engine) can be created.
    """

    def _rapidocr() -> OcrEngine:
        if device == "cpu" and cache_dir is None:
            return RapidOcr()
        return RapidOcr(device, cache_dir)

    if name == "windows":
        return WindowsOcr()
    if name == "rapidocr":
        return _rapidocr()
    if name != "auto":
        raise OcrUnavailable(f"unknown OCR engine {name!r}")

    try:
        return WindowsOcr()
    except OcrUnavailable as exc:
        log.info("Windows OCR unavailable (%s); falling back to RapidOCR", exc)
    try:
        return _rapidocr()
    except OcrUnavailable as exc:
        raise OcrUnavailable(f"no OCR engine available (tried windows, rapidocr): {exc}") from exc
