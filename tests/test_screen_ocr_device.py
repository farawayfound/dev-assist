"""Offline tests for OCR acceleration (exocortex.screen.ocr) and the accurate/sampling engine
split in the screen reader (exocortex.screen.reader), added for hands-free mode.

Covers, without ever touching real OpenVINO devices, real DirectML/onnxruntime, real WinRT, real
GPU hardware or a real rapidocr install:

- ``resolve_ocr_device``: auto/explicit device selection against an injected ``devices`` dict
  (iGPU-vs-dGPU-vs-plain-Intel-GPU naming, npu/cpu fallbacks, unknown values, missing-device
  warnings), PLUS (added 2026-09-25 for the AMD Radeon 8060S / DirectML support) the DirectML step
  against injected ``gpus`` lists and a ``dml`` flag: a non-Intel OpenVINO GPU name is excluded
  from the OpenVINO step, DirectML is chosen only when no NVIDIA adapter is present, the device-id
  heuristic (0 with one adapter, ``len - 1`` when a UMA one exists among several, else 0), and an
  Intel OpenVINO iGPU still wins over DirectML when both are available.
- ``directml_available``: True/False from an injected fake ``onnxruntime`` module's
  ``get_available_providers()``, False when the import fails, and that the result is cached.
- ``describe_ocr_device``: the OpenVINO / DirectML / onnxruntime description strings.
- ``openvino_devices``: returns ``{}`` when ``openvino`` cannot be imported.
- ``npu_width_bucket`` / ``pad_static_batch`` / ``pad_det_input``: the static-shape padding math
  the NPU (and now DirectML) paths depend on.
- ``engine_label``: the "name" vs "name/device" formatting.
- The Windows-OCR-first crash guard: a :class:`~exocortex.screen.ocr.RapidOcr` created after a
  :class:`~exocortex.screen.ocr.WindowsOcr` in the same process must refuse immediately (before
  ever importing rapidocr), and ``WindowsOcr.__init__`` must set that guard flag.
- ``create_engine``: device/cache_dir passthrough to RapidOcr, and that "auto" still prefers
  Windows OCR.
- ``_OpenVinoSession``: det-output cropping back to the real frame size, static (NPU) compiling
  once per (batch, width-bucket) shape, dynamic (CPU/iGPU) compiling once at construction, and
  the CPU-only INFERENCE_NUM_THREADS config.
- ``_OrtSession`` (DirectML): the same det-crop / rec-slice bucketing as ``_OpenVinoSession`` over
  a fake ``onnxruntime.InferenceSession``, and that it is built with the DML execution provider,
  ``device_id`` and the required session options (``enable_mem_pattern=False``,
  ``ORT_SEQUENTIAL``).
- ``RapidOcr.__init__``'s dispatch on the resolved device spec (a fake ``rapidocr`` module in
  ``sys.modules``): a ``"dml:<id>"`` spec calls ``_attach_directml``, an OpenVINO device name
  calls ``_attach_openvino``, ``""`` calls neither, and either attach method raising falls back to
  plain CPU with a warning -- never an exception.
- The screen reader's accurate-engine-built-first rule (:mod:`exocortex.screen.reader`): the
  accurate engine is created before the sampling engine, ``capture_now(accurate=True)`` routes
  through it and marks the event, a same-named sampling engine reuses the accurate instance
  instead of creating a second one, a failed accurate engine falls back to sampling silently,
  ``warm()`` is called once per engine after creation, and ``stop()`` closes both engines.

Fakes only: a fake ``devices`` dict stands in for ``openvino_devices()``, a fake ``core`` object
(``read_model``/``compile_model``) stands in for ``openvino.Core()``, a fake ``onnxruntime``
module stands in for onnxruntime's ``InferenceSession`` / provider list, a fake ``rapidocr``
module stands in for the real package in the dispatch tests, a fake ``winrt.windows.media.ocr``
module is injected via ``sys.modules`` for the WindowsOcr guard test, synthetic
:class:`~exocortex.gpu.GpuInfo` instances stand in for :func:`~exocortex.gpu.detect_gpus`, and the
reader tests reuse ``ScriptedGrabber``/``ScriptedEngine``/``FakeClock``/``make_reader`` from
tests/test_screen_reader.py against a real ``EventBus``.
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from exocortex.bus import EventBus, ScreenTextEvent
from exocortex.screen import ocr
from tests.test_screen_reader import (
    _BLACK,
    _WHITE,
    _cfg,
    _drain_until,
    make_reader,  # noqa: F401 - imported for pytest fixture discovery
)
from tests.test_screen_reader import ScriptedEngine, ScriptedGrabber

# --------------------------------------------------------------------------------- #
# resolve_ocr_device
# --------------------------------------------------------------------------------- #

_IGPU_AND_NPU = {
    "CPU": "Intel(R) Core(TM) Ultra 9 285H",
    "GPU.0": "Intel(R) Arc(TM) 140T GPU (32GB) (iGPU)",
    "NPU": "Intel(R) AI Boost",
}
_NPU_ONLY = {"CPU": "Intel(R) Core(TM) Ultra 9 285H", "NPU": "Intel(R) AI Boost"}
_CPU_ONLY = {"CPU": "Intel(R) Core(TM) Ultra 9 285H"}
_DGPU_ONLY = {"CPU": "Intel(R) Core(TM) Ultra 9 285H", "GPU.0": "NVIDIA GeForce RTX 5090 Laptop GPU (dGPU)"}
_INTEL_GPU_NO_SUFFIX = {"GPU.0": "Intel(R) Iris Xe Graphics"}
_INTEL_ARC_DGPU_ONLY = {"CPU": "13th Gen Intel(R) Core(TM) i9-14900K", "GPU.0": "Intel(R) Arc(TM) B580 Graphics (dGPU)"}
_INTEL_IGPU_AND_ARC_DGPU = {
    "CPU": "Intel(R) Core(TM) Ultra 9 285H",
    "GPU.0": "Intel(R) Arc(TM) 140T GPU (32GB) (iGPU)",
    "GPU.1": "Intel(R) Arc(TM) Pro B70 Graphics (dGPU)",
}


@pytest.mark.parametrize(
    "requested, devices, expected",
    [
        ("auto", _IGPU_AND_NPU, ("igpu", "GPU.0")),
        ("auto", _NPU_ONLY, ("npu", "NPU")),
        ("auto", _CPU_ONLY, ("cpu", "CPU")),
        ("auto", {}, ("cpu", "")),
        ("auto", _DGPU_ONLY, ("cpu", "CPU")),  # a dGPU-only GPU.0 is never picked as igpu
        ("auto", _INTEL_GPU_NO_SUFFIX, ("igpu", "GPU.0")),  # Intel GPU without "(iGPU)" still accepted
    ],
)
def test_resolve_ocr_device_auto_with_explicit_devices(requested, devices, expected) -> None:
    # dml=False: these rows only exercise the OpenVINO-devices-dict ladder (the original
    # behaviour, pre-DirectML) and must stay hermetic -- without this, a real machine that
    # happens to have onnxruntime-directml installed and a real non-NVIDIA GPU would fall
    # through to the DirectML step instead of "cpu" for _NPU_ONLY / _CPU_ONLY / {} / _DGPU_ONLY.
    assert ocr.resolve_ocr_device(requested, devices, dml=False) == expected


def test_resolve_ocr_device_explicit_igpu_missing_falls_back_to_cpu_with_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        result = ocr.resolve_ocr_device("igpu", _CPU_ONLY, dml=False)
    assert result == ("cpu", "CPU")
    assert any("igpu" in r.message.lower() for r in caplog.records)


def test_resolve_ocr_device_explicit_npu_missing_falls_back_to_cpu_with_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        result = ocr.resolve_ocr_device("npu", _CPU_ONLY)
    assert result == ("cpu", "CPU")
    assert any("npu" in r.message.lower() for r in caplog.records)


def test_resolve_ocr_device_unknown_value_is_treated_as_auto(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        result = ocr.resolve_ocr_device("bogus", _IGPU_AND_NPU, dml=False)
    assert result == ocr.resolve_ocr_device("auto", _IGPU_AND_NPU, dml=False)
    assert any("unknown" in r.message.lower() for r in caplog.records)


# --------------------------------------------------------------------------------- #
# resolve_ocr_device: DirectML step (non-Intel GPUs, incl. this AMD Radeon 8060S box)
# --------------------------------------------------------------------------------- #

_AMD_IGPU_ONLY = {"CPU": "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S", "GPU": "gfx1151 (iGPU)"}


def _gpu(vendor: str, uma: bool, index: int = 0) -> Any:
    from exocortex.gpu import GpuInfo

    return GpuInfo(
        index=index, name=f"fake {vendor} adapter {index}", vendor=vendor, vendor_id=0,
        device_id=0, dedicated_mb=1024, shared_mb=2048 if uma else 0, uma=uma, luid=f"luid{index}",
    )


def test_resolve_ocr_device_amd_openvino_gpu_is_excluded_from_rule_a() -> None:
    """A non-Intel OpenVINO GPU name ("gfx1151 (iGPU)", no "intel" in it) must never be picked by
    rule (a) -- see the module docstring's measured OpenVINO-on-AMD crash -- regardless of dml."""
    assert ocr.resolve_ocr_device("auto", _AMD_IGPU_ONLY, dml=False) == ("cpu", "CPU")
    assert ocr.resolve_ocr_device("igpu", _AMD_IGPU_ONLY, dml=False) == ("cpu", "CPU")


def test_resolve_ocr_device_amd_uma_only_with_dml_uses_directml_device_0() -> None:
    gpus = [_gpu("amd", uma=True)]
    assert ocr.resolve_ocr_device("auto", _AMD_IGPU_ONLY, gpus=gpus, dml=True) == ("igpu", "dml:0")
    assert ocr.resolve_ocr_device("igpu", _AMD_IGPU_ONLY, gpus=gpus, dml=True) == ("igpu", "dml:0")


def test_resolve_ocr_device_amd_uma_plus_nvidia_discrete_never_uses_dml() -> None:
    """OCR must never land on the NVIDIA GPU (it holds Whisper and llama-server): an NVIDIA
    adapter anywhere in ``gpus`` disables the DirectML step entirely, even with an AMD UMA
    adapter also present and dml=True."""
    gpus = [_gpu("nvidia", uma=False), _gpu("amd", uma=True, index=1)]
    assert ocr.resolve_ocr_device("auto", _AMD_IGPU_ONLY, gpus=gpus, dml=True) == ("cpu", "CPU")


def test_resolve_ocr_device_intel_igpu_present_uses_openvino_even_with_dml_true() -> None:
    """Rule (a) still wins over (b): an Intel OpenVINO iGPU is used exactly as before, whether or
    not DirectML is also available."""
    gpus = [_gpu("intel", uma=True)]
    assert ocr.resolve_ocr_device("auto", _IGPU_AND_NPU, gpus=gpus, dml=True) == ("igpu", "GPU.0")


def test_resolve_ocr_device_no_openvino_at_all_plus_amd_and_dml_uses_directml() -> None:
    gpus = [_gpu("amd", uma=True)]
    assert ocr.resolve_ocr_device("auto", {}, gpus=gpus, dml=True) == ("igpu", "dml:0")


def test_resolve_ocr_device_two_amd_adapters_with_a_uma_one_picks_last_index() -> None:
    gpus = [_gpu("amd", uma=False, index=0), _gpu("amd", uma=True, index=1)]
    assert ocr.resolve_ocr_device("auto", {}, gpus=gpus, dml=True) == ("igpu", "dml:1")


def test_resolve_ocr_device_two_amd_adapters_neither_uma_picks_index_0_kind_gpu() -> None:
    gpus = [_gpu("amd", uma=False, index=0), _gpu("amd", uma=False, index=1)]
    assert ocr.resolve_ocr_device("auto", {}, gpus=gpus, dml=True) == ("gpu", "dml:0")


def test_resolve_ocr_device_single_amd_dgpu_no_uma_picks_index_0_kind_gpu() -> None:
    gpus = [_gpu("amd", uma=False)]
    assert ocr.resolve_ocr_device("auto", {}, gpus=gpus, dml=True) == ("gpu", "dml:0")


def test_resolve_ocr_device_dml_true_but_no_gpus_falls_back_to_cpu() -> None:
    assert ocr.resolve_ocr_device("auto", {}, gpus=[], dml=True) == ("cpu", "")


def test_resolve_ocr_device_explicit_igpu_dml_unavailable_falls_back_to_cpu_with_warning(caplog) -> None:
    gpus = [_gpu("amd", uma=True)]
    with caplog.at_level(logging.WARNING):
        result = ocr.resolve_ocr_device("igpu", _AMD_IGPU_ONLY, gpus=gpus, dml=False)
    assert result == ("cpu", "CPU")
    assert any("igpu" in r.message.lower() for r in caplog.records)


# --------------------------------------------------------------------------------- #
# directml_available
# --------------------------------------------------------------------------------- #


@pytest.fixture
def clean_dml_cache(monkeypatch):
    monkeypatch.setattr(ocr, "_dml_available", None)


def test_directml_available_true_when_provider_listed(monkeypatch, clean_dml_cache) -> None:
    fake_ort = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider", "DmlExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    assert ocr.directml_available() is True


def test_directml_available_false_when_provider_missing(monkeypatch, clean_dml_cache) -> None:
    fake_ort = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    assert ocr.directml_available() is False


def test_directml_available_false_when_onnxruntime_not_importable(monkeypatch, clean_dml_cache) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert ocr.directml_available() is False


def test_directml_available_is_cached(monkeypatch, clean_dml_cache) -> None:
    calls = {"n": 0}

    def get_providers():
        calls["n"] += 1
        return ["DmlExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(get_available_providers=get_providers))
    assert ocr.directml_available() is True
    assert ocr.directml_available() is True
    assert calls["n"] == 1


# --------------------------------------------------------------------------------- #
# describe_ocr_device
# --------------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kind, spec, expected",
    [
        ("igpu", "GPU.0", "igpu via OpenVINO GPU.0"),
        ("igpu", "dml:0", "igpu via DirectML device 0"),
        ("gpu", "dml:1", "gpu via DirectML device 1"),
        ("npu", "NPU", "npu via OpenVINO NPU"),
        ("cpu", "CPU", "cpu via OpenVINO CPU"),
        ("cpu", "", "cpu via onnxruntime"),
    ],
)
def test_describe_ocr_device(kind, spec, expected) -> None:
    assert ocr.describe_ocr_device(kind, spec) == expected


# --------------------------------------------------------------------------------- #
# openvino_devices
# --------------------------------------------------------------------------------- #


def test_openvino_devices_returns_empty_dict_when_openvino_not_importable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openvino", None)
    assert ocr.openvino_devices() == {}


# --------------------------------------------------------------------------------- #
# npu_width_bucket / pad_static_batch / pad_det_input
# --------------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "width, expected_bucket",
    [(300, 320), (320, 320), (321, 480), (4096, 4096), (4200, 4608)],
)
def test_npu_width_bucket(width, expected_bucket) -> None:
    assert ocr.npu_width_bucket(width) == expected_bucket


def test_pad_static_batch_rec_pads_batch_to_six_and_width_to_bucket_returns_real_n() -> None:
    x = np.random.rand(3, 3, 48, 300).astype(np.float32)
    padded, n = ocr.pad_static_batch(x, "rec")
    assert n == 3
    assert padded.shape == (6, 3, 48, ocr.npu_width_bucket(300))
    assert np.array_equal(padded[:3, :, :, :300], x)
    assert not np.any(padded[3:])


def test_pad_static_batch_cls_pads_batch_only() -> None:
    x = np.random.rand(3, 3, 48, 48).astype(np.float32)
    padded, n = ocr.pad_static_batch(x, "cls")
    assert n == 3
    assert padded.shape == (6, 3, 48, 48)  # width untouched for cls
    assert np.array_equal(padded[:3], x)


@pytest.mark.parametrize("kind", ["rec", "cls"])
def test_pad_static_batch_batch_of_seven_rounds_up_to_twelve(kind) -> None:
    width = 300 if kind == "rec" else 48
    x = np.zeros((7, 3, 48, width), dtype=np.float32)
    padded, n = ocr.pad_static_batch(x, kind)
    assert n == 7
    assert padded.shape[0] == 12


def test_pad_det_input_pads_bottom_right_to_multiples_of_256_and_returns_original_hw() -> None:
    x = np.random.rand(1, 3, 100, 150).astype(np.float32)
    padded, h, w = ocr.pad_det_input(x)
    assert padded.shape == (1, 3, 256, 256)
    assert (h, w) == (100, 150)
    assert np.array_equal(padded[:, :, :100, :150], x)
    assert not np.any(padded[:, :, 100:, :])
    assert not np.any(padded[:, :, :, 150:])


def test_pad_det_input_returns_input_unchanged_when_already_a_multiple_of_256() -> None:
    x = np.zeros((1, 3, 256, 512), dtype=np.float32)
    padded, h, w = ocr.pad_det_input(x)
    assert padded is x  # no copy made when nothing needs padding
    assert (h, w) == (256, 512)


# --------------------------------------------------------------------------------- #
# engine_label
# --------------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "engine, expected",
    [
        (SimpleNamespace(name="windows", device="cpu"), "windows"),
        (SimpleNamespace(name="rapidocr", device="igpu"), "rapidocr/igpu"),
        (SimpleNamespace(name="rapidocr", device="cpu"), "rapidocr"),
    ],
)
def test_engine_label(engine, expected) -> None:
    assert ocr.engine_label(engine) == expected


# --------------------------------------------------------------------------------- #
# Windows-OCR-first crash guard (module docstring's process-order rule)
# --------------------------------------------------------------------------------- #


@pytest.fixture
def clean_windows_ocr_flag(monkeypatch):
    """Pins ocr._windows_ocr_created to False for the test and restores whatever it was
    before, even though WindowsOcr.__init__ mutates that module global directly (bypassing
    monkeypatch's own tracking) -- so tests touching it stay independent of each other."""
    monkeypatch.setattr(ocr, "_windows_ocr_created", False)


def test_rapidocr_refuses_after_windows_ocr_before_importing_rapidocr(monkeypatch, clean_windows_ocr_flag) -> None:
    monkeypatch.setattr(ocr, "_windows_ocr_created", True)
    # Poison the rapidocr import: if RapidOcr.__init__ reached "from rapidocr import RapidOCR"
    # before the guard, this would raise a *different* OcrUnavailable ("not installed"), so the
    # message assertion below proves the guard fired first -- no real rapidocr needed either way.
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    with pytest.raises(ocr.OcrUnavailable, match="(?i)windows ocr"):
        ocr.RapidOcr()


def test_windows_ocr_init_sets_the_process_order_guard_flag(monkeypatch, clean_windows_ocr_flag) -> None:
    class _FakeWinOcrEngineCls:
        max_image_dimension = 10000

        @staticmethod
        def try_create_from_user_profile_languages():
            return SimpleNamespace()  # anything not None = "an engine was created"

    monkeypatch.setitem(sys.modules, "winrt.windows.media.ocr", SimpleNamespace(OcrEngine=_FakeWinOcrEngineCls))

    assert ocr._windows_ocr_created is False
    engine = ocr.WindowsOcr()
    try:
        assert ocr._windows_ocr_created is True
    finally:
        engine.close()


# --------------------------------------------------------------------------------- #
# create_engine: device/cache_dir passthrough, "auto" still prefers Windows OCR
# --------------------------------------------------------------------------------- #


def test_create_engine_passes_device_and_cache_dir_through_to_rapidocr(monkeypatch) -> None:
    recorded: list[tuple[str, Any]] = []

    def fake_rapidocr(device: str = "cpu", cache_dir: Any = None) -> SimpleNamespace:
        recorded.append((device, cache_dir))
        return SimpleNamespace(name="rapidocr")

    monkeypatch.setattr(ocr, "RapidOcr", fake_rapidocr)
    engine = ocr.create_engine("rapidocr", device="igpu", cache_dir="C:/cache")
    assert recorded == [("igpu", "C:/cache")]
    assert engine.name == "rapidocr"


def test_create_engine_auto_still_prefers_windows_with_device_and_cache_dir(monkeypatch) -> None:
    monkeypatch.setattr(ocr, "WindowsOcr", lambda: SimpleNamespace(name="windows"))
    monkeypatch.setattr(
        ocr, "RapidOcr", lambda device="cpu", cache_dir=None: (_ for _ in ()).throw(AssertionError("should not be called"))
    )
    engine = ocr.create_engine("auto", device="igpu", cache_dir="C:/cache")
    assert engine.name == "windows"


# --------------------------------------------------------------------------------- #
# _OpenVinoSession
# --------------------------------------------------------------------------------- #


class _FakeModel:
    def __init__(self) -> None:
        self.reshape_calls: list[Any] = []

    def clone(self) -> "_FakeModel":
        return _FakeModel()

    def reshape(self, spec: dict[int, list[int]]) -> None:
        self.reshape_calls.append(spec)


class _FakeInferRequest:
    """Records every infer() input and derives its "output" shape from it via a per-test
    callable, so det/rec/cls output shapes can be modelled without a real OpenVINO model."""

    def __init__(self, output_shape_fn: Callable[[tuple[int, ...]], tuple[int, ...]]) -> None:
        self._output_shape_fn = output_shape_fn
        self.infer_calls: list[np.ndarray] = []
        self._last_shape: tuple[int, ...] | None = None

    def infer(self, inputs: dict[int, np.ndarray]) -> None:
        x = inputs[0]
        self.infer_calls.append(x)
        self._last_shape = tuple(int(d) for d in x.shape)

    def get_output_tensor(self, index: int) -> SimpleNamespace:
        shape = self._output_shape_fn(self._last_shape)
        data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
        return SimpleNamespace(data=data)


class _FakeCompiledModel:
    def __init__(self, output_shape_fn: Callable[[tuple[int, ...]], tuple[int, ...]]) -> None:
        self._output_shape_fn = output_shape_fn
        self.requests: list[_FakeInferRequest] = []

    def create_infer_request(self) -> _FakeInferRequest:
        req = _FakeInferRequest(self._output_shape_fn)
        self.requests.append(req)
        return req


class _FakeCore:
    """Stands in for openvino.Core(): read_model()/compile_model(), recording every compile
    call's (model, device, config) so tests can assert caching behaviour and per-device config."""

    def __init__(self, output_shape_fn: Callable[[tuple[int, ...]], tuple[int, ...]]) -> None:
        self._output_shape_fn = output_shape_fn
        self.compile_calls: list[tuple[Any, str, dict[str, str]]] = []

    def read_model(self, path: str) -> _FakeModel:
        return _FakeModel()

    def compile_model(self, model: Any, device: str, config: dict[str, str]) -> _FakeCompiledModel:
        self.compile_calls.append((model, device, dict(config)))
        return _FakeCompiledModel(self._output_shape_fn)


def _det_shape_fn(input_shape: tuple[int, ...]) -> tuple[int, ...]:
    return (input_shape[0], 1, input_shape[2], input_shape[3])


def _rec_shape_fn(input_shape: tuple[int, ...]) -> tuple[int, ...]:
    return (input_shape[0], 25, 6625)


def test_openvino_session_det_crops_output_back_to_original_frame_size() -> None:
    core = _FakeCore(_det_shape_fn)
    session = ocr._OpenVinoSession(core, Path("fake_det.onnx"), "CPU", static=False, kind="det")

    out = session(np.random.rand(1, 3, 100, 150).astype(np.float32))

    assert out.shape == (1, 1, 100, 150)  # cropped back from the 256-padded compute shape


def test_openvino_session_dynamic_compiles_once_at_construction_regardless_of_input_shape() -> None:
    core = _FakeCore(_det_shape_fn)
    session = ocr._OpenVinoSession(core, Path("fake_det.onnx"), "CPU", static=False, kind="det")
    assert len(core.compile_calls) == 1  # compiled once, at construction

    session(np.zeros((1, 3, 100, 100), dtype=np.float32))
    session(np.zeros((1, 3, 300, 300), dtype=np.float32))
    assert len(core.compile_calls) == 1  # dynamic shapes: never recompiles


def test_openvino_session_static_rec_compiles_once_per_batch_width_bucket() -> None:
    core = _FakeCore(_rec_shape_fn)
    session = ocr._OpenVinoSession(core, Path("fake_rec.onnx"), "NPU", static=True, kind="rec")
    assert core.compile_calls == []  # static: nothing compiled yet at construction

    out1 = session(np.zeros((3, 3, 48, 300), dtype=np.float32))  # bucket 320
    assert out1.shape == (3, 25, 6625)  # sliced back to the real batch size (3), not the padded 6
    assert len(core.compile_calls) == 1

    session(np.zeros((3, 3, 48, 305), dtype=np.float32))  # same (batch, bucket) key -> reused
    assert len(core.compile_calls) == 1

    session(np.zeros((3, 3, 48, 400), dtype=np.float32))  # bucket 480 -> a new shape, new compile
    assert len(core.compile_calls) == 2


@pytest.mark.parametrize("device, expect_threads", [("CPU", True), ("GPU.0", False)])
def test_openvino_session_compile_config_has_inference_num_threads_only_on_cpu(device, expect_threads) -> None:
    core = _FakeCore(_det_shape_fn)
    ocr._OpenVinoSession(core, Path("fake_det.onnx"), device, static=False, kind="det")

    config = core.compile_calls[0][2]
    assert ("INFERENCE_NUM_THREADS" in config) is expect_threads


# --------------------------------------------------------------------------------- #
# _OrtSession (DirectML)
# --------------------------------------------------------------------------------- #


class _FakeIoInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeOrtSessionOptions:
    def __init__(self) -> None:
        self.enable_mem_pattern = True     # ORT's own default; _OrtSession must flip this to False
        self.execution_mode = None
        self.graph_optimization_level = None


class _FakeInferenceSession:
    """Stands in for ``onnxruntime.InferenceSession``: records the sess_options and providers it
    was built with and every ``run()`` input, and derives its "output" from the (already padded)
    input via a per-test callable -- same idea as ``_FakeCompiledModel`` above, for onnxruntime's
    API shape instead of OpenVINO's."""

    def __init__(self, model_path: Any, output_shape_fn: Callable[[tuple[int, ...]], tuple[int, ...]],
                 sess_options: Any = None, providers: Any = None) -> None:
        self.model_path = model_path
        self.sess_options = sess_options
        self.providers = providers
        self._output_shape_fn = output_shape_fn
        self.run_calls: list[np.ndarray] = []

    def get_inputs(self) -> list[_FakeIoInfo]:
        return [_FakeIoInfo("x")]

    def get_outputs(self) -> list[_FakeIoInfo]:
        return [_FakeIoInfo("y")]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        x = input_feed["x"]
        self.run_calls.append(x)
        shape = self._output_shape_fn(tuple(int(d) for d in x.shape))
        return [np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)]


def _fake_ort_module(output_shape_fn: Callable[[tuple[int, ...]], tuple[int, ...]]) -> tuple[Any, list[_FakeInferenceSession]]:
    created: list[_FakeInferenceSession] = []

    def inference_session(model_path: Any, sess_options: Any = None, providers: Any = None) -> _FakeInferenceSession:
        sess = _FakeInferenceSession(model_path, output_shape_fn, sess_options, providers)
        created.append(sess)
        return sess

    module = SimpleNamespace(
        InferenceSession=inference_session,
        SessionOptions=_FakeOrtSessionOptions,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="ORT_SEQUENTIAL"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="ORT_ENABLE_ALL"),
    )
    return module, created


def test_ort_session_builds_dml_provider_with_device_id_and_required_session_options() -> None:
    module, created = _fake_ort_module(_det_shape_fn)
    ocr._OrtSession(module, Path("fake_det.onnx"), 2, kind="det")

    assert len(created) == 1
    sess = created[0]
    assert sess.providers == [("DmlExecutionProvider", {"device_id": 2}), "CPUExecutionProvider"]
    assert sess.sess_options.enable_mem_pattern is False
    assert sess.sess_options.execution_mode == "ORT_SEQUENTIAL"
    assert sess.sess_options.graph_optimization_level == "ORT_ENABLE_ALL"


def test_ort_session_det_crops_output_back_to_original_frame_size() -> None:
    module, created = _fake_ort_module(_det_shape_fn)
    session = ocr._OrtSession(module, Path("fake_det.onnx"), 0, kind="det")

    out = session(np.random.rand(1, 3, 100, 150).astype(np.float32))

    assert out.shape == (1, 1, 100, 150)               # cropped back from the 256-padded shape
    assert created[0].run_calls[0].shape == (1, 3, 256, 256)   # padded before being run


def test_ort_session_rec_slices_batch_back_to_real_size() -> None:
    module, created = _fake_ort_module(_rec_shape_fn)
    session = ocr._OrtSession(module, Path("fake_rec.onnx"), 0, kind="rec")

    out = session(np.zeros((3, 3, 48, 300), dtype=np.float32))

    assert out.shape == (3, 25, 6625)                          # sliced back to the real batch (3)
    assert created[0].run_calls[0].shape[0] == ocr._NPU_BATCH  # padded batch (6) was what ran


def test_ort_session_cls_pads_batch_only_not_width() -> None:
    module, created = _fake_ort_module(lambda shape: (shape[0], 2))
    session = ocr._OrtSession(module, Path("fake_cls.onnx"), 0, kind="cls")

    out = session(np.zeros((3, 3, 48, 48), dtype=np.float32))

    assert out.shape == (3, 2)
    assert created[0].run_calls[0].shape == (ocr._NPU_BATCH, 3, 48, 48)  # width untouched for cls


def test_ort_session_warm_rec_buckets_only_runs_for_rec_kind() -> None:
    module, created = _fake_ort_module(_rec_shape_fn)
    det_session = ocr._OrtSession(module, Path("fake_det.onnx"), 0, kind="det")
    det_session.warm_rec_buckets(buckets=(320,))
    assert created[0].run_calls == []                          # kind != "rec": a no-op

    rec_session = ocr._OrtSession(module, Path("fake_rec.onnx"), 0, kind="rec")
    rec_session.warm_rec_buckets(buckets=(320, 480))
    assert len(created[1].run_calls) == 2


# --------------------------------------------------------------------------------- #
# directml_available
# --------------------------------------------------------------------------------- #


@pytest.fixture
def clean_dml_cache(monkeypatch):
    """Pins ``ocr._dml_available`` to unprobed (``None``) for the test, mirroring
    ``clean_windows_ocr_flag`` above -- ``directml_available()`` caches its result forever
    otherwise, so tests touching it would leak into each other."""
    monkeypatch.setattr(ocr, "_dml_available", None)


def test_directml_available_true_when_provider_listed(monkeypatch, clean_dml_cache) -> None:
    fake_ort = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider", "DmlExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    assert ocr.directml_available() is True


def test_directml_available_false_when_provider_missing(monkeypatch, clean_dml_cache) -> None:
    fake_ort = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    assert ocr.directml_available() is False


def test_directml_available_false_when_onnxruntime_not_importable(monkeypatch, clean_dml_cache) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert ocr.directml_available() is False


def test_directml_available_is_cached_after_first_call(monkeypatch, clean_dml_cache) -> None:
    calls = {"n": 0}

    def get_providers() -> list[str]:
        calls["n"] += 1
        return ["DmlExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(get_available_providers=get_providers))
    assert ocr.directml_available() is True
    assert ocr.directml_available() is True
    assert calls["n"] == 1


# --------------------------------------------------------------------------------- #
# describe_ocr_device
# --------------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kind, spec, expected",
    [
        ("igpu", "GPU.0", "igpu via OpenVINO GPU.0"),
        ("igpu", "dml:0", "igpu via DirectML device 0"),
        ("gpu", "dml:1", "gpu via DirectML device 1"),
        ("npu", "NPU", "npu via OpenVINO NPU"),
        ("cpu", "CPU", "cpu via OpenVINO CPU"),
        ("cpu", "", "cpu via onnxruntime"),
    ],
)
def test_describe_ocr_device(kind, spec, expected) -> None:
    assert ocr.describe_ocr_device(kind, spec) == expected


# --------------------------------------------------------------------------------- #
# RapidOcr.__init__: dispatch on the resolved device spec
# --------------------------------------------------------------------------------- #


class _FakeRapidEngine:
    """Stands in for ``rapidocr.RapidOCR`` -- just enough (``text_det`` / ``text_rec`` with a
    settable ``.session``) for ``_attach_directml`` / ``_attach_openvino`` to have somewhere to
    swap their session objects into, without a real rapidocr install or model files."""

    def __init__(self, **kwargs: Any) -> None:
        self.text_det = SimpleNamespace(session=None)
        self.text_rec = SimpleNamespace(session=None)


@pytest.fixture
def fake_rapidocr_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(RapidOCR=_FakeRapidEngine))


def test_rapidocr_init_dispatches_to_directml_for_a_dml_spec(
    monkeypatch, fake_rapidocr_module, clean_windows_ocr_flag
) -> None:
    monkeypatch.setattr(ocr, "resolve_ocr_device", lambda device: ("igpu", "dml:3"))
    calls: list[int] = []
    monkeypatch.setattr(ocr.RapidOcr, "_attach_directml", lambda self, device_id: calls.append(device_id))

    engine = ocr.RapidOcr("igpu")

    assert calls == [3]
    assert (engine.device, engine.ov_device) == ("igpu", "dml:3")
    assert engine.accelerator == "dml:3"


def test_rapidocr_init_dispatches_to_openvino_for_an_openvino_device_name(
    monkeypatch, fake_rapidocr_module, clean_windows_ocr_flag
) -> None:
    monkeypatch.setattr(ocr, "resolve_ocr_device", lambda device: ("igpu", "GPU.0"))
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(ocr.RapidOcr, "_attach_openvino",
                        lambda self, ov_device, cache_dir: calls.append((ov_device, cache_dir)))

    engine = ocr.RapidOcr("igpu")

    assert calls == [("GPU.0", None)]
    assert (engine.device, engine.ov_device) == ("igpu", "GPU.0")


def test_rapidocr_init_calls_neither_attach_method_for_an_empty_spec(
    monkeypatch, fake_rapidocr_module, clean_windows_ocr_flag
) -> None:
    monkeypatch.setattr(ocr, "resolve_ocr_device", lambda device: ("cpu", ""))
    monkeypatch.setattr(ocr.RapidOcr, "_attach_directml",
                        lambda self, device_id: (_ for _ in ()).throw(AssertionError("must not be called")))
    monkeypatch.setattr(ocr.RapidOcr, "_attach_openvino",
                        lambda self, ov_device, cache_dir: (_ for _ in ()).throw(AssertionError("must not be called")))

    engine = ocr.RapidOcr("cpu")

    assert (engine.device, engine.ov_device) == ("cpu", "")
    assert engine.accelerator == ""


def test_rapidocr_init_directml_failure_falls_back_to_cpu_with_warning(
    monkeypatch, fake_rapidocr_module, clean_windows_ocr_flag, caplog
) -> None:
    monkeypatch.setattr(ocr, "resolve_ocr_device", lambda device: ("igpu", "dml:0"))
    monkeypatch.setattr(ocr.RapidOcr, "_attach_directml",
                        lambda self, device_id: (_ for _ in ()).throw(RuntimeError("boom")))

    with caplog.at_level(logging.WARNING):
        engine = ocr.RapidOcr("igpu")

    assert (engine.device, engine.ov_device) == ("cpu", "")
    assert any("directml" in r.message.lower() for r in caplog.records)


def test_rapidocr_init_openvino_failure_falls_back_to_cpu_with_warning(
    monkeypatch, fake_rapidocr_module, clean_windows_ocr_flag, caplog
) -> None:
    monkeypatch.setattr(ocr, "resolve_ocr_device", lambda device: ("igpu", "GPU.0"))
    monkeypatch.setattr(ocr.RapidOcr, "_attach_openvino",
                        lambda self, ov_device, cache_dir: (_ for _ in ()).throw(RuntimeError("boom")))

    with caplog.at_level(logging.WARNING):
        engine = ocr.RapidOcr("igpu")

    assert (engine.device, engine.ov_device) == ("cpu", "")
    assert any("openvino" in r.message.lower() for r in caplog.records)


# --------------------------------------------------------------------------------- #
# RapidOcr._recognize / warm(): accelerator failure AFTER attach (not just during it)
# --------------------------------------------------------------------------------- #


class _FlakyAccelEngine:
    """Stands in for RapidOCR's callable engine once an accelerator is attached: ``text_det``/
    ``text_rec`` carry whichever session object is currently "installed" (a plain marker here,
    not a real session), and calling the engine raises on its SECOND call while an accelerator
    session is installed -- e.g. a DirectML 'device lost' mid-session -- but always succeeds once
    the CPU sessions are installed, mirroring a real fallback actually working."""

    def __init__(self) -> None:
        self.text_det = SimpleNamespace(session="cpu-det-session")
        self.text_rec = SimpleNamespace(session="cpu-rec-session")
        self.calls = 0

    def __call__(self, x: np.ndarray) -> SimpleNamespace:
        self.calls += 1
        if self.text_det.session == "accel-det-session" and self.calls >= 2:
            raise RuntimeError("simulated DXGI_ERROR_DEVICE_REMOVED")
        return SimpleNamespace(txts=(), boxes=None, scores=None)


def _bare_rapidocr(engine: _FlakyAccelEngine, *, device: str, ov_device: str) -> ocr.RapidOcr:
    """A RapidOcr instance with an accelerator already attached, built without running
    ``__init__`` (no real rapidocr/model files needed) -- ``_cpu_sessions`` is set exactly as
    ``_attach_directml``/``_attach_openvino`` set it, before they swap the accelerator in."""
    instance = object.__new__(ocr.RapidOcr)
    instance._engine = engine
    instance._cpu_sessions = ("cpu-det-session", "cpu-rec-session")
    instance._engine.text_det.session = "accel-det-session"
    instance._engine.text_rec.session = "accel-rec-session"
    instance.device = device
    instance.ov_device = ov_device
    instance._lock = threading.Lock()
    return instance


def test_recognize_falls_back_to_cpu_when_accelerator_fails_mid_session(caplog) -> None:
    engine = _FlakyAccelEngine()
    instance = _bare_rapidocr(engine, device="igpu", ov_device="dml:0")

    first = instance.recognize(np.zeros((8, 8, 3), dtype=np.uint8))
    assert first.text == ""  # first call: accelerator still healthy

    with caplog.at_level(logging.WARNING):
        second = instance.recognize(np.zeros((8, 8, 3), dtype=np.uint8))

    assert second.text == ""  # never raises -- degrades to CPU instead
    assert (instance.device, instance.ov_device) == ("cpu", "")
    assert instance._engine.text_det.session == "cpu-det-session"
    assert instance._engine.text_rec.session == "cpu-rec-session"
    assert any("accelerator inference failed" in r.message for r in caplog.records)


def test_recognize_reraises_when_already_on_cpu() -> None:
    """No accelerator attached (device == "cpu"): a failure has nowhere to fall back to, so it
    must propagate like any ordinary onnxruntime-on-CPU error."""

    class _AlwaysFails:
        def __call__(self, x: np.ndarray) -> SimpleNamespace:
            raise RuntimeError("cpu inference boom")

    instance = object.__new__(ocr.RapidOcr)
    instance._engine = _AlwaysFails()
    instance.device = "cpu"
    instance.ov_device = ""
    instance._lock = threading.Lock()

    with pytest.raises(RuntimeError, match="cpu inference boom"):
        instance.recognize(np.zeros((8, 8, 3), dtype=np.uint8))


def test_warm_swallows_a_det_warmup_failure_instead_of_raising(caplog) -> None:
    """warm()'s own det-model warm-up pass (a `recognize()` call) must degrade the same way as a
    live tick, not crash engine construction -- guarding it the same way ``warm_rec_buckets()``
    is already guarded a few lines above it."""

    class _AlwaysFailsRecognize:
        def __call__(self, x: np.ndarray) -> SimpleNamespace:
            raise RuntimeError("boom")

    instance = object.__new__(ocr.RapidOcr)
    instance._engine = _AlwaysFailsRecognize()
    instance.device = "cpu"
    instance.ov_device = ""
    instance._lock = threading.Lock()
    instance.compile_ms = []

    with caplog.at_level(logging.DEBUG):
        elapsed = instance.warm(64, 64)  # must not raise

    assert isinstance(elapsed, float)


# --------------------------------------------------------------------------------- #
# ScreenReader: accurate-engine-built-first rule (hands-free mode)
# --------------------------------------------------------------------------------- #


def test_accurate_engine_is_created_before_the_sampling_engine_on_first_tick(make_reader) -> None:
    cfg = _cfg(mode="auto", interval_s=0.03, ocr_engine="sampling")
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    creation_order: list[str] = []
    engines = {"accurate": ScriptedEngine(("acc",)), "sampling": ScriptedEngine(("samp",))}

    def factory(name: str) -> ScriptedEngine:
        creation_order.append(name)
        return engines[name]

    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=factory)
    reader.set_accurate_engine("accurate")
    reader.start()

    _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    assert creation_order == ["accurate", "sampling"]
    assert reader.accurate_engine_ready is True


def test_capture_now_accurate_uses_accurate_engine_then_next_tick_is_not_accurate(make_reader) -> None:
    cfg = _cfg(mode="auto", interval_s=0.05, ocr_engine="sampling")
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    engines = {"accurate": ScriptedEngine(("accurate text",)), "sampling": ScriptedEngine(("sampling text",))}
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engines[name])
    reader.set_accurate_engine("accurate")
    reader.start()

    # start() may already have produced one ordinary (non-accurate) tick before capture_now()
    # below is even seen by the worker thread, so wait specifically for the accurate one rather
    # than assuming it is the first ScreenTextEvent published.
    reader.capture_now(accurate=True)
    first = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) and e.accurate for e in evs))
    accurate_events = [e for e in first if isinstance(e, ScreenTextEvent) and e.accurate]
    assert accurate_events[0].text == "accurate text"

    grabber.image = _WHITE.copy()  # force a real change so the next normal tick publishes
    later = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    later_text = [e for e in later if isinstance(e, ScreenTextEvent)]
    assert later_text
    assert later_text[-1].accurate is False
    assert later_text[-1].text == "sampling text"


def test_capture_now_accurate_without_accurate_engine_set_still_publishes_via_sampling(make_reader) -> None:
    """No accurate engine: the sampling engine answers the request. The event is still flagged
    ``accurate=True`` -- the flag means "this answers a capture_now(accurate=True) request" (the
    hands-free controller waits for exactly that, whatever engine served it); ``engine`` says
    which engine actually produced the text."""
    cfg = _cfg(mode="auto", interval_s=0.05, ocr_engine="sampling")
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    engine = ScriptedEngine(("sampling text",))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()  # note: no set_accurate_engine()
    # the first, ordinary sample (the worker's first tick) is never flagged
    first = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    assert [e.accurate for e in first if isinstance(e, ScreenTextEvent)] == [False]

    reader.capture_now(accurate=True)
    events = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) and e.accurate for e in evs))
    answer = next(e for e in events if isinstance(e, ScreenTextEvent) and e.accurate)
    assert answer.text == "sampling text"
    assert answer.engine == "fake"
    assert reader.accurate_engine_ready is False


def test_accurate_engine_creation_failure_falls_back_to_sampling(make_reader) -> None:
    cfg = _cfg(mode="auto", interval_s=0.03, ocr_engine="sampling")
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    sampling_engine = ScriptedEngine(("sampling text",))

    def factory(name: str) -> ScriptedEngine:
        if name == "accurate":
            raise RuntimeError("accurate engine boom")
        return sampling_engine

    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=factory)
    reader.set_accurate_engine("accurate")
    reader.start()

    events = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    text_events = [e for e in events if isinstance(e, ScreenTextEvent)]
    assert text_events and text_events[0].text == "sampling text"
    assert reader.accurate_engine_ready is False


def test_sampling_engine_matching_accurate_name_reuses_the_same_instance(make_reader) -> None:
    cfg = _cfg(mode="auto", interval_s=0.03, ocr_engine="same")
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    creations: list[str] = []
    engine = ScriptedEngine(("same text",))

    def factory(name: str) -> ScriptedEngine:
        creations.append(name)
        return engine

    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=factory)
    reader.set_accurate_engine("same")
    reader.start()

    _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    assert creations == ["same"]  # one instance, not created twice for the same name


def test_screen_text_event_engine_field_carries_the_engine_label(make_reader) -> None:
    cfg = _cfg(mode="auto", interval_s=0.03, ocr_engine="sampling")
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    engine = ScriptedEngine(("hi",))
    engine.device = "igpu"  # ScriptedEngine.name == "fake" (class attribute)
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()

    events = _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    text_events = [e for e in events if isinstance(e, ScreenTextEvent)]
    assert text_events[0].engine == "fake/igpu"


def test_engine_warm_method_called_once_with_frame_size_after_creation(make_reader) -> None:
    cfg = _cfg(mode="auto", interval_s=0.03, ocr_engine="sampling")
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())  # 64x64
    engine = ScriptedEngine(("hi",))
    warm_calls: list[tuple[int, int]] = []
    engine.warm = lambda w, h: warm_calls.append((w, h))
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engine)
    reader.start()

    _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    assert warm_calls == [(64, 64)]

    grabber.image = _WHITE.copy()  # a later, changed tick must not warm again
    _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    assert warm_calls == [(64, 64)]


def test_stop_closes_both_accurate_and_sampling_engines(make_reader) -> None:
    cfg = _cfg(mode="auto", interval_s=0.03, ocr_engine="sampling")
    bus = EventBus()
    q = bus.subscribe()
    grabber = ScriptedGrabber(_BLACK.copy())
    engines = {"accurate": ScriptedEngine(("acc",)), "sampling": ScriptedEngine(("samp",))}
    reader = make_reader(cfg, bus, grabber_factory=lambda: grabber, engine_factory=lambda name: engines[name])
    reader.set_accurate_engine("accurate")
    reader.start()

    _drain_until(q, lambda evs: any(isinstance(e, ScreenTextEvent) for e in evs))
    reader.stop()

    assert engines["accurate"].closed == 1
    assert engines["sampling"].closed == 1


def test_resolve_ocr_device_openvino_cpu_only_plus_amd_and_dml_uses_directml() -> None:
    """OpenVINO installed but seeing only the CPU (no Intel GPU/NPU) must not block the DirectML step."""
    gpus = [_gpu("amd", uma=True)]
    assert ocr.resolve_ocr_device("auto", _CPU_ONLY, gpus=gpus, dml=True) == ("igpu", "dml:0")
    assert ocr.resolve_ocr_device("igpu", _CPU_ONLY, gpus=gpus, dml=True) == ("igpu", "dml:0")


# --------------------------------------------------------------------------------- #
# resolve_ocr_device: rule (a2), Intel Arc DISCRETE GPUs (added 2026-09-26)
#
# No Intel GPU exists on this box, so every case here is driven by a synthetic ``devices`` dict
# (standing in for openvino_devices()) and synthetic GpuInfo lists (standing in for
# exocortex.gpu.detect_gpus()) -- never a real OpenVINO device or a real adapter.
# --------------------------------------------------------------------------------- #


def test_resolve_ocr_device_intel_arc_dgpu_alone_is_used_via_openvino_as_kind_gpu() -> None:
    """An Intel(R) Arc(TM) ... (dGPU) name, with no Intel iGPU in the mix and no NVIDIA adapter in
    ``gpus`` (an Intel-only desktop: an Arc B580 next to its Core i9's own display controller,
    which OpenVINO does not enumerate as a GPU device at all here) -- the Arc IS the machine's
    GPU, so rule (a2) accepts it, with kind "gpu" (not "igpu": it is not integrated)."""
    gpus = [_gpu("intel", uma=False)]
    assert ocr.resolve_ocr_device("auto", _INTEL_ARC_DGPU_ONLY, gpus=gpus) == ("gpu", "GPU.0")
    assert ocr.resolve_ocr_device("igpu", _INTEL_ARC_DGPU_ONLY, gpus=gpus) == ("gpu", "GPU.0")


def test_resolve_ocr_device_intel_arc_dgpu_beats_directml_on_an_amd_igpu() -> None:
    """Arc dGPU + AMD iGPU (no NVIDIA): OpenVINO on the Arc (a2) is chosen over DirectML on the AMD
    adapter (b) -- (a2) is checked before the DirectML step -- even with dml=True."""
    gpus = [_gpu("intel", uma=False), _gpu("amd", uma=True, index=1)]
    assert ocr.resolve_ocr_device("auto", _INTEL_ARC_DGPU_ONLY, gpus=gpus, dml=True) == ("gpu", "GPU.0")


def test_resolve_ocr_device_intel_arc_dgpu_never_used_when_an_nvidia_adapter_is_present() -> None:
    """An NVIDIA adapter anywhere in ``gpus`` blocks rule (a2) exactly as it blocks DirectML: OCR
    must never land on a machine's NVIDIA GPU, even though the OpenVINO device name here says
    Intel and (dGPU) -- unlike the DirectML step, this must hold even with dml=True/False, since
    (a2) does not consult ``dml`` at all."""
    gpus = [_gpu("nvidia", uma=False), _gpu("intel", uma=True, index=1)]
    assert ocr.resolve_ocr_device("auto", _IGPU_AND_NPU, gpus=gpus) == ("igpu", "GPU.0")  # unchanged: rule (a) still wins
    arc_plus_nvidia = [_gpu("nvidia", uma=False), _gpu("intel", uma=False, index=1)]
    assert ocr.resolve_ocr_device("auto", _INTEL_ARC_DGPU_ONLY, gpus=arc_plus_nvidia, dml=True) == ("cpu", "CPU")


def test_resolve_ocr_device_intel_igpu_and_intel_arc_dgpu_both_listed_keeps_using_the_igpu() -> None:
    """Rule (a2) only applies when (a) found no Intel iGPU: with both an Intel iGPU (GPU.0) and an
    Intel Arc dGPU (GPU.1) visible to OpenVINO -- e.g. a laptop's Arc 140T iGPU plus an Arc Pro
    dGPU over Thunderbolt/OCuLink -- the iGPU is used exactly as before this rule existed."""
    gpus = [_gpu("intel", uma=True), _gpu("intel", uma=False, index=1)]
    assert ocr.resolve_ocr_device("auto", _INTEL_IGPU_AND_ARC_DGPU, gpus=gpus) == ("igpu", "GPU.0")


def test_resolve_ocr_device_nvidia_dgpu_only_still_never_picked_by_rule_a_or_a2() -> None:
    """The existing rule-(a) exclusion (an NVIDIA dGPU is never routed through OpenVINO) still
    holds now that (a2) exists: it only ever accepts an INTEL dGPU name."""
    assert ocr.resolve_ocr_device("auto", _DGPU_ONLY, gpus=[_gpu("nvidia", uma=False)], dml=False) == ("cpu", "CPU")


def test_resolve_ocr_device_intel_arc_dgpu_with_empty_gpus_list_fails_closed_to_cpu() -> None:
    """Regression for the (a2) fail-open bug: an empty/failed hardware enumeration (``gpus=[]``,
    exactly what :func:`exocortex.gpu.detect_gpus` returns on a DXGI hiccup, per its own docstring
    -- advisory, never raises) is inconclusive, not proof there is no NVIDIA adapter, so it must
    block (a2) exactly like it already blocks the DirectML step (b) (see
    ``test_resolve_ocr_device_dml_true_but_no_gpus_falls_back_to_cpu`` above) instead of being read
    as "confirmed no NVIDIA adapter" and letting OpenVINO through."""
    assert ocr.resolve_ocr_device("auto", _INTEL_ARC_DGPU_ONLY, gpus=[], dml=False) == ("cpu", "CPU")
    assert ocr.resolve_ocr_device("igpu", _INTEL_ARC_DGPU_ONLY, gpus=[], dml=False) == ("cpu", "CPU")


# --------------------------------------------------------------------------------- #
# resolve_ocr_device: rule (a)'s no-suffix fallback, cross-checked against ``gpus``
# (added for the (a2)-bypass finding: a genuinely discrete Intel GPU whose OpenVINO name omits
# the "(dGPU)" tag must not silently skip (a2)'s NVIDIA gate).
# --------------------------------------------------------------------------------- #

_INTEL_ARC_NO_SUFFIX = {"CPU": "i9", "GPU.0": "Intel(R) Arc(TM) B580 Graphics"}  # no (dGPU)/(iGPU) tag


def test_resolve_ocr_device_no_suffix_intel_name_confirmed_discrete_plus_nvidia_falls_to_cpu() -> None:
    """The bug this locks in: an Intel dGPU whose OpenVINO FULL_DEVICE_NAME omits the "(dGPU)" tag
    must not bypass the NVIDIA safety gate. With ``gpus`` showing the only Intel adapter is
    discrete (non-UMA) AND an NVIDIA adapter present, the no-suffix name must not be accepted by
    rule (a) -- and, since it also lacks the literal "(dGPU)" tag rule (a2) requires, nothing picks
    it up, so this must fall all the way to cpu (never the NVIDIA adapter)."""
    gpus = [_gpu("nvidia", uma=False), _gpu("intel", uma=False, index=1)]
    assert ocr.resolve_ocr_device("auto", _INTEL_ARC_NO_SUFFIX, gpus=gpus, dml=False) == ("cpu", "CPU")
    assert ocr.resolve_ocr_device("auto", _INTEL_ARC_NO_SUFFIX, gpus=gpus, dml=True) == ("cpu", "CPU")


def test_resolve_ocr_device_no_suffix_intel_name_confirmed_discrete_no_nvidia_uses_directml() -> None:
    """Same confirmed-discrete no-suffix Intel name, but no NVIDIA adapter anywhere: rule (a) still
    withholds it (its name is not proven safe), but with no NVIDIA present the DirectML step (b) is
    free to pick the discrete Intel adapter up instead -- never through OpenVINO (unverified for a
    plain "GPU" name), but also never blocked from any acceleration at all."""
    gpus = [_gpu("intel", uma=False)]
    assert ocr.resolve_ocr_device("auto", _INTEL_ARC_NO_SUFFIX, gpus=gpus, dml=True) == ("gpu", "dml:0")


def test_resolve_ocr_device_no_suffix_intel_name_integrated_adapter_keeps_using_igpu() -> None:
    """Parity guard: when the correlated Intel adapter is actually integrated (UMA) -- the classic
    older-OpenVINO Iris-Xe-style laptop this fallback was written for -- rule (a) must keep
    accepting the no-suffix name exactly as before, even with an NVIDIA adapter also present (a
    real NVIDIA laptop with an Iris Xe iGPU must behave byte-identically to main)."""
    gpus = [_gpu("nvidia", uma=False), _gpu("intel", uma=True, index=1)]
    assert ocr.resolve_ocr_device("auto", _INTEL_ARC_NO_SUFFIX, gpus=gpus, dml=False) == ("igpu", "GPU.0")


def test_resolve_ocr_device_no_suffix_intel_name_empty_gpus_list_keeps_old_behavior() -> None:
    """Parity guard: an empty/failed hardware enumeration cannot correlate the no-suffix name to
    any adapter at all, so it proves nothing and must NOT withhold rule (a)'s pre-existing
    behaviour (unlike (a2), which requires proof of safety, this pre-existing branch requires proof
    of danger before it changes -- see the fix-safety note in the module docstring)."""
    assert ocr.resolve_ocr_device("auto", _INTEL_ARC_NO_SUFFIX, gpus=[], dml=False) == ("igpu", "GPU.0")


def test_resolve_ocr_device_no_suffix_intel_name_no_intel_adapter_in_gpus_keeps_old_behavior() -> None:
    """Parity guard: ``gpus`` enumerated fine but has no Intel entry at all (can't correlate) --
    must not withhold rule (a)'s fallback either."""
    gpus = [_gpu("amd", uma=True)]
    assert ocr.resolve_ocr_device("auto", _INTEL_ARC_NO_SUFFIX, gpus=gpus, dml=False) == ("igpu", "GPU.0")


def test_no_suffix_intel_gpu_is_confirmed_discrete_mixed_adapters_is_false() -> None:
    """Direct unit test of the helper: a mix of an integrated and a discrete Intel adapter is not
    a confident "this GPU is discrete" signal (can't tell which one the OpenVINO name refers to),
    so it must return False (parity-preserving)."""
    gpus = [_gpu("intel", uma=True), _gpu("intel", uma=False, index=1)]
    assert ocr._no_suffix_intel_gpu_is_confirmed_discrete(gpus) is False


def test_no_suffix_intel_gpu_is_confirmed_discrete_all_discrete_is_true() -> None:
    assert ocr._no_suffix_intel_gpu_is_confirmed_discrete([_gpu("intel", uma=False)]) is True


def test_no_suffix_intel_gpu_is_confirmed_discrete_empty_list_is_false() -> None:
    assert ocr._no_suffix_intel_gpu_is_confirmed_discrete([]) is False
