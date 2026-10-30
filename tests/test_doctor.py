"""Unit tests for exocortex.doctor -- offline and deterministic.

Every heavy probe (audio device enumeration, the OCR subprocess probes, screen caret tracking,
LlamaServerManager.start, the LM Studio / Ollama clients, route resolution, GPU discovery,
--list-devices) is monkeypatched, so `run_doctor()` completes in well under 2 s regardless of
what hardware or software this machine actually has -- no real GPU, audio device, llama-server
process or network call is touched. Real tests against this box's hardware are `@pytest.mark.live`
elsewhere (see tests/test_asr.py, scripts/bench_asr.py); nothing here needs that marker.
"""
from __future__ import annotations

import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from exocortex import doctor
from exocortex.config import Config

# ---- fake GPU discovery (exocortex.gpu) --------------------------------------------------------


@dataclass
class _FakeGpu:
    name: str
    vendor: str
    budget_mb: int = 48000


def _patch_gpu(monkeypatch: pytest.MonkeyPatch, vendor: str = "amd") -> None:
    from exocortex import gpu

    fake = _FakeGpu(name="Fake Radeon 8060S", vendor=vendor)
    monkeypatch.setattr(gpu, "describe", lambda *a, **k: f"{fake.name} [{vendor}, integrated/UMA, 32590 MiB dedicated + 16204 MiB shared]")
    monkeypatch.setattr(gpu, "detect_gpus", lambda *a, **k: [fake])
    monkeypatch.setattr(gpu, "primary_gpu", lambda *a, **k: fake)
    monkeypatch.setattr(gpu, "gpu_vendor", lambda *a, **k: vendor)
    monkeypatch.setattr(gpu, "vram_free_mb", lambda: 20000)
    monkeypatch.setattr(gpu, "vram_total_mb", lambda: 48000, raising=False)
    monkeypatch.setattr(gpu, "vram_source", lambda: "pdh", raising=False)


# ---- fake audio devices (exocortex.audio.capture) ----------------------------------------------


@dataclass
class _FakeAudioDevice:
    index: int
    name: str
    sample_rate: int
    channels: int
    is_loopback: bool
    is_input: bool


def _patch_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    from exocortex.audio import capture

    devices = [
        _FakeAudioDevice(0, "Fake Speakers (loopback)", 48000, 2, True, False),
        _FakeAudioDevice(1, "Fake Microphone", 16000, 1, False, True),
    ]
    monkeypatch.setattr(capture, "list_devices", lambda: devices)
    monkeypatch.setattr(capture, "resolve_device", lambda kind, spec: devices[0] if kind == "system" else devices[1])


# ---- fake screen reading (capture/foreground/ocr) + the OCR subprocess probes ------------------


class _FakeCompletedProcess:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_subprocess_run(*args: Any, **kwargs: Any) -> _FakeCompletedProcess:
    return _FakeCompletedProcess(0, "12|34|rapidocr|ExoCortex screen reading check: budget 1.2M", "")


def _patch_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    from exocortex.screen import foreground, ocr

    monkeypatch.setattr(foreground, "monitor_rects", lambda: [(0, 0, 1920, 1080)])
    monkeypatch.setattr(foreground, "foreground_window", lambda: None)
    monkeypatch.setattr(ocr, "openvino_devices", lambda: {"GPU": "gfx1151 (iGPU)"})
    monkeypatch.setattr(ocr, "resolve_ocr_device", lambda requested, devices=None, **kw: ("igpu", "GPU"))
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run)


def _patch_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    from exocortex.screen import caret

    class _FakeTracker:
        def __init__(self, *a: Any, **k: Any) -> None:
            self.detail = "fake caret tracker (patched)"
            self.last_reject = None

        def attach(self) -> bool:
            return True

        def probe(self) -> None:
            return None

        def detach(self) -> None:
            pass

    monkeypatch.setattr(caret, "CaretTracker", _FakeTracker)


# ---- fake llama.cpp backend (exocortex.llm.server / .client) ------------------------------------


class _FakeServerBinary:
    exe = Path("C:/fake/llama.cpp/llama-server.exe")
    dll_dirs: tuple = ()
    flavor = "vulkan"
    version = "2.46.0"


class _FakeGgufModel:
    key = "fake/model"
    name = "model"
    path = Path("C:/fake/models/fake/model.gguf")
    size_bytes = 1_000_000_000
    display = "fake/model  (1.0 GB)"


class _FakeLlamaServerManager:
    def __init__(self, cfg: Any, **k: Any) -> None:
        self.cfg = cfg
        self.device = "Vulkan0"
        self.note = "fake device note"

    def start(self, *a: Any, **k: Any) -> str:
        return "http://127.0.0.1:19999"

    def stop(self) -> None:
        pass

    def build_args(self, model: Any, port: int, device: Any = None) -> list[str]:
        return ["--threads", "8"]


@dataclass
class _FakeStats:
    ttft_s: float = 0.05
    tok_per_s: float = 42.0


class _FakeLMStudioClient:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def set_base_url(self, url: str) -> None:
        self._base_url = url

    def stream_completion(self, prompt: str, max_tokens: int = 24, stop: Any = None) -> Any:
        yield "ready"

    @property
    def last_stats(self) -> _FakeStats:
        return _FakeStats()

    def close(self) -> None:
        pass

    def health(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(reachable=True, detail="", model_listed=True, model_loaded=True)

    def list_models(self) -> list[types.SimpleNamespace]:
        return [types.SimpleNamespace(id="fake-model")]


def _patch_llamacpp(monkeypatch: pytest.MonkeyPatch) -> None:
    from exocortex.llm import client as client_mod
    from exocortex.llm import server as server_mod

    monkeypatch.setattr(server_mod, "find_server_binary", lambda *a, **k: _FakeServerBinary())
    monkeypatch.setattr(server_mod, "lmstudio_models_dir", lambda cfg: Path("C:/fake/models"))
    monkeypatch.setattr(server_mod, "scan_models", lambda md: [_FakeGgufModel()])
    monkeypatch.setattr(server_mod, "resolve_model", lambda spec, md: _FakeGgufModel())
    monkeypatch.setattr(server_mod, "LlamaServerManager", _FakeLlamaServerManager)
    monkeypatch.setattr(server_mod, "backend_ladder", lambda vendor, requested: (requested, "cpu"), raising=False)
    monkeypatch.setattr(server_mod, "list_devices", lambda binary: [], raising=False)
    monkeypatch.setattr(server_mod, "choose_model", lambda spec, md, budget: (_FakeGgufModel(), "fits budget"), raising=False)
    monkeypatch.setattr(client_mod, "LMStudioClient", _FakeLMStudioClient)


class _FakeOllamaClient:
    def health(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(reachable=True, detail="")

    def list_models(self) -> list[types.SimpleNamespace]:
        return [types.SimpleNamespace(id="llama3", size_bytes=4_000_000_000)]

    def resolve_model(self, spec: str) -> str:
        return "llama3"

    def stream_completion(self, prompt: str, max_tokens: int = 24, stop: Any = None) -> Any:
        yield "ready"

    @property
    def last_stats(self) -> _FakeStats:
        return _FakeStats()

    def close(self) -> None:
        pass


def _patch_ollama_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from exocortex.llm import client as client_mod

    monkeypatch.setattr(client_mod, "make_client", lambda cfg, backend: _FakeOllamaClient(), raising=False)


@dataclass
class _FakeRoute:
    backend: str
    reason: str = "ok"
    checked: list = field(default_factory=list)
    binary: Any = None
    model_count: int = 0


def _patch_route(monkeypatch: pytest.MonkeyPatch, backend: str, reason: str = "ok") -> _FakeRoute:
    """`exocortex.llm.route` does not exist yet (another agent's in-progress module): install a
    fake one into sys.modules AND as the `exocortex.llm` package attribute, since `from exocortex.llm
    import route` only skips the real import when both are set."""
    import exocortex.llm as llm_pkg

    rt = _FakeRoute(backend=backend, reason=reason)
    fake = types.ModuleType("exocortex.llm.route")
    fake.resolve_route = lambda cfg: rt          # type: ignore[attr-defined]
    fake.describe_route = lambda r: f"route -> {r.backend}: {r.reason}"   # type: ignore[attr-defined]
    fake.Route = _FakeRoute                      # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "exocortex.llm.route", fake)
    monkeypatch.setattr(llm_pkg, "route", fake, raising=False)
    return rt


# =================================================================================================
# tests
# =================================================================================================


def test_llm_config_bare_defaults_match_main_on_nvidia_relevant_fields() -> None:
    """Regression for the config.py "auto"/"auto" default-swap finding.

    A fresh/no-config install (no config.toml anywhere in load_config()'s search path) must load
    the same model on an NVIDIA machine as it always did: the specific, tuned "maple-preview" key,
    not whatever pick_default_model()'s VRAM-budget search happens to land on. backend="auto" is
    intentionally kept (it resolves to "llamacpp" on a working NVIDIA install via
    exocortex.llm.route.resolve_route, so it is not a behaviour change there) -- only the model
    default regressed and is locked down here.
    """
    cfg = Config()
    assert cfg.llm.model == "maple-preview"
    assert cfg.llm.backend == "auto"


def test_run_doctor_completes_offline_fast_and_returns_a_deterministic_int(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import time

    _patch_gpu(monkeypatch, vendor="amd")
    _patch_audio(monkeypatch)
    _patch_screen(monkeypatch)
    _patch_completion(monkeypatch)
    _patch_llamacpp(monkeypatch)
    _patch_route(monkeypatch, backend="llamacpp")

    cfg = Config()
    assert cfg.llm.backend == "auto"    # the default this doctor must route

    t0 = time.perf_counter()
    result = doctor.run_doctor(cfg)
    elapsed = time.perf_counter() - t0

    assert result in (0, 1)
    assert elapsed < 2.0, f"run_doctor took {elapsed:.2f}s -- a heavy probe was not monkeypatched"
    out = capsys.readouterr().out
    assert "ExoCortex doctor" in out

    # running it again must produce the exact same verdict (deterministic, no live-machine drift)
    result2 = doctor.run_doctor(cfg)
    assert result2 == result


def test_gpu_section_prints_vendor_and_memory_source_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_gpu(monkeypatch, vendor="amd")
    r = doctor._Report()
    doctor._check_gpu(r, Config())
    out = capsys.readouterr().out
    assert "== GPU ==" in out
    assert "[amd]" in out
    assert "VRAM reading via pdh: 20000 MiB free / 48000 MiB total" in out


def test_gpu_section_warns_when_no_memory_reading_is_available(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from exocortex import gpu

    _patch_gpu(monkeypatch, vendor="amd")
    monkeypatch.setattr(gpu, "vram_free_mb", lambda: None)
    r = doctor._Report()
    doctor._check_gpu(r, Config())
    out = capsys.readouterr().out
    assert "no GPU memory reading on this machine; the GPU pressure guard stays off" in out


def test_cuda_lines_appear_only_when_vendor_is_nvidia(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_gpu(monkeypatch, vendor="amd")
    r = doctor._Report()
    doctor._check_gpu(r, Config())
    out = capsys.readouterr().out
    assert "ctranslate2 sees" not in out
    assert "NVIDIA pip DLL folders" not in out
    assert "no NVIDIA GPU: CUDA libraries are not needed" in out

    _patch_gpu(monkeypatch, vendor="nvidia")
    r2 = doctor._Report()
    doctor._check_gpu(r2, Config())
    out2 = capsys.readouterr().out
    assert "ctranslate2 sees" in out2 or "ctranslate2 CUDA probe failed" in out2
    assert "no NVIDIA GPU" not in out2


# ---- Intel Arc discrete-GPU advisories (exocortex.gpu.intel_arc_notes; no such hardware on any
# dev box this branch was built on -- unit-tested here with a synthetic Intel GpuInfo) ----------


@dataclass
class _FakeIntelGpu:
    name: str = "Intel Arc(TM) B580 Graphics"
    vendor: str = "intel"
    vendor_id: int = 0x8086
    device_id: int = 0xE20B
    dedicated_mb: int = 12288
    uma: bool = False
    budget_mb: int = 12288


def _patch_intel_gpu(monkeypatch: pytest.MonkeyPatch, driver: str | None) -> _FakeIntelGpu:
    from exocortex import gpu

    fake = _FakeIntelGpu()
    monkeypatch.setattr(gpu, "describe", lambda *a, **k: f"{fake.name} [intel, discrete, 12288 MiB dedicated + 0 MiB shared]")
    monkeypatch.setattr(gpu, "detect_gpus", lambda *a, **k: [fake])
    monkeypatch.setattr(gpu, "primary_gpu", lambda *a, **k: fake)
    monkeypatch.setattr(gpu, "gpu_vendor", lambda *a, **k: "intel")
    monkeypatch.setattr(gpu, "vram_free_mb", lambda: 10000)
    monkeypatch.setattr(gpu, "vram_total_mb", lambda: 12288, raising=False)
    monkeypatch.setattr(gpu, "vram_source", lambda: "pdh", raising=False)
    monkeypatch.setattr(gpu, "driver_version", lambda g: driver)
    return fake


def test_gpu_section_warns_for_an_old_driver_on_a_discrete_intel_primary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_intel_gpu(monkeypatch, driver="31.0.101.5522")
    r = doctor._Report()
    doctor._check_gpu(r, Config())
    out = capsys.readouterr().out.lower()
    assert "update the intel arc driver" in out


def test_gpu_section_is_clean_for_a_current_driver_on_a_discrete_intel_primary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_intel_gpu(monkeypatch, driver="32.0.101.6795")
    r = doctor._Report()
    doctor._check_gpu(r, Config())
    out = capsys.readouterr().out.lower()
    assert "update the intel arc driver" not in out
    assert "cooperative-matrix" in out


def test_gpu_section_warns_when_driver_is_unknown_on_a_discrete_intel_primary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: a Battlemage card whose driver could not be read (registry lookup failed, or
    the version string didn't parse) must not be reported as if Xe2 cooperative-matrix support
    were confirmed -- that note is unverified/unknown, so it must go out at WARN, not OK."""
    _patch_intel_gpu(monkeypatch, driver=None)
    r = doctor._Report()
    doctor._check_gpu(r, Config())
    out = capsys.readouterr().out
    lower = out.lower()
    assert "could not read the driver version" in lower
    for gpu_line in out.splitlines():
        if "could not read the driver version" in gpu_line.lower():
            assert gpu_line.startswith("[WARN]"), gpu_line
            break
    else:
        pytest.fail("expected 'could not read the driver version' note line not found in output")


def test_gpu_section_never_touches_intel_notes_for_a_non_intel_primary_without_a_uma_attr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`_FakeGpu` (used by `_patch_gpu` for the AMD/NVIDIA tests above) has no `.uma` attribute --
    the vendor == 'intel' check must short-circuit before that would ever be read, else this
    would AttributeError instead of a clean report."""
    _patch_gpu(monkeypatch, vendor="amd")
    r = doctor._Report()
    doctor._check_gpu(r, Config())    # must not raise
    out = capsys.readouterr().out
    assert "primary GPU lookup failed" not in out


def test_route_section_appears_for_backend_auto(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_route(monkeypatch, backend="llamacpp", reason="found a runtime pack and a GGUF")
    monkeypatch.setitem(doctor._LLM_DISPATCH, "llamacpp", lambda r, cfg: None)   # isolate dispatch only

    cfg = Config()
    assert cfg.llm.backend == "auto"
    r = doctor._Report()
    doctor._check_llm(r, cfg)
    out = capsys.readouterr().out
    assert "LLM backend (auto route)" in out
    assert "route -> llamacpp: found a runtime pack and a GGUF" in out


def test_check_llm_dispatches_to_the_ollama_check_for_an_ollama_route(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_route(monkeypatch, backend="ollama", reason="Ollama answered")
    calls: list[str] = []
    monkeypatch.setitem(doctor._LLM_DISPATCH, "ollama", lambda r, cfg: calls.append("ollama"))
    monkeypatch.setitem(doctor._LLM_DISPATCH, "llamacpp", lambda r, cfg: calls.append("llamacpp"))
    monkeypatch.setitem(doctor._LLM_DISPATCH, "openai", lambda r, cfg: calls.append("openai"))

    cfg = Config()
    r = doctor._Report()
    doctor._check_llm(r, cfg)
    assert calls == ["ollama"]


def test_check_llm_ollama_runs_the_full_ollama_check_when_dispatched_directly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_ollama_client(monkeypatch)
    cfg = Config()
    cfg.llm.backend = "ollama"
    r = doctor._Report()
    doctor._check_llm_ollama(r, cfg)
    out = capsys.readouterr().out
    assert "Ollama reachable" in out
    assert "llama3" in out
    assert r.fails == 0


def test_check_llm_none_route_fails_with_the_route_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_route(monkeypatch, backend="none", reason="install LM Studio or Ollama")
    cfg = Config()
    r = doctor._Report()
    doctor._check_llm(r, cfg)
    out = capsys.readouterr().out
    assert "install LM Studio or Ollama" in out
    assert r.fails == 1


def test_check_llm_explicit_backend_bypasses_route_resolution(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-"auto" [llm].backend must dispatch directly -- exocortex.llm.route must not even be
    imported, exactly like before this feature existed."""
    called: list[str] = []
    monkeypatch.setitem(doctor._LLM_DISPATCH, "openai", lambda r, cfg: called.append("openai"))
    cfg = Config()
    cfg.llm.backend = "openai"
    r = doctor._Report()
    doctor._check_llm(r, cfg)
    assert called == ["openai"]


def test_llamacpp_check_missing_new_server_contract_degrades_to_warn_not_crash(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """backend_ladder/list_devices/choose_model are another agent's in-progress contract: when
    they are absent, the check must still complete (WARN lines), never raise."""
    from exocortex.llm import server as server_mod

    _patch_gpu(monkeypatch, vendor="amd")
    monkeypatch.setattr(server_mod, "find_server_binary", lambda *a, **k: _FakeServerBinary())
    monkeypatch.setattr(server_mod, "lmstudio_models_dir", lambda cfg: Path("C:/fake/models"))
    monkeypatch.setattr(server_mod, "scan_models", lambda md: [_FakeGgufModel()])
    monkeypatch.setattr(server_mod, "resolve_model", lambda spec, md: _FakeGgufModel())
    monkeypatch.setattr(server_mod, "LlamaServerManager", _FakeLlamaServerManager)
    monkeypatch.delattr(server_mod, "backend_ladder", raising=False)
    monkeypatch.delattr(server_mod, "list_devices", raising=False)
    monkeypatch.delattr(server_mod, "choose_model", raising=False)
    from exocortex.llm import client as client_mod

    monkeypatch.setattr(client_mod, "LMStudioClient", _FakeLMStudioClient)

    cfg = Config()
    r = doctor._Report()
    doctor._check_llm_llamacpp(r, cfg)   # must not raise
    out = capsys.readouterr().out
    assert "backend_ladder() not available yet" in out
    assert "list_devices() not available yet" in out
    assert "[llm].model" in out          # fell back to resolve_model()


# ---- 2026-09-26: ServerBinary.source, exocortex.llm.runtimes, [llm].llama_env, mgr.fallbacks and
# the .tools/llama.cpp upstream-runtime listing -- all of it another agent's in-progress contract
# except the directory listing itself, so every case here is exercised via getattr(..., None)
# degrades (module/field absent, exactly the state of this branch today) and via a fake module
# installed into sys.modules the same way _patch_route() fakes exocortex.llm.route below. --------


def test_llamacpp_check_warns_when_binary_source_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_gpu(monkeypatch, vendor="amd")
    _patch_llamacpp(monkeypatch)
    r = doctor._Report()
    doctor._check_llm_llamacpp(r, Config())
    out = capsys.readouterr().out
    assert "ServerBinary.source not available yet" in out


def test_llamacpp_check_prints_binary_source_when_present(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_gpu(monkeypatch, vendor="amd")
    _patch_llamacpp(monkeypatch)
    monkeypatch.setattr(_FakeServerBinary, "source", "lmstudio", raising=False)
    r = doctor._Report()
    doctor._check_llm_llamacpp(r, Config())
    out = capsys.readouterr().out
    assert "binary source: lmstudio" in out


def test_llamacpp_check_resolves_the_binary_with_the_configured_runtime_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The doctor must name -- and run --list-devices against -- the binary the app will actually
    start. With [llm].llama_source = "upstream" (an RTX 50 laptop whose only LM Studio CUDA pack is
    the CUDA 11 build) that is the upstream build; an argument-less lookup found the LM Studio pack
    instead and reported a working CUDA 12 setup as one that sees no GPU. Offline: no `ensure`."""
    from exocortex.llm import server as server_mod

    _patch_gpu(monkeypatch, vendor="nvidia")
    _patch_llamacpp(monkeypatch)
    seen: dict = {}

    def fake_find(explicit: str, **kwargs: object) -> _FakeServerBinary:
        seen.update(kwargs)
        return _FakeServerBinary()

    monkeypatch.setattr(server_mod, "find_server_binary", fake_find)
    cfg = Config()
    cfg.llm.llama_source = "upstream"
    cfg.llm.llama_upstream_build = "b11200"
    cfg.llm.runtime_dir = "D:/runtimes"
    doctor._check_llm_llamacpp(doctor._Report(), cfg)
    assert seen["source"] == "upstream"
    assert seen["upstream_build"] == "b11200"
    assert seen["upstream_root"] == Path("D:/runtimes")
    assert seen["vendor"] == "nvidia"
    assert "ensure" not in seen, "the doctor must never download a runtime"


def _force_runtimes_module_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate exocortex.llm.runtimes not existing (this branch's state for most of its
    development): dropping only the sys.modules entry is not enough once the module has been
    imported for real at least once in this process, since Python also caches it as an attribute
    of the `exocortex.llm` package -- `from exocortex.llm import runtimes` would still find that
    attribute without ever consulting sys.modules. Removing both forces the import to fail again,
    exactly as it does on a checkout that predates that module landing."""
    import exocortex.llm as llm_pkg

    monkeypatch.delattr(llm_pkg, "runtimes", raising=False)
    monkeypatch.setitem(sys.modules, "exocortex.llm.runtimes", None)


def test_llamacpp_check_degrades_runtimes_module_to_warn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """describe_runtime()/vulkan_env_switches() must degrade to WARN, never raise, on a checkout
    where exocortex.llm.runtimes does not exist (most of this branch's development; forced here
    even though the module has since landed for real -- see _force_runtimes_module_absent)."""
    _force_runtimes_module_absent(monkeypatch)
    _patch_gpu(monkeypatch, vendor="amd")
    _patch_llamacpp(monkeypatch)
    r = doctor._Report()
    doctor._check_llm_llamacpp(r, Config())
    out = capsys.readouterr().out
    assert "exocortex.llm.runtimes.vulkan_env_switches() not available yet" in out


def test_llamacpp_check_uses_runtimes_module_once_it_exists(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Once exocortex.llm.runtimes lands (faked here into sys.modules, same technique as
    _patch_route below for exocortex.llm.route), describe_runtime()/vulkan_env_switches() are
    used instead of the WARN fallback. Both take the SAME folder (the one holding
    llama-server.exe) per that module's real contract, so this also pins that call shape down."""
    import exocortex.llm as llm_pkg

    calls: list[Any] = []
    fake_runtimes = types.ModuleType("exocortex.llm.runtimes")
    fake_runtimes.describe_runtime = lambda folder: f"upstream build at {folder}"  # type: ignore[attr-defined]

    def _switches(folder: Any) -> list[str]:
        calls.append(folder)
        return ["GGML_VK_DISABLE_F16", "GGML_VK_VISIBLE_DEVICES"]

    fake_runtimes.vulkan_env_switches = _switches  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "exocortex.llm.runtimes", fake_runtimes)
    monkeypatch.setattr(llm_pkg, "runtimes", fake_runtimes, raising=False)

    _patch_gpu(monkeypatch, vendor="amd")
    _patch_llamacpp(monkeypatch)
    monkeypatch.setattr(_FakeServerBinary, "source", "upstream", raising=False)

    r = doctor._Report()
    doctor._check_llm_llamacpp(r, Config())
    out = capsys.readouterr().out
    assert "upstream build at" in out
    assert "ggml-vulkan.dll knows: GGML_VK_DISABLE_F16, GGML_VK_VISIBLE_DEVICES" in out
    assert calls == [_FakeServerBinary.exe.parent]   # same folder describe_runtime() was given


def test_llamacpp_check_prints_llama_env_when_set(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_gpu(monkeypatch, vendor="amd")
    _patch_llamacpp(monkeypatch)
    cfg = Config()
    cfg.llm.llama_env = {"GGML_VK_DISABLE_F16": "1"}   # exocortex.llm, in-progress field
    r = doctor._Report()
    doctor._check_llm_llamacpp(r, cfg)
    out = capsys.readouterr().out
    assert "[llm].llama_env" in out and "GGML_VK_DISABLE_F16" in out


def test_llamacpp_check_prints_no_llama_env_line_when_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_gpu(monkeypatch, vendor="amd")
    _patch_llamacpp(monkeypatch)
    r = doctor._Report()
    doctor._check_llm_llamacpp(r, Config())
    out = capsys.readouterr().out
    assert "[llm].llama_env" not in out


def test_llamacpp_check_prints_fallbacks_recorded_by_start(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_gpu(monkeypatch, vendor="amd")
    _patch_llamacpp(monkeypatch)
    from exocortex.llm import server as server_mod

    class _FakeManagerWithFallbacks(_FakeLlamaServerManager):
        def __init__(self, cfg: Any, **k: Any) -> None:
            super().__init__(cfg, **k)
            self.fallbacks = ["amd-rocm 2.29.1: no GPU device"]

    monkeypatch.setattr(server_mod, "LlamaServerManager", _FakeManagerWithFallbacks)
    r = doctor._Report()
    doctor._check_llm_llamacpp(r, Config())
    out = capsys.readouterr().out
    assert "runtime pack skipped: amd-rocm 2.29.1: no GPU device" in out


def test_llamacpp_check_lists_installed_upstream_runtimes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """<root>/<build>/<kind> is exocortex.llm.runtimes.runtime_dir()'s own layout; a fake
    describe_runtime() (module presence/absence is covered separately above) pins the exact text
    down without depending on that module's real formatting."""
    import exocortex.llm as llm_pkg
    import exocortex.config as config_mod

    (tmp_path / ".tools" / "llama.cpp" / "b11200" / "vulkan").mkdir(parents=True)
    monkeypatch.setattr(config_mod, "project_root", lambda: tmp_path)
    fake_runtimes = types.ModuleType("exocortex.llm.runtimes")
    fake_runtimes.describe_runtime = lambda folder: f"llama.cpp b11200 vulkan (upstream, {folder})"  # type: ignore[attr-defined]
    fake_runtimes.RUNTIME_KINDS = ("vulkan", "sycl", "openvino", "cpu")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "exocortex.llm.runtimes", fake_runtimes)
    monkeypatch.setattr(llm_pkg, "runtimes", fake_runtimes, raising=False)

    _patch_gpu(monkeypatch, vendor="amd")
    _patch_llamacpp(monkeypatch)
    r = doctor._Report()
    doctor._check_llm_llamacpp(r, Config())
    out = capsys.readouterr().out
    assert f"upstream runtime installed: llama.cpp b11200 vulkan (upstream, {tmp_path / '.tools' / 'llama.cpp' / 'b11200' / 'vulkan'})" in out


def test_llamacpp_check_lists_upstream_runtimes_by_dirname_when_runtimes_module_is_absent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import exocortex.config as config_mod

    (tmp_path / ".tools" / "llama.cpp" / "b11200" / "vulkan").mkdir(parents=True)
    monkeypatch.setattr(config_mod, "project_root", lambda: tmp_path)
    _force_runtimes_module_absent(monkeypatch)
    _patch_gpu(monkeypatch, vendor="amd")
    _patch_llamacpp(monkeypatch)
    r = doctor._Report()
    doctor._check_llm_llamacpp(r, Config())
    out = capsys.readouterr().out
    assert "upstream runtime installed: vulkan build b11200" in out


def test_llamacpp_check_warns_when_no_upstream_runtimes_are_installed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import exocortex.config as config_mod

    monkeypatch.setattr(config_mod, "project_root", lambda: tmp_path)
    _patch_gpu(monkeypatch, vendor="amd")
    _patch_llamacpp(monkeypatch)
    r = doctor._Report()
    doctor._check_llm_llamacpp(r, Config())
    out = capsys.readouterr().out
    assert "no upstream llama.cpp runtimes under" in out


def test_llamacpp_check_always_prints_the_upstream_opt_in_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_gpu(monkeypatch, vendor="amd")
    _patch_llamacpp(monkeypatch)
    r = doctor._Report()
    doctor._check_llm_llamacpp(r, Config())
    out = capsys.readouterr().out
    assert "llama_backend = 'sycl'" in out
    assert "llama_source = 'upstream'" in out


def test_as_json_includes_gpu_vendor_gpus_and_route(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    _patch_gpu(monkeypatch, vendor="amd")
    _patch_route(monkeypatch, backend="llamacpp")
    data = json.loads(doctor.as_json(Config()))
    assert data["gpu_vendor"] == "amd"
    assert data["gpus"] == ["Fake Radeon 8060S"]
    assert data["route"] == "llamacpp"


# =====================================================================================
# 2026-09-27 (portable-install): the model-library section of the doctor
# =====================================================================================

def _patch_library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, models: list, profiles: dict | None = None,
                   report: object | None = None) -> None:
    from exocortex import hardware
    from exocortex.llm import library

    monkeypatch.setattr(library, "library_roots", lambda cfg: [("library", tmp_path / "llm"), ("lmstudio", tmp_path / "lm")])
    monkeypatch.setattr(library, "scan_library", lambda cfg, **kw: list(models))
    monkeypatch.setattr(library, "load_profiles", lambda cfg: dict(profiles or {}))
    monkeypatch.setattr(library, "library_dir", lambda cfg: tmp_path / "llm")
    monkeypatch.setattr(hardware, "report_path", lambda cfg: tmp_path / "llm" / "hardware.json")
    monkeypatch.setattr(hardware, "load", lambda path: report)


def test_model_library_section_lists_models_profiles_and_the_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from types import SimpleNamespace

    models = [SimpleNamespace(key="unsloth/gemma-4-12b-it", name="gemma-4-12b-it", quant="Q4_K_M", size_bytes=7.1e9, source="library"),
              SimpleNamespace(key="deepgrove/maple-preview", name="maple-preview", quant="TQ1_0", size_bytes=5.0e9, source="lmstudio")]
    report = SimpleNamespace(generated_at="2026-09-27T12:00", budget_mb=16003, recommended_policy="auto")
    _patch_library(monkeypatch, tmp_path, models=models, profiles={"unsloth/gemma-4-12b-it": object()}, report=report)
    cfg = Config()
    cfg.llm.model = "maple-preview"
    r = doctor._Report()
    doctor._check_model_library(r, cfg)
    out = capsys.readouterr().out
    assert "== Model library" in out
    assert "2 model(s): 1 from library, 1 from lmstudio" in out
    assert "unsloth/gemma-4-12b-it [Q4_K_M] 7.1 GB (library)" in out
    assert "1 tuning profile(s)" in out and "profiles.json" in out
    assert "hardware report" in out and "budget 16003 MiB, policy auto" in out
    assert "is not in the library" not in out          # maple-preview matches by name
    assert r.fails == 0


def test_model_library_section_warns_on_an_empty_library_and_a_missing_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _patch_library(monkeypatch, tmp_path, models=[], report=None)
    cfg = Config()
    cfg.llm.model = "maple-preview"
    r = doctor._Report()
    doctor._check_model_library(r, cfg)
    out = capsys.readouterr().out
    assert "[WARN] no GGUF model anywhere yet" in out and "Models..." in out
    assert "[WARN] [llm].model = 'maple-preview' is not in the library" in out
    assert "[WARN] no hardware report at" in out
    assert r.fails == 0


def test_doctor_warns_about_a_legacy_server_address_the_migration_dropped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review round two, 2026-09-27: an old config with BOTH base_url and ollama_url customised
    keeps one in server_url; the doctor names the other and how to get it back."""
    _patch_route(monkeypatch, backend="llamacpp")
    _patch_llamacpp(monkeypatch)
    _patch_gpu(monkeypatch, vendor="nvidia")
    cfg = Config()
    cfg.llm.server_url = "http://192.168.1.50:8081"
    cfg.llm.dropped_legacy_url = "http://192.168.1.50:11434"
    r = doctor._Report()
    doctor._check_llm(r, cfg)
    out = capsys.readouterr().out
    assert "migrated from an older config" in out
    assert "dropped 'http://192.168.1.50:11434'" in out and "kept server_url = 'http://192.168.1.50:8081'" in out
    cfg.llm.dropped_legacy_url = ""
    doctor._check_llm(doctor._Report(), cfg)
    assert "migrated from an older config" not in capsys.readouterr().out
