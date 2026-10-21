"""Environment diagnostics: `python -m exocortex --doctor` (or scripts/doctor.py).

Checks, in order: Python/packages, GPU (vendor-neutral discovery + VRAM reading, then CUDA-only
checks on an NVIDIA machine), ONNX models, audio devices, screen reading, completion mode, the
LLM backend (route-aware for [llm].backend = "auto"), and overlay support. Prints a report with
OK / WARN / FAIL markers; returns 1 if anything FAILed.

2026-09-25: this file was built and tuned on NVIDIA laptops (RTX 5080/5090); the GPU section and
the LLM backend checks below now also cover AMD (Vulkan/ROCm iGPUs and dGPUs, verified on a Ryzen
AI MAX+ 395 / Radeon 8060S), Intel iGPUs and CPU-only machines, and Ollama as an [llm].backend.
Every probe that depends on another module's not-yet-landed contract (`exocortex.llm.route`,
`exocortex.llm.server.backend_ladder`/`list_devices`/`choose_model`, `exocortex.llm.client.
make_client`, `exocortex.screen.ocr.directml_available`/`describe_ocr_device`) is wrapped so a
missing function degrades to a WARN line instead of crashing the report -- see the module's
`getattr(..., None)` guards throughout.

2026-09-26: Intel Arc DISCRETE GPUs (Alchemist A-series, Battlemage B-series/Pro B60/B70) --
`exocortex.gpu.driver_version`/`intel_arc_notes` -- get a driver-version line per adapter from
`gpu.describe()` (Intel-only: `describe()` gates the clause on `vendor == "intel"`, so NVIDIA and
AMD adapters' lines stay byte-identical to before this branch) and, on an Intel discrete primary,
advisory OK/WARN lines (WARN: an Alchemist/Battlemage driver older than the Xe2 cooperative-matrix
threshold, or a driver that could not be read/parsed at all -- an unknown status is not an OK).
No Intel Arc GPU exists on either dev box this branch was built on, so this path is exercised only
by unit tests with synthetic `GpuInfo`. The llama.cpp section also now reports the chosen binary's
`source` and, for an upstream (non-LM-Studio) runtime, `exocortex.llm.runtimes.describe_runtime`/
`vulkan_env_switches` -- another agent's in-progress module, same `getattr(..., None)` degrade --
plus `[llm].llama_env`, `LlamaServerManager.fallbacks`, and what is already downloaded under
`.tools/llama.cpp`.
"""
from __future__ import annotations

import importlib
import json
import platform
import sys
import time
from pathlib import Path

from exocortex import cuda_env
from exocortex.config import Config

_OK, _WARN, _FAIL = "OK  ", "WARN", "FAIL"


class _Report:
    def __init__(self) -> None:
        self.fails = 0

    def line(self, level: str, text: str) -> None:
        if level == _FAIL:
            self.fails += 1
        print(f"[{level}] {text}", flush=True)

    def section(self, title: str) -> None:
        print(f"\n== {title} ==", flush=True)


def _check_packages(r: _Report) -> None:
    r.section("Python & packages")
    r.line(_OK, f"Python {platform.python_version()} at {sys.executable}")
    for mod, pkg in (("faster_whisper", "faster-whisper"), ("ctranslate2", "ctranslate2"), ("pyaudiowpatch", "pyaudiowpatch"),
                     ("sherpa_onnx", "sherpa-onnx"), ("PySide6", "PySide6"), ("httpx", "httpx"), ("numpy", "numpy"),
                     ("soundfile", "soundfile")):
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            r.line(_OK, f"{pkg} {ver}")
        except Exception as e:  # noqa: BLE001
            r.line(_FAIL, f"{pkg} not importable: {e}  ->  pip install -r requirements.txt")


def _check_hf_cache(r: _Report, cfg: Config) -> None:
    """Whisper model presence in the Hugging Face cache -- vendor-independent, used on every
    machine whether or not it has an NVIDIA GPU."""
    try:
        from huggingface_hub import scan_cache_dir  # type: ignore[import-not-found]

        repos = {r_.repo_id for r_ in scan_cache_dir().repos}
        hit = [x for x in repos if cfg.asr.model.replace("large-v3-turbo", "large-v3-turbo") in x or "whisper" in x]
        if hit:
            r.line(_OK, f"Whisper models in HF cache: {', '.join(sorted(hit))}")
        else:
            r.line(_WARN, f"no Whisper model cached yet - '{cfg.asr.model}' will download on first run (~1.6 GB)")
    except Exception:  # noqa: BLE001
        r.line(_WARN, "could not inspect the Hugging Face cache")


def _check_cuda(r: _Report, cfg: Config) -> None:
    """NVIDIA-only checks -- today's exact wording, unchanged. Called from `_check_gpu` only when
    `exocortex.gpu.gpu_vendor()` is "nvidia"; every other vendor never sees these lines."""
    dirs = cuda_env.setup_cuda_dlls()
    if dirs:
        r.line(_OK, f"NVIDIA pip DLL folders on PATH: {len(dirs)}")
    else:
        r.line(_WARN, "no nvidia/*/bin folders found in site-packages (pip install nvidia-cublas-cu12 nvidia-cudnn-cu12)")
    try:
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        if n > 0:
            types = sorted(ctranslate2.get_supported_compute_types("cuda"))
            r.line(_OK, f"ctranslate2 sees {n} CUDA device(s); compute types: {', '.join(types)}")
        else:
            r.line(_WARN, "ctranslate2 sees no CUDA device - Whisper will run on CPU (slower)")
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"ctranslate2 CUDA probe failed: {e}")
    _check_hf_cache(r, cfg)


def _check_gpu(r: _Report, cfg: Config) -> None:
    """Vendor-neutral GPU discovery (`exocortex.gpu`) + the VRAM reading behind the pressure
    guard, then -- only on an NVIDIA machine -- today's unchanged CUDA checks. On every other
    vendor (AMD, Intel, no GPU) Whisper's CPU path is reported instead, with the thread count
    `exocortex.speech.asr.default_cpu_threads()` will actually use."""
    r.section("GPU")
    try:
        from exocortex import gpu
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"gpu module not importable: {e}")
        return

    try:
        desc = gpu.describe()
        for line in desc.splitlines():
            r.line(_OK, line)
    except Exception as e:  # noqa: BLE001
        r.line(_WARN, f"gpu.describe() failed: {e}")

    vendor = "none"
    try:
        primary = gpu.primary_gpu()
        vendor = gpu.gpu_vendor()
        if primary is not None:
            r.line(_OK, f"primary adapter: {primary.name} [{vendor}]")
            # Intel Arc DISCRETE GPUs only (B570/B580/B770/Pro B60/B70, A-series): advisory notes
            # from exocortex.gpu.intel_arc_notes -- no Intel GPU exists on any dev box this branch
            # was built on, so these are unit-tested with synthetic GpuInfo (tests/test_doctor.py),
            # never exercised live here.
            if vendor == "intel" and not getattr(primary, "uma", False):
                try:
                    driver = gpu.driver_version(primary)
                except Exception:  # noqa: BLE001 - advisory only; a missing driver is not a failure
                    driver = None
                for note in gpu.intel_arc_notes(primary, driver):
                    low = note.lower()
                    needs_attention = "update the intel arc driver" in low or "could not read the driver version" in low
                    level = _WARN if needs_attention else _OK
                    r.line(level, note)
        else:
            r.line(_WARN, f"no primary GPU adapter detected [{vendor}]")
    except Exception as e:  # noqa: BLE001
        r.line(_WARN, f"primary GPU lookup failed: {e}")

    try:
        vram_source_fn = getattr(gpu, "vram_source", None)
        vram_total_fn = getattr(gpu, "vram_total_mb", None)
        source = vram_source_fn() if callable(vram_source_fn) else "none"
        free_mb = gpu.vram_free_mb()
        total_mb = vram_total_fn() if callable(vram_total_fn) else None
        if free_mb is not None:
            extra = f" / {total_mb} MiB total" if total_mb is not None else ""
            r.line(_OK, f"VRAM reading via {source}: {free_mb} MiB free{extra}")
        else:
            r.line(_WARN, "no GPU memory reading on this machine; the GPU pressure guard stays off")
    except Exception as e:  # noqa: BLE001
        r.line(_WARN, f"VRAM reading failed: {e}")

    if vendor == "nvidia":
        _check_cuda(r, cfg)
        return
    # 2026-09-26 (asr-sherpa): engine-aware -- name whichever engine/model resolve_engine() picks
    # (faster-whisper or sherpa-onnx's Parakeet-TDT), instead of assuming Whisper unconditionally.
    try:
        from exocortex.speech.asr import default_cpu_threads, resolve_engine, resolve_sherpa_model

        threads = default_cpu_threads(vendor)
        engine = resolve_engine(cfg.asr)
    except Exception:  # noqa: BLE001
        threads, engine = 4, "faster-whisper"
    if engine == "sherpa":
        model_name = resolve_sherpa_model(cfg.asr)
        r.line(_OK, f"no NVIDIA GPU: CUDA libraries are not needed; ASR engine 'sherpa' runs "
                    f"{model_name} (sherpa-onnx) on the CPU (int8, {threads} threads)")
    else:
        r.line(_OK, f"no NVIDIA GPU: CUDA libraries are not needed; ASR engine 'faster-whisper' "
                    f"runs {cfg.asr.model} on the CPU (int8, {threads} threads)")
    _check_hf_cache(r, cfg)


def _check_models(r: _Report, cfg: Config) -> None:
    r.section("ONNX models (VAD / speaker)")
    d = cfg.models_dir()
    for name in ("silero_vad.onnx", cfg.diarization.model):
        p = d / name
        if p.is_file() and p.stat().st_size > 100_000:
            r.line(_OK, f"{p} ({p.stat().st_size // 1024} kB)")
        else:
            r.line(_WARN, f"{p} missing - it will be downloaded on first run")
    _check_model_library(r, cfg)


def _check_model_library(r: _Report, cfg: Config) -> None:
    """2026-09-27 (portable-install): the app's own GGUF library (plus LM Studio's folder when
    present), the per-model tuning profiles and the hardware report `python -m exocortex.setup`
    wrote -- what a fresh machine needs before the LLM section below can succeed."""
    r.section("Model library (GGUF: own library + LM Studio's folder)")
    try:
        from exocortex.llm import library
    except Exception as e:  # noqa: BLE001
        r.line(_WARN, f"exocortex.llm.library not importable: {e}")
        return
    try:
        roots = library.library_roots(cfg.llm)
        models = library.scan_library(cfg.llm)
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"library scan failed: {e}")
        return
    for source, root in roots:
        r.line(_OK, f"{source}: {root}" + ("" if root.is_dir() else " (not created yet)"))
    by_source: dict[str, int] = {}
    for m in models:
        by_source[m.source] = by_source.get(m.source, 0) + 1
    if models:
        summary = ", ".join(f"{n} from {src}" for src, n in sorted(by_source.items()))
        r.line(_OK, f"{len(models)} model(s): {summary}")
        for m in models[:12]:
            r.line(_OK, f"  {m.key} [{m.quant or '?'}] {m.size_bytes / 1e9:.1f} GB ({m.source})")
        if len(models) > 12:
            r.line(_OK, f"  ... and {len(models) - 12} more")
    else:
        r.line(_WARN, "no GGUF model anywhere yet -- run.bat offers to pull one, or use ... > Models... in the overlay")
    wanted = (cfg.llm.model or "").strip()
    if wanted and wanted.lower() != "auto":
        low = wanted.lower()
        if not any(low in (m.key.lower(), m.name.lower()) for m in models):
            r.line(_WARN, f"[llm].model = {wanted!r} is not in the library; the app picks the best fit automatically")
    try:
        profiles = library.load_profiles(cfg.llm)
        if profiles:
            r.line(_OK, f"{len(profiles)} tuning profile(s) in {library.library_dir(cfg.llm) / library.PROFILES_FILE}: "
                        + ", ".join(sorted(profiles)[:6]))
    except Exception as e:  # noqa: BLE001
        r.line(_WARN, f"profiles.json unreadable: {e}")
    try:
        from exocortex import hardware

        path = hardware.report_path(cfg.llm)
        report = hardware.load(path)
        if report is None:
            r.line(_WARN, f"no hardware report at {path} yet (run.bat / python -m exocortex.setup writes it)")
        else:
            r.line(_OK, f"hardware report {path} ({report.generated_at}): budget {report.budget_mb} MiB, "
                        f"policy {report.recommended_policy}")
    except Exception as e:  # noqa: BLE001
        r.line(_WARN, f"hardware report unreadable: {e}")


def _check_audio(r: _Report, cfg: Config) -> None:
    r.section("Audio devices (WASAPI)")
    try:
        from exocortex.audio import capture
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"capture module not importable: {e}")
        return
    try:
        devices = capture.list_devices()
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"device enumeration failed: {e}")
        return
    loops = [d for d in devices if d.is_loopback]
    mics = [d for d in devices if d.is_input and not d.is_loopback]
    r.line(_OK if loops else _FAIL, f"{len(loops)} loopback device(s): " + "; ".join(f"[{d.index}] {d.name} {d.sample_rate} Hz {d.channels}ch" for d in loops))
    r.line(_OK if mics else _WARN, f"{len(mics)} microphone(s): " + "; ".join(f"[{d.index}] {d.name} {d.sample_rate} Hz" for d in mics))
    for kind, spec in (("system", cfg.audio.system_device), ("mic", cfg.audio.mic_device)):
        try:
            d = capture.resolve_device(kind, spec)  # type: ignore[arg-type]
            r.line(_OK, f"{kind} -> [{d.index}] {d.name} ({d.sample_rate} Hz, {d.channels} ch)")
        except Exception as e:  # noqa: BLE001
            r.line(_FAIL, f"{kind} device '{spec}' not resolvable: {e}")


def _timed_completion(r: _Report, client: "object") -> None:
    from exocortex.llm.prompts import NO_THINK_PREFIX, STOP, render_chatml

    prompt = render_chatml("You are a concise assistant.", "Reply with the single word: ready", NO_THINK_PREFIX)
    try:
        t0 = time.perf_counter()
        text = "".join(client.stream_completion(prompt, max_tokens=24, stop=STOP))  # type: ignore[attr-defined]
        dt = time.perf_counter() - t0
        st = client.last_stats  # type: ignore[attr-defined]
        ttft = f"{st.ttft_s:.2f} s" if st else "?"
        tps = f"{st.tok_per_s:.0f} tok/s" if st else "?"
        r.line(_OK, f"completion in {dt:.2f} s (ttft {ttft}, {tps}): {text.strip()[:60]!r}")
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"test completion failed: {e}")


def _check_llm_llamacpp(r: _Report, cfg: Config) -> None:
    """The llamacpp backend: the chosen runtime pack, the backend ladder for this machine's GPU
    vendor, `--list-devices`, and the model `choose_model` (or the older `resolve_model`) picks.

    `exocortex.llm.server.backend_ladder` / `list_devices` / `choose_model`, and the `vendor` /
    `backend` keyword args of `find_server_binary`, are another agent's in-progress contract
    (SPEC.md): every one of them is looked up with `getattr(..., None)` / a `TypeError` fallback
    so this check degrades to a WARN line, never a crash, until they land.

    2026-09-26: also reports `ServerBinary.source` ("lmstudio" | "upstream", another agent's
    in-progress field) and, for an upstream runtime, `exocortex.llm.runtimes.describe_runtime()` /
    `vulkan_env_switches()` (that whole module is in progress -- same degrade-to-WARN pattern),
    `[llm].llama_env` when set, `LlamaServerManager.fallbacks` after `start()`, what is already
    under `project_root()/.tools/llama.cpp`, and how to opt into an upstream runtime.
    """
    r.section("LLM backend: llama.cpp (llama-server spawned by ExoCortex)")
    try:
        from exocortex.llm.client import LMStudioClient
        from exocortex.llm import server as llm_server
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"llm modules not importable: {e}")
        return
    try:
        from exocortex import gpu

        vendor = gpu.gpu_vendor()
    except Exception:  # noqa: BLE001
        vendor = "none"

    try:
        try:
            b = llm_server.find_server_binary(cfg.llm.llama_server_path, vendor=vendor, backend=cfg.llm.llama_backend)
        except TypeError:
            b = llm_server.find_server_binary(cfg.llm.llama_server_path)
        framework = getattr(b, "framework", None) or b.flavor
        reason = getattr(b, "reason", "")
        r.line(_OK, f"llama-server: {b.exe} ({b.flavor}, framework {framework}, runtime {b.version})"
                    + (f" -- {reason}" if reason else ""))
        source = getattr(b, "source", None)
        if source:
            r.line(_OK, f"binary source: {source}")
        else:
            r.line(_WARN, "ServerBinary.source not available yet (exocortex.llm.server, in progress)")
        if vendor == "nvidia":
            if b.flavor not in ("nvidia-cuda12", "nvidia-cuda", "custom"):
                r.line(_WARN, f"runtime flavor '{b.flavor}' is not a CUDA build - install LM Studio's CUDA llama.cpp runtime for GPU speed")
        elif vendor != "none" and b.flavor in ("cpu", "avx2", "other"):
            r.line(_WARN, "the CPU llama.cpp runtime was chosen -- install LM Studio's Vulkan (AMD/Intel) "
                          "llama.cpp runtime for GPU speed")
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"llama-server binary: {e}")
        return

    # Upstream llama.cpp runtimes (exocortex.llm.runtimes, another agent's in-progress module):
    # self-contained builds downloaded from github.com/ggml-org/llama.cpp releases, meant for a
    # GPU framework LM Studio's own runtime catalog does not package (SYCL/OpenVINO on Intel Arc
    # today; see SPEC.md). `describe_fn`/`switches_fn` degrade to a WARN line until that module
    # lands -- exactly the getattr(..., None) pattern the rest of this function already uses for
    # exocortex.llm.server's in-progress pieces.
    try:
        from exocortex.llm import runtimes as runtimes_mod
    except Exception:  # noqa: BLE001 - module does not exist yet on this branch
        runtimes_mod = None
    describe_fn = getattr(runtimes_mod, "describe_runtime", None) if runtimes_mod else None
    switches_fn = getattr(runtimes_mod, "vulkan_env_switches", None) if runtimes_mod else None

    # describe_runtime()/vulkan_env_switches() both take the SAME folder: the one holding
    # llama-server.exe (and, for a Vulkan build, ggml-vulkan*.dll next to it) -- exe.parent for a
    # ServerBinary, upstream or not (an LM Studio pack's ggml-vulkan*.dll lives there too, so this
    # line is informative even when source != "upstream").
    folder = b.exe.parent if getattr(b, "exe", None) else None
    if source == "upstream":
        if callable(describe_fn) and folder is not None:
            try:
                r.line(_OK, describe_fn(folder))
            except Exception as e:  # noqa: BLE001
                r.line(_WARN, f"describe_runtime() failed: {e}")
        else:
            r.line(_WARN, "exocortex.llm.runtimes.describe_runtime() not available yet (in progress)")

    if callable(switches_fn) and folder is not None:
        try:
            switches = switches_fn(folder)
            if switches:
                r.line(_OK, f"ggml-vulkan.dll knows: {', '.join(switches)}")
            else:
                r.line(_WARN, "no GGML_VK_* env switches found (no ggml-vulkan*.dll in this runtime, or none named this way)")
        except Exception as e:  # noqa: BLE001
            r.line(_WARN, f"vulkan_env_switches() failed: {e}")
    else:
        r.line(_WARN, "exocortex.llm.runtimes.vulkan_env_switches() not available yet (in progress)")

    llama_env = getattr(cfg.llm, "llama_env", None)
    if llama_env:
        r.line(_OK, f"[llm].llama_env: {llama_env}")

    # What is already downloaded under .tools/llama.cpp, one level down from build/kind
    # (exocortex.llm.runtimes.runtime_dir(root, kind, build) == <root>/<build>/<kind>): every
    # <build>/<kind> pair recognised as a runtime kind (RUNTIME_KINDS when the module is present,
    # else the fixed list this branch's research settled on) is reported via describe_runtime()
    # when available, else a plain "<kind> build <build>" derived straight from the path.
    try:
        from exocortex.config import project_root

        tools_dir = project_root() / ".tools" / "llama.cpp"
        kinds = getattr(runtimes_mod, "RUNTIME_KINDS", None) or ("vulkan", "sycl", "openvino", "cpu")
        installed = (sorted(p for p in tools_dir.glob("*/*") if p.is_dir() and p.name in kinds)
                     if tools_dir.is_dir() else [])
        if installed:
            for d in installed:
                info = None
                if callable(describe_fn):
                    try:
                        info = describe_fn(d)
                    except Exception:  # noqa: BLE001
                        info = None
                if not info:
                    info = f"{d.name} build {d.parent.name}"
                r.line(_OK, f"upstream runtime installed: {info}")
        else:
            r.line(_WARN, f"no upstream llama.cpp runtimes under {tools_dir} yet")
    except Exception as e:  # noqa: BLE001
        r.line(_WARN, f"could not list upstream runtimes: {e}")
    r.line(_OK, "opt in to an upstream runtime (e.g. SYCL/OpenVINO for Intel Arc) with "
                "[llm].llama_backend = 'sycl' and [llm].llama_source = 'upstream'")

    ladder_fn = getattr(llm_server, "backend_ladder", None)
    if callable(ladder_fn):
        try:
            ladder = ladder_fn(vendor, cfg.llm.llama_backend)
            r.line(_OK, f"backend ladder for {vendor} ({cfg.llm.llama_backend}): {' > '.join(ladder)}")
        except Exception as e:  # noqa: BLE001
            r.line(_WARN, f"backend_ladder() failed: {e}")
    else:
        r.line(_WARN, "backend_ladder() not available yet (exocortex.llm.server, in progress)")

    list_devices_fn = getattr(llm_server, "list_devices", None)
    if callable(list_devices_fn):
        try:
            devices = list_devices_fn(b)
            if devices:
                for d in devices:
                    r.line(_OK, f"  --list-devices: {d.name} - {d.description} "
                                f"({d.free_mb} MiB free / {d.total_mb} MiB)")
            else:
                r.line(_WARN, "--list-devices returned no devices")
        except Exception as e:  # noqa: BLE001
            r.line(_WARN, f"--list-devices failed: {e}")
    else:
        r.line(_WARN, "list_devices() not available yet (exocortex.llm.server, in progress)")

    md = llm_server.lmstudio_models_dir(cfg.llm)
    models = llm_server.scan_models(md)
    r.line(_OK if models else _FAIL, f"model library {md}: {len(models)} GGUF model(s)")
    for m in models[:15]:
        r.line(_OK, f"  {m.display}")

    choose_model_fn = getattr(llm_server, "choose_model", None)
    m = None
    try:
        if callable(choose_model_fn):
            try:
                primary = gpu.primary_gpu()
                budget_mb = primary.budget_mb if primary is not None else 0
            except Exception:  # noqa: BLE001
                budget_mb = 0
            m, note = choose_model_fn(cfg.llm.model, md, budget_mb)
            r.line(_OK, f"[llm].model '{cfg.llm.model}' -> {m.path}" + (f" ({note})" if note else ""))
        else:
            m = llm_server.resolve_model(cfg.llm.model, md)
            r.line(_OK, f"[llm].model '{cfg.llm.model}' -> {m.path}")
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"[llm].model '{cfg.llm.model}': {e}")
        return

    media_fn = getattr(llm_server, "media_kind", None)
    if callable(media_fn) and m is not None:
        try:
            kind, fs, root = media_fn(m.path)
            est = llm_server.estimate_startup_timeout_s(cfg.llm.llama_startup_timeout_s, m.size_bytes, kind)
            slow = kind in ("removable", "remote", "cdrom")
            r.line(_WARN if slow else _OK,
                   f"model volume {root or '?'} is {kind}{f' ({fs})' if fs else ''}; startup deadline "
                   f"{est:.0f} s ([llm].llama_startup_timeout_s = {cfg.llm.llama_startup_timeout_s:.0f})"
                   + (" -- a cold load of this file can take minutes; an internal drive loads it in seconds" if slow else ""))
        except Exception as e:  # noqa: BLE001
            r.line(_WARN, f"media probe failed: {e}")

    import tempfile

    log_dir = Path(tempfile.mkdtemp(prefix="exocortex-doctor-"))
    mgr = llm_server.LlamaServerManager(cfg.llm, log_dir=log_dir)
    try:
        t0 = time.perf_counter()
        url = mgr.start()
        args = mgr.build_args(m, 1)
        threads = args[args.index("--threads") + 1] if "--threads" in args else "?"
        r.line(_OK, f"spawned llama-server at {url} in {time.perf_counter() - t0:.1f} s "
                    f"(threads {threads}, ctx {cfg.llm.ctx_size}, slots {cfg.llm.parallel})")
        device_used = getattr(mgr, "device", None)
        note_used = getattr(mgr, "note", None)
        if device_used:
            r.line(_OK, f"device: {device_used}")
        if note_used:
            r.line(_OK, f"note: {note_used}")
        for w in getattr(mgr, "warnings", None) or []:
            r.line(_WARN, f"llama-server: {w}")
        for fb in getattr(mgr, "fallbacks", None) or []:
            r.line(_WARN, f"runtime pack skipped: {fb}")
        client = LMStudioClient(cfg.llm)
        client.set_base_url(url)
        _timed_completion(r, client)
        client.close()
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"llama-server failed to start: {e}")
    finally:
        mgr.stop()


def _check_llm_ollama(r: _Report, cfg: Config) -> None:
    """The ollama backend, via `exocortex.llm.client.make_client(cfg.llm, "ollama")` -- another
    agent's in-progress contract; a WARN line stands in until it lands."""
    ollama_url = cfg.llm.effective_server_url("ollama")
    r.section(f"LLM backend: Ollama at {ollama_url}")
    try:
        from exocortex.llm import client as llm_client
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"llm client not importable: {e}")
        return
    make_client = getattr(llm_client, "make_client", None)
    if not callable(make_client):
        r.line(_WARN, "make_client() not available yet (exocortex.llm.client, in progress); cannot probe Ollama")
        return
    try:
        client = make_client(cfg.llm, "ollama")
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"could not build an Ollama client: {e}")
        return
    try:
        h = client.health()
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"health check raised: {e}")
        return
    if not h.reachable:
        r.line(_FAIL, f"Ollama not reachable at {ollama_url}: {h.detail}")
        return
    r.line(_OK, f"Ollama reachable at {ollama_url}")
    try:
        models = client.list_models()
        ids = []
        for mdl in models:
            size = getattr(mdl, "size_bytes", None)
            ids.append(f"{mdl.id} ({size / 1e9:.1f} GB)" if size else mdl.id)
        r.line(_OK, f"{len(ids)} model(s): {', '.join(ids[:12])}{' ...' if len(ids) > 12 else ''}")
    except Exception as e:  # noqa: BLE001
        r.line(_WARN, f"could not list models: {e}")
    resolve_fn = getattr(client, "resolve_model", None)
    if callable(resolve_fn):
        try:
            chosen = resolve_fn(cfg.llm.model)
            r.line(_OK, f"[llm].model '{cfg.llm.model}' -> {chosen}")
        except Exception as e:  # noqa: BLE001
            r.line(_WARN, f"model resolution failed: {e}")
    _timed_completion(r, client)
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        close_fn()


def _check_llm_openai(r: _Report, cfg: Config) -> None:
    base_url = cfg.llm.effective_server_url("openai")
    r.section(f"LLM backend: OpenAI-compatible server at {base_url}")
    if "localhost" in base_url:
        r.line(_WARN, "server_url uses 'localhost' - use 127.0.0.1 to avoid a ~2 s IPv6 fallback per connection")
    try:
        from exocortex.llm.client import LMStudioClient
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"llm client not importable: {e}")
        return
    client = LMStudioClient(cfg.llm)
    try:
        h = client.health()
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"health check raised: {e}")
        return
    if not h.reachable:
        r.line(_FAIL, f"server not reachable at {base_url}: {h.detail}")
        return
    r.line(_OK, f"server reachable at {base_url}")
    try:
        ids = [m.id for m in client.list_models()]
        r.line(_OK, f"{len(ids)} model(s): {', '.join(ids[:12])}{' ...' if len(ids) > 12 else ''}")
    except Exception as e:  # noqa: BLE001
        r.line(_WARN, f"could not list models: {e}")
    if not h.model_listed:
        r.line(_FAIL, f"model '{cfg.llm.model}' is not served there (check [llm].model in config.toml)")
        return
    if h.model_loaded is False:
        r.line(_WARN, f"model '{cfg.llm.model}' is not loaded yet - the first request will load it")
    else:
        r.line(_OK, f"model '{cfg.llm.model}' available")
    _timed_completion(r, client)
    client.close()


_LLM_DISPATCH = {"llamacpp": _check_llm_llamacpp, "ollama": _check_llm_ollama, "openai": _check_llm_openai}


def _check_llm(r: _Report, cfg: Config) -> None:
    """[llm].backend "auto": resolve the route first (`exocortex.llm.route.resolve_route`, another
    agent's in-progress module -- a WARN + a fallback to "llamacpp" stands in until it lands) and
    dispatch on it; any other backend value dispatches directly, as before."""
    dropped = getattr(cfg.llm, "dropped_legacy_url", "")
    if dropped:
        # review round two, 2026-09-27: an old config.toml customised BOTH base_url and ollama_url;
        # only one fits the single server_url, and the other would otherwise vanish unnoticed.
        r.section("LLM server address (migrated from an older config)")
        r.line(_WARN, f"the old config set two server addresses; kept server_url = {cfg.llm.server_url!r}, "
                      f"dropped {dropped!r}. To use the dropped one, set [llm].backend explicitly "
                      f'("ollama" or "openai") and server_url = "{dropped}".')
    backend = cfg.llm.backend.lower()
    if backend != "auto":
        fn = _LLM_DISPATCH.get(backend)
        if fn is None:
            r.section("LLM backend")
            r.line(_FAIL, f"unknown [llm].backend '{cfg.llm.backend}'")
            return
        fn(r, cfg)
        return
    try:
        from exocortex.llm import route
    except Exception as e:  # noqa: BLE001
        r.section("LLM backend (auto route)")
        r.line(_WARN, f"exocortex.llm.route not available yet ({e}); falling back to backend 'llamacpp'")
        _check_llm_llamacpp(r, cfg)
        return
    r.section("LLM backend (auto route)")
    try:
        rt = route.resolve_route(cfg.llm)
        r.line(_OK, route.describe_route(rt))
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"route resolution failed: {e}")
        return
    if rt.backend == "none":
        r.line(_FAIL, rt.reason)
        return
    fn = _LLM_DISPATCH.get(rt.backend)
    if fn is None:
        r.line(_FAIL, f"resolved route backend '{rt.backend}' is not one this doctor knows how to check")
        return
    fn(r, cfg)


def _check_overlay(r: _Report, cfg: Config) -> None:
    r.section("Overlay")
    if sys.platform != "win32":
        r.line(_WARN, "not Windows - global hotkeys are Windows-only")
        return
    r.line(_OK, "visible copilot: the overlay is a normal always-on-top window, shown in the "
                "taskbar and captured by screen sharing like any other app")
    try:
        from exocortex.ui import win32  # noqa: F401

        r.line(_OK, "win32 helpers importable")
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"win32 helpers not importable: {e}")


def _check_screen(r: _Report, cfg: Config) -> None:
    r.section("Screen reading (capture + OCR)")
    if cfg.screen.mode == "off":
        r.line(_WARN, "screen reading is off ([screen].mode)")
    try:
        from exocortex.screen import capture, foreground, ocr
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"screen modules not importable: {e}")
        return
    try:
        mons = foreground.monitor_rects()
        r.line(_OK if mons else _FAIL, f"{len(mons)} monitor(s): " + "; ".join(f"{w}x{h}@({x},{y})" for x, y, w, h in mons))
        fg = foreground.foreground_window()
        r.line(_OK, f"foreground window: {fg.title[:50]!r} {fg.rect}" if fg else "foreground window: none (desktop)")
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"screen enumeration failed: {e}")
    # OpenVINO devices for RapidOCR (the iGPU / NPU path; see exocortex/screen/ocr.py), plus
    # DirectML (the AMD/Intel iGPU fallback when OpenVINO sees no matching device).
    try:
        devices = ocr.openvino_devices()
        try:
            from exocortex import gpu as gpu_mod

            gpus = gpu_mod.detect_gpus()
        except Exception:  # noqa: BLE001
            gpus = None
        try:
            kind, ov_device = ocr.resolve_ocr_device(cfg.screen.ocr_device, devices, gpus=gpus)
        except TypeError:
            # resolve_ocr_device() without the gpus= kwarg yet (exocortex.screen.ocr, in progress).
            kind, ov_device = ocr.resolve_ocr_device(cfg.screen.ocr_device, devices)
        if devices:
            r.line(_OK, "OpenVINO devices: " + "; ".join(f"{d} = {n}" for d, n in devices.items()))
        else:
            r.line(_WARN, "OpenVINO not installed or no devices: RapidOCR runs on the CPU via onnxruntime "
                          "(pip install openvino for the iGPU / NPU)")
        describe_fn = getattr(ocr, "describe_ocr_device", None)
        if callable(describe_fn):
            r.line(_OK, describe_fn(kind, ov_device))
        else:
            r.line(_OK, f"[screen].ocr_device = {cfg.screen.ocr_device} -> {kind}" + (f" ({ov_device})" if ov_device else " (onnxruntime)"))
        dml_fn = getattr(ocr, "directml_available", None)
        if callable(dml_fn):
            try:
                r.line(_OK, f"DirectML available: {dml_fn()}")
            except Exception as e:  # noqa: BLE001
                r.line(_WARN, f"DirectML probe failed: {e}")
        else:
            r.line(_WARN, "directml_available() not available yet (exocortex.screen.ocr, in progress)")
    except Exception as e:  # noqa: BLE001
        r.line(_WARN, f"OpenVINO probe failed: {e}")
    # Each engine is probed in its own subprocess: creating RapidOCR AFTER Windows OCR (WinRT) in
    # the same process crashes the interpreter (verified on this machine; the app only ever builds
    # RapidOCR first), and a failing accelerator must not take the report down.
    probe = (
        "import sys, time, numpy as np\n"
        "from PIL import Image, ImageDraw\n"
        "sys.path.insert(0, %r)\n"
        "from exocortex.screen import ocr\n"
        "img = Image.new('RGB', (900, 200), 'white'); d = ImageDraw.Draw(img)\n"
        "try:\n    from PIL import ImageFont; font = ImageFont.truetype('arial.ttf', 30)\n"
        "except Exception:\n    font = None\n"
        "d.text((20, 40), 'ExoCortex screen reading check: budget 1.2M', fill='black', font=font)\n"
        "bgr = np.ascontiguousarray(np.array(img)[:, :, ::-1])\n"
        "t0 = time.perf_counter(); eng = ocr.create_engine(%r, device=%r, cache_dir=%r); init = (time.perf_counter() - t0) * 1000\n"
        "eng.recognize(bgr)\n"
        "res = eng.recognize(bgr); text = ocr.lines_to_text(res.lines)\n"
        "print('%%.0f|%%.0f|%%s|%%s' %% (res.elapsed_ms, init, ocr.engine_label(eng), text.replace(chr(10), ' ')[:60]))\n"
    )
    import subprocess

    root = str(Path(__file__).resolve().parent.parent)
    cache_dir = str(cfg.models_dir() / "ov_cache")
    for name in ("windows", "rapidocr"):
        try:
            out = subprocess.run([sys.executable, "-c", probe % (root, name, cfg.screen.ocr_device, cache_dir)],
                                 capture_output=True, text=True, timeout=300,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if out.returncode != 0:
                err = (out.stderr.strip().splitlines() or ["(no output)"])[-1]
                r.line(_FAIL if name == cfg.screen.ocr_engine else _WARN, f"OCR '{name}' unavailable: {err[:160]}")
                continue
            ms, init, label, text = out.stdout.strip().split("|", 3)
            ok = "budget" in text.lower()
            r.line(_OK if ok else _WARN, f"OCR '{label}': {ms} ms (+{init} ms init incl. warm-up) -> {text!r}")
        except Exception as e:  # noqa: BLE001
            r.line(_WARN, f"OCR '{name}' probe failed: {e}")
    r.line(_OK, f"configured engine: {cfg.screen.ocr_engine} (auto = Windows OCR if available, else rapidocr); "
                f"hands-free accurate re-reads: {'rapidocr' if cfg.handsfree.accurate_ocr else 'off'}")


def _check_completion(r: _Report, cfg: Config) -> None:
    """Completion mode: UI Automation / MSAA caret tracking (reports what has focus, never its text)."""
    r.section("Completion mode (caret tracking)")
    try:
        import comtypes  # noqa: F401
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"comtypes not importable: {e}  ->  pip install comtypes")
        return
    try:
        from exocortex.screen.caret import CaretTracker
    except Exception as e:  # noqa: BLE001
        r.line(_FAIL, f"caret module not importable: {e}")
        return
    tracker = CaretTracker(cfg.completion.prefix_chars, cfg.completion.suffix_chars)
    t0 = time.perf_counter()
    ok = tracker.attach()
    ms = (time.perf_counter() - t0) * 1000
    if not ok:
        r.line(_FAIL, f"caret tracking unavailable: {tracker.detail}")
        return
    r.line(_OK, f"{tracker.detail} (init {ms:.0f} ms)")
    t1 = time.perf_counter()
    info = tracker.probe()
    ms = (time.perf_counter() - t1) * 1000
    if info is None:
        r.line(_OK, f"focused control is not a text field right now ({tracker.last_reject or 'nothing focused'}; probe {ms:.1f} ms)")
    else:
        r.line(_OK, f"focused text field: {info.app} {info.field_name!r} via {info.source}, caret at ({info.x},{info.y}) "
                    f"h={info.h}px, font {info.font_pt or '?'}pt {info.font_name!r} (probe {ms:.1f} ms)")
    tracker.detach()
    r.line(_OK, f"debounce {cfg.completion.debounce_ms} ms, poll {cfg.completion.poll_ms} ms, "
                f"max {cfg.completion.max_tokens} tokens / {cfg.completion.max_lines} lines"
                + (" - starts in completion mode" if cfg.completion.enabled else ""))


def _check_context(r: _Report, cfg: Config) -> None:
    """2026-10-20: the repository root ([context].repo_root), the Claude Code toolkit discovered
    in it (exocortex.context.toolkit) and the dashboard's tracking settings."""
    r.section("Context: repository, Claude Code toolkit, dashboard")
    c, d = cfg.context, cfg.dashboard
    explicit = (c.repo_root or "").strip()
    try:
        from exocortex.context.detect import recent_projects
        from exocortex.context.provider import ContextProvider
        from exocortex.context.toolkit import find_toolkit_dir

        provider = ContextProvider(cfg)
        root = provider.repo_root
        if root is None:
            if explicit:
                r.line(_FAIL, f"[context].repo_root does not exist: {explicit}")
            elif not c.repo_auto:
                r.line(_WARN, "no repository: [context].repo_root is empty and auto-detection is off -- no "
                              "repository lookups and no toolkit discovery (Settings > Context)")
            else:
                r.line(_WARN, f"no repository: [context].repo_root is empty and Claude Code has not run in a project "
                              f"in the last {c.repo_auto_max_age_h} h (View > Repository, or Settings > Context)")
        else:
            how = "[context].repo_root" if provider.repo_source == "config" else "auto-detected from Claude Code's session logs"
            r.line(_OK, f"repo root: {root} ({how}; lookups {'on' if c.repo_search else 'off'}: "
                        f"{c.repo_max_files} files / {c.repo_max_chars} chars per lookup, on mention only)")
        if c.repo_auto and not explicit:
            others = [p for p, _at in recent_projects(max_age_s=max(0, int(c.repo_auto_max_age_h)) * 3600.0, limit=4)
                      if root is None or p != root]
            if others:
                r.line(_OK, "  other recent Claude Code projects: " + ", ".join(str(p) for p in others[:3]))
        cat = provider.catalog
        if not c.toolkit:
            r.line(_WARN, "toolkit context off ([context].toolkit = false)")
        elif cat.is_empty:
            where = f"{root}/.claude" if root is not None else "(no repository)"
            r.line(_WARN, f"no Claude Code toolkit found under {where}; install one in the repo with "
                          "`node \"$HOME/claude-toolkit/install.mjs\" --set live`, or hand-copy skills into .claude/skills")
        else:
            extras = []
            if cat.loop:
                extras.append(f"working loop: {len(cat.loop)} rules")
            if cat.commands:
                extras.append("repo commands: " + ", ".join(k for k, _c in cat.commands))
            r.line(_OK, f"toolkit: {cat.summary() or 'house rules only'}" + (f" ({'; '.join(extras)})" if extras else ""))
            for kind in ("skill", "agent", "workflow"):
                names = [p.command for p in cat.by_kind(kind)]
                if names:
                    r.line(_OK, f"  {kind}s: " + ", ".join(names))
            example = cat.example_command()
            if example:
                r.line(_OK, f"  suggested prompts open with its commands, e.g. {example}")
        clone = find_toolkit_dir(c.toolkit_dir)
        if clone is not None:
            r.line(_OK, f"toolkit clone: {clone} (richer part descriptions)")
        else:
            r.line(_WARN, "toolkit clone not found (~/claude-toolkit, ~/.claude/dev-toolkit, [context].toolkit_dir): "
                          "descriptions come from the installed files only -- fine")
    except Exception as e:  # noqa: BLE001
        r.line(_WARN, f"toolkit discovery failed: {e}")
    r.line(_OK, f"dashboard: tracking {'on' if d.track else 'off'} (a refresh at most every {d.refresh_interval_s} s "
                f"after {d.min_new_words} new words), view {'on' if d.enabled else 'off'} at start "
                f"({cfg.hotkeys.toggle_dashboard or 'no hotkey'}), items {'persisted' if d.persist else 'in memory'}")


def run_doctor(cfg: Config) -> int:
    r = _Report()
    print("ExoCortex doctor")
    print(f"config: {cfg.path or '(defaults)'}  models: {cfg.models_dir()}  sessions: {cfg.sessions_dir()}")
    _check_packages(r)
    _check_gpu(r, cfg)
    _check_models(r, cfg)
    _check_audio(r, cfg)
    _check_screen(r, cfg)
    _check_completion(r, cfg)
    _check_context(r, cfg)
    _check_llm(r, cfg)
    _check_overlay(r, cfg)
    print()
    print("All good." if r.fails == 0 else f"{r.fails} problem(s) found.")
    return 0 if r.fails == 0 else 1


def as_json(cfg: Config) -> str:
    """Machine-readable subset for bug reports."""
    data: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "models_dir": str(cfg.models_dir()),
    }
    try:
        from exocortex import gpu

        data["gpu_vendor"] = gpu.gpu_vendor()
        data["gpus"] = [g.name for g in gpu.detect_gpus()]
    except Exception:  # noqa: BLE001
        data["gpu_vendor"] = "unknown"
        data["gpus"] = []
    try:
        from exocortex.llm import route

        data["route"] = route.resolve_route(cfg.llm).backend
    except Exception:  # noqa: BLE001
        data["route"] = cfg.llm.backend
    return json.dumps(data)


if __name__ == "__main__":  # pragma: no cover
    from exocortex.config import load_config

    sys.exit(run_doctor(load_config()))
