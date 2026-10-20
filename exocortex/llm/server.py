"""Spawn and supervise a private llama.cpp `llama-server` for the LLM backend.

LM Studio is used purely as a *library*: its downloaded GGUF files (``downloadsFolder`` in
``~/.lmstudio/settings.json``, e.g. ``E:\\AI\\models``) and its llama.cpp runtime packs under
``~/.lmstudio/extensions/backends/llama.cpp-win-x86_64-<flavor>-avx2-<ver>/llama-server.exe``. No
LM Studio process has to run: ExoCortex starts ``llama-server`` itself, hidden, on 127.0.0.1 and a
private port, inside a Windows Job Object so it dies with the app.

This module used to assume every machine has an NVIDIA GPU and a CUDA runtime pack (that
assumption is what this branch removes -- never what a machine WITHOUT one falls back to, which
stays the same before and after). Three choices are now vendor-aware, each with its own function
so it can be unit-tested without hardware:

* **Which runtime pack** (:func:`backend_ladder`, :func:`_scan_packs`, :func:`find_server_binary`)
  -- LM Studio installs one pack per GPU framework side by side (CUDA / CUDA 12 / ROCm / Vulkan /
  CPU-only "avx2"), and the right one depends on the machine's GPU vendor, never on which pack
  happens to sort last alphabetically. NVIDIA's ladder (CUDA 12 > CUDA > Vulkan > CPU) is
  UNCHANGED from before this branch: this is the hard constraint the branch was reviewed against,
  because an RTX laptop's tuning must not move. AMD's ladder puts Vulkan *ahead* of ROCm: Vulkan
  runs on every AMD GPU (the 780M and 8060S iGPUs included) while the ROCm pack only covers the
  ``gpu.targets`` its manifest lists, so Vulkan is the one that never leaves a machine stranded.
  Picking the wrong pack is not slow, it is a crash: LM Studio's CUDA pack's ``llama-server.exe``
  exits with ``0xC0000135`` (DLL not found) on a machine with no NVIDIA driver -- that was the bug
  a vendor-blind "prefer CUDA" picker produced here, and it is what this ladder fixes.
* **Which GPU device** (:func:`parse_list_devices`, :func:`list_devices`, :func:`choose_device`)
  -- a framework can see more than one adapter (a laptop's discrete + integrated GPU under
  Vulkan); ``--device`` picks the one :func:`exocortex.gpu.primary_gpu` would run the model on.
  NVIDIA behaviour is unchanged here too: CUDA builds are never given a ``--device`` flag.
* **Which model** (:func:`pick_default_model`, :func:`choose_model`) -- ``[llm].model = "auto"``
  picks the largest GGUF in the library that leaves room for the KV cache and activations within
  the primary GPU's memory budget (:attr:`exocortex.gpu.GpuInfo.budget_mb`), falling back to the
  smallest model when nothing fits. The default stays the named ``maple-preview`` (a fresh NVIDIA
  install loads exactly what it loaded before); a configured name that is not in the library --
  ``maple-preview`` on this AMD box included -- falls back to the automatic choice with a warning
  instead of failing the start.
* **How long to wait** (:func:`media_kind`, :func:`estimate_startup_timeout_s`) -- the startup
  deadline is the configured ``llama_startup_timeout_s`` or, if longer, a cold read of the GGUF at
  the volume's read floor plus 60 s: a 16.8 GB model on the exFAT card in this box's card reader
  measured a 323 s cold load (2026-09-26), well past the 180 s default that stands on a fixed disk.

Verified on this machine (2026-09-20, RTX 5080/5090 laptops): maple-preview loads in ~3 s, 60 ms
time-to-first-token on a cached prefix, ~62 tok/s. The TQ1_0 ternary tensors run on the CPU
(llama.cpp has no CUDA kernels for them yet), so ``threads`` matters for that model.

Verified on 2026-09-25 on the Ryzen AI MAX+ 395 / Radeon 8060S (no NVIDIA driver on this box):
``find_server_binary()`` with no arguments picks the Vulkan pack ``llama.cpp-win-x86_64-vulkan-
avx2-2.46.0`` (the newest of five installed Vulkan versions; reason "vendor amd -> vulkan (first
choice)"); ``list_devices()`` on it reports exactly 1 device (``Vulkan0: AMD Radeon(TM) 8060S
Graphics``, 48971/46523 MiB total/free) and the ROCm pack also reports exactly 1 device
(``ROCm0``, 44920/44765 MiB) -- with only one device on either framework, :func:`choose_device`
adds no ``--device`` flag, exactly as a single-GPU NVIDIA laptop gets none today.
``choose_model("auto", lmstudio_models_dir(), budget_mb=48793)`` picked
``lmstudio-community/Qwen3.8-27B`` (Qwen3.8-27B-Q4_K_M.gguf, 16.8 GB) over the complete but
slightly smaller ``Muse-Glimmer-30B-KQuant-17GB-Q4_K_M`` (16.75 GB) and the 1-bit
``Bonsai-27B-Q1_0`` (3.5 GB) -- the largest file that still clears the 0.75x budget margin. A live
``LlamaServerManager(LLMConfig()).start()`` on that Vulkan pack with that model: ready in 102.4 s
(mmap + Vulkan buffer upload of 16.8 GB dominate; ``model loaded`` in the log at 1:42), ``/health``
200, one ``/v1/completions`` (20 tokens) returned coherent (non-garbage) output in 5.7 s at
~1.8 tok/s prompt-eval / ~11.7 tok/s generation -- no CUDA kernels here, so this is Vulkan on a
UMA iGPU, not a discrete GPU's numbers. ``stop()`` in a ``finally`` left no llama-server.exe
process behind (checked with ``tasklist``).

2026-09-26 (intel-integration): Intel Arc DISCRETE GPUs (A770/A750 "Alchemist", B580/B570
"Battlemage", Pro B70) with no local hardware to test against. Four additions, none of which move
the NVIDIA or AMD paths above:

* **A second runtime source.** LM Studio's own runtime catalog ships CUDA/ROCm/Vulkan/CPU packs
  only -- no SYCL or OpenVINO pack exists there for Intel Arc, and the llama.cpp community's
  current recommendation for Arc is Vulkan anyway (SYCL lags as of 2026-07). :func:`find_server_binary`
  gained ``source`` ("auto" | "lmstudio" | "upstream") and now consults :mod:`exocortex.llm.runtimes`
  -- self-contained Windows zips fetched straight from ``github.com/ggml-org/llama.cpp`` releases --
  as a second source: "upstream" fetches only (never CUDA/ROCm: NVIDIA and AMD keep their tuned LM
  Studio packs); "auto" tries LM Studio first and only fetches upstream when no LM Studio pack
  exists for *any* GPU flavor of the vendor's ladder (or every one already failed this run),
  before finally dropping to the CPU pack. New flavors "intel-sycl" (upstream kind "sycl") and
  "openvino" are explicit opt-ins via ``[llm].llama_backend`` -- the Intel ladder itself stays
  ``vulkan, cpu`` (Vulkan is the community-recommended default, not SYCL/OpenVINO).
* **The quit-during-load race** (docs/UPSTREAM_SUGGESTIONS.md finding #1): a size-scaled startup
  deadline up to ~10 minutes made this worse, not better. `start()`/`_attempt()` no longer hold
  `self._lock` across the blocking `_wait_ready()` poll; a `threading.Event` (`self._stop_event`)
  lets a concurrent `stop()` interrupt the wait within one poll tick (0.25 s) instead of the whole
  timeout, and `start()` then raises `LlamaServerError("stopped")` promptly instead of a bogus
  "did not become ready" timeout, without trying the next runtime pack. `stop()` sets the event but
  deliberately leaves clearing it to the next `start()`'s own top -- clearing it itself would race
  whichever `_wait_ready()` loop it just interrupted.
* **A conservative retry within a pack.** Nobody can test an Intel Arc driver's Vulkan
  flash-attention / quantized-KV support here, and older "Alchemist" drivers are documented to have
  fp16/garbage-output issues under llama.cpp's Vulkan backend. Before falling through to the NEXT
  flavor, a GPU pack (any framework but CUDA) that exits while loading gets ONE retry of the SAME
  pack with `--flash-attn auto` and an f16 KV cache (`build_args(..., conservative=True)`) -- this
  turns an untested vendor's flaky first attempt into a working server instead of an unnecessary
  demotion to the CPU pack, without touching the user's own `flash_attn`/`kv_cache_type` config.
* **`[llm].llama_env`** merges extra environment variables into llama-server's process (after
  PATH) -- an Arc "Alchemist" workaround (`GGML_VK_DISABLE_COOPMAT`, etc. -- see
  `exocortex.llm.runtimes.vulkan_env_switches`, which reads the exact switches a given
  `ggml-vulkan.dll` build actually recognises) or SYCL's `ONEAPI_DEVICE_SELECTOR`. Framework
  "sycl" also gets `ZES_ENABLE_SYSMAN=1` by default (SYCL.md: enables free-VRAM reporting) unless
  the user's own `llama_env` already sets that key.

No Intel GPU exists on either dev box this branch was built on (RTX 5080/5090 laptops, the Ryzen
AI MAX+ 395 / Radeon 8060S box); every Intel-specific code path above is unit-tested with
synthetic data only (fake `GpuInfo`, fake `--list-devices` text, a monkeypatched `runtimes`
module). Live on the AMD box (2026-09-26), all with runtimes.py's real b11200 downloads already on
disk under `.tools/llama.cpp` and LM Studio untouched:

* `find_server_binary(vendor="intel", source="upstream", upstream_root=Path(".tools/llama.cpp"))`
  -> flavor "vulkan", source "upstream", reason "vendor intel -> vulkan (upstream llama.cpp
  b11200)", exactly as designed (Intel's ladder is vulkan > cpu; no download needed, already on
  disk). A full `LlamaServerManager(LLMConfig(llama_source="upstream", llama_backend="vulkan",
  model="lmstudio-community/Bonsai-27B")).start()` on this box (vendor auto-detects "amd" here,
  backend forced to vulkan) loaded the upstream Vulkan build end to end: ready in 87.4 s (a cold
  3.8 GB read off the E: exFAT card), `/health` 200, one `/v1/completions` (2 tokens, "OK")
  answered in 4.09 s at ~3.7 tok/s prompt-eval / ~11.8 tok/s generation (a fresh, cold-cache read,
  not a warm-model number), `fallbacks == []`, `stop()` left no `llama-server.exe` behind.
* `LLMConfig(llama_source="upstream", llama_backend="sycl", model="lmstudio-community/Bonsai-27B")`
  on this AMD box: the already-installed upstream SYCL runtime lists no device, the one
  conservative retry (flash-attn auto + f16 KV) sees none either (device listing does not depend
  on that flag), then the fall-through lands on the upstream Vulkan build (already installed too)
  and loads normally. `mgr.fallbacks == ["retrying intel-sycl with flash-attn auto + f16 KV",
  "intel-sycl b11200: sycl llama-server lists no GPU device on this machine"]` -- exactly the
  "untested vendor degrades to a working server" behaviour this branch adds.
* Quit-during-load: starting the slow-loading 16.8 GB Qwen3.8-27B (LM Studio's ROCm pack, a
  600 s configured timeout) on a thread and calling `stop()` 3.0 s in: `start()` raised
  `LlamaServerError("stopped")` essentially instantly (~0.6 ms after the stop event was set) --
  the fix (not holding `self._lock` across `_wait_ready`) is what makes that possible at all.
  `stop()` itself took ~5.0 s to return to ITS caller, not the "~1 s" hoped for -- dominated by the
  pre-existing (unchanged by this branch) `terminate() -> wait(5) -> kill() -> wait(5)` teardown of
  a process with a large in-flight mmap read from slow removable media, which this box's OS took
  just over the first 5 s budget to tear down. No `llama-server.exe` remained a few seconds later.
  The number that matters is the ABSENCE of a ~600 s block: before this fix, the same scenario
  would have hung `stop()` behind the whole remaining startup timeout, because `start()`/`_attempt()`
  held the lock `stop()` needs for its entire duration.

2026-09-27 (portable-install): the goal of this branch is a fresh machine with no LM Studio and no
Ollama installed at all still getting a working llama-server from ONE run of run.bat/run.sh --
NVIDIA and AMD no longer get a free pass out of the upstream-fetch mechanism Intel/other vendors
already had (see the "second runtime source" bullet above). The rule is now vendor-neutral: source
"auto" is still the newest LM Studio pack for the first ladder flavor that HAS one (a machine with
any LM Studio pack for its own vendor stays byte-identical to before this branch -- no ensure()
call, no network, ever); only when NO LM Studio pack exists for ANY GPU flavor of the ladder does
the first fetchable GPU flavor get an upstream build, before finally the CPU pack (LM Studio's or
upstream). :func:`_upstream_kind_for` replaces the old static ``_UPSTREAM_KIND_OF`` lookup for the
CUDA/ROCm flavors it used to refuse outright: NVIDIA now resolves to ``"cuda13"`` or ``"cuda12"``
by the installed driver's major version
(:func:`exocortex.llm.runtimes.cuda_kind_for_driver`/:func:`exocortex.gpu.nvidia_driver_version`),
AMD to ``"rocm"`` -- both fetched from :data:`exocortex.llm.runtimes.PINNED_ASSETS` exactly like
Vulkan/SYCL/OpenVINO always were. Library integration moved onto :mod:`exocortex.llm.library` (a
sibling module landing in this same tree): :meth:`LlamaServerManager.available_models` and the
model-selection helpers (:func:`resolve_from`/:func:`choose_from`, the model-list halves
:func:`resolve_model`/:func:`choose_model` now share) work over its richer ``LibraryModel`` scan
when it is importable, falling back to this module's own :func:`scan_models` when it is not --
every reach into it goes through the single :func:`_library` accessor (lazy, guarded with
``try``/``except ImportError``) so this module keeps importing, and every pre-existing test keeps
passing, whether or not that module has landed yet -- see :func:`_library`'s own docstring for why
this is a dedicated function rather than each call site's own inline import.
:meth:`LlamaServerManager.build_args` also applies a per-model tuning profile
(``library.profile_for``) on top of ``[llm]``'s own values when one exists;
:attr:`LlamaServerManager.profile` records what was applied, for the doctor/UI.
:meth:`LlamaServerManager._can_fall_through` now allows an upstream-sourced CUDA pack (never one
from an LM Studio pack or an explicit ``llama_server_path``) to fall through to the next flavor --
a driver too old for the pinned CUDA 13 build should land on CUDA 12, then Vulkan, then CPU
instead of a hard failure.
"""
from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from exocortex import gpu
from exocortex.config import LLMConfig, project_root
from exocortex.llm import runtimes

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------
# Backend (runtime pack) selection: vendor -> flavor ladder
# ---------------------------------------------------------------------------------------

#: Per-vendor flavor ladders, most preferred first. NVIDIA's ordering (CUDA 12 > CUDA > Vulkan >
#: CPU) is UNCHANGED from before this module became vendor-aware -- see the module docstring.
#: AMD puts ROCm ahead of Vulkan, measured 2026-09-26 on the Radeon 8060S with Qwen3.8-27B Q4_K_M
#: (16.8 GB, idle machine, LM Studio packs rocm 2.29.1 vs vulkan 2.46.0): prompt processing
#: 300 vs 160 tok/s, time-to-first-token on a cached prefix 0.4 s vs 3.4 s, generation equal
#: (11.2 vs 12.4 tok/s), incremental prompt 273 vs 92 tok/s -- this app sends a long transcript +
#: screen text with every answer, so prompt speed is what the user feels. LM Studio only installs
#: the ROCm pack on GPUs in its ``gpu.targets`` list (the 780M / gfx1103 is not; the 8060S /
#: gfx1151 is), and :meth:`LlamaServerManager.start` falls through to the next pack when a GPU
#: pack sees no device or exits while loading, so Vulkan -- which runs on every AMD GPU -- is
#: still what any AMD machine without a usable ROCm ends up on.
_VENDOR_LADDER: dict[str, tuple[str, ...]] = {
    "nvidia": ("nvidia-cuda12", "nvidia-cuda", "vulkan", "cpu"),
    "amd": ("amd-rocm", "vulkan", "cpu"),
    # Intel: Vulkan stays the DEFAULT ladder for both the Arc iGPUs (140V/140T) this app already
    # ran on and the Arc DISCRETE GPUs this branch adds support for (A770/A750, B580/B570, Pro
    # B70) -- the llama.cpp community's own recommendation for Arc as of 2026-07, and the only
    # framework LM Studio's runtime catalog packages for this vendor at all. "intel-sycl" and
    # "openvino" (Intel-only, upstream-only -- see _UPSTREAM_KIND_OF) are explicit opt-ins via
    # [llm].llama_backend, never on by default: nobody here can measure whether they help or hurt
    # on real Arc hardware.
    "intel": ("vulkan", "cpu"),
    # "none" = DXGI enumerated no hardware adapter (or failed). That is NOT proof of a GPU-less
    # machine -- a driver hiccup or a locked-down session on an NVIDIA laptop must not demote it to
    # the CPU pack -- so this is the pre-vendor-aware order: whatever pack LM Studio installed is
    # itself evidence of the hardware (it only offers packs the machine can run), best one first.
    "none": ("nvidia-cuda12", "nvidia-cuda", "vulkan", "cpu"),
}
#: Any vendor not listed above ("other", "qualcomm", ...): Vulkan is the only GPU framework that
#: could apply, then the CPU pack.
_DEFAULT_LADDER: tuple[str, ...] = ("vulkan", "cpu")
#: [llm].llama_backend overrides: the flavor(s) this moves to the front of the ladder. "sycl" and
#: "openvino" (added 2026-09-26) are Intel-only frameworks LM Studio's runtime catalog never
#: packages (no SYCL/OpenVINO pack exists there) -- they only ever resolve through
#: exocortex.llm.runtimes' upstream builds (see _UPSTREAM_KIND_OF, find_server_binary).
_REQUESTED_HEAD: dict[str, tuple[str, ...]] = {
    "cuda": ("nvidia-cuda12", "nvidia-cuda"),
    "rocm": ("amd-rocm",),
    "vulkan": ("vulkan",),
    "sycl": ("intel-sycl",),
    "openvino": ("openvino",),
}
#: What to tell the user, per vendor, when no pack in their ladder is present. 2026-09-27
#: (portable-install): every vendor can now self-heal via run.bat's upstream fetch (see the module
#: docstring), so this points there first and [llm].llama_server_path second, instead of naming an
#: LM Studio runtime to install by hand -- flavoured per vendor only in which GPU framework it
#: names, since that is the one thing that actually differs between them.
_INSTALL_HINT: dict[str, str] = {
    "nvidia": "run run.bat (it fetches the CUDA llama.cpp runtime for this GPU) or set [llm].llama_server_path",
    "amd": "run run.bat (it fetches the ROCm/Vulkan llama.cpp runtime for this GPU) or set [llm].llama_server_path",
    "intel": "run run.bat (it fetches the Vulkan llama.cpp runtime for this GPU) or set [llm].llama_server_path",
    "none": "run run.bat (it fetches the llama.cpp runtime for this machine) or set [llm].llama_server_path",
}
#: ServerBinary.framework, per flavor.
_FRAMEWORK_OF: dict[str, str] = {
    "nvidia-cuda12": "cuda", "nvidia-cuda": "cuda", "amd-rocm": "rocm", "vulkan": "vulkan", "cpu": "cpu",
    "intel-sycl": "sycl", "openvino": "openvino",
}
#: Vendor DLL package directory (under backends/vendor/) for a flavor, when the manifest does not
#: name one itself. ROCm's vendor package is versioned (win-llama-rocm-vendor-v6 today) and is
#: resolved by globbing instead (see `_vendor_names_for`).
_VENDOR_FOR: dict[str, str] = {
    "nvidia-cuda12": "win-llama-cuda12-vendor-v2",
    "nvidia-cuda": "win-llama-cuda-vendor-v2",
    "vulkan": "win-llama-vulkan-vendor-v2",
}
#: flavor -> exocortex.llm.runtimes.RUNTIME_KINDS, for the flavors whose upstream kind never
#: depends on anything about the machine beyond the flavor itself. CUDA and ROCm are handled by
#: :func:`_upstream_kind_for` instead (NVIDIA's upstream kind depends on the installed driver
#: version; AMD's is a fixed "rocm") -- see that function and the module docstring's 2026-09-27
#: (portable-install) note for why NVIDIA/AMD are no longer excluded from upstream fetching the
#: way they were before this branch.
_UPSTREAM_KIND_OF: dict[str, str] = {
    "vulkan": "vulkan",
    "cpu": "cpu",
    "intel-sycl": "sycl",
    "openvino": "openvino",
}


def _upstream_kind_for(flavor: str, vendor: str) -> str | None:
    """The :mod:`exocortex.llm.runtimes` ``RUNTIME_KINDS`` value to fetch for `flavor` on a
    `vendor` machine, or ``None`` when this flavor is never fetched upstream (there is none such
    left as of 2026-09-27 (portable-install) -- every GPU flavor in every ladder can now be
    fetched when no LM Studio pack covers it; see the module docstring).

    NVIDIA (``"nvidia-cuda12"``/``"nvidia-cuda"``): ``exocortex.llm.runtimes.cuda_kind_for_driver
    (exocortex.gpu.nvidia_driver_version())`` -- ``"cuda13"`` for a driver whose major version is
    recent enough, else ``"cuda12"``. AMD (``"amd-rocm"``): always ``"rocm"``. Everything else:
    the fixed :data:`_UPSTREAM_KIND_OF` map (vulkan/cpu/sycl/openvino).

    Agents B (``gpu.nvidia_driver_version``) and C (``runtimes.cuda_kind_for_driver``) build these
    two functions in this same tree concurrently with this module -- both are looked up as plain
    attributes (not imported at module load time; :mod:`exocortex.gpu` and
    :mod:`exocortex.llm.runtimes` are already imported above, only these two specific names might
    not exist on them yet) and any ``AttributeError`` falls back to ``"cuda12"``, the safer/older
    of the two pinned CUDA builds, so this module keeps working (and every existing test keeps
    passing) whether or not those two pieces have landed yet. A driver-probe failure of any other
    kind is swallowed the same way NVML/DXGI probes elsewhere in this app are (advisory only).
    """
    if flavor in ("nvidia-cuda12", "nvidia-cuda"):
        try:
            driver = gpu.nvidia_driver_version()
        except AttributeError:
            driver = None
            gated = "cuda12"
        except Exception:  # noqa: BLE001 - a driver probe must never block runtime routing
            driver = None
            gated = None
        else:
            gated = None
        if gated is None:
            try:
                gated = runtimes.cuda_kind_for_driver(driver)
            except AttributeError:
                gated = "cuda12"
        if flavor == "nvidia-cuda12":
            return gated
        # The ladder's second CUDA rung ("nvidia-cuda") is the fall-through for a driver that
        # passed the coarse ">= 580" check but cannot actually run the pinned 13.4 build: it
        # resolves to the 12.4 build then -- and to nothing at all when the first rung already
        # was 12.4, so the fall-through never re-runs the identical binary under another name
        # (a review finding of 2026-09-27) and goes straight on to Vulkan.
        return "cuda12" if gated == "cuda13" else None
    if flavor == "amd-rocm":
        return "rocm"
    return _UPSTREAM_KIND_OF.get(flavor)
#: GPU frameworks eligible for the one-time same-pack conservative retry (see
#: LlamaServerManager.start): every non-CPU, non-CUDA framework this app can spawn. CUDA is
#: excluded (NVIDIA behaviour never changes); "custom" (an explicit llama_server_path) is excluded
#: by flavor, not framework, checked separately -- an explicit path is the user's own choice too.
_GPU_FRAMEWORKS_FOR_RETRY: frozenset[str] = frozenset({"rocm", "vulkan", "sycl", "openvino"})


def backend_ladder(vendor: str, requested: str = "auto") -> tuple[str, ...]:
    """Flavor preference order for `vendor` ("nvidia" | "amd" | "intel"; "none" = no adapter
    detected -> the legacy CUDA 12 > CUDA > Vulkan > CPU order, see _VENDOR_LADDER; any other
    vendor -> Vulkan > CPU), optionally overridden by `requested` ([llm].llama_backend): "auto" (default) is the
    vendor's own ladder; "cuda" / "rocm" / "vulkan" / "sycl" / "openvino" move that flavor (CUDA:
    both CUDA flavors) to the front, with the vendor ladder following as a deduplicated fallback;
    "sycl"/"openvino" are Intel-only, upstream-only flavors (LM Studio's runtime catalog packages
    neither) that any vendor can still request explicitly; "cpu" means CPU only, no GPU fallback
    (an explicit choice, not a missing pack). An unrecognised `requested` is logged and treated as
    "auto"."""
    vendor_ladder = _VENDOR_LADDER.get(vendor, _DEFAULT_LADDER)
    req = (requested or "auto").strip().lower()
    if req in ("", "auto"):
        return vendor_ladder
    if req == "cpu":
        return ("cpu",)
    head = _REQUESTED_HEAD.get(req)
    if head is None:
        log.warning("llm: unknown [llm].llama_backend %r; using auto", requested)
        return vendor_ladder
    out = list(head)
    for f in vendor_ladder:
        if f not in out:
            out.append(f)
    return tuple(out)


class LlamaServerError(RuntimeError):
    """Raised when the server binary/model cannot be found or the server fails to start."""


#: The exact message _wait_ready()/start() use for a deliberate stop() during load, so the caller
#: can tell it apart from an ordinary failure (a genuine timeout, a crash) without racing the
#: shared _stop_event's state -- checked by matching this string, never by re-reading the event,
#: since a concurrent stop() may have already cleared it (for the NEXT start()) by the time the
#: loading thread's except-handler runs.
_STOPPED = "stopped"

#: Prefix _wait_ready() puts on a genuine deadline-timeout message (as opposed to a process-exit
#: message, e.g. "llama-server exited with code N ..."), checked -- like _STOPPED -- by matching
#: the message string, so start()'s except-handler can tell a pack that merely outlasted its
#: startup deadline (still loading, not crashed) apart from one that actually died while loading.
#: 2026-09-26 (finding #5, docs/UPSTREAM_SUGGESTIONS.md review): the one-time conservative retry is
#: for the crash case only (see the module docstring); before this, a legitimately slow load that
#: just hit `estimate_startup_timeout_s` got a second full-length wait of the SAME pack before any
#: fall-through, up to doubling (chained across a ladder, ~4x) the worst-case time on slow media.
_TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class GgufModel:
    """A GGUF file in the model library."""

    key: str            # "deepgrove/maple-preview"
    name: str           # "maple-preview"
    path: Path
    size_bytes: int

    @property
    def display(self) -> str:
        return f"{self.key}  ({self.size_bytes / 1e9:.1f} GB)"


@dataclass(frozen=True)
class ServerBinary:
    exe: Path
    dll_dirs: tuple[Path, ...]
    flavor: str         # e.g. "nvidia-cuda12" | "amd-rocm" | "vulkan" | "intel-sycl" | "openvino" | "cpu" | "custom"
    version: str
    framework: str = "custom"  # "cuda" | "rocm" | "vulkan" | "sycl" | "openvino" | "cpu" | "custom"
    reason: str = ""    # one human sentence: why this pack, e.g. "vendor amd -> vulkan (first choice)"
    # 2026-09-26: which runtime source this binary came from -- "lmstudio" (LM Studio's own runtime
    # pack, the only source before this branch), "upstream" (an official llama.cpp release build
    # fetched by exocortex.llm.runtimes, e.g. Intel's SYCL/OpenVINO or a newer Vulkan than LM
    # Studio ships), or "custom" (an explicit [llm].llama_server_path). Surfaced by the doctor.
    source: str = "lmstudio"


@dataclass(frozen=True)
class Pack:
    """One discovered llama.cpp runtime pack, before it is picked (or not) for a vendor."""

    exe: Path
    flavor: str
    version: str
    dll_dirs: tuple[Path, ...]


# ---------------------------------------------------------------------------------------
def lmstudio_home() -> Path:
    return Path(os.environ.get("LMSTUDIO_HOME") or (Path.home() / ".lmstudio"))


def lmstudio_models_dir(cfg: LLMConfig | None = None) -> Path:
    """LM Studio's download folder (``downloadsFolder`` in settings.json) or its default."""
    if cfg is not None and cfg.lmstudio_models_dir:
        return Path(cfg.lmstudio_models_dir).expanduser()
    settings = lmstudio_home() / "settings.json"
    try:
        data = json.loads(settings.read_text(encoding="utf-8"))
        folder = data.get("downloadsFolder")
        if folder:
            return Path(folder)
    except (OSError, ValueError):
        pass
    return lmstudio_home() / "models"


def scan_models(models_dir: Path) -> list[GgufModel]:
    """All non-projector GGUF files in the library, as ``publisher/repo`` keys (sorted).

    LM Studio lays models out as ``<publisher>/<repo>/<file>.gguf``; only the top three levels are
    searched (a full ``rglob`` over a large library with unrelated subtrees took ~20 s cold here).
    """
    out: list[GgufModel] = []
    if not models_dir.is_dir():
        return out
    found: set[Path] = set()
    for pattern in ("*.gguf", "*/*.gguf", "*/*/*.gguf"):
        found.update(models_dir.glob(pattern))
    for p in sorted(found):
        if p.name.lower().startswith("mmproj") or "mmproj" in p.name.lower():
            continue
        try:
            rel = p.relative_to(models_dir)
        except ValueError:
            rel = Path(p.name)
        parts = rel.parts
        if len(parts) >= 3:
            publisher, repo = parts[0], parts[1]
        elif len(parts) == 2:
            publisher, repo = parts[0], p.stem
        else:
            publisher, repo = "local", p.stem
        name = re.sub(r"[-_.]?gguf$", "", repo, flags=re.IGNORECASE)
        out.append(GgufModel(key=f"{publisher}/{name}", name=name, path=p, size_bytes=p.stat().st_size))
    return out


def resolve_from(models: "Sequence[Any]", spec: str, *, where: str = "") -> "Any":
    """The model-LIST half of :func:`resolve_model` -- matches `spec` against `models` by exact
    key / name / file stem / file name (case-insensitive), then a unique case-insensitive
    substring of key/file name, factored out 2026-09-27 (portable-install) so it also works on
    :class:`exocortex.llm.library.LibraryModel` (agent A's richer library scan), not only
    :class:`GgufModel` -- any object with ``.key``/``.name``/``.path``/``.size_bytes`` works.
    `where` names the root(s) searched, used only in the "not found" error text. Raises
    :class:`LlamaServerError` naming every known key when nothing matches."""
    low = spec.strip().lower()
    for m in models:
        if m.key.lower() == low or m.name.lower() == low or m.path.stem.lower() == low or m.path.name.lower() == low:
            return m
    subs = [m for m in models if low in m.key.lower() or low in m.path.name.lower()]
    if len(subs) == 1:
        return subs[0]
    if len(subs) > 1:
        # prefer the smallest file among ambiguous matches only if names are identical modulo quant
        subs.sort(key=lambda m: m.size_bytes)
        log.warning("model spec %r is ambiguous (%s); using %s", spec, ", ".join(m.key for m in subs), subs[0].key)
        return subs[0]
    known = ", ".join(m.key for m in models) or "(no GGUF files found)"
    raise LlamaServerError(f"model '{spec}' not found in {where}. Known models: {known}")


def resolve_model(spec: str, models_dir: Path) -> GgufModel:
    """Turn ``maple-preview`` / ``deepgrove/maple-preview`` / a file name / an absolute path into a
    GgufModel -- an absolute ``.gguf`` path first, else :func:`resolve_from` over
    :func:`scan_models`'s list."""
    spec = spec.strip()
    p = Path(spec).expanduser()
    if p.suffix.lower() == ".gguf" and p.is_file():
        return GgufModel(key=p.stem, name=p.stem, path=p, size_bytes=p.stat().st_size)
    models = scan_models(models_dir)
    return resolve_from(models, spec, where=str(models_dir))


def pick_default_model(models: "Sequence[Any]", budget_mb: int | None) -> "Any | None":
    """``None`` if `models` is empty. With a `budget_mb`, the largest model whose ``size_bytes``
    stays within 75% of it (room left over for the KV cache and activations), or -- if none fit --
    the smallest model available. Without a budget (no GPU detected), the largest model."""
    if not models:
        return None
    if budget_mb is not None:
        limit_bytes = 0.75 * budget_mb * 2**20
        fitting = [m for m in models if m.size_bytes <= limit_bytes]
        if fitting:
            return max(fitting, key=lambda m: m.size_bytes)
        return min(models, key=lambda m: m.size_bytes)
    return max(models, key=lambda m: m.size_bytes)


def choose_from(models: "Sequence[Any]", spec: str, budget_mb: int | None, *, where: str = "") -> "tuple[Any, str]":
    """The model-LIST half of :func:`choose_model` -- factored out 2026-09-27 (portable-install)
    so :meth:`LlamaServerManager._attempt` can run it over
    :func:`exocortex.llm.library.scan_library`'s ``LibraryModel`` list exactly as easily as over
    :func:`scan_models`'s ``GgufModel`` list. `where` names the root(s) searched, used in the "no
    GGUF models found" error text and passed through to :func:`resolve_from`.

    ``"auto"`` (any case) or ``""``: :func:`pick_default_model` over `models`, noted. Any other
    spec: :func:`resolve_from` (raises for a name that truly does not exist); when that raises
    here, fall back to the automatic choice with a WARNING instead of failing the start over one
    bad model name. :class:`LlamaServerError` is raised only when `models` is empty, naming
    `where`.
    """
    text = (spec or "").strip()

    def _auto() -> "tuple[Any, str]":
        model = pick_default_model(models, budget_mb)
        if model is None:
            raise LlamaServerError(f"no GGUF models found in {where}")
        budget_txt = f"{budget_mb} MiB" if budget_mb is not None else "no GPU budget"
        gb = model.size_bytes / 1e9
        fits = budget_mb is None or model.size_bytes <= 0.75 * budget_mb * 2**20
        if fits:
            note = f"auto: {model.key} ({gb:.1f} GB) fits {budget_txt}"
        else:
            note = (
                f"auto: {model.key} ({gb:.1f} GB) is the smallest model available "
                f"and may not fit the {budget_mb} MiB GPU budget"
            )
        return model, note

    if text == "" or text.lower() == "auto":
        return _auto()
    try:
        return resolve_from(models, text, where=where), ""
    except LlamaServerError:
        model, _note = _auto()
        log.warning("model %r not found; using %s", spec, model.key)
        return model, f"model '{spec}' not found; using {model.key}"


def choose_model(spec: str, models_dir: Path, budget_mb: int | None) -> tuple[GgufModel, str]:
    """Resolve `spec` ([llm].model) to a model plus a status note (possibly "") -- :func:`choose_from`
    over :func:`scan_models`'s list, keeping this function's own pre-2026-09-27 signature and "no
    GGUF models found ...; download a GGUF in LM Studio." error text unchanged."""
    models = scan_models(models_dir)
    where = f"{models_dir}; download a GGUF in LM Studio."
    return choose_from(models, spec, budget_mb, where=where)


def _read_version(d: Path) -> str:
    try:
        data = json.loads((d / "backend-manifest.json").read_text(encoding="utf-8"))
        return str(data.get("version", ""))
    except (OSError, ValueError):
        m = re.search(r"(\d+\.\d+\.\d+)$", d.name)
        return m.group(1) if m else "0"


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)) or (0,)


def _read_manifest(d: Path) -> dict | None:
    try:
        data = json.loads((d / "backend-manifest.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _flavor_from_dirname(name: str) -> str:
    """Directory-name fallback when the manifest is missing/malformed or names no GPU framework."""
    low = name.lower()
    if "cuda12" in low:
        return "nvidia-cuda12"
    if "rocm" in low:
        return "amd-rocm"
    if "cuda" in low:
        return "nvidia-cuda"
    if "vulkan" in low:
        return "vulkan"
    return "cpu"


def _flavor_of(d: Path, manifest: dict | None) -> str:
    """The manifest's ``gpu.framework`` is definitive when present ("CUDA" -> "nvidia-cuda12" if
    "cuda12" is in the directory name else "nvidia-cuda"; "ROCm" -> "amd-rocm"; "Vulkan" ->
    "vulkan"); a manifest with no usable ``gpu`` framework (absent, null, or a genuine CPU-only
    pack -- LM Studio's own CPU manifests carry no "gpu" key at all) falls back to directory-name
    tokens, which resolve a real CPU pack to "cpu" the same way."""
    if isinstance(manifest, dict):
        gpu_info = manifest.get("gpu")
        if isinstance(gpu_info, dict):
            framework = str(gpu_info.get("framework") or "").strip().lower()
            if framework == "cuda":
                return "nvidia-cuda12" if "cuda12" in d.name.lower() else "nvidia-cuda"
            if framework == "rocm":
                return "amd-rocm"
            if framework == "vulkan":
                return "vulkan"
    return _flavor_from_dirname(d.name)


def _vendor_names_for(flavor: str, manifest: dict | None, backends: Path) -> list[str]:
    """Vendor DLL package name(s) for `flavor`: the manifest's own ``vendor_lib_package_names``
    when present and non-empty, else the fixed map (CUDA/CUDA12/Vulkan) or, for ROCm, the newest
    ``win-llama-rocm-vendor-v<N>`` under ``backends/vendor`` (its package is versioned)."""
    if isinstance(manifest, dict):
        names = manifest.get("vendor_lib_package_names")
        if isinstance(names, list) and names:
            return [str(n) for n in names]
    if flavor == "amd-rocm":
        candidates = list((backends / "vendor").glob("win-llama-rocm-vendor-v*")) if backends.is_dir() else []
        if not candidates:
            return []

        def _v(p: Path) -> int:
            m = re.search(r"-v(\d+)$", p.name)
            return int(m.group(1)) if m else -1

        candidates.sort(key=_v)
        return [candidates[-1].name]
    name = _VENDOR_FOR.get(flavor)
    return [name] if name else []


def _scan_packs(backends: Path) -> list[Pack]:
    """Every ``llama.cpp-*`` directory under `backends` that has a ``llama-server.exe``, with its
    flavor, version and DLL search directories (its own dir plus its vendor DLL package(s), and
    that package's ``bin`` subfolder when it has one -- the ROCm vendor pack keeps its DLLs there)."""
    packs: list[Pack] = []
    if not backends.is_dir():
        return packs
    for d in sorted(backends.iterdir()):
        if not d.is_dir() or not d.name.startswith("llama.cpp-"):
            continue
        exe = d / "llama-server.exe"
        if not exe.is_file():
            continue
        manifest = _read_manifest(d)
        flavor = _flavor_of(d, manifest)
        version = _read_version(d)
        dll_dirs: list[Path] = [d]
        for vname in _vendor_names_for(flavor, manifest, backends):
            vdir = backends / "vendor" / vname
            if vdir.is_dir():
                dll_dirs.append(vdir)
                bindir = vdir / "bin"
                if bindir.is_dir():
                    dll_dirs.append(bindir)
        packs.append(Pack(exe=exe, flavor=flavor, version=version, dll_dirs=tuple(dll_dirs)))
    return packs


def _lmstudio_reason(vendor: str, flavor: str, i: int, ladder: tuple[str, ...],
                      exclude: "frozenset[str] | set[str]", chosen: Pack) -> str:
    """The exact reason text find_server_binary has always produced for an LM Studio pack --
    unchanged so the "auto" + LM-Studio-packs-present result stays byte-identical to before this
    branch (the owner's hard rule for NVIDIA, extended here to every vendor's LM Studio path)."""
    if exclude:
        return f"vendor {vendor} -> {flavor} (after {', '.join(sorted(exclude))} failed on this machine)"
    if i == 0:
        return f"vendor {vendor} -> {flavor} (first choice)"
    hint = _INSTALL_HINT.get(vendor, "run run.bat or set [llm].llama_server_path")
    log.warning("llm: preferred backend %r not available for vendor %r; using %r instead (%s)",
                ladder[0], vendor, flavor, chosen.exe)
    return f"vendor {vendor} -> {flavor} (no {ladder[0]} pack installed; {hint})"


def _upstream_reason(vendor: str, flavor: str, build: str, *, no_lmstudio_pack_anywhere: bool) -> str:
    if no_lmstudio_pack_anywhere:
        return f"vendor {vendor} -> {flavor} (upstream llama.cpp {build}; no LM Studio pack for any {vendor} GPU flavor)"
    return f"vendor {vendor} -> {flavor} (upstream llama.cpp {build})"


def _upstream_binary(folder: Path, flavor: str, build: str, reason: str) -> ServerBinary:
    return ServerBinary(exe=folder / "llama-server.exe", dll_dirs=(folder,), flavor=flavor, version=build,
                         framework=_FRAMEWORK_OF.get(flavor, "custom"), reason=reason, source="upstream")


def _try_upstream(kind: str, root: Path, build: str,
                   ensure: "Callable[[str, Path, str], Path] | None") -> Path | None:
    """:func:`exocortex.llm.runtimes.installed_runtime`, or -- only when `ensure` was given, i.e.
    downloading is actually allowed here -- :func:`exocortex.llm.runtimes.ensure_runtime` via that
    injected callable (so tests, and any caller that only wants to know what is already on disk,
    never trigger a network call). ``None`` on a download failure (logged), never raises."""
    folder = runtimes.installed_runtime(root, kind, build)
    if folder is not None or ensure is None:
        return folder
    try:
        return ensure(kind, root, build)
    except runtimes.UpstreamRuntimeCancelled:
        # The app is quitting mid-download: that is a stop, not "this runtime is unavailable, try
        # the next flavor" -- propagate it as the same sentinel start() checks for.
        raise LlamaServerError(_STOPPED)
    except (runtimes.UpstreamRuntimeError, OSError) as exc:
        log.warning("llm: could not fetch the upstream %s runtime (build %s): %s", kind, build, exc)
        return None


def find_server_binary(explicit: str = "", *, vendor: str | None = None, backend: str = "auto",
                       exclude: "frozenset[str] | set[str]" = frozenset(),
                       source: str | None = None, upstream_root: "Path | None" = None,
                       upstream_build: str | None = None,
                       ensure: "Callable[[str, Path, str], Path] | None" = None,
                       stop_check: "Callable[[], bool] | None" = None) -> ServerBinary:
    """Locate llama-server.exe.

    An explicit path/dir (flavor "custom", framework sniffed from its DLLs) always wins, exactly
    as before this branch. Otherwise, for each flavor of :func:`backend_ladder` (`vendor`,
    `backend`) not in `exclude` (flavors that already failed on this machine --
    :meth:`LlamaServerManager.start`'s fall-through):

    * `source` "lmstudio": the newest LM Studio pack for that flavor, or try the next flavor.
      (This is today's/pre-2026-09-26 behaviour, unchanged -- the default when `source` is
      omitted or "auto" and any GPU flavor of the ladder has an LM Studio pack.)
    * `source` "upstream": an already-installed upstream build (under `upstream_root`, default
      ``project_root()/.tools/llama.cpp``, or ``Path(cfg.runtime_dir)`` when
      :meth:`LlamaServerManager.start` finds one set), else -- only for a flavor with an upstream
      kind (:func:`_upstream_kind_for`; every flavor has one as of 2026-09-27, including CUDA and
      ROCm -- see the module docstring) and only when `ensure` was given -- a freshly fetched one;
      else the next flavor.
    * `source` "auto" (default): the LM Studio pack first; when a flavor has none, it is
      remembered as the first "fetchable" GPU flavor (has an upstream kind) and the next flavor is
      tried. Only once EVERY GPU flavor of the ladder has been tried with no LM Studio pack found
      (including one already in `exclude`) does the remembered fetchable flavor get the same
      installed-or-fetch treatment as explicit "upstream" -- before the CPU pack, not after.

    The CPU pack is tried last regardless of `source`: LM Studio's own CPU pack for every source,
    plus (source "auto"/"upstream" only -- "lmstudio" never fetches) the upstream CPU build when
    LM Studio has none.

    `vendor` defaults to :func:`exocortex.gpu.gpu_vendor`. Downloading is synchronous but never
    implicit: it only happens through the injected `ensure` callable
    (``ensure(kind, root, build) -> Path``, raising :class:`exocortex.llm.runtimes.UpstreamRuntimeError`
    on failure) -- :meth:`LlamaServerManager.start` supplies one wired to report progress via
    `on_status`; this function itself makes no network call and stays synchronous either way.
    Raises :class:`LlamaServerError` naming what to install for THIS vendor when nothing usable
    was found anywhere.

    `stop_check` (optional): polled between flavors -- and once more before the deferred
    "auto"-fetch and the final CPU rung -- so a concurrent :meth:`LlamaServerManager.stop` lands
    within one flavor's worth of (fast, local-disk) work instead of only being noticed back in
    :meth:`LlamaServerManager.start`'s own loop. Raises :class:`LlamaServerError` (``"stopped"``)
    the moment it returns true. NOTE: this does NOT interrupt a single upstream download already
    in flight -- :mod:`exocortex.llm.runtimes`'s ``ensure_runtime`` has no cancellation hook of its
    own yet, so a stop() during an actual multi-MB fetch still blocks until that fetch finishes or
    fails (docs/UPSTREAM_SUGGESTIONS.md finding #3 is only partially closed here).
    """
    if explicit:
        p = Path(explicit).expanduser()
        exe = p if p.is_file() else p / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
        if not exe.is_file():
            raise LlamaServerError(f"llama-server not found at {explicit}")
        framework = sniff_framework(exe.parent)
        return ServerBinary(exe=exe, dll_dirs=(exe.parent,), flavor="custom", version=_read_version(exe.parent),
                             framework=framework, reason=f"explicit path {explicit} ({framework} build)",
                             source="custom")
    if vendor is None:
        vendor = gpu.gpu_vendor()
    src = (source or "auto").strip().lower()
    if src not in ("auto", "lmstudio", "upstream"):
        log.warning("llm: unknown [llm].llama_source %r; using auto", source)
        src = "auto"
    root = Path(upstream_root) if upstream_root is not None else (project_root() / ".tools" / "llama.cpp")
    build = upstream_build or runtimes.PINNED_BUILD

    backends = lmstudio_home() / "extensions" / "backends"
    ladder = backend_ladder(vendor, backend)
    by_flavor: dict[str, list[Pack]] = {}
    for pk in _scan_packs(backends):
        by_flavor.setdefault(pk.flavor, []).append(pk)
    for group in by_flavor.values():
        group.sort(key=lambda pk: _version_tuple(pk.version), reverse=True)

    fetchable_flavor: str | None = None      # "auto" only: first GPU flavor with no LM Studio pack
    for i, flavor in enumerate(ladder):
        if stop_check is not None and stop_check():
            raise LlamaServerError(_STOPPED)
        if flavor == "cpu" or flavor in exclude:
            continue
        if src == "lmstudio":
            group = by_flavor.get(flavor)
            if group:
                return ServerBinary(exe=group[0].exe, dll_dirs=group[0].dll_dirs, flavor=flavor,
                                     version=group[0].version, framework=_FRAMEWORK_OF.get(flavor, "custom"),
                                     reason=_lmstudio_reason(vendor, flavor, i, ladder, exclude, group[0]),
                                     source="lmstudio")
            continue
        if src == "upstream":
            kind = _upstream_kind_for(flavor, vendor)
            if kind is None:               # no flavor left this can happen for as of 2026-09-27
                continue
            folder = _try_upstream(kind, root, build, ensure)
            if folder is not None:
                return _upstream_binary(folder, flavor, build, _upstream_reason(vendor, flavor, build,
                                                                                 no_lmstudio_pack_anywhere=False))
            continue
        # src == "auto"
        group = by_flavor.get(flavor)
        if group:
            return ServerBinary(exe=group[0].exe, dll_dirs=group[0].dll_dirs, flavor=flavor,
                                 version=group[0].version, framework=_FRAMEWORK_OF.get(flavor, "custom"),
                                 reason=_lmstudio_reason(vendor, flavor, i, ladder, exclude, group[0]),
                                 source="lmstudio")
        # 2026-09-27 (portable-install): this used to gate on `vendor not in ("nvidia", "amd")` --
        # NVIDIA and AMD already had first-class LM Studio pack families for every flavor their own
        # ladder can reach (CUDA/CUDA12, ROCm), so "auto"'s upstream-fetch fallback was reserved for
        # vendors LM Studio's catalog left with NOTHING at all (Intel's SYCL/OpenVINO, "other"/
        # "none"). That made "no LM Studio installed at all" a hard, offline failure for NVIDIA/AMD
        # even though every GPU flavor of their ladder now HAS an upstream kind
        # (:func:`_upstream_kind_for`) -- exactly the case a from-scratch install (no LM Studio, no
        # Ollama) hits on first run. The gate is gone: any vendor's first GPU flavor with no LM
        # Studio pack becomes the remembered fetchable flavor. A machine that DOES have an LM
        # Studio pack for its own vendor never reaches this line for that flavor (the `group` check
        # above returns first), so byte-identical behaviour for an already-provisioned NVIDIA/AMD
        # machine is preserved by that check, not by this one.
        if fetchable_flavor is None and _upstream_kind_for(flavor, vendor) is not None:
            fetchable_flavor = flavor

    if src == "auto" and fetchable_flavor is not None:
        if stop_check is not None and stop_check():
            raise LlamaServerError(_STOPPED)
        kind = _upstream_kind_for(fetchable_flavor, vendor)
        assert kind is not None            # guaranteed by the gate above that set fetchable_flavor
        folder = _try_upstream(kind, root, build, ensure)
        if folder is not None:
            return _upstream_binary(folder, fetchable_flavor, build,
                                     _upstream_reason(vendor, fetchable_flavor, build, no_lmstudio_pack_anywhere=True))

    # Finally, the CPU pack: LM Studio's own for every source; the upstream CPU build (source
    # "auto"/"upstream" only -- "lmstudio" never fetches, "packs only" by design) when LM Studio
    # has none.
    if "cpu" not in exclude:
        if stop_check is not None and stop_check():
            raise LlamaServerError(_STOPPED)
        cpu_group = by_flavor.get("cpu")
        if cpu_group:
            i = ladder.index("cpu") if "cpu" in ladder else len(ladder)
            return ServerBinary(exe=cpu_group[0].exe, dll_dirs=cpu_group[0].dll_dirs, flavor="cpu",
                                 version=cpu_group[0].version, framework="cpu",
                                 reason=_lmstudio_reason(vendor, "cpu", i, ladder, exclude, cpu_group[0]),
                                 source="lmstudio")
        # 2026-09-27 (portable-install): the same gate removal as the GPU-flavor fetch above --
        # "auto" now fetches an upstream CPU build for every vendor when LM Studio has none
        # installed, not only for the vendors LM Studio's catalog already left with nothing.
        if src == "upstream" or src == "auto":
            folder = _try_upstream("cpu", root, build, ensure)
            if folder is not None:
                return _upstream_binary(folder, "cpu", build,
                                         _upstream_reason(vendor, "cpu", build, no_lmstudio_pack_anywhere=True))

    hint = _INSTALL_HINT.get(vendor, "run run.bat or set [llm].llama_server_path")
    raise LlamaServerError(
        f"no llama.cpp runtime pack for a {vendor} GPU found under {backends} (tried {', '.join(ladder)}); "
        f"{hint}."
    )


#: Sequential-read floors (MiB/s) used to size the startup timeout by where the GGUF lives.
#: Measured 2026-09-26 on the Ryzen AI MAX+ 395 box: its model library sits on an exFAT card in the
#: PCIe card reader ("Removable"), and a cold load of the 16.8 GB Qwen3.8-27B Q4_K_M took 323 s
#: (~52 MiB/s) -- far past the 180 s default, so the app would have given up on a model that was
#: loading fine. An internal NVMe reads the same file in well under a minute.
_READ_FLOOR_MIB_S: dict[str, float] = {"removable": 30.0, "remote": 40.0, "cdrom": 10.0, "fixed": 300.0,
                                       "ramdisk": 2000.0, "unknown": 100.0}
_DRIVE_TYPES = {0: "unknown", 1: "unknown", 2: "removable", 3: "fixed", 4: "remote", 5: "cdrom", 6: "ramdisk"}


def media_kind(path: Path) -> tuple[str, str, str]:
    """``(kind, filesystem, root)`` of the volume holding `path` -- kind is ``"removable"`` /
    ``"fixed"`` / ``"remote"`` / ``"cdrom"`` / ``"ramdisk"`` / ``"unknown"`` (``GetDriveTypeW``),
    filesystem e.g. ``"exFAT"`` / ``"NTFS"`` (``GetVolumeInformationW``), root e.g. ``"E:\\"``.
    ``("unknown", "", "")`` off Windows or on any failure; never raises."""
    if sys.platform != "win32":
        return "unknown", "", ""
    try:
        import ctypes
        from ctypes import wintypes

        anchor = Path(path).resolve().anchor  # "E:\\" (or a UNC root)
        if not anchor:
            return "unknown", "", ""
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
        k32.GetDriveTypeW.restype = wintypes.UINT
        kind = _DRIVE_TYPES.get(int(k32.GetDriveTypeW(anchor)), "unknown")
        fs = ctypes.create_unicode_buffer(64)
        k32.GetVolumeInformationW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
                                              ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                                              ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR, wintypes.DWORD]
        k32.GetVolumeInformationW.restype = wintypes.BOOL
        ok = k32.GetVolumeInformationW(anchor, None, 0, None, None, None, fs, 64)
        return kind, (fs.value if ok else ""), anchor
    except Exception:  # noqa: BLE001 - advisory only
        log.debug("llm: media_kind(%s) failed", path, exc_info=True)
        return "unknown", "", ""


def estimate_startup_timeout_s(configured_s: float, size_bytes: int, kind: str) -> float:
    """The startup deadline actually used: never less than `configured_s` ([llm].llama_startup_timeout_s,
    180 s by default -- unchanged for every model that loads within it), and never less than the
    time a cold read of `size_bytes` would take at the volume kind's read floor plus 60 s of slack.
    A 16.8 GB GGUF: fixed disk -> 117 s (the 180 s default stands); removable card -> 594 s."""
    floor = _READ_FLOOR_MIB_S.get(kind, _READ_FLOOR_MIB_S["unknown"])
    estimate = size_bytes / (floor * 2**20) + 60.0
    return max(float(configured_s), estimate)


_LOG_WARNING_PATTERNS: tuple[tuple[str, str], ...] = (
    ("cache_reuse is not supported",
     "llama.cpp disabled --cache-reuse for this context (parallel slots with a unified KV cache): a prompt "
     "prefix that moves is re-evaluated instead of shifted; set [llm].parallel = 1 to get KV shifting back"),
    ("cannot be used with preferred buffer type",
     "some tensors have no GPU kernels in this build and run on the CPU (see llama-server.log)"),
)


def scan_log_warnings(text: str) -> list[str]:
    """Human-readable warnings for the llama-server log lines that matter operationally (measured
    on both the Vulkan and ROCm packs on 2026-09-26: ``cache_reuse is not supported by this
    context, it will be disabled`` appears on every start with ``--parallel 2``)."""
    out: list[str] = []
    low = text.lower()
    for needle, message in _LOG_WARNING_PATTERNS:
        if needle in low and message not in out:
            out.append(message)
    return out


_SNIFF_DLLS: tuple[tuple[str, str], ...] = (("ggml-cuda*", "cuda"), ("ggml-hip*", "rocm"), ("ggml-vulkan*", "vulkan"),
                                            ("ggml-sycl*", "sycl"), ("ggml-openvino*", "openvino"),
                                            ("ggml-cpu*", "cpu"))


def sniff_framework(exe_dir: Path) -> str:
    """The GPU framework of a llama.cpp folder given by hand ([llm].llama_server_path), from the
    ggml backend DLLs next to the executable: ``"cuda"`` / ``"rocm"`` / ``"vulkan"`` / ``"sycl"`` /
    ``"openvino"`` / ``"cpu"``, or ``"custom"`` when none is recognisable. A CUDA folder found this
    way is treated exactly like a CUDA pack (never given ``--device``)."""
    try:
        for pattern, framework in _SNIFF_DLLS:
            if any(exe_dir.glob(pattern + ".dll")) or any(exe_dir.glob(pattern + ".so")):
                return framework
    except OSError:
        pass
    return "custom"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _physical_cores() -> int | None:
    """Physical core count via ``GetLogicalProcessorInformationEx(RelationProcessorCore)``.

    ``os.cpu_count()`` reports LOGICAL processors, and the old ``cores // 2`` guess assumed SMT;
    on a no-SMT chip (this machine's Core Ultra 9 285H: 16 cores, 16 logical) it halved the real
    count and starved the llama-server thread pool. ``None`` on any failure (non-Windows, an odd
    kernel32) -- the caller falls back to the old heuristic.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = k32.GetLogicalProcessorInformationEx
        fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        fn.restype = wintypes.BOOL
        relation_core = 0                      # RelationProcessorCore
        needed = wintypes.DWORD(0)
        if fn(relation_core, None, ctypes.byref(needed)) or needed.value == 0:
            return None                        # success with a null buffer would be nonsense
        buf = (ctypes.c_char * needed.value)()
        if not fn(relation_core, buf, ctypes.byref(needed)):
            return None
        raw = bytes(buf)
        count = 0
        offset = 0
        # Variable-size SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX records: DWORD Relationship at
        # +0, DWORD Size at +4; walking by Size is the documented iteration.
        while offset + 8 <= len(raw):
            relationship = int.from_bytes(raw[offset:offset + 4], "little")
            size = int.from_bytes(raw[offset + 4:offset + 8], "little")
            if size <= 0:
                return None
            if relationship == relation_core:
                count += 1
            offset += size
        return count or None
    except Exception:  # noqa: BLE001 - a counting failure must never stop the server starting
        return None


def default_threads() -> int:
    cores = os.cpu_count() or 8
    physical = _physical_cores()
    if physical is None:
        physical = max(4, cores // 2)  # old heuristic: assumes SMT (logical = 2x physical)
    return max(4, physical - 2)


# ---------------------------------------------------------------------------------------
# GPU device selection ("--device" among several a framework sees)
# ---------------------------------------------------------------------------------------

_CUDA_FLAVORS = frozenset({"nvidia-cuda12", "nvidia-cuda"})

#: "Vulkan0: AMD Radeon(TM) 8060S Graphics (48971 MiB, 46523 MiB free)" -- name, description, and
#: an optional memory group (either half may be absent: some builds print no parentheses at all).
_DEVICE_LINE_RE = re.compile(
    r"^\s*([A-Za-z]+\d+):\s*(.+?)(?:\s*\((\d+)\s*MiB(?:,\s*(\d+)\s*MiB\s*free)?\))?\s*$"
)


@dataclass(frozen=True)
class LlamaDevice:
    name: str                  # "Vulkan0", "ROCm0", "CUDA0"
    description: str           # "AMD Radeon(TM) 8060S Graphics"
    total_mb: int | None
    free_mb: int | None


def parse_list_devices(text: str) -> list[LlamaDevice]:
    """Parse ``llama-server --list-devices`` output. Tolerates CRLF, extra log lines before/after
    the device lines, and a missing memory parenthetical."""
    devices: list[LlamaDevice] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        m = _DEVICE_LINE_RE.match(raw)
        if not m:
            continue
        name, desc, total, free = m.groups()
        devices.append(LlamaDevice(
            name=name, description=desc.strip(),
            total_mb=int(total) if total else None,
            free_mb=int(free) if free else None,
        ))
    return devices


_list_devices_cache: dict[str, list[LlamaDevice]] = {}


def list_devices(binary: ServerBinary, timeout_s: float = 20.0) -> list[LlamaDevice]:
    """Run ``[binary.exe, "--list-devices"]`` (its ``dll_dirs`` prepended to PATH, its own
    directory as cwd, no console window) and parse the result. ``[]`` on any failure or timeout
    (logged at debug -- device listing is advisory, llama.cpp starts fine without ``--device``).
    Cached per exe path."""
    key = str(binary.exe)
    if key in _list_devices_cache:
        return _list_devices_cache[key]
    devices: list[LlamaDevice] = []
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(str(d) for d in binary.dll_dirs) + os.pathsep + env.get("PATH", "")
    creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            [str(binary.exe), "--list-devices"], cwd=str(binary.exe.parent), env=env,
            capture_output=True, timeout=timeout_s, creationflags=creation,
        )
        text = ((proc.stdout or b"") + b"\n" + (proc.stderr or b"")).decode("utf-8", errors="replace")
        devices = parse_list_devices(text)
    except Exception:  # noqa: BLE001 - advisory only; a failure here must never stop the server
        log.debug("llm: --list-devices failed for %s", binary.exe, exc_info=True)
        devices = []
    _list_devices_cache[key] = devices
    return devices


def _normalise_device_name(s: str) -> str:
    s = s.lower().replace("(tm)", "").replace("(r)", "")
    return re.sub(r"\s+", " ", s).strip()


def choose_device(devices: list[LlamaDevice], primary: "gpu.GpuInfo | None", requested: str = "auto",
                   *, flavor: str) -> str | None:
    """The ``--device`` value to pass (``None`` = let llama.cpp decide):

    * `requested` ``""``: always ``None``.
    * `requested` an explicit name: that name, even if it is not in `devices` (warns then).
    * `requested` ``"auto"``: ``None`` for a CUDA `flavor` (unchanged NVIDIA behaviour) or when
      `devices` has 0 or 1 entries; else the device whose description names `primary`'s GPU
      (case-insensitive, "(TM)"/"(R)"/whitespace normalised, substring either way), else the
      device reporting the most ``total_mb``.
    """
    if requested == "":
        return None
    if requested.strip().lower() != "auto":
        wanted = requested.strip()
        for d in devices:
            if d.name.lower() == wanted.lower():
                return d.name                      # llama.cpp matches --device exactly: hand back its own spelling
        log.warning("llm: requested device %r not among --list-devices output (%s); using it anyway",
                    requested, ", ".join(d.name for d in devices) or "none")
        return wanted
    if flavor in _CUDA_FLAVORS:
        return None
    if len(devices) <= 1:
        return None
    if primary is not None:
        pname = _normalise_device_name(primary.name)
        if pname:
            for d in devices:
                dname = _normalise_device_name(d.description)
                if pname in dname or dname in pname:
                    return d.name
    best = max(devices, key=lambda d: d.total_mb if d.total_mb is not None else -1)
    return best.name


# ---------------------------------------------------------------------------------------
class _JobObject:
    """Windows Job Object that kills the child when our process exits (any way)."""

    def __init__(self) -> None:
        self.handle = None
        if sys.platform != "win32":
            return
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32 = k32
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                       "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # JobObjectExtendedLimitInformation
        if not ok:
            log.warning("SetInformationJobObject failed: %s", ctypes.get_last_error())
            k32.CloseHandle(job)
            return
        self.handle = job

    def assign(self, pid: int) -> None:
        if self.handle is None or sys.platform != "win32":
            return
        import ctypes

        k32 = self._k32
        PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
        h = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not h:
            log.warning("OpenProcess(%s) failed: %s", pid, ctypes.get_last_error())
            return
        if not k32.AssignProcessToJobObject(self.handle, h):
            log.warning("AssignProcessToJobObject failed: %s", ctypes.get_last_error())
        k32.CloseHandle(h)


def _library() -> "Any | None":
    """Lazy, guarded import of :mod:`exocortex.llm.library` -- the module, or ``None`` when it is
    not (yet) importable. Every caller below used to write its own bare
    ``try: from exocortex.llm import library; except ImportError: ...`` inline; factored into this
    one indirection (2026-09-27, portable-install) because that inline form stops being
    interceptable by a test the moment ANY caller in the process has imported the real module
    successfully once -- CPython then caches ``library`` as an ATTRIBUTE of the ``exocortex.llm``
    package, and a later ``from exocortex.llm import library`` reads that cached attribute straight
    off the package object without even consulting ``sys.modules`` again, so a test that patches
    only ``sys.modules["exocortex.llm.library"]`` (the standard trick, and the right one for forcing
    an ``ImportError`` -- see the module's own "no profile module" tests) has no effect once the
    package attribute is already warm. Tests that need to INJECT a fake module (as opposed to
    simulating an absent one) monkeypatch this function directly
    (``monkeypatch.setattr(srv, "_library", lambda: fake)``), which sidesteps the caching question
    entirely rather than fighting it."""
    try:
        from exocortex.llm import library
    except ImportError:
        return None
    return library


class LlamaServerManager:
    """Start/stop/swap a private llama-server. Thread-safe; all methods block until done."""

    def __init__(self, cfg: LLMConfig, *, log_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.log_dir = log_dir
        self._proc: subprocess.Popen[bytes] | None = None
        self._job = _JobObject()
        self._lock = threading.RLock()
        # 2026-09-26: set by stop(), watched by _wait_ready()'s poll loop so a concurrent stop()
        # (e.g. the user quitting while a big model is still loading) interrupts the wait within
        # one poll tick instead of blocking behind the whole startup timeout (docs/UPSTREAM_SUGGESTIONS.md
        # finding #1) -- see the module docstring's "quit-during-load" paragraph. Cleared at the
        # TOP of start() (not by stop() itself -- see stop()'s docstring for why that would race).
        self._stop_event = threading.Event()
        self.binary: ServerBinary | None = None
        self.model: GgufModel | None = None
        self.note: str = ""
        self.device: str | None = None
        self.media: tuple[str, str, str] = ("unknown", "", "")   # (kind, filesystem, root) of the model's volume
        self.timeout_s: float = float(cfg.llama_startup_timeout_s)  # the deadline actually used by the last start()
        self.warnings: list[str] = []                             # operational warnings scanned from llama-server.log
        self.fallbacks: list[str] = []                            # packs skipped by start()'s fall-through, and why
        self.port: int = 0
        self.log_path: Path | None = None
        # 2026-09-27 (portable-install): the per-model ModelProfile build_args() applied on the
        # last call (exocortex.llm.library.profile_for), or None when no profile module/entry
        # exists for that model -- surfaced for the doctor/UI, never consulted by this class itself.
        self.profile: "Any | None" = None

    # ---- discovery --------------------------------------------------------------------
    def available_models(self) -> list:
        """The model library: :func:`exocortex.llm.library.scan_library` (this app's own GGUF
        library plus LM Studio's, agent A's richer scan -- richer ``LibraryModel`` objects) when
        that module is importable, else this module's own :func:`scan_models` over LM Studio's
        library alone (2026-09-27 (portable-install): the pre-``library`` behaviour, kept as the
        fallback while that module is still landing in this same tree -- see the module
        docstring). Every caller that only reads ``.key``/``.name``/``.path``/``.size_bytes``
        works unchanged either way."""
        library = _library()
        if library is None:
            return scan_models(lmstudio_models_dir(self.cfg))
        return library.scan_library(self.cfg)

    def _model_search_where(self) -> str:
        """Text naming where :meth:`available_models` looked, for :func:`choose_from`'s "no GGUF
        models found" error -- this app's own library dir plus LM Studio's when
        :mod:`exocortex.llm.library` is importable, else the old LM-Studio-only text (that module
        not having landed yet in this tree)."""
        library = _library()
        if library is None:
            return f"{lmstudio_models_dir(self.cfg)}; download a GGUF in LM Studio."
        try:
            lib_dir = library.library_dir(self.cfg)
        except Exception:  # noqa: BLE001 - this is error-message text, never worth failing a start over
            return f"{lmstudio_models_dir(self.cfg)}; download a GGUF in LM Studio."
        return f"{lib_dir} or LM Studio's library -- open the ... menu > Models... to pull one"

    def _profile_for(self, model: "Any") -> "Any | None":
        """:func:`exocortex.llm.library.profile_for` (self.cfg, model.key), or ``None`` when
        :mod:`exocortex.llm.library` is not importable yet or has no entry for this model --
        imported lazily so this module keeps importing (and every pre-existing test keeps
        passing) whether or not that sibling module has landed in this tree yet."""
        library = _library()
        if library is None:
            return None
        try:
            return library.profile_for(self.cfg, model.key)
        except Exception:  # noqa: BLE001 - a bad profiles.json must never block a start
            log.warning("llm: profile_for(%s) failed", model.key, exc_info=True)
            return None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ---- lifecycle --------------------------------------------------------------------
    def build_args(self, model: "Any", port: int, *, device: str | None = None,
                   conservative: bool = False) -> list[str]:
        """`conservative=True` (2026-09-26): the one-time same-pack retry :meth:`start` performs
        when an untested vendor's GPU pack exits while loading (see the module docstring) --
        `--flash-attn auto` instead of the configured on/off, and no `--cache-type-k/v` flags at
        all (llama-server's own default is an unquantized f16 KV cache). `cfg.flash_attn` /
        `cfg.kv_cache_type` themselves are never touched; a later normal start still uses them.

        2026-09-27 (portable-install): a per-model tuning profile
        (:func:`exocortex.llm.library.profile_for`, ``self.profile`` after this call) overrides
        ``ctx_size``/``gpu_layers``/``kv_cache_type``/``flash_attn``/``threads``/``parallel`` field
        by field wherever it sets a non-``None`` value (``None`` means "use the global [llm]
        value", so a profile that only pins ``ctx_size`` leaves everything else exactly as before);
        its ``extra_args`` are appended after ``cfg.llama_extra_args``. The CPU-flavor "gpu_layers
        0" rule and `conservative`'s own overrides still take precedence over the profile -- a
        profile is a per-model DEFAULT, not a way to force GPU layers onto a CPU pack or to defeat
        the conservative retry."""
        cfg = self.cfg
        profile = self._profile_for(model)
        self.profile = profile

        def _pf(name: str, cfg_value: Any) -> Any:
            if profile is None:
                return cfg_value
            v = getattr(profile, name, None)
            return cfg_value if v is None else v

        threads_cfg = _pf("threads", cfg.threads)
        threads = threads_cfg if threads_cfg and threads_cfg > 0 else default_threads()
        flavor = self.binary.flavor if self.binary is not None else None
        gpu_layers_cfg = _pf("gpu_layers", cfg.gpu_layers)
        gpu_layers = "0" if flavor == "cpu" else str(gpu_layers_cfg)  # a CPU pack ignores the flag anyway
        flash_attn_cfg = bool(_pf("flash_attn", cfg.flash_attn))
        flash_attn_value = "auto" if conservative else ("on" if flash_attn_cfg else "off")
        parallel_cfg = _pf("parallel", cfg.parallel)
        ctx_size_cfg = _pf("ctx_size", cfg.ctx_size)
        args = [
            "--model", str(model.path),
            "--host", "127.0.0.1", "--port", str(port),
            "--no-webui",
            "--ctx-size", str(ctx_size_cfg),
            "--n-gpu-layers", gpu_layers,
            "--n-cpu-moe", "0",
            "--batch-size", "2048", "--ubatch-size", "512",
            "--threads", str(threads),
            "--parallel", str(max(1, parallel_cfg)),
            "--flash-attn", flash_attn_value,
            "--kv-offload",
            "--load-mode", cfg.load_mode or "mmap",
            # Reuse cached KV chunks even when they moved in the prompt (KV shifting): the <screen>
            # block sits after the growing transcript, so without this it would be re-evaluated on
            # every answer even when the screen did not change.
            "--cache-reuse", "256",
        ]
        # KV cache quantisation: q8_0 by default (see LLMConfig.kv_cache_type -- the same GGUF
        # runs under llama-swap with q8_0, and the freed VRAM is paging margin on the display GPU).
        # Skipped entirely under `conservative` -- an f16 KV cache, llama-server's own default.
        kv_cache_type_cfg = _pf("kv_cache_type", cfg.kv_cache_type)
        kv_type = (kv_cache_type_cfg or "").strip().lower()
        if not conservative and kv_type not in ("", "f16"):
            v_type = kv_cache_type_cfg
            if not flash_attn_cfg:
                # llama.cpp refuses to start: "quantized V cache requires flash_attn to be enabled"
                # (measured 2026-09-26 on the Vulkan pack; a core check, not a vendor one). K may
                # stay quantized; only V has to be f16 without flash attention.
                v_type = "f16"
                log.warning("llm: flash_attn is off, so the V cache cannot be %s; using f16 for V (K stays %s)",
                            kv_cache_type_cfg, kv_cache_type_cfg)
            args += ["--cache-type-k", kv_cache_type_cfg, "--cache-type-v", v_type]
        if parallel_cfg > 1:
            args.append("--kv-unified")
        args += list(cfg.llama_extra_args)
        if profile is not None and getattr(profile, "extra_args", None):
            args += list(profile.extra_args)
        if device:
            args += ["--device", device]
        if profile is not None and any(
            getattr(profile, name, None) is not None
            for name in ("ctx_size", "gpu_layers", "kv_cache_type", "flash_attn", "threads", "parallel")
        ):
            log.info("llm: profile for %s: ctx=%s gpu_layers=%s kv=%s", model.key, ctx_size_cfg, gpu_layers,
                     kv_cache_type_cfg)
        return args

    def start(self, model_spec: str | None = None, *, on_status: "callable[[str], None] | None" = None) -> str:  # type: ignore[name-defined]
        """Start the server for `model_spec` (default cfg.model); returns the base URL.

        Fall-through: when the chosen pack is a GPU framework other than CUDA and was picked from
        the ladder (not an explicit ``llama_server_path``), a pack that sees no device
        (``--list-devices`` empty) or exits / times out while loading is put aside and the next
        flavor of the ladder is tried -- ROCm -> Vulkan -> CPU on an AMD machine, Vulkan -> CPU
        on Intel. CUDA packs and explicit paths raise exactly as before (NVIDIA behaviour and the
        user's explicit choice are never second-guessed). :attr:`fallbacks` records what was
        skipped and why, for the doctor / status line.

        2026-09-26: before falling through to the NEXT flavor, a GPU pack that fails gets ONE
        retry of the SAME pack with conservative flags (``build_args(..., conservative=True)`` --
        see the module docstring); ``[llm].llama_source``/``llama_upstream_build`` are threaded
        into :func:`find_server_binary` on every attempt, with an `ensure` callable that reports
        download progress through `on_status` (find_server_binary itself never downloads unless
        this is given -- see its own docstring). This method no longer holds ``self._lock`` for
        its own duration: a concurrent :meth:`stop` (e.g. quitting mid-load) sets
        ``self._stop_event``, which :meth:`_attempt`/:meth:`_wait_ready` notice within one poll
        tick and raise :class:`LlamaServerError` (``"stopped"``) instead of trying the next pack.
        """
        if self.running:
            self.stop()
        self._stop_event.clear()
        vendor = gpu.gpu_vendor()
        exclude: set[str] = set()
        retried: set[str] = set()          # flavors that already had their one conservative retry
        self.fallbacks = []
        # 2026-09-27 (portable-install): [llm].runtime_dir (agent D), when set, is where run.bat's
        # own upstream fetch installs a runtime and where find_server_binary() should look for one
        # -- getattr'd (not a plain attribute access) so this manager keeps working before that
        # config field has landed in this tree; "" (unset, or the field missing) keeps
        # find_server_binary()'s own default (project_root()/.tools/llama.cpp).
        runtime_dir = getattr(self.cfg, "runtime_dir", "")
        upstream_root = Path(runtime_dir) if runtime_dir else None

        def _ensure(kind: str, root: Path, build: str) -> Path:
            def _progress(name: str, downloaded: int, total: int) -> None:
                if on_status is not None:
                    pct = f" {downloaded * 100 // total}%" if total else ""
                    on_status(f"downloading {name}{pct} ({downloaded // 2**20} MiB)")
            # cancelled= is polled per 64 KiB chunk: stop() during a 30-120 MB fetch interrupts it
            # within one chunk instead of after the whole download (UpstreamRuntimeCancelled ->
            # _try_upstream -> LlamaServerError("stopped")).
            return runtimes.ensure_runtime(kind, root, build=build, progress=_progress,
                                           cancelled=self._stop_event.is_set)

        while True:
            if self._stop_event.is_set():
                raise LlamaServerError(_STOPPED)
            # NOT under self._lock: with source "auto"/"upstream" this can download tens to (for
            # SYCL) ~120 MB over the network, which must never block a concurrent stop() -- only
            # the assignment itself (self.binary is one of the fields _attempt/stop coordinate on)
            # is protected, exactly as briefly as self._proc/self.model/self.port below.
            binary = find_server_binary(self.cfg.llama_server_path, vendor=vendor,
                                        backend=self.cfg.llama_backend, exclude=exclude,
                                        source=self.cfg.llama_source, upstream_root=upstream_root,
                                        upstream_build=self.cfg.llama_upstream_build, ensure=_ensure,
                                        stop_check=self._stop_event.is_set)
            with self._lock:
                self.binary = binary
            try:
                return self._attempt(model_spec, on_status=on_status)
            except LlamaServerError as exc:
                if exc.args and exc.args[0] == _STOPPED:
                    raise
                # 2026-09-26 (finding #2, docs/UPSTREAM_SUGGESTIONS.md review): the one-time
                # conservative retry is Intel-only -- nobody can test an Intel Arc driver's Vulkan/
                # SYCL/OpenVINO behaviour here, but ROCm and Vulkan are shared frameworks, so
                # gating on framework alone also retried AMD's ROCm pack (and, narrowly, a Vulkan
                # pack an NVIDIA machine fell back to), changing the AMD rocm > vulkan > cpu
                # fall-through the owner's hard rule says must not move. Restricting this to
                # vendor == "intel" keeps NVIDIA/AMD's fall-through attempt counts and timing
                # byte-identical to before this branch.
                # (finding #5): also never retry a GENUINE `_wait_ready` timeout -- only a same-pack
                # crash-while-loading is what this retry is for (see the module docstring); a pack
                # that was simply still loading would otherwise wait its own full timeout a SECOND
                # time before falling through, up to ~doubling (chained across a ladder, ~4x) the
                # worst-case time on exactly the slow-media boxes this branch targets.
                timed_out = bool(exc.args) and str(exc.args[0]).startswith(_TIMED_OUT)
                if (vendor == "intel" and not timed_out
                        and self.binary.framework in _GPU_FRAMEWORKS_FOR_RETRY and self.binary.flavor != "custom"
                        and self.binary.flavor not in retried):
                    retried.add(self.binary.flavor)
                    why = f"retrying {self.binary.flavor} with flash-attn auto + f16 KV"
                    self.fallbacks.append(why)
                    log.warning("llm: %s -- %s", exc, why)
                    if on_status:
                        on_status(why)
                    try:
                        return self._attempt(model_spec, on_status=on_status, conservative=True)
                    except LlamaServerError as exc2:
                        if exc2.args and exc2.args[0] == _STOPPED:
                            raise
                        exc = exc2
                if not self._can_fall_through(exclude):
                    raise
                exclude.add(self.binary.flavor)
                why = f"{self.binary.flavor} {self.binary.version}: {exc}"
                self.fallbacks.append(why)
                log.warning("llm: %s -- trying the next runtime pack", why)
                if on_status:
                    on_status(f"{self.binary.flavor} llama.cpp failed here; trying the next runtime pack")
                try:
                    find_server_binary(self.cfg.llama_server_path, vendor=vendor,
                                       backend=self.cfg.llama_backend, exclude=exclude,
                                       source=self.cfg.llama_source, upstream_root=upstream_root,
                                       upstream_build=self.cfg.llama_upstream_build,
                                       ensure=_ensure, stop_check=self._stop_event.is_set)
                except LlamaServerError as probe_exc:
                    if probe_exc.args and probe_exc.args[0] == _STOPPED:
                        raise                  # a stop() landed on this probe: report it, not the old failure
                    raise exc from None       # nothing left to try: report the real failure

    def _can_fall_through(self, exclude: set[str]) -> bool:
        """Whether the ladder may move on to the next flavor after `self.binary` failed.

        Unchanged (the owner's hard rule): a CUDA pack from an LM Studio pack or from an explicit
        `llama_server_path` (``source == "custom"``, checked by ``flavor == "custom"`` above) never
        falls through -- NVIDIA behaviour and the user's own explicit choice are never
        second-guessed. 2026-09-27 (portable-install): a CUDA pack whose `source` is `"upstream"`
        MAY now fall through -- a driver too old for the pinned CUDA 13 build should land on
        CUDA 12, then Vulkan, then CPU instead of a hard failure, exactly like every other GPU
        framework already does."""
        b = self.binary
        if b is None or b.flavor == "custom" or b.flavor in exclude:
            return False
        if b.framework == "cuda" and b.source != "upstream":
            return False
        return len(exclude) < 3               # at most three packs per start()

    def _attempt(self, model_spec: str | None, *, on_status: "callable[[str], None] | None" = None,  # type: ignore[name-defined]
                conservative: bool = False) -> str:
        """One attempt with ``self.binary``: model / media / device / spawn / wait. Raises
        :class:`LlamaServerError` on any failure (the caller decides about falling through and
        about the one conservative retry).

        2026-09-26: only holds ``self._lock`` while mutating ``self.model``/``self.port``/
        ``self._proc`` (``self.binary`` is set by :meth:`start`, under the same short lock) --
        never across :meth:`_wait_ready`'s blocking poll -- so a concurrent :meth:`stop` is never
        stuck behind a slow load (docs/UPSTREAM_SUGGESTIONS.md finding #1). `conservative=True` is
        the one same-pack retry `start()` performs before falling through to the next flavor: it
        only changes the flags :meth:`build_args` produces, never the model/device/media chosen.
        """
        primary = gpu.primary_gpu()
        budget_mb = gpu.primary_budget_mb()
        # 2026-09-27 (portable-install): available_models() (this app's own library + LM Studio's,
        # via exocortex.llm.library when it is importable) replaces a direct lmstudio_models_dir()
        # scan here -- see the module docstring and available_models()'s own docstring.
        model, note = choose_from(self.available_models(), model_spec or self.cfg.model, budget_mb,
                                  where=self._model_search_where())
        with self._lock:
            self.model = model
        self.note = note
        if self.note:
            log.info("llm: %s", self.note)
            if on_status:
                on_status(self.note)
        if budget_mb is not None and model.size_bytes > 0.9 * budget_mb * 2**20:
            warn = (f"{model.key} ({model.size_bytes / 1e9:.1f} GB) may not fit in GPU memory "
                    f"({budget_mb} MiB budget)")
            log.warning("llm: %s", warn)
            if on_status:
                on_status(warn)
        self.media = media_kind(model.path)
        self.timeout_s = estimate_startup_timeout_s(self.cfg.llama_startup_timeout_s, model.size_bytes,
                                                    self.media[0])
        if self.media[0] in ("removable", "remote", "cdrom"):
            slow = (f"{model.name} ({model.size_bytes / 1e9:.1f} GB) lives on a {self.media[0]} "
                    f"volume ({self.media[2]} {self.media[1]}): a cold load can take several minutes "
                    f"(waiting up to {self.timeout_s:.0f} s); move the model library to the internal drive "
                    "for fast starts")
            log.warning("llm: %s", slow)
            if on_status:
                on_status(slow)
        self.warnings = []
        device_name = None
        if self.binary.framework != "cuda" and self.cfg.llama_device != "":
            devices = list_devices(self.binary)
            if (not devices and self.binary.framework in ("rocm", "vulkan", "sycl", "openvino")
                    and self.binary.flavor != "custom"):
                # A GPU pack that sees no GPU (a ROCm pack left over for a GPU it does not
                # support, a Vulkan/SYCL/OpenVINO driver that is not there): do not even spawn it.
                raise LlamaServerError(f"{self.binary.framework} llama-server lists no GPU device on this machine")
            device_name = choose_device(devices, primary, self.cfg.llama_device, flavor=self.binary.flavor)
        self.device = device_name
        with self._lock:
            self.port = self.cfg.llama_port or free_port()
        args = self.build_args(model, self.port, device=device_name, conservative=conservative)
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(str(d) for d in self.binary.dll_dirs) + os.pathsep + env.get("PATH", "")
        # 2026-09-26: framework "sycl" gets ZES_ENABLE_SYSMAN=1 by default (SYCL.md: enables
        # free-VRAM reporting) unless [llm].llama_env already names that key; llama_env itself is
        # merged in last (after PATH), so it can override anything set above.
        llama_env = self.cfg.llama_env or {}
        if self.binary.framework == "sycl" and "ZES_ENABLE_SYSMAN" not in llama_env:
            env["ZES_ENABLE_SYSMAN"] = "1"
        for k, v in llama_env.items():
            env[str(k)] = str(v)
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.log_dir / "llama-server.log"
            args += ["--log-file", str(self.log_path)]
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        log.info("starting %s (%s %s) model=%s port=%s device=%s", self.binary.exe.name, self.binary.flavor,
                 self.binary.version, model.key, self.port, self.device)
        if on_status:
            on_status(f"loading {model.name} ({model.size_bytes / 1e9:.1f} GB) with llama.cpp {self.binary.version}")
        if self._stop_event.is_set():
            raise LlamaServerError(_STOPPED)
        try:
            proc = subprocess.Popen(
                [str(self.binary.exe), *args], env=env, cwd=str(self.binary.exe.parent),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                creationflags=creation,
            )
        except OSError as e:
            raise LlamaServerError(f"could not start {self.binary.exe}: {e}") from e
        with self._lock:
            self._proc = proc
        # 2026-09-26 (finding #6, docs/UPSTREAM_SUGGESTIONS.md review): a concurrent stop() that
        # lands strictly between the pre-Popen check above and this assignment reads self._proc as
        # still None (Popen had not returned yet), so it terminates nothing and returns -- leaving
        # the process we just spawned tracked only by self._proc, with nothing left to clean it up.
        # Re-checking the event right here, under the same short lock discipline, closes that
        # window: if we lost the race, we tear down the process ourselves before reporting
        # "stopped", exactly as stop() would have.
        if self._stop_event.is_set():
            with self._lock:
                if self._proc is proc:
                    self._proc = None
            self._terminate_process(proc)
            raise LlamaServerError(_STOPPED)
        self._job.assign(proc.pid)
        self._wait_ready()
        return self.base_url

    def _wait_ready(self) -> None:
        """Poll ``self._proc``/``/health`` until ready, the process exits, the timeout elapses, or
        :attr:`_stop_event` is set (a concurrent :meth:`stop`) -- whichever comes first. Uses
        ``_stop_event.wait(0.25)`` instead of ``time.sleep(0.25)`` so a stop() is noticed within
        one tick rather than only at the next iteration's start, and never re-reads ``self._proc``
        (which stop() may already have cleared for the next start()) once captured locally."""
        proc = self._proc
        assert proc is not None
        deadline = time.monotonic() + self.timeout_s
        url = f"{self.base_url}/health"
        with httpx.Client(timeout=2.0) as c:
            while time.monotonic() < deadline:
                if self._stop_event.is_set():
                    raise LlamaServerError(_STOPPED)
                code = proc.poll()
                if code is not None:
                    tail = self._log_tail()
                    raise LlamaServerError(f"llama-server exited with code {code} while loading {self.model.name if self.model else ''}.{tail}")
                try:
                    r = c.get(url)
                    if r.status_code == 200:
                        log.info("llama-server ready at %s", self.base_url)
                        self._collect_log_warnings()
                        return
                except httpx.HTTPError:
                    pass
                if self._stop_event.wait(0.25):
                    raise LlamaServerError(_STOPPED)
        # 2026-09-26 (finding #4, docs/UPSTREAM_SUGGESTIONS.md review): this is a genuine deadline
        # timeout, NOT a caller-requested stop() -- calling self.stop() here (as before) would set
        # self._stop_event, which start()'s own retry/fall-through logic and _attempt()'s pre-Popen
        # guard would then (wrongly) treat as "the user asked to stop", short-circuiting the
        # conservative retry AND the whole rocm > vulkan > cpu fall-through ladder on the very
        # first genuine timeout. _kill_hung_process() does the same terminate/kill teardown without
        # touching _stop_event, which stays reserved for an actual external stop().
        self._kill_hung_process()
        raise LlamaServerError(f"{_TIMED_OUT}: llama-server did not become ready within {self.timeout_s:.0f}s")

    def _collect_log_warnings(self) -> None:
        """Read the operational warnings out of llama-server.log once the server is up (only when a
        log_dir was given); logged once each and kept on ``self.warnings`` for the doctor / UI."""
        if not self.log_path or not self.log_path.is_file():
            return
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        for w in scan_log_warnings(text):
            if w not in self.warnings:
                self.warnings.append(w)
                log.warning("llm: %s", w)

    def _log_tail(self, lines: int = 12) -> str:
        if not self.log_path or not self.log_path.is_file():
            return ""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
            return "\n  " + "\n  ".join(text)
        except OSError:
            return ""

    def stop(self) -> None:
        """Interrupt any in-progress :meth:`start` (sets ``self._stop_event`` FIRST, before doing
        anything that could block, so a load stuck in :meth:`_wait_ready` notices within one poll
        tick) and terminate the process it finds in ``self._proc``, if any.

        Deliberately does NOT clear ``self._stop_event`` itself: :meth:`start` clears it at its own
        top before doing anything else, so a later `start()` is never affected either way, and
        clearing it HERE would race an in-flight `_wait_ready()` -- if the clear ran before that
        loop's next `is_set()` check, it would never notice the stop and would poll all the way to
        the real timeout instead. Leaving the event set until the next `start()` closes that
        window completely.
        """
        self._stop_event.set()
        self._kill_hung_process()

    def _kill_hung_process(self) -> None:
        """The same teardown :meth:`stop` performs (grab-and-clear ``self._proc`` under the lock,
        then ``terminate()`` -> ``wait(5)`` -> ``kill()`` fallback on whatever it found) WITHOUT
        touching ``self._stop_event`` -- used by :meth:`_wait_ready`'s own deadline-timeout cleanup
        (finding #4), which is not a caller-requested stop and must not be indistinguishable from
        one. :meth:`stop` itself still sets the event (that is what an external stop IS), then
        delegates the actual process teardown here."""
        with self._lock:
            p = self._proc
            self._proc = None
        if p is not None and p.poll() is None:
            self._terminate_process(p)

    @staticmethod
    def _terminate_process(p: "subprocess.Popen[bytes]") -> None:
        log.info("stopping llama-server pid %s", p.pid)
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                p.kill()
                p.wait(timeout=5)
            except Exception:  # noqa: BLE001
                log.exception("could not kill llama-server")

    def swap_model(self, model_spec: str, *, on_status: "callable[[str], None] | None" = None) -> str:  # type: ignore[name-defined]
        """Restart with another model; returns the new base URL.

        2026-09-26: no longer wraps :meth:`stop`/:meth:`start` in ``self._lock`` itself -- each
        already takes it exactly where needed, and holding it here would put the same
        quit-during-load block back in front of a concurrent :meth:`stop` (this method runs the
        new model's full load, which can take just as long as the first `start()`)."""
        self.stop()
        self.cfg.model = model_spec
        return self.start(model_spec, on_status=on_status)

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.stop()
        except Exception:  # noqa: BLE001
            pass
