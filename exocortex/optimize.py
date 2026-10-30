"""The Optimize planner (2026-10-07): pure arithmetic that turns *this machine's* facts into a plan
for where each job runs and how llama-server is launched, tailored to one goal -- ``latency``,
``balanced``, ``context`` or ``quality``.

Why this exists. The 2026-10-07 re-benchmark of the owner's laptop (Core Ultra 9 285H, Arc 140T
iGPU, AI Boost NPU, RTX 5090 Laptop 24 GB, 64 GB DDR5-7467) found three things no single knob
could express:

* Speech ran Whisper large-v3-turbo int8 on four CPU threads: 6.2 s per utterance (``[asr]
  device = "cpu"`` was set on 2026-09-25 to keep ~2 GB off the display GPU, and ``engine = "auto"``
  still picks Whisper whenever CUDA exists). Parakeet-TDT on the CPU does the same utterance in
  0.15 s with no GPU memory at all; Whisper on CUDA in 0.11 s for 874 MiB. Speech placement is a
  real latency lever and a real VRAM lever, and the two goals disagree about it.
* The language model's KV cache for the hybrid ``qwen35`` architecture is ~34.8 KiB per token at
  q8_0 (16 attention layers of 64), not the 138 KiB the dense formula gives -- four times too
  much. Sizing the context window needs the hybrid-aware math, and a 12 % margin until a real
  footprint has been measured on this machine.
* The 5090 is power-capped (115 W enforced of 130 W, ~700 MHz of 3090 while generating, 83-88 C).
  Batch-size, thread and KV-type knobs barely move the numbers; only measured wins may become
  launch flags. Speculative decoding off the model's own MTP head (``--spec-type draft-mtp``, the
  GGUF carries the head) helped generation by 15-30 %; n-gram speculation hurt.

Design rules (the tests pin them):

* **Pure.** No I/O, no threads, no Qt, no imports of the heavy stacks. Everything the planner knows
  arrives in :class:`PlanInputs`; everything it decides leaves in a :class:`Plan` of plain data.
  ``exocortex.optimize_bench`` gathers evidence, ``exocortex.optimize_run`` drives analysis /
  benchmark / apply on the engine, ``exocortex.ui.optimize_panel`` draws it.
* **Arithmetic without evidence, flags only with it.** VRAM fit and the ASR / OCR choices follow
  from the hardware report, the GGUF header and shipped "typical" numbers. A llama-server launch
  flag (``--ubatch-size``, ``--spec-type``, ``--threads-batch``, ``--load-mode``, ``--parallel 1``)
  is proposed only when a measured run on this machine beat its baseline by more than the noise
  floor.
* **Never exceed the budget.** ``need(max_ctx) <= avail < need(max_ctx + CTX_STEP)`` exactly, for
  every tier of evidence; the reserve is never below the VRAM-guard floor + 256 MiB.
* **Explain every row.** Each :class:`Item` says what runs now, what is proposed, when it takes
  effect, what evidence it rests on (measured / recorded / estimated), and why.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

# ============================================================================================
# constants: seed values, one place. "typical" numbers are the 2026-10-07 sweep on the owner's
# machine; a measured row from THIS machine always wins over them.
# ============================================================================================
MIB = 2 ** 20
#: bytes per KV element, same table as exocortex.llm.library._KV_BYTES_PER_ELEMENT (unknown -> f16)
ELEM: dict[str, float] = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
CTX_STEP = 2048
CTX_FLOOR = 8192
#: SEED: llama-server's CUDA compute buffer by --ubatch-size (MiB), interpolated / clamped
COMPUTE_MB: dict[int, float] = {512: 700.0, 1024: 1100.0, 2048: 1900.0}
#: SEED: CUDA context + cuBLAS workspaces (MiB)
RUNTIME_MB = 400.0
#: a hybrid model whose header lacks the ssm.* keys: per-slot recurrent state (MiB)
STATE_MB_PER_SLOT_FALLBACK = 150.0
#: relative uncertainty added to the ctx-dependent part per evidence tier
UNC: dict[str, float] = {"fit": 0.04, "calibrated": 0.04, "header": 0.12, "pessimistic": 0.0}
GOALS: tuple[str, ...] = ("latency", "balanced", "context", "quality")
GOAL_SLACK_MB: dict[str, int] = {"latency": 1280, "balanced": 256, "quality": 256, "context": 0}
#: share of the fitting window each goal actually plans (context takes it all)
KV_SHARE: dict[str, float] = {"latency": 0.60, "balanced": 0.60, "quality": 0.60, "context": 1.00}
#: screen / hands-free prompt budgets relative to the app DEFAULTS (so a plan is a fixed point:
#: applying it and planning again proposes nothing new)
BUDGET_SCALE: dict[str, float] = {"latency": 0.6, "balanced": 1.0, "quality": 1.0, "context": 1.6}
DEFAULT_SCREEN_CHARS = 2500          # ScreenConfig.max_text_chars default
DEFAULT_HANDSFREE_CHARS = 6000       # HandsFreeConfig.context_chars default
PP_TYPICAL_TPS = 900.0        # prompt tokens / s, 2026-10-07 sweep (power-capped 5090, 27B Q4_K_M)
TG_TYPICAL_TPS = 20.0         # generated tokens / s, same
TRANSCRIPT_TOK_PER_MIN = 220  # spoken words -> tokens, for "hours of meeting the window holds"
TRANSCRIPT_SHARE = 0.75       # share of the answer budget the transcript typically fills
UNCACHED_PROMPT_TOKENS = 1000 # typical_answer_s model: screen block + new speech re-read per answer
ANSWER_TOKENS = 150
NOISE_FLOOR = 0.08            # minimum relative gain before a knob is adopted
SPEC_GAIN = 0.15              # ... for speculative-decoding groups (they also cost prefill)
REREAD_TARGET_S = 6.0         # latency goal: a cold re-read of the whole window should stay under this
KEEPUP_SHARE = 0.5            # ASR must finish an utterance within this share of the partial interval
KEEPUP_NO_PARTIALS_S = 1.5
QUALITY_MIN_CTX = 32768
QUALITY_OVERLAP_GAIN = 0.03
#: ASR priors: label -> (engine, device, compute_type, threads, median_s, vram_mb, overlap); "typical (2026-10-07)"
PRIORS_ASR: dict[str, tuple[str, str, str, int, float, int, float]] = {
    "sherpa-cpu": ("sherpa", "cpu", "int8", 4, 0.15, 0, 0.99),
    "whisper-cuda-int8_float16": ("faster-whisper", "cuda", "int8_float16", 0, 0.110, 874, 1.00),
    "whisper-cuda-float16": ("faster-whisper", "cuda", "float16", 0, 0.125, 2374, 1.00),
    "whisper-cpu-int8": ("faster-whisper", "cpu", "int8", 4, 6.2, 0, 1.00),
}
#: OCR priors: device -> (wall ms, cpu ms) for RapidOCR on a dense 2560x1600 frame, measured 2026-10-07
PRIORS_OCR: dict[str, tuple[float, float]] = {"igpu": (1334.0, 1344.0), "npu": (2648.0, 1109.0), "cpu": (3200.0, 13719.0)}
#: llama-server flags the planner owns inside ModelProfile.extra_args; anything else is left alone
MANAGED_FLAGS: tuple[str, ...] = ("--ubatch-size", "--spec-type", "--spec-draft-n-max", "--threads-batch", "--load-mode")
#: the languages the multilingual Parakeet bundle (parakeet-tdt-0.6b-v3) covers, besides "en"
SHERPA_LANGUAGES: frozenset[str] = frozenset({
    "en", "auto", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt",
    "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
})
#: --spec-type values that need no draft model file
SPEC_SELF_TYPES: frozenset[str] = frozenset({"draft-mtp", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache"})

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ============================================================================================
# data model: frozen, defaulted, JSON round-trippable
# ============================================================================================
def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_dict(v) for k, v in obj.items()}
    return obj


def _from_dict(cls: type, data: Any) -> Any:
    """Rebuild ``cls`` from :func:`_to_dict` output, tolerating missing / unknown keys (an older or
    foreign JSON never raises here -- a bad field just keeps its default)."""
    if not isinstance(data, dict):
        return cls()
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        nested = _NESTED.get((cls.__name__, f.name))
        try:
            if nested is not None:
                inner, many = nested
                if many:
                    v = tuple(_from_dict(inner, x) for x in (v or []))
                else:
                    v = _from_dict(inner, v) if v is not None else None
            elif isinstance(v, list) and f.name == "flags":
                v = tuple(tuple(x) for x in v)
            elif isinstance(v, list) and f.name in ("extra", "blocked", "warnings", "summary"):
                v = tuple(v)
            kwargs[f.name] = v
        except Exception:  # noqa: BLE001 - keep the default
            continue
    try:
        return cls(**kwargs)
    except TypeError:
        return cls()


@dataclass(frozen=True)
class Settings:
    goal: str = "balanced"
    reserve_vram_mb: int = 0
    ctx_cap: int = 0
    asr_gpu: str = "auto"


@dataclass(frozen=True)
class GpuFacts:
    name: str = ""
    total_mb: int | None = None
    free_mb: int | None = None
    baseline_mb: int | None = None      # VRAM in use when the app started (everything but this app); None = unknown
    stable: bool = True                 # the NVML samples agreed within 256 MiB
    swap_resident: bool | None = None   # llama-swap holds a model; None = could not ask
    power_capped: bool = False          # NVML reports a power-cap clock event (laptop TGP)
    power_w: float | None = None
    power_limit_w: float | None = None
    sm_mhz: int | None = None
    sm_max_mhz: int | None = None
    temperature_c: int | None = None


@dataclass(frozen=True)
class Machine:
    cpu: str = ""
    physical_cores: int = 0
    ram_total_mb: int = 0
    ram_free_mb: int = 0
    mem_mts: int | None = None          # DDR transfer rate, informational
    gpu: GpuFacts | None = None
    igpu: str = ""                      # OpenVINO device name of an Intel/AMD integrated GPU ("" = none)
    npu: str = ""                       # OpenVINO NPU name ("" = none)
    devices_known: bool = False         # the devices probe has answered (or its cache is fresh)
    ct2_cuda: bool = False              # CTranslate2 sees a CUDA device (Whisper can run on the GPU)
    sherpa_ready: bool = False          # the Parakeet bundle is on disk (nothing is ever downloaded by a plan)
    runtime: str = ""                   # e.g. "nvidia-cuda12 2.44.0"
    guard_floor_mb: int = 1500


@dataclass(frozen=True)
class ModelFacts:
    key: str = ""
    size_bytes: int = 0
    arch: str = ""
    n_layer: int = 0
    n_ctx_train: int = 0
    n_head_kv: int | None = None
    head_dim: int | None = None
    kv_heads_total: int | None = None
    full_attention_interval: int | None = None
    nextn_layers: int = 0
    ssm_conv: int | None = None
    ssm_state: int | None = None
    ssm_inner: int | None = None
    ssm_groups: int | None = None

    @property
    def has_mtp(self) -> bool:
        return self.nextn_layers >= 1

    @property
    def is_hybrid(self) -> bool:
        return bool(self.full_attention_interval and self.full_attention_interval > 1)

    @property
    def attn_layers(self) -> int | None:
        """Layers holding a per-token KV cache (all of them for a dense model)."""
        if self.n_layer <= 0:
            return None
        body = max(0, self.n_layer - self.nextn_layers)
        if self.is_hybrid:
            return body // int(self.full_attention_interval)  # type: ignore[arg-type]
        return body


@dataclass(frozen=True)
class LaunchConfig:
    """The llama-server launch the app would do right now (global [llm] overlaid by the profile)."""
    ctx: int = 0
    kv: str = "q8_0"
    parallel: int = 2
    ubatch: int = 512
    threads: int = 0
    flash: bool = True
    gpu_layers: int = 999999
    flags: tuple[tuple[str, str], ...] = ()   # managed flags in effect, sorted
    extra: tuple[str, ...] = ()


@dataclass(frozen=True)
class LlmRun:
    """One measured llama-server configuration (live probe, imported sweep, or an apply)."""
    model: str = ""
    ctx: int = 0
    kv: str = "q8_0"
    parallel: int = 2
    ubatch: int = 512
    flags: tuple[tuple[str, str], ...] = ()   # managed flags that differ from the built-ins, sorted
    pp: float | None = None                   # prompt tokens / s on a fresh prompt
    tg: float | None = None                   # generated tokens / s
    cached_ttft_ms: float | None = None       # wall time to the first token with the prefix cached
    incr_prompt_n: float | None = None        # prompt tokens re-evaluated for an incremental append
    spread: float = 0.0                       # relative run-to-run spread (max of pp, tg)
    gpu_used_mb: float | None = None          # this app's footprint: VRAM used with the server minus before
    load_s: float | None = None
    ok: bool = True
    clean_log: bool = True
    at: str = ""
    source: str = "probe"                     # probe | sweep | apply
    stale: bool = False


@dataclass(frozen=True)
class AsrRow:
    label: str = ""
    engine: str = ""
    device: str = ""
    compute: str = ""
    threads: int = 0
    median_s: float = 0.0
    overlap: float | None = None
    vram_mb: int | None = None
    vram_source: str = "prior"
    at: str = ""
    source: str = "prior"                     # probe | sweep | prior
    stale: bool = False


@dataclass(frozen=True)
class OcrRow:
    label: str = ""
    device: str = ""
    median_ms: float = 0.0
    cpu_ms: float | None = None
    at: str = ""
    source: str = "prior"
    stale: bool = False


@dataclass(frozen=True)
class Footprint:
    model: str = ""
    ctx: int = 0
    kv: str = "q8_0"
    ubatch: int = 512
    parallel: int = 2
    flags: tuple[tuple[str, str], ...] = ()
    app_mb: float = 0.0
    source: str = "probe"
    at: str = ""


@dataclass(frozen=True)
class Evidence:
    llm: tuple[LlmRun, ...] = ()
    asr: tuple[AsrRow, ...] = ()
    ocr: tuple[OcrRow, ...] = ()
    footprints: tuple[Footprint, ...] = ()


@dataclass(frozen=True)
class CfgFacts:
    """The Config numbers the planner reads, flattened so the tests need no Config object."""
    lane_share: float = 0.3
    reserve_tokens: int = 256
    chars_per_token: float = 3.6
    budget_mode: str = "auto"
    max_answer_tokens: int = 700
    screen_max_text_chars: int = 2500
    handsfree_context_chars: int = 6000
    accurate_ocr: bool = True
    partial_interval_s: float = 1.2
    partials_enabled: bool = True
    asr_engine: str = "auto"
    asr_device: str = "auto"
    asr_compute: str = "auto"
    asr_threads: int = 0
    asr_language: str = "en"
    hotwords: bool = False
    asr_now: str = ""                   # label of what runs now ("whisper-cpu-int8") or ""
    ocr_device: str = "auto"
    ocr_now: str = ""                   # the device "auto" resolves to ("igpu") or ""


@dataclass(frozen=True)
class PlanInputs:
    settings: Settings = Settings()
    machine: Machine = Machine()
    model: ModelFacts | None = None
    launch: LaunchConfig | None = None
    running: bool = False               # the app's own llama-server is up (calibration possible)
    backend: str = "llamacpp"
    profile: dict | None = None         # the saved ModelProfile for model.key (asdict), or None
    cfg: CfgFacts = CfgFacts()
    evidence: Evidence = Evidence()
    now_epoch: float = 0.0


@dataclass(frozen=True)
class Item:
    id: str = ""                        # llm | speech | ocr | screen | handsfree | speakers | npu | gpu
    label: str = ""
    now: str = ""
    proposed: str = ""
    changed: bool = False
    applies: str = "none"               # model-restart | live | next-launch | none
    evidence_kind: str = "estimated"    # measured | recorded | estimated
    evidence_text: str = ""
    why: str = ""
    settings: dict[str, object] = field(default_factory=dict)   # dialog paths -> values
    profile: dict[str, object] = field(default_factory=dict)    # ModelProfile field changes
    tick: bool = True
    enabled: bool = True


@dataclass(frozen=True)
class Plan:
    goal: str = "balanced"
    model_key: str = ""
    digest: str = ""
    created_at: float = 0.0
    items: tuple[Item, ...] = ()
    settings: dict[str, object] = field(default_factory=dict)   # union of the ticked items' settings
    profile: dict[str, object] | None = None                    # the llm item's profile changes, or None
    vram: dict[str, object] = field(default_factory=dict)       # the budget breakdown (MiB) + tier
    predicted: dict[str, object] = field(default_factory=dict)  # felt numbers
    devices: dict[str, list[str]] = field(default_factory=dict) # device -> what runs there
    blocked: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    summary: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "Plan":
        return _from_dict(cls, data)


@dataclass(frozen=True)
class FreshFacts:
    free_mb: int | None = None
    loaded_key: str = ""
    swap_resident: bool | None = None
    now_epoch: float = 0.0


@dataclass(frozen=True)
class Activity:
    llm_ready: bool = False
    backend: str = "llamacpp"
    server_running: bool = False
    assist_busy: bool = False
    swap_busy: bool = False
    transcriber_ready: bool = False
    speech_idle_s: float = 1e9
    swap_resident: bool | None = None
    handsfree_on: bool = False
    paused: bool = False


_NESTED: dict[tuple[str, str], tuple[type, bool]] = {
    ("Machine", "gpu"): (GpuFacts, False),
    ("Evidence", "llm"): (LlmRun, True),
    ("Evidence", "asr"): (AsrRow, True),
    ("Evidence", "ocr"): (OcrRow, True),
    ("Evidence", "footprints"): (Footprint, True),
    ("Plan", "items"): (Item, True),
    ("PlanInputs", "settings"): (Settings, False),
    ("PlanInputs", "machine"): (Machine, False),
    ("PlanInputs", "model"): (ModelFacts, False),
    ("PlanInputs", "launch"): (LaunchConfig, False),
    ("PlanInputs", "cfg"): (CfgFacts, False),
    ("PlanInputs", "evidence"): (Evidence, False),
}


def to_dict(obj: Any) -> Any:
    """Plain JSON-able data for any dataclass / tuple here."""
    return _to_dict(obj)


def from_dict(cls: type, data: Any) -> Any:
    """Inverse of :func:`to_dict` for the dataclasses of this module (tolerant, never raises)."""
    return _from_dict(cls, data)


# ============================================================================================
# launch config helpers
# ============================================================================================
def parse_launch_args(args: Sequence[str]) -> dict[str, str]:
    """``--flag value`` pairs from a llama-server argument list, LAST occurrence wins (that is how
    llama.cpp's own parser resolves a repeated flag); a flag followed by another flag or by nothing
    is a boolean and reads ``"1"``."""
    out: dict[str, str] = {}
    i = 0
    while i < len(args):
        a = str(args[i])
        if a.startswith("--"):
            if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                out[a] = str(args[i + 1])
                i += 2
                continue
            out[a] = "1"
        i += 1
    return out


def merge_extra_args(existing: Sequence[str], managed: dict[str, str | None]) -> list[str]:
    """``existing`` with every MANAGED flag (and its value) removed, in order, then ``managed``
    appended as ``--flag value`` (a ``None`` value removes without re-adding). Foreign flags such
    as ``--n-cpu-moe 4`` are preserved exactly."""
    out: list[str] = []
    i = 0
    ex = [str(x) for x in existing]
    while i < len(ex):
        a = ex[i]
        if a in MANAGED_FLAGS or a in managed:
            i += 1
            if i < len(ex) and not ex[i].startswith("--"):
                i += 1
            continue
        out.append(a)
        i += 1
    for flag in sorted(managed):
        value = managed[flag]
        if value is None:
            continue
        out += [flag, str(value)]
    return out


def managed_flags_of(args: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """The managed flags present in ``args`` as a sorted tuple of pairs (the identity of a run)."""
    parsed = parse_launch_args(args)
    return tuple(sorted((k, v) for k, v in parsed.items() if k in MANAGED_FLAGS))


def effective_launch(llm_cfg: Any, profile: dict | None) -> LaunchConfig:
    """What ``LlamaServerManager.build_args`` would launch: the global ``[llm]`` values, each one
    overridden field by field by a non-``None`` profile value (its ``_pf`` rule), the profile's
    ``extra_args`` appended after ``llama_extra_args`` (a later ``--ctx-size`` etc. wins). A test
    pins this against the real ``build_args`` so the two cannot drift."""
    prof = profile or {}

    def _pf(name: str, default: Any) -> Any:
        v = prof.get(name)
        return default if v is None else v

    extra = list(getattr(llm_cfg, "llama_extra_args", []) or []) + list(prof.get("extra_args") or [])
    parsed = parse_launch_args(extra)
    ctx = int(_pf("ctx_size", getattr(llm_cfg, "ctx_size", 8192)))
    parallel = int(_pf("parallel", getattr(llm_cfg, "parallel", 2)))
    ubatch = 512
    try:
        ctx = int(parsed.get("--ctx-size", ctx))
        parallel = int(parsed.get("--parallel", parallel))
        ubatch = int(parsed.get("--ubatch-size", ubatch))
    except ValueError:
        pass
    kv = str(_pf("kv_cache_type", getattr(llm_cfg, "kv_cache_type", "q8_0")) or "f16").strip().lower()
    kv = parsed.get("--cache-type-k", kv).strip().lower()
    threads = int(_pf("threads", getattr(llm_cfg, "threads", 0)) or 0)
    flash = bool(_pf("flash_attn", getattr(llm_cfg, "flash_attn", True)))
    gpu_layers = int(_pf("gpu_layers", getattr(llm_cfg, "gpu_layers", 999999)))
    return LaunchConfig(ctx=ctx, kv=kv, parallel=max(1, parallel), ubatch=ubatch, threads=threads, flash=flash,
                        gpu_layers=gpu_layers, flags=managed_flags_of(extra), extra=tuple(extra))


# ============================================================================================
# VRAM math
# ============================================================================================
def kv_bytes_per_token(m: ModelFacts, kv: str) -> tuple[int | None, str]:
    """``(bytes per context token, tier)``. Hybrid models (``full_attention_interval`` in the
    header) count only their attention layers: ``2 * attn_layers * n_head_kv * head_dim * elem``
    -> tier ``"header"``. A dense model uses ``kv_heads_total`` (already summed over layers) ->
    also ``"header"``. A hybrid header missing ``n_head_kv`` or ``head_dim`` falls back to the
    dense formula over ``kv_heads_total`` -- an upper bound, tier ``"pessimistic"``. No head counts
    at all: ``(None, "unknown")``."""
    elem = ELEM.get((kv or "f16").strip().lower(), ELEM["f16"])
    if m.is_hybrid and m.n_head_kv and m.head_dim and m.attn_layers:
        return int(round(2 * m.attn_layers * m.n_head_kv * m.head_dim * elem)), "header"
    if m.kv_heads_total and m.head_dim:
        return int(round(2 * m.kv_heads_total * m.head_dim * elem)), ("pessimistic" if m.is_hybrid else "header")
    return None, "unknown"


def recurrent_mb(m: ModelFacts, slots: int) -> float:
    """Fixed recurrent state of a hybrid model's non-attention layers for ``slots`` sequences
    (MiB): per layer ``inner * state * 4`` (S) + ``(conv - 1) * (inner + 2 * groups * state) * 4``
    (R). ~150 MiB per slot for Qwen3.5/3.8-27B. 0 for a dense model."""
    if not m.is_hybrid or m.attn_layers is None:
        return 0.0
    n_rec = max(0, m.n_layer - m.nextn_layers - m.attn_layers)
    if not (m.ssm_inner and m.ssm_state and m.ssm_conv):
        return STATE_MB_PER_SLOT_FALLBACK * max(1, slots)
    groups = m.ssm_groups or 1
    per_layer = m.ssm_inner * m.ssm_state * 4 + (m.ssm_conv - 1) * (m.ssm_inner + 2 * groups * m.ssm_state) * 4
    return n_rec * per_layer * max(1, slots) / MIB


def compute_mb(ubatch: int) -> float:
    """SEED compute-buffer estimate for a ``--ubatch-size`` (MiB), linear between the table points."""
    pts = sorted(COMPUTE_MB.items())
    if ubatch <= pts[0][0]:
        return pts[0][1]
    for (u0, v0), (u1, v1) in zip(pts, pts[1:]):
        if u0 <= ubatch <= u1:
            return v0 + (v1 - v0) * (ubatch - u0) / (u1 - u0)
    return pts[-1][1]


def fit_footprint(points: Sequence[Footprint], header_bpt: int | None) -> tuple[float, float] | None:
    """Least-squares ``app_mb = a + b * ctx`` over measured footprints of one configuration;
    ``None`` with fewer than two points or a ctx range narrower than 1.5x. A slope outside 0.5x to
    1.6x of the header slope is rejected (something else moved on the GPU during a measurement)."""
    pts = [(p.ctx, p.app_mb) for p in points if p.ctx > 0 and p.app_mb > 0]
    if len(pts) < 2:
        return None
    xs = [x for x, _ in pts]
    if min(xs) <= 0 or max(xs) / min(xs) < 1.5:
        return None
    n = len(pts)
    mx = sum(xs) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    a = my - b * mx
    if b <= 0:
        return None
    if header_bpt:
        hb = header_bpt / MIB
        if not (0.5 * hb <= b <= 1.6 * hb):
            return None
    return a, b


@dataclass(frozen=True)
class VramModel:
    """``need(ctx) = fixed + ctx * mpt + unc * (ctx * mpt + var_fixed) + reserve``; ``max_ctx`` is
    its exact inverse rounded down to :data:`CTX_STEP`."""
    fixed_mb: float
    mib_per_token: float
    var_fixed_mb: float
    unc: float
    tier: str
    reserve_mb: float
    avail_mb: float
    parts: dict[str, float] = field(default_factory=dict)

    def need(self, ctx: int) -> float:
        var = ctx * self.mib_per_token
        return self.fixed_mb + var + self.unc * (var + self.var_fixed_mb) + self.reserve_mb

    def max_ctx(self) -> int:
        if self.mib_per_token <= 0:
            return 0
        room = self.avail_mb - self.reserve_mb - self.fixed_mb - self.unc * self.var_fixed_mb
        if room <= 0:
            return 0
        ctx = int(room / (self.mib_per_token * (1 + self.unc)))
        return ctx - ctx % CTX_STEP


def _reserve_mb(settings: Settings, goal: str, guard_floor_mb: int) -> float:
    floor = guard_floor_mb + 256
    if settings.reserve_vram_mb and settings.reserve_vram_mb > 0:
        return float(max(settings.reserve_vram_mb, floor))
    return float(floor + GOAL_SLACK_MB.get(goal, 256))


def _others_mb(gpu: GpuFacts, own_now_mb: float | None) -> float:
    """VRAM held by everything that is not this app."""
    total = float(gpu.total_mb or 0)
    if gpu.baseline_mb is not None and 0 <= gpu.baseline_mb <= 0.5 * total:
        return float(gpu.baseline_mb)
    used = total - float(gpu.free_mb or 0)
    if own_now_mb is not None:
        return max(0.0, used - own_now_mb)
    return max(0.0, used)


def vram_model(inp: PlanInputs, *, goal: str, kv: str, ubatch: int, parallel: int,
               asr_mb: float = 0.0, extra_mb: float = 0.0) -> VramModel | None:
    """The VRAM model for one candidate launch, picking the best evidence tier available:
    ``fit`` (>= 2 measured footprints of this configuration), ``calibrated`` (the server is up and
    its footprint is known, so the fixed overhead is measured), ``header`` (GGUF arithmetic plus
    seeds), ``pessimistic`` (dense upper bound). ``None`` when the GPU totals or the head counts
    are unknown -- the caller then gives advice instead of a number."""
    m, gpu = inp.model, inp.machine.gpu
    if m is None or gpu is None or not gpu.total_mb:
        return None
    bpt, tier = kv_bytes_per_token(m, kv)
    if bpt is None:
        return None
    unc = UNC["pessimistic"] if tier == "pessimistic" else UNC["header"]
    weights = m.size_bytes / MIB
    if m.n_layer and inp.launch is not None and 0 < inp.launch.gpu_layers < m.n_layer:
        weights *= inp.launch.gpu_layers / m.n_layer
    state = recurrent_mb(m, parallel)
    mpt = bpt / MIB
    reserve = _reserve_mb(inp.settings, goal, inp.machine.guard_floor_mb)

    # how much this app holds right now (running server): used - others
    own_now: float | None = None
    launch = inp.launch
    if inp.running and launch is not None and gpu.free_mb is not None and gpu.baseline_mb is not None:
        own_now = (gpu.total_mb - gpu.free_mb) - gpu.baseline_mb
        if not (1.0 < own_now < 0.95 * gpu.total_mb):
            own_now = None
    if not gpu.stable:
        own_now = None

    parts: dict[str, float] = {"weights": weights, "state": state, "asr": asr_mb, "extra": extra_mb}
    pts = [p for p in inp.evidence.footprints
           if p.model == m.key and p.kv == kv and p.ubatch == ubatch and p.parallel == parallel]
    fitted = fit_footprint(pts, bpt)
    if fitted is not None:
        a, b = fitted
        # Measured footprints only ever LOWER the overhead estimate, never the physics: the slope
        # is floored at the header's bytes-per-token (the 2026-10-07 sweep saw the same footprint
        # at 81,920 and 131,072 tokens, then a GPU filled to the last 57 MiB at 204,800 -- the
        # cache is paid in full once the window is used) and the intercept at weights + recurrent
        # state. The fit contributes what it can measure: how small the fixed buffers really are.
        b = max(b, mpt)
        a = max(a, weights + state)
        fixed = a + asr_mb + extra_mb
        parts["overhead"] = max(0.0, a - weights - state)
        model = VramModel(fixed, b, 0.0, UNC["fit"], "fit", reserve, 0.0, parts)
    elif own_now is not None and launch is not None:
        cur_bpt, _t = kv_bytes_per_token(m, launch.kv)
        cur_kv = (cur_bpt or bpt) / MIB * launch.ctx
        overhead = own_now - weights - cur_kv - recurrent_mb(m, launch.parallel)
        if 300.0 <= overhead <= 3000.0:
            fixed = weights + state + overhead + (compute_mb(ubatch) - compute_mb(launch.ubatch)) + asr_mb + extra_mb
            parts["overhead"] = overhead
            model = VramModel(fixed, mpt, state, UNC["calibrated"], "calibrated", reserve, 0.0, parts)
        else:
            model = None
    else:
        model = None
    if model is None:
        comp = compute_mb(ubatch)
        fixed = weights + state + comp + RUNTIME_MB + asr_mb + extra_mb
        parts["overhead"] = comp + RUNTIME_MB
        model = VramModel(fixed, mpt, state + comp, unc, tier, reserve, 0.0, parts)
    avail = float(gpu.total_mb) - _others_mb(gpu, own_now)
    return VramModel(model.fixed_mb, model.mib_per_token, model.var_fixed_mb, model.unc, model.tier,
                     model.reserve_mb, avail, model.parts)


# ============================================================================================
# evidence helpers
# ============================================================================================
def typical_answer_s(r: LlmRun) -> float | None:
    """A felt answer time for one measured run: ~1000 uncached prompt tokens, a 150-token answer
    and the cached time-to-first-token. ``None`` when pp or tg is missing."""
    if not r.pp or not r.tg or r.pp <= 0 or r.tg <= 0:
        return None
    return UNCACHED_PROMPT_TOKENS / r.pp + ANSWER_TOKENS / r.tg + (r.cached_ttft_ms or 0.0) / 1000.0


def _usable(r: LlmRun) -> bool:
    return r.ok and r.clean_log and not r.stale and typical_answer_s(r) is not None


def _baseline_runs(ev: Evidence, model: str, kv: str, parallel: int) -> list[LlmRun]:
    return [r for r in ev.llm if _usable(r) and r.model == model and r.kv == kv and r.parallel == parallel and not r.flags]


def ctx_is_free(ev: Evidence, model: str, kv: str, parallel: int) -> bool | None:
    """Does a longer window cost answer time on this machine? From two usable baseline runs with
    ctx at least 2x apart: within 3 % -> ``True``; the longer one worse by more than
    ``max(3 %, 2 x spread)`` -> ``False``; otherwise, or without data, ``None``."""
    runs = sorted(_baseline_runs(ev, model, kv, parallel), key=lambda r: r.ctx)
    if len(runs) < 2:
        return None
    lo, hi = runs[0], runs[-1]
    if lo.ctx <= 0 or hi.ctx / lo.ctx < 2.0:
        return None
    a, b = typical_answer_s(lo), typical_answer_s(hi)
    if a is None or b is None or a <= 0:
        return None
    worse = (b - a) / a
    if worse <= 0.03:
        return True
    if worse > max(0.03, 2 * max(lo.spread, hi.spread)):
        return False
    return None


def best_run(ev: Evidence, model: str, kv: str, parallel: int, ctx: int) -> LlmRun | None:
    """The usable baseline run nearest ``ctx`` (for the felt numbers), or ``None``."""
    runs = _baseline_runs(ev, model, kv, parallel)
    if not runs:
        return None
    return min(runs, key=lambda r: (abs(r.ctx - ctx), 0 if r.source == "probe" else 1))


# ============================================================================================
# choices: speech, OCR, knobs
# ============================================================================================
@dataclass(frozen=True)
class AsrChoice:
    row: AsrRow | None
    settings: dict[str, object]
    vram_mb: float
    why: str
    evidence_kind: str
    warnings: tuple[str, ...] = ()


def _asr_rows(inp: PlanInputs) -> list[AsrRow]:
    """Evidence rows first (fresh probes, then sweeps), then the priors for labels not measured."""
    rows: list[AsrRow] = []
    seen: set[str] = set()
    for src in ("probe", "sweep"):
        for r in inp.evidence.asr:
            if r.source == src and not r.stale and r.label not in seen and r.median_s > 0:
                rows.append(r)
                seen.add(r.label)
    for label, (engine, device, compute, threads, median, vram, overlap) in PRIORS_ASR.items():
        if label not in seen:
            rows.append(AsrRow(label=label, engine=engine, device=device, compute=compute, threads=threads,
                               median_s=median, overlap=overlap, vram_mb=vram, vram_source="prior", source="prior"))
    return rows


def _asr_label_now(cfg: CfgFacts, machine: Machine) -> str:
    if cfg.asr_now:
        return cfg.asr_now
    engine = cfg.asr_engine
    if engine == "auto":
        engine = "faster-whisper" if (machine.ct2_cuda or cfg.asr_device == "cuda") else "sherpa"
    if engine == "sherpa":
        return "sherpa-cpu"
    device = cfg.asr_device
    if device == "auto":
        device = "cuda" if machine.ct2_cuda else "cpu"
    if device == "cpu":
        return "whisper-cpu-int8"
    compute = cfg.asr_compute if cfg.asr_compute != "auto" else "float16"
    return f"whisper-cuda-{compute}"


def _asr_settings_for(row: AsrRow) -> dict[str, object]:
    if row.engine == "sherpa":
        return {"asr.engine": "sherpa", "asr.cpu_threads": int(row.threads or 4)}
    out: dict[str, object] = {"asr.engine": "faster-whisper", "asr.device": row.device, "asr.compute_type": row.compute}
    if row.device == "cpu":
        out["asr.cpu_threads"] = int(row.threads or 4)
    return out


def choose_asr(inp: PlanInputs, goal: str, ctx_after: Callable[[float], int] | None = None) -> AsrChoice:
    """Speech placement: the lowest-VRAM engine that keeps up with the partial interval (median
    per utterance <= 50 % of ``[asr].partial_interval_s``), ties within 0.1 s going to the faster.
    Parakeet (sherpa) needs its bundle on disk, an English or European language and no hotwords;
    Whisper on CUDA needs CTranslate2's CUDA and ``asr_gpu != "never"``. ``latency`` takes the
    fastest viable engine instead when it is > 0.1 s per utterance quicker and the window stays
    >= 32k tokens (speed is what that goal is for; ``asr_gpu = "never"`` still keeps Whisper off
    the GPU -- the owner's 2026-09-25 rule, now a switch). ``quality`` prefers a GPU Whisper row
    only when it is measurably more accurate (>= 0.03 word overlap)."""
    cfg, machine = inp.cfg, inp.machine
    rows = _asr_rows(inp)
    now_label = _asr_label_now(cfg, machine)
    now_row = next((r for r in rows if r.label == now_label), None)
    keepup = KEEPUP_SHARE * cfg.partial_interval_s if cfg.partials_enabled else KEEPUP_NO_PARTIALS_S
    warnings: list[str] = []
    viable: list[AsrRow] = []
    for r in rows:
        if r.engine == "sherpa":
            if not machine.sherpa_ready:
                continue
            if cfg.hotwords or (cfg.asr_language or "en").lower() not in SHERPA_LANGUAGES:
                continue
        elif r.device == "cuda":
            if not machine.ct2_cuda or inp.settings.asr_gpu == "never":
                continue
        if r.median_s > keepup:
            continue
        viable.append(r)
    if not viable:
        why = (f"nothing installed keeps up with {cfg.partial_interval_s:.1f} s partials"
               if now_row is None else
               f"{now_row.label} takes {now_row.median_s:.2f} s per utterance; no faster engine is available here")
        return AsrChoice(now_row, {}, float((now_row.vram_mb if now_row else 0) or 0), why, "estimated", (why,))
    viable.sort(key=lambda r: (int(r.vram_mb or 0), r.median_s))
    best = viable[0]
    for r in viable[1:]:
        if int(r.vram_mb or 0) == int(best.vram_mb or 0) and r.median_s + 0.10 < best.median_s:
            best = r
    gpu_rows = [r for r in viable if r.device == "cuda"]
    if goal == "latency":
        # The latency goal is the one place speed buys GPU memory: the fastest viable engine wins
        # when it is clearly faster (> 0.1 s per utterance) and the window stays usable. "never"
        # (asr_gpu) already removed the GPU rows above, so the owner's rule still has a switch.
        fastest = min(viable, key=lambda r: (r.median_s, int(r.vram_mb or 0)))
        if fastest is not best and best.median_s - fastest.median_s > 0.10 and \
                (ctx_after is None or int(fastest.vram_mb or 0) == 0 or ctx_after(float(fastest.vram_mb or 0)) >= QUALITY_MIN_CTX):
            best = fastest
    if goal == "quality" and gpu_rows and best.device != "cuda":
        gpu_best = min(gpu_rows, key=lambda r: (int(r.vram_mb or 0), r.median_s))
        gain = (gpu_best.overlap or 0) - (best.overlap or 0)
        if gain >= QUALITY_OVERLAP_GAIN and (ctx_after is None or ctx_after(float(gpu_best.vram_mb or 0)) >= QUALITY_MIN_CTX):
            best = gpu_best
    cost_line = ""
    if gpu_rows and best.device != "cuda":
        g = min(gpu_rows, key=lambda r: int(r.vram_mb or 0))
        tokens = ""
        if ctx_after is not None and inp.model is not None:
            bpt, _t = kv_bytes_per_token(inp.model, inp.launch.kv if inp.launch else "q8_0")
            if bpt:
                tokens = f", about {int((g.vram_mb or 0) * MIB / bpt) // 1000}k tokens of window"
        cost_line = f" Whisper on the GPU would take {g.median_s:.2f} s but holds {(g.vram_mb or 0) / 1024:.2f} GB{tokens}."
    kinds = {"probe": "measured", "sweep": "recorded", "prior": "estimated"}
    if best.label == now_label:
        why = f"{best.label}: {best.median_s:.2f} s per utterance, {int(best.vram_mb or 0)} MiB of GPU memory -- already the best fit.{cost_line}"
        return AsrChoice(best, {}, float(best.vram_mb or 0), why, kinds.get(best.source, "estimated"), tuple(warnings))
    now_txt = f"{now_row.label} ({now_row.median_s:.2f} s per utterance)" if now_row else now_label
    why = (f"{now_txt} -> {best.label}: {best.median_s:.2f} s per utterance, {int(best.vram_mb or 0)} MiB of GPU memory, "
           f"word overlap {best.overlap if best.overlap is not None else '?'}.{cost_line}")
    if now_row is not None and now_row.median_s > keepup:
        warnings.append(f"Speech recognition cannot keep up today: {now_row.label} takes {now_row.median_s:.1f} s per "
                        f"utterance against {cfg.partial_interval_s:.1f} s partials.")
    return AsrChoice(best, _asr_settings_for(best), float(best.vram_mb or 0), why, kinds.get(best.source, "estimated"), tuple(warnings))


@dataclass(frozen=True)
class OcrChoice:
    device: str
    settings: dict[str, object]
    why: str
    evidence_kind: str


def choose_ocr(inp: PlanInputs) -> OcrChoice:
    """RapidOCR device (the accurate re-read engine): the lowest ``wall + 0.1 * cpu`` among the
    devices OpenVINO reports (iGPU, NPU, CPU), measured rows first, priors otherwise. No change when
    the winner is what ``auto`` already resolves to."""
    m, cfg = inp.machine, inp.cfg
    allowed = ["cpu"] + (["igpu"] if m.igpu else []) + (["npu"] if m.npu else [])
    rows: dict[str, OcrRow] = {}
    for r in inp.evidence.ocr:
        if r.device in allowed and not r.stale and r.median_ms > 0 and (r.device not in rows or r.source == "probe"):
            rows[r.device] = r
    for dev in allowed:
        if dev not in rows and dev in PRIORS_OCR:
            wall, cpu = PRIORS_OCR[dev]
            rows[dev] = OcrRow(label=f"rapidocr/{dev}", device=dev, median_ms=wall, cpu_ms=cpu)

    def _score(r: OcrRow) -> float:
        return r.median_ms + 0.1 * (r.cpu_ms or 0.0)

    if not m.devices_known:
        return OcrChoice(cfg.ocr_now or "auto", {}, "not checked yet: run the benchmark to probe the OpenVINO devices", "estimated")
    winner = min(rows.values(), key=_score)
    now = cfg.ocr_now or ("igpu" if m.igpu else "npu" if m.npu else "cpu")
    kind = "measured" if winner.source == "probe" else "recorded" if winner.source == "sweep" else "estimated"
    table = ", ".join(f"{r.device} {r.median_ms / 1000:.1f} s" for r in sorted(rows.values(), key=_score))
    if winner.device == now and cfg.ocr_device == "auto":
        return OcrChoice(now, {}, f"auto already picks the {now} ({table}); Windows OCR keeps sampling on the CPU", kind)
    if winner.device == (cfg.ocr_device if cfg.ocr_device != "auto" else now):
        return OcrChoice(winner.device, {}, f"{winner.device} is the fastest RapidOCR device here ({table})", kind)
    return OcrChoice(winner.device, {"screen.ocr_device": winner.device},
                     f"{now} -> {winner.device}: fastest accurate re-read ({table})", kind)


@dataclass(frozen=True)
class KnobChoice:
    flags: dict[str, str] = field(default_factory=dict)   # managed flag -> value to apply
    parallel: int | None = None                           # a slot-count change (single-slot speculation)
    extra_mb: float = 0.0                                 # measured VRAM cost of the adopted knobs
    gains: dict[str, float] = field(default_factory=dict) # group -> relative gain
    notes: tuple[str, ...] = ()
    evidence_kind: str = "estimated"


#: knob groups: name -> (flags that identify the group, metric, goals, min gain)
_KNOB_GROUPS: dict[str, tuple[tuple[str, ...], tuple[str, ...], float]] = {
    "ubatch": (("--ubatch-size",), ("latency", "balanced"), NOISE_FLOOR),
    "spec": (("--spec-type", "--spec-draft-n-max"), ("latency", "balanced"), SPEC_GAIN),
    "threads_batch": (("--threads-batch",), ("latency", "balanced"), NOISE_FLOOR),
    "load_mode": (("--load-mode",), ("latency", "balanced", "context", "quality"), NOISE_FLOOR),
}


def _group_of(flags: tuple[tuple[str, str], ...]) -> str | None:
    names = {f for f, _v in flags}
    for group, (gflags, _goals, _gain) in _KNOB_GROUPS.items():
        if names and names <= set(gflags):
            return group
    return None


def choose_knobs(inp: PlanInputs, goal: str) -> KnobChoice:
    """Launch flags with a MEASURED win on this machine, else nothing. A candidate run differs from
    its baseline (same model, kv, ctx; no managed flags) in exactly one knob group; it is adopted
    when ``gain >= max(group floor, 2 x spread)`` on :func:`typical_answer_s`, its log was clean,
    and the goal allows the group. ``--spec-type draft-mtp`` additionally needs the model's MTP
    head. A single-slot run (``--parallel 1`` + MTP) is the ``single_slot_spec`` group, latency
    goal only: a notes job then waits behind a live answer, but KV shifting returns."""
    m, launch = inp.model, inp.launch
    if m is None or launch is None:
        return KnobChoice()
    runs = [r for r in inp.evidence.llm if r.model == m.key and _usable(r)]
    base_by_ctx: dict[tuple[str, int, int], LlmRun] = {}
    for r in runs:
        if not r.flags:
            base_by_ctx[(r.kv, r.parallel, r.ctx)] = r
    flags: dict[str, str] = {}
    gains: dict[str, float] = {}
    notes: list[str] = []
    rejected: dict[str, tuple[float, float]] = {}   # group -> (best gain seen, the floor it had to clear)
    extra_mb = 0.0
    parallel: int | None = None
    kinds: set[str] = set()
    for r in runs:
        if not r.flags:
            continue
        group = _group_of(r.flags)
        single = r.parallel == 1 and launch.parallel > 1 and group == "spec"
        if group is None:
            continue
        gflags, goals, min_gain = _KNOB_GROUPS[group]
        if single:
            if goal != "latency":
                continue
            base = base_by_ctx.get((r.kv, launch.parallel, r.ctx))
        else:
            if goal not in goals:
                continue
            base = base_by_ctx.get((r.kv, r.parallel, r.ctx))
        if base is None or r.kv != launch.kv:
            continue
        spec_type = dict(r.flags).get("--spec-type", "")
        if spec_type == "draft-mtp" and not m.has_mtp:
            continue
        if spec_type and spec_type not in SPEC_SELF_TYPES:
            continue
        a, b = typical_answer_s(base), typical_answer_s(r)
        if a is None or b is None or a <= 0:
            continue
        gain = (a - b) / a
        floor = max(min_gain, 2 * max(base.spread, r.spread))
        key = "single_slot_spec" if single else group
        if gain < floor:
            best = rejected.get(key)
            if best is None or gain > best[0]:
                rejected[key] = (gain, floor)
            continue
        if key in gains and gains[key] >= gain:
            continue
        gains[key] = gain
        for f, v in r.flags:
            flags[f] = v
        if single:
            parallel = 1
        if r.gpu_used_mb is not None and base.gpu_used_mb is not None:
            extra_mb = max(extra_mb, r.gpu_used_mb - base.gpu_used_mb)
        kinds.add(r.source)
    for key, (gain, floor) in rejected.items():
        if key not in gains:
            notes.append(f"{key} measured {gain * 100:+.0f} % (needs +{floor * 100:.0f} % at this run-to-run noise) -- not applied")
    if parallel == 1:
        notes.append("single slot: a notes or summary job then waits behind a live answer; KV shifting "
                     "returns, so a window advance no longer re-reads everything")
    kind = "measured" if "probe" in kinds or "apply" in kinds else "recorded" if kinds else "estimated"
    return KnobChoice(flags, parallel, max(0.0, extra_mb), gains, tuple(notes), kind)


# ============================================================================================
# the plan
# ============================================================================================
def _floor_step(ctx: float) -> int:
    c = int(ctx)
    return max(0, c - c % CTX_STEP)


def working_set_floor(cfg: CfgFacts, parallel: int) -> int:
    """The smallest window the app's own prompt pieces need: system prompt + screen block +
    reference / lookup blocks + the biggest answer budget + reserve, with the notes lane's share
    when two slots share one KV pool."""
    screen_tokens = cfg.screen_max_text_chars / max(2.0, cfg.chars_per_token)
    body = 900 + screen_tokens + 1200 + 600 + cfg.max_answer_tokens + cfg.reserve_tokens
    if parallel > 1:
        body /= max(0.1, 1.0 - cfg.lane_share)
    need = int(math.ceil(body / CTX_STEP) * CTX_STEP)
    return max(CTX_FLOOR, need)


def window_numbers(ctx: int, cfg: CfgFacts, parallel: int, pp: float | None) -> dict[str, object]:
    """Felt numbers for a window of ``ctx`` tokens: the answer lane's tokens, how many hours of
    meeting that holds, a cold re-read time at ``pp`` tokens/s."""
    lane = ctx * (1.0 - cfg.lane_share) if parallel > 1 else float(ctx)
    answer_tokens = max(0, int(lane) - cfg.reserve_tokens)
    speed = pp if pp and pp > 0 else PP_TYPICAL_TPS
    return {
        "ctx": ctx,
        "answer_tokens": answer_tokens,
        "hours_held": round(answer_tokens * TRANSCRIPT_SHARE / TRANSCRIPT_TOK_PER_MIN / 60.0, 2),
        "cold_reread_s": round(answer_tokens / speed, 1),
        "pp_basis": "measured" if pp else "typical",
    }


def plan_digest(goal: str, model_key: str, profile: dict | None, settings: dict) -> str:
    raw = json.dumps({"goal": goal, "model": model_key, "profile": profile, "settings": settings},
                     sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _clamp(v: int, lo: int, hi: int, step: int) -> int:
    v = max(lo, min(hi, v))
    return v - (v - lo) % step


def recommend(inp: PlanInputs, goal: str | None = None) -> Plan:
    """The plan for ``goal`` (default: ``inp.settings.goal``). Pure; see the module docstring for
    the policy. Every decision lands as an :class:`Item` with its evidence and reason."""
    goal = (goal or inp.settings.goal or "balanced").lower()
    if goal not in GOALS:
        goal = "balanced"
    items: list[Item] = []
    warnings: list[str] = []
    blocked: list[str] = []
    summary: list[str] = []
    m, launch, cfg, machine = inp.model, inp.launch, inp.cfg, inp.machine
    gpu = machine.gpu
    model_key = m.key if m else ""

    if gpu is not None and gpu.swap_resident is True:
        blocked.append("The shared model server (llama-swap) holds a model on the GPU; unload it before applying")

    # ---- speech first: its GPU memory comes off the budget ------------------------------
    def _ctx_after(asr_mb: float) -> int:
        vm = vram_model(inp, goal="quality", kv=launch.kv if launch else "q8_0",
                        ubatch=launch.ubatch if launch else 512, parallel=launch.parallel if launch else 2,
                        asr_mb=asr_mb)
        return vm.max_ctx() if vm else 0

    asr = choose_asr(inp, goal, _ctx_after)
    warnings.extend(asr.warnings)
    now_label = _asr_label_now(cfg, machine)
    items.append(Item(
        id="speech", label="Speech recognition", now=now_label,
        proposed=asr.row.label if asr.row else now_label, changed=bool(asr.settings),
        applies="next-launch" if asr.settings else "none", evidence_kind=asr.evidence_kind,
        evidence_text=(f"{asr.row.median_s:.2f} s/utterance, {int(asr.row.vram_mb or 0)} MiB" if asr.row else ""),
        why=asr.why, settings=dict(asr.settings), tick=bool(asr.settings),
    ))
    if asr.settings and asr.row is not None:
        summary.append(f"Speech: {now_label} -> {asr.row.label} ({asr.row.median_s:.2f} s per utterance)")

    # ---- the language model -------------------------------------------------------------
    profile_changes: dict[str, object] | None = None
    vram_out: dict[str, object] = {}
    predicted: dict[str, object] = {}
    kv = launch.kv if launch else "q8_0"
    if inp.backend != "llamacpp":
        items.append(Item(id="llm", label="Answers (language model)", now=f"served by {inp.backend}",
                          proposed="unchanged", applies="none", enabled=False,
                          why=f"the {inp.backend} backend manages its own model and context", tick=False))
    elif m is None or launch is None or gpu is None or not gpu.total_mb:
        items.append(Item(id="llm", label="Answers (language model)", now=f"{launch.ctx if launch else '?'} tokens",
                          proposed="advice only", applies="none", enabled=False, tick=False,
                          why="the GPU memory or the model header could not be read; use Models > Tuning > Auto-tune"))
    else:
        knobs = choose_knobs(inp, goal)
        parallel = knobs.parallel or launch.parallel
        ubatch = int(knobs.flags.get("--ubatch-size", launch.ubatch))
        kv = "f16" if goal == "quality" else "q8_0"
        vm = vram_model(inp, goal=goal, kv=kv, ubatch=ubatch, parallel=parallel, asr_mb=asr.vram_mb, extra_mb=knobs.extra_mb)
        if goal == "quality" and vm is not None and vm.max_ctx() < QUALITY_MIN_CTX:
            warnings.append(f"a 16-bit KV cache would leave only {vm.max_ctx()} tokens of window; keeping q8_0")
            kv = "q8_0"
            vm = vram_model(inp, goal=goal, kv=kv, ubatch=ubatch, parallel=parallel, asr_mb=asr.vram_mb, extra_mb=knobs.extra_mb)
        if vm is None:
            items.append(Item(id="llm", label="Answers (language model)", now=f"{launch.ctx} tokens, {launch.kv}",
                              proposed="advice only", applies="none", enabled=False, tick=False,
                              why="the model header carries no attention head counts; use Models > Tuning > Auto-tune"))
        else:
            fit = vm.max_ctx()
            floor = working_set_floor(cfg, parallel)
            top = min(m.n_ctx_train or fit, inp.settings.ctx_cap or 10 ** 9)
            top = max(floor, top - top % CTX_STEP) if top >= CTX_STEP else floor
            share = KV_SHARE[goal]
            ctx = _floor_step(fit * share)
            free_ctx = ctx_is_free(inp.evidence, m.key, kv, parallel)
            run = best_run(inp.evidence, m.key, kv, parallel, ctx)
            pp = run.pp if run is not None else None
            if goal == "latency" and free_ctx is False and pp:
                reread = int(math.ceil((REREAD_TARGET_S * pp + cfg.reserve_tokens) / max(0.1, 1.0 - cfg.lane_share)))
                ctx = min(ctx, _floor_step(reread))
            ctx = _clamp(max(ctx, floor), floor, max(floor, top), CTX_STEP)
            basis = {"fit": "measured footprints", "calibrated": "calibrated against the running server",
                     "header": "GGUF header arithmetic", "pessimistic": "dense upper bound"}[vm.tier]
            vram_out = {
                "tier": vm.tier, "total_mb": gpu.total_mb, "avail_mb": round(vm.avail_mb), "others_mb": round(gpu.total_mb - vm.avail_mb),
                "reserve_mb": round(vm.reserve_mb), "weights_mb": round(vm.parts.get("weights", 0)),
                "state_mb": round(vm.parts.get("state", 0)), "overhead_mb": round(vm.parts.get("overhead", 0)),
                "asr_mb": round(asr.vram_mb), "extra_mb": round(knobs.extra_mb),
                "kv_mb": round(ctx * vm.mib_per_token), "kv_bytes_per_token": int(round(vm.mib_per_token * MIB)),
                "max_ctx": fit, "need_mb": round(vm.need(ctx)), "free_after_mb": round(vm.avail_mb - vm.need(ctx) + vm.reserve_mb),
                "margin_pct": round(vm.unc * 100), "free_at_analysis_mb": gpu.free_mb,
            }
            predicted = window_numbers(ctx, cfg, parallel, pp)
            if run is not None:
                predicted["typical_answer_s"] = round(typical_answer_s(run) or 0.0, 2)
                predicted["pp"] = run.pp
                predicted["tg"] = run.tg
                predicted["cached_ttft_ms"] = run.cached_ttft_ms
            if fit < floor:
                items.append(Item(id="llm", label="Answers (language model)", now=f"{launch.ctx} tokens, {launch.kv}",
                                  proposed="advice only", applies="none", enabled=False, tick=False,
                                  evidence_kind="measured" if vm.tier in ("fit", "calibrated") else "estimated",
                                  evidence_text=f"fits {fit} tokens, needs {floor}",
                                  why=(f"the model does not fit next to the other jobs with a usable window "
                                       f"({fit} < {floor} tokens); use Models > Tuning > Auto-tune for partial offload")))
                warnings.append("the language model does not fit the GPU with a usable context window")
            else:
                existing_extra = list((inp.profile or {}).get("extra_args") or [])
                new_extra = merge_extra_args(existing_extra, dict(knobs.flags)) if knobs.flags else existing_extra
                prof: dict[str, object] = {}
                cur_flags = dict(launch.flags)
                if ctx != launch.ctx:
                    prof["ctx_size"] = ctx
                if kv != launch.kv:
                    prof["kv_cache_type"] = kv
                if not launch.flash:
                    prof["flash_attn"] = True
                if knobs.parallel is not None and knobs.parallel != launch.parallel:
                    prof["parallel"] = knobs.parallel
                if knobs.flags and any(cur_flags.get(f) != v for f, v in knobs.flags.items()):
                    prof["extra_args"] = list(new_extra)
                if prof:
                    prof["note"] = f"Optimize ({goal}), {basis}"
                    prof["generated_from"] = f"optimize:{goal}:{vm.tier}"
                pinned = bool(inp.profile) and inp.profile.get("auto") is False and inp.profile.get("ctx_size") is not None
                tick = bool(prof) and not blocked
                if pinned and ctx < launch.ctx:
                    tick = False
                    warnings.append(f"the saved profile pins {launch.ctx} tokens; the smaller {ctx}-token window is proposed but left unticked")
                flag_txt = " ".join(f"{f} {v}" for f, v in sorted(knobs.flags.items()))
                now_txt = f"{launch.ctx} tokens, {launch.kv} KV, {launch.parallel} slot{'s' if launch.parallel > 1 else ''}" + \
                          (" " + " ".join(f"{f} {v}" for f, v in launch.flags) if launch.flags else "")
                prop_txt = f"{ctx} tokens, {kv} KV, {parallel} slot{'s' if parallel > 1 else ''}" + (f" {flag_txt}" if flag_txt else "")
                why_parts = [
                    f"{vram_out['avail_mb']} MiB usable of {gpu.total_mb} ({vram_out['others_mb']} held by others); "
                    f"weights {vram_out['weights_mb']}, buffers {vram_out['overhead_mb'] + vram_out['state_mb']}, speech {vram_out['asr_mb']}, "
                    f"{vram_out['reserve_mb']} kept free; KV {vram_out['kv_bytes_per_token'] // 1024} KiB/token -> up to {fit} tokens fit ({basis}, {vram_out['margin_pct']} % margin).",
                ]
                if goal == "context":
                    why_parts.append("context: the whole fitting window.")
                elif goal == "latency":
                    why_parts.append("latency: 60 % of the fitting window and 1.3 GB more kept free" +
                                     ("; shortened further because a longer window measurably costs answer time." if free_ctx is False else
                                      "; the window is not shrunk on a guess (no measured latency cost)."))
                else:
                    why_parts.append(f"{goal}: 60 % of the fitting window (room for the display and the shared server).")
                if knobs.gains:
                    why_parts.append("measured wins: " + ", ".join(f"{k} {v * 100:+.0f} %" for k, v in knobs.gains.items()) + ".")
                if knobs.notes:
                    why_parts.append("; ".join(knobs.notes) + ".")
                if kv == "q8_0" and goal != "quality":
                    q4 = vram_model(inp, goal=goal, kv="q4_0", ubatch=ubatch, parallel=parallel, asr_mb=asr.vram_mb, extra_mb=knobs.extra_mb)
                    if q4 is not None and q4.max_ctx() > fit:
                        why_parts.append(f"q4_0 KV would allow {q4.max_ctx()} tokens at a quality cost; set it on Models > Tuning if wanted.")
                items.append(Item(
                    id="llm", label="Answers (language model)", now=now_txt, proposed=prop_txt, changed=bool(prof),
                    applies="model-restart" if prof else "none",
                    evidence_kind=("measured" if vm.tier in ("fit", "calibrated") else "estimated") if not knobs.gains else knobs.evidence_kind,
                    evidence_text=f"{vm.tier}: fits {fit}, plan {ctx}; need {vram_out['need_mb']} MiB",
                    why=" ".join(why_parts), profile=prof, tick=tick, enabled=not blocked,
                ))
                if prof:
                    profile_changes = prof
                    summary.append(f"Window: {launch.ctx} -> {ctx} tokens ({kv} KV, {parallel} slot{'s' if parallel > 1 else ''})" +
                                   (f", flags {flag_txt}" if flag_txt else ""))

    # ---- prompt budgets (screen / hands-free) ------------------------------------------
    scale = BUDGET_SCALE[goal]
    if scale != 1.0:
        new_screen = _clamp(int(round(DEFAULT_SCREEN_CHARS * scale / 250.0) * 250), 500, 8000, 250)
        new_hf = _clamp(int(round(DEFAULT_HANDSFREE_CHARS * scale / 500.0) * 500), 1000, 20000, 500)
        pp_now = float(predicted.get("pp") or PP_TYPICAL_TPS)
        for iid, label, path, cur, new in (
            ("screen", "Screen text per answer", "screen.max_text_chars", cfg.screen_max_text_chars, new_screen),
            ("handsfree", "Hands-free screen context", "handsfree.context_chars", cfg.handsfree_context_chars, new_hf),
        ):
            if new == cur:
                continue
            delta_s = (new - cur) / max(2.0, cfg.chars_per_token) / pp_now
            items.append(Item(
                id=iid, label=label, now=f"{cur} chars", proposed=f"{new} chars", changed=True, applies="live",
                evidence_kind="measured" if predicted.get("pp") else "estimated",
                evidence_text=f"{delta_s:+.1f} s per answer at {pp_now:.0f} tok/s",
                why=(f"{goal}: {'less' if new < cur else 'more'} screen text in every prompt is about {delta_s:+.1f} s per uncached answer "
                     f"at {pp_now:.0f} prompt tokens/s{' (typical)' if not predicted.get('pp') else ''}."),
                settings={path: new},
            ))

    # ---- OCR --------------------------------------------------------------------------
    ocr = choose_ocr(inp)
    items.append(Item(id="ocr", label="Accurate OCR (RapidOCR)", now=cfg.ocr_now or cfg.ocr_device, proposed=ocr.device,
                      changed=bool(ocr.settings), applies="next-launch" if ocr.settings else "none",
                      evidence_kind=ocr.evidence_kind, why=ocr.why, settings=dict(ocr.settings), tick=bool(ocr.settings)))

    # ---- static rows ---------------------------------------------------------------------
    items.append(Item(id="speakers", label="Speaker identification", now="CPU, 2 threads", proposed="CPU, 2 threads",
                      applies="none", tick=False, evidence_kind="estimated",
                      why="WeSpeaker CAM++ over onnxruntime: the installed sherpa/onnxruntime wheels have no accelerated provider, and it costs ~10 ms per utterance"))
    if machine.npu:
        items.append(Item(id="npu", label="NPU", now="idle", proposed="idle", applies="none", tick=False, evidence_kind="estimated",
                          why="only OCR can run there and it is about 2x slower than the iGPU (2.6 s vs 1.3 s); it stays the fallback when the iGPU is absent"))
    if gpu is not None and gpu.power_capped:
        cap = f"{gpu.power_w:.0f} W of a {gpu.power_limit_w:.0f} W limit" if gpu.power_w and gpu.power_limit_w else "its power limit"
        clk = f", {gpu.sm_mhz} MHz of {gpu.sm_max_mhz}" if gpu.sm_mhz and gpu.sm_max_mhz else ""
        items.append(Item(id="gpu", label="GPU power", now=f"power-capped ({cap}{clk})", proposed="see note", applies="none", tick=False,
                          evidence_kind="measured",
                          why=("the GPU runs at its power limit while generating, so batch and thread knobs cannot make it faster from "
                               "inside the app; a higher performance mode in the laptop's power tool (and a cool desk) is the only lever left")))

    # ---- devices strip -------------------------------------------------------------------
    devices: dict[str, list[str]] = {}
    gpu_name = gpu.name if gpu and gpu.name else "GPU"
    if vram_out:
        devices[gpu_name] = [f"weights {vram_out['weights_mb']} MiB", f"KV cache {vram_out['kv_mb']} MiB",
                             f"buffers {vram_out['overhead_mb'] + vram_out['state_mb']} MiB", f"kept free {vram_out['reserve_mb']} MiB"]
    cpu_lines = [f"llama-server threads ({launch.threads or 'auto'})" if launch else "llama-server threads",
                 "speaker identification (2 threads)", "Windows OCR sampling"]
    speech_row = asr.row if asr.row is not None else None
    if speech_row is not None and speech_row.device != "cuda":
        cpu_lines.insert(1, f"speech: {speech_row.label} ({speech_row.threads or 4} threads)")
    elif speech_row is not None:
        devices.setdefault(gpu_name, []).append(f"speech: {speech_row.label} ({int(speech_row.vram_mb or 0)} MiB)")
    devices[machine.cpu or "CPU"] = cpu_lines
    if machine.igpu:
        devices[machine.igpu] = [f"accurate OCR ({ocr.device})" if ocr.device == "igpu" else "idle"]
    if machine.npu:
        devices[machine.npu] = [f"accurate OCR ({ocr.device})" if ocr.device == "npu" else "idle: OCR is 2x slower here than on the iGPU"]

    settings_union: dict[str, object] = {}
    for it in items:
        if it.tick and it.enabled:
            settings_union.update(it.settings)
    digest = plan_digest(goal, model_key, profile_changes, settings_union)
    if not summary:
        summary.append("nothing to change: this machine already runs the way this goal wants")
    return Plan(goal=goal, model_key=model_key, digest=digest, created_at=inp.now_epoch, items=tuple(items),
                settings=settings_union, profile=profile_changes, vram=vram_out, predicted=predicted, devices=devices,
                blocked=tuple(blocked), warnings=tuple(dict.fromkeys(warnings)), summary=tuple(summary))


def plan_all(inp: PlanInputs) -> dict[str, Plan]:
    return {g: recommend(inp, g) for g in GOALS}


def validate_plan(plan: Plan, fresh: FreshFacts, *, max_age_s: float = 900.0, vram_tolerance_mb: int = 1536) -> tuple[str, ...]:
    """Reasons NOT to apply ``plan`` now: the GPU budget moved, the shared server woke up, another
    model got loaded, or the plan is older than 15 minutes."""
    reasons: list[str] = []
    if plan.blocked:
        reasons.extend(plan.blocked)
    if fresh.swap_resident is True:
        reasons.append("the shared model server holds a model on the GPU")
    if plan.model_key and fresh.loaded_key and fresh.loaded_key != plan.model_key:
        reasons.append(f"a different model is loaded ({fresh.loaded_key})")
    if plan.created_at and fresh.now_epoch and fresh.now_epoch - plan.created_at > max_age_s:
        reasons.append("the plan is older than 15 minutes; analyze again")
    if plan.vram and fresh.free_mb is not None:
        then = plan.vram.get("free_at_analysis_mb")
        # something else took more than the tolerance since the analysis (llama-swap waking up, a
        # game, a second model): the budget the plan was computed against is gone
        if isinstance(then, (int, float)) and fresh.free_mb < then - vram_tolerance_mb:
            reasons.append("GPU memory moved since the analysis")
    return tuple(dict.fromkeys(reasons))


def gate(scope: str, a: Activity, *, force: bool = False) -> tuple[bool, str]:
    """May a benchmark (``"llm"`` / ``"speech"``) or an apply run now? ``(ok, reason)``."""
    if scope in ("llm", "apply"):
        if a.backend != "llamacpp":
            return False, f"the {a.backend} backend is not served by ExoCortex"
        if not a.llm_ready or not a.server_running:
            return False, "the language model is not ready yet"
        if a.swap_busy:
            return False, "a model change is still running"
        if a.assist_busy:
            return False, "an answer is being generated"
        if a.swap_resident is True:
            return False, "the shared model server holds a model on the GPU"
        if a.speech_idle_s < 3.0 and not (force and scope == "llm"):
            return False, "someone is speaking; wait for a quiet moment"
        return True, ""
    if scope == "speech":
        if not a.transcriber_ready:
            return False, "speech recognition is not ready yet"
        if a.speech_idle_s < 3.0 and not force:
            return False, "someone is speaking; wait for a quiet moment"
        return True, ""
    return False, f"unknown scope {scope!r}"


def describe(plan: Plan) -> str:
    """The plan as text (the CLI, the log, golden tests)."""
    lines = [f"Optimize for {plan.goal}" + (f" -- {plan.model_key}" if plan.model_key else "") + f"  [{plan.digest}]"]
    for s in plan.summary:
        lines.append(f"  * {s}")
    if plan.vram:
        v = plan.vram
        lines.append(f"  GPU memory ({v.get('tier')}): {v.get('avail_mb')} MiB usable, need {v.get('need_mb')} MiB, "
                     f"up to {v.get('max_ctx')} tokens fit, {v.get('kv_bytes_per_token', 0) // 1024} KiB/token")
    if plan.predicted:
        p = plan.predicted
        lines.append(f"  Window {p.get('ctx')} tokens: {p.get('answer_tokens')} for answers, ~{p.get('hours_held')} h of meeting, "
                     f"cold re-read {p.get('cold_reread_s')} s ({p.get('pp_basis')} speed)")
    for it in plan.items:
        mark = "[x]" if it.tick and it.enabled else "[ ]"
        glyph = {"measured": "*", "recorded": "o", "estimated": "~"}.get(it.evidence_kind, "~")
        lines.append(f"  {mark} {glyph} {it.label}: {it.now} -> {it.proposed} ({it.applies})")
        lines.append(f"        {it.why}")
    for w in plan.warnings:
        lines.append(f"  ! {w}")
    for b in plan.blocked:
        lines.append(f"  BLOCKED: {b}")
    return "\n".join(lines)


def parse_server_log(text: str) -> dict[str, Any] | None:
    """Optional: KV / state sizes from a llama-server log that prints them (``llama_kv_cache: size
    = N MiB``, ``llama_memory_recurrent: size = N MiB``). The LM Studio pack's ``--log-file`` does
    not, so this normally returns ``None``. ANSI colour codes are stripped first."""
    if not text:
        return None
    clean = _ANSI_RE.sub("", text)
    out: dict[str, Any] = {}
    m = re.search(r"llama_kv_cache:\s*size\s*=\s*([\d.]+)\s*MiB\s*\((\d+)\s*cells,\s*(\d+)\s*layers", clean)
    if m:
        out["kv_mb"], out["cells"], out["kv_layers"] = float(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"llama_memory_recurrent:\s*size\s*=\s*([\d.]+)\s*MiB", clean)
    if m:
        out["recurrent_mb"] = float(m.group(1))
    m = re.search(r"\bn_ctx\s*=\s*(\d+)", clean)
    if m:
        out["n_ctx"] = int(m.group(1))
    return out or None


# ============================================================================================
# facts from the app's own objects (thin adapters; still no I/O)
# ============================================================================================
def model_facts_from_meta(key: str, size_bytes: int, meta: Any) -> ModelFacts:
    """A :class:`ModelFacts` from ``exocortex.llm.library.GgufMeta`` (duck-typed; missing
    attributes read as unknown)."""
    g = lambda name, default=None: getattr(meta, name, default) if meta is not None else default  # noqa: E731
    return ModelFacts(
        key=key, size_bytes=int(size_bytes or 0), arch=str(g("architecture", "") or ""),
        n_layer=int(g("n_layer", 0) or 0), n_ctx_train=int(g("n_ctx_train", 0) or 0),
        n_head_kv=g("n_head_kv"), head_dim=g("head_dim"), kv_heads_total=g("kv_heads_total"),
        full_attention_interval=g("full_attention_interval"), nextn_layers=int(g("nextn_predict_layers", 0) or 0),
        ssm_conv=g("ssm_conv_kernel"), ssm_state=g("ssm_state_size"), ssm_inner=g("ssm_inner_size"), ssm_groups=g("ssm_group_count"),
    )


def cfg_facts_from_config(cfg: Any, *, asr_now: str = "", ocr_now: str = "") -> CfgFacts:
    """A :class:`CfgFacts` from a ``Config`` (duck-typed)."""
    llm, ctx, scr, hf, asr = cfg.llm, cfg.context, cfg.screen, cfg.handsfree, cfg.asr
    return CfgFacts(
        lane_share=float(getattr(ctx, "lane_share", 0.3)), reserve_tokens=int(getattr(ctx, "reserve_tokens", 256)),
        chars_per_token=float(getattr(ctx, "chars_per_token", 3.6)), budget_mode=str(getattr(ctx, "budget_mode", "auto")),
        max_answer_tokens=int(max(getattr(llm, "assist_max_tokens", 220), getattr(llm, "ask_max_tokens", 400),
                                  getattr(llm, "solve_max_tokens", 700), getattr(llm, "text_max_tokens", 400))),
        screen_max_text_chars=int(getattr(scr, "max_text_chars", 2500)), handsfree_context_chars=int(getattr(hf, "context_chars", 6000)),
        accurate_ocr=bool(getattr(hf, "accurate_ocr", True)), partial_interval_s=float(getattr(asr, "partial_interval_s", 1.2)),
        partials_enabled=bool(getattr(asr, "partials_enabled", True)), asr_engine=str(getattr(asr, "engine", "auto")),
        asr_device=str(getattr(asr, "device", "auto")), asr_compute=str(getattr(asr, "compute_type", "auto")),
        asr_threads=int(getattr(asr, "cpu_threads", 0) or 0), asr_language=str(getattr(asr, "language", "en") or "en"),
        hotwords=bool(getattr(asr, "hotwords", None)), asr_now=asr_now, ocr_device=str(getattr(scr, "ocr_device", "auto")), ocr_now=ocr_now,
    )


def settings_from_config(cfg: Any) -> Settings:
    o = getattr(cfg, "optimize", None)
    if o is None:
        return Settings()
    return Settings(goal=str(getattr(o, "goal", "balanced")), reserve_vram_mb=int(getattr(o, "reserve_vram_mb", 0) or 0),
                    ctx_cap=int(getattr(o, "ctx_cap", 0) or 0), asr_gpu=str(getattr(o, "asr_gpu", "auto")))


__all__ = [name for name in dir() if not name.startswith("_")]
