"""Evidence for the Optimize tab (2026-10-07): the benchmark store, the importer for the sweep
scripts' JSON, the live probe of the RUNNING llama-server, the VRAM / llama-swap helpers and the
killable child processes that probe speech and devices.

Why this module exists. ``exocortex.optimize`` is pure arithmetic: it only knows what it is handed.
This file is where the numbers come from, under four rules the 2026-10-07 re-benchmark taught:

* **Measure the server that is already running; never start another.** A second llama-server on the
  5090 crashed both (the card is shared with a llama-swap service that can pin a 17 GB model at any
  moment). :func:`probe_llm` talks HTTP to the app's own server, pins its requests to one idle slot
  (``id_slot``) so the live conversation's KV cache is not evicted, and erases that slot at the end.
  It never imports ``exocortex.llm.server``.
* **Anything that imports a native stack runs in a child process.** numpy on a worker thread deadlocked
  startup once (loader lock), RapidOCR created after WinRT OCR crashes the interpreter, and
  ctranslate2 / OpenVINO / sherpa each want their own DLL search order. ``python -m
  exocortex.optimize_bench probe-devices|probe-asr`` therefore does the heavy imports lazily inside
  ``main()``, and :func:`run_child` launches it with ``CREATE_NO_WINDOW`` at below-normal priority,
  a hard deadline and a ``--out`` file, and kills it the moment the caller cancels, speech resumes
  or the shared GPU fills up. A probe never downloads anything (``HF_HUB_OFFLINE=1``; the sherpa
  stand-in raises instead of fetching a bundle).
* **Every number carries its provenance and ages.** :class:`Store` keeps ``benchmarks.json`` next to
  the GGUF library (atomic writes, soft-failing reads, caps, dedup). Rows from another machine
  fingerprint (GPU, driver, runtime, RAM) or older than 30 days are loaded with ``stale=True`` so the
  planner may rank with them but never adopt a launch flag from them. The apply journal lives in its
  own file so a corrupt benchmark store can never drop a pending rollback.
* **The sweep scripts are never imported.** ``scripts/`` is not part of the package; the transcript
  builders and the SSE ``timings`` reader are COPIED here from ``scripts/bench_llm.py``, and the
  ASR truth table from ``scripts/bench_asr.py``. The importers read the JSON those scripts write.

Every probe fails soft: a refused connection, a 404 on ``/slots``, a child that hangs, a garbage
result file -- each ends as ``{"error": reason}`` (plus whatever was measured so far), never as an
exception into the engine.
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import logging
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from exocortex import optimize
from exocortex.config import project_root
from exocortex.llm.prompts import NO_THINK_PREFIX, STOP, approx_tokens, render_chatml, system_assist, user_assist
from exocortex.optimize import AsrRow, Evidence, Footprint, LlmRun, OcrRow

log = logging.getLogger(__name__)

BENCH_FILE = "benchmarks.json"
JOURNAL_FILE = "optimize_journal.json"
SCHEMA = 1
STALE_AFTER_S = 30 * 86400
DEVICES_FRESH_S = 7 * 86400
#: per-kind row caps (newest kept)
CAPS: dict[str, int] = {"llm": 60, "asr": 40, "ocr": 20, "footprints": 80}
JOURNAL_KEEP = 10
#: one lock for every file this module writes (the engine's threads and the CLI importer share it)
_LOCK = threading.RLock()


# ============================================================================================
# time helpers
# ============================================================================================
def iso_now(epoch: float | None = None) -> str:
    """Local time as an ISO 8601 string with its UTC offset (the ``at`` fields), to the second."""
    ts = time.time() if epoch is None else float(epoch)
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def parse_at(text: Any) -> float | None:
    """Epoch seconds for an ``at`` string (naive values read as local time); ``None`` for anything
    that is not a parsable ISO 8601 timestamp."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return datetime.fromisoformat(text.strip()).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


# ============================================================================================
# machine fingerprint
# ============================================================================================
_FP_KEYS = ("cpu", "gpu", "driver", "runtime", "ram_mb")
_FP_GPU_KEYS = ("gpu", "driver", "runtime")   # what a measured VRAM footprint depends on
_FP_RAM_TOLERANCE = 0.02


def _runtime_label(runtime: Any) -> str:
    if isinstance(runtime, dict):
        return " ".join(str(x) for x in (runtime.get("flavor"), runtime.get("version")) if x).strip()
    return str(runtime or "")


def fingerprint(hw_report: dict, runtime: str = "") -> dict:
    """``{"cpu", "gpu", "driver", "runtime", "ram_mb"}`` from a ``exocortex.hardware.to_dict()``
    report (``cpu_name``, the primary entry of ``gpus``, ``nvidia_driver``, ``runtime`` flavor +
    version, ``ram_total_mb``). ``runtime`` overrides the report's. A missing key reads as ``""`` /
    ``0``: an unknown part never counts as a difference (see :func:`_fp_differs`)."""
    hw: Any = hw_report
    if dataclasses.is_dataclass(hw) and not isinstance(hw, type):
        hw = dataclasses.asdict(hw)
    if not isinstance(hw, dict):
        hw = {}
    gpus = [g for g in (hw.get("gpus") or []) if isinstance(g, dict)]
    primary = None
    idx = hw.get("primary_index")
    if idx is not None:
        primary = next((g for g in gpus if g.get("index") == idx), None)
    if primary is None and gpus:
        primary = gpus[0]
    try:
        ram = int(hw.get("ram_total_mb") or 0)
    except (TypeError, ValueError):
        ram = 0
    return {
        "cpu": str(hw.get("cpu_name") or ""),
        "gpu": str((primary or {}).get("name") or ""),
        "driver": str(hw.get("nvidia_driver") or ""),
        "runtime": runtime or _runtime_label(hw.get("runtime")),
        "ram_mb": ram,
    }


def _fp_differs(stored: Any, given: Any, keys: tuple[str, ...] = _FP_KEYS) -> bool:
    """True only on POSITIVE evidence that two fingerprints are different machines: some key present
    and non-empty on both sides with different values (RAM within 2 %). A missing / empty part is
    "unknown", never a difference, so a half-filled fingerprint cannot mark everything stale."""
    if not isinstance(stored, dict) or not isinstance(given, dict) or not stored or not given:
        return False
    for key in keys:
        a, b = stored.get(key), given.get(key)
        if a in (None, "", 0) or b in (None, "", 0):
            continue
        if key == "ram_mb":
            try:
                fa, fb = float(a), float(b)
            except (TypeError, ValueError):
                continue
            if abs(fa - fb) > _FP_RAM_TOLERANCE * max(fa, fb):
                return True
            continue
        if str(a).strip().lower() != str(b).strip().lower():
            return True
    return False


# ============================================================================================
# the store
# ============================================================================================
def _empty() -> dict[str, Any]:
    return {"schema": SCHEMA, "fingerprint": {}, "devices": {}, "llm": [], "asr": [], "ocr": [], "footprints": []}


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as JSON to ``path`` through a temp file + ``os.replace`` (like
    ``library.save_profiles``). Raises ``OSError`` on failure after removing the temp file; a
    ``PermissionError`` on the replace (a virus scanner holding the target on Windows) is retried."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        for attempt in range(3):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.05)
    except Exception:  # noqa: BLE001 - remove the temp file, then re-raise: the caller must know
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> Any | None:
    """Parsed JSON at ``path``; ``None`` when missing, unreadable or corrupt (logged, never raised)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        log.warning("optimize: could not read %s (%s); treating it as empty", path, exc)
        return None
    try:
        return json.loads(text)
    except ValueError as exc:
        log.warning("optimize: %s is corrupt (%s); treating it as empty", path, exc)
        return None


def _flags_key(flags: Any) -> tuple[tuple[str, str], ...]:
    try:
        return tuple((str(a), str(b)) for a, b in (flags or []))
    except (TypeError, ValueError):
        return (("?", "?"),)


def _llm_key(d: dict) -> tuple:
    return (d.get("model"), d.get("ctx"), d.get("kv"), d.get("ubatch"), d.get("parallel"), _flags_key(d.get("flags")))


def _too_old(at: Any, now_epoch: float) -> bool:
    ep = parse_at(at)
    return bool(ep is not None and now_epoch and now_epoch - ep > STALE_AFTER_S)


class Store:
    """``<directory>/benchmarks.json`` (measurements) and ``<directory>/optimize_journal.json``
    (the apply journal). In production ``directory`` is ``library.library_dir(cfg.llm)``.

    Reads soft-fail to an empty store (a corrupt or foreign file logs a warning, never raises);
    writes are atomic and raise ``OSError`` if the disk refuses them -- the caller (the controller)
    decides whether that matters; for the journal it does. All writes serialise on a module lock."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    @property
    def path(self) -> Path:
        return self.directory / BENCH_FILE

    @property
    def journal_path(self) -> Path:
        return self.directory / JOURNAL_FILE

    # ---- raw load / save --------------------------------------------------------------------
    def load(self) -> dict:
        """The store as a dict with all six keys present. Missing, corrupt, non-object or
        foreign-schema files read as an empty store."""
        with _LOCK:
            raw = _read_json(self.path)
        out = _empty()
        if raw is None:
            return out
        if not isinstance(raw, dict):
            log.warning("optimize: %s is not a JSON object; treating it as empty", self.path)
            return out
        if raw.get("schema", SCHEMA) != SCHEMA:
            log.warning("optimize: %s has schema %r (this build reads %d); treating it as empty",
                        self.path, raw.get("schema"), SCHEMA)
            return out
        for key in ("fingerprint", "devices"):
            if isinstance(raw.get(key), dict):
                out[key] = raw[key]
        for key in CAPS:
            if isinstance(raw.get(key), list):
                out[key] = [r for r in raw[key] if isinstance(r, dict)]
        return out

    def save(self, data: dict) -> None:
        """Atomically replace ``benchmarks.json`` with ``data`` (raises ``OSError`` on failure)."""
        with _LOCK:
            _atomic_write_json(self.path, data)

    # ---- adding rows ------------------------------------------------------------------------
    def _add(self, kind: str, row: dict, key_fn: Callable[[dict], Any] | None) -> None:
        with _LOCK:
            data = self.load()
            rows = data[kind]
            if key_fn is not None:
                k = key_fn(row)
                rows = [r for r in rows if key_fn(r) != k]
            rows.append(row)
            data[kind] = rows[-CAPS[kind]:]
            self.save(data)

    def add_llm(self, run: LlmRun) -> None:
        """One measured llama-server configuration. Deduplicated per ``(model, ctx, kv, ubatch,
        parallel, flags)``, newest wins; the newest 60 are kept."""
        self._add("llm", optimize.to_dict(run), _llm_key)

    def add_asr(self, row: AsrRow) -> None:
        """One speech variant. Deduplicated per ``label``, newest wins; the newest 40 are kept."""
        self._add("asr", optimize.to_dict(row), lambda d: d.get("label"))

    def add_ocr(self, row: OcrRow) -> None:
        """One OCR device. Deduplicated per ``device``, newest wins; the newest 20 are kept."""
        self._add("ocr", optimize.to_dict(row), lambda d: d.get("device"))

    def add_footprint(self, fp: Footprint) -> None:
        """One measured app footprint (VRAM held with the server up). The newest 80 are kept."""
        self._add("footprints", optimize.to_dict(fp), None)

    # ---- fingerprint / devices --------------------------------------------------------------
    def set_fingerprint(self, fp: dict) -> None:
        """Record the machine the NEW rows are measured on. If it differs from the stored one, every
        existing row is flagged ``stale`` on disk first, so the old measurements stay stale even
        after the stored fingerprint has caught up with the machine."""
        new = dict(fp or {})
        with _LOCK:
            data = self.load()
            if data["fingerprint"] == new:
                return
            if _fp_differs(data["fingerprint"], new):
                for kind in CAPS:
                    for row in data[kind]:
                        row["stale"] = True
            data["fingerprint"] = new
            self.save(data)

    def set_devices(self, devices: dict, at_epoch: float) -> None:
        """Cache the ``probe-devices`` child's answer (valid for :data:`DEVICES_FRESH_S`)."""
        with _LOCK:
            data = self.load()
            data["devices"] = {**dict(devices), "at": iso_now(at_epoch)}
            self.save(data)

    def devices(self, fingerprint: dict | None, now_epoch: float) -> dict | None:
        """The cached devices answer if it is under 7 days old and the machine fingerprint has not
        changed; ``None`` otherwise (probe again). The ``at`` stamp is not part of the result."""
        data = self.load()
        cached = data["devices"]
        if not cached:
            return None
        ep = parse_at(cached.get("at"))
        if ep is None or now_epoch - ep > DEVICES_FRESH_S:
            return None
        if _fp_differs(data["fingerprint"], fingerprint):
            return None
        return {k: v for k, v in cached.items() if k != "at"}

    # ---- reading evidence -------------------------------------------------------------------
    def evidence(self, fingerprint: dict | None, now_epoch: float) -> Evidence:
        """Everything stored as an :class:`~exocortex.optimize.Evidence`. A row is ``stale`` when it
        was flagged so on disk, when the stored fingerprint differs from ``fingerprint`` (the
        machine changed since the last record), or when its ``at`` is older than 30 days.

        Footprints have no ``stale`` field, so a footprint measured on a different GPU / driver /
        runtime is dropped from the result instead (it is kept on disk). Age alone never drops one:
        what a model costs in VRAM does not rot."""
        data = self.load()
        machine_changed = _fp_differs(data["fingerprint"], fingerprint)
        gpu_changed = _fp_differs(data["fingerprint"], fingerprint, _FP_GPU_KEYS)

        def _rows(kind: str, cls: type) -> tuple:
            out = []
            for d in data[kind]:
                row = optimize.from_dict(cls, d)
                if d.get("stale") or machine_changed or _too_old(d.get("at"), now_epoch):
                    row = dataclasses.replace(row, stale=True)
                out.append(row)
            return tuple(out)

        footprints: tuple[Footprint, ...] = ()
        if not gpu_changed:
            footprints = tuple(optimize.from_dict(Footprint, d) for d in data["footprints"] if not d.get("stale"))
        return Evidence(llm=_rows("llm", LlmRun), asr=_rows("asr", AsrRow), ocr=_rows("ocr", OcrRow),
                        footprints=footprints)

    # ---- the apply journal (a separate file) ------------------------------------------------
    def journal(self) -> list[dict]:
        """The last 10 journal entries, oldest first (``[]`` when missing or corrupt)."""
        with _LOCK:
            raw = _read_json(self.journal_path)
        entries = raw.get("entries") if isinstance(raw, dict) else None
        if not isinstance(entries, list):
            return []
        return [e for e in entries if isinstance(e, dict)][-JOURNAL_KEEP:]

    def journal_write(self, entries: list[dict]) -> None:
        """Atomically replace the journal with the last 10 of ``entries`` (raises ``OSError``)."""
        keep = [e for e in entries if isinstance(e, dict)][-JOURNAL_KEEP:]
        with _LOCK:
            _atomic_write_json(self.journal_path, {"schema": SCHEMA, "entries": keep})


# ============================================================================================
# importers: the sweep scripts' JSON -> evidence
# ============================================================================================
@dataclass
class ImportReport:
    imported: int = 0                       # records (runs / speech rows / OCR rows) added to the store
    skipped: int = 0                        # records or files that could not be used
    warnings: list[str] = field(default_factory=list)


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _int(v: Any, default: int = 0) -> int:
    f = _num(v)
    return int(f) if f is not None else default


#: launch flags that are fields of the run rather than part of its flag identity
_DEDICATED_FLAGS = frozenset({"--ctx-size", "--parallel", "--cache-type-k", "--cache-type-v"})
#: managed flag values equal to what the app launches with anyway (so they do not "differ")
_BUILTIN_FLAG_VALUES = {"--ubatch-size": "512", "--load-mode": "mmap"}
#: flags that move weights or KV between GPU and host: a footprint measured with them says nothing
#: about the normal launch, so it is not recorded
_MEMORY_MOVING_FLAGS = frozenset({"--no-kv-offload", "-nkvo", "--cpu-moe", "--n-cpu-moe", "--override-tensor",
                                  "--n-gpu-layers", "--device", "--fit"})
_BAD_LOG_RE = re.compile(r"error|failed|\boom\b|out[ _]of[ _]memory|not supported", re.IGNORECASE)
#: warning lines llama-server prints on EVERY healthy launch of this app's configuration; they must
#: not mark a run unclean (that would veto every footprint and every knob): "cache_reuse is not
#: supported by this context" (two slots share a unified KV cache, so KV shifting is off -- the app
#: logs the same thing itself) and "common_fit_params: failed to fit params ... n_gpu_layers already
#: set by user" (--fit backs off because the app pins -ngl).
_BENIGN_LOG_RE = re.compile(r"cache_reuse is not supported|common_fit_params: failed to fit params", re.IGNORECASE)
_ANSI_LOG_RE = re.compile(r"\x1b\[[0-9;]*m")


def _rel_spread(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    med = statistics.median(vals)
    return (max(vals) - min(vals)) / med if med else 0.0


def _llm_spread(obj: dict, medians: dict) -> float:
    """Relative run-to-run spread of pp and tg: from the per-round timings when the file has them,
    else from the script's absolute ``spread_pp`` / ``spread_tg`` over the medians."""
    pp: list[float] = []
    tg: list[float] = []
    rounds = obj.get("rounds")
    if isinstance(rounds, list):
        for r in rounds:
            try:
                t = r["a_fresh"]["timings"]
            except (KeyError, TypeError):
                continue
            if not isinstance(t, dict):
                continue
            a, b = _num(t.get("prompt_per_second")), _num(t.get("predicted_per_second"))
            if a is not None:
                pp.append(a)
            if b is not None:
                tg.append(b)
    if len(pp) >= 2 or len(tg) >= 2:
        return round(max(_rel_spread(pp), _rel_spread(tg)), 4)
    out = 0.0
    for spread_key, med_key in (("spread_pp", "pp"), ("spread_tg", "tg")):
        s, m = _num(medians.get(spread_key)), _num(medians.get(med_key))
        if s is not None and m:
            out = max(out, s / m)
    return round(out, 4)


def import_bench_llm_json(obj: dict, *, at: str = "", profile: dict | None = None) -> tuple[LlmRun | None, Footprint | None]:
    """One ``scripts/bench_llm.py --protocol full --json`` result as ``(LlmRun, Footprint)``.

    The variant is read from ``requested.llama_extra_args`` parsed LAST-WINS (``--ctx-size``,
    ``--parallel``, ``--ubatch-size``, ``--cache-type-k``) over the plain ``requested.*`` values, so
    a repeated flag overrides the earlier one exactly like llama.cpp's own parser. ``flags`` is the
    run's flag identity: the managed flags (:func:`optimize.managed_flags_of`) minus values equal to
    the built-ins (``--ubatch-size 512``, ``--load-mode mmap``), ``requested.load_mode`` when it is
    not the default, PLUS any other non-dedicated flag (``--no-kv-offload``, ``--ctx-checkpoints``,
    ...). The last part is deliberate: without it such a run would carry the baseline's identity,
    replace it in the store and be taken for it by the planner; with it the run is recorded but
    inert. ``ok`` is ``not aborted`` and medians present; ``clean_log`` is false when a captured log
    warning looks like an error / OOM / "not supported". The footprint (VRAM the app held with the
    server up) is returned only for a clean, completed run without memory-moving flags.
    ``(None, None)`` for something that is not such a result."""
    if not isinstance(obj, dict):
        return None, None
    requested = obj.get("requested") if isinstance(obj.get("requested"), dict) else {}
    medians = obj.get("medians") if isinstance(obj.get("medians"), dict) else {}
    if not requested and not medians:
        return None, None

    extra = [str(x) for x in (requested.get("llama_extra_args") or [])]
    parsed = optimize.parse_launch_args(extra)
    managed = dict(optimize.managed_flags_of(extra))
    flags = {k: v for k, v in managed.items() if _BUILTIN_FLAG_VALUES.get(k) != v}
    load_mode = str(requested.get("load_mode") or "")
    if load_mode and load_mode != "mmap" and "--load-mode" not in parsed:
        flags["--load-mode"] = load_mode
    for k, v in parsed.items():
        if k not in managed and k not in _DEDICATED_FLAGS:
            flags[k] = v
    flag_t = tuple(sorted(flags.items()))

    # `requested.*` are the GLOBAL [llm] values the script ran with; the server actually launched
    # with the model's tuning PROFILE overlaid on them (LlamaServerManager.build_args), so a caller
    # that knows that profile passes it here and it takes precedence -- a repeated flag in the extra
    # args still wins over both (the owner's 81,920-token profile vs the 8,192 global on 2026-10-07).
    prof = profile if isinstance(profile, dict) else {}
    ctx = _int(parsed.get("--ctx-size"), _int(prof.get("ctx_size"), _int(requested.get("ctx_size"))))
    kv = str(parsed.get("--cache-type-k") or prof.get("kv_cache_type") or requested.get("kv_cache_type") or "q8_0").strip().lower()
    parallel = _int(parsed.get("--parallel"), _int(prof.get("parallel"), _int(requested.get("parallel"), 2))) or 1
    ubatch = _int(parsed.get("--ubatch-size"), 512)
    model_info = obj.get("model") if isinstance(obj.get("model"), dict) else {}
    model = str(model_info.get("key") or requested.get("model") or "")

    aborted = bool(obj.get("aborted"))
    pp, tg = _num(medians.get("pp")), _num(medians.get("tg"))
    ok = not aborted and pp is not None and tg is not None
    warnings = []
    signals = obj.get("log_signals")
    if isinstance(signals, dict) and isinstance(signals.get("warnings"), list):
        warnings = [str(w) for w in signals["warnings"]]
    clean = not any(_BAD_LOG_RE.search(_ANSI_LOG_RE.sub("", w)) and not _BENIGN_LOG_RE.search(w) for w in warnings)
    used = _num(obj.get("gpu_used_mb"))
    if used is not None and used <= 0:
        used = None                          # a non-positive "VRAM used" is another process's noise, not a measurement

    run = LlmRun(
        model=model, ctx=ctx, kv=kv, parallel=parallel, ubatch=ubatch, flags=flag_t, pp=pp, tg=tg,
        cached_ttft_ms=_num(medians.get("cached_ttft_ms")), incr_prompt_n=_num(medians.get("incremental_prompt_n")),
        spread=_llm_spread(obj, medians), gpu_used_mb=used, load_s=_num(obj.get("load_s")),
        ok=ok, clean_log=clean, at=at, source="sweep",
    )
    footprint = None
    if ok and clean and used is not None and used > 0 and ctx > 0 and not (set(flags) & _MEMORY_MOVING_FLAGS):
        footprint = Footprint(model=model, ctx=ctx, kv=kv, ubatch=ubatch, parallel=parallel, flags=flag_t,
                              app_mb=used, source="sweep", at=at)
    return run, footprint


# ---- speech ---------------------------------------------------------------------------------
_COMPUTE_TOKENS = frozenset({"float16", "int8_float16", "int8", "float32", "bfloat16"})
_THREADS_RE = re.compile(r"^(\d+)\s*(?:thr|threads?|t)$")
_CANONICAL_THREADS = 4          # the thread count the unsuffixed canonical cpu labels stand for
_VRAM_IMPLAUSIBLE_MB = 8000     # an ASR engine cannot hold more than this: a concurrent job's noise
_VRAM_CPU_IMPLAUSIBLE_MB = 512  # ... and a CPU engine holds none at all


def _asr_identity(label: str, engine: str = "", device: str = "", compute: str = "",
                  threads: int = 0) -> tuple[str, str, str, str, int]:
    """``(canonical label, engine, device, compute, threads)`` for a speech variant. The label's own
    words win ("whisper-turbo cuda int8_float16", "parakeet cpu 4thr", "sherpa-cpu-8thr", the
    canonical "whisper-cpu-int8"); the row's fields fill what the label does not say. An unreadable
    variant keeps its label unchanged and the fields as given."""
    tokens = [t for t in re.split(r"[\s\-/]+", label.lower()) if t]
    kind = ""
    if any(t in ("sherpa", "parakeet") for t in tokens):
        kind = "sherpa"
    elif any("whisper" in t for t in tokens):
        kind = "faster-whisper"
    else:
        low = engine.lower()
        kind = "sherpa" if ("sherpa" in low or "parakeet" in low) else ("faster-whisper" if "whisper" in low else "")
    dev = next((("cuda" if t == "gpu" else t) for t in tokens if t in ("cuda", "cpu", "gpu")), "") or device.lower()
    dev = "cuda" if dev == "gpu" else dev
    comp = next((t for t in tokens if t in _COMPUTE_TOKENS), "") or compute
    thr = 0
    for t in tokens:
        m = _THREADS_RE.match(t)
        if m:
            thr = int(m.group(1))
            break
    thr = thr or threads
    if kind == "sherpa":
        suffix = f"-{thr}thr" if thr and thr != _CANONICAL_THREADS else ""
        return f"sherpa-cpu{suffix}", "sherpa", "cpu", "int8", thr
    if kind == "faster-whisper" and dev == "cuda" and comp:
        return f"whisper-cuda-{comp}", kind, "cuda", comp, 0
    if kind == "faster-whisper" and dev == "cpu":
        comp = comp or "int8"
        suffix = f"-{thr}thr" if thr and thr != _CANONICAL_THREADS else ""
        return f"whisper-cpu-{comp}{suffix}", kind, "cpu", comp, thr
    return label.strip(), kind or engine, dev, comp, thr


def _asr_prior_vram(label: str) -> int | None:
    """The shipped VRAM prior for a canonical label (a thread-suffixed one uses its base's)."""
    base = re.sub(r"-\d+thr$", "", label)
    for key in (label, base):
        if key in optimize.PRIORS_ASR:
            return int(optimize.PRIORS_ASR[key][5])
    return None


_WRAPPER_KEYS = ("results", "rows", "runs", "variants", "asr", "ocr", "llm", "items")


def _records(obj: Any) -> list[dict]:
    """The result dicts inside a parsed JSON value: a list of dicts, a single dict, or a dict that
    wraps such a list under a conventional key."""
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in _WRAPPER_KEYS:
            inner = obj.get(key)
            if isinstance(inner, list) and any(isinstance(x, dict) for x in inner) \
                    and not ("medians" in obj or "requested" in obj):
                return [x for x in inner if isinstance(x, dict)]
        return [obj]
    return []


def import_asr_json(obj: list | dict, *, at: str = "") -> list[AsrRow]:
    """Speech rows from the lead's ``bench_asr_devices.py`` output (a list of dicts with ``label``,
    ``engine``, ``device``, ``compute_type``, ``cpu_threads``, ``median_s``, ``overlap_mean``,
    ``vram_delta_mb``, ``load_s``, ``error``). Rows with an ``error`` or no positive ``median_s`` are
    left out. Labels are normalised to the planner's canonical ones ("whisper-turbo cuda int8_float16"
    -> "whisper-cuda-int8_float16", "whisper-turbo cpu int8 4thr" -> "whisper-cpu-int8", "parakeet cpu
    4thr" -> "sherpa-cpu"; other thread counts keep a suffix such as "sherpa-cpu-8thr"). A negative
    or over-8000 MiB VRAM delta is a concurrent job's noise (the Parakeet "-3011 MiB"), so the
    shipped prior for that label is used instead with ``vram_source="prior"``; a CPU engine's delta
    over 512 MiB is rejected the same way. ``source="sweep"``."""
    out: list[AsrRow] = []
    for rec in _records(obj):
        if rec.get("error"):
            continue
        median = _num(rec.get("median_s", rec.get("median")))
        label_in = str(rec.get("label") or rec.get("id") or "").strip()
        if median is None or median <= 0 or not label_in:
            continue
        label, engine, device, compute, threads = _asr_identity(
            label_in, str(rec.get("engine") or ""), str(rec.get("device") or ""),
            str(rec.get("compute_type") or rec.get("compute") or ""), _int(rec.get("cpu_threads", rec.get("threads"))))
        overlap = None
        for key in ("overlap_mean", "overlap", "word_overlap"):
            overlap = _num(rec.get(key))
            if overlap is not None:
                break
        delta = None
        for key in ("vram_delta_mb", "vram_mb", "vram_delta"):
            delta = _num(rec.get(key))
            if delta is not None:
                break
        limit = _VRAM_CPU_IMPLAUSIBLE_MB if device == "cpu" else _VRAM_IMPLAUSIBLE_MB
        if delta is None or delta < 0 or delta > limit:
            prior = _asr_prior_vram(label)
            vram, source = (prior, "prior") if prior is not None else (None, "none")
        else:
            vram, source = int(round(delta)), "nvml"
        out.append(AsrRow(label=label, engine=engine, device=device, compute=compute, threads=threads,
                          median_s=round(median, 4), overlap=overlap, vram_mb=vram, vram_source=source,
                          at=at, source="sweep"))
    return out


def import_ocr_json(obj: list | dict, *, at: str = "") -> list[OcrRow]:
    """OCR rows from ``scripts/bench_ocr.py``'s per-engine result dicts (``label`` such as
    ``rapidocr/igpu``, ``wall_ms[]``, ``cpu_ms[]``, ``ok``). Windows OCR is not a RapidOCR device and
    is left out. Not part of the lead's contract; it lets one ``import`` command take the whole
    sweep directory."""
    out: list[OcrRow] = []
    for rec in _records(obj):
        label = str(rec.get("label") or "")
        walls = [w for w in (_num(x) for x in (rec.get("wall_ms") or [])) if w is not None]
        if rec.get("ok") is False or not label.startswith("rapidocr/") or not walls:
            continue
        cpus = [c for c in (_num(x) for x in (rec.get("cpu_ms") or [])) if c is not None]
        out.append(OcrRow(label=label, device=label.split("/", 1)[1], median_ms=round(statistics.median(walls), 1),
                          cpu_ms=round(statistics.median(cpus), 1) if cpus else None, at=at, source="sweep"))
    return out


def _classify(rec: dict) -> str:
    if "medians" in rec or "requested" in rec:
        return "llm"
    if "median_s" in rec:
        return "asr"
    if "wall_ms" in rec:
        return "ocr"
    return ""


def import_files(paths: list[Path], store: Store, *, profile_lookup: Callable[[str], dict | None] | None = None) -> ImportReport:
    """Read each JSON file (a directory means its ``*.json``), sniff what it is and add it to
    ``store``: a dict with ``medians`` / ``requested`` is a ``bench_llm.py`` run; dicts with
    ``median_s`` are speech rows; dicts with ``wall_ms`` are OCR rows (a list, or a dict wrapping
    one, may mix them). Anything else is skipped with a warning. Rows are dated by the file's
    modification time. ``profile_lookup(model_key)`` (the CLI passes the library's saved tuning
    profile) tells :func:`import_bench_llm_json` the context / KV / slots the server really launched
    with. Never raises."""
    report = ImportReport()
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    for path in files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            report.skipped += 1
            report.warnings.append(f"{path.name}: unreadable ({exc})")
            continue
        try:
            at = iso_now(path.stat().st_mtime)
        except OSError:
            at = iso_now()
        records = _records(raw)
        if not records:
            report.skipped += 1
            report.warnings.append(f"{path.name}: not a benchmark result")
            continue
        for rec in records:
            kind = _classify(rec)
            try:
                if kind == "llm":
                    key = str((rec.get("model") or {}).get("key") or (rec.get("requested") or {}).get("model") or "")                         if isinstance(rec.get("model"), dict) or isinstance(rec.get("requested"), dict) else ""
                    prof = None
                    if profile_lookup is not None and key:
                        try:
                            prof = profile_lookup(key)
                        except Exception:  # noqa: BLE001 - a profile lookup failure must not stop the import
                            prof = None
                    run, fp = import_bench_llm_json(rec, at=at, profile=prof)
                    if run is None:
                        raise ValueError("no usable fields")
                    store.add_llm(run)
                    if fp is not None:
                        store.add_footprint(fp)
                    report.imported += 1
                elif kind == "asr":
                    rows = import_asr_json([rec], at=at)
                    if not rows:
                        raise ValueError("a failed or empty speech row")
                    for row in rows:
                        store.add_asr(row)
                    report.imported += len(rows)
                elif kind == "ocr":
                    rows_o = import_ocr_json([rec], at=at)
                    if not rows_o:
                        raise ValueError("not a RapidOCR timing")
                    for row_o in rows_o:
                        store.add_ocr(row_o)
                    report.imported += len(rows_o)
                else:
                    raise ValueError("not a benchmark result")
            except OSError as exc:
                report.skipped += 1
                report.warnings.append(f"{path.name}: could not write the store ({exc})")
            except Exception as exc:  # noqa: BLE001 - a malformed record skips that record, never the import
                report.skipped += 1
                report.warnings.append(f"{path.name}: skipped ({exc})")
    return report


# ============================================================================================
# the live LLM probe (HTTP to the RUNNING server only)
# ============================================================================================
QUESTION = "What would you say is the main trade-off between Kafka and RabbitMQ for our use case?"
_FILLER_TOPICS = [
    "the retry backoff strategy for the ingestion workers",
    "whether we need exactly-once delivery or at-least-once is good enough here",
    "how partitioning should map onto the downstream consumers",
    "the schema registry and how we handle breaking changes",
    "back-pressure once the warehouse copy job falls behind",
    "the on-call runbook if a partition rebalance stalls",
    "cost of running a 3-broker cluster versus a managed offering",
    "how we replay a bad batch without double-processing everything downstream",
    "the dead-letter queue policy and who triages it",
    "whether the ordering guarantee actually matters for this event type",
    "compaction settings for the changelog topic",
    "how client library upgrades get rolled out without downtime",
    "the monitoring dashboards we still need before this ships",
    "load testing results from last week's staging run",
    "how this affects the SLA we already promised the analytics team",
]
_SPEAKERS = ("Speaker 1", "Me", "Speaker 2")
PROBE_EXTRA_TOKENS = 200       # request C appends this much transcript before the question
_NOISY_SPREAD = 0.25


def build_transcript(target_tokens: int, round_seed: int) -> str:
    """A synthetic meeting transcript of roughly ``target_tokens`` tokens (``approx_tokens``). The
    wording is a deterministic function of ``round_seed`` and every line carries it, so two seeds
    never share a 256-token chunk that ``--cache-reuse`` could shift out of an earlier prompt.
    Copied from ``scripts/bench_llm.py`` (the package never imports ``scripts/``)."""
    lines: list[str] = []
    idx = 0
    minute = 0
    second = 0
    budget = max(0, target_tokens - approx_tokens(QUESTION) - 40)
    while approx_tokens("\n".join(lines)) < budget:
        topic = _FILLER_TOPICS[(idx + round_seed) % len(_FILLER_TOPICS)]
        speaker = _SPEAKERS[idx % len(_SPEAKERS)]
        minute = idx * 7 // 60
        second = (idx * 7) % 60
        lines.append(
            f"[{minute:02d}:{second:02d}] {speaker}: (round {round_seed}, item {idx}) let's talk about "
            f"{topic} -- I think it changes the latency and cost trade-offs for the new ingestion "
            "service, and we should decide before the next sprint."
        )
        idx += 1
    lines.append(f"[{minute:02d}:{(second + 3) % 60:02d}] Speaker 1: {QUESTION}")
    return "\n".join(lines)


def extra_transcript_lines(extra_tokens: int, round_seed: int, start_idx: int) -> str:
    """About ``extra_tokens`` more transcript lines continuing from ``start_idx`` (never repeating
    lines :func:`build_transcript` produced for the same seed). Copied from ``scripts/bench_llm.py``."""
    lines: list[str] = []
    idx = start_idx
    while approx_tokens("\n".join(lines)) < extra_tokens:
        topic = _FILLER_TOPICS[(idx + round_seed) % len(_FILLER_TOPICS)]
        speaker = _SPEAKERS[idx % len(_SPEAKERS)]
        minute = idx * 7 // 60
        second = (idx * 7) % 60
        lines.append(
            f"[{minute:02d}:{second:02d}] {speaker}: (round {round_seed}, item {idx}, follow-up) one "
            f"more thing on {topic} before we move on."
        )
        idx += 1
    return "\n".join(lines)


def _insert_before_question(transcript: str, extra: str) -> str:
    lines = transcript.split("\n")
    return "\n".join(lines[:-1] + [extra, lines[-1]])


class _ProbeError(RuntimeError):
    """A failure the probe reports as ``{"error": str(exc)}``."""


class _NoIdleSlot(_ProbeError):
    """Every slot is generating: measuring now would disturb the answer in progress."""


class _Cancelled(Exception):
    pass


def _client_for(http: Any, timeout_s: float) -> tuple[Any, bool]:
    """``(client, owned)``. ``http`` may be None (a fresh ``httpx.Client``), an object with
    ``get`` / ``post`` / ``stream`` (used as is, never closed here), or a class / zero-argument
    callable returning one (its product is owned, so it is closed afterwards)."""
    if http is None:
        return httpx.Client(timeout=httpx.Timeout(timeout_s, connect=5.0)), True
    if inspect.isclass(http) or (callable(http) and not hasattr(http, "get")):
        return http(), True
    return http, False


def _close(client: Any, owned: bool) -> None:
    if owned and hasattr(client, "close"):
        try:
            client.close()
        except Exception:  # noqa: BLE001 - closing a probe client must never raise
            log.debug("optimize: closing the probe client failed", exc_info=True)


def _path_of(url: str) -> str:
    return "/" + url.split("/", 3)[3] if url.count("/") >= 3 else url


def _get_json(client: Any, url: str, timeout_s: float) -> Any:
    r = client.get(url, timeout=timeout_s)
    if r.status_code >= 400:
        raise _ProbeError(f"HTTP {r.status_code} from {_path_of(url)}")
    try:
        return r.json()
    except ValueError as exc:
        raise _ProbeError(f"{_path_of(url)} did not return JSON ({exc})") from exc


def _cached_tokens(slot: dict) -> int:
    """How much a slot has cached, from whichever field this llama-server build reports (0 when it
    reports none, which makes every slot equal and the lowest id win)."""
    for key in ("n_cached", "n_past", "n_prompt_tokens_cache", "n_tokens"):
        v = _num(slot.get(key))
        if v is not None:
            return int(v)
    for key in ("cache_tokens", "prompt"):
        v = slot.get(key)
        if isinstance(v, list):
            return len(v)
        if isinstance(v, str):
            return len(v) // 4
    return 0


def _pick_slot(slots: Any) -> tuple[int, int, int | None]:
    """``(slot id, slot count, n_ctx or None)``: the idle slot with the fewest cached tokens."""
    if not isinstance(slots, list) or not any(isinstance(s, dict) for s in slots):
        raise _ProbeError("/slots returned an unexpected shape")
    rows = [s for s in slots if isinstance(s, dict) and _num(s.get("id")) is not None]
    idle = []
    for s in rows:
        busy = bool(s["is_processing"]) if "is_processing" in s else (s.get("state") not in (None, 0, "idle"))
        if not busy:
            idle.append(s)
    n_ctx = next((int(v) for v in (_num(s.get("n_ctx")) for s in rows) if v), None)
    if not idle:
        raise _NoIdleSlot("every slot is busy; wait for the answer in progress to finish")
    best = min(idle, key=lambda s: (_cached_tokens(s), int(_num(s.get("id")) or 0)))
    return int(_num(best.get("id")) or 0), len(rows), n_ctx


def _stream_completion(client: Any, base: str, prompt: str, *, max_tokens: int, slot: int | None,
                       timeout_s: float = 180.0) -> tuple[float | None, dict]:
    """POST ``/v1/completions`` with ``stream=True`` and return ``(wall seconds to the first content
    delta or None, llama-server's own "timings" object off the final SSE chunk or {})``. The
    ``timings`` reader is the one in ``scripts/bench_llm.py``; ``id_slot`` pins the request."""
    body: dict[str, Any] = {"model": "", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
                            "stop": STOP, "stream": True, "cache_prompt": True}
    if slot is not None:
        body["id_slot"] = slot
    timings: dict[str, Any] = {}
    ttft: float | None = None
    t0 = time.perf_counter()
    with client.stream("POST", base + "/v1/completions", json=body, timeout=timeout_s) as r:
        if r.status_code >= 400:
            detail = ""
            try:
                r.read()
                detail = f": {r.text[:160]}"
            except Exception:  # noqa: BLE001 - the status code is what matters
                pass
            raise _ProbeError(f"HTTP {r.status_code} from /v1/completions{detail}")
        for line in r.iter_lines():
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            line = line.strip("\r")
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("error"):
                raise _ProbeError(f"server error: {str(chunk['error'])[:160]}")
            choices = chunk.get("choices") or []
            text = choices[0].get("text", "") if choices and isinstance(choices[0], dict) else ""
            if text and ttft is None:
                ttft = time.perf_counter() - t0
            if isinstance(chunk.get("timings"), dict):
                timings = chunk["timings"]
    return ttft, timings


def _median(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    return float(statistics.median(clean)) if clean else None


def probe_llm(base_url: str, *, n_ctx: int | None = None, rounds: int = 2, prompt_tokens: int | None = None,
              max_tokens: int = 96, persona: str = "", style: str = "bullets",
              cancel: Callable[[], bool] | None = None, progress: Callable[[str, float], None] | None = None,
              http: Any = None) -> dict:
    """Measure the RUNNING llama-server over HTTP (it never starts, restarts or constructs one).

    ``GET /props`` gives ``n_ctx`` (``default_generation_settings.n_ctx``) and the slot count; ``GET
    /slots`` picks the idle slot with the fewest cached tokens and every request is pinned to it with
    ``"id_slot"`` so the live conversation's KV cache survives. Per round: (A) a fresh, round-seeded
    transcript prompt of ``prompt_tokens`` (default ``min(1500, n_ctx // 4)``) gives ``pp`` and
    ``tg`` from the final chunk's ``timings``; (B) the same prompt again gives ``cached_ttft_ms`` (wall
    time to the first content delta); (C) A plus about 200 more transcript tokens before the
    question gives ``incr_prompt_n`` (``timings.prompt_n``) and ``incr_pp``. The slot is erased
    (``POST /slots/N?action=erase``, best effort) at the end, also after a failure or a cancel.

    Returns ``{"pp", "tg", "cached_ttft_ms", "incr_prompt_n", "incr_pp"`` (medians over the rounds),
    ``"spread"`` (relative max-min over the rounds, worst of pp / tg), ``"noisy"`` (spread > 25 %),
    ``"n_ctx"``, ``"slot"``, ``"rounds"`` (raw per-round numbers), ``"warnings"``, ``"prompt_tokens"``,
    ``"cancelled"``, ``"error"``: None}``. Any failure sets ``error`` and keeps what was measured
    before it; a cancel sets ``error = "cancelled"``. ``cancel`` is polled before every request,
    ``progress(message, fraction)`` after each. ``http`` is an httpx-compatible client or a factory
    for one (tests inject an ``httpx.MockTransport`` client). Never raises."""
    rounds = max(1, int(rounds))
    base = base_url.rstrip("/")
    result: dict[str, Any] = {
        "pp": None, "tg": None, "cached_ttft_ms": None, "incr_prompt_n": None, "incr_pp": None,
        "spread": 0.0, "noisy": False, "n_ctx": n_ctx, "slot": None, "rounds": [], "warnings": [],
        "prompt_tokens": None, "cancelled": False, "error": None,
    }
    client, owned = _client_for(http, 180.0)
    slot: int | None = None
    total_requests = 3 * rounds
    done_requests = 0

    def _check_cancel() -> None:
        if cancel is not None:
            try:
                stop = bool(cancel())
            except Exception:  # noqa: BLE001 - a broken predicate must not abort the probe
                stop = False
            if stop:
                raise _Cancelled()

    def _say(message: str) -> None:
        if progress is not None:
            try:
                progress(message, done_requests / total_requests)
            except Exception:  # noqa: BLE001 - a UI callback must never break a measurement
                log.debug("optimize: progress callback raised", exc_info=True)

    try:
        _check_cancel()
        props_ctx: int | None = None
        try:
            props = _get_json(client, base + "/props", 5.0)
            settings = props.get("default_generation_settings") if isinstance(props, dict) else None
            v = _num(settings.get("n_ctx")) if isinstance(settings, dict) else None
            props_ctx = int(v) if v else None
        except _ProbeError as exc:    # a transport error (nobody listening) propagates: that IS the answer
            result["warnings"].append(f"could not read /props ({exc}); the context size is unknown")
        if n_ctx is None:
            n_ctx = props_ctx
        try:
            slot, slot_count, slots_ctx = _pick_slot(_get_json(client, base + "/slots", 5.0))
            if n_ctx is None:
                n_ctx = slots_ctx
            result["slot"] = slot
            if slot_count < 2:
                result["warnings"].append("the server has a single slot: the probe shares it with the live "
                                          "conversation, whose cache it will evict")
        except _NoIdleSlot:
            raise
        except _ProbeError as exc:
            result["warnings"].append(f"/slots unavailable ({exc}): the probe cannot pin a slot and may evict "
                                      "the live conversation's cache")
        result["n_ctx"] = n_ctx

        ptok = int(prompt_tokens) if prompt_tokens else min(1500, max(256, (n_ctx or 6000) // 4))
        if n_ctx:
            ptok = min(ptok, max(128, n_ctx - PROBE_EXTRA_TOKENS - max_tokens - 600))
        result["prompt_tokens"] = ptok
        seed_base = time.time_ns() // 1000 % 1_000_000
        system = system_assist(persona, style)

        for r in range(rounds):
            seed = seed_base + r
            rec: dict[str, Any] = {"round": r}
            result["rounds"].append(rec)
            transcript = build_transcript(ptok, round_seed=seed)
            prompt_a = render_chatml(system, user_assist(transcript, QUESTION), NO_THINK_PREFIX)

            _check_cancel()
            _say(f"round {r + 1}/{rounds}: fresh prompt")
            ttft_a, t_a = _stream_completion(client, base, prompt_a, max_tokens=max_tokens, slot=slot)
            done_requests += 1
            rec.update(pp=_num(t_a.get("prompt_per_second")), tg=_num(t_a.get("predicted_per_second")),
                       prompt_n=_num(t_a.get("prompt_n")), ttft_ms=None if ttft_a is None else ttft_a * 1000)

            _check_cancel()
            _say(f"round {r + 1}/{rounds}: cached repeat")
            ttft_b, t_b = _stream_completion(client, base, prompt_a, max_tokens=max_tokens, slot=slot)
            done_requests += 1
            rec.update(cached_ttft_ms=None if ttft_b is None else ttft_b * 1000, cached_prompt_n=_num(t_b.get("prompt_n")))

            _check_cancel()
            _say(f"round {r + 1}/{rounds}: incremental")
            extra = extra_transcript_lines(PROBE_EXTRA_TOKENS, round_seed=seed, start_idx=10_000 + r * 1000)
            prompt_c = render_chatml(system, user_assist(_insert_before_question(transcript, extra), QUESTION),
                                     NO_THINK_PREFIX)
            _ttft_c, t_c = _stream_completion(client, base, prompt_c, max_tokens=max_tokens, slot=slot)
            done_requests += 1
            rec.update(incr_prompt_n=_num(t_c.get("prompt_n")), incr_pp=_num(t_c.get("prompt_per_second")))
            _say(f"round {r + 1}/{rounds} done")
    except _Cancelled:
        result["cancelled"] = True
        result["error"] = "cancelled"
    except _ProbeError as exc:
        result["error"] = str(exc)
    except httpx.HTTPError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    except Exception as exc:  # noqa: BLE001 - a probe reports, it never raises into the engine
        log.warning("optimize: LLM probe failed", exc_info=True)
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if slot is not None:
            try:
                client.post(f"{base}/slots/{slot}?action=erase", timeout=5.0)
            except Exception:  # noqa: BLE001 - best effort: an unerased slot only keeps a cache
                log.debug("optimize: slot erase failed", exc_info=True)
        _close(client, owned)

    rec_rows = result["rounds"]
    result["pp"] = _median([r.get("pp") for r in rec_rows])
    result["tg"] = _median([r.get("tg") for r in rec_rows])
    result["cached_ttft_ms"] = _median([r.get("cached_ttft_ms") for r in rec_rows])
    result["incr_prompt_n"] = _median([r.get("incr_prompt_n") for r in rec_rows])
    result["incr_pp"] = _median([r.get("incr_pp") for r in rec_rows])
    spread = max(_rel_spread([r.get("pp") for r in rec_rows if r.get("pp") is not None]),
                 _rel_spread([r.get("tg") for r in rec_rows if r.get("tg") is not None]))
    result["spread"] = round(spread, 4)
    result["noisy"] = spread > _NOISY_SPREAD
    if result["error"] is None and result["pp"] is None and result["tg"] is None:
        result["error"] = "the server returned no timings (is this a llama-server?)"
    return result


# ============================================================================================
# small helpers: VRAM sampling, llama-swap, canary
# ============================================================================================
VRAM_STABLE_MB = 256


def _sample(n: int, gap_s: float, read: Callable[[], int | None] | None,
            sleep: Callable[[float], None] | None) -> list[int]:
    if read is None:
        from exocortex import gpu
        read = gpu.vram_free_mb
    pause = sleep or time.sleep
    vals: list[int] = []
    for i in range(max(1, n)):
        if i:
            pause(gap_s)
        try:
            v = read()
        except Exception:  # noqa: BLE001 - an NVML hiccup is just a missing sample
            v = None
        if v is not None:
            vals.append(int(v))
    return vals


def sample_vram(n: int = 3, gap_s: float = 1.0, read: Callable[[], int | None] | None = None,
                sleep: Callable[[float], None] | None = None) -> tuple[int | None, bool]:
    """``(median free MiB or None, stable)`` from ``n`` readings ``gap_s`` apart (default reader
    ``gpu.vram_free_mb``). ``stable`` means every reading was obtained and they agree within
    256 MiB: something else moved on the GPU otherwise, and a calibration built on them is suspect."""
    vals = _sample(n, gap_s, read, sleep)
    if not vals:
        return None, False
    stable = len(vals) >= min(max(1, n), 2) and (max(vals) - min(vals)) <= VRAM_STABLE_MB
    return int(round(statistics.median(vals))), stable


def swap_resident(url: str, timeout_s: float = 1.5, http_get: Callable | None = None) -> bool | None:
    """Does the llama-swap service at ``url`` hold a model on the GPU right now? ``GET url/running``
    -> ``len(running) > 0``; ``None`` when there is no such service or it answers something else (a
    machine without llama-swap is normal, not an error). ``http_get(url, timeout=...)`` returns an
    object with ``status_code`` and ``json()`` (default ``httpx.get``)."""
    if not url:
        return None
    getter = http_get or httpx.get
    try:
        r = getter(url.rstrip("/") + "/running", timeout=timeout_s)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:  # noqa: BLE001 - unreachable / malformed both mean "could not ask"
        return None
    rows = data.get("running") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return None
    return len(rows) > 0


def canary(base_url: str, timeout_s: float = 10.0, http: Any = None) -> tuple[bool, float]:
    """A one-token ``/v1/completions``: ``(answered with choices, wall milliseconds)``. Used after a
    relaunch to prove the server really generates. Never raises."""
    client, owned = _client_for(http, timeout_s)
    t0 = time.perf_counter()
    ok = False
    try:
        r = client.post(base_url.rstrip("/") + "/v1/completions", timeout=timeout_s,
                        json={"model": "", "prompt": "Hello", "max_tokens": 1, "temperature": 0.0, "stream": False})
        if r.status_code == 200:
            data = r.json()
            ok = isinstance(data, dict) and bool(data.get("choices"))
    except Exception:  # noqa: BLE001 - a dead server is the answer, not an error
        ok = False
    finally:
        _close(client, owned)
    return ok, (time.perf_counter() - t0) * 1000.0


# ============================================================================================
# child processes
# ============================================================================================
def python_cmd() -> str:
    """The interpreter that runs the children (this one: same venv, same packages)."""
    return sys.executable


def _safe(pred: Callable[[], bool] | None, default: bool) -> bool:
    """``pred()`` as a bool; ``default`` when there is none or it raises (a broken predicate must
    neither kill a child nor keep one alive)."""
    if pred is None:
        return default
    try:
        return bool(pred())
    except Exception:  # noqa: BLE001 - see above
        return default


def run_child(argv_tail: list[str], *, out_path: Path, deadline_s: float, cancel: Callable[[], bool] | None = None,
              still_ok: Callable[[], bool] | None = None, procs: set | None = None, python: str | None = None,
              popen: Any = None, cwd: Path | None = None) -> dict:
    """Run ``python -m exocortex.optimize_bench <argv_tail> --out <out_path>`` in the project root and
    return the JSON it wrote. The child gets ``HF_HUB_OFFLINE=1``, no stdio, no console window and
    below-normal priority (Windows), and is KILLED when ``cancel()`` is true, when ``still_ok()``
    turns false (speech resumed, free VRAM collapsed), or after ``deadline_s``; it is polled every
    0.5 s. The result file is read only after the child has exited. Failures return ``{"error":
    reason}``: ``"cancelled"``, ``"timeout after N s"``, ``"no result file"``, ``"unreadable
    result"`` ... The running ``Popen`` is in ``procs`` (added / discarded) so a shutdown can kill it."""
    out_path = Path(out_path)
    cmd = [python or python_cmd(), "-m", "exocortex.optimize_bench", *argv_tail, "--out", str(out_path)]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}
    kwargs: dict[str, Any] = {"cwd": str(cwd or project_root()), "env": env, "stdin": subprocess.DEVNULL,
                              "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = (getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                                   | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000))
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.unlink(missing_ok=True)      # never read a previous run's answer
    except OSError as exc:
        return {"error": f"cannot prepare the result file: {exc}"}
    try:
        proc = (popen or subprocess.Popen)(cmd, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a missing interpreter / policy block is a result, not a crash
        return {"error": f"could not start the child: {exc}"}
    if procs is not None:
        procs.add(proc)

    def _kill() -> None:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001 - already gone
            pass
        try:
            proc.wait(5)
        except Exception:  # noqa: BLE001 - the OS will reap it
            pass

    reason = ""
    started = time.monotonic()
    try:
        while True:
            try:
                proc.wait(0.5)
                break                                   # exited
            except subprocess.TimeoutExpired:
                pass
            if _safe(cancel, False):
                reason = "cancelled"
            elif not _safe(still_ok, True):
                reason = "cancelled"
            elif time.monotonic() - started > deadline_s:
                reason = f"timeout after {deadline_s:g} s"
            if reason:
                _kill()
                return {"error": reason}
    finally:
        if procs is not None:
            procs.discard(proc)

    try:
        text = out_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        rc = getattr(proc, "returncode", None)
        return {"error": "no result file" + (f" (child exit code {rc})" if rc not in (None, 0) else "")}
    except OSError as exc:
        return {"error": f"unreadable result ({exc})"}
    try:
        data = json.loads(text)
    except ValueError as exc:
        return {"error": f"unreadable result ({exc})"}
    if not isinstance(data, dict):
        return {"error": "unreadable result (not a JSON object)"}
    return data


# ---- probe-devices --------------------------------------------------------------------------
def _classify_openvino(devices: dict[str, str]) -> tuple[str, str]:
    """``(igpu name, npu name)`` from OpenVINO's ``{device: full name}``: the integrated GPU is a
    ``GPU.x`` whose name says "(iGPU)", else an Intel one that does not say "(dGPU)"; the NPU is the
    ``NPU`` device. ``""`` when absent."""
    igpu = npu = ""
    for key, name in devices.items():
        k, n = str(key).upper(), str(name)
        if k.startswith("NPU") and not npu:
            npu = n
        elif k.startswith("GPU") and not igpu:
            low = n.lower()
            if "(igpu)" in low or ("intel" in low and "(dgpu)" not in low):
                igpu = n
    return igpu, npu


def _probe_openvino() -> dict[str, str]:
    from exocortex.screen.ocr import openvino_devices

    return {str(k): str(v) for k, v in openvino_devices().items()}


def _probe_ct2_cuda() -> bool:
    from exocortex.cuda_env import setup_cuda_dlls

    setup_cuda_dlls()
    import ctranslate2

    return int(ctranslate2.get_cuda_device_count()) > 0


def _bundle_ready(model_dir: Path) -> bool:
    try:
        from exocortex.models_dl import _sherpa_model_ready
    except Exception:  # noqa: BLE001 - private helper moved: fall back to the same rule
        _sherpa_model_ready = None
    if _sherpa_model_ready is not None:
        return bool(_sherpa_model_ready(model_dir))
    return (model_dir / "tokens.txt").is_file() and all(any(model_dir.glob(p)) for p in
                                                        ("encoder*.onnx", "decoder*.onnx", "joiner*.onnx"))


def _probe_sherpa(models_dir: Path, names: tuple[str, ...]) -> dict[str, bool]:
    return {name: _bundle_ready(Path(models_dir) / name) for name in names}


def _probe_mem_mts() -> int | None:
    if sys.platform != "win32":
        return None
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
         "Get-CimInstance Win32_PhysicalMemory | Select -First 1 -ExpandProperty ConfiguredClockSpeed"],
        capture_output=True, text=True, timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    text = proc.stdout.strip().splitlines()
    return int(text[0].strip()) if text and text[0].strip().isdigit() else None


def _probe_power() -> dict | None:
    from exocortex import gpu

    ps = gpu.power_state()
    if ps is None:
        return None
    out = dataclasses.asdict(ps)
    out["reasons"] = list(ps.reasons)
    out["power_capped"] = ps.power_capped
    out["thermal"] = ps.thermal
    return out


def collect_devices(models_dir: Path | None = None, sherpa_model: str = "") -> dict:
    """The ``probe-devices`` child's answer: ``{"openvino", "igpu", "npu", "ct2_cuda", "sherpa_ready",
    "sherpa_models", "mem_mts", "power", "errors"}``. Each part is probed on its own and fails soft
    to its empty value (the failure is named in ``errors``)."""
    from exocortex.config import ASRConfig

    models = Path(models_dir) if models_dir else project_root() / "models"
    wanted = sherpa_model or ASRConfig().sherpa_model
    errors: dict[str, str] = {}

    def _try(name: str, fn: Callable[[], Any], default: Any) -> Any:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - one broken probe must not blank the others
            errors[name] = f"{type(exc).__name__}: {exc}"
            return default

    openvino = _try("openvino", _probe_openvino, {})
    igpu, npu = _classify_openvino(openvino)
    bundles = _try("sherpa", lambda: _probe_sherpa(models, tuple(dict.fromkeys(
        (wanted, "parakeet-tdt-0.6b-v2-int8", "parakeet-tdt-0.6b-v3-int8")))), {})
    return {
        "openvino": openvino, "igpu": igpu, "npu": npu,
        "ct2_cuda": bool(_try("ct2_cuda", _probe_ct2_cuda, False)),
        "sherpa_ready": bool(bundles.get(wanted, False)), "sherpa_models": bundles,
        "mem_mts": _try("mem_mts", _probe_mem_mts, None),
        "power": _try("power", _probe_power, None),
        "errors": errors,
    }


# ---- probe-asr ------------------------------------------------------------------------------
#: ``(start s, end s, speaker, text)`` for tests/data/two_speakers_16k.wav -- a literal copy of
#: ``scripts/bench_asr.py::TWO_SPEAKERS_TRUTH`` (itself a copy of ``tests/conftest.py``)
TWO_SPEAKERS_TRUTH: list[tuple[float, float, str, str]] = [
    (0.13, 0.98, "A", "Hi everyone!"),
    (1.35, 2.29, "A", "Thanks for joining."),
    (3.17, 6.86, "A", "Today we need to decide on the message queue for the new ingestion service."),
    (7.65, 8.18, "B", "Sure."),
    (9.03, 13.81, "B", "What would you say is the main trade-off between Kafka and RabbitMQ for our use case?"),
    (14.63, 15.38, "A", "Good question."),
    (16.23, 19.89, "A", "Let's also make sure someone owns the migration plan by Friday."),
    (20.71, 22.26, "B", "I can take that action item"),
    (23.11, 26.74, "B", "Can you walk me through how you'd design a rate limiter for the public API?"),
]
ASR_VARIANTS: dict[str, dict[str, str]] = {
    "whisper-cuda-int8_float16": {"engine": "faster-whisper", "device": "cuda", "compute": "int8_float16"},
    "whisper-cuda-float16": {"engine": "faster-whisper", "device": "cuda", "compute": "float16"},
    "whisper-cpu-int8": {"engine": "faster-whisper", "device": "cpu", "compute": "int8"},
    "sherpa-cpu": {"engine": "sherpa", "device": "cpu", "compute": "int8"},
}
ASR_UTTERANCES = 4
ASR_PASSES = 2
VRAM_TRIPLE_SPREAD_MB = 200


def _word_overlap(hypothesis: str, truth: str) -> float:
    """Fraction of ``truth``'s words also in ``hypothesis`` (case-insensitive alphanumeric tokens)."""
    words = lambda s: set(re.findall(r"[a-z0-9']+", s.lower()))  # noqa: E731
    truth_words = words(truth)
    if not truth_words:
        return 1.0
    return len(words(hypothesis) & truth_words) / len(truth_words)


def _offline_sherpa(name: str, models_dir: Path, progress: Any = None) -> Path:
    """The ``ensure_sherpa_model`` stand-in a probe hands the Transcriber: the bundle's directory if
    it is already extracted, else ``FileNotFoundError``. A benchmark must never download 480 MB."""
    model_dir = Path(models_dir) / name
    if _bundle_ready(model_dir):
        return model_dir
    raise FileNotFoundError(f"sherpa model {name} is not on disk at {model_dir}; a probe never downloads")


def _load_clip(path: Path) -> tuple[Any, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr)


def _vram_cost(before: list[int], after: list[int]) -> tuple[int | None, str]:
    """MiB the load cost from median-of-3 free-VRAM triples, or ``(None, "none")`` when a triple is
    short or spreads more than 200 MiB (something else moved) or the cost is negative."""
    if len(before) < 3 or len(after) < 3:
        return None, "none"
    if max(before) - min(before) > VRAM_TRIPLE_SPREAD_MB or max(after) - min(after) > VRAM_TRIPLE_SPREAD_MB:
        return None, "none"
    delta = statistics.median(before) - statistics.median(after)
    if delta < 0:
        return None, "none"
    return int(round(delta)), "nvml"


def probe_asr(variant: str, *, models_dir: Path, clip: Path, threads: int = 4, language: str = "en",
              sherpa_model: str = "", transcriber_cls: Any = None, load_audio: Callable | None = None,
              read_vram: Callable[[], int | None] | None = None, sleep: Callable[[float], None] | None = None) -> dict:
    """Load one speech variant through the app's own ``Transcriber`` and time it on the four longest
    utterances of the truth-table clip, two passes (the load already ran the warm-up). Returns
    ``{"label", "engine", "device", "compute_type", "threads", "load_s", "median_s", "max_s",
    "overlap", "vram_mb", "vram_source", "error"}`` -- no transcript text. A CUDA variant that
    silently fell back to the CPU (the Transcriber does that when the load fails) is reported as an
    error, not as a CPU timing under a CUDA label. VRAM (CUDA variants only) is the median-of-3 NVML
    free before minus after the load, rejected when negative or when either triple spreads over
    200 MiB. Never raises; never downloads."""
    spec = ASR_VARIANTS.get(variant)
    result: dict[str, Any] = {
        "label": variant, "engine": (spec or {}).get("engine", ""), "device": (spec or {}).get("device", ""),
        "compute_type": (spec or {}).get("compute", ""), "threads": 0, "load_s": None, "median_s": None,
        "max_s": None, "overlap": None, "vram_mb": None, "vram_source": "none", "error": None,
    }
    if spec is None:
        result["error"] = f"unknown variant {variant!r}; known: {', '.join(ASR_VARIANTS)}"
        return result
    try:
        clip = Path(clip)
        if not clip.is_file():
            result["error"] = f"clip not found: {clip}"
            return result
        from exocortex.config import ASRConfig

        is_cpu = spec["device"] == "cpu"
        cfg_kwargs: dict[str, Any] = {"engine": spec["engine"], "device": spec["device"], "compute_type": spec["compute"],
                                      "language": language, "cpu_threads": int(threads) if is_cpu else 0}
        if sherpa_model:
            cfg_kwargs["sherpa_model"] = sherpa_model
        cfg = ASRConfig(**cfg_kwargs)
        result["threads"] = cfg.cpu_threads
        if transcriber_cls is None:
            from exocortex.speech.asr import Transcriber as transcriber_cls  # noqa: N813
        audio, sr = (load_audio or _load_clip)(clip)
        if sr != 16000:
            result["error"] = f"the clip must be 16 kHz (got {sr})"
            return result

        before: list[int] = []
        if not is_cpu:
            before = _sample(3, 0.5, read_vram, sleep)
        t = transcriber_cls(cfg, models_dir=Path(models_dir), ensure_sherpa_model=_offline_sherpa)
        t0 = time.perf_counter()
        t.load()
        result["load_s"] = round(time.perf_counter() - t0, 2)
        engine, device, compute = t.engine_used, t.device_used, t.compute_type_used
        result.update(engine=engine or spec["engine"], device=device or spec["device"],
                      compute_type=compute or spec["compute"])
        if engine != spec["engine"] or (spec["engine"] != "sherpa" and (device, compute) != (spec["device"], spec["compute"])):
            result["error"] = (f"{variant} is unavailable: the Transcriber loaded {engine}/{device}/{compute} instead "
                               "(a model that is not on disk is never downloaded)")
            return result
        if not is_cpu:
            result["vram_mb"], result["vram_source"] = _vram_cost(before, _sample(3, 0.5, read_vram, sleep))

        longest = sorted(TWO_SPEAKERS_TRUTH, key=lambda row: row[1] - row[0], reverse=True)[:ASR_UTTERANCES]
        times: list[float] = []
        overlaps: list[float] = []
        for _pass in range(ASR_PASSES):
            for start, end, _speaker, truth in longest:
                res = t.transcribe(audio[int(start * sr): int(end * sr)])
                times.append(float(res.elapsed_s))
                overlaps.append(_word_overlap(res.text, truth))
        result["median_s"] = round(statistics.median(times), 4)
        result["max_s"] = round(max(times), 4)
        result["overlap"] = round(statistics.mean(overlaps), 3)
    except Exception as exc:  # noqa: BLE001 - the child reports a failure as data
        log.warning("optimize: probe-asr %s failed", variant, exc_info=True)
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# ============================================================================================
# CLI: child entry points + the lead's "import" command
# ============================================================================================
def _emit(result: dict, out: str | None) -> int:
    if not out:
        print(json.dumps(result, indent=2, default=str))
        return 0
    try:
        _atomic_write_json(Path(out), result)
    except OSError as exc:
        print(f"cannot write {out}: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """``probe-devices [--out F]``, ``probe-asr --variant L --clip WAV [--models-dir D --threads N
    --language en --sherpa-model M --out F]`` (the children :func:`run_child` launches; each writes
    one JSON object and exits 0 even when the probe failed -- the failure is the ``error`` field) and
    ``import FILE|DIR... [--store-dir D | --config C]`` (add sweep JSON to the benchmark store)."""
    ap = argparse.ArgumentParser(prog="python -m exocortex.optimize_bench", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_dev = sub.add_parser("probe-devices", help="OpenVINO devices, CUDA for CTranslate2, sherpa bundle, RAM speed, GPU power")
    p_dev.add_argument("--out")
    p_dev.add_argument("--models-dir")
    p_dev.add_argument("--sherpa-model", default="")
    p_asr = sub.add_parser("probe-asr", help="time one speech variant on the truth-table clip")
    p_asr.add_argument("--variant", required=True)
    p_asr.add_argument("--models-dir")
    p_asr.add_argument("--clip", required=True)
    p_asr.add_argument("--threads", type=int, default=4)
    p_asr.add_argument("--language", default="en")
    p_asr.add_argument("--sherpa-model", default="")
    p_asr.add_argument("--out")
    p_imp = sub.add_parser("import", help="add bench_llm / bench_asr / bench_ocr JSON to the benchmark store")
    p_imp.add_argument("files", nargs="+")
    p_imp.add_argument("--store-dir")
    p_imp.add_argument("--config")
    args = ap.parse_args(argv)

    if args.cmd == "probe-devices":
        try:
            result = collect_devices(Path(args.models_dir) if args.models_dir else None, args.sherpa_model)
        except Exception as exc:  # noqa: BLE001 - always answer
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return _emit(result, args.out)

    if args.cmd == "probe-asr":
        os.environ["HF_HUB_OFFLINE"] = "1"      # a probe never downloads, even when launched by hand
        try:
            models = Path(args.models_dir) if args.models_dir else project_root() / "models"
            result = probe_asr(args.variant, models_dir=models, clip=Path(args.clip), threads=args.threads,
                               language=args.language, sherpa_model=args.sherpa_model)
        except Exception as exc:  # noqa: BLE001 - always answer
            result = {"label": args.variant, "error": f"{type(exc).__name__}: {exc}"}
        return _emit(result, args.out)

    # import
    if args.store_dir:
        directory = Path(args.store_dir)
    else:
        from exocortex.config import load_config
        from exocortex.llm.library import library_dir

        directory = library_dir(load_config(args.config).llm)

    def _lookup(key: str) -> dict | None:
        """The saved tuning profile of ``key`` (the context the server really launched with)."""
        try:
            import dataclasses as _dc

            from exocortex.config import load_config as _lc
            from exocortex.llm.library import profile_for

            prof = profile_for(_lc(args.config).llm, key)
            return _dc.asdict(prof) if prof is not None else None
        except Exception:  # noqa: BLE001 - no config / library: import with the global values
            return None

    store = Store(directory)
    report = import_files([Path(f) for f in args.files], store, profile_lookup=_lookup)
    try:
        from exocortex import hardware as _hw
        from exocortex.config import load_config as _lc2

        cfg_ = _lc2(args.config)
        rep = _hw.load(_hw.report_path(cfg_.llm)) or _hw.collect(cfg_.llm)
        d = _hw.to_dict(rep)
        rt = d.get("runtime") or {}
        store.set_fingerprint(fingerprint(d, f"{rt.get('flavor', '')} {rt.get('version', '')}".strip()))
    except Exception:  # noqa: BLE001 - the fingerprint is a bonus; the rows are what matters
        log.debug("import: fingerprint not recorded", exc_info=True)
    print(f"imported {report.imported} record(s), skipped {report.skipped}, into {directory / BENCH_FILE}")
    for w in report.warnings:
        print(f"  ! {w}")
    return 0 if report.imported else 1


__all__ = [
    "BENCH_FILE", "JOURNAL_FILE", "STALE_AFTER_S", "DEVICES_FRESH_S", "Store", "ImportReport", "fingerprint",
    "iso_now", "parse_at", "import_bench_llm_json", "import_asr_json", "import_ocr_json", "import_files",
    "probe_llm", "run_child", "sample_vram", "swap_resident", "canary", "python_cmd", "main",
    "probe_asr", "collect_devices", "build_transcript", "extra_transcript_lines", "ASR_VARIANTS",
]


if __name__ == "__main__":
    sys.exit(main())
