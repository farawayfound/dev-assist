"""The Optimize controller (2026-10-07): the engine-side driver behind Settings > Optimize.

:mod:`exocortex.optimize` is pure arithmetic and :mod:`exocortex.optimize_bench` only gathers
numbers; this module is the one that touches the live engine. It owns three jobs, each on its own
named daemon thread, and one small piece of state the Optimize tab polls:

* **analyze** (``optimize-analyze``): collect :class:`~exocortex.optimize.PlanInputs` for THIS
  machine -- the hardware report, live NVML readings (free VRAM, power state), the OpenVINO /
  CTranslate2 / Parakeet device probe (a child process, cached a week), the loaded model's GGUF
  header, its saved tuning profile, the config, and the benchmark store -- and run
  :func:`~exocortex.optimize.plan_all`. The result lands in the snapshot as plain dicts.
* **benchmark** (``optimize-bench``): measure the RUNNING llama-server over HTTP
  (:func:`~exocortex.optimize_bench.probe_llm`; it never starts a second server -- two on the 5090
  crashed both) and the speech engines in killable child processes, save every number to
  ``benchmarks.json`` as it arrives, and re-plan after each step. Gated by
  :func:`~exocortex.optimize.gate` so it never runs over a live answer or while someone speaks.
* **apply** (on the engine's own ``llm-swap`` thread via ``Engine._start_swap``): the model part of a
  plan as a transaction -- journal ``pending`` first, write the merged profile, relaunch, verify
  (context window, a canary completion, free VRAM above the guard floor), then journal
  ``verified`` -- or restore the previous profile, relaunch again and journal ``rolled-back``.
  :meth:`OptimizeController.recover` runs at start-up: a ``pending`` entry means the app died
  mid-apply, and the old profile is restored BEFORE llama-server launches, so a profile the model
  cannot start with never crash-loops every launch. :meth:`undo` replays the last verified entry's
  previous profile through the same transaction.

Why polling, not the bus: the overlay cannot be edited for this feature, and it forwards no new
event kinds to the settings dialog. :meth:`snapshot` therefore returns a copy of an immutable dict
that the tab reads on a timer; :class:`~exocortex.bus.OptimizeEvent` is published too, for the log
and anything else that cares. ``_activity`` is the ONLY place that peeks at engine internals, every
attribute through ``getattr`` with a default, so a fake engine in the tests (or a future refactor)
degrades the gates instead of raising.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from exocortex import optimize as O
from exocortex.bus import ComponentStatusEvent, OptimizeEvent
from exocortex.config import project_root

log = logging.getLogger(__name__)

#: how long one speech probe child may run before it is killed
ASR_CHILD_DEADLINE_S = 180.0
DEVICES_CHILD_DEADLINE_S = 30.0
#: the apply transaction's verification limits
CANARY_TIMEOUT_S = 5.0
VERIFY_VRAM_SAMPLES = 3
#: a free-VRAM collapse of this much while a child runs means the shared server woke up: kill it
VRAM_COLLAPSE_MB = 3072
#: how many plans (by digest) are remembered for ``apply``
PLAN_CACHE = 8
#: ASR variants the speech step may try, in order; CUDA ones only with room on the GPU
SPEECH_VARIANTS = ("sherpa-cpu", "whisper-cuda-int8_float16")
WHISPER_CUDA_MB = 874
CLIP = project_root() / "tests" / "data" / "two_speakers_16k.wav"


def _asr_label(engine_used: str, device_used: str, compute_used: str) -> str:
    """The planner's canonical label for what the Transcriber loaded."""
    if not engine_used:
        return ""
    if engine_used == "sherpa":
        return "sherpa-cpu"
    if device_used == "cpu":
        return "whisper-cpu-int8"
    return f"whisper-cuda-{compute_used or 'float16'}"


def _call(obj: Any, name: str, *args: Any, default: Any = None) -> Any:
    """``obj.name(*args)`` (or the attribute itself when it is not callable), ``default`` on any
    failure or absence -- the engine-internals accessor used everywhere here."""
    try:
        attr = getattr(obj, name, None)
        if attr is None:
            return default
        return attr(*args) if callable(attr) else attr
    except Exception:  # noqa: BLE001 - a peek at engine state must never take the controller down
        return default


class OptimizeController:
    """See the module docstring. ``engine`` is the :class:`exocortex.app.Engine` (duck-typed:
    ``cfg``, ``bus``, ``server``, ``client``, ``assistant``, ``transcriber``, ``pipeline``,
    ``transcript``, ``clock()``, ``hardware_report()``, ``loaded_model()``, ``_start_swap(work,
    what)``, ``_relaunch_blocking(key)``, ``_status(component, state, detail)``, ``_stopped``)."""

    def __init__(self, engine: Any, *, store: Any = None, now: Callable[[], float] = time.time) -> None:
        self._engine = engine
        self._now = now
        self._store = store
        self._lock = threading.RLock()
        self._busy = threading.Lock()              # benchmark and apply are mutually exclusive
        self._analyze_lock = threading.Lock()
        self._cancel = threading.Event()
        self._procs: set = set()
        self._threads: list[threading.Thread] = []
        self._plans: dict[str, O.Plan] = {}        # digest -> plan (last PLAN_CACHE)
        self._plan_order: list[str] = []
        self._baseline: tuple[int, int] | None = None   # (vram_total_mb, vram_used_by_others_mb) at start
        self._run_id = 0
        self._pending_analyze: tuple[str, dict | None, str] | None = None
        self._devices: dict | None = None
        self._snap: dict[str, Any] = {
            "engine": {"ready": False, "backend": "", "model": "", "swap_resident": None},
            "facts": {}, "models": [], "plans": {}, "selected": "balanced",
            "gates": {"llm": [False, "not analysed yet"], "speech": [False, "not analysed yet"], "apply": [False, "not analysed yet"]},
            "analysis": {"state": "idle", "message": "", "age_s": 0.0},
            "run": {"state": "idle", "run_id": 0, "step": "", "fraction": 0.0, "message": ""},
            "measured": {"llm": None, "asr": [], "ocr": [], "at": "", "stale": False},
            "apply": {"state": "idle", "message": "", "digest": "", "undo": False},
        }
        self._analysis_at = 0.0

    # ------------------------------------------------------------------ plumbing -----------
    @property
    def _cfg(self) -> Any:
        return self._engine.cfg

    def _bench(self) -> Any:
        from exocortex import optimize_bench as ob

        return ob

    def _get_store(self) -> Any:
        if self._store is None:
            from exocortex.llm import library

            self._store = self._bench().Store(Path(library.library_dir(self._cfg.llm)))
        return self._store

    def _update(self, **parts: Any) -> None:
        with self._lock:
            snap = dict(self._snap)
            for key, value in parts.items():
                if isinstance(value, dict) and isinstance(snap.get(key), dict):
                    merged = dict(snap[key])
                    merged.update(value)
                    snap[key] = merged
                else:
                    snap[key] = value
            self._snap = snap

    def _publish(self, kind: str, *, run_id: int = 0, step: str = "", fraction: float = 0.0, detail: str = "") -> None:
        bus = getattr(self._engine, "bus", None)
        if bus is None:
            return
        try:
            bus.publish(OptimizeEvent(kind=kind, run_id=run_id, step=step, fraction=fraction, detail=detail))  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            log.debug("OptimizeEvent publish failed", exc_info=True)

    def _toast(self, state: str, detail: str) -> None:
        _call(self._engine, "_status", "app", state, detail)

    def _stopped(self) -> bool:
        ev = getattr(self._engine, "_stopped", None)
        return bool(ev is not None and _call(ev, "is_set", default=False))

    def snapshot(self) -> dict[str, Any]:
        """A deep copy of the cached state (cheap: plain data). Never raises."""
        try:
            with self._lock:
                snap = copy.deepcopy(self._snap)
            snap["analysis"]["age_s"] = round(self._now() - self._analysis_at, 1) if self._analysis_at else 0.0
            return snap
        except Exception:  # noqa: BLE001
            log.exception("optimize snapshot failed")
            return {}

    # ------------------------------------------------------------------ start / stop --------
    def note_baseline(self) -> None:
        """Remember how much VRAM everything else holds right now (called from ``Engine.start``
        before llama-server launches, after ``--replace`` freed llama-swap). Discarded when more than
        half the card is in use -- that is a resident shared model, not the desktop."""
        backend = getattr(getattr(self._engine, "route", None), "backend", None) or self._cfg.llm.backend
        if backend not in ("llamacpp", "none", "auto"):
            return
        try:
            from exocortex import gpu

            total, free = gpu.vram_total_mb(), gpu.vram_free_mb()
        except Exception:  # noqa: BLE001
            return
        if total is None or free is None:
            return
        used = max(0, int(total) - int(free))
        if used > 0.5 * total:
            log.info("optimize: %d MiB in use before launch -- not a desktop baseline, ignored", used)
            return
        self._baseline = (int(total), used)

    def recover(self) -> None:
        """Roll back an apply the previous run died in the middle of (a ``pending`` journal entry),
        BEFORE llama-server launches."""
        try:
            store = self._get_store()
            entries = store.journal()
        except Exception:  # noqa: BLE001
            log.debug("optimize: no journal to recover", exc_info=True)
            return
        changed = False
        for entry in entries:
            if entry.get("state") != "pending":
                continue
            key = str(entry.get("key") or "")
            try:
                self._write_profile(key, entry.get("previous"))
                entry["state"] = "recovered"
                entry["detail"] = "restored at start-up: the previous apply did not finish"
                changed = True
                log.warning("optimize: restored the previous profile of %s (an apply did not finish)", key)
                self._publish("recovered", detail=key)
                self._toast("info", f"Optimize: the unfinished change to {key} was undone")
            except Exception:  # noqa: BLE001
                log.exception("optimize: could not restore the profile of %s", key)
        if changed:
            try:
                store.journal_write(entries)
            except Exception:  # noqa: BLE001
                log.exception("optimize: journal write failed during recovery")
        self._update(apply={"undo": self._can_undo(entries)})

    def shutdown(self) -> None:
        """Cancel everything, kill the probe children and join the workers (at most ~3 s)."""
        self._cancel.set()
        for proc in list(self._procs):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        deadline = time.monotonic() + 3.0
        for t in list(self._threads):
            remaining = deadline - time.monotonic()
            if remaining > 0 and t.is_alive():
                t.join(timeout=remaining)

    # ------------------------------------------------------------------ activity / gates ----
    def _activity(self) -> O.Activity:
        e = self._engine
        server = getattr(e, "server", None)
        running = bool(_call(server, "running", default=False)) if server is not None else False
        llm_ready = bool(_call(getattr(e, "_llm_ready", None), "is_set", default=False))
        assistant = getattr(e, "assistant", None)
        assist_busy = getattr(assistant, "_current_job", None) is not None
        swap_thread = getattr(e, "_swap_thread", None)
        swap_busy = bool(swap_thread is not None and _call(swap_thread, "is_alive", default=False))
        pipeline = getattr(e, "pipeline", None)
        asr_busy = bool(_call(getattr(pipeline, "_asr_busy", None), "is_set", default=False))
        idle_s = 1e9
        try:
            transcript = getattr(e, "transcript", None)
            last = transcript.last(n=1) if transcript is not None else []
            if last:
                idle_s = max(0.0, float(e.clock()) - float(getattr(last[-1], "t_end", 0.0)))
        except Exception:  # noqa: BLE001
            idle_s = 1e9
        if asr_busy:
            idle_s = 0.0
        backend = self._backend()
        return O.Activity(
            llm_ready=llm_ready, backend=str(backend), server_running=running, assist_busy=assist_busy,
            swap_busy=swap_busy, transcriber_ready=getattr(e, "transcriber", None) is not None,
            speech_idle_s=idle_s, swap_resident=self._swap_resident(),
            handsfree_on=bool(getattr(getattr(e, "handsfree", None), "enabled", False)),
            paused=bool(getattr(e, "_paused", False)),
        )

    def _backend(self) -> str:
        """The resolved backend: the route's, else [llm].backend; "auto" / "none" read as llamacpp
        (the only backend this engine spawns a server for)."""
        backend = getattr(getattr(self._engine, "route", None), "backend", None) or self._cfg.llm.backend
        return "llamacpp" if backend in ("auto", "none", "", None) else str(backend)

    def _swap_resident(self) -> bool | None:
        url = (getattr(self._cfg.llm, "swap_unload_url", "") or "").strip()
        if not url:
            return None
        try:
            return self._bench().swap_resident(url)
        except Exception:  # noqa: BLE001
            return None

    def _gates(self, act: O.Activity | None = None) -> dict[str, list]:
        act = act or self._activity()
        return {scope: list(O.gate(scope, act)) for scope in ("llm", "speech", "apply")}

    # ------------------------------------------------------------------ facts ----------------
    def _models(self) -> list:
        from exocortex.llm import library

        try:
            return list(library.scan_library(self._cfg.llm, with_meta=True))
        except Exception:  # noqa: BLE001
            log.exception("optimize: library scan failed")
            return []

    def _profile_dict(self, key: str) -> dict | None:
        from exocortex.llm import library

        try:
            p = library.profile_for(self._cfg.llm, key)
        except Exception:  # noqa: BLE001
            return None
        return dataclasses.asdict(p) if p is not None else None

    def _ensure_devices(self, fp: dict | None) -> dict | None:
        """The cached device probe, or a fresh ``probe-devices`` child (30 s cap)."""
        store = self._get_store()
        cached = store.devices(fp, self._now())
        if cached:
            self._devices = cached
            return cached
        ob = self._bench()
        out = Path(store.path).parent / "optimize_devices.json" if hasattr(store, "path") else project_root() / "models" / "optimize_devices.json"
        result = ob.run_child(["probe-devices", "--models-dir", str(self._cfg.models_dir()), "--sherpa-model", self._cfg.asr.sherpa_model],
                              out_path=out, deadline_s=DEVICES_CHILD_DEADLINE_S, cancel=self._cancel.is_set, procs=self._procs)
        if result and not result.get("error"):
            try:
                store.set_devices(result, self._now())
            except Exception:  # noqa: BLE001
                log.debug("optimize: devices cache write failed", exc_info=True)
            self._devices = result
            return result
        log.info("optimize: device probe failed: %s", (result or {}).get("error"))
        return None

    def _gather(self, settings: O.Settings, model_key: str) -> tuple[O.PlanInputs, dict, list, dict | None]:
        """Everything the planner needs plus the facts block of the snapshot."""
        from exocortex import gpu
        from exocortex.llm import library

        e, cfg = self._engine, self._cfg
        ob = self._bench()
        hw = _call(e, "hardware_report", default={}) or {}
        runtime = hw.get("runtime") or {}
        runtime_label = f"{runtime.get('flavor', '')} {runtime.get('version', '')}".strip()
        fp = None
        try:
            fp = ob.fingerprint(hw, runtime_label)
        except Exception:  # noqa: BLE001
            fp = None

        total = _call(gpu, "vram_total_mb") or hw.get("vram_total_mb")
        free_med, stable = ob.sample_vram(n=3, gap_s=0.2)
        free = free_med if free_med is not None else _call(gpu, "vram_free_mb")
        power = _call(gpu, "power_state")
        power_d = None
        if power is not None:
            power_d = {**dataclasses.asdict(power), "power_capped": bool(power.power_capped), "thermal": bool(power.thermal)}
        swap = self._swap_resident()
        baseline_mb = self._baseline[1] if self._baseline else None
        gpus = hw.get("gpus") or []
        primary = hw.get("primary_index")
        gpu_name = ""
        if isinstance(primary, int) and 0 <= primary < len(gpus):
            gpu_name = str(gpus[primary].get("name", ""))
        elif gpus:
            gpu_name = str(gpus[0].get("name", ""))
        gf = None
        if total:
            gf = O.GpuFacts(name=gpu_name, total_mb=int(total), free_mb=int(free) if free is not None else None,
                            baseline_mb=baseline_mb, stable=bool(stable), swap_resident=swap,
                            power_capped=bool(power_d and power_d.get("power_capped")),
                            power_w=power_d.get("power_w") if power_d else None,
                            power_limit_w=power_d.get("power_limit_w") if power_d else None,
                            sm_mhz=power_d.get("sm_mhz") if power_d else None,
                            sm_max_mhz=power_d.get("sm_max_mhz") if power_d else None,
                            temperature_c=power_d.get("temperature_c") if power_d else None)

        devices = self._ensure_devices(fp) if not self._cancel.is_set() else self._devices
        dev = devices or {}
        machine = O.Machine(
            cpu=str(hw.get("cpu_name") or ""), physical_cores=int(hw.get("physical_cores") or 0),
            ram_total_mb=int(hw.get("ram_total_mb") or 0), ram_free_mb=int(hw.get("ram_free_mb") or 0),
            mem_mts=dev.get("mem_mts"), gpu=gf, igpu=str(dev.get("igpu") or ""), npu=str(dev.get("npu") or ""),
            devices_known=bool(devices), ct2_cuda=bool(dev.get("ct2_cuda")), sherpa_ready=bool(dev.get("sherpa_ready")),
            runtime=runtime_label, guard_floor_mb=int(getattr(cfg.llm, "vram_guard_free_mb", 1500) or 1500),
        )

        models = self._models()
        loaded = str(_call(e, "loaded_model", default="") or "")
        key = model_key or loaded or cfg.llm.model
        match = next((m for m in models if m.key == key), None)
        mf = O.model_facts_from_meta(match.key, match.size_bytes, match.meta) if match is not None else None
        profile = self._profile_dict(key) if match is not None else None
        launch = O.effective_launch(cfg.llm, profile) if match is not None else None
        server = getattr(e, "server", None)
        running = bool(server is not None and _call(server, "running", default=False) and loaded == key)
        backend = self._backend()

        tr = getattr(e, "transcriber", None)
        asr_now = _asr_label(str(_call(tr, "engine_used", default="") or ""), str(_call(tr, "device_used", default="") or ""),
                             str(_call(tr, "compute_type_used", default="") or "")) if tr is not None else ""
        ocr_now = ""
        try:
            from exocortex.screen.ocr import resolve_ocr_device

            ocr_now = str(resolve_ocr_device(cfg.screen.ocr_device)[0])
        except Exception:  # noqa: BLE001
            ocr_now = ""
        cf = O.cfg_facts_from_config(cfg, asr_now=asr_now, ocr_now=ocr_now)
        evidence = O.Evidence()
        try:
            evidence = self._get_store().evidence(fp, self._now())
        except Exception:  # noqa: BLE001
            log.debug("optimize: evidence load failed", exc_info=True)

        inp = O.PlanInputs(settings=settings, machine=machine, model=mf, launch=launch, running=running,
                           backend=str(backend), profile=profile, cfg=cf, evidence=evidence, now_epoch=self._now())
        facts: dict[str, Any] = {
            "cpu": machine.cpu, "physical_cores": machine.physical_cores, "ram_total_mb": machine.ram_total_mb,
            "ram_free_mb": machine.ram_free_mb, "mem_mts": machine.mem_mts,
            "gpu": dataclasses.asdict(gf) if gf else {},
            "vram": {"total_mb": gf.total_mb if gf else None, "free_mb": gf.free_mb if gf else None,
                     "others_mb": baseline_mb, "guard_floor_mb": machine.guard_floor_mb},
            "igpu": machine.igpu, "npu": machine.npu, "devices": sorted((dev.get("openvino") or {}).keys()),
            "devices_known": machine.devices_known, "runtime": runtime_label, "power": power_d or {},
            "ct2_cuda": machine.ct2_cuda, "sherpa_ready": machine.sherpa_ready,
        }
        model_rows = [{"key": m.key, "size_gb": round(m.size_bytes / 1e9, 2),
                       "ctx_train": int(getattr(m.meta, "n_ctx_train", 0) or 0) if getattr(m, "meta", None) else 0}
                      for m in models]
        measured = self._measured(evidence)
        engine_block = {"ready": bool(_call(getattr(e, "_llm_ready", None), "is_set", default=False)), "backend": str(backend),
                        "model": loaded, "swap_resident": swap}
        return inp, facts, model_rows, {"measured": measured, "engine": engine_block}

    @staticmethod
    def _measured(ev: O.Evidence) -> dict[str, Any]:
        probes = [r for r in ev.llm if r.source in ("probe", "apply") and r.pp]
        llm = max(probes, key=lambda r: r.at) if probes else None
        asr = [O.to_dict(r) for r in ev.asr if r.source == "probe"]
        ocr = [O.to_dict(r) for r in ev.ocr if r.source == "probe"]
        at = max([llm.at] if llm else [] + [r["at"] for r in asr], default="")
        return {"llm": O.to_dict(llm) if llm else None, "asr": asr, "ocr": ocr, "at": at,
                "stale": bool(llm.stale) if llm else False}

    # ------------------------------------------------------------------ analyze --------------
    def analyze(self, goal: str = "", options: dict[str, object] | None = None, model_key: str = "") -> None:
        """Recompute every goal's plan on the ``optimize-analyze`` thread. A request arriving while
        one runs is kept (latest wins) and run right after."""
        with self._lock:
            self._pending_analyze = (goal, dict(options or {}), model_key)
        if not self._analyze_lock.acquire(blocking=False):
            return
        t = threading.Thread(target=self._analyze_loop, name="optimize-analyze", daemon=True)
        self._threads.append(t)
        self._update(analysis={"state": "running", "message": "Analysing this machine..."})
        t.start()

    def _analyze_loop(self) -> None:
        try:
            while True:
                with self._lock:
                    req = self._pending_analyze
                    self._pending_analyze = None
                if req is None:
                    break
                goal, options, model_key = req
                try:
                    self._analyze_once(goal, options, model_key)
                except Exception as exc:  # noqa: BLE001
                    log.exception("optimize: analysis failed")
                    self._update(analysis={"state": "error", "message": f"analysis failed: {exc}"})
                    self._publish("error", step="facts", detail=str(exc))
                if self._cancel.is_set() or self._stopped():
                    break
        finally:
            self._analyze_lock.release()

    def _settings_from(self, goal: str, options: dict | None) -> O.Settings:
        base = O.settings_from_config(self._cfg)
        o = options or {}
        try:
            return O.Settings(
                goal=str(goal or o.get("goal") or base.goal),
                reserve_vram_mb=int(o.get("reserve_vram_mb", base.reserve_vram_mb) or 0),
                ctx_cap=int(o.get("ctx_cap", base.ctx_cap) or 0),
                asr_gpu=str(o.get("asr_gpu", base.asr_gpu) or "auto"),
            )
        except (TypeError, ValueError):
            return dataclasses.replace(base, goal=str(goal or base.goal))

    def _analyze_once(self, goal: str, options: dict | None, model_key: str) -> None:
        settings = self._settings_from(goal, options)
        inp, facts, models, extra = self._gather(settings, model_key)
        plans = O.plan_all(inp)
        with self._lock:
            for p in plans.values():
                self._plans[p.digest] = p
                self._plan_order.append(p.digest)
            while len(self._plan_order) > PLAN_CACHE * 4:
                old = self._plan_order.pop(0)
                if old not in self._plan_order:
                    self._plans.pop(old, None)
        self._analysis_at = self._now()
        act = self._activity()
        self._update(
            facts=facts, models=models, plans={g: p.to_dict() for g, p in plans.items()}, selected=settings.goal,
            gates=self._gates(act), analysis={"state": "done", "message": ""},
            measured=extra["measured"], engine=extra["engine"],
            apply={"undo": self._can_undo()},
        )
        self._publish("analysis", detail=settings.goal)

    # ------------------------------------------------------------------ benchmark -----------
    def benchmark(self, scopes: list[str] | None = None, force: bool = False) -> bool:
        scopes = [s for s in (scopes or ["llm", "speech"]) if s in ("llm", "speech")]
        if not scopes:
            return False
        act = self._activity()
        if not any(O.gate(s, act, force=force)[0] for s in scopes):
            reason = "; ".join(f"{s}: {O.gate(s, act, force=force)[1]}" for s in scopes)
            self._update(run={"state": "idle", "message": reason}, gates=self._gates(act))
            return False
        if not self._busy.acquire(blocking=False):
            self._update(run={"message": "a benchmark or apply is already running"})
            return False
        self._cancel.clear()
        self._run_id += 1
        rid = self._run_id
        self._update(run={"state": "running", "run_id": rid, "step": "facts", "fraction": 0.0, "message": "starting"})
        t = threading.Thread(target=self._bench_run, args=(rid, scopes, force), name="optimize-bench", daemon=True)
        self._threads.append(t)
        t.start()
        return True

    def cancel(self) -> None:
        self._cancel.set()
        for proc in list(self._procs):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def _progress(self, rid: int, step: str, fraction: float, message: str = "") -> None:
        self._update(run={"state": "running", "run_id": rid, "step": step, "fraction": round(fraction, 3), "message": message})
        self._publish("progress", run_id=rid, step=step, fraction=fraction, detail=message)

    def _bench_run(self, rid: int, scopes: list[str], force: bool) -> None:
        ob = self._bench()
        store = self._get_store()
        partial = False
        try:
            self._publish("started", run_id=rid)
            settings = O.settings_from_config(self._cfg)
            inp, facts, models, extra = self._gather(settings, "")
            self._update(facts=facts, models=models)
            total_steps = len(scopes)
            done_steps = 0
            fp = None
            try:
                fp = ob.fingerprint(_call(self._engine, "hardware_report", default={}) or {}, inp.machine.runtime)
                store.set_fingerprint(fp)
            except Exception:  # noqa: BLE001
                pass

            if "llm" in scopes and not self._cancel.is_set():
                ok, reason = O.gate("llm", self._activity(), force=force)
                if ok and inp.launch is not None and inp.model is not None:
                    self._progress(rid, "llm", 0.05, "measuring the running language model")
                    try:
                        self._probe_llm_step(rid, inp, store)
                    except Exception as exc:  # noqa: BLE001
                        log.exception("optimize: llm probe failed")
                        partial = True
                        self._progress(rid, "llm", 0.45, f"language model probe failed: {exc}")
                else:
                    partial = True
                    self._progress(rid, "llm", 0.45, f"language model skipped: {reason or 'no model loaded'}")
                done_steps += 1

            if "speech" in scopes and not self._cancel.is_set():
                ok, reason = O.gate("speech", self._activity(), force=force)
                if ok:
                    try:
                        self._probe_speech_step(rid, inp, store, done_steps / total_steps)
                    except Exception as exc:  # noqa: BLE001
                        log.exception("optimize: speech probe failed")
                        partial = True
                        self._progress(rid, "speech", 0.95, f"speech probe failed: {exc}")
                else:
                    partial = True
                    self._progress(rid, "speech", 0.95, f"speech skipped: {reason}")
                done_steps += 1

            if self._cancel.is_set():
                self._update(run={"state": "cancelled", "run_id": rid, "fraction": 1.0, "message": "cancelled"})
                self._publish("cancelled", run_id=rid)
            else:
                msg = "done (some steps were skipped)" if partial else "done"
                self._update(run={"state": "done", "run_id": rid, "step": "", "fraction": 1.0, "message": msg})
                self._publish("done", run_id=rid, detail=msg)
            try:
                self._analyze_once(self._cfg.optimize.goal, None, "")
            except Exception:  # noqa: BLE001
                log.exception("optimize: re-plan after the benchmark failed")
        except Exception as exc:  # noqa: BLE001
            log.exception("optimize: benchmark run failed")
            self._update(run={"state": "error", "run_id": rid, "message": f"benchmark failed: {exc}"})
            self._publish("error", run_id=rid, detail=str(exc))
        finally:
            self._busy.release()

    def _base_url(self) -> str:
        server = getattr(self._engine, "server", None)
        url = _call(server, "base_url", default="") if server is not None else ""
        if not url:
            url = _call(getattr(self._engine, "client", None), "base_url", default="") or ""
        return str(url)

    def _probe_llm_step(self, rid: int, inp: O.PlanInputs, store: Any) -> None:
        from exocortex import gpu

        ob = self._bench()
        launch, m = inp.launch, inp.model
        assert launch is not None and m is not None
        result = ob.probe_llm(self._base_url(), n_ctx=launch.ctx, rounds=2, persona=self._cfg.assist.persona,
                              style=self._cfg.assist.style, cancel=self._cancel.is_set,
                              progress=lambda msg, frac: self._progress(rid, "llm", 0.05 + 0.4 * float(frac or 0.0), str(msg)))
        if result.get("error"):
            raise RuntimeError(str(result["error"]))
        app_mb: float | None = None
        total, free = gpu.vram_total_mb(), gpu.vram_free_mb()
        if self._baseline and total is not None and free is not None:
            own = (int(total) - int(free)) - self._baseline[1]
            if 1 < own < 0.95 * int(total):
                app_mb = float(own)
        at = ob.iso_now(self._now())
        run = O.LlmRun(model=m.key, ctx=int(result.get("n_ctx") or launch.ctx), kv=launch.kv, parallel=launch.parallel,
                       ubatch=launch.ubatch, flags=launch.flags, pp=result.get("pp"), tg=result.get("tg"),
                       cached_ttft_ms=result.get("cached_ttft_ms"), incr_prompt_n=result.get("incr_prompt_n"),
                       spread=float(result.get("spread") or 0.0), gpu_used_mb=app_mb, ok=True,
                       clean_log=True, at=at, source="probe")
        store.add_llm(run)
        if app_mb is not None:
            store.add_footprint(O.Footprint(model=m.key, ctx=run.ctx, kv=launch.kv, ubatch=launch.ubatch, parallel=launch.parallel,
                                            flags=launch.flags, app_mb=app_mb, source="probe", at=at))
        self._progress(rid, "llm", 0.45, f"language model: {run.pp:.0f} tok/s read, {run.tg:.1f} tok/s written" if run.pp and run.tg else "language model measured")

    def _speech_variants(self, inp: O.PlanInputs) -> list[str]:
        cfg, machine = inp.cfg, inp.machine
        variants: list[str] = []
        now = cfg.asr_now
        if now in self._bench().ASR_VARIANTS:
            variants.append(now)
        if machine.sherpa_ready and "sherpa-cpu" not in variants:
            variants.append("sherpa-cpu")
        gpu = machine.gpu
        if machine.ct2_cuda and inp.settings.asr_gpu != "never" and gpu is not None and gpu.free_mb is not None \
                and gpu.free_mb - WHISPER_CUDA_MB - 512 >= machine.guard_floor_mb and "whisper-cuda-int8_float16" not in variants:
            variants.append("whisper-cuda-int8_float16")
        return variants

    def _probe_speech_step(self, rid: int, inp: O.PlanInputs, store: Any, base_fraction: float) -> None:
        from exocortex import gpu

        ob = self._bench()
        if not CLIP.is_file():
            self._progress(rid, "speech", 0.95, "speech skipped: the test clip is not installed")
            return
        variants = self._speech_variants(inp)
        if not variants:
            self._progress(rid, "speech", 0.95, "speech skipped: no engine to measure")
            return
        free0 = gpu.vram_free_mb()

        def still_ok() -> bool:
            act = self._activity()
            if act.speech_idle_s < 3.0:
                return False
            f = gpu.vram_free_mb()
            return not (free0 is not None and f is not None and free0 - f > VRAM_COLLAPSE_MB)

        out_dir = Path(store.path).parent if hasattr(store, "path") else project_root() / "models"
        threads = int(inp.cfg.asr_threads or 4)
        for i, variant in enumerate(variants):
            if self._cancel.is_set():
                return
            frac = 0.5 + 0.45 * (i / max(1, len(variants)))
            self._progress(rid, "speech", frac, f"measuring speech: {variant}")
            result = ob.run_child(
                ["probe-asr", "--variant", variant, "--models-dir", str(self._cfg.models_dir()), "--clip", str(CLIP),
                 "--threads", str(threads), "--language", self._cfg.asr.language, "--sherpa-model", self._cfg.asr.sherpa_model],
                out_path=out_dir / f"optimize_asr_{variant}.json", deadline_s=ASR_CHILD_DEADLINE_S,
                cancel=self._cancel.is_set, still_ok=still_ok, procs=self._procs,
            )
            if not result or result.get("error"):
                reason = (result or {}).get("error", "no result")
                if reason == "cancelled":
                    return
                log.info("optimize: speech probe %s failed: %s", variant, reason)
                continue
            try:
                row = O.AsrRow(label=str(result.get("label") or variant), engine=str(result.get("engine") or ""),
                               device=str(result.get("device") or ""), compute=str(result.get("compute_type") or ""),
                               threads=int(result.get("threads") or 0), median_s=float(result.get("median_s") or 0.0),
                               overlap=result.get("overlap"), vram_mb=result.get("vram_mb"),
                               vram_source=str(result.get("vram_source") or "none"), at=ob.iso_now(self._now()), source="probe")
                if row.median_s > 0:
                    store.add_asr(row)
            except (TypeError, ValueError):
                log.info("optimize: speech probe %s returned an unusable result", variant)

    # ------------------------------------------------------------------ apply ---------------
    def _write_profile(self, key: str, values: dict | None) -> None:
        from exocortex.llm import library

        if values is None:
            library.set_profile(self._cfg.llm, key, None)
            return
        known = {f.name for f in dataclasses.fields(library.ModelProfile)}
        library.set_profile(self._cfg.llm, key, library.ModelProfile(**{k: v for k, v in values.items() if k in known}))

    def _journal_add(self, entry: dict) -> list[dict]:
        store = self._get_store()
        entries = store.journal()
        entries.append(entry)
        store.journal_write(entries)
        return entries

    def _journal_set(self, entry_id: str, state: str, detail: str = "") -> None:
        store = self._get_store()
        entries = store.journal()
        for e in entries:
            if e.get("id") == entry_id:
                e["state"] = state
                e["detail"] = detail
        store.journal_write(entries)
        self._update(apply={"undo": self._can_undo(entries)})

    def _can_undo(self, entries: list[dict] | None = None) -> bool:
        """True while the newest VERIFIED journal entry is an apply (an undo that followed it has
        already put the previous profile back; a rolled-back apply changed nothing)."""
        try:
            entries = entries if entries is not None else self._get_store().journal()
        except Exception:  # noqa: BLE001
            return False
        for e in reversed(entries):
            if e.get("state") == "verified":
                return e.get("kind", "apply") == "apply"
        return False

    def apply(self, digest: str) -> bool:
        with self._lock:
            plan = self._plans.get(digest)
        if plan is None:
            self._update(apply={"state": "refused", "message": "the plan changed; analyse again", "digest": digest})
            return False
        act = self._activity()
        ok, reason = O.gate("apply", act)
        if not ok:
            self._update(apply={"state": "refused", "message": reason, "digest": digest})
            return False
        from exocortex import gpu

        loaded = str(_call(self._engine, "loaded_model", default="") or "")
        # a plan for a model that is NOT loaded is only saved (no relaunch), so "a different model
        # is loaded" is not a reason to refuse it; every other check still applies
        fresh = O.FreshFacts(free_mb=_call(gpu, "vram_free_mb"), loaded_key=loaded if loaded == plan.model_key else "",
                             swap_resident=act.swap_resident, now_epoch=self._now())
        reasons = O.validate_plan(plan, fresh)
        if reasons:
            self._update(apply={"state": "refused", "message": "; ".join(reasons), "digest": digest})
            return False
        if not plan.profile:
            self._update(apply={"state": "verified", "message": "no model change in this plan", "digest": digest})
            return True
        return self._transaction(plan.model_key, dict(plan.profile), digest=digest, goal=plan.goal, kind="apply")

    def undo(self) -> bool:
        try:
            entries = self._get_store().journal()
        except Exception:  # noqa: BLE001
            return False
        last = next((e for e in reversed(entries) if e.get("state") == "verified"), None)
        if last is not None and last.get("kind", "apply") != "apply":
            last = None                                   # the newest verified change IS an undo: nothing further to undo
        if last is None:
            self._update(apply={"state": "refused", "message": "nothing to undo"})
            return False
        key = str(last.get("key") or "")
        previous = last.get("previous")
        ok, reason = O.gate("apply", self._activity())
        if not ok:
            self._update(apply={"state": "refused", "message": reason})
            return False
        return self._transaction(key, previous, digest=str(last.get("digest") or ""), goal=str(last.get("goal") or ""),
                                 kind="undo", replace=True, undo_of=str(last.get("id") or ""))

    def _transaction(self, key: str, changes: dict | None, *, digest: str, goal: str, kind: str,
                     replace: bool = False, undo_of: str = "") -> bool:
        """Journal -> write the profile -> relaunch (if that model is loaded) -> verify -> journal
        ``verified``; any failure restores the previous profile, relaunches again and journals
        ``rolled-back``. ``replace=True`` writes ``changes`` as the whole profile (undo); otherwise
        they are merged into the saved one."""
        if not self._busy.acquire(blocking=False):
            self._update(apply={"state": "refused", "message": "a benchmark or apply is already running", "digest": digest})
            return False
        released = False
        try:
            previous = self._profile_dict(key)
            if replace:
                new = dict(changes) if changes is not None else None
            else:
                new = {**(previous or {}), **(changes or {})}
                new["auto"] = False
            entry_id = f"{int(self._now() * 1000)}-{kind}"
            entry = {"id": entry_id, "at": self._bench().iso_now(self._now()), "goal": goal, "key": key, "digest": digest,
                     "previous": previous, "new": new, "state": "pending", "detail": "", "kind": kind, "undo_of": undo_of}
            loaded = str(_call(self._engine, "loaded_model", default="") or "")
            if loaded != key:
                # the plan is for a model that is not loaded: save it, it takes effect when that model loads
                self._journal_add(entry)
                self._write_profile(key, new)
                self._journal_set(entry_id, "verified", "saved; takes effect when that model loads")
                self._update(apply={"state": "verified", "message": f"saved for {key}; takes effect when it loads", "digest": digest})
                self._publish("applied", detail=key)
                self._busy.release()
                released = True
                return True

            self._update(apply={"state": "running", "message": "applying: reloading the model...", "digest": digest, "undo": False})

            def work() -> None:
                try:
                    self._apply_work(entry, key, previous, new, digest, kind)
                finally:
                    self._busy.release()

            started = bool(_call(self._engine, "_start_swap", work, "optimize", default=False))
            if not started:
                self._update(apply={"state": "refused", "message": "a model change is still running; try again", "digest": digest})
                self._busy.release()
                released = True
                return False
            released = True   # the worker releases
            return True
        except Exception as exc:  # noqa: BLE001
            log.exception("optimize: apply could not start")
            self._update(apply={"state": "error", "message": f"apply failed: {exc}", "digest": digest})
            return False
        finally:
            if not released:
                self._busy.release()

    def _apply_work(self, entry: dict, key: str, previous: dict | None, new: dict | None, digest: str, kind: str) -> None:
        """Runs on the llm-swap thread."""
        from exocortex import gpu

        ob = self._bench()
        entry_id = str(entry["id"])
        try:
            self._journal_add(entry)
            self._write_profile(key, new)
        except Exception as exc:  # noqa: BLE001
            log.exception("optimize: could not write the profile")
            self._update(apply={"state": "error", "message": f"could not write the profile: {exc}", "digest": digest})
            return
        err = _call(self._engine, "_relaunch_blocking", key, default="relaunch unavailable")
        failure = err
        if failure is None:
            failure = self._verify(key, new, kind)
        if failure is None:
            self._journal_set(entry_id, "verified", "")
            self._record_apply_footprint(key, new, ob)
            self._update(apply={"state": "verified", "message": "applied and verified", "digest": digest, "undo": kind == "apply"})
            self._publish("applied", step="apply", detail=key)
            return
        log.warning("optimize: apply of %s failed (%s); restoring the previous profile", key, failure)
        detail = str(failure)
        try:
            self._write_profile(key, previous)
            if not self._stopped():
                err2 = _call(self._engine, "_relaunch_blocking", key, default="relaunch unavailable")
                if err2:
                    detail += f"; restore failed too: {err2}"
                    self._toast("error", f"Optimize: could not restore the previous model setup ({err2}); open Models to fix it")
        except Exception as exc:  # noqa: BLE001
            log.exception("optimize: restore failed")
            detail += f"; restore failed too: {exc}"
        self._journal_set(entry_id, "rolled-back", detail)
        self._update(apply={"state": "rolled-back", "message": f"rolled back: {detail}", "digest": digest})
        self._publish("rolled-back", step="verify", detail=detail)
        self._toast("error", f"Optimize: the change was rolled back ({detail})")

    def _verify(self, key: str, new: dict | None, kind: str) -> str | None:
        """``None`` when the relaunched server passes every check, else the reason."""
        from exocortex import gpu

        ob = self._bench()
        if kind == "apply" and new and new.get("ctx_size"):
            got = _call(getattr(self._engine, "client", None), "context_window")
            if got is not None and int(got) != int(new["ctx_size"]):
                return f"the server came up with a {got}-token window instead of {new['ctx_size']}"
        ok, ms = ob.canary(self._base_url(), timeout_s=CANARY_TIMEOUT_S)
        if not ok:
            return "the model did not answer a one-token test within 5 s"
        free, _stable = ob.sample_vram(n=VERIFY_VRAM_SAMPLES, gap_s=0.5)
        floor = int(getattr(self._cfg.llm, "vram_guard_free_mb", 1500) or 0)
        if free is not None and free < floor:
            return f"only {free} MiB of GPU memory left free (the guard floor is {floor})"
        return None

    def _record_apply_footprint(self, key: str, new: dict | None, ob: Any) -> None:
        try:
            from exocortex import gpu

            total, free = gpu.vram_total_mb(), gpu.vram_free_mb()
            if not (self._baseline and total is not None and free is not None):
                return
            own = (int(total) - int(free)) - self._baseline[1]
            if not (1 < own < 0.95 * int(total)):
                return
            launch = O.effective_launch(self._cfg.llm, self._profile_dict(key))
            at = ob.iso_now(self._now())
            self._get_store().add_footprint(O.Footprint(model=key, ctx=launch.ctx, kv=launch.kv, ubatch=launch.ubatch,
                                                        parallel=launch.parallel, flags=launch.flags, app_mb=float(own),
                                                        source="apply", at=at))
        except Exception:  # noqa: BLE001
            log.debug("optimize: apply footprint not recorded", exc_info=True)
