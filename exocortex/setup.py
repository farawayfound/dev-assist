"""First-run setup: everything a fresh machine needs beyond ``pip install`` (2026-09-27).

``run.bat`` (and ``run.vbs`` through it) call ``python -m exocortex.setup`` on EVERY launch, right
after the pip step, so a single run on a new machine ends with a working app and no LM Studio or
Ollama anywhere in the picture. Four idempotent steps, each one line of console output when
nothing needs doing (~1 s in total once everything is present):

1. **The llama.cpp runtime** for this GPU -- :func:`exocortex.llm.server.find_server_binary` with an
   ``ensure`` callback: LM Studio's pack when one is installed (unchanged behaviour), otherwise the
   pinned official llama.cpp build for the vendor's ladder (CUDA 13.4 / 12.4, ROCm, Vulkan, CPU;
   :mod:`exocortex.llm.runtimes`), downloaded with a progress line.
2. **The hardware report** (:mod:`exocortex.hardware`) saved to ``<library>/hardware.json`` -- what
   the Models window's Hardware tab shows and what the auto-tuned model profiles are computed from.
3. **The speech models**: Silero VAD + the speaker-embedding model (:func:`exocortex.models_dl.ensure_models`),
   plus the ASR model the engine will resolve to -- the sherpa-onnx Parakeet bundle off CUDA, or the
   faster-whisper checkpoint (through faster-whisper's own Hugging Face cache) on an NVIDIA box.
   Without this the very first meeting would spend its opening minutes downloading.
4. **At least one LLM.** When the library (:mod:`exocortex.llm.library`, the app's own
   ``models/llm`` folder plus LM Studio's library when present) holds no model, the console asks
   which one to pull: a short list recommended for THIS machine's GPU budget (real sizes fetched
   from the Hub, the quant chosen by :func:`exocortex.llm.library.choose_quant`), a Hub search, or
   "not now". The chosen model is pulled whole (every shard, sha256-verified, resumable), gets an
   auto-tuned profile, and becomes ``[llm].model``. The prompt is skipped -- with a one-line hint
   to use the overlay's ``... > Models...`` window -- when stdin is not a console, under ``--quiet``
   (what run.vbs passes for its hidden catch-up runs) or ``EXO_SETUP_NONINTERACTIVE=1``; ``--yes``
   takes the first recommendation without asking (scripted installs).

Exit status: 0, except 1 when no llama.cpp runtime could be found or fetched at all (run.bat
prints a warning and still launches: the overlay transcribes without a model, and the engine
retries the fetch at start). ``--dry-run`` prints what each step would do and touches nothing.
Every network- or disk-heavy step is injectable (see :func:`run_setup`) so tests run offline.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from exocortex.config import Config, load_config, project_root, save_config

log = logging.getLogger(__name__)

_PREFIX = "[ExoCortex]"
_DRY = "[dryrun]"


@dataclass(frozen=True)
class Recommendation:
    """One curated Hub repo: shown when it fits the machine's GPU budget (``min_budget_mb`` -- the
    smallest budget its smallest sensible quant still fits with the 0.75 margin
    :func:`exocortex.llm.library.choose_quant` keeps; 0 = fine on a CPU too)."""

    repo_id: str
    why: str
    min_budget_mb: int


#: Curated 2026-09-27 from the Hub's most-downloaded GGUF repos for each size class (sizes checked
#: live: gemma-4-12B QAT Q4_0 6.98 GB, gemma-4-12b Q4_K_M 7.12 GB, Qwen3.8-9B Q4_K_M 5.78 GB,
#: Qwen3.8-27B UD-Q4_K_M 16.46 GB, gemma-4-E4B Q4_K_M 5.34 GB, Qwen3.8-4B Q4_K_M 2.78 GB,
#: gemma-4-E2B Q4_K_M 3.11 GB). Order = preference within a band; :func:`recommendations_for`
#: keeps the ones whose min_budget_mb the machine clears, best first. Apache-2.0 / Gemma terms,
#: none gated.
RECOMMENDED: tuple[Recommendation, ...] = (
    Recommendation("unsloth/Qwen3.8-27B-GGUF",
                   "27B dense Qwen: the strongest general model that fits a 24 GB card at Q4", 22_528),
    Recommendation("lmstudio-community/gemma-4-12B-it-QAT-GGUF",
                   "12B Gemma, quantization-aware Q4_0 (7 GB): the model this app is tuned on", 9_728),
    Recommendation("unsloth/gemma-4-12b-it-GGUF",
                   "12B Gemma with every quant from IQ2 to Q8, for a tighter or roomier budget", 6_144),
    Recommendation("empero-ai/Qwen3.8-9B-Distill-GGUF",
                   "9B Qwen distilled from the 27B: strong on code, 5.8 GB at Q4_K_M", 8_192),
    Recommendation("lmstudio-community/gemma-4-E4B-it-GGUF",
                   "Gemma 4 E4B (5.3 GB at Q4_K_M): the small-GPU pick", 7_168),
    Recommendation("empero-ai/Qwen3.8-4B-Distill-GGUF",
                   "4B Qwen (2.8 GB at Q4_K_M): fast on a small GPU or the CPU alone", 0),
    Recommendation("unsloth/gemma-4-E2B-it-GGUF",
                   "Gemma 4 E2B (3.1 GB at Q4_K_M): the lightest reasonable choice", 0),
)


def recommendations_for(budget_mb: int | None, *, limit: int = 4) -> list[Recommendation]:
    """The curated repos this machine can run, best first (a ``None`` budget = no GPU: the
    CPU-capable entries only)."""
    budget = budget_mb if budget_mb is not None else 0
    out = [r for r in RECOMMENDED if r.min_budget_mb <= budget]
    return out[:limit]


@dataclass
class SetupResult:
    runtime: str = ""        # one line: which llama.cpp binary, or why none
    hardware: str = ""       # where the report went
    speech: str = ""         # what was checked / downloaded
    model: str = ""          # the outcome of step 4
    rc: int = 0
    notes: list[str] = field(default_factory=list)


# ---- console helpers -------------------------------------------------------------------------
def _fmt_gb(n: int | float) -> str:
    return f"{n / 1e9:.1f} GB"


class _Progress:
    """A single, in-place console progress line (``\\r``), throttled to ~5 updates a second."""

    def __init__(self, out: Callable[[str], None], *, label: str) -> None:
        self._out = out
        self._label = label
        self._last = 0.0
        self._t0 = time.monotonic()
        self._shown = False

    def __call__(self, name: str, done: int, total: int) -> None:
        now = time.monotonic()
        if now - self._last < 0.2 and done < total:
            return
        self._last = now
        pct = f"{done * 100 // total:3d}%" if total else "   ?"
        rate = done / max(now - self._t0, 1e-6)
        line = f"\r{_PREFIX} {self._label} {name} {pct} ({_fmt_gb(done)}/{_fmt_gb(total)}, {rate / 1e6:.0f} MB/s)   "
        self._out(line)
        self._shown = True

    def finish(self) -> None:
        if self._shown:
            self._out("\n")


def _stdout_write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _say(out: Callable[[str], None], text: str) -> None:
    out(f"{_PREFIX} {text}\n")


def _interactive(stdin_is_tty: bool | None) -> bool:
    if os.environ.get("EXO_SETUP_NONINTERACTIVE") == "1":
        return False
    if stdin_is_tty is not None:
        return stdin_is_tty
    try:
        return sys.stdin is not None and sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


# ---- step 1: the llama.cpp runtime -----------------------------------------------------------
def _ensure_runtime(cfg: Config, *, dry_run: bool, out: Callable[[str], None],
                    find_binary: Callable[..., Any] | None = None) -> tuple[str, bool]:
    """(one-line description, ok). Only fetches through ``ensure``; never raises."""
    from exocortex import gpu
    from exocortex.llm import runtimes, server

    find = find_binary or server.find_server_binary
    vendor = gpu.gpu_vendor()
    root = Path(cfg.llm.runtime_dir) if getattr(cfg.llm, "runtime_dir", "") else None
    bar = _Progress(out, label="downloading")

    def _ensure(kind: str, upstream_root: Path, build: str) -> Path:
        if dry_run:
            raise runtimes.UpstreamRuntimeError(f"{_DRY} would fetch the official llama.cpp build {build} ({kind})")
        _say(out, f"no llama.cpp runtime for this {vendor} GPU is installed; fetching the official build ({kind} {build})")
        try:
            return runtimes.ensure_runtime(kind, upstream_root, build=build, progress=bar)
        finally:
            bar.finish()

    kwargs: dict[str, Any] = dict(vendor=vendor, backend=cfg.llm.llama_backend, source=cfg.llm.llama_source,
                                  upstream_build=cfg.llm.llama_upstream_build, ensure=_ensure)
    if root is not None:
        kwargs["upstream_root"] = root
    try:
        try:
            binary = find(cfg.llm.llama_server_path, **kwargs)
        except TypeError:
            # an older server.find_server_binary without the newer keyword arguments
            binary = find(cfg.llm.llama_server_path, vendor=vendor, backend=cfg.llm.llama_backend, ensure=_ensure)
    except Exception as exc:  # noqa: BLE001 - LlamaServerError or a download failure
        text = str(exc)
        if _DRY in text:
            return text.split(_DRY, 1)[1].strip(), True
        return f"no llama.cpp runtime: {text}", False
    source = getattr(binary, "source", "lmstudio")
    return f"llama.cpp {binary.flavor} {binary.version} ({source}): {binary.exe}", True


# ---- step 2: the hardware report -------------------------------------------------------------
def _hardware_report(cfg: Config, *, dry_run: bool, out: Callable[[str], None],
                     collect: Callable[..., Any] | None = None) -> tuple[str, dict[str, Any] | None]:
    try:
        from exocortex import hardware
    except ImportError as exc:  # pragma: no cover - only while the module is missing
        return f"hardware report unavailable ({exc})", None
    fn = collect or hardware.collect
    try:
        report = fn(cfg.llm, probe_runtime=True)
    except Exception as exc:  # noqa: BLE001
        return f"hardware report failed: {exc}", None
    out(hardware.describe(report) + "\n")
    path = hardware.report_path(cfg.llm)
    if dry_run:
        return f"{_DRY} would write {path}", hardware.to_dict(report)
    try:
        hardware.save(report, path)
    except OSError as exc:
        return f"could not write {path}: {exc}", hardware.to_dict(report)
    return f"hardware report saved to {path}", hardware.to_dict(report)


# ---- step 3: the speech models ---------------------------------------------------------------
def _ensure_whisper(model_name: str, *, dry_run: bool) -> str:
    """faster-whisper's checkpoint through its own Hugging Face cache; a local CTranslate2
    directory needs nothing."""
    if Path(model_name).expanduser().is_dir():
        return f"whisper {model_name}: local directory"
    try:
        from faster_whisper import download_model
    except ImportError:
        return "faster-whisper not installed; skipped"
    try:
        download_model(model_name, local_files_only=True)
        return f"whisper {model_name}: cached"
    except Exception:  # noqa: BLE001 - not cached yet
        pass
    if dry_run:
        return f"{_DRY} would download whisper {model_name} (~1.6 GB)"
    download_model(model_name)
    return f"whisper {model_name}: downloaded"


def _ensure_speech(cfg: Config, *, dry_run: bool, out: Callable[[str], None],
                   ensure_models: Callable[..., Any] | None = None,
                   ensure_sherpa: Callable[..., Any] | None = None,
                   ensure_whisper: Callable[..., str] | None = None) -> str:
    from exocortex import models_dl
    from exocortex.speech import asr

    models_dir = cfg.models_dir()
    parts: list[str] = []
    missing = [name for name in models_dl.MODEL_URLS if not (models_dir / name).is_file()]
    if dry_run:
        parts.append(f"{_DRY} would download {', '.join(missing)}" if missing else "VAD + speaker models present")
    else:
        bar = _Progress(out, label="downloading")
        try:
            (ensure_models or models_dl.ensure_models)(models_dir, progress=bar)
        finally:
            bar.finish()
        parts.append(f"VAD + speaker models {'downloaded' if missing else 'present'}")
    try:
        engine = asr.resolve_engine(cfg.asr)
    except Exception as exc:  # noqa: BLE001
        parts.append(f"ASR engine unresolved ({exc})")
        return "; ".join(parts)
    if engine == "sherpa":
        name = asr.resolve_sherpa_model(cfg.asr)
        ready = models_dl._sherpa_model_ready(models_dir / name)
        if dry_run:
            parts.append(f"sherpa {name}: present" if ready else f"{_DRY} would download sherpa {name} (~0.5 GB)")
        else:
            bar = _Progress(out, label="downloading")
            try:
                (ensure_sherpa or models_dl.ensure_sherpa_model)(name, models_dir, progress=bar)
            finally:
                bar.finish()
            parts.append(f"sherpa {name}: {'present' if ready else 'downloaded'}")
    else:
        parts.append((ensure_whisper or _ensure_whisper)(cfg.asr.model, dry_run=dry_run))
    return "; ".join(parts)


# ---- step 4: at least one LLM ------------------------------------------------------------------
def _describe_gpu(hw: dict[str, Any] | None) -> str:
    if not hw:
        return "this machine"
    gpus = hw.get("gpus") or []
    idx = hw.get("primary_index")
    name = ""
    if isinstance(idx, int) and 0 <= idx < len(gpus):
        name = str(gpus[idx].get("name", ""))
    budget = hw.get("budget_mb")
    if name and budget:
        return f"{name} ({budget} MiB)"
    if budget:
        return f"a {hw.get('vendor', '')} GPU with {budget} MiB"
    return "this machine (no GPU: CPU only)"


def _pull(cfg: Config, repo_id: str, quant: str, *, budget_mb: int | None, policy: str,
          out: Callable[[str], None], pull: Callable[..., Any]) -> str:
    _say(out, f"pulling {repo_id} ({quant}) into {cfg.llm.library_dir or project_root() / 'models' / 'llm'} ...")
    bar = _Progress(out, label="pulling")
    try:
        model = pull(repo_id, quant, cfg.llm, budget_mb=budget_mb, policy=policy, progress=bar)
    finally:
        bar.finish()
    cfg.llm.model = model.key
    path = cfg.path or (project_root() / "config.toml")
    try:
        save_config(cfg, path)
        saved = f"; [llm].model = {model.key} saved to {path}"
    except OSError as exc:
        saved = f"; could not save config ({exc})"
    return f"pulled {model.key} ({_fmt_gb(model.size_bytes)}){saved}"


def _ensure_model(cfg: Config, hw: dict[str, Any] | None, *, dry_run: bool, quiet: bool, yes: bool,
                  interactive: bool, out: Callable[[str], None], ask: Callable[[str], str],
                  scan: Callable[..., Any] | None = None, repo_info: Callable[..., Any] | None = None,
                  search: Callable[..., Any] | None = None, pull: Callable[..., Any] | None = None) -> str:
    from exocortex.llm import library

    scan_fn = scan or library.scan_library
    try:
        models = scan_fn(cfg.llm)
    except Exception as exc:  # noqa: BLE001
        return f"could not scan the model library: {exc}"
    if models:
        keys = [m.key for m in models]
        wanted = (cfg.llm.model or "").strip()
        if wanted and wanted.lower() != "auto" and not any(
                wanted.lower() in (k.lower(), k.split("/")[-1].lower()) for k in keys):
            return (f"{len(models)} model(s) in the library; [llm].model = {wanted!r} is not one of them "
                    f"-- the app will pick automatically (or choose in ... > Models...)")
        return f"{len(models)} model(s) in the library" + (f"; default {wanted}" if wanted else "")

    hint = "no LLM in the library yet -- open the overlay's ... > Models... window to pull one"
    if dry_run:
        return f"{_DRY} would offer to pull a model (library empty)"
    if quiet or not (interactive or yes):
        return hint + (" (prompt skipped: not an interactive console)" if not quiet else " (prompt skipped: --quiet)")

    budget = hw.get("budget_mb") if hw else None
    policy = str((hw or {}).get("recommended_policy") or "auto")
    info_fn = repo_info or library.hf_repo_info
    search_fn = search or library.hf_search
    pull_fn = pull or library.pull_model

    _say(out, f"No LLM model in the library yet ({library.library_dir(cfg.llm)}).")
    _say(out, f"Recommended for {_describe_gpu(hw)}:")
    options: list[tuple[str, str, int]] = []      # (repo, quant, size)
    for rec in recommendations_for(budget):
        try:
            info = info_fn(rec.repo_id)
            choice = library.choose_quant(info.quants, budget_mb=budget, policy=policy)
        except Exception as exc:  # noqa: BLE001 - offline, rate-limited, a moved repo
            log.debug("setup: %s unavailable: %s", rec.repo_id, exc)
            continue
        if choice is None:
            continue
        options.append((rec.repo_id, choice.quant, choice.size_bytes))
        out(f"  [{len(options)}] {rec.repo_id:<46} {choice.quant:<10} {_fmt_gb(choice.size_bytes):>8}   {rec.why}\n")
    if not options and not yes:
        out("  (the Hugging Face Hub could not be reached: no recommendations)\n")
    out("  [S] search the Hugging Face Hub by name\n")
    out("  [N] not now (the app opens its Models window instead)\n")

    if yes:
        if not options:
            return hint + " (--yes: the Hub was unreachable)"
        repo, quant, _size = options[0]
        return _pull(cfg, repo, quant, budget_mb=budget, policy=policy, out=out, pull=pull_fn)

    while True:
        answer = ask("Pull which one? [1]: ").strip().lower()
        if answer == "":
            answer = "1"
        if answer in ("n", "no", "q", "quit", "later"):
            return hint + " (declined)"
        if answer == "s":
            query = ask("Search the Hub for: ").strip()
            if not query:
                continue
            try:
                found = search_fn(query, limit=10)
            except Exception as exc:  # noqa: BLE001
                out(f"  search failed: {exc}\n")
                continue
            if not found:
                out("  nothing found\n")
                continue
            for i, repo in enumerate(found, 1):
                out(f"  [{i}] {repo.id:<56} {repo.downloads:>9} downloads\n")
            pick = ask("Which one? [1]: ").strip() or "1"
            if not pick.isdigit() or not 1 <= int(pick) <= len(found):
                continue
            repo_id = found[int(pick) - 1].id
            try:
                info = info_fn(repo_id)
                choice = library.choose_quant(info.quants, budget_mb=budget, policy=policy)
            except Exception as exc:  # noqa: BLE001
                out(f"  could not read {repo_id}: {exc}\n")
                continue
            if choice is None:
                out(f"  {repo_id} has no GGUF quant that fits this machine\n")
                continue
            return _pull(cfg, repo_id, choice.quant, budget_mb=budget, policy=policy, out=out, pull=pull_fn)
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            repo, quant, _size = options[int(answer) - 1]
            return _pull(cfg, repo, quant, budget_mb=budget, policy=policy, out=out, pull=pull_fn)
        out("  please answer with a number, S or N\n")


# ---- the whole thing -------------------------------------------------------------------------
def run_setup(cfg: Config, *, quiet: bool = False, yes: bool = False, dry_run: bool = False,
              do_runtime: bool = True, do_speech: bool = True, do_model: bool = True,
              out: Callable[[str], None] = _stdout_write, stdin_is_tty: bool | None = None,
              ask: Callable[[str], str] = input, **probes: Any) -> SetupResult:
    """Run the four steps; ``probes`` are the injectable network/disk functions of each step
    (``find_binary``, ``collect``, ``ensure_models``, ``ensure_sherpa``, ``ensure_whisper``,
    ``scan``, ``repo_info``, ``search``, ``pull``)."""
    result = SetupResult()
    interactive = _interactive(stdin_is_tty)

    if do_runtime:
        result.runtime, ok = _ensure_runtime(cfg, dry_run=dry_run, out=out, find_binary=probes.get("find_binary"))
        if not ok:
            result.rc = 1
        _say(out, f"runtime: {result.runtime}")

    result.hardware, hw = _hardware_report(cfg, dry_run=dry_run, out=out, collect=probes.get("collect"))
    _say(out, result.hardware)

    if do_speech:
        try:
            result.speech = _ensure_speech(cfg, dry_run=dry_run, out=out, ensure_models=probes.get("ensure_models"),
                                           ensure_sherpa=probes.get("ensure_sherpa"),
                                           ensure_whisper=probes.get("ensure_whisper"))
        except Exception as exc:  # noqa: BLE001 - offline: the engine retries at start
            result.speech = f"speech models: {exc} (the app retries at start)"
            result.notes.append(result.speech)
        _say(out, f"speech: {result.speech}")

    if do_model:
        try:
            result.model = _ensure_model(cfg, hw, dry_run=dry_run, quiet=quiet, yes=yes, interactive=interactive,
                                         out=out, ask=ask, scan=probes.get("scan"), repo_info=probes.get("repo_info"),
                                         search=probes.get("search"), pull=probes.get("pull"))
        except KeyboardInterrupt:
            result.model = "model pull interrupted (partial files are kept; the next run resumes)"
        except Exception as exc:  # noqa: BLE001
            log.exception("setup: model step failed")
            result.model = f"model pull failed: {exc}"
            result.notes.append(result.model)
        _say(out, f"model: {result.model}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m exocortex.setup",
                                     description="ExoCortex first-run setup: llama.cpp runtime, hardware report, "
                                                 "speech models, and at least one LLM.")
    parser.add_argument("--config", help="path to config.toml")
    parser.add_argument("--quiet", action="store_true", help="never prompt (run.vbs's hidden runs)")
    parser.add_argument("--yes", action="store_true", help="pull the first recommended model without asking")
    parser.add_argument("--dry-run", action="store_true", help="print what would be done; download nothing")
    parser.add_argument("--no-runtime", action="store_true", help="skip the llama.cpp runtime step")
    parser.add_argument("--no-speech", action="store_true", help="skip the speech-model step")
    parser.add_argument("--no-model", action="store_true", help="skip the LLM step")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    result = run_setup(cfg, quiet=args.quiet, yes=args.yes, dry_run=args.dry_run, do_runtime=not args.no_runtime,
                       do_speech=not args.no_speech, do_model=not args.no_model)
    return result.rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
