"""One hardware snapshot for the portable-install launcher: what GPU/CPU/RAM/disk this machine
has, which llama.cpp runtime pack it would actually get right now, and a recommended model-fit
policy -- everything the run script needs to print before it prompts the person to pull a model,
and everything :mod:`exocortex.llm.library`'s ``choose_quant``/``auto_profile`` need to size a
pull and a tuning profile for THIS machine, not a generic one.

2026-09-27 (portable-install): why this module exists -- before it, "does this machine have a
usable GPU runtime" was only answered by actually trying to spawn llama-server
(:func:`exocortex.llm.server.find_server_binary`) deep inside :class:`LlamaServerManager`, and
"how much RAM/VRAM is there" was scattered across :mod:`exocortex.gpu` calls made from a handful of
different call sites (the doctor, the server manager, the config defaults). A fresh machine with
no LM Studio pack and no upstream runtime yet needs to know all of this BEFORE the run script tries
to download anything -- to decide which runtime kind to fetch, how big a model quant fits, and what
to tell the person. :func:`collect` is that single, side-effect-free (besides the optional
``--save``) snapshot; every individual probe inside it is injectable and wrapped so a failure never
takes the whole report down -- it only adds a line to :attr:`HardwareReport.notes` and leaves that
one field ``None``, exactly the shape a launcher script wants (show what IS known, one line per
missing fact, and keep going).

:func:`collect` never downloads anything: its runtime probe is
``exocortex.llm.server.find_server_binary(..., ensure=None)``, i.e. "what would be picked right
now" -- an LM Studio pack, an already-installed upstream build, or neither (``runtime = None`` plus
a note naming what is missing). The run script decides whether to fetch a runtime from that.

``python -m exocortex.hardware`` prints :func:`describe`; ``--json`` prints :func:`to_dict`;
``--save`` additionally writes :func:`report_path` (under the model library directory, so it sits
next to the GGUFs it was measured for) via :func:`exocortex.config.load_config`.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import platform
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from exocortex import gpu
from exocortex.config import LLMConfig, project_root

log = logging.getLogger(__name__)


@dataclass
class HardwareReport:
    """Everything the portable-install launcher and :mod:`exocortex.llm.library` need to know
    about this machine. Every field can be ``None``/``[]``/``""`` -- a probe that failed or found
    nothing is recorded in :attr:`notes` instead of raising; this dataclass round-trips through
    :func:`to_dict`/:func:`save`/:func:`load` as plain JSON (no ``Path``/dataclass values inside
    ``gpus``/``runtime``, only dicts and scalars)."""

    os: str
    cpu_name: str
    physical_cores: int | None
    logical_cores: int | None
    ram_total_mb: int | None
    ram_free_mb: int | None
    gpus: list[dict]
    primary_index: int | None
    vendor: str
    budget_mb: int | None
    vram_total_mb: int | None
    vram_free_mb: int | None
    nvidia_driver: str | None
    library_dir: str
    library_free_mb: int | None
    runtime: dict | None
    recommended_policy: str
    generated_at: str
    notes: list[str] = field(default_factory=list)


def _resolve_library_dir(cfg: LLMConfig, notes: list[str]) -> Path:
    """:func:`exocortex.llm.library.library_dir` if that module is importable (it may not be yet --
    agents work on this branch in parallel), else its own fallback -- same default the real
    function documents (``project_root()/models/llm``). Imported lazily so this module's own
    import stays side-effect-free even while ``exocortex.llm.library`` does not exist yet."""
    try:
        # importlib (not "from exocortex.llm import library"): the package attribute is cached by
        # whichever import ran first, so only a sys.modules lookup honours a test's
        # sys.modules["exocortex.llm.library"] = None deterministically, whatever ran before it.
        import importlib

        library = importlib.import_module("exocortex.llm.library")
    except ImportError as exc:
        notes.append(f"exocortex.llm.library not importable yet ({exc}); using the default library path")
        return project_root() / "models" / "llm"
    try:
        return Path(library.library_dir(cfg))
    except Exception as exc:  # noqa: BLE001 - a bug in library_dir() must not take the report down
        notes.append(f"library.library_dir() failed: {exc}")
        return project_root() / "models" / "llm"


def _disk_free_mb(path: Path, disk: Callable[[str], Any], notes: list[str]) -> int | None:
    """``disk(...)`` (``shutil.disk_usage``-shaped: an object/tuple with a ``.free`` in bytes) for
    the nearest EXISTING parent of `path` (a not-yet-created library directory still lives on some
    real volume) -- ``None`` when nothing on the chain exists or the probe fails."""
    probe = path
    seen = set()
    while probe not in seen:
        seen.add(probe)
        if probe.exists():
            try:
                usage = disk(str(probe))
                return int(usage.free // (1024 * 1024))
            except Exception as exc:  # noqa: BLE001
                notes.append(f"disk usage probe for {probe} failed: {exc}")
                return None
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    notes.append(f"no existing parent directory found to measure free space for {path}")
    return None


def _recommended_policy(budget_mb: int | None) -> str:
    """"quality" (>= 20 GB budget) / "auto" (6-20 GB) / "fit" (smaller, or the budget is unknown --
    the conservative default when this machine's GPU memory could not be sized at all)."""
    if budget_mb is None:
        return "fit"
    if budget_mb >= 20480:
        return "quality"
    if budget_mb >= 6144:
        return "auto"
    return "fit"


def collect(
    cfg: LLMConfig,
    *,
    probe_runtime: bool = True,
    detect: "Callable[[], list[gpu.GpuInfo]] | None" = None,
    memory: "Callable[[], tuple[int | None, int | None]] | None" = None,
    cores: "Callable[[], int | None] | None" = None,
    driver: "Callable[[list[gpu.GpuInfo] | None], str | None] | None" = None,
    disk: "Callable[[str], Any]" = shutil.disk_usage,
    find_binary: "Callable[..., Any] | None" = None,
) -> HardwareReport:
    """One hardware snapshot for `cfg` (the ``[llm]`` section). Never raises: every probe below is
    wrapped so a failure lands as a line in the report's ``notes`` and leaves that field ``None``
    (or the module default) instead of taking the whole call down -- exactly what a launcher wants
    to show ("here is what I found; here is what I could not tell").

    Every probe is injectable for tests (never touches real hardware when a fake is passed):
    `detect` (default :func:`exocortex.gpu.detect_gpus`), `memory` (default
    :func:`exocortex.gpu.system_memory_mb`), `cores` (default
    :func:`exocortex.llm.server._physical_cores`, imported lazily), `driver` (default
    :func:`exocortex.gpu.nvidia_driver_version`), `disk` (default :func:`shutil.disk_usage`), and
    `find_binary` (default :func:`exocortex.llm.server.find_server_binary`, imported lazily, called
    with ``ensure=None`` -- this NEVER downloads a runtime, only reports what is already usable).
    `probe_runtime=False` skips that last probe entirely (``runtime`` stays ``None``, no note --
    the caller did not ask), for callers that only want the CPU/RAM/GPU facts quickly.
    """
    notes: list[str] = []

    detect = detect or gpu.detect_gpus
    memory = memory or gpu.system_memory_mb
    driver = driver or gpu.nvidia_driver_version
    if cores is None:
        try:
            from exocortex.llm.server import _physical_cores as cores
        except Exception as exc:  # noqa: BLE001
            notes.append(f"exocortex.llm.server not importable for core counting ({exc})")
            cores = lambda: None  # noqa: E731

    try:
        cpu = gpu.cpu_name()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"cpu_name() failed: {exc}")
        cpu = ""

    try:
        physical_cores = cores()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"physical core count probe failed: {exc}")
        physical_cores = None
    try:
        logical_cores = os.cpu_count()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"logical core count probe failed: {exc}")
        logical_cores = None

    try:
        ram_total_mb, ram_free_mb = memory()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"system_memory_mb() failed: {exc}")
        ram_total_mb, ram_free_mb = None, None

    try:
        gpus = list(detect())
    except Exception as exc:  # noqa: BLE001
        notes.append(f"GPU detection failed: {exc}")
        gpus = []
    gpu_dicts = [dataclasses.asdict(g) for g in gpus]

    try:
        primary = gpu.primary_gpu(gpus)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"primary_gpu() failed: {exc}")
        primary = None
    primary_index = primary.index if primary is not None else None
    budget_mb = primary.budget_mb if primary is not None else None

    try:
        vendor = gpu.gpu_vendor(gpus)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"gpu_vendor() failed: {exc}")
        vendor = "none"

    try:
        vram_total_mb = gpu.vram_total_mb()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"vram_total_mb() failed: {exc}")
        vram_total_mb = None
    try:
        vram_free_mb = gpu.vram_free_mb()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"vram_free_mb() failed: {exc}")
        vram_free_mb = None

    try:
        nvidia_driver = driver(gpus)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"nvidia driver lookup failed: {exc}")
        nvidia_driver = None

    library_dir_path = _resolve_library_dir(cfg, notes)
    library_free_mb = _disk_free_mb(library_dir_path, disk, notes)

    runtime: dict | None = None
    if probe_runtime:
        try:
            fb = find_binary
            if fb is None:
                from exocortex.llm.server import find_server_binary as fb
            probe_kwargs: dict = {"vendor": vendor, "backend": cfg.llama_backend, "source": cfg.llama_source,
                                  "upstream_build": cfg.llama_upstream_build}
            runtime_dir = getattr(cfg, "runtime_dir", "")
            if runtime_dir:
                # A build the setup step installed under a custom [llm].runtime_dir must show up
                # here too (the default root is what find_server_binary assumes otherwise).
                probe_kwargs["upstream_root"] = Path(runtime_dir).expanduser()
            binary = fb("", **probe_kwargs)
            runtime = {
                "flavor": binary.flavor, "version": binary.version, "source": binary.source,
                "exe": str(binary.exe), "framework": binary.framework, "reason": binary.reason,
            }
        except Exception as exc:  # noqa: BLE001 - "no runtime yet" is expected on a fresh machine
            notes.append(f"no llama.cpp runtime available yet: {exc}")
            runtime = None

    return HardwareReport(
        os=platform.platform(),
        cpu_name=cpu,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        ram_total_mb=ram_total_mb,
        ram_free_mb=ram_free_mb,
        gpus=gpu_dicts,
        primary_index=primary_index,
        vendor=vendor,
        budget_mb=budget_mb,
        vram_total_mb=vram_total_mb,
        vram_free_mb=vram_free_mb,
        nvidia_driver=nvidia_driver,
        library_dir=str(library_dir_path),
        library_free_mb=library_free_mb,
        runtime=runtime,
        recommended_policy=_recommended_policy(budget_mb),
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        notes=notes,
    )


def to_dict(report: HardwareReport) -> dict:
    """Plain-JSON representation of `report` (``dataclasses.asdict``)."""
    return dataclasses.asdict(report)


def report_path(cfg: LLMConfig) -> Path:
    """``<library_dir>/hardware.json`` -- where :func:`collect`'s snapshot is saved, next to the
    GGUFs it describes. Uses the same lazy, tolerant lookup as :func:`collect` itself."""
    return _resolve_library_dir(cfg, []) / "hardware.json"


def save(report: HardwareReport, path: Path) -> Path:
    """Writes `report` as JSON to `path` (parents created as needed) and returns `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(report), indent=2), encoding="utf-8")
    return path


def load(path: Path) -> HardwareReport | None:
    """The report at `path`, or ``None`` when it does not exist or does not parse -- never raises
    (a stale/corrupt hardware.json from an older schema must never crash the launcher; it just
    re-collects)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in dataclasses.fields(HardwareReport)}
        return HardwareReport(**{k: v for k, v in data.items() if k in known})
    except Exception:  # noqa: BLE001
        log.debug("hardware: could not load %s", path, exc_info=True)
        return None


def describe(report: HardwareReport) -> str:
    """The block the run script / doctor print: one fact per line, concise -- CPU/RAM, the GPU
    line (name, vendor, budget, live VRAM reading, NVIDIA driver if known), the runtime line, the
    library line, the recommended policy, then every note."""
    lines: list[str] = []
    lines.append(f"OS: {report.os}")
    cores = f"{report.physical_cores if report.physical_cores is not None else '?'} physical / " \
            f"{report.logical_cores if report.logical_cores is not None else '?'} logical cores"
    lines.append(f"CPU: {report.cpu_name or 'unknown'} ({cores})")
    ram_free = report.ram_free_mb if report.ram_free_mb is not None else "?"
    ram_total = report.ram_total_mb if report.ram_total_mb is not None else "?"
    lines.append(f"RAM: {ram_free} MiB free / {ram_total} MiB total")

    if report.gpus and report.primary_index is not None and report.primary_index < len(report.gpus):
        name = report.gpus[report.primary_index].get("name", "?")
    elif report.gpus:
        name = report.gpus[0].get("name", "?")
    else:
        name = None
    if name:
        gpu_line = f"GPU: {name} [{report.vendor}]"
        if report.budget_mb is not None:
            gpu_line += f", budget {report.budget_mb} MiB"
        if report.vram_free_mb is not None:
            gpu_line += f", {report.vram_free_mb} MiB free"
            if report.vram_total_mb is not None:
                gpu_line += f" / {report.vram_total_mb} MiB total"
        if report.nvidia_driver:
            gpu_line += f", driver {report.nvidia_driver}"
        lines.append(gpu_line)
    else:
        lines.append(f"GPU: none detected [{report.vendor}]")

    if report.runtime:
        rt = report.runtime
        lines.append(
            f"Runtime: {rt.get('flavor')} {rt.get('version')} ({rt.get('source')} / {rt.get('framework')})"
        )
    else:
        lines.append("Runtime: none found yet")

    lib_line = f"Library: {report.library_dir}"
    if report.library_free_mb is not None:
        lib_line += f" ({report.library_free_mb} MiB free)"
    lines.append(lib_line)

    lines.append(f"Recommended model-fit policy: {report.recommended_policy}")
    for note in report.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``describe()`` by default; ``--json`` prints :func:`to_dict`; ``--save`` additionally
    writes :func:`report_path`. Loads the config via :func:`exocortex.config.load_config`. Always
    exits 0 -- this may be shelled out to by the run script."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m exocortex.hardware", add_help=True,
                                      description="Hardware snapshot for ExoCortex (run script / doctor).")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--save", action="store_true", help="also write the report next to the model library")
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 0

    from exocortex.config import load_config

    cfg = load_config()
    report = collect(cfg.llm)
    if args.save:
        try:
            save(report, report_path(cfg.llm))
        except Exception as exc:  # noqa: BLE001 - --save is a bonus, not the reason to fail
            log.warning("hardware: could not save report: %s", exc)

    if args.json:
        print(json.dumps(to_dict(report)))
    else:
        print(describe(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
