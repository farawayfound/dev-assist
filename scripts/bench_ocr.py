"""Benchmark the OCR engines (module docstring of ``exocortex.screen.ocr``) on one real screen
frame: Windows OCR, RapidOCR through OpenVINO on the iGPU / NPU / CPU, RapidOCR through
onnxruntime's DirectML execution provider ("dml", a non-Intel GPU), and RapidOCR's own
onnxruntime-CPU path with both accelerators forced off.

    .venv\\Scripts\\python.exe scripts\\bench_ocr.py [--config config.toml] [--runs 5]
                                                     [--devices igpu,npu,cpu,onnxruntime,dml]

Grabs the primary monitor once (``exocortex.screen.capture.ScreenGrabber``, mode ``"monitor"``)
and saves it to a temp ``.npy`` file, then times each engine against that same frame: Windows OCR
(``exocortex.screen.ocr.WindowsOcr``) always, and RapidOCR (``exocortex.screen.ocr.RapidOcr``)
once per ``--devices`` entry -- "igpu" / "npu" / "cpu" let :func:`~exocortex.screen.ocr.resolve_ocr_device`
pick the accelerator as the app does (OpenVINO, or -- on a non-Intel GPU -- DirectML); "onnxruntime"
monkeypatches ``resolve_ocr_device`` to return ``("cpu", "")`` in that child so RapidOCR falls back
to its own plain onnxruntime-CPU path; "dml" monkeypatches it to return ``("igpu", "dml:0")``,
forcing onnxruntime's DirectML execution provider on adapter 0 regardless of what this machine's
OpenVINO install or GPU list would otherwise pick (onnxruntime-directml must be installed for
this one -- see ``requirements-directml.txt`` -- it replaces the plain CPU ``onnxruntime`` package
faster-whisper also depends on, so restore that afterwards: see the module docstring of
``exocortex.screen.ocr`` and the AMD section below).
``--devices`` defaults to every kind :func:`~exocortex.screen.ocr.resolve_ocr_device` can satisfy
on this machine (igpu and/or npu, if OpenVINO -- or DirectML, on a non-Intel GPU -- can serve them,
plus cpu) with "onnxruntime" always added.

Each engine runs in its own subprocess, RapidOCR variants first and Windows OCR last (belt and
braces: the module docstring's process-order rule -- a RapidOCR created AFTER a WindowsOcr in one
process crashes the interpreter -- cannot bite across separate processes, and one crashing or
unavailable backend does not take the others down, same approach as ``exocortex/doctor.py``'s
``_check_screen``). Each child prints one line of JSON on stdout that the parent parses; anything
else on stdout/stderr is the engine's own noise and is ignored unless the child's exit code is
non-zero, in which case its last stderr line is shown as the failure reason.

For each engine the table reports: init ms (engine construction, incl. any OpenVINO compile),
warm-up ms (the first ``recognize()`` call -- NPU/iGPU kernel builds not yet paid by init land
here), median and min wall ms over ``--runs`` further calls, the median process CPU ms per call
(``time.process_time()`` -- onnxruntime spreads work over every core, so wall and CPU time tell
different stories), the line count, and the first three recognised lines.

Measured on 2026-09-23 (Core Ultra 9 285H, Arc 140T iGPU, AI Boost NPU, 64 GB RAM, Windows 11
26200; primary monitor 2560x1600, a code editor + terminal; OpenVINO cache at models/ov_cache
already warm; default --devices, --runs 5):

    engine                  init ms  warm ms  median ms   min ms   cpu ms  lines  first line(s)
    -------------------------------------------------------------------------------------------
    rapidocr/igpu              2110     5448       1421     1413     1500    204  'File Edit Selection' / 'View Go Run' / 'Terminal Help'
    rapidocr/npu                1978    14453       2842     2830     1109    205  'File Edit Selection' / 'View Go Run' / 'Terminal Help'
    rapidocr/cpu                2119     2672       2381     2353     9875    205  'File Edit Selection' / 'View Go Run' / 'Terminal Help'
    rapidocr/onnxruntime         582     5698       4033     3473    50422    205  'File Edit Selection' / 'View Go Run' / 'Terminal Help'
    windows                       10      152        139      138      141    212  'File Edit Selection View' / 'device = "cpu' / 'Go'

(init ms with a warm on-disk OpenVINO cache still pays a per-run device attach, not a from-scratch
kernel compile; warm-up ms is noisier than the doctor.py synthetic probe because it is the first
real accelerator dispatch for THIS frame's actual shapes -- across a dozen runs on this machine
rapidocr/igpu's warm-up ranged ~2.8-10.3 s. One run out of that dozen came back with a single
garbled line instead of ~200: that was the iGPU loading a *reshaped* (static-shape) model back
from OpenVINO's cache, which returns a wrong blob on the GPU plugin; the engine now keeps the
dynamic-shape model on the iGPU (only the inputs are bucketed) and the misread no longer occurs
-- see the "static"/"bucket" notes on ``exocortex.screen.ocr._OpenVinoSession``.
CPU time is the ``time.process_time()`` of THIS process; onnxruntime spreads work over every
core, so the wall/CPU gap on "cpu" and "onnxruntime" reflects that, not idle time. Despite the
higher numbers here than the module docstrings' synthetic-image figures, the relative order
matches: igpu fastest and cheapest on CPU, then npu, then the OpenVINO CPU plugin, then plain
onnxruntime -- this frame is a much busier real desktop capture (200+ lines) than that probe.)

AMD Radeon 8060S (measured 2026-09-25, Ryzen AI MAX+ 395, 64 GB UMA, Windows 11 26200, OpenVINO
2026.4.0, ``core.available_devices`` = ``['CPU', 'GPU']`` with GPU FULL_DEVICE_NAME "gfx1151
(iGPU)"; primary monitor 2560x1600, a code editor + terminal; ``--runs 3``):

    --devices igpu,cpu,onnxruntime (onnxruntime-directml NOT installed -- plain CPU onnxruntime):

    engine                  init ms  warm ms  median ms   min ms   cpu ms  lines  first line(s)
    -------------------------------------------------------------------------------------------
    rapidocr/igpu          FAILED: exit 3221226505: ^
    rapidocr/cpu               1074     1304        841      745     3969    109  'Recycle Bin' / 'Steam' / 'X'
    rapidocr/onnxruntime        546     1766       1073      980    15422    109  'Recycle Bin' / 'Steam' / 'X'
    windows                      14      203        130      128      125     93  'Lum' / 'isual' / 'Recycle Bin'

    --devices dml,cpu,onnxruntime, RUN 1 (cold; onnxruntime-directml==1.24.4 installed in place
    of plain onnxruntime -- see requirements-directml.txt):

    engine                  init ms  warm ms  median ms   min ms   cpu ms  lines  first line(s)
    -------------------------------------------------------------------------------------------
    rapidocr/dml                750     1176        937      932      953    153  'File Edit Selection View Go Run Terminal Help' / '?' / 'exo-cortex'
    rapidocr/cpu                803     1321       1028     1019     5031    151  'File Edit Selection View Go Run Terminal Help' / 'exo-cortex' / '?'
    rapidocr/onnxruntime        535     2384       1534     1533    22266    152  'File Edit Selection View Go Run Terminal Help' / 'exo-cortex' / '?'
    windows                      14      149        134      133      125    105  'File Edit Selection' / '08' / 'agentld" : " a18fee92f4f69ecd3" , " label" : "gpu-memory" , "phase" .'

    --devices dml,cpu,onnxruntime, RUN 2 (immediately after RUN 1, same frame; DirectML has no
    on-disk kernel cache, so this shows per-PROCESS cost only, not a cross-run warm-up):

    engine                  init ms  warm ms  median ms   min ms   cpu ms  lines  first line(s)
    -------------------------------------------------------------------------------------------
    rapidocr/dml                742     1256       1015      993     1375    155  'File Edit Selection View Go Run Terminal Help' / '?' / 'exo-cortex'
    rapidocr/cpu                850     1521       1157     1085     6078    155  'File Edit Selection View Go Run Terminal Help' / '?' / 'exo-cortex'
    rapidocr/onnxruntime        530     2347       1551     1541    22609    152  'File Edit Selection View Go Run Terminal Help' / 'exo-cortex' / '?'
    windows                      14      155        140      137      141    105  'File Edit Selection' / '08' / 'agentld" : " a18fee92f4f69ecd3" , " label" : "gpu-memory" , "phase" .'

rapidocr/igpu (OpenVINO's GPU plugin) is not a typo above: ``RapidOcr("igpu", ...)`` reliably
CRASHES the interpreter on this hardware, cold and with a warm on-disk ``CACHE_DIR`` alike (exit
3221226505 = 0xC0000409, ``STATUS_STACK_BUFFER_OVERRUN``; run outside this script's subprocess
isolation for the full message: the plugin's kernel compiler emits Intel GEN/vISA inline assembly
-- literal ``.decl`` / ``.implicit_PSEUDO_INPUT`` directives, an Intel-only ISA -- that the AMD
OpenCL ICD's shader compiler cannot parse). ``exocortex.screen.ocr.resolve_ocr_device`` therefore
never routes a non-Intel OpenVINO GPU through OpenVINO (see its docstring); DirectML is this
machine's replacement.

Correctness: rapidocr/dml's line count (153, 155) and first lines match rapidocr/cpu (OpenVINO CPU
plugin: 151, 155) and rapidocr/onnxruntime (plain CPU: 152, 152) within the module's +-10 % bar --
DirectML on this AMD iGPU reads the same screen correctly. Speed/cost: rapidocr/dml is the fastest
wall time of the three AND by far the cheapest in CPU time (~950-1400 ms vs ~5000-6100 ms for the
OpenVINO CPU plugin and ~22000-22600 ms for plain onnxruntime) -- consistent across both runs.
DECISION (2026-09-25): DirectML is this machine's "auto"/"igpu" default (OpenVINO GPU is not an
option here at all; DirectML clearly beats both CPU paths) -- see ``resolve_ocr_device``'s
docstring in ``exocortex/screen/ocr.py``. A human can flip it back once OpenVINO's GPU plugin is
fixed for AMD (or measurements on a different AMD/Intel-iGPU-less machine say otherwise).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from exocortex.config import load_config  # noqa: E402
from exocortex.screen import ocr  # noqa: E402
from exocortex.screen.capture import ScreenGrabber  # noqa: E402

RAPIDOCR_DEVICES: tuple[str, ...] = ("igpu", "npu", "cpu", "onnxruntime", "dml")

# One child per engine: creating RapidOCR AFTER WindowsOcr in a single process crashes the
# interpreter (exocortex/screen/ocr.py's module docstring), and an unavailable/crashing backend
# must not take the whole benchmark down. "onnxruntime" monkeypatches resolve_ocr_device so
# RapidOCR takes its own plain-CPU path instead of OpenVINO's; "dml" monkeypatches it the other
# way, forcing onnxruntime's DirectML execution provider on adapter 0 regardless of what OpenVINO
# or exocortex.gpu.detect_gpus() see on this machine (added 2026-09-25 for the AMD Radeon 8060S,
# where OpenVINO's own GPU plugin crashes -- see exocortex/screen/ocr.py's module docstring).
_CHILD = """
import json, sys, time
sys.path.insert(0, {root!r})
import numpy as np
from exocortex.screen import ocr as _ocr

kind, device, cache_dir, frame_path, runs = {kind!r}, {device!r}, {cache_dir!r}, {frame_path!r}, {runs!r}
# The label reflects what was asked for, not ocr.engine_label(engine): that collapses the OpenVINO
# CPU plugin and plain onnxruntime to the same "rapidocr" (both report device "cpu"), which is
# exactly the distinction "onnxruntime" exists to show.
label = "windows" if kind == "windows" else "rapidocr/" + device
if device == "onnxruntime":
    _ocr.resolve_ocr_device = lambda requested="auto", devices=None: ("cpu", "")
    device = "cpu"
elif device == "dml":
    _ocr.resolve_ocr_device = lambda requested="auto", devices=None: ("igpu", "dml:0")
    device = "igpu"

image = np.load(frame_path)
out = {{"ok": True, "label": label}}
try:
    t0 = time.perf_counter()
    engine = _ocr.WindowsOcr() if kind == "windows" else _ocr.RapidOcr(device, cache_dir)
    out["init_ms"] = (time.perf_counter() - t0) * 1000.0
except Exception as exc:
    print(json.dumps({{"ok": False, "label": label, "error": "%s: %s" % (type(exc).__name__, exc)}}))
    sys.exit(0)

t0 = time.perf_counter()
result = engine.recognize(image)
out["warm_ms"] = (time.perf_counter() - t0) * 1000.0

wall_ms, cpu_ms = [], []
for _ in range(runs):
    c0 = time.process_time()
    t0 = time.perf_counter()
    result = engine.recognize(image)
    wall_ms.append((time.perf_counter() - t0) * 1000.0)
    cpu_ms.append((time.process_time() - c0) * 1000.0)

lines = [ln.text for ln in result.lines]
out.update(wall_ms=wall_ms, cpu_ms=cpu_ms, num_lines=len(lines), first_lines=lines[:3])
print(json.dumps(out))
"""


def _default_devices() -> list[str]:
    """Every RapidOCR device kind this machine can satisfy through :func:`resolve_ocr_device` as
    the app would call it -- "igpu" here can mean an OpenVINO GPU (Intel) OR onnxruntime's
    DirectML execution provider (a non-Intel GPU, if ``onnxruntime-directml`` is installed and
    :func:`~exocortex.screen.ocr.directml_available` says so) -- plus plain "cpu", and
    "onnxruntime" (RapidOCR's own CPU path, both accelerators forced off). Pass ``--devices``
    explicitly (e.g. ``dml,cpu,onnxruntime``) to force a specific accelerator regardless of what
    this machine would pick on its own."""
    devices = ocr.openvino_devices()
    out = [kind for kind in ("igpu", "npu") if ocr.resolve_ocr_device(kind, devices)[0] == kind]
    out.append("cpu")
    out.append("onnxruntime")
    return out


def _run_engine(kind: str, device: str, cache_dir: Path, frame_path: Path, runs: int) -> dict:
    """Run one engine ("windows" or a RapidOCR device kind) in its own subprocess; returns the
    child's parsed JSON result, or ``{"ok": False, "error": ...}`` if it crashed / produced
    nothing usable."""
    label = "windows" if kind == "windows" else f"rapidocr/{device}"
    code = _CHILD.format(root=str(ROOT), kind=kind, device=device, cache_dir=str(cache_dir),
                          frame_path=str(frame_path), runs=runs)
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               timeout=300, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return {"ok": False, "label": label, "error": "timed out"}
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    err = (proc.stderr.strip().splitlines() or ["(no output)"])[-1]
    return {"ok": False, "label": label, "error": f"exit {proc.returncode}: {err[:200]}"}


def _print_table(rows: list[dict]) -> None:
    header = f"{'engine':<22} {'init ms':>8} {'warm ms':>8} {'median ms':>10} {'min ms':>8} {'cpu ms':>8} {'lines':>6}  first line(s)"
    print(header)
    print("-" * len(header))
    for row in rows:
        if not row.get("ok"):
            print(f"{row.get('label', '?'):<22} FAILED: {row.get('error', '?')}")
            continue
        wall, cpu = row["wall_ms"], row["cpu_ms"]
        first = " / ".join(repr(line) for line in row["first_lines"])
        print(f"{row['label']:<22} {row['init_ms']:>8.0f} {row['warm_ms']:>8.0f} "
              f"{statistics.median(wall):>10.0f} {min(wall):>8.0f} {statistics.median(cpu):>8.0f} "
              f"{row['num_lines']:>6}  {first}")


def main() -> int:
    # A screen frame's OCR text can contain any Unicode glyph (arrows, box-drawing, emoji in a
    # chat app...); Windows' default console code page (cp1252) chokes on those when a table row
    # is printed. errors="replace" keeps the benchmark's exit code and every other row intact
    # instead of crashing on the one row with an exotic character.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001 - best effort only (e.g. a captured/redirected stream)
            pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config")
    ap.add_argument("--runs", type=int, default=5, help="timed recognize() calls per engine, after warm-up")
    ap.add_argument("--devices", help=f"comma-separated RapidOCR device kinds, any of {RAPIDOCR_DEVICES} "
                                       "(default: every kind this machine's OpenVINO can satisfy, plus onnxruntime)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    devices = [d.strip().lower() for d in args.devices.split(",")] if args.devices else _default_devices()
    for d in devices:
        if d not in RAPIDOCR_DEVICES:
            ap.error(f"--devices: {d!r} is not one of {RAPIDOCR_DEVICES}")

    grabber = ScreenGrabber()
    try:
        frame = grabber.grab("monitor", monitor=0)
    finally:
        grabber.close()
    if frame is None:
        print("could not capture the primary monitor", file=sys.stderr)
        return 1
    print(f"captured {frame.source_name}: {frame.rect[2]}x{frame.rect[3]}\n")

    cache_dir = cfg.models_dir() / "ov_cache"
    with tempfile.TemporaryDirectory(prefix="exocortex-bench-ocr-") as tmp:
        frame_path = Path(tmp) / "frame.npy"
        np.save(frame_path, frame.image)

        rows = [_run_engine("rapidocr", device, cache_dir, frame_path, args.runs) for device in devices]
        rows.append(_run_engine("windows", "cpu", cache_dir, frame_path, args.runs))

    _print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
