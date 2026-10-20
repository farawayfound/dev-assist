# Installing ExoCortex

## Prerequisites

- **Windows 11.**
- **Nothing else.** As of 2026-09-27, one `run.bat` run sets up a working local LLM on its own --
  the llama.cpp runtime for this GPU, the speech models, and (on an empty model library) a prompt
  to pull one. **LM Studio and Ollama are both optional**: LM Studio, if already installed, is
  still read as a second model/runtime source (its packs are preferred when present); Ollama, if
  already running, is still a valid `[llm].backend = "auto"` target. Neither is required for a
  fresh install to work end to end.
- **Python is optional.** If a system Python (3.11-3.13) is already installed, `run.bat` uses it.
  If not, `run.bat` downloads a small, private one for itself (via [uv](https://astral.sh/uv)) --
  nothing is installed system-wide, and nothing outside the project folder is touched.

That's it. No CUDA toolkit, no manually-installed onnxruntime build, no `pip install` by hand, and
no other program to install first.

## What `run.bat` does, step by step

1. **Find (or fetch) a Python interpreter.** It tries `py -3.12`, then `py -3.13`, then `py -3.11`,
   then a bare `python` if its version is 3.11-3.13. If none of those exist, it downloads
   [uv](https://astral.sh/uv) (a single small .exe) into `.tools\uv\`, uses it to install Python
   3.12 privately, and creates `.venv` with it (`uv venv --seed`, which seeds `pip` into the venv
   so every later step is identical either way). This only happens once -- `.venv` is reused after
   that.
2. **Install the vendor-neutral core** (`requirements.txt`): faster-whisper, PySide6, RapidOCR,
   OpenVINO (for Intel iGPU/NPU OCR, and also usable for AMD's OpenVINO GPU plugin), and so on --
   nothing here is specific to one GPU vendor.
3. **Detect the GPU** (`exocortex.gpu --vendor` / `--extras`, DXGI-based, no drivers required to
   ask) and install this machine's extras on top:
   - **NVIDIA** -> `requirements-nvidia.txt` (the CUDA/cuDNN wheels faster-whisper needs to run
     Whisper on the GPU).
   - **AMD** -> `requirements-directml.txt` (`onnxruntime-directml`, for OCR on the iGPU/dGPU;
     this *replaces* the plain CPU `onnxruntime` package, so it is uninstalled first).
   - **Intel / no GPU** -> nothing extra; OpenVINO (already installed in step 2) and the CPU cover
     these machines.
4. **Write a stamp** (`.venv\exocortex-setup.stamp`) recording the GPU vendor and the date. On
   every later run, `run.bat` compares this stamp's timestamp against `requirements.txt`,
   `requirements-nvidia.txt`, `requirements-directml.txt` and `run.bat` itself, and re-probes the
   currently detected GPU vendor (`exocortex.gpu --vendor`, a few ms, no pip resolve) against the
   vendor word recorded in the stamp; if no watched file changed and the vendor still matches, it
   prints `Setup is up to date.` and skips straight to launching. If the vendor changed (a driver
   update, a hybrid-graphics/Optimus switch, a dock/monitor change) it reinstalls this machine's
   GPU extras even though no file on disk moved.
5. **First-run setup** (`python -m exocortex.setup`, `exocortex/setup.py`, 2026-09-27) -- right
   after step 4, on EVERY launch, not only the first: this is what makes LM Studio and Ollama
   optional. Four idempotent steps (~1 s total once everything is present):
   1. **The llama.cpp runtime** for this GPU -- an LM Studio pack when one is installed, else the
      pinned official llama.cpp build for the vendor's ladder, fetched with a progress line
      (`exocortex.llm.runtimes`; see the README's "Runtime sources" for the exact rule and the
      pinned asset list).
   2. **A hardware report** (`exocortex.hardware`) saved to `<library>/hardware.json`.
   3. **The speech models** -- Silero VAD + the speaker-embedding model, plus the ASR model
      (sherpa-onnx's Parakeet bundle off CUDA, faster-whisper's checkpoint on an NVIDIA box).
   4. **At least one LLM**, when the library holds none: a short list recommended for this
      machine's GPU budget (real sizes from the Hub), a Hub search, or "not now". See "Flags"
      below for how the prompt is skipped or auto-answered.
   A problem in this step (offline, a refused download) is reported and never blocks the launch --
   the overlay still transcribes, and the app retries the runtime fetch at start; the step only
   exits non-zero when no llama.cpp runtime could be found or fetched at all.
6. **Launch** the overlay windowless (`pythonw.exe -m exocortex --replace`) and closes its own
   console immediately, unless a flag below says otherwise.

## Flags

- **`run.bat --setup-only`** -- runs steps 1-5 above and exits without launching the overlay.
  This is what `run.vbs` uses for the one-time visible install.
- **`run.bat --doctor`** -- runs `python -m exocortex --doctor` in a normal, visible console (GPU,
  audio devices, models, llama-server), then pauses so the window stays readable, then exits. Use
  this instead of a plain `run.bat` when you want to see what it found.
- **`EXO_SETUP_QUIET=1`** -- skips step 5's interactive "pull a model" prompt; the app opens its
  Models window (`... > Models...`) instead. `run.vbs` sets this itself for its hidden catch-up
  runs (nobody can answer a prompt in a hidden window). `python -m exocortex.setup --quiet` is the
  same flag spelled as a CLI argument.
- **`EXO_SETUP_NONINTERACTIVE=1`** -- the same skip, keyed off the environment rather than stdin;
  useful for a CI/scripted run that has a real console but must never block on input.
- **`--yes`** (`python -m exocortex.setup --yes`) -- pulls the first recommended model without
  asking, instead of skipping the prompt. Combine with a config that already sets `[llm].model` if
  you want a specific model instead of the top recommendation.
- **`EXO_SETUP_DRYRUN=1`** -- makes every pip/uv/download/setup step (including step 5) print the
  command or action it would take instead of running it; used by `tests\test_run_script.py` so the
  test suite needs no network access and takes seconds, not minutes.

## `run.vbs`

Double-click `run.vbs` (or make a shortcut to it) for a completely windowless start: no console,
nothing in the taskbar except the overlay itself. It decides on its own, every launch, whether
setup needs to run again -- by checking `.venv`, the setup stamp's presence, its timestamp against
`requirements.txt`/`requirements-nvidia.txt`/`requirements-directml.txt`/`run.bat`, and (once a
stamp exists) the same GPU-vendor check `run.bat` uses -- rather than only checking whether the
stamp file happens to exist:

- **No `.venv` yet** (a genuinely fresh install): runs `run.bat --setup-only` in a normal, visible
  window so you can watch the one-time install.
- **`.venv` already works, but setup is stale or was never stamped** (upgrading from an older copy
  of this project, a `requirements*.txt`/`run.bat` change, or a changed GPU vendor): runs the same
  `run.bat --setup-only` for real, but with its console window hidden -- an already-working
  install never sees an unexpected console just to catch up.
- **Everything is current**: launches the overlay directly, with nothing shown at all.

## What runs where

| | NVIDIA | AMD | Intel iGPU | Intel Arc (discrete) | No GPU |
|---|---|---|---|---|---|
| **LLM** (`llama-server`) | LM Studio's CUDA pack when installed, else the pinned upstream CUDA 13.4 build (driver >= 580, native Blackwell/RTX-50 kernels) or CUDA 12.4 otherwise, fetched automatically -- **live-verified** 2026-09-27 on an RTX 5080 Laptop with no LM Studio pack at all: ready 40.0 s after `start()` (both CUDA zips downloaded), 318 tok/s prompt / 473 tok/s generation | ROCm pack where LM Studio installs it (2x faster prompt processing than Vulkan here), falling through to the Vulkan pack when ROCm lists no device or fails to load; else the pinned upstream ROCm build (**unverified**: not yet fetched live under this rule) | Vulkan pack | Vulkan pack from LM Studio, or -- when LM Studio has no pack for this vendor at all -- the same pinned upstream Vulkan build fetched automatically (`[llm].llama_source = "auto"`, the default); SYCL/OpenVINO are opt-in only (`llama_backend = "sycl"` / `"openvino"`), upstream-only, and **unverified**: no Intel Arc GPU exists on either dev box this app was built on -- see the README's "Intel Arc verification checklist" | CPU pack |
| **Speech recognition** (`[asr].engine = "auto"`) | faster-whisper `large-v3-turbo`, CUDA float16 (unchanged) | sherpa-onnx Parakeet-TDT 0.6B, CPU int8 (~0.2 s per utterance; Whisper on this CPU takes ~6 s) | sherpa-onnx Parakeet-TDT, CPU int8 | sherpa-onnx Parakeet-TDT, CPU int8 (CTranslate2 has no Intel GPU path either) | sherpa-onnx Parakeet-TDT, CPU int8 |
| **OCR** (RapidOCR / Windows OCR) | Windows OCR or RapidOCR on CPU -- never the NVIDIA card | `onnxruntime-directml` on the iGPU/dGPU (OpenVINO's GPU plugin crashes on AMD hardware -- see `exocortex/screen/ocr.py`) | OpenVINO on the iGPU or NPU | OpenVINO on the Arc card itself, once no Intel iGPU and no NVIDIA adapter share the machine (rule a2 in `exocortex/screen/ocr.py`) -- otherwise DirectML | CPU |

**The upstream llama.cpp runtime download.** When `[llm].llama_source` resolves to fetching an
official build (the "auto" fallback above, or an explicit `"upstream"`), ExoCortex downloads a
pinned release zip from `github.com/ggml-org/llama.cpp` (`exocortex/llm/runtimes.py`), verifies it
against both its exact byte size and its published sha256 before extracting, and unpacks it under
`<repo>/.tools/llama.cpp/<build>/<kind>/` (e.g. `.tools\llama.cpp\b11200\cuda13`; `[llm].runtime_dir`
overrides the root). **As of 2026-09-27 this covers every vendor, CUDA and ROCm included** -- the
rule is "an LM Studio pack wins first, for any flavor of this vendor's ladder; upstream only once
none exists at all" (see the README's "Runtime sources" for the exact rule, an LM Studio machine
stays byte-identical). Most flavors are one zip (~19-121 MB); CUDA is two -- the `llama-*` binaries
plus a same-release `cudart-llama-*` zip with the matching CUDA runtime DLLs, both required.

Ollama, when used instead of LM Studio, does its own model storage and GPU routing regardless of
vendor; ExoCortex just talks to it over HTTP (`[llm].server_url`, the field `base_url`/`ollama_url`
migrate into).

See the README's "Intel Arc verification checklist" for the exact steps to confirm this branch's
Intel Arc discrete-GPU support once real hardware is available -- everything Arc-specific in the
table above is a unit-tested design today, not a measurement.
