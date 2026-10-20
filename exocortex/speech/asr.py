"""faster-whisper / sherpa-onnx wrapper: loads an ASR model and transcribes short utterances
handed to it by the VAD segmenter, applying drop rules for silence / hallucination artefacts.

2026-09-26 (asr-sherpa): `resolve_engine()` picks which model `Transcriber.load()` builds --
"faster-whisper" (unchanged, see below) or "sherpa" (`SherpaTransducerModel`, a sherpa-onnx
`OfflineRecognizer` over a NeMo Parakeet-TDT transducer) -- per `cfg.engine` and
`exocortex.config.ASRConfig`'s comment for the numbers behind the "auto" choice. Both models
satisfy the same `_ModelLike.transcribe(audio, **kwargs) -> (segments, info)` duck type, so
`Transcriber.transcribe()` (drop rules, hotword prompt, beam size, ...) needed no change at all:
`SherpaTransducerModel.transcribe()` simply ignores the faster-whisper-specific kwargs
(language/beam_size/initial_prompt/...) it is handed.

Device routing (`_resolve_device_and_compute_type`): "auto" -> cuda+float16 when
`ctranslate2.get_cuda_device_count() > 0`, else cpu+int8; a CUDA load failure retries on
cpu/int8. This is unchanged by the 2026-09-25 iGPU/AMD work below -- CTranslate2 (the faster-
whisper backend) has no ROCm/Vulkan/OpenVINO device of its own, so every non-NVIDIA machine
(AMD, Intel-only, no GPU) runs Whisper on the CPU regardless of what GPU sits next to it.

What changed is the CPU *thread count* (`default_cpu_threads`): on the NVIDIA laptops this was
tuned for (RTX 5080 / 5090 + Intel Arc iGPU/NPU), Whisper's CPU threads share the core count with
llama-server's CPU-bound tensors (the TQ1_0 ternary kernels have no CUDA implementation), so the
historical default of 4 threads is kept exactly -- `default_cpu_threads("nvidia", ...)` always
returns 4, whatever the core count. On every other vendor, Whisper is the only heavy CPU consumer
in the process (llama.cpp runs on the AMD/Intel GPU via Vulkan/ROCm), so it gets more of the
machine: `max(4, min(8, physical_cores // 2))` -- 8 threads on this dev box (a 16-core/32-thread
Ryzen AI MAX+ 395), verified 2026-09-25 to keep int8 large-v3-turbo transcribes under the
utterance length where 4 threads did not (see `scripts/bench_asr.py`).

`exocortex.cuda_env.setup_cuda_dlls()` MUST run before `ctranslate2`/`faster_whisper` are
imported, so those imports happen lazily inside `Transcriber.load()`, not at module import
time (this module has no import-time side effects and is safe to import without a GPU).
"""
from __future__ import annotations

import logging
import os
import re
import string
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from exocortex.config import ASRConfig, project_root

log = logging.getLogger(__name__)

# Utterances shorter than this are checked against the hallucination blocklist (final only).
_HALLUCINATION_MAX_AUDIO_S = 1.5

# Normalised (lowercase, punctuation-stripped) texts Whisper is known to emit on
# silence/noise-only clips. See SPEC.md "Silence hallucination".
HALLUCINATIONS: frozenset[str] = frozenset(
    {
        "thank you",
        "thanks for watching",
        "thank you for watching",
        "you",
        "bye",
        "the end",
        "subtitles by the amara.org community",
    }
)

# Characters left over when a "text" is really just punctuation/music-note noise
# (e.g. Whisper emitting "..." or "♪♪♪" for non-speech audio).
_PUNCT_OR_MUSIC = set(string.punctuation) | {"♪", "♫", "♩", "“", "”", "…"}

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class AsrResult:
    """Outcome of one `Transcriber.transcribe()` call."""

    text: str
    avg_logprob: float
    no_speech_prob: float
    audio_s: float
    elapsed_s: float
    dropped: bool
    reason: str


class _SegmentLike(Protocol):
    text: str
    start: float
    end: float
    avg_logprob: float
    no_speech_prob: float


class _ModelLike(Protocol):
    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Any, Any]: ...


ModelFactory = Callable[[str, str, str, int], _ModelLike]


def _normalise(text: str) -> str:
    """Lowercase, strip surrounding whitespace, drop punctuation, collapse spaces."""
    stripped = "".join(ch for ch in text if ch not in string.punctuation)
    return _WHITESPACE_RE.sub(" ", stripped).strip().lower()


# HALLUCINATIONS entries are written with natural punctuation (e.g. "amara.org"); compare
# against normalised forms of both sides so `_normalise()`-stripped transcript text still matches.
_NORMALISED_HALLUCINATIONS: frozenset[str] = frozenset(_normalise(h) for h in HALLUCINATIONS)


def _is_only_punct_or_music(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return all(ch in _PUNCT_OR_MUSIC or ch.isspace() for ch in stripped)


def _build_hotwords_prompt(hotwords: list[str]) -> str | None:
    """Render `cfg.asr.hotwords` as a short glossary prompt, or None when empty."""
    words = [w.strip() for w in hotwords if w and w.strip()]
    if not words:
        return None
    return "Glossary: " + ", ".join(words) + "."


def default_cpu_threads(vendor: str | None = None, physical: int | None = None) -> int:
    """CTranslate2 thread count for Whisper running on the CPU.

    ``vendor`` defaults to ``exocortex.gpu.gpu_vendor()``; ``physical`` defaults to
    ``exocortex.llm.server._physical_cores()``, falling back to ``os.cpu_count() // 2`` (assumes
    SMT, the old heuristic) and then to 4 if even that is unavailable.

    ``"nvidia"`` always returns 4: the historical default, unchanged, because on those laptops
    the CPU also feeds llama-server's CPU-bound tensors (see module docstring). Every other
    vendor -- AMD, Intel-only, no GPU -- gets ``max(4, min(8, physical // 2))``: 8 on this 16-core
    Ryzen AI MAX+ 395.
    """
    if vendor is None:
        try:
            from exocortex.gpu import gpu_vendor

            vendor = gpu_vendor()
        except Exception:  # noqa: BLE001 - discovery is advisory; default to the safe old value
            log.debug("asr: gpu_vendor() unavailable", exc_info=True)
            vendor = "none"
    if vendor == "nvidia":
        return 4
    if physical is None:
        try:
            from exocortex.llm.server import _physical_cores

            physical = _physical_cores()
        except Exception:  # noqa: BLE001
            log.debug("asr: _physical_cores() unavailable", exc_info=True)
            physical = None
        if physical is None:
            cores = os.cpu_count()
            physical = (cores // 2) if cores else 4
    return max(4, min(8, physical // 2))


def _default_model_factory(model_size_or_path: str, device: str, compute_type: str, cpu_threads: int) -> _ModelLike:
    """Builds the real faster-whisper `WhisperModel`. Imported lazily so this module has
    no CTranslate2/CUDA import-time side effects."""
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_size_or_path,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )


def _cuda_available_default() -> bool:
    """The same probe `Transcriber._resolve_device_and_compute_type()` uses for `device="auto"`:
    True if CTranslate2 sees a CUDA device, after `cuda_env.setup_cuda_dlls()` has put NVIDIA's
    pip-installed DLLs on PATH. False (never raises) on any machine without CUDA/ctranslate2."""
    try:
        from exocortex.cuda_env import setup_cuda_dlls

        setup_cuda_dlls()
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 - no CUDA/ctranslate2 available
        return False


# 2026-09-26 (asr-sherpa): the measured decision behind resolve_engine()'s "auto" on a non-CUDA
# machine -- see exocortex.config.ASRConfig.engine's comment for the numbers (scripts/bench_asr.py
# on this dev box, Ryzen AI MAX+ 395 / Radeon 8060S, no NVIDIA GPU) and the decision rule
# (sherpa's mean word overlap >= whisper's - 0.05 AND its median latency <= 50% of whisper's).
_AUTO_ENGINE_WITHOUT_CUDA = "sherpa"


def resolve_engine(cfg: ASRConfig, cuda_available: Callable[[], bool] = _cuda_available_default) -> str:
    """Resolve `cfg.engine` ("auto" | "faster-whisper" | "sherpa") to a concrete engine name.

    An explicit `"faster-whisper"` or `"sherpa"` is returned unchanged (an explicit choice is
    never second-guessed, even on hardware where it is the wrong one -- that is what `"auto"` is
    for). `"auto"`: `"faster-whisper"` when `cfg.device == "cuda"` (an explicit CUDA request) or
    when `cuda_available()` (the default: CTranslate2 sees a CUDA device -- the exact probe
    `Transcriber._resolve_device_and_compute_type()` uses for `device="auto"`), so this never
    changes anything on the NVIDIA laptops; otherwise `_AUTO_ENGINE_WITHOUT_CUDA` (see above).
    `cuda_available` is injectable so this stays a pure function for tests.
    """
    if cfg.engine in ("faster-whisper", "sherpa"):
        return cfg.engine
    if cfg.device == "cuda" or cuda_available():
        return "faster-whisper"
    return _AUTO_ENGINE_WITHOUT_CUDA


def resolve_sherpa_model(cfg: ASRConfig) -> str:
    """Resolve `cfg.sherpa_model`, swapping the English-only default for the multilingual v3
    bundle when `cfg.language` asks for anything else -- an explicit `sherpa_model` (i.e. not
    still at its default) is never overridden, matching how every other "soft default" in this
    codebase behaves (see e.g. AssistConfig.primary_language)."""
    if cfg.sherpa_model == "parakeet-tdt-0.6b-v2-int8" and cfg.language not in ("en", "auto", ""):
        return "parakeet-tdt-0.6b-v3-int8"
    return cfg.sherpa_model


@dataclass(frozen=True)
class _SherpaSegment:
    """`_SegmentLike` for a sherpa-onnx transducer result: always exactly one per utterance
    (or zero, for empty text), so `start`/`end` simply span the whole clip and `avg_logprob` /
    `no_speech_prob` are fixed at values that never trip `Transcriber._apply_drop_rules`'
    logprob/no_speech thresholds (the blocklist and empty-text rules still apply normally)."""

    text: str
    start: float
    end: float
    avg_logprob: float
    no_speech_prob: float


def _find_onnx(model_dir: Path, stem: str) -> Path:
    """The `<stem>*.onnx` file under `model_dir` (e.g. stem "encoder" -> "encoder.int8.onnx"),
    preferring an int8 variant when more than one candidate is present."""
    candidates = sorted(model_dir.glob(f"{stem}*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"no {stem}*.onnx found under {model_dir}")
    int8 = [c for c in candidates if "int8" in c.name.lower()]
    return (int8 or candidates)[0]


class SherpaTransducerModel:
    """`_ModelLike` over a sherpa-onnx `OfflineRecognizer` built from a NeMo transducer (NVIDIA
    Parakeet-TDT) ONNX export -- the non-CUDA CPU speech-recognition engine (see the module and
    `ASRConfig.engine` docstrings for why: no Whisper GPU path exists off CUDA, and a transducer
    architecture is fast on the CPU where Whisper's fixed 30 s encoder window is not).

    File names are resolved by globbing `model_dir` (the directory
    `exocortex.models_dl.ensure_sherpa_model()` extracts the model archive into) rather than
    hard-coded, since sherpa-onnx model releases vary their exact file names (e.g.
    "encoder.int8.onnx" vs "encoder.onnx") -- see `_find_onnx`.

    `provider`: constructed with "directml" once; some sherpa-onnx wheels are built with the
    DirectML onnxruntime execution provider and some are not, so a construction failure there
    (not a transcribe-time failure -- that would be a bug elsewhere) falls back to "cpu" and logs
    at debug. `self.provider` records whichever one this class itself asked for/fell back to.

    CAVEAT (verified 2026-09-26 on this dev box, sherpa-onnx 1.13.8): "not available" does not
    always surface as a raised exception here. This wheel has no DirectML execution provider
    compiled in, yet `from_transducer(provider="directml")` returns successfully anyway -- the
    native (C++) layer prints a "... Fallback to cpu!" line and silently builds a CPU session
    instead of raising. A reliable, portable way to detect that from Python was tried (redirecting
    the OS-level stderr file descriptor around the call) and abandoned: it only works if this is
    the *first* sherpa-onnx native call in the whole process, because the C++ runtime appears to
    bind its own stderr target once, at native-module-load time, not per call -- and in the real
    app sherpa-onnx is already loaded earlier for VAD/diarization, so the redirection would almost
    never see the message there either. Net effect: on a machine like this one, `self.provider`
    stays `"directml"` even though the session actually runs on the CPU -- the measured latency
    numbers this work is based on (scripts/bench_asr.py) are exactly that CPU number regardless of
    what this attribute says, so nothing about the "auto" decision depends on it being accurate.
    A future sherpa-onnx release that raises on an unsupported provider (or exposes the resolved
    provider on the recognizer) would make this attribute trustworthy without any change here.
    """

    def __init__(self, model_dir: Path, num_threads: int, provider: str = "cpu") -> None:
        model_dir = Path(model_dir)
        encoder = _find_onnx(model_dir, "encoder")
        decoder = _find_onnx(model_dir, "decoder")
        joiner = _find_onnx(model_dir, "joiner")
        tokens = model_dir / "tokens.txt"
        if not tokens.is_file():
            raise FileNotFoundError(f"tokens.txt not found under {model_dir}")

        import sherpa_onnx

        def _build(prov: str):
            return sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
                tokens=str(tokens),
                num_threads=num_threads,
                provider=prov,
                model_type="nemo_transducer",
                decoding_method="greedy_search",
            )

        try:
            self._recognizer = _build(provider)
            self.provider = provider
        except Exception:  # noqa: BLE001 - this provider is unavailable in this wheel; fall back
            if provider == "cpu":
                raise
            log.debug("SherpaTransducerModel: provider=%r unavailable; falling back to cpu",
                      provider, exc_info=True)
            self._recognizer = _build("cpu")
            self.provider = "cpu"

        self._warned_kwargs = False

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[list[_SherpaSegment], None]:
        """Ignores every faster-whisper-specific kwarg (language, beam_size, initial_prompt, ...)
        -- logged at debug once, since `Transcriber.transcribe()` always passes them. Returns a
        one-element segment list, or `[]` for empty (no-speech) text."""
        if kwargs and not self._warned_kwargs:
            log.debug("SherpaTransducerModel.transcribe: ignoring kwargs %s (not applicable to a "
                      "NeMo transducer)", sorted(kwargs))
            self._warned_kwargs = True

        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, np.asarray(audio, dtype=np.float32))
        self._recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        if not text:
            return [], None
        return [_SherpaSegment(text=text, start=0.0, end=len(audio) / 16000.0,
                                avg_logprob=0.0, no_speech_prob=0.0)], None


SherpaModelFactory = Callable[[Path, int], _ModelLike]


def _sherpa_model_factory(model_dir: Path, num_threads: int) -> _ModelLike:
    """Builds the real `SherpaTransducerModel`, trying provider "directml" first (see its
    docstring for the cpu fallback)."""
    return SherpaTransducerModel(model_dir, num_threads, provider="directml")


class Transcriber:
    """faster-whisper / sherpa-onnx wrapper (see module docstring for the 2026-09-26 engine split).

    `load()` resolves the engine (`resolve_engine(cfg)`) and dispatches to `_load_faster_whisper`
    ("auto": cuda+float16 if `ctranslate2.get_cuda_device_count() > 0` else cpu+int8) or
    `_load_sherpa` (always cpu+int8; downloads/extracts the model bundle first via
    `exocortex.models_dl.ensure_sherpa_model` if needed). Either way: loads the model and runs a
    warm-up transcribe of 1 s of silence. Thread-safe via an internal lock (one transcribe at a
    time).

    `transcribe()` uses `beam_size` from config (1 for partials), a fixed `language`,
    `vad_filter=False` (segmentation already happened upstream),
    `condition_on_previous_text=False`, `without_timestamps=True`, and an `initial_prompt`
    built from `cfg.hotwords` when any are configured -- unchanged by the engine split: a sherpa
    model simply ignores these faster-whisper-specific kwargs (`SherpaTransducerModel.transcribe`).
    Drop rules (see `AsrResult.reason`): empty text, `no_speech_prob` above threshold,
    `avg_logprob` below threshold, or (for final results on clips shorter than 1.5 s) a
    normalised-text match against a small hallucination blocklist.
    """

    def __init__(self, cfg: ASRConfig, on_status: Callable[[str, str], None] | None = None,
                 *, model_factory: ModelFactory = _default_model_factory,
                 sherpa_model_factory: SherpaModelFactory = _sherpa_model_factory,
                 models_dir: Path | None = None,
                 progress: Callable[[str, int, int], None] | None = None,
                 ensure_sherpa_model: Callable[[str, Path, Callable[[str, int, int], None] | None], Path]
                 | None = None) -> None:
        """`on_status(state, detail)` mirrors ComponentStatusEvent states ("loading",
        "ready", "error"). `model_factory(model_size_or_path, device, compute_type,
        cpu_threads) -> model` is injectable for tests; defaults to the real
        faster-whisper `WhisperModel`. `sherpa_model_factory(model_dir, num_threads) -> model` is
        the sherpa equivalent, used when the engine resolves to "sherpa". `models_dir` defaults to
        `exocortex.config.project_root() / "models"` (matches `Config.models_dir()`'s own
        default). `progress(name, downloaded, total)` is forwarded to
        `exocortex.models_dl.ensure_sherpa_model` for the sherpa model download; when not given,
        download progress is still reported (throttled to whole percent) through `on_status`
        rather than left silent. `ensure_sherpa_model` is injectable for tests; defaults to the
        real `exocortex.models_dl.ensure_sherpa_model` (imported lazily)."""
        self._cfg = cfg
        self._on_status = on_status
        self._model_factory = model_factory
        self._sherpa_model_factory = sherpa_model_factory
        self._models_dir = models_dir if models_dir is not None else project_root() / "models"
        self._progress = progress if progress is not None else self._status_progress
        self._ensure_sherpa_model = ensure_sherpa_model
        self._model: _ModelLike | None = None
        self._lock = threading.Lock()
        self._device_used = ""
        self._compute_type_used = ""
        self._cpu_threads_used = 0
        self._engine_used = ""
        self._model_used = ""
        self._last_progress_pct = -1

    @property
    def device_used(self) -> str:
        """Resolved device ("cuda" or "cpu") after `load()`."""
        return self._device_used

    @property
    def compute_type_used(self) -> str:
        """Resolved compute type (e.g. "float16", "int8") after `load()`."""
        return self._compute_type_used

    @property
    def cpu_threads_used(self) -> int:
        """CTranslate2 / sherpa-onnx thread count handed to the model factory after `load()`
        (0 before it; also 0 on CUDA, where no CPU thread count applies)."""
        return self._cpu_threads_used

    @property
    def engine_used(self) -> str:
        """Resolved engine ("faster-whisper" or "sherpa") after `load()` (`""` before it)."""
        return self._engine_used

    @property
    def model_used(self) -> str:
        """The model name actually loaded (`cfg.model` or the resolved sherpa model bundle
        name) after `load()` (`""` before it)."""
        return self._model_used

    def _status_progress(self, name: str, downloaded: int, total: int) -> None:
        """Default sherpa model-download progress reporter: forwards to `on_status` as a
        "loading" detail, throttled to whole-percent steps so a multi-hundred-MB download does
        not flood the status callback with one call per 64 kB chunk."""
        if total > 0:
            pct = min(100, int(downloaded * 100 / total))
            if pct == self._last_progress_pct:
                return
            self._last_progress_pct = pct
            self._status("loading", f"downloading {name} ... {pct}%")
        else:
            self._status("loading", f"downloading {name} ... {downloaded / 1_000_000:.0f} MB")

    def _status(self, state: str, detail: str) -> None:
        if self._on_status is not None:
            try:
                self._on_status(state, detail)
            except Exception:  # noqa: BLE001 - a broken status callback must not break ASR
                log.exception("on_status callback raised")

    def _resolve_device_and_compute_type(self) -> tuple[str, str]:
        device = self._cfg.device
        if device == "auto":
            try:
                from exocortex.cuda_env import setup_cuda_dlls

                setup_cuda_dlls()
                import ctranslate2

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:  # noqa: BLE001 - no CUDA/ctranslate2 available -> CPU
                log.exception("CUDA device probe failed; falling back to cpu")
                device = "cpu"

        compute_type = self._cfg.compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        return device, compute_type

    def load(self) -> None:
        """Resolve the engine (`resolve_engine(cfg)`) and dispatch to `_load_faster_whisper` or
        `_load_sherpa`. Records `engine_used` before dispatching either way."""
        self._engine_used = resolve_engine(self._cfg)
        if self._engine_used == "sherpa":
            self._load_sherpa()
        else:
            self._load_faster_whisper()

    def _load_faster_whisper(self) -> None:
        """Resolve device/compute type, load the model, and run a warm-up transcribe.
        Falls back to CPU/int8 if loading on CUDA raises. Unchanged by the 2026-09-26 engine
        split -- same factory, same kwargs, same fallback -- as required for the NVIDIA path."""
        from exocortex.cuda_env import setup_cuda_dlls

        setup_cuda_dlls()

        device, compute_type = self._resolve_device_and_compute_type()
        cpu_threads = self._cfg.cpu_threads if self._cfg.cpu_threads > 0 else default_cpu_threads()
        self._cpu_threads_used = cpu_threads
        self._status("loading", f"loading {self._cfg.model} on {device}/{compute_type}")
        try:
            model = self._model_factory(self._cfg.model, device, compute_type, cpu_threads)
        except Exception as exc:  # noqa: BLE001 - fall back to CPU on any load failure
            log.exception("failed to load ASR model on %s/%s", device, compute_type)
            self._status("error", f"failed to load on {device}/{compute_type}: {exc}")
            device, compute_type = "cpu", "int8"
            self._status("loading", f"retrying on {device}/{compute_type}")
            model = self._model_factory(self._cfg.model, device, compute_type, cpu_threads)

        self._model = model
        self._device_used = device
        self._compute_type_used = compute_type
        self._model_used = self._cfg.model

        silence = np.zeros(16000, dtype=np.float32)
        self.transcribe(silence, partial=False)

        detail = f"{self._cfg.model} on {device}/{compute_type}"
        if device == "cpu":
            detail += f" ({cpu_threads} threads)"
        self._status("ready", detail)

    def _load_sherpa(self) -> None:
        """Resolve the sherpa model bundle, download/extract it if needed
        (`exocortex.models_dl.ensure_sherpa_model`), load it via `_sherpa_model_factory`, and run
        the same warm-up transcribe as the faster-whisper path. Always cpu/int8 -- sherpa-onnx's
        model here is int8-quantised and this path exists precisely because no GPU is in play."""
        model_name = resolve_sherpa_model(self._cfg)
        self._model_used = model_name
        cpu_threads = self._cfg.cpu_threads if self._cfg.cpu_threads > 0 else default_cpu_threads()
        self._cpu_threads_used = cpu_threads

        ensure_fn = self._ensure_sherpa_model
        if ensure_fn is None:
            from exocortex.models_dl import ensure_sherpa_model as ensure_fn

        self._status("loading", f"downloading {model_name} (sherpa-onnx)")
        model_dir = ensure_fn(model_name, self._models_dir, self._progress)

        self._status("loading", f"loading {model_name} (sherpa-onnx) on cpu/int8, {cpu_threads} threads")
        model = self._sherpa_model_factory(model_dir, cpu_threads)

        self._model = model
        self._device_used = "cpu"
        self._compute_type_used = "int8"

        silence = np.zeros(16000, dtype=np.float32)
        self.transcribe(silence, partial=False)

        self._status("ready", f"{model_name} (sherpa-onnx) on cpu/int8, {cpu_threads} threads")

    def transcribe(self, audio: np.ndarray, *, partial: bool = False) -> AsrResult:
        """Transcribe `audio` (float32 mono, 16 kHz). Raises `RuntimeError` if `load()`
        has not been called yet."""
        if self._model is None:
            raise RuntimeError("Transcriber.load() must be called before transcribe()")

        audio_s = len(audio) / 16000.0
        beam_size = 1 if partial else self._cfg.beam_size
        prompt = _build_hotwords_prompt(self._cfg.hotwords)

        t0 = time.monotonic()
        with self._lock:
            segments, _info = self._model.transcribe(
                audio,
                language=self._cfg.language,
                beam_size=beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
                initial_prompt=prompt,
                temperature=0.0,
            )
            segments = list(segments)
        elapsed_s = time.monotonic() - t0

        text = " ".join(seg.text.strip() for seg in segments if seg.text and seg.text.strip())
        text = _WHITESPACE_RE.sub(" ", text).strip()

        total_dur = sum(max(0.0, seg.end - seg.start) for seg in segments)
        if total_dur > 0:
            avg_logprob = sum(seg.avg_logprob * max(0.0, seg.end - seg.start) for seg in segments) / total_dur
        elif segments:
            avg_logprob = sum(seg.avg_logprob for seg in segments) / len(segments)
        else:
            avg_logprob = 0.0
        no_speech_prob = max((seg.no_speech_prob for seg in segments), default=0.0)

        dropped, reason = self._apply_drop_rules(text, avg_logprob, no_speech_prob, audio_s, partial)

        return AsrResult(
            text=text,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            audio_s=audio_s,
            elapsed_s=elapsed_s,
            dropped=dropped,
            reason=reason,
        )

    def _apply_drop_rules(
        self, text: str, avg_logprob: float, no_speech_prob: float, audio_s: float, partial: bool
    ) -> tuple[bool, str]:
        if not text:
            return True, "empty text"
        if no_speech_prob > self._cfg.no_speech_threshold:
            return True, f"no_speech_prob {no_speech_prob:.2f} > {self._cfg.no_speech_threshold:.2f}"
        if avg_logprob < self._cfg.logprob_threshold:
            return True, f"avg_logprob {avg_logprob:.2f} < {self._cfg.logprob_threshold:.2f}"
        if not partial and audio_s < _HALLUCINATION_MAX_AUDIO_S:
            normalised = _normalise(text)
            if normalised in _NORMALISED_HALLUCINATIONS:
                return True, f"hallucination blocklist: {normalised!r}"
            if _is_only_punct_or_music(text):
                return True, "text is only punctuation/music symbols"
        return False, ""
