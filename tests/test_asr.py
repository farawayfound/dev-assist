"""Unit tests for exocortex.speech.asr — offline, using a fake WhisperModel injected via
`model_factory`. See tests/test_asr.py -m live for tests against the real large-v3-turbo model.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from exocortex.config import ASRConfig
from exocortex.speech.asr import HALLUCINATIONS, AsrResult, Transcriber, default_cpu_threads

from conftest import TWO_SPEAKERS_TRUTH


@dataclass
class FakeSegment:
    """Stand-in for faster_whisper.transcribe.Segment; only the fields Transcriber reads."""

    text: str
    start: float = 0.0
    end: float = 1.0
    avg_logprob: float = -0.3
    no_speech_prob: float = 0.1


@dataclass
class FakeModel:
    """Stand-in for faster_whisper.WhisperModel. `segments` is returned (as a fresh list)
    on every `transcribe()` call; mutate it between calls to change the next result.
    Every call's kwargs are recorded in `calls`."""

    segments: list[FakeSegment] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    cpu_threads_seen: int | None = None    # set by make_transcriber's factory on load()

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[list[FakeSegment], None]:
        self.calls.append(kwargs)
        return list(self.segments), None


def make_transcriber(cfg: ASRConfig | None = None, model: FakeModel | None = None,
                      on_status=None) -> tuple[Transcriber, FakeModel]:
    """Builds a Transcriber wired to `model` (or a fresh FakeModel) via model_factory, and
    loads it (warm-up call included in model.calls[0]). The factory records the `cpu_threads`
    it was called with on `model.cpu_threads_seen`.

    Pins `cfg.engine = "faster-whisper"` (2026-09-26 asr-sherpa): every test in this module
    exercises the faster-whisper path via `FakeModel`, and must do so on any machine regardless
    of what `resolve_engine`'s real CUDA probe finds there -- `cfg.device == "cuda"` already
    forces this by itself, but a plain `ASRConfig()` (device/engine both "auto") on a non-CUDA
    machine would otherwise resolve to "sherpa" and never call `model_factory` at all. Tests of
    `resolve_engine` itself live in test_asr_sherpa.py, not here."""
    cfg = cfg or ASRConfig()
    cfg.engine = "faster-whisper"
    model = model if model is not None else FakeModel()

    def factory(model_size_or_path: str, device: str, compute_type: str, cpu_threads: int) -> FakeModel:
        model.cpu_threads_seen = cpu_threads
        return model

    t = Transcriber(cfg, on_status, model_factory=factory)
    t.load()
    return t, model


def audio_of(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16000), dtype=np.float32)


# ---- drop rules --------------------------------------------------------------------------

def test_drop_empty_text() -> None:
    t, model = make_transcriber()
    model.segments = [FakeSegment(text="")]
    result = t.transcribe(audio_of(2.0))
    assert result.dropped is True
    assert result.reason == "empty text"
    assert result.text == ""


def test_drop_no_speech_prob_above_threshold() -> None:
    cfg = ASRConfig(no_speech_threshold=0.7)
    t, model = make_transcriber(cfg)
    model.segments = [FakeSegment(text="hello there", no_speech_prob=0.9, avg_logprob=-0.2)]
    result = t.transcribe(audio_of(2.0))
    assert result.dropped is True
    assert "no_speech_prob" in result.reason
    assert result.no_speech_prob == 0.9


def test_drop_avg_logprob_below_threshold() -> None:
    cfg = ASRConfig(logprob_threshold=-1.2)
    t, model = make_transcriber(cfg)
    model.segments = [FakeSegment(text="hello there", avg_logprob=-2.0, no_speech_prob=0.1)]
    result = t.transcribe(audio_of(2.0))
    assert result.dropped is True
    assert "avg_logprob" in result.reason
    assert result.avg_logprob == -2.0


@pytest.mark.parametrize("phrase", sorted(HALLUCINATIONS))
def test_drop_hallucination_blocklist_short_clip(phrase: str) -> None:
    t, model = make_transcriber()
    model.segments = [FakeSegment(text=phrase.capitalize() + ".")]
    result = t.transcribe(audio_of(1.0))  # < 1.5 s
    assert result.dropped is True
    assert "hallucination" in result.reason


def test_hallucination_blocklist_not_applied_to_long_clip() -> None:
    t, model = make_transcriber()
    model.segments = [FakeSegment(text="Thank you.")]
    result = t.transcribe(audio_of(3.0))  # >= 1.5 s: blocklist not checked
    assert result.dropped is False


def test_hallucination_blocklist_not_applied_to_partials() -> None:
    t, model = make_transcriber()
    model.segments = [FakeSegment(text="Thank you.")]
    result = t.transcribe(audio_of(1.0), partial=True)
    assert result.dropped is False


def test_drop_punctuation_only_text() -> None:
    t, model = make_transcriber()
    model.segments = [FakeSegment(text="... ")]
    result = t.transcribe(audio_of(1.0))
    assert result.dropped is True
    assert "punctuation" in result.reason


def test_normal_short_text_not_dropped() -> None:
    t, model = make_transcriber()
    model.segments = [FakeSegment(text="Yes, sure.")]
    result = t.transcribe(audio_of(1.0))
    assert result.dropped is False
    assert result.reason == ""
    assert result.text == "Yes, sure."


# ---- hotwords / initial_prompt -----------------------------------------------------------

def test_hotwords_build_glossary_prompt() -> None:
    cfg = ASRConfig(hotwords=["Kafka", "RabbitMQ"])
    t, model = make_transcriber(cfg)
    model.segments = [FakeSegment(text="hello")]
    t.transcribe(audio_of(2.0))
    assert model.calls[-1]["initial_prompt"] == "Glossary: Kafka, RabbitMQ."


def test_no_hotwords_means_no_prompt() -> None:
    cfg = ASRConfig(hotwords=[])
    t, model = make_transcriber(cfg)
    model.segments = [FakeSegment(text="hello")]
    t.transcribe(audio_of(2.0))
    assert model.calls[-1]["initial_prompt"] is None


# ---- fixed transcribe() kwargs -------------------------------------------------------------

def test_transcribe_passes_fixed_kwargs() -> None:
    cfg = ASRConfig(language="en")
    t, model = make_transcriber(cfg)
    model.segments = [FakeSegment(text="hello")]
    t.transcribe(audio_of(2.0))
    call = model.calls[-1]
    assert call["language"] == "en"
    assert call["vad_filter"] is False
    assert call["condition_on_previous_text"] is False
    assert call["without_timestamps"] is True
    assert call["temperature"] == 0.0


# ---- partial vs final beam size ------------------------------------------------------------

def test_partial_uses_beam_size_one() -> None:
    cfg = ASRConfig(beam_size=5)
    t, model = make_transcriber(cfg)
    model.segments = [FakeSegment(text="hello")]
    t.transcribe(audio_of(2.0), partial=True)
    assert model.calls[-1]["beam_size"] == 1


def test_final_uses_configured_beam_size() -> None:
    cfg = ASRConfig(beam_size=5)
    t, model = make_transcriber(cfg)
    model.segments = [FakeSegment(text="hello")]
    t.transcribe(audio_of(2.0), partial=False)
    assert model.calls[-1]["beam_size"] == 5


# ---- text joining / whitespace -------------------------------------------------------------

def test_segments_joined_with_single_spaces() -> None:
    t, model = make_transcriber()
    model.segments = [
        FakeSegment(text=" Hello  "),
        FakeSegment(text="there   friend "),
        FakeSegment(text="how are  you?"),
    ]
    result = t.transcribe(audio_of(3.0))
    assert result.text == "Hello there friend how are you?"


def test_empty_segments_skipped_when_joining() -> None:
    t, model = make_transcriber()
    model.segments = [FakeSegment(text="Hello"), FakeSegment(text="   "), FakeSegment(text="world")]
    result = t.transcribe(audio_of(3.0))
    assert result.text == "Hello world"


# ---- avg_logprob / no_speech_prob aggregation -----------------------------------------------

def test_avg_logprob_is_duration_weighted() -> None:
    t, model = make_transcriber()
    model.segments = [
        FakeSegment(text="a", start=0.0, end=1.0, avg_logprob=-0.2, no_speech_prob=0.1),
        FakeSegment(text="b", start=1.0, end=4.0, avg_logprob=-0.8, no_speech_prob=0.05),
    ]
    result = t.transcribe(audio_of(4.0))
    assert result.avg_logprob == pytest.approx(-0.65)
    assert result.no_speech_prob == pytest.approx(0.1)
    assert result.dropped is False


def test_audio_s_reflects_input_length() -> None:
    t, model = make_transcriber()
    model.segments = [FakeSegment(text="hello")]
    result = t.transcribe(audio_of(2.5))
    assert result.audio_s == pytest.approx(2.5)


# ---- device / compute_type resolution + fallback --------------------------------------------

def test_explicit_cpu_device_uses_int8_compute_type() -> None:
    cfg = ASRConfig(device="cpu", compute_type="auto")
    t, model = make_transcriber(cfg)
    assert t.device_used == "cpu"
    assert t.compute_type_used == "int8"


def test_explicit_cuda_device_uses_float16_compute_type() -> None:
    cfg = ASRConfig(device="cuda", compute_type="auto")
    t, model = make_transcriber(cfg)
    assert t.device_used == "cuda"
    assert t.compute_type_used == "float16"


def test_auto_device_resolves_via_ctranslate2_cuda_count(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCt2:
        @staticmethod
        def get_cuda_device_count() -> int:
            return 0

    monkeypatch.setitem(sys.modules, "ctranslate2", _FakeCt2())
    cfg = ASRConfig(device="auto", compute_type="auto")
    t, model = make_transcriber(cfg)
    assert t.device_used == "cpu"
    assert t.compute_type_used == "int8"


def test_auto_device_picks_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCt2:
        @staticmethod
        def get_cuda_device_count() -> int:
            return 1

    monkeypatch.setitem(sys.modules, "ctranslate2", _FakeCt2())
    cfg = ASRConfig(device="auto", compute_type="auto")
    t, model = make_transcriber(cfg)
    assert t.device_used == "cuda"
    assert t.compute_type_used == "float16"


def test_load_falls_back_to_cpu_when_cuda_factory_raises() -> None:
    cfg = ASRConfig(device="cuda", compute_type="auto")
    statuses: list[tuple[str, str]] = []
    cpu_model = FakeModel()

    def factory(model_size_or_path: str, device: str, compute_type: str, cpu_threads: int) -> FakeModel:
        if device == "cuda":
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        return cpu_model

    t = Transcriber(cfg, lambda state, detail: statuses.append((state, detail)), model_factory=factory)
    t.load()

    assert t.device_used == "cpu"
    assert t.compute_type_used == "int8"
    states = [s for s, _ in statuses]
    assert states == ["loading", "error", "loading", "ready"]
    # the warm-up transcribe ran against the fallback (cpu) model
    assert len(cpu_model.calls) == 1


def test_load_runs_warmup_transcribe_of_one_second_silence() -> None:
    t, model = make_transcriber()
    assert len(model.calls) == 1


def test_transcribe_before_load_raises() -> None:
    cfg = ASRConfig()

    def factory(model_size_or_path: str, device: str, compute_type: str, cpu_threads: int) -> FakeModel:
        return FakeModel()

    t = Transcriber(cfg, model_factory=factory)
    with pytest.raises(RuntimeError):
        t.transcribe(audio_of(1.0))


def test_on_status_reports_loading_then_ready() -> None:
    statuses: list[tuple[str, str]] = []
    cfg = ASRConfig(device="cpu")
    make_transcriber(cfg, on_status=lambda state, detail: statuses.append((state, detail)))
    states = [s for s, _ in statuses]
    assert states == ["loading", "ready"]


# ---- default_cpu_threads (vendor-aware CPU thread count) ------------------------------------

def test_default_cpu_threads_nvidia_is_always_4_regardless_of_cores() -> None:
    assert default_cpu_threads("nvidia", physical=4) == 4
    assert default_cpu_threads("nvidia", physical=64) == 4


def test_default_cpu_threads_amd_16_cores_returns_8() -> None:
    assert default_cpu_threads("amd", physical=16) == 8


def test_default_cpu_threads_amd_8_cores_returns_4() -> None:
    assert default_cpu_threads("amd", physical=8) == 4


def test_default_cpu_threads_amd_40_cores_is_capped_at_8() -> None:
    assert default_cpu_threads("amd", physical=40) == 8


def test_default_cpu_threads_intel_uses_the_same_non_nvidia_rule() -> None:
    assert default_cpu_threads("intel", physical=12) == 6


def test_default_cpu_threads_no_gpu_uses_the_same_non_nvidia_rule() -> None:
    assert default_cpu_threads("none", physical=16) == 8


def test_default_cpu_threads_vendor_defaults_to_gpu_vendor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("exocortex.gpu.gpu_vendor", lambda: "nvidia")
    assert default_cpu_threads(physical=32) == 4
    monkeypatch.setattr("exocortex.gpu.gpu_vendor", lambda: "amd")
    assert default_cpu_threads(physical=32) == 8


def test_default_cpu_threads_physical_defaults_to_physical_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("exocortex.llm.server._physical_cores", lambda: 16)
    assert default_cpu_threads("amd") == 8


def test_default_cpu_threads_falls_back_to_cpu_count_when_physical_cores_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_physical_cores()` returns None (non-Windows, or an odd kernel32): fall back to
    `os.cpu_count() // 2` (assumes SMT), then to 4 if even `os.cpu_count()` is unavailable."""
    monkeypatch.setattr("exocortex.llm.server._physical_cores", lambda: None)
    import exocortex.speech.asr as asr_mod

    monkeypatch.setattr(asr_mod.os, "cpu_count", lambda: 32)
    assert default_cpu_threads("amd") == 8   # 32 // 2 = 16 physical guess -> min(8, 16 // 2)

    monkeypatch.setattr(asr_mod.os, "cpu_count", lambda: None)
    assert default_cpu_threads("amd") == 4   # no reading at all -> the floor


def test_default_cpu_threads_gpu_vendor_lookup_failure_falls_back_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> str:
        raise RuntimeError("no DXGI")

    monkeypatch.setattr("exocortex.gpu.gpu_vendor", boom)
    assert default_cpu_threads(physical=16) == 8    # vendor "none" -> the non-NVIDIA rule


# ---- Transcriber.load() resolves and records cpu_threads --------------------------------------

def test_transcriber_uses_configured_cpu_threads_when_positive() -> None:
    cfg = ASRConfig(cpu_threads=6)
    t, model = make_transcriber(cfg)
    assert model.cpu_threads_seen == 6
    assert t.cpu_threads_used == 6


def test_transcriber_uses_default_cpu_threads_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    import exocortex.speech.asr as asr_mod

    monkeypatch.setattr(asr_mod, "default_cpu_threads", lambda: 7)
    cfg = ASRConfig(cpu_threads=0)
    t, model = make_transcriber(cfg)
    assert model.cpu_threads_seen == 7
    assert t.cpu_threads_used == 7


def test_ready_status_detail_includes_thread_count_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import exocortex.speech.asr as asr_mod

    monkeypatch.setattr(asr_mod, "default_cpu_threads", lambda: 5)
    statuses: list[tuple[str, str]] = []
    cfg = ASRConfig(device="cpu", cpu_threads=0)
    make_transcriber(cfg, on_status=lambda state, detail: statuses.append((state, detail)))
    ready_detail = next(detail for state, detail in statuses if state == "ready")
    assert "5 threads" in ready_detail


def test_ready_status_detail_omits_thread_count_on_cuda() -> None:
    cfg = ASRConfig(device="cuda", cpu_threads=4)
    statuses: list[tuple[str, str]] = []
    make_transcriber(cfg, on_status=lambda state, detail: statuses.append((state, detail)))
    ready_detail = next(detail for state, detail in statuses if state == "ready")
    assert "threads" not in ready_detail


def test_asr_result_is_frozen_dataclass() -> None:
    result = AsrResult(text="hi", avg_logprob=-0.1, no_speech_prob=0.1, audio_s=1.0,
                        elapsed_s=0.01, dropped=False, reason="")
    with pytest.raises(Exception):
        result.text = "changed"  # type: ignore[misc]


# ---- live: real large-v3-turbo on CUDA ------------------------------------------------------

def _word_overlap(hypothesis: str, truth: str) -> float:
    """Fraction of truth's words also present in hypothesis (case-insensitive, alnum tokens)."""
    words = lambda s: set(re.findall(r"[a-z0-9']+", s.lower()))
    truth_words = words(truth)
    if not truth_words:
        return 1.0
    return len(words(hypothesis) & truth_words) / len(truth_words)


@pytest.mark.live
class TestLiveTranscription:
    """Loads the real faster-whisper large-v3-turbo model on CUDA and checks accuracy/speed
    against the truth table for tests/data/two_speakers_16k.wav. Run with:
    pytest tests/test_asr.py -m live -q
    """

    @pytest.fixture(scope="class")
    def transcriber(self) -> Transcriber:
        cfg = ASRConfig(model="large-v3-turbo", device="cuda", compute_type="auto")
        t = Transcriber(cfg)
        t.load()  # includes the warm-up transcribe
        return t

    @pytest.mark.parametrize(
        "start,end,speaker,truth_text",
        TWO_SPEAKERS_TRUTH,
        ids=[t[3] for t in TWO_SPEAKERS_TRUTH],
    )
    def test_truth_table_slice(
        self,
        transcriber: Transcriber,
        two_speakers_audio: tuple[np.ndarray, int],
        start: float,
        end: float,
        speaker: str,
        truth_text: str,
    ) -> None:
        audio, sr = two_speakers_audio
        assert sr == 16000
        clip = audio[int(start * sr): int(end * sr)]

        result = transcriber.transcribe(clip)

        assert _word_overlap(result.text, truth_text) >= 0.7, (
            f"got {result.text!r}, expected overlap with {truth_text!r}"
        )
        assert result.elapsed_s < 1.0

    def test_device_and_compute_type_used(self, transcriber: Transcriber) -> None:
        assert transcriber.device_used == "cuda"
        assert transcriber.compute_type_used == "float16"
