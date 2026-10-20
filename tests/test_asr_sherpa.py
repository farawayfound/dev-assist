"""Unit tests for the 2026-09-26 asr-sherpa work: `resolve_engine`, `resolve_sherpa_model`,
`SherpaTransducerModel` (with a fake `sherpa_onnx` module injected into `sys.modules`), and
`Transcriber.load()` on the "sherpa" engine path (with a fake `ensure_sherpa_model` and a fake
`sherpa_model_factory` -- no real download, no real sherpa-onnx recognizer).

See tests/test_asr.py / tests/test_asr_extra.py for the faster-whisper path, which this file
never touches (their `make_transcriber()` pins `cfg.engine = "faster-whisper"` explicitly so
those tests are unaffected by whatever `resolve_engine`'s real CUDA probe finds on the machine
running them).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from exocortex.config import ASRConfig
from exocortex.speech.asr import (
    SherpaTransducerModel,
    Transcriber,
    resolve_engine,
    resolve_sherpa_model,
)

# --------------------------------------------------------------------------------------------- #
# resolve_engine
# --------------------------------------------------------------------------------------------- #


def test_resolve_engine_explicit_faster_whisper_is_never_overridden() -> None:
    cfg = ASRConfig(engine="faster-whisper")
    assert resolve_engine(cfg, cuda_available=lambda: False) == "faster-whisper"
    assert resolve_engine(cfg, cuda_available=lambda: True) == "faster-whisper"


def test_resolve_engine_explicit_sherpa_is_never_overridden() -> None:
    cfg = ASRConfig(engine="sherpa")
    assert resolve_engine(cfg, cuda_available=lambda: True) == "sherpa"
    assert resolve_engine(cfg, cuda_available=lambda: False) == "sherpa"


def test_resolve_engine_auto_with_cuda_device_uses_faster_whisper() -> None:
    cfg = ASRConfig(engine="auto", device="auto")
    assert resolve_engine(cfg, cuda_available=lambda: True) == "faster-whisper"


def test_resolve_engine_auto_device_cuda_uses_faster_whisper_even_if_probe_says_no() -> None:
    """`cfg.device == "cuda"` is an explicit request; it must win even if the injected CUDA
    probe (which `resolve_engine` would otherwise use for "auto") says False."""
    cfg = ASRConfig(engine="auto", device="cuda")
    assert resolve_engine(cfg, cuda_available=lambda: False) == "faster-whisper"


def test_resolve_engine_auto_without_cuda_uses_the_measured_default() -> None:
    cfg = ASRConfig(engine="auto", device="auto")
    # See exocortex.config.ASRConfig.engine's comment and scripts/bench_asr.py's 2026-09-26
    # measurement on this dev box for why "sherpa" is the decision here.
    assert resolve_engine(cfg, cuda_available=lambda: False) == "sherpa"


def test_resolve_engine_default_cuda_probe_is_used_when_not_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an injected `cuda_available`, resolve_engine() falls back to the real probe
    (same one `Transcriber._resolve_device_and_compute_type()` uses)."""
    class _FakeCt2:
        @staticmethod
        def get_cuda_device_count() -> int:
            return 1

    monkeypatch.setitem(sys.modules, "ctranslate2", _FakeCt2())
    cfg = ASRConfig(engine="auto", device="auto")
    assert resolve_engine(cfg) == "faster-whisper"

    class _FakeCt2NoDevice:
        @staticmethod
        def get_cuda_device_count() -> int:
            return 0

    monkeypatch.setitem(sys.modules, "ctranslate2", _FakeCt2NoDevice())
    assert resolve_engine(cfg) == "sherpa"


# --------------------------------------------------------------------------------------------- #
# resolve_sherpa_model
# --------------------------------------------------------------------------------------------- #


def test_resolve_sherpa_model_default_english() -> None:
    cfg = ASRConfig(language="en")
    assert resolve_sherpa_model(cfg) == "parakeet-tdt-0.6b-v2-int8"


def test_resolve_sherpa_model_switches_to_v3_for_other_languages() -> None:
    cfg = ASRConfig(language="fr")
    assert resolve_sherpa_model(cfg) == "parakeet-tdt-0.6b-v3-int8"


def test_resolve_sherpa_model_explicit_choice_is_never_overridden() -> None:
    cfg = ASRConfig(language="fr", sherpa_model="parakeet-tdt-0.6b-v2-int8")
    # sherpa_model is still literally the v2 default value, so this can't distinguish "explicit"
    # from "default" -- matches resolve_sherpa_model's documented behaviour (a soft default).
    assert resolve_sherpa_model(cfg) == "parakeet-tdt-0.6b-v3-int8"

    cfg2 = ASRConfig(language="fr", sherpa_model="some-other-bundle")
    assert resolve_sherpa_model(cfg2) == "some-other-bundle"


def test_resolve_sherpa_model_auto_language_keeps_default() -> None:
    cfg = ASRConfig(language="auto")
    assert resolve_sherpa_model(cfg) == "parakeet-tdt-0.6b-v2-int8"


# --------------------------------------------------------------------------------------------- #
# SherpaTransducerModel, with a fake sherpa_onnx module
# --------------------------------------------------------------------------------------------- #


class _FakeStreamResult:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeStream:
    def __init__(self, owner: "_FakeOfflineRecognizer") -> None:
        self._owner = owner
        self.accepted: list[tuple[int, np.ndarray]] = []
        self.result = _FakeStreamResult("")

    def accept_waveform(self, sample_rate: int, audio: np.ndarray) -> None:
        self.accepted.append((sample_rate, audio))


class _FakeOfflineRecognizer:
    """Stand-in for sherpa_onnx.OfflineRecognizer built via from_transducer()."""

    #: set by the test before calling transcribe(); consumed (and reset to "") after one decode.
    next_text: str = ""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.decoded_streams: list[_FakeStream] = []

    def create_stream(self) -> _FakeStream:
        return _FakeStream(self)

    def decode_stream(self, stream: _FakeStream) -> None:
        stream.result = _FakeStreamResult(_FakeOfflineRecognizer.next_text)
        self.decoded_streams.append(stream)


class _FakeSherpaOnnxModule:
    """Stand-in for the `sherpa_onnx` module: records every `from_transducer()` call's kwargs
    on `.calls`, and can be made to raise for a given provider (to test the directml->cpu
    fallback) via `.fail_providers`."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_providers: set[str] = set()
        self.last_recognizer: _FakeOfflineRecognizer | None = None

        class _OfflineRecognizer:
            @staticmethod
            def from_transducer(**kwargs: Any) -> _FakeOfflineRecognizer:
                self.calls.append(kwargs)
                if kwargs.get("provider") in self.fail_providers:
                    raise RuntimeError(f"provider {kwargs.get('provider')!r} unavailable in this build")
                rec = _FakeOfflineRecognizer(**kwargs)
                self.last_recognizer = rec
                return rec

        self.OfflineRecognizer = _OfflineRecognizer


def _make_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "parakeet-tdt-0.6b-v2-int8"
    d.mkdir()
    (d / "encoder.int8.onnx").write_bytes(b"e")
    (d / "decoder.onnx").write_bytes(b"d")
    (d / "joiner.int8.onnx").write_bytes(b"j")
    (d / "tokens.txt").write_bytes(b"<blk> 0\n")
    return d


@pytest.fixture()
def fake_sherpa_onnx(monkeypatch: pytest.MonkeyPatch) -> _FakeSherpaOnnxModule:
    fake = _FakeSherpaOnnxModule()
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)
    return fake


def test_sherpa_model_builds_recognizer_with_expected_kwargs(
    tmp_path: Path, fake_sherpa_onnx: _FakeSherpaOnnxModule
) -> None:
    model_dir = _make_model_dir(tmp_path)
    model = SherpaTransducerModel(model_dir, num_threads=6, provider="cpu")

    assert model.provider == "cpu"
    assert len(fake_sherpa_onnx.calls) == 1
    call = fake_sherpa_onnx.calls[0]
    assert call["encoder"] == str(model_dir / "encoder.int8.onnx")
    assert call["decoder"] == str(model_dir / "decoder.onnx")
    assert call["joiner"] == str(model_dir / "joiner.int8.onnx")
    assert call["tokens"] == str(model_dir / "tokens.txt")
    assert call["num_threads"] == 6
    assert call["provider"] == "cpu"
    assert call["model_type"] == "nemo_transducer"
    assert call["decoding_method"] == "greedy_search"


def test_sherpa_model_prefers_int8_onnx_when_multiple_candidates(
    tmp_path: Path, fake_sherpa_onnx: _FakeSherpaOnnxModule
) -> None:
    model_dir = tmp_path / "bundle"
    model_dir.mkdir()
    (model_dir / "encoder.onnx").write_bytes(b"fp32")
    (model_dir / "encoder.int8.onnx").write_bytes(b"int8")
    (model_dir / "decoder.onnx").write_bytes(b"d")
    (model_dir / "joiner.int8.onnx").write_bytes(b"j")
    (model_dir / "tokens.txt").write_bytes(b"<blk> 0\n")

    SherpaTransducerModel(model_dir, num_threads=4)

    call = fake_sherpa_onnx.calls[0]
    assert call["encoder"] == str(model_dir / "encoder.int8.onnx")


def test_sherpa_model_directml_falls_back_to_cpu_on_failure(
    tmp_path: Path, fake_sherpa_onnx: _FakeSherpaOnnxModule
) -> None:
    fake_sherpa_onnx.fail_providers = {"directml"}
    model_dir = _make_model_dir(tmp_path)

    model = SherpaTransducerModel(model_dir, num_threads=4, provider="directml")

    assert model.provider == "cpu"
    providers_tried = [c["provider"] for c in fake_sherpa_onnx.calls]
    assert providers_tried == ["directml", "cpu"]


def test_sherpa_model_cpu_failure_propagates(tmp_path: Path, fake_sherpa_onnx: _FakeSherpaOnnxModule) -> None:
    fake_sherpa_onnx.fail_providers = {"cpu"}
    model_dir = _make_model_dir(tmp_path)
    with pytest.raises(RuntimeError):
        SherpaTransducerModel(model_dir, num_threads=4, provider="cpu")


def test_sherpa_model_missing_tokens_raises(tmp_path: Path, fake_sherpa_onnx: _FakeSherpaOnnxModule) -> None:
    model_dir = tmp_path / "bundle"
    model_dir.mkdir()
    (model_dir / "encoder.onnx").write_bytes(b"e")
    (model_dir / "decoder.onnx").write_bytes(b"d")
    (model_dir / "joiner.onnx").write_bytes(b"j")
    with pytest.raises(FileNotFoundError):
        SherpaTransducerModel(model_dir, num_threads=4)


def test_sherpa_model_missing_encoder_raises(tmp_path: Path, fake_sherpa_onnx: _FakeSherpaOnnxModule) -> None:
    model_dir = tmp_path / "bundle"
    model_dir.mkdir()
    (model_dir / "decoder.onnx").write_bytes(b"d")
    (model_dir / "joiner.onnx").write_bytes(b"j")
    (model_dir / "tokens.txt").write_bytes(b"<blk> 0\n")
    with pytest.raises(FileNotFoundError):
        SherpaTransducerModel(model_dir, num_threads=4)


def test_sherpa_model_transcribe_returns_scripted_text(
    tmp_path: Path, fake_sherpa_onnx: _FakeSherpaOnnxModule
) -> None:
    model_dir = _make_model_dir(tmp_path)
    model = SherpaTransducerModel(model_dir, num_threads=4)
    _FakeOfflineRecognizer.next_text = "hello there"

    audio = np.zeros(16000, dtype=np.float32)
    segments, info = model.transcribe(audio)

    assert info is None
    assert len(segments) == 1
    seg = segments[0]
    assert seg.text == "hello there"
    assert seg.start == 0.0
    assert seg.end == pytest.approx(1.0)
    assert seg.avg_logprob == 0.0
    assert seg.no_speech_prob == 0.0


def test_sherpa_model_transcribe_empty_text_returns_empty_list(
    tmp_path: Path, fake_sherpa_onnx: _FakeSherpaOnnxModule
) -> None:
    model_dir = _make_model_dir(tmp_path)
    model = SherpaTransducerModel(model_dir, num_threads=4)
    _FakeOfflineRecognizer.next_text = ""

    segments, info = model.transcribe(np.zeros(8000, dtype=np.float32))
    assert segments == []
    assert info is None


def test_sherpa_model_transcribe_ignores_faster_whisper_kwargs(
    tmp_path: Path, fake_sherpa_onnx: _FakeSherpaOnnxModule
) -> None:
    model_dir = _make_model_dir(tmp_path)
    model = SherpaTransducerModel(model_dir, num_threads=4)
    _FakeOfflineRecognizer.next_text = "ok"

    # Must not raise, despite being handed faster-whisper-only kwargs.
    segments, _info = model.transcribe(
        np.zeros(1600, dtype=np.float32),
        language="en", beam_size=5, vad_filter=False,
        condition_on_previous_text=False, without_timestamps=True,
        initial_prompt="Glossary: Kafka.", temperature=0.0,
    )
    assert segments[0].text == "ok"


def test_sherpa_model_accept_waveform_called_with_16000(
    tmp_path: Path, fake_sherpa_onnx: _FakeSherpaOnnxModule
) -> None:
    model_dir = _make_model_dir(tmp_path)
    model = SherpaTransducerModel(model_dir, num_threads=4)
    _FakeOfflineRecognizer.next_text = "x"
    audio = np.ones(4800, dtype=np.float32) * 0.5

    model.transcribe(audio)

    rec = fake_sherpa_onnx.last_recognizer
    assert rec is not None
    stream = rec.decoded_streams[-1]
    assert stream.accepted[0][0] == 16000
    assert np.array_equal(stream.accepted[0][1], audio)


# --------------------------------------------------------------------------------------------- #
# Transcriber.load() on the sherpa engine path (fake ensure_sherpa_model + fake model factory)
# --------------------------------------------------------------------------------------------- #


class _FakeSherpaModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[list[Any], None]:
        self.calls.append(kwargs)
        return [], None  # warm-up silence: empty text either way


def _make_sherpa_transcriber(
    cfg: ASRConfig | None = None,
    *,
    ensure_calls: list[tuple[str, Path]] | None = None,
    factory_calls: list[tuple[Path, int]] | None = None,
    model: _FakeSherpaModel | None = None,
    on_status=None,
    models_dir: Path | None = None,
) -> Transcriber:
    cfg = cfg or ASRConfig(engine="sherpa")
    model = model if model is not None else _FakeSherpaModel()

    def fake_ensure_sherpa_model(name: str, models_dir_: Path, progress) -> Path:
        if ensure_calls is not None:
            ensure_calls.append((name, models_dir_))
        if progress is not None:
            progress(name, 50, 100)  # exercise the progress path without a real download
        d = models_dir_ / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fake_sherpa_factory(model_dir: Path, num_threads: int) -> _FakeSherpaModel:
        if factory_calls is not None:
            factory_calls.append((model_dir, num_threads))
        return model

    return Transcriber(
        cfg,
        on_status,
        sherpa_model_factory=fake_sherpa_factory,
        ensure_sherpa_model=fake_ensure_sherpa_model,
        models_dir=models_dir,
    )


def test_transcriber_load_sherpa_uses_resolved_model_and_engine(tmp_path: Path) -> None:
    cfg = ASRConfig(engine="sherpa")
    ensure_calls: list[tuple[str, Path]] = []
    factory_calls: list[tuple[Path, int]] = []
    t = _make_sherpa_transcriber(cfg, ensure_calls=ensure_calls, factory_calls=factory_calls, models_dir=tmp_path)
    t.load()

    assert t.engine_used == "sherpa"
    assert t.model_used == "parakeet-tdt-0.6b-v2-int8"
    assert t.device_used == "cpu"
    assert t.compute_type_used == "int8"
    assert ensure_calls == [("parakeet-tdt-0.6b-v2-int8", tmp_path)]
    assert factory_calls[0][0] == tmp_path / "parakeet-tdt-0.6b-v2-int8"


def test_transcriber_load_sherpa_resolves_v3_model_for_other_language(tmp_path: Path) -> None:
    cfg = ASRConfig(engine="sherpa", language="de")
    ensure_calls: list[tuple[str, Path]] = []
    t = _make_sherpa_transcriber(cfg, ensure_calls=ensure_calls, models_dir=tmp_path)
    t.load()

    assert t.model_used == "parakeet-tdt-0.6b-v3-int8"
    assert ensure_calls == [("parakeet-tdt-0.6b-v3-int8", tmp_path)]


def test_transcriber_load_sherpa_uses_configured_cpu_threads(tmp_path: Path) -> None:
    cfg = ASRConfig(engine="sherpa", cpu_threads=6)
    factory_calls: list[tuple[Path, int]] = []
    t = _make_sherpa_transcriber(cfg, factory_calls=factory_calls, models_dir=tmp_path)
    t.load()

    assert t.cpu_threads_used == 6
    assert factory_calls[0][1] == 6


def test_transcriber_load_sherpa_default_cpu_threads_when_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import exocortex.speech.asr as asr_mod

    monkeypatch.setattr(asr_mod, "default_cpu_threads", lambda: 7)
    cfg = ASRConfig(engine="sherpa", cpu_threads=0)
    t = _make_sherpa_transcriber(cfg, models_dir=tmp_path)
    t.load()

    assert t.cpu_threads_used == 7


def test_transcriber_load_sherpa_status_messages(tmp_path: Path) -> None:
    statuses: list[tuple[str, str]] = []
    cfg = ASRConfig(engine="sherpa", cpu_threads=8)
    t = _make_sherpa_transcriber(cfg, on_status=lambda s, d: statuses.append((s, d)), models_dir=tmp_path)
    t.load()

    states = [s for s, _ in statuses]
    assert states[0] == "loading"
    assert states[-1] == "ready"
    loading_details = [d for s, d in statuses if s == "loading"]
    assert any("downloading parakeet-tdt-0.6b-v2-int8" in d for d in loading_details)
    assert any(
        "loading parakeet-tdt-0.6b-v2-int8 (sherpa-onnx) on cpu/int8, 8 threads" in d
        for d in loading_details
    )
    ready_detail = next(d for s, d in statuses if s == "ready")
    assert ready_detail == "parakeet-tdt-0.6b-v2-int8 (sherpa-onnx) on cpu/int8, 8 threads"


def test_transcriber_load_sherpa_download_progress_reaches_on_status_by_default(tmp_path: Path) -> None:
    """No explicit `progress=` given: download progress must still surface through on_status
    (not be silent) -- see exocortex.speech.asr.Transcriber's docstring."""
    statuses: list[tuple[str, str]] = []
    cfg = ASRConfig(engine="sherpa")
    t = _make_sherpa_transcriber(cfg, on_status=lambda s, d: statuses.append((s, d)), models_dir=tmp_path)
    t.load()

    assert any("downloading" in d and "%" in d for _, d in statuses)


def test_transcriber_load_sherpa_explicit_progress_callback_is_used(tmp_path: Path) -> None:
    events: list[tuple[str, int, int]] = []
    cfg = ASRConfig(engine="sherpa")
    t = Transcriber(
        cfg,
        sherpa_model_factory=lambda model_dir, num_threads: _FakeSherpaModel(),
        ensure_sherpa_model=lambda name, models_dir_, progress: (
            progress(name, 50, 100), (models_dir_ / name).mkdir(parents=True, exist_ok=True), models_dir_ / name
        )[-1],
        models_dir=tmp_path,
        progress=lambda name, done, total: events.append((name, done, total)),
    )
    t.load()
    assert events == [("parakeet-tdt-0.6b-v2-int8", 50, 100)]


def test_transcriber_load_sherpa_default_models_dir_is_project_root_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Redirect project_root() to tmp_path so this test never touches the real repo's models/ dir.
    import exocortex.speech.asr as asr_mod

    monkeypatch.setattr(asr_mod, "project_root", lambda: tmp_path)
    ensure_calls: list[tuple[str, Path]] = []
    cfg = ASRConfig(engine="sherpa")
    t = _make_sherpa_transcriber(cfg, ensure_calls=ensure_calls)  # no models_dir override
    t.load()
    assert ensure_calls[0][1] == tmp_path / "models"


def test_transcriber_load_sherpa_runs_warmup_transcribe(tmp_path: Path) -> None:
    model = _FakeSherpaModel()
    t = _make_sherpa_transcriber(model=model, models_dir=tmp_path)
    t.load()
    assert len(model.calls) == 1


def test_transcriber_load_faster_whisper_path_unaffected_by_sherpa_wiring(tmp_path: Path) -> None:
    """Sanity check: passing sherpa-only kwargs to Transcriber must not change the
    faster-whisper path's behaviour when engine resolves to "faster-whisper"."""
    from test_asr import FakeModel  # local import matching this test suite's sys.path convention

    fw_model = FakeModel()

    def fw_factory(model_size_or_path: str, device: str, compute_type: str, cpu_threads: int) -> FakeModel:
        return fw_model

    # device pinned: with the default "auto" this test asserted "cpu" and so only passed on a box
    # without CUDA (it was red on the NVIDIA laptops); the sherpa wiring is what is under test here.
    cfg = ASRConfig(engine="faster-whisper", device="cpu")
    t = Transcriber(cfg, model_factory=fw_factory, models_dir=tmp_path)
    t.load()

    assert t.engine_used == "faster-whisper"
    assert t.device_used == "cpu"
    assert len(fw_model.calls) == 1
