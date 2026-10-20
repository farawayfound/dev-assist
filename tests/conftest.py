"""Shared pytest fixtures. Tests run offline by default (see pytest.ini)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exocortex.cuda_env import setup_cuda_dlls  # noqa: E402

setup_cuda_dlls()

DATA = ROOT / "tests" / "data"
MODELS = ROOT / "models"
TWO_SPEAKERS_WAV = DATA / "two_speakers_16k.wav"
SILERO = MODELS / "silero_vad.onnx"
SPEAKER_MODEL = MODELS / "wespeaker_en_voxceleb_CAM++.onnx"


@pytest.fixture(scope="session")
def two_speakers_audio() -> tuple[np.ndarray, int]:
    """(float32 mono samples, 16000). Two Windows TTS voices, 27 s, four sentences each way."""
    import soundfile as sf

    audio, sr = sf.read(str(TWO_SPEAKERS_WAV), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


@pytest.fixture(scope="session")
def silero_path() -> str:
    if not SILERO.is_file():
        pytest.skip("models/silero_vad.onnx missing")
    return str(SILERO)


@pytest.fixture(scope="session")
def speaker_model_path() -> str:
    if not SPEAKER_MODEL.is_file():
        pytest.skip("speaker embedding model missing")
    return str(SPEAKER_MODEL)


# Ground truth for tests/data/two_speakers_16k.wav (VAD segments measured with the defaults in SPEC.md)
# speaker A = "Microsoft David", speaker B = "Microsoft Zira"
TWO_SPEAKERS_TRUTH = [
    # (approx start, approx end, speaker, text)
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


@pytest.fixture(autouse=True)
def _never_save_the_real_config(monkeypatch):
    """An Engine built on a programmatic ``Config()`` (``cfg.path`` is None) must never write the
    developer's own ``config.toml`` on ``stop()`` (``Engine.save_config`` falls back to the
    project file when no path is set -- that is the first-launch behaviour of the real app, not
    something a test may do). Tests that check saving set ``cfg.path`` to a tmp file."""
    from exocortex import app as app_mod

    original = app_mod.Engine.save_config

    def guarded(self):
        if getattr(self.cfg, "path", None) is None:
            return
        original(self)

    monkeypatch.setattr(app_mod.Engine, "save_config", guarded)


@pytest.fixture(autouse=True)
def _no_real_swap_unload(monkeypatch):
    """exocortex.instance.free_llama_swap talks to the machine's real llama-swap on 127.0.0.1:8081
    and can UNLOAD a model the developer is using; no test may ever do that. Engine.stop() and
    main() both call it, so it is stubbed to a no-op for every test; wiring tests install their
    own recorder on top, and the unit tests for free_llama_swap itself hold a direct reference to
    the original function (imported at collection time), which this attribute swap leaves intact."""
    import exocortex.instance as _inst

    monkeypatch.setattr(_inst, "free_llama_swap", lambda *a, **k: "")
