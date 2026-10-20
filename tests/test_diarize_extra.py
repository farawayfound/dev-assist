"""Additional verifier-written tests for exocortex.speech.diarize.SpeakerClusterer.

Focused on boundary conditions, silent-failure modes, and rules not exercised by
tests/test_diarize.py: exact duration thresholds, non-finite embeddings, pre-existing
renames, and max_speakers edge cases. All offline, using the real speaker embedding model
via the `speaker_model_path` fixture (skipped automatically if the model file is absent)
with `_embed` monkeypatched for deterministic vectors where needed.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from exocortex.speech.diarize import SpeakerClusterer, _l2_normalize


@pytest.fixture()
def clusterer(speaker_model_path: str) -> SpeakerClusterer:
    return SpeakerClusterer(speaker_model_path)


# ---- _l2_normalize helper ----------------------------------------------------------------


def test_l2_normalize_unit_vector_is_idempotent() -> None:
    v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    out = _l2_normalize(v)
    assert out.dtype == np.float32
    assert np.isclose(float(np.linalg.norm(out)), 1.0, atol=1e-6)


def test_l2_normalize_near_zero_vector_returned_unchanged() -> None:
    v = np.zeros(4, dtype=np.float32)
    out = _l2_normalize(v)
    assert np.array_equal(out, v)


def test_l2_normalize_nan_vector_becomes_zero_not_nan() -> None:
    v = np.array([np.nan, 1.0, 0.0, 0.0], dtype=np.float32)
    out = _l2_normalize(v)
    assert np.all(np.isfinite(out)), "a NaN embedding must not propagate as NaN"
    assert np.array_equal(out, np.zeros(4, dtype=np.float32))


def test_l2_normalize_inf_vector_becomes_zero_not_nan() -> None:
    v = np.array([np.inf, 0.0, 0.0, 0.0], dtype=np.float32)
    out = _l2_normalize(v)
    assert np.all(np.isfinite(out))


# ---- non-finite embedding robustness (assign() must never raise / never wedge) -----------


def test_non_finite_embedding_keeps_cap_fallback_folding_working(
    speaker_model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once `max_speakers` is exhausted by NaN-derived embeddings, later calls must still fold
    into an existing centroid (per SPEC's cap-fallback rule) instead of permanently degrading
    to plain continuity with no centroid ever updated again.

    Before the fix, a NaN embedding was stored as a NaN centroid; every later cosine similarity
    against it was NaN, and `nan > x` is always False, so `_best_match` could never again select
    it as `best_id`. Once every one of the `max_speakers` slots was filled this way, `best_id`
    stayed None forever and the cap-fallback branch (`if best_id is not None: fold in`) could
    never fire - every subsequent call fell through to continuity only, silently unable to ever
    update a centroid's enrollment count again.
    """
    clusterer = SpeakerClusterer(speaker_model_path, threshold=0.5, enroll_min_s=1.0, max_speakers=2)
    nan_vec = np.array([np.nan, np.nan], dtype=np.float32)
    monkeypatch.setattr(clusterer, "_embed", lambda audio: nan_vec)
    long_audio = np.zeros(int(1.2 * 16000), dtype=np.float32)

    ids = [clusterer.assign(long_audio)[0] for _ in range(4)]

    assert ids[0] != ids[1], "the first two calls fill the two available speaker slots"
    assert len(clusterer.stats()) == 2, "max_speakers=2 must not be exceeded"
    # The key regression check: once the cap is hit, later calls must fold into (increment)
    # an existing centroid rather than being permanently stuck on continuity with counts frozen.
    counts_after = {sid: n for sid, _label, n in clusterer.stats()}
    assert sum(counts_after.values()) == 4, "every call must land on a real centroid, none lost to a NaN dead end"
    assert max(counts_after.values()) >= 2, "the cap-fallback must actually fold a call into an existing centroid"


def test_non_finite_embedding_never_raises(speaker_model_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    clusterer = SpeakerClusterer(speaker_model_path)
    monkeypatch.setattr(clusterer, "_embed", lambda audio: np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32))
    for _ in range(5):
        speaker_id, label = clusterer.assign(np.zeros(int(2.0 * 16000), dtype=np.float32))
        assert isinstance(speaker_id, str) and speaker_id
        assert isinstance(label, str) and label


# ---- exact duration boundaries -------------------------------------------------------------


def _counting_embed(calls: list[int]) -> Callable[[np.ndarray], np.ndarray]:
    def _embed(audio: np.ndarray) -> np.ndarray:
        calls.append(1)
        return np.array([1.0, 0.0], dtype=np.float32)

    return _embed


def test_min_embed_duration_boundary_is_not_skipped(speaker_model_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audio at exactly 0.3 s (the _MIN_EMBED_S guard) must still be embedded, not skipped."""
    clusterer = SpeakerClusterer(speaker_model_path, enroll_min_s=1.0)
    calls: list[int] = []
    monkeypatch.setattr(clusterer, "_embed", _counting_embed(calls))

    exactly_at_guard = np.zeros(int(0.3 * 16000), dtype=np.float32)
    clusterer.assign(exactly_at_guard)
    assert len(calls) == 1, "0.3s exactly is not 'shorter than 0.3s' and must still be embedded"


def test_just_under_min_embed_duration_is_skipped(speaker_model_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    clusterer = SpeakerClusterer(speaker_model_path, enroll_min_s=1.0)
    calls: list[int] = []
    monkeypatch.setattr(clusterer, "_embed", _counting_embed(calls))

    just_under = np.zeros(int(0.3 * 16000) - 1, dtype=np.float32)
    speaker_id, label = clusterer.assign(just_under)
    assert len(calls) == 0, "audio shorter than 0.3s must skip embedding entirely"
    assert speaker_id == "spk1" and label == "Speaker 1"


def test_enroll_min_s_boundary_creates_speaker(speaker_model_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audio at exactly enroll_min_s must be treated as enrollable (>= enroll_min_s), not short."""
    clusterer = SpeakerClusterer(speaker_model_path, threshold=0.5, enroll_min_s=1.0)
    monkeypatch.setattr(clusterer, "_embed", lambda audio: np.array([1.0, 0.0], dtype=np.float32))

    exactly_enroll_min = np.zeros(int(1.0 * 16000), dtype=np.float32)
    speaker_id, _ = clusterer.assign(exactly_enroll_min)
    assert clusterer.stats() == [("spk1", "Speaker 1", 1)], "exactly enroll_min_s must create/update a centroid"


# ---- embedding extraction failure -----------------------------------------------------------


def test_embed_raising_falls_back_to_continuity_without_crashing(
    speaker_model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clusterer = SpeakerClusterer(speaker_model_path, threshold=0.5, enroll_min_s=1.0)

    def _boom(audio: np.ndarray) -> np.ndarray:
        raise RuntimeError("onnxruntime exploded")

    monkeypatch.setattr(clusterer, "_embed", _boom)
    long_audio = np.zeros(int(1.2 * 16000), dtype=np.float32)

    speaker_id, label = clusterer.assign(long_audio)
    assert speaker_id == "spk1" and label == "Speaker 1"
    # A failed embedding must never create/update a centroid (continuity may still register a
    # default label for "spk1" with enrollment_count 0, per SpeakerClusterer.stats()'s contract).
    assert [s for s in clusterer.stats() if s[2] > 0] == []


# ---- rename() before the id has ever been produced by assign() -----------------------------


def test_rename_before_creation_is_preserved_when_speaker_is_later_created(
    speaker_model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clusterer = SpeakerClusterer(speaker_model_path, threshold=0.5, enroll_min_s=1.0)
    clusterer.rename("spk1", "Alice")

    monkeypatch.setattr(clusterer, "_embed", lambda audio: np.array([1.0, 0.0], dtype=np.float32))
    long_audio = np.zeros(int(1.2 * 16000), dtype=np.float32)

    speaker_id, label = clusterer.assign(long_audio)
    assert speaker_id == "spk1"
    assert label == "Alice", "a rename set before the speaker was ever created must survive creation"


# ---- max_speakers edge cases -----------------------------------------------------------------


def test_max_speakers_zero_never_creates_a_speaker(speaker_model_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    clusterer = SpeakerClusterer(speaker_model_path, threshold=0.5, enroll_min_s=1.0, max_speakers=0)
    monkeypatch.setattr(clusterer, "_embed", lambda audio: np.array([1.0, 0.0], dtype=np.float32))
    long_audio = np.zeros(int(1.2 * 16000), dtype=np.float32)

    for _ in range(3):
        speaker_id, label = clusterer.assign(long_audio)
        assert speaker_id == "spk1" and label == "Speaker 1"
    created = [s for s in clusterer.stats() if s[2] > 0]
    assert created == [], "max_speakers=0 must never create a real centroid"


def test_fold_into_existing_centroid_increments_enrollment_count(
    speaker_model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clusterer = SpeakerClusterer(speaker_model_path, threshold=0.5, enroll_min_s=1.0, max_speakers=1)
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)  # orthogonal: won't match v1 at threshold=0.5
    embeds = iter([v1, v2])
    monkeypatch.setattr(clusterer, "_embed", lambda audio: next(embeds))
    long_audio = np.zeros(int(1.2 * 16000), dtype=np.float32)

    id1, _ = clusterer.assign(long_audio)
    assert clusterer.stats() == [(id1, "Speaker 1", 1)]

    id2, _ = clusterer.assign(long_audio)
    assert id2 == id1, "with max_speakers=1 the only centroid absorbs every later utterance"
    assert clusterer.stats() == [(id1, "Speaker 1", 2)], "folding into the cap must still bump the count"


# ---- dtype / shape robustness -----------------------------------------------------------------


def test_assign_accepts_float64_and_int16_input(speaker_model_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    clusterer = SpeakerClusterer(speaker_model_path, enroll_min_s=1.0)
    monkeypatch.setattr(clusterer, "_embed", lambda audio: np.array([1.0, 0.0], dtype=np.float32))

    float64_audio = np.zeros(int(1.2 * 16000), dtype=np.float64)
    speaker_id, label = clusterer.assign(float64_audio)
    assert speaker_id and label

    int16_audio = np.zeros(int(1.2 * 16000), dtype=np.int16)
    speaker_id2, label2 = clusterer.assign(int16_audio)
    assert speaker_id2 and label2


def test_assign_accepts_column_vector_shape(speaker_model_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A (N, 1) mono array (as some audio libraries hand back) must reshape cleanly to (N,)."""
    clusterer = SpeakerClusterer(speaker_model_path, enroll_min_s=1.0)
    monkeypatch.setattr(clusterer, "_embed", lambda audio: np.array([1.0, 0.0], dtype=np.float32))

    column = np.zeros((int(1.2 * 16000), 1), dtype=np.float32)
    speaker_id, label = clusterer.assign(column)
    assert speaker_id and label
