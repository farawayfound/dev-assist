"""Tests for exocortex.speech.diarize.SpeakerClusterer.

All tests run offline on CPU using the real sherpa-onnx CAM++ embedding model in `models/`
(skipped automatically if that file is missing, via the `speaker_model_path` fixture) plus
`tests/data/two_speakers_16k.wav` and the ground-truth segment table in `tests/conftest.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

from exocortex.speech.diarize import SpeakerClusterer
from tests.conftest import TWO_SPEAKERS_TRUTH


def _slice(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    return audio[int(start * sr) : int(end * sr)]


@pytest.fixture()
def clusterer(speaker_model_path: str) -> SpeakerClusterer:
    return SpeakerClusterer(speaker_model_path)


# ---- real-audio truth-table tests -------------------------------------------------------------


def test_two_speakers_cluster_correctly(clusterer: SpeakerClusterer, two_speakers_audio: tuple[np.ndarray, int]) -> None:
    """Long utterances of A all get one id, long utterances of B all get another, exactly 2 speakers."""
    audio, sr = two_speakers_audio
    results: list[tuple[str, float, str, str]] = []

    for start, end, truth_speaker, _text in TWO_SPEAKERS_TRUTH:
        duration = end - start
        seg = _slice(audio, sr, start, end)
        speaker_id, label = clusterer.assign(seg)
        results.append((truth_speaker, duration, speaker_id, label))

    long_ids_a = {sid for spk, dur, sid, _label in results if spk == "A" and dur >= 1.5}
    long_ids_b = {sid for spk, dur, sid, _label in results if spk == "B" and dur >= 1.5}

    assert len(long_ids_a) == 1, f"speaker A's long utterances should map to one id, got {long_ids_a}"
    assert len(long_ids_b) == 1, f"speaker B's long utterances should map to one id, got {long_ids_b}"
    assert long_ids_a != long_ids_b, "speaker A and B should not share an id"

    created = [s for s in clusterer.stats() if s[2] > 0]
    assert len(created) == 2, f"expected exactly 2 speakers to be created, got {created}"


def test_short_utterances_never_create_a_speaker(
    clusterer: SpeakerClusterer, two_speakers_audio: tuple[np.ndarray, int]
) -> None:
    audio, sr = two_speakers_audio
    for start, end, _truth_speaker, _text in TWO_SPEAKERS_TRUTH:
        duration = end - start
        n_before = len({s for s in clusterer.stats() if s[2] > 0})
        seg = _slice(audio, sr, start, end)
        clusterer.assign(seg)
        n_after = len({s for s in clusterer.stats() if s[2] > 0})
        if duration < 1.5:
            assert n_after == n_before, "a short utterance must never create a new speaker"


def test_rename_reflects_in_labels_and_later_assign(
    clusterer: SpeakerClusterer, two_speakers_audio: tuple[np.ndarray, int]
) -> None:
    audio, sr = two_speakers_audio
    # Speaker A's first long utterance.
    start, end, _spk, _text = TWO_SPEAKERS_TRUTH[2]
    speaker_id, _label = clusterer.assign(_slice(audio, sr, start, end))

    clusterer.rename(speaker_id, "Alice")
    assert clusterer.labels()[speaker_id] == "Alice"

    # A later long utterance from the same speaker should come back with the new label.
    start2, end2, _spk2, _text2 = TWO_SPEAKERS_TRUTH[6]
    later_id, later_label = clusterer.assign(_slice(audio, sr, start2, end2))
    assert later_id == speaker_id
    assert later_label == "Alice"


def test_reset_clears_centroids_and_labels(
    clusterer: SpeakerClusterer, two_speakers_audio: tuple[np.ndarray, int]
) -> None:
    audio, sr = two_speakers_audio
    start, end, _spk, _text = TWO_SPEAKERS_TRUTH[2]
    speaker_id, _label = clusterer.assign(_slice(audio, sr, start, end))
    clusterer.rename(speaker_id, "Alice")

    clusterer.reset()

    assert clusterer.labels() == {}
    assert clusterer.stats() == []
    # Numbering starts over: the next long utterance becomes spk1/"Speaker 1" again, not "Alice".
    new_id, new_label = clusterer.assign(_slice(audio, sr, start, end))
    assert new_id == "spk1"
    assert new_label == "Speaker 1"


def test_random_noise_does_not_crash(clusterer: SpeakerClusterer) -> None:
    rng = np.random.default_rng(0)
    for duration_s in (0.05, 0.2, 0.5, 2.0, 4.0):
        audio = (rng.standard_normal(int(duration_s * 16000)).astype(np.float32)) * 0.05
        speaker_id, label = clusterer.assign(audio)
        assert isinstance(speaker_id, str) and speaker_id
        assert isinstance(label, str) and label


def test_empty_audio_does_not_crash(clusterer: SpeakerClusterer) -> None:
    speaker_id, label = clusterer.assign(np.zeros(0, dtype=np.float32))
    assert speaker_id == "spk1"
    assert label == "Speaker 1"


# ---- fake-extractor tests for enroll_min_s / max_speakers rules ------------------------------


def test_enroll_min_s_rule_with_fake_embeddings(speaker_model_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    clusterer = SpeakerClusterer(speaker_model_path, threshold=0.5, enroll_min_s=1.0)

    v_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_a_close = np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32)  # cos sim to v_a = 0.8 >= 0.5+0.15
    v_far = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # cos sim to v_a = 0.0

    embeds = iter([v_a, v_a_close, v_far])
    monkeypatch.setattr(clusterer, "_embed", lambda audio: next(embeds))

    long_audio = np.zeros(int(1.2 * 16000), dtype=np.float32)
    short_audio = np.zeros(int(0.5 * 16000), dtype=np.float32)  # >= 0.3s (not skipped) but < enroll_min_s

    spk1_id, _ = clusterer.assign(long_audio)
    assert spk1_id == "spk1"
    assert clusterer.stats() == [("spk1", "Speaker 1", 1)]

    # Short utterance close enough to spk1 (0.8 >= 0.65): matched, but must NOT update the centroid.
    matched_id, _ = clusterer.assign(short_audio)
    assert matched_id == "spk1"
    assert clusterer.stats() == [("spk1", "Speaker 1", 1)], "a short utterance must never update a centroid"

    # Short utterance far from spk1 (0.0 < 0.65): falls back to continuity (the most recent id, spk1),
    # and still creates nothing new.
    continuity_id, _ = clusterer.assign(short_audio)
    assert continuity_id == "spk1"
    assert clusterer.stats() == [("spk1", "Speaker 1", 1)]


def test_max_speakers_rule_with_fake_embeddings(speaker_model_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    clusterer = SpeakerClusterer(speaker_model_path, threshold=0.5, enroll_min_s=1.0, max_speakers=2)

    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)  # orthogonal to both v1 and v2

    embeds = iter([v1, v2, v3])
    monkeypatch.setattr(clusterer, "_embed", lambda audio: next(embeds))

    long_audio = np.zeros(int(1.2 * 16000), dtype=np.float32)

    id1, _ = clusterer.assign(long_audio)
    id2, _ = clusterer.assign(long_audio)
    assert {id1, id2} == {"spk1", "spk2"}
    assert id1 != id2

    # Cap reached: v3 matches neither existing centroid, but must be folded into one of them
    # instead of creating spk3.
    id3, _ = clusterer.assign(long_audio)
    assert id3 in {"spk1", "spk2"}

    created = [s for s in clusterer.stats() if s[2] > 0]
    assert len(created) == 2, f"max_speakers=2 must not be exceeded, got {created}"
