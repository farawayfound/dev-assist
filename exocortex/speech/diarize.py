"""Online speaker diarization via cosine clustering over sherpa-onnx speaker embeddings.

`SpeakerClusterer` keeps a small set of speaker "centroids" (L2-normalised mean embedding
vectors, one per speaker) and assigns each incoming utterance to the closest one, creating new
speakers on demand. State lives entirely in memory for the duration of a session; a meeting has
at most a handful of speakers so no persistence or approximate-nearest-neighbour machinery is
needed. Embedding extraction runs on CPU (~16 ms/utterance) via sherpa-onnx and is called from
the ASR worker thread only (see SPEC.md), so this class is not internally locked.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import sherpa_onnx

log = logging.getLogger(__name__)

_SPEAKER_NUM_RE = re.compile(r"(\d+)$")

# Utterances shorter than this are too short for sherpa-onnx to embed reliably (SPEC.md:
# "utterances < 1 s are unreliable"); skip embedding extraction entirely for them.
_MIN_EMBED_S = 0.3

# Extra margin required, on top of `threshold`, for a short-but-not-tiny utterance
# (< enroll_min_s) to be matched to an existing speaker instead of falling back to continuity.
_SHORT_MATCH_MARGIN = 0.15


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Scale `vec` to unit L2 norm (float32). A near-zero vector is returned unchanged.

    A non-finite vector (NaN/Inf, e.g. from a degenerate embedding on pathological audio) is
    replaced with all zeros rather than returned as-is: an unguarded NaN would otherwise be
    stored as a centroid and propagate into every later cosine similarity against it as NaN.
    Since `nan > x` is always False, `_best_match` would then never again treat that centroid
    as a candidate - once `max_speakers` is reached, an unmatched utterance is supposed to fold
    into the closest existing centroid (SPEC.md), but with a NaN centroid in the mix `best_id`
    can stay None forever, silently breaking that fallback (it degrades to plain continuity,
    permanently unable to update any centroid again). A zero vector instead compares as a
    clean, well-defined 0.0 similarity, so matching and folding keep working normally.
    """
    vec = np.asarray(vec, dtype=np.float32)
    if not np.all(np.isfinite(vec)):
        log.warning("speaker embedding contains non-finite values; treating as zero vector")
        return np.zeros_like(vec)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return vec
    return vec / norm


class SpeakerClusterer:
    """Online cosine clustering over sherpa-onnx speaker embeddings (L2-normalised).

    `assign(audio)` returns `(speaker_id, speaker_label)`. Rules:

    - audio shorter than 0.3 s: skip embedding extraction entirely and fall back straight to
      the continuity rule below (no centroid comparison is possible without an embedding).
    - duration < `enroll_min_s`: an embedding *is* computed but the utterance never creates or
      updates a speaker centroid. It is assigned to the best-matching centroid only when
      similarity >= `threshold + 0.15` (a stricter bar than a normal match, since short clips
      are unreliable); otherwise it falls back to the most recently assigned speaker
      ("continuity"), or "spk1" if nothing has been assigned yet.
    - duration >= `enroll_min_s`: assigned to the best-matching centroid when similarity >=
      `threshold`, updating that centroid as a re-normalised running mean
      (`c = normalize(c*n + e)`, `n` incremented); otherwise a new speaker is created
      (ids "spk1", "spk2", ... up to `max_speakers`). Once the cap is reached, an unmatched
      long utterance is folded into the closest existing centroid instead of creating a new one.

    Speaker labels default to "Speaker N" and can be overridden with `rename()`. `reset()`
    clears every centroid *and* every rename/label, so speaker numbering and labels both start
    over from scratch.
    """

    def __init__(
        self,
        model_path: str,
        *,
        threshold: float = 0.45,
        enroll_min_s: float = 1.5,
        max_speakers: int = 8,
        num_threads: int = 2,
        sample_rate: int = 16000,
    ) -> None:
        """Load the sherpa-onnx speaker embedding extractor from `model_path`."""
        self._threshold = threshold
        self._enroll_min_s = enroll_min_s
        self._max_speakers = max_speakers
        self._sample_rate = sample_rate
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path, num_threads=num_threads, provider="cpu"
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self._centroids: dict[str, tuple[np.ndarray, int]] = {}
        self._labels: dict[str, str] = {}
        self._last_id: str | None = None
        log.info(
            "SpeakerClusterer loaded model=%s dim=%d threshold=%.2f enroll_min_s=%.2f max_speakers=%d",
            model_path,
            self._extractor.dim,
            threshold,
            enroll_min_s,
            max_speakers,
        )

    # ---- public API ---------------------------------------------------------------------

    def assign(self, audio: np.ndarray) -> tuple[str, str]:
        """Assign one utterance of mono float32 audio (at `sample_rate`) to a speaker.

        Returns `(speaker_id, speaker_label)`. Never raises: embedding failures are logged and
        fall back to the continuity rule so the caller always gets a usable assignment.
        """
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration_s = audio.shape[0] / self._sample_rate

        if duration_s < _MIN_EMBED_S:
            return self._finish(self._continuity_id())

        try:
            embedding = _l2_normalize(self._embed(audio))
        except Exception:
            log.exception("speaker embedding extraction failed; falling back to continuity")
            return self._finish(self._continuity_id())

        best_id, best_sim = self._best_match(embedding)

        if duration_s < self._enroll_min_s:
            if best_id is not None and best_sim >= self._threshold + _SHORT_MATCH_MARGIN:
                return self._finish(best_id)
            return self._finish(self._continuity_id())

        if best_id is not None and best_sim >= self._threshold:
            self._update_centroid(best_id, embedding)
            return self._finish(best_id)

        if len(self._centroids) < self._max_speakers:
            new_id = f"spk{len(self._centroids) + 1}"
            self._centroids[new_id] = (embedding, 1)
            log.info("diarization: created %s (best existing sim=%.3f)", new_id, best_sim)
            return self._finish(new_id)

        if best_id is not None:
            log.debug("diarization: max_speakers reached, folding into %s (sim=%.3f)", best_id, best_sim)
            self._update_centroid(best_id, embedding)
            return self._finish(best_id)

        return self._finish(self._continuity_id())

    def rename(self, speaker_id: str, label: str) -> None:
        """Set a user-chosen label for `speaker_id`; remembered until the next `reset()`."""
        self._labels[speaker_id] = label

    def labels(self) -> dict[str, str]:
        """Return a copy of the current `{speaker_id: label}` mapping for every known speaker."""
        return dict(self._labels)

    def stats(self) -> list[tuple[str, str, int]]:
        """Return `(speaker_id, label, enrollment_count)` per known speaker, for debugging.

        `enrollment_count` is 0 for a speaker id that has only ever been produced by the
        continuity fallback and has no centroid of its own yet.
        """
        ids = set(self._labels) | set(self._centroids)
        return [
            (sid, self._labels.get(sid, self._default_label(sid)), self._centroids.get(sid, (None, 0))[1])
            for sid in sorted(ids, key=self._sort_key)
        ]

    def reset(self) -> None:
        """Clear every speaker centroid and every rename. Labels revert to "Speaker N" defaults."""
        self._centroids.clear()
        self._labels.clear()
        self._last_id = None

    # ---- internals ------------------------------------------------------------------------

    def _embed(self, audio: np.ndarray) -> np.ndarray:
        """Run the sherpa-onnx extractor once and return the raw embedding vector.

        Kept as its own method (rather than inlined into `assign`) so tests can monkeypatch
        this single call site with fixed vectors instead of loading real audio through ONNX.
        """
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=self._sample_rate, waveform=audio)
        stream.input_finished()
        return np.asarray(self._extractor.compute(stream), dtype=np.float32)

    def _best_match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Return the id and cosine similarity of the closest centroid, or `(None, -1.0)`."""
        best_id: str | None = None
        best_sim = -1.0
        for speaker_id, (centroid, _count) in self._centroids.items():
            sim = float(np.dot(centroid, embedding))
            if sim > best_sim:
                best_id, best_sim = speaker_id, sim
        return best_id, best_sim

    def _update_centroid(self, speaker_id: str, embedding: np.ndarray) -> None:
        """Fold `embedding` into `speaker_id`'s centroid via a re-normalised running mean."""
        centroid, count = self._centroids[speaker_id]
        merged = centroid * count + embedding
        self._centroids[speaker_id] = (_l2_normalize(merged), count + 1)

    def _continuity_id(self) -> str:
        """The most recently assigned speaker id, or "spk1" if nothing has been assigned yet."""
        return self._last_id if self._last_id is not None else "spk1"

    def _finish(self, speaker_id: str) -> tuple[str, str]:
        """Register a default label for `speaker_id` if new, remember it, return `(id, label)`."""
        if speaker_id not in self._labels:
            self._labels[speaker_id] = self._default_label(speaker_id)
        self._last_id = speaker_id
        return speaker_id, self._labels[speaker_id]

    @staticmethod
    def _default_label(speaker_id: str) -> str:
        match = _SPEAKER_NUM_RE.search(speaker_id)
        return f"Speaker {match.group(1)}" if match else speaker_id

    @staticmethod
    def _sort_key(speaker_id: str) -> tuple[int, int | str]:
        match = _SPEAKER_NUM_RE.search(speaker_id)
        return (0, int(match.group(1))) if match else (1, speaker_id)
