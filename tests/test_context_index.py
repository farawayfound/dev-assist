"""Tests for exocortex.context: tokenizer, query_terms, ContextIndex (BM25) and TokenEstimator."""
from __future__ import annotations

import threading

import pytest

from exocortex.context import ContextIndex, ContextStore, TokenEstimator, chunk_text, query_terms, tokenize


def _add(store: ContextStore, name: str, text: str, mode: str = "reference"):
    return store.add_text(name, text, mode=mode, max_chars=100000, digest_wanted=False)


ARCH = """# Architecture

The audio pipeline captures system audio and the microphone, then streams frames to the speech recognizer.

# Threads

Each worker thread owns a dedicated lock. The assistant worker serialises answer jobs.

# Storage

Sessions are written atomically to the output folder as markdown and jsonl files.
"""

OPS = """# Deployment

The installer ships a bundled llama server runtime and downloads the model on first launch.

# Troubleshooting

If the GPU runs out of memory lower the context size or pick a smaller quantisation.
"""


@pytest.fixture()
def store() -> ContextStore:
    s = ContextStore(None)
    _add(s, "arch.md", ARCH)
    _add(s, "ops.md", OPS)
    return s


# ---- tokenizer / query_terms ---------------------------------------------------

def test_headings_only_document_keeps_every_heading_in_order() -> None:
    """Review 2026-10-06: a document that is nothing but headings yields one chunk of ALL of them
    (sibling headings used to be lost with the heading stack), so each stays retrievable."""
    chunks = chunk_text("# A\n# B\n## C\n## D", max_chars=500, doc_id="d", doc_name="h.md")
    assert len(chunks) == 1
    assert chunks[0].text.split("\n") == ["A", "B", "C", "D"]


def test_tokenize_casefolds_drops_short_and_stop_words() -> None:
    assert tokenize("The Quick brown FOX, a b of it!") == ["quick", "brown", "fox"]


def test_tokenize_splits_on_non_alphanumerics_and_underscore() -> None:
    assert tokenize("lookup_top_k=3; ctx-size") == ["lookup", "top", "ctx", "size"]


def test_tokenize_crude_stemming() -> None:
    assert tokenize("services") == ["service"]
    assert tokenize("running") == ["runn"]       # crude on purpose, but deterministic
    assert tokenize("jumped") == ["jump"]
    assert tokenize("classes") == ["class"]
    assert tokenize("class") == ["class"]
    assert tokenize("uses") == ["uses"]           # stem would be < 4 chars: left alone
    assert tokenize("bed") == ["bed"]


def test_tokenize_is_deterministic() -> None:
    text = "Threads and locks: the assistant worker serialises answer jobs."
    assert tokenize(text) == tokenize(text)


def test_query_terms_weights() -> None:
    q = query_terms("deployment installer", "[00:05] Speaker 1: gpu memory", "screen words")
    assert q["deployment"] == 1.0 and q["installer"] == 1.0
    assert q["gpu"] == 0.5 and q["memory"] == 0.5          # the "[00:05] Speaker 1:" prefix is not a topic
    assert q["screen"] == 0.25 and q["word"] == 0.25
    assert "speaker" not in q and "00" not in q


def test_query_terms_takes_the_max_weight_when_a_term_repeats() -> None:
    q = query_terms("threads", "threads locks", "locks threads")
    assert q["thread"] == 1.0 and q["lock"] == 0.5


def test_query_terms_empty_inputs() -> None:
    assert query_terms("") == {}
    assert query_terms("the a of") == {}


# ---- ContextIndex.search -------------------------------------------------------

def test_best_passage_ranks_first(store: ContextStore) -> None:
    idx = ContextIndex(store, chunk_chars=300)
    hits = idx.search(query_terms("how are the worker threads and locks organised"), top_k=3, min_score=0.1)
    assert hits and hits[0][0].title == "Threads" and hits[0][0].doc_name == "arch.md"
    scores = [s for _c, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_other_document_found_by_its_own_terms(store: ContextStore) -> None:
    idx = ContextIndex(store, chunk_chars=300)
    hits = idx.search(query_terms("what if the GPU runs out of memory"), top_k=1, min_score=0.1)
    assert hits[0][0].doc_name == "ops.md" and hits[0][0].title == "Troubleshooting"


def test_min_score_filters_everything_when_too_high(store: ContextStore) -> None:
    idx = ContextIndex(store, chunk_chars=300)
    q = query_terms("worker threads")
    assert idx.search(q, top_k=3, min_score=0.1)
    assert idx.search(q, top_k=3, min_score=1000.0) == []


def test_unrelated_query_matches_nothing(store: ContextStore) -> None:
    idx = ContextIndex(store, chunk_chars=300)
    assert idx.search(query_terms("pineapple pizza topping"), top_k=3, min_score=0.0001) == []


def test_empty_query_and_zero_top_k(store: ContextStore) -> None:
    idx = ContextIndex(store, chunk_chars=300)
    assert idx.search({}, top_k=3, min_score=0.0) == []
    assert idx.search(query_terms("threads"), top_k=0, min_score=0.0) == []


def test_one_passage_per_document_when_enough_documents_match() -> None:
    s = ContextStore(None)
    _add(s, "a.md", "# One\nrelay protocol details\n\n# Two\nrelay protocol again relay\n")
    _add(s, "b.md", "# Only\nrelay protocol summary\n")
    idx = ContextIndex(s, chunk_chars=200)
    hits = idx.search(query_terms("relay protocol"), top_k=2, min_score=0.0)
    assert sorted(c.doc_name for c, _s in hits) == ["a.md", "b.md"]


def test_more_passages_per_document_only_when_fewer_documents_match_than_top_k() -> None:
    s = ContextStore(None)
    _add(s, "a.md", "# One\nrelay protocol details\n\n# Two\nrelay protocol again relay\n\n# Three\nrelay notes")
    idx = ContextIndex(s, chunk_chars=200)
    hits = idx.search(query_terms("relay protocol"), top_k=3, min_score=0.0)
    assert len(hits) == 3 and {c.doc_name for c, _s in hits} == {"a.md"}
    assert len(idx.search(query_terms("relay protocol"), top_k=1, min_score=0.0)) == 1


def test_only_enabled_reference_documents_are_indexed() -> None:
    s = ContextStore(None)
    ref = _add(s, "ref.md", "# Topic\nkubernetes cluster autoscaling")
    inj = _add(s, "inj.md", "# Topic\nkubernetes cluster autoscaling", mode="inject")
    idx = ContextIndex(s, chunk_chars=200)
    q = query_terms("kubernetes autoscaling")
    assert [c.doc_id for c, _s in idx.search(q, top_k=5, min_score=0.0)] == [ref.id]
    s.set_enabled(ref.id, False)
    assert idx.search(q, top_k=5, min_score=0.0) == []          # lazily rebuilt after the store changed
    s.set_enabled(ref.id, True)
    s.set_mode(inj.id, "reference", digest_wanted=False)
    assert {c.doc_id for c, _s in idx.search(q, top_k=5, min_score=0.0)} == {ref.id, inj.id}


def test_index_rebuilds_after_add_and_remove(store: ContextStore) -> None:
    idx = ContextIndex(store, chunk_chars=300)
    q = query_terms("zeppelin hangar")
    assert idx.search(q, top_k=3, min_score=0.0) == []
    d = _add(store, "air.md", "# Hangars\nThe zeppelin hangar is large.")
    assert [c.doc_id for c, _s in idx.search(q, top_k=3, min_score=0.0)] == [d.id]
    store.remove(d.id)
    assert idx.search(q, top_k=3, min_score=0.0) == []


def test_chunk_chars_change_rebuilds(store: ContextStore) -> None:
    idx = ContextIndex(store, chunk_chars=2000)
    before = idx.size
    idx.chunk_chars = 40
    assert idx.size > before


def test_heading_words_are_searchable() -> None:
    s = ContextStore(None)
    _add(s, "a.md", "# Zebra Crossing\nWalk when the light is green.\n\n# Misc\nSomething else entirely.")
    idx = ContextIndex(s, chunk_chars=200)
    hits = idx.search(query_terms("zebra"), top_k=1, min_score=0.0)
    assert hits and hits[0][0].title == "Zebra Crossing"


def test_ties_break_by_document_then_chunk_order() -> None:
    s = ContextStore(None)
    first = _add(s, "first.md", "# H\nsame words here")
    second = _add(s, "second.md", "# H\nsame words here but");      # different text -> different id
    idx = ContextIndex(s, chunk_chars=200)
    hits = idx.search({"same": 1.0, "words": 1.0}, top_k=2, min_score=0.0)
    ranked = [c.doc_id for c, _s in hits]
    assert set(ranked) == {first.id, second.id}
    # identical queries -> identical ordering, every time (pure and deterministic)
    for _ in range(5):
        assert [c.doc_id for c, _s in idx.search({"same": 1.0, "words": 1.0}, top_k=2, min_score=0.0)] == ranked


def test_query_term_weight_scales_the_score(store: ContextStore) -> None:
    idx = ContextIndex(store, chunk_chars=300)
    (_c1, s1), = idx.search({"thread": 1.0}, top_k=1, min_score=0.0)
    (_c2, s2), = idx.search({"thread": 0.5}, top_k=1, min_score=0.0)
    assert s1 == pytest.approx(2 * s2)


def test_bm25_prefers_the_shorter_passage_for_equal_term_counts() -> None:
    s = ContextStore(None)
    _add(s, "short.md", "# S\nflux capacitor")
    _add(s, "long.md", "# L\nflux capacitor " + "filler words that pad this passage out considerably " * 6)
    idx = ContextIndex(s, chunk_chars=2000)
    hits = idx.search({"flux": 1.0}, top_k=2, min_score=0.0)
    assert [c.doc_name for c, _s in hits] == ["short.md", "long.md"]


def test_concurrent_searches_and_store_changes_do_not_crash(store: ContextStore) -> None:
    idx = ContextIndex(store, chunk_chars=300)
    errors: list[Exception] = []

    def search() -> None:
        try:
            for _ in range(30):
                idx.search(query_terms("threads locks"), top_k=3, min_score=0.0)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def mutate() -> None:
        try:
            for i in range(30):
                _add(store, f"x{i}.md", f"# T{i}\nthreads locks number {i}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=search), threading.Thread(target=search), threading.Thread(target=mutate)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors


# ---- TokenEstimator ------------------------------------------------------------

def test_tokens_and_chars_basic() -> None:
    est = TokenEstimator(4.0, auto=True)
    assert est.tokens("") == 0
    assert est.tokens("x") == 2                  # ceil(1/4) + 1
    assert est.tokens("x" * 400) == 101
    assert est.tokens_for_len(0) == 0
    assert est.chars(100) == 400 and est.chars(0) == 0 and est.chars(-5) == 0


def test_initial_value_is_clamped() -> None:
    assert TokenEstimator(0.5, auto=True).chars_per_token == 2.0
    assert TokenEstimator(99, auto=True).chars_per_token == 6.0
    assert TokenEstimator(float("nan"), auto=True).chars_per_token == 3.6


def test_observe_moves_toward_the_sample_by_ema() -> None:
    est = TokenEstimator(4.0, auto=True)
    est.observe(3000, 1000)                       # sample 3.0
    assert est.chars_per_token == pytest.approx(0.7 * 4.0 + 0.3 * 3.0)
    for _ in range(60):
        est.observe(3000, 1000)
    assert est.chars_per_token == pytest.approx(3.0, abs=0.01)


def test_observe_is_clamped_to_the_range() -> None:
    low = TokenEstimator(4.0, auto=True)
    for _ in range(80):
        low.observe(1000, 5000)                   # 0.2 chars/token
    assert low.chars_per_token == pytest.approx(2.0, abs=0.01) and low.chars_per_token >= 2.0
    high = TokenEstimator(4.0, auto=True)
    for _ in range(80):
        high.observe(50000, 100)                  # 500 chars/token
    assert high.chars_per_token <= 6.0 and high.chars_per_token == pytest.approx(6.0, abs=0.01)


def test_observe_ignored_when_auto_off_or_numbers_tiny_or_missing() -> None:
    off = TokenEstimator(4.0, auto=False)
    off.observe(3000, 1000)
    assert off.chars_per_token == 4.0
    on = TokenEstimator(4.0, auto=True)
    on.observe(49, 1000)
    on.observe(3000, 49)
    on.observe(3000, None)  # type: ignore[arg-type]
    assert on.chars_per_token == 4.0


def test_auto_flag_can_be_flipped_live() -> None:
    est = TokenEstimator(4.0, auto=False)
    est.auto = True
    est.observe(3000, 1000)
    assert est.chars_per_token < 4.0


def test_note_overflow_shrinks_by_15_percent_and_floors_at_two() -> None:
    est = TokenEstimator(4.0, auto=True)
    est.note_overflow()
    assert est.chars_per_token == pytest.approx(3.4)
    for _ in range(20):
        est.note_overflow()
    assert est.chars_per_token == 2.0


def test_note_overflow_applies_even_with_calibration_off() -> None:
    est = TokenEstimator(4.0, auto=False)
    est.note_overflow()
    assert est.chars_per_token == pytest.approx(3.4)


def test_set_base_resets_and_clamps() -> None:
    est = TokenEstimator(4.0, auto=True)
    est.set_base(3.0)
    assert est.chars_per_token == 3.0
    est.set_base(100)
    assert est.chars_per_token == 6.0


def test_estimator_is_thread_safe() -> None:
    est = TokenEstimator(4.0, auto=True)

    def work() -> None:
        for _ in range(200):
            est.observe(3000, 1000)
            est.note_overflow()
            est.tokens("hello world")

    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert 2.0 <= est.chars_per_token <= 6.0


def test_a_single_coincidental_term_needs_twice_the_floor() -> None:
    """Review 2026-10-06: small talk ("thanks everyone, see you next week") pulled README
    passages in on one rare word. One matched term now needs twice `min_score`; two terms do not."""
    store = ContextStore(None)
    store.add_text("readme.md", "# Thanks\n\nThanks to every contributor.\n\n# Queue\n\nThe message queue "
                   "handles retries and backpressure for the ingestion service.\n\n# Other\n\nUnrelated filler "
                   "text about nothing in particular, repeated. " * 3, mode="reference", max_chars=100000,
                   digest_wanted=False)
    index = ContextIndex(store, chunk_chars=200)
    one_term = index.search({"thank": 1.0}, top_k=3, min_score=0.0)
    assert one_term and one_term[0][1] > 0
    floor = one_term[0][1] * 0.75                   # below the single-term score, above half of it
    assert index.search({"thank": 1.0}, top_k=3, min_score=floor) == []            # one term: needs 2x floor
    assert index.search({"thank": 1.0}, top_k=3, min_score=floor / 2.0)            # ... which this clears
    two = index.search({t: 1.0 for t in tokenize("queue retries")}, top_k=3, min_score=floor)   # two terms: the floor is enough
    assert two and "retries" in two[0][0].text
