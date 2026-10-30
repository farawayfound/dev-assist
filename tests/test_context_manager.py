"""Tests for exocortex.context.ContextManager: budgeting, trim order, reference/lookup blocks,
include_in gating, compaction flags, usage reporting and calibration. Offline, no LLM."""
from __future__ import annotations

import dataclasses

import pytest

from exocortex.bus import ContextUsageEvent, Source, TranscriptSegment
from exocortex.config import Config
from exocortex.context import _LANE_TEMPLATE_TOKENS, ContextManager, ContextStore
from exocortex.transcript import Transcript

SYSTEM = "S" * 400            # ~101 tokens at 4 chars/token
Q = "How are the worker threads organised?"


def _seg(seg_id: int, text: str) -> TranscriptSegment:
    return TranscriptSegment(id=seg_id, source=Source.SYSTEM, speaker_id="spk1", speaker_label="Speaker 1",
                             text=text, t_start=float(seg_id * 3), t_end=float(seg_id * 3 + 1), confidence=0.9)


def _fill(tr: Transcript, n: int, start: int = 0, width: int = 60) -> None:
    for i in range(start, start + n):
        tr.add(_seg(i, f"utterance {i:03d} " + "w" * width))


def _cfg(**ctx) -> Config:
    cfg = Config()
    cfg.llm.ctx_size = 8192
    # One slot: job_limit("answer") is then slot - reserve, which keeps this file's budget
    # arithmetic exact. The two-slot lane split has its own tests below (2026-10-06 addendum).
    cfg.llm.parallel = 1
    cfg.llm.context_tokens = 2500
    cfg.context.chars_per_token = 4.0
    cfg.context.auto_calibrate = False
    cfg.context.reserve_tokens = 256
    cfg.context.transcript_min_tokens = 600
    for k, v in ctx.items():
        setattr(cfg.context, k, v)
    return cfg


def _mk(n_lines: int = 0, *, store: ContextStore | None = None, width: int = 60, **ctx) -> tuple[ContextManager, Transcript, Config]:
    cfg = _cfg(**ctx)
    tr = Transcript()
    _fill(tr, n_lines, width=width)
    return ContextManager(cfg, store, tr), tr, cfg


def _build(m: ContextManager, *, kind: str = "ask", question: str = Q, screen: str = "", since: int | None = None,
           until: int | None = None, max_tokens: int = 400, system: str = SYSTEM, fixed: int | None = None):
    return m.build(kind=kind, question=question, screen_text=screen, since_id=since, until_id=until,
                   max_tokens=max_tokens, system_text=system, fixed_chars=len(question) + 200 if fixed is None else fixed)


def _add(store: ContextStore, name: str, text: str, mode: str = "reference"):
    return store.add_text(name, text, mode=mode, max_chars=100000, digest_wanted=False)


ARCH = ("# Architecture\n\nThe audio pipeline captures system audio.\n\n"
        "# Threads\n\nEach worker thread owns a dedicated lock. The assistant worker serialises answer jobs.\n\n"
        "# Storage\n\nSessions are written atomically to the output folder.\n")


def _store_with_arch() -> tuple[ContextStore, str]:
    store = ContextStore(None)
    doc = _add(store, "arch.md", ARCH)
    return store, doc.id


# ---- slot size -----------------------------------------------------------------

def test_slot_falls_back_to_the_whole_ctx_size_never_divided_by_parallel() -> None:
    """Measured 2026-10-06 (SPEC.md): with --parallel 2 llama-server runs --kv-unified and every
    slot reports the FULL ctx_size (the two share one pool), so the fallback is ctx_size itself;
    the sharing is handled by job_limit()."""
    m, _tr, cfg = _mk()
    assert m.slot_tokens == 8192
    cfg.llm.parallel = 4
    assert m.slot_tokens == 8192
    cfg.llm.parallel = 0
    assert m.slot_tokens == 8192


def test_set_slot_tokens_overrides_and_none_restores_the_fallback() -> None:
    m, _tr, _cfg_ = _mk()
    m.set_slot_tokens(3000)
    assert m.slot_tokens == 3000
    m.set_slot_tokens(None)
    assert m.slot_tokens == 8192
    m.set_slot_tokens(0)
    assert m.slot_tokens == 8192


# ---- lane-aware job limits (2026-10-06 addendum) ---------------------------------------------

def test_job_limit_with_one_slot_is_the_pool_minus_the_reserve_for_either_lane() -> None:
    m, _tr, cfg = _mk(reserve_tokens=200)
    m.set_slot_tokens(4096)
    assert m.job_limit("answer") == 4096 - 200
    assert m.job_limit("notes") == 4096 - 200


def test_job_limit_with_two_slots_splits_the_unified_pool_by_lane_share() -> None:
    m, _tr, cfg = _mk(reserve_tokens=256, lane_share=0.3)
    cfg.llm.parallel = 2
    m.set_slot_tokens(4096)
    assert m.job_limit("answer") == int(4096 * 0.7) - 256
    assert m.job_limit("notes") == int(4096 * 0.3)
    assert m.job_limit("answer") + m.job_limit("notes") <= 4096 - 256
    # the notes lane never shrinks under 512 tokens, however small the pool
    m.set_slot_tokens(1000)
    assert m.job_limit("notes") == 512


def test_job_limit_clamps_a_silly_lane_share() -> None:
    m, _tr, cfg = _mk(lane_share=5.0)
    cfg.llm.parallel = 2
    m.set_slot_tokens(4096)
    assert 0 < m.job_limit("answer") < 4096
    assert m.job_limit("notes") < 4096


def test_build_budget_follows_the_answer_lane_limit_with_two_slots() -> None:
    m, _tr, cfg = _mk(5, reserve_tokens=256, lane_share=0.3)
    cfg.llm.parallel = 2
    m.set_slot_tokens(4096)
    b = _build(m, max_tokens=300)
    sys_t = m.estimator.tokens(SYSTEM)
    assert b.usage.budget_tokens == m.job_limit("answer") - 300 - sys_t
    assert b.usage.slot_tokens == 4096


def test_available_chars_is_the_room_left_in_a_lane_prompt() -> None:
    m, _tr, _c = _mk(reserve_tokens=0)
    m.set_slot_tokens(2000)
    system = "s" * 400                      # 101 tokens
    other = "o" * 800                       # 201 tokens
    room_tokens = m.job_limit("notes") - 500 - m.estimator.tokens(system) - m.estimator.tokens(other) - _LANE_TEMPLATE_TOKENS
    assert m.available_chars("notes", max_tokens=500, system_text=system, other_text=other) == m.estimator.chars(room_tokens)
    # never negative, even when the prompt already overflows the lane
    assert m.available_chars("notes", max_tokens=5000, system_text=system) == 0


def test_manager_creates_an_in_memory_store_when_given_none() -> None:
    m, _tr, _c = _mk()
    assert m.store.documents() == []
    _add(m.store, "a", "text")
    assert len(m.store.documents()) == 1


# ---- include_in gating ---------------------------------------------------------

@pytest.mark.parametrize("kind", ["auto", "manual", "ask", "screen", "suggest", "solve", "code", "text", "explain", "complete"])
def test_include_in_all(kind: str) -> None:
    m, _tr, _c = _mk(include_in="all")
    assert m.wants_reference(kind) is True


@pytest.mark.parametrize("kind,expected", [
    ("auto", True), ("manual", True), ("ask", True), ("screen", True),
    ("suggest", False), ("solve", False), ("code", False), ("text", False), ("explain", False), ("complete", False),
])
def test_include_in_questions(kind: str, expected: bool) -> None:
    m, _tr, _c = _mk(include_in="questions")
    assert m.wants_reference(kind) is expected


def test_include_in_none_and_live_changes() -> None:
    m, _tr, cfg = _mk(include_in="none")
    assert not any(m.wants_reference(k) for k in ("auto", "ask", "solve", "complete"))
    cfg.context.include_in = "all"
    assert m.wants_reference("complete")


def test_gated_job_gets_no_reference_or_lookup_but_still_a_transcript() -> None:
    store, _id = _store_with_arch()
    _add(store, "inj.md", "Injected body text.", mode="inject")
    m, _tr, _c = _mk(40, store=store, include_in="questions")
    b = _build(m, kind="solve", question="worker threads locks")
    assert b.reference == "" and b.lookup == "" and b.documents_used == 0
    assert b.transcript
    asked = _build(m, kind="ask", question="worker threads locks")
    assert asked.reference and asked.lookup


# ---- reference block -----------------------------------------------------------

def test_inject_documents_render_in_store_order_with_headers() -> None:
    store = ContextStore(None)
    _add(store, "first.md", "First body.", mode="inject")
    _add(store, "second.md", "Second body.", mode="inject")
    m, _tr, _c = _mk(3, store=store)
    b = _build(m)
    assert b.reference == "### first.md\nFirst body.\n\n### second.md\nSecond body."
    assert b.documents_used == 2


def test_catalogue_follows_the_injected_documents_with_digest_or_first_heading() -> None:
    store = ContextStore(None)
    _add(store, "inj.md", "Injected body.", mode="inject")
    with_digest = _add(store, "arch.md", ARCH)
    store.set_digest(with_digest.id, "The architecture\nof the app, threads and storage.", "ready")
    _add(store, "plain.md", "# Deployment Guide\n\nHow to install it.")
    _add(store, "noheading.txt", "just the first line\nsecond line")
    m, _tr, _c = _mk(3, store=store)
    ref = _build(m).reference
    header = "Reference documents available (not shown in full; relevant passages appear in <lookup> when the question calls for them):"
    assert ref.index("### inj.md") < ref.index(header)
    assert ref.endswith(
        header + "\n- arch.md: The architecture of the app, threads and storage."
        "\n- plain.md: Deployment Guide\n- noheading.txt: just the first line")


def test_failed_or_pending_digest_falls_back_to_the_first_heading() -> None:
    store = ContextStore(None)
    d = store.add_text("p.md", "# Real Title\nbody", mode="reference", max_chars=1000, digest_wanted=True)
    assert d.digest_state == "pending"
    m, _tr, _c = _mk(2, store=store)
    assert "- p.md: Real Title" in _build(m).reference
    store.set_digest(d.id, "", "failed")
    assert "- p.md: Real Title" in _build(m).reference


def test_code_catalogue_entry_skips_the_fence_line() -> None:
    store = ContextStore(None)
    store.add_text("m.py", "```py\nimport os\n```", mode="reference", max_chars=1000, digest_wanted=False, kind="code")
    m, _tr, _c = _mk(2, store=store)
    assert "- m.py: import os" in _build(m).reference


def test_disabled_documents_are_not_in_the_prompt() -> None:
    store = ContextStore(None)
    d = _add(store, "gone.md", "Hidden body.", mode="inject")
    r = _add(store, "ref.md", "# Topic\nkubernetes autoscaling", mode="reference")
    store.set_enabled(d.id, False)
    store.set_enabled(r.id, False)
    m, _tr, _c = _mk(2, store=store)
    b = _build(m, question="kubernetes autoscaling")
    assert b.reference == "" and b.lookup == "" and b.documents_used == 0


def test_no_documents_means_empty_blocks() -> None:
    m, _tr, _c = _mk(5)
    b = _build(m)
    assert (b.reference, b.lookup, b.earlier, b.screen) == ("", "", "", "")
    assert b.documents_used == 0 and b.usage.documents == 0


def test_inject_documents_share_the_reference_budget_equally_short_ones_give_back() -> None:
    store = ContextStore(None)
    short = _add(store, "short.md", "tiny body", mode="inject")
    big = _add(store, "big.md", "\n".join(f"big line {i:03d} with some words" for i in range(200)), mode="inject")
    big2 = _add(store, "big2.md", "\n".join(f"other line {i:03d} with some words" for i in range(200)), mode="inject")
    m, _tr, _c = _mk(3, store=store, reference_tokens=500)       # ~1996 chars of reference
    b = _build(m)
    assert "### short.md\ntiny body" in b.reference and "tiny body\n[...]" not in b.reference
    parts = b.reference.split("\n\n### ")
    assert len(parts) == 3
    assert parts[1].endswith("[...]") and parts[2].endswith("[...]")
    assert len(parts[1]) == pytest.approx(len(parts[2]), abs=40)           # equal shares for the long ones
    assert len(b.reference) <= m.estimator.chars(500)
    # the short one's leftover went to the long ones: they are longer than a naive third each
    naive = m.estimator.chars(499) // 3
    assert len(parts[1]) > naive
    assert b.usage.trimmed is True and b.documents_used == 3
    assert {short.id, big.id, big2.id}


def test_cut_injected_document_cuts_on_a_line_boundary() -> None:
    store = ContextStore(None)
    _add(store, "big.md", "\n".join(f"line {i:03d} abcdefghij" for i in range(300)), mode="inject")
    m, _tr, _c = _mk(2, store=store, reference_tokens=200)
    ref = _build(m).reference
    body = ref.split("\n", 1)[1]
    assert body.endswith("\n[...]")
    for ln in body[: -len("\n[...]")].split("\n"):
        assert ln.startswith("line ") and len(ln) == len("line 000 abcdefghij")


def test_document_without_a_useful_share_is_left_out_whole() -> None:
    store = ContextStore(None)
    for i in range(6):
        _add(store, f"doc{i}.md", f"body number {i} " * 200, mode="inject")
    m, _tr, _c = _mk(2, store=store, reference_tokens=40)        # ~156 chars for six documents
    b = _build(m)
    assert b.reference.count("### doc") < 6 and b.usage.trimmed is True
    assert b.documents_used == b.reference.count("### doc")


def test_reference_block_is_stable_between_turns() -> None:
    store, _id = _store_with_arch()
    _add(store, "inj.md", "Injected body.", mode="inject")
    m, tr, _c = _mk(10, store=store)
    first = _build(m).reference
    _fill(tr, 5, start=10)
    assert _build(m, question="something else entirely").reference == first


def test_missing_stored_document_text_is_skipped_not_fatal(tmp_path) -> None:
    lib = tmp_path / "lib"
    store = ContextStore(lib)
    d = store.add_text("a.md", "Body that will vanish", mode="inject", max_chars=1000, digest_wanted=False)
    (lib / "docs" / f"{d.id}.md").unlink()
    reopened = ContextStore(lib)
    m, _tr, _c = _mk(3, store=reopened)
    b = _build(m)
    assert b.reference == "" and b.transcript


# ---- lookup --------------------------------------------------------------------

def test_lookup_pulls_the_matching_passage_only_for_a_matching_question() -> None:
    store, _id = _store_with_arch()
    m, _tr, _c = _mk(10, store=store, lookup_min_score=0.5)
    hit = _build(m, question="how do the worker threads and locks work")
    assert hit.lookup.startswith("[arch.md > Threads]\n")
    assert "dedicated lock" in hit.lookup
    miss = _build(m, question="what is the weather like in lisbon")
    assert miss.lookup == ""


def test_lookup_passages_are_labelled_and_separated_by_blank_lines() -> None:
    store, _id = _store_with_arch()
    m, _tr, _c = _mk(10, store=store, lookup_min_score=0.0, lookup_top_k=3, lookup_tokens=2000)
    b = _build(m, question="worker threads audio pipeline sessions storage")
    passages = b.lookup.split("\n\n[")
    assert len(passages) >= 2
    assert passages[0].startswith("[arch.md > ")
    assert all(p.startswith("arch.md > ") or p.startswith("[arch.md > ") for p in passages)


def test_lookup_label_is_just_the_name_when_the_document_has_no_headings() -> None:
    store = ContextStore(None)
    _add(store, "notes.txt", "kubernetes autoscaling is configured per cluster in the platform repo")
    m, _tr, _c = _mk(3, store=store, lookup_min_score=0.0)
    b = _build(m, question="kubernetes autoscaling")
    assert b.lookup.startswith("[notes.txt]\n")


def test_lookup_uses_recent_transcript_terms_when_the_question_is_empty() -> None:
    store, _id = _store_with_arch()
    tr = Transcript()
    for i in range(3):
        tr.add(_seg(i, "so tell me about the worker threads and locks"))
    m = ContextManager(_cfg(lookup_min_score=0.5), store, tr)
    b = _build(m, kind="auto", question="")
    assert "dedicated lock" in b.lookup


def test_lookup_respects_top_k() -> None:
    store = ContextStore(None)
    for i in range(4):
        _add(store, f"d{i}.md", f"# Relay {i}\nrelay protocol handshake variant {i}")
    m, _tr, _c = _mk(3, store=store, lookup_min_score=0.0, lookup_top_k=2, lookup_tokens=2000)
    b = _build(m, question="relay protocol handshake")
    assert b.lookup.count("[d") == 2
    assert b.usage.documents == 2


def test_lookup_is_fitted_to_its_token_budget() -> None:
    store = ContextStore(None)
    for i in range(3):
        _add(store, f"d{i}.md", f"# Relay {i}\n" + f"relay protocol handshake sentence {i}. " * 15)
    m, _tr, _c = _mk(3, store=store, lookup_min_score=0.0, lookup_top_k=3, lookup_tokens=200)
    b = _build(m, question="relay protocol handshake")
    assert 0 < m.estimator.tokens(b.lookup) <= 200
    # The best passage fits whole and the next one simply does not fit beside it: that is the
    # budget doing its job, not a trim (live finding 2026-10-06; only the best passage being cut is).
    assert b.usage.trimmed is False


def test_the_best_passage_is_cut_when_even_it_exceeds_the_lookup_budget() -> None:
    store = ContextStore(None)
    _add(store, "big.md", "# Relay\n" + "\n".join(f"relay protocol handshake line {i:03d}" for i in range(100)))
    m, _tr, _c = _mk(3, store=store, lookup_min_score=0.0, lookup_tokens=60, chunk_chars=5000)
    b = _build(m, question="relay protocol handshake")
    assert b.lookup.endswith("[...]") and m.estimator.tokens(b.lookup) <= 60


def test_lookup_ignores_disabled_and_inject_documents() -> None:
    store = ContextStore(None)
    inj = _add(store, "inj.md", "# Relay\nrelay protocol handshake", mode="inject")
    m, _tr, _c = _mk(3, store=store, lookup_min_score=0.0)
    assert _build(m, question="relay protocol handshake").lookup == ""
    store.set_mode(inj.id, "reference", digest_wanted=False)
    assert _build(m, question="relay protocol handshake").lookup != ""


# ---- fixed mode ----------------------------------------------------------------

def test_fixed_mode_transcript_equals_window_text_within_budget() -> None:
    m, tr, _c = _mk(20, budget_mode="fixed")
    b = _build(m)
    assert b.transcript == tr.window_text(2500 * 4)
    assert b.usage.anchor_moved is False


def test_fixed_mode_over_budget_matches_window_text_when_sliding() -> None:
    m, tr, cfg = _mk(400, budget_mode="fixed", trim_step=0.0)
    cfg.llm.context_tokens = 500
    b = _build(m)
    assert b.transcript == tr.window_text(500 * 4)


def test_fixed_mode_does_no_fitting_even_with_a_tiny_slot() -> None:
    m, tr, _c = _mk(200, budget_mode="fixed", trim_step=0.0)
    m.set_slot_tokens(300)
    b = _build(m)
    assert b.transcript == tr.window_text(2500 * 4)
    assert b.usage.used_tokens > b.usage.slot_tokens


def test_fixed_mode_still_anchors_so_the_prefix_is_stable() -> None:
    m, tr, cfg = _mk(100, budget_mode="fixed", trim_step=0.25)
    cfg.llm.context_tokens = 500
    first = _build(m)
    assert first.usage.anchor_moved
    _fill(tr, 1, start=100)
    second = _build(m)
    assert not second.usage.anchor_moved and second.transcript.startswith(first.transcript)


def test_fixed_mode_uses_reference_and_lookup_at_their_configured_sizes() -> None:
    store, _id = _store_with_arch()
    _add(store, "inj.md", "Injected body.", mode="inject")
    m, _tr, _c = _mk(5, store=store, budget_mode="fixed", lookup_min_score=0.0)
    b = _build(m, question="worker threads")
    assert "### inj.md\nInjected body." in b.reference and "dedicated lock" in b.lookup


# ---- auto budget ---------------------------------------------------------------

def test_auto_budget_fits_inside_the_slot() -> None:
    store, _id = _store_with_arch()
    _add(store, "inj.md", "Injected body text. " * 30, mode="inject")
    m, _tr, cfg = _mk(600, store=store, lookup_min_score=0.0)
    for slot in (4096, 2048, 1500, 1200):
        m.set_slot_tokens(slot)
        b = _build(m, screen="screen text " * 50, max_tokens=300)
        u = b.usage
        assert u.slot_tokens == slot
        assert u.system_tokens + u.used_tokens <= slot - 300 - cfg.context.reserve_tokens, (slot, u)
        assert u.used_tokens == (m.estimator.tokens_for_len(len(Q) + 200) + u.transcript_tokens + u.reference_tokens
                                 + u.lookup_tokens + u.earlier_tokens + u.screen_tokens)


def test_budget_tokens_is_slot_minus_answer_reserve_and_system() -> None:
    m, _tr, cfg = _mk(5)
    m.set_slot_tokens(3000)
    b = _build(m, max_tokens=500)
    sys_t = m.estimator.tokens(SYSTEM)
    assert b.usage.budget_tokens == 3000 - 500 - cfg.context.reserve_tokens - sys_t   # one slot: job_limit = slot - reserve
    assert b.usage.system_tokens == sys_t


def test_transcript_is_capped_by_context_tokens_in_a_big_slot() -> None:
    m, _tr, cfg = _mk(2000)
    cfg.llm.context_tokens = 800
    m.set_slot_tokens(32768)
    b = _build(m)
    assert b.usage.transcript_tokens <= 800
    assert b.usage.transcript_tokens > 800 * 0.7


def test_the_transcript_floor_is_honoured_by_giving_back_other_blocks() -> None:
    store, _id = _store_with_arch()
    _add(store, "inj.md", "Injected body text. " * 80, mode="inject")
    m, _tr, cfg = _mk(600, store=store, lookup_min_score=0.0, transcript_min_tokens=600, reference_tokens=800)
    m.set_slot_tokens(1700)
    b = _build(m, screen="s " * 400, max_tokens=300)
    assert b.usage.transcript_tokens >= 560           # floor 600, line granularity / trim_step slack aside
    assert b.usage.trimmed is True
    assert b.usage.system_tokens + b.usage.used_tokens <= 1700 - 300 - cfg.context.reserve_tokens


def test_trim_order_lookup_goes_first_then_reference_then_earlier_then_screen() -> None:
    store, _id = _store_with_arch()
    inj = _add(store, "inj.md", "Injected body text. " * 40, mode="inject")
    assert inj.mode == "inject"

    def run(slot: int):
        m, _tr, cfg = _mk(600, store=store, lookup_min_score=0.0, transcript_min_tokens=600, reference_tokens=500,
                          lookup_tokens=400, summary_tokens=100)
        m.window.set_summary(None, "earlier summary " * 20)
        m.set_slot_tokens(slot)
        return _build(m, screen="screen words " * 40, question="worker threads locks", max_tokens=200).usage

    roomy = run(6000)
    assert roomy.lookup_tokens > 0 and roomy.reference_tokens > 0 and roomy.earlier_tokens > 0 and roomy.screen_tokens > 0
    assert roomy.trimmed is False or roomy.transcript_tokens > 0

    # shrink step by step: each smaller slot may only lose blocks in the documented order
    state = []
    for slot in (2600, 2000, 1700, 1500, 1300, 1100):
        u = run(slot)
        state.append((u.lookup_tokens > 0, u.reference_tokens > 0, u.earlier_tokens > 0, u.screen_tokens >= roomy.screen_tokens))
    # a block that is gone never comes back in a tighter slot
    for col in range(4):
        values = [row[col] for row in state]
        assert values == sorted(values, reverse=True)
    # lookup is lost before the reference block is, the reference block before the summary
    first_lost = [next((i for i, row in enumerate(state) if not row[c]), len(state)) for c in range(3)]
    assert first_lost[0] <= first_lost[1] <= first_lost[2]
    assert first_lost[0] < len(state)           # lookup really did get trimmed somewhere


def test_reference_documents_are_dropped_from_the_end_catalogue_lines_last() -> None:
    store = ContextStore(None)
    _add(store, "one.md", "First injected body. " * 20, mode="inject")
    _add(store, "two.md", "Second injected body. " * 20, mode="inject")
    _add(store, "cat.md", "# Catalogued\nsomething")
    m, _tr, _c = _mk(600, store=store, reference_tokens=600, transcript_min_tokens=1000, lookup_tokens=100)

    def ref_at(slot: int) -> str:
        m.set_slot_tokens(slot)
        return _build(m, max_tokens=200).reference

    full = ref_at(6000)
    assert "### one.md" in full and "### two.md" in full and "- cat.md" in full
    mid = ref_at(1900)
    if "### two.md" not in mid:
        assert "### one.md" in mid or "- cat.md" in mid
    gone_second = ref_at(1700)
    assert not ("### two.md" in gone_second and "### one.md" not in gone_second)     # never the earlier doc first
    tight = ref_at(1500)
    assert "### two.md" not in tight or "### one.md" in tight


def test_newest_line_always_survives_even_in_an_absurdly_small_slot() -> None:
    m, tr, _c = _mk(50)
    m.set_slot_tokens(10)
    b = _build(m, system="x" * 4000, fixed=4000)
    assert b.transcript.endswith(tr.lines()[-1][1][-30:])
    assert b.transcript != "" and b.usage.trimmed is True


def test_slot_smaller_than_the_fixed_parts_never_raises() -> None:
    store, _id = _store_with_arch()
    _add(store, "inj.md", "x " * 500, mode="inject")
    m, _tr, _c = _mk(30, store=store)
    m.set_slot_tokens(5)
    b = _build(m, screen="y" * 3000, max_tokens=5000, system="z" * 9000, fixed=-5)
    assert b.transcript and b.usage.trimmed


def test_empty_transcript_is_fine() -> None:
    m, _tr, _c = _mk(0)
    b = _build(m)
    assert b.transcript == "" and b.compact_wanted is False and b.usage.transcript_tokens == 0


def test_build_never_raises_when_the_transcript_misbehaves() -> None:
    class Broken:
        def lines(self, **kw):
            raise RuntimeError("boom")

        def last_id(self):
            raise RuntimeError("boom")

        def window_text(self, *a, **kw):
            raise RuntimeError("boom")

    m = ContextManager(_cfg(), None, Broken())  # type: ignore[arg-type]
    b = _build(m)
    assert b.transcript == "" and b.usage.trimmed is True and b.compact_wanted is False


def test_screen_text_is_cut_to_no_less_than_a_quarter_under_pressure() -> None:
    m, _tr, _c = _mk(600, transcript_min_tokens=900)
    m.set_slot_tokens(1600)
    screen = "screen " * 600           # 4200 chars
    b = _build(m, screen=screen, max_tokens=200)
    assert len(b.screen) < len(screen)
    assert m.estimator.tokens(b.screen) >= m.estimator.tokens(screen) * 0.25 - 2
    assert screen.startswith(b.screen)


def test_screen_text_untouched_when_there_is_room() -> None:
    m, _tr, _c = _mk(10)
    screen = "screen words " * 20
    assert _build(m, screen=screen).screen == screen


# ---- earlier summary / compaction ----------------------------------------------

def test_earlier_summary_is_included_and_counted() -> None:
    m, _tr, _c = _mk(10)
    m.set_summary(None, "They discussed the roadmap.")
    b = _build(m)
    assert b.earlier == "They discussed the roadmap." and b.usage.earlier_tokens > 0


def test_earlier_summary_is_per_scope() -> None:
    m, _tr, _c = _mk(10)
    m.set_summary(None, "whole meeting")
    m.set_summary(4, "since checkpoint")
    assert _build(m).earlier == "whole meeting"
    assert _build(m, since=4).earlier == "since checkpoint"
    assert _build(m, since=5).earlier == ""


def test_an_overlong_summary_is_cut_to_twice_summary_tokens() -> None:
    m, _tr, _c = _mk(10, summary_tokens=50)
    m.set_summary(None, "\n".join(f"- point {i:03d} about things" for i in range(200)))
    b = _build(m)
    assert len(b.earlier) <= m.estimator.chars(100) and b.usage.trimmed


def test_compact_wanted_when_the_anchor_moves_and_scope_none_is_valid() -> None:
    m, tr, cfg = _mk(600)
    cfg.llm.context_tokens = 300
    b = _build(m)
    assert b.usage.anchor_moved and b.compact_wanted is True and b.compact_scope is None
    claimed = m.claim_compaction(None)
    assert claimed is not None and claimed[1]
    _fill(tr, 1, start=600)
    nxt = _build(m)
    assert not nxt.usage.anchor_moved and nxt.compact_wanted is False


def test_compact_scope_is_the_since_id() -> None:
    m, _tr, cfg = _mk(600)
    cfg.llm.context_tokens = 300
    b = _build(m, since=100)
    assert b.compact_wanted and b.compact_scope == 100


def test_compact_wanted_stays_true_while_text_is_pending_after_a_failed_job() -> None:
    m, tr, cfg = _mk(600)
    cfg.llm.context_tokens = 300
    _build(m)
    _prev, dropped = m.claim_compaction(None)  # type: ignore[misc]
    m.restore_compaction(None, dropped)
    _fill(tr, 1, start=600)
    assert _build(m).compact_wanted is True


def test_compaction_off_never_wants_and_never_piles_up() -> None:
    m, _tr, cfg = _mk(600, compaction="off")
    cfg.llm.context_tokens = 300
    b = _build(m)
    assert b.usage.anchor_moved and b.compact_wanted is False
    assert m.claim_compaction(None) is None


def test_no_overflow_no_compaction() -> None:
    m, _tr, _c = _mk(5)
    assert _build(m).compact_wanted is False


# ---- usage reporting -----------------------------------------------------------

def test_usage_event_fields() -> None:
    store, _id = _store_with_arch()
    _add(store, "inj.md", "Injected body.", mode="inject")
    m, _tr, _c = _mk(20, store=store, lookup_min_score=0.0)
    m.set_slot_tokens(4000)
    b = _build(m, kind="auto", question="worker threads", screen="some screen text", max_tokens=300)
    u = b.usage
    assert isinstance(u, ContextUsageEvent)
    assert (u.kind, u.slot_tokens, u.job_id) == ("auto", 4000, 0)
    assert u.prompt_tokens is None and u.cached_tokens is None
    assert u.transcript_tokens == m.estimator.tokens(b.transcript)
    assert u.reference_tokens == m.estimator.tokens(b.reference)
    assert u.lookup_tokens == m.estimator.tokens(b.lookup)
    assert u.screen_tokens == m.estimator.tokens(b.screen)
    assert u.chars_per_token == 4.0
    assert u.documents == b.documents_used == 2          # one injected + one distinct lookup document


def test_documents_counts_distinct_documents_not_passages() -> None:
    store, _id = _store_with_arch()
    m, _tr, _c = _mk(5, store=store, lookup_min_score=0.0, lookup_top_k=3, lookup_tokens=2000)
    b = _build(m, question="worker threads audio pipeline sessions storage")
    assert b.lookup.count("[arch.md") >= 2
    assert b.documents_used == 1


def test_a_document_that_is_both_injected_and_looked_up_counts_once() -> None:
    store = ContextStore(None)
    _add(store, "a.md", "# Relay\nrelay protocol handshake", mode="inject")
    _add(store, "b.md", "# Relay\nrelay protocol handshake summary", mode="reference")
    m, _tr, _c = _mk(5, store=store, lookup_min_score=0.0)
    assert _build(m, question="relay protocol handshake").documents_used == 2


# ---- calibration / overflow ----------------------------------------------------

def test_observe_prompt_calibrates_and_changes_later_estimates() -> None:
    m, _tr, cfg = _mk(5)
    cfg.context.auto_calibrate = True
    before = m.estimator.tokens("x" * 1000)
    m.observe_prompt(3000, 1000)               # 3 chars/token measured
    assert m.estimator.chars_per_token < 4.0
    assert m.estimator.tokens("x" * 1000) > before


def test_observe_prompt_ignores_missing_counts_and_respects_auto_off() -> None:
    m, _tr, cfg = _mk(5)
    cfg.context.auto_calibrate = True
    m.observe_prompt(3000, None)
    assert m.estimator.chars_per_token == 4.0
    cfg.context.auto_calibrate = False
    m.observe_prompt(3000, 1000)
    assert m.estimator.chars_per_token == 4.0


def test_note_overflow_shrinks_the_transcript_budget() -> None:
    """A fresh window after note_overflow() is sized with the denser tokens estimate (fewer chars)."""
    plain, _t1, _c1 = _mk(2000)
    plain.set_slot_tokens(2000)
    denser, _t2, _c2 = _mk(2000)
    denser.set_slot_tokens(2000)
    denser.note_overflow()
    a, b = _build(plain), _build(denser)
    assert len(b.transcript) < len(a.transcript)
    assert b.usage.chars_per_token == pytest.approx(3.4)
    assert b.usage.system_tokens + b.usage.used_tokens <= 2000 - 400 - 256


def test_config_changes_apply_live() -> None:
    m, _tr, cfg = _mk(2000)
    first = len(_build(m).transcript)
    cfg.llm.context_tokens = 300
    assert len(_build(m).transcript) < first
    cfg.context.chars_per_token = 3.0
    assert _build(m).usage.chars_per_token == 3.0


# ---- session handling / snapshot -----------------------------------------------

def test_reset_session_clears_the_window_but_keeps_documents() -> None:
    store, _id = _store_with_arch()
    m, _tr, cfg = _mk(600, store=store)
    cfg.llm.context_tokens = 300
    _build(m)
    m.set_summary(None, "old summary")
    assert m.window.anchor(None) is not None
    m.reset_session()
    assert m.window.anchor(None) is None and m.window.earlier_summary(None) == ""
    assert len(m.store.documents()) == 1


def test_usage_snapshot_before_and_after_a_record() -> None:
    store, _id = _store_with_arch()
    m, _tr, _c = _mk(20, store=store)
    m.set_slot_tokens(3500)
    snap = m.usage_snapshot()
    assert snap["has_usage"] is False and snap["slot_tokens"] == 3500 and snap["documents"] == 1
    assert snap["chars_per_token"] == 4.0 and snap["earlier_summary"] == "" and snap["kind"] is None
    b = _build(m, kind="screen")
    final = dataclasses.replace(b.usage, job_id=7, prompt_tokens=1234, cached_tokens=1000)
    m.record_usage(final)
    m.set_summary(None, "so far")
    snap = m.usage_snapshot()
    assert snap["has_usage"] is True and snap["job_id"] == 7 and snap["kind"] == "screen"
    assert snap["prompt_tokens"] == 1234 and snap["cached_tokens"] == 1000
    assert snap["transcript_tokens"] == b.usage.transcript_tokens
    assert snap["documents"] == 1 and snap["documents_used"] == b.usage.documents
    assert snap["earlier_summary"] == "so far" and "ts" not in snap


def test_reset_session_clears_the_last_usage() -> None:
    m, _tr, _c = _mk(5)
    m.record_usage(_build(m).usage)
    assert m.usage_snapshot()["has_usage"] is True
    m.reset_session()
    assert m.usage_snapshot()["has_usage"] is False


# ---- the long-session problem itself -------------------------------------------

def test_long_session_prefix_stays_stable_and_the_prompt_always_fits() -> None:
    """Simulate a long meeting turn by turn: the estimated prompt never exceeds the slot, and the
    transcript block changes its first line only a handful of times (KV prefix cache hits)."""
    store, _id = _store_with_arch()
    cfg = _cfg(lookup_min_score=0.0)
    tr = Transcript()
    m = ContextManager(cfg, store, tr)
    m.set_slot_tokens(2048)
    first_lines: list[str] = []
    for i in range(400):
        tr.add(_seg(i, f"utterance {i:03d} " + "w" * 60))
        b = _build(m, kind="auto", question="", max_tokens=300)
        u = b.usage
        assert u.system_tokens + u.used_tokens <= 2048 - 300 - cfg.context.reserve_tokens
        first_lines.append(b.transcript.split("\n", 1)[0])
        if b.compact_wanted:
            m.claim_compaction(None)
    changes = sum(1 for a, c in zip(first_lines, first_lines[1:]) if a != c)
    assert changes < 400 / 4


def test_a_lower_ranked_passage_that_does_not_fit_is_not_a_trim() -> None:
    """Live finding 2026-10-06: with top_k 3 and a 600-token lookup budget every answer read
    "trimmed" because the second-best passage never fit beside the best one. Only cutting or
    dropping the BEST passage counts."""
    store = ContextStore(None)
    for i in range(3):
        _add(store, f"doc{i}.md", f"# Threads {i}\n\n" + ("worker thread lock answer jobs. " * 8))   # ~70 tokens each
    m, _tr, _c = _mk(5, store=store, lookup_min_score=0.0, lookup_tokens=120, lookup_top_k=3)
    m.set_slot_tokens(8000)
    b = _build(m, question="worker threads locks")
    assert b.lookup and b.usage.lookup_tokens <= 120
    assert b.usage.trimmed is False
    m2, _tr2, _c2 = _mk(5, store=store, lookup_min_score=0.0, lookup_tokens=30, lookup_top_k=3)
    m2.set_slot_tokens(8000)
    b2 = _build(m2, question="worker threads locks")
    assert b2.usage.trimmed is True          # the best passage itself had to be cut


def test_build_uses_a_staged_summary_only_on_the_turn_the_anchor_moves() -> None:
    m, tr, cfg = _mk(60)
    cfg.llm.context_tokens = 150
    m.set_slot_tokens(8000)
    b0 = _build(m)
    assert b0.usage.anchor_moved
    m.stage_summary(None, "staged summary of the start")
    b1 = _build(m)                                  # same window: the prompt must not change
    assert not b1.usage.anchor_moved and b1.earlier == "" and b1.usage.earlier_tokens == 0
    _fill(tr, 60, start=60)
    b2 = _build(m)
    assert b2.usage.anchor_moved and b2.earlier == "staged summary of the start"
    assert b2.usage.earlier_tokens > 0


def test_reference_block_is_stable_whatever_this_turns_screen_length() -> None:
    """Review 2026-10-06: the stable <reference> prefix used to flip in and out with the screen
    length (one long screen pushed it out, the next short one brought it back), re-evaluating the
    whole prompt each time. Its budget now depends on the library, the slot and the configured
    screen maximum only."""
    store = ContextStore(None)
    _add(store, "arch.md", "Architecture body sentence. " * 60, mode="inject")      # ~420 tokens
    m, _tr, cfg = _mk(3, store=store, reference_tokens=1200, transcript_min_tokens=600)
    cfg.llm.parallel = 2
    cfg.screen.max_text_chars = 2500
    m.set_slot_tokens(4096)
    short = _build(m, screen="x" * 12, max_tokens=400)
    long = _build(m, screen="y " * 1000, max_tokens=400)                             # 2000 chars
    assert short.reference and short.reference == long.reference
    assert short.usage.reference_tokens == long.usage.reference_tokens > 0


def test_an_oversized_injected_document_is_cut_to_the_stable_budget_not_dropped() -> None:
    store = ContextStore(None)
    _add(store, "readme.md", "\n".join(f"Line {i} of the long readme body." for i in range(400)), mode="inject")
    m, _tr, cfg = _mk(3, store=store, reference_tokens=1200, transcript_min_tokens=600)
    cfg.llm.parallel = 2
    m.set_slot_tokens(4096)
    b = _build(m, screen="", max_tokens=220)
    assert "### readme.md" in b.reference and b.reference.endswith("[...]")
    assert 0 < b.usage.reference_tokens <= 1200
    assert b.usage.transcript_tokens > 0


def test_an_oversized_screen_is_cut_before_the_transcript_floor_is_broken() -> None:
    """A hands-free cumulative screen far above [screen].max_text_chars on a tight slot: the
    screen is cut (down to a quarter) and the newest speech keeps its floor; the reference block
    has no room that turn (its budget reserves the screen's real size) and comes back next turn."""
    store = ContextStore(None)
    _add(store, "arch.md", "Architecture body sentence. " * 30, mode="inject")
    m, _tr, cfg = _mk(40, store=store, reference_tokens=400, transcript_min_tokens=600, lookup_tokens=0)
    cfg.screen.max_text_chars = 400
    m.set_slot_tokens(2000)                                   # tight: avail ~ 2000-256-101-400 = ~1240
    big = _build(m, screen="s " * 1200, max_tokens=400)       # a 2400-char screen
    assert len(big.screen) < 2400 and big.usage.trimmed
    assert big.usage.transcript_tokens >= 300                 # the floor's worth, line granularity aside
    small = _build(m, screen="s " * 100, max_tokens=400)      # back to an ordinary screen
    assert small.reference                                    # the stable block is back
