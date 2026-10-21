"""DashboardService + the assistant's "dashboard" lane (exocortex.dashboard.service,
exocortex.llm.assistant, 2026-10-20): a refresh end to end against the scripted FakeClient, the
ids the model is shown and how its changes are applied, the background due logic with a
hand-moved clock, cancellation / retry, manual edits, the insights mirror, notes, reset."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from exocortex.bus import (
    AssistDoneEvent,
    AssistStartedEvent,
    DashboardUpdatedEvent,
    EventBus,
    InsightsUpdatedEvent,
    NotesUpdatedEvent,
    Source,
)
from exocortex.config import Config
from exocortex.dashboard import service as service_mod
from exocortex.dashboard.service import DashboardService
from exocortex.llm import assistant as assistant_mod
from exocortex.llm.assistant import Assistant
from exocortex.transcript import Transcript
from tests.test_assistant import FakeClient, drain_until, make_segment

FIRST = (
    "## Questions to ask\n- NEW: Should export() paginate past 1000 rows?\n- NEW: Is the cache per tenant?\n"
    "## Blind spots & follow-ups\n- NEW: No test for an empty payload\n"
    "## Next steps\n- NEW: Write the failing test for export()\n"
    "## Suggested prompts\n- NEW: /repro-fix export() returns 500 on an empty tenant\n"
)
SECOND = (
    "## Questions to ask\n- ANSWERED Q2: yes, keyed by tenant id\n"
    "## Blind spots & follow-ups\n- (no change)\n"
    "## Next steps\n- DONE N1: it is in and red\n- NEW: Make it green\n"
    "## Suggested prompts\n- DROP P1\n"
)


class ManualClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture(autouse=True)
def _park_the_timers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assistant_mod, "INSIGHT_TICK_S", 3600.0)
    monkeypatch.setattr(service_mod, "TICK_S", 3600.0)


def _say(transcript: Transcript, seg_id: int, text: str, *, me: bool = False) -> None:
    transcript.add(make_segment(seg_id, text=text, t_start=float(seg_id), t_end=seg_id + 1.0,
                                source=Source.MIC if me else Source.SYSTEM,
                                speaker_id="me" if me else "spk1", speaker_label="Me" if me else "Sam"))


def _start(cfg: Config, answers: list[str], *, screen: str = "", clock: Any = None, context: Any = None,
           folder: Path | None = None) -> tuple[EventBus, Transcript, FakeClient, Assistant, DashboardService]:
    bus = EventBus()
    transcript = Transcript()
    client = FakeClient(token_batches=[[a] for a in answers])
    clk = clock or ManualClock()
    assistant = Assistant(cfg, bus, client, transcript, clock=clk)
    assistant.set_screen_provider(lambda: screen)
    service = DashboardService(cfg, bus, transcript, assistant, clk, context=context,
                               screen_provider=lambda: screen, folder=folder)
    assistant.start()
    service.start()
    return bus, transcript, client, assistant, service


def _done(job_id: int):
    return lambda e: isinstance(e, AssistDoneEvent) and e.job_id == job_id


def _stop(assistant: Assistant, service: DashboardService) -> None:
    service.stop()
    assistant.stop()


def test_a_refresh_shows_the_open_items_with_ids_and_applies_the_changes() -> None:
    cfg = Config()
    cfg.assist.session_context = "billing service"
    bus, transcript, client, assistant, service = _start(cfg, [FIRST, SECOND], screen="def export(tenant):")
    sub = bus.subscribe()
    try:
        _say(transcript, 1, "We need per-tenant rate limiting on the export endpoint.")
        job = service.refresh_now()
        events = drain_until(sub, _done(job))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert started.kind == "dashboard" and started.question == "Dashboard refresh"
        updated = [e for e in events if isinstance(e, DashboardUpdatedEvent)]
        assert updated and updated[-1].detail == "updated: +5 new" and not updated[-1].auto
        open_texts = [i.text for i in updated[-1].items if i.status == "open"]
        assert open_texts == ["Should export() paginate past 1000 rows?", "Is the cache per tenant?",
                              "No test for an empty payload", "Write the failing test for export()",
                              "/repro-fix export() returns 500 on an empty tenant"]
        mirrored = next(e for e in events if isinstance(e, InsightsUpdatedEvent))
        assert mirrored.questions == "- Should export() paginate past 1000 rows?\n- Is the cache per tenant?"
        assert mirrored.blindspots == "- No test for an empty payload"
        assert assistant.insights == {"questions": mirrored.questions, "blindspots": mirrored.blindspots}
        assert assistant.answers == []                                     # never an answer
        prompt = client.calls[0]["prompt"]
        assert "keeping a small DASHBOARD current" in prompt and "Session context (what the team is working on): billing service" in prompt
        assert "<current>\nQuestions to ask:\n- (none yet)" in prompt and "per-tenant rate limiting" in prompt
        assert "<transcript_earlier>\n" not in prompt and "def export(tenant):" in prompt
        assert client.calls[0]["max_tokens"] == cfg.llm.dashboard_max_tokens

        _say(transcript, 2, "The cache is keyed by tenant id, so yes.")
        job2 = service.refresh_now()
        events = drain_until(sub, _done(job2))
        prompt2 = client.calls[1]["prompt"]
        assert "- Q1: Should export() paginate past 1000 rows?" in prompt2 and "- Q2: Is the cache per tenant?" in prompt2
        assert "- N1: Write the failing test for export()" in prompt2 and "- P1: /repro-fix" in prompt2
        assert "<transcript_earlier>\n[00:01] Sam: We need per-tenant" in prompt2
        assert "<transcript_new>\n[00:02] Sam: The cache is keyed" in prompt2
        updated = [e for e in events if isinstance(e, DashboardUpdatedEvent)][-1]
        assert updated.detail == "updated: +1 new, 2 answered / done, 1 dropped"
        store = service.store
        assert store.get(2).status == "answered" and store.get(2).answer == "yes, keyed by tenant id"
        assert store.get(4).status == "answered" and store.get(4).answer == "it is in and red"
        assert store.get(5).status == "dropped" and [i.text for i in store.open_items("nextstep")] == ["Make it green"]
        md = service.markdown()
        assert "- [x] Is the cache per tenant? — yes, keyed by tenant id" in md and "- [ ] Make it green" in md
    finally:
        _stop(assistant, service)


def test_a_refresh_can_be_limited_to_one_section() -> None:
    cfg = Config()
    bus, transcript, client, assistant, service = _start(cfg, ["## Questions to ask\n- NEW: Only this?\n## Next steps\n- NEW: ignored\n"])
    sub = bus.subscribe()
    try:
        job = service.refresh_now(("questions",))
        drain_until(sub, _done(job))
        prompt = client.calls[0]["prompt"]
        assert "Questions to ask (Q1, Q2, ...)" in prompt and "Next steps (N1" not in prompt
        assert [i.text for i in service.store.items()] == ["Only this?"]
    finally:
        _stop(assistant, service)


def test_a_skip_or_empty_answer_still_moves_the_watermark() -> None:
    cfg = Config()
    bus, transcript, client, assistant, service = _start(cfg, ["(nothing to do)", FIRST])
    sub = bus.subscribe()
    try:
        _say(transcript, 1, "hello there everyone")
        job = service.refresh_now()
        events = drain_until(sub, _done(job))
        assert events[-1].skipped and [e for e in events if isinstance(e, DashboardUpdatedEvent)][-1].detail == "nothing new"
        assert service._seg_id == 1 and service.store.items() == []
        _say(transcript, 2, "now something technical about export()")
        job = service.refresh_now()
        drain_until(sub, _done(job))
        assert "<transcript_earlier>\n[00:01] Sam: hello there everyone" in client.calls[1]["prompt"]
        assert "<transcript_new>\n[00:02] Sam: now something" in client.calls[1]["prompt"]
    finally:
        _stop(assistant, service)


def test_background_refreshes_follow_the_interval_new_speech_and_the_screen() -> None:
    cfg = Config()
    cfg.dashboard.refresh_interval_s = 100
    cfg.dashboard.min_new_words = 5
    clock = ManualClock()
    screen = {"text": ""}
    bus = EventBus()
    transcript = Transcript()
    client = FakeClient(token_batches=[[FIRST], [SECOND], [SECOND]])
    assistant = Assistant(cfg, bus, client, transcript, clock=clock)
    assistant.set_screen_provider(lambda: screen["text"])
    service = DashboardService(cfg, bus, transcript, assistant, clock, screen_provider=lambda: screen["text"])
    sub = bus.subscribe()
    try:
        assert service.tick() is False                      # the assistant is not running yet
        assistant.start()
        service.start()
        assert service.tick() is False                      # the first interval has not passed
        clock.advance(101)
        assert service.tick() is False                      # nothing new was said
        _say(transcript, 1, "one two three four")
        assert service.tick() is False                      # 4 < 5 words
        _say(transcript, 2, "five six")
        assert service.tick() is True                       # due: queued as a background refresh
        events = drain_until(sub, lambda e: isinstance(e, DashboardUpdatedEvent) and e.auto and e.detail.startswith("updated"))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert started.kind == "dashboard"
        assert service.tick() is False                      # throttled: the interval restarted
        clock.advance(101)
        assert service.tick() is False                      # no new words since the pass
        screen["text"] = "\n".join(f"line {n} of brand new code" for n in range(8))
        assert service.tick() is True                       # the screen changed substantially
        drain_until(sub, lambda e: isinstance(e, DashboardUpdatedEvent) and e.auto and e.detail.startswith("updated"))
        clock.advance(101)
        assert service.tick() is False                      # that screen was seen
        cfg.dashboard.track = False
        _say(transcript, 3, "lots of new words here to make it due again for sure")
        assert service.tick() is False                      # tracking off
        cfg.dashboard.track = True
        service.set_paused(True)
        assert service.tick() is False                      # paused
        service.set_paused(False)
        cfg.handsfree.manual = True
        assert service.tick() is False                      # manual advance: nothing automatic
        cfg.handsfree.manual = False
        cfg.dashboard.refresh_interval_s = 0
        assert service.tick() is False                      # on demand only
        cfg.dashboard.refresh_interval_s = 100
        assert service.tick() is True
    finally:
        _stop(assistant, service)


def test_a_cancelled_background_refresh_is_retried_sooner_and_a_manual_one_wins() -> None:
    cfg = Config()
    cfg.dashboard.refresh_interval_s = 120
    clock = ManualClock(5000.0)
    bus, transcript, client, assistant, service = _start(cfg, [FIRST], clock=clock)
    try:
        fake_job = type("Job", (), {"id": 77, "auto": True, "started": 5000.0, "sections": ()})()
        service._jobs[77] = ({}, ("question",))
        service._last_start = 5000.0
        service._settle(fake_job, ok=False)
        assert service._last_start == pytest.approx(5000.0 - (120 - service_mod.RETRY_S))
        assert 77 not in service._jobs
        # dashboard_now: a background pass never replaces a queued on-demand one; on demand replaces anything
        assistant._dashboard_slot.clear()
        assert assistant.dashboard_now(auto=False) is not None
        assert assistant.dashboard_now(auto=True) is None
        assert assistant.dashboard_now(auto=False) is not None
        assistant._dashboard_slot.clear()
        assert assistant.dashboard_now(auto=True) is not None and assistant.dashboard_now(auto=True) is not None
        assistant._drop_auto_insights()
        assert assistant._dashboard_slot.try_get() is None
    finally:
        _stop(assistant, service)


def test_the_insights_background_pass_yields_to_the_dashboard() -> None:
    cfg = Config()
    cfg.assist.insight_interval_s = 1
    clock = ManualClock()
    bus, transcript, client, assistant, service = _start(cfg, [FIRST], clock=clock)
    try:
        _say(transcript, 1, " ".join(["word"] * 100))
        clock.advance(10)
        assert assistant.dashboard_tracking() and assistant._maybe_auto_insights() is False
        cfg.dashboard.track = False
        assert assistant._maybe_auto_insights() is True
    finally:
        _stop(assistant, service)


def test_manual_edits_add_resolve_dismiss_reopen_and_notes_land_in_the_store(tmp_path: Path) -> None:
    cfg = Config()
    bus, transcript, client, assistant, service = _start(cfg, [FIRST], folder=tmp_path)
    sub = bus.subscribe()
    try:
        assert service.add_item("question", "Who owns the migration?") and not service.add_item("nope", "x")
        assert not service.add_item("question", "   ")
        item = service.store.items("question")[0]
        assert item.source == "user"
        assert service.item_action(item.id, "answer", "Sam does")
        assert service.store.get(item.id).status == "answered" and service.store.get(item.id).answer == "Sam does"
        assert service.item_action(item.id, "reopen") and service.store.get(item.id).status == "open"
        assert service.item_action(item.id, "edit", "Who owns the migration plan?")
        assert service.item_action(item.id, "dismiss") and service.store.get(item.id).status == "dropped"
        assert not service.item_action(item.id, "bogus") and not service.item_action(999, "dismiss")
        events = [e for e in __import__("exocortex.bus", fromlist=["drain"]).drain(sub) if isinstance(e, DashboardUpdatedEvent)]
        assert [e.detail for e in events] == ["added", "marked answered", "reopened", "edited", "dismissed"]
        assert (tmp_path / "dashboard.md").is_file() and (tmp_path / "dashboard.sqlite").is_file()
        bus.publish(NotesUpdatedEvent(markdown="## Summary\nnotes!", generated_at=33.0))
        service.tick()
        assert service.store.notes() == ("## Summary\nnotes!", 33.0)
        snap = [e for e in __import__("exocortex.bus", fromlist=["drain"]).drain(sub) if isinstance(e, DashboardUpdatedEvent)][-1]
        assert snap.notes == "## Summary\nnotes!" and snap.notes_at == 33.0 and snap.detail == "notes updated"
    finally:
        _stop(assistant, service)


def test_reset_starts_a_fresh_store(tmp_path: Path) -> None:
    cfg = Config()
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    bus, transcript, client, assistant, service = _start(cfg, [FIRST], folder=first)
    sub = bus.subscribe()
    try:
        service.add_item("question", "old one?")
        service._seg_id = 5
        service.reset(second)
        assert service.store.items() == [] and service.store.path == second / "dashboard.sqlite" and service._seg_id is None
        ev = [e for e in __import__("exocortex.bus", fromlist=["drain"]).drain(sub) if isinstance(e, DashboardUpdatedEvent)][-1]
        assert ev.detail == "new session" and ev.items == ()
        service.reset(None)
        assert service.store.path is None and service.folder is None
    finally:
        _stop(assistant, service)


def test_the_context_provider_feeds_the_refresh_prompt(tmp_path: Path) -> None:
    class _Context:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def refresh(self) -> bool:
            return False

        def toolkit_summary(self) -> str:
            return "1 skill"

        def toolkit_block(self) -> str:
            return "<toolkit>\nSkills: /recon [topic]: look around.\n</toolkit>"

        def toolkit_lookup(self, *texts: str) -> str:
            self.calls.append("lookup")
            return "<toolkit_detail>\n### /recon [topic] (skill)\nbody\n</toolkit_detail>" if any("/recon" in t for t in texts) else ""

        def repo_block(self, *texts: str, time_budget_s: float = 1.5) -> str:
            self.calls.append("repo")
            return '<repo root="r">\nbilling.py\n  L1: def export(tenant):\n</repo>' if any("export" in t for t in texts) else ""

    cfg = Config()
    ctx = _Context()
    bus, transcript, client, assistant, service = _start(cfg, [FIRST], context=ctx)
    sub = bus.subscribe()
    try:
        _say(transcript, 1, "let's run /recon on export() first")
        job = service.refresh_now()
        drain_until(sub, _done(job))
        prompt = client.calls[0]["prompt"]
        system_part, user_part = prompt.split("<|im_start|>user\n", 1)
        assert "<toolkit>\nSkills: /recon" in system_part and "Use ONLY the skills" in system_part
        assert '<repo root="r">' in user_part and "<toolkit_detail>\n### /recon" in user_part
        assert user_part.index("<current>") < user_part.index("<screen>") < user_part.index("<repo") < user_part.index("<toolkit_detail>")
        assert ctx.calls == ["lookup", "repo"]
    finally:
        _stop(assistant, service)


def test_answers_get_the_toolkit_catalog_and_repo_lookups_through_the_assistant() -> None:
    class _Context:
        def toolkit_block(self) -> str:
            return "<toolkit>\nSkills: /recon [topic]: look around.\n</toolkit>"

        def toolkit_lookup(self, *texts: str) -> str:
            return "<toolkit_detail>\nDETAIL\n</toolkit_detail>" if any("/recon" in t for t in texts) else ""

        def repo_block(self, *texts: str, time_budget_s: float = 1.5) -> str:
            return '<repo root="r">\nbilling.py\n  L1: def export(tenant):\n</repo>' if any("export" in t for t in texts) else ""

    cfg = Config()
    bus = EventBus()
    transcript = Transcript()
    client = FakeClient(token_batches=[["ok"], ["ok"], ["ok"]])
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    assistant.set_context_provider(_Context())
    assistant.start()
    sub = bus.subscribe()
    try:
        assistant.ask("what does export() do and should I /recon it?")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        prompt = client.calls[0]["prompt"]
        system_part, user_part = prompt.split("<|im_start|>user\n", 1)
        assert "<toolkit>\nSkills: /recon" in system_part and system_part.rstrip().endswith("</toolkit><|im_end|>")
        assert '<repo root="r">' in user_part and "<toolkit_detail>\nDETAIL" in user_part
        assert user_part.index("<repo") < user_part.index("<question>")
        cfg.context.toolkit_in_answers = False
        assistant.ask("and export() again?")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id > 1)
        assert "<toolkit>" not in client.calls[1]["prompt"] and '<repo root="r">' in client.calls[1]["prompt"]
        assistant.generate_notes(force=True)
        drain_until(sub, lambda e: isinstance(e, NotesUpdatedEvent))
        assert "<toolkit>" not in client.calls[2]["prompt"]          # notes never carry it
    finally:
        assistant.stop()
