"""`[assist].consider_given_code` (2026-10-29): the given-code rule in the solve / code / text system
prompts and the asks of their user turns, the builders' byte-identical default (None = no rule, the
prompt as it was), and the Assistant plumbing -- the setting read live, a "fresh" code job (the
Code-from-scratch gesture) always told to start from scratch, its label. Offline; the assistant
tests reuse tests/test_assistant.py's FakeClient and helpers."""
from __future__ import annotations

import threading
import time

from exocortex.bus import AssistDoneEvent, AssistStartedEvent, EventBus
from exocortex.config import Config
from exocortex.llm import prompts
from exocortex.llm.assistant import Assistant, _AssistJob
from exocortex.transcript import Transcript
from tests.test_assistant import FakeClient, drain_until

# --------------------------------------------------------------------------------------------
# the builders
# --------------------------------------------------------------------------------------------


def test_no_rule_by_default_the_prompts_render_as_before() -> None:
    for build in (prompts.system_solve, prompts.system_code, prompts.system_text):
        assert build("p") == build("p", consider_given_code=None)
        assert "Given code" not in build("p")
    assert prompts.user_code("T", "s", "prev", "do it") == prompts.user_code("T", "s", "prev", "do it", consider_given_code=None)
    assert prompts.user_text("T", "s", "", "") == prompts.user_text("T", "s", "", "", consider_given_code=None)
    assert prompts.user_solve("T", "s") == prompts.user_solve("T", "s", consider_given_code=None)


def test_consider_true_tells_the_model_to_work_with_the_given_code() -> None:
    for build in (prompts.system_solve, prompts.system_code):
        out = build("", consider_given_code=True)
        assert "Given code:" in out and "WORK WITH IT" in out
        assert "complete what is missing and fix only what is wrong" in out
        assert "starter code" in out and "signature kept" in out
        assert "never substituted" in out
        assert "from scratch" not in out.split("Given code:")[1].split("\n")[0]
        # placed with the other code rules: after the library rule, before the output-shape rules
        assert out.index("Libraries:") < out.index("Given code:") < out.index("No preamble")


def test_consider_false_tells_the_model_to_start_from_scratch() -> None:
    for build in (prompts.system_solve, prompts.system_code):
        out = build("", consider_given_code=False)
        assert "Given code: write the best solution from scratch" in out
        assert "not a template" in out and "entry point" in out
        assert "WORK WITH IT" not in out


def test_text_gets_only_the_work_with_it_wording() -> None:
    on = prompts.system_text("", consider_given_code=True)
    assert "walk through THAT code" in on and "how to complete or fix it" in on
    assert "Given code:" not in on                                 # the code rule's wording is not for prose
    assert prompts.system_text("", consider_given_code=False) == prompts.system_text("")


def test_user_asks_gain_one_sentence_after_the_ask() -> None:
    assert prompts.user_code("", "", consider_given_code=True).endswith(
        "Output the finished code now. If code for this task is already written on the screen, complete "
        "or fix that code rather than starting over.")
    assert prompts.user_code("", "", consider_given_code=False).endswith(
        "Output the finished code now. Write it from scratch: code already on screen is context, not a template.")
    assert prompts.user_solve("", "", consider_given_code=True).endswith(
        "complete or fix that code rather than starting over.")
    assert prompts.user_solve("", "", consider_given_code=False).endswith("context, not a template.")
    assert prompts.user_text("", "", consider_given_code=True).endswith(
        "Output the walkthrough now. If code for this task is already on screen, walk through completing or "
        "fixing that code.")
    assert prompts.user_text("", "") == prompts.user_text("", "", consider_given_code=False)


def test_a_given_focus_is_kept_and_the_sentence_follows_it() -> None:
    out = prompts.user_code("T", "s", "", "do it", consider_given_code=True)
    assert out.endswith("do it If code for this task is already written on the screen, complete or fix that "
                        "code rather than starting over.")
    assert prompts.user_text("T", "s", "", "explain", consider_given_code=True).endswith(
        "explain If code for this task is already on screen, walk through completing or fixing that code.")


def test_rules_are_stable_across_calls_for_the_kv_cache() -> None:
    for consider in (True, False):
        assert prompts.system_code("p", consider_given_code=consider) == prompts.system_code("p", consider_given_code=consider)
        assert prompts.system_solve("p", consider_given_code=consider) == prompts.system_solve("p", consider_given_code=consider)
    assert prompts._given_code_rule(None) == "" and prompts._given_code_lines(None) == []
    assert prompts._given_code_ask("code", None) == "" and prompts._given_code_ask("text", False) == ""


# --------------------------------------------------------------------------------------------
# the Assistant: the setting read live, the fresh flag
# --------------------------------------------------------------------------------------------


def test_default_config_considers_given_code() -> None:
    assert Config().assist.consider_given_code is True


def _assistant(cfg: Config, bus: EventBus | None = None, client: FakeClient | None = None) -> Assistant:
    return Assistant(cfg, bus or EventBus(), client or FakeClient(token_batches=[["x"]], chunk_delay=0.0),
                     Transcript(), clock=time.monotonic)


def _job(kind: str, job_id: int = 1, **kw: object) -> _AssistJob:
    return _AssistJob(id=job_id, kind=kind, question="q", cancel=threading.Event(), **kw)  # type: ignore[arg-type]


def test_assistant_system_prompts_follow_the_setting_and_the_fresh_flag() -> None:
    cfg = Config()
    a = _assistant(cfg)
    code, solve, text = _job("code"), _job("solve", 2), _job("text", 3)
    fresh = _job("code", 4, fresh=True)
    assert "WORK WITH IT" in a._system_for(code) and "WORK WITH IT" in a._system_for(solve)
    assert "walk through THAT code" in a._system_for(text)
    assert "write the best solution from scratch" in a._system_for(fresh)
    assert "WORK WITH IT" not in a._system_for(fresh)
    cfg.assist.consider_given_code = False                 # read live: no restart, no re-creation
    assert "write the best solution from scratch" in a._system_for(code)
    assert "write the best solution from scratch" in a._system_for(solve)
    assert "walk through THAT code" not in a._system_for(text)
    assert "write the best solution from scratch" in a._system_for(fresh)
    assert a._consider_given_code(code) is False and a._consider_given_code(fresh) is False
    cfg.assist.consider_given_code = True
    assert a._consider_given_code(code) is True and a._consider_given_code(fresh) is False


def test_assistant_user_turns_carry_the_matching_sentence() -> None:
    cfg = Config()
    a = _assistant(cfg)
    assert a._user_for(_job("code"), "T", "screen", previous="").endswith("rather than starting over.")
    assert a._user_for(_job("code", fresh=True), "T", "screen", previous="").endswith("not a template.")
    assert a._user_for(_job("text"), "T", "screen", previous="").endswith("completing or fixing that code.")
    assert a._user_for(_job("solve"), "T", "screen").endswith("rather than starting over.")
    cfg.assist.consider_given_code = False
    assert a._user_for(_job("code"), "T", "screen", previous="").endswith("not a template.")
    assert a._user_for(_job("text"), "T", "screen", previous="").endswith("Output the walkthrough now.")


def test_code_fresh_job_runs_as_a_code_job_with_its_own_label_and_prompt() -> None:
    cfg = Config()
    bus = EventBus()
    client = FakeClient(token_batches=[["def f(): pass"]], chunk_delay=0.0)
    a = _assistant(cfg, bus, client)
    sub = bus.subscribe()
    a.start()
    try:
        job_id = a.code(fresh=True)
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert started.job_id == job_id and started.kind == "code"
        assert started.question == "the task being discussed · from scratch"
        prompt = client.calls[-1]["prompt"]
        assert "write the best solution from scratch" in prompt and "WORK WITH IT" not in prompt
        # an ordinary Code press right after: the setting is on, so it works with the given code
        a.code("")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        started2 = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert started2.question == "the task being discussed"
        assert "WORK WITH IT" in client.calls[-1]["prompt"]
    finally:
        a.stop()


def test_fresh_is_ignored_for_other_kinds_and_a_label_wins() -> None:
    cfg = Config()
    a = _assistant(cfg)
    job_id = a.code("focus text", label="Hands-free label", fresh=True)
    assert isinstance(job_id, int)
    assert a._consider_given_code(_job("text", fresh=True)) is True        # only a code job can be fresh
