"""Tests for the `label` plumbing added to `Engine.code`/`Engine.text` and `_HandsFreeHooks`
(exocortex/app.py): a deferred request now carries a 3-tuple `(kind, text, label)` and is
replayed as `assistant.code(text, label=label)` / `assistant.text(text, label=label)` once the
LLM is ready, and `_HandsFreeHooks.code`/`.text` forward both `focus` and `label` through to the
matching `Engine` method unchanged.

`engine_env` / `_wait` / `_peek` / `app_mod` are imported from tests/test_app.py rather than
redefined here (same fakes, same LLM-unreachable-by-design setup), exactly as
tests/test_assistant_text.py imports its fixtures from tests/test_assistant.py -- see that
file's module docstring. This mirrors tests/test_app.py's own
`test_code_is_deferred_and_replayed_once_the_llm_is_ready` / `test_text_is_deferred_and_replayed_once_the_llm_is_ready`,
adding the `label` dimension those two do not cover.
"""
from __future__ import annotations

from exocortex.bus import AssistStartedEvent, EventBus
from exocortex.config import Config
from tests.test_app import _peek, _wait, app_mod, engine_env  # noqa: F401 -- engine_env is a fixture

# --------------------------------------------------------------------------------------------
# Engine.code / Engine.text: deferred 3-tuple, replayed with the label
# --------------------------------------------------------------------------------------------


def test_code_with_label_is_deferred_as_a_3_tuple_and_replays_with_label(
    engine_env: tuple[Config, EventBus],
) -> None:
    cfg, bus = engine_env
    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        eng.code("some focus instruction", label="the coding task on screen")
        assert (
            ("code", "some focus instruction", "the coding task on screen") in eng._pending
            or eng._llm_ready.is_set()
        )
        assert _wait(lambda: eng._llm_ready.is_set())
        assert _wait(
            lambda: any(
                isinstance(e, AssistStartedEvent) and e.kind == "code" and e.question == "the coding task on screen"
                for e in _peek(q)
            )
        )
    finally:
        eng.stop()


def test_text_with_label_is_deferred_as_a_3_tuple_and_replays_with_label(
    engine_env: tuple[Config, EventBus],
) -> None:
    cfg, bus = engine_env
    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        eng.text("walk through it", label="the walkthrough task")
        assert (
            ("text", "walk through it", "the walkthrough task") in eng._pending
            or eng._llm_ready.is_set()
        )
        assert _wait(lambda: eng._llm_ready.is_set())
        assert _wait(
            lambda: any(
                isinstance(e, AssistStartedEvent) and e.kind == "text" and e.question == "the walkthrough task"
                for e in _peek(q)
            )
        )
    finally:
        eng.stop()


def test_code_without_label_still_defers_and_replays(engine_env: tuple[Config, EventBus]) -> None:
    """No regression: the plain (no-label) path -- an empty label in the 3-tuple -- still works."""
    cfg, bus = engine_env
    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        eng.code("fix the bug")
        assert ("code", "fix the bug", "") in eng._pending or eng._llm_ready.is_set()
        assert _wait(lambda: eng._llm_ready.is_set())
        assert _wait(
            lambda: any(
                isinstance(e, AssistStartedEvent) and e.kind == "code" and e.question == "fix the bug"
                for e in _peek(q)
            )
        )
    finally:
        eng.stop()


# --------------------------------------------------------------------------------------------
# _HandsFreeHooks.code / .text: forward (focus, label) unchanged
# --------------------------------------------------------------------------------------------


class _FakeEngineForHooks:
    """Stand-in for `Engine`: only the two methods `_HandsFreeHooks.code`/`.text` call."""

    def __init__(self) -> None:
        self.code_calls: list[tuple[str, str]] = []
        self.text_calls: list[tuple[str, str]] = []

    def code(self, focus: str = "", label: str = "") -> int | None:
        self.code_calls.append((focus, label))
        return 42

    def text(self, focus: str = "", label: str = "") -> int | None:
        self.text_calls.append((focus, label))
        return 43


def test_hands_free_hooks_code_forwards_focus_and_label() -> None:
    fake = _FakeEngineForHooks()
    hooks = app_mod._HandsFreeHooks(fake)
    result = hooks.code("some long focus", "the coding task on screen")
    assert fake.code_calls == [("some long focus", "the coding task on screen")]
    assert result == 42


def test_hands_free_hooks_text_forwards_focus_and_label() -> None:
    fake = _FakeEngineForHooks()
    hooks = app_mod._HandsFreeHooks(fake)
    result = hooks.text("some long focus", "the walkthrough task")
    assert fake.text_calls == [("some long focus", "the walkthrough task")]
    assert result == 43


def test_hands_free_hooks_code_and_text_default_to_blank_focus_and_label() -> None:
    fake = _FakeEngineForHooks()
    hooks = app_mod._HandsFreeHooks(fake)
    hooks.code()
    hooks.text()
    assert fake.code_calls == [("", "")]
    assert fake.text_calls == [("", "")]
