"""The seam between the two features that were built in parallel on separate machines:
explanations (a solve becomes a *plan* the completer types out) and multiple-choice answers
(the answer must end on a "**Answer: N - text**" verdict line).

They meet on a single "solve" job, and pull in opposite directions: the verdict is appended to
the answer, while the plan must keep the raw text -- a verdict line typed into an editor as
ghost text would be nonsense. These tests pin that split, plus the two false-positive risks the
combination creates (a coding problem read as a quiz; the model's own fenced answer read as one).
"""
from __future__ import annotations

from exocortex.bus import AssistDoneEvent
from exocortex.config import Config
from exocortex.llm.choices import find_choices
from exocortex.plan import align_plan, split_plan
from tests.test_assistant_complete import FakeClient, drain_until
from tests.test_assistant_explain import PROBLEM, ScriptedClient, chunks, make

#: The <choices> BLOCK (the volatile user-turn part, `choices.choices_block`). The system prompts
#: mention "<choices>" unconditionally -- the rule has to be byte-identical every turn or it would
#: defeat the KV prefix cache -- so the tag itself says nothing about whether a set was detected.
BLOCK_MARK = "This is a multiple-choice question."

#: A multiple-choice question of the kind the screen reader hands to a "solve" job.
QUIZ = "\n".join((
    "Which time complexity describes binary search on a sorted array?",
    "1. O(n)",
    "2. O(log n)",
    "3. O(n log n)",
    "4. O(1)",
    "Submit",
))

#: An explanations-mode answer to `QUIZ`: walkthrough, blank line, then the prose answer (a
#: question takes the "direct answer" branch of the plan shape, so there is no fenced block).
QUIZ_ANSWER = (
    "Walkthrough:\n"
    "The array is sorted, so every comparison halves the range left to search.\n"
    "Halving n until one element remains takes log2(n) steps.\n"
    "Option 2 is the only logarithmic choice.\n"
    "\n"
    "Binary search discards half of the remaining elements per step, so it runs in "
    "logarithmic time."
)


# =====================================================================================
# no forced verdict where there is no question to pick from
# =====================================================================================
def test_a_coding_problem_is_not_read_as_a_multiple_choice_question() -> None:
    """LeetCode's numbered title ("1. Two Sum") and its "Example 1:/Output:" rows must not look
    like an option run: a verdict appended to a solved coding problem would be nonsense, and the
    <choices> block would tell the model to pick an option instead of writing the code."""
    assert find_choices("\n".join(PROBLEM), question="") is None


def test_the_models_own_plan_answer_is_not_read_as_a_multiple_choice_question() -> None:
    """A fenced solution can contain enumerated-looking lines; re-detecting on the answer would
    append a verdict to code the completer then has to type."""
    from tests.test_plan import ANSWER

    assert find_choices(ANSWER, question="") is None


# =====================================================================================
# explanations + a real option set on one solve job
# =====================================================================================
def test_solve_appends_the_verdict_but_the_plan_keeps_the_raw_answer() -> None:
    """The Done event (Assist panel, export, console) ends on the verdict; `plan()` -- what the
    completer offers as ghost text -- must not contain it."""
    cfg = Config()
    cfg.explain.enabled = True
    client = ScriptedClient([chunks(QUIZ_ANSWER)])
    a, _bus, sub = make(cfg, client, screen=QUIZ)
    try:
        a.solve("")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text)
        done = [e for e in events if isinstance(e, AssistDoneEvent) and e.text][-1]

        assert done.text.rstrip().endswith("**Answer: 2 — O(log n)**")
        assert BLOCK_MARK in client.calls[-1]["prompt"]

        view = a.plan()
        assert view is not None and view.done
        assert "Answer:" not in view.solution
        assert view.solution == split_plan(QUIZ_ANSWER).solution
        assert a.plan_walkthrough.startswith("The array is sorted")
        # the answer history keeps the finalised text, so the verdict survives an export
        assert a.answers[-1][2].rstrip().endswith("**Answer: 2 — O(log n)**")
    finally:
        a.stop()


def test_a_coding_solve_with_explanations_still_aligns_for_ghost_text() -> None:
    """The plan path is untouched when no option set is found: the fenced solution still parses
    and `align_plan` still finds the caret in the header the user has typed."""
    from tests.test_plan import ANSWER, SOLUTION

    cfg = Config()
    cfg.explain.enabled = True
    client = ScriptedClient([chunks(ANSWER)])
    a, _bus, sub = make(cfg, client)
    try:
        a.solve("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text)
        view = a.plan()
        assert view is not None and view.is_code and view.solution == SOLUTION
        assert BLOCK_MARK not in client.calls[-1]["prompt"]
        cursor = align_plan(view.solution, "class Solution:\n    def twoSum(self, nums, target):\n")
        assert cursor is not None and cursor.rest
    finally:
        a.stop()


# =====================================================================================
# the lanes that must never ask for a verdict
# =====================================================================================
def test_explain_and_complete_jobs_never_detect_choices() -> None:
    """An explanation of the solution, and inline ghost text, must never end on a verdict line:
    the explanation is prose about code, and a completion is typed into the user's editor."""
    cfg = Config()
    cfg.explain.enabled = True
    client = ScriptedClient([chunks(QUIZ_ANSWER), ["- it halves ", "the range."]])
    a, _bus, sub = make(cfg, client, screen=QUIZ)
    try:
        a.solve("")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text)
        a.plan_typed(a.plan().id)
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.text.startswith("- it halves"))
        done = [e for e in events if isinstance(e, AssistDoneEvent) and e.text.startswith("- it halves")][-1]
        assert "Answer:" not in done.text
        assert BLOCK_MARK not in client.calls[-1]["prompt"]
    finally:
        a.stop()


def test_a_completion_on_a_quiz_screen_is_not_given_a_verdict() -> None:
    """Ghost text is typed straight into the user's editor, so a quiz visible on screen must
    never turn a completion into a verdict line."""
    cfg = Config()
    cfg.completion.enabled = True
    client = FakeClient(tokens=["O(log n)"])
    a, _bus, sub = make(cfg, client, screen=QUIZ)
    try:
        job_id = a.complete("The answer is ", "", "chrome")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == job_id)
        done = [e for e in events if isinstance(e, AssistDoneEvent) and e.job_id == job_id][-1]
        assert BLOCK_MARK not in client.calls[-1]["prompt"]
        assert "Answer:" not in done.text
    finally:
        a.stop()
