"""The dashboard refresh prompt and its parser (exocortex.dashboard.prompts, 2026-10-20; reworked
2026-10-21 for constraints, goals and a task / sub-task tree)."""
from __future__ import annotations

from exocortex.bus import DashboardItem
from exocortex.dashboard import prompts as dp
from exocortex.llm import prompts as lp


def _item(item_id: int, kind: str, text: str, parent: int | None = None, created: float = 0.0) -> DashboardItem:
    return DashboardItem(id=item_id, kind=kind, text=text, parent_id=parent, created_at=created)


def test_sections_and_kinds() -> None:
    assert dp.normalize_sections(()) == () and dp.normalize_sections("questions") == ("questions",)
    assert dp.normalize_sections(("nextsteps", "QUESTIONS", "nope")) == ("nextsteps", "questions")
    assert dp.normalize_sections(("blindspot", "prompt", "constraint")) == ("nextsteps", "constraints", "blindspots")   # kind names work too
    assert dp.kinds_for(()) == ("nextstep", "goal", "constraint", "question", "blindspot", "prompt")
    assert dp.kinds_for((), suggest_prompts=False) == ("nextstep", "goal", "constraint", "question", "blindspot")
    assert dp.kinds_for(("questions",)) == ("question",) and dp.kinds_for(("constraints",)) == ("constraint",)
    assert dp.kinds_for(("nextsteps",), suggest_prompts=False) == ("nextstep",)
    # goals and prompts name action items, so a refresh of either shows the tasks too
    assert dp.kinds_for(("goals",)) == ("nextstep", "goal") and dp.kinds_for(("goals",), suggest_prompts=False) == ("nextstep", "goal")


def test_render_current_lists_the_task_tree_and_says_what_goals_and_prompts_are_for() -> None:
    text, aliases = dp.render_current(
        {"nextstep": [_item(13, "nextstep", "step b", parent=12), _item(10, "nextstep", "Task 1"),
                      _item(11, "nextstep", "step a", parent=10), _item(12, "nextstep", "Task 2")],
         "goal": [_item(20, "goal", "g for task 2", parent=12), _item(21, "goal", "overall")],
         "constraint": [_item(30, "constraint", "no new deps")],
         "prompt": [_item(40, "prompt", "/recon a", parent=11)]},
        ("nextstep", "goal", "constraint", "question", "prompt"),
    )
    assert text.splitlines() == [
        "Action items:", "- N1: Task 1", "  - N2: step a", "- N3: Task 2", "  - N4: step b",
        "Goals:", "- G1 (for N3): g for task 2", "- G2: overall",
        "Constraints:", "- C1: no new deps",
        "Questions to ask:", "- (none yet)",
        "Suggested prompts:", "- P1 (for N2): /recon a",
    ]
    assert aliases == {"N1": 10, "N2": 11, "N3": 12, "N4": 13, "G1": 20, "G2": 21, "C1": 30, "P1": 40}
    assert dp.render_open_markdown([_item(1, "question", "a"), DashboardItem(id=2, kind="question", text="b", status="answered")]) == "- a"


def test_render_current_caps_what_the_model_is_shown_and_keeps_orphans_at_the_top_level() -> None:
    many = [_item(i, "nextstep", f"Task {i}", created=float(i)) for i in range(1, 7)]
    text, aliases = dp.render_current({"nextstep": many}, ("nextstep",), limit=2)
    assert list(aliases) == ["N1", "N2", "N3", "N4"]                      # x2 for action items: the earliest four
    tree = [_item(1, "nextstep", "Task 1", created=1.0), _item(2, "nextstep", "Task 2", created=2.0),
            _item(3, "nextstep", "a", parent=2, created=3.0), _item(4, "nextstep", "b", parent=2, created=4.0),
            _item(5, "nextstep", "c", parent=2, created=5.0), _item(6, "nextstep", "Task 3", created=6.0)]
    text, aliases = dp.render_current({"nextstep": tree}, ("nextstep",), limit=1)
    assert text.splitlines() == ["Action items:", "- N1: Task 1", "- N2: Task 2", "  - N3: a", "  - N4: b", "  - N5: c"]   # whole tasks
    assert len(dp.render_current({"nextstep": many}, ("nextstep",))[1]) == 6          # limit 0 = everything
    qs = [_item(i, "question", f"q{i}") for i in range(1, 6)]
    assert list(dp.render_current({"question": qs}, ("question",), limit=2)[1]) == ["Q1", "Q2"]
    orphan, _ = dp.render_current({"nextstep": [_item(5, "nextstep", "lost sub-task", parent=99)]}, ("nextstep",))
    assert orphan.splitlines() == ["Action items:", "- N1: lost sub-task"]


def test_section_of_heading_is_loose_about_wording() -> None:
    kinds = {
        "Action items": "nextstep", "Next steps": "nextstep", "Tasks": "nextstep",
        "Goals (acceptance criteria)": "goal", "Acceptance criteria": "goal",
        "Constraints & guardrails": "constraint", "Rules": "constraint",
        "Questions to ask": "question", "Blind spots & follow-ups": "blindspot",
        "Suggested prompts": "prompt", "Next steps & prompts": "prompt", "Summary": None,
    }
    for title, kind in kinds.items():
        assert dp.section_of_heading(title) == kind, title


def test_parse_update_reads_every_verb_shape() -> None:
    answer = """## Action items
- NEW: Task 1: segment evaluation
- NEW under N1: parse the filter expression
- NEW for N1: map filters to customers
- NEW N1: bare alias form
- under N1: no verb at all
- DONE N2: the parser is in
- N3 done: types written
## Goals
- NEW for N1: attribute filters return the matching customer ids
- NEW: every response is JSON
- MET G1: the unit test passes
- G2 met: observed 200
## Constraints
- NEW: TypeScript on Node 22, no new dependencies
- DROP C1
- DONE C2: lifted
## Questions to ask
- ANSWERED Q2: yes, the cache is keyed by tenant id
- Q1 answered: it retries three times
- KEEP Q3
- Q4
Another new one without a dash
## Blind spots & follow-ups
- DROP B1, B3
- RESOLVED B2 - test added
- (no change)
## Suggested prompts
- NEW for N1: /recon segment evaluation
- NEW: /repro-fix export() returns 500 on an empty tenant
- [x] P1 ran
- [ ] /check
- DROP P2
"""
    u = dp.parse_update(answer, max_new=4)
    A = dp.Add
    assert u.adds == [
        A("nextstep", "Task 1: segment evaluation"), A("nextstep", "parse the filter expression", "N1"),
        A("nextstep", "map filters to customers", "N1"), A("nextstep", "bare alias form", "N1"),
        A("nextstep", "no verb at all", "N1"),
        A("goal", "attribute filters return the matching customer ids", "N1"), A("goal", "every response is JSON"),
        A("constraint", "TypeScript on Node 22, no new dependencies"),
        A("question", "Another new one without a dash"),
        A("prompt", "/recon segment evaluation", "N1"), A("prompt", "/repro-fix export() returns 500 on an empty tenant"),
        A("prompt", "/check"),
    ]
    assert u.resolves == [
        ("N2", "the parser is in"), ("N3", "types written"), ("G1", "the unit test passes"), ("G2", "observed 200"),
        ("C2", "lifted"), ("Q2", "yes, the cache is keyed by tenant id"), ("Q1", "it retries three times"),
        ("B2", "test added"), ("P1", "ran"),
    ]
    assert u.drops == ["C1", "B1", "B3", "P2"] and u.keeps == ["Q3", "Q4"] and u.unparsed == [] and not u.is_empty


def test_a_target_only_counts_for_kinds_that_can_have_one() -> None:
    u = dp.parse_update("## Questions to ask\n- NEW for N1: is it cached?\n## Constraints\n- NEW under N2: stay on Node 22\n")
    assert u.adds == [dp.Add("question", "for N1: is it cached?"), dp.Add("constraint", "under N2: stay on Node 22")]


def test_parse_update_caps_new_items_per_kind_and_per_task_and_ignores_other_headings() -> None:
    answer = ("## Action items\n- NEW: a\n- NEW: b\n- NEW: c\n- NEW under N1: s1\n- NEW under N1: s2\n- NEW under N1: s3\n"
              "- NEW under N1: s1\n## Summary\n- NEW: not a section\n## Goals\n- ANSWERED: no id here\n")
    u = dp.parse_update(answer, max_new=2)
    assert u.adds == [dp.Add("nextstep", "a"), dp.Add("nextstep", "b"), dp.Add("nextstep", "s1", "N1"),
                      dp.Add("nextstep", "s2", "N1")]
    assert u.unparsed == ["ANSWERED: no id here"] and u.resolves == []
    only = dp.parse_update("## Questions to ask\n- NEW: q\n## Goals\n- NEW: n\n", kinds=("question",))
    assert only.adds == [dp.Add("question", "q")]
    assert dp.parse_update("(nothing to do)").is_empty and dp.parse_update("").is_empty
    assert dp.parse_update("- a bare line before any heading").adds == [dp.Add("nextstep", "a bare line before any heading")]


def test_system_prompt_is_config_only_and_names_the_toolkit_command_format_when_given() -> None:
    a = dp.system_dashboard(persona="me", role="navigator", context="billing", max_new=3)
    b = dp.system_dashboard(persona="me", role="navigator", context="billing", max_new=3)
    assert a == b                                                   # byte-identical for the KV prefix cache
    assert "The user is the NAVIGATOR" in a and "Session context (what the team is working on): billing" in a
    assert "About me: me" in a and "at most 3 per heading" in a and lp.SKIP_MARKER in a
    for heading in dp.HEADINGS.values():
        assert f'"{heading}"' in a
    for rule in ("PRIMARY TASK", "NEW under N2", "acceptance criteria", "MET G3", "guardrails that MUST be respected", "DROP C2"):
        assert rule in a, rule
    assert lp.TOOLKIT_RULE not in a and "plain, self-contained request" in a
    with_tk = dp.system_dashboard(toolkit="<toolkit>\nSkills: /recon [topic]: look around.\n</toolkit>", example="/recon [topic]")
    assert lp.TOOLKIT_RULE in with_tk and "Start every prompt with a slash command from <toolkit> (like /recon [topic])" in with_tk
    assert "never invent one" in with_tk and "constraints and acceptance criteria" in with_tk
    assert with_tk.rstrip().endswith("</toolkit>")
    no_example = dp.system_dashboard(toolkit="<toolkit>\nSkills: /recon\n</toolkit>")
    assert "Start every prompt with a slash command from <toolkit> or" in no_example
    no_prompts = dp.system_dashboard(kinds=("nextstep", "goal", "constraint", "question", "blindspot"))
    assert '"## Suggested prompts"' not in no_prompts and "Suggested prompts:" not in no_prompts
    one = dp.system_dashboard(kinds=("constraint",))
    assert '"## Constraints"' in one and '"## Goals"' not in one and "PRIMARY TASK" not in one


def test_user_prompt_layout_keeps_the_volatile_blocks_last() -> None:
    user = dp.user_dashboard(current="Questions to ask:\n- Q1: x", new="[00:10] Me: hi", earlier="[00:01] Sam: hello",
                             screen="def f(): pass", repo='<repo root="r">\na.py\n  L1: x\n</repo>',
                             toolkit_detail="<toolkit_detail>\n### /recon [topic] (skill)\nbody\n</toolkit_detail>")
    order = [user.index(tag) for tag in ("<current>", "<transcript_earlier>", "<transcript_new>", "<screen>", "<repo", "<toolkit_detail>")]
    assert order == sorted(order) and user.endswith("as instructed.")
    bare = dp.user_dashboard(current="", new="")
    assert "(nothing tracked yet)" in bare and "(nothing new was said)" in bare and "(nothing readable on screen)" in bare
    assert "<transcript_earlier>" not in bare and "<repo" not in bare and "<toolkit_detail>" not in bare
