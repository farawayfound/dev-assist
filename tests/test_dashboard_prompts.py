"""The dashboard refresh prompt and its parser (exocortex.dashboard.prompts, 2026-10-20)."""
from __future__ import annotations

from exocortex.bus import DashboardItem
from exocortex.dashboard import prompts as dp
from exocortex.llm import prompts as lp


def _item(item_id: int, kind: str, text: str) -> DashboardItem:
    return DashboardItem(id=item_id, kind=kind, text=text)


def test_sections_and_kinds() -> None:
    assert dp.normalize_sections(()) == () and dp.normalize_sections("questions") == ("questions",)
    assert dp.normalize_sections(("nextsteps", "QUESTIONS", "nope")) == ("questions", "nextsteps")
    assert dp.normalize_sections(("blindspot", "prompt")) == ("blindspots", "nextsteps")   # kind names work too
    assert dp.kinds_for(()) == ("question", "blindspot", "nextstep", "prompt")
    assert dp.kinds_for((), suggest_prompts=False) == ("question", "blindspot", "nextstep")
    assert dp.kinds_for(("questions",)) == ("question",)
    assert dp.kinds_for(("nextsteps",), suggest_prompts=False) == ("nextstep",)


def test_render_current_assigns_aliases_in_display_order() -> None:
    text, aliases = dp.render_current(
        {"question": [_item(10, "question", "Is it cached?"), _item(12, "question", "Which tenant?")],
         "prompt": [_item(7, "prompt", "/recon caching")]},
        ("question", "blindspot", "prompt"),
    )
    assert aliases == {"Q1": 10, "Q2": 12, "P1": 7}
    assert text.splitlines() == [
        "Questions to ask:", "- Q1: Is it cached?", "- Q2: Which tenant?",
        "Blind spots & follow-ups:", "- (none yet)",
        "Suggested prompts:", "- P1: /recon caching",
    ]
    assert dp.render_open_markdown([_item(1, "question", "a"), DashboardItem(id=2, kind="question", text="b", status="answered")]) == "- a"


def test_parse_update_reads_every_verb_shape() -> None:
    answer = """## Questions to ask
- NEW: Should export() paginate past 1000 rows?
- ANSWERED Q2: yes, the cache is keyed by tenant id
- Q1 answered: it retries three times
- KEEP Q3
- Q4
Another new one without a dash
## Blind spots & follow-ups
- DROP B1, B3
- RESOLVED B2 - test added
- (no change)
## Next steps
- DONE N1: the failing test is in
- Run the suite again
## Suggested prompts
- NEW: /repro-fix export() returns 500 on an empty tenant
- [x] P1 ran
- [ ] /check
- new: /repro-fix export() returns 500 on an empty tenant
"""
    u = dp.parse_update(answer, max_new=4)
    assert u.adds == [
        ("question", "Should export() paginate past 1000 rows?"),
        ("question", "Another new one without a dash"),
        ("nextstep", "Run the suite again"),
        ("prompt", "/repro-fix export() returns 500 on an empty tenant"),
        ("prompt", "/check"),
    ]
    assert u.resolves == [("Q2", "yes, the cache is keyed by tenant id"), ("Q1", "it retries three times"),
                          ("B2", "test added"), ("N1", "the failing test is in"), ("P1", "ran")]
    assert u.drops == ["B1", "B3"] and u.keeps == ["Q3", "Q4"] and u.unparsed == [] and not u.is_empty


def test_parse_update_caps_new_items_and_ignores_other_headings() -> None:
    answer = "## Questions to ask\n- NEW: a\n- NEW: b\n- NEW: c\n## Summary\n- NEW: not a section\n## Next steps\n- ANSWERED: no id here\n"
    u = dp.parse_update(answer, max_new=2)
    assert u.adds == [("question", "a"), ("question", "b")]
    assert u.unparsed == ["ANSWERED: no id here"] and u.resolves == []
    only = dp.parse_update("## Questions to ask\n- NEW: q\n## Next steps\n- NEW: n\n", kinds=("question",))
    assert only.adds == [("question", "q")]
    assert dp.parse_update("(nothing to do)").is_empty and dp.parse_update("").is_empty
    assert dp.parse_update("- a bare line before any heading").adds == [("question", "a bare line before any heading")]


def test_system_prompt_is_config_only_and_carries_the_toolkit_rule_when_given() -> None:
    a = dp.system_dashboard(persona="me", role="navigator", context="billing", max_new=3)
    b = dp.system_dashboard(persona="me", role="navigator", context="billing", max_new=3)
    assert a == b                                                   # byte-identical for the KV prefix cache
    assert "The user is the NAVIGATOR" in a and "Session context (what the team is working on): billing" in a
    assert "About me: me" in a and "at most 3 per heading" in a and lp.SKIP_MARKER in a
    for heading in dp.HEADINGS.values():
        assert f'"{heading}"' in a
    assert lp.TOOLKIT_RULE not in a and "plain, self-contained request" in a
    with_tk = dp.system_dashboard(toolkit="<toolkit>\nSkills: /recon [topic]: look around.\n</toolkit>")
    assert lp.TOOLKIT_RULE in with_tk and "Use ONLY the skills, agents and workflows listed in <toolkit>" in with_tk
    assert with_tk.rstrip().endswith("</toolkit>")
    no_prompts = dp.system_dashboard(kinds=("question", "blindspot", "nextstep"))
    assert '"## Suggested prompts"' not in no_prompts and "Suggested prompts:" not in no_prompts


def test_user_prompt_layout_keeps_the_volatile_blocks_last() -> None:
    user = dp.user_dashboard(current="Questions to ask:\n- Q1: x", new="[00:10] Me: hi", earlier="[00:01] Sam: hello",
                             screen="def f(): pass", repo='<repo root="r">\na.py\n  L1: x\n</repo>',
                             toolkit_detail="<toolkit_detail>\n### /recon [topic] (skill)\nbody\n</toolkit_detail>")
    order = [user.index(tag) for tag in ("<current>", "<transcript_earlier>", "<transcript_new>", "<screen>", "<repo", "<toolkit_detail>")]
    assert order == sorted(order) and user.endswith("as instructed.")
    bare = dp.user_dashboard(current="", new="")
    assert "(nothing tracked yet)" in bare and "(nothing new was said)" in bare and "(nothing readable on screen)" in bare
    assert "<transcript_earlier>" not in bare and "<repo" not in bare and "<toolkit_detail>" not in bare
