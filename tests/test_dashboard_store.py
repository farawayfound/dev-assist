"""The dashboard's database (exocortex.dashboard.store, 2026-10-20): items with a status and an
answer, duplicate folding, one refresh's apply(), the view snapshot, the notes, persistence --
and (2026-10-21) the task tree (sub-tasks, goals and prompts hanging off an action item), the
constraints, the migration of a database written before parent_id existed."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from exocortex.dashboard.store import KINDS, DashboardStore, normalize_text


def test_normalize_text_folds_case_punctuation_and_whitespace() -> None:
    assert normalize_text("Is the  cache per-tenant?") == normalize_text("is the cache per tenant")
    assert normalize_text("") == ""


def test_kinds_list_tasks_before_the_things_that_refer_to_them() -> None:
    assert KINDS == ("nextstep", "goal", "constraint", "question", "blindspot", "prompt")


def test_add_folds_duplicates_by_status() -> None:
    s = DashboardStore()
    try:
        q = s.add_item("question", "Is the cache per tenant?", now=1.0)
        assert q is not None and q.status == "open" and q.source == "model" and q.created_at == 1.0 and q.parent_id is None
        assert s.add_item("question", "is the cache per tenant", now=2.0).id == q.id       # open twice: one item
        assert s.add_item("blindspot", "Is the cache per tenant?", now=2.0).id != q.id     # another kind: its own
        s.resolve(q.id, "yes", now=3.0)
        assert s.add_item("question", "Is the cache per tenant?", now=4.0) is None         # answered: not re-raised
        s.drop(q.id, now=5.0)
        reopened = s.add_item("question", "Is the cache per tenant?", now=6.0)             # dropped: reopened
        assert reopened is not None and reopened.id == q.id and reopened.status == "open" and reopened.answer == ""
        assert s.add_item("question", "   ", now=7.0) is None and s.add_item("nope", "x", now=7.0) is None
    finally:
        s.close()


def test_status_changes_edit_and_ordering() -> None:
    s = DashboardStore()
    try:
        a = s.add_item("nextstep", "Run the suite", now=1.0)
        b = s.add_item("nextstep", "Write the failing test", now=2.0)
        c = s.add_item("nextstep", "Open the PR", now=3.0, source="user")
        assert [i.id for i in s.open_items("nextstep")] == [a.id, b.id, c.id]              # oldest first
        assert [i.id for i in s.open_items("nextstep", limit=2)] == [a.id, b.id]
        s.resolve(a.id, "green", now=4.0)
        s.resolve(b.id, "", now=5.0)
        done = s.items("nextstep", "answered")
        assert [i.id for i in done] == [b.id, a.id] and done[1].answer == "green"           # newest resolved first
        s.reopen(a.id, now=6.0)
        assert s.get(a.id).status == "open" and s.get(a.id).answer == ""
        edited = s.edit(c.id, "  Open   the draft PR ", now=7.0)
        assert edited.text == "Open the draft PR" and s.edit(c.id, "", now=8.0).text == "Open the draft PR"
        assert s.get(999) is None and s.resolve(999, "x", now=1.0) is None
        assert s.counts()["nextstep"] == (2, 1)
    finally:
        s.close()


def test_apply_resolves_drops_then_adds_with_a_cap_per_kind() -> None:
    s = DashboardStore()
    try:
        q1 = s.add_item("question", "q1", now=1.0)
        b1 = s.add_item("blindspot", "b1", now=1.0)
        r = s.apply(
            [("question", "q2"), ("question", "q1"), ("question", "q3"), ("question", "q4"), ("bogus", "x")],
            [(q1.id, "answered in chat"), (999, "nobody")],
            [b1.id, b1.id],
            now=2.0, max_new_per_kind=2,
        )
        assert (r.added, r.resolved, r.dropped) == (2, 1, 1)
        # skipped: "q1" (answered just before, never raised again), "q4" (over the cap), the bogus
        # kind, the unknown id 999, and b1 dropped a second time
        assert r.skipped == 5 and r.changed and "+2 new" in r.describe() and "1 answered" in r.describe()
        assert [i.text for i in s.open_items("question")] == ["q2", "q3"]
        assert s.get(q1.id).status == "answered" and s.get(b1.id).status == "dropped"
        again = s.apply([], [], [], now=3.0)
        assert not again.changed and again.describe() == "nothing new"
    finally:
        s.close()


def test_snapshot_shows_open_items_and_the_most_recent_resolved_ones() -> None:
    s = DashboardStore()
    try:
        for n in range(5):
            item = s.add_item("question", f"q{n}", now=float(n))
            if n < 4:
                s.resolve(item.id, f"a{n}", now=10.0 + n)
        dropped = s.add_item("question", "gone", now=20.0)
        s.drop(dropped.id, now=21.0)
        snap = s.snapshot(resolved_keep=2)
        assert [i.text for i in snap.of("question", "open")] == ["q4"]
        assert [i.text for i in snap.of("question", "answered")] == ["q3", "q2"]
        assert all(i.status != "dropped" for i in snap.items)
        assert [i.text for i in s.snapshot(show_resolved=False).items] == ["q4"]
    finally:
        s.close()


# ------------------------------------------------------------------ the task tree --------------
def test_sub_tasks_goals_and_prompts_hang_off_an_action_item() -> None:
    s = DashboardStore()
    try:
        task = s.add_item("nextstep", "Task 1: segment evaluation", now=1.0)
        sub = s.add_item("nextstep", "parse the filter expression", now=2.0, parent_id=task.id)
        deeper = s.add_item("nextstep", "handle nested attributes", now=3.0, parent_id=sub.id)
        goal = s.add_item("goal", "returns the matching customer ids", now=4.0, parent_id=task.id)
        overall = s.add_item("goal", "every response is JSON", now=4.0)
        prompt = s.add_item("prompt", "/recon segment evaluation", now=5.0, parent_id=sub.id)
        assert task.parent_id is None and sub.parent_id == task.id
        assert deeper.parent_id == task.id                       # the tree is one level deep: re-attached to the task
        assert goal.parent_id == task.id and overall.parent_id is None and prompt.parent_id == sub.id
        assert [i.id for i in s.children(task.id)] == [sub.id, deeper.id, goal.id]
        assert [i.id for i in s.children(sub.id)] == [prompt.id]
    finally:
        s.close()


def test_a_parent_must_be_a_live_action_item_and_only_some_kinds_take_one() -> None:
    s = DashboardStore()
    try:
        task = s.add_item("nextstep", "Task 1", now=1.0)
        q = s.add_item("question", "who owns it?", now=1.0)
        assert s.add_item("question", "another?", now=2.0, parent_id=task.id).parent_id is None   # questions never nest
        assert s.add_item("constraint", "no new dependencies", now=2.0, parent_id=task.id).parent_id is None
        assert s.add_item("goal", "g1", now=2.0, parent_id=q.id).parent_id is None                # not an action item
        assert s.add_item("goal", "g2", now=2.0, parent_id=12345).parent_id is None               # no such item
        s.drop(task.id, now=3.0)
        assert s.add_item("nextstep", "late sub-task", now=4.0, parent_id=task.id).parent_id is None   # a dismissed task
    finally:
        s.close()


def test_duplicates_are_folded_within_the_same_parent_only() -> None:
    s = DashboardStore()
    try:
        t1 = s.add_item("nextstep", "Task 1", now=1.0)
        t2 = s.add_item("nextstep", "Task 2", now=1.0)
        a = s.add_item("nextstep", "write the tests", now=2.0, parent_id=t1.id)
        assert s.add_item("nextstep", "Write the tests.", now=3.0, parent_id=t1.id).id == a.id      # same parent: folded
        b = s.add_item("nextstep", "write the tests", now=3.0, parent_id=t2.id)
        assert b.id != a.id and b.parent_id == t2.id                                              # another task: its own
        assert s.add_item("nextstep", "write the tests", now=4.0).id not in (a.id, b.id)          # a task of its own too
    finally:
        s.close()


def test_dismissing_a_task_dismisses_what_is_still_open_under_it() -> None:
    s = DashboardStore()
    try:
        task = s.add_item("nextstep", "Task 1", now=1.0)
        open_sub = s.add_item("nextstep", "open step", now=2.0, parent_id=task.id)
        done_sub = s.add_item("nextstep", "done step", now=2.0, parent_id=task.id)
        s.resolve(done_sub.id, "finished", now=3.0)
        goal = s.add_item("goal", "criterion", now=2.0, parent_id=task.id)
        prompt = s.add_item("prompt", "/scope x", now=2.0, parent_id=open_sub.id)
        other = s.add_item("nextstep", "Task 2", now=2.0)
        s.drop(task.id, now=4.0)
        assert [s.get(i.id).status for i in (task, open_sub, goal, prompt)] == ["dropped"] * 4
        assert s.get(done_sub.id).status == "answered" and s.get(other.id).status == "open"   # history and others stay
        s.resolve(other.id, "done", now=5.0)                                                    # completing does not cascade
        s.add_item("nextstep", "kept step", now=5.0, parent_id=other.id)
        assert [i.text for i in s.open_items("nextstep")] == ["kept step"]
    finally:
        s.close()


def test_finishing_a_task_finishes_its_open_steps_and_retires_their_prompts() -> None:
    s = DashboardStore()
    try:
        task = s.add_item("nextstep", "Task 1", now=1.0)
        a = s.add_item("nextstep", "step a", now=2.0, parent_id=task.id)
        b = s.add_item("nextstep", "step b", now=2.0, parent_id=task.id)
        s.resolve(b.id, "early", now=3.0)
        goal = s.add_item("goal", "criterion", now=2.0, parent_id=task.id)
        p_task = s.add_item("prompt", "/scope task", now=2.0, parent_id=task.id)
        p_step = s.add_item("prompt", "/recon step", now=2.0, parent_id=a.id)
        s.resolve(task.id, "shipped", now=4.0)
        assert s.get(a.id).status == "answered" and s.get(a.id).answer == "with the task"
        assert s.get(b.id).answer == "early"                                                     # already done: untouched
        assert s.get(goal.id).status == "open"                                                   # met is a separate claim
        assert s.get(p_task.id).status == "dropped" and s.get(p_step.id).status == "dropped"
        s.resolve(a.id, "again", now=5.0)                                                        # a sub-task: no cascade
        assert s.get(task.id).answer == "shipped"
    finally:
        s.close()


def test_a_sub_task_of_a_step_whose_task_is_gone_is_a_task_of_its_own() -> None:
    s = DashboardStore()
    try:
        task = s.add_item("nextstep", "Task 1", now=1.0)
        step = s.add_item("nextstep", "step", now=2.0, parent_id=task.id)
        s.drop(task.id, now=3.0)
        s.reopen(step.id, now=4.0)                                   # the step came back; its task did not
        assert s.add_item("nextstep", "deeper", now=5.0, parent_id=step.id).parent_id is None
    finally:
        s.close()


def test_apply_adds_may_carry_a_parent_and_are_capped_per_parent() -> None:
    s = DashboardStore()
    try:
        t1 = s.add_item("nextstep", "Task 1", now=1.0)
        t2 = s.add_item("nextstep", "Task 2", now=1.0)
        r = s.apply(
            [("nextstep", "a", t1.id), ("nextstep", "b", t1.id), ("nextstep", "c", t1.id),    # over the cap of 2 for Task 1
             ("nextstep", "d", t2.id), ("goal", "g", t1.id), ("goal", "g-overall"), ("nextstep", "top")],
            [], [], now=2.0, max_new_per_kind=2,
        )
        assert (r.added, r.skipped) == (6, 1)
        assert [i.text for i in s.children(t1.id)] == ["a", "b", "g"] and [i.text for i in s.children(t2.id)] == ["d"]
        assert s.get(s.items("goal")[1].id).parent_id is None and "top" in [i.text for i in s.open_items("nextstep")]
    finally:
        s.close()


def test_apply_adds_may_name_a_parent_added_earlier_in_the_same_batch() -> None:
    s = DashboardStore()
    try:
        r = s.apply(
            [("nextstep", "Task 1", None, "N1"), ("nextstep", "step a", "N1", "N2"), ("goal", "done when green", "N1"),
             ("prompt", "/recon a", "N2"), ("goal", "names nothing real", "N9")],
            [], [], now=1.0,
        )
        task, sub = s.get(1), s.get(2)
        assert r.added == 5 and task.parent_id is None and sub.parent_id == task.id
        assert [(i.text, i.parent_id) for i in s.items("goal")] == [("done when green", task.id), ("names nothing real", None)]
        assert s.items("prompt")[0].parent_id == sub.id
        # the same batch again: the task folds into the open duplicate, and the new sub-task still lands under it
        s.apply([("nextstep", "Task 1", None, "N1"), ("nextstep", "step b", "N1")], [], [], now=2.0)
        assert [i.text for i in s.children(task.id, "open") if i.kind == "nextstep"] == ["step a", "step b"]
    finally:
        s.close()


def test_constraints_are_in_force_until_dropped() -> None:
    s = DashboardStore()
    try:
        c = s.add_item("constraint", "TypeScript on Node 22, no new dependencies", now=1.0)
        assert c.status == "open" and s.counts()["constraint"] == (1, 0)
        s.drop(c.id, now=2.0)
        assert s.counts()["constraint"] == (0, 0) and s.snapshot().of("constraint") == []
    finally:
        s.close()


# ------------------------------------------------------------------ notes, export, files ------
def test_notes_meta_and_markdown_export() -> None:
    s = DashboardStore()
    try:
        assert s.notes() == ("", None) and s.get_meta("x") is None and s.get_meta("x", "d") == "d"
        s.set_notes("## Summary\nhi", 12.5)
        s.set_notes("## Summary\nhi again", 13.0)
        assert s.notes() == ("## Summary\nhi again", 13.0)
        s.set_meta("seg", "7")
        s.set_meta("seg", "8")
        assert s.get_meta("seg") == "8"
        s.add_item("question", "Open one?", now=1.0)
        d = s.add_item("question", "Done one?", now=1.0)
        s.resolve(d.id, "yes", now=2.0)
        s.add_item("prompt", "/recon update deal", now=3.0)
        md = s.to_markdown()
        assert md.startswith("# Dashboard\n\n## Session notes\n\n## Summary\nhi again\n")
        assert "- [ ] Open one?" in md and "- [x] Done one? — yes" in md and "- `/recon update deal`" in md
        assert "## Blind spots & follow-ups\n\n- (none)" in md and "## Constraints\n\n- (none)" in md
        s.clear()
        assert s.items() == [] and s.notes() == ("", None) and s.get_meta("seg") is None
    finally:
        s.close()


def test_markdown_shows_the_task_tree_the_goals_and_the_constraints() -> None:
    s = DashboardStore()
    try:
        t1 = s.add_item("nextstep", "Task 1: segment evaluation", now=1.0)
        sub = s.add_item("nextstep", "parse the filter", now=2.0, parent_id=t1.id)
        done = s.add_item("nextstep", "write the types", now=3.0, parent_id=t1.id)
        s.resolve(done.id, "in types.ts", now=4.0)
        s.add_item("prompt", "/recon filters", now=5.0, parent_id=sub.id)
        s.add_item("goal", "returns matching customer ids", now=6.0, parent_id=t1.id)
        s.add_item("goal", "all responses are JSON", now=6.0)
        s.add_item("constraint", "no new dependencies", now=7.0)
        md = s.to_markdown()
        assert ("## Action items\n\n- [ ] Task 1: segment evaluation\n"
                "  - [ ] parse the filter\n    - Prompt: `/recon filters`\n"
                "  - [x] write the types — in types.ts\n") in md
        assert "- [ ] returns matching customer ids  (for: Task 1: segment evaluation)" in md
        assert "- [ ] all responses are JSON\n" in md and "## Constraints\n\n- no new dependencies\n" in md
        assert "## Suggested prompts" not in md             # every prompt hangs off a task
        order = [md.index(h) for h in ("## Session notes", "## Action items", "## Goals", "## Constraints",
                                       "## Questions to ask", "## Blind spots & follow-ups")]
        assert order == sorted(order)
    finally:
        s.close()


def test_a_file_backed_store_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "session" / "dashboard.sqlite"
    s = DashboardStore(path)
    try:
        task = s.add_item("nextstep", "Task 1", now=1.0)
        s.add_item("goal", "it works", now=1.0, parent_id=task.id)
        s.add_item("blindspot", "No test for an empty payload", now=1.0)
        s.set_notes("notes", 2.0)
    finally:
        s.close()
    assert path.is_file()
    again = DashboardStore(path)
    try:
        assert [i.text for i in again.items("blindspot")] == ["No test for an empty payload"]
        assert [i.parent_id for i in again.items("goal")] == [task.id]
        assert again.notes() == ("notes", 2.0) and again.path == path
    finally:
        again.close()


def test_a_database_from_before_parent_id_is_migrated_in_place(tmp_path: Path) -> None:
    path = tmp_path / "dashboard.sqlite"
    old = sqlite3.connect(str(path))
    old.executescript(
        "CREATE TABLE items (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, text TEXT NOT NULL, "
        "norm TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open', answer TEXT NOT NULL DEFAULT '', "
        "source TEXT NOT NULL DEFAULT 'model', created_at REAL NOT NULL, updated_at REAL NOT NULL);"
        "INSERT INTO items (kind, text, norm, created_at, updated_at) VALUES ('nextstep', 'Old step', 'old step', 1, 1);"
    )
    old.commit()
    old.close()
    s = DashboardStore(path)
    try:
        legacy = s.open_items("nextstep")
        assert [(i.text, i.parent_id) for i in legacy] == [("Old step", None)]
        sub = s.add_item("nextstep", "a sub-task", now=2.0, parent_id=legacy[0].id)
        assert sub.parent_id == legacy[0].id
    finally:
        s.close()
