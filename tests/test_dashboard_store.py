"""The dashboard's database (exocortex.dashboard.store, 2026-10-20): items with a status and an
answer, duplicate folding, one refresh's apply(), the view snapshot, the notes, persistence."""
from __future__ import annotations

from pathlib import Path

from exocortex.dashboard.store import KINDS, DashboardStore, normalize_text


def test_normalize_text_folds_case_punctuation_and_whitespace() -> None:
    assert normalize_text("Is the  cache per-tenant?") == normalize_text("is the cache per tenant")
    assert normalize_text("") == ""


def test_add_folds_duplicates_by_status() -> None:
    s = DashboardStore()
    try:
        q = s.add_item("question", "Is the cache per tenant?", now=1.0)
        assert q is not None and q.status == "open" and q.source == "model" and q.created_at == 1.0
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
        q = s.add_item("question", "Open one?", now=1.0)
        d = s.add_item("question", "Done one?", now=1.0)
        s.resolve(d.id, "yes", now=2.0)
        s.add_item("prompt", "/recon update deal", now=3.0)
        md = s.to_markdown()
        assert md.startswith("# Dashboard\n\n## Session notes\n\n## Summary\nhi again\n")
        assert "- [ ] Open one?" in md and "- [x] Done one? — yes" in md and "- [ ] /recon update deal" in md
        for kind in KINDS:
            assert kind != "blindspot" or "## Blind spots & follow-ups\n\n- (none)" in md
        s.clear()
        assert s.items() == [] and s.notes() == ("", None) and s.get_meta("seg") is None
    finally:
        s.close()


def test_a_file_backed_store_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "session" / "dashboard.sqlite"
    s = DashboardStore(path)
    try:
        s.add_item("blindspot", "No test for an empty payload", now=1.0)
        s.set_notes("notes", 2.0)
    finally:
        s.close()
    assert path.is_file()
    again = DashboardStore(path)
    try:
        assert [i.text for i in again.items("blindspot")] == ["No test for an empty payload"]
        assert again.notes() == ("notes", 2.0) and again.path == path
    finally:
        again.close()
