"""Repository lookups (exocortex.project.repo / provider, 2026-10-20): identifier extraction,
the file index, the budgeted search and the provider's config gating."""
from __future__ import annotations

from pathlib import Path

from exocortex.config import Config
from exocortex.project.provider import ProjectContext
from exocortex.project.repo import RepoHit, RepoIndex, extract_terms, render_hits


def test_extract_terms_keeps_identifier_shapes_only() -> None:
    terms = extract_terms(
        "We should make insights_now take the ScreenReader and fix exocortex/llm/questions.py; "
        "`render_insights` too; the DashboardStore class; MAX_ITEMS; export(tenant)",
        "def export(tenant):",
    )
    assert terms[0] == "render_insights"                       # back-ticked wins
    for expected in ("insights_now", "ScreenReader", "exocortex/llm/questions.py", "questions.py", "DashboardStore", "MAX_ITEMS"):
        assert expected in terms, expected
    assert "export" in terms and "the" not in terms and "should" not in terms
    assert extract_terms("so what do you think about the plan for tomorrow and the thing we said") == []
    assert extract_terms("visit https://example.com/x/y and 1.2.3 then", limit=3) == []
    assert len(extract_terms("a_b c_d e_f g_h i_j", limit=2)) == 2


def _tree(root: Path) -> None:
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "billing.py").write_text(
        "def export(tenant):\n    cache = TenantCache(tenant)\n    return cache.rows()\n\nclass TenantCache:\n    pass\n",
        encoding="utf-8",
    )
    (root / "src" / "pkg" / "other.py").write_text("x = 1\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_billing.py").write_text("from pkg.billing import export\n\ndef test_export():\n    assert export('t') == []\n", encoding="utf-8")
    (root / "node_modules" / "dep").mkdir(parents=True)
    (root / "node_modules" / "dep" / "index.js").write_text("export const TenantCache = 1\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (root / "big.py").write_text("TenantCache\n" * 40000, encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG\x00\x00")
    (root / "README.md").write_text("# Billing\n\nexport() paginates.\n", encoding="utf-8")


def test_index_lists_text_files_and_prunes_excludes(tmp_path: Path) -> None:
    _tree(tmp_path)
    clock = {"t": 0.0}
    index = RepoIndex(tmp_path, refresh_s=60, clock=lambda: clock["t"])
    files = {str(p).replace("\\", "/") for p in index.files()}
    assert files == {"README.md", "src/pkg/billing.py", "src/pkg/other.py", "tests/test_billing.py"}   # big.py too large, png binary
    (tmp_path / "new.py").write_text("TenantCache\n", encoding="utf-8")
    assert "new.py" not in {str(p) for p in index.files()}                  # cached
    clock["t"] = 61
    assert "new.py" in {str(p) for p in index.files()}                      # refreshed on the cadence
    assert index.available and RepoIndex(tmp_path / "missing").files() == [] and not RepoIndex(tmp_path / "missing").available
    capped = RepoIndex(tmp_path, max_files=2)
    assert len(capped.files()) == 2 and capped.truncated


def test_search_ranks_by_terms_matched_and_name_hits(tmp_path: Path) -> None:
    _tree(tmp_path)
    index = RepoIndex(tmp_path)
    hits = index.search(["TenantCache", "billing.py", "export"], max_files=3)
    by_file: dict[str, list[RepoHit]] = {}
    for h in hits:
        by_file.setdefault(h.path, []).append(h)
    assert list(by_file)[0] == "src/pkg/billing.py"                         # name match + every term
    assert by_file["src/pkg/billing.py"][0].line == 1 and by_file["src/pkg/billing.py"][0].term == "export"
    assert all(not p.startswith("node_modules") and p != "big.py" for p in by_file)
    assert len(by_file) <= 3
    assert index.search(["zzz_never_there"]) == [] and index.search(["ab"]) == []
    sensitive = index.search(["tenantcache"], max_files=5)                 # lower-case: case-insensitive
    assert any(h.path == "src/pkg/billing.py" for h in sensitive)
    strict = index.search(["Tenantcache"], max_files=5)                    # an upper-case letter: exact
    assert strict == []


def test_render_hits_only_whole_files_within_the_budget() -> None:
    hits = [
        RepoHit("a.py", 1, "def a():", "a"), RepoHit("a.py", 9, "return a", "a"),
        RepoHit("b.py", 0, "", "b"), RepoHit("c.py", 3, "c = 1", "c"),
    ]
    assert render_hits(hits, 1000) == "a.py\n  L1: def a():\n  L9: return a\nb.py\nc.py\n  L3: c = 1"
    assert render_hits(hits, 30) == "a.py\n  L1: def a():\nb.py"             # a.py's 2nd line and c.py's chunk do not fit
    assert render_hits(hits, 24) == "a.py\n  L1: def a():"
    assert render_hits(hits, 10) == "" and render_hits([], 100) == ""


def test_provider_respects_the_config(tmp_path: Path) -> None:
    _tree(tmp_path)
    cfg = Config()
    cfg.project.repo_root = str(tmp_path)
    cfg.project.toolkit = True
    provider = ProjectContext(cfg, home=tmp_path / "nohome")
    assert provider.repo_available and provider.repo_root == tmp_path
    block = provider.repo_block("where is TenantCache used?", "", time_budget_s=2.0)
    assert block.startswith(f'<repo root="{tmp_path.name}">') and "src/pkg/billing.py" in block and block.endswith("</repo>")
    assert provider.repo_block("nothing identifier like here") == ""
    cfg.project.repo_search = False
    assert provider.repo_block("TenantCache") == ""
    cfg.project.repo_search = True
    assert provider.toolkit_block() == "" and provider.toolkit_lookup("/recon") == "" and not provider.has_toolkit
    assert provider.toolkit_summary() == ""
    # a toolkit installed later is picked up by refresh()
    (tmp_path / ".claude" / "skills" / "recon").mkdir(parents=True)
    (tmp_path / ".claude" / "skills" / "recon" / "SKILL.md").write_text(
        "---\nname: recon\ndescription: look around\nargument-hint: '[topic]'\n---\nRead-only.", encoding="utf-8")
    assert provider.refresh(force=True) is True and provider.has_toolkit
    assert provider.toolkit_block().startswith("<toolkit>\n") and "/recon [topic]: look around" in provider.toolkit_block()
    assert "### /recon [topic] (skill)" in provider.toolkit_lookup("try /recon first")
    assert provider.toolkit_summary().startswith("1 skill (")
    cfg.project.toolkit = False
    assert provider.toolkit_block() == "" and provider.toolkit_lookup("/recon") == ""
    cfg.project.repo_root = ""
    provider.reconfigure()
    assert not provider.repo_available and provider.repo_block("TenantCache") == ""
