"""Offline tests for exocortex.workspace (2026-10-29, multi-file tasks): how screen readings are sorted
into per-file views -- the same file scrolled, another file by title or by shared lines / identifiers,
the task statement in a browser beside the IDE, a glance at a chat, a run of foreign task-like captures
(a new problem without a checkpoint), eviction, staleness -- and how the OTHER views render for the
<workspace> block."""
from __future__ import annotations

from exocortex import workspace as ws
from exocortex.workspace import View, Workspace, containment, file_in_title, identifiers, looks_like_task

CHROME = ["EXPLORER", "src", "repo.py", "models.py", "tests", "test_repo.py", "main - myproj - Visual Studio Code"]

REPO_PY = CHROME + [
    "class UserRepository:",
    "    def __init__(self, session: Session) -> None:",
    "        self.session = session",
    "    def get_user(self, user_id: int) -> User:",
    "        return self.session.query(User).get(user_id)",
    "    def list_users(self) -> list[User]:",
    "        return self.session.query(User).all()",
]
REPO_PY_SCROLLED = CHROME + [
    "    def list_users(self) -> list[User]:",
    "        return self.session.query(User).all()",
    "    def save_user(self, user: User) -> None:",
    "        self.session.add(user)",
    "        self.session.commit()",
    "    def delete_user(self, user_id: int) -> None:",
    "        self.session.delete(self.get_user(user_id))",
]
MODELS_PY = CHROME + [
    "from dataclasses import dataclass",
    "@dataclass",
    "class User:",
    "    user_id: int",
    "    first_name: str",
    "    email_address: str",
]
TEST_REPO_PY = CHROME + [
    "from repo import UserRepository",
    "def test_get_user(session):",
    "    repository = UserRepository(session)",
    "    assert repository.get_user(1).email_address == 'a@b.c'",
]
BRIEF = [
    "Take-home assignment",
    "Implement a UserRepository class with get_user, list_users, save_user and delete_user methods",
    "backed by the SQLAlchemy session passed to its constructor. Add a dataclass User in models.py",
    "and unit tests in test_repo.py. Please keep the public method names as given.",
]
CHAT = ["Alex: hey are we still on for lunch?", "Sam: sure, noon works", "Alex: great see you then", "Sam: bring the slides"]
LEETCODE_A = ["Problems", "Description", "Solutions", "Submissions", "Run", "Submit", "Python3",
              "1. Two Sum", "Given an array of integers nums and an integer target, return indices",
              "of the two numbers such that they add up to target.",
              "class Solution:", "    def twoSum(self, nums: List[int], target: int) -> List[int]:", "        pass"]
LEETCODE_B = ["Problems", "Description", "Solutions", "Submissions", "Run", "Submit", "Python3",
              "9. Palindrome Number", "Given an integer x, return true if x is a palindrome, and false otherwise.",
              "class Solution:", "    def isPalindrome(self, x: int) -> bool:", "        pass"]
OTHER_PROJECT = [
    "EXPLORER", "lib", "parser.rs", "lexer.rs", "Cargo.toml",
    "pub struct Lexer<'a> {", "    source: &'a str,", "    position: usize,", "}",
    "impl<'a> Lexer<'a> {", "    pub fn next_token(&mut self) -> Option<Token> {",
]
OTHER_PROJECT_2 = OTHER_PROJECT[:5] + [
    "    pub fn peek_char(&self) -> Option<char> {", "        self.source[self.position..].chars().next()", "    }",
    "    fn skip_whitespace(&mut self) {", "        while let Some(c) = self.peek_char() {",
]


def text(lines: list[str]) -> str:
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def test_file_in_title_finds_the_file_in_editor_titles_and_nothing_elsewhere() -> None:
    assert file_in_title("● main.py - myproj - Visual Studio Code") == "main.py"
    assert file_in_title("myproj – Main.java") == "main.java"
    assert file_in_title("Repository.cs - MySolution - Microsoft Visual Studio") == "repository.cs"
    assert file_in_title("Two Sum - LeetCode") == ""
    assert file_in_title("") == ""
    assert file_in_title("www.example.com - Google Chrome") == ""   # a domain is not a file


def test_identifiers_keep_names_and_drop_generic_words() -> None:
    ids = identifiers("Implement a UserRepository with get_user and list_users; return the Solution integers")
    assert {"UserRepository", "get_user", "list_users"} <= ids
    assert "Solution" not in ids and "integers" not in ids and "return" not in ids


def test_containment_is_the_share_of_the_capture_found_in_the_view() -> None:
    assert containment({"a", "b", "c", "d"}, {"a", "b"}) == 0.5
    assert containment(set(), {"a"}) == 0.0


def test_looks_like_task_needs_code_or_a_statement() -> None:
    assert looks_like_task(REPO_PY, text(REPO_PY))
    assert looks_like_task(BRIEF, text(BRIEF))
    assert not looks_like_task(CHAT, text(CHAT))


# ---------------------------------------------------------------------------------------------
# sorting captures into views
# ---------------------------------------------------------------------------------------------
def test_the_same_file_scrolled_is_one_view_that_accumulates_its_lines() -> None:
    w = Workspace()
    v1 = w.observe(text(REPO_PY), title="repo.py - myproj - Visual Studio Code", process="Code.exe", now=1.0)
    v2 = w.observe(text(REPO_PY_SCROLLED), title="repo.py - myproj - Visual Studio Code", process="Code.exe", now=8.0)
    assert v1 is v2 and w.last_outcome == "same"
    assert len(w.views()) == 1 and v1.file == "repo.py" and v1.captures == 2
    merged = v1.text(10_000)
    assert "def get_user" in merged and "def delete_user" in merged      # earlier + later parts kept
    assert merged.index("def get_user") < merged.index("def save_user")  # first-seen order
    assert v1.last_seen == 8.0 and v1.first_seen == 1.0


def test_a_scroll_without_a_file_in_the_title_is_still_the_same_view_by_overlap() -> None:
    w = Workspace()
    a = w.observe(text(LEETCODE_A), title="Two Sum - LeetCode", now=1.0)
    scrolled = LEETCODE_A[:9] + ["Example 1:", "Input: nums = [2,7,11,15], target = 9", "Output: [0,1]"]
    b = w.observe(text(scrolled), title="Two Sum - LeetCode", now=2.0)
    assert a is b and w.last_outcome == "same"


def test_another_file_named_in_the_title_is_another_view_of_the_task() -> None:
    w = Workspace()
    repo = w.observe(text(REPO_PY), title="repo.py - myproj - Visual Studio Code", now=1.0)
    models = w.observe(text(MODELS_PY), title="models.py - myproj - Visual Studio Code", now=2.0)
    tests = w.observe(text(TEST_REPO_PY), title="test_repo.py - myproj - Visual Studio Code", now=3.0)
    assert repo is not models and models is not tests and w.last_outcome == "new"
    assert [v.file for v in w.views()] == ["repo.py", "models.py", "test_repo.py"]
    assert w.current is tests
    back = w.observe(text(REPO_PY), title="repo.py - myproj - Visual Studio Code", now=4.0)
    assert back is repo and w.current is repo and len(w.views()) == 3


def test_a_statement_in_the_browser_joins_the_ide_files_of_the_task() -> None:
    """The take-home brief in a browser tab (no file in its title, no IDE chrome) and the files in the
    editor are one task: the files join by their titles, the brief stays beside them."""
    w = Workspace()
    brief = w.observe(text(BRIEF), title="Assignment - Google Chrome", process="chrome.exe", now=1.0)
    assert brief is not None and brief.name == "Assignment - Google Chrome"
    for i, (lines, title) in enumerate(((REPO_PY, "repo.py - myproj - Visual Studio Code"),
                                        (REPO_PY_SCROLLED, "repo.py - myproj - Visual Studio Code"),
                                        (MODELS_PY, "models.py - myproj - Visual Studio Code"),
                                        (TEST_REPO_PY, "test_repo.py - myproj - Visual Studio Code"),
                                        (REPO_PY, "repo.py - myproj - Visual Studio Code"))):
        w.observe(text(lines), title=title, process="Code.exe", now=10.0 + i)
    assert brief in w.views() and len(w.views()) == 4
    others = w.others_text(max_chars=4000, now=20.0)
    assert "[Assignment - Google Chrome]" in others and "UserRepository class" in others


def test_a_pane_without_a_file_joins_by_shared_lines_or_identifiers() -> None:
    w = Workspace()
    w.observe(text(REPO_PY), title="repo.py - myproj - Visual Studio Code", now=1.0)
    # the same IDE, title unreadable: the shared file tree makes it another pane of the task
    v = w.observe(text(MODELS_PY), title="", now=2.0)
    assert v is not None and w.last_outcome == "new" and len(w.views()) == 2
    # a page sharing only identifiers (the class and a method name) with the code
    page = ["Design notes", "UserRepository owns the session; get_user raises when the id is unknown.",
            "list_users must not load relationships eagerly.", "Review by Friday."]
    v2 = w.observe(text(page), title="Notes - Google Chrome", now=3.0)
    assert v2 is not None and w.last_outcome == "new" and len(w.views()) == 3


def test_a_glance_at_a_chat_is_foreign_and_changes_nothing() -> None:
    w = Workspace()
    repo = w.observe(text(REPO_PY), title="repo.py - myproj - Visual Studio Code", now=1.0)
    for i in range(6):
        assert w.observe(text(CHAT + [f"Sam: line {i} of small talk"]), title="Slack", now=2.0 + i) is None
        assert w.last_outcome == "foreign"
    assert w.views() == [repo] and w.current is repo


def test_a_run_of_foreign_task_captures_restarts_the_workspace() -> None:
    """Problem A was never checkpointed; the user moved to a different project. Four consecutive
    task-like captures that belong together and share nothing with A restart the workspace."""
    w = Workspace()
    w.observe(text(LEETCODE_A), title="Two Sum - LeetCode", now=1.0)
    outcomes = []
    for i in range(ws.FOREIGN_RESET):
        lines = OTHER_PROJECT if i % 2 == 0 else OTHER_PROJECT_2
        w.observe(text(lines), title="lexer - Google Chrome", now=10.0 + i)   # a code review page: no file in the title
        outcomes.append(w.last_outcome)
    assert outcomes[:-1] == ["foreign"] * (ws.FOREIGN_RESET - 1) and outcomes[-1] == "reset"
    names = [v.name for v in w.views()]
    assert all("LeetCode" not in n for n in names) and len(w.views()) in (1, 2)
    assert "lexer.rs" in (w.current.text(10_000) if w.current else "")
    assert "twoSum" not in w.others_text(max_chars=4000, now=20.0)


def test_a_new_project_in_the_same_ide_restarts_the_workspace_after_a_run() -> None:
    """The IDE chrome (file tree, menus) is shared by every project, so a file of ANOTHER project
    would otherwise read as related; the project named in the title tells them apart."""
    w = Workspace()
    w.observe(text(REPO_PY), title="repo.py - myproj - Visual Studio Code", now=1.0)
    w.observe(text(MODELS_PY), title="models.py - myproj - Visual Studio Code", now=2.0)
    other = CHROME[:1] + ["lib", "parser.rs", "lexer.rs", "Cargo.toml", "pub struct Lexer<'a> {",
                          "    source: &'a str,", "    position: usize,", "}"]
    outcomes = []
    for i in range(ws.FOREIGN_RESET):
        w.observe(text(other), title="lexer.rs - rustlexer - Visual Studio Code", now=10.0 + i)
        outcomes.append(w.last_outcome)
    assert outcomes == ["foreign"] * (ws.FOREIGN_RESET - 1) + ["reset"]
    assert [v.file for v in w.views()] == ["lexer.rs"] and w.views()[0].project == "rustlexer"
    # the same project's other file is simply another view (no reset)
    w.observe(text(other[:5] + ["pub struct Parser {", "    tokens: Vec<Token>,", "}"]),
              title="parser.rs - rustlexer - Visual Studio Code", now=20.0)
    assert w.last_outcome == "new" and len(w.views()) == 2


def test_project_in_title_reads_the_common_editor_title_shapes() -> None:
    from exocortex.workspace import project_in_title

    assert project_in_title("● main.py - myproj - Visual Studio Code") == "myproj"
    assert project_in_title("myproj – Main.java") == "myproj"
    assert project_in_title("Repository.cs - MySolution - Microsoft Visual Studio") == "mysolution"
    assert project_in_title("Two Sum - LeetCode") == ""
    assert project_in_title("notes.txt - Notepad") == ""        # a file, no project


def test_an_edited_line_keeps_only_its_latest_reading_while_scrolled_lines_stay() -> None:
    w = Workspace()
    first = CHROME + ["def total(items):", "    retrun sum(items)", "    # TODO: handle None"]
    fixed = CHROME + ["def total(items):", "    return sum(items)", "    # TODO: handle None"]
    v = w.observe(text(first), title="util.py - myproj - Visual Studio Code", now=1.0)
    w.observe(text(fixed), title="util.py - myproj - Visual Studio Code", now=2.0)
    body = v.text(10_000)
    assert "return sum(items)" in body and "retrun" not in body
    scrolled = CHROME + ["def mean(items):", "    return total(items) / len(items)"]
    w.observe(text(scrolled), title="util.py - myproj - Visual Studio Code", now=3.0)
    body = v.text(10_000)
    assert "def total(items):" in body and "def mean(items):" in body       # scrolled away, kept


def test_a_foreign_run_broken_by_an_unrelated_capture_starts_over() -> None:
    w = Workspace()
    w.observe(text(LEETCODE_A), title="Two Sum - LeetCode", now=1.0)
    for i in range(ws.FOREIGN_RESET - 1):
        w.observe(text(OTHER_PROJECT), title="lexer - Google Chrome", now=10.0 + i)
    w.observe(text(CHAT), title="Slack", now=20.0)             # not task-like: the run is dropped
    for i in range(ws.FOREIGN_RESET - 1):
        w.observe(text(OTHER_PROJECT), title="lexer - Google Chrome", now=30.0 + i)
    assert w.last_outcome == "foreign" and len(w.views()) == 1   # still problem A; no reset yet


def test_the_next_problem_on_the_same_site_shares_its_chrome_and_joins_as_a_view() -> None:
    """LeetCode problem B after problem A without a checkpoint: the shared page chrome relates them
    (documented limit -- the checkpoint is the explicit boundary); A ages out of the block with time."""
    w = Workspace()
    a = w.observe(text(LEETCODE_A), title="Two Sum - LeetCode", now=1.0)
    b = w.observe(text(LEETCODE_B), title="Palindrome Number - LeetCode", now=2.0)
    assert a is not b and w.last_outcome == "new"
    assert "twoSum" in w.others_text(max_chars=4000, now=3.0)
    assert w.others_text(max_chars=4000, now=2.0 + ws.STALE_S + 1) == ""


def test_reset_forgets_everything() -> None:
    w = Workspace()
    w.observe(text(REPO_PY), title="repo.py - myproj - Visual Studio Code", now=1.0)
    w.observe(text(MODELS_PY), title="models.py - myproj - Visual Studio Code", now=2.0)
    w.reset()
    assert w.views() == [] and w.current is None and w.others_text() == ""
    v = w.observe(text(TEST_REPO_PY), title="test_repo.py - myproj - Visual Studio Code", now=3.0)
    assert w.views() == [v]


def test_empty_or_tiny_captures_teach_nothing() -> None:
    w = Workspace()
    assert w.observe("", now=1.0) is None and w.last_outcome == "empty"
    assert w.observe("OK\nCancel", now=2.0) is None and w.last_outcome == "empty"
    assert w.views() == []


def test_eviction_keeps_the_newest_views_and_the_current_one() -> None:
    w = Workspace()
    for i in range(ws.MAX_VIEWS + 3):
        lines = CHROME + [f"class Thing{i}:", f"    def method_{i}(self) -> int:", f"        return {i}"]
        w.observe(text(lines), title=f"thing{i}.py - myproj - Visual Studio Code", now=float(i))
    views = w.views()
    assert len(views) == ws.MAX_VIEWS
    assert views[0].file == "thing3.py" and views[-1].file == f"thing{ws.MAX_VIEWS + 2}.py"
    assert w.current is views[-1]


def test_observe_is_idempotent_for_a_repeated_reading() -> None:
    w = Workspace()
    v = w.observe(text(REPO_PY), title="repo.py - myproj - Visual Studio Code", now=1.0)
    again = w.observe(text(REPO_PY), title="repo.py - myproj - Visual Studio Code", now=1.0)
    assert again is v and len(w.views()) == 1 and v.text(10_000).count("class UserRepository:") == 1


# ---------------------------------------------------------------------------------------------
# rendering the OTHER views
# ---------------------------------------------------------------------------------------------
def test_others_text_lists_the_other_views_newest_first_under_headers_and_skips_the_current() -> None:
    w = Workspace()
    w.observe(text(REPO_PY), title="repo.py - myproj - Visual Studio Code", now=1.0)
    w.observe(text(MODELS_PY), title="models.py - myproj - Visual Studio Code", now=2.0)
    cur = w.observe(text(TEST_REPO_PY), title="test_repo.py - myproj - Visual Studio Code", now=3.0)
    out = w.others_text(max_chars=4000, now=3.0)
    assert out.index("[models.py]") < out.index("[repo.py]")          # newest first
    assert "[test_repo.py]" not in out and "test_get_user" not in out  # the current view is the <screen>
    assert "class UserRepository:" in out and "email_address: str" in out
    # an explicit current view wins over the remembered one
    out2 = w.others_text(current=w.views()[0], max_chars=4000, now=3.0)
    assert "[test_repo.py]" in out2 and "[repo.py]" not in out2
    assert cur is w.current


def test_others_text_respects_the_budget_and_shares_it() -> None:
    w = Workspace()
    for i in range(3):
        lines = CHROME + [f"class Thing{i}:"] + [f"    def method_{i}_{j}(self) -> int:  # line {j}" for j in range(40)]
        w.observe(text(lines), title=f"thing{i}.py - myproj - Visual Studio Code", now=float(i))
    w.observe(text(REPO_PY), title="repo.py - myproj - Visual Studio Code", now=10.0)
    out = w.others_text(max_chars=1500, now=10.0)
    assert len(out) <= 1500
    assert out.count("[thing") == 3                                   # every view got a share
    assert w.others_text(max_chars=0, now=10.0) == ""


def test_a_view_renders_its_name_and_a_cut_that_ends_on_a_line() -> None:
    v = View(file="", title="  Two   Sum - LeetCode ")
    assert v.name == "Two Sum - LeetCode"
    assert View().name == "pane"
    w = Workspace()
    view = w.observe(text(REPO_PY), title="repo.py - myproj - Visual Studio Code", now=1.0)
    cut = view.text(60)
    assert len(cut) <= 60 and not cut.endswith(" ") and "\n" in view.text(10_000)
