"""Tests for exocortex.llm.choices: option-set detection, prompt annotation and answer
finalisation for multiple-choice questions. Fully offline, no `live` marker."""
from __future__ import annotations

import pytest

from exocortex.llm import prompts
from exocortex.llm.choices import (
    LETTER,
    NONE,
    NUMBER,
    ROMAN,
    Choice,
    ChoiceSet,
    annotate_choices,
    choices_block,
    finalize_answer,
    find_choices,
    format_answer_line,
)

# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def labels_texts(cs: ChoiceSet | None) -> list[tuple[str, str]] | None:
    return None if cs is None else [(c.label, c.text) for c in cs.choices]


def cs_of(labels: list[str], texts: list[str], style: str) -> ChoiceSet:
    return ChoiceSet(tuple(Choice(lab, t) for lab, t in zip(labels, texts)), style, (0, len(texts) - 1))


NUMBERED = cs_of(["1", "2", "3", "4"], ["4", "8", "16", "32"], NUMBER)
LETTERED = cs_of(["A", "B", "C", "D"], ["4", "6", "7", "9"], LETTER)
NAMED = cs_of(["A", "B", "C", "D"], ["Deadlock", "Livelock", "Starvation", "Race condition"], LETTER)
RADIO = cs_of(["", "", ""], ["Python is compiled", "Python is interpreted", "Python has no GC"], NONE)
TRUE_FALSE = cs_of(["", ""], ["True", "False"], NONE)


# --------------------------------------------------------------------------------------------
# find_choices -- positives (the shapes real quiz / exam / interview pages OCR into)
# --------------------------------------------------------------------------------------------

DETECT_POSITIVE: list[tuple[str, str, str, list[tuple[str, str]]]] = [
    (
        "numbered run under a question (the reported '1. 4' case)",
        "Question 3 of 20\nWhat is 2 ** 2?\n1. 4\n2. 8\n3. 16\n4. 32\nSubmit",
        NUMBER, [("1", "4"), ("2", "8"), ("3", "16"), ("4", "32")],
    ),
    (
        "OCR dropped the space after the enumerator: '1.4' is option 1, text 4",
        "What is the result of 2 XOR 6?\n1.4\n2.8\n3.16\n4.32",
        NUMBER, [("1", "4"), ("2", "8"), ("3", "16"), ("4", "32")],
    ),
    (
        "lettered options, dot",
        "Which of the following is a prime number?\nA. 4\nB. 6\nC. 7\nD. 9",
        LETTER, [("A", "4"), ("B", "6"), ("C", "7"), ("D", "9")],
    ),
    (
        "lettered options in parentheses, lower case",
        "Which one?\n(a) foo bar\n(b) baz\n(c) qux",
        LETTER, [("a", "foo bar"), ("b", "baz"), ("c", "qux")],
    ),
    (
        "radio glyph in front of a lettered option",
        "Q5. Which sorts are stable?\n○ A. merge sort\n○ B. quicksort\n○ C. heapsort",
        LETTER, [("A", "merge sort"), ("B", "quicksort"), ("C", "heapsort")],
    ),
    (
        "two options per OCR row, joined with two spaces",
        "Time complexity of binary search?\nA) O(n)  B) O(log n)\nC) O(1)  D) O(n log n)\nNext",
        LETTER, [("A", "O(n)"), ("B", "O(log n)"), ("C", "O(1)"), ("D", "O(n log n)")],
    ),
    (
        "a whole horizontal answer row in one OCR line, single spaces",
        "Which is a mammal? A) Frog B) Whale C) Shark D) Salmon",
        LETTER, [("A", "Frog"), ("B", "Whale"), ("C", "Shark"), ("D", "Salmon")],
    ),
    (
        "radio buttons, OCR'd as an empty circle glyph",
        "Select the correct statement.\n○ Python is compiled\n○ Python is interpreted\n○ Python has no GC",
        NONE, [("", "Python is compiled"), ("", "Python is interpreted"), ("", "Python has no GC")],
    ),
    (
        "radio buttons misread as the letter O",
        "Select the correct statement.\nO Python is compiled\nO Python is interpreted\nO Python has no GC",
        NONE, [("", "Python is compiled"), ("", "Python is interpreted"), ("", "Python has no GC")],
    ),
    (
        "checkboxes, select all that apply",
        "Which are even? (select all that apply)\n☐ 1\n☐ 2\n☐ 3\n☐ 4",
        NONE, [("", "1"), ("", "2"), ("", "3"), ("", "4")],
    ),
    (
        "true / false radios under a statement (no question mark, no question word)",
        "The smallest element of a min-heap is always at the root.\n○ True\n○ False",
        NONE, [("", "True"), ("", "False")],
    ),
    (
        "bare True / False lines right under a question",
        "True or False: The mitochondria is the powerhouse of the cell.\nTrue\nFalse",
        NONE, [("", "True"), ("", "False")],
    ),
    (
        "lettered true / false on one row under a statement",
        "In a min-heap, the smallest element is always at the root.\na) True  b) False",
        LETTER, [("a", "True"), ("b", "False")],
    ),
    (
        "roman numerals with the question below the statements",
        "Consider the following statements:\ni. Lookup is O(n log n).\nii. Collisions never occur.\n"
        "iii. Worst case is O(n).\nWhich statement is correct?",
        ROMAN, [("i", "Lookup is O(n log n)."), ("ii", "Collisions never occur."), ("iii", "Worst case is O(n).")],
    ),
    (
        "circled digits",
        "How many sides does a hexagon have?\n① 4\n② 5\n③ 6\n④ 7",
        NUMBER, [("1", "4"), ("2", "5"), ("3", "6"), ("4", "7")],
    ),
    (
        "label OCR'd onto its own line, text on the next",
        "What is the capital of France?\nA\nParis\nB\nLondon\nC\nMadrid\nFlag for review",
        LETTER, [("A", "Paris"), ("B", "London"), ("C", "Madrid")],
    ),
    (
        "OCR read '3.' and '4.' as 'C.' and 'D.' mid-run",
        "Which planet is known as the Red Planet?\n1. Mercury\n2. Venus\nC. Earth\nD. Mars\nSubmit",
        NUMBER, [("1", "Mercury"), ("2", "Venus"), ("3", "Earth"), ("4", "Mars")],
    ),
    (
        "cyrillic labels",
        "Сколько дней в неделе?\nА) 5\nБ) 6\nВ) 7\nГ) 8",
        LETTER, [("А", "5"), ("Б", "6"), ("В", "7"), ("Г", "8")],
    ),
    (
        "a wrapped last option keeps its lower-case continuation, not the chrome below it",
        "Select all valid HTTP methods.\n□ GET\n□ POST\n□ PATCH — used to apply partial\nmodifications to a resource\nCheck  Show Answer",
        NONE, [("", "GET"), ("", "POST"), ("", "PATCH — used to apply partial modifications to a resource")],
    ),
    (
        "code with a line-number gutter between the question and the options",
        "What does this print?\n1 def f():\n2     return 4\n3 print(f())\nA) 4\nB) None\nC) error",
        LETTER, [("A", "4"), ("B", "None"), ("C", "error")],
    ),
    (
        "'2 XOR 6 = ?' -- a bare question mark right above short numeric options",
        "2 XOR 6 = ?\n1. 4\n2. 8\n3. 16",
        NUMBER, [("1", "4"), ("2", "8"), ("3", "16")],
    ),
    (
        "an 'Options:' header between the question and numbered options",
        "Which is prime?\nOptions:\n1. 4\n2. 6\n3. 7",
        NUMBER, [("1", "4"), ("2", "6"), ("3", "7")],
    ),
    (
        "an 'Answer choices:' header alone licenses radio options",
        "Answer choices:\n○ Python is compiled\n○ Python is interpreted\n○ Python has no GC",
        NONE, [("", "Python is compiled"), ("", "Python is interpreted"), ("", "Python has no GC")],
    ),
    (
        "the question sits below the list ('Which of these …?')",
        "Users have reported several possible causes:\n1. A stale lock file\n2. An outdated GPU driver\n"
        "3. A corrupted config.toml\nWhich of these is the most common cause?",
        NUMBER, [("1", "A stale lock file"), ("2", "An outdated GPU driver"), ("3", "A corrupted config.toml")],
    ),
]


@pytest.mark.parametrize(("name", "text", "style", "expected"), DETECT_POSITIVE, ids=[c[0] for c in DETECT_POSITIVE])
def test_find_choices_detects(name: str, text: str, style: str, expected: list[tuple[str, str]]) -> None:
    cs = find_choices(text)
    assert cs is not None, name
    assert cs.style == style
    assert labels_texts(cs) == expected


def test_find_choices_typed_question_with_single_spaces() -> None:
    cs = find_choices("Which is prime? A) 4 B) 6 C) 7", typed=True)
    assert labels_texts(cs) == [("A", "4"), ("B", "6"), ("C", "7")]
    assert cs is not None and cs.lead == "Which is prime?"


def test_find_choices_span_covers_the_run_only() -> None:
    cs = find_choices("Question 3 of 20\nWhat is 2 ** 2?\n1. 4\n2. 8\n3. 16\n4. 32\nSubmit")
    assert cs is not None
    assert cs.span == (2, 5)
    assert cs.cue == "What is 2 ** 2?"


def test_find_choices_prefers_the_run_whose_cue_matches_the_question() -> None:
    page = (
        "Question 1: Which is a prime number?\nA) 4\nB) 6\nC) 7\nD) 8\n\n"
        "Question 2: Which is an even number?\nA) 3\nB) 4\nC) 5\nD) 7"
    )
    first = find_choices(page, question="Which is a prime number?")
    assert first is not None and first.texts == ("4", "6", "7", "8")
    second = find_choices(page, question="Which is an even number?")
    assert second is not None and second.texts == ("3", "4", "5", "7")
    # Without a question the last run wins (the options sit under the question being read).
    assert find_choices(page).texts == ("3", "4", "5", "7")


# --------------------------------------------------------------------------------------------
# find_choices -- negatives (what a shared screen shows most of the time)
# --------------------------------------------------------------------------------------------

DETECT_NEGATIVE: list[tuple[str, str]] = [
    ("numbered instruction steps, no question", "Steps:\n1. Clone the repo\n2. Install deps\n3. Run tests"),
    ("unrelated question above steps", "Have you tried turning it off and on again?\n1. Go to settings\n2. Click security\n3. Click reset password\n4. Check your email"),
    ("slide bullets", "Key Benefits\n• Faster performance\n• Lower cost\n• Easier maintenance"),
    ("lettered agenda", "Meeting Agenda\nA. Budget review\nB. Roadmap update\nC. Hiring plan"),
    ("two-way settings toggle", "A) Light Mode\nB) Dark Mode"),
    ("radio buttons on a settings page", "Theme\n○ Light\n○ Dark"),
    ("a table of decimals under a question about values", "Compute the values below. Which value is largest?\n1.5\n2.3\n3.7"),
    ("arithmetic lines", "Complete each multiplication.\n1.4 * 2 = 2.8\n3.6 / 2 = 1.8\n2.5 * 4 = 10.0"),
    ("LeetCode examples and constraints", "Two Sum\nExample 1:\nInput: nums = [2,7,11,15], target = 9\nOutput: [0,1]\nExample 2:\nInput: nums = [3,2,4]\nOutput: [1,2]\nConstraints:\n2 <= nums.length <= 10^4"),
    ("code gutter alone", "1 def f():\n2     return 4\n3 print(f())"),
    ("reasons introduced by a colon after a question", "Why does my Flask app hang under load?\nThere are three common reasons this happens:\n1. The dev server is single-threaded\n2. A blocking call holds the GIL\n3. The connection pool is exhausted"),
    ("repro steps followed by an unrelated question", "Repro steps:\n1. Open the app\n2. Toggle dark mode\n3. Restart the app\nCan someone confirm on their machine?"),
    ("a site poll", "Have you seen this question in an interview before?\n○ Yes, in the last 6 months\n○ Yes, more than 6 months ago\n○ No"),
    ("a ranking task", "Rank the following sorting algorithms from fastest to slowest:\n1. Bubble Sort\n2. Merge Sort\n3. Quicksort"),
    ("a changelog", "v2.0.1\n- Fixed crash on startup\n- Faster OCR\n- Updated deps"),
    ("keyboard shortcuts", "Settings > Keyboard Shortcuts\n• Ctrl+Shift+Space — Toggle overlay\n• Ctrl+Enter — Ask a question\n• Ctrl+R — Reset"),
    ("a table of contents", "Contents\n1. Introduction\n2. Installation\n3. Usage\n4. FAQ"),
    ("an agenda with 'Q3' in the title", "Q3 Planning.pptx\nAgenda\n1. Review Q3 metrics\n2. Discuss roadmap\n3. Staffing updates\n4. Open Q&A"),
    ("roman section headers", "Statement of Work.pdf\ni. Scope of Work\nii. Deliverables\niii. Timeline\niv. Payment Terms"),
    ("decimals and versions in prose", "Requires Python 3.12, about 1.4 GB of RAM and v2.0.1 of the driver. What should I do?"),
    ("a single option", "Which one?\nA. only one"),
    ("empty", ""),
]


@pytest.mark.parametrize(("name", "text"), DETECT_NEGATIVE, ids=[c[0] for c in DETECT_NEGATIVE])
def test_find_choices_rejects(name: str, text: str) -> None:
    assert find_choices(text) is None, name


# --------------------------------------------------------------------------------------------
# annotate_choices / choices_block
# --------------------------------------------------------------------------------------------


def test_annotate_choices_rewrites_only_the_option_lines() -> None:
    screen = "Question 3 of 20\nWhat is 2 ** 2?\n1. 4\n2. 8\n3. 16\n4. 32\nSubmit"
    cs = find_choices(screen)
    assert cs is not None
    assert annotate_choices(screen, cs) == (
        "Question 3 of 20\nWhat is 2 ** 2?\nChoice 1: 4\nChoice 2: 8\nChoice 3: 16\nChoice 4: 32\nSubmit"
    )


def test_annotate_choices_keeps_text_that_shared_a_row_with_the_options() -> None:
    screen = "Q3. What is 2+2?\nA) 3  B) 4  C) 5  Submit"
    cs = find_choices(screen)
    assert cs is not None
    assert annotate_choices(screen, cs) == "Q3. What is 2+2?\nChoice A: 3\nChoice B: 4\nChoice C: 5\nSubmit"
    typed = "Which is prime? A) 4 B) 6 C) 7"
    cs = find_choices(typed, typed=True)
    assert cs is not None
    assert annotate_choices(typed, cs) == "Which is prime?\nChoice A: 4\nChoice B: 6\nChoice C: 7"


def test_annotate_choices_leaves_unlabelled_runs_alone() -> None:
    screen = "Select the correct statement.\n○ Python is compiled\n○ Python is interpreted"
    cs = find_choices(screen)
    assert cs is not None and not cs.labeled
    assert annotate_choices(screen, cs) == screen


def test_choices_block_lists_options_and_asks_for_the_final_line() -> None:
    block = choices_block(NUMBERED)
    assert block.startswith("<choices>\n") and block.endswith("</choices>\n\n")
    assert "1: 4\n2: 8\n3: 16\n4: 32" in block
    assert '"Answer: <label> — <option text>"' in block
    assert "do not default to the first one" in block
    unlabelled = choices_block(RADIO)
    assert "- Python is interpreted" in unlabelled
    assert '"Answer: <option text>"' in unlabelled


def test_prompt_builders_place_the_block_after_the_screen_and_before_the_ask() -> None:
    block = choices_block(NUMBERED)
    solve = prompts.user_solve("hello", "screen text", "", block)
    assert solve.index("</screen>") < solve.index("<choices>") < solve.index("Solve whatever task")
    ask = prompts.user_ask("hello", "which one?", "screen text", block)
    assert ask.index("</screen>") < ask.index("<choices>") < ask.index("<question>")
    assist = prompts.user_assist("hello", "which one?", "", block)
    assert assist.index("</transcript>") < assist.index("<choices>") < assist.index("<question>")
    # Without a block the prompts are byte-identical to what they were.
    assert prompts.user_solve("hello", "screen text") == prompts.user_solve("hello", "screen text", "", "")


def test_system_prompts_carry_the_enumerator_rule_and_stay_stable() -> None:
    for system in (prompts.system_assist("", "bullets"), prompts.system_ask(""), prompts.system_solve("")):
        assert "never the decimal number 1.4" in system
        assert "<choices>" in system
    # The solve prompt separates the answer protocol from the [[CHOICE]] fork protocol.
    assert prompts.CHOICE_PREFIX in prompts.system_solve("").split("Enumerated options")[1]
    # Same (persona, style) -> same bytes, whatever was detected this turn (KV prefix cache).
    assert prompts.system_solve("me") == prompts.system_solve("me")


# --------------------------------------------------------------------------------------------
# finalize_answer -- an explicit Answer line is normalised
# --------------------------------------------------------------------------------------------

EXPLICIT: list[tuple[str, ChoiceSet, str, str]] = [
    ("label only", LETTERED, "7 is prime.\nAnswer: C", "**Answer: C — 7**"),
    ("bold label with a period", LETTERED, "7 is prime.\n**Answer: C.**", "**Answer: C — 7**"),
    ("final answer in parentheses", LETTERED, "7 is prime.\nFinal answer: (c)", "**Answer: C — 7**"),
    ("'option 3'", NUMBERED, "2^4 is 16.\nAnswer: option 3", "**Answer: 3 — 16**"),
    ("text only, labelled set", NUMBERED, "2^4 is 16.\nAnswer: 16", "**Answer: 3 — 16**"),
    ("the decimal misread of label + text", NUMBERED, "2 squared is 1.4.\nAnswer: 1.4", "**Answer: 1 — 4**"),
    ("label and text", NUMBERED, "So it is 4.\nAnswer: 1 — 4", "**Answer: 1 — 4**"),
    ("label disagrees with its text: the text wins", NUMBERED, "It is 16.\nAnswer: 2 — 16", "**Answer: 3 — 16**"),
    ("several labels", NAMED, "Both apply.\nAnswer: A, C", "**Answer: A, C — Deadlock; Starvation**"),
    ("unlabelled: the text", RADIO, "Bytecode runs on a VM.\nAnswer: Python is interpreted", "**Answer: Python is interpreted**"),
    ("true / false", TRUE_FALSE, "Yes it is.\nAnswer: True", "**Answer: True**"),
    ("an Answer line in the middle is moved to the end", LETTERED, "Answer: C\nBecause 7 is prime.", "**Answer: C — 7**"),
    ("a bare value that is also a label reads as the value", NUMBERED, "2 squared is 4.\nAnswer: 4", "**Answer: 1 — 4**"),
]


@pytest.mark.parametrize(("name", "cs", "text", "last"), EXPLICIT, ids=[c[0] for c in EXPLICIT])
def test_finalize_answer_normalises_an_explicit_answer_line(name: str, cs: ChoiceSet, text: str, last: str) -> None:
    out = finalize_answer(text, cs)
    assert out.rstrip().split("\n")[-1] == last
    assert out.count("Answer:") == 1, "the model's own Answer line is replaced, not duplicated"


def test_finalize_answer_keeps_the_body_and_separates_the_verdict_by_a_blank_line() -> None:
    out = finalize_answer("- 4 is 2 squared\n- 8 is 2 cubed, not this\nAnswer: 1", NUMBERED)
    assert out == "- 4 is 2 squared\n- 8 is 2 cubed, not this\n\n**Answer: 1 — 4**"


# --------------------------------------------------------------------------------------------
# finalize_answer -- no Answer line: the verdict is inferred from the body
# --------------------------------------------------------------------------------------------

INFERRED: list[tuple[str, ChoiceSet, str, str]] = [
    ("'so C is correct'", LETTERED, "4, 6 and 9 are composite. 7 has no divisors besides 1 and itself, so C is correct.", "**Answer: C — 7**"),
    ("'the correct option is A.'", NAMED, "Two threads each wait on the other's lock.\nThe correct option is A.", "**Answer: A — Deadlock**"),
    ("'so option 1'", NUMBERED, "2 squared is 4, so option 1.", "**Answer: 1 — 4**"),
    ("'both A and C are valid'", NAMED, "Deadlock and starvation both block progress.\nBoth A and C are valid.", "**Answer: A, C — Deadlock; Starvation**"),
    ("a rejected option is skipped", LETTERED, "B is wrong because 6 is even. 7 is prime, so it is right.", "**Answer: C — 7**"),
    ("the last value named after 'the answer is'", cs_of(["", "", "", ""], ["4", "8", "16", "32"], NONE), "The sequence doubles: 4, 8, 16, 32 -- so the third value, the answer is 16.", "**Answer: 16**"),
    ("an unlabelled option named in prose", RADIO, "Bytecode runs on a VM, so Python is interpreted.", "**Answer: Python is interpreted**"),
    ("'is false' for a true / false set", TRUE_FALSE, "TCP does not guarantee low latency. So the claim is false.", "**Answer: False**"),
    ("a sentence that is just the option", cs_of(["A", "B", "C", "D"], ["A stack", "A queue", "A hash map", "None of the above"], LETTER),
     "A hash map gives O(1) lookups but not O(1) delete by position, so no listed structure fits.\nNone of the above.", "**Answer: D — None of the above**"),
    ("never commits", NAMED, "Each of these can stall a system in its own way; it really depends on the workload.", "**Answer: (unclear)**"),
    ("says nothing usable", NUMBERED, "I'd need to check.", "**Answer: (unclear)**"),
]


@pytest.mark.parametrize(("name", "cs", "text", "last"), INFERRED, ids=[c[0] for c in INFERRED])
def test_finalize_answer_infers_the_verdict_from_the_body(name: str, cs: ChoiceSet, text: str, last: str) -> None:
    out = finalize_answer(text, cs)
    assert out.rstrip().split("\n")[-1] == last
    assert out.startswith(text.rstrip())


def test_format_answer_line() -> None:
    assert format_answer_line([], NUMBERED) == "**Answer: (unclear)**"
    assert format_answer_line([NUMBERED.choices[2]], NUMBERED) == "**Answer: 3 — 16**"
    assert format_answer_line(list(RADIO.choices[1:]), RADIO) == "**Answer: Python is interpreted; Python has no GC**"


def test_choice_set_lookups() -> None:
    assert NUMBERED.by_label("3") is NUMBERED.choices[2]
    assert LETTERED.by_label("(c)") is LETTERED.choices[2]
    assert LETTERED.by_label("c") is LETTERED.choices[2]
    assert LETTERED.by_label("") is None
    assert RADIO.by_label("") is None
    assert NUMBERED.labels == ("1", "2", "3", "4") and NUMBERED.texts == ("4", "8", "16", "32")
    assert RADIO.labeled is False and NUMBERED.labeled is True
