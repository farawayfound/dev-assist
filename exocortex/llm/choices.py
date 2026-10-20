"""Multiple-choice questions: find the option set, hand it to the model unambiguously, and make
sure the answer ends with the chosen option.

Three pure, deterministic pieces (no threads, no Qt, no model calls -- unit-tested like
``exocortex.plan``):

* :func:`find_choices` scans screen text (or a typed question) for a run of answer options:
  labelled ``A. / A) / (A)``, ``1. / 1) / (1)``, ``i. ii.``, circled ``①`` / ``Ⓐ``, or unlabelled
  radio / checkbox glyphs (and their OCR misreads). Several options OCR'd onto one row
  (``A) 4  B) 8  C) 16``) are split; a label OCR'd onto its own line is paired with the next line.
* :func:`annotate_choices` rewrites the detected option lines inside the screen text as
  ``Choice 1: 4`` so the token pattern ``1. 4`` -- which a small model happily reads as the decimal
  1.4 -- never reaches the prompt; :func:`choices_block` restates the options right before the ask.
* :func:`finalize_answer` post-processes the model's full answer: a trailing ``Answer:`` line is
  normalised to the option it names (``Answer: 1.4`` with options ``1. 4 / 2. 8`` is choice 1), a
  missing one is inferred from the body, and the result is appended as the last line, bold:
  ``**Answer: 3 — 16**`` (``**Answer: 16**`` when the options carry no labels, ``**Answer: (unclear)**``
  when nothing can be told).

Evidence rules (:func:`_accept`). Every run needs a *question* next to it, because agendas, slide
bullets, tables of contents and numbered instruction steps look exactly like option runs and are
what this app reads off a shared screen most of the time:

* a **question cue** -- a line with a question word (which / what / select / choose / …) or a
  "Question N" header within a few lines above, or directly below ("Which of these is …?") --
  licenses any style;
* lettered runs and radio / checkbox glyphs also accept a *statement* right above them (a
  sentence ending in ``.``, ``?`` or ``:``: "The smallest element is always at the root.") and a
  closed vocabulary (True / False, Yes / No);
* numbered runs whose options are all short (values, not steps) accept a bare ``?`` directly
  above ("2 XOR 6 = ?");
* a cue that is a site poll / survey (``questions.DEFAULT_IGNORE_PATTERNS``) or a ranking task
  ("Rank the following …") disqualifies the run, and so does a list introduced by a plain colon
  ("There are three reasons:") unless the question sits below it.

When several runs qualify (a scrolled page with two questions), the one whose cue line shares the
most words with the question being answered wins; without a question, the last one does.
"""
from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from exocortex.llm.questions import DEFAULT_IGNORE_PATTERNS

log = logging.getLogger(__name__)

#: Option label styles.
LETTER, NUMBER, ROMAN, NONE = "letter", "number", "roman", "none"

#: Fewest / most options a run may have.
MIN_OPTIONS = 2
MAX_OPTIONS = 10
#: An option text longer than this is prose, not an option (a cue relaxes it to the larger value).
_MAX_TEXT = 120
_MAX_TEXT_WITH_CUE = 240
#: How many non-blank lines above a run are searched for a question-word cue, per style (a code
#: block often sits between "What does this print?" and the lettered options).
_CUE_WINDOW = {LETTER: 30, NUMBER: 12, ROMAN: 12, NONE: 8}
#: How many non-blank lines below a run may hold the question ("Which of these is …?").
_CUE_BELOW = 2
#: Options this short (in words) are values ("4", "Merge sort"), not instruction steps ("Go to
#: settings"); only such runs may ride on a bare "?" directly above them.
_SHORT_OPTION_WORDS = 2

#: Label alphabets: Latin, and Cyrillic (А Б В Г … as used by Russian-language quizzes).
_ALPHABETS = ("ABCDEFGHIJ", "АБВГДЕЖЗИК")
_LETTER_CLASS = "".join(_ALPHABETS) + "".join(a.lower() for a in _ALPHABETS)
_ROMANS = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x")
_ROMAN_INDEX = {r: k for k, r in enumerate(_ROMANS)}

#: Circled / parenthesised unicode enumerators, mapped to plain labels.
_UNICODE_LABELS: dict[str, str] = {}
for _k, _ch in enumerate("①②③④⑤⑥⑦⑧⑨⑩"):
    _UNICODE_LABELS[_ch] = str(_k + 1)
for _k, _ch in enumerate("⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽"):
    _UNICODE_LABELS[_ch] = str(_k + 1)
for _k, _ch in enumerate("ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿ"):
    _UNICODE_LABELS[_ch] = _ALPHABETS[0][_k]
for _k, _ch in enumerate("ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙ"):
    _UNICODE_LABELS[_ch] = _ALPHABETS[0][_k]
for _k, _ch in enumerate("⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥"):
    _UNICODE_LABELS[_ch] = _ALPHABETS[0][_k]
_UNICODE_CLASS = "".join(re.escape(c) for c in _UNICODE_LABELS)

#: Radio / checkbox glyphs as OCR reports them …
_RADIO_GLYPHS = "○◯●◉◎⊙□☐☑☒■◻◼▢◇◆"
#: … and plain bullets (the default slide bullet), which only ever count next to a question word.
_BULLET_GLYPHS = "•·▪▫‣⁃◦"
_RADIO_RE = re.compile(
    rf"^(?:(?P<glyph>[{re.escape(_RADIO_GLYPHS)}]|\(\s*[•·xX]?\s*\)|\[\s*[xX✓✔]?\s*\]|[oO0](?=\s)))\s*(?P<text>\S.*)$"
)
_BULLET_RE = re.compile(rf"^(?P<glyph>[{re.escape(_BULLET_GLYPHS)}]|[-*–—])\s+(?P<text>\S.*)$")
#: A leading radio glyph in front of a labelled option ("○ A. 4"), dropped before matching.
_LEADING_GLYPH_RE = re.compile(rf"^(?:[{re.escape(_RADIO_GLYPHS + _BULLET_GLYPHS)}]|\(\s*\)|\[\s*\])\s*")

#: Labelled options. The label is followed by ".", ")" or ":" (or sits in parentheses: "(A)").
#: A single space may separate the label from its punctuation ("A ) 4", "1 . 4" -- OCR); the
#: option text may follow with or without a space ("A.4").
_LETTER_RE = re.compile(rf"^\(?(?P<label>[{_LETTER_CLASS}])\s?[.):](?!\S*\w\.)\s*(?P<text>\S.*)$")
_NUMBER_RE = re.compile(r"^\(?(?P<label>\d{1,2})\s?[.):]\s+(?P<text>\S.*)$")
#: "1.4" -- an OCR-dropped space (option 1, text "4"); only accepted as a run of >= 3.
_NUMBER_TIGHT_RE = re.compile(r"^(?P<label>\d{1,2})[.)](?P<text>[^\s.].*)$")
_ROMAN_RE = re.compile(r"^\(?(?P<label>[ivx]{1,4}|[IVX]{1,4})\s?[.):]\s+(?P<text>\S.*)$")
_UNICODE_RE = re.compile(rf"^(?P<label>[{_UNICODE_CLASS}])\s*(?P<text>\S.*)$")
#: A label alone on its line (OCR split the label off its text): "A", "(B)", "C.", "1.", "2)".
_BARE_LETTER_RE = re.compile(rf"^\(?(?P<label>[{_LETTER_CLASS}])[.):]?\)?$")
_BARE_NUMBER_RE = re.compile(r"^\(?(?P<label>\d{1,2})[.)]$")

#: Where an inline "A) 4  B) 8" row may be split: before a label (or glyph) preceded by spaces.
_INLINE_LABEL = (
    rf"(?:\(?[{_LETTER_CLASS}][.):]|\(?\d{{1,2}}[.):]|\(?(?:[ivx]{{1,4}}|[IVX]{{1,4}})[.):]|"
    rf"[{_UNICODE_CLASS}]|[{re.escape(_RADIO_GLYPHS)}]|\(\s*[•·xX]?\s*\)|\[\s*[xX✓✔]?\s*\])"
)
_INLINE_SPLIT_WIDE_RE = re.compile(rf"\s{{2,}}(?={_INLINE_LABEL}\s?\S)")
_INLINE_SPLIT_TIGHT_RE = re.compile(rf"\s+(?={_INLINE_LABEL}\s?\S)")

#: A line with a question word or a "Question N" header (strong cue); `_CUE_RE` also takes a
#: bare trailing "?" (weak cue: fine for lettered runs, not for numbered steps).
_CUE_WORDS = (
    r"\b(which|what|who|when|where|why|how|select|choose|pick|identify|determine|evaluate|"
    r"complete|fill in|true or false|following|correct|options?|choices|alternatives)\b|"
    r"^\s*(question|q)\.?\s*\d+\s*(?:[.):]|of\s+\d+|$)|\bquestion\s+\d+\s+of\s+\d+\b"
)
_CUE_WORD_RE = re.compile(_CUE_WORDS, re.IGNORECASE)
_CUE_RE = re.compile(r"\?\s*$|" + _CUE_WORDS, re.IGNORECASE)
#: A cue line that asks for an ordering / matching, not a pick: the run is not an option set.
_NOT_MCQ_CUE_RE = re.compile(
    r"\b(rank|ranking|arrange|reorder|drag|order the|in the correct order|in order of|match each|"
    r"put (?:these|the following|them) in)\b",
    re.IGNORECASE,
)
_IGNORE_RES = tuple(re.compile(p, re.IGNORECASE) for p in DEFAULT_IGNORE_PATTERNS)
#: Chrome / status lines that must not be swallowed as a wrapped option text.
_CHROME_RE = re.compile(
    r"^(submit|next|previous|prev|back|skip|save|clear|reset|check( answer)?|show answer|explanation|"
    r"question \d+|q\d+|\d+ of \d+|\d+/\d+|time (left|remaining).*|\d{1,2}:\d{2}(:\d{2})?|flag.*|"
    r"reply|quote|share|edit|follow|report|hint)\W*$",
    re.IGNORECASE,
)
#: Option vocabularies that are a question all by themselves.
_CLOSED_VOCABULARIES = (
    {"true", "false"}, {"yes", "no"}, {"agree", "disagree"}, {"correct", "incorrect"},
    {"true", "false", "cannot be determined"}, {"yes", "no", "maybe"},
)
_WORD_RE = re.compile(r"[a-zа-я0-9]{3,}", re.IGNORECASE)
_STOP_WORDS = {"the", "and", "for", "are", "was", "this", "that", "with", "what", "which", "from", "does", "how"}


@dataclass(frozen=True)
class Choice:
    """One answer option: its label ("A", "3", "iv"; "" when the options carry none) and text."""

    label: str
    text: str


@dataclass(frozen=True)
class ChoiceSet:
    """The option set of a multiple-choice question as found in a text."""

    choices: tuple[Choice, ...]
    style: str                 # LETTER | NUMBER | ROMAN | NONE
    span: tuple[int, int]      # first and last line index (inclusive) of the run in the source text
    lead: str = ""             # text that shared the run's first line, before the first option
    trail: str = ""            # text that shared the run's last line, after the last option
    cue: str = ""              # the question line that licensed the run ("" when a statement did)

    @property
    def labeled(self) -> bool:
        return self.style != NONE

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(c.label for c in self.choices)

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(c.text for c in self.choices)

    def by_label(self, label: str) -> Choice | None:
        key = _norm_label(label)
        if not key:
            return None
        for c in self.choices:
            if _norm_label(c.label) == key:
                return c
        return None


# ------------------------------------------------------------------------------ detection ----
@dataclass(frozen=True)
class _Item:
    """A classified line: which style it could be, its label/text, and its source line."""

    style: str
    label: str
    text: str
    line: int
    tight: bool = False        # matched the dropped-space "1.4" form
    bare: bool = False         # the label stood alone; the text is on the next line
    glyph: str = ""            # NONE style: "radio" (○ □ ☐ …) or "bullet" (• - *)


def _norm_label(label: str) -> str:
    return label.strip("() .").casefold()


def _label_index(style: str, label: str) -> int | None:
    """Position of `label` in its style's sequence (A=0, 1=0, i=0); None when it is no label."""
    key = _norm_label(label)
    if style == LETTER:
        if len(key) != 1:
            return None
        for alphabet in _ALPHABETS:
            low = alphabet.casefold()
            if key in low:
                return low.index(key)
        return None
    if style == NUMBER:
        return int(key) - 1 if key.isdigit() and 1 <= int(key) <= 99 else None
    if style == ROMAN:
        return _ROMAN_INDEX.get(key)
    return None


def _classify(line: str, idx: int) -> list[_Item]:
    """Every reading of `line` as an option line (a line like "i) foo" is both a letter and a roman)."""
    stripped = line.strip()
    if not stripped:
        return []
    m = _UNICODE_RE.match(stripped)
    if m:
        label = _UNICODE_LABELS[m.group("label")]
        return [_Item(NUMBER if label.isdigit() else LETTER, label, m.group("text").strip(), idx)]
    m = _RADIO_RE.match(stripped)
    if m:
        inner = m.group("text").strip()
        labelled = [it for it in _classify_labelled(inner, idx) if not it.bare]
        if labelled:
            return labelled                    # "○ A. 4": the label wins
        return [_Item(NONE, "", inner, idx, glyph="radio")]
    m = _BULLET_RE.match(stripped)
    if m:
        return [_Item(NONE, "", m.group("text").strip(), idx, glyph="bullet")]
    return _classify_labelled(_LEADING_GLYPH_RE.sub("", stripped), idx)


def _classify_labelled(stripped: str, idx: int) -> list[_Item]:
    out: list[_Item] = []
    m = _LETTER_RE.match(stripped)
    if m:
        out.append(_Item(LETTER, m.group("label"), m.group("text").strip(), idx))
    m = _ROMAN_RE.match(stripped)
    if m and _norm_label(m.group("label")) in _ROMAN_INDEX:
        out.append(_Item(ROMAN, m.group("label"), m.group("text").strip(), idx))
    m = _NUMBER_RE.match(stripped)
    if m:
        out.append(_Item(NUMBER, m.group("label"), m.group("text").strip(), idx))
    else:
        m = _NUMBER_TIGHT_RE.match(stripped)
        if m and 1 <= int(m.group("label")) <= 12:
            out.append(_Item(NUMBER, m.group("label"), m.group("text").strip(), idx, tight=True))
    if out:
        return out
    m = _BARE_LETTER_RE.match(stripped)
    if m:
        out.append(_Item(LETTER, m.group("label"), "", idx, bare=True))
    m = _BARE_NUMBER_RE.match(stripped)
    if m:
        out.append(_Item(NUMBER, m.group("label"), "", idx, bare=True))
    return out


def _sequential_pieces(pieces: list[str]) -> bool:
    """Whether at least two of `pieces` read as options of one style, in sequence. Dropped-space
    numbers ("2.8" in "1.4 * 2 = 2.8") never justify a split: they are decimals until a whole
    run of lines says otherwise."""
    readings = [[it for it in _classify(p, 0) if not it.bare and not it.tight] for p in pieces]
    for style in (LETTER, NUMBER, ROMAN, NONE):
        seq: list[int] = []
        for items in readings:
            for it in items:
                if it.style == style and (style == NONE or _label_index(style, it.label) is not None):
                    seq.append(_label_index(style, it.label) if style != NONE else len(seq))
                    break
        if len(seq) >= 2 and all(b == a + 1 for a, b in zip(seq, seq[1:])):
            return True
    return False


def _split_inline(line: str, typed: bool) -> list[str]:
    """Split a row holding several options ("A) 4  B) 8"); the row itself when it holds one.

    OCR rows join items with two spaces (`lines_to_text`); a typed question, or an OCR line that
    already merged a horizontal answer row, uses single spaces -- that split is tried second.
    """
    for pattern in ((_INLINE_SPLIT_TIGHT_RE,) if typed else (_INLINE_SPLIT_WIDE_RE, _INLINE_SPLIT_TIGHT_RE)):
        pieces = [p.strip() for p in pattern.split(line) if p.strip()]
        if len(pieces) >= 2 and _sequential_pieces(pieces):
            if pattern is _INLINE_SPLIT_WIDE_RE:
                # Chrome that shared the row with the last option ("C) 5  Submit") is its own piece.
                tail = re.split(r"\s{2,}", pieces[-1], maxsplit=1)
                pieces[-1:] = [p for p in tail if p.strip()]
            return pieces
    return [line]


def _nonblank_above(lines: Sequence[str], before: int, count: int) -> list[str]:
    out: list[str] = []
    for k in range(before - 1, -1, -1):
        text = lines[k].strip()
        if text:
            out.append(text)
            if len(out) >= count:
                break
    return out


def _nonblank_below(lines: Sequence[str], after: int, count: int) -> list[str]:
    out: list[str] = []
    for k in range(after + 1, len(lines)):
        text = lines[k].strip()
        if text:
            out.append(text)
            if len(out) >= count:
                break
    return out


@dataclass(frozen=True)
class _Cue:
    line: str | None           # the nearest question cue line (None when there is none)
    strong: bool               # it carries a question word / header (not just a trailing "?")
    below: bool                # it sits under the run ("Which of these …?")
    adjacent: bool             # it is the first real (non-chrome) line above the run
    intro_colon: bool          # the line right above the run ends in ":" without being a cue
    statement: bool            # the line right above the run is a sentence (ends in . ? : !)

    @property
    def found(self) -> bool:
        return self.line is not None


def _is_sentence(text: str) -> bool:
    return text.rstrip().endswith("?") or (text.rstrip()[-1:] in ".:!" and len(text.split()) >= 4)


def _cue(lines: Sequence[str], run: list[_Item], style: str) -> _Cue:
    first, last = run[0].line, run[-1].line
    above = _nonblank_above(lines, first, _CUE_WINDOW[style])
    real_above = [t for t in above if not _CHROME_RE.match(t)]
    nearest = real_above[0] if real_above else ""
    # A prose sentence ending in ":" introduces a list ("There are three reasons:"); a short
    # header ("Options:", "Answer choices:") introduces the options and is not held against them.
    intro_colon = nearest.endswith(":") and not _CUE_RE.search(nearest) and len(nearest.split()) >= 4
    statement = bool(nearest) and _is_sentence(nearest)
    # A question above the run outranks one below it (on a page with two questions the next
    # question's stem sits right under this run's options).
    for k, text in enumerate(real_above):
        if _CUE_WORD_RE.search(text):
            return _Cue(text, True, False, k == 0, intro_colon, statement)
    for text in _nonblank_below(lines, last, _CUE_BELOW):
        # Under the run a bare "?" is not enough ("Can someone confirm?" after repro steps): the
        # line must carry a question word ("Which of these is …?", "Which statement is correct?").
        if _CUE_WORD_RE.search(text):
            return _Cue(text, True, True, False, intro_colon, statement)
    for k, text in enumerate(real_above):
        if text.rstrip().endswith("?") and k < _CUE_WINDOW[NUMBER]:
            return _Cue(text, False, False, k == 0, intro_colon, statement)
    return _Cue(None, False, False, False, intro_colon, statement)


def _runs(items: list[_Item], style: str) -> list[list[_Item]]:
    """Maximal runs of `style` items with sequential labels (one gap tolerated), in text order.

    A number run also absorbs a lettered item whose position continues it ("1. 2. C. D." --
    OCR read "3." and "4." as "C." and "D."); the letter is relabelled with its number.
    """
    runs: list[list[_Item]] = []
    cur: list[_Item] = []
    expected: int | None = None
    for it in items:
        if style == NONE:
            if it.style != NONE:
                continue
            if cur and it.glyph != cur[-1].glyph:
                runs.append(cur)                 # "-" bullets and "○" glyphs are different runs
                cur = []
            cur.append(it)
            continue
        if it.style != style:
            if style == NUMBER and it.style == LETTER and len(cur) >= 2 and expected is not None \
                    and _label_index(LETTER, it.label) == expected and not it.bare:
                cur.append(replace(it, style=NUMBER, label=str(expected + 1)))
                expected += 1
            continue
        pos = _label_index(style, it.label)
        if pos is None:
            continue
        if cur and expected is not None and pos in (expected, expected + 1) and it.bare == cur[-1].bare:
            cur.append(it)
            expected = pos + 1
            continue
        if cur:
            runs.append(cur)
        cur = [it]
        expected = pos + 1
    if cur:
        runs.append(cur)
    return runs


def _gather_text(lines: Sequence[str], run: list[_Item], max_text: int) -> list[Choice] | None:
    """Option texts for `run`: a bare label takes the next line; wrapped continuation lines between
    two options are folded in, and after the last option only lines that start in lower case (page
    chrome is capitalised). None when a text is unusable or another option style sits in between."""
    out: list[Choice] = []
    for k, it in enumerate(run):
        text = it.text
        start = it.line + 1
        if it.bare:
            if start >= len(lines) or not lines[start].strip() or _classify(lines[start], start) \
                    or _CHROME_RE.match(lines[start].strip()):
                return None
            text = lines[start].strip()
            start += 1
        stop = run[k + 1].line if k + 1 < len(run) else len(lines)
        for j in range(start, stop):
            extra = lines[j].strip()
            if not extra:
                if k + 1 < len(run):
                    continue
                break
            if _classify(extra, j):
                if k + 1 < len(run):
                    return None            # another option style in between: not one run
                break
            if _CHROME_RE.match(extra) or (k + 1 == len(run) and not extra[0].islower()):
                if k + 1 < len(run):
                    continue
                break
            text = f"{text} {extra}"
        text = text.strip()
        if not text or len(text) > max_text:
            return None
        out.append(Choice(it.label if it.style != NONE else "", text))
    return out


def _closed_vocabulary(texts: Sequence[str]) -> bool:
    keys = {t.casefold().rstrip(".") for t in texts}
    return any(keys <= vocab and len(keys) >= 2 for vocab in _CLOSED_VOCABULARIES)


def _overlap(a: str, b: str) -> int:
    wa = {w.casefold() for w in _WORD_RE.findall(a)} - _STOP_WORDS
    wb = {w.casefold() for w in _WORD_RE.findall(b)} - _STOP_WORDS
    return len(wa & wb)


def find_choices(text: str, *, typed: bool = False, question: str = "") -> ChoiceSet | None:
    """The multiple-choice option set in `text`, or None.

    `typed` marks text the user typed or spoke (an ask-box question) rather than OCR screen
    text: options may then sit on one line separated by single spaces ("A) 4 B) 8 C) 16").
    `question` is the question being answered, if known: when several runs qualify (a scrolled
    page with two questions), the run whose cue line shares the most words with it wins;
    otherwise the last run does (the options sit under the question; worked examples and page
    chrome come before it).
    """
    if not text or not text.strip():
        return None
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines: list[str] = []
    origin: list[int] = []       # index of the raw line each (possibly split) line came from
    for i, raw in enumerate(raw_lines):
        for piece in _split_inline(raw, typed):
            lines.append(piece)
            origin.append(i)

    items: list[_Item] = []
    for idx, line in enumerate(lines):
        items.extend(_classify(line, idx))

    candidates: list[ChoiceSet] = []
    for style in (LETTER, NUMBER, ROMAN, NONE):
        for run in _runs(items, style):
            cs = _accept(lines, origin, run, style)
            if cs is not None and not any(c.span == cs.span for c in candidates):
                candidates.append(cs)
    if not candidates:
        return _yes_no(lines, origin)
    candidates.sort(key=lambda c: c.span[0])
    if len(candidates) > 1:
        log.debug("find_choices: %d candidate option runs at lines %s", len(candidates), [c.span for c in candidates])
        if question.strip():
            scored = [(_overlap(c.cue, question), k, c) for k, c in enumerate(candidates)]
            best_score, _k, best = max(scored, key=lambda t: (t[0], t[1]))
            if best_score > 0:
                return best
    return candidates[-1]


def _accept(lines: Sequence[str], origin: Sequence[int], run: list[_Item], style: str) -> ChoiceSet | None:
    """Turn a candidate run into a ChoiceSet when it passes the evidence rules (module docstring)."""
    if len(run) < MIN_OPTIONS or len(run) > MAX_OPTIONS:
        return None
    first = run[0]
    cue = _cue(lines, run, style)
    if cue.line is not None and (_NOT_MCQ_CUE_RE.search(cue.line) or any(p.search(cue.line) for p in _IGNORE_RES)):
        return None                     # a ranking task, or a site poll / survey
    max_text = _MAX_TEXT_WITH_CUE if cue.found else _MAX_TEXT
    choices = _gather_text(lines, run, max_text)
    if choices is None:
        return None
    texts = [c.text for c in choices]
    if len({t.casefold() for t in texts}) != len(texts):
        return None                     # duplicate option texts: not an option set (or a misread)
    closed = _closed_vocabulary(texts)
    short = all(len(t.split()) <= _SHORT_OPTION_WORDS for t in texts)
    bare = any(it.bare for it in run)
    radio = style == NONE and first.glyph == "radio"

    if style == LETTER or radio:
        licensed = cue.found or closed or cue.statement
    else:                               # numbers, romans, bullets, bare labels: steps and agendas look the same
        licensed = cue.strong or closed or (cue.found and cue.adjacent and short and not bare)
    if bare and not cue.strong:
        licensed = False
    if not licensed:
        return None
    if cue.intro_colon and not cue.below and style != LETTER and not closed:
        return None                     # "There are three reasons:" followed by a numbered list

    if style == NUMBER:
        tight = any(it.tight for it in run)
        if tight and (len(run) < 3 or not all(it.tight for it in run) or not cue.strong):
            return None
        if tight and all(re.fullmatch(r"\d", t) for t in texts):
            return None                 # "1.5 / 2.3 / 3.7" are decimals, not options 1..3
        if tight and any("=" in t or len(t.split()) > 4 for t in texts):
            return None                 # "1.4 * 2 = 2.8" is arithmetic, not option 1
        if _label_index(NUMBER, first.label) not in (0, 1):
            return None
        if bare and len(run) < 3:
            return None
    elif style == LETTER:
        idx0 = _label_index(LETTER, first.label)
        if idx0 not in (0, 1) or (len(run) == 2 and idx0 != 0):
            return None
        if bare and len(run) < 3:
            return None
        if not cue.found and any(len(t) > _MAX_TEXT for t in texts):
            return None
    elif style == ROMAN:
        if _label_index(ROMAN, first.label) != 0:
            return None

    last_line = run[-1].line + (1 if run[-1].bare else 0)
    last_line = min(last_line, len(lines) - 1)
    lead = _shared_line_text(lines, origin, first.line, before=True)
    trail = _shared_line_text(lines, origin, last_line, before=False)
    return ChoiceSet(
        tuple(choices), style, (origin[first.line], origin[last_line]),
        lead=lead, trail=trail, cue=cue.line or "",
    )


def _shared_line_text(lines: Sequence[str], origin: Sequence[int], idx: int, *, before: bool) -> str:
    """Pieces of the raw line that `lines[idx]` came from, before (or after) it -- the question
    stem that shared the row with the first option, or chrome after the last one."""
    raw = origin[idx]
    if before:
        pieces = [lines[k] for k in range(idx - 1, -1, -1) if origin[k] == raw][::-1]
    else:
        pieces = [lines[k] for k in range(idx + 1, len(lines)) if origin[k] == raw]
    return "  ".join(p.strip() for p in pieces if p.strip())


def _yes_no(lines: Sequence[str], origin: Sequence[int]) -> ChoiceSet | None:
    """Bare "Yes / No" or "True / False" lines right under a question."""
    for k in range(len(lines) - 1):
        a, b = lines[k].strip().rstrip("."), lines[k + 1].strip().rstrip(".")
        if a.casefold() != b.casefold() and _closed_vocabulary((a, b)):
            cues = [t for t in _nonblank_above(lines, k, 3) if _CUE_RE.search(t)]
            if cues:
                return ChoiceSet((Choice("", a), Choice("", b)), NONE, (origin[k], origin[k + 1]), cue=cues[0])
    return None


# ------------------------------------------------------------------------------ prompting ----
def annotate_choices(text: str, cs: ChoiceSet) -> str:
    """`text` (the screen text or typed question `cs` was found in) with the option run rewritten
    as one ``Choice <label>: <text>`` line per option. Unlabelled runs are left as they are:
    nothing in them reads as a decimal."""
    if not cs.labeled:
        return text
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    first, last = cs.span
    if not (0 <= first <= last < len(lines)):
        return text
    rewritten = [f"Choice {c.label}: {c.text}" for c in cs.choices]
    if cs.lead:
        rewritten.insert(0, cs.lead)
    if cs.trail:
        rewritten.append(cs.trail)
    return "\n".join(lines[:first] + rewritten + lines[last + 1:])


def answer_line_format(cs: ChoiceSet) -> str:
    """The final-line format the model is asked for."""
    if cs.labeled:
        return '"Answer: <label> — <option text>"'
    return '"Answer: <option text>"'


def choices_block(cs: ChoiceSet) -> str:
    """The volatile <choices> block for the end of the user turn (right before the ask)."""
    if cs.labeled:
        rows = "\n".join(f"{c.label}: {c.text}" for c in cs.choices)
        head = "This is a multiple-choice question. The options are listed as \"label: option text\":"
        tail = (
            "Decide which one option is most likely correct -- do not default to the first one just "
            "because it is listed first -- in at most two short sentences, then make the very last "
            f"line, with nothing after it, exactly {answer_line_format(cs)} (comma-separate labels only "
            "if the question explicitly allows more than one)."
        )
    else:
        rows = "\n".join(f"- {c.text}" for c in cs.choices)
        head = "This is a multiple-choice question. The options are:"
        tail = (
            "Decide which one option is most likely correct -- do not default to the first one just "
            "because it is listed first -- in at most two short sentences, then make the very last "
            f"line, with nothing after it, exactly {answer_line_format(cs)}, quoting the option text "
            "exactly as listed above."
        )
    return f"<choices>\n{head}\n{rows}\n{tail}\n</choices>\n\n"


# ------------------------------------------------------------------------------ the answer ---
_ANSWER_LINE_RE = re.compile(
    r"^\W*(?:my\s+|the\s+)?(?:final\s+)?(?:answer|correct\s+(?:answer|option|choice)|"
    r"(?:the\s+)?answer\s+is)\s*(?:is\s*)?[:\-–—]\s*(?P<payload>.+?)\s*$",
    re.IGNORECASE,
)
_UNCLEAR = "(unclear)"
_NEGATIVE_RE = re.compile(
    r"\b(not|isn'?t|aren'?t|wrong|incorrect|rule[sd]? out|eliminat\w*|never|neither|nor|false|cannot|can'?t|"
    r"doesn'?t|don'?t|fails?|except)\b",
    re.IGNORECASE,
)
#: A sentence that commits to an option. "pick" / "choose" / "select" are deliberately absent:
#: they are the question's words ("whichever you pick"), not a verdict.
_POSITIVE_RE = re.compile(
    r"\b(correct|right|answer|therefore|thus|hence|so|best|true|yes|must be|is the|"
    r"gives|yields|equals|matches|fits|valid|holds)\b",
    re.IGNORECASE,
)
_MULTI_RE = re.compile(r"\b(both|all|and|as well as|plus)\b|&", re.IGNORECASE)
_LABEL_WORDS = r"(?:option|choice|answer|letter|alternative|number)"
_LABEL_MENTION_RE = re.compile(
    rf"(?:\b{_LABEL_WORDS}\s+(?:is|would be|should be|must be|being)\s*\(?(?P<a>[A-Ja-j]|\d{{1,2}}|[ivx]{{1,4}})\)?(?![\w.])|"
    rf"\b{_LABEL_WORDS}\s*\(?(?P<b>[A-Ja-j]|\d{{1,2}}|[ivx]{{1,4}})\)?(?![\w.])|"
    r"\((?P<c>[A-Ja-j]|\d{1,2}|[ivx]{1,4})\)|"
    r"(?-i:(?<![\w.])(?P<d>[A-J])(?:[.)](?=\s|$)|(?=[,;:!?]?\s*$)|(?=\s+(?:is|are|was|would|should|seems|looks|fits|matches|gives|because|since|and|,|:|—|-))))|"
    r"^\s*(?P<e>[A-Ja-j]|\d{1,2})[.):]\s+)",
    re.IGNORECASE | re.MULTILINE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _strip_md(s: str) -> str:
    return s.replace("**", "").replace("__", "").replace("`", "").strip()


def _norm_text(s: str) -> str:
    return " ".join(_strip_md(s).split()).casefold().strip(" .,;:!\"'")


def _text_pattern(text: str) -> re.Pattern[str]:
    """A whole-word / whole-number pattern for an option text inside prose."""
    t = re.escape(" ".join(text.split()))
    if re.fullmatch(r"[-+]?\d[\d,]*(\.\d+)?%?", text.strip()):
        return re.compile(rf"(?<![\w.\-])(?<!\d)(?<!\d\.){t}(?!\w)(?!\.\d)", re.IGNORECASE)
    return re.compile(rf"(?<!\w){t}(?!\w)", re.IGNORECASE)


def _match_text(payload: str, cs: ChoiceSet) -> Choice | None:
    """The option whose text best matches `payload` (exact, then contained, then fuzzy)."""
    p = _norm_text(payload)
    if not p:
        return None
    for c in cs.choices:
        if _norm_text(c.text) == p:
            return c
    contained = [c for c in cs.choices if len(_norm_text(c.text)) >= 2 and _text_pattern(c.text).search(payload)]
    if len(contained) == 1:
        return contained[0]
    if len(contained) > 1:
        return max(contained, key=lambda c: len(c.text))
    covering = [c for c in cs.choices if len(p) >= 3 and p in _norm_text(c.text)]
    if len(covering) == 1:
        return covering[0]
    if len(p) >= 6:
        scored = [(difflib.SequenceMatcher(None, p, _norm_text(c.text)).ratio(), c) for c in cs.choices]
        ratio, c = max(scored, key=lambda pair: pair[0])
        if ratio >= 0.85:
            return c
    return None


def _resolve_part(part: str, cs: ChoiceSet, body: str) -> Choice | None:
    """One comma-separated part of an Answer payload -> the option it names."""
    s = _strip_md(part).strip().strip("()[]").strip()
    s = re.sub(rf"^{_LABEL_WORDS}\s+", "", s, flags=re.IGNORECASE)
    s = s.strip().rstrip(".:;!")
    if not s:
        return None
    if cs.labeled:
        # "B — 16" / "B) 16" / "B: 16" / "B. 16": label first, the rest is (hopefully) its text.
        m = re.match(
            rf"^\(?(?P<label>[{_LETTER_CLASS}]|\d{{1,2}}|[ivx]{{1,4}}|[IVX]{{1,4}})\)?(?:\s*[-–—:.)]\s*|\s+)(?P<rest>.+)$", s
        )
        if m and cs.by_label(m.group("label")) is not None:
            c = cs.by_label(m.group("label"))
            assert c is not None
            by_text = _match_text(m.group("rest"), cs)
            if by_text is None or by_text is c:
                return c
            return by_text             # label and text disagree ("2 — 16" when 16 is option 3): trust the text
        # "1.4" with options 1: 4 / 2: 8 -- the decimal misread of a label + text.
        m = re.fullmatch(r"(?P<label>\d{1,2})\.(?P<rest>\S.*)", s)
        if m and cs.style == NUMBER:
            c = cs.by_label(m.group("label"))
            if c is not None and _norm_text(c.text) == _norm_text(m.group("rest")) and _match_text(s, cs) is None:
                return c
        c = cs.by_label(s)
        if c is not None:
            by_text = _match_text(s, cs)
            if by_text is None or by_text is c:
                return c
            # A bare token that is both a label and an option text ("4" with options 1: 4 … 4: 32):
            # let the body break the tie, else read it as the value the model computed.
            inferred = _infer_from_body(body, cs)
            if len(inferred) == 1 and inferred[0] in (c, by_text):
                return inferred[0]
            return by_text
    return _match_text(s, cs)


def _split_payload(payload: str) -> list[str]:
    parts = re.split(r"\s*(?:,|;|/|&|\band\b|\bor\b|\bboth\b)\s*", payload, flags=re.IGNORECASE)
    return [p for p in (x.strip() for x in parts) if p]


def _resolve_payload(payload: str, cs: ChoiceSet, body: str) -> list[Choice]:
    whole = _resolve_part(payload, cs, body)
    if whole is not None and _norm_text(payload) == _norm_text(whole.text):
        return [whole]                    # the option text itself contains a comma / "and"
    found: list[Choice] = []
    for part in _split_payload(payload):
        c = _resolve_part(part, cs, body)
        if c is not None and c not in found:
            found.append(c)
    if found:
        return found
    return [whole] if whole is not None else []


def _mentions(sent: str, cs: ChoiceSet) -> list[tuple[int, Choice, bool]]:
    """(position, option, by_label) for every option named in `sent`, in text order."""
    out: list[tuple[int, Choice, bool]] = []
    if cs.labeled:
        for m in _LABEL_MENTION_RE.finditer(sent):
            label = m.group("a") or m.group("b") or m.group("c") or m.group("d") or m.group("e")
            c = cs.by_label(label) if label else None
            if c is not None:
                out.append((m.start(), c, True))
    for c in cs.choices:
        if _norm_text(c.text):
            pattern = _text_pattern(c.text)
            out.extend((m.start(), c, False) for m in pattern.finditer(sent))
    out.sort(key=lambda t: t[0])
    return out


def _infer_from_body(body: str, cs: ChoiceSet) -> list[Choice]:
    """The option(s) the prose commits to.

    Sentences are read in order; one that only rejects an option ("A is wrong") is skipped and
    that option remembered as rejected. The verdict is the last *committing* sentence: one with a
    positive cue ("so", "correct", "the answer is …"), or one that simply states an option or a
    label. Within it the last-named option wins, or all named labels when it says "both A and C".
    A body that never commits yields nothing (the caller writes "(unclear)").
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(body) if s and s.strip()]
    verdict: list[Choice] = []
    rejected: set[Choice] = set()
    for sent in sentences:
        mentioned = _mentions(sent, cs)
        if not mentioned:
            continue
        bare = _norm_text(sent)
        states_option = any(_norm_text(c.text) == bare or (cs.labeled and _norm_label(c.label) == bare) for c in cs.choices)
        negative = bool(_NEGATIVE_RE.search(sent)) and not states_option
        positive = bool(_POSITIVE_RE.search(sent)) or states_option
        if negative and not positive:
            if len({c for _p, c, _l in mentioned}) == 1:
                rejected.add(mentioned[0][1])
            continue
        if not positive:
            continue
        labels = [c for _p, c, by_label in mentioned if by_label]
        distinct_labels = list(dict.fromkeys(labels))
        if len(distinct_labels) >= 2 and _MULTI_RE.search(sent) and not negative:
            verdict = distinct_labels
        else:
            verdict = [mentioned[-1][1]]
    return [c for c in verdict if c not in rejected] or verdict


def format_answer_line(chosen: Sequence[Choice], cs: ChoiceSet) -> str:
    """The canonical bold last line for `chosen` (empty -> "(unclear)")."""
    if not chosen:
        return f"**Answer: {_UNCLEAR}**"
    if cs.labeled:
        labels = ", ".join(c.label for c in chosen)
        texts = "; ".join(c.text for c in chosen)
        return f"**Answer: {labels} — {texts}**"
    return "**Answer: " + "; ".join(c.text for c in chosen) + "**"


def finalize_answer(text: str, cs: ChoiceSet) -> str:
    """`text` with its multiple-choice verdict as the last line (see the module docstring)."""
    body_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    payloads: list[str] = []
    kept: list[str] = []
    for line in body_lines:
        m = _ANSWER_LINE_RE.match(line)
        if m and len(_strip_md(m.group("payload"))) <= 200:
            payloads.append(m.group("payload"))
        else:
            kept.append(line)
    body = "\n".join(kept).rstrip()
    chosen: list[Choice] = []
    for payload in reversed(payloads):
        chosen = _resolve_payload(payload, cs, body)
        if chosen:
            break
    if not chosen:
        chosen = _infer_from_body(body, cs)
    line = format_answer_line(chosen, cs)
    return f"{body}\n\n{line}" if body else line
