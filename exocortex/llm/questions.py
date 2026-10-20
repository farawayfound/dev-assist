"""Heuristics to detect questions in live transcript text and pull out the one to answer.

Deliberately simple and fast (no model call): this runs on the ASR worker thread for every SYSTEM
final segment, so it has to be cheap. `is_question` is a lightweight lexical check; `extract_question`
turns a handful of recent segment texts into the single question string handed to the LLM prompt
builders in `prompts.py`.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

#: Leading filler words stripped (one at a time, left to right) before checking for an
#: interrogative start, e.g. "so, what do you think" -> "what do you think".
_FILLERS = {"so", "okay", "ok", "um", "uh", "hmm", "well", "and", "but", "like", "oh"}

#: Single-word interrogative sentence-starters.
_INTERROGATIVE_STARTS = {
    "what", "why", "how", "when", "where", "who", "which",
    "can", "could", "would", "should", "will",
    "do", "does", "did",
    "is", "are", "was", "were",
    "have", "has", "had",
    "explain", "describe",
}

#: Multi-word interrogative sentence-starters (checked as a prefix, e.g. "tell me about ...",
#: "walk me through ...").
_INTERROGATIVE_PHRASES = ("tell me", "walk me")

#: Split on '.', '?', '!' while keeping the punctuation attached to the sentence it ends.
_SENTENCE_RE = re.compile(r"[^.?!]+[.?!]*")

#: extract_question() never returns more than this many characters.
_MAX_QUESTION_CHARS = 400


def _strip_fillers(text: str) -> str:
    """Remove leading filler words one at a time, returning what's left (may be empty).

    A token that normalises to an empty string (pure punctuation such as "--" or "...",
    common in live ASR output for false starts / filled pauses) is also skipped, so it
    never blocks the interrogative check on the real word that follows it."""
    words = text.split()
    i = 0
    while i < len(words):
        norm = _normalize_word(words[i])
        if norm in _FILLERS or norm == "":
            i += 1
            continue
        break
    return " ".join(words[i:])


def _normalize_word(word: str) -> str:
    """Lower-case a word, strip surrounding punctuation (including dash characters, so
    standalone "--"/"—" tokens normalise to ""), and undo simple contractions ("what's" ->
    "what", "doesn't" -> "does") so contracted interrogatives still match."""
    w = word.strip(",.!?;:\"'-–—").lower()
    if w.endswith("'s"):
        w = w[:-2]
    elif w.endswith("n't"):
        w = w[:-3]
    return w


def _starts_interrogative(text: str) -> bool:
    """Whether `text` opens with an interrogative word or phrase."""
    t = text.strip().lower()
    if not t:
        return False
    if t.startswith(_INTERROGATIVE_PHRASES):
        return True
    first = _normalize_word(t.split(" ", 1)[0])
    return first in _INTERROGATIVE_STARTS


def is_question(text: str, min_words: int = 3) -> bool:
    """True if `text` reads as a question worth answering.

    Rules (see SPEC.md llm/questions.py):
    - texts with fewer than `min_words` words are never questions (filters out bare backchannels
      and rhetorical tags like "right?" / "you know?");
    - otherwise, True if the text ends with "?", or if it starts (after stripping leading fillers
      such as "so", "okay", "um", "and") with an interrogative word/phrase.
    """
    t = (text or "").strip()
    if not t:
        return False
    if len(t.split()) < min_words:
        return False
    if t.endswith("?"):
        return True
    return _starts_interrogative(_strip_fillers(t))


def _split_sentences(text: str) -> list[str]:
    """Split `text` into trimmed sentences on '.', '?', '!' boundaries."""
    return [m.strip() for m in _SENTENCE_RE.findall(text) if m.strip()]


def extract_question(segments_text: list[str]) -> str:
    """Pull the question to answer out of the last few segment texts (oldest -> newest).

    Looks only at the newest (last) segment, split into sentences: finds the most recent sentence
    that reads as a question and prepends the immediately preceding sentence from the same segment
    for context (e.g. "Quick question. What's the timeline?"). Trailing, non-question sentences
    after the question are dropped. If no sentence in the newest segment is a question, the whole
    newest segment text is returned unchanged. The result is capped at ~400 characters.
    """
    if not segments_text:
        return ""
    newest = segments_text[-1].strip()
    sentences = _split_sentences(newest)
    if not sentences:
        return newest[:_MAX_QUESTION_CHARS]

    q_idx = None
    for i in range(len(sentences) - 1, -1, -1):
        if is_question(sentences[i], min_words=1):
            q_idx = i
            break

    if q_idx is None:
        log.debug("extract_question: no question sentence found, returning newest text as-is")
        return newest[:_MAX_QUESTION_CHARS]

    start = q_idx - 1 if q_idx > 0 else q_idx
    result = " ".join(sentences[start : q_idx + 1]).strip()
    return result[:_MAX_QUESTION_CHARS]


# ---------------------------------------------------------------- relevance (focus) -----
#: Default patterns for on-screen prompts that are never worth answering: site polls and surveys
#: (LeetCode's "Have you seen this in an interview before?"), feedback widgets, UI chrome.
#: Case-insensitive regexes, matched anywhere in the question. Extend via `[assist].ignore_patterns`.
DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    r"\bhave you (seen|encountered|solved|gotten|got|been asked) this\b",
    r"\bseen this (question|problem|one)?\s*(in|at|during) (a |an |the )?(real )?interview",
    r"\b(did|do) you (like|enjoy) (this|the|it|our)\b",
    r"\bdid you find (this|it)\b",
    r"\bwas this (helpful|useful|clear)\b",
    r"\b(rate|rating) (this|the|your)\b",
    r"\bhow (would|do|did) you rate\b",
    r"\bwould you recommend\b",
    r"\bhow (was|is) your (day|experience|week)\b",
    r"\bwhat did you think (of|about) (this|the) (problem|question|contest|experience)\b",
    r"\b(feedback|survey|poll)\b",
    r"\bare you (enjoying|satisfied|happy with)\b",
    r"\bhow (difficult|hard|easy) (was|is|did you find) this\b",
    r"\bshare your (thoughts|experience|feedback)\b",
    r"\b(upvote|downvote|subscribe|sign up|log in|sign in|premium|unlock|try now|learn more)\b",
    r"\bwant to (see|read|know) more\b",
    r"\b(cookies?|privacy) (policy|preferences|settings)\b",
)

#: Words/tokens that mark a question as technical (theory, concepts, design, math, code). A
#: personal-experience question that carries none of these is treated as opinion/trivia.
_TECH_CUES = (
    "algorithm", "complexity", "big-o", "o(n", "o(1", "o(log", "runtime", "run time", "time complexity",
    "space complexity", "array", "list", "tree", "graph", "hash", "map", "set", "heap", "stack", "queue",
    "linked", "node", "pointer", "index", "sort", "search", "binary", "recursion", "recursive", "iterative",
    "dynamic programming", "dp", "memo", "greedy", "backtrack", "bfs", "dfs", "dijkstra", "trie", "bit",
    "string", "substring", "palindrome", "matrix", "vector", "prime", "modulo", "mod ", "gcd", "probability",
    "expected value", "derivative", "integral", "equation", "theorem", "proof", "prove", "lemma", "formula",
    "function", "class", "method", "variable", "loop", "return", "compile", "syntax", "bug", "exception",
    "error", "test", "unit test", "python", "java", "c++", "c#", "javascript", "typescript", "golang", " go ",
    "rust", "sql", "regex", "api", "rest", "grpc", "http", "tcp", "udp", "dns", "protocol", "socket",
    "thread", "process", "lock", "mutex", "async", "concurren", "parallel", "memory", "cpu", "gpu", "cache",
    "latency", "throughput", "scale", "scalab", "shard", "replica", "consisten", "availability", "partition",
    "database", "index", "transaction", "acid", "cap theorem", "kafka", "queue", "load balanc", "cdn",
    "design", "architect", "system", "service", "microservice", "container", "kubernetes", "docker",
    "cloud", "aws", "azure", "gcp", "linux", "kernel", "compiler", "interpreter", "garbage", "jvm",
    "encrypt", "hash", "token", "auth", "oauth", "jwt", "security", "vulnerab", "model", "neural",
    "gradient", "tensor", "embedding", "transformer", "llm", "regression", "classif", "statistic",
    "variance", "mean", "median", "distribution", "bayes", "optimi", "implement", "optimal", "edge case",
    "input", "output", "constraint", "invariant", "approach", "solution", "solve", "trade-off", "tradeoff",
    "why does", "how does", "how would you", "explain", "difference between", "what is the", "define",
)
#: Questions that ask about the user's own feelings, history or preferences.
_PERSONAL_RE = re.compile(
    r"^(have|has|had|do|does|did|are|were|would|will|can|could) you\b|"
    r"\bhow (do|did|does) you feel\b|\bwhat do you think about your\b|\byour (favou?rite|experience|feelings?)\b",
    re.IGNORECASE,
)
_CODE_TOKEN_RE = re.compile(r"[A-Za-z_]\w*\(|\[\]|==|!=|->|=>|::|\b[a-z]+_[a-z]+\b|\b[a-z]+[A-Z]\w+\b|\bO\(")


def has_technical_cue(text: str) -> bool:
    """True when `text` mentions a technical concept, a code-ish token or a maths term."""
    low = " " + " ".join(text.split()).lower() + " "
    if any(cue in low for cue in _TECH_CUES):
        return True
    return bool(_CODE_TOKEN_RE.search(text))


#: The `[assist].focus` levels, least to most permissive.
FOCUS_LEVELS: tuple[str, ...] = ("technical", "auto", "all")

#: How a line that is not a question can still read as a TASK (a slide's "Reverse a linked
#: list", "Given an array..., return..."): an opening task verb, after an optional list marker /
#: "Q3:" / "Task 2." prefix.
_TASK_LINE_RE = re.compile(
    r"^(?:(?:\d+[.):]|[-*\u2022]|q\d*[.):]?|question\s*\d*[.):]?|task\s*\d*[.):]?|problem\s*\d*[.):]?)\s*)?"
    r"(?:please\s+)?"
    r"(?:implement|write|design|build|create|find|return|compute|count|print|reverse|merge|sort|"
    r"determine|calculate|check|convert|remove|insert|delete|validate|generate|parse|solve|"
    r"minimi[sz]e|maximi[sz]e|explain|describe|compare|define|list|name|give|show|prove|derive|"
    r"estimate|identify|choose|select|given|suppose|consider|assume|you are given|your task|"
    r"complete|fix|debug|optimi[sz]e|refactor|rewrite|translate|output|simulate|predict|evaluate|"
    r"analy[sz]e|summari[sz]e|discuss|justify|tell me|walk me|what|why|how|when|where|which|who)\b",
    re.IGNORECASE,
)


def focus_level(focus: str | None) -> str:
    """A `[assist].focus` value folded to one of `FOCUS_LEVELS`: "technical" and "all" as
    written, anything else (including "" and the old default) is "auto"."""
    value = str(focus or "").strip().lower()
    return value if value in ("technical", "all") else "auto"


def is_technical_focus(focus: str | None) -> bool:
    """Whether a `[assist].focus` value is the strict "technical" level."""
    return focus_level(focus) == "technical"


#: Function words that make a line read as a sentence rather than a list of labels (a menu, a
#: navigation bar, a tag cloud: "Education Jobs Tech Crypto Network").
_FUNCTION_WORDS = frozenset(
    "the a an of to in on at for with and or that this these those is are was were be been by from "
    "as it its into than then there their they we you your our he she his her not no if when how what "
    "why which who whose where".split()
)


def is_sentence_like(text: str) -> bool:
    """Whether a line has the shape of a sentence: at least 4 words, one of them a function word
    (`_FUNCTION_WORDS`). A navigation bar or a list of tags is not."""
    words = [w.strip(",.;:!?()[]\"'").lower() for w in (text or "").split()]
    words = [w for w in words if w]
    return len(words) >= 4 and any(w in _FUNCTION_WORDS for w in words)


def is_task_line(text: str) -> bool:
    """Whether one line reads as something to answer or do: a question (`is_question`, 2+
    words), an opening task verb ("Implement...", "Given an array...", "Q3: Explain...") or a
    sentence (`is_sentence_like`) carrying a technical cue (`has_technical_cue`) -- a bare list
    of technical-sounding labels (a site's navigation bar, "Education Jobs Tech Crypto Network")
    is not. What the "auto" focus wants of new text on screen."""
    norm = " ".join((text or "").split())
    if not norm:
        return False
    if is_question(norm, min_words=2) or _TASK_LINE_RE.match(norm):
        return True
    return is_sentence_like(norm) and has_technical_cue(norm)


def is_relevant(text: str, focus: str = "auto", ignore_patterns: list[str] | tuple[str, ...] | None = None) -> bool:
    """Decide whether an on-screen question (or, in hands-free mode, a new line of text) deserves
    a model call, per the `[assist].focus` level (`focus_level`):

    1. Anything matching an ignore pattern (defaults + `ignore_patterns`) is dropped at every
       level: polls, surveys, feedback and site chrome.
    2. "all": everything else passes -- read everything, answer as much as possible.
    3. "auto" (default, code-biased): a question about the user's own feelings/history ("have
       you", "do you like") is dropped unless it carries technical substance
       (`has_technical_cue`); everything else passes.
    4. "technical": additionally requires technical substance in the text itself.
    """
    norm = " ".join(text.split())
    if not norm:
        return False
    patterns = list(DEFAULT_IGNORE_PATTERNS) + list(ignore_patterns or ())
    for pat in patterns:
        try:
            if re.search(pat, norm, re.IGNORECASE):
                return False
        except re.error:
            log.warning("ignore pattern %r is not a valid regex; skipped", pat)
    level = focus_level(focus)
    if level == "all":
        return True
    if _PERSONAL_RE.search(norm) and not has_technical_cue(norm):
        return False
    if level == "technical" and not has_technical_cue(norm):
        return False
    return True

