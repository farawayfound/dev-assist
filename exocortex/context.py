"""Context library + dynamic context window (Qt-free, LLM-free, pure Python).

Added 2026-10-06 for the Models window's "Context" tab (see SPEC.md, "Context library + dynamic
context window"). Everything the Assistant needs to put *background material* and a *well-sized
transcript* into a prompt lives here, behind one object, :class:`ContextManager`.

Why this module exists
----------------------
1. **Reference documents.** The owner wants to upload a README or a design / architecture doc
   before a presentation and have the model answer audience questions from it. A document is
   *extracted* to plain text (:func:`extract_text`), kept in a small on-disk library
   (:class:`ContextStore`) and reaches the model in one of two modes: ``"inject"`` (its text sits
   in a stable ``<reference>`` block of every prompt) or ``"reference"`` (only a one-line
   catalogue entry is always present; the passages that match the question at hand are pulled
   into a volatile ``<lookup>`` block by a local BM25 index, :class:`ContextIndex`). Documents
   always rank *below* the live transcript -- the prompts say so; this module only decides what
   text goes where and how much of it.
2. **A window that does not thrash the KV cache.** ``Transcript.window_text`` front-trims one
   line at a time once the transcript outgrows its budget, so from then on the prompt prefix
   changes on every turn and llama.cpp re-evaluates thousands of tokens per answer.
   :class:`TranscriptWindow` is *anchored*: it keeps the same start line and only advances it in
   big steps (``[context].trim_step``), so the next many prompts extend the previous one
   byte-for-byte. What falls out is remembered (a bounded buffer) so a model-written running
   summary can replace it (``<earlier>``).
3. **A prompt that fits the real context.** The server was started with a fixed per-request
   context; the old prompt builder never looked at it. :meth:`ContextManager.build` sizes every
   block to it (token counts estimated from a calibrated characters-per-token ratio,
   :class:`TokenEstimator`) and trims in a fixed priority order: lookup passages, reference
   block, earlier summary, screen text, and only then the oldest transcript. The newest speech
   and the question are never trimmed.

Design rules kept throughout: stdlib only (``pypdf`` is imported lazily, for ``.pdf`` alone);
every public object is thread-safe (the Assistant's worker, the Qt thread and the health check
all reach it); every disk write is atomic (temp file + ``os.replace``); ``ContextManager.build``
never raises -- a bad input degrades to the smallest sane prompt and sets ``trimmed``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import html.parser
import json
import logging
import math
import os
import re
import threading
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from xml.etree import ElementTree as ET

from exocortex.bus import ContextUsageEvent
from exocortex.config import Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from exocortex.transcript import Transcript

log = logging.getLogger(__name__)

ContextMode = Literal["inject", "reference"]
CONTEXT_MODES = ("inject", "reference")
DOCUMENT_KINDS = ("markdown", "text", "code", "html", "docx", "pptx", "pdf")

MANIFEST_FILE = "manifest.json"
DOCS_DIR = "docs"
MANIFEST_VERSION = 1
_MAX_FILE_BYTES = 64 * 1024 * 1024          # refuse to even read a file bigger than this
_MAX_MEMBER_BYTES = 64 * 1024 * 1024        # ... or an office-document XML part this big
_MIN_TRANSCRIPT_CHARS = 240                 # the newest speech always gets at least this much room
_SCREEN_FLOOR = 0.25                        # the screen text is only ever cut down to this share
_CATALOGUE_SHARE = 0.4                      # of the reference budget, when injected documents share it
_CATALOGUE_LINE_CHARS = 400
#: Tokens held back in `available_chars` for a lane prompt's own template (the ChatML tags, the
#: block tags and the trailing instruction of a notes / compaction / digest prompt -- ~100 tokens
#: measured on the notes template) so the block it sizes never pushes the prompt over the lane.
_LANE_TEMPLATE_TOKENS = 128

_MARKDOWN_EXT = {".md", ".markdown"}
_TEXT_EXT = {".txt", ".rst", ".log", ".csv", ".json", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
_CODE_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".cs", ".sql", ".sh", ".ps1", ".bat", ".rb", ".php", ".kt", ".swift", ".m",
}
_HTML_EXT = {".html", ".htm"}


class ContextError(Exception):
    """A document could not be read, extracted or stored. The message is shown to the user."""


# --------------------------------------------------------------------------- documents
@dataclass
class ContextDocument:
    """One entry of the context library. ``path`` is the stored copy of the *extracted* text
    (always utf-8), never the original file: the original can move or change, the library must
    keep working. ``id`` is derived from the text, so adding the same text twice is one entry."""

    id: str
    name: str
    kind: str
    path: str
    source_path: str
    chars: int
    enabled: bool = True
    mode: str = "reference"
    digest: str = ""
    digest_state: str = "n/a"   # "pending" | "ready" | "failed" | "n/a" (inject mode, or digest off)
    added_at: str = ""
    sha256: str = ""

    def to_dict(self) -> dict[str, object]:
        """Plain JSON-able dict (the manifest entry; also what ``context_documents()`` returns)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "ContextDocument":
        """Tolerant loader: unknown keys are ignored with a warning (a manifest written by a
        newer build must not break an older one), missing optional keys take their defaults.
        Raises :class:`ContextError` when the entry has no usable ``id`` / ``name`` / ``path``."""
        if not isinstance(d, dict):
            raise ContextError("document entry is not an object")
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, object] = {}
        for key, value in d.items():
            if key not in known:
                log.warning("context: unknown document field %r ignored", key)
                continue
            kwargs[key] = value
        for required in ("id", "name", "path"):
            if not isinstance(kwargs.get(required), str) or not kwargs[required]:
                raise ContextError(f"document entry has no usable {required!r}")
        try:
            kwargs["chars"] = int(kwargs.get("chars", 0) or 0)  # type: ignore[call-overload]
            kwargs["enabled"] = bool(kwargs.get("enabled", True))
        except (TypeError, ValueError) as exc:
            raise ContextError(f"document entry is malformed: {exc}") from exc
        for text_key in ("kind", "source_path", "mode", "digest", "digest_state", "added_at", "sha256"):
            if text_key in kwargs and not isinstance(kwargs[text_key], str):
                kwargs[text_key] = str(kwargs[text_key])
        kwargs.setdefault("kind", "text")
        kwargs.setdefault("source_path", "")
        return cls(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- extraction
def _normalise(text: str) -> str:
    """Line endings to ``\\n``, trailing whitespace stripped per line, 3+ consecutive blank lines
    collapsed to 2, no leading / trailing blank lines. Idempotent."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n").replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip("\n")


def _cut(text: str, max_chars: int) -> tuple[str, bool]:
    """`text` cut to at most `max_chars` on a line boundary (a hard cut when the first line is
    itself longer). ``max_chars <= 0`` means no limit. -> (text, was_cut)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head = text[:max_chars]
    if text[max_chars] == "\n":
        return head.rstrip(), True
    idx = head.rfind("\n")
    if idx > 0:
        head = head[:idx]
    return head.rstrip(), True


def _decode(data: bytes, *, strict_binary_check: bool) -> str:
    """bytes -> str: utf-8 (BOM tolerated), utf-16 when it carries a BOM; undecodable bytes become
    U+FFFD. With `strict_binary_check` (an unknown extension) a NUL byte or 5 % or more
    replacement characters means "this is not text" -> :class:`ContextError`."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    text = data.decode("utf-8-sig", errors="replace")
    if strict_binary_check:
        bad = text.count("�")
        if b"\x00" in data[:4096] or (text and bad / len(text) >= 0.05):
            raise ContextError("this file does not look like text; supported types are "
                               "markdown, text, code, HTML, DOCX, PPTX and PDF")
    return text


def _read_bytes(path: Path) -> bytes:
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            raise ContextError(f"{path.name} is larger than {_MAX_FILE_BYTES // (1024 * 1024)} MB")
        return path.read_bytes()
    except OSError as exc:
        raise ContextError(f"could not read {path.name}: {exc}") from exc


class _HTMLText(html.parser.HTMLParser):
    """HTML -> readable text: scripts / styles dropped, headings become markdown headings (so
    chunking can split on them), list items become "- " lines, block tags become line breaks."""

    _SKIP = {"script", "style", "noscript", "template", "svg"}
    _BLOCK = {"p", "div", "section", "article", "header", "footer", "main", "nav", "aside", "ul", "ol",
              "table", "tr", "blockquote", "pre", "form", "figure", "figcaption", "dl", "dt", "dd", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._pre_depth = 0

    @staticmethod
    def _is_heading(tag: str) -> bool:
        return len(tag) == 2 and tag[0] == "h" and tag[1] in "123456"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self.parts.append("\n# ")
        elif self._is_heading(tag):
            self.parts.append("\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag in self._BLOCK:
            self.parts.append("\n")
            if tag == "pre":
                self._pre_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "pre":
            self._pre_depth = max(0, self._pre_depth - 1)
            self.parts.append("\n")
        elif tag in self._BLOCK or tag == "title" or self._is_heading(tag):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self.parts.append(data if self._pre_depth else re.sub(r"\s+", " ", data))


#: ``HTMLParser.close()`` rescans an unterminated tag from every later position (quadratic: a 32 KiB
#: run of ``<a `` takes half a minute), while ``feed()`` is linear. A tail this long is one tag or
#: comment that never ended, which HTML5 drops at the end of the input too, so ``close()`` is skipped.
_MAX_OPEN_TAIL = 4096


def _html_to_text(raw: str) -> str:
    parser = _HTMLText()
    try:
        parser.feed(raw)
        if len(parser.rawdata) <= _MAX_OPEN_TAIL:
            parser.close()
    except Exception as exc:  # noqa: BLE001 - html.parser is lenient; a crash must still surface as ContextError
        raise ContextError(f"could not parse the HTML: {exc}") from exc
    text = "".join(parser.parts)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    lines = [re.sub(r"^\|\s*", "", ln) for ln in lines]  # a table row starts with its first cell's " | "
    return "\n".join(lines)


def html_to_text(raw: str) -> str:
    """HTML -> readable markdown-ish text (headings, list items, table cells kept; scripts and
    styles dropped). The public name of :func:`_html_to_text` (2026-10-07, capabilities:
    ``exocortex.webfetch`` turns a fetched page into text with it). Raises
    :class:`ContextError` when the parser fails."""
    return _html_to_text(raw)


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def _zip_member(zf: zipfile.ZipFile, name: str) -> bytes:
    info = zf.getinfo(name)
    if info.file_size > _MAX_MEMBER_BYTES:
        raise ContextError(f"{name} inside the archive is too large")
    return zf.read(name)


def _docx_paragraph(p: ET.Element) -> str:
    parts: list[str] = []
    for el in p.iter():
        if el.tag == _W + "t":
            parts.append(el.text or "")
        elif el.tag == _W + "tab":
            parts.append("\t")
        elif el.tag in (_W + "br", _W + "cr"):
            parts.append("\n")
    text = "".join(parts).strip()
    if not text:
        return ""
    ppr = p.find(_W + "pPr")
    if ppr is not None:
        style = ppr.find(_W + "pStyle")
        val = style.get(_W + "val", "") if style is not None else ""
        m = re.fullmatch(r"(?i)heading\s*([1-9])", val or "")
        if m:
            return "#" * int(m.group(1)) + " " + text
        if (val or "").lower() == "title":
            return "# " + text
        if ppr.find(_W + "numPr") is not None:
            return "- " + text
    return text


def _docx_block(el: ET.Element, out: list[str]) -> None:
    if el.tag == _W + "p":
        line = _docx_paragraph(el)
        if line:
            out.append(line)
    elif el.tag == _W + "tbl":
        for tr in el.findall(_W + "tr"):
            cells = []
            for tc in tr.findall(_W + "tc"):
                cell = " ".join(t for t in (_docx_paragraph(p) for p in tc.iter(_W + "p")) if t)
                cells.append(cell.strip())
            if any(cells):
                out.append(" | ".join(cells))
    elif el.tag in (_W + "sdt", _W + "sdtContent"):
        for child in el:
            _docx_block(child, out)


def _docx_to_text(path: Path) -> str:
    # Any failure inside the zip / XML machinery (a corrupt deflate stream raises zlib.error, an
    # unsupported compression method NotImplementedError, an encrypted member RuntimeError,
    # absurd nesting RecursionError...) is "not a readable .docx", never a crash (review 2026-10-06).
    try:
        with zipfile.ZipFile(path) as zf:
            root = ET.fromstring(_zip_member(zf, "word/document.xml"))
        body = root.find(_W + "body")
        out: list[str] = []
        for child in (body if body is not None else []):
            _docx_block(child, out)
    except Exception as exc:  # noqa: BLE001 - every parser failure becomes a ContextError
        raise ContextError(f"{path.name} is not a readable .docx file: {exc}") from exc
    return "\n\n".join(out)


def _pptx_to_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            slides = []
            for name in zf.namelist():
                m = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
                if m:
                    slides.append((int(m.group(1)), name))
            slides.sort()
            out: list[str] = []
            for pos, (_num, name) in enumerate(slides, start=1):
                root = ET.fromstring(_zip_member(zf, name))
                lines = []
                for p in root.iter(_A + "p"):
                    text = "".join(
                        (el.text or "") if el.tag == _A + "t" else " "
                        for el in p.iter() if el.tag in (_A + "t", _A + "br")
                    ).strip()
                    if text:
                        lines.append(text)
                out.append(f"## Slide {pos}\n" + "\n".join(lines))
    except Exception as exc:  # noqa: BLE001 - every parser failure becomes a ContextError (see _docx_to_text)
        raise ContextError(f"{path.name} is not a readable .pptx file: {exc}") from exc
    return "\n\n".join(out)


def _pdf_to_text(path: Path, max_chars: int) -> str:
    try:
        import pypdf  # imported lazily: only .pdf needs it
    except ImportError as exc:
        raise ContextError("reading PDFs needs the pypdf package: pip install pypdf") from exc
    try:
        reader = pypdf.PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001 - pypdf raises several types for a bad password
                raise ContextError(f"{path.name} is password protected") from exc
        out: list[str] = []
        total = 0
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            out.append(f"## Page {number}\n{text}")
            total += len(out[-1])
            if max_chars > 0 and total > max_chars * 2:
                break  # far past the cut; no need to parse the rest of a huge PDF
    except ContextError:
        raise
    except Exception as exc:  # noqa: BLE001 - pypdf raises its own zoo of errors on malformed files
        raise ContextError(f"could not read {path.name}: {exc}") from exc
    return "\n\n".join(out)


def _fence(text: str, lang: str) -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{lang}\n{text}\n{fence}"


def extract_text(path: Path, *, max_chars: int) -> tuple[str, str]:
    """``(text, kind)`` for a file. Plain text is normalised (``\\n`` line endings, trailing
    whitespace stripped, 3+ blank lines collapsed to 2) and cut at `max_chars` on a line boundary
    (a warning is logged); code is stored fenced as ```` ```<ext>\\n...\\n``` ````. Office
    documents are read with zipfile + xml.etree (no extra dependency), PDFs with pypdf (a clear
    :class:`ContextError` naming ``pip install pypdf`` when it is missing). Unknown extensions
    are accepted when they decode as text (< 5 % replacement characters), else ContextError.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in _MARKDOWN_EXT:
        kind, text = "markdown", _decode(_read_bytes(path), strict_binary_check=False)
    elif ext in _TEXT_EXT:
        kind, text = "text", _decode(_read_bytes(path), strict_binary_check=False)
    elif ext in _CODE_EXT:
        kind, text = "code", _decode(_read_bytes(path), strict_binary_check=False)
    elif ext in _HTML_EXT:
        kind, text = "html", _html_to_text(_decode(_read_bytes(path), strict_binary_check=False))
    elif ext == ".docx":
        kind, text = "docx", _docx_to_text(path)
    elif ext == ".pptx":
        kind, text = "pptx", _pptx_to_text(path)
    elif ext == ".pdf":
        kind, text = "pdf", _pdf_to_text(path, max_chars)
    else:
        kind, text = "text", _decode(_read_bytes(path), strict_binary_check=True)
    text = _normalise(text)
    text, was_cut = _cut(text, max_chars)
    if was_cut:
        log.warning("context: %s is longer than %d characters; the rest was cut", path.name, max_chars)
    if kind == "code" and text:
        text = _fence(text, ext.lstrip("."))
    return text, kind


# --------------------------------------------------------------------------- chunking
@dataclass(frozen=True)
class Chunk:
    """One retrievable passage of a document."""
    doc_id: str
    doc_name: str
    index: int
    title: str
    text: str


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _pack(pieces: list[str], sep: str, max_chars: int) -> list[str]:
    """Greedy-pack `pieces` joined by `sep` into strings of at most `max_chars`."""
    out: list[str] = []
    cur = ""
    for piece in pieces:
        if not cur:
            cur = piece
        elif len(cur) + len(sep) + len(piece) <= max_chars:
            cur = cur + sep + piece
        else:
            out.append(cur)
            cur = piece
    if cur:
        out.append(cur)
    return out


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Split one over-long paragraph on line boundaries, then sentence boundaries, then hard."""
    pieces: list[str] = []
    for line in paragraph.split("\n"):
        if len(line) <= max_chars:
            pieces.append(line)
            continue
        for sentence in _SENTENCE_RE.split(line):
            if len(sentence) <= max_chars:
                pieces.append(sentence)
            else:
                pieces.extend(sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars))
    # lines and sentences share one packing pass; joining with "\n" is harmless for sentences
    return [p for p in _pack(pieces, "\n", max_chars) if p.strip()]


def chunk_text(text: str, *, max_chars: int, doc_id: str, doc_name: str) -> list[Chunk]:
    """Split on markdown headings and blank-line paragraphs, packing consecutive paragraphs of the
    same heading section up to `max_chars`; a single paragraph longer than that is split on line,
    then sentence boundaries. A chunk never spans two headings. ``title`` is the heading path
    ("Architecture > Threads") or the document name before the first heading. Fenced code blocks
    are never split on blank lines or mistaken for headings. Never returns an empty chunk; a
    blank text -> ``[]``. The heading line itself is not repeated in the chunk text (it is the
    title); a document that is nothing but headings yields one chunk made of them."""
    if not text or not text.strip():
        return []
    max_chars = max(20, int(max_chars))
    sections: list[tuple[str, list[str]]] = []   # (title, paragraphs)
    stack: list[tuple[int, str]] = []
    all_headings: list[str] = []                 # every heading in document order (headings-only docs)
    title = doc_name
    paragraphs: list[str] = []
    current: list[str] = []
    in_fence = False

    def flush_paragraph() -> None:
        if current:
            para = "\n".join(current).strip("\n")
            if para.strip():
                paragraphs.append(para)
            current.clear()

    def flush_section() -> None:
        flush_paragraph()
        if paragraphs:
            sections.append((title, list(paragraphs)))
            paragraphs.clear()

    for line in text.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            current.append(line)
            continue
        if in_fence:
            current.append(line)
            continue
        m = _HEADING_RE.match(line)
        if m and m.group(2):
            flush_section()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
            all_headings.append(m.group(2))
            title = " > ".join(t for _lvl, t in stack)
            continue
        if not line.strip():
            flush_paragraph()
        else:
            current.append(line)
    flush_section()

    chunks: list[Chunk] = []

    def emit(chunk_title: str, body: str) -> None:
        body = body.strip()
        if body:
            chunks.append(Chunk(doc_id, doc_name, len(chunks), chunk_title, body))

    for sec_title, paras in sections:
        buf: list[str] = []
        for para in paras:
            if len(para) > max_chars:
                for packed in _pack(buf, "\n\n", max_chars):
                    emit(sec_title, packed)
                buf = []
                for piece in _split_long(para, max_chars):
                    emit(sec_title, piece)
            else:
                buf.append(para)
        for packed in _pack(buf, "\n\n", max_chars):
            emit(sec_title, packed)

    if not chunks:  # headings only: one chunk of every heading, in order, so each stays retrievable
        emit(doc_name, "\n".join(all_headings) or text.strip())
    return chunks


# --------------------------------------------------------------------------- store
def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write(path: Path, text: str) -> None:
    """Temp file next to `path`, then ``os.replace`` (all-or-nothing, like every other write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class ContextStore:
    """The on-disk library of reference documents. Thread-safe (one RLock).

    Layout: ``<dir>/manifest.json`` (``{"version": 1, "documents": [...]}``, atomic writes) and
    ``<dir>/docs/<id>.md|.txt`` (the extracted text of each document). The directory is created on
    the first write, never at construction; a corrupt manifest is set aside as
    ``manifest.json.corrupt`` and the store starts empty (a bad library file must never stop the
    app). ``ContextStore(None)`` keeps everything in memory (tests, headless runs).

    ``version`` increases on every change; the index rebuilds lazily when it moved.
    """

    def __init__(self, directory: Path | None) -> None:
        self._dir = Path(directory) if directory is not None else None
        self._lock = threading.RLock()
        self._docs: dict[str, ContextDocument] = {}
        self._texts: dict[str, str] = {}      # id -> extracted text (the whole cache for an in-memory store)
        self._version = 0
        self._load()

    # ---- persistence
    def _manifest_path(self) -> Path | None:
        return self._dir / MANIFEST_FILE if self._dir is not None else None

    def _load(self) -> None:
        path = self._manifest_path()
        if path is None or not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("documents", []), list):
                raise ValueError("manifest is not an object with a 'documents' list")
        except (OSError, ValueError):
            log.warning("context: could not read %s; starting with an empty library", path, exc_info=True)
            try:
                os.replace(path, path.with_name(path.name + ".corrupt"))
            except OSError:
                pass
            return
        if data.get("version") != MANIFEST_VERSION:
            log.warning("context: manifest version %r (expected %d); loading it anyway",
                        data.get("version"), MANIFEST_VERSION)
        for raw in data.get("documents", []):
            try:
                doc = ContextDocument.from_dict(raw)
            except ContextError as exc:
                log.warning("context: skipping a manifest entry: %s", exc)
                continue
            self._docs[doc.id] = doc

    def _save(self) -> None:
        """Write the manifest. A failure is logged, not raised: the in-memory library stays
        usable for the session, only a restart would lose the change."""
        path = self._manifest_path()
        if path is None:
            return
        payload = {"version": MANIFEST_VERSION, "documents": [d.to_dict() for d in self._docs.values()]}
        try:
            _atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False))
        except OSError:
            log.warning("context: could not write %s; changes will not survive a restart", path, exc_info=True)

    def _bump(self) -> None:
        self._version += 1

    def reload(self) -> None:
        """Re-read the manifest from disk (drops the text cache of an on-disk store)."""
        with self._lock:
            if self._dir is None:
                return
            self._docs.clear()
            self._texts.clear()
            self._load()
            self._bump()

    # ---- reads
    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def documents(self) -> list[ContextDocument]:
        """Copies of every document, in the order they were added."""
        with self._lock:
            return [dataclasses.replace(d) for d in self._docs.values()]

    def get(self, doc_id: str) -> ContextDocument | None:
        with self._lock:
            doc = self._docs.get(doc_id)
            return dataclasses.replace(doc) if doc is not None else None

    def enabled(self, mode: str | None = None) -> list[ContextDocument]:
        """Enabled documents in added order, optionally only those in `mode`."""
        with self._lock:
            return [dataclasses.replace(d) for d in self._docs.values()
                    if d.enabled and (mode is None or d.mode == mode)]

    def pending_digests(self) -> list[str]:
        """Ids whose digest still has to be written (enabled or not), in added order."""
        with self._lock:
            return [d.id for d in self._docs.values() if d.digest_state == "pending"]

    def text(self, doc_id: str) -> str:
        """The stored extracted text of a document ("" when unknown or the copy is missing)."""
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                return ""
            cached = self._texts.get(doc_id)
            if cached is not None:
                return cached
            if self._dir is None:
                return ""
            stored = Path(doc.path)
            if not stored.is_file():  # the library folder was moved: resolve by file name
                stored = self._dir / DOCS_DIR / stored.name
            try:
                text = stored.read_text(encoding="utf-8")
            except OSError:
                log.warning("context: stored copy of %r is missing (%s)", doc.name, stored)
                return ""
            self._texts[doc_id] = text
            return text

    # ---- adding
    def add_file(self, path: Path, *, mode: str, max_chars: int, digest_wanted: bool) -> ContextDocument:
        """Extract `path` (see :func:`extract_text`) and add it. Raises :class:`ContextError`."""
        path = Path(path)
        if not path.is_file():
            raise ContextError(f"{path} is not a file")
        text, kind = extract_text(path, max_chars=max_chars)
        return self._add(path.name, text, kind, mode, str(path), digest_wanted)

    def add_text(self, name: str, text: str, *, mode: str, max_chars: int, digest_wanted: bool,
                 kind: str = "markdown") -> ContextDocument:
        """Add pasted text. Adding the same text again (same id) replaces the entry but keeps its
        enabled flag, mode and digest. Raises :class:`ContextError` for empty text / bad mode."""
        text = _normalise(text or "")
        text, was_cut = _cut(text, max_chars)
        if was_cut:
            log.warning("context: %s is longer than %d characters; the rest was cut", name, max_chars)
        return self._add(name, text, kind, mode, "", digest_wanted)

    def _add(self, name: str, text: str, kind: str, mode: str, source_path: str,
             digest_wanted: bool) -> ContextDocument:
        if mode not in CONTEXT_MODES:
            raise ContextError(f"unknown mode {mode!r} (expected 'inject' or 'reference')")
        if kind not in DOCUMENT_KINDS:
            raise ContextError(f"unknown document kind {kind!r}")
        if not text.strip():
            raise ContextError(f"{name or 'the document'} has no text to add")
        name = (name or "").strip() or "Untitled"
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc_id = sha[:12]
        with self._lock:
            if self._dir is not None:
                ext = ".txt" if kind == "text" else ".md"
                target = self._dir / DOCS_DIR / f"{doc_id}{ext}"
                try:
                    _atomic_write(target, text)
                except OSError as exc:
                    raise ContextError(f"could not store {name}: {exc}") from exc
                stored = str(target)
            else:
                stored = f"memory://{doc_id}"
            previous = self._docs.get(doc_id)
            if previous is not None and self._dir is not None and previous.path and previous.path != stored:
                # The same text re-added as another kind stores under another extension (.md vs
                # .txt): drop the old copy, or remove() would later leave it behind (review, 2026-10-06).
                try:
                    Path(previous.path).unlink(missing_ok=True)
                except OSError:
                    log.warning("context: could not delete the superseded copy %s", previous.path, exc_info=True)
            doc = ContextDocument(
                id=doc_id, name=name, kind=kind, path=stored, source_path=source_path, chars=len(text),
                enabled=True, mode=mode, digest="", digest_state="n/a", added_at=_now_iso(), sha256=sha,
            )
            if previous is not None:
                doc.enabled, doc.mode = previous.enabled, previous.mode
                doc.digest, doc.digest_state, doc.added_at = previous.digest, previous.digest_state, previous.added_at
            if doc.mode == "reference" and digest_wanted and doc.digest_state == "n/a":
                doc.digest_state = "pending"
            self._docs[doc_id] = doc
            self._texts[doc_id] = text
            self._bump()
            self._save()
            return dataclasses.replace(doc)

    # ---- changing
    def remove(self, doc_id: str) -> bool:
        with self._lock:
            doc = self._docs.pop(doc_id, None)
            if doc is None:
                return False
            self._texts.pop(doc_id, None)
            if self._dir is not None:
                for candidate in (Path(doc.path), self._dir / DOCS_DIR / Path(doc.path).name):
                    try:
                        candidate.unlink(missing_ok=True)
                    except OSError:
                        log.warning("context: could not delete %s", candidate, exc_info=True)
            self._bump()
            self._save()
            return True

    def set_enabled(self, doc_id: str, enabled: bool) -> bool:
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                return False
            if doc.enabled != bool(enabled):
                doc.enabled = bool(enabled)
                self._bump()
                self._save()
            return True

    def set_mode(self, doc_id: str, mode: str, *, digest_wanted: bool) -> bool:
        """Switch a document between inject and reference. To "inject" its digest state becomes
        "n/a" (the digest text is kept in case it goes back); to "reference" a kept digest is
        "ready" again, otherwise "pending" when `digest_wanted` (else "n/a")."""
        if mode not in CONTEXT_MODES:
            log.warning("context: ignoring unknown mode %r", mode)
            return False
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                return False
            if doc.mode == mode and not (mode == "reference" and doc.digest_state == "n/a" and digest_wanted):
                return True
            doc.mode = mode
            if mode == "inject":
                doc.digest_state = "n/a"
            elif doc.digest:
                doc.digest_state = "ready"
            else:
                doc.digest_state = "pending" if digest_wanted else "n/a"
            self._bump()
            self._save()
            return True

    def set_digest(self, doc_id: str, digest: str, state: str) -> bool:
        """Store a finished digest ("ready") or record that writing it failed ("failed")."""
        if state not in ("ready", "failed"):
            return False
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                return False
            text = (digest or "").strip()
            if state == "failed" and not text and doc.digest:
                # A re-digest that failed (model down, empty answer) keeps the digest it already
                # had, and that digest stays "ready" for the catalogue; the ContextLibraryEvent the
                # caller publishes still says "digest failed" (review, 2026-10-06).
                doc.digest_state = "ready"
            else:
                doc.digest = text
                doc.digest_state = state
            self._bump()
            self._save()
            return True

    def mark_digest_pending(self, doc_id: str) -> bool:
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                return False
            doc.digest_state = "pending"
            self._bump()
            self._save()
            return True


# --------------------------------------------------------------------------- index
_STOPWORDS = frozenset(
    "a an the and or but if of to in on at by for with from as is are was were be been being it its this "
    "that these those do does did not no so than then there their they them he she we you your our i me my "
    "can could should would will shall may might must have has had what which who whom how when where why "
    "about into over under also just very more most some any each other such".split()
)
_TOKEN_RE = re.compile(r"[^\W_]+")


def _stem(token: str) -> str:
    """Crude suffix stemmer: trailing "ing" / "ed" / "es" / "s" when the stem keeps >= 4 chars.
    "es" is only dropped after s/x/z/ch/sh ("classes" -> "class"), otherwise "s" alone is
    ("services" -> "service"), so the singular and the plural meet. Deterministic, not idempotent."""
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    if token.endswith("es") and len(token) - 2 >= 4 and token[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) - 1 >= 4:
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Casefolded alphanumeric tokens of >= 2 chars, stop words dropped, crudely stemmed."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.casefold()):
        if len(tok) < 2 or tok in _STOPWORDS:
            continue
        out.append(_stem(tok))
    return out


_LINE_PREFIX_RE = re.compile(r"^\[[^\]]*\]\s*[^:]{0,60}:\s*")


def query_terms(question: str, recent_transcript: str = "", screen_text: str = "") -> dict[str, float]:
    """Retrieval query: the question's terms weigh 1.0, the most recent transcript lines' 0.5, the
    screen text's 0.25 (the maximum when a term occurs in several). Pure and deterministic. The
    "[mm:ss] Label:" prefix of transcript lines is ignored (a timestamp is not a topic)."""
    terms: dict[str, float] = {}
    recent = "\n".join(_LINE_PREFIX_RE.sub("", ln) for ln in (recent_transcript or "").split("\n"))
    for text, weight in ((question or "", 1.0), (recent, 0.5), (screen_text or "", 0.25)):
        for term in tokenize(text):
            if terms.get(term, 0.0) < weight:
                terms[term] = weight
    return terms


class ContextIndex:
    """BM25 (k1 = 1.5, b = 0.75) over the passages of the ENABLED reference-mode documents.

    The tokenizer is :func:`tokenize`. The index is rebuilt lazily whenever the store's version or
    `chunk_chars` changed, so enabling / disabling / adding a document takes effect on the next
    search without any bookkeeping by the caller. Each passage is indexed together with its
    heading path (a question about "threads" should find the "Threads" section).

    Note on tiny libraries: BM25's idf is low when there are only a handful of passages, so a
    one-passage library scores below the default ``lookup_min_score`` unless several query terms
    match; that is the intended behaviour (no evidence, no injection).
    """

    K1 = 1.5
    B = 0.75

    def __init__(self, store: ContextStore, *, chunk_chars: int) -> None:
        self._store = store
        self._chunk_chars = int(chunk_chars)
        self._lock = threading.RLock()
        self._built_for: tuple[int, int] | None = None
        self._chunks: list[Chunk] = []
        self._lens: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._avgdl = 0.0

    @property
    def chunk_chars(self) -> int:
        return self._chunk_chars

    @chunk_chars.setter
    def chunk_chars(self, value: int) -> None:
        with self._lock:
            self._chunk_chars = int(value)

    @property
    def size(self) -> int:
        """Number of indexed passages (after bringing the index up to date)."""
        with self._lock:
            self._ensure()
            return len(self._chunks)

    def _ensure(self) -> None:
        key = (self._store.version, self._chunk_chars)
        if key == self._built_for:
            return
        chunks: list[Chunk] = []
        lens: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        for doc in self._store.enabled("reference"):
            for chunk in chunk_text(self._store.text(doc.id), max_chars=self._chunk_chars,
                                    doc_id=doc.id, doc_name=doc.name):
                tf = Counter(tokenize(chunk.title + "\n" + chunk.text))
                idx = len(chunks)
                chunks.append(chunk)
                lens.append(sum(tf.values()))
                for term, count in tf.items():
                    postings.setdefault(term, []).append((idx, count))
        self._chunks, self._lens, self._postings = chunks, lens, postings
        self._avgdl = (sum(lens) / len(lens)) if lens else 0.0
        self._built_for = key

    def search(self, query_terms: dict[str, float], *, top_k: int, min_score: float) -> list[tuple[Chunk, float]]:
        """Best passages for the weighted query, highest score first (ties: document order, then
        passage order). At most one passage per document unless fewer than `top_k` documents
        match; only scores >= `min_score`. Keys of `query_terms` must come from
        :func:`query_terms` / :func:`tokenize` (they are looked up as-is, case-folded)."""
        if top_k <= 0 or not query_terms:
            return []
        with self._lock:
            self._ensure()
            n = len(self._chunks)
            if n == 0:
                return []
            scores: dict[int, float] = {}
            matched: dict[int, int] = {}      # distinct query terms each passage matched
            for term, weight in query_terms.items():
                posting = self._postings.get(term.casefold())
                if not posting:
                    continue
                df = len(posting)
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                for idx, tf in posting:
                    norm = self.K1 * (1.0 - self.B + self.B * self._lens[idx] / (self._avgdl or 1.0))
                    scores[idx] = scores.get(idx, 0.0) + weight * idf * tf * (self.K1 + 1.0) / (tf + norm)
                    matched[idx] = matched.get(idx, 0) + 1
            # A passage that matched a SINGLE query term ("thanks", "slide", "next" in small talk
            # against a README) must clear twice the floor: one rare term alone scores high in
            # BM25 while saying nothing about relevance (review 2026-10-06, measured on this
            # repo's README: such coincidences scored 2.9-5.2 against real hits of 9+).
            ranked = sorted(
                ((s, i) for i, s in scores.items()
                 if s >= min_score and (matched.get(i, 0) >= 2 or s >= 2.0 * min_score)),
                key=lambda p: (-p[0], p[1]),
            )
            picked: list[tuple[float, int]] = []
            seen_docs: set[str] = set()
            for score, idx in ranked:
                doc_id = self._chunks[idx].doc_id
                if doc_id not in seen_docs and len(picked) < top_k:
                    picked.append((score, idx))
                    seen_docs.add(doc_id)
            if len(picked) < top_k:
                taken = {i for _s, i in picked}
                for score, idx in ranked:
                    if idx not in taken and len(picked) < top_k:
                        picked.append((score, idx))
            picked.sort(key=lambda p: (-p[0], p[1]))
            return [(self._chunks[i], s) for s, i in picked]


# --------------------------------------------------------------------------- token estimator
class TokenEstimator:
    """Characters-per-token estimate. Starts from ``[context].chars_per_token`` and, when `auto`,
    is calibrated from the server's own prompt-token counts with an exponential moving average
    (alpha 0.3), always clamped to [2.0, 6.0]. Thread-safe."""

    MIN_CPT = 2.0
    MAX_CPT = 6.0
    ALPHA = 0.3

    def __init__(self, chars_per_token: float, *, auto: bool) -> None:
        self._lock = threading.Lock()
        self._cpt = self._clamp(chars_per_token)
        self.auto = auto

    @classmethod
    def _clamp(cls, value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 3.6
        if value != value:  # NaN
            value = 3.6
        return min(cls.MAX_CPT, max(cls.MIN_CPT, value))

    @property
    def chars_per_token(self) -> float:
        with self._lock:
            return self._cpt

    def set_base(self, chars_per_token: float) -> None:
        """Reset the estimate (the owner changed ``[context].chars_per_token``)."""
        with self._lock:
            self._cpt = self._clamp(chars_per_token)

    def tokens_for_len(self, n_chars: int) -> int:
        """Estimated tokens of `n_chars` characters: ``ceil(n / cpt) + 1`` (0 for nothing)."""
        if n_chars <= 0:
            return 0
        return math.ceil(n_chars / self.chars_per_token) + 1

    def tokens(self, text: str) -> int:
        return self.tokens_for_len(len(text)) if text else 0

    def chars(self, tokens: int) -> int:
        """Characters that fit in `tokens` tokens: ``floor(tokens * cpt)``."""
        return max(0, math.floor(tokens * self.chars_per_token))

    def observe(self, prompt_chars: int, prompt_tokens: int | None) -> None:
        """Fold one measured prompt into the estimate. Ignored when calibration is off or either
        number is under 50 (a tiny prompt says nothing about the ratio)."""
        if not self.auto or prompt_tokens is None or prompt_chars < 50 or prompt_tokens < 50:
            return
        sample = min(self.MAX_CPT, max(self.MIN_CPT, prompt_chars / prompt_tokens))
        with self._lock:
            self._cpt = self._clamp((1.0 - self.ALPHA) * self._cpt + self.ALPHA * sample)

    def note_overflow(self) -> None:
        """The server rejected a prompt as too long: assume tokens are denser (cpt * 0.85, floored
        at 2.0). Applies even with calibration off -- an overflow is a hard fact, not a sample."""
        with self._lock:
            self._cpt = max(self.MIN_CPT, self._cpt * 0.85)


# --------------------------------------------------------------------------- transcript window
@dataclass(frozen=True)
class WindowResult:
    """One rendered transcript window. `windowed` (extension) says the window does not start at
    the beginning of its scope (earlier lines were dropped now or on an earlier turn, or the one
    line shown is cut)."""
    text: str
    anchor_moved: bool
    dropped_text: str
    scope: int | None
    chars: int
    line_count: int
    windowed: bool = False


@dataclass
class _Scope:
    anchor: int | None = None     # id of the last dropped segment; the window starts after it
    summary: str = ""             # the ACTIVE summary: what the <earlier> block carries right now
    staged: str = ""              # a newer summary waiting for the next anchor move (see stage_summary)
    pending: str = ""             # dropped lines not yet folded into the summary
    limit: int = 8000             # bound of `pending` (4 x the last max_chars)


def _bound_tail(text: str, limit: int) -> str:
    """The newest `limit` characters of `text`, starting on a line boundary where possible."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    idx = tail.find("\n")
    return tail[idx + 1:] if 0 <= idx < len(tail) - 1 else tail


class TranscriptWindow:
    """An anchored window over a :class:`~exocortex.transcript.Transcript`, per SCOPE (the
    checkpoint boundary ``since_id``; ``None`` = the whole meeting).

    Why anchored: a sliding window changes the prompt's first line every turn, which defeats
    llama.cpp's prefix cache. Here the start only moves when the window overflows, and then by at
    least ``trim_step`` of its size, so for the following turns the text is the previous text plus
    new lines at the end (a prefix extension). Per scope it also remembers the pending compaction
    text (what was dropped) and the running summary the model wrote for it. Thread-safe.
    """

    def __init__(self, transcript: "Transcript") -> None:
        self._transcript = transcript
        self._lock = threading.RLock()
        self._scopes: dict[int | None, _Scope] = {}

    def _scope(self, scope: int | None) -> _Scope:
        st = self._scopes.get(scope)
        if st is None:
            st = self._scopes[scope] = _Scope()
        return st

    def render(self, *, since_id: int | None, until_id: int | None, max_chars: int, trim_step: float) -> WindowResult:
        """The window text (oldest -> newest "[mm:ss] Label: text" lines) within `max_chars`.

        While the lines after the anchor fit, they are returned whole and the anchor stays. On
        overflow whole lines are dropped from the front until the rest fits in
        ``max_chars * (1 - trim_step)`` (at least one line, never the newest), the anchor moves to
        the last dropped line and the dropped text joins the pending buffer. If the newest line
        alone exceeds `max_chars` its tail is shown, as ``window_text`` does.
        ``trim_step <= 0`` slides (drops only what is needed), like the old behaviour.
        """
        with self._lock:
            st = self._scope(since_id)
            last = self._transcript.last_id()
            if st.anchor is not None and (last is None or last < st.anchor):
                # The transcript was cleared / replaced under us: start this scope afresh.
                st = self._scopes[since_id] = _Scope()
            start = since_id
            if st.anchor is not None:
                start = st.anchor if since_id is None else max(since_id, st.anchor)
            lines = self._transcript.lines(since_id=start, until_id=until_id)
            already = st.anchor is not None
            if max_chars <= 0 or not lines:
                return WindowResult("", False, "", since_id, 0, 0, windowed=already)
            texts = [t for _i, t in lines]
            total = sum(len(t) for t in texts) + len(texts) - 1
            if total <= max_chars:
                return WindowResult("\n".join(texts), False, "", since_id, total, len(texts), windowed=already)

            if len(texts) == 1:
                tail = texts[0][-max_chars:]
                return WindowResult(tail, False, "", since_id, len(tail), 1, windowed=True)
            step = min(max(float(trim_step), 0.0), 0.9)
            target = int(max_chars * (1.0 - step))
            kept = 0
            acc = 0
            for t in reversed(texts):
                extra = len(t) + (1 if kept else 0)
                if acc + extra > target:
                    break
                acc += extra
                kept += 1
            kept = max(1, min(kept, len(texts) - 1))     # drop at least one line, keep at least the newest
            dropped = lines[: len(texts) - kept]
            shown = texts[len(texts) - kept:]
            text = "\n".join(shown)
            if len(text) > max_chars:                    # the newest line alone is longer than the budget
                text = text[-max_chars:]
            dropped_text = "\n".join(t for _i, t in dropped)
            st.anchor = dropped[-1][0]
            st.limit = max(4 * max_chars, 1)
            st.pending = _bound_tail(f"{st.pending}\n{dropped_text}" if st.pending else dropped_text, st.limit)
            if st.staged:
                # The prefix changes on this very turn anyway: the moment a newer summary may
                # replace the active one at no extra cost (measured 2026-10-06: installing it as
                # soon as the job finished re-evaluated the whole prompt on two extra turns).
                st.summary, st.staged = st.staged, ""
            log.debug("context window: anchor -> %s (%d lines dropped, %d kept)", st.anchor, len(dropped), len(shown))
            return WindowResult(text, True, dropped_text, since_id, len(text), len(shown), windowed=True)

    def compact_now(self, scope: int | None, *, keep_lines: int = 2) -> int:
        """Force a compaction of `scope`'s window *now* (the overlay's "Compact" control, like the
        ``/compact`` command): drop everything in the current window except the newest `keep_lines`
        lines into the pending buffer and move the anchor past them, exactly as an overflow would,
        so the next prompt carries only the newest speech and the caller's compaction job folds
        the rest into this scope's running summary. Returns the number of characters dropped (0 =
        the window already held `keep_lines` or fewer lines, nothing to do). Unlike overflow this
        ignores the budget -- it is the user asking to free room immediately."""
        keep = max(1, int(keep_lines))
        with self._lock:
            st = self._scope(scope)
            last = self._transcript.last_id()
            if st.anchor is not None and (last is None or last < st.anchor):
                st = self._scopes[scope] = _Scope()      # transcript cleared under us: start afresh
            start = scope
            if st.anchor is not None:
                start = st.anchor if scope is None else max(scope, st.anchor)
            lines = self._transcript.lines(since_id=start, until_id=None)
            if len(lines) <= keep:
                return 0
            dropped = lines[: len(lines) - keep]
            dropped_text = "\n".join(t for _i, t in dropped)
            if not dropped_text.strip():
                return 0
            st.anchor = dropped[-1][0]
            # Raise the pending bound so this forced drop is never truncated before its job runs
            # (render sizes `limit` to the budget; a user-driven compact can exceed that).
            st.limit = max(st.limit, 4 * (len(dropped_text) + len(st.pending)), 1)
            st.pending = _bound_tail(f"{st.pending}\n{dropped_text}" if st.pending else dropped_text, st.limit)
            if st.staged:
                st.summary, st.staged = st.staged, ""
            log.info("context window: forced compaction of scope %s (%d lines -> pending)",
                     scope, len(dropped))
            return len(dropped_text)

    def anchor(self, scope: int | None) -> int | None:
        with self._lock:
            st = self._scopes.get(scope)
            return st.anchor if st is not None else None

    def has_pending(self, scope: int | None) -> bool:
        """True when dropped text for `scope` still waits to be folded into the summary."""
        with self._lock:
            st = self._scopes.get(scope)
            return bool(st and st.pending.strip())

    def earlier_summary(self, scope: int | None) -> str:
        with self._lock:
            st = self._scopes.get(scope)
            return st.summary if st is not None else ""

    def set_summary(self, scope: int | None, text: str) -> None:
        """Install `text` as the ACTIVE summary at once (the next prompt carries it, and pays a
        prefix re-evaluation for it). The compaction job uses `stage_summary` instead."""
        with self._lock:
            st = self._scope(scope)
            st.summary = text or ""
            st.staged = ""

    def stage_summary(self, scope: int | None, text: str) -> None:
        """Hold `text` as the NEXT summary of `scope` (2026-10-06): it replaces the active one
        inside `render` at the next anchor move -- the one moment the prompt prefix changes
        anyway -- so a finished compaction never costs a re-evaluation of its own. Until then the
        prompt keeps the previous summary (or none); `claim_compaction` folds into the staged
        text, so chained jobs build on each other."""
        with self._lock:
            self._scope(scope).staged = text or ""

    def staged_summary(self, scope: int | None) -> str:
        with self._lock:
            st = self._scopes.get(scope)
            return st.staged if st is not None else ""

    def claim_compaction(self, scope: int | None, max_chars: int | None = None) -> tuple[str, str] | None:
        """``(previous_summary, pending_dropped_text)`` and the claimed text leaves the pending
        buffer; None when nothing is pending. With `max_chars` (2026-10-06 addendum: the room one
        compaction prompt has in the notes lane) only the OLDEST part that fits is claimed, on
        whole lines, at least one line, and the rest stays pending -- the caller re-queues itself
        while `has_pending` says so, so a long backlog is folded in order, chunk by chunk, with
        nothing skipped (the same shape as the notes job's backlog). A job that fails must hand
        the text back (`restore_compaction`)."""
        with self._lock:
            st = self._scopes.get(scope)
            if st is None or not st.pending.strip():
                return None
            latest = st.staged or st.summary
            if max_chars is None or max_chars <= 0 or len(st.pending) <= max_chars:
                pending, st.pending = st.pending, ""
                return latest, pending
            lines = st.pending.split("\n")
            kept: list[str] = []
            used = 0
            for line in lines:
                extra = len(line) + (1 if kept else 0)
                if kept and used + extra > max_chars:
                    break
                kept.append(line)
                used += extra
            claimed = "\n".join(kept)
            st.pending = "\n".join(lines[len(kept):])
            return latest, claimed

    def restore_compaction(self, scope: int | None, dropped: str) -> None:
        """Give claimed text back (prepended: it is older than anything dropped since)."""
        if not dropped:
            return
        with self._lock:
            st = self._scope(scope)
            joined = f"{dropped}\n{st.pending}" if st.pending else dropped
            st.pending = _bound_tail(joined, st.limit)

    def reset(self) -> None:
        """A new session: every anchor, summary and pending buffer is gone."""
        with self._lock:
            self._scopes.clear()

    def forget_scope(self, scope: int | None) -> None:
        """A checkpoint scope that is no longer current."""
        with self._lock:
            self._scopes.pop(scope, None)


# --------------------------------------------------------------------------- manager
@dataclass
class ContextBundle:
    """What one prompt gets. Every text is "" when absent. `compact_scope` may legitimately be
    None (the whole meeting) -- `compact_wanted` is the flag that says whether to compact."""
    transcript: str
    reference: str
    lookup: str
    earlier: str
    screen: str
    usage: ContextUsageEvent
    compact_scope: int | None
    compact_wanted: bool
    documents_used: int


@dataclass
class _ReferenceBlock:
    """The stable <reference> text as droppable pieces: injected documents (in store order), then
    the catalogue of reference-mode documents. `drop_one` removes pieces in the order the budget
    planner gives them back: injected documents from the last to the first, then catalogue lines
    from the last to the first (the header goes with the last line)."""
    docs: list[tuple[str, str]] = field(default_factory=list)     # (doc_id, "### name\ntext")
    cat_header: str = ""
    cat_lines: list[str] = field(default_factory=list)
    trimmed: bool = False

    def text(self) -> str:
        parts = [body for _id, body in self.docs]
        if self.cat_lines:
            parts.append(self.cat_header + "\n" + "\n".join(self.cat_lines))
        return "\n\n".join(parts)

    def injected_ids(self) -> list[str]:
        return [doc_id for doc_id, _b in self.docs]

    def drop_one(self) -> bool:
        if self.docs:
            self.docs.pop()
        elif self.cat_lines:
            self.cat_lines.pop()
        else:
            return False
        self.trimmed = True
        return True


_CATALOGUE_HEADER = ("Reference documents available (not shown in full; relevant passages appear in "
                     "<lookup> when the question calls for them):")
_QUESTION_KINDS = frozenset({"auto", "manual", "ask", "screen"})
_CUT_MARK = "\n[...]"


def _first_line(text: str) -> str:
    """First markdown heading, else first non-empty non-fence line, of `text` (for a digest-less
    catalogue entry)."""
    fallback = ""
    for line in text[:3000].split("\n"):
        stripped = line.strip()
        if not stripped or _FENCE_RE.match(stripped):
            continue
        m = _HEADING_RE.match(stripped)
        if m and m.group(2):
            return m.group(2)
        fallback = fallback or stripped
    return fallback


class ContextManager:
    """Owns the store, the index, the estimator and the window: the ONE object the Assistant
    talks to. `cfg` is read live on every call, so settings apply at once.

    ``set_slot_tokens(n)`` is what the Assistant calls from its health check with the server's
    per-request context (llama-server's ``/props`` ``n_ctx``). Measured 2026-10-06 (SPEC.md): with
    ``--parallel 2`` llama-server runs ``--kv-unified``, every slot reports the FULL ``ctx_size``
    and the two slots share that one pool -- so the fallback when the server does not say is
    ``cfg.llm.ctx_size`` itself (never divided by ``parallel``), and :meth:`job_limit` is what
    keeps an answer and a notes-lane job (notes, a compaction summary, a document digest) that run
    at the same time inside the pool together (``[context].lane_share``).
    """

    def __init__(self, cfg: Config, store: ContextStore | None, transcript: "Transcript") -> None:
        self._cfg = cfg
        self._transcript = transcript
        self.store: ContextStore = store if store is not None else ContextStore(None)
        self.window = TranscriptWindow(transcript)
        c = cfg.context
        self.estimator = TokenEstimator(c.chars_per_token, auto=c.auto_calibrate)
        self.index = ContextIndex(self.store, chunk_chars=c.chunk_chars)
        self._cfg_cpt = c.chars_per_token
        self._lock = threading.Lock()          # slot size + last usage
        self._slot: int | None = None
        self._usage: ContextUsageEvent | None = None

    # ---- server context
    def set_slot_tokens(self, n: int | None) -> None:
        with self._lock:
            self._slot = int(n) if n is not None and int(n) > 0 else None

    @property
    def slot_tokens(self) -> int:
        with self._lock:
            slot = self._slot
        if slot is not None:
            return slot
        return max(1, int(self._cfg.llm.ctx_size))

    def job_limit(self, lane: str) -> int:
        """The tokens one job of `lane` ("answer" | "notes") may occupy in the server's pool, prompt
        and answer together (2026-10-06 addendum). With ``[llm].parallel <= 1`` the pool is one
        request's own: ``slot_tokens - reserve_tokens`` for either lane. With two slots over one
        unified pool the answer lane gets ``1 - lane_share`` of it (minus the reserve) and the
        notes lane ``lane_share`` (at least 512), so the two never overflow the pool together."""
        c = self._cfg.context
        slot = self.slot_tokens
        reserve = max(0, int(c.reserve_tokens))
        if int(self._cfg.llm.parallel) <= 1:
            return max(1, slot - reserve)
        share = min(max(float(c.lane_share), 0.05), 0.95)
        if lane == "notes":
            return max(512, int(slot * share))
        return max(1, int(slot * (1.0 - share)) - reserve)

    def available_chars(self, lane: str, *, max_tokens: int, system_text: str, other_text: str = "") -> int:
        """The characters one more text block may take in a prompt of `lane` that already carries
        `system_text`, `other_text` and an answer budget of `max_tokens` (2026-10-06 addendum):
        the notes job's transcript chunk, the compaction job's dropped text, a digest's excerpt."""
        est = self.estimator
        room = (self.job_limit(lane) - max(0, int(max_tokens)) - est.tokens(system_text or "")
                - est.tokens(other_text or "") - _LANE_TEMPLATE_TOKENS)
        return est.chars(max(0, room))

    # ---- policy
    def wants_reference(self, kind: str) -> bool:
        """Does a job of this kind get the reference material? ``include_in``: "all" -> every kind;
        "questions" -> auto / manual / ask / screen only ("complete" counts as "all" only);
        "none" -> never. An unknown value behaves like "all"."""
        mode = self._cfg.context.include_in
        if mode == "none":
            return False
        if mode == "questions":
            return kind in _QUESTION_KINDS
        return True

    def _sync(self) -> None:
        """Pull live config values into the long-lived helpers."""
        c = self._cfg.context
        self.estimator.auto = c.auto_calibrate
        if c.chars_per_token != self._cfg_cpt:
            self._cfg_cpt = c.chars_per_token
            self.estimator.set_base(c.chars_per_token)
        self.index.chunk_chars = c.chunk_chars

    # ---- pieces
    def _cap(self, tokens: int) -> int:
        """Characters for a block that must estimate to at most `tokens` tokens."""
        return self.estimator.chars(max(0, tokens - 1))

    def _render_reference(self, budget_tokens: int) -> _ReferenceBlock:
        """The <reference> block within `budget_tokens`: the catalogue first (small, and the model
        needs it to know what exists), then each injected document gets an equal share of what is
        left (leftover from short documents goes to the longer ones, one redistribution pass); a
        document that cannot get a useful share is left out whole."""
        block = _ReferenceBlock()
        inject = self.store.enabled("inject")
        catalogue = self.store.enabled("reference")
        if not inject and not catalogue:
            return block
        budget = self._cap(budget_tokens)
        if budget <= 0:
            block.trimmed = True
            return block

        cat_lines: list[str] = []
        for doc in catalogue:
            blurb = ""
            if doc.digest_state == "ready" and doc.digest:
                blurb = " ".join(doc.digest.split())
            if not blurb:
                blurb = _first_line(self.store.text(doc.id))
            cat_lines.append(f"- {doc.name}: {blurb}"[:_CATALOGUE_LINE_CHARS].rstrip())
        cat_cap = int(budget * _CATALOGUE_SHARE) if inject else budget
        while cat_lines and len(_CATALOGUE_HEADER) + 1 + sum(len(x) + 1 for x in cat_lines) > cat_cap:
            cat_lines.pop()
            block.trimmed = True
        used = 0
        if cat_lines:
            block.cat_header = _CATALOGUE_HEADER
            block.cat_lines = cat_lines
            used = len(_CATALOGUE_HEADER) + 1 + sum(len(x) + 1 for x in cat_lines) + 2

        entries: list[tuple[ContextDocument, str, str]] = []
        for doc in inject:
            text = self.store.text(doc.id)
            if text.strip():
                entries.append((doc, f"### {doc.name}\n", text))
        if entries:
            room = budget - used - 2 * (len(entries) - 1)
            base = max(0, room // len(entries))
            leftover = 0
            cut_count = 0
            for _d, header, text in entries:
                full = len(header) + len(text)
                if full <= base:
                    leftover += base - full
                else:
                    cut_count += 1
            extra = leftover // cut_count if cut_count else 0
            for doc, header, text in entries:
                full = len(header) + len(text)
                share = base if full <= base else base + extra
                if full <= share:
                    block.docs.append((doc.id, header + text))
                    continue
                room_text = share - len(header) - len(_CUT_MARK)
                if room_text < 40:
                    block.trimmed = True
                    continue
                cut, _ = _cut(text, room_text)
                block.docs.append((doc.id, header + cut + _CUT_MARK))
                block.trimmed = True
        return block

    def _recent_lines(self, since_id: int | None, until_id: int | None, n: int = 3) -> str:
        try:
            lines = self._transcript.lines(since_id=since_id, until_id=until_id)
        except Exception:  # noqa: BLE001 - a retrieval hint must never break a prompt
            return ""
        return "\n".join(t for _i, t in lines[-n:])

    def _lookup_passages(self, question: str, screen_text: str, since_id: int | None,
                         until_id: int | None) -> list[tuple[str, str]]:
        """Matching passages as ``(doc_id, "[doc > title]\\ntext")``, best first. Empty when the
        library has no enabled reference-mode document or nothing scores above the threshold."""
        c = self._cfg.context
        if not self.store.enabled("reference"):
            return []
        terms = query_terms(question, self._recent_lines(since_id, until_id), (screen_text or "")[:1000])
        if not terms:
            return []
        out: list[tuple[str, str]] = []
        for chunk, _score in self.index.search(terms, top_k=int(c.lookup_top_k), min_score=float(c.lookup_min_score)):
            label = chunk.doc_name if chunk.title == chunk.doc_name else f"{chunk.doc_name} > {chunk.title}"
            out.append((chunk.doc_id, f"[{label}]\n{chunk.text}"))
        return out

    def _fit_lookup(self, passages: list[tuple[str, str]],
                    budget_tokens: int) -> tuple[list[tuple[str, str]], bool]:
        """Greedy-fit passages (best first) into the lookup budget; the best one is cut on a line
        boundary when even it does not fit. -> (kept, best_was_cut): a lower-ranked passage that
        simply does not fit beside the better ones is the normal case (top_k is a ceiling, the
        budget the real limit) and is NOT reported as a trim -- measured live, that made every
        answer read "trimmed" in the Context tab; only cutting (or dropping) the best passage is."""
        kept: list[tuple[str, str]] = []
        for doc_id, body in passages:
            candidate = [*kept, (doc_id, body)]
            if self.estimator.tokens("\n\n".join(b for _d, b in candidate)) <= budget_tokens:
                kept = candidate
                continue
            if not kept:
                room = self._cap(budget_tokens) - len(_CUT_MARK)
                if room >= 40:
                    cut, _ = _cut(body, room)
                    kept = [(doc_id, cut + _CUT_MARK)]
                return kept, True
            return kept, False
        return kept, False

    # ---- the main entry point
    def build(self, *, kind: str, question: str, screen_text: str, since_id: int | None,
              until_id: int | None, max_tokens: int, system_text: str, fixed_chars: int) -> ContextBundle:
        """Size and assemble every context block for one prompt. Never raises: on any internal
        failure it falls back to the smallest sane prompt (just the newest transcript lines) with
        ``usage.trimmed`` set.

        Auto budget (``[context].budget_mode = "auto"``): ``avail = slot - max_tokens - reserve -
        system - fixed``; the transcript gets ``min([llm].context_tokens, avail - screen -
        reference - lookup - earlier)``. When that is under ``transcript_min_tokens`` the other
        blocks give back, in order: lookup passages, reference (injected documents last-first,
        then catalogue lines), the earlier summary, the screen text (down to 25 %). If the floor
        still cannot be met the transcript simply gets what is left -- its newest line always
        survives. Fixed mode keeps the old ``context_tokens * 4`` transcript and fits nothing.
        """
        try:
            return self._build(kind=kind, question=question or "", screen_text=screen_text or "",
                               since_id=since_id, until_id=until_id, max_tokens=max_tokens,
                               system_text=system_text or "", fixed_chars=fixed_chars)
        except Exception:  # noqa: BLE001 - the Assistant's worker must always get a prompt
            log.warning("context: build failed; falling back to a minimal prompt", exc_info=True)
            return self._fallback(kind, since_id, until_id)

    def _fallback(self, kind: str, since_id: int | None, until_id: int | None) -> ContextBundle:
        try:
            text = self._transcript.window_text(1200, until_id=until_id, since_id=since_id)
        except Exception:  # noqa: BLE001
            text = ""
        tokens = self.estimator.tokens(text)
        usage = ContextUsageEvent(job_id=0, kind=kind, slot_tokens=self.slot_tokens, trimmed=True,  # type: ignore[arg-type]
                                  transcript_tokens=tokens, used_tokens=tokens,
                                  chars_per_token=self.estimator.chars_per_token)
        return ContextBundle(text, "", "", "", "", usage, since_id, False, 0)

    def _build(self, *, kind: str, question: str, screen_text: str, since_id: int | None,
               until_id: int | None, max_tokens: int, system_text: str, fixed_chars: int) -> ContextBundle:
        self._sync()
        c = self._cfg.context
        llm = self._cfg.llm
        est = self.estimator
        auto = c.budget_mode != "fixed"
        slot = self.slot_tokens
        system_t = est.tokens(system_text)
        fixed_t = est.tokens_for_len(max(0, int(fixed_chars)))
        # The answer lane's share of the pool (job_limit already holds back the reserve and, with
        # two slots, the notes lane's share) minus the answer and the system prompt: the user turn.
        budget_t = max(0, self.job_limit("answer") - max(0, int(max_tokens)) - system_t)
        avail = budget_t - fixed_t
        trimmed = False

        # ---- earlier summary (kept to a sane size even when the model overshot). A staged
        # summary is promoted by the window at an anchor move, so it is budgeted for here (the
        # larger of the two) and only used below when this build did move the anchor.
        earlier_active = self.window.earlier_summary(since_id)
        earlier_staged = self.window.staged_summary(since_id)
        earlier_cap = est.chars(max(1, int(c.summary_tokens)) * 2)
        earlier = earlier_staged if len(earlier_staged) >= len(earlier_active) else earlier_active
        if len(earlier) > earlier_cap:
            earlier, _ = _cut(earlier, earlier_cap)
            trimmed = True
        screen = screen_text
        wants = self.wants_reference(kind)
        block = _ReferenceBlock()
        lookup: list[tuple[str, str]] = []

        def lookup_text() -> str:
            return "\n\n".join(b for _d, b in lookup)

        def render_lookup(budget: int) -> None:
            nonlocal lookup, trimmed
            if wants and budget > 0:
                lookup, cut_any = self._fit_lookup(
                    self._lookup_passages(question, screen_text, since_id, until_id), budget)
                trimmed = trimmed or cut_any

        if auto:
            ctx_tokens = max(1, int(llm.context_tokens))
            floor = min(max(0, int(c.transcript_min_tokens)), ctx_tokens)
            screen_t0 = est.tokens(screen)
            earlier_t = est.tokens(earlier)
            # The <reference> block is the PREFIX of the user turn, so its size must not follow
            # this turn's screen length (review 2026-10-06: a long screen one turn and a short one
            # the next made the block flip in and out, re-evaluating the whole prompt each time).
            # Its budget is therefore computed from stable quantities only -- the lane, the
            # transcript floor, the summary and room for the LARGEST screen this job may carry
            # (the configured maximum; a hands-free cumulative screen that exceeds it costs that
            # one turn) -- and documents are CUT to that budget inside `_render_reference`, never
            # dropped whole because one turn's screen happened to be long.
            screen_reserve = max(screen_t0, est.tokens_for_len(int(self._cfg.screen.max_text_chars))) if screen else 0
            if wants:
                ref_budget = min(int(c.reference_tokens), max(0, avail - floor - earlier_t - screen_reserve))
                block = self._render_reference(ref_budget)
                trimmed = trimmed or block.trimmed
                # The lookup passages get what the SAME reserve leaves after the reference block
                # (not what this turn's shorter screen happens to leave): they are volatile and
                # documented as the first thing to go, so they must never outlive the reference
                # in a tight slot; the slack of a short screen goes to the transcript instead.
                render_lookup(min(int(c.lookup_tokens),
                                  max(0, avail - floor - earlier_t - screen_reserve - est.tokens(block.text()))))

            def free_tokens() -> int:
                return (avail - est.tokens(screen) - est.tokens(block.text())
                        - est.tokens(lookup_text()) - est.tokens(earlier))

            if free_tokens() < floor:
                # Only when even the stable budgets above could not leave the floor (a tiny
                # slot): give back in order -- lookup passages (volatile), the summary, the screen
                # text (volatile, down to 25 %), and only then the stable reference block.
                trimmed = True
                while lookup and free_tokens() < floor:        # 1. lookup passages, worst first
                    lookup.pop()
                if earlier and free_tokens() < floor:           # 2. the earlier summary
                    earlier = ""
                if screen and free_tokens() < floor:            # 3. screen text, down to 25 %
                    keep = max(math.ceil(screen_t0 * _SCREEN_FLOOR), screen_t0 - (floor - free_tokens()))
                    if keep < screen_t0:
                        screen = screen[: self._cap(keep)]
                while free_tokens() < floor and block.drop_one():   # 4. the reference block, last
                    pass
            tb = max(0, min(ctx_tokens, free_tokens()))
            slot_limited = tb < ctx_tokens
            # The newest speech always gets a little room even in a degenerate slot -- but never
            # more than a deliberately tiny [llm].context_tokens allows (review, 2026-10-06).
            max_chars = max(self._cap(tb), min(_MIN_TRANSCRIPT_CHARS, self._cap(ctx_tokens)))
        else:
            if wants:
                block = self._render_reference(int(c.reference_tokens))
                trimmed = trimmed or block.trimmed
                render_lookup(int(c.lookup_tokens))
            slot_limited = False
            max_chars = max(1, int(llm.context_tokens) * 4)

        trim_step = float(c.trim_step)
        if slot_limited and max_chars > 0:
            # The window is small because the SERVER's context is: a step-trim leaves
            # max_chars * (1 - trim_step) on screen and that must not fall under the transcript
            # floor (the newest speech keeps its minimum), even if a tight slot then trims in
            # smaller steps. When [llm].context_tokens is the limit the owner chose that size, so
            # the full step applies.
            floor_chars = self._cap(min(max(0, int(c.transcript_min_tokens)), max(1, int(llm.context_tokens))))
            trim_step = min(trim_step, max(0.0, 1.0 - floor_chars / max_chars))
        result = self.window.render(since_id=since_id, until_id=until_id, max_chars=max_chars,
                                    trim_step=trim_step)
        if result.anchor_moved or result.windowed:
            trimmed = True
        if earlier:
            # What the window holds NOW (promoted on an anchor move, else the previous one), within
            # the same cap; dropped entirely if the budget pass above gave the summary back.
            now = self.window.earlier_summary(since_id)
            earlier = _cut(now, earlier_cap)[0] if len(now) > earlier_cap else now

        compact_wanted = False
        if c.compaction == "summary":
            compact_wanted = result.anchor_moved or self.window.has_pending(since_id)
        elif result.dropped_text:
            self.window.claim_compaction(since_id)  # compaction off: never let the buffer pile up

        reference_text = block.text()
        lookup_str = lookup_text()
        injected = block.injected_ids()
        used_docs = len(injected) + len({d for d, _b in lookup} - set(injected))
        transcript_t = est.tokens(result.text)
        reference_t = est.tokens(reference_text)
        lookup_t = est.tokens(lookup_str)
        earlier_t = est.tokens(earlier)
        screen_t = est.tokens(screen)
        usage = ContextUsageEvent(
            job_id=0, kind=kind,  # type: ignore[arg-type]
            slot_tokens=slot, budget_tokens=budget_t,
            used_tokens=fixed_t + transcript_t + reference_t + lookup_t + earlier_t + screen_t,
            transcript_tokens=transcript_t, reference_tokens=reference_t, lookup_tokens=lookup_t,
            earlier_tokens=earlier_t, screen_tokens=screen_t, system_tokens=system_t,
            trimmed=trimmed, anchor_moved=result.anchor_moved, chars_per_token=est.chars_per_token,
            documents=used_docs,
        )
        return ContextBundle(result.text, reference_text, lookup_str, earlier, screen, usage,
                             since_id, compact_wanted, used_docs)

    # ---- feedback from the Assistant
    def observe_prompt(self, prompt_chars: int, prompt_tokens: int | None) -> None:
        """Calibrate the characters-per-token estimate from the server's own count."""
        if prompt_tokens is None:
            return
        self._sync()
        self.estimator.observe(prompt_chars, prompt_tokens)

    def note_overflow(self) -> None:
        self.estimator.note_overflow()

    def claim_compaction(self, scope: int | None, max_chars: int | None = None) -> tuple[str, str] | None:
        return self.window.claim_compaction(scope, max_chars)

    def compact_now(self, scope: int | None, *, keep_lines: int = 2) -> int:
        """Force `scope`'s window to shed everything but its newest lines into the pending buffer
        now (see :meth:`TranscriptWindow.compact_now`); returns the characters dropped."""
        return self.window.compact_now(scope, keep_lines=keep_lines)

    def restore_compaction(self, scope: int | None, dropped: str) -> None:
        self.window.restore_compaction(scope, dropped)

    def set_summary(self, scope: int | None, text: str) -> None:
        self.window.set_summary(scope, text)

    def stage_summary(self, scope: int | None, text: str) -> None:
        self.window.stage_summary(scope, text)

    def reset_session(self) -> None:
        """A new session: the window forgets its anchors and summaries; documents stay."""
        self.window.reset()
        with self._lock:
            self._usage = None

    def record_usage(self, usage: ContextUsageEvent) -> None:
        with self._lock:
            self._usage = usage

    def usage_snapshot(self) -> dict[str, object]:
        """For ``EngineActions.context_usage()``: the last usage event's fields (zeros / None before
        the first answer; `has_usage` says which), plus `slot_tokens`, the live `chars_per_token`,
        the whole-meeting `earlier_summary`, `documents` (ENABLED documents -- the event's own
        count of contributing ones is `documents_used`)."""
        with self._lock:
            ev = self._usage
        snap: dict[str, object] = {
            "has_usage": ev is not None, "job_id": 0, "kind": None, "budget_tokens": 0, "used_tokens": 0,
            "transcript_tokens": 0, "reference_tokens": 0, "lookup_tokens": 0, "earlier_tokens": 0,
            "screen_tokens": 0, "system_tokens": 0, "prompt_tokens": None, "cached_tokens": None,
            "trimmed": False, "anchor_moved": False, "documents_used": 0,
        }
        if ev is not None:
            data = dataclasses.asdict(ev)
            data.pop("ts", None)
            snap.update(data)
            snap["documents_used"] = data.get("documents", 0)
        snap["slot_tokens"] = self.slot_tokens
        snap["chars_per_token"] = self.estimator.chars_per_token
        snap["earlier_summary"] = self.window.earlier_summary(None)
        snap["documents"] = len(self.store.enabled())
        return snap
