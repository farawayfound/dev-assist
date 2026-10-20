"""A tiny, dependency-free Markdown-to-HTML renderer for the overlay's rich text views.

Only the subset of Markdown the overlay actually produces/consumes is supported:
headings (``# `` .. ``###### ``), unordered lists (``- `` / ``* ``), ordered lists
(``1. ``), checkbox list items (``- [ ] `` / ``- [x] ``), ``**bold**``, ``*italic*``,
inline ``` `code` ```, paragraphs and line breaks. All input is treated as plain text:
HTML metacharacters are escaped before any Markdown substitution, so nothing in the
transcript, notes or an LLM answer can inject markup into a ``QTextBrowser``.

No external dependencies; safe to feed straight into ``QTextBrowser.setHtml``/``setText``.
"""
from __future__ import annotations

import html
import logging
import re

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_UL_RE = re.compile(r"^[-*]\s+(.*)$")
_OL_RE = re.compile(r"^(\d+)[.)]\s+(.*)$")
_CHECKBOX_RE = re.compile(r"^\[([ xX])\]\s*(.*)$")

_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")

_CHECKED_GLYPH = "☑"  # ballot box with check
_UNCHECKED_GLYPH = "☐"  # ballot box

#: Style for a fenced code block (```...```). white-space:pre-wrap keeps indentation and line
#: breaks while still wrapping long lines inside the narrow overlay; the tint/border set it apart
#: from prose. Used by autonomous "solve" answers, which are mostly code.
_PRE_STYLE = (
    "white-space:pre-wrap; font-family:Consolas,'Cascadia Mono','Courier New',monospace; "
    "background:rgba(127,127,127,0.14); border:1px solid rgba(127,127,127,0.28); "
    "border-radius:4px; padding:6px 8px; margin:4px 0;"
)


def _inline(text: str) -> str:
    """Escape HTML then apply inline formatting (code, bold, italic) to one logical line."""
    escaped = html.escape(text, quote=True)

    codes: list[str] = []

    def _stash_code(m: re.Match[str]) -> str:
        codes.append(f"<code>{m.group(1)}</code>")
        return f"\x00{len(codes) - 1}\x00"

    escaped = _CODE_RE.sub(_stash_code, escaped)
    escaped = _BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    escaped = _ITALIC_RE.sub(r"<em>\1</em>", escaped)
    escaped = _PLACEHOLDER_RE.sub(lambda m: codes[int(m.group(1))], escaped)
    return escaped


def _list_item_html(content: str) -> str:
    """Render one ``- ``/``* `` list item, recognising an optional ``[ ]``/``[x]`` checkbox."""
    cb = _CHECKBOX_RE.match(content)
    if cb:
        glyph = _CHECKED_GLYPH if cb.group(1).lower() == "x" else _UNCHECKED_GLYPH
        return f'<li style="list-style:none;">{glyph}&nbsp;{_inline(cb.group(2))}</li>'
    return f"<li>{_inline(content)}</li>"


def md_to_html(text: str) -> str:
    """Render a small Markdown subset to HTML suitable for ``QTextBrowser``.

    Supports paragraphs (blank-line separated, single newlines become ``<br>``),
    ``#``..``######`` headings, ``-``/``*``/``1.`` lists (with ``- [ ]``/``- [x]``
    checkboxes), ``**bold**``, ``*italic*`` and inline `` `code` ``. Everything else
    is treated as literal text and HTML-escaped.
    """
    if not text:
        return ""

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[str] = []
    para_lines: list[str] = []
    list_items: list[str] = []
    list_tag: str | None = None        # "ul", "ol" or 'ol start="N"' (the open list's tag)
    ol_next = 1   # the number the next "N. item" line must carry to join the open <ol>
    in_code = False
    code_lines: list[str] = []
    fence_marker = ""

    def flush_code() -> None:
        nonlocal in_code, fence_marker
        # Preserve inner lines verbatim (escaped), no inline formatting, no stripping.
        body = html.escape("\n".join(code_lines), quote=True)
        blocks.append(f'<pre style="{_PRE_STYLE}">{body}</pre>')
        code_lines.clear()
        in_code = False
        fence_marker = ""

    def flush_para() -> None:
        if para_lines:
            blocks.append("<p>" + "<br>\n".join(_inline(l) for l in para_lines) + "</p>")
            para_lines.clear()

    def flush_list() -> None:
        nonlocal list_tag, ol_next
        if list_items and list_tag:
            close = list_tag.split(" ", 1)[0]          # 'ol start="3"' closes as </ol>
            blocks.append(f"<{list_tag}>" + "".join(list_items) + f"</{close}>")
        list_items.clear()
        list_tag = None
        ol_next = 1

    for raw in lines:
        fence = _FENCE_RE.match(raw)
        if in_code:
            # Only a fence of the same type (```/~~~) closes the block; everything else is code.
            if fence is not None and fence.group(1)[0] == fence_marker and not fence.group(2).strip():
                flush_code()
            else:
                code_lines.append(raw)
            continue
        if fence is not None:
            flush_para()
            flush_list()
            in_code = True
            fence_marker = fence.group(1)[0]
            continue

        stripped = raw.strip()
        if not stripped:
            flush_para()
            flush_list()
            continue

        m = _HEADING_RE.match(stripped)
        if m:
            flush_para()
            flush_list()
            level = len(m.group(1))
            blocks.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            continue

        m = _UL_RE.match(stripped)
        if m:
            flush_para()
            if list_tag != "ul":
                flush_list()
                list_tag = "ul"
            list_items.append(_list_item_html(m.group(1)))
            continue

        m = _OL_RE.match(stripped)
        if m:
            number = int(m.group(1))
            if list_tag == "ol" and number == ol_next:
                list_items.append(f"<li>{_inline(m.group(2))}</li>")
                ol_next += 1
                continue
            # An enumerator out of sequence -- a lone "3. 16" naming a multiple-choice option, an
            # item after a blank line, "2." right after prose -- starts a new list that keeps its
            # number (QTextBrowser honours <ol start>), never one renumbered from 1.
            flush_para()
            flush_list()
            list_tag = "ol" if number == 1 else f'ol start="{number}"'
            list_items.append(f"<li>{_inline(m.group(2))}</li>")
            ol_next = number + 1
            continue

        flush_list()
        para_lines.append(stripped)

    if in_code:  # unterminated fence at EOF: still render what we collected
        flush_code()
    flush_para()
    flush_list()
    return "\n".join(blocks)
