"""Tests for exocortex.context: text extraction, chunking, ContextDocument and ContextStore.

Everything is offline; office documents are built with zipfile in the tests, the PDF test writes a
minimal one by hand (pypdf is installed, the test skips if it is not)."""
from __future__ import annotations

import json
import threading
import zipfile
from pathlib import Path

import pytest

from exocortex.context import (
    DOCUMENT_KINDS,
    Chunk,
    ContextDocument,
    ContextError,
    ContextStore,
    chunk_text,
    extract_text,
)

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _write(tmp_path: Path, name: str, content: str | bytes) -> Path:
    p = tmp_path / name
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8", newline="")
    return p


def _docx(tmp_path: Path, body_xml: str, name: str = "d.docx") -> Path:
    p = tmp_path / name
    xml = f'<?xml version="1.0"?><w:document xmlns:w="{_W}"><w:body>{body_xml}</w:body></w:document>'
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("word/document.xml", xml)
    return p


def _pptx(tmp_path: Path, slides: dict[int, list[str]], name: str = "d.pptx") -> Path:
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        for num, paragraphs in slides.items():
            paras = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in paragraphs)
            zf.writestr(f"ppt/slides/slide{num}.xml",
                        f'<?xml version="1.0"?><p:sld xmlns:p="x" xmlns:a="{_A}"><p:cSld>{paras}</p:cSld></p:sld>')
    return p


def _minimal_pdf(path: Path, pages: list[str]) -> None:
    """A tiny valid PDF with one text line per page (Helvetica, no compression)."""
    objs: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(pages):
        page_no, content_no = 3 + 2 * i, 4 + 2 * i
        objs.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents {content_no} 0 R "
                     f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>").encode())
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
        objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for idx, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


# ---- extract_text -------------------------------------------------------------

def test_readding_the_same_text_as_another_kind_leaves_no_orphaned_copy(tmp_path) -> None:
    """Review 2026-10-06: the stored copy's extension follows the kind (.txt / .md) while the id
    follows the text alone, so the superseded copy is deleted on re-add and remove() leaves
    nothing behind."""
    store = ContextStore(tmp_path)
    a = store.add_text("notes", "same text twice over", mode="inject", max_chars=10000, digest_wanted=False, kind="text")
    b = store.add_text("notes", "same text twice over", mode="inject", max_chars=10000, digest_wanted=False, kind="markdown")
    assert a.id == b.id
    docs_dir = tmp_path / "docs"
    assert sorted(p.name for p in docs_dir.iterdir()) == [f"{b.id}.md"]
    assert store.remove(b.id)
    assert list(docs_dir.iterdir()) == []


def test_markdown_kind_and_normalisation(tmp_path: Path) -> None:
    p = _write(tmp_path, "a.md", "# Title  \r\nline one   \r\n\r\n\r\n\r\n\r\nline two\r\n")
    text, kind = extract_text(p, max_chars=1000)
    assert kind == "markdown"
    assert text == "# Title\nline one\n\n\nline two"   # CRLF -> LF, trailing blanks stripped, 4 blanks -> 2


@pytest.mark.parametrize("name", ["a.txt", "a.rst", "a.log", "a.csv", "a.json", "a.toml", "a.yaml", "a.yml", "a.ini", "a.cfg"])
def test_text_extensions(tmp_path: Path, name: str) -> None:
    text, kind = extract_text(_write(tmp_path, name, "hello\nworld"), max_chars=100)
    assert (text, kind) == ("hello\nworld", "text")


def test_code_is_fenced_with_language(tmp_path: Path) -> None:
    text, kind = extract_text(_write(tmp_path, "m.py", "def f():\n    return 1\n"), max_chars=100)
    assert kind == "code"
    assert text == "```py\ndef f():\n    return 1\n```"


def test_code_containing_a_fence_gets_a_longer_fence(tmp_path: Path) -> None:
    text, _ = extract_text(_write(tmp_path, "m.js", "const s = '```';"), max_chars=100)
    assert text.startswith("````js\n") and text.endswith("\n````")


def test_html_strips_tags_scripts_and_styles(tmp_path: Path) -> None:
    html = ("<html><head><title>Doc</title><style>p{color:red}</style></head><body>"
            "<h2>Intro</h2><p>Hello <b>world</b></p><script>alert(1)</script>"
            "<ul><li>one</li><li>two</li></ul><table><tr><td>a</td><td>b</td></tr></table></body></html>")
    text, kind = extract_text(_write(tmp_path, "a.html", html), max_chars=1000)
    assert kind == "html"
    assert "alert" not in text and "color:red" not in text
    assert "## Intro" in text and "Hello world" in text
    assert "- one" in text and "- two" in text
    assert "a | b" in text


def test_docx_paragraphs_headings_and_tables(tmp_path: Path) -> None:
    body = (
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Architecture</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>First </w:t></w:r><w:r><w:t>paragraph.</w:t></w:r></w:p>'
        '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Name</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Role</w:t></w:r></w:p></w:tc></w:tr>'
        '<w:tr><w:tc><w:p><w:r><w:t>Bus</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Events</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
        '<w:p><w:r><w:t>Last.</w:t></w:r></w:p>'
    )
    text, kind = extract_text(_docx(tmp_path, body), max_chars=1000)
    assert kind == "docx"
    assert text.split("\n\n") == ["# Architecture", "First paragraph.", "Name | Role", "Bus | Events", "Last."]


def test_docx_bad_zip_raises_context_error(tmp_path: Path) -> None:
    with pytest.raises(ContextError):
        extract_text(_write(tmp_path, "bad.docx", b"not a zip"), max_chars=100)


def test_pptx_slides_in_numeric_order(tmp_path: Path) -> None:
    p = _pptx(tmp_path, {10: ["Ten"], 2: ["Two", "more"], 1: ["One"]})
    text, kind = extract_text(p, max_chars=1000)
    assert kind == "pptx"
    assert text == "## Slide 1\nOne\n\n## Slide 2\nTwo\nmore\n\n## Slide 3\nTen"


def test_pdf_pages_get_headings(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    p = tmp_path / "a.pdf"
    _minimal_pdf(p, ["Alpha page", "Beta page"])
    text, kind = extract_text(p, max_chars=1000)
    assert kind == "pdf"
    assert "## Page 1\nAlpha page" in text and "## Page 2\nBeta page" in text


def test_pdf_without_pypdf_names_the_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    real_import = builtins.__import__

    def fake(name, *a, **kw):
        if name == "pypdf":
            raise ImportError("no pypdf")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ContextError, match="pip install pypdf"):
        extract_text(_write(tmp_path, "a.pdf", b"%PDF-1.4"), max_chars=100)


def test_garbage_pdf_is_a_context_error(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    with pytest.raises(ContextError):
        extract_text(_write(tmp_path, "a.pdf", b"definitely not a pdf"), max_chars=100)


def test_unknown_extension_accepted_when_it_decodes(tmp_path: Path) -> None:
    text, kind = extract_text(_write(tmp_path, "notes.weird", "just some words"), max_chars=100)
    assert (text, kind) == ("just some words", "text")


def test_unknown_binary_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ContextError):
        extract_text(_write(tmp_path, "blob.bin", bytes(range(256)) * 8), max_chars=100)


def test_unknown_with_many_replacement_chars_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ContextError):
        extract_text(_write(tmp_path, "x.dat", b"ok" + b"\xc3\x28" * 30), max_chars=100)


def test_cut_at_max_chars_on_a_line_boundary(tmp_path: Path) -> None:
    content = "\n".join(f"line number {i:03d}" for i in range(100))
    text, _ = extract_text(_write(tmp_path, "a.txt", content), max_chars=100)
    assert len(text) <= 100
    assert text.endswith("line number 005") or text.split("\n")[-1].startswith("line number")
    assert all(ln.startswith("line number ") and len(ln) == 15 for ln in text.split("\n"))


def test_missing_file_is_a_context_error(tmp_path: Path) -> None:
    with pytest.raises(ContextError):
        extract_text(tmp_path / "nope.md", max_chars=100)


def test_utf16_with_bom_is_decoded(tmp_path: Path) -> None:
    p = _write(tmp_path, "u.txt", "héllo wörld".encode("utf-16"))
    text, _ = extract_text(p, max_chars=100)
    assert text == "héllo wörld"


# ---- chunk_text ---------------------------------------------------------------

def _chunks(text: str, max_chars: int = 200) -> list[Chunk]:
    return chunk_text(text, max_chars=max_chars, doc_id="d1", doc_name="Doc")


def test_blank_text_yields_no_chunks() -> None:
    assert _chunks("") == []
    assert _chunks("   \n\n  ") == []


def test_chunks_split_on_headings_with_heading_path_titles() -> None:
    text = "intro line\n\n# Architecture\nbus text\n\n## Threads\nthread text\n\n## Locks\nlock text\n\n# Setup\nsetup text"
    chunks = _chunks(text)
    assert [c.title for c in chunks] == ["Doc", "Architecture", "Architecture > Threads", "Architecture > Locks", "Setup"]
    assert [c.text for c in chunks] == ["intro line", "bus text", "thread text", "lock text", "setup text"]
    assert [c.index for c in chunks] == [0, 1, 2, 3, 4]
    assert all(c.doc_id == "d1" and c.doc_name == "Doc" for c in chunks)


def test_paragraphs_under_one_heading_are_packed_up_to_max_chars() -> None:
    text = "# H\n" + "\n\n".join(["a" * 40] * 5)
    chunks = _chunks(text, max_chars=100)
    assert [len(c.text) for c in chunks] == [40 + 2 + 40, 40 + 2 + 40, 40]
    assert all(c.title == "H" for c in chunks)


def test_a_chunk_never_spans_two_headings() -> None:
    chunks = _chunks("# A\nx\n# B\ny", max_chars=1000)
    assert [(c.title, c.text) for c in chunks] == [("A", "x"), ("B", "y")]


def test_long_paragraph_is_split_on_sentence_boundaries() -> None:
    sentence = "This is a sentence of moderate length."
    text = " ".join([sentence] * 10)
    chunks = _chunks(text, max_chars=100)
    assert len(chunks) > 3
    assert all(len(c.text) <= 100 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)
    assert " ".join(c.text.replace("\n", " ") for c in chunks) == text


def test_long_paragraph_of_lines_splits_on_line_boundaries() -> None:
    text = "\n".join(f"line {i:02d} " + "x" * 20 for i in range(12))
    chunks = _chunks(text, max_chars=80)
    assert all(len(c.text) <= 80 for c in chunks)
    assert "\n".join(c.text for c in chunks) == text


def test_a_single_huge_token_is_hard_split() -> None:
    chunks = _chunks("z" * 450, max_chars=100)
    assert [len(c.text) for c in chunks] == [100, 100, 100, 100, 50]


def test_fenced_code_is_not_split_on_blank_lines_or_hashes() -> None:
    text = "# Code\n```py\n# a comment, not a heading\n\nx = 1\n```\n"
    chunks = _chunks(text, max_chars=500)
    assert len(chunks) == 1
    assert chunks[0].title == "Code"
    assert "# a comment, not a heading" in chunks[0].text and "x = 1" in chunks[0].text


def test_heading_only_document_still_yields_a_chunk() -> None:
    chunks = _chunks("# Just a title")
    assert len(chunks) == 1 and chunks[0].text == "Just a title"


def test_no_chunk_is_ever_empty() -> None:
    text = "# A\n\n\n\n## B\n\n\n# C\nbody\n\n\n\n"
    assert all(c.text.strip() for c in _chunks(text))


# ---- ContextDocument ----------------------------------------------------------

def _doc(**over) -> ContextDocument:
    base = dict(id="abc123abc123", name="n.md", kind="markdown", path="/x/abc.md", source_path="/src/n.md", chars=5)
    base.update(over)
    return ContextDocument(**base)


def test_document_to_dict_from_dict_round_trip() -> None:
    d = _doc(enabled=False, mode="inject", digest="dd", digest_state="ready", added_at="2026-10-06T10:00:00+00:00", sha256="ff")
    assert ContextDocument.from_dict(d.to_dict()) == d


def test_from_dict_ignores_unknown_keys_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    raw = _doc().to_dict()
    raw["future_field"] = 1
    with caplog.at_level("WARNING"):
        doc = ContextDocument.from_dict(raw)
    assert doc == _doc()
    assert "future_field" in caplog.text


def test_from_dict_requires_id_name_path() -> None:
    for missing in ("id", "name", "path"):
        raw = _doc().to_dict()
        del raw[missing]
        with pytest.raises(ContextError):
            ContextDocument.from_dict(raw)


def test_from_dict_defaults_optional_fields() -> None:
    doc = ContextDocument.from_dict({"id": "i", "name": "n", "path": "p"})
    assert doc.enabled is True and doc.mode == "reference" and doc.digest_state == "n/a" and doc.kind == "text"


# ---- ContextStore (in memory) -------------------------------------------------

def _add(store: ContextStore, name: str = "A", text: str = "alpha text", **over):
    kw = dict(mode="reference", max_chars=10000, digest_wanted=True)
    kw.update(over)
    return store.add_text(name, text, **kw)


def test_add_text_id_is_sha_prefix_and_defaults() -> None:
    import hashlib
    store = ContextStore(None)
    doc = _add(store, "Notes", "hello there")
    assert doc.id == hashlib.sha256(b"hello there").hexdigest()[:12]
    assert doc.sha256 == hashlib.sha256(b"hello there").hexdigest()
    assert (doc.name, doc.kind, doc.chars, doc.enabled, doc.mode) == ("Notes", "markdown", 11, True, "reference")
    assert doc.digest_state == "pending" and doc.added_at
    assert store.text(doc.id) == "hello there"
    assert doc.kind in DOCUMENT_KINDS


def test_digest_state_na_for_inject_or_digest_off() -> None:
    store = ContextStore(None)
    assert _add(store, "A", "one", mode="inject").digest_state == "n/a"
    assert _add(store, "B", "two", digest_wanted=False).digest_state == "n/a"
    assert _add(store, "C", "three").digest_state == "pending"


def test_same_text_readd_keeps_id_and_state() -> None:
    store = ContextStore(None)
    first = _add(store, "A", "same text", mode="inject")
    store.set_enabled(first.id, False)
    again = _add(store, "Renamed", "same text", mode="reference")
    assert again.id == first.id
    assert len(store.documents()) == 1
    assert again.name == "Renamed"
    assert again.enabled is False and again.mode == "inject"   # kept from the existing entry


def test_readd_keeps_a_finished_digest() -> None:
    store = ContextStore(None)
    d = _add(store, "A", "same text")
    store.set_digest(d.id, "a digest", "ready")
    again = _add(store, "A", "same text")
    assert again.digest == "a digest" and again.digest_state == "ready"


def test_documents_in_added_order_and_are_copies() -> None:
    store = ContextStore(None)
    ids = [_add(store, n, t).id for n, t in (("A", "aa"), ("B", "bb"), ("C", "cc"))]
    docs = store.documents()
    assert [d.id for d in docs] == ids
    docs[0].enabled = False
    assert store.get(ids[0]).enabled is True


def test_add_text_rejects_empty_bad_mode_bad_kind() -> None:
    store = ContextStore(None)
    with pytest.raises(ContextError):
        _add(store, "A", "   \n  ")
    with pytest.raises(ContextError):
        _add(store, "A", "x", mode="sideways")
    with pytest.raises(ContextError):
        _add(store, "A", "x", kind="zip")
    assert store.documents() == []


def test_add_text_cuts_at_max_chars() -> None:
    store = ContextStore(None)
    doc = _add(store, "A", "\n".join(["x" * 9] * 30), max_chars=50)
    assert doc.chars <= 50
    assert store.text(doc.id) == "\n".join(["x" * 9] * 5)


def test_remove_set_enabled_set_mode_set_digest() -> None:
    store = ContextStore(None)
    d = _add(store, "A", "aa")
    v0 = store.version
    assert store.set_enabled(d.id, False) and store.get(d.id).enabled is False
    assert store.version > v0
    assert store.set_mode(d.id, "inject", digest_wanted=True) and store.get(d.id).digest_state == "n/a"
    assert store.set_mode(d.id, "reference", digest_wanted=True) and store.get(d.id).digest_state == "pending"
    assert store.set_digest(d.id, "  the digest ", "ready")
    assert store.get(d.id).digest == "the digest" and store.get(d.id).digest_state == "ready"
    # a failed re-digest keeps the good digest it had (review, 2026-10-06)
    assert store.set_digest(d.id, "", "failed") and store.get(d.id).digest_state == "ready"
    assert store.get(d.id).digest == "the digest"
    assert store.remove(d.id) and store.get(d.id) is None and store.text(d.id) == ""


def test_changes_to_unknown_ids_return_false() -> None:
    store = ContextStore(None)
    assert not store.remove("nope")
    assert not store.set_enabled("nope", True)
    assert not store.set_mode("nope", "inject", digest_wanted=True)
    assert not store.set_digest("nope", "x", "ready")
    assert not store.mark_digest_pending("nope")


def test_set_mode_rejects_unknown_mode_and_bad_digest_state() -> None:
    store = ContextStore(None)
    d = _add(store)
    assert store.set_mode(d.id, "weird", digest_wanted=True) is False
    assert store.set_digest(d.id, "x", "pending") is False


def test_back_to_reference_with_kept_digest_is_ready_again() -> None:
    store = ContextStore(None)
    d = _add(store)
    store.set_digest(d.id, "kept", "ready")
    store.set_mode(d.id, "inject", digest_wanted=True)
    store.set_mode(d.id, "reference", digest_wanted=True)
    assert store.get(d.id).digest_state == "ready" and store.get(d.id).digest == "kept"


def test_back_to_reference_without_digest_wanted_stays_na() -> None:
    store = ContextStore(None)
    d = _add(store, mode="inject", digest_wanted=False)
    store.set_mode(d.id, "reference", digest_wanted=False)
    assert store.get(d.id).digest_state == "n/a"


def test_pending_digests_and_mark_pending() -> None:
    store = ContextStore(None)
    a = _add(store, "A", "aa")
    b = _add(store, "B", "bb", mode="inject")
    c = _add(store, "C", "cc")
    store.set_enabled(c.id, False)
    assert store.pending_digests() == [a.id, c.id]      # disabled ones count too
    store.set_digest(a.id, "d", "ready")
    assert store.pending_digests() == [c.id]
    assert store.mark_digest_pending(b.id) and store.pending_digests() == [b.id, c.id]


def test_enabled_filters_by_flag_and_mode() -> None:
    store = ContextStore(None)
    a = _add(store, "A", "aa", mode="inject")
    b = _add(store, "B", "bb")
    c = _add(store, "C", "cc")
    store.set_enabled(c.id, False)
    assert [d.id for d in store.enabled()] == [a.id, b.id]
    assert [d.id for d in store.enabled("inject")] == [a.id]
    assert [d.id for d in store.enabled("reference")] == [b.id]


def test_version_bumps_on_every_change() -> None:
    store = ContextStore(None)
    seen = [store.version]
    d = _add(store)
    seen.append(store.version)
    store.set_enabled(d.id, False)
    seen.append(store.version)
    store.set_digest(d.id, "x", "ready")
    seen.append(store.version)
    store.remove(d.id)
    seen.append(store.version)
    assert seen == sorted(set(seen))


def test_store_is_thread_safe_under_concurrent_adds() -> None:
    store = ContextStore(None)
    errors: list[Exception] = []

    def work(n: int) -> None:
        try:
            for i in range(20):
                _add(store, f"d{n}-{i}", f"text {n} {i}")
                store.documents()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(store.documents()) == 80


# ---- ContextStore (on disk) ---------------------------------------------------

def test_directory_is_not_created_at_construction(tmp_path: Path) -> None:
    target = tmp_path / "lib"
    ContextStore(target)
    assert not target.exists()


def test_manifest_round_trip_and_stored_copy(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    store = ContextStore(lib)
    src = _write(tmp_path, "README.md", "# Readme\nSome body text.\n")
    d = store.add_file(src, mode="inject", max_chars=10000, digest_wanted=True)
    t = store.add_text("pasted", "plain pasted words", mode="reference", max_chars=10000, digest_wanted=True, kind="text")
    store.set_digest(t.id, "digest of pasted", "ready")

    manifest = json.loads((lib / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 1 and len(manifest["documents"]) == 2
    assert (lib / "docs" / f"{d.id}.md").read_text(encoding="utf-8") == "# Readme\nSome body text."
    assert (lib / "docs" / f"{t.id}.txt").exists()
    assert d.source_path == str(src) and t.source_path == ""

    reopened = ContextStore(lib)
    assert [x.to_dict() for x in reopened.documents()] == [x.to_dict() for x in store.documents()]
    assert reopened.text(d.id) == "# Readme\nSome body text."
    assert reopened.get(t.id).digest == "digest of pasted"


def test_remove_deletes_stored_copy_and_updates_manifest(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    store = ContextStore(lib)
    d = store.add_text("A", "aaa", mode="inject", max_chars=100, digest_wanted=False)
    stored = lib / "docs" / f"{d.id}.md"
    assert stored.exists()
    assert store.remove(d.id)
    assert not stored.exists()
    assert json.loads((lib / "manifest.json").read_text(encoding="utf-8"))["documents"] == []


def test_no_temp_files_left_behind(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    store = ContextStore(lib)
    store.add_text("A", "aaa", mode="inject", max_chars=100, digest_wanted=False)
    assert not list(lib.rglob("*.tmp"))


def test_corrupt_manifest_gives_empty_store_without_raising(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "manifest.json").write_text("{ this is not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        store = ContextStore(lib)
    assert store.documents() == []
    assert "empty library" in caplog.text
    d = store.add_text("A", "aaa", mode="inject", max_chars=100, digest_wanted=False)   # still usable
    assert ContextStore(lib).get(d.id) is not None


def test_manifest_with_wrong_shape_is_treated_as_corrupt(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "manifest.json").write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    assert ContextStore(lib).documents() == []


def test_bad_manifest_entries_are_skipped_others_kept(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    good = _doc().to_dict()
    (lib / "manifest.json").write_text(json.dumps({"version": 1, "documents": [good, {"name": "no id"}, 7]}), encoding="utf-8")
    store = ContextStore(lib)
    assert [d.id for d in store.documents()] == ["abc123abc123"]


def test_text_falls_back_to_docs_dir_when_the_library_moved(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    store = ContextStore(lib)
    d = store.add_text("A", "moved text", mode="inject", max_chars=100, digest_wanted=False)
    moved = tmp_path / "moved"
    lib.rename(moved)
    assert ContextStore(moved).text(d.id) == "moved text"


def test_missing_stored_copy_gives_empty_text_not_an_error(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    store = ContextStore(lib)
    d = store.add_text("A", "gone soon", mode="inject", max_chars=100, digest_wanted=False)
    (lib / "docs" / f"{d.id}.md").unlink()
    assert ContextStore(lib).text(d.id) == ""


def test_reload_picks_up_external_manifest_edits(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    a = ContextStore(lib)
    d = a.add_text("A", "aaa", mode="inject", max_chars=100, digest_wanted=False)
    b = ContextStore(lib)
    b.set_enabled(d.id, False)
    assert a.get(d.id).enabled is True
    v = a.version
    a.reload()
    assert a.get(d.id).enabled is False and a.version > v


def test_add_file_missing_path_is_a_context_error(tmp_path: Path) -> None:
    store = ContextStore(None)
    with pytest.raises(ContextError):
        store.add_file(tmp_path / "ghost.md", mode="inject", max_chars=100, digest_wanted=False)


def test_add_file_of_empty_file_is_a_context_error(tmp_path: Path) -> None:
    store = ContextStore(None)
    with pytest.raises(ContextError):
        store.add_file(_write(tmp_path, "e.md", "\n\n"), mode="inject", max_chars=100, digest_wanted=False)


def test_add_file_code_keeps_fenced_text(tmp_path: Path) -> None:
    store = ContextStore(None)
    d = store.add_file(_write(tmp_path, "x.py", "print(1)\n"), mode="inject", max_chars=100, digest_wanted=False)
    assert d.kind == "code" and store.text(d.id) == "```py\nprint(1)\n```"


def test_a_failed_redigest_keeps_the_previous_digest() -> None:
    """Review 2026-10-06: Re-digest with the model down used to wipe the good digest."""
    store = ContextStore(None)
    doc = store.add_text("a.md", "alpha body text", mode="reference", max_chars=10000, digest_wanted=True)
    assert store.set_digest(doc.id, "A fine digest", "ready")
    assert store.mark_digest_pending(doc.id)
    assert store.set_digest(doc.id, "", "failed")
    got = store.get(doc.id)
    assert got is not None and got.digest == "A fine digest" and got.digest_state == "ready"
    # a document that never had one is marked failed with no text
    doc2 = store.add_text("b.md", "beta body text", mode="reference", max_chars=10000, digest_wanted=True)
    assert store.set_digest(doc2.id, "", "failed")
    got2 = store.get(doc2.id)
    assert got2 is not None and got2.digest == "" and got2.digest_state == "failed"


def test_malformed_office_files_raise_context_error_not_zlib_or_runtime_errors(tmp_path: Path) -> None:
    """Review 2026-10-06: a corrupt deflate stream raised zlib.error, an encrypted member RuntimeError."""
    import struct

    docx = tmp_path / "broken.docx"
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")
    data = bytearray(docx.read_bytes())
    # flip bytes inside the member's compressed data (after the local header + name)
    for i in range(40, 46):
        data[i] ^= 0xFF
    docx.write_bytes(bytes(data))
    with pytest.raises(ContextError):
        extract_text(docx, max_chars=10000)
    enc = tmp_path / "enc.docx"
    with zipfile.ZipFile(enc, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")
    raw = bytearray(enc.read_bytes())
    # set the "encrypted" general-purpose bit in the local file header (offset 6) and the central directory
    raw[6] |= 1
    cd = raw.find(b"PK\x01\x02")
    raw[cd + 8] |= 1
    enc.write_bytes(bytes(raw))
    with pytest.raises(ContextError):
        extract_text(enc, max_chars=10000)
    pptx = tmp_path / "deep.pptx"
    with zipfile.ZipFile(pptx, "w") as zf:
        zf.writestr("ppt/slides/slide1.xml", "<p:sld>" + "<a:p>" * 3000 + "</a:p>" * 3000 + "</p:sld>")
    try:
        extract_text(pptx, max_chars=10000)
    except ContextError:
        pass                                    # either a clean ContextError...
    # ...or a successful parse of a legal (if silly) file; never any other exception type
