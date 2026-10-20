"""Tests for exocortex.llm.library (all offline: synthetic GGUF headers on disk, httpx.MockTransport
for every Hugging Face call -- never a real network/GPU/LM Studio touch, per the house rules)."""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import time
from pathlib import Path

import httpx
import pytest

import exocortex.gpu as gpu_mod
from exocortex.config import LLMConfig
from exocortex.llm import server as server_mod
from exocortex.llm.library import (
    PROFILES_FILE,
    GgufMeta,
    HfFile,
    LibraryModel,
    ModelProfile,
    PullCancelled,
    PullError,
    QuantOption,
    auto_profile,
    choose_quant,
    delete_model,
    gguf_metadata,
    group_quants,
    hf_repo_info,
    hf_search,
    load_profiles,
    profile_for,
    pull_model,
    save_profiles,
    scan_library,
    set_profile,
)

# =====================================================================================
# A minimal, from-scratch GGUF header writer (independent of library.py's own parser --
# this is the GGUF wire format itself, not an implementation detail of the module under test).
# =====================================================================================

_T_UINT8, _T_INT8, _T_UINT16, _T_INT16 = 0, 1, 2, 3
_T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = 4, 5, 6, 7
_T_STRING, _T_ARRAY = 8, 9
_T_UINT64, _T_INT64, _T_FLOAT64 = 10, 11, 12

_STRUCT_FMT = {
    _T_UINT8: "<B", _T_INT8: "<b", _T_UINT16: "<H", _T_INT16: "<h",
    _T_UINT32: "<I", _T_INT32: "<i", _T_FLOAT32: "<f", _T_BOOL: "<B",
    _T_UINT64: "<Q", _T_INT64: "<q", _T_FLOAT64: "<d",
}


def _pack_gguf_string(s: str) -> bytes:
    data = s.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _pack_gguf_value(vtype: int, value: object) -> bytes:
    """`value` is a scalar/str for scalar/string types, or `(elem_type, items)` for an array."""
    if vtype == _T_STRING:
        return _pack_gguf_string(value)  # type: ignore[arg-type]
    if vtype == _T_ARRAY:
        elem_type, items = value  # type: ignore[misc]
        out = struct.pack("<I", elem_type) + struct.pack("<Q", len(items))
        for item in items:
            out += _pack_gguf_value(elem_type, item)
        return out
    return struct.pack(_STRUCT_FMT[vtype], value)


def _build_gguf_header(kvs: list[tuple[str, int, object]], *, version: int = 3, n_tensors: int = 0,
                        magic: bytes = b"GGUF") -> bytes:
    out = magic + struct.pack("<I", version) + struct.pack("<Q", n_tensors) + struct.pack("<Q", len(kvs))
    for key, vtype, value in kvs:
        out += _pack_gguf_string(key) + struct.pack("<I", vtype) + _pack_gguf_value(vtype, value)
    return out


@pytest.fixture(autouse=True)
def _no_real_lmstudio_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every test in this file is offline: scan_library()/pull_model() always probe LM Studio's
    folder too (exocortex.llm.library.library_roots), so without this a test that forgets its own
    monkeypatch would silently fall through to whatever real ``D:\\AI\\models``-style folder exists
    on the machine running the suite -- slow, non-deterministic, and exactly what the house rules
    forbid touching. A test that wants a specific (fake) LM Studio folder still overrides this with
    its own ``monkeypatch.setattr(server_mod, "lmstudio_models_dir", ...)`` afterwards."""
    monkeypatch.setattr(server_mod, "lmstudio_models_dir", lambda cfg=None: tmp_path / "no-real-lmstudio-dir")


# ---- gguf_metadata ------------------------------------------------------------------------

def test_gguf_metadata_round_trip_reads_header_and_skips_50k_string_array_cheaply(tmp_path: Path) -> None:
    vocab = [f"tok{i}" for i in range(50_000)]
    kvs: list[tuple[str, int, object]] = [
        ("general.architecture", _T_STRING, "testarch"),
        ("general.name", _T_STRING, "Synthetic Test Model"),
        ("general.file_type", _T_UINT32, 2),
        ("general.size_label", _T_STRING, "12B"),
        ("testarch.block_count", _T_UINT32, 48),
        ("testarch.context_length", _T_UINT32, 8192),
        ("testarch.embedding_length", _T_UINT32, 4096),
        ("testarch.attention.head_count", _T_UINT32, 32),
        ("testarch.attention.head_count_kv", _T_ARRAY, (_T_UINT32, [8] * 48)),
        ("testarch.attention.key_length", _T_UINT32, 128),
        ("tokenizer.ggml.tokens", _T_ARRAY, (_T_STRING, vocab)),  # must be skipped, not materialised
    ]
    path = tmp_path / "synthetic-test-model.gguf"
    path.write_bytes(_build_gguf_header(kvs))

    start = time.perf_counter()
    meta = gguf_metadata(path)
    elapsed = time.perf_counter() - start

    assert meta.architecture == "testarch"
    assert meta.n_layer == 48
    assert meta.n_ctx_train == 8192
    assert meta.n_embd == 4096
    assert meta.n_head == 32
    assert meta.n_head_kv is None                      # per-layer array, not a scalar
    assert meta.kv_heads_total == 8 * 48
    assert meta.head_dim == 128                          # explicit key_length wins over n_embd // n_head
    assert meta.params == 12_000_000_000
    assert meta.name == "Synthetic Test Model"
    assert meta.file_type == 2
    assert elapsed < 1.0, "a 50k-entry string array must be skipped cheaply, not materialised"


def test_gguf_metadata_falls_back_to_file_type_when_filename_has_no_quant_token(tmp_path: Path) -> None:
    kvs = [
        ("general.architecture", _T_STRING, "testarch"),
        ("general.file_type", _T_UINT32, 8),  # Q5_0 per the ggml_ftype table
    ]
    path = tmp_path / "modelwithnoquanttokeninitsname.gguf"
    path.write_bytes(_build_gguf_header(kvs))
    assert gguf_metadata(path).quant == "Q5_0"


def test_gguf_metadata_prefers_filename_quant_token_over_file_type(tmp_path: Path) -> None:
    kvs = [("general.architecture", _T_STRING, "testarch"), ("general.file_type", _T_UINT32, 8)]  # Q5_0
    path = tmp_path / "model-Q4_K_M.gguf"
    path.write_bytes(_build_gguf_header(kvs))
    assert gguf_metadata(path).quant == "Q4_K_M"


def test_gguf_metadata_rejects_non_gguf_file(tmp_path: Path) -> None:
    path = tmp_path / "not-a-model.gguf"
    path.write_bytes(b"NOTG" + b"\x00" * 64)
    with pytest.raises(ValueError, match="not a GGUF file"):
        gguf_metadata(path)


def test_gguf_metadata_rejects_unsupported_version(tmp_path: Path) -> None:
    path = tmp_path / "future-version.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 99) + struct.pack("<Q", 0) + struct.pack("<Q", 0))
    with pytest.raises(ValueError, match="version"):
        gguf_metadata(path)


# =====================================================================================
# scan_library
# =====================================================================================

def test_scan_library_groups_shards_attaches_mmproj_skips_ancillary_and_lists_lmstudio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    own_dir = tmp_path / "own"
    lmstudio_dir = tmp_path / "lmstudio"

    repo_a = own_dir / "PublisherA" / "RepoA-GGUF"
    repo_a.mkdir(parents=True)
    (repo_a / "RepoA-Q4_K_M-00001-of-00002.gguf").write_bytes(b"a" * 1000)
    (repo_a / "RepoA-Q4_K_M-00002-of-00002.gguf").write_bytes(b"b" * 500)
    (repo_a / "mmproj-RepoA-BF16.gguf").write_bytes(b"m" * 100)

    repo_b = own_dir / "PublisherB" / "RepoB-GGUF"
    repo_b.mkdir(parents=True)
    (repo_b / "RepoB-Q8_0.gguf").write_bytes(b"c" * 300)
    (repo_b / "mtp-RepoB-head.gguf").write_bytes(b"d" * 50)          # must never be listed
    (repo_b / "RepoB-imatrix.gguf").write_bytes(b"e" * 50)           # must never be listed

    repo_c = lmstudio_dir / "PublisherC" / "RepoC-GGUF"
    repo_c.mkdir(parents=True)
    (repo_c / "RepoC-Q4_0.gguf").write_bytes(b"f" * 200)

    cfg = LLMConfig(library_dir=str(own_dir))
    monkeypatch.setattr(server_mod, "lmstudio_models_dir", lambda cfg=None: lmstudio_dir)

    models = scan_library(cfg, with_meta=False)
    by_key = {m.key: m for m in models}

    assert set(by_key) == {"PublisherA/RepoA", "PublisherB/RepoB", "PublisherC/RepoC"}

    a = by_key["PublisherA/RepoA"]
    assert a.source == "library"
    assert [p.name for p in a.shards] == ["RepoA-Q4_K_M-00001-of-00002.gguf", "RepoA-Q4_K_M-00002-of-00002.gguf"]
    assert a.size_bytes == 1500
    assert a.mmproj == repo_a / "mmproj-RepoA-BF16.gguf"

    b = by_key["PublisherB/RepoB"]
    assert b.source == "library"
    assert len(b.shards) == 1
    assert b.mmproj is None

    c = by_key["PublisherC/RepoC"]
    assert c.source == "lmstudio"
    assert c.mmproj is None

    assert len(models) == 3   # mtp-/imatrix files never turn into their own listed models


def test_scan_library_dedups_by_resolved_path_and_own_library_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_dir = tmp_path / "shared"
    (shared_dir / "Pub" / "Repo-GGUF").mkdir(parents=True)
    (shared_dir / "Pub" / "Repo-GGUF" / "Repo-Q4_K_M.gguf").write_bytes(b"x" * 10)

    cfg = LLMConfig(library_dir=str(shared_dir))
    # The (unusual, but possible) config where LM Studio's folder resolves to the SAME directory
    # as this app's own library -- the same file must be listed exactly once, as "library".
    monkeypatch.setattr(server_mod, "lmstudio_models_dir", lambda cfg=None: shared_dir)

    models = scan_library(cfg, with_meta=False)
    assert len(models) == 1
    assert models[0].source == "library"


def test_scan_library_missing_roots_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = LLMConfig(library_dir=str(tmp_path / "does-not-exist"))
    monkeypatch.setattr(server_mod, "lmstudio_models_dir", lambda cfg=None: tmp_path / "also-missing")
    assert scan_library(cfg, with_meta=False) == []


# =====================================================================================
# group_quants (shaped like unsloth/gemma-4-12b-it-GGUF)
# =====================================================================================

def test_group_quants_unsloth_shaped_repo() -> None:
    quant_names = [
        "BF16", "F16", "F32", "Q2_K", "Q2_K_S", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1",
        "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "IQ4_XS", "IQ4_NL",
    ]
    files = [HfFile(path=f"gemma-4-12b-it-{q}.gguf", size=1_000_000, sha256="") for q in quant_names]
    # a 21st quant, sharded in two parts -- must group into ONE QuantOption with summed size
    files += [
        HfFile(path="gemma-4-12b-it-UD-Q4_K_XL-00001-of-00002.gguf", size=2_000_000, sha256=""),
        HfFile(path="gemma-4-12b-it-UD-Q4_K_XL-00002-of-00002.gguf", size=1_500_000, sha256=""),
    ]
    # never grouped: a subfolder, 3 mmproj files, and an imatrix file
    files.append(HfFile(path="MTP/gemma-4-12b-it-MTP-Q4_K_M.gguf", size=999, sha256=""))
    files += [HfFile(path=f"mmproj-gemma-4-12b-it-{q}.gguf", size=500, sha256="") for q in ("F16", "Q8_0", "BF16")]
    files.append(HfFile(path="gemma-4-12b-it.imatrix.gguf", size=100, sha256=""))

    options = group_quants(files)

    assert len(options) == 21
    by_quant = {o.quant: o for o in options}
    assert set(by_quant) == set(quant_names) | {"UD-Q4_K_XL"}

    xl = by_quant["UD-Q4_K_XL"]
    assert len(xl.files) == 2
    assert xl.size_bytes == 3_500_000
    assert [f.path for f in xl.files] == [
        "gemma-4-12b-it-UD-Q4_K_XL-00001-of-00002.gguf",
        "gemma-4-12b-it-UD-Q4_K_XL-00002-of-00002.gguf",
    ]
    for q in quant_names:
        assert by_quant[q].size_bytes == 1_000_000
        assert len(by_quant[q].files) == 1


# =====================================================================================
# choose_quant
# =====================================================================================

_QUANT_SIZES = {
    "Q8_0": 12_750_000_000, "Q6_K": 9_900_000_000, "Q5_K_M": 8_550_000_000, "Q4_K_M": 7_200_000_000,
    "Q4_0": 6_750_000_000, "Q3_K_M": 5_850_000_000, "Q2_K": 4_200_000_000, "IQ2_M": 2_500_000_000,
}


def _quant_options(sizes: dict[str, int]) -> tuple[QuantOption, ...]:
    return tuple(
        QuantOption(quant=q, files=(HfFile(path=f"{q}.gguf", size=sz, sha256=""),), size_bytes=sz)
        for q, sz in sizes.items()
    )


@pytest.mark.parametrize("budget_mb,policy,expected", [
    (16003, "auto", "Q4_K_M"),      # first of _AUTO_PREF present that fits 75% of budget
    (16003, "quality", "Q6_K"),     # LARGEST fitting quant in _QUALITY_PREF (Q8_0 does not fit)
    (16003, "fit", "Q6_K"),         # largest non-float quant that fits at all
    (8192, "auto", "Q3_K_M"),       # Q4_K_M/Q4_0 no longer fit at this smaller budget
    (8192, "quality", "Q3_K_M"),
    (8192, "fit", "Q3_K_M"),
    (4096, "auto", "IQ2_M"),        # only the smallest quant fits this tight a budget
    (4096, "quality", "IQ2_M"),
    (4096, "fit", "IQ2_M"),
    (None, "auto", "Q4_K_M"),       # no GPU budget: first preference-list quant present wins outright
    (None, "quality", "Q8_0"),
    (None, "fit", "Q8_0"),          # "fit" with no budget: the largest quant overall, regardless of order
])
def test_choose_quant_policies_across_budgets(budget_mb: "int | None", policy: str, expected: str) -> None:
    options = _quant_options(_QUANT_SIZES)
    chosen = choose_quant(options, budget_mb=budget_mb, policy=policy)
    assert chosen is not None and chosen.quant == expected


@pytest.mark.parametrize("budget_mb", [16003, 8192, 4096, None])
def test_choose_quant_named_policy_ignores_budget(budget_mb: "int | None") -> None:
    options = _quant_options(_QUANT_SIZES)
    chosen = choose_quant(options, budget_mb=budget_mb, policy="Q4_0")
    assert chosen is not None and chosen.quant == "Q4_0"


def test_choose_quant_unknown_named_policy_returns_none() -> None:
    options = _quant_options(_QUANT_SIZES)
    assert choose_quant(options, budget_mb=16003, policy="NOT_A_REAL_QUANT") is None


@pytest.mark.parametrize("policy", ["auto", "quality", "fit", "Q4_0"])
@pytest.mark.parametrize("budget_mb", [16003, None])
def test_choose_quant_single_qat_quant_repo(budget_mb: "int | None", policy: str) -> None:
    """A repo with exactly one (QAT) quant must be returned by every policy, whenever it fits."""
    qat = (QuantOption(quant="Q4_0", files=(HfFile(path="m-Q4_0.gguf", size=6_750_000_000, sha256=""),),
                        size_bytes=6_750_000_000),)
    chosen = choose_quant(qat, budget_mb=budget_mb, policy=policy)
    assert chosen is not None and chosen.quant == "Q4_0"


# =====================================================================================
# hf_search / hf_repo_info (httpx.MockTransport, no real network)
# =====================================================================================

def test_hf_search_sends_expected_query_params_and_parses_results() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json=[
            {"id": "org/modelA", "downloads": 100, "likes": 5, "lastModified": "2026-01-01",
             "gated": False, "private": False},
            {"modelId": "org/modelB", "downloads": 50, "likes": 1, "lastModified": "2026-02-02",
             "gated": True, "private": True},
        ])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    repos = hf_search("gemma", limit=10, http=client)

    assert captured["params"] == {
        "search": "gemma", "filter": "gguf", "sort": "downloads", "direction": "-1", "limit": "10",
    }
    assert len(repos) == 2
    assert repos[0].id == "org/modelA" and repos[0].downloads == 100 and not repos[0].gated
    assert repos[1].id == "org/modelB" and repos[1].gated is True and repos[1].private is True


def _repo_data_json(repo_id: str, **gguf_overrides: object) -> dict:
    gguf = {"architecture": "gemma4", "context_length": 8192, "total": 12_000_000_000}
    gguf.update(gguf_overrides)
    return {"id": repo_id, "sha": "abc123", "gguf": gguf}


def test_hf_repo_info_parses_metadata_and_tree_with_sha256_from_lfs_oid() -> None:
    repo_id = "unsloth/gemma-4-12b-it-GGUF"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/tree/main" in url:
            return httpx.Response(200, json=[
                {"type": "file", "path": "gemma-4-12b-it-Q4_K_M.gguf", "size": 7_000_000_000,
                 "lfs": {"oid": "deadbeef" * 8}},
                {"type": "file", "path": "README.md", "size": 500},  # no lfs -> sha256 ""
                {"type": "file", "path": "mmproj-gemma-4-12b-it-F16.gguf", "size": 900_000_000,
                 "lfs": {"oid": "cafebabe" * 8}},
                {"type": "directory", "path": "MTP"},
            ])
        if url.endswith(f"/api/models/{repo_id}"):
            return httpx.Response(200, json=_repo_data_json(repo_id))
        raise AssertionError(f"unexpected request: {url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    info = hf_repo_info(repo_id, http=client)

    assert info.id == repo_id
    assert info.sha == "abc123"
    assert info.architecture == "gemma4"
    assert info.params == 12_000_000_000
    assert info.context_length == 8192

    by_path = {f.path: f for f in info.files}
    assert by_path["gemma-4-12b-it-Q4_K_M.gguf"].sha256 == "deadbeef" * 8
    assert by_path["README.md"].sha256 == ""
    assert len(info.mmproj) == 1 and info.mmproj[0].path == "mmproj-gemma-4-12b-it-F16.gguf"
    assert len(info.quants) == 1 and info.quants[0].quant == "Q4_K_M"


def test_hf_repo_info_gated_and_private_flags_do_not_break_parsing() -> None:
    repo_id = "someone/gated-repo-GGUF"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/tree/main" in url:
            return httpx.Response(200, json=[
                {"type": "file", "path": "gated-repo-Q4_0.gguf", "size": 100, "lfs": {"oid": "1" * 64}},
            ])
        data = _repo_data_json(repo_id)
        data["gated"] = "auto"
        data["private"] = True
        return httpx.Response(200, json=data)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    info = hf_repo_info(repo_id, http=client)
    assert info.id == repo_id
    assert len(info.quants) == 1


# =====================================================================================
# pull_model (httpx.MockTransport, no real network)
# =====================================================================================

def _lfs(content: bytes) -> dict[str, str]:
    return {"oid": hashlib.sha256(content).hexdigest()}


def _dispatch_hf(repo_id: str, tree_entries: list[dict], resolve_handler):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if f"/api/models/{repo_id}/tree/main" in url:
            return httpx.Response(200, json=tree_entries)
        if url.endswith(f"/api/models/{repo_id}"):
            return httpx.Response(200, json=_repo_data_json(repo_id))
        if f"/{repo_id}/resolve/main/" in url:
            return resolve_handler(request)
        raise AssertionError(f"unexpected request: {url}")

    return handler


def test_pull_model_happy_path_writes_file_and_returns_library_model(tmp_path: Path) -> None:
    repo_id = "acme/tiny-model-GGUF"
    content = b"0123456789" * 30_000
    tree = [{"type": "file", "path": "tiny-model-Q4_K_M.gguf", "size": len(content), "lfs": _lfs(content)}]

    def resolve(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, headers={"content-length": str(len(content))})

    client = httpx.Client(transport=httpx.MockTransport(_dispatch_hf(repo_id, tree, resolve)),
                           follow_redirects=True)
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))

    model = pull_model(repo_id, "Q4_K_M", cfg, http=client)

    dest = tmp_path / "lib" / "acme" / "tiny-model-GGUF" / "tiny-model-Q4_K_M.gguf"
    assert dest.read_bytes() == content
    assert model.key == "acme/tiny-model"
    assert model.source == "library"
    assert not dest.with_name(dest.name + ".part").exists()


def test_pull_model_follows_redirect_to_a_cdn_url(tmp_path: Path) -> None:
    repo_id = "acme/tiny-model-GGUF"
    content = b"abcdefghij" * 30_000
    tree = [{"type": "file", "path": "tiny-model-Q4_0.gguf", "size": len(content), "lfs": _lfs(content)}]
    cdn_url = "https://cdn.example.invalid/blob/tiny-model-Q4_0.gguf"
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen_urls.append(url)
        if f"/api/models/{repo_id}/tree/main" in url:
            return httpx.Response(200, json=tree)
        if url.endswith(f"/api/models/{repo_id}"):
            return httpx.Response(200, json=_repo_data_json(repo_id))
        if f"/{repo_id}/resolve/main/" in url:
            return httpx.Response(302, headers={"location": cdn_url})
        if url == cdn_url:
            return httpx.Response(200, content=content, headers={"content-length": str(len(content))})
        raise AssertionError(f"unexpected request: {url}")

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))

    model = pull_model(repo_id, "Q4_0", cfg, http=client)

    dest = tmp_path / "lib" / "acme" / "tiny-model-GGUF" / "tiny-model-Q4_0.gguf"
    assert dest.read_bytes() == content
    assert cdn_url in seen_urls


def test_pull_model_resumes_from_a_half_written_part_via_range_206(tmp_path: Path) -> None:
    repo_id = "acme/tiny-model-GGUF"
    content = b"0123456789" * 30_000
    tree = [{"type": "file", "path": "tiny-model-Q4_K_M.gguf", "size": len(content), "lfs": _lfs(content)}]
    dest_root = tmp_path / "lib" / "acme" / "tiny-model-GGUF"
    dest_root.mkdir(parents=True)
    split = len(content) // 2
    (dest_root / "tiny-model-Q4_K_M.gguf.part").write_bytes(content[:split])
    seen_ranges: list[str | None] = []

    def resolve(request: httpx.Request) -> httpx.Response:
        seen_ranges.append(request.headers.get("range"))
        return httpx.Response(206, content=content[split:])

    client = httpx.Client(transport=httpx.MockTransport(_dispatch_hf(repo_id, tree, resolve)),
                           follow_redirects=True)
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))

    pull_model(repo_id, "Q4_K_M", cfg, http=client)

    assert (dest_root / "tiny-model-Q4_K_M.gguf").read_bytes() == content
    assert seen_ranges == [f"bytes={split}-"]


def test_pull_model_200_answer_to_a_range_request_restarts_from_zero(tmp_path: Path) -> None:
    repo_id = "acme/tiny-model-GGUF"
    content = b"0123456789" * 30_000
    tree = [{"type": "file", "path": "tiny-model-Q4_K_M.gguf", "size": len(content), "lfs": _lfs(content)}]
    dest_root = tmp_path / "lib" / "acme" / "tiny-model-GGUF"
    dest_root.mkdir(parents=True)
    (dest_root / "tiny-model-Q4_K_M.gguf.part").write_bytes(b"garbage-not-a-prefix-of-content")

    def resolve(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("range")  # a Range WAS requested
        # ... but the server ignores it and answers with the whole file from byte 0.
        return httpx.Response(200, content=content, headers={"content-length": str(len(content))})

    client = httpx.Client(transport=httpx.MockTransport(_dispatch_hf(repo_id, tree, resolve)),
                           follow_redirects=True)
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))

    pull_model(repo_id, "Q4_K_M", cfg, http=client)

    assert (dest_root / "tiny-model-Q4_K_M.gguf").read_bytes() == content


def test_pull_model_sha256_mismatch_deletes_part_and_raises(tmp_path: Path) -> None:
    repo_id = "acme/tiny-model-GGUF"
    real_content = b"0123456789" * 30_000
    wrong_content = b"9876543210" * 30_000  # same size, different bytes -> sha256 mismatch
    tree = [{"type": "file", "path": "tiny-model-Q4_K_M.gguf", "size": len(real_content), "lfs": _lfs(real_content)}]

    def resolve(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=wrong_content, headers={"content-length": str(len(wrong_content))})

    client = httpx.Client(transport=httpx.MockTransport(_dispatch_hf(repo_id, tree, resolve)),
                           follow_redirects=True)
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))

    with pytest.raises(PullError, match="sha256 mismatch"):
        pull_model(repo_id, "Q4_K_M", cfg, http=client)

    dest_root = tmp_path / "lib" / "acme" / "tiny-model-GGUF"
    assert not (dest_root / "tiny-model-Q4_K_M.gguf").exists()
    assert not (dest_root / "tiny-model-Q4_K_M.gguf.part").exists()


def test_pull_model_cancelled_mid_download_raises_and_keeps_the_part(tmp_path: Path) -> None:
    repo_id = "acme/tiny-model-GGUF"
    content = b"0123456789" * 200_000  # several 64 KiB chunks
    tree = [{"type": "file", "path": "tiny-model-Q4_K_M.gguf", "size": len(content), "lfs": _lfs(content)}]

    def resolve(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, headers={"content-length": str(len(content))})

    client = httpx.Client(transport=httpx.MockTransport(_dispatch_hf(repo_id, tree, resolve)),
                           follow_redirects=True)
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))
    seen: list[int] = []

    def cancelled() -> bool:
        return len(seen) >= 2

    with pytest.raises(PullCancelled):
        pull_model(repo_id, "Q4_K_M", cfg, http=client,
                    progress=lambda name, done, total: seen.append(done), cancelled=cancelled)

    part = tmp_path / "lib" / "acme" / "tiny-model-GGUF" / "tiny-model-Q4_K_M.gguf.part"
    assert part.is_file()
    assert 0 < part.stat().st_size < len(content)


def test_pull_model_insufficient_disk_space_raises_naming_the_shortfall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_id = "acme/tiny-model-GGUF"
    content = b"0123456789" * 30_000  # 300,000 bytes needed
    tree = [{"type": "file", "path": "tiny-model-Q4_K_M.gguf", "size": len(content), "lfs": _lfs(content)}]

    def resolve(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be reached
        raise AssertionError("must not download when there isn't enough disk space")

    client = httpx.Client(transport=httpx.MockTransport(_dispatch_hf(repo_id, tree, resolve)),
                           follow_redirects=True)
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))

    class _TinyDisk:
        free = 100
        total = 1_000
        used = 900

    monkeypatch.setattr(shutil, "disk_usage", lambda path: _TinyDisk())

    with pytest.raises(PullError, match=r"not enough disk space.*need 0\.00 GB.*GB free"):
        pull_model(repo_id, "Q4_K_M", cfg, http=client)


def test_pull_model_include_mmproj_pulls_the_smallest_one(tmp_path: Path) -> None:
    repo_id = "acme/tiny-vision-GGUF"
    content = b"0123456789" * 30_000
    small_proj = b"m" * 10_000
    large_proj = b"M" * 50_000
    tree = [
        {"type": "file", "path": "tiny-vision-Q4_K_M.gguf", "size": len(content), "lfs": _lfs(content)},
        {"type": "file", "path": "mmproj-tiny-vision-F16.gguf", "size": len(large_proj), "lfs": _lfs(large_proj)},
        {"type": "file", "path": "mmproj-tiny-vision-Q8_0.gguf", "size": len(small_proj), "lfs": _lfs(small_proj)},
    ]
    served = {
        "tiny-vision-Q4_K_M.gguf": content,
        "mmproj-tiny-vision-F16.gguf": large_proj,
        "mmproj-tiny-vision-Q8_0.gguf": small_proj,
    }

    def resolve(request: httpx.Request) -> httpx.Response:
        name = str(request.url).rsplit("/", 1)[-1]
        data = served[name]
        return httpx.Response(200, content=data, headers={"content-length": str(len(data))})

    client = httpx.Client(transport=httpx.MockTransport(_dispatch_hf(repo_id, tree, resolve)),
                           follow_redirects=True)
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))

    model = pull_model(repo_id, "Q4_K_M", cfg, include_mmproj=True, http=client)

    dest_root = tmp_path / "lib" / "acme" / "tiny-vision-GGUF"
    assert (dest_root / "mmproj-tiny-vision-Q8_0.gguf").is_file()
    assert not (dest_root / "mmproj-tiny-vision-F16.gguf").exists()
    assert model.mmproj == dest_root / "mmproj-tiny-vision-Q8_0.gguf"


def test_pull_model_skips_an_already_completed_file_on_rerun(tmp_path: Path) -> None:
    repo_id = "acme/tiny-model-GGUF"
    content = b"0123456789" * 30_000
    tree = [{"type": "file", "path": "tiny-model-Q4_K_M.gguf", "size": len(content), "lfs": _lfs(content)}]
    resolve_calls: list[str] = []

    def resolve(request: httpx.Request) -> httpx.Response:
        resolve_calls.append(str(request.url))
        return httpx.Response(200, content=content, headers={"content-length": str(len(content))})

    client = httpx.Client(transport=httpx.MockTransport(_dispatch_hf(repo_id, tree, resolve)),
                           follow_redirects=True)
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))

    pull_model(repo_id, "Q4_K_M", cfg, http=client)
    assert len(resolve_calls) == 1

    pull_model(repo_id, "Q4_K_M", cfg, http=client)  # rerun: the file is already complete on disk
    assert len(resolve_calls) == 1, "a completed file must not be re-downloaded"


def test_pull_model_writes_an_auto_profile_when_budget_mb_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Hardware probing is guarded/lazy in library.py, but must still never touch a real GPU here.
    monkeypatch.setattr(gpu_mod, "gpu_vendor", lambda *a, **k: "cpu")
    monkeypatch.setattr(gpu_mod, "system_memory_mb", lambda: (16_000, 8_000))
    monkeypatch.setattr(server_mod, "_physical_cores", lambda: 8)

    repo_id = "acme/tiny-model-GGUF"
    content = b"0123456789" * 30_000
    tree = [{"type": "file", "path": "tiny-model-Q4_K_M.gguf", "size": len(content), "lfs": _lfs(content)}]

    def resolve(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, headers={"content-length": str(len(content))})

    client = httpx.Client(transport=httpx.MockTransport(_dispatch_hf(repo_id, tree, resolve)),
                           follow_redirects=True)
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))

    model = pull_model(repo_id, "Q4_K_M", cfg, budget_mb=8192, http=client)

    profiles = load_profiles(cfg)
    assert model.key in profiles
    assert profiles[model.key].auto is True


# =====================================================================================
# profiles.json (load_profiles / save_profiles / profile_for / set_profile)
# =====================================================================================

def test_profiles_round_trip_and_atomic_save_leaves_no_tmp_file(tmp_path: Path) -> None:
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))
    profiles = {
        "acme/tiny-model": ModelProfile(ctx_size=8192, gpu_layers=32, kv_cache_type="q8_0",
                                         flash_attn=True, threads=12, auto=False, note="hand-tuned",
                                         extra_args=["--foo", "bar"]),
        "acme/other-model": ModelProfile(auto=True, generated_from="test"),
    }

    path = save_profiles(cfg, profiles)

    assert path == tmp_path / "lib" / PROFILES_FILE
    assert not any(p.name.endswith(".tmp") or ".tmp." in p.name for p in path.parent.iterdir())
    assert load_profiles(cfg) == profiles


def test_load_profiles_tolerates_unknown_fields(tmp_path: Path) -> None:
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / PROFILES_FILE).write_text(json.dumps({
        "acme/tiny-model": {"ctx_size": 4096, "bogus_field_from_the_future": "??"},
    }), encoding="utf-8")

    loaded = load_profiles(cfg)
    assert loaded["acme/tiny-model"].ctx_size == 4096


def test_load_profiles_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))
    assert load_profiles(cfg) == {}


def test_load_profiles_unparseable_file_returns_empty_dict(tmp_path: Path) -> None:
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / PROFILES_FILE).write_text("{not valid json", encoding="utf-8")
    assert load_profiles(cfg) == {}


def test_set_profile_and_profile_for_round_trip(tmp_path: Path) -> None:
    cfg = LLMConfig(library_dir=str(tmp_path / "lib"))
    assert profile_for(cfg, "acme/tiny-model") is None

    set_profile(cfg, "acme/tiny-model", ModelProfile(ctx_size=2048))
    assert profile_for(cfg, "acme/tiny-model").ctx_size == 2048

    set_profile(cfg, "acme/tiny-model", None)
    assert profile_for(cfg, "acme/tiny-model") is None


# =====================================================================================
# delete_model
# =====================================================================================

def test_delete_model_removes_shards_mmproj_and_now_empty_dirs(tmp_path: Path) -> None:
    repo_dir = tmp_path / "lib" / "acme" / "tiny-model-GGUF"
    repo_dir.mkdir(parents=True)
    shard1 = repo_dir / "tiny-model-Q4_K_M-00001-of-00002.gguf"
    shard2 = repo_dir / "tiny-model-Q4_K_M-00002-of-00002.gguf"
    mmproj = repo_dir / "mmproj-tiny-model-BF16.gguf"
    for p in (shard1, shard2, mmproj):
        p.write_bytes(b"x")

    model = LibraryModel(key="acme/tiny-model", name="tiny-model", path=shard1, size_bytes=3,
                          source="library", publisher="acme", repo="tiny-model-GGUF", quant="Q4_K_M",
                          shards=(shard1, shard2), mmproj=mmproj, meta=None)

    removed = delete_model(model)

    assert not shard1.exists() and not shard2.exists() and not mmproj.exists()
    assert not repo_dir.exists()
    assert not (tmp_path / "lib" / "acme").exists()   # <publisher> dir removed too, now empty
    assert (tmp_path / "lib").exists()                 # but never climbs further than that
    assert set(removed) >= {shard1, shard2, mmproj}


def test_delete_model_refuses_lmstudio_source(tmp_path: Path) -> None:
    p = tmp_path / "model.gguf"
    p.write_bytes(b"x")
    model = LibraryModel(key="pub/name", name="name", path=p, size_bytes=1, source="lmstudio",
                          publisher="pub", repo="name-GGUF", quant="unknown", shards=(p,), mmproj=None, meta=None)

    with pytest.raises(PermissionError):
        delete_model(model)
    assert p.exists()


# =====================================================================================
# auto_profile (a synthetic 12B/48-layer model, matching the real gemma-4-12B header shape)
# =====================================================================================

_SYNTHETIC_12B_META = GgufMeta(
    architecture="gemma4", n_layer=48, n_ctx_train=262144, n_embd=3840, n_head=16,
    n_head_kv=None, kv_heads_total=328, head_dim=512, params=12_000_000_000,
    file_type=2, quant="Q4_0", name="synthetic-12b",
)
_SYNTHETIC_12B_WEIGHTS = 6_800_000_000


def _synthetic_model(meta: GgufMeta | None) -> LibraryModel:
    path = Path("synthetic-12b-Q4_0.gguf")
    return LibraryModel(key="acme/synthetic-12b", name="synthetic-12b", path=path,
                         size_bytes=_SYNTHETIC_12B_WEIGHTS, source="library", publisher="acme",
                         repo="synthetic-12b-GGUF", quant="Q4_0", shards=(path,), mmproj=None, meta=meta)


def test_auto_profile_full_gpu_fit_at_a_generous_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_mod, "default_threads", lambda: 14)
    cfg = LLMConfig()
    model = _synthetic_model(_SYNTHETIC_12B_META)

    profile = auto_profile(model, budget_mb=16003, ram_total_mb=None, physical_cores=None,
                            gpu_framework="nvidia", base=cfg)

    assert profile.ctx_size == 12288
    assert profile.ctx_size % 2048 == 0
    assert profile.gpu_layers is None       # every layer on the GPU -- no partial-offload count
    assert profile.kv_cache_type == "q8_0"
    assert profile.flash_attn is True
    assert profile.auto is True


def test_auto_profile_partial_offload_at_a_tighter_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_mod, "default_threads", lambda: 14)
    cfg = LLMConfig()
    model = _synthetic_model(_SYNTHETIC_12B_META)

    profile = auto_profile(model, budget_mb=8192, ram_total_mb=None, physical_cores=None,
                            gpu_framework="nvidia", base=cfg)

    assert profile.ctx_size == 4096
    assert profile.gpu_layers is not None
    assert 0 < profile.gpu_layers < 48
    assert profile.auto is True


def test_auto_profile_no_gpu_forces_cpu_only_4096_ctx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_mod, "default_threads", lambda: 14)
    cfg = LLMConfig()
    model = _synthetic_model(_SYNTHETIC_12B_META)

    profile = auto_profile(model, budget_mb=None, ram_total_mb=None, physical_cores=None,
                            gpu_framework="cpu", base=cfg)

    assert profile.ctx_size == 4096
    assert profile.gpu_layers == 0
    assert profile.kv_cache_type == "f16"
    assert profile.flash_attn is False
    assert profile.auto is True


def test_auto_profile_without_gguf_metadata_defers_to_global_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_mod, "default_threads", lambda: 14)
    cfg = LLMConfig()
    model = _synthetic_model(None)   # a header this module could not parse

    profile = auto_profile(model, budget_mb=16003, ram_total_mb=None, physical_cores=None,
                            gpu_framework="nvidia", base=cfg)

    assert profile.ctx_size is None
    assert profile.gpu_layers is None
    assert profile.kv_cache_type == "q8_0"
    assert profile.auto is True
    assert "no usable GGUF metadata" in profile.note
