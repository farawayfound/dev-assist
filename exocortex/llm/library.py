"""ExoCortex's own GGUF model library: discovery, GGUF header metadata, Hugging Face search/pull,
and per-model tuning profiles.

2026-09-27 (portable-install): the owner's brief for this branch is a machine with NEITHER LM
Studio NOR Ollama installed still getting a working local LLM in one run. That means the app can
no longer treat "LM Studio's downloads folder" as *the* model library -- it needs one of its own
(``<project_root>/models/llm/<publisher>/<repo>/*.gguf``, the same layout LM Studio uses, so a
model dropped in by hand or moved between the two just works), with LM Studio's own folder read as
an OPTIONAL extra source when it happens to exist (:func:`library_roots`). Everything else in this
module exists to make that library self-sufficient: reading a GGUF's own header (:func:`gguf_metadata`)
instead of depending on LM Studio's UI to tell you what a file *is*; searching and pulling whole
models straight from Hugging Face (:func:`hf_search`, :func:`hf_repo_info`, :func:`pull_model`) --
deliberately a WHOLE quant's worth of shards, never a pick-a-file browse of a repo that often holds
a dozen quants and a dozen shards each; and turning a hardware budget into a concrete llama-server
tuning (:func:`auto_profile`) that the "Models..." window's Tuning tab can then let the owner
override per model (:class:`ModelProfile`, :func:`load_profiles`/:func:`save_profiles`).

Owned entirely by this module (agent A of five landing at once, 2026-09-27 -- see SPEC.md's
"Coding rules" and the cross-module interface block in the branch's task description for the
other four): :mod:`exocortex.llm.server` (GgufModel/LlamaServerManager), :mod:`exocortex.hardware`
(HardwareReport), :mod:`exocortex.llm.runtimes` (llama.cpp runtime fetch) and :mod:`exocortex.config`
(LLMConfig's new ``library_dir``/``runtime_dir``/``server_url`` fields) are each somebody else's
files this branch reads but never edits. Because those four are being written in the same working
tree at the same time, every touch point into them here is deliberately late-bound: `library_dir`/
`library_roots`/`pull_model`/`load_profiles`/`save_profiles` read `cfg.library_dir` off whatever
:class:`exocortex.config.LLMConfig` instance they are handed (a plain attribute access -- Python
dataclass instances take an attribute that was never declared as a field just as happily, so this
works whether or not agent D's field has landed yet in *this* checkout) rather than importing
anything of agent D's; and every reach into :mod:`exocortex.llm.server` (for
``lmstudio_models_dir``/``default_threads``) or :mod:`exocortex.gpu` (for hardware context) is a
lazy, function-body import wrapped so a missing/failing probe degrades to ``None`` instead of
raising -- both because agent E's server.py is expected to import THIS module (`available_models()`
-> ``library.scan_library``, per the interface block), which would make a module-level import back
into server.py an actual circular import, and because none of this module's own behaviour should
ever depend on those other four branches having landed in a particular order.

Every function here is a pure computation over paths/bytes/HTTP responses it is handed -- nothing
imports touches the GPU, the network, or LM Studio at MODULE-IMPORT time (the house rule every
module here follows), and :func:`hf_search`/:func:`hf_repo_info`/:func:`pull_model` all take an
optional `http` (an already-open ``httpx.Client``) precisely so tests can hand them one wired to
``httpx.MockTransport`` and never touch a real network, exactly like
:mod:`exocortex.models_dl`/:mod:`exocortex.llm.runtimes` already do for their own downloads.

GGUF header parsing (:func:`gguf_metadata`) reads ONLY the ``GGUF`` magic, the version, the tensor/
KV counts and then the KV section itself -- it stops at the first byte of the tensor-info section
and never touches tensor data, so it is cheap enough to run over an entire library on every scan
(``with_meta=True``). A KV value this module does not care about (most of them -- a 32-vocab-entry
tokenizer array included) is never materialised: fixed-size scalars are skipped with a single
``seek``, and a string/array value is skipped by reading only its length prefix(es), never its
bytes, which is what "cheap" means for a 262144-entry Gemma vocabulary here.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import shutil
import struct
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, fields
from datetime import date, timezone
from pathlib import Path
from typing import Any

import httpx

from exocortex.config import LLMConfig, project_root

log = logging.getLogger(__name__)

#: The library's own folder under the project root, when [llm].library_dir is not set: see
#: library_dir() below. "llm" (not "models/gguf" or similar) because a sibling non-LLM model
#: folder already exists at <project_root>/models (VAD/speaker/OCR/ASR models) -- this is that
#: same models/ tree's own subdirectory for LLM GGUFs, laid out exactly like LM Studio's library
#: (<publisher>/<repo>/*.gguf) so a file moved between the two "just works" either way.
LIBRARY_DIRNAME = "llm"

#: The per-model tuning-profile store's file name, under library_dir().
PROFILES_FILE = "profiles.json"


def library_dir(cfg: LLMConfig) -> Path:
    """This app's own GGUF library: ``cfg.library_dir`` (an explicit override) or
    ``<project_root>/models/llm``. Never LM Studio's folder -- see :func:`library_roots` for that."""
    override = getattr(cfg, "library_dir", "") or ""
    if override:
        return Path(override).expanduser()
    return project_root() / "models" / LIBRARY_DIRNAME


def library_roots(cfg: LLMConfig) -> list[tuple[str, Path]]:
    """Every directory :func:`scan_library` reads, in scan order: this app's own library first
    (source ``"library"`` -- the only one :func:`delete_model` may ever touch), then LM Studio's
    download folder (source ``"lmstudio"``) IF it happens to exist on this machine -- LM Studio is
    optional now (2026-09-27), so a machine that never installed it simply gets one root."""
    roots: list[tuple[str, Path]] = [("library", library_dir(cfg))]
    from exocortex.llm import server as _server  # lazy: see the module docstring's import note

    try:
        lm_dir = _server.lmstudio_models_dir(cfg)
    except Exception:  # noqa: BLE001 - a library scan must never fail over LM Studio probing
        log.debug("library: could not resolve LM Studio's models dir", exc_info=True)
        lm_dir = None
    if lm_dir is not None and lm_dir.is_dir():
        roots.append(("lmstudio", lm_dir))
    return roots


# =========================================================================================
# GGUF header parsing (magic + version + KV section only -- never tensor data)
# =========================================================================================

_GGUF_MAGIC = b"GGUF"

# GGUF metadata value types (ggml's gguf_type enum), 0-12 -- see the module/task docstrings.
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16 = 0, 1, 2, 3
_T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = 4, 5, 6, 7
_T_STRING, _T_ARRAY = 8, 9
_T_UINT64, _T_INT64, _T_FLOAT64 = 10, 11, 12

_SCALAR_STRUCT: dict[int, str] = {
    _T_UINT8: "<B", _T_INT8: "<b", _T_UINT16: "<H", _T_INT16: "<h",
    _T_UINT32: "<I", _T_INT32: "<i", _T_FLOAT32: "<f", _T_BOOL: "<B",
    _T_UINT64: "<Q", _T_INT64: "<q", _T_FLOAT64: "<d",
}
_SCALAR_SIZE: dict[int, int] = {t: struct.calcsize(fmt) for t, fmt in _SCALAR_STRUCT.items()}


def _read_exact(f: Any, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError(f"unexpected end of file (wanted {n} bytes, got {len(data)})")
    return data


def _read_u32(f: Any) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f: Any) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_string(f: Any) -> str:
    n = _read_u64(f)
    return _read_exact(f, n).decode("utf-8", errors="replace")


def _read_scalar(f: Any, vtype: int) -> Any:
    fmt = _SCALAR_STRUCT[vtype]
    value = struct.unpack(fmt, _read_exact(f, _SCALAR_SIZE[vtype]))[0]
    return bool(value) if vtype == _T_BOOL else value


def _skip_array(f: Any, elem_type: int, count: int) -> None:
    """Skip `count` elements of `elem_type` without materialising them -- a single ``seek`` for a
    fixed-size element (this is what makes a 262144-entry vocabulary array cheap to pass over), or
    one length read + seek per element for strings (their length prefixes still have to be read to
    know how far to skip, but their bytes never do)."""
    if elem_type in _SCALAR_SIZE:
        f.seek(_SCALAR_SIZE[elem_type] * count, 1)
    elif elem_type == _T_STRING:
        for _ in range(count):
            n = _read_u64(f)
            f.seek(n, 1)
    elif elem_type == _T_ARRAY:
        for _ in range(count):
            sub_type = _read_u32(f)
            sub_count = _read_u64(f)
            _skip_array(f, sub_type, sub_count)
    else:
        raise ValueError(f"unknown GGUF array element type {elem_type}")


def _read_value(f: Any, vtype: int, *, want_value: bool) -> Any:
    """Read (or, when `want_value` is False, cheaply skip) one KV value of type `vtype`."""
    if vtype in _SCALAR_SIZE:
        if want_value:
            return _read_scalar(f, vtype)
        f.seek(_SCALAR_SIZE[vtype], 1)
        return None
    if vtype == _T_STRING:
        if want_value:
            return _read_string(f)
        n = _read_u64(f)
        f.seek(n, 1)
        return None
    if vtype == _T_ARRAY:
        elem_type = _read_u32(f)
        count = _read_u64(f)
        if want_value:
            return [_read_value(f, elem_type, want_value=True) for _ in range(count)]
        _skip_array(f, elem_type, count)
        return None
    raise ValueError(f"unknown GGUF value type {vtype}")


#: Suffixes of the per-architecture KV keys this module reads (e.g. "gemma4.block_count",
#: "qwen3.attention.head_count_kv") -- matched by suffix so the architecture never has to be known
#: in advance (a single pass over the KV section, no two-pass "read architecture first" dance).
_SUFFIX_BLOCK_COUNT = ".block_count"
_SUFFIX_CONTEXT_LENGTH = ".context_length"
_SUFFIX_EMBEDDING_LENGTH = ".embedding_length"
_SUFFIX_HEAD_COUNT = ".attention.head_count"
_SUFFIX_HEAD_COUNT_KV = ".attention.head_count_kv"
_SUFFIX_KEY_LENGTH = ".attention.key_length"


@dataclass(frozen=True)
class GgufMeta:
    """The handful of GGUF header fields the tuning math (:func:`auto_profile`,
    :func:`kv_bytes_per_token`) and the library UI need -- never the full KV dump."""

    architecture: str
    n_layer: int | None
    n_ctx_train: int | None
    n_embd: int | None
    n_head: int | None
    # None whenever <arch>.attention.head_count_kv was a per-layer array (e.g. gemma4's
    # [8, 8, 8, 8, 8, 1, ...]) -- there is then no single scalar to report here; the array is
    # summed into kv_heads_total instead, which is what the KV-cache-size math actually needs.
    n_head_kv: int | None
    kv_heads_total: int | None
    head_dim: int | None
    params: int | None
    file_type: int | None
    quant: str
    name: str


_SIZE_LABEL_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMBT])$", re.IGNORECASE)
_SIZE_LABEL_MULT: dict[str, float] = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def _params_from_size_label(label: Any) -> int | None:
    """``general.size_label`` ("12B", "270M", ...) parsed into an approximate parameter count.
    Never raises: an unrecognised label (e.g. an MoE "8x7B") is simply ``None``."""
    if not isinstance(label, str):
        return None
    m = _SIZE_LABEL_RE.match(label.strip())
    if not m:
        return None
    return int(round(float(m.group(1)) * _SIZE_LABEL_MULT[m.group(2).upper()]))


#: Quant token straight from the file name -- tried BEFORE the file_type table, since the name is
#: what a human (and choose_quant/group_quants) actually compares against, and it survives quants
#: the file_type enum below has no entry for (unsloth's "UD-*" dynamic quants).
_QUANT_FILENAME_RE = re.compile(
    r"Q\d_K_[SML]|Q\d_[01K]|IQ\d_(?:XXS|XS|S|M|NL)|UD-[A-Z0-9_]+|BF16|F16|F32|TQ\d_0|Q\d_K",
    re.IGNORECASE,
)


def _quant_from_filename(name: str) -> str | None:
    m = _QUANT_FILENAME_RE.search(name)
    return m.group(0).upper() if m else None


#: llama.cpp's ggml_ftype enum (general.file_type), the subset with a stable quant-name mapping --
#: the fallback for a file whose name carries no recognisable quant token at all.
_FILE_TYPE_QUANT: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
}


def _quant_from_file_type(file_type: Any) -> str | None:
    try:
        return _FILE_TYPE_QUANT.get(int(file_type)) if file_type is not None else None
    except (TypeError, ValueError):
        return None


def gguf_metadata(path: Path) -> GgufMeta:
    """Parse `path`'s GGUF header (magic + version + KV section only, never tensor data or
    tensor-shape info). Raises :class:`ValueError` if `path` is not a GGUF file (bad magic) or
    carries a GGUF version this module does not know (only 2 and 3 exist as of this writing)."""
    path = Path(path)
    architecture = name = None
    file_type = size_label = None
    n_layer = n_ctx_train = n_embd = n_head = head_dim = None
    head_count_kv_raw: Any = None

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != _GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file (magic was {magic!r})")
        version = _read_u32(f)
        if version not in (2, 3):
            raise ValueError(f"{path}: unsupported GGUF version {version} (only 2 and 3 are known)")
        _n_tensors = _read_u64(f)   # unused here, but must be consumed to reach n_kv
        n_kv = _read_u64(f)

        for _ in range(n_kv):
            key = _read_string(f)
            vtype = _read_u32(f)
            if key == "general.architecture":
                architecture = _read_value(f, vtype, want_value=True)
            elif key == "general.name":
                name = _read_value(f, vtype, want_value=True)
            elif key == "general.file_type":
                file_type = _read_value(f, vtype, want_value=True)
            elif key == "general.size_label":
                size_label = _read_value(f, vtype, want_value=True)
            elif key.endswith(_SUFFIX_BLOCK_COUNT):
                n_layer = _read_value(f, vtype, want_value=True)
            elif key.endswith(_SUFFIX_CONTEXT_LENGTH):
                n_ctx_train = _read_value(f, vtype, want_value=True)
            elif key.endswith(_SUFFIX_EMBEDDING_LENGTH):
                n_embd = _read_value(f, vtype, want_value=True)
            elif key.endswith(_SUFFIX_HEAD_COUNT):
                n_head = _read_value(f, vtype, want_value=True)
            elif key.endswith(_SUFFIX_HEAD_COUNT_KV):
                head_count_kv_raw = _read_value(f, vtype, want_value=True)
            elif key.endswith(_SUFFIX_KEY_LENGTH):
                head_dim = _read_value(f, vtype, want_value=True)
            else:
                _read_value(f, vtype, want_value=False)   # everything else: skip cheaply

    def _int_or_none(v: Any) -> int | None:
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    n_layer = _int_or_none(n_layer)
    n_ctx_train = _int_or_none(n_ctx_train)
    n_embd = _int_or_none(n_embd)
    n_head = _int_or_none(n_head)
    head_dim = _int_or_none(head_dim)

    n_head_kv: int | None = None
    kv_heads_total: int | None = None
    if isinstance(head_count_kv_raw, list):
        nums = [int(x) for x in head_count_kv_raw if isinstance(x, (int, float))]
        kv_heads_total = sum(nums) if nums else None
    else:
        n_head_kv = _int_or_none(head_count_kv_raw)
        if n_head_kv is not None:
            kv_heads_total = n_head_kv * n_layer if n_layer else n_head_kv

    if head_dim is None and n_embd is not None and n_head:
        head_dim = n_embd // n_head

    quant = _quant_from_filename(path.name) or _quant_from_file_type(file_type) or "unknown"

    return GgufMeta(
        architecture=str(architecture) if architecture is not None else "unknown",
        n_layer=n_layer, n_ctx_train=n_ctx_train, n_embd=n_embd, n_head=n_head,
        n_head_kv=n_head_kv, kv_heads_total=kv_heads_total, head_dim=head_dim,
        params=_params_from_size_label(size_label), file_type=_int_or_none(file_type),
        quant=quant, name=str(name) if name is not None else path.stem,
    )


# =========================================================================================
# The library itself: scan / delete
# =========================================================================================

@dataclass(frozen=True)
class LibraryModel:
    """One model in the library -- a single GGUF, a group of shards, or (with its mmproj
    attached) a vision model, discovered by :func:`scan_library`."""

    key: str            # "publisher/repo" -- what [llm].model matches against
    name: str           # "repo" with a trailing "-GGUF" stripped
    path: Path          # the first shard (or the only file)
    size_bytes: int      # sum across all shards
    source: str          # "library" | "lmstudio"
    publisher: str
    repo: str
    quant: str
    shards: tuple[Path, ...]
    mmproj: Path | None
    meta: GgufMeta | None

    @property
    def display(self) -> str:
        return f"{self.key}  ({self.size_bytes / 1e9:.1f} GB)"


#: "<stem>-00001-of-00005.gguf" -- the sharded-GGUF naming every llama.cpp/HF split uses.
_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d+)-of-(?P<total>\d+)$", re.IGNORECASE)


def _iter_gguf_files(root: Path) -> list[Path]:
    """Every ``*.gguf`` up to 3 levels deep -- exactly :func:`exocortex.llm.server.scan_models`'s own
    depth limit (a full ``rglob`` over a large library with unrelated subtrees is what that
    module's docstring measured at ~20 s cold; this mirrors it rather than repeating that cost)."""
    found: set[Path] = set()
    for pattern in ("*.gguf", "*/*.gguf", "*/*/*.gguf"):
        found.update(root.glob(pattern))
    return sorted(found)


def _publisher_repo_name(root: Path, path: Path) -> tuple[str, str, str]:
    """(publisher, repo, name) exactly like ``exocortex.llm.server.scan_models`` derives them, so a
    ``[llm].model`` value that resolved against LM Studio's library keeps resolving against this
    one too."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    parts = rel.parts
    if len(parts) >= 3:
        publisher, repo = parts[0], parts[1]
    elif len(parts) == 2:
        publisher, repo = parts[0], path.stem
    else:
        publisher, repo = "local", path.stem
    name = re.sub(r"[-_.]?gguf$", "", repo, flags=re.IGNORECASE)
    return publisher, repo, name


def _best_mmproj_match(mmproj_name: str, candidates: list[LibraryModel]) -> LibraryModel | None:
    """Which model in the same folder an ``mmproj*.gguf`` file belongs to: the one whose quant
    token (or, failing that, name) appears in the mmproj file's own name, else -- most repos ship
    exactly one non-mmproj model per folder anyway -- the largest model there."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    low = mmproj_name.lower()
    for m in candidates:
        if m.quant != "unknown" and m.quant.lower() in low:
            return m
    for m in candidates:
        if m.name.lower() in low:
            return m
    return max(candidates, key=lambda m: m.size_bytes)


def _scan_root(root: Path, source: str, with_meta: bool) -> list[LibraryModel]:
    root = Path(root)
    if not root.is_dir():
        return []

    by_dir: dict[Path, list[Path]] = {}
    for p in _iter_gguf_files(root):
        by_dir.setdefault(p.parent, []).append(p)

    models: list[LibraryModel] = []
    for _directory, paths in by_dir.items():
        mmproj_paths: list[Path] = []
        candidates: list[Path] = []
        for p in sorted(paths):
            low = p.name.lower()
            if "mmproj" in low:
                mmproj_paths.append(p)
            elif low.startswith("mtp-") or "imatrix" in low:
                continue                                   # never listed as models
            else:
                candidates.append(p)

        shard_groups: dict[str, list[tuple[int, Path]]] = {}
        singles: list[Path] = []
        for p in candidates:
            m = _SHARD_RE.match(p.stem)
            if m:
                shard_groups.setdefault(m.group("stem"), []).append((int(m.group("idx")), p))
            else:
                singles.append(p)

        groups: list[list[Path]] = [p for p in ([s] for s in singles)]
        for items in shard_groups.values():
            items.sort(key=lambda t: t[0])
            groups.append([p for _idx, p in items])

        built: list[LibraryModel] = []
        for shard_paths in groups:
            rep_path = shard_paths[0]
            publisher, repo, name = _publisher_repo_name(root, rep_path)
            quant = _quant_from_filename(rep_path.name) or "unknown"
            meta: GgufMeta | None = None
            if with_meta:
                try:
                    meta = gguf_metadata(rep_path)
                except (OSError, ValueError):
                    log.debug("library: could not read GGUF metadata for %s", rep_path, exc_info=True)
                if meta is not None:
                    quant = meta.quant
            try:
                size = sum(sp.stat().st_size for sp in shard_paths)
            except OSError:
                size = 0
            built.append(LibraryModel(
                key=f"{publisher}/{name}", name=name, path=rep_path, size_bytes=size,
                source=source, publisher=publisher, repo=repo, quant=quant,
                shards=tuple(shard_paths), mmproj=None, meta=meta,
            ))

        # Attach each mmproj file to its best-matching sibling model; when several map to the same
        # model (an F16 and a Q8_0 mmproj in the same folder), keep the smallest -- deterministic,
        # and the same "smallest mmproj" convention pull_model's include_mmproj uses.
        mmproj_for: dict[int, Path] = {}
        for mp in mmproj_paths:
            target = _best_mmproj_match(mp.name, built)
            if target is None:
                continue
            idx = built.index(target)
            current = mmproj_for.get(idx)
            if current is None or mp.stat().st_size < current.stat().st_size:
                mmproj_for[idx] = mp
        for idx, mp in mmproj_for.items():
            built[idx] = dataclasses.replace(built[idx], mmproj=mp)

        models.extend(built)

    return models


def scan_library(cfg: LLMConfig, *, with_meta: bool = False) -> list[LibraryModel]:
    """Every model across :func:`library_roots` (own library first, then LM Studio's folder if it
    exists), deduplicated by resolved path -- a file reachable from both roots (an unusual config
    where they point at the same place) is only ever listed once, under whichever source scanned
    it first. `with_meta=True` additionally reads each model's GGUF header
    (:func:`gguf_metadata`); a file that fails to parse gets ``meta=None`` rather than aborting the
    whole scan."""
    out: list[LibraryModel] = []
    seen: set[Path] = set()
    for source, root in library_roots(cfg):
        for model in _scan_root(root, source, with_meta):
            try:
                resolved = model.path.resolve()
            except OSError:
                resolved = model.path
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(model)
    return sorted(out, key=lambda m: m.key)


def delete_model(model: LibraryModel) -> list[Path]:
    """Delete `model`'s shards and mmproj (and its now-empty ``<publisher>/<repo>`` folder pair,
    never climbing further up than that fixed two-level layout). Only ``source == "library"``
    models may be deleted this way; anything else (LM Studio's own folder) raises
    :class:`PermissionError` -- this app never deletes another program's files."""
    if model.source != "library":
        raise PermissionError(
            f"cannot delete {model.key!r}: it lives in the {model.source!r} library, not this "
            "app's own -- only files under this app's models/llm folder can be deleted from here"
        )
    targets = list(model.shards) if model.shards else [model.path]
    if model.mmproj is not None:
        targets.append(model.mmproj)

    removed: list[Path] = []
    for p in targets:
        try:
            p.unlink()
            removed.append(p)
        except FileNotFoundError:
            pass

    directory = targets[0].parent if targets else model.path.parent
    for _ in range(2):   # <publisher>/<repo>, never further
        try:
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
                removed.append(directory)
                directory = directory.parent
            else:
                break
        except OSError:
            break
    return removed


# =========================================================================================
# Hugging Face Hub: search, repo info, quant grouping/choice, whole-model pulls
# =========================================================================================

_HF_BASE = "https://huggingface.co"
_HF_TIMEOUT = httpx.Timeout(30.0, read=120.0)


def _hf_headers() -> dict[str, str]:
    headers = {"User-Agent": "exocortex/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _new_hf_client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, headers=_hf_headers(), timeout=_HF_TIMEOUT)


@dataclass(frozen=True)
class HfRepo:
    id: str
    downloads: int
    likes: int
    modified: str
    gated: bool
    private: bool


@dataclass(frozen=True)
class HfFile:
    path: str
    size: int
    sha256: str        # "" when the file is not an LFS object (no digest to check)


@dataclass(frozen=True)
class QuantOption:
    quant: str
    files: tuple[HfFile, ...]     # this quant's shard(s), in shard order
    size_bytes: int


@dataclass(frozen=True)
class HfRepoInfo:
    id: str
    sha: str
    files: tuple[HfFile, ...]
    architecture: str
    params: int | None
    context_length: int | None
    quants: tuple[QuantOption, ...]
    mmproj: tuple[HfFile, ...]


def hf_search(query: str, *, limit: int = 25, http: httpx.Client | None = None) -> list[HfRepo]:
    """Search Hugging Face's model hub for GGUF repos matching `query`, newest-downloads first."""
    owns = http is None
    client = http or _new_hf_client()
    try:
        resp = client.get(f"{_HF_BASE}/api/models", params={
            "search": query, "filter": "gguf", "sort": "downloads", "direction": -1, "limit": limit,
        })
        resp.raise_for_status()
        data = resp.json()
    finally:
        if owns:
            client.close()

    out: list[HfRepo] = []
    for item in data if isinstance(data, list) else []:
        out.append(HfRepo(
            id=str(item.get("id") or item.get("modelId") or ""),
            downloads=int(item.get("downloads") or 0),
            likes=int(item.get("likes") or 0),
            modified=str(item.get("lastModified") or ""),
            gated=bool(item.get("gated")),
            private=bool(item.get("private")),
        ))
    return out


#: Files never offered as a quant to pull: side files (mmproj), draft/speculative-decoding heads
#: ("mtp-"), and calibration data (imatrix) -- none of these are a model on their own.
def _is_ancillary_gguf(name: str) -> bool:
    low = name.lower()
    return "mmproj" in low or low.startswith("mtp-") or "imatrix" in low


def group_quants(files: Sequence[HfFile]) -> tuple[QuantOption, ...]:
    """Top-level ``*.gguf`` files (never a subfolder such as unsloth's ``MTP/`` or ``BF16/``, and
    never mmproj/mtp-/imatrix), grouped into one :class:`QuantOption` per quant token found in the
    file names, each option's shards in shard order. A file whose name carries no recognisable
    quant token at all (see :data:`_QUANT_FILENAME_RE`) is skipped -- there is nothing to group it
    under."""
    by_quant: dict[str, list[HfFile]] = {}
    for f in files:
        if "/" in f.path:
            continue
        base = Path(f.path).name
        if not base.lower().endswith(".gguf") or _is_ancillary_gguf(base):
            continue
        quant = _quant_from_filename(base)
        if quant is None:
            continue
        by_quant.setdefault(quant, []).append(f)

    def _shard_idx(f: HfFile) -> int:
        m = _SHARD_RE.match(Path(f.path).stem)
        return int(m.group("idx")) if m else 0

    options = []
    for quant, flist in by_quant.items():
        ordered = tuple(sorted(flist, key=_shard_idx))
        options.append(QuantOption(quant=quant, files=ordered, size_bytes=sum(x.size for x in ordered)))
    return tuple(sorted(options, key=lambda o: o.quant))


def hf_repo_info(repo_id: str, *, http: httpx.Client | None = None) -> HfRepoInfo:
    """Everything :func:`pull_model` needs to know about `repo_id`: its file listing, its
    (self-reported) architecture/context length, and its GGUF quants grouped by
    :func:`group_quants`."""
    owns = http is None
    client = http or _new_hf_client()
    try:
        resp = client.get(f"{_HF_BASE}/api/models/{repo_id}")
        resp.raise_for_status()
        data = resp.json()
        tree_resp = client.get(f"{_HF_BASE}/api/models/{repo_id}/tree/main", params={"recursive": "true"})
        tree_resp.raise_for_status()
        tree = tree_resp.json()
    finally:
        if owns:
            client.close()

    files: list[HfFile] = []
    for item in tree if isinstance(tree, list) else []:
        if item.get("type") != "file":
            continue
        lfs = item.get("lfs") or {}
        files.append(HfFile(path=str(item.get("path") or ""), size=int(item.get("size") or 0),
                             sha256=str(lfs.get("oid") or "")))

    gguf_info = data.get("gguf") or {}
    context_length = gguf_info.get("context_length")
    total_params = gguf_info.get("total")
    mmproj = tuple(f for f in files if "/" not in f.path and "mmproj" in Path(f.path).name.lower())

    return HfRepoInfo(
        id=str(data.get("id") or repo_id),
        sha=str(data.get("sha") or ""),
        files=tuple(files),
        architecture=str(gguf_info.get("architecture") or ""),
        params=int(total_params) if isinstance(total_params, (int, float)) else None,
        context_length=int(context_length) if isinstance(context_length, (int, float)) else None,
        quants=group_quants(files),
        mmproj=mmproj,
    )


#: choose_quant policy "auto": preference order, most-preferred first -- see the task brief this
#: module was built from for the exact list (a 4-bit-ish default that keeps generation on the GPU
#: without demanding the biggest quant a repo has).
_AUTO_PREF: tuple[str, ...] = (
    "Q4_K_M", "UD-Q4_K_XL", "Q4_K_S", "IQ4_XS", "IQ4_NL", "Q4_0", "Q4_1",
    "Q3_K_M", "UD-Q3_K_XL", "IQ3_XXS", "UD-IQ3_XXS", "Q3_K_S",
    "Q2_K", "UD-Q2_K_XL", "IQ2_M", "UD-IQ2_M",
)
#: policy "quality": the same tail as _AUTO_PREF, with the larger-than-Q4_K_M quants prepended.
_QUALITY_PREF: tuple[str, ...] = ("Q8_0", "Q6_K", "UD-Q6_K_XL", "Q5_K_M", "UD-Q5_K_XL", "Q5_K_S") + _AUTO_PREF
_FLOAT_QUANTS = frozenset({"BF16", "F16", "F32"})


def choose_quant(options: Sequence[QuantOption], *, budget_mb: int | None, policy: str = "auto") -> QuantOption | None:
    """Pick one of `options` for this machine.

    * A named policy (e.g. ``"Q6_K"``, anything not "auto"/"quality"/"fit"): that exact quant, or
      ``None`` if `options` has no such quant.
    * ``"auto"`` (default): the first of :data:`_AUTO_PREF` that both exists in `options` and fits
      75% of `budget_mb` (room left for the KV cache/activations); failing that, the largest
      non-float quant that fits; without a `budget_mb` (no GPU detected), the fit check is skipped
      entirely -- the first preference-list quant present wins regardless of size.
    * ``"quality"``: the LARGEST of :data:`_QUALITY_PREF` that fits (not merely the first in list
      order -- quality wants the best quant this machine can run, not the first one that happens to
      fit); same "any other quant" and no-budget fallbacks as "auto".
    * ``"fit"``: the largest non-float quant that fits at all, full stop.

    A repo with exactly one non-float quant (a QAT release, e.g. only ``Q4_0``) falls out of this
    the same way any other single-quant repo would: it is returned whenever it fits, by whichever
    of the rules above it happens to match.
    """
    by_name = {o.quant: o for o in options}
    limit_bytes = 0.75 * budget_mb * 2**20 if budget_mb is not None else None

    def fits(o: QuantOption) -> bool:
        return limit_bytes is None or o.size_bytes <= limit_bytes

    def largest_non_float(pool: Iterable[QuantOption]) -> QuantOption | None:
        candidates = [o for o in pool if o.quant not in _FLOAT_QUANTS]
        return max(candidates, key=lambda o: o.size_bytes, default=None)

    pol = (policy or "auto").strip()
    if pol not in ("auto", "quality", "fit"):
        return by_name.get(pol)

    if pol == "fit":
        if limit_bytes is None:
            return largest_non_float(options)
        fitting = [o for o in options if o.quant not in _FLOAT_QUANTS and fits(o)]
        return max(fitting, key=lambda o: o.size_bytes, default=None)

    pref = _QUALITY_PREF if pol == "quality" else _AUTO_PREF
    if limit_bytes is None:
        for name in pref:
            if name in by_name:
                return by_name[name]
        return largest_non_float(options)

    if pol == "quality":
        candidates = [by_name[n] for n in pref if n in by_name and fits(by_name[n])]
        if candidates:
            return max(candidates, key=lambda o: o.size_bytes)
    else:
        for name in pref:
            o = by_name.get(name)
            if o is not None and fits(o):
                return o

    return largest_non_float(o for o in options if fits(o))


# ---- pulling a whole model ---------------------------------------------------------------------

class PullError(RuntimeError):
    """A Hugging Face model pull could not complete: a bad repo/quant name, a network/integrity
    failure, or not enough disk space."""


class PullCancelled(PullError):
    """The caller's `cancelled` hook fired mid-download. The ``.part`` file already on disk is
    kept (not deleted) so the next :func:`pull_model` call resumes instead of starting over."""


_PULL_CHUNK = 1 << 16


def _pull_one_file(client: httpx.Client, repo_id: str, hf_file: HfFile, dest: Path,
                   progress: "Callable[[str, int, int], None] | None",
                   cancelled: "Callable[[], bool] | None") -> None:
    """Stream one repo file to `dest`, resuming an existing ``dest.part`` via an HTTP Range
    request. A file already complete at `dest` (exact size match) is a no-op -- no request made."""
    if dest.is_file() and hf_file.size and dest.stat().st_size == hf_file.size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    url = f"{_HF_BASE}/{repo_id}/resolve/main/{hf_file.path}"

    resume_from = part.stat().st_size if part.is_file() else 0
    hasher = hashlib.sha256()
    headers: dict[str, str] = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
        with open(part, "rb") as existing:
            for chunk in iter(lambda: existing.read(1 << 20), b""):
                hasher.update(chunk)

    try:
        with client.stream("GET", url, headers=headers) as resp:
            if resume_from > 0 and resp.status_code != 206:
                # The server ignored the Range request (some CDNs do for small/edge-cached
                # objects): restart from scratch rather than silently duplicating bytes.
                resume_from = 0
                hasher = hashlib.sha256()
            resp.raise_for_status()
            total = hf_file.size or resume_from + int(resp.headers.get("content-length") or 0)
            downloaded = resume_from
            with open(part, "ab" if resume_from > 0 else "wb") as out:
                for chunk in resp.iter_bytes(_PULL_CHUNK):
                    out.write(chunk)
                    hasher.update(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(hf_file.path, downloaded, total)
                    if cancelled is not None and cancelled():
                        raise PullCancelled(f"pull of {hf_file.path} cancelled after {downloaded} bytes")
    except PullCancelled:
        raise
    except (httpx.HTTPError, OSError) as exc:
        raise PullError(f"failed to download {hf_file.path} from {repo_id}: {exc}") from exc

    actual_size = part.stat().st_size
    if hf_file.size and actual_size != hf_file.size:
        part.unlink(missing_ok=True)
        raise PullError(f"{hf_file.path}: size mismatch (expected {hf_file.size}, got {actual_size})")
    if hf_file.sha256:
        digest = hasher.hexdigest()
        if digest.lower() != hf_file.sha256.lower():
            part.unlink(missing_ok=True)
            raise PullError(f"{hf_file.path}: sha256 mismatch (expected {hf_file.sha256}, got {digest})")
    part.replace(dest)


#: One path component as the Hub may name it (repo owner, repo name, file name): letters, digits,
#: and the usual separators -- never "..", never a slash, never empty.
_SAFE_COMPONENT = re.compile(r"^[\w][\w.\- +@()\[\]]*$")


def _check_path_component(component: str, *, what: str) -> None:
    """Refuse a component that could leave the library folder when joined onto it: empty, ".",
    "..", anything with a path separator or a drive/absolute prefix, or characters outside the
    Hub's own naming rules. Raises :class:`PullError`."""
    if (not component or component in (".", "..") or "/" in component or "\\" in component
            or ":" in component or not _SAFE_COMPONENT.match(component)):
        raise PullError(f"refusing unsafe {what}: it must be a single plain path component")


def pull_model(repo_id: str, quant: str, cfg: LLMConfig, *, include_mmproj: bool = False,
               policy: str = "auto", budget_mb: int | None = None,
               progress: "Callable[[str, int, int], None] | None" = None,
               cancelled: "Callable[[], bool] | None" = None,
               http: httpx.Client | None = None) -> LibraryModel:
    """Pull one whole quant of `repo_id` -- every shard, sha256-verified, resumable -- into this
    app's own library (never LM Studio's folder: only `source == "library"` models are ever
    written here). `quant` is a quant name (``"Q4_K_M"``) or ``"auto"``/``""`` to let
    :func:`choose_quant` (`policy`, `budget_mb`) pick one for this machine. `include_mmproj` also
    pulls the smallest mmproj file the repo offers, if any. Raises :class:`PullError` (or its
    :class:`PullCancelled` subclass) on anything that stops the pull; returns the resulting
    :class:`LibraryModel`, re-discovered from disk exactly the way :func:`scan_library` would find
    it (so pulling and scanning never disagree about shards/mmproj/key)."""
    owns_client = http is None
    client = http if http is not None else _new_hf_client()
    try:
        info = hf_repo_info(repo_id, http=client)
        quant_spec = (quant or "").strip()
        if not quant_spec or quant_spec.lower() == "auto":
            chosen = choose_quant(info.quants, budget_mb=budget_mb, policy=policy)
            if chosen is None:
                raise PullError(
                    f"no quant of {repo_id} fits this machine (budget {budget_mb} MiB, policy {policy!r})"
                )
        else:
            chosen = next((q for q in info.quants if q.quant.lower() == quant_spec.lower()), None)
            if chosen is None:
                available = ", ".join(sorted(q.quant for q in info.quants)) or "(no GGUF quants found)"
                raise PullError(f"quant {quant!r} not found in {repo_id}; available: {available}")

        files_to_fetch: list[HfFile] = list(chosen.files)
        if include_mmproj and info.mmproj:
            files_to_fetch.append(min(info.mmproj, key=lambda f: f.size))

        publisher, _, repo = repo_id.partition("/")
        if not repo:
            publisher, repo = "local", repo_id
        # Defense in depth (2026-09-27 review): the repo id and every file name come from the
        # Hub's JSON; each becomes exactly one path component under the library, never a
        # traversal ("..", an absolute path, a separator) however the answer was shaped.
        for component in (publisher, repo):
            _check_path_component(component, what=f"repo id {repo_id!r}")
        file_names = [Path(f.path).name for f in files_to_fetch]
        for name in file_names:
            _check_path_component(name, what=f"file name {name!r} in {repo_id}")
        dest_root = library_dir(cfg) / publisher / repo
        dest_root.mkdir(parents=True, exist_ok=True)

        # Free-space check net of what is already on disk: a resumed .part, or a shard that
        # finished in an earlier run, must not make a nearly-done pull look unaffordable.
        needed = 0
        for hf_file, name in zip(files_to_fetch, file_names):
            dest = dest_root / name
            if dest.is_file() and dest.stat().st_size == hf_file.size:
                continue
            part = dest.with_name(dest.name + ".part")
            have = part.stat().st_size if part.is_file() else 0
            needed += max(0, hf_file.size - min(have, hf_file.size))
        free = shutil.disk_usage(dest_root).free
        if free < needed:
            raise PullError(
                f"not enough disk space to pull {repo_id} ({chosen.quant}): need {needed / 1e9:.2f} GB more, "
                f"only {free / 1e9:.2f} GB free at {dest_root}"
            )

        for hf_file, name in zip(files_to_fetch, file_names):
            _pull_one_file(client, repo_id, hf_file, dest_root / name, progress, cancelled)
    finally:
        if owns_client:
            client.close()

    model = next((m for m in scan_library(cfg, with_meta=budget_mb is not None)
                  if m.path.parent == dest_root), None)
    if model is None:
        raise PullError(f"pulled {repo_id} ({chosen.quant}) but could not find it afterwards in {dest_root}")

    if budget_mb is not None:
        ram_total_mb, physical_cores, framework = _hardware_context_for_profile()
        profile = auto_profile(model, budget_mb=budget_mb, ram_total_mb=ram_total_mb,
                               physical_cores=physical_cores, gpu_framework=framework, base=cfg)
        set_profile(cfg, model.key, profile)

    return model


# =========================================================================================
# Per-model tuning profiles
# =========================================================================================

@dataclass
class ModelProfile:
    """Per-model overrides of the global ``[llm]`` tuning -- ``None`` means "use the global value".
    ``auto=True`` marks a profile :func:`auto_profile` generated (the Tuning tab shows it as a
    suggestion); the owner editing any field there should flip it to ``False`` so a later re-pull
    or hardware change never silently overwrites a hand-tuned value."""

    ctx_size: int | None = None
    gpu_layers: int | None = None
    kv_cache_type: str | None = None
    flash_attn: bool | None = None
    threads: int | None = None
    parallel: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    thinking: str | None = None
    extra_args: list[str] = field(default_factory=list)
    auto: bool = True
    note: str = ""
    generated_from: str = ""


def load_profiles(cfg: LLMConfig) -> dict[str, ModelProfile]:
    """``{model_key: ModelProfile}`` from ``profiles.json`` under :func:`library_dir`. Missing or
    unparseable file -> ``{}``; an unknown field in a stored profile is dropped with a warning
    rather than failing the whole load (the same tolerance :mod:`exocortex.config` gives an unknown
    TOML key)."""
    path = library_dir(cfg) / PROFILES_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("library: could not read %s; starting with no saved profiles", path, exc_info=True)
        return {}
    if not isinstance(data, dict):
        return {}

    known = {f.name for f in fields(ModelProfile)}
    out: dict[str, ModelProfile] = {}
    for key, raw in data.items():
        if not isinstance(raw, dict):
            continue
        kwargs = {}
        for k, v in raw.items():
            if k not in known:
                log.warning("library: unknown profile field %r for %r ignored", k, key)
                continue
            kwargs[k] = v
        try:
            out[key] = ModelProfile(**kwargs)
        except TypeError:
            log.warning("library: could not load the saved profile for %r", key, exc_info=True)
    return out


def save_profiles(cfg: LLMConfig, profiles: dict[str, ModelProfile]) -> Path:
    """Write `profiles` to ``profiles.json`` under :func:`library_dir`, atomically (temp file +
    ``replace``, like every other on-disk write in this app)."""
    path = library_dir(cfg) / PROFILES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {key: dataclasses.asdict(p) for key, p in profiles.items()}
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def profile_for(cfg: LLMConfig, key: str) -> ModelProfile | None:
    return load_profiles(cfg).get(key)


def set_profile(cfg: LLMConfig, key: str, profile: ModelProfile | None) -> None:
    """Set (or, with `profile=None`, remove) `key`'s saved profile and save the store at once."""
    profiles = load_profiles(cfg)
    if profile is None:
        profiles.pop(key, None)
    else:
        profiles[key] = profile
    save_profiles(cfg, profiles)


#: KV cache bytes per element, per exact ``[llm].kv_cache_type`` string this app writes -- an
#: unrecognised type (a value the owner typed by hand) falls back to f16's size, the safe (larger)
#: estimate.
_KV_BYTES_PER_ELEMENT: dict[str, float] = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}


def kv_bytes_per_token(meta: GgufMeta, kv_cache_type: str) -> int | None:
    """Bytes of KV cache ONE token costs across every layer: ``2 * kv_heads_total * head_dim *
    bytes_per_element(kv_cache_type)`` (the 2 is K and V). ``None`` when `meta` lacks the head
    counts needed (no ``attention.head_count_kv``/``key_length`` in the header)."""
    if meta.kv_heads_total is None or meta.head_dim is None:
        return None
    per_element = _KV_BYTES_PER_ELEMENT.get((kv_cache_type or "f16").strip().lower(),
                                            _KV_BYTES_PER_ELEMENT["f16"])
    return int(round(2 * meta.kv_heads_total * meta.head_dim * per_element))


_MIB = 2 ** 20


def _hardware_context_for_profile() -> tuple[int | None, int | None, str]:
    """Best-effort ``(ram_total_mb, physical_cores, gpu_framework)`` for :func:`auto_profile`,
    gathered without hard-depending on :mod:`exocortex.gpu`/:mod:`exocortex.hardware` (agents B's
    own branch, landing in this same working tree at the same time, 2026-09-27) having grown a
    particular probe yet: any probe that is missing, not callable, or raises yields ``None``/
    ``"cpu"`` here rather than failing the pull that wants this context."""
    ram_total_mb: int | None = None
    framework = "cpu"
    try:
        from exocortex import gpu as _gpu

        vendor = _gpu.gpu_vendor()
        if vendor and vendor not in ("none", ""):
            framework = vendor
        sysmem = getattr(_gpu, "system_memory_mb", None)
        if callable(sysmem):
            ram_total_mb, _free = sysmem()
    except Exception:  # noqa: BLE001 - hardware probing must never fail a pull
        log.debug("library: gpu hardware probe failed", exc_info=True)

    physical_cores: int | None = None
    try:
        from exocortex.llm import server as _server

        physical_cores = _server._physical_cores()
    except Exception:  # noqa: BLE001
        log.debug("library: physical-core probe failed", exc_info=True)

    return ram_total_mb, physical_cores, framework


def auto_profile(model: LibraryModel, *, budget_mb: int | None, ram_total_mb: int | None,
                 physical_cores: int | None, gpu_framework: str, base: LLMConfig) -> ModelProfile:
    """Generate a tuning profile for `model` from a hardware budget.

    Without a GPU (`budget_mb` is ``None``): CPU-only, ctx 4096, no offload. Without GGUF metadata
    (`model.meta` is ``None`` -- a header this module could not parse): no opinion on ctx/layers,
    just the KV cache type and thread count; the global ``[llm]`` values stand for the rest. With
    both: the largest 2048-aligned context in ``[4096, min(n_ctx_train or 32768, 32768)]`` whose
    weights plus that much KV cache plus a 768 MiB activation margin fit 75% of `budget_mb`
    (leaving headroom for the display's own use of the same GPU); when even 4096 does not fit,
    partial GPU offload instead, sized so the offloaded layers' share of the weights plus a 4096
    KV cache still clears the same margin.

    `ram_total_mb`/`physical_cores` (from :func:`_hardware_context_for_profile`, or supplied by a
    caller with its own :mod:`exocortex.hardware` report) are not used by this branch's own math
    today (threads comes from :func:`exocortex.llm.server.default_threads` instead, which reads
    the machine directly) -- they are accepted here so a caller already holding a
    ``HardwareReport`` never has to compute anything twice, and so this signature does not need to
    change if a future refinement wants them. `base` is likewise accepted but unused today (kept
    for the same forward-compatibility reason, and for symmetry with this branch's other per-model
    constructors) -- nothing here reads or writes `base`'s own fields.
    """
    del ram_total_mb, physical_cores, base   # accepted for the API reasons in the docstring above

    from exocortex.llm import server as _server

    threads = _server.default_threads()
    is_cpu = (gpu_framework or "").strip().lower() == "cpu"
    kv_cache_type = "f16" if is_cpu else "q8_0"
    flash_attn = not is_cpu
    budget_txt = f"{budget_mb} MiB {gpu_framework}" if budget_mb is not None else "no GPU"
    generated_from = f"{budget_txt} budget, {date.today().isoformat()}"

    if budget_mb is None:
        return ModelProfile(
            ctx_size=4096, gpu_layers=0, kv_cache_type=kv_cache_type, flash_attn=flash_attn,
            threads=threads, parallel=None, auto=True,
            note=f"no GPU detected; {model.key} runs fully on CPU at a 4096-token context",
            generated_from=generated_from,
        )

    meta = model.meta
    if meta is None or meta.n_layer is None:
        return ModelProfile(
            ctx_size=None, gpu_layers=None, kv_cache_type=kv_cache_type, flash_attn=flash_attn,
            threads=threads, parallel=None, auto=True,
            note=f"no usable GGUF metadata for {model.key}; using the global [llm] context/layers",
            generated_from=generated_from,
        )

    weights = model.size_bytes
    budget_bytes = budget_mb * _MIB
    margin = int(0.75 * budget_bytes)
    kv_per_token = kv_bytes_per_token(meta, kv_cache_type)
    max_ctx = min(meta.n_ctx_train or 32768, 32768)

    def fits(ctx: int) -> bool:
        kv_bytes = 0 if kv_per_token is None else kv_per_token * ctx
        return weights + kv_bytes + 768 * _MIB <= margin

    ctx = max_ctx - (max_ctx % 2048)
    while ctx >= 4096 and not fits(ctx):
        ctx -= 2048

    if ctx >= 4096 and fits(ctx):
        return ModelProfile(
            ctx_size=ctx, gpu_layers=None, kv_cache_type=kv_cache_type, flash_attn=flash_attn,
            threads=threads, parallel=None, auto=True,
            note=(f"{model.key} ({weights / 1e9:.1f} GB) fits {budget_mb} MiB with room for a "
                  f"{ctx}-token KV cache; every layer runs on the GPU"),
            generated_from=generated_from,
        )

    kv_at_4096 = 0 if kv_per_token is None else kv_per_token * 4096
    remaining = margin - kv_at_4096 - 768 * _MIB
    gpu_layers = 0
    if weights > 0 and meta.n_layer:
        gpu_layers = max(0, min(meta.n_layer, math.floor(meta.n_layer * remaining / weights)))
    return ModelProfile(
        ctx_size=4096, gpu_layers=gpu_layers, kv_cache_type=kv_cache_type, flash_attn=flash_attn,
        threads=threads, parallel=None, auto=True,
        note=(f"{model.key} ({weights / 1e9:.1f} GB) does not fully fit {budget_mb} MiB even at a "
              f"4096-token context; offloading {gpu_layers}/{meta.n_layer} layers to the GPU"),
        generated_from=generated_from,
    )
