"""Fetch an official llama.cpp Windows runtime build as an alternative to LM Studio's packs.

2026-09-27 (portable-install). Until now this module was deliberately never asked for a CUDA or
ROCm kind -- NVIDIA and AMD machines always got their runtime from an installed LM Studio pack,
and this module only ever covered the flavors LM Studio's catalog does not (SYCL, OpenVINO) or
might ship an older build of (Vulkan, CPU). That assumption breaks the moment ExoCortex has to
install on a machine with NO LM Studio at all: the rule from here on is "an LM Studio pack wins
when one is present for a given GPU flavor; upstream covers every vendor -- NVIDIA and AMD
included -- when it is not". ``exocortex.llm.server`` decides WHEN to reach for upstream (LM
Studio pack first, this module only as the fallback); this module only knows how to fetch one.
``RUNTIME_KINDS`` therefore now includes ``"cuda12"``, ``"cuda13"`` and ``"rocm"`` alongside the
existing four. NVIDIA's CUDA 13.4 build needs driver >= 580 (see :func:`cuda_kind_for_driver`) and
is the only build with native kernels for Blackwell (RTX 50-series) GPUs -- CUDA 12.4 still runs
on Blackwell via PTX JIT, but 13.4 is preferred once the driver supports it. AMD's ROCm 10.0 build
requires no counterpart LM Studio pack lookup fallback of its own; it is one self-contained zip.

Older intel-integration note, still true for those three flavors: LM Studio's own runtime catalog
(``~/.lmstudio/.internal/internal-engine-index.json``) offers only CUDA, ROCm, Vulkan and CPU
packs -- there is no SYCL or OpenVINO pack, so an Intel Arc discrete GPU (A770/A750 "Alchemist",
B580/B570 "Battlemage", Pro B70) only ever gets LM Studio's Vulkan pack today, which matches the
llama.cpp community's current recommendation for Arc anyway (Vulkan over SYCL as of July 2026).
Nothing here is Intel-specific in its mechanics -- it downloads, verifies and unpacks one or more
zips -- the Intel tie-in is entirely in which `kind` the caller asks for and in
:func:`vulkan_env_switches`, which the doctor uses to surface the Alchemist-era Vulkan environment
switches (see below).

This is a SEPARATE runtime source from ``exocortex.llm.server`` (LM Studio packs under
``~/.lmstudio/extensions/backends``): callers that want an upstream build ask this module for a
folder holding ``llama-server.exe`` and hand that folder to ``server.py`` the same way an explicit
``[llm].llama_server_path`` is used today (this module never touches ``server.py`` or
``config.py`` -- see SPEC.md's "GPU vendor routing" section for the contract boundary).

Integrity is mandatory, not advisory: every asset this module installs is verified by BOTH its
exact byte size and its sha256 before extraction, and a build other than the pinned one is only
ever installed from an asset GitHub's release API itself attests a sha256 digest for (its
``"digest": "sha256:..."`` field) -- an asset with no digest is refused outright, never installed
on a "probably fine" basis. Extraction guards against zip-slip (path traversal), absolute member
names, and unix-symlink-mode entries, the same threat model as ``exocortex.models_dl``'s tar.bz2
extraction, adapted for zip's ``ZipInfo`` (no equivalent "exactly one top-level directory" rule,
since upstream's zips are not consistent about wrapping their contents in one -- see
``installed_runtime``, which searches one level deep for whichever layout shows up).

Most kinds are one zip. ``cuda12`` and ``cuda13`` are each TWO zips -- the ``llama-*`` binaries
plus a same-release ``cudart-llama-*`` zip carrying the matching CUDA runtime DLLs (``cudart64_*``,
``cublas64_*``, ...) that upstream does not statically link -- so :data:`PINNED_ASSETS` maps every
kind to a *tuple* of assets, and :func:`ensure_runtime` downloads, verifies and extracts each one
of a kind into the same install directory before checking once for ``llama-server.exe``.

``PINNED_ASSETS`` records the Windows assets of build b11200 (2026-09-26's release) with their
sizes and sha256 digests, fetched by hand from
``https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11200`` -- installing the pinned
build never makes a network call before the download itself. Any other `build` resolves its
assets from that same release API at call time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import stat
import time
import uuid
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

#: What can be fetched. "sycl" and "openvino" exist upstream but not in LM Studio's catalog (see
#: module docstring); "vulkan", "cpu", "cuda12"/"cuda13" (CUDA 12.4 / 13.4) and "rocm" mirror LM
#: Studio's own flavor names for the same frameworks so a caller can compare a
#: `ServerBinary.framework` against a `kind` directly -- "cuda12"/"cuda13" both map to framework
#: "cuda" the way LM Studio's two CUDA packs do.
RUNTIME_KINDS: tuple[str, ...] = ("vulkan", "sycl", "openvino", "cpu", "cuda12", "cuda13", "rocm")

#: The llama.cpp release this module knows the asset sizes/digests for without a network call.
PINNED_BUILD = "b11200"

_GITHUB_REPO = "ggml-org/llama.cpp"
_RELEASES_BASE = f"https://github.com/{_GITHUB_REPO}/releases/download"
_API_RELEASE_TAG = f"https://api.github.com/repos/{_GITHUB_REPO}/releases/tags/{{build}}"

_CHUNK_SIZE = 1 << 16
_TIMEOUT = httpx.Timeout(30.0, read=60.0)

_SERVER_EXE = "llama-server.exe"


class UpstreamRuntimeError(RuntimeError):
    """Raised when an upstream llama.cpp runtime build cannot be resolved, downloaded, verified,
    or extracted. This module never raises anything else but this and OSError."""


class UpstreamRuntimeCancelled(UpstreamRuntimeError):
    """The caller's `cancelled()` hook returned True mid-download (the app is quitting): the
    partial file is deleted and nothing is installed. A subclass so callers that only care about
    "did it fail" keep working, while :class:`exocortex.llm.server.LlamaServerManager` can tell a
    quit apart from a real download failure and stop instead of trying the next runtime pack."""


@dataclass(frozen=True)
class RuntimeAsset:
    """One llama.cpp release zip: enough to download it and prove it arrived intact."""

    kind: str       # one of RUNTIME_KINDS
    build: str      # e.g. "b11200"
    name: str       # e.g. "llama-b11200-bin-win-vulkan-x64.zip"
    url: str
    size: int       # expected exact byte size
    sha256: str     # expected hex digest, lowercase


def _pinned_url(name: str) -> str:
    return f"{_RELEASES_BASE}/{PINNED_BUILD}/{name}"


#: b11200 Windows assets (2026-09-26 / 2026-09-27), fetched from the GitHub release API by hand --
#: see module docstring. Sizes in bytes, sha256 as published in each asset's "digest" field. Every
#: kind maps to a TUPLE of assets: one for the four single-zip kinds, two for "cuda12"/"cuda13"
#: (the llama-* binaries plus the same-release cudart-llama-* runtime-DLL zip).
PINNED_ASSETS: dict[str, tuple[RuntimeAsset, ...]] = {
    "vulkan": (
        RuntimeAsset(
            kind="vulkan", build=PINNED_BUILD, name="llama-b11200-bin-win-vulkan-x64.zip",
            url=_pinned_url("llama-b11200-bin-win-vulkan-x64.zip"),
            size=33_062_078,
            sha256="670a1f9bf5272c43dbbd7b0004a12dd3d3f6f575246fd23b733f06c064f1b96b",
        ),
    ),
    "sycl": (
        RuntimeAsset(
            kind="sycl", build=PINNED_BUILD, name="llama-b11200-bin-win-sycl-x64.zip",
            url=_pinned_url("llama-b11200-bin-win-sycl-x64.zip"),
            size=120_815_015,
            sha256="4617886da18964323846d509a46ec5a32dac6d9ece633a292531b3b11dbcd965",
        ),
    ),
    "openvino": (
        RuntimeAsset(
            kind="openvino", build=PINNED_BUILD, name="llama-b11200-bin-win-openvino-2026.4-x64.zip",
            url=_pinned_url("llama-b11200-bin-win-openvino-2026.4-x64.zip"),
            size=88_333_295,
            sha256="fb9d5374886aa3d5719a48a2cf15b695b34344e94454a7477db1bfc6959a173b",
        ),
    ),
    "cpu": (
        RuntimeAsset(
            kind="cpu", build=PINNED_BUILD, name="llama-b11200-bin-win-cpu-x64.zip",
            url=_pinned_url("llama-b11200-bin-win-cpu-x64.zip"),
            size=19_154_722,
            sha256="b958c2f249b59335048a57993802b292faeffaee55555b71e59616d6c2c399ca",
        ),
    ),
    "cuda12": (
        RuntimeAsset(
            kind="cuda12", build=PINNED_BUILD, name="llama-b11200-bin-win-cuda-12.4-x64.zip",
            url=_pinned_url("llama-b11200-bin-win-cuda-12.4-x64.zip"),
            size=263_038_371,
            sha256="8f9fcdb185dcd99a63cdfa3716f1ab12584060baef96a48bc6db157a4db843d1",
        ),
        RuntimeAsset(
            kind="cuda12", build=PINNED_BUILD, name="cudart-llama-bin-win-cuda-12.4-x64.zip",
            url=_pinned_url("cudart-llama-bin-win-cuda-12.4-x64.zip"),
            size=391_443_627,
            sha256="8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
        ),
    ),
    "cuda13": (
        RuntimeAsset(
            kind="cuda13", build=PINNED_BUILD, name="llama-b11200-bin-win-cuda-13.4-x64.zip",
            url=_pinned_url("llama-b11200-bin-win-cuda-13.4-x64.zip"),
            size=152_319_780,
            sha256="ac88b6102fb9cb6344f897ddfa7400e67ff8d687ad34c124ca5764293ef5ef3f",
        ),
        RuntimeAsset(
            kind="cuda13", build=PINNED_BUILD, name="cudart-llama-bin-win-cuda-13.4-x64.zip",
            url=_pinned_url("cudart-llama-bin-win-cuda-13.4-x64.zip"),
            size=423_535_356,
            sha256="738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
        ),
    ),
    "rocm": (
        RuntimeAsset(
            kind="rocm", build=PINNED_BUILD, name="llama-b11200-bin-win-rocm-10.0-x64.zip",
            url=_pinned_url("llama-b11200-bin-win-rocm-10.0-x64.zip"),
            size=257_329_562,
            sha256="ca0aabc00240dd4588cbf7afb65d4df311458bb2f6391884aec00269ed652ef2",
        ),
    ),
}

#: Asset-name patterns used to pick the right asset(s) out of a release, mirroring the naming
#: upstream has used since it started shipping per-flavor Windows zips: "win-<kind>-x64", except
#: OpenVINO (whose name carries the OpenVINO package version, "win-openvino-<ver>-x64") and
#: CUDA (whose name carries the CUDA toolkit version, "win-cuda-<ver>-x64", and which needs BOTH
#: the "llama-*" binaries and the same-release "cudart-llama-*" runtime-DLL zip -- see
#: :func:`_fetch_release_asset`, which requires every pattern of a kind to match exactly one
#: digest-bearing asset). Each kind maps to a tuple of one-or-more patterns.
_ASSET_NAME_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "vulkan": (re.compile(r"win-vulkan-x64\.zip$", re.IGNORECASE),),
    "sycl": (re.compile(r"win-sycl-x64\.zip$", re.IGNORECASE),),
    "cpu": (re.compile(r"win-cpu-x64\.zip$", re.IGNORECASE),),
    "openvino": (re.compile(r"win-openvino-[\d.]+-x64\.zip$", re.IGNORECASE),),
    "cuda12": (
        re.compile(r"^llama-.*-bin-win-cuda-12\.\d+-x64\.zip$", re.IGNORECASE),
        re.compile(r"^cudart-llama-bin-win-cuda-12\.\d+-x64\.zip$", re.IGNORECASE),
    ),
    "cuda13": (
        re.compile(r"^llama-.*-bin-win-cuda-13\.\d+-x64\.zip$", re.IGNORECASE),
        re.compile(r"^cudart-llama-bin-win-cuda-13\.\d+-x64\.zip$", re.IGNORECASE),
    ),
    "rocm": (re.compile(r"win-rocm-[\d.]+-x64\.zip$", re.IGNORECASE),),
}

#: The filename of the small per-install manifest written under a runtime's top-level directory
#: (``<root>/<build>/<kind>/.exocortex-runtime.json``) recording every asset installed for it, so
#: :func:`describe_runtime` can list them back even for a multi-zip kind like "cuda12". Absent for
#: a runtime installed before this manifest existed (or one dropped in by hand) -- callers must
#: treat that as "unknown assets", never as an error.
_MANIFEST_NAME = ".exocortex-runtime.json"


def _client(http_client_factory: "Callable[[], httpx.Client] | None") -> httpx.Client:
    """The httpx client used for both the GitHub API call and the download itself. A separate
    function so tests can inject a client wired to ``httpx.MockTransport``."""
    if http_client_factory is not None:
        return http_client_factory()
    return httpx.Client(follow_redirects=True, timeout=_TIMEOUT)


def runtime_dir(root: Path, kind: str, build: str = PINNED_BUILD) -> Path:
    """Where `kind`'s `build` is (or would be) installed: ``<root>/<build>/<kind>``."""
    return Path(root) / build / kind


def _find_server_exe_dir(rdir: Path) -> Path | None:
    """The folder holding ``llama-server.exe`` directly under `rdir`, or one level down, or
    ``None``. Shared by :func:`installed_runtime` (checking a final ``runtime_dir``) and
    :func:`ensure_runtime` (checking a private extraction-temp directory before it is renamed
    into place -- see the module docstring's note on upstream's inconsistent zip layouts).
    """
    if (rdir / _SERVER_EXE).is_file():
        return rdir
    if not rdir.is_dir():
        return None
    try:
        children = sorted(p for p in rdir.iterdir() if p.is_dir())
    except OSError:
        return None
    for child in children:
        if (child / _SERVER_EXE).is_file():
            return child
    return None


def installed_runtime(root: Path, kind: str, build: str = PINNED_BUILD) -> Path | None:
    """The folder holding ``llama-server.exe`` for `kind`/`build`, or ``None``.

    Upstream's zips are not consistent about wrapping their contents in one top-level folder --
    some unpack straight into the root, some nest one level down -- so this checks
    :func:`runtime_dir` itself first, then searches one level deep.
    """
    return _find_server_exe_dir(runtime_dir(root, kind, build))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_release_asset(kind: str, build: str,
                          http_client_factory: "Callable[[], httpx.Client] | None") -> tuple[RuntimeAsset, ...]:
    """Resolve `kind`'s asset(s) for `build` from GitHub's release API: one release fetch, then
    every pattern in :data:`_ASSET_NAME_PATTERNS[kind]` (two for "cuda12"/"cuda13" -- the llama-*
    binaries and the matching cudart-llama-* runtime-DLL zip; one for every other kind) must match
    exactly one asset, in pattern order. Refuses (raises :class:`UpstreamRuntimeError`) any asset
    the API does not itself attest a sha256 digest for -- this is the only integrity guarantee
    available before download, so it is not optional -- and names whichever pattern (e.g. the
    cudart-* one) found no digest-bearing match, so a caller can tell which of the kind's assets
    is missing from the release."""
    url = _API_RELEASE_TAG.format(build=build)
    try:
        with _client(http_client_factory) as client:
            resp = client.get(url, headers={"Accept": "application/vnd.github+json"})
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise UpstreamRuntimeError(f"could not fetch llama.cpp release {build!r} from GitHub: {exc}") from exc

    assets = data.get("assets") or []
    resolved: list[RuntimeAsset] = []
    for pattern in _ASSET_NAME_PATTERNS[kind]:
        match = None
        for a in assets:
            name = str(a.get("name") or "")
            if pattern.search(name):
                match = a
                break
        if match is None:
            raise UpstreamRuntimeError(
                f"no asset found in llama.cpp release {build!r} matching {pattern.pattern!r} "
                f"(needed for kind {kind!r})"
            )
        name = str(match.get("name") or "")
        digest = str(match.get("digest") or "")
        if not digest.lower().startswith("sha256:"):
            raise UpstreamRuntimeError(
                f"refusing to install {name}: GitHub's release API provides no sha256 digest for "
                "this asset"
            )
        asset_url = str(match.get("browser_download_url") or f"{_RELEASES_BASE}/{build}/{name}")
        size = int(match.get("size") or 0)
        resolved.append(RuntimeAsset(kind=kind, build=build, name=name, url=asset_url, size=size,
                                      sha256=digest.split(":", 1)[1].strip().lower()))
    return tuple(resolved)


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract `zip_path` into `dest_dir`, preserving whatever internal layout the zip has
    (unlike ``models_dl``'s tar.bz2 helper, upstream's zips are not guaranteed to wrap their
    contents in a single top-level directory -- see :func:`installed_runtime`).

    Every member is validated BEFORE any extraction happens (so a bad member later in the archive
    never leaves earlier members written to disk): rejects absolute paths, ``..`` traversal, and
    unix-symlink-mode entries (a zip can carry these via ``external_attr`` even though Windows
    itself has no concept of a zip symlink), and re-checks with a resolved-path containment test
    as belt-and-suspenders.
    """
    dest_dir = Path(dest_dir)
    dest_resolved = dest_dir.resolve()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.infolist()
            for info in members:
                name = info.filename
                parts = Path(name).parts
                if not name or name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name) or ".." in parts:
                    raise UpstreamRuntimeError(f"refusing to extract unsafe zip member: {name!r}")
                mode = (info.external_attr >> 16) & 0xFFFF
                if mode and stat.S_ISLNK(mode):
                    raise UpstreamRuntimeError(f"refusing to extract symlink zip member: {name!r}")
                target = (dest_dir / name).resolve()
                if dest_resolved != target and dest_resolved not in target.parents:
                    raise UpstreamRuntimeError(f"refusing to extract outside target dir: {name!r}")
            for info in members:
                zf.extract(info, dest_dir)
    except zipfile.BadZipFile as exc:
        raise UpstreamRuntimeError(f"{zip_path} is not a valid zip: {exc}") from exc


def assets_for(kind: str, build: str = PINNED_BUILD, *,
                http_client_factory: "Callable[[], httpx.Client] | None" = None) -> tuple[RuntimeAsset, ...]:
    """The asset(s) `kind`'s `build` installs from: :data:`PINNED_ASSETS` for the pinned build
    (no network call), else a live lookup via :func:`_fetch_release_asset`. Raises
    :class:`UpstreamRuntimeError` for an unknown `kind` or an unresolvable release."""
    if kind not in RUNTIME_KINDS:
        raise UpstreamRuntimeError(f"unknown runtime kind {kind!r}; must be one of {RUNTIME_KINDS}")
    if build == PINNED_BUILD and kind in PINNED_ASSETS:
        return PINNED_ASSETS[kind]
    return _fetch_release_asset(kind, build, http_client_factory)


def cuda_kind_for_driver(driver: "str | None") -> str:
    """Which CUDA runtime `kind` ("cuda12" or "cuda13") an NVIDIA driver version can run: "cuda13"
    once the driver's leading major version is >= 580 (the minimum for CUDA 13.4, and the only
    build with native Blackwell/RTX-50 kernels -- see module docstring), else "cuda12". A `driver`
    that is ``None``, empty, or does not start with an integer (an unparseable string from
    :func:`exocortex.gpu.nvidia_driver_version`) conservatively resolves to "cuda12"."""
    if driver:
        m = re.match(r"^\s*(\d+)", driver)
        if m and int(m.group(1)) >= 580:
            return "cuda13"
    return "cuda12"


def ensure_runtime(kind: str, root: Path, *, build: str = PINNED_BUILD,
                    progress: "Callable[[str, int, int], None] | None" = None,
                    http_client_factory: "Callable[[], httpx.Client] | None" = None,
                    asset: "RuntimeAsset | Sequence[RuntimeAsset] | None" = None,
                    cancelled: "Callable[[], bool] | None" = None) -> Path:
    """Ensure `kind`'s `build` is installed under `root`; return the folder holding
    ``llama-server.exe``.

    Idempotent: an already-installed runtime (:func:`installed_runtime`) is returned at once, no
    network call made. Otherwise: `asset` if given (a single :class:`RuntimeAsset` or a sequence
    of them -- normalised to a tuple either way; most kinds are one asset, "cuda12"/"cuda13" are
    two), else :func:`assets_for` (:data:`PINNED_ASSETS` for the pinned build, else a live lookup
    via :func:`_fetch_release_asset`).

    Download and extraction happen in a private per-attempt temp directory
    (``<root>/<build>/.<kind>-<uuid>.tmp``), never directly in the final ``runtime_dir`` -- so a
    concurrent caller's own :func:`installed_runtime` check can never observe a partially
    downloaded or partially extracted runtime as installed, and two callers racing to install the
    same `kind`/`build` never write into the same zip/part file or extract into the same
    destination at the same time. Every asset of the kind is downloaded, verified and extracted
    into that SAME temp directory, one after another (each streamed to
    ``<tmp_dir>/<name>.part`` with per-asset `progress` callbacks -- `name` is that asset's zip);
    a mismatch on either byte size or sha256 for ANY asset deletes that asset's ``.part`` and
    raises naming expected vs. actual, before anything of that asset is extracted, and aborts the
    whole install (the assets already extracted from an earlier asset of the same kind are
    discarded along with the rest of the temp directory -- see below). Extraction
    (:func:`_safe_extract_zip`) guards against path traversal; only AFTER every asset of the kind
    has extracted does this check once for ``llama-server.exe`` (one level deep -- see
    :func:`installed_runtime`) -- if it is not found, the temp directory is removed and this
    raises rather than leaving a half-installed runtime that looks present. Only once a complete,
    verified runtime is sitting in the temp directory is it moved into its final ``runtime_dir``
    location with a single atomic ``Path.replace()`` -- which never overwrites an existing
    directory (Windows refuses that), so if a concurrent caller already finished installing the
    same `kind`/`build` first, this loses the race harmlessly: it discards its own (redundant)
    temp directory and returns the winner's folder instead of raising. On a successful install, a
    small manifest (``<rdir>/.exocortex-runtime.json`` -- see :data:`_MANIFEST_NAME`) records
    every asset's filename, for :func:`describe_runtime` to read back later.

    `cancelled` (optional) is polled before each asset's download, after every chunk and before
    each asset's extraction: when it returns True -- :class:`exocortex.llm.server.LlamaServerManager.stop`
    sets its stop event while a fetch is in flight -- the partial file and the temp directory are
    removed and :class:`UpstreamRuntimeCancelled` is raised within one chunk (64 KiB) of the
    request, whichever asset of the kind was in flight at the time.

    Every filesystem operation that creates a directory or touches disk beyond the temp
    directory's own lifetime (creating ``<root>/<build>``, creating the temp directory itself,
    extracting into it) is wrapped so a raw ``OSError`` (e.g. a non-writable install root,
    disk full, or a Windows ``MAX_PATH`` violation on a deeply nested zip member) is always
    reported as :class:`UpstreamRuntimeError` with the temp directory cleaned up first, never left
    to propagate as a bare ``OSError`` with debris on disk.
    """
    if kind not in RUNTIME_KINDS:
        raise UpstreamRuntimeError(f"unknown runtime kind {kind!r}; must be one of {RUNTIME_KINDS}")
    root = Path(root)

    existing = installed_runtime(root, kind, build)
    if existing is not None:
        log.debug("llm runtimes: %s %s already installed at %s", build, kind, existing)
        return existing

    if asset is None:
        resolved: tuple[RuntimeAsset, ...] = assets_for(kind, build, http_client_factory=http_client_factory)
    elif isinstance(asset, RuntimeAsset):
        resolved = (asset,)
    else:
        resolved = tuple(asset)

    rdir = runtime_dir(root, kind, build)
    build_dir = rdir.parent
    tmp_dir = build_dir / f".{kind}-{uuid.uuid4().hex}.tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise UpstreamRuntimeError(f"could not create install directory {tmp_dir}: {exc}") from exc

    try:
        # The returned folder is the exe-holding directory INSIDE tmp_dir; we rename tmp_dir
        # itself (not that inner folder) onto rdir below, so runtime_dir()/installed_runtime()'s
        # one-level-deep search still works the same way against the final location as it would
        # have before this change.
        _download_and_extract(resolved, tmp_dir, progress, http_client_factory, cancelled)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    _write_manifest(tmp_dir, kind, build, resolved)

    try:
        tmp_dir.replace(rdir)
    except OSError as exc:
        # Someone else won the race and already installed this kind/build. Windows refuses to
        # replace an existing directory, so this is the expected way to lose that race -- prefer
        # the winner's install over our own (equally valid, now-redundant) one.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        winner = installed_runtime(root, kind, build)
        if winner is not None:
            return winner
        raise UpstreamRuntimeError(f"could not install {kind} {build} into {rdir}: {exc}") from exc

    folder = installed_runtime(root, kind, build)
    if folder is None:  # pragma: no cover - _download_and_extract already guarantees this
        shutil.rmtree(rdir, ignore_errors=True)
        names = ", ".join(a.name for a in resolved)
        raise UpstreamRuntimeError(f"installed {names} but no {_SERVER_EXE} found under {rdir}")
    return folder


def _write_manifest(tmp_dir: Path, kind: str, build: str, assets: tuple[RuntimeAsset, ...]) -> None:
    """Record every asset installed for this attempt as ``<tmp_dir>/.exocortex-runtime.json``,
    written into the per-attempt temp directory so it moves into place with the same atomic
    ``Path.replace()`` as the rest of the install. Best-effort: a write failure here must never
    fail an otherwise-successful install, so any :class:`OSError` is swallowed."""
    manifest = {"kind": kind, "build": build, "assets": [a.name for a in assets]}
    try:
        (tmp_dir / _MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    except OSError:
        log.debug("llm runtimes: could not write %s under %s", _MANIFEST_NAME, tmp_dir, exc_info=True)


def _download_and_extract(resolved: "tuple[RuntimeAsset, ...]", tmp_dir: Path,
                           progress: "Callable[[str, int, int], None] | None",
                           http_client_factory: "Callable[[], httpx.Client] | None",
                           cancelled: "Callable[[], bool] | None" = None) -> Path:
    """Download and extract every asset in `resolved` into `tmp_dir`, in order, then return the
    folder that ends up holding ``llama-server.exe``. Raises :class:`UpstreamRuntimeError` for
    every failure mode (download, integrity or extraction of any one asset, or a missing exe once
    all of them have extracted) -- callers are responsible for cleaning up `tmp_dir` itself, since
    a caller may want to inspect it first. The exe check happens exactly once, after the LAST
    asset -- most kinds have exactly one asset (which typically carries the exe itself), but
    "cuda12"/"cuda13" have two (the llama-* binaries, then a cudart-llama-* runtime-DLL zip that
    carries no exe of its own), and checking after every asset would wrongly fail on the first."""
    for one in resolved:
        _download_and_extract_one(one, tmp_dir, progress, http_client_factory, cancelled)

    folder = _find_server_exe_dir(tmp_dir)
    if folder is None:
        names = ", ".join(a.name for a in resolved)
        raise UpstreamRuntimeError(f"extracted {names} but no {_SERVER_EXE} found afterwards")
    return folder


def _download_and_extract_one(resolved: RuntimeAsset, tmp_dir: Path,
                               progress: "Callable[[str, int, int], None] | None",
                               http_client_factory: "Callable[[], httpx.Client] | None",
                               cancelled: "Callable[[], bool] | None" = None) -> None:
    """Download one `resolved` asset and extract it into `tmp_dir`. Raises
    :class:`UpstreamRuntimeError` (or :class:`UpstreamRuntimeCancelled`) for a download, integrity
    or extraction failure of THIS asset; does not check for ``llama-server.exe`` itself -- that is
    :func:`_download_and_extract`'s job, once every asset of the kind has extracted."""
    zip_path = tmp_dir / resolved.name
    part_path = zip_path.with_name(zip_path.name + ".part")

    def _check_cancelled() -> None:
        # Raises only; the .part is unlinked by the caller AFTER the file handle is closed --
        # deleting an open file is refused on Windows (WinError 32).
        if cancelled is not None and cancelled():
            raise UpstreamRuntimeCancelled(f"download of {resolved.name} cancelled after {downloaded} bytes")

    t0 = time.monotonic()
    downloaded = 0
    _check_cancelled()
    try:
        with _client(http_client_factory) as client, client.stream("GET", resolved.url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length") or resolved.size or 0)
            with open(part_path, "wb") as f:
                for chunk in resp.iter_bytes(_CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(resolved.name, downloaded, total)
                    _check_cancelled()
    except UpstreamRuntimeCancelled:
        part_path.unlink(missing_ok=True)
        raise
    except (httpx.HTTPError, OSError) as exc:
        part_path.unlink(missing_ok=True)
        raise UpstreamRuntimeError(f"failed to download {resolved.name} from {resolved.url}: {exc}") from exc
    _check_cancelled()
    elapsed = time.monotonic() - t0

    actual_size = part_path.stat().st_size
    if resolved.size and actual_size != resolved.size:
        part_path.unlink(missing_ok=True)
        raise UpstreamRuntimeError(
            f"{resolved.name}: size mismatch (expected {resolved.size} bytes, got {actual_size})"
        )
    actual_sha256 = _sha256_file(part_path)
    if resolved.sha256 and actual_sha256.lower() != resolved.sha256.lower():
        part_path.unlink(missing_ok=True)
        raise UpstreamRuntimeError(
            f"{resolved.name}: sha256 mismatch (expected {resolved.sha256}, got {actual_sha256})"
        )
    part_path.replace(zip_path)
    log.info("llm runtimes: downloaded %s (%.1f MB) in %.1fs", resolved.name, actual_size / 1e6, elapsed)

    try:
        _safe_extract_zip(zip_path, tmp_dir)
    except OSError as exc:
        raise UpstreamRuntimeError(f"failed to extract {resolved.name}: {exc}") from exc
    finally:
        zip_path.unlink(missing_ok=True)


def describe_runtime(folder: Path) -> str:
    """A one-line human description of an installed runtime, e.g.
    ``"llama.cpp b11200 vulkan (upstream, C:\\...\\b11200\\vulkan) [llama-b11200-bin-win-vulkan-x64.zip]"``.

    `kind`/`build` are read back from the folder's path (see :func:`runtime_dir` /
    :func:`installed_runtime` for the two layouts this recognises); when a ``build-info*`` file
    exists in the folder (some llama.cpp packages carry one), its first non-empty line overrides
    the path-derived build number, since that is the more authoritative source when it exists.
    When the runtime's top-level directory carries the ``.exocortex-runtime.json`` manifest
    :func:`ensure_runtime` writes at install time (see :data:`_MANIFEST_NAME`), every asset name
    it lists is appended -- this is the only way to tell a multi-zip kind like "cuda12" apart from
    a single-zip one in the description. A runtime installed before this manifest existed (or
    dropped in by hand) has none, and the description is exactly as it was before this manifest
    existed: no trailing bracket, never an error.
    """
    folder = Path(folder)
    kind = "unknown"
    build = "unknown"
    rdir = folder
    if folder.name in RUNTIME_KINDS:
        kind = folder.name
        build = folder.parent.name or build
        rdir = folder
    elif folder.parent.name in RUNTIME_KINDS:
        kind = folder.parent.name
        build = folder.parent.parent.name or build
        rdir = folder.parent
    try:
        for candidate in sorted(folder.glob("build-info*")):
            if not candidate.is_file():
                continue
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
            first_line = text.splitlines()[0].strip() if text else ""
            if first_line:
                build = first_line
            break
    except OSError:
        pass
    desc = f"llama.cpp {build} {kind} (upstream, {folder})"
    assets = _read_manifest_assets(rdir)
    if assets:
        desc += f" [{', '.join(assets)}]"
    return desc


def _read_manifest_assets(rdir: Path) -> list[str]:
    """The asset filenames :func:`ensure_runtime` recorded for `rdir` in its
    ``.exocortex-runtime.json`` manifest, or ``[]`` when it is absent, unreadable, or malformed --
    never raises, since a missing manifest (any runtime installed before it existed) is the
    expected, common case, not an error."""
    try:
        text = (rdir / _MANIFEST_NAME).read_text(encoding="utf-8")
        data = json.loads(text)
        names = data.get("assets") or []
        return [str(n) for n in names]
    except (OSError, ValueError):
        return []


_VK_ENV_RE = re.compile(rb"GGML_VK_[A-Z0-9_]+")


def vulkan_env_switches(folder: Path) -> list[str]:
    """The ``GGML_VK_*`` environment switches this Vulkan build actually recognises, scanned
    straight out of ``ggml-vulkan*.dll``'s bytes (no header ships these officially -- this is
    llama.cpp's own documented way of discovering them, and the only way that stays correct as
    upstream adds or renames switches between builds). ``[]`` if `folder` has no such DLL.

    Older Intel Arc "Alchemist" drivers have had fp16/garbage-output issues in llama.cpp's Vulkan
    backend that some of these switches exist to work around (e.g. disabling f16 or cooperative
    matrix); this function does not know which switches mean what, only which strings the binary
    itself will actually read from the environment -- the doctor prints them for that reason.
    """
    folder = Path(folder)
    found: set[str] = set()
    try:
        dlls = list(folder.glob("ggml-vulkan*.dll"))
    except OSError:
        return []
    for dll in dlls:
        try:
            data = dll.read_bytes()
        except OSError:
            continue
        for m in _VK_ENV_RE.finditer(data):
            found.add(m.group().decode("ascii"))
    return sorted(found)
