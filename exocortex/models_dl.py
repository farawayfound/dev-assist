"""Downloads the ONNX models used for VAD and speaker diarization, if not already present.

Only ever talks to the fixed URLs in ``MODEL_URLS`` (GitHub release assets); never re-downloads
a file that already exists locally.

2026-09-26 (asr-sherpa): ``SHERPA_MODELS`` / ``ensure_sherpa_model`` add a second, archive-based
download: sherpa-onnx ASR model bundles (encoder/decoder/joiner ONNX + tokens.txt) ship as a
single ``.tar.bz2`` per k2-fsa/sherpa-onnx's "asr-models" GitHub release, extracted under
``<models_dir>/<name>/`` -- see ``exocortex.speech.asr.SherpaTransducerModel``.
"""
from __future__ import annotations

import logging
import shutil
import tarfile
from collections.abc import Callable, Iterable
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

# NOTE: "recongition" (sic) is the upstream sherpa-onnx release-asset path; do not "fix" the typo.
MODEL_URLS: dict[str, str] = {
    "silero_vad.onnx": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
    "wespeaker_en_voxceleb_CAM++.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx"
    ),
}

# 2026-09-26 (asr-sherpa): sherpa-onnx ASR model archives (k2-fsa/sherpa-onnx release "asr-models").
# Both URLs verified with an httpx GET (follow_redirects=True, stream, no body read) on 2026-09-26:
# v2 (English only) 482468385 bytes, v3 (25 European languages) 487170055 bytes.
SHERPA_MODELS: dict[str, str] = {
    "parakeet-tdt-0.6b-v2-int8": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
    ),
    "parakeet-tdt-0.6b-v3-int8": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
    ),
}

# Files that must be present (by glob) for an extracted sherpa model directory to count as ready.
_SHERPA_REQUIRED_GLOBS = ("encoder*.onnx", "decoder*.onnx", "joiner*.onnx")

_MIN_SIZE_BYTES = 100_000  # 100 kB floor; catches HTML error pages saved in place of the real asset
_CHUNK_SIZE = 1 << 16
_TIMEOUT = httpx.Timeout(30.0, read=60.0)


class ModelDownloadError(RuntimeError):
    """Raised when a model file could not be downloaded or failed validation after download."""


def _make_client() -> httpx.Client:
    """Factory for the httpx client used to download models.

    A separate function so tests can monkeypatch it to return a client wired to
    ``httpx.MockTransport`` instead of making real network calls.
    """
    return httpx.Client(follow_redirects=True, timeout=_TIMEOUT)


def _download_one(name: str, url: str, dest: Path, progress: Callable[[str, int, int], None] | None) -> None:
    """Stream `url` to `dest`.part, then rename to `dest` once validated."""
    part = dest.with_name(dest.name + ".part")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with _make_client() as client, client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length") or 0)
            downloaded = 0
            with open(part, "wb") as f:
                for chunk in resp.iter_bytes(_CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(name, downloaded, total)
    except (httpx.HTTPError, OSError) as exc:
        part.unlink(missing_ok=True)
        raise ModelDownloadError(f"failed to download {name} from {url}: {exc}") from exc

    size = part.stat().st_size
    if size <= _MIN_SIZE_BYTES:
        part.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"downloaded {name} looks wrong ({size} bytes, expected > {_MIN_SIZE_BYTES}); "
            f"deleted the partial file. Check the URL: {url}"
        )
    part.replace(dest)
    log.info("downloaded %s (%d bytes) -> %s", name, size, dest)


def ensure_models(
    models_dir: Path,
    names: Iterable[str] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Path]:
    """Ensure each of `names` (default: every model in MODEL_URLS) exists under `models_dir`.

    Files that already exist are never re-downloaded. Missing files are streamed to a
    ``.part`` file and atomically renamed once their size clears a sanity floor (catches
    truncated downloads or HTML error pages saved in place of the real binary); a failed
    or invalid download deletes the partial file and raises ModelDownloadError.

    Returns {name: path} for every requested model.
    """
    models_dir = Path(models_dir)
    wanted = list(names) if names is not None else list(MODEL_URLS)
    result: dict[str, Path] = {}
    for name in wanted:
        if name not in MODEL_URLS:
            raise KeyError(f"unknown model name: {name!r}; known models: {sorted(MODEL_URLS)}")
        dest = models_dir / name
        if dest.is_file():
            log.debug("%s already present at %s; skipping download", name, dest)
        else:
            log.info("downloading %s ...", name)
            _download_one(name, MODEL_URLS[name], dest, progress)
        result[name] = dest
    return result


def _sherpa_model_ready(model_dir: Path) -> bool:
    """True if `model_dir` already holds a usable sherpa-onnx transducer model bundle."""
    if not model_dir.is_dir() or not (model_dir / "tokens.txt").is_file():
        return False
    return all(any(model_dir.glob(pattern)) for pattern in _SHERPA_REQUIRED_GLOBS)


def _safe_extract_tar_bz2(archive: Path, dest_dir: Path) -> None:
    """Extract `archive` (a `.tar.bz2` with exactly one top-level directory, as every
    k2-fsa/sherpa-onnx model release asset is packaged) into `dest_dir`, which ends up holding
    that top-level directory's contents directly (not nested one level down).

    Guards against path traversal: every member name is rejected outright if it is absolute or
    contains a `..` component, and (belt-and-suspenders) every member's resolved extraction path
    is checked to still be inside the temporary extraction root before `extractall` runs. Also
    rejects an archive that does not have exactly one top-level directory, since a differently
    laid-out archive would otherwise silently scatter files under `models_dir`.
    """
    tmp_root = dest_dir.parent / (dest_dir.name + ".extracting")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:bz2") as tar:
            members = tar.getmembers()
            top_level: set[str] = set()
            for member in members:
                parts = Path(member.name).parts
                if member.name.startswith("/") or ".." in parts or (parts and parts[0] in ("", ".")):
                    raise ModelDownloadError(f"refusing to extract unsafe archive member: {member.name!r}")
                if parts:
                    top_level.add(parts[0])
            if len(top_level) != 1:
                raise ModelDownloadError(
                    f"expected exactly one top-level directory in {archive}, found {sorted(top_level)}"
                )
            tmp_root_resolved = tmp_root.resolve()
            for member in members:
                target = (tmp_root / member.name).resolve()
                if tmp_root_resolved not in (target, *target.parents):
                    raise ModelDownloadError(f"refusing to extract outside target dir: {member.name!r}")
            try:
                tar.extractall(tmp_root, members=members, filter="data")
            except TypeError:  # pragma: no cover - Python < 3.12 has no `filter` kwarg
                tar.extractall(tmp_root, members=members)

        extracted = tmp_root / next(iter(top_level))
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        extracted.replace(dest_dir)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def ensure_sherpa_model(
    name: str,
    models_dir: Path,
    progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Ensure the sherpa-onnx ASR model bundle `name` (a key of `SHERPA_MODELS`) is extracted
    under `models_dir/<name>/`, and return that directory.

    Idempotent: a directory that already holds `encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx`
    and `tokens.txt` is returned at once, no network call made. Otherwise the archive is
    streamed to `<models_dir>/<name>.tar.bz2.part` with the same streaming/progress/validation
    as `_download_one` (via that same function), extracted with `_safe_extract_tar_bz2`, and the
    archive deleted once extraction succeeds -- an extraction failure also deletes it, so a
    corrupt or unsafe archive never lingers on disk to be picked up by a later idempotence check.
    """
    if name not in SHERPA_MODELS:
        raise KeyError(f"unknown sherpa model: {name!r}; known models: {sorted(SHERPA_MODELS)}")
    models_dir = Path(models_dir)
    model_dir = models_dir / name
    if _sherpa_model_ready(model_dir):
        log.debug("sherpa model %s already present at %s; skipping download", name, model_dir)
        return model_dir

    archive = models_dir / f"{name}.tar.bz2"
    log.info("downloading sherpa model %s ...", name)
    _download_one(name, SHERPA_MODELS[name], archive, progress)
    try:
        _safe_extract_tar_bz2(archive, model_dir)
    finally:
        archive.unlink(missing_ok=True)

    if not _sherpa_model_ready(model_dir):
        raise ModelDownloadError(f"extracted sherpa model {name} but {model_dir} is missing required files")
    log.info("extracted sherpa model %s -> %s", name, model_dir)
    return model_dir
