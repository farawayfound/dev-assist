"""Tests for exocortex.models_dl (all offline via httpx.MockTransport)."""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import httpx
import pytest

from exocortex import models_dl
from exocortex.models_dl import (
    MODEL_URLS,
    SHERPA_MODELS,
    ModelDownloadError,
    ensure_models,
    ensure_sherpa_model,
)

GOOD_CONTENT = b"x" * 200_000  # well above the 100 kB validation floor
SMALL_CONTENT = b"nope"  # too small -> must be rejected and cleaned up


def _install_mock_client(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Point exocortex.models_dl._make_client() at an httpx.MockTransport."""

    def factory() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)

    monkeypatch.setattr(models_dl, "_make_client", factory)


def test_ensure_models_downloads_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(200, content=GOOD_CONTENT)

    _install_mock_client(monkeypatch, handler)

    result = ensure_models(tmp_path, names=["silero_vad.onnx"])

    dest = tmp_path / "silero_vad.onnx"
    assert result == {"silero_vad.onnx": dest}
    assert dest.is_file()
    assert dest.stat().st_size == len(GOOD_CONTENT)
    assert not dest.with_name(dest.name + ".part").exists()
    assert requested_urls == [MODEL_URLS["silero_vad.onnx"]]


def test_ensure_models_never_redownloads_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = tmp_path / "silero_vad.onnx"
    existing.write_bytes(b"already-downloaded-bytes")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be called
        raise AssertionError("must not attempt a download for a file that already exists")

    _install_mock_client(monkeypatch, handler)

    result = ensure_models(tmp_path, names=["silero_vad.onnx"])

    assert result == {"silero_vad.onnx": existing}
    assert existing.read_bytes() == b"already-downloaded-bytes"


def test_ensure_models_defaults_to_every_known_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=GOOD_CONTENT)

    _install_mock_client(monkeypatch, handler)

    result = ensure_models(tmp_path)

    assert set(result) == set(MODEL_URLS)
    for name, path in result.items():
        assert path == tmp_path / name
        assert path.is_file()


def test_ensure_models_rejects_and_cleans_up_undersized_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=SMALL_CONTENT)

    _install_mock_client(monkeypatch, handler)

    with pytest.raises(ModelDownloadError):
        ensure_models(tmp_path, names=["silero_vad.onnx"])

    assert not (tmp_path / "silero_vad.onnx").exists()
    assert not (tmp_path / "silero_vad.onnx.part").exists()


def test_ensure_models_cleans_up_on_http_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"not found")

    _install_mock_client(monkeypatch, handler)

    with pytest.raises(ModelDownloadError):
        ensure_models(tmp_path, names=["silero_vad.onnx"])

    assert not (tmp_path / "silero_vad.onnx").exists()
    assert not (tmp_path / "silero_vad.onnx.part").exists()


def test_ensure_models_reports_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=GOOD_CONTENT, headers={"content-length": str(len(GOOD_CONTENT))}
        )

    _install_mock_client(monkeypatch, handler)

    events: list[tuple[str, int, int]] = []
    ensure_models(
        tmp_path, names=["silero_vad.onnx"], progress=lambda name, done, total: events.append((name, done, total))
    )

    assert events, "progress callback should have been invoked at least once"
    last_name, last_done, last_total = events[-1]
    assert last_name == "silero_vad.onnx"
    assert last_done == len(GOOD_CONTENT)
    assert last_total == len(GOOD_CONTENT)


def test_ensure_models_unknown_name_raises_key_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        ensure_models(tmp_path, names=["not_a_real_model.onnx"])


def test_model_urls_contains_expected_entries() -> None:
    assert MODEL_URLS["silero_vad.onnx"] == (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
    )
    # NOTE: "recongition" is an intentional reproduction of the upstream misspelling.
    assert "recongition" in MODEL_URLS["wespeaker_en_voxceleb_CAM++.onnx"]


# ---- 2026-09-26 (asr-sherpa): ensure_sherpa_model (archive download + tar.bz2 extraction) -------

SHERPA_NAME = "parakeet-tdt-0.6b-v2-int8"
_ARCHIVE_TOP_DIR = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"


def test_sherpa_models_contains_expected_entries() -> None:
    assert SHERPA_MODELS["parakeet-tdt-0.6b-v2-int8"] == (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
    )
    assert SHERPA_MODELS["parakeet-tdt-0.6b-v3-int8"] == (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
    )


def _make_tar_bz2(top_dir: str, files: dict[str, bytes]) -> bytes:
    """Builds a synthetic `.tar.bz2` in memory: `top_dir/<name>` -> `content` for each entry."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name=f"{top_dir}/{name}")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _make_tar_bz2_raw(members: dict[str, bytes]) -> bytes:
    """Like `_make_tar_bz2`, but `members` keys are used as literal tar entry names (no
    top-dir prefixing) -- for building archives with unsafe/malformed layouts."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


import os as _os  # noqa: E402 - local alias, used only for the random (incompressible) test payloads

_GOOD_SHERPA_FILES = {
    # Random, not repeated bytes: bz2 would otherwise compress a repeated byte string down under
    # the download's 100 kB validation floor, defeating the point of these files being "large".
    "encoder.int8.onnx": _os.urandom(150_000),
    "decoder.onnx": _os.urandom(150_000),
    "joiner.int8.onnx": _os.urandom(150_000),
    "tokens.txt": b"<blk> 0\na 1\n",
}


def test_ensure_sherpa_model_downloads_and_extracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_bytes = _make_tar_bz2(_ARCHIVE_TOP_DIR, _GOOD_SHERPA_FILES)
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(200, content=archive_bytes)

    _install_mock_client(monkeypatch, handler)

    model_dir = ensure_sherpa_model(SHERPA_NAME, tmp_path)

    assert model_dir == tmp_path / SHERPA_NAME
    assert requested_urls == [SHERPA_MODELS[SHERPA_NAME]]
    assert (model_dir / "encoder.int8.onnx").read_bytes() == _GOOD_SHERPA_FILES["encoder.int8.onnx"]
    assert (model_dir / "decoder.onnx").is_file()
    assert (model_dir / "joiner.int8.onnx").is_file()
    assert (model_dir / "tokens.txt").is_file()
    # the archive and its .part file must both be gone once extraction succeeds
    assert not (tmp_path / f"{SHERPA_NAME}.tar.bz2").exists()
    assert not (tmp_path / f"{SHERPA_NAME}.tar.bz2.part").exists()


def test_ensure_sherpa_model_reports_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_bytes = _make_tar_bz2(_ARCHIVE_TOP_DIR, _GOOD_SHERPA_FILES)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=archive_bytes, headers={"content-length": str(len(archive_bytes))})

    _install_mock_client(monkeypatch, handler)

    events: list[tuple[str, int, int]] = []
    ensure_sherpa_model(SHERPA_NAME, tmp_path, progress=lambda name, done, total: events.append((name, done, total)))

    assert events
    assert events[-1] == (SHERPA_NAME, len(archive_bytes), len(archive_bytes))


def test_ensure_sherpa_model_is_idempotent_when_already_extracted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / SHERPA_NAME
    model_dir.mkdir(parents=True)
    for name, content in _GOOD_SHERPA_FILES.items():
        (model_dir / name).write_bytes(content)

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be called
        raise AssertionError("must not attempt a download when the model dir is already complete")

    _install_mock_client(monkeypatch, handler)

    result = ensure_sherpa_model(SHERPA_NAME, tmp_path)

    assert result == model_dir
    # untouched: still the original bytes, not re-extracted
    assert (model_dir / "encoder.int8.onnx").read_bytes() == _GOOD_SHERPA_FILES["encoder.int8.onnx"]


def test_ensure_sherpa_model_rejects_path_traversal_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unsafe = dict(_GOOD_SHERPA_FILES)
    archive_bytes = _make_tar_bz2_raw(
        {f"{_ARCHIVE_TOP_DIR}/{k}": v for k, v in unsafe.items()}
        | {"../evil.txt": b"pwned"}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=archive_bytes)

    _install_mock_client(monkeypatch, handler)

    with pytest.raises(ModelDownloadError):
        ensure_sherpa_model(SHERPA_NAME, tmp_path)

    # cleaned up: neither the archive nor a partial extraction is left behind
    assert not (tmp_path / f"{SHERPA_NAME}.tar.bz2").exists()
    assert not (tmp_path / SHERPA_NAME).exists()
    assert not (tmp_path / f"{SHERPA_NAME}.extracting").exists()


def test_ensure_sherpa_model_rejects_absolute_path_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_bytes = _make_tar_bz2_raw(
        {f"{_ARCHIVE_TOP_DIR}/{k}": v for k, v in _GOOD_SHERPA_FILES.items()} | {"/etc/passwd": b"pwned"}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=archive_bytes)

    _install_mock_client(monkeypatch, handler)

    with pytest.raises(ModelDownloadError):
        ensure_sherpa_model(SHERPA_NAME, tmp_path)

    assert not (tmp_path / SHERPA_NAME).exists()


def test_ensure_sherpa_model_rejects_multiple_top_level_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_bytes = _make_tar_bz2_raw(
        {
            f"{_ARCHIVE_TOP_DIR}/encoder.int8.onnx": _GOOD_SHERPA_FILES["encoder.int8.onnx"],
            "some-other-dir/decoder.onnx": _GOOD_SHERPA_FILES["decoder.onnx"],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=archive_bytes)

    _install_mock_client(monkeypatch, handler)

    with pytest.raises(ModelDownloadError):
        ensure_sherpa_model(SHERPA_NAME, tmp_path)


def test_ensure_sherpa_model_rejects_undersized_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=SMALL_CONTENT)

    _install_mock_client(monkeypatch, handler)

    with pytest.raises(ModelDownloadError):
        ensure_sherpa_model(SHERPA_NAME, tmp_path)

    assert not (tmp_path / SHERPA_NAME).exists()
    assert not (tmp_path / f"{SHERPA_NAME}.tar.bz2").exists()
    assert not (tmp_path / f"{SHERPA_NAME}.tar.bz2.part").exists()


def test_ensure_sherpa_model_unknown_name_raises_key_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        ensure_sherpa_model("not-a-real-model", tmp_path)
