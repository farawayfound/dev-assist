"""Tests for exocortex.llm.runtimes (all offline via httpx.MockTransport and synthetic zips)."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from exocortex.llm import runtimes
from exocortex.llm.runtimes import (
    PINNED_ASSETS,
    PINNED_BUILD,
    RUNTIME_KINDS,
    RuntimeAsset,
    UpstreamRuntimeError,
    assets_for,
    cuda_kind_for_driver,
    describe_runtime,
    UpstreamRuntimeCancelled,
    ensure_runtime,
    installed_runtime,
    runtime_dir,
    vulkan_env_switches,
)


def _make_zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _asset_for(kind: str, build: str, data: bytes, url: str = "https://example.invalid/x.zip") -> RuntimeAsset:
    return RuntimeAsset(
        kind=kind, build=build, name="llama-test-bin-win-vulkan-x64.zip", url=url,
        size=len(data), sha256=hashlib.sha256(data).hexdigest(),
    )


def _client_factory(handler):
    def factory() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    return factory


def _zip_dispatch_handler(zips: dict[str, bytes]):
    """A MockTransport handler that serves whichever of `zips` the request URL names (by asset
    name being a substring of the URL) -- used for the multi-asset cuda12/cuda13 tests, where
    each asset has its own URL."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for name, data in zips.items():
            if name in url:
                return httpx.Response(200, content=data, headers={"content-length": str(len(data))})
        raise AssertionError(f"unexpected request: {url}")
    return handler


#: cuda12's two assets: the llama-* binaries (carries llama-server.exe) and the cudart-llama-*
#: runtime-DLL zip (carries none -- see the module docstring on why the exe check happens only
#: once, after every asset of the kind has extracted).
CUDA_LLAMA_ZIP = _make_zip_bytes({"llama-server.exe": b"fake cuda exe", "ggml-cuda.dll": b"dll"})
CUDA_CUDART_ZIP = _make_zip_bytes({"cudart64_12.dll": b"cudart dll bytes"})


def _cuda12_assets(llama_zip: bytes = CUDA_LLAMA_ZIP,
                    cudart_zip: bytes = CUDA_CUDART_ZIP) -> tuple[RuntimeAsset, RuntimeAsset]:
    llama = RuntimeAsset(
        kind="cuda12", build="btest", name="llama-b11200-bin-win-cuda-12.4-x64.zip",
        url="https://example.invalid/llama-b11200-bin-win-cuda-12.4-x64.zip",
        size=len(llama_zip), sha256=hashlib.sha256(llama_zip).hexdigest(),
    )
    cudart = RuntimeAsset(
        kind="cuda12", build="btest", name="cudart-llama-bin-win-cuda-12.4-x64.zip",
        url="https://example.invalid/cudart-llama-bin-win-cuda-12.4-x64.zip",
        size=len(cudart_zip), sha256=hashlib.sha256(cudart_zip).hexdigest(),
    )
    return (llama, cudart)


GOOD_ZIP = _make_zip_bytes({"llama-server.exe": b"fake exe bytes", "ggml-vulkan.dll": b"dll"})


# ---- happy path / idempotence -------------------------------------------------------------

def test_ensure_runtime_happy_path(tmp_path: Path) -> None:
    asset = _asset_for("vulkan", "btest", GOOD_ZIP)
    events: list[tuple[str, int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=GOOD_ZIP, headers={"content-length": str(len(GOOD_ZIP))})

    folder = ensure_runtime(
        "vulkan", tmp_path, build="btest", asset=asset,
        http_client_factory=_client_factory(handler),
        progress=lambda name, done, total: events.append((name, done, total)),
    )

    assert folder == runtime_dir(tmp_path, "vulkan", "btest")
    assert (folder / "llama-server.exe").is_file()
    assert not (folder / asset.name).exists()  # zip deleted after extraction
    assert not (folder / (asset.name + ".part")).exists()
    assert events, "progress callback should have been invoked"
    assert events[-1][1] == len(GOOD_ZIP)


def test_ensure_runtime_is_idempotent_no_network(tmp_path: Path) -> None:
    rdir = runtime_dir(tmp_path, "vulkan", "btest")
    rdir.mkdir(parents=True)
    (rdir / "llama-server.exe").write_bytes(b"already here")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not make a network call when already installed")

    folder = ensure_runtime("vulkan", tmp_path, build="btest",
                             http_client_factory=_client_factory(handler))
    assert folder == rdir


def test_installed_runtime_finds_nested_top_level_folder(tmp_path: Path) -> None:
    rdir = runtime_dir(tmp_path, "vulkan", PINNED_BUILD)
    nested = rdir / "llama-b11200-bin-win-vulkan-x64"
    nested.mkdir(parents=True)
    (nested / "llama-server.exe").write_bytes(b"exe")

    found = installed_runtime(tmp_path, "vulkan")
    assert found == nested


def test_installed_runtime_none_when_absent(tmp_path: Path) -> None:
    assert installed_runtime(tmp_path, "vulkan") is None


# ---- integrity checks -----------------------------------------------------------------------

def test_ensure_runtime_rejects_size_mismatch(tmp_path: Path) -> None:
    good_asset = _asset_for("vulkan", "btest", GOOD_ZIP)
    wrong_size_asset = RuntimeAsset(
        kind=good_asset.kind, build=good_asset.build, name=good_asset.name, url=good_asset.url,
        size=good_asset.size + 5, sha256=good_asset.sha256,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=GOOD_ZIP)

    with pytest.raises(UpstreamRuntimeError, match="size mismatch"):
        ensure_runtime("vulkan", tmp_path, build="btest", asset=wrong_size_asset,
                        http_client_factory=_client_factory(handler))

    rdir = runtime_dir(tmp_path, "vulkan", "btest")
    assert not any(rdir.iterdir()) if rdir.exists() else True
    assert not (rdir / wrong_size_asset.name).exists()
    assert not (rdir / (wrong_size_asset.name + ".part")).exists()


def test_ensure_runtime_rejects_sha256_mismatch(tmp_path: Path) -> None:
    good_asset = _asset_for("vulkan", "btest", GOOD_ZIP)
    wrong_hash_asset = RuntimeAsset(
        kind=good_asset.kind, build=good_asset.build, name=good_asset.name, url=good_asset.url,
        size=good_asset.size, sha256="0" * 64,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=GOOD_ZIP)

    with pytest.raises(UpstreamRuntimeError, match="sha256 mismatch"):
        ensure_runtime("vulkan", tmp_path, build="btest", asset=wrong_hash_asset,
                        http_client_factory=_client_factory(handler))

    rdir = runtime_dir(tmp_path, "vulkan", "btest")
    assert not (rdir / wrong_hash_asset.name).exists()
    assert not (rdir / (wrong_hash_asset.name + ".part")).exists()


def test_ensure_runtime_zip_without_server_exe_rejected_and_cleaned(tmp_path: Path) -> None:
    bad_zip = _make_zip_bytes({"readme.txt": b"no server here"})
    asset = _asset_for("vulkan", "btest", bad_zip)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=bad_zip)

    with pytest.raises(UpstreamRuntimeError, match="llama-server.exe"):
        ensure_runtime("vulkan", tmp_path, build="btest", asset=asset,
                        http_client_factory=_client_factory(handler))

    rdir = runtime_dir(tmp_path, "vulkan", "btest")
    assert not rdir.exists()


@pytest.mark.parametrize("bad_name", ["../evil.exe", "C:/x.exe", "/etc/passwd"])
def test_ensure_runtime_rejects_traversal_members(tmp_path: Path, bad_name: str) -> None:
    evil_zip = _make_zip_bytes({"llama-server.exe": b"exe", bad_name: b"evil"})
    asset = _asset_for("vulkan", "btest", evil_zip)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=evil_zip)

    with pytest.raises(UpstreamRuntimeError):
        ensure_runtime("vulkan", tmp_path, build="btest", asset=asset,
                        http_client_factory=_client_factory(handler))

    # nothing written outside tmp_path's runtime tree
    rdir = runtime_dir(tmp_path, "vulkan", "btest")
    assert not rdir.exists() or not any(rdir.rglob("evil.exe"))
    assert not Path("evil.exe").exists()


# ---- OSError wrapping / partial-install cleanup (mkdir + extraction) -------------------------

def test_ensure_runtime_wraps_mkdir_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A PermissionError creating the install temp dir (e.g. a non-writable install root) must be
    reported as UpstreamRuntimeError, not propagate as a raw OSError -- and must never reach the
    network."""
    asset = _asset_for("vulkan", "btest", GOOD_ZIP)

    def bad_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError("no access")

    monkeypatch.setattr(Path, "mkdir", bad_mkdir)

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not download when the install directory cannot be created")

    with pytest.raises(UpstreamRuntimeError, match="install directory"):
        ensure_runtime("vulkan", tmp_path, build="btest", asset=asset,
                        http_client_factory=_client_factory(handler))


def test_ensure_runtime_extraction_oserror_wrapped_and_cleaned_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OSError raised mid-extraction (disk full, MAX_PATH violation, ...) must be reported as
    UpstreamRuntimeError, and must leave no partially-extracted runtime behind that a later
    installed_runtime() call would mistake for a completed install."""
    asset = _asset_for("vulkan", "btest", GOOD_ZIP)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=GOOD_ZIP)

    orig_extract = zipfile.ZipFile.extract
    calls = {"n": 0}

    def flaky_extract(self: zipfile.ZipFile, member: object, path: object = None,
                       pwd: object = None) -> object:
        calls["n"] += 1
        if calls["n"] == 2:  # let llama-server.exe (member 1) land, then fail on the next member
            raise OSError("disk full")
        return orig_extract(self, member, path, pwd)

    monkeypatch.setattr(zipfile.ZipFile, "extract", flaky_extract)

    with pytest.raises(UpstreamRuntimeError, match="failed to extract"):
        ensure_runtime("vulkan", tmp_path, build="btest", asset=asset,
                        http_client_factory=_client_factory(handler))

    rdir = runtime_dir(tmp_path, "vulkan", "btest")
    assert not rdir.exists(), "no partially-extracted runtime should be left at the final location"
    assert installed_runtime(tmp_path, "vulkan", "btest") is None, (
        "a broken, partially-extracted runtime must not be reported as installed"
    )
    build_dir = rdir.parent
    # the private temp dir used for the failed attempt must be cleaned up too, not just rdir
    assert not build_dir.exists() or not any(build_dir.iterdir())


# ---- concurrent-install safety (extract-to-temp + atomic rename) ------------------------------

def test_installed_runtime_never_observes_partial_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent caller polling installed_runtime() while an install is mid-extraction must
    never see it as installed -- extraction must happen off to the side (a private temp
    directory), not directly in the final runtime_dir."""
    asset = _asset_for("vulkan", "btest", GOOD_ZIP)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=GOOD_ZIP)

    orig_extract = zipfile.ZipFile.extract
    observed_during_extraction: list[Path | None] = []

    def probing_extract(self: zipfile.ZipFile, member: object, path: object = None,
                         pwd: object = None) -> object:
        result = orig_extract(self, member, path, pwd)
        observed_during_extraction.append(installed_runtime(tmp_path, "vulkan", "btest"))
        return result

    monkeypatch.setattr(zipfile.ZipFile, "extract", probing_extract)

    folder = ensure_runtime("vulkan", tmp_path, build="btest", asset=asset,
                             http_client_factory=_client_factory(handler))

    assert len(observed_during_extraction) == 2  # one probe per member of GOOD_ZIP
    assert all(obs is None for obs in observed_during_extraction), (
        "a concurrent installed_runtime() check observed a partially-extracted runtime as installed"
    )
    assert (folder / "llama-server.exe").is_file()


def test_ensure_runtime_loses_install_race_returns_winners_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If another (concurrent) installer finishes first, this caller's atomic rename onto the
    same final directory must fail harmlessly and this must return the winner's folder untouched,
    instead of raising or corrupting/overwriting what the winner installed."""
    asset = _asset_for("vulkan", "btest", GOOD_ZIP)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=GOOD_ZIP)

    rdir = runtime_dir(tmp_path, "vulkan", "btest")
    orig_replace = Path.replace

    def replace_after_a_winner_installs(self: Path, target: object) -> object:
        target_path = Path(target)
        if target_path == rdir and not target_path.exists():
            # Simulate a concurrent process completing its own install of the same kind/build
            # in the gap between our extraction finishing and our own rename.
            target_path.mkdir(parents=True)
            (target_path / "llama-server.exe").write_bytes(b"winner's exe")
            (target_path / "ggml-vulkan.dll").write_bytes(b"winner's dll")
        return orig_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace_after_a_winner_installs)

    folder = ensure_runtime("vulkan", tmp_path, build="btest", asset=asset,
                             http_client_factory=_client_factory(handler))

    assert folder == rdir
    assert (folder / "llama-server.exe").read_bytes() == b"winner's exe", (
        "the winner's install must be preserved untouched, not overwritten"
    )
    build_dir = rdir.parent
    leftovers = [p for p in build_dir.iterdir() if p != rdir]
    assert leftovers == [], "the losing installer's temp directory must be cleaned up"


# ---- GitHub release-API resolution (non-pinned build) ---------------------------------------

def test_ensure_runtime_refuses_asset_with_no_digest(tmp_path: Path) -> None:
    release_json = {
        "assets": [
            {
                "name": "llama-bother-bin-win-vulkan-x64.zip",
                "size": 12345,
                "browser_download_url": "https://example.invalid/llama-bother-bin-win-vulkan-x64.zip",
                # no "digest" key at all
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "api.github.com" in str(request.url)
        return httpx.Response(200, json=release_json)

    with pytest.raises(UpstreamRuntimeError, match="digest"):
        ensure_runtime("vulkan", tmp_path, build="bother",
                        http_client_factory=_client_factory(handler))


def test_ensure_runtime_resolves_via_release_api_with_digest(tmp_path: Path) -> None:
    sha = hashlib.sha256(GOOD_ZIP).hexdigest()
    release_json = {
        "assets": [
            {
                "name": "llama-bother-bin-win-vulkan-x64.zip",
                "size": len(GOOD_ZIP),
                "browser_download_url": "https://example.invalid/llama-bother-bin-win-vulkan-x64.zip",
                "digest": f"sha256:{sha}",
            },
            {
                "name": "llama-bother-bin-win-cpu-x64.zip",
                "size": 999,
                "browser_download_url": "https://example.invalid/llama-bother-bin-win-cpu-x64.zip",
                "digest": f"sha256:{'1' * 64}",
            },
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if "api.github.com" in str(request.url):
            return httpx.Response(200, json=release_json)
        return httpx.Response(200, content=GOOD_ZIP)

    folder = ensure_runtime("vulkan", tmp_path, build="bother",
                             http_client_factory=_client_factory(handler))
    assert (folder / "llama-server.exe").is_file()


def test_ensure_runtime_no_matching_asset_in_release(tmp_path: Path) -> None:
    release_json = {"assets": [{"name": "llama-bother-bin-win-cpu-x64.zip", "size": 1,
                                 "digest": f"sha256:{'1' * 64}"}]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=release_json)

    with pytest.raises(UpstreamRuntimeError, match="no 'vulkan' asset|no \"vulkan\" asset|vulkan"):
        ensure_runtime("vulkan", tmp_path, build="bother",
                        http_client_factory=_client_factory(handler))


# ---- misc -------------------------------------------------------------------------------------

def test_unknown_kind_rejected(tmp_path: Path) -> None:
    with pytest.raises(UpstreamRuntimeError):
        ensure_runtime("metal", tmp_path)


def test_runtime_kinds_and_pinned_assets_agree() -> None:
    assert set(PINNED_ASSETS) == set(RUNTIME_KINDS)
    for kind, assets in PINNED_ASSETS.items():
        assert isinstance(assets, tuple)
        assert len(assets) >= 1
        for asset in assets:
            assert asset.kind == kind
            assert asset.build == PINNED_BUILD
            assert len(asset.sha256) == 64
            assert asset.size > 0
            assert asset.url.startswith("https://github.com/ggml-org/llama.cpp/releases/download/")


def test_cuda_and_rocm_kinds_have_expected_asset_counts() -> None:
    # cuda12/cuda13 are two assets each (llama-* binaries + cudart-llama-* runtime DLLs); every
    # other kind (including "rocm", which is one self-contained zip) is a single asset.
    assert len(PINNED_ASSETS["cuda12"]) == 2
    assert len(PINNED_ASSETS["cuda13"]) == 2
    for kind in ("vulkan", "sycl", "openvino", "cpu", "rocm"):
        assert len(PINNED_ASSETS[kind]) == 1


def test_describe_runtime_from_path(tmp_path: Path) -> None:
    folder = runtime_dir(tmp_path, "vulkan", "b11200")
    folder.mkdir(parents=True)
    (folder / "llama-server.exe").write_bytes(b"exe")
    desc = describe_runtime(folder)
    assert "b11200" in desc
    assert "vulkan" in desc
    assert str(folder) in desc


def test_describe_runtime_reads_build_info_file(tmp_path: Path) -> None:
    folder = runtime_dir(tmp_path, "vulkan", "b11200")
    folder.mkdir(parents=True)
    (folder / "llama-server.exe").write_bytes(b"exe")
    (folder / "build-info.txt").write_text("b99999\n", encoding="utf-8")
    desc = describe_runtime(folder)
    assert "b99999" in desc


def test_describe_runtime_nested_folder(tmp_path: Path) -> None:
    rdir = runtime_dir(tmp_path, "sycl", "b11200")
    nested = rdir / "llama-b11200-bin-win-sycl-x64"
    nested.mkdir(parents=True)
    (nested / "llama-server.exe").write_bytes(b"exe")
    desc = describe_runtime(nested)
    assert "sycl" in desc
    assert "b11200" in desc


# ---- vulkan_env_switches ----------------------------------------------------------------------

def test_vulkan_env_switches_scans_dll_strings(tmp_path: Path) -> None:
    dll = tmp_path / "ggml-vulkan.dll"
    payload = (
        b"\x00\x00garbage\x00GGML_VK_DISABLE_F16\x00more junk\x00"
        b"GGML_VK_VISIBLE_DEVICES\x00\x01\x02GGML_VK_DISABLE_F16\x00tail"
    )
    dll.write_bytes(payload)

    switches = vulkan_env_switches(tmp_path)
    assert switches == sorted({"GGML_VK_DISABLE_F16", "GGML_VK_VISIBLE_DEVICES"})


def test_vulkan_env_switches_empty_when_no_dll(tmp_path: Path) -> None:
    assert vulkan_env_switches(tmp_path) == []


def test_vulkan_env_switches_empty_folder_does_not_raise(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    assert vulkan_env_switches(missing) == []


# ---- cancellation ---------------------------------------------------------------------------

def test_ensure_runtime_cancelled_mid_download_removes_the_part_and_installs_nothing(tmp_path: Path) -> None:
    """stop() during a fetch: the `cancelled` hook flips after the first chunk; the .part is gone,
    no runtime dir exists, and the error is the Cancelled subclass (a stop, not a failure)."""
    big = GOOD_ZIP * 2000                      # several 64 KiB chunks
    asset = _asset_for("vulkan", "btest", big)
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=big, headers={"content-length": str(len(big))})

    def cancelled() -> bool:
        return len(seen) >= 1                  # True from the second poll on

    with pytest.raises(UpstreamRuntimeCancelled):
        ensure_runtime("vulkan", tmp_path, build="btest", asset=asset,
                       http_client_factory=_client_factory(handler),
                       progress=lambda name, done, total: seen.append(done), cancelled=cancelled)
    assert installed_runtime(tmp_path, "vulkan", "btest") is None
    assert not list(tmp_path.rglob("*.part")) and not list(tmp_path.rglob("*.tmp"))
    assert seen and seen[-1] < len(big)        # it stopped early, not after the whole file


def test_ensure_runtime_cancelled_before_the_first_byte(tmp_path: Path) -> None:
    asset = _asset_for("vulkan", "btest", GOOD_ZIP)

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be reached
        raise AssertionError("no request expected when already cancelled")

    with pytest.raises(UpstreamRuntimeCancelled):
        ensure_runtime("vulkan", tmp_path, build="btest", asset=asset,
                       http_client_factory=_client_factory(handler), cancelled=lambda: True)
    assert installed_runtime(tmp_path, "vulkan", "btest") is None


# ---- multi-asset kinds (cuda12/cuda13) -------------------------------------------------------

def test_ensure_runtime_cuda12_happy_path_downloads_both_assets(tmp_path: Path) -> None:
    llama_asset, cudart_asset = _cuda12_assets()
    handler = _zip_dispatch_handler({llama_asset.name: CUDA_LLAMA_ZIP, cudart_asset.name: CUDA_CUDART_ZIP})
    events: list[tuple[str, int, int]] = []

    folder = ensure_runtime(
        "cuda12", tmp_path, build="btest", asset=(llama_asset, cudart_asset),
        http_client_factory=_client_factory(handler),
        progress=lambda name, done, total: events.append((name, done, total)),
    )

    assert folder == runtime_dir(tmp_path, "cuda12", "btest")
    assert (folder / "llama-server.exe").is_file()
    assert (folder / "ggml-cuda.dll").is_file()
    assert (folder / "cudart64_12.dll").is_file()
    # progress fired for BOTH assets, keyed by that asset's own zip name
    assert {name for name, _, _ in events} == {llama_asset.name, cudart_asset.name}
    # no leftover zips or .part files from either asset
    assert not list(folder.rglob("*.zip"))
    assert not list(folder.rglob("*.part"))


def test_ensure_runtime_cuda12_asset_passed_as_single_runtime_asset(tmp_path: Path) -> None:
    """`asset` still accepts a single RuntimeAsset (not just a sequence) for a kind whose pinned
    assets happen to be one -- exercised here via "rocm", the new single-asset kind."""
    zip_bytes = _make_zip_bytes({"llama-server.exe": b"rocm exe"})
    asset = RuntimeAsset(kind="rocm", build="btest", name="llama-b11200-bin-win-rocm-10.0-x64.zip",
                          url="https://example.invalid/rocm.zip", size=len(zip_bytes),
                          sha256=hashlib.sha256(zip_bytes).hexdigest())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=zip_bytes)

    folder = ensure_runtime("rocm", tmp_path, build="btest", asset=asset,
                             http_client_factory=_client_factory(handler))
    assert (folder / "llama-server.exe").is_file()


def test_ensure_runtime_cuda12_cudart_sha_mismatch_leaves_nothing_installed(tmp_path: Path) -> None:
    """A failure on the SECOND asset (cudart) must discard the whole attempt, including whatever
    the FIRST asset (llama-*, which already extracted llama-server.exe) had put in place."""
    llama_asset, good_cudart = _cuda12_assets()
    bad_cudart = RuntimeAsset(kind=good_cudart.kind, build=good_cudart.build, name=good_cudart.name,
                               url=good_cudart.url, size=good_cudart.size, sha256="0" * 64)
    handler = _zip_dispatch_handler({llama_asset.name: CUDA_LLAMA_ZIP, bad_cudart.name: CUDA_CUDART_ZIP})

    with pytest.raises(UpstreamRuntimeError, match="sha256 mismatch"):
        ensure_runtime("cuda12", tmp_path, build="btest", asset=(llama_asset, bad_cudart),
                        http_client_factory=_client_factory(handler))

    assert installed_runtime(tmp_path, "cuda12", "btest") is None
    build_dir = runtime_dir(tmp_path, "cuda12", "btest").parent
    assert not build_dir.exists() or not any(build_dir.iterdir())


def test_ensure_runtime_cancelled_during_second_asset(tmp_path: Path) -> None:
    """stop() while the SECOND asset (cudart) is downloading: the first asset's download and
    extraction must have already happened (seen fired for it), but nothing ends up installed --
    the whole attempt, including the completed first asset, is discarded."""
    llama_asset, cudart_asset = _cuda12_assets()
    big_cudart = CUDA_CUDART_ZIP * 2000  # several 64 KiB chunks
    cudart_asset = RuntimeAsset(kind=cudart_asset.kind, build=cudart_asset.build, name=cudart_asset.name,
                                 url=cudart_asset.url, size=len(big_cudart),
                                 sha256=hashlib.sha256(big_cudart).hexdigest())
    handler = _zip_dispatch_handler({llama_asset.name: CUDA_LLAMA_ZIP, cudart_asset.name: big_cudart})
    seen: list[str] = []

    def cancelled() -> bool:
        return cudart_asset.name in seen  # True only once the SECOND asset has started downloading

    with pytest.raises(UpstreamRuntimeCancelled):
        ensure_runtime("cuda12", tmp_path, build="btest", asset=(llama_asset, cudart_asset),
                        http_client_factory=_client_factory(handler),
                        progress=lambda name, done, total: seen.append(name), cancelled=cancelled)

    assert llama_asset.name in seen, "the first asset should have downloaded before cancellation"
    assert cudart_asset.name in seen, "cancellation should have happened mid-way through the second asset"
    assert installed_runtime(tmp_path, "cuda12", "btest") is None
    build_dir = runtime_dir(tmp_path, "cuda12", "btest").parent
    assert not build_dir.exists() or not any(build_dir.iterdir())


def test_fetch_release_asset_cuda12_missing_cudart_names_it(tmp_path: Path) -> None:
    """Both patterns of a multi-asset kind must match: when the release has the llama-* binaries
    but no cudart-llama-* zip, the error must name the missing one ("cudart")."""
    release_json = {
        "assets": [
            {
                "name": "llama-bother-bin-win-cuda-12.4-x64.zip",
                "size": len(CUDA_LLAMA_ZIP),
                "browser_download_url": "https://example.invalid/llama-bother-bin-win-cuda-12.4-x64.zip",
                "digest": f"sha256:{hashlib.sha256(CUDA_LLAMA_ZIP).hexdigest()}",
            },
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=release_json)

    with pytest.raises(UpstreamRuntimeError, match="cudart"):
        ensure_runtime("cuda12", tmp_path, build="bother", http_client_factory=_client_factory(handler))


def test_assets_for_pinned_build_no_network(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not make a network call for the pinned build")

    assets = assets_for("cuda13", http_client_factory=_client_factory(handler))
    assert assets == PINNED_ASSETS["cuda13"]


# ---- cuda_kind_for_driver ---------------------------------------------------------------------

@pytest.mark.parametrize("driver,expected", [
    ("592.14", "cuda13"),
    ("580.00", "cuda13"),
    ("579.99", "cuda12"),
    ("552.22", "cuda12"),
    (None, "cuda12"),
    ("garbage", "cuda12"),
])
def test_cuda_kind_for_driver_table(driver: "str | None", expected: str) -> None:
    assert cuda_kind_for_driver(driver) == expected


# ---- install manifest / describe_runtime -------------------------------------------------------

def test_ensure_runtime_writes_manifest_and_describe_runtime_reads_it_back(tmp_path: Path) -> None:
    llama_asset, cudart_asset = _cuda12_assets()
    handler = _zip_dispatch_handler({llama_asset.name: CUDA_LLAMA_ZIP, cudart_asset.name: CUDA_CUDART_ZIP})

    folder = ensure_runtime("cuda12", tmp_path, build="btest", asset=(llama_asset, cudart_asset),
                             http_client_factory=_client_factory(handler))

    rdir = runtime_dir(tmp_path, "cuda12", "btest")
    manifest_path = rdir / ".exocortex-runtime.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "cuda12"
    assert manifest["build"] == "btest"
    assert set(manifest["assets"]) == {llama_asset.name, cudart_asset.name}

    desc = describe_runtime(folder)
    assert llama_asset.name in desc
    assert cudart_asset.name in desc


def test_describe_runtime_absent_manifest_unchanged(tmp_path: Path) -> None:
    """A runtime installed before the manifest existed (or dropped in by hand) must describe
    exactly as it did before manifests existed -- no trailing bracket, never an error."""
    folder = runtime_dir(tmp_path, "vulkan", "b11200")
    folder.mkdir(parents=True)
    (folder / "llama-server.exe").write_bytes(b"exe")
    desc = describe_runtime(folder)
    assert "[" not in desc
    assert "b11200" in desc and "vulkan" in desc
