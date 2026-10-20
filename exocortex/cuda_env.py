"""CUDA DLL discovery for pip-installed NVIDIA libraries (Windows).

CTranslate2 (faster-whisper) loads cuBLAS/cuDNN lazily with plain LoadLibrary,
which only honours the standard search order (PATH), not os.add_dll_directory.
This module MUST be imported and `setup_cuda_dlls()` called BEFORE importing
`ctranslate2` or `faster_whisper`.

Verified on an NVIDIA laptop (RTX 5080): with the nvidia/*/bin folders prepended to PATH,
`ctranslate2.get_cuda_device_count()` returns 1 and float16 inference runs on the GPU. Without
the PATH change the encoder fails with "Library cublas64_12.dll is not found or cannot be
loaded". `setup_cuda_dlls()` is unconditional and safe to call on every machine: on a machine
with no NVIDIA GPU (this file's other verification point, 2026-09-25, on a Ryzen AI MAX+ 395 /
Radeon 8060S box) `find_nvidia_dll_dirs()` simply finds nothing under site-packages (no
`nvidia-cublas-cu12` etc. get installed there without an NVIDIA GPU at setup time) and returns
`[]` -- Whisper falls through to the CPU path in `exocortex.speech.asr` and this module is a
no-op. Nothing here is NVIDIA-specific in a way that needs a vendor check; it is simply useless,
and harmless, off NVIDIA.
"""
from __future__ import annotations

import glob
import os
import sys

_done: list[str] | None = None


def find_nvidia_dll_dirs() -> list[str]:
    dirs: list[str] = []
    for p in list(sys.path) + [os.path.dirname(os.__file__)]:
        if not p or not os.path.isdir(p):
            continue
        for d in glob.glob(os.path.join(p, "nvidia", "*", "bin")):
            if os.path.isdir(d) and d not in dirs:
                dirs.append(d)
    return dirs


def setup_cuda_dlls() -> list[str]:
    """Prepend NVIDIA pip DLL folders to PATH (idempotent). Returns the folders."""
    global _done
    if _done is not None:
        return _done
    dirs = find_nvidia_dll_dirs()
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            for d in dirs:
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _done = dirs
    return dirs
