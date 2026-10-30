"""Tests for exocortex.ui.resources.ResourceSampler: the Monitor section's CPU / RAM / VRAM
reader. Like the rest of the hardening, every field fails soft (``None``) and nothing raises,
on any platform -- these tests run under the offscreen platform and off Windows in CI."""
from __future__ import annotations

import sys

from exocortex.ui.resources import ResourceSample, ResourceSampler


def test_sample_never_raises_and_returns_a_sample() -> None:
    s = ResourceSampler()
    sample = s.sample()
    assert isinstance(sample, ResourceSample)
    # A second sample (so CPU has an interval to difference against on Windows).
    sample2 = s.sample()
    assert isinstance(sample2, ResourceSample)


def test_fields_have_the_right_types_or_none() -> None:
    sample = ResourceSampler().sample()
    assert sample.cpu_percent is None or isinstance(sample.cpu_percent, float)
    assert sample.proc_ram_mb is None or isinstance(sample.proc_ram_mb, int)
    assert sample.ram_used_mb is None or isinstance(sample.ram_used_mb, int)
    assert sample.ram_total_mb is None or isinstance(sample.ram_total_mb, int)
    assert sample.vram_used_mb is None or isinstance(sample.vram_used_mb, int)
    assert sample.vram_total_mb is None or isinstance(sample.vram_total_mb, int)
    assert isinstance(sample.vram_source, str)
    for util in (sample.gpu_percent, sample.igpu_percent, sample.npu_percent):
        assert util is None or (isinstance(util, float) and 0.0 <= util <= 100.0)


def test_cpu_percent_is_a_percentage_when_present() -> None:
    s = ResourceSampler()
    s.sample()
    cpu = s.sample().cpu_percent
    assert cpu is None or (0.0 <= cpu <= 100.0)


def test_off_windows_every_hardware_field_is_none() -> None:
    if sys.platform == "win32":
        import pytest

        pytest.skip("Windows reads real values")
    sample = ResourceSampler().sample()
    assert sample.cpu_percent is None
    assert sample.proc_ram_mb is None and sample.ram_used_mb is None and sample.ram_total_mb is None
    assert sample.vram_used_mb is None and sample.vram_total_mb is None
    assert sample.gpu_percent is None and sample.igpu_percent is None and sample.npu_percent is None
