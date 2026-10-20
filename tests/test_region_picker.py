"""Offscreen tests for the screen-region picker."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from exocortex.ui.region_picker import RegionPicker, logical_to_physical, virtual_desktop_rect  # noqa: E402


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_virtual_desktop_rect_is_non_empty(app: QApplication) -> None:
    r = virtual_desktop_rect()
    assert r.width() > 0 and r.height() > 0


def test_logical_to_physical_identity_at_dpr_1(app: QApplication) -> None:
    x, y, w, h = logical_to_physical(QRect(10, 20, 300, 200))
    # offscreen platform reports dpr 1.0
    assert (w, h) == (300, 200) and x == 10 and y == 20


def test_drag_emits_selection(app: QApplication) -> None:
    picker = RegionPicker()
    got: list[tuple[int, int, int, int]] = []
    picker.selected.connect(lambda x, y, w, h: got.append((x, y, w, h)))
    picker.begin()
    QTest.qWait(20)
    QTest.mousePress(picker, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(50, 60))
    QTest.mouseMove(picker, QPoint(250, 210))
    QTest.mouseRelease(picker, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(250, 210))
    QTest.qWait(20)
    assert len(got) == 1
    x, y, w, h = got[0]
    assert (w, h) == (200, 150)
    assert (x, y) == (picker.x() + 50, picker.y() + 60)
    assert not picker.isVisible()


def test_escape_cancels_and_tiny_selection_cancels(app: QApplication) -> None:
    picker = RegionPicker()
    cancelled: list[int] = []
    picker.cancelled.connect(lambda: cancelled.append(1))
    picker.begin()
    QTest.qWait(20)
    QTest.keyClick(picker, Qt.Key.Key_Escape)
    assert cancelled == [1] and not picker.isVisible()

    picker.begin()
    QTest.mousePress(picker, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(10, 10))
    QTest.mouseRelease(picker, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(12, 12))
    QTest.qWait(20)
    assert cancelled == [1, 1]


def test_picker_shows_when_begun(app: QApplication) -> None:
    """dev-assist is a visible copilot: the picker just appears (no capture gating)."""
    picker = RegionPicker()
    picker.begin()
    QTest.qWait(10)
    assert picker.isVisible()
    picker._cancel()
