"""Offscreen tests for the ghost-text window: font matching, layout, show/hide."""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from exocortex.bus import CompletionEvent
from exocortex.config import Config
from exocortex.ui import ghost as ghost_mod
from exocortex.ui.ghost import GhostText


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["ghost-tests"])


def test_show_matches_font_and_places_at_caret(qapp: QApplication) -> None:
    g = GhostText(Config())
    g.show_completion(CompletionEvent(visible=True, text=" on the mat", x=500, y=400, line_h=23, font_px=13.3,
                                      font_name="Segoe UI", current_line="the cat sat", app="x"))
    assert g.isVisible()
    assert g._font.pixelSize() == 13
    assert g._lines == [" on the mat"] and g._offsets == [0.0]
    geo = g.geometry()
    assert abs(geo.x() - 498) <= 2 and abs(geo.y() - 399) <= 2      # dpr 1 offscreen: caret minus padding
    assert geo.height() >= 23
    g.hide_completion()
    assert not g.isVisible()
    g.close()


def test_font_falls_back_to_caret_height_and_is_clamped(qapp: QApplication) -> None:
    g = GhostText(Config())
    g.show_completion(CompletionEvent(visible=True, text="x", x=0, y=0, line_h=26, font_px=0.0))
    assert g._font.pixelSize() == 20                                   # 26 / 1.3
    g.show_completion(CompletionEvent(visible=True, text="x", x=0, y=0, line_h=20, font_px=60.0))
    assert g._font.pixelSize() == 19                                   # capped at 95 % of the line height
    g.close()


def test_multiline_continuations_align_to_line_start(qapp: QApplication) -> None:
    g = GhostText(Config())
    g.show_completion(CompletionEvent(visible=True, text="# todo\n    return 1", x=300, y=100, line_h=22,
                                      font_px=16.0, font_name="Consolas", mono=True, current_line="    "))
    assert len(g._lines) == 2
    assert g._offsets[0] == 0.0 and g._offsets[1] < 0.0              # second line shifted back by the indent width
    assert g.geometry().x() < 300                                      # window widened to the left to fit it
    assert g.geometry().height() >= 44
    g.close()


def test_blank_or_invisible_event_hides(qapp: QApplication) -> None:
    g = GhostText(Config())
    g.show_completion(CompletionEvent(visible=True, text="abc", x=10, y=10, line_h=20, font_px=14.0))
    assert g.isVisible()
    g.show_completion(CompletionEvent(visible=True, text="   ", x=10, y=10, line_h=20, font_px=14.0))
    assert not g.isVisible()
    g.show_completion(CompletionEvent(visible=False))
    assert not g.isVisible()
    g.close()


def test_hidden_input_caret_does_not_shrink_the_font(qapp: QApplication) -> None:
    """A Monaco-style hidden 1-px textarea reports a 1-px caret: the font must come from the
    reported font size (or the default), never be capped to that caret."""
    cfg = Config()
    g = GhostText(cfg)
    g.show_completion(CompletionEvent(visible=True, text="x", x=0, y=0, line_h=1, font_px=18.0))
    assert g._font.pixelSize() == 18
    g.show_completion(CompletionEvent(visible=True, text="x", x=0, y=0, line_h=1, font_px=0.0))
    assert g._font.pixelSize() == round(cfg.completion.default_font_pt * 96 / 72)   # 15 px at dpr 1
    g.close()

