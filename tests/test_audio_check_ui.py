"""Offscreen tests for Settings > Audio's Check box (exocortex.ui.audio_check): exo-cortex's Audio page
with this fork's checks -- the Windows mute of a microphone a Test found, and the speaker-routing
check -- with every device / Windows call faked (tests/test_settings_audio.py's fakes for the page)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Any

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox

from exocortex.audio.check import RoutingResult
from exocortex.audio.endpoint import EndpointState
from exocortex.ui import audio_check as check_mod
from exocortex.ui import audio_panel as panel_mod
from exocortex.ui.audio_check import AudioCheckPanel, CheckedAudioPanel
from exocortex.ui.settings import _DeviceCombo
from tests.test_settings import qapp  # noqa: F401 -- reused session fixture
from tests.test_settings_audio import MIC_A, _dialog, _FakeProbe, _settle


def _wait(predicate: Any, timeout_ms: int = 3000) -> bool:
    for _ in range(timeout_ms // 20):
        if predicate():
            return True
        QTest.qWait(20)
    return predicate()


class _Windows:
    """The endpoint calls, faked: `states` is what Windows reports (the last one wins)."""

    def __init__(self, *states: EndpointState) -> None:
        self.states: list[EndpointState] = list(states)
        self.asked: list[str] = []
        self.unmuted: list[tuple[str, bool]] = []

    def state(self, name: str) -> EndpointState | None:
        self.asked.append(name)
        return self.states[-1] if self.states else None

    def set_muted(self, name: str, muted: bool) -> bool:
        self.unmuted.append((name, muted))
        self.states.append(EndpointState(name=name, muted=muted, volume=0.6))
        return True


@pytest.fixture()
def windows(monkeypatch: pytest.MonkeyPatch) -> _Windows:
    fake = _Windows()
    monkeypatch.setattr(check_mod, "_default_endpoint_state", fake.state)
    monkeypatch.setattr(check_mod, "_default_set_endpoint_muted", fake.set_muted)
    return fake


# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #


def test_the_audio_page_is_exo_cortexs_panel_with_the_check_box(qapp: QApplication, monkeypatch: pytest.MonkeyPatch,
                                                                windows: _Windows) -> None:
    dlg = _dialog(monkeypatch)
    try:
        page = dlg._audio_panel
        assert isinstance(page, CheckedAudioPanel) and isinstance(dlg.audio_check, AudioCheckPanel)
        assert page.check is dlg.audio_check and page.check_box.title() == "Check"
        body = page.scroll.widget().layout()
        order = [body.itemAt(i).widget() for i in range(body.count())]
        assert order.index(page.check_box) > order.index(page.vad_box)       # under the device boxes...
        assert body.indexOf(page.check_box) < body.count() - 2               # ...above Refresh + the status line
        assert dlg.audio_check.windows_label.isHidden() and dlg.audio_check.unmute_btn.isHidden()
        assert windows.asked == []                                           # nothing asked before a Test
    finally:
        dlg.deleteLater()


def test_a_mic_test_from_a_windows_muted_mic_offers_to_unmute_it(qapp: QApplication, monkeypatch: pytest.MonkeyPatch,
                                                                windows: _Windows) -> None:
    windows.states.append(EndpointState(name=MIC_A.name, muted=True, volume=0.58))
    monkeypatch.setattr(panel_mod, "probe_device", _FakeProbe("silent"))
    dlg = _dialog(monkeypatch)
    try:
        page, check = dlg._audio_panel, dlg.audio_check
        assert page.start_test("mic")
        _settle(page)
        assert windows.asked == [MIC_A.name]                                 # the device the Test opened
        assert "muted" in check.windows_label.text() and not check.unmute_btn.isHidden()
        check.unmute_btn.click()
        assert windows.unmuted == [(MIC_A.name, False)]
        assert check.windows_label.isHidden() and check.unmute_btn.isHidden()
    finally:
        dlg.deleteLater()


def test_a_low_windows_input_volume_is_reported_without_an_unmute_button(
        qapp: QApplication, monkeypatch: pytest.MonkeyPatch, windows: _Windows) -> None:
    windows.states.append(EndpointState(name=MIC_A.name, muted=False, volume=0.0))
    monkeypatch.setattr(panel_mod, "probe_device", _FakeProbe("silent"))
    dlg = _dialog(monkeypatch)
    try:
        page, check = dlg._audio_panel, dlg.audio_check
        assert page.start_test("mic")
        _settle(page)
        assert "0%" in check.windows_label.text() and check.unmute_btn.isHidden()
    finally:
        dlg.deleteLater()


def test_a_system_test_or_a_failed_one_leaves_windows_alone(qapp: QApplication, monkeypatch: pytest.MonkeyPatch,
                                                           windows: _Windows) -> None:
    monkeypatch.setattr(panel_mod, "probe_device", _FakeProbe("ok"))
    dlg = _dialog(monkeypatch)
    try:
        page = dlg._audio_panel
        assert page.start_test("system")
        _settle(page)
        monkeypatch.setattr(panel_mod, "probe_device", _FakeProbe(raise_exc=RuntimeError("boom")))
        assert page.start_test("mic")
        _settle(page)
        assert windows.asked == []
    finally:
        dlg.deleteLater()


# --------------------------------------------------------------------------- #
# the routing check
# --------------------------------------------------------------------------- #


class _Rig:
    def __init__(self) -> None:
        self.mic = _DeviceCombo()
        self.system = _DeviceCombo()
        self.echo = QCheckBox()
        self.echo.setChecked(True)
        self.routing_result: RoutingResult | Exception | None = None
        self.windows = _Windows()
        self.panel = AudioCheckPanel(self.mic, self.system, self.echo, routing=self._routing,
                                     endpoint_state=self.windows.state, set_endpoint_muted=self.windows.set_muted)

    def _routing(self, system: str, mic: str) -> RoutingResult:
        self.routed_with = (system, mic)
        if isinstance(self.routing_result, Exception):
            raise self.routing_result
        assert self.routing_result is not None
        return self.routing_result


def _routing_result(**over: Any) -> RoutingResult:
    base = dict(output_device="Speakers A", system_device="Speakers A [Loopback]", mic_device="Mic A",
                system_before_db=-180.0, system_during_db=-15.0, mic_before_db=-70.0, mic_during_db=-70.0,
                mic_silent=False)
    base.update(over)
    return RoutingResult(**base)


def test_clean_routing_offers_to_turn_echo_suppression_off(qapp: QApplication) -> None:
    rig = _Rig()
    rig.mic.set_spec("Headset")
    rig.routing_result = _routing_result()
    rig.panel.check_routing()
    assert not rig.panel.route_btn.isEnabled()
    assert _wait(lambda: rig.panel.route_btn.isEnabled())
    assert rig.routed_with == ("default", "Headset")                         # the pickers' specs, applied or not
    text = rig.panel.result.text()
    assert "reach the output feed" in text and "does not hear the speakers" in text and "Clean routing" in text
    assert not rig.panel.echo_btn.isHidden() and "off" in rig.panel.echo_btn.text()
    rig.panel.echo_btn.click()
    assert not rig.echo.isChecked() and not rig.panel.echo_btn.isEnabled()


def test_bleed_offers_to_turn_echo_suppression_back_on(qapp: QApplication) -> None:
    rig = _Rig()
    rig.echo.setChecked(False)
    rig.routing_result = _routing_result(mic_during_db=-40.0)
    rig.panel.check_routing()
    assert _wait(lambda: rig.panel.route_btn.isEnabled())
    assert "transcribed as Me" in rig.panel.result.text() and "Clean routing" not in rig.panel.result.text()
    rig.panel.echo_btn.click()
    assert rig.echo.isChecked()


def test_routing_problems_are_explained(qapp: QApplication) -> None:
    rig = _Rig()
    rig.routing_result = _routing_result(system_during_db=-178.0, mic_silent=True)
    rig.panel.check_routing()
    assert _wait(lambda: rig.panel.route_btn.isEnabled())
    text = rig.panel.result.text()
    assert "did not reach" in text and "digital silence" in text and "Test the microphone" in text
    assert rig.panel.echo_btn.isHidden()                                     # nothing to recommend

    rig.routing_result = ValueError("no system device matching 'x'")
    rig.panel.check_routing()
    assert _wait(lambda: rig.panel.route_btn.isEnabled())
    assert "Could not run the check" in rig.panel.result.text()


def test_the_mute_check_without_a_test_reads_the_pickers_spec(qapp: QApplication) -> None:
    rig = _Rig()
    rig.mic.set_spec("Yeti")
    rig.windows.states.append(EndpointState(name="Yeti Stereo Microphone", muted=True, volume=0.5))
    rig.panel.refresh_endpoint()
    assert rig.windows.asked == ["Yeti"] and not rig.panel.unmute_btn.isHidden()
