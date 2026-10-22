"""Offscreen tests for Settings > Audio: the device pickers and the Check panel
(exocortex.ui.audio_check), with every device / Windows call faked."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Any

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox

from exocortex.audio.capture import DeviceInfo
from exocortex.audio.check import RoutingResult
from exocortex.audio.endpoint import EndpointState
from exocortex.bus import Source
from exocortex.config import Config
from exocortex.ui import settings as settings_mod
from exocortex.ui.audio_check import AudioCheckPanel
from exocortex.ui.settings import FIELDS, SettingsDialog, _DeviceCombo, make_field_widget


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-tests"])


def _wait(predicate: Any, timeout_ms: int = 3000) -> bool:
    for _ in range(timeout_ms // 20):
        if predicate():
            return True
        QTest.qWait(20)
    return predicate()


def _dev(index: int, name: str, *, is_input: bool = True, loopback: bool = False, default: bool = False) -> DeviceInfo:
    return DeviceInfo(index=index, name=name, is_loopback=loopback, is_input=is_input,
                      sample_rate=48000, channels=2, is_default=default)


DEVICES = [
    _dev(12, "Odyssey G61SH (NVIDIA High Definition Audio)", is_input=False),
    _dev(15, "Microphone Array (Realtek(R) Audio)", default=True),
    _dev(16, "Odyssey G61SH (NVIDIA High Definition Audio) [Loopback]", loopback=True, default=True),
    _dev(19, "Headset Microphone (Jabra)"),
    _dev(20, "Headset Earphone (Jabra) [Loopback]", loopback=True),
]


def _field(key: str) -> Any:
    return next(f for f in FIELDS["Audio"] if f.key == key)


# --------------------------------------------------------------------------- #
# the device pickers
# --------------------------------------------------------------------------- #


def test_audio_tab_leads_with_the_two_device_pickers() -> None:
    keys = [f.key for f in FIELDS["Audio"]]
    assert keys[:2] == ["mic_device", "system_device"]
    assert {f.kind for f in FIELDS["Audio"][:2]} == {"mic_device", "system_device"}
    assert not any(f.restart for f in FIELDS["Audio"][:2])      # applied at once


def test_device_combo_lists_only_its_kind_and_keeps_the_choice(qapp: QApplication,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(settings_mod, "_list_audio_devices", lambda: calls.append(1) or list(DEVICES))
    mic = make_field_widget(_field("mic_device"))
    system = make_field_widget(_field("system_device"))
    assert isinstance(mic, _DeviceCombo) and mic.isEditable()
    assert calls == []                                          # nothing enumerated until it drops down
    mic.setEditText("Headset Microphone (Jabra)")
    mic.refresh()
    system.refresh()
    assert [mic.itemText(i) for i in range(mic.count())] == [
        "default", "Microphone Array (Realtek(R) Audio)", "Headset Microphone (Jabra)"]
    assert [system.itemText(i) for i in range(system.count())] == [
        "default", "Odyssey G61SH (NVIDIA High Definition Audio) [Loopback]", "Headset Earphone (Jabra) [Loopback]"]
    assert mic.currentText() == "Headset Microphone (Jabra)"
    assert "Microphone Array (Realtek(R) Audio)" in str(mic.itemData(0, 3))      # the default, named
    assert "the Windows default" in str(mic.itemData(1, 3))
    mic.setEditText("jabra")                                    # part of a name, as config.toml allows
    mic.refresh()
    assert mic.currentText() == "jabra"


def test_device_combo_survives_a_failing_enumeration(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> list[DeviceInfo]:
        raise RuntimeError("pyaudiowpatch missing")

    monkeypatch.setattr(settings_mod, "_list_audio_devices", boom)
    combo = make_field_widget(_field("mic_device"))
    combo.refresh()
    assert combo.currentText() == "default" and combo.count() == 1


def test_dialog_loads_and_applies_a_device_choice(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings_mod, "_list_audio_devices", lambda: list(DEVICES))
    cfg = Config()
    cfg.audio.system_device = "Headset Earphone"
    dlg = SettingsDialog(cfg)
    got: list[dict[str, Any]] = []
    dlg.applied.connect(got.append)
    try:
        assert isinstance(dlg.audio_check, AudioCheckPanel)
        assert dlg._widgets["audio.system_device"].currentText() == "Headset Earphone"
        assert dlg.changes() == {}
        dlg._widgets["audio.mic_device"].refresh()
        dlg._widgets["audio.mic_device"].setCurrentText("Headset Microphone (Jabra)")
        changed = dlg.apply()
        assert changed == {"audio.mic_device": "Headset Microphone (Jabra)"}
        assert cfg.audio.mic_device == "Headset Microphone (Jabra)" and got == [changed]
        assert "next launch" not in dlg.note.text()
    finally:
        dlg.deleteLater()


# --------------------------------------------------------------------------- #
# the Check panel
# --------------------------------------------------------------------------- #


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class _Capture:
    def __init__(self, source: Source, spec: str, **kw: Any) -> None:
        self.source, self.spec, self.kw = source, spec, kw
        self.on_level = kw["on_level"]
        self.stopped = False

    def start(self) -> DeviceInfo:
        if self.spec == "broken":
            raise ValueError("no mic device matching 'broken'")
        return _dev(1, f"{self.source.value} device ({self.spec})", loopback=self.source is Source.SYSTEM)

    def stop(self) -> None:
        self.stopped = True


class _Rig:
    def __init__(self) -> None:
        self.mic = _DeviceCombo("mic")
        self.system = _DeviceCombo("system")
        self.echo = QCheckBox()
        self.echo.setChecked(True)
        self.captures: list[_Capture] = []
        self.states: list[EndpointState | None] = []
        self.unmuted: list[tuple[str, bool]] = []
        self.routing_result: RoutingResult | Exception | None = None
        self.clock = _Clock()
        self.panel = AudioCheckPanel(
            self.mic, self.system, self.echo,
            capture_factory=self._factory, routing=self._routing,
            endpoint_state=self._state, set_endpoint_muted=self._set_muted, clock=self.clock,
        )

    def _factory(self, source: Source, spec: str, **kw: Any) -> _Capture:
        cap = _Capture(source, spec, **kw)
        self.captures.append(cap)
        return cap

    def _routing(self, system: str, mic: str) -> RoutingResult:
        self.routed_with = (system, mic)
        if isinstance(self.routing_result, Exception):
            raise self.routing_result
        assert self.routing_result is not None
        return self.routing_result

    def _state(self, name: str) -> EndpointState | None:
        self.state_names = getattr(self, "state_names", []) + [name]
        return self.states[-1] if self.states else None

    def _set_muted(self, name: str, muted: bool) -> bool:
        self.unmuted.append((name, muted))
        self.states.append(EndpointState(name=name, muted=muted, volume=0.6))
        return True

    def capture(self, source: Source) -> _Capture:
        return [c for c in self.captures if c.source is source][-1]

    def start(self) -> None:
        self.panel.test_btn.setChecked(True)
        assert _wait(lambda: len(self.captures) == 2 and all(k in self.panel._names for k in ("mic", "system")))


def _routing_result(**over: Any) -> RoutingResult:
    base = dict(output_device="Speakers A", system_device="Speakers A [Loopback]", mic_device="Mic A",
                system_before_db=-180.0, system_during_db=-15.0, mic_before_db=-70.0, mic_during_db=-70.0,
                mic_silent=False)
    base.update(over)
    return RoutingResult(**base)


def test_test_devices_opens_the_chosen_devices_and_meters_them(qapp: QApplication) -> None:
    rig = _Rig()
    rig.mic.setEditText("Headset")
    rig.start()
    mic, system = rig.capture(Source.MIC), rig.capture(Source.SYSTEM)
    assert mic.spec == "Headset" and system.spec == "default"
    assert mic.kw["keepalive"] is False and mic.kw["target_rate"] == 16000
    assert rig.panel.test_btn.text() == "Stop test"

    mic.on_level(0.05, 0.3)
    assert _wait(lambda: rig.panel.meters["mic"].value() > 0)
    rig.panel._tick()
    assert "hearing you" in rig.panel.statuses["mic"].text()
    assert "nothing playing" in rig.panel.statuses["system"].text()

    rig.panel.test_btn.setChecked(False)
    assert _wait(lambda: mic.stopped and system.stopped)
    assert rig.panel.meters["mic"].value() == 0 and rig.panel.statuses["mic"].text() == ""


def test_a_silent_mic_shows_the_windows_mute_and_unmutes_it(qapp: QApplication) -> None:
    rig = _Rig()
    rig.states.append(EndpointState(name="Microphone Array (Realtek(R) Audio)", muted=True, volume=0.58))
    rig.start()
    assert not rig.panel.windows_label.isHidden() and not rig.panel.unmute_btn.isHidden()   # known at open
    assert "muted" in rig.panel.windows_label.text()
    mic = rig.capture(Source.MIC)
    for _ in range(20):
        mic.on_level(1e-10, 1e-9)
        rig.clock.t += 0.1
    QTest.qWait(50)
    rig.panel._tick()
    assert "digital silence" in rig.panel.statuses["mic"].text()

    rig.panel.unmute_btn.click()
    assert rig.unmuted == [("mic device (default)", False)]
    assert rig.panel.windows_label.isHidden() and rig.panel.unmute_btn.isHidden()
    rig.panel.test_btn.setChecked(False)


def test_a_low_windows_input_volume_is_reported_without_an_unmute_button(qapp: QApplication) -> None:
    rig = _Rig()
    rig.states.append(EndpointState(name="Mic A", muted=False, volume=0.0))
    rig.start()
    assert "0%" in rig.panel.windows_label.text() and rig.panel.unmute_btn.isHidden()
    rig.panel.test_btn.setChecked(False)


def test_a_device_that_does_not_open_says_why(qapp: QApplication) -> None:
    rig = _Rig()
    rig.mic.setEditText("broken")
    rig.panel.test_btn.setChecked(True)
    assert _wait(lambda: "could not open" in rig.panel.statuses["mic"].text())
    rig.panel.test_btn.setChecked(False)


def test_picking_another_device_while_testing_reopens(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings_mod, "_list_audio_devices", lambda: list(DEVICES))
    rig = _Rig()
    rig.start()
    first = rig.capture(Source.MIC)
    rig.mic.refresh()
    rig.mic.setCurrentIndex(2)
    rig.mic.activated.emit(2)
    assert _wait(lambda: rig.capture(Source.MIC) is not first and "Jabra" in rig.capture(Source.MIC).spec)
    assert _wait(lambda: first.stopped)
    rig.panel.test_btn.setChecked(False)


def test_hiding_the_panel_stops_the_test(qapp: QApplication) -> None:
    rig = _Rig()
    rig.panel.show()
    rig.start()
    rig.panel.hide()
    assert not rig.panel.testing
    assert _wait(lambda: all(c.stopped for c in rig.captures))


def test_clean_routing_offers_to_turn_echo_suppression_off(qapp: QApplication) -> None:
    rig = _Rig()
    rig.mic.setEditText("Headset")
    rig.routing_result = _routing_result()
    rig.panel.check_routing()
    assert not rig.panel.route_btn.isEnabled()
    assert _wait(lambda: rig.panel.route_btn.isEnabled())
    assert rig.routed_with == ("default", "Headset")
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
    assert "did not reach" in text and "digital silence" in text
    assert rig.panel.echo_btn.isHidden()                        # nothing to recommend

    rig.routing_result = ValueError("no system device matching 'x'")
    rig.panel.check_routing()
    assert _wait(lambda: rig.panel.route_btn.isEnabled())
    assert "Could not run the check" in rig.panel.result.text()
