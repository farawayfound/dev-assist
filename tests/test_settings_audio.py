"""Offscreen tests for Settings > Audio (2026-10-29): the ``device`` field kind (``_DeviceCombo``), the
reworked ``FIELDS["Audio"]``, and the page built by ``exocortex.ui.audio_panel.AudioPanel`` -- device
pickers filled from a background scan, a Test button and level meter per source, the status line.

``list_devices`` / ``probe_device`` are swapped on the panel module for fakes; the panel's worker
threads are real and the tests wait for them, then poll once (``_settle``). Nothing touches a device,
pyaudiowpatch or the developer's config.toml (``Config()`` objects with no path).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QScrollArea

from exocortex.audio.capture import DeviceInfo, ProbeResult
from exocortex.config import Config
from exocortex.ui import audio_panel as panel_mod
from exocortex.ui.audio_panel import NOT_TESTED_TEXT, AudioPanel
from exocortex.ui.settings import FIELDS, Field, SettingsDialog, _DeviceCombo, all_fields, get_field_value, \
    make_field_widget, set_field_value
from exocortex.ui.theme import COLORS
from tests.test_settings import qapp  # noqa: F401 -- reused session fixture

LOOP_A = DeviceInfo(index=14, name="Odyssey G95SD (NVIDIA High Definition Audio) [Loopback]", is_loopback=True,
                    is_input=True, sample_rate=48000, channels=2, is_default=True)
LOOP_B = DeviceInfo(index=16, name="Speakers (Realtek(R) Audio) [Loopback]", is_loopback=True, is_input=True,
                    sample_rate=48000, channels=2, is_default=False)
MIC_A = DeviceInfo(index=1, name="Microphone Array (AMD Audio Device)", is_loopback=False, is_input=True,
                   sample_rate=48000, channels=2, is_default=True)
SPEAKERS = DeviceInfo(index=15, name="Speakers (Realtek(R) Audio)", is_loopback=False, is_input=False,
                      sample_rate=48000, channels=2, is_default=False)
DEVICES = [LOOP_A, MIC_A, SPEAKERS, LOOP_B]


def _probe_result(kind: str, spec: str, *, signal: str = "ok") -> ProbeResult:
    device = LOOP_A if kind == "system" else MIC_A
    if signal == "error":
        return ProbeResult(kind, spec, False, None, 0.0, 0, 0.0, 0.0, "no such device")
    if signal == "none":
        return ProbeResult(kind, spec, True, device, 1.5, 0, 0.0, 0.0)
    if signal == "silent":
        return ProbeResult(kind, spec, True, device, 1.5, 40, 5e-5, 1e-4)
    return ProbeResult(kind, spec, True, device, 1.5, 40, 0.2, 0.3)


class _FakeProbe:
    """Stands in for ``probe_device``: records the call, feeds two levels, returns a canned result."""

    def __init__(self, signal: str = "ok", *, raise_exc: Exception | None = None,
                 gate: threading.Event | None = None) -> None:
        self.signal = signal
        self.raise_exc = raise_exc
        self.gate = gate                                              # when given: block until the test opens it
        self.calls: list[dict[str, Any]] = []

    def __call__(self, kind: str, spec: str, *, seconds: float, keepalive: bool, on_level: Any = None) -> ProbeResult:
        self.calls.append({"kind": kind, "spec": spec, "seconds": seconds, "keepalive": keepalive})
        if on_level is not None:
            on_level(0.1, 0.2)
        if self.gate is not None:
            assert self.gate.wait(5.0), "the test never opened the gate"
        if on_level is not None:
            on_level(0.25, 0.4)
        if self.raise_exc is not None:
            raise self.raise_exc
        return _probe_result(kind, spec, signal=self.signal)


class _Actions:
    """Only an object whose CLASS defines audio_status (and optimize_snapshot, for _resolve_actions) counts."""

    def __init__(self, status: dict[str, Any]) -> None:
        self.status = status

    def audio_status(self) -> dict[str, Any]:
        return self.status

    def optimize_snapshot(self) -> dict[str, Any]:
        return {}


def _settle(panel: AudioPanel, timeout: float = 5.0) -> None:
    """Wait for the panel's worker threads, then deliver their results the way the timer would."""
    deadline = time.monotonic() + timeout
    while panel.pending() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not panel.pending(), "a panel job did not finish in time"
    panel._poll()


def _dialog(monkeypatch: pytest.MonkeyPatch, cfg: Config | None = None, *, devices: Any = None,
            actions: Any = None) -> SettingsDialog:
    monkeypatch.setattr(panel_mod, "list_devices", (lambda: list(DEVICES)) if devices is None else devices)
    dlg = SettingsDialog(cfg or Config(), None, actions=actions)
    assert isinstance(dlg._audio_panel, AudioPanel)
    _settle(dlg._audio_panel)
    return dlg


def _rows(combo: _DeviceCombo) -> list[tuple[str, str]]:
    return [(combo.itemText(i), str(combo.itemData(i))) for i in range(combo.count())]


# ---------------------------------------------------------------------------------------------
# the fields
# ---------------------------------------------------------------------------------------------
def test_audio_tab_has_device_pickers_live_switches_and_restart_only_vad_knobs() -> None:
    by_path = {f.path: f for f in all_fields()}
    for path in ("audio.system_device", "audio.mic_device"):
        assert by_path[path].kind == "device" and by_path[path] in FIELDS["Audio"]
    for path in ("audio.capture_system", "audio.capture_mic", "audio.loopback_keepalive"):
        assert by_path[path].kind == "bool" and not by_path[path].restart       # applied live since 2026-10-29
    for path in ("audio.vad_threshold", "audio.min_silence_s", "audio.min_speech_s", "audio.max_utterance_s",
                 "audio.pre_roll_s"):
        assert by_path[path].kind == "float" and by_path[path].restart
    keys = [f.key for f in FIELDS["Audio"]]
    assert keys[:3] == ["capture_system", "system_device", "loopback_keepalive"]
    assert keys[3:6] == ["capture_mic", "mic_device", "mic_start_muted"]
    assert list(FIELDS)[-1] == "Optimize" and list(FIELDS)[0] == "Completion"


# ---------------------------------------------------------------------------------------------
# _DeviceCombo
# ---------------------------------------------------------------------------------------------
def test_device_kind_builds_an_editable_combo_whose_value_is_a_spec(qapp: QApplication) -> None:
    f = Field("audio", "mic_device", "Microphone", "device", help="h")
    w = make_field_widget(f)
    assert isinstance(w, _DeviceCombo) and w.isEditable() and w.toolTip() == "h"
    assert get_field_value(w, f) == "default" and _rows(w) == [("Default", "default")]
    for spec in ("Odyssey", "14", "default"):
        set_field_value(w, f, spec)
        assert get_field_value(w, f) == spec
    set_field_value(w, f, "")
    assert get_field_value(w, f) == "default"
    set_field_value(w, f, "Default")
    assert get_field_value(w, f) == "default"


def test_device_combo_rows_carry_specs_and_keep_typed_text(qapp: QApplication) -> None:
    w = _DeviceCombo()
    w.set_spec("Odyssey")                                             # a substring from the config
    w.set_devices([LOOP_A, LOOP_B], LOOP_A.name)
    assert _rows(w) == [(f"Default  ({LOOP_A.name})", "default"), (LOOP_A.name, LOOP_A.name),
                        (LOOP_B.name, LOOP_B.name)]
    assert w.spec() == "Odyssey" and w.currentIndex() == -1          # shown as typed, never "corrected"
    assert "index 14" in w.itemData(1, 3) and "Windows default" in w.itemData(1, 3)   # ToolTipRole

    w.set_spec(LOOP_B.name)                                           # a full name: its row is selected
    assert w.currentIndex() == 2 and w.spec() == LOOP_B.name
    w.setCurrentIndex(0)                                              # the "Default (...)" row stores "default"
    assert w.spec() == "default"
    w.setCurrentIndex(1)
    w.set_devices([LOOP_B], None)                                     # a rescan that lost the chosen device
    assert w.spec() == LOOP_A.name and w.currentIndex() == -1         # kept as text, still what resolves


# ---------------------------------------------------------------------------------------------
# the page: scan, pickers, status line
# ---------------------------------------------------------------------------------------------
def test_dialog_builds_the_audio_page_and_fills_the_pickers_from_the_scan(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    dlg = _dialog(monkeypatch)
    try:
        panel = dlg._audio_panel
        assert dlg.tabs.widget(list(FIELDS).index("Audio")) is panel
        sys_combo, mic_combo = dlg._widgets["audio.system_device"], dlg._widgets["audio.mic_device"]
        assert _rows(sys_combo) == [(f"Default  ({LOOP_A.name})", "default"), (LOOP_A.name, LOOP_A.name),
                                    (LOOP_B.name, LOOP_B.name)]
        assert _rows(mic_combo) == [(f"Default  ({MIC_A.name})", "default"), (MIC_A.name, MIC_A.name)]
        assert panel.status_label.text() == "2 loopback devices, 1 microphone."
        assert panel.refresh_btn.isEnabled()
        # every form widget found a home on the page (none left in the never-shown scroll shell)
        for f in FIELDS["Audio"]:
            w = dlg._widgets[f.path]
            assert w.parentWidget() is not None and w.window() is dlg
        assert isinstance(dlg._widgets["audio.vad_threshold"], QDoubleSpinBox)
        assert panel.vad_box.isVisibleTo(panel)
    finally:
        dlg.close()
        qapp.processEvents()


def test_scan_failure_degrades_to_typing(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> list[DeviceInfo]:
        raise RuntimeError("pyaudiowpatch is required for audio capture but is not installed")

    dlg = _dialog(monkeypatch, devices=_boom)
    try:
        panel = dlg._audio_panel
        assert panel.status_label.text().startswith("Device list unavailable: pyaudiowpatch is required")
        assert "can still be typed" in panel.status_label.text()
        combo = dlg._widgets["audio.system_device"]
        assert combo.isEditable() and _rows(combo) == [("Default", "default")]
        combo.set_spec("Odyssey")
        assert dlg.changes() == {"audio.system_device": "Odyssey"}
    finally:
        dlg.close()
        qapp.processEvents()


def test_rescan_refills_and_status_shows_what_the_app_has_open(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    status = {"running": True, "system": {"enabled": True, "spec": "default", "device": LOOP_A.name},
              "mic": {"enabled": False, "spec": "default", "device": ""}}
    dlg = _dialog(monkeypatch, actions=_Actions(status))
    try:
        panel = dlg._audio_panel
        assert panel.status_label.text() == (f"2 loopback devices, 1 microphone. Open now: system: {LOOP_A.name}; "
                                             "mic: off.")
        monkeypatch.setattr(panel_mod, "list_devices", lambda: [LOOP_B, MIC_A])
        assert panel.rescan()
        assert not panel.refresh_btn.isEnabled()                      # disabled at once; the fake scan may already be done
        _settle(panel)
        assert _rows(dlg._widgets["audio.system_device"]) == [("Default", "default"), (LOOP_B.name, LOOP_B.name)]
        assert panel.status_label.text().startswith("1 loopback device, 1 microphone.")
        status["running"] = False
        dlg.load()                                                    # the dialog's load() refreshes the panel
        assert panel.status_label.text().endswith("Audio capture is not running.")
    finally:
        dlg.close()
        qapp.processEvents()


def test_catch_all_fakes_do_not_count_as_an_engine(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    class _CatchAll:
        def __getattr__(self, name: str) -> Any:
            return lambda *a, **k: {"running": True}

    dlg = _dialog(monkeypatch, actions=_CatchAll())
    try:
        assert dlg._audio_panel._actions is None
        assert dlg._audio_panel.status_label.text() == "2 loopback devices, 1 microphone."
    finally:
        dlg.close()
        qapp.processEvents()


# ---------------------------------------------------------------------------------------------
# Test button
# ---------------------------------------------------------------------------------------------
def test_test_button_probes_the_widgets_values_and_reports(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()
    probe = _FakeProbe("ok", gate=gate)
    monkeypatch.setattr(panel_mod, "probe_device", probe)
    dlg = _dialog(monkeypatch)
    try:
        panel = dlg._audio_panel
        dlg._widgets["audio.system_device"].set_spec("Yeti")          # not applied: the test must still use it
        dlg._widgets["audio.loopback_keepalive"].setChecked(False)
        assert panel.results["system"].text() == NOT_TESTED_TEXT
        panel.test_btns["system"].click()
        assert panel.probe_running() and not panel.start_test("mic")  # one at a time
        assert not panel.test_btns["system"].isEnabled() and not panel.test_btns["mic"].isEnabled()
        assert panel.results["system"].text() == "Testing 'Yeti' for 1.5 s…"
        deadline = time.monotonic() + 2.0                             # the first level has landed: the meter moves
        while time.monotonic() < deadline and panel.meters["system"].value() == 0:
            panel._poll()
            time.sleep(0.01)
        assert panel.meters["system"].value() > 0
        gate.set()
        _settle(panel)
        assert probe.calls == [{"kind": "system", "spec": "Yeti", "seconds": 1.5, "keepalive": False}]
        res = panel.last_result["system"]
        assert res.signal == "ok" and panel.results["system"].text() == res.describe()
        assert COLORS["ok"] in panel.results["system"].styleSheet()
        assert panel.meters["system"].value() > 0                     # the final rms
        assert panel.test_btns["system"].isEnabled() and panel.test_btns["mic"].isEnabled()
        assert dlg.changes() == {"audio.system_device": "Yeti", "audio.loopback_keepalive": False}   # nothing applied

        dlg._widgets["audio.system_device"].set_spec("default")       # a new device: the verdict no longer applies
        assert panel.results["system"].text() == NOT_TESTED_TEXT and "system" not in panel.last_result
    finally:
        dlg.close()
        qapp.processEvents()


@pytest.mark.parametrize("signal, color", [("none", "warn"), ("silent", "warn"), ("error", "err")])
def test_test_button_tones_follow_the_verdict(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch, signal: str, color: str
) -> None:
    monkeypatch.setattr(panel_mod, "probe_device", _FakeProbe(signal))
    dlg = _dialog(monkeypatch)
    try:
        panel = dlg._audio_panel
        assert panel.start_test("mic")
        _settle(panel)
        assert COLORS[color] in panel.results["mic"].styleSheet()
        assert panel.results["mic"].text() == panel.last_result["mic"].describe()
    finally:
        dlg.close()
        qapp.processEvents()


def test_test_button_survives_a_raising_probe(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(panel_mod, "probe_device", _FakeProbe(raise_exc=RuntimeError("portaudio exploded")))
    dlg = _dialog(monkeypatch)
    try:
        panel = dlg._audio_panel
        assert panel.start_test("mic")
        _settle(panel)
        assert panel.results["mic"].text() == "Test failed: portaudio exploded"
        assert COLORS["err"] in panel.results["mic"].styleSheet()
        assert panel.test_btns["mic"].isEnabled() and "mic" not in panel.last_result
    finally:
        dlg.close()
        qapp.processEvents()


# ---------------------------------------------------------------------------------------------
# the form underneath still owns load / changes / apply / restore
# ---------------------------------------------------------------------------------------------
def test_picking_a_device_round_trips_into_config_by_full_name(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config()
    dlg = _dialog(monkeypatch, cfg)
    got: list[dict[str, Any]] = []
    dlg.applied.connect(got.append)
    try:
        combo = dlg._widgets["audio.system_device"]
        combo.setCurrentIndex(2)                                      # LOOP_B's row
        dlg._widgets["audio.capture_mic"].setChecked(False)
        assert dlg.changes() == {"audio.system_device": LOOP_B.name, "audio.capture_mic": False}
        changed = dlg.apply()
        assert got == [changed] and cfg.audio.system_device == LOOP_B.name and cfg.audio.capture_mic is False
        assert dlg.note.text() == "Applied."                          # live fields: no "(restart)" note
        assert dlg.changes() == {}
        dlg.load()
        assert combo.currentIndex() == 2 and combo.spec() == LOOP_B.name

        dlg._widgets["audio.vad_threshold"].setValue(0.6)
        dlg.apply()
        assert "next launch" in dlg.note.text() and "Speech probability" in dlg.note.text()

        dlg.restore_defaults()
        assert combo.spec() == "default" and combo.currentIndex() == 0
        assert dlg.changes()["audio.system_device"] == "default"
    finally:
        dlg.close()
        qapp.processEvents()


def test_plain_form_fallback_when_the_panel_cannot_be_built(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("no panel today")

    monkeypatch.setattr(panel_mod.AudioPanel, "__init__", _boom)
    dlg = SettingsDialog(Config(), None)
    try:
        assert dlg._audio_panel is None
        page = dlg.tabs.widget(list(FIELDS).index("Audio"))
        assert isinstance(page, QScrollArea)
        assert isinstance(dlg._widgets["audio.mic_device"], _DeviceCombo)
        assert isinstance(dlg._widgets["audio.capture_mic"], QCheckBox)
        dlg._widgets["audio.mic_device"].set_spec("Yeti")
        assert dlg.changes() == {"audio.mic_device": "Yeti"}
    finally:
        dlg.close()
        qapp.processEvents()
