"""The Settings > Audio page (2026-10-29): pick the capture devices by hand and test them.

Why a panel of its own. ``FIELDS["Audio"]`` stays an ordinary tab -- the dialog builds its
:class:`~exocortex.ui.settings.FieldForm` as usual and hands it here -- but a device is not a value
one types blind: the page lists what WASAPI actually offers (loopback devices for system audio,
input devices for the microphone), says which device "Default" resolves to, shows what the running
app has open, and lets each source be *tested* before it is applied: a real
:func:`exocortex.audio.capture.probe_device` run (resolve, open, downmix, resample -- the production
path) for a moment, with a live level meter and a plain-language verdict. The same three rules as
:mod:`exocortex.ui.optimize_panel` shaped it:

* **The form stays the source of truth.** The panel only moves the form's widgets into its own
  layout (as the very last step of construction, so a failure earlier leaves the plain form usable
  as the fallback page); ``load`` / ``changes`` / ``apply`` / ``restore_defaults`` keep working
  untouched, and OK / Apply persist the values like any other tab. A test reads the *widgets* (the
  spec in the picker, the keep-alive box), not the config, so a device can be checked before it is
  applied.
* **Poll, never push.** A device scan and a probe each run on a short daemon thread that writes
  into a plain :class:`_Job` holder; a 100 ms ``QTimer`` reads the holder (the live level, then the
  result) and stops when nothing is pending. No worker thread ever touches a widget, so closing the
  dialog mid-test cannot reach a dead object -- the thread finishes into a holder nobody reads.
* **Degrade, never raise.** Without ``pyaudiowpatch`` (off Windows, tests) the scan fails and the
  status line says so; the pickers stay editable (a name or an index can still be typed) and Test
  reports the open error in words. Without an engine (a headless dialog, the overlay tests'
  catch-all fakes) the "open now" part of the status line is simply absent. Every ``actions.*``
  call is wrapped.

Applying is the dialog's: Apply / OK write the values and ``Engine.apply_settings`` reopens the
affected capture live (``SpeechPipeline.restart_source``); the panel never touches the pipeline.
Tests swap :func:`list_devices` / :func:`probe_device` on this module for fakes.

Qt trap noted once more: ``clicked(bool)`` must never be connected straight to a method with an
optional argument, so every button goes through ``lambda _=False: ...``.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from exocortex.audio.capture import candidates_for, list_devices, probe_device
from exocortex.config import Config
from exocortex.ui.theme import COLORS
from exocortex.ui.widgets import level_to_percent

log = logging.getLogger(__name__)

POLL_MS = 100              # holder poll while a scan or a probe runs
PROBE_SECONDS = 1.5        # how long Test listens

KINDS: tuple[str, ...] = ("system", "mic")
TITLES: dict[str, str] = {"system": "System audio (the other side)", "mic": "Microphone (you)"}
DEVICE_PATH: dict[str, str] = {"system": "audio.system_device", "mic": "audio.mic_device"}
ENABLE_PATH: dict[str, str] = {"system": "audio.capture_system", "mic": "audio.capture_mic"}
EXTRA_PATHS: dict[str, tuple[str, ...]] = {"system": ("audio.loopback_keepalive",), "mic": ("audio.mic_start_muted",)}
VAD_PATHS: tuple[str, ...] = ("audio.vad_threshold", "audio.min_silence_s", "audio.min_speech_s",
                              "audio.max_utterance_s", "audio.pre_roll_s")

SCANNING_TEXT = "Looking for audio devices…"
NOT_TESTED_TEXT = "Not tested yet."
TESTING_TEXT = "Testing {spec!r} for {seconds:.1f} s…"
NO_DEVICES_TEXT = "Device list unavailable: {error} A device name or index can still be typed."
NOT_RUNNING_TEXT = "Audio capture is not running."

_TONE_COLORS: dict[str, str] = {"ok": COLORS["ok"], "warn": COLORS["warn"], "err": COLORS["err"],
                                "dim": COLORS["text_faint"]}
_SIGNAL_TONE: dict[str, str] = {"ok": "ok", "silent": "warn", "none": "warn", "error": "err"}

_PANEL_QSS = f"""
QLabel#hint {{ color: {COLORS['text_faint']}; }}
QGroupBox {{ border: 1px solid {COLORS['border']}; border-radius: 6px; margin-top: 10px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {COLORS['text_dim']}; }}
QProgressBar {{ background: {COLORS['panel']}; border: 1px solid {COLORS['border']}; border-radius: 4px; }}
QProgressBar::chunk {{ background: {COLORS['accent']}; border-radius: 3px; }}
"""


class _Job:
    """One background job -- a device scan or a probe -- and what it produced. Written by its own
    daemon thread, read by the panel's timer; ``done`` flips last, after ``result`` / ``error``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.done = threading.Event()
        self.result: Any = None
        self.error = ""
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._level = (0.0, 0.0)

    def set_level(self, rms: float, peak: float) -> None:
        """The probe's ``on_level`` relay (called from the capture's consumer thread)."""
        with self._lock:
            self._level = (float(rms), float(peak))

    def level(self) -> tuple[float, float]:
        with self._lock:
            return self._level

    def run(self, fn: Callable[[], Any]) -> None:
        def _run() -> None:
            try:
                self.result = fn()
            except Exception as exc:  # noqa: BLE001 - a job's failure is a result, never a crash
                log.exception("%s failed", self.name)
                self.error = str(exc) or type(exc).__name__
            finally:
                self.done.set()

        self.thread = threading.Thread(target=_run, name=self.name, daemon=True)
        self.thread.start()


class AudioPanel(QWidget):
    """The Audio tab's page. ``form`` is the dialog's :class:`~exocortex.ui.settings.FieldForm` for
    ``FIELDS["Audio"]`` (its widgets are moved in here); ``host`` is the dialog; ``actions`` is the
    engine (used only for ``audio_status()``) or ``None``. See the module docstring."""

    def __init__(self, cfg: Config, actions: Any, form: Any, host: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        # only an object whose CLASS defines audio_status counts (the overlay tests' catch-all fakes
        # answer hasattr for every name and must not look like an engine)
        self._actions = actions if callable(getattr(type(actions), "audio_status", None)) else None
        self._form = form
        self._host = host

        self._devices: list[Any] = []
        self._scan_error = ""
        self._status: dict[str, Any] = {}
        self._scan: _Job | None = None
        self._probe: _Job | None = None
        self._probe_kind = ""
        self.last_result: dict[str, Any] = {}

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

        self.setStyleSheet(_PANEL_QSS)
        self._build()
        self._place_form_widgets()          # last structural step: see the module docstring
        self._wire()
        self.refresh()
        self.rescan()

    # =========================================================================================
    # construction
    # =========================================================================================
    @staticmethod
    def _label(text: str = "", *, name: str = "", parent: QWidget | None = None) -> QLabel:
        lab = QLabel(text, parent)
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if name:
            lab.setObjectName(name)
        return lab

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll)
        body = QWidget(self.scroll)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self.scroll.setWidget(body)

        self.boxes: dict[str, QGroupBox] = {}
        self.forms: dict[str, QFormLayout] = {}
        self.device_rows: dict[str, QHBoxLayout] = {}
        self.test_btns: dict[str, QPushButton] = {}
        self.meters: dict[str, QProgressBar] = {}
        self.results: dict[str, QLabel] = {}
        for kind in KINDS:
            box = QGroupBox(TITLES[kind], body)
            form = QFormLayout(box)
            form.setSpacing(6)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            row = QHBoxLayout()                 # the picker is inserted at 0 by _place_form_widgets
            btn = QPushButton("Test", box)
            btn.setToolTip(f"Open this device for {PROBE_SECONDS:.1f} s the way the app does and report what arrives.")
            row.addWidget(btn)
            meter = QProgressBar(box)
            meter.setRange(0, 100)
            meter.setTextVisible(False)
            meter.setFixedHeight(8)
            result = self._label(NOT_TESTED_TEXT, name="hint", parent=box)
            self.boxes[kind], self.forms[kind], self.device_rows[kind] = box, form, row
            self.test_btns[kind], self.meters[kind], self.results[kind] = btn, meter, result
            lay.addWidget(box)

        self.vad_box = QGroupBox("Voice detection (restart)", body)
        self.vad_form = QFormLayout(self.vad_box)
        self.vad_form.setSpacing(6)
        self.vad_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        lay.addWidget(self.vad_box)

        self.other_form = QFormLayout()
        self.other_form.setSpacing(6)
        self.other_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        lay.addLayout(self.other_form)

        bottom = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh devices", body)
        self.refresh_btn.setToolTip("Enumerate the WASAPI devices again (after plugging something in).")
        bottom.addWidget(self.refresh_btn)
        self.status_label = self._label(SCANNING_TEXT, name="hint", parent=body)
        bottom.addWidget(self.status_label, 1)
        lay.addLayout(bottom)
        lay.addStretch(1)

    def _row_label(self, path: str, parent: QWidget | None) -> QLabel:
        f = self._fields[path]
        label = QLabel(f.label + (" (restart)" if f.restart else ""), parent)
        label.setWordWrap(True)
        if f.help:
            label.setToolTip(f.help)
        return label

    def _add_row(self, form: QFormLayout, path: str, placed: set[str]) -> None:
        w = self._form.widgets.get(path)
        if w is None:
            return
        form.addRow(self._row_label(path, form.parentWidget()), w)
        placed.add(path)

    def _place_form_widgets(self) -> None:
        """Move the form's widgets into this page: per source its on/off switch, the device picker
        (next to Test), the meter and verdict, then the source's own switch; the voice-detection
        knobs in their box; anything else below. Done last so a failure while building leaves
        ``form`` whole for the fallback page."""
        self._fields = {f.path: f for f in self._form.fields}
        widgets = self._form.widgets
        placed: set[str] = set()
        for kind in KINDS:
            form = self.forms[kind]
            self._add_row(form, ENABLE_PATH[kind], placed)
            combo = widgets.get(DEVICE_PATH[kind])
            if combo is not None:
                self.device_rows[kind].insertWidget(0, combo, 1)
                form.addRow(self._row_label(DEVICE_PATH[kind], self.boxes[kind]), self.device_rows[kind])
                placed.add(DEVICE_PATH[kind])
            form.addRow(self.meters[kind])
            form.addRow(self.results[kind])
            for path in EXTRA_PATHS[kind]:
                self._add_row(form, path, placed)
        for path in VAD_PATHS:
            self._add_row(self.vad_form, path, placed)
        self.vad_box.setVisible(any(p in placed for p in VAD_PATHS))
        for f in self._form.fields:
            if f.path not in placed:
                self._add_row(self.other_form, f.path, placed)

    def _wire(self) -> None:
        for kind in KINDS:
            self.test_btns[kind].clicked.connect(lambda _=False, k=kind: self.start_test(k))
            combo = self._combo(kind)
            if combo is not None:
                combo.currentTextChanged.connect(lambda _t="", k=kind: self._on_spec_edited(k))
        self.refresh_btn.clicked.connect(lambda _=False: self.rescan())

    # =========================================================================================
    # small accessors
    # =========================================================================================
    def _combo(self, kind: str) -> Any:
        return self._form.widgets.get(DEVICE_PATH[kind])

    def _set_result(self, kind: str, text: str, tone: str) -> None:
        label = self.results[kind]
        label.setText(text)
        label.setStyleSheet(f"color: {_TONE_COLORS.get(tone, COLORS['text_faint'])};")

    def pending(self) -> bool:
        """Whether a scan or a probe is still running (the timer polls while this is true)."""
        return any(j is not None and not j.done.is_set() for j in (self._probe, self._scan))

    def probe_running(self) -> bool:
        return self._probe is not None and not self._probe.done.is_set()

    # =========================================================================================
    # actions: scan, test
    # =========================================================================================
    def rescan(self) -> bool:
        """Enumerate the devices again on a worker thread; the pickers refill when it is done.
        Returns False when a scan is already running."""
        if self._scan is not None and not self._scan.done.is_set():
            return False
        job = _Job("audio-scan")
        self._scan = job
        self.refresh_btn.setEnabled(False)
        self._render_status()
        job.run(lambda: list(list_devices()))
        self._timer.start()
        return True

    def start_test(self, kind: str) -> bool:
        """Probe ``kind``'s device as the WIDGETS describe it right now (the picker's spec, the
        keep-alive box) on a worker thread; the meter follows the live level and the verdict lands
        when it is done. One probe at a time; returns False when one is already running."""
        if kind not in KINDS or self.probe_running():
            return False
        combo = self._combo(kind)
        spec = str(combo.spec()) if combo is not None and hasattr(combo, "spec") else "default"
        ka = self._form.widgets.get("audio.loopback_keepalive")
        keepalive = bool(ka.isChecked()) if isinstance(ka, QCheckBox) else bool(self._cfg.audio.loopback_keepalive)
        job = _Job(f"audio-probe-{kind}")
        self._probe, self._probe_kind = job, kind
        for btn in self.test_btns.values():
            btn.setEnabled(False)
        self.meters[kind].setValue(0)
        self._set_result(kind, TESTING_TEXT.format(spec=spec, seconds=PROBE_SECONDS), "dim")
        job.run(lambda: probe_device(kind, spec, seconds=PROBE_SECONDS, keepalive=keepalive,  # type: ignore[arg-type]
                                     on_level=job.set_level))
        self._timer.start()
        return True

    def _on_spec_edited(self, kind: str) -> None:
        """A different device is being described: the last verdict no longer applies to it."""
        if self.probe_running() and self._probe_kind == kind:
            return
        self._set_result(kind, NOT_TESTED_TEXT, "dim")
        self.meters[kind].setValue(0)
        self.last_result.pop(kind, None)

    # =========================================================================================
    # polling (the only place worker results reach widgets)
    # =========================================================================================
    def _poll(self) -> None:
        probe = self._probe
        if probe is not None:
            rms, _peak = probe.level()
            self.meters[self._probe_kind].setValue(level_to_percent(rms))
            if probe.done.is_set():
                self._finish_probe(probe)
        scan = self._scan
        if scan is not None and scan.done.is_set():
            self._finish_scan(scan)
        if not self.pending():
            self._timer.stop()

    def _finish_probe(self, job: _Job) -> None:
        self._probe = None
        kind = self._probe_kind
        for btn in self.test_btns.values():
            btn.setEnabled(True)
        res = job.result
        if job.error or res is None:
            self._set_result(kind, f"Test failed: {job.error or 'no result'}", "err")
            return
        self.last_result[kind] = res
        self.meters[kind].setValue(level_to_percent(float(getattr(res, "rms", 0.0) or 0.0)))
        tone = _SIGNAL_TONE.get(str(getattr(res, "signal", "error")), "err")
        try:
            text = str(res.describe())
        except Exception:  # noqa: BLE001 - a foreign result object: show what we can
            text = str(res)
        self._set_result(kind, text, tone)

    def _finish_scan(self, job: _Job) -> None:
        self._scan = None
        self.refresh_btn.setEnabled(True)
        if job.error:
            self._scan_error, self._devices = job.error, []
        else:
            self._scan_error, self._devices = "", list(job.result or [])
        for kind in KINDS:
            combo = self._combo(kind)
            if combo is None or not hasattr(combo, "set_devices"):
                continue
            cands = candidates_for(kind, self._devices)  # type: ignore[arg-type]
            default = next((d.name for d in cands if d.is_default), None)
            combo.set_devices(cands, default)
        self._render_status()

    # =========================================================================================
    # status line: devices found + what the running app has open
    # =========================================================================================
    def refresh(self) -> None:
        """Re-read what the running app has open (``actions.audio_status``) and redraw the status
        line. Never starts any work; the dialog calls it on every ``load()``."""
        self._status = self._read_status()
        self._render_status()

    def _read_status(self) -> dict[str, Any]:
        if self._actions is None:
            return {}
        try:
            status = self._actions.audio_status()
        except Exception:  # noqa: BLE001 - an engine fault must never take the Settings dialog down
            log.exception("actions.audio_status() failed")
            return {}
        return status if isinstance(status, dict) else {}

    def _render_status(self) -> None:
        parts: list[str] = []
        if self._scan is not None and not self._scan.done.is_set():
            parts.append(SCANNING_TEXT)
        elif self._scan_error:
            parts.append(NO_DEVICES_TEXT.format(error=self._scan_error.rstrip(".") + "."))
        else:
            n_loop = len(candidates_for("system", self._devices))  # type: ignore[arg-type]
            n_mic = len(candidates_for("mic", self._devices))      # type: ignore[arg-type]
            parts.append(f"{n_loop} loopback device{'' if n_loop == 1 else 's'}, "
                         f"{n_mic} microphone{'' if n_mic == 1 else 's'}.")
        status = self._status
        if status:
            if status.get("running"):
                open_now: list[str] = []
                for kind in KINDS:
                    d = status.get(kind) if isinstance(status.get(kind), dict) else {}
                    if not d.get("enabled"):
                        open_now.append(f"{kind}: off")
                    else:
                        open_now.append(f"{kind}: {d.get('device') or 'not open'}")
                parts.append("Open now: " + "; ".join(open_now) + ".")
            else:
                parts.append(NOT_RUNNING_TEXT)
        self.status_label.setText(" ".join(parts))

    # =========================================================================================
    # Qt events
    # =========================================================================================
    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.refresh()
        if self.pending() and not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().hideEvent(event)
        self._timer.stop()                     # a pending result is picked up on the next show
