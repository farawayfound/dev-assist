"""The Check panel under Settings > Audio: live meters, the Windows mute, the speaker-routing check.

Speakers are attributed by source -- the mic feed is "Me", the output feed is clustered into
Speaker 1, 2, ... -- so the transcript is only as right as the routing. This panel shows it:

* **Test devices** opens the devices the two pickers above name (not yet applied, so a choice can
  be tried before it is kept) and shows a meter and a verdict for each. A mic delivering digital
  silence gets the reason when Windows knows it (the endpoint is muted) and an **Unmute** button.
* **Check speaker routing** plays a short chirp on the output behind the chosen loopback and
  measures where it arrives (:func:`exocortex.audio.check.routing_check`): it should reach the
  output feed and not the mic. The verdict says whether echo suppression is still worth having.

Unlike :mod:`exocortex.ui.settings` this module does know about threads: devices are opened, closed
and the routing check run on daemon threads, and every result comes back through a queued Qt signal.
Every device / Windows call is injectable for tests.
"""
from __future__ import annotations

import html
import logging
import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QHideEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from exocortex.audio.check import LevelTracker, RoutingResult
from exocortex.bus import Source
from exocortex.ui.theme import COLORS
from exocortex.ui.widgets import level_to_percent

log = logging.getLogger(__name__)

_KINDS = ("mic", "system")
_SOURCE = {"mic": Source.MIC, "system": Source.SYSTEM}
_TICK_MS = 300
#: Windows input volume under this is reported (it reads as near-silence, not as a fault).
_LOW_VOLUME = 0.05

_MIC_VERDICTS = {
    "waiting": ("listening...", "text_dim"),
    "ok": ("hearing you", "ok"),
    "quiet": ("very quiet: speak up, or raise the input volume in Windows", "warn"),
    "silent": ("digital silence: muted in Windows, mic privacy, or a mic-mute key", "err"),
}
_SYSTEM_VERDICTS = {
    "waiting": ("nothing playing (play something)", "text_dim"),
    "ok": ("hearing the output", "ok"),
    "quiet": ("nothing playing (play something)", "text_dim"),
    "silent": ("nothing playing (play something)", "text_dim"),
}


def _default_capture_factory(source: Source, spec: str, **kwargs: Any) -> Any:
    from exocortex.audio.capture import AudioCapture

    return AudioCapture(source, spec, **kwargs)


def _default_routing(system_spec: str, mic_spec: str) -> RoutingResult:
    from exocortex.audio.check import routing_check

    return routing_check(system_spec, mic_spec)


def _default_endpoint_state(name: str) -> Any:
    from exocortex.audio.endpoint import capture_endpoint_state

    return capture_endpoint_state(name)


def _default_set_endpoint_muted(name: str, muted: bool) -> bool:
    from exocortex.audio.endpoint import set_capture_muted

    return set_capture_muted(name, muted)


def _span(text: str, color: str) -> str:
    return f'<span style="color:{COLORS[color]}">{html.escape(text)}</span>'


class AudioCheckPanel(QWidget):
    """Meters + checks for the devices ``mic_combo`` / ``system_combo`` name; ``echo_box`` is the
    Audio tab's echo-suppression checkbox, which the routing verdict may offer to flip (the form
    still applies it on Apply / OK, like any other edit)."""

    _level = Signal(str, float, float)       # kind, rms, peak     (capture threads)
    _opened = Signal(int, str, str, str)     # generation, kind, device name, error
    _routed = Signal(object, str)            # RoutingResult | None, error

    def __init__(
        self,
        mic_combo: QComboBox,
        system_combo: QComboBox,
        echo_box: QCheckBox,
        parent: QWidget | None = None,
        *,
        capture_factory: Callable[..., Any] | None = None,
        routing: Callable[[str, str], RoutingResult] | None = None,
        endpoint_state: Callable[[str], Any] | None = None,
        set_endpoint_muted: Callable[[str, bool], bool] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(parent)
        self._combos = {"mic": mic_combo, "system": system_combo}
        self._echo_box = echo_box
        self._factory = capture_factory or _default_capture_factory
        self._routing = routing or _default_routing
        self._endpoint_state = endpoint_state or _default_endpoint_state
        self._set_endpoint_muted = set_endpoint_muted or _default_set_endpoint_muted
        tracker_kwargs = {"clock": clock} if clock is not None else {}
        self._trackers = {k: LevelTracker(**tracker_kwargs) for k in _KINDS}
        self._captures: dict[str, Any] = {}
        self._names: dict[str, str] = {}
        self._verdicts: dict[str, str] = {}
        self._lock = threading.Lock()
        self._generation = 0
        self._echo_target: bool | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 10, 0, 0)
        outer.setSpacing(6)
        title = QLabel("Check", self)
        title.setStyleSheet("font-weight: 600;")
        outer.addWidget(title)
        intro = QLabel("The mic is transcribed as Me, the output feed as everyone else: these show whether "
                       "each feed hears what it should.", self)
        intro.setObjectName("help")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        self.meters: dict[str, QProgressBar] = {}
        self.statuses: dict[str, QLabel] = {}
        for row, (kind, label) in enumerate((("mic", "Microphone"), ("system", "Output feed"))):
            meter = QProgressBar(self)
            meter.setRange(0, 100)
            meter.setTextVisible(False)
            meter.setFixedWidth(120)
            meter.setFixedHeight(10)
            status = QLabel("", self)
            status.setWordWrap(True)
            self.meters[kind] = meter
            self.statuses[kind] = status
            grid.addWidget(QLabel(label, self), row, 0)
            grid.addWidget(meter, row, 1)
            grid.addWidget(status, row, 2)
        grid.setColumnStretch(2, 1)
        outer.addLayout(grid)

        win_row = QHBoxLayout()
        self.windows_label = QLabel("", self)
        self.windows_label.setWordWrap(True)
        self.windows_label.setStyleSheet(f"color: {COLORS['err']};")
        self.unmute_btn = QPushButton("Unmute in Windows", self)
        self.unmute_btn.clicked.connect(self._unmute)
        win_row.addWidget(self.windows_label, 1)
        win_row.addWidget(self.unmute_btn)
        outer.addLayout(win_row)
        self.windows_label.hide()
        self.unmute_btn.hide()

        buttons = QHBoxLayout()
        self.test_btn = QPushButton("Test devices", self)
        self.test_btn.setCheckable(True)
        self.test_btn.setToolTip("Open the devices chosen above (applied or not) and show what each one hears.")
        self.test_btn.toggled.connect(self._on_test_toggled)
        self.route_btn = QPushButton("Check speaker routing", self)
        self.route_btn.setToolTip("Plays a short chirp (~1 s) on the output, then measures whether it reached "
                                  "the output feed and whether the mic heard it. Stay quiet while it runs.")
        self.route_btn.clicked.connect(self.check_routing)
        buttons.addWidget(self.test_btn)
        buttons.addWidget(self.route_btn)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        self.result = QLabel("", self)
        self.result.setWordWrap(True)
        self.result.setTextFormat(Qt.TextFormat.RichText)
        outer.addWidget(self.result)
        self.echo_btn = QPushButton("", self)
        self.echo_btn.clicked.connect(self._apply_echo_recommendation)
        self.echo_btn.hide()
        echo_row = QHBoxLayout()
        echo_row.addWidget(self.echo_btn)
        echo_row.addStretch(1)
        outer.addLayout(echo_row)

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._level.connect(self._on_level)
        self._opened.connect(self._on_opened)
        self._routed.connect(self._on_routed)
        for combo in self._combos.values():
            combo.activated.connect(lambda _i: self._restart_if_testing())

    # ---- live test -------------------------------------------------------------------------
    @property
    def testing(self) -> bool:
        return self.test_btn.isChecked()

    def _spec(self, kind: str) -> str:
        return self._combos[kind].currentText().strip() or "default"

    def _on_test_toggled(self, on: bool) -> None:
        self.test_btn.setText("Stop test" if on else "Test devices")
        if on:
            self._start_test()
        else:
            self._stop_test()

    def _restart_if_testing(self) -> None:
        if self.testing:
            self._stop_test()
            self._start_test()

    def _start_test(self) -> None:
        with self._lock:
            self._generation += 1
            generation = self._generation
        for kind in _KINDS:
            self._trackers[kind].reset()
            self._names.pop(kind, None)
            self._verdicts.pop(kind, None)
            self.meters[kind].setValue(0)
            self._set_status(kind, "opening...", "text_dim")
            threading.Thread(target=self._open_worker, args=(generation, kind, self._spec(kind)),
                             name=f"audio-check-{kind}", daemon=True).start()
        self._timer.start()

    def _open_worker(self, generation: int, kind: str, spec: str) -> None:
        try:
            capture = self._factory(
                _SOURCE[kind], spec, target_rate=16000, frame_samples=512,
                on_frames=lambda _frame: None,
                on_level=lambda rms, peak: self._level.emit(kind, rms, peak),
                on_status=None, keepalive=False,
            )
            device = capture.start()
        except Exception as exc:  # noqa: BLE001 - reported beside the meter
            log.info("audio check: could not open %s %r: %s", kind, spec, exc)
            self._opened.emit(generation, kind, "", str(exc))
            return
        with self._lock:
            stale = generation != self._generation
            if not stale:
                self._captures[kind] = capture
        if stale:            # stopped or restarted while this one was opening
            _stop_quietly(capture)
            return
        self._opened.emit(generation, kind, str(getattr(device, "name", "")), "")

    def _stop_test(self) -> None:
        with self._lock:
            self._generation += 1
            captures = list(self._captures.values())
            self._captures.clear()
        self._timer.stop()
        for kind in _KINDS:
            self.meters[kind].setValue(0)
            self._set_status(kind, "", "text_dim")
        if captures:
            threading.Thread(target=lambda: [_stop_quietly(c) for c in captures],
                             name="audio-check-stop", daemon=True).start()

    def _on_level(self, kind: str, rms: float, peak: float) -> None:
        if not self.testing or kind not in self._names:
            return
        self._trackers[kind].push(rms, peak)
        self.meters[kind].setValue(level_to_percent(rms))

    def _on_opened(self, generation: int, kind: str, name: str, error: str) -> None:
        if generation != self._generation:
            return
        if error:
            self._set_status(kind, f"could not open: {error}", "err")
            return
        self._names[kind] = name
        self.statuses[kind].setToolTip(name)
        if kind == "mic":
            self.refresh_endpoint()
        self._tick()

    def _tick(self) -> None:
        for kind, name in self._names.items():
            verdict = self._trackers[kind].verdict()
            if verdict == "waiting":
                self.meters[kind].setValue(0)
            text, color = (_MIC_VERDICTS if kind == "mic" else _SYSTEM_VERDICTS)[verdict]
            self._set_status(kind, f"{name}: {text}", color)
            if kind == "mic" and verdict == "silent" and self._verdicts.get(kind) != "silent":
                self.refresh_endpoint()      # muted since the test started? say so
            self._verdicts[kind] = verdict

    def _set_status(self, kind: str, text: str, color: str) -> None:
        self.statuses[kind].setText(text)
        self.statuses[kind].setStyleSheet(f"color: {COLORS[color]};")

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802 - Qt override
        # the tab switched away or the dialog closed: never leave devices open behind it
        if self.testing:
            self.test_btn.setChecked(False)
        super().hideEvent(event)

    # ---- the Windows mute --------------------------------------------------------------------
    def refresh_endpoint(self) -> None:
        """Show the mic's Windows mute / input volume when either explains a silent mic."""
        name = self._names.get("mic") or self._spec("mic")
        try:
            state = self._endpoint_state(name)
        except Exception:  # noqa: BLE001
            log.debug("endpoint state unavailable", exc_info=True)
            state = None
        if state is not None and state.muted:
            self.windows_label.setText(f"Windows has {state.name!r} muted: it opens fine but delivers silence, "
                                       "so nothing you say is transcribed.")
            self.windows_label.show()
            self.unmute_btn.show()
        elif state is not None and state.volume < _LOW_VOLUME:
            self.windows_label.setText(f"Windows input volume for {state.name!r} is {round(state.volume * 100)}%: "
                                       "raise it in Settings > System > Sound > Input.")
            self.windows_label.show()
            self.unmute_btn.hide()
        else:
            self.windows_label.hide()
            self.unmute_btn.hide()

    def _unmute(self) -> None:
        name = self._names.get("mic") or self._spec("mic")
        if not self._set_endpoint_muted(name, False):
            self.windows_label.setText("Could not unmute it: use Settings > System > Sound > Input in Windows.")
            return
        self._trackers["mic"].reset()
        self.refresh_endpoint()

    # ---- the routing check ---------------------------------------------------------------------
    def check_routing(self) -> None:
        """Play the chirp and measure where it lands (on a daemon thread; ~2 s)."""
        self.route_btn.setEnabled(False)
        self.echo_btn.hide()
        self.result.setText(_span("Playing a short test sound: stay quiet for two seconds...", "text_dim"))
        system, mic = self._spec("system"), self._spec("mic")
        threading.Thread(target=self._routing_worker, args=(system, mic), name="audio-routing-check",
                         daemon=True).start()

    def _routing_worker(self, system: str, mic: str) -> None:
        try:
            result = self._routing(system, mic)
        except Exception as exc:  # noqa: BLE001 - shown in the panel
            log.warning("routing check failed", exc_info=True)
            self._routed.emit(None, str(exc))
            return
        self._routed.emit(result, "")

    def _on_routed(self, result: RoutingResult | None, error: str) -> None:
        self.route_btn.setEnabled(True)
        if result is None:
            self.result.setText(_span(f"Could not run the check: {error}", "err"))
            return
        lines: list[str] = []
        if result.output_ok:
            lines.append(_span(f"The speakers reach the output feed ({result.system_device}): "
                               f"{result.system_rise_db:+.0f} dB.", "ok"))
        else:
            lines.append(_span(f"The test sound ({result.output_device}) did not reach {result.system_device} "
                               f"({result.system_rise_db:+.0f} dB): pick the [Loopback] of the device your "
                               "speakers or headset play through.", "err"))
        bleed = result.bleed
        if bleed == "none":
            lines.append(_span(f"The mic does not hear the speakers ({result.mic_rise_db:+.0f} dB).", "ok"))
        elif bleed == "slight":
            lines.append(_span(f"The mic hears the speakers faintly ({result.mic_rise_db:+.0f} dB): a loud "
                               "remote voice can end up in the transcript as Me. Headphones fix it.", "warn"))
        elif bleed == "strong":
            lines.append(_span(f"The mic hears the speakers ({result.mic_rise_db:+.0f} dB): remote voices are "
                               "transcribed as Me too. Use a headset or headphones.", "err"))
        elif result.mic_silent:
            lines.append(_span("The mic delivered digital silence, so whether it hears the speakers could not be "
                               "measured: unmute it (Test devices) and check again.", "err"))
        else:
            lines.append(_span("No microphone to measure.", "warn"))
        if result.isolated:
            lines.append(_span("Clean routing: Me (the mic) and everyone else (the output feed, clustered into "
                               "Speaker 1, 2, ...) are attributed exactly.", "ok"))
        self.result.setText("<br>".join(lines))
        self._recommend_echo(result)

    def _recommend_echo(self, result: RoutingResult) -> None:
        echo_on = self._echo_box.isChecked()
        if result.isolated and echo_on:
            self._echo_target = False
            self.echo_btn.setText("Turn echo suppression off (nothing to echo; it can only drop real lines)")
        elif result.bleed in ("slight", "strong") and not echo_on:
            self._echo_target = True
            self.echo_btn.setText("Turn echo suppression on (drops the speaker sound the mic transcribes)")
        else:
            self._echo_target = None
            self.echo_btn.hide()
            return
        self.echo_btn.setEnabled(True)
        self.echo_btn.show()

    def _apply_echo_recommendation(self) -> None:
        if self._echo_target is None:
            return
        self._echo_box.setChecked(self._echo_target)
        self.echo_btn.setText(f"Echo suppression {'on' if self._echo_target else 'off'}: Apply or OK keeps it")
        self.echo_btn.setEnabled(False)
        self._echo_target = None


def _stop_quietly(capture: Any) -> None:
    try:
        capture.stop()
    except Exception:  # noqa: BLE001
        log.debug("audio check: capture.stop() raised", exc_info=True)
