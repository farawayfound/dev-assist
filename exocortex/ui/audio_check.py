"""The Check box under Settings > Audio: the Windows mute of a silent mic, and the speaker routing.

Speakers are attributed by source -- the mic feed is "Me", the output feed is clustered into
Speaker 1, 2, ... -- so the transcript is only as right as the routing. exo-cortex's Audio page
(:class:`exocortex.ui.audio_panel.AudioPanel`, 2026-10-29) picks each device and Tests it; this
module adds what that page does not check, as :class:`CheckedAudioPanel` -- the page the Settings
dialog builds:

* **The Windows mute.** A WASAPI capture endpoint that Windows has muted (or turned to 0 %) opens
  without an error and delivers digital silence. After every microphone Test the page asks Windows
  (:mod:`exocortex.audio.endpoint`) and, when that is the reason, says so and offers **Unmute in
  Windows**.
* **Check speaker routing** plays a short chirp on the output behind the chosen loopback and
  measures where it arrives (:func:`exocortex.audio.check.routing_check`): it should reach the
  output feed and not the mic. The verdict says whether echo suppression is still worth having.

The routing check runs on a daemon thread and comes back through a queued Qt signal; the endpoint
calls are a few milliseconds on the Qt thread. Every device / Windows call is injectable for tests.
"""
from __future__ import annotations

import html
import logging
import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from exocortex.audio.check import RoutingResult
from exocortex.ui.audio_panel import AudioPanel
from exocortex.ui.theme import COLORS

log = logging.getLogger(__name__)

#: Windows input volume under this is reported (it reads as near-silence, not as a fault).
_LOW_VOLUME = 0.05


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
    """The Windows-mute row and the routing check for the devices ``mic_combo`` / ``system_combo``
    name (their ``spec()`` when they have one, else their text); ``echo_box`` is the Audio tab's
    echo-suppression checkbox, which the routing verdict may offer to flip (the form still applies
    it on Apply / OK, like any other edit)."""

    _routed = Signal(object, str)            # RoutingResult | None, error

    def __init__(
        self,
        mic_combo: QComboBox,
        system_combo: QComboBox,
        echo_box: QCheckBox,
        parent: QWidget | None = None,
        *,
        routing: Callable[[str, str], RoutingResult] | None = None,
        endpoint_state: Callable[[str], Any] | None = None,
        set_endpoint_muted: Callable[[str, bool], bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self._combos = {"mic": mic_combo, "system": system_combo}
        self._echo_box = echo_box
        self._routing = routing or _default_routing
        self._endpoint_state = endpoint_state or _default_endpoint_state
        self._set_endpoint_muted = set_endpoint_muted or _default_set_endpoint_muted
        self._mic_name = ""
        self._echo_target: bool | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        intro = QLabel("The mic is transcribed as Me, the output feed as everyone else: the routing check "
                       "shows whether each feed hears what it should.", self)
        intro.setObjectName("hint")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        win_row = QHBoxLayout()
        self.windows_label = QLabel("", self)
        self.windows_label.setWordWrap(True)
        self.windows_label.setStyleSheet(f"color: {COLORS['err']};")
        self.unmute_btn = QPushButton("Unmute in Windows", self)
        self.unmute_btn.clicked.connect(lambda _=False: self._unmute())
        win_row.addWidget(self.windows_label, 1)
        win_row.addWidget(self.unmute_btn)
        outer.addLayout(win_row)
        self.windows_label.hide()
        self.unmute_btn.hide()

        buttons = QHBoxLayout()
        self.route_btn = QPushButton("Check speaker routing", self)
        self.route_btn.setToolTip("Plays a short chirp (~1 s) on the output, then measures whether it reached "
                                  "the output feed and whether the mic heard it. Stay quiet while it runs.")
        self.route_btn.clicked.connect(lambda _=False: self.check_routing())
        buttons.addWidget(self.route_btn)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        self.result = QLabel("", self)
        self.result.setWordWrap(True)
        self.result.setTextFormat(Qt.TextFormat.RichText)
        outer.addWidget(self.result)
        self.echo_btn = QPushButton("", self)
        self.echo_btn.clicked.connect(lambda _=False: self._apply_echo_recommendation())
        self.echo_btn.hide()
        echo_row = QHBoxLayout()
        echo_row.addWidget(self.echo_btn)
        echo_row.addStretch(1)
        outer.addLayout(echo_row)

        self._routed.connect(self._on_routed)

    def _spec(self, kind: str) -> str:
        combo = self._combos[kind]
        spec = combo.spec() if callable(getattr(combo, "spec", None)) else combo.currentText()
        return str(spec or "").strip() or "default"

    # ---- the Windows mute --------------------------------------------------------------------
    def refresh_endpoint(self, name: str = "") -> None:
        """Show the mic's Windows mute / input volume when either explains a silent mic. `name` is
        the device a Test opened ("" = the picker's spec)."""
        self._mic_name = name or self._spec("mic")
        try:
            state = self._endpoint_state(self._mic_name)
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
        name = self._mic_name or self._spec("mic")
        if not self._set_endpoint_muted(name, False):
            self.windows_label.setText("Could not unmute it: use Settings > System > Sound > Input in Windows.")
            return
        self.refresh_endpoint(name)

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
            result, error = self._routing(system, mic), ""
        except Exception as exc:  # noqa: BLE001 - shown in the panel
            log.warning("routing check failed", exc_info=True)
            result, error = None, str(exc)
        try:
            self._routed.emit(result, error)
        except RuntimeError:     # the dialog closed while the chirp played: nobody is listening
            log.debug("routing check finished after its panel was gone")

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
                               "measured: Test the microphone above, unmute it, and check again.", "err"))
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


class CheckedAudioPanel(AudioPanel):
    """The Settings > Audio page: exo-cortex's :class:`~exocortex.ui.audio_panel.AudioPanel` (device
    pickers, a Test per source, voice detection, what is open now) with a **Check** box above its
    status line -- :class:`AudioCheckPanel` -- and, after every microphone Test, the Windows mute of
    the device it opened. Without the pickers or the echo box (a trimmed form) the box is left out
    and the page is the plain AudioPanel."""

    check: AudioCheckPanel | None = None

    def _place_form_widgets(self) -> None:
        super()._place_form_widgets()
        widgets = self._form.widgets
        mic, system = widgets.get("audio.mic_device"), widgets.get("audio.system_device")
        echo = widgets.get("audio.echo_suppression")
        if mic is None or system is None or not isinstance(echo, QCheckBox):
            return
        try:
            body = self.scroll.widget()
            box = QGroupBox("Check", body)
            QVBoxLayout(box).addWidget(AudioCheckPanel(mic, system, echo, box))
            lay = body.layout()
            lay.insertWidget(max(0, lay.count() - 2), box)    # above Refresh devices + the status line
            self.check = box.findChild(AudioCheckPanel)
            self.check_box = box
        except Exception:  # noqa: BLE001 - the checks are an extra: the page works without them
            log.exception("audio Check box unavailable")
            self.check = None

    def _finish_probe(self, job: Any) -> None:
        kind = self._probe_kind
        super()._finish_probe(job)
        result = job.result
        if kind != "mic" or self.check is None or job.error or result is None:
            return
        device = getattr(result, "device", None)
        self.check.refresh_endpoint(str(getattr(device, "name", "") or ""))
