"""Start the real overlay for a few seconds, render each tab to PNG, then quit cleanly.

    .venv\\Scripts\\python.exe scripts\\ui_snapshot.py [out_dir] [--seconds 12]

Useful to eyeball the UI without a meeting. The PNGs are widget renders (QWidget.grab), so they
work even though the window is excluded from screen capture. Also prints the capture-affinity
state and which hotkeys failed to register.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default=str(ROOT / "docs" / "screenshots"))
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--config")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    from exocortex.app import Engine
    from exocortex.bus import EventBus
    from exocortex.config import load_config
    from exocortex.logging_setup import setup_logging
    from exocortex.ui import win32
    from exocortex.ui.overlay import OverlayWindow

    setup_logging("INFO")
    cfg = load_config(args.config)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    bus = EventBus()
    engine = Engine(cfg, bus, on_quit=app.quit)
    overlay = OverlayWindow(cfg, bus, engine)
    engine.start()
    overlay.show()

    tabs = ["live", "assist", "notes", "screen", "handsfree"]
    state = {"i": 0}

    def report() -> None:
        hwnd = int(overlay.winId())
        print("display affinity:", hex(win32.get_display_affinity(hwnd) or 0), "->", win32.describe_affinity(hwnd), flush=True)
        hk = getattr(overlay, "_hotkeys", None)
        failed = getattr(hk, "failed", None)
        print("hotkeys failed:", failed, flush=True)

    def snap() -> None:
        name = tabs[state["i"]]
        try:
            if name == "handsfree":
                # The hands-free VIEW only (the engine's controller is not started): the output
                # pane with the title bar and the switch, as the user sees it while it runs.
                overlay._set_hands_free(True, notify_engine=False)
            else:
                overlay._show_tab(state["i"])
        except Exception as e:  # noqa: BLE001
            print("tab switch failed:", e, flush=True)
        app.processEvents()
        overlay.toast.hide()   # a mode-change toast would cover the title bar in the capture
        pix = overlay.grab()
        path = out / f"overlay_{name}.png"
        pix.save(str(path))
        print("saved", path, pix.width(), "x", pix.height(), flush=True)
        if name == "handsfree":
            overlay._set_hands_free(False, notify_engine=False)
        state["i"] += 1
        if state["i"] < len(tabs):
            QTimer.singleShot(400, snap)
        else:
            QTimer.singleShot(300, finish)

    def finish() -> None:
        overlay.close_silently()
        engine.stop()
        app.quit()

    QTimer.singleShot(int(args.seconds * 1000) - 2500, report)
    QTimer.singleShot(int(args.seconds * 1000) - 2000, snap)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
