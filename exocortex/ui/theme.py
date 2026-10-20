"""Visual theme for the overlay: a dark, translucent "glass" panel that stays legible over any meeting app.

Exposes:
- COLORS: named colours used by widgets (hex strings; ALPHA variants are rgba()).
- build_qss(font_size): the complete Qt style sheet for the overlay.
- speaker_color(speaker_id): stable, distinct colour per speaker ("me" gets the accent colour).
"""
from __future__ import annotations

COLORS: dict[str, str] = {
    "bg": "rgba(18, 20, 26, 235)",          # panel background (alpha applied via window opacity too)
    "bg_solid": "#12141a",
    "panel": "rgba(255, 255, 255, 12)",      # card surfaces
    "panel_hover": "rgba(255, 255, 255, 22)",
    "border": "rgba(255, 255, 255, 28)",
    "text": "#e8eaf0",
    "text_dim": "#9aa0ad",
    "text_faint": "#6b7180",
    "accent": "#7dd3fc",                      # sky-300: "Me", primary buttons, active tab
    "accent_dim": "rgba(125, 211, 252, 40)",
    "ok": "#86efac",                          # green-300
    "warn": "#fcd34d",                        # amber-300
    "err": "#fca5a5",                         # red-300
    "question": "rgba(252, 211, 77, 26)",     # highlight for detected questions
    "answer_bg": "rgba(125, 211, 252, 14)",
}

# Distinct, readable hues for remote speakers (avoid the accent hue used by "Me").
_SPEAKER_PALETTE: list[str] = [
    "#f9a8d4",  # pink-300
    "#c4b5fd",  # violet-300
    "#fdba74",  # orange-300
    "#6ee7b7",  # emerald-300
    "#fde68a",  # yellow-200
    "#a5b4fc",  # indigo-300
    "#fca5a5",  # red-300
    "#99f6e4",  # teal-200
]


def speaker_color(speaker_id: str) -> str:
    """Stable colour for a speaker id ("me" → accent; "spk3" → palette[2]; unknown → hash)."""
    if speaker_id == "me":
        return COLORS["accent"]
    if speaker_id.startswith("spk") and speaker_id[3:].isdigit():
        return _SPEAKER_PALETTE[(int(speaker_id[3:]) - 1) % len(_SPEAKER_PALETTE)]
    return _SPEAKER_PALETTE[sum(map(ord, speaker_id)) % len(_SPEAKER_PALETTE)]


def build_qss(font_size: int = 11) -> str:
    """Complete style sheet for the overlay window and its widgets (object names are documented)."""
    fs = int(font_size)
    small = max(8, fs - 2)
    return f"""
    QWidget {{
        color: {COLORS['text']};
        font-family: "Segoe UI Variable", "Segoe UI", "Inter", sans-serif;
        font-size: {fs}pt;
        background: transparent;
    }}
    QFrame#root {{
        background: {COLORS['bg']};
        border: 1px solid {COLORS['border']};
        border-radius: 14px;
    }}
    QFrame#titlebar {{
        background: transparent;
        border-bottom: 1px solid {COLORS['border']};
    }}
    QLabel#title {{
        font-weight: 600;
        letter-spacing: 0.5px;
        color: {COLORS['text']};
    }}
    QLabel#subtitle, QLabel.dim {{
        color: {COLORS['text_dim']};
        font-size: {small}pt;
    }}
    QPushButton {{
        background: {COLORS['panel']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 4px 10px;
        color: {COLORS['text']};
    }}
    QPushButton:hover {{ background: {COLORS['panel_hover']}; }}
    QPushButton:pressed {{ background: {COLORS['accent_dim']}; }}
    QPushButton:disabled {{ color: {COLORS['text_faint']}; }}
    QPushButton#primary {{
        background: {COLORS['accent_dim']};
        border: 1px solid {COLORS['accent']};
        color: {COLORS['accent']};
        font-weight: 600;
    }}
    QPushButton#icon {{
        border: none;
        background: transparent;
        padding: 2px 6px;
        font-size: {fs + 1}pt;
        color: {COLORS['text_dim']};
    }}
    QPushButton#icon:hover {{ color: {COLORS['text']}; background: {COLORS['panel']}; }}
    QPushButton#icon:checked {{ color: {COLORS['accent']}; }}
    QPushButton#tab {{
        border: none;
        background: transparent;
        color: {COLORS['text_dim']};
        padding: 6px 12px;
        border-radius: 0px;
        border-bottom: 2px solid transparent;
    }}
    QPushButton#tab:checked {{
        color: {COLORS['accent']};
        border-bottom: 2px solid {COLORS['accent']};
    }}
    QPushButton#tab:hover {{ color: {COLORS['text']}; }}
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background: {COLORS['panel']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 6px 8px;
        selection-background-color: {COLORS['accent_dim']};
    }}
    QLineEdit:focus, QTextEdit:focus {{ border: 1px solid {COLORS['accent']}; }}
    QTextBrowser {{
        background: transparent;
        border: none;
    }}
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: rgba(255,255,255,50);
        border-radius: 4px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{ background: rgba(255,255,255,90); }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: rgba(255,255,255,40);
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        width: 12px;
        margin: -5px 0;
        background: {COLORS['accent']};
        border-radius: 6px;
    }}
    QProgressBar {{
        background: rgba(255,255,255,20);
        border: none;
        border-radius: 2px;
        height: 4px;
    }}
    QProgressBar::chunk {{ background: {COLORS['ok']}; border-radius: 2px; }}
    QFrame#statusbar {{
        border-top: 1px solid {COLORS['border']};
        background: transparent;
    }}
    QLabel#status {{
        color: {COLORS['text_dim']};
        font-size: {small}pt;
    }}
    QLabel#toast {{
        background: rgba(0,0,0,170);
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 6px 10px;
        color: {COLORS['text']};
    }}
    QToolTip {{
        background: {COLORS['bg_solid']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['border']};
        padding: 4px;
    }}
    QMenu {{
        background: {COLORS['bg_solid']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 4px;
    }}
    QMenu::item {{ padding: 5px 14px; border-radius: 6px; }}
    QMenu::item:selected {{ background: {COLORS['panel_hover']}; }}
    """
