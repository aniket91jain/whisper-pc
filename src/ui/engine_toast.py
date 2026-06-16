"""Tiny transient 'transcribed by X' hint, shown just above the recording
bubble's anchor at paste time.

Deliberately minimal: a small rounded pill with one short engine name, in the
same colour the Transcript-History badge uses for that engine, that appears for
a moment and fades out. It is a hint, not a notification — no interaction, never
steals focus, and disappears on its own.

The label/colour mapping is shared with the history-window badge via
``_engine_badge`` so the toast and the history list always speak the same
vocabulary (ElevenLabs / Groq + Polish / Gemini / On-device).
"""

from __future__ import annotations

import os
import sys

from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QRectF
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QPainter
from PyQt5.QtWidgets import QWidget

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui.transcript_history_window import _engine_badge

# How long the pill stays fully visible before it starts fading, and how long
# the fade itself takes (ms). Short — this is a glance, not a read.
_VISIBLE_MS = 1400
_FADE_MS = 450

# Vertical gap (px) between the top of the recording bubble and the bottom of
# the toast pill, so the hint floats just above the mic circle.
_GAP_ABOVE_BUBBLE = 10

_PAD_H = 12
_PAD_V = 5


class EngineToast(QWidget):
    """A self-dismissing pill that names the STT engine of the last dictation.

    Construct once at startup (cheap, hidden). Call ``show_engine(label)`` from
    the GUI thread when a transcription lands; everything else is automatic."""

    def __init__(self, bubble_diameter: int, bottom_offset: int,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Frameless, click-through-ish (never activates), always on top — same
        # window recipe as the recording bubble so it layers correctly.
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        # Bubble anchor geometry — used to position the pill just above it.
        self._bubble_diameter = bubble_diameter
        self._bottom_offset = bottom_offset

        self._font = QFont('Segoe UI', 9, QFont.Bold)
        self._fm = QFontMetrics(self._font)

        self._label = ''
        self._fg = QColor('#404040')
        self._bg = QColor('#ffffff')

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._begin_fade)

        self._fade = QPropertyAnimation(self, b'windowOpacity', self)
        self._fade.setDuration(_FADE_MS)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self.hide)

        self._noactivate_applied = False

    # ---- Public API -------------------------------------------------------

    def show_engine(self, engine: str) -> None:
        """Show the pill for the given raw engine label (e.g. 'elevenlabs-stream').
        No-op when the engine maps to nothing recognisable."""
        badge = _engine_badge(engine or '')
        if badge is None:
            return
        label, fg, bg = badge
        self._label = label
        self._fg = fg
        # Solid, slightly translucent dark-tinted pill so light text reads on
        # any desktop. Use the engine colour as a saturated fill.
        self._bg = QColor(fg)
        self._bg.setAlpha(242)
        self._fg = QColor('#ffffff')

        text_w = self._fm.horizontalAdvance(label)
        text_h = self._fm.height()
        self.setFixedSize(text_w + 2 * _PAD_H, text_h + 2 * _PAD_V)
        self._position_above_bubble()

        # Restart the visible-then-fade cycle from full opacity.
        self._fade.stop()
        self.setWindowOpacity(1.0)
        self.show()
        self.raise_()
        self._apply_noactivate_once()
        self.update()
        self._hide_timer.start(_VISIBLE_MS)

    # ---- Internals --------------------------------------------------------

    def _begin_fade(self) -> None:
        self._fade.stop()
        self._fade.start()

    def _position_above_bubble(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.geometry()
        # Recording bubble window height = diameter + 2*padding (12); its top
        # edge sits bottom_offset above the screen bottom. We don't import the
        # bubble's padding constant — _GAP_ABOVE_BUBBLE absorbs the small slack.
        bubble_window_h = self._bubble_diameter + 24
        bubble_top = geo.y() + geo.height() - bubble_window_h - self._bottom_offset
        x = geo.x() + (geo.width() - self.width()) // 2
        y = bubble_top - self.height() - _GAP_ABOVE_BUBBLE
        self.move(x, y)

    def _apply_noactivate_once(self) -> None:
        if self._noactivate_applied:
            return
        try:
            import ctypes
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            hwnd = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE
            )
            self._noactivate_applied = True
        except Exception:
            pass

    def paintEvent(self, event):  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(0, 0, self.width(), self.height())
        radius = self.height() / 2
        painter.setBrush(self._bg)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, radius, radius)
        painter.setFont(self._font)
        painter.setPen(self._fg)
        painter.drawText(rect, Qt.AlignCenter, self._label)
