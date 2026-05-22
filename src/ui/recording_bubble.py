"""Three-state recording bubble — visual port of Whisper Mobile's
WhisperAccessibilityService floating button.

Visual spec (from whisper-mobile WhisperAccessibilityService.kt):
  - Circle background color by state:
      IDLE         -> #D9F2EBF6 (pale lavender, ~85% opacity)  [hidden in our use]
      RECORDING    -> #DDEF4444 (red, ~87% opacity), alpha-breathing 1.0↔0.4
      PAUSED       -> #DD16A34A (darker green, stable)
      TRANSCRIBING -> #DD6B6B6B (grey) + blue (#1E88E5) circular progress ring
  - Microphone glyph centered, dark (#1C1C1E) for contrast on any state color.
  - Pulse animation while RECORDING: window opacity 1.0 → 0.4 → 1.0 over 500ms
    each phase (matches startPulse() / stopPulse() in the Kotlin source).

Position: bottom-center of the primary screen, ~120 px above the bottom edge.
This replaces the old StatusWindow position so the user sees the same anchor
they already learned.

Interactions (per memory spec project_whisper_pc_three_state_bubble.md):
  - Single click  → pauseToggleRequested  (RECORDING ↔ PAUSED)
  - Double click  → endRequested          (end + transcribe)
"""

from __future__ import annotations

import math
import os
import sys

from PyQt5.QtCore import Qt, QTimer, QRect, QRectF, pyqtSignal
from PyQt5.QtGui import (QBrush, QColor, QGuiApplication, QPainter,
                          QPainterPath, QPen, QPixmap)
from PyQt5.QtWidgets import QApplication, QWidget

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


_BUBBLE_DIAMETER = 64       # actual circle diameter, like mobile BTN_DP=52 scaled for PC
_WINDOW_PADDING = 12        # transparent room for shadow + transcribing ring
_BOTTOM_OFFSET = 120        # px above the screen bottom edge (matches StatusWindow)
_DOUBLE_CLICK_MS = 260

# Pulse: mobile's startPulse runs 500ms 1.0→0.4, then 500ms 0.4→1.0, repeat.
_PULSE_TICK_MS = 20                                 # ~50 fps
_PULSE_STEP = (1.0 - 0.4) / (500 / _PULSE_TICK_MS)  # alpha delta per tick

# Transcribing spinner: blue ring rotating around the circle.
_SPINNER_TICK_MS = 16
_SPINNER_DEG_PER_TICK = 4


class RecordingBubble(QWidget):
    """Floating circular indicator with state-based color + mobile-style
    pulse + dark mic glyph in the center."""

    # Mobile color constants (RGBA, alpha encoded in the QColor.alpha()).
    _COLOR_IDLE = QColor(0xF2, 0xEB, 0xF6, 0xD9)          # pale lavender (not visible in our flow)
    _COLOR_RECORDING = QColor(0xEF, 0x44, 0x44, 0xDD)     # red
    _COLOR_PAUSED = QColor(0x16, 0xA3, 0x4A, 0xDD)        # darker green
    _COLOR_TRANSCRIBING = QColor(0x6B, 0x6B, 0x6B, 0xDD)  # grey
    _COLOR_RING = QColor(0x1E, 0x88, 0xE5)                # transcribing spinner
    _COLOR_MIC = QColor(0x1C, 0x1C, 0x1E)                 # dark mic glyph
    _COLOR_SHADOW = QColor(0, 0, 0, 60)                   # drop shadow

    pauseToggleRequested = pyqtSignal()
    endRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        size = _BUBBLE_DIAMETER + _WINDOW_PADDING * 2
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)

        # Pre-load the mic pixmap once, rendered dark like mobile's ic_mic.
        # Inner mic occupies ~55% of the circle diameter to match mobile padding.
        self._mic_pixmap = self._load_mic_pixmap(
            int(_BUBBLE_DIAMETER * 0.55)
        )

        self._state = 'idle'
        self._pulse_alpha = 1.0
        self._pulse_direction_down = True
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(_PULSE_TICK_MS)
        self._pulse_timer.timeout.connect(self._on_pulse_tick)

        self._spinner_angle = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(_SPINNER_TICK_MS)
        self._spinner_timer.timeout.connect(self._on_spinner_tick)

        # Single vs double-click discrimination.
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(_DOUBLE_CLICK_MS)
        self._click_timer.timeout.connect(self._fire_single_click)
        self._pending_single_click = False

        self._noactivate_applied = False
        self._position_bottom_center()

    # ---- State management --------------------------------------------------

    def set_state(self, state: str) -> None:
        """idle / recording / paused / transcribing — drives color + animation."""
        if state == self._state:
            return
        self._state = state

        # Re-position before showing in case the user changed screens.
        if state in ('recording', 'paused', 'transcribing'):
            self._position_bottom_center()

        # Pulse only while RECORDING.
        if state == 'recording':
            self._pulse_alpha = 1.0
            self._pulse_direction_down = True
            self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
            # Reset window opacity if a pulse left it dim.
            self.setWindowOpacity(1.0)

        # Spinner ring only while TRANSCRIBING.
        if state == 'transcribing':
            self._spinner_angle = 0
            self._spinner_timer.start()
        else:
            self._spinner_timer.stop()

        if state == 'idle':
            self.hide()
        else:
            self.show()
            self.raise_()
            self._apply_noactivate_once()
        self.update()

    # ---- Position ---------------------------------------------------------

    def _position_bottom_center(self) -> None:
        # Use full geometry (NOT availableGeometry) to match the legacy
        # StatusWindow anchor exactly — the user is anchored to that position.
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.geometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + geo.height() - self.height() - _BOTTOM_OFFSET
        self.move(x, y)

    def _apply_noactivate_once(self) -> None:
        """WS_EX_NOACTIVATE — clicking the bubble must not steal focus from
        the paste target (mirrors transcript_history_window)."""
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

    # ---- Mic pixmap loading + tinting -------------------------------------

    @staticmethod
    def _load_mic_pixmap(target_size: int) -> QPixmap:
        """Load microphone.png from assets at the requested size. The asset
        already ships with a dark-grey mic on transparent bg, so no tinting
        is needed — just scale it."""
        # Resolve from the project root: src/ui/ -> .. -> .. -> assets/
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [
            os.path.join(root, 'assets', 'microphone.png'),
            # Fallback: when launched from src/, the CWD-relative path works too.
            os.path.join('assets', 'microphone.png'),
        ]
        for path in candidates:
            if os.path.isfile(path):
                px = QPixmap(path)
                if not px.isNull():
                    return px.scaled(
                        target_size, target_size,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation,
                    )
        return QPixmap()

    # ---- Animation ticks --------------------------------------------------

    def _on_pulse_tick(self) -> None:
        # Mirror mobile startPulse: 1.0 → 0.4 over 500ms, 0.4 → 1.0 over 500ms.
        if self._pulse_direction_down:
            self._pulse_alpha -= _PULSE_STEP
            if self._pulse_alpha <= 0.4:
                self._pulse_alpha = 0.4
                self._pulse_direction_down = False
        else:
            self._pulse_alpha += _PULSE_STEP
            if self._pulse_alpha >= 1.0:
                self._pulse_alpha = 1.0
                self._pulse_direction_down = True
        self.setWindowOpacity(self._pulse_alpha)

    def _on_spinner_tick(self) -> None:
        self._spinner_angle = (self._spinner_angle + _SPINNER_DEG_PER_TICK) % 360
        self.update()

    # ---- Paint ------------------------------------------------------------

    def paintEvent(self, event):  # noqa: D401, ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Circle geometry: centered in the widget; padding reserved for shadow + ring.
        cx = self.width() // 2
        cy = self.height() // 2
        r = _BUBBLE_DIAMETER // 2

        # Drop shadow — soft, two-pass.
        for offset, alpha in ((4, 30), (2, 50)):
            shadow = QColor(0, 0, 0, alpha)
            painter.setBrush(shadow)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(
                QRectF(cx - r, cy - r + offset, r * 2, r * 2)
            )

        # Background circle by state.
        color = {
            'idle': self._COLOR_IDLE,
            'recording': self._COLOR_RECORDING,
            'paused': self._COLOR_PAUSED,
            'transcribing': self._COLOR_TRANSCRIBING,
        }.get(self._state, self._COLOR_IDLE)

        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        # Mic glyph centered.
        if not self._mic_pixmap.isNull():
            mw = self._mic_pixmap.width()
            mh = self._mic_pixmap.height()
            painter.drawPixmap(cx - mw // 2, cy - mh // 2, self._mic_pixmap)
        else:
            # Fallback if the asset failed to load: draw a simple mic shape.
            painter.setBrush(self._COLOR_MIC)
            painter.setPen(Qt.NoPen)
            body_w = int(r * 0.5)
            body_h = int(r * 0.85)
            painter.drawRoundedRect(
                cx - body_w // 2, cy - body_h // 2,
                body_w, body_h, body_w // 2, body_w // 2,
            )

        # Transcribing spinner ring — saturated blue, rotating arc.
        if self._state == 'transcribing':
            ring_r = r + 4
            pen = QPen(self._COLOR_RING, 3)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            arc_rect = QRectF(cx - ring_r, cy - ring_r, ring_r * 2, ring_r * 2)
            # Qt arc angles are 1/16 of a degree.
            start = (90 - self._spinner_angle) * 16
            painter.drawArc(arc_rect, start, -90 * 16)

    # ---- Click handling: single vs double ---------------------------------

    def mousePressEvent(self, event):  # noqa: D401
        if event.button() != Qt.LeftButton:
            return
        if self._click_timer.isActive():
            self._click_timer.stop()
            self._pending_single_click = False
            self.endRequested.emit()
        else:
            self._pending_single_click = True
            self._click_timer.start()
        event.accept()

    def _fire_single_click(self) -> None:
        if self._pending_single_click:
            self._pending_single_click = False
            self.pauseToggleRequested.emit()
