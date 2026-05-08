import sys
import os
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QHBoxLayout, QPushButton

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui.base_window import BaseWindow

# The icon-only window is sized for a 48px pixmap; the no-speech message needs
# more horizontal room for the "Nothing transcribable detected" text.
_ICON_SIZE = (80, 110)
_MESSAGE_SIZE = (300, 70)
_NO_SPEECH_DISMISS_MS = 2500


class StatusWindow(BaseWindow):
    statusSignal = pyqtSignal(str)
    closeSignal = pyqtSignal()

    def __init__(self):
        """
        Initialize the status window.
        """
        super().__init__('WhisperWriter Status', _ICON_SIZE[0], _ICON_SIZE[1],
                         show_title_bar=False, background_alpha=180)
        self.initStatusUI()
        self.statusSignal.connect(self.updateStatus)
        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self.close)

    def initStatusUI(self):
        """
        Initialize the status user interface.
        """
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)

        # Top row: tiny close button right-aligned. Closing also stops the
        # current recording (closeSignal is wired to stop_result_thread).
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.addStretch(1)
        close_button = QPushButton('×')
        close_button.setFixedSize(18, 18)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #404040;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #000000;
            }
        """)
        close_button.clicked.connect(self.handleCloseButton)
        top_row.addWidget(close_button)
        self.main_layout.addLayout(top_row)

        # Mic / pencil icon, generously padded above and below.
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(48, 48)
        microphone_path = os.path.join('assets', 'microphone.png')
        pencil_path = os.path.join('assets', 'pencil.png')
        self.microphone_pixmap = QPixmap(microphone_path).scaled(
            48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.pencil_pixmap = QPixmap(pencil_path).scaled(
            48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.icon_label.setPixmap(self.microphone_pixmap)
        self.icon_label.setAlignment(Qt.AlignCenter)

        icon_row = QHBoxLayout()
        icon_row.setContentsMargins(0, 0, 0, 0)
        icon_row.addStretch(1)
        icon_row.addWidget(self.icon_label)
        icon_row.addStretch(1)

        # Message label, shown in place of the icon for transient feedback like
        # "Nothing transcribable detected". Hidden during normal recording/transcribing.
        self.message_label = QLabel()
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setFont(QFont('Segoe UI', 10))
        self.message_label.setStyleSheet('color: #404040;')
        self.message_label.setWordWrap(True)
        self.message_label.hide()

        self.main_layout.addStretch(1)
        self.main_layout.addLayout(icon_row)
        self.main_layout.addWidget(self.message_label)
        self.main_layout.addStretch(1)

    def show(self):
        """
        Position the window in the bottom center of the screen and show it.
        """
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        window_width = self.width()
        window_height = self.height()

        x = (screen_width - window_width) // 2
        y = screen_height - window_height - 120

        self.move(x, y)
        super().show()

    def _show_icon_mode(self):
        """Hide the message label, show the icon, and restore icon-mode size."""
        self.message_label.hide()
        self.icon_label.show()
        if (self.width(), self.height()) != _ICON_SIZE:
            self.setFixedSize(*_ICON_SIZE)

    def _show_message_mode(self, text):
        """Swap the icon for a message and resize the window to fit it."""
        self.icon_label.hide()
        self.message_label.setText(text)
        self.message_label.show()
        if (self.width(), self.height()) != _MESSAGE_SIZE:
            self.setFixedSize(*_MESSAGE_SIZE)

    def closeEvent(self, event):
        """
        Emit the close signal when the window is closed.
        """
        self.closeSignal.emit()
        super().closeEvent(event)

    @pyqtSlot(str)
    def updateStatus(self, status):
        """
        Update the status window based on the given status.
        """
        if status == 'recording':
            self._dismiss_timer.stop()
            self._show_icon_mode()
            self.icon_label.setPixmap(self.microphone_pixmap)
            self.show()
        elif status == 'transcribing':
            self._dismiss_timer.stop()
            self._show_icon_mode()
            self.icon_label.setPixmap(self.pencil_pixmap)
        elif status == 'no_speech':
            self._show_message_mode('Nothing transcribable detected')
            self.show()
            self._dismiss_timer.start(_NO_SPEECH_DISMISS_MS)
            return

        if status in ('idle', 'error', 'cancel'):
            self._dismiss_timer.stop()
            self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    status_window = StatusWindow()
    status_window.show()

    # Simulate status updates
    QTimer.singleShot(3000, lambda: status_window.statusSignal.emit('transcribing'))
    QTimer.singleShot(6000, lambda: status_window.statusSignal.emit('idle'))

    sys.exit(app.exec_())
