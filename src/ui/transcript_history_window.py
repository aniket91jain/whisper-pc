import os
import sys
import numpy as np
import soundfile as sf
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QScrollArea, QPushButton, QFrame, QApplication,
                              QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QCursor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui.base_window import BaseWindow
from transcription import transcribe, TranscriptionAPIError
from utils import ConfigManager


def _parse_log(log_path):
    """Read transcript_log.txt; return list of dicts with kind='ok'."""
    if not os.path.isfile(log_path):
        return []
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    entries = []
    for block in content.strip().split('\n\n'):
        lines = block.strip().split('\n')
        timestamp = polished = ''
        for line in lines:
            s = line.strip()
            if s.startswith('[') and s.endswith(']'):
                timestamp = s[1:-1]
            elif s.startswith('POLISHED:'):
                polished = s[9:].strip()
        if polished:
            entries.append({'kind': 'ok', 'timestamp': timestamp, 'text': polished})
    return entries


def _parse_failed_log(log_path):
    """Read failed_log.txt; return list of dicts with kind='failed'."""
    if not os.path.isfile(log_path):
        return []
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    entries = []
    for block in content.strip().split('\n\n'):
        lines = block.strip().split('\n')
        timestamp = audio_rel = error = ''
        for line in lines:
            s = line.strip()
            if s.startswith('[') and s.endswith(']'):
                timestamp = s[1:-1]
            elif s.startswith('AUDIO:'):
                audio_rel = s[6:].strip()
            elif s.startswith('ERROR:'):
                error = s[6:].strip()
        if audio_rel:
            entries.append({'kind': 'failed', 'timestamp': timestamp,
                            'audio_rel': audio_rel, 'error': error})
    return entries


def _remove_failed_entry(log_path, audio_rel):
    """Rewrite failed_log.txt without the entry whose AUDIO line matches audio_rel."""
    if not os.path.isfile(log_path):
        return
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    kept_blocks = []
    for block in content.strip().split('\n\n'):
        if not block.strip():
            continue
        # Skip blocks pointing at the audio we just succeeded on
        if any(line.strip() == f'AUDIO:    {audio_rel}' or
               line.strip() == f'AUDIO: {audio_rel}'
               for line in block.split('\n')):
            continue
        kept_blocks.append(block.strip())
    with open(log_path, 'w', encoding='utf-8') as f:
        if kept_blocks:
            f.write('\n\n'.join(kept_blocks) + '\n\n')


def _simulate_paste():
    """Simulate Ctrl+V in whichever window currently has keyboard focus."""
    try:
        from pynput.keyboard import Key, Controller as _KbController
        kb = _KbController()
        kb.press(Key.ctrl)
        kb.press('v')
        kb.release('v')
        kb.release(Key.ctrl)
    except Exception:
        pass


class TranscriptCard(QFrame):
    def __init__(self, timestamp, polished, parent=None):
        super().__init__(parent)
        self._polished = polished
        self.setObjectName('TranscriptCard')
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._set_style(hovered=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(3)

        ts_label = QLabel(timestamp)
        ts_label.setFont(QFont('Segoe UI', 8))
        ts_label.setStyleSheet('color: #999; background: transparent;')
        layout.addWidget(ts_label)

        self._text_label = QLabel(polished)
        self._text_label.setFont(QFont('Segoe UI', 10))
        self._text_label.setWordWrap(True)
        self._text_label.setStyleSheet('color: #2c2c2c; background: transparent;')
        layout.addWidget(self._text_label)

        self._status_label = QLabel()
        self._status_label.setFont(QFont('Segoe UI', 8))
        self._status_label.setStyleSheet('color: #3a863a; background: transparent;')
        self._status_label.hide()
        layout.addWidget(self._status_label)

    def _set_style(self, hovered):
        bg, border = ('#eaf4ea', '#5aac5a') if hovered else ('#f7f7f7', '#e0e0e0')
        self.setStyleSheet(f'''
            QFrame#TranscriptCard {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}
        ''')

    def enterEvent(self, event):
        self._set_style(hovered=True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._set_style(hovered=False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            QApplication.clipboard().setText(self._polished)
            # Paste into whichever app still has focus (50ms lets clipboard settle)
            QTimer.singleShot(50, _simulate_paste)
            self._status_label.setText('✓  Pasted at cursor')
            self._status_label.show()
            QTimer.singleShot(2000, self._status_label.hide)
        event.accept()  # don't bubble up to BaseWindow drag handler


class FailedTranscriptCard(QFrame):
    """Red-tinted card for an API-failed recording. Includes a Retry button."""

    retryRequested = pyqtSignal(str, object)  # audio_abs_path, self

    def __init__(self, timestamp, audio_abs_path, audio_rel, error_text, parent=None):
        super().__init__(parent)
        self.audio_abs_path = audio_abs_path
        self.audio_rel = audio_rel
        self.setObjectName('FailedTranscriptCard')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._set_style(hovered=False)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(8)

        text_col = QVBoxLayout()
        text_col.setSpacing(3)

        ts_label = QLabel(timestamp)
        ts_label.setFont(QFont('Segoe UI', 8))
        ts_label.setStyleSheet('color: #999; background: transparent;')
        text_col.addWidget(ts_label)

        body_text = '⚠  ' + (error_text or 'Transcription failed')
        if not os.path.isfile(audio_abs_path):
            body_text += '  (audio missing)'
        self._body = QLabel(body_text)
        self._body.setFont(QFont('Segoe UI', 10))
        self._body.setWordWrap(True)
        self._body.setStyleSheet('color: #8a2828; background: transparent;')
        text_col.addWidget(self._body)

        self._sub = QLabel('')
        self._sub.setFont(QFont('Segoe UI', 8))
        self._sub.setStyleSheet('color: #b04040; background: transparent;')
        self._sub.hide()
        text_col.addWidget(self._sub)

        outer.addLayout(text_col, stretch=1)

        self._retry_btn = QPushButton('↻  Retry')
        self._retry_btn.setFont(QFont('Segoe UI', 9))
        self._retry_btn.setFixedHeight(28)
        self._retry_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._retry_btn.setStyleSheet('''
            QPushButton {
                background: #fff;
                border: 1px solid #d77;
                border-radius: 4px;
                padding: 0 12px;
                color: #8a2828;
            }
            QPushButton:hover { background: #ffeaea; }
            QPushButton:disabled { background: #f5f5f5; color: #999; border-color: #ccc; }
        ''')
        self._retry_btn.setEnabled(os.path.isfile(audio_abs_path))
        self._retry_btn.clicked.connect(self._on_retry_clicked)
        outer.addWidget(self._retry_btn, alignment=Qt.AlignVCenter)

    def _set_style(self, hovered):
        bg, border = ('#fbd9d9', '#d77') if hovered else ('#fdecec', '#e8b8b8')
        self.setStyleSheet(f'''
            QFrame#FailedTranscriptCard {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}
        ''')

    def enterEvent(self, event):
        self._set_style(hovered=True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._set_style(hovered=False)
        super().leaveEvent(event)

    def _on_retry_clicked(self):
        self.set_retrying(True)
        self.retryRequested.emit(self.audio_abs_path, self)

    def set_retrying(self, retrying):
        if retrying:
            self._retry_btn.setEnabled(False)
            self._retry_btn.setText('…  Retrying')
            self._sub.hide()
        else:
            self._retry_btn.setEnabled(True)
            self._retry_btn.setText('↻  Retry')

    def show_retry_error(self, reason):
        self.set_retrying(False)
        self._sub.setText(f'Retry failed: {reason}')
        self._sub.show()


class RetryWorker(QThread):
    """Re-runs transcription on a saved WAV file in a background thread."""

    successSignal = pyqtSignal(str, str)  # audio_abs_path, polished_text
    errorSignal = pyqtSignal(str, str)    # audio_abs_path, reason

    def __init__(self, audio_abs_path, local_model):
        super().__init__()
        self.audio_abs_path = audio_abs_path
        self.local_model = local_model

    def run(self):
        try:
            data, _ = sf.read(self.audio_abs_path, dtype='int16')
            if data.ndim > 1:
                data = data[:, 0]
            audio = np.ascontiguousarray(data, dtype=np.int16)
            result = transcribe(audio, self.local_model)
            self.successSignal.emit(self.audio_abs_path, result or '')
        except TranscriptionAPIError as e:
            self.errorSignal.emit(self.audio_abs_path, e.reason)
        except Exception as e:
            self.errorSignal.emit(self.audio_abs_path, f'{type(e).__name__}: {e}')


class TranscriptHistoryWindow(BaseWindow):
    def __init__(self, log_path, failed_log_path=None, local_model=None, input_simulator=None):
        super().__init__('Transcript History', 540, 680)
        self._log_path = log_path
        self._failed_log_path = failed_log_path
        # Project root = directory containing transcript_log.txt; failed/<wav>
        # lives under there too (set in result_thread._persist_failed_recording).
        self._project_root = os.path.dirname(os.path.abspath(log_path))
        self._local_model = local_model
        self._input_simulator = input_simulator
        self._retry_workers = []  # keep QThread refs alive while running

        # Float above all other windows; Tool keeps it off the taskbar.
        # WA_ShowWithoutActivating prevents stealing focus when show() is called.
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        self._init_content()
        self._load()

    def showEvent(self, event):
        super().showEvent(event)
        # WS_EX_NOACTIVATE: window receives mouse events but never becomes the
        # active (keyboard-focus) window when clicked — so clicks paste into the
        # previously focused app via _simulate_paste().
        try:
            import ctypes
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            hwnd = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE)
        except Exception:
            pass  # Non-Windows or ctypes unavailable; window still stays on top

    def _init_content(self):
        header_row = QHBoxLayout()
        hint = QLabel('Click any entry to paste at cursor')
        hint.setFont(QFont('Segoe UI', 9))
        hint.setStyleSheet('color: #666;')
        header_row.addWidget(hint)
        header_row.addStretch()

        refresh_btn = QPushButton('↻  Refresh')
        refresh_btn.setFont(QFont('Segoe UI', 9))
        refresh_btn.setFixedHeight(28)
        refresh_btn.setCursor(QCursor(Qt.PointingHandCursor))
        refresh_btn.setStyleSheet('''
            QPushButton {
                background: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 0 10px;
                color: #404040;
            }
            QPushButton:hover { background: #e0e0e0; }
        ''')
        refresh_btn.clicked.connect(self._load)
        header_row.addWidget(refresh_btn)
        self.main_layout.addLayout(header_row)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet('QScrollArea { border: none; background: transparent; }')

        self._container = QWidget()
        self._container.setStyleSheet('background: transparent;')
        self._cards_layout = QVBoxLayout(self._container)
        self._cards_layout.setContentsMargins(0, 0, 6, 0)
        self._cards_layout.setSpacing(8)
        self._cards_layout.addStretch()

        self._scroll.setWidget(self._container)
        self.main_layout.addWidget(self._scroll)

    def _load(self):
        while self._cards_layout.count() > 1:
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        ok_entries = _parse_log(self._log_path)
        failed_entries = _parse_failed_log(self._failed_log_path) if self._failed_log_path else []
        entries = ok_entries + failed_entries
        # Newest first by timestamp string (ISO-like format sorts correctly)
        entries.sort(key=lambda e: e.get('timestamp', ''), reverse=True)

        if not entries:
            label = QLabel('No transcriptions yet.\nDictate something and come back.')
            label.setFont(QFont('Segoe UI', 10))
            label.setStyleSheet('color: #aaa;')
            label.setAlignment(Qt.AlignCenter)
            self._cards_layout.insertWidget(0, label)
            return

        for i, entry in enumerate(entries):
            if entry['kind'] == 'ok':
                card = TranscriptCard(entry['timestamp'], entry['text'], self._container)
            else:
                audio_abs = os.path.join(self._project_root, entry['audio_rel'].replace('/', os.sep))
                card = FailedTranscriptCard(
                    entry['timestamp'], audio_abs, entry['audio_rel'],
                    entry['error'], self._container,
                )
                card.retryRequested.connect(self._on_retry_requested)
            self._cards_layout.insertWidget(i, card)

    def _on_retry_requested(self, audio_abs_path, card):
        if self._local_model is None and not ConfigManager.get_config_value('model_options', 'use_api'):
            card.show_retry_error('Local model not loaded')
            return
        worker = RetryWorker(audio_abs_path, self._local_model)
        worker.successSignal.connect(lambda path, text, c=card: self._on_retry_success(path, text, c))
        worker.errorSignal.connect(lambda path, reason, c=card: self._on_retry_error(path, reason, c))
        worker.finished.connect(lambda w=worker: self._retry_workers.remove(w) if w in self._retry_workers else None)
        self._retry_workers.append(worker)
        worker.start()

    def _on_retry_success(self, audio_abs_path, text, card):
        # On a successful retry, transcript_log.txt has already been appended to
        # by llm_polish() inside transcribe(). Type the text into the focused app
        # (matching first-pass behavior), then clean up the failed entry + WAV.
        if text and self._input_simulator is not None:
            try:
                self._input_simulator.typewrite(text)
            except Exception as e:
                ConfigManager.console_print(f'typewrite failed during retry: {e}')

        try:
            if os.path.isfile(audio_abs_path):
                os.remove(audio_abs_path)
        except Exception as e:
            ConfigManager.console_print(f'Could not delete failed audio {audio_abs_path}: {e}')

        if self._failed_log_path:
            _remove_failed_entry(self._failed_log_path, card.audio_rel)

        self._load()

    def _on_retry_error(self, audio_abs_path, reason, card):
        card.show_retry_error(reason)

    def closeEvent(self, event):
        self.hide()
        event.ignore()
