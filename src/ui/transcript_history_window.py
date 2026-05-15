import os
import sys
import time
import ctypes as _ctypes
from datetime import datetime as _datetime
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


# --- Diagnostic instrumentation (popup_diag.log) ------------------------------
# Lightweight timing/handle/memory snapshots written next to transcript_log.txt.
# Goal: localize whether the popup hang is parse, widget creation, or GDI/USER
# handle pressure on Windows. Remove once root cause is fixed.

_DIAG_PATH = None


def _diag_init(log_path):
    global _DIAG_PATH
    _DIAG_PATH = os.path.join(os.path.dirname(os.path.abspath(log_path)), 'popup_diag.log')


def _diag(msg):
    if _DIAG_PATH is None:
        return
    try:
        ts = _datetime.now().isoformat(timespec='milliseconds')
        with open(_DIAG_PATH, 'a', encoding='utf-8') as f:
            f.write(f'[{ts}] {msg}\n')
    except Exception:
        pass


def _handle_counts():
    """Return (GDI, USER) handle counts for the current process, or (None, None)."""
    try:
        user32 = _ctypes.windll.user32
        user32.GetGuiResources.restype = _ctypes.c_uint
        user32.GetGuiResources.argtypes = [_ctypes.c_void_p, _ctypes.c_uint]
        kernel32 = _ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = _ctypes.c_void_p
        hproc = kernel32.GetCurrentProcess()
        gdi = user32.GetGuiResources(hproc, 0)  # GR_GDIOBJECTS
        usr = user32.GetGuiResources(hproc, 1)  # GR_USEROBJECTS
        return gdi, usr
    except Exception:
        return None, None


def _mem_mb():
    """Return working-set in MB via psapi, or None."""
    try:
        class _PMC(_ctypes.Structure):
            _fields_ = [
                ('cb', _ctypes.c_ulong),
                ('PageFaultCount', _ctypes.c_ulong),
                ('PeakWorkingSetSize', _ctypes.c_size_t),
                ('WorkingSetSize', _ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', _ctypes.c_size_t),
                ('QuotaPagedPoolUsage', _ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', _ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', _ctypes.c_size_t),
                ('PagefileUsage', _ctypes.c_size_t),
                ('PeakPagefileUsage', _ctypes.c_size_t),
            ]
        pmc = _PMC()
        pmc.cb = _ctypes.sizeof(_PMC)
        hproc = _ctypes.windll.kernel32.GetCurrentProcess()
        if _ctypes.windll.psapi.GetProcessMemoryInfo(hproc, _ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / (1024 * 1024)
    except Exception:
        return None
    return None


def _fmt_mem(m):
    return f'{m:.1f}MB' if m is not None else 'n/a'


# Render only the most-recent N transcripts on first show; "Show older" reveals
# more in fixed-size chunks. Keeps the popup snappy regardless of log length.
INITIAL_RENDER_LIMIT = 25
LOAD_MORE_INCREMENT = 25


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

        _diag_init(log_path)
        gdi0, usr0 = _handle_counts()
        _diag(f'__init__ start gdi={gdi0} usr={usr0} mem={_fmt_mem(_mem_mb())} '
              f'log_size={os.path.getsize(log_path) if os.path.isfile(log_path) else 0}B')
        t0 = time.perf_counter()
        self._init_content()
        t_init = time.perf_counter() - t0
        t0 = time.perf_counter()
        self._load()
        t_load = time.perf_counter() - t0
        _diag(f'__init__ done init_content={t_init:.3f}s load={t_load:.3f}s')

    def showEvent(self, event):
        t0 = time.perf_counter()
        gdi0, usr0 = _handle_counts()
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
        gdi1, usr1 = _handle_counts()
        _diag(f'showEvent: {time.perf_counter()-t0:.3f}s '
              f'gdi={gdi0}->{gdi1} usr={usr0}->{usr1} mem={_fmt_mem(_mem_mb())}')

    def _init_content(self):
        header_row = QHBoxLayout()
        self._hint_label = QLabel('Click any entry to paste at cursor')
        self._hint_label.setFont(QFont('Segoe UI', 9))
        self._hint_label.setStyleSheet('color: #666;')
        header_row.addWidget(self._hint_label)
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

        # All parsed entries (newest first); only a prefix is materialized as cards.
        self._all_entries = []
        self._rendered_count = 0
        self._show_more_btn = None

    def _load(self):
        """Re-parse logs from disk and render the most-recent INITIAL_RENDER_LIMIT
        entries. Older entries stay in self._all_entries until the user clicks
        'Show older'."""
        t_total_start = time.perf_counter()
        gdi0, usr0 = _handle_counts()
        mem0 = _mem_mb()

        cleared = self._clear_cards()
        t_clear = time.perf_counter() - t_total_start

        t = time.perf_counter()
        ok_entries = _parse_log(self._log_path)
        t_parse_ok = time.perf_counter() - t

        t = time.perf_counter()
        failed_entries = _parse_failed_log(self._failed_log_path) if self._failed_log_path else []
        t_parse_failed = time.perf_counter() - t

        entries = ok_entries + failed_entries
        # Newest first by timestamp string (ISO-like format sorts correctly)
        entries.sort(key=lambda e: e.get('timestamp', ''), reverse=True)
        self._all_entries = entries
        self._rendered_count = 0

        if not entries:
            label = QLabel('No transcriptions yet.\nDictate something and come back.')
            label.setFont(QFont('Segoe UI', 10))
            label.setStyleSheet('color: #aaa;')
            label.setAlignment(Qt.AlignCenter)
            self._cards_layout.insertWidget(0, label)
            _diag(f'_load: empty total={time.perf_counter()-t_total_start:.3f}s '
                  f'cleared={cleared}')
            return

        t = time.perf_counter()
        rendered = self._render_more(INITIAL_RENDER_LIMIT)
        t_cards = time.perf_counter() - t

        gdi1, usr1 = _handle_counts()
        mem1 = _mem_mb()
        _diag(
            f'_load: total={time.perf_counter()-t_total_start:.3f}s '
            f'cleared={cleared} clear={t_clear:.3f}s '
            f'parse_ok={t_parse_ok:.3f}s({len(ok_entries)}) '
            f'parse_failed={t_parse_failed:.3f}s({len(failed_entries)}) '
            f'rendered={rendered}/{len(entries)} cards={t_cards:.3f}s '
            f'gdi={gdi0}->{gdi1} usr={usr0}->{usr1} '
            f'mem={_fmt_mem(mem0)}->{_fmt_mem(mem1)}'
        )

    def _clear_cards(self):
        """Remove all card widgets and any 'Show older' footer; keep the trailing
        stretch. Returns the number of widgets cleared."""
        cleared = 0
        # Iterate from the front; the trailing stretch (a QSpacerItem, no widget)
        # is naturally skipped because takeAt advances the live index.
        i = 0
        while i < self._cards_layout.count():
            item = self._cards_layout.itemAt(i)
            if item is None:
                break
            if item.widget() is not None:
                taken = self._cards_layout.takeAt(i)
                taken.widget().deleteLater()
                cleared += 1
            else:
                i += 1
        self._show_more_btn = None
        return cleared

    def _render_more(self, n):
        """Materialize the next `n` entries from self._all_entries as cards,
        starting at self._rendered_count. Returns how many cards were actually
        added. Updates the 'Show older' footer."""
        if self._show_more_btn is not None:
            # Drop the existing footer; we'll re-add (or skip) below.
            idx = self._cards_layout.indexOf(self._show_more_btn)
            if idx >= 0:
                self._cards_layout.takeAt(idx)
            self._show_more_btn.deleteLater()
            self._show_more_btn = None

        start = self._rendered_count
        end = min(start + n, len(self._all_entries))
        # Cards live before the trailing stretch; insert at len-1 (stretch index).
        for i in range(start, end):
            entry = self._all_entries[i]
            if entry['kind'] == 'ok':
                card = TranscriptCard(entry['timestamp'], entry['text'], self._container)
            else:
                audio_abs = os.path.join(
                    self._project_root, entry['audio_rel'].replace('/', os.sep)
                )
                card = FailedTranscriptCard(
                    entry['timestamp'], audio_abs, entry['audio_rel'],
                    entry['error'], self._container,
                )
                card.retryRequested.connect(self._on_retry_requested)
            insert_at = self._cards_layout.count() - 1  # before the stretch
            self._cards_layout.insertWidget(insert_at, card)
        added = end - start
        self._rendered_count = end

        remaining = len(self._all_entries) - self._rendered_count
        if remaining > 0:
            self._show_more_btn = QPushButton(
                f'Show {min(LOAD_MORE_INCREMENT, remaining)} older  '
                f'({self._rendered_count} of {len(self._all_entries)} shown)'
            )
            self._show_more_btn.setFont(QFont('Segoe UI', 9))
            self._show_more_btn.setFixedHeight(32)
            self._show_more_btn.setCursor(QCursor(Qt.PointingHandCursor))
            self._show_more_btn.setStyleSheet('''
                QPushButton {
                    background: #f0f0f0;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    padding: 0 12px;
                    color: #404040;
                }
                QPushButton:hover { background: #e0e0e0; }
            ''')
            self._show_more_btn.clicked.connect(self._on_show_more_clicked)
            insert_at = self._cards_layout.count() - 1
            self._cards_layout.insertWidget(insert_at, self._show_more_btn)
        return added

    def _on_show_more_clicked(self):
        t = time.perf_counter()
        added = self._render_more(LOAD_MORE_INCREMENT)
        _diag(f'_show_more: added={added} now_rendered={self._rendered_count}/'
              f'{len(self._all_entries)} t={time.perf_counter()-t:.3f}s')

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
