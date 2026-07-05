import os
import sys
import time
import ctypes as _ctypes
from datetime import datetime as _datetime
import numpy as np
import soundfile as sf
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QListView, QPushButton, QApplication,
                              QStyledItemDelegate, QStyle, QMenu, QDialog,
                              QPlainTextEdit, QDialogButtonBox)
from PyQt5.QtCore import (Qt, QTimer, QThread, QSize, QRect, QModelIndex,
                          QAbstractListModel, pyqtSignal)
from PyQt5.QtGui import QFont, QCursor, QColor, QPainter, QFontMetrics, QPen

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui.base_window import BaseWindow
from transcription import transcribe, TranscriptionAPIError
from utils import ConfigManager


# --- Diagnostic instrumentation (popup_diag.log) ------------------------------
# Lightweight timing/handle/memory snapshots next to transcript_log.txt. Kept
# through one verification cycle of the Ditto-style rebuild so we can confirm
# the new architecture is stable; remove once that's done.

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


# Set ctypes signatures so handle/memory queries don't truncate HANDLE returns
# on 64-bit Windows. Without these, the diag log just records gdi=0 usr=0.
try:
    _ctypes.windll.kernel32.GetCurrentProcess.restype = _ctypes.c_void_p
    _ctypes.windll.user32.GetGuiResources.argtypes = [_ctypes.c_void_p, _ctypes.c_ulong]
    _ctypes.windll.user32.GetGuiResources.restype = _ctypes.c_ulong
    _ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
        _ctypes.c_void_p, _ctypes.c_void_p, _ctypes.c_ulong,
    ]
    _ctypes.windll.psapi.GetProcessMemoryInfo.restype = _ctypes.c_int
except Exception:
    pass


def _handle_counts():
    try:
        hproc = _ctypes.windll.kernel32.GetCurrentProcess()
        gdi = _ctypes.windll.user32.GetGuiResources(hproc, 0)  # GR_GDIOBJECTS
        usr = _ctypes.windll.user32.GetGuiResources(hproc, 1)  # GR_USEROBJECTS
        return gdi, usr
    except Exception:
        return None, None


def _mem_mb():
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


# --- Log parsing --------------------------------------------------------------

def _parse_log_slice(content):
    """Parse polished-transcript blocks from a text slice. Captures RAW
    (pre-polish STT output), POLISHED (typed text), and ENGINE (which backend
    produced the entry — added 2026-05-22)."""
    entries = []
    for block in content.strip().split('\n\n'):
        lines = block.strip().split('\n')
        timestamp = raw = polished = engine = ''
        for line in lines:
            s = line.strip()
            if s.startswith('[') and s.endswith(']'):
                timestamp = s[1:-1]
            elif s.startswith('ENGINE:'):
                engine = s[7:].strip()
            elif s.startswith('RAW:'):
                raw = s[4:].strip()
            elif s.startswith('POLISHED:'):
                polished = s[9:].strip()
        if polished:
            entries.append({'kind': 'ok', 'timestamp': timestamp,
                            'text': polished, 'raw': raw, 'engine': engine})
    return entries


def _parse_failed_slice(content):
    """Parse failed-transcript blocks from a text slice."""
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
            entries.append({
                'kind': 'failed',
                'timestamp': timestamp,
                'audio_rel': audio_rel,
                'error': error,
                'retry_state': 'idle',
                'retry_error': '',
            })
    return entries


def _read_whole(path):
    if not path or not os.path.isfile(path):
        return ''
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


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


# --- Retry worker -------------------------------------------------------------

class RetryWorker(QThread):
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
            if result and result.strip():
                self.successSignal.emit(self.audio_abs_path, result)
            else:
                # An empty retry is NOT a success — the recording still yielded
                # no text. Report it as an error so the failed entry AND the
                # audio are kept for another try, instead of being cleaned up.
                # (Pre-2026-07-05 this emitted success on empty, and the success
                # handler then hard-deleted the only copy of the audio.)
                self.errorSignal.emit(self.audio_abs_path, 'Retry produced no text (still empty)')
        except TranscriptionAPIError as e:
            self.errorSignal.emit(self.audio_abs_path, e.reason)
        except Exception as e:
            self.errorSignal.emit(self.audio_abs_path, f'{type(e).__name__}: {e}')


# --- Model + roles ------------------------------------------------------------

KindRole = Qt.UserRole + 1        # 'ok' or 'failed'
TimestampRole = Qt.UserRole + 2
TextRole = Qt.UserRole + 3        # polished text (ok only)
AudioRelRole = Qt.UserRole + 4    # failed only
AudioAbsRole = Qt.UserRole + 5    # failed only
ErrorRole = Qt.UserRole + 6       # failed only
RetryStateRole = Qt.UserRole + 7  # 'idle' / 'retrying'
RetryErrorRole = Qt.UserRole + 8  # last retry-failure reason
RawRole = Qt.UserRole + 9         # pre-polish STT output (ok only)
EngineRole = Qt.UserRole + 10     # engine label e.g. 'elevenlabs-stream'


class TranscriptHistoryModel(QAbstractListModel):
    """Newest-first list of transcripts. Tail-reads on refresh()."""

    def __init__(self, log_path, failed_log_path, project_root, parent=None):
        super().__init__(parent)
        self._log_path = log_path
        self._failed_log_path = failed_log_path
        self._project_root = project_root
        self._entries = []
        self._ok_last_size = 0
        self._failed_last_size = 0
        self._reload()

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._entries)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        i = index.row()
        if i < 0 or i >= len(self._entries):
            return None
        e = self._entries[i]
        if role == KindRole:
            return e['kind']
        if role == TimestampRole:
            return e.get('timestamp', '')
        if role == TextRole:
            return e.get('text', '')
        if role == AudioRelRole:
            return e.get('audio_rel', '')
        if role == AudioAbsRole:
            audio_rel = e.get('audio_rel')
            if audio_rel:
                return os.path.join(self._project_root, audio_rel.replace('/', os.sep))
            return ''
        if role == ErrorRole:
            return e.get('error', '')
        if role == RetryStateRole:
            return e.get('retry_state', 'idle')
        if role == RetryErrorRole:
            return e.get('retry_error', '')
        if role == RawRole:
            return e.get('raw', '')
        if role == EngineRole:
            return e.get('engine', '')
        if role == Qt.DisplayRole:
            return e.get('text') or e.get('error') or ''
        return None

    def _reload(self):
        ok_content = _read_whole(self._log_path)
        failed_content = _read_whole(self._failed_log_path)
        merged = _parse_log_slice(ok_content) + _parse_failed_slice(failed_content)
        merged.sort(key=lambda e: e.get('timestamp', ''), reverse=True)

        self.beginResetModel()
        self._entries = merged
        self._ok_last_size = os.path.getsize(self._log_path) if os.path.isfile(self._log_path) else 0
        self._failed_last_size = (os.path.getsize(self._failed_log_path)
                                  if self._failed_log_path and os.path.isfile(self._failed_log_path)
                                  else 0)
        self.endResetModel()

    def refresh(self):
        """Tail-read new entries from both logs. Falls back to full reload if
        either file shrank (truncation, rotation, retry-success cleanup)."""
        cur_ok_size = os.path.getsize(self._log_path) if os.path.isfile(self._log_path) else 0
        cur_fail_size = (os.path.getsize(self._failed_log_path)
                         if self._failed_log_path and os.path.isfile(self._failed_log_path)
                         else 0)

        if cur_ok_size < self._ok_last_size or cur_fail_size < self._failed_last_size:
            self._reload()
            return

        new_entries = []
        if cur_ok_size > self._ok_last_size:
            with open(self._log_path, 'r', encoding='utf-8') as f:
                f.seek(self._ok_last_size)
                tail = f.read()
            new_entries.extend(_parse_log_slice(tail))
            self._ok_last_size = cur_ok_size
        if self._failed_log_path and cur_fail_size > self._failed_last_size:
            with open(self._failed_log_path, 'r', encoding='utf-8') as f:
                f.seek(self._failed_last_size)
                tail = f.read()
            new_entries.extend(_parse_failed_slice(tail))
            self._failed_last_size = cur_fail_size

        if not new_entries:
            return

        new_entries.sort(key=lambda e: e.get('timestamp', ''), reverse=True)
        self.beginInsertRows(QModelIndex(), 0, len(new_entries) - 1)
        self._entries = new_entries + self._entries
        self.endInsertRows()

    def set_retry_state(self, row, state, error_text=''):
        if 0 <= row < len(self._entries):
            self._entries[row]['retry_state'] = state
            self._entries[row]['retry_error'] = error_text
            idx = self.index(row)
            self.dataChanged.emit(idx, idx)

    def find_failed_row_by_audio_rel(self, audio_rel):
        for i, e in enumerate(self._entries):
            if e['kind'] == 'failed' and e.get('audio_rel') == audio_rel:
                return i
        return -1


# --- Engine badge mapping -----------------------------------------------------

def _engine_badge(engine):
    """Map a PC engine label (the ENGINE: line in transcript_log.txt) to the
    same friendly badge the mobile app shows: (label, foreground QColor,
    background QColor). Returns None when there's no engine recorded (older
    entries that pre-date ENGINE logging).

    The mapping keys off what actually *produced* the text, so a fallback chain
    like 'groq→gemini-fallback' shows as 'Gemini' and 'groq→local-whisper-
    fallback' shows as 'On-device' — matching the phone, where GEMINI_AUDIO and
    ANDROID_LOCAL get their own badges regardless of what was tried first."""
    if not engine:
        return None
    e = engine.lower()

    def _badge(label, hex_fg):
        fg = QColor(hex_fg)
        bg = QColor(hex_fg)
        bg.setAlpha(38)  # ~15% tint over the card, mirrors mobile's 0x33 chips
        return label, fg, bg

    if 'gemini' in e:
        return _badge('Gemini', '#4527A0')          # purple
    if 'local' in e:                                  # local-whisper fallback
        return _badge('On-device', '#455A64')        # grey
    if e.startswith('elevenlabs'):
        return _badge('ElevenLabs', '#0D47A1')       # blue
    if e.startswith('groq'):
        return _badge('Groq + Polish', '#00695C')    # teal
    # Unknown / future engine: show the raw label in a neutral chip.
    return _badge(engine, '#808080')


# --- Delegate -----------------------------------------------------------------

class TranscriptItemDelegate(QStyledItemDelegate):
    """Paints one row: timestamp + body inside a rounded card.
    Failed rows include a 'Retry' pill on the right side; the list view turns
    a click in that rect into a retryClicked signal."""

    MARGIN_H = 12
    MARGIN_V = 8
    INNER_SPACING = 3
    INTER_CARD_GAP = 6
    BTN_W = 78
    BTN_H = 26
    BTN_INSET_R = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ts_font = QFont('Segoe UI', 8)
        self._body_font = QFont('Segoe UI', 10)
        self._btn_font = QFont('Segoe UI', 9)
        self._chip_font = QFont('Segoe UI', 8, QFont.Bold)
        self._ts_fm = QFontMetrics(self._ts_font)
        self._body_fm = QFontMetrics(self._body_font)
        self._chip_fm = QFontMetrics(self._chip_font)

    # Engine chip geometry (matches the mobile provider chip proportions).
    CHIP_PAD_H = 8
    CHIP_PAD_V = 2

    @classmethod
    def card_rect(cls, option_rect):
        return QRect(
            option_rect.left(),
            option_rect.top(),
            option_rect.width(),
            option_rect.height() - cls.INTER_CARD_GAP,
        )

    @classmethod
    def retry_rect(cls, card_rect):
        return QRect(
            card_rect.right() - cls.BTN_INSET_R - cls.BTN_W,
            card_rect.top() + (card_rect.height() - cls.BTN_H) // 2,
            cls.BTN_W,
            cls.BTN_H,
        )

    def paint(self, painter, option, index):
        kind = index.data(KindRole) or 'ok'
        timestamp = index.data(TimestampRole) or ''
        hovered = bool(option.state & QStyle.State_MouseOver)

        if kind == 'ok':
            body_text = index.data(TextRole) or ''
            if hovered:
                bg, border, body_color = QColor('#eaf4ea'), QColor('#5aac5a'), QColor('#2c2c2c')
            else:
                bg, border, body_color = QColor('#f7f7f7'), QColor('#e0e0e0'), QColor('#2c2c2c')
        else:
            err = index.data(ErrorRole) or 'Transcription failed'
            retry_err = index.data(RetryErrorRole) or ''
            body_text = '⚠  ' + err
            if retry_err:
                body_text += f'\nRetry failed: {retry_err}'
            if hovered:
                bg, border, body_color = QColor('#fbd9d9'), QColor('#d77'), QColor('#8a2828')
            else:
                bg, border, body_color = QColor('#fdecec'), QColor('#e8b8b8'), QColor('#8a2828')

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        card = self.card_rect(option.rect)
        painter.setBrush(bg)
        painter.setPen(QPen(border, 1))
        painter.drawRoundedRect(card, 8, 8)

        right_reserve = (self.BTN_W + self.BTN_INSET_R + 4) if kind == 'failed' else 0
        text_left = card.left() + self.MARGIN_H
        text_right = card.right() - self.MARGIN_H - right_reserve
        text_width = max(text_right - text_left, 10)

        painter.setFont(self._ts_font)
        painter.setPen(QColor('#999'))
        ts_y = card.top() + self.MARGIN_V + self._ts_fm.ascent()
        painter.drawText(text_left, ts_y, timestamp)

        # Engine badge (ok rows only) — top-right of the card, on the timestamp
        # row. Names which STT engine/channel produced this entry, matching the
        # mobile history chips. Failed rows reserve the right side for Retry.
        if kind == 'ok':
            badge = _engine_badge(index.data(EngineRole) or '')
            if badge is not None:
                label, fg, bg = badge
                chip_w = self._chip_fm.horizontalAdvance(label) + 2 * self.CHIP_PAD_H
                chip_h = self._chip_fm.height() + 2 * self.CHIP_PAD_V
                chip_left = card.right() - self.MARGIN_H - chip_w
                chip_top = card.top() + self.MARGIN_V + (self._ts_fm.height() - chip_h) // 2
                chip = QRect(chip_left, chip_top, chip_w, chip_h)
                painter.setBrush(bg)
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(chip, chip_h // 2, chip_h // 2)
                painter.setFont(self._chip_font)
                painter.setPen(fg)
                painter.drawText(chip, Qt.AlignCenter, label)

        painter.setFont(self._body_font)
        painter.setPen(body_color)
        body_top = card.top() + self.MARGIN_V + self._ts_fm.height() + self.INNER_SPACING
        body_rect = QRect(text_left, body_top, text_width,
                          card.bottom() - body_top - self.MARGIN_V)
        painter.drawText(body_rect, Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignTop, body_text)

        if kind == 'failed':
            state = index.data(RetryStateRole) or 'idle'
            audio_abs = index.data(AudioAbsRole) or ''
            audio_present = bool(audio_abs) and os.path.isfile(audio_abs)
            btn = self.retry_rect(card)
            if state == 'retrying':
                btn_bg, btn_border, btn_fg, btn_label = (
                    QColor('#f5f5f5'), QColor('#ccc'), QColor('#999'), '…  Retrying'
                )
            elif not audio_present:
                btn_bg, btn_border, btn_fg, btn_label = (
                    QColor('#f5f5f5'), QColor('#ccc'), QColor('#999'), '↻ Retry'
                )
            else:
                btn_bg, btn_border, btn_fg, btn_label = (
                    QColor('#ffffff'), QColor('#d77'), QColor('#8a2828'), '↻ Retry'
                )
            painter.setBrush(btn_bg)
            painter.setPen(QPen(btn_border, 1))
            painter.drawRoundedRect(btn, 4, 4)
            painter.setFont(self._btn_font)
            painter.setPen(btn_fg)
            painter.drawText(btn, Qt.AlignCenter, btn_label)

        painter.restore()

    def sizeHint(self, option, index):
        kind = index.data(KindRole) or 'ok'
        if kind == 'ok':
            body_text = index.data(TextRole) or ''
        else:
            body_text = '⚠  ' + (index.data(ErrorRole) or 'Transcription failed')
            retry_err = index.data(RetryErrorRole) or ''
            if retry_err:
                body_text += f'\nRetry failed: {retry_err}'

        width = option.rect.width()
        if width <= 0:
            view = self.parent()
            if isinstance(view, QListView):
                width = view.viewport().width()
            if width <= 0:
                width = 520

        right_reserve = (self.BTN_W + self.BTN_INSET_R + 4) if kind == 'failed' else 0
        text_width = max(width - (2 * self.MARGIN_H) - right_reserve, 10)
        body_rect = self._body_fm.boundingRect(
            0, 0, text_width, 100000,
            Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignTop,
            body_text,
        )
        height = (self.MARGIN_V + self._ts_fm.height() + self.INNER_SPACING +
                  body_rect.height() + self.MARGIN_V + self.INTER_CARD_GAP)
        if kind == 'failed':
            min_h = self.BTN_H + 2 * self.MARGIN_V + self.INTER_CARD_GAP
            height = max(height, min_h)
        return QSize(width, max(height, 48))


# --- List view (custom click handling) ----------------------------------------

class _RawPolishedDialog(QDialog):
    """Modal showing both RAW (pre-polish STT output) and POLISHED side by side.
    Selectable, copyable text so the user can grab either for re-use."""

    def __init__(self, timestamp, raw, polished, parent=None, engine=''):
        super().__init__(parent)
        self.setWindowTitle(f'Transcript — {timestamp}')
        self.resize(720, 460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        if engine:
            engine_label = QLabel(f'ENGINE: {engine}')
            engine_label.setFont(QFont('Segoe UI', 9, QFont.Bold))
            engine_label.setStyleSheet('color: #1E88E5;')
            layout.addWidget(engine_label)

        raw_label = QLabel('RAW (pre-polish STT output):')
        raw_label.setFont(QFont('Segoe UI', 9, QFont.Bold))
        layout.addWidget(raw_label)
        raw_edit = QPlainTextEdit(raw or '(empty — pre-polish text not recorded for this entry)')
        raw_edit.setReadOnly(True)
        raw_edit.setFont(QFont('Segoe UI', 10))
        layout.addWidget(raw_edit, stretch=1)

        polished_label = QLabel('POLISHED (what got typed):')
        polished_label.setFont(QFont('Segoe UI', 9, QFont.Bold))
        layout.addWidget(polished_label)
        polished_edit = QPlainTextEdit(polished or '')
        polished_edit.setReadOnly(True)
        polished_edit.setFont(QFont('Segoe UI', 10))
        layout.addWidget(polished_edit, stretch=1)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        layout.addWidget(btns)


class TranscriptListView(QListView):
    """Click on an OK row → okClicked(text). Click on a failed-row retry pill
    → retryClicked(row). Right-click → context menu (copy RAW, view RAW vs
    POLISHED, etc.). Selection is disabled to keep the visual quiet."""

    okClicked = pyqtSignal(str)
    retryClicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListView.NoSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(QListView.ScrollPerPixel)
        self.setUniformItemSizes(False)
        self.setFocusPolicy(Qt.NoFocus)
        # MouseTracking on the viewport drives QStyle::State_MouseOver for hover.
        self.viewport().setMouseTracking(True)
        self.setMouseTracking(True)
        self.setStyleSheet(
            'QListView { background: transparent; border: none; }'
            'QListView::item { background: transparent; }'
        )

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        idx = self.indexAt(event.pos())
        if not idx.isValid():
            super().mousePressEvent(event)
            return

        rect = self.visualRect(idx)
        kind = idx.data(KindRole)
        if kind == 'ok':
            self.okClicked.emit(idx.data(TextRole) or '')
            event.accept()
            return
        if kind == 'failed':
            card = TranscriptItemDelegate.card_rect(rect)
            btn = TranscriptItemDelegate.retry_rect(card)
            if btn.contains(event.pos()):
                self.retryClicked.emit(idx.row())
                event.accept()
                return
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Word-wrap heights depend on viewport width; force re-layout.
        self.scheduleDelayedItemsLayout()

    def contextMenuEvent(self, event):
        idx = self.indexAt(event.pos())
        if not idx.isValid():
            return
        kind = idx.data(KindRole)
        menu = QMenu(self)

        if kind == 'ok':
            polished = idx.data(TextRole) or ''
            raw = idx.data(RawRole) or ''
            ts = idx.data(TimestampRole) or ''

            act_copy_polished = menu.addAction('Copy polished text')
            act_copy_raw = menu.addAction('Copy raw (pre-polish) transcript')
            act_view = menu.addAction('View RAW + POLISHED…')
            # Disable raw-side actions when there's no raw recorded (older
            # entries that pre-date dual-line logging, or any future entry
            # logged without a RAW line).
            if not raw:
                act_copy_raw.setEnabled(False)
                act_copy_raw.setText('Copy raw  (not recorded)')

            chosen = menu.exec_(event.globalPos())
            if chosen is act_copy_polished:
                QApplication.clipboard().setText(polished)
            elif chosen is act_copy_raw:
                QApplication.clipboard().setText(raw)
            elif chosen is act_view:
                engine = idx.data(EngineRole) or ''
                dlg = _RawPolishedDialog(ts, raw, polished,
                                          parent=self.window(), engine=engine)
                dlg.exec_()
            return

        if kind == 'failed':
            err = idx.data(ErrorRole) or ''
            audio_rel = idx.data(AudioRelRole) or ''
            audio_abs = idx.data(AudioAbsRole) or ''
            act_copy_err = menu.addAction('Copy error message')
            act_copy_path = menu.addAction('Copy audio file path')
            if not audio_rel:
                act_copy_path.setEnabled(False)

            chosen = menu.exec_(event.globalPos())
            if chosen is act_copy_err:
                QApplication.clipboard().setText(err)
            elif chosen is act_copy_path:
                QApplication.clipboard().setText(audio_abs or audio_rel)


# --- The window ---------------------------------------------------------------

class TranscriptHistoryWindow(BaseWindow):
    """Persistent singleton — constructed once at app startup, hidden when not
    in use, shown on hotkey/tray. Virtualized QListView keeps every operation
    cheap regardless of log size."""

    def __init__(self, log_path, failed_log_path=None, local_model=None, input_simulator=None):
        super().__init__('Transcript History', 540, 680)
        self._log_path = log_path
        self._failed_log_path = failed_log_path
        self._project_root = os.path.dirname(os.path.abspath(log_path))
        self._local_model = local_model
        self._input_simulator = input_simulator
        self._retry_workers = []

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        _diag_init(log_path)
        gdi0, usr0 = _handle_counts()
        log_size = os.path.getsize(log_path) if os.path.isfile(log_path) else 0
        _diag(f'__init__ start gdi={gdi0} usr={usr0} mem={_fmt_mem(_mem_mb())} '
              f'log_size={log_size}B')
        t0 = time.perf_counter()

        self._init_content()
        self._model = TranscriptHistoryModel(
            log_path, failed_log_path, self._project_root, parent=self,
        )
        self._list_view.setModel(self._model)
        self._list_view.okClicked.connect(self._on_ok_clicked)
        self._list_view.retryClicked.connect(self._on_retry_clicked)

        gdi1, usr1 = _handle_counts()
        _diag(f'__init__ done in {time.perf_counter()-t0:.3f}s '
              f'rows={self._model.rowCount()} gdi={gdi0}->{gdi1} usr={usr0}->{usr1} '
              f'mem={_fmt_mem(_mem_mb())}')

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
        refresh_btn.clicked.connect(self.refresh)
        header_row.addWidget(refresh_btn)
        self.main_layout.addLayout(header_row)

        self._list_view = TranscriptListView(self)
        self._list_view.setItemDelegate(TranscriptItemDelegate(self._list_view))
        self.main_layout.addWidget(self._list_view, stretch=1)

        self._status_label = QLabel('')
        self._status_label.setFont(QFont('Segoe UI', 9))
        self._status_label.setStyleSheet('color: #3a863a;')
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.hide()
        self.main_layout.addWidget(self._status_label)

    def showEvent(self, event):
        t0 = time.perf_counter()
        gdi0, usr0 = _handle_counts()
        super().showEvent(event)
        # WS_EX_NOACTIVATE: window receives mouse events but never becomes the
        # active (keyboard-focus) window, so clicks paste into the previously
        # focused app via _simulate_paste().
        try:
            import ctypes
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            hwnd = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE)
        except Exception:
            pass
        gdi1, usr1 = _handle_counts()
        _diag(f'showEvent: {time.perf_counter()-t0:.3f}s '
              f'gdi={gdi0}->{gdi1} usr={usr0}->{usr1} mem={_fmt_mem(_mem_mb())}')

    def closeEvent(self, event):
        self.hide()
        event.ignore()

    def refresh(self):
        t0 = time.perf_counter()
        before = self._model.rowCount()
        self._model.refresh()
        after = self._model.rowCount()
        _diag(f'refresh: {time.perf_counter()-t0:.3f}s rows={before}->{after}')

    def _on_ok_clicked(self, polished_text):
        if not polished_text:
            return
        QApplication.clipboard().setText(polished_text)
        # 50ms lets the clipboard settle before Ctrl+V fires.
        QTimer.singleShot(50, _simulate_paste)
        self._status_label.setText('✓  Pasted at cursor')
        self._status_label.show()
        QTimer.singleShot(2000, self._status_label.hide)

    def _on_retry_clicked(self, row):
        m = self._model
        if row < 0 or row >= m.rowCount():
            return
        idx = m.index(row)
        if idx.data(KindRole) != 'failed':
            return
        if idx.data(RetryStateRole) == 'retrying':
            return

        audio_abs = idx.data(AudioAbsRole) or ''
        audio_rel = idx.data(AudioRelRole) or ''
        if not audio_abs or not os.path.isfile(audio_abs):
            m.set_retry_state(row, 'idle', 'audio missing')
            return
        if self._local_model is None and not ConfigManager.get_config_value('model_options', 'use_api'):
            m.set_retry_state(row, 'idle', 'local model not loaded')
            return

        m.set_retry_state(row, 'retrying', '')

        worker = RetryWorker(audio_abs, self._local_model)
        worker.successSignal.connect(
            lambda p, t, ar=audio_rel: self._on_retry_success(p, t, ar)
        )
        worker.errorSignal.connect(
            lambda p, r, ar=audio_rel: self._on_retry_error(p, r, ar)
        )
        worker.finished.connect(
            lambda w=worker: self._retry_workers.remove(w) if w in self._retry_workers else None
        )
        self._retry_workers.append(worker)
        worker.start()

    def _on_retry_success(self, audio_abs, text, audio_rel):
        # llm_polish() inside transcribe() already appended to transcript_log.
        if text and self._input_simulator is not None:
            try:
                self._input_simulator.typewrite(text)
            except Exception as e:
                ConfigManager.console_print(f'typewrite failed during retry: {e}')

        # Do NOT delete the audio here. It lives in the recordings/ archive and
        # is pruned by age at startup. Hard-deleting on retry-success is what
        # destroyed the only copy of a recording on 2026-07-05 (an empty result
        # was mis-treated as success, then this os.remove ran). The failed entry
        # is removed below so the item leaves the "failed" list; the audio stays
        # safely archived.
        if self._failed_log_path:
            _remove_failed_entry(self._failed_log_path, audio_rel)

        # Failed-log shrank → refresh() detects it and triggers a full reload,
        # surfacing the new ok entry too.
        self.refresh()

    def _on_retry_error(self, audio_abs, reason, audio_rel):
        row = self._model.find_failed_row_by_audio_rel(audio_rel)
        if row >= 0:
            self._model.set_retry_state(row, 'idle', reason)
