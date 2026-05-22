"""Detect Windows workstation lock / unlock and emit Qt signals.

Polls `OpenInputDesktop(DESKTOP_SWITCHDESKTOP)` every `poll_interval_ms` (default
5 s). The call returns a valid handle when the input desktop is the default
(unlocked) and NULL when the lock screen is in front. Cheap — microseconds per
call — and works without any special permissions or window handles.

WTSRegisterSessionNotification would give zero-latency events, but it requires
hooking a window's message procedure (or installing a Qt native event filter),
which is materially more code for a sub-5-second improvement the user won't
notice. Polling is the right trade-off here.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from PyQt5.QtCore import QObject, QTimer, pyqtSignal


_DESKTOP_SWITCHDESKTOP = 0x0100


def _query_locked_now() -> bool:
    """Return True if the workstation is locked right now (best-effort).

    On any Win32 / ctypes error we report 'unlocked' — better to warm the
    mic redundantly than to leave the user cold because of a probe glitch.
    """
    try:
        user32 = ctypes.windll.user32
        user32.OpenInputDesktop.restype = wintypes.HANDLE
        user32.OpenInputDesktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        user32.CloseDesktop.restype = wintypes.BOOL
        user32.CloseDesktop.argtypes = [wintypes.HANDLE]
        hdesk = user32.OpenInputDesktop(0, False, _DESKTOP_SWITCHDESKTOP)
        if not hdesk:
            return True  # NULL → can't see the input desktop → locked
        user32.CloseDesktop(hdesk)
        return False
    except Exception:
        return False


class ScreenLockMonitor(QObject):
    """Periodically polls the workstation lock state. Emits `locked_changed`
    when the state flips. `is_locked` reflects the most recent observation.
    """

    lockedChanged = pyqtSignal(bool)

    DEFAULT_POLL_MS = 5_000

    def __init__(self, parent: QObject | None = None, poll_interval_ms: int = DEFAULT_POLL_MS):
        super().__init__(parent)
        self._poll_interval_ms = poll_interval_ms
        self._timer = QTimer(self)
        self._timer.setInterval(self._poll_interval_ms)
        self._timer.timeout.connect(self._tick)
        self._is_locked = _query_locked_now()

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def is_locked(self) -> bool:
        return self._is_locked

    def _tick(self) -> None:
        now_locked = _query_locked_now()
        if now_locked == self._is_locked:
            return
        self._is_locked = now_locked
        self.lockedChanged.emit(now_locked)
