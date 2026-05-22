"""Coordinates the always-warm audio capture + ElevenLabs session pool.

Goal: privacy-friendly warm-mic. Mic stays cold while the workstation is
locked (mic-in-use indicator off, no surprise listening). When the user
unlocks the screen, the mic warms up so the very first dictation has zero
PortAudio init / AGC settle / WS handshake latency. After `idle_minutes`
of no activity the mic cools down again — typically because the user has
walked away without locking.

Events that mark "activity":
  - Screen unlock observed by ScreenLockMonitor
  - Hotkey-driven dictation completed (the user is still here, give them
    another 10 min of warm mic for follow-ups)

Events that force cool:
  - Screen lock observed by ScreenLockMonitor
  - Idle timer expires after `idle_minutes` since last activity

A dictation triggered while cold still works — `ResultThread` falls back
to its legacy per-hotkey InputStream open. The first words *may* be
clipped on that cold dictation, but the very act of dictating marks
activity, so the *next* dictation is warm.

Thread model: all methods run on the Qt main thread. The mic/pool start()
and stop() calls are fast (microseconds for the locks; the actual heavy
work — InputStream open, WS handshake — happens on background threads
they own).
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import QObject, QTimer

from utils import ConfigManager


_LOG = logging.getLogger(__name__)


class WarmupCoordinator(QObject):
    """Drives `AudioCaptureService.start()/stop()` and
    `ElevenLabsSessionPool.start()/stop()` based on lock state + idle timer.
    """

    DEFAULT_IDLE_MINUTES = 10

    def __init__(
        self,
        audio_capture_service,
        session_pool,
        screen_lock_monitor,
        idle_minutes: int = DEFAULT_IDLE_MINUTES,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._service = audio_capture_service
        self._pool = session_pool
        self._monitor = screen_lock_monitor
        self._is_warm = False
        self._suspended_for_recording = False
        # When True, a cool event was requested while a dictation was in
        # flight. We don't honour it immediately (that would yank the mic
        # out from under the active recording — see the 2026-05-23 bug where
        # the idle timer fired mid-dictation and silently truncated audio).
        # Re-evaluated in resume_after_recording once the dictation finishes.
        self._cool_pending = False
        self._idle_minutes = int(idle_minutes)

        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.setInterval(self._idle_minutes * 60 * 1000)
        self._idle_timer.timeout.connect(self._on_idle_timeout)

        if self._monitor is not None:
            self._monitor.lockedChanged.connect(self._on_lock_changed)

    # ---- lifecycle ----

    def start(self) -> None:
        """Wire up the monitor and warm the mic if the screen is unlocked
        right now. Called once from `WhisperPCApp.initialize_components`."""
        if self._monitor is not None:
            self._monitor.start()
            initial_locked = self._monitor.is_locked()
        else:
            initial_locked = False
        if initial_locked:
            ConfigManager.console_print(
                'WarmupCoordinator: screen is locked at startup; mic stays cold'
            )
        else:
            ConfigManager.console_print(
                'WarmupCoordinator: screen unlocked at startup; warming mic'
            )
            self._warm()

    def shutdown(self) -> None:
        """Final tear-down at app exit. Pool/service close handled by main.cleanup."""
        self._idle_timer.stop()
        if self._monitor is not None:
            try:
                self._monitor.stop()
            except Exception:
                pass

    # ---- external events ----

    def note_dictation_started(self) -> None:
        """A dictation is starting. Counts as activity — reset the idle
        timer so we don't time-out the warm window mid-recording.

        Pair with `suspend_for_recording`: this resets the timer; that
        also blocks the cool path from running. Both are needed because
        a long dictation could outlast the timer that was already running."""
        if self._monitor is not None and self._monitor.is_locked():
            return
        # Reset the timer to a fresh full window. Don't call _warm() — the
        # service is presumably already warm (that's why we're not in the
        # legacy cold path); and suspend_for_recording was just called.
        self._idle_timer.start()

    def note_dictation_completed(self) -> None:
        """A hotkey-driven dictation just finished. Reset the idle timer
        and (if currently cold and unlocked) start warming for the next
        follow-up."""
        if self._monitor is not None and self._monitor.is_locked():
            return  # ignore — user dictating while locked is implausible, but be safe
        self._warm()  # idempotent: warm if not already warm + reset idle timer

    def suspend_for_recording(self) -> None:
        """ResultThread is starting up. Hold off on cooling — if the idle
        timer or a lock event fires mid-dictation, defer it via _cool_pending
        rather than killing the mic under the active recording.

        Symptom of the bug this guards against: dictation silently truncates
        mid-sentence, transcription fires on whatever was captured up to that
        point. Fixed 2026-05-23 after a 10-min-idle expiry interrupted a
        live dictation."""
        self._suspended_for_recording = True

    def resume_after_recording(self) -> None:
        """ResultThread has stopped. Resume normal warmup logic; honour any
        cool that was deferred during the dictation. Called from
        `result_thread.finished` (runs on the Qt main thread)."""
        self._suspended_for_recording = False
        # If a cool was requested mid-dictation, re-evaluate now that we
        # know the current state. Don't trust the stale flag — re-check
        # the actual lock status; if the user is still here (unlocked),
        # they just dictated, which extends the warm window.
        if self._cool_pending:
            self._cool_pending = False
            if self._monitor is not None and self._monitor.is_locked():
                self._cool()
                return
            # Idle-timer-driven cool, deferred and now resolved by activity.
            # _warm() restarts the timer with a fresh window.
            self._warm()
            return
        # Normal path — no deferred cool to clear.
        if self._monitor is not None and self._monitor.is_locked():
            return
        self._warm()

    # ---- lock state ----

    def _on_lock_changed(self, is_locked: bool) -> None:
        if is_locked:
            ConfigManager.console_print('WarmupCoordinator: screen locked; cooling mic')
            self._cool()
        else:
            ConfigManager.console_print('WarmupCoordinator: screen unlocked; warming mic')
            self._warm()

    # ---- idle timer ----

    def _on_idle_timeout(self) -> None:
        ConfigManager.console_print(
            f'WarmupCoordinator: {self._idle_minutes} min of inactivity; cooling mic'
        )
        self._cool()

    # ---- warm / cool ----

    def _warm(self) -> None:
        # Always reset the idle countdown, even if already warm — every
        # touch_activity / unlock should buy another `idle_minutes` of warmth.
        self._idle_timer.start()

        if self._suspended_for_recording:
            return  # actual start() deferred to resume_after_recording

        if self._service is not None and not self._service.is_alive():
            ok = self._service.start()
            if not ok:
                ConfigManager.console_print(
                    'WarmupCoordinator: AudioCaptureService.start() failed; '
                    'legacy per-hotkey mic open will be used'
                )
        if self._pool is not None:
            self._pool.start()
        self._is_warm = True

    def _cool(self) -> None:
        if self._suspended_for_recording:
            # Active dictation — we'd close the InputStream out from under
            # the consumer, truncating the user's audio mid-word. Defer
            # until resume_after_recording, which will re-evaluate state.
            ConfigManager.console_print(
                'WarmupCoordinator: cool requested mid-dictation; deferring'
            )
            self._cool_pending = True
            return
        self._idle_timer.stop()
        if self._pool is not None:
            try:
                self._pool.stop()
            except Exception:
                pass
        if self._service is not None and self._service.is_alive():
            try:
                self._service.stop()
            except Exception:
                pass
        self._is_warm = False
        self._cool_pending = False

    def is_warm(self) -> bool:
        return self._is_warm
