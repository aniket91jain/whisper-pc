"""Pre-opened ElevenLabs Realtime session pool.

Holds (at most) one warm `Session` ready to receive audio. On `acquire()`,
hands out the warm session and kicks off opening a replacement in the
background. The hot path — pressing the hotkey to dictate — pays zero
WebSocket-handshake latency 99% of the time.

When idle, the warm session is kept alive by the `Session._keepalive_loop`
which pings every 10 s (ElevenLabs RT idle timeout is ~15 s). Once acquired,
the caller is responsible for `commit()` / `cancel()` per usual.

Keyterm changes (user edits proper-nouns config) invalidate the warm
session via a version stamp — the next acquire opens fresh.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from utils import ConfigManager


_LOG = logging.getLogger(__name__)


class ElevenLabsSessionPool:
    """Singleton-in-practice pool of one warm `Session`.

    Lifecycle:
        pool = ElevenLabsSessionPool()
        pool.start()                    # at app launch
        session = pool.acquire()        # on hotkey — may return None
        if session is None:
            # no warm session; caller falls through to burst path
        else:
            session.send_audio_chunk(...)
            session.commit(on_result)
        pool.stop()                     # at app exit
    """

    # How long acquire() will wait for an in-flight open to complete before
    # giving up and returning None. Tight — caller must not stall on the
    # hotkey path. If the pool isn't warm yet, fall back to burst.
    _ACQUIRE_WAIT_S = 0.05

    def __init__(self):
        self._lock = threading.Lock()
        self._session = None
        self._session_keyterms: tuple[str, ...] = ()
        self._opening_thread: Optional[threading.Thread] = None
        self._opening_started_at = 0.0
        self._stopped = False
        self._enabled = False

    # ---- lifecycle ----

    def start(self) -> None:
        """Open the first warm session in the background. Returns immediately.

        Safe to call after a previous stop() — the WarmupCoordinator cycles
        the pool on screen lock/unlock and idle timeout."""
        self._stopped = False
        if not self._streaming_enabled():
            ConfigManager.console_print(
                'ElevenLabsSessionPool: streaming disabled (engine=groq or '
                'stt_streaming_mode=false); pool not started'
            )
            return
        if not self._api_key():
            ConfigManager.console_print(
                'ElevenLabsSessionPool: ELEVENLABS_API_KEY not set; pool not started'
            )
            return
        self._enabled = True
        self._kick_replenish()

    def stop(self) -> None:
        """Cancel the warm session and stop opening replacements.

        Reversible — calling start() again re-arms the pool. Used by the
        WarmupCoordinator on screen lock and idle timeout."""
        self._stopped = True
        self._enabled = False
        with self._lock:
            session = self._session
            self._session = None
        if session is not None:
            try:
                session.cancel()
            except Exception:
                pass

    def is_enabled(self) -> bool:
        return self._enabled

    # ---- hot path ----

    def acquire(self) -> Optional[object]:
        """Return the warm Session, kick off a replacement in background.

        Returns None when no session is warm. Caller must then fall through
        to the burst path (audio still captured via AudioCaptureService;
        commit happens via transcribe_burst at end of recording).

        Briefly waits (up to _ACQUIRE_WAIT_S) if an open is in flight —
        catches the common case where the user hits hotkey right after
        app launch before the first session has finished opening.
        """
        if self._stopped or not self._enabled:
            return None

        # Verify keyterms still match — user may have edited proper-nouns
        # since the warm session opened. If they have, invalidate now.
        current_keyterms = self._current_keyterms()
        with self._lock:
            if self._session is not None and self._session_keyterms != current_keyterms:
                ConfigManager.console_print(
                    'ElevenLabsSessionPool: keyterms changed since warm session opened; '
                    'discarding and reopening'
                )
                stale = self._session
                self._session = None
                try:
                    stale.cancel()
                except Exception:
                    pass

        # Try the warm session first.
        session = self._take_warm_session()
        if session is not None:
            ConfigManager.console_print(
                'ElevenLabsSessionPool: handed out warm session, replenishing'
            )
            self._kick_replenish()
            return session

        # No warm session. Give an in-flight open a tiny window to finish.
        if self._opening_thread is not None and self._opening_thread.is_alive():
            self._opening_thread.join(timeout=self._ACQUIRE_WAIT_S)
            session = self._take_warm_session()
            if session is not None:
                ConfigManager.console_print(
                    f'ElevenLabsSessionPool: warm session caught up within '
                    f'{self._ACQUIRE_WAIT_S*1000:.0f}ms'
                )
                self._kick_replenish()
                return session

        # Still nothing — kick a replenish (in case the previous attempt died
        # without scheduling a retry) and tell the caller to use burst.
        self._kick_replenish()
        return None

    def try_acquire_nowait(self):
        """Return the warm Session only if one is instantly ready; never blocks.

        Unlike acquire(), this does NOT join an in-flight open — it returns None
        immediately when no session is warm. Used on the dictation resume path,
        which runs while recording is paused: any blocking there would drop
        post-resume audio, so the caller instead opens a fresh (buffering)
        session inline when this returns None. Kicks a background replenish so a
        warm session is ready for next time.
        """
        if self._stopped or not self._enabled:
            return None
        session = self._take_warm_session()
        if session is not None:
            ConfigManager.console_print(
                'ElevenLabsSessionPool: handed out warm session (nowait), replenishing'
            )
        self._kick_replenish()
        return session

    # ---- internals ----

    def _take_warm_session(self):
        """Atomically pull the warm session out of the pool if it's healthy.

        Health-check: must report is_ready() True. A session whose WS died
        is discarded silently."""
        with self._lock:
            session = self._session
            self._session = None
        if session is None:
            return None
        try:
            if not session.is_ready():
                ConfigManager.console_print(
                    'ElevenLabsSessionPool: warm session is no longer ready; discarding'
                )
                try:
                    session.cancel()
                except Exception:
                    pass
                return None
        except Exception:
            return None
        return session

    def _kick_replenish(self) -> None:
        """Start a background open if not already in flight."""
        if self._stopped or not self._enabled:
            return
        with self._lock:
            if self._session is not None:
                return  # already warm
            if self._opening_thread is not None and self._opening_thread.is_alive():
                return  # already opening
            self._opening_thread = threading.Thread(
                target=self._open_session_blocking,
                name='elevenlabs-pool-replenish',
                daemon=True,
            )
            self._opening_started_at = time.monotonic()
            self._opening_thread.start()

    def _open_session_blocking(self) -> None:
        """Synchronously open + warm one session, store in pool. Background thread."""
        if self._stopped:
            return
        api_key = self._api_key()
        if not api_key:
            return
        keyterms = self._current_keyterms()
        try:
            from engine.stt.elevenlabs_rt import Session
        except Exception as e:
            ConfigManager.console_print(f'ElevenLabsSessionPool: import failed: {e}')
            return

        ready_event = threading.Event()
        ready_ok = {'ok': False}

        def on_ready(ok: bool) -> None:
            ready_ok['ok'] = ok
            ready_event.set()

        try:
            session = Session(api_key, list(keyterms))
            session.start(on_ready=on_ready)
        except Exception as e:
            ConfigManager.console_print(f'ElevenLabsSessionPool: Session.start raised: {e}')
            return

        # 5 s ceiling for the initial open. Larger than the in-record 2 s
        # because this is a background replenish — slow open here doesn't
        # block the user.
        if not ready_event.wait(timeout=5.0):
            ConfigManager.console_print(
                'ElevenLabsSessionPool: session_started not received within 5s; aborting'
            )
            try:
                session.cancel()
            except Exception:
                pass
            return

        if not ready_ok['ok']:
            ConfigManager.console_print('ElevenLabsSessionPool: session failed to open')
            try:
                session.cancel()
            except Exception:
                pass
            return

        # Keepalive every 10 s so the warm session survives long idle gaps
        # between hotkey presses.
        try:
            session.start_keepalive()
        except Exception as e:
            ConfigManager.console_print(f'ElevenLabsSessionPool: start_keepalive raised: {e}')

        elapsed_ms = int((time.monotonic() - self._opening_started_at) * 1000)
        ConfigManager.console_print(
            f'ElevenLabsSessionPool: warm session ready (open took {elapsed_ms}ms)'
        )

        # Stash if we still want it. (Stop may have raced.)
        with self._lock:
            if self._stopped:
                try:
                    session.cancel()
                except Exception:
                    pass
                return
            # Race-safety: another open could've completed first. Drop ours.
            if self._session is not None:
                try:
                    session.cancel()
                except Exception:
                    pass
                return
            self._session = session
            self._session_keyterms = keyterms

    # ---- helpers ----

    @staticmethod
    def _streaming_enabled() -> bool:
        engine = ConfigManager.get_config_value('model_options', 'stt_engine') or 'groq'
        if engine != 'elevenlabs':
            return False
        streaming_enabled = ConfigManager.get_config_value('model_options', 'stt_streaming_mode')
        return streaming_enabled is not False  # default True

    @staticmethod
    def _api_key() -> Optional[str]:
        return os.getenv('ELEVENLABS_API_KEY') or ConfigManager.get_config_value(
            'model_options', 'elevenlabs_api_key'
        )

    @staticmethod
    def _current_keyterms() -> tuple[str, ...]:
        try:
            from transcription import _build_elevenlabs_keyterms
            return tuple(_build_elevenlabs_keyterms())
        except Exception:
            return ()
