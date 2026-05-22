"""Always-warm audio capture service.

Owns one `sounddevice.InputStream` opened at app startup and continuously
filling a ring buffer. On hotkey, `attach_consumer()` returns a FIFO that
starts with the ring's last `preroll_ms` of audio and then receives every
subsequent frame until detached.

Goal: when the hotkey fires, the mic is already capturing — no PortAudio
init, no driver power-up, no AGC settle, no leading-words dropped. The
~200 ms of pre-roll catches the leading phoneme when the user starts
speaking fractionally before the keypress.

Threading:
  - Service is created on the main GUI thread, started + stopped from there.
  - `sounddevice` callback runs on its own internal thread; we push each
    frame into the ring (under lock) and fan out copies to every attached
    consumer's queue.
  - Consumers are drained from `ResultThread.run()` (a QThread), one frame
    at a time, ordered.

NEVER call into QWidgets from here — this module is Qt-agnostic. Bubble
state changes must be marshaled via the existing `_BubbleStateSignal` in
main.py (see CRITICAL note in result_thread.py + main.py).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd

from utils import ConfigManager


_LOG = logging.getLogger(__name__)


class CaptureConsumer:
    """FIFO of PCM16 mono frames drained from the parent service.

    Each frame is a 1-D `np.ndarray` of `int16` of length `frame_size`.
    `get_frame(timeout)` blocks up to `timeout` seconds and returns the
    next frame, or `None` if the consumer is closed (service shut down)
    or the timeout elapsed with no frame available.
    """

    # Soft cap on the consumer queue. With 30 ms frames that's 9 s of
    # pending audio — only triggers if the downstream WS send is
    # catastrophically slow. We drop *newest* on overflow so the existing
    # ordered prefix stays intact (Scribe RT does better with truncated
    # tail than with a hole in the middle).
    _MAX_QUEUE = 300

    def __init__(self, frame_size: int, sample_rate: int):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._frames: deque[np.ndarray] = deque()
        self._data_ready = threading.Event()
        self._closed = False
        self._dropped = 0

    def _push_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            if self._closed:
                return
            if len(self._frames) >= self._MAX_QUEUE:
                self._dropped += 1
                if self._dropped == 1 or self._dropped % 50 == 0:
                    _LOG.warning(
                        f'CaptureConsumer queue full ({self._MAX_QUEUE} frames); '
                        f'dropped {self._dropped} so far'
                    )
                return
            self._frames.append(frame)
        self._data_ready.set()

    def _push_preroll(self, frames: list[np.ndarray]) -> None:
        if not frames:
            return
        with self._lock:
            if self._closed:
                return
            for f in frames:
                if len(self._frames) >= self._MAX_QUEUE:
                    break
                self._frames.append(f)
        self._data_ready.set()

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        end = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._frames:
                    frame = self._frames.popleft()
                    if not self._frames:
                        self._data_ready.clear()
                    return frame
                if self._closed:
                    return None
            remaining = end - time.monotonic()
            if remaining <= 0:
                return None
            self._data_ready.wait(timeout=remaining)
            self._data_ready.clear()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._data_ready.set()

    def is_closed(self) -> bool:
        with self._lock:
            return self._closed


class AudioCaptureService:
    """Always-open InputStream + ring buffer. Singleton in practice.

    Lifecycle:
        service = AudioCaptureService(sample_rate=16000, device=...)
        service.start()                       # at app launch
        consumer = service.attach_consumer(preroll_ms=200)   # on hotkey
        while ...:
            frame = consumer.get_frame()
        service.detach_consumer(consumer)     # on stop_recording
        service.stop()                        # at app exit
    """

    DEFAULT_FRAME_DURATION_MS = 30           # matches existing VAD frame size
    DEFAULT_RING_SECONDS = 1.0               # 1 s of audio = ~33 frames @ 30ms

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
        ring_seconds: float = DEFAULT_RING_SECONDS,
        device: Optional[str | int] = None,
    ):
        self.sample_rate = int(sample_rate)
        self.frame_duration_ms = int(frame_duration_ms)
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000.0)
        self.ring_max_frames = max(1, int(ring_seconds * self.sample_rate / self.frame_size))
        self.device = device

        self._stream: Optional[sd.InputStream] = None
        self._ring: deque[np.ndarray] = deque(maxlen=self.ring_max_frames)
        self._ring_lock = threading.Lock()
        self._consumers: list[CaptureConsumer] = []
        self._consumers_lock = threading.Lock()
        self._alive = False
        self._failed_reason: Optional[str] = None

    # ---- lifecycle ----

    def start(self) -> bool:
        """Open the InputStream. Returns True on success, False if the audio
        device couldn't be opened (no mic, exclusive lock by another app,
        unsupported sample rate). Caller should fall back to per-hotkey
        opening when this returns False."""
        if self._alive:
            return True
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=self.frame_size,
                device=self.device,
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as e:
            self._failed_reason = str(e)
            self._stream = None
            self._alive = False
            ConfigManager.console_print(
                f'AudioCaptureService open failed: {e}; '
                f'falling back to per-hotkey mic open'
            )
            return False
        self._alive = True
        ConfigManager.console_print(
            f'AudioCaptureService started (sr={self.sample_rate}, '
            f'frame={self.frame_size} samples, ring={self.ring_max_frames} frames)'
        )
        return True

    def stop(self) -> None:
        self._alive = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                _LOG.warning(f'InputStream close raised: {e}')
            self._stream = None
        with self._consumers_lock:
            consumers = list(self._consumers)
            self._consumers.clear()
        for c in consumers:
            c.close()
        with self._ring_lock:
            self._ring.clear()

    def is_alive(self) -> bool:
        return self._alive and self._stream is not None

    def failed_reason(self) -> Optional[str]:
        return self._failed_reason

    # ---- consumer attach / detach ----

    def attach_consumer(self, preroll_ms: int = 200) -> CaptureConsumer:
        """Return a fresh CaptureConsumer seeded with the ring's last
        `preroll_ms` of captured audio. Subsequent frames go to all
        attached consumers."""
        c = CaptureConsumer(self.frame_size, self.sample_rate)
        if preroll_ms > 0:
            preroll_frame_count = max(
                1, int(preroll_ms / 1000.0 * self.sample_rate / self.frame_size)
            )
            with self._ring_lock:
                # Last N frames; deque slicing requires list conversion.
                if self._ring:
                    snapshot = list(self._ring)[-preroll_frame_count:]
                else:
                    snapshot = []
            c._push_preroll(snapshot)
        with self._consumers_lock:
            self._consumers.append(c)
        return c

    def detach_consumer(self, consumer: CaptureConsumer) -> None:
        with self._consumers_lock:
            try:
                self._consumers.remove(consumer)
            except ValueError:
                pass
        consumer.close()

    # ---- internals ----

    def _on_audio(self, indata, frames, time_info, status) -> None:
        """sounddevice callback (its own thread). Frame is shape (frame_size, 1)
        int16 because we asked for channels=1, dtype='int16', blocksize=frame_size."""
        if status:
            # Status flags include input_overflow / input_underflow. We log
            # but don't auto-restart on v1 — if it becomes persistent the
            # user can restart Whisper PC. (Mobile handles device-change events
            # explicitly; v2 enhancement on PC.)
            _LOG.warning(f'audio cb status: {status}')

        # Flatten to 1-D int16. .copy() because indata is a numpy view into a
        # PortAudio buffer that gets reused after this callback returns.
        try:
            frame = indata[:, 0].copy()
        except Exception as e:
            _LOG.warning(f'audio cb flatten failed: {e}')
            return

        with self._ring_lock:
            self._ring.append(frame)

        with self._consumers_lock:
            # Each consumer receives its own copy. With one active consumer
            # this is one extra memcpy of 480 int16s = 960 bytes per 30 ms —
            # ~32 KB/s. Negligible.
            for c in self._consumers:
                c._push_frame(frame.copy())
