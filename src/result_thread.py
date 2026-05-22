import datetime
import os
import time
import traceback
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import wave
import webrtcvad
from PyQt5.QtCore import QThread, QMutex, pyqtSignal
from collections import deque
from threading import Event

from transcription import transcribe, transcribe_streaming_result, TranscriptionAPIError
from utils import ConfigManager


class ResultThread(QThread):
    """
    A thread class for handling audio recording, transcription, and result processing.

    This class manages the entire process of:
    1. Recording audio from the microphone
    2. Detecting speech and silence
    3. Saving the recorded audio as numpy array
    4. Transcribing the audio
    5. Emitting the transcription result

    Signals:
        statusSignal: Emits the current status of the thread (e.g., 'recording', 'transcribing', 'idle')
        resultSignal: Emits the transcription result
        failedSignal: Emits (audio_path, error_reason) when the API fails after recording
    """

    statusSignal = pyqtSignal(str)
    resultSignal = pyqtSignal(str)
    failedSignal = pyqtSignal(str, str)

    def __init__(self, local_model=None, audio_capture_service=None, session_pool=None):
        """
        Initialize the ResultThread.

        :param local_model: Local transcription model (if applicable)
        :param audio_capture_service: Optional always-warm AudioCaptureService.
            When alive, capture skips the per-hotkey sd.InputStream open and
            drains frames from a consumer attached at run() start (pre-roll
            included). Falls back to per-hotkey open if None / not alive.
        :param session_pool: Optional ElevenLabsSessionPool. When acquired,
            replaces the synchronous in-record _open_streaming_session_if_enabled
            (~300-1500 ms WS handshake) with zero-latency hand-off of a warm
            session. Falls back to in-record open if None / acquire returns None.
        """
        super().__init__()
        self.local_model = local_model
        self._audio_capture_service = audio_capture_service
        self._session_pool = session_pool
        self.is_recording = False
        self.is_running = True
        # Pause flag — when True, the audio capture loop discards new frames
        # so the resulting recording omits the paused span. The InputStream
        # itself keeps running (avoids re-init latency); we just drop frames.
        self.is_paused = False
        self.sample_rate = None
        self.mutex = QMutex()
        # v0.3.2 PC: streaming-during-recording session. Open before recording
        # starts (when stt_engine=elevenlabs AND stt_streaming_mode=true), fed
        # per-frame in _record_audio, committed in run() after stop. None when
        # streaming is disabled or for the Groq path.
        self._stream_session = None
        self._stream_failed_reason: str | None = None

    def stop_recording(self):
        """Stop the current recording session."""
        self.mutex.lock()
        self.is_recording = False
        # If we were paused, transcription must still proceed on whatever
        # audio was captured before the pause — clearing the flag ensures
        # the recording loop exits and the captured frames go to transcribe.
        self.is_paused = False
        self.mutex.unlock()

    def pause_recording(self):
        """Pause the audio-capture loop. Frames received while paused are
        discarded so the resulting recording skips over the paused span."""
        self.mutex.lock()
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            self.mutex.unlock()
            self.statusSignal.emit('paused')
        else:
            self.mutex.unlock()

    def resume_recording(self):
        """Resume capture after a pause."""
        self.mutex.lock()
        if self.is_recording and self.is_paused:
            self.is_paused = False
            self.mutex.unlock()
            self.statusSignal.emit('recording')
        else:
            self.mutex.unlock()

    def toggle_pause(self):
        """Single-shortcut helper: pause if recording, resume if paused."""
        self.mutex.lock()
        was_paused = self.is_paused
        is_rec = self.is_recording
        self.mutex.unlock()
        if not is_rec:
            return
        if was_paused:
            self.resume_recording()
        else:
            self.pause_recording()

    def stop(self):
        """Stop the entire thread execution."""
        self.mutex.lock()
        self.is_running = False
        self.mutex.unlock()
        self.statusSignal.emit('idle')
        self.wait()

    def run(self):
        """Main execution method for the thread."""
        audio_data = None
        consumer = None
        try:
            if not self.is_running:
                return

            self.mutex.lock()
            self.is_recording = True
            self.mutex.unlock()

            # v0.4 PC: prefer the always-warm path — the audio capture service
            # has been recording into a ring since app startup, so there's no
            # mic-open / AGC-settle delay here. We just attach a consumer
            # which seeds with `preroll_ms` of pre-hotkey audio (catching the
            # leading phoneme when the user starts speaking on the keypress).
            #
            # If the service isn't alive (always_warm_mic=false, or open
            # failed at startup), fall through to the legacy in-record open
            # in _record_audio.
            use_warm_capture = (
                self._audio_capture_service is not None
                and self._audio_capture_service.is_alive()
            )
            if use_warm_capture:
                preroll_ms = int(
                    ConfigManager.get_config_value('recording_options', 'preroll_ms') or 200
                )
                consumer = self._audio_capture_service.attach_consumer(preroll_ms=preroll_ms)
                ConfigManager.console_print(
                    f'Attached warm-capture consumer (preroll={preroll_ms}ms)'
                )

            # ElevenLabs session: prefer the pre-opened warm session from the
            # pool (zero latency). Fall back to in-record open (the legacy
            # ~300-1500 ms blocking handshake) when the pool isn't enabled or
            # has no warm session ready.
            self._acquire_warm_session_or_open()

            self.statusSignal.emit('recording')
            ConfigManager.console_print('Recording...')
            if consumer is not None:
                audio_data = self._record_audio_from_consumer(consumer)
            else:
                audio_data = self._record_audio()

            if not self.is_running:
                self._cancel_streaming_session()
                return

            if audio_data is None:
                self._cancel_streaming_session()
                self.statusSignal.emit('idle')
                return

            self.statusSignal.emit('transcribing')
            ConfigManager.console_print('Transcribing...')

            start_time = time.time()
            streaming_text = self._commit_streaming_session(audio_data)
            if streaming_text is not None:
                # ElevenLabs streaming delivered text; just polish.
                result = transcribe_streaming_result(streaming_text)
            else:
                # No streaming, or streaming failed. Force Groq when streaming
                # was attempted-and-failed so we don't re-hit ElevenLabs burst
                # on the same audio that just killed the streaming session.
                # (Mirrors mobile v0.3.4 forceGroq path.)
                streaming_failed = self._stream_failed_reason is not None
                self._stream_failed_reason = None
                result = transcribe(audio_data, self.local_model, force_groq=streaming_failed)
            end_time = time.time()

            transcription_time = end_time - start_time
            ConfigManager.console_print(f'Transcription completed in {transcription_time:.2f} seconds. Post-processed line: {result}')

            if not self.is_running:
                return

            if not result.strip():
                # Whisper produced nothing usable (silence, hallucination filter,
                # RMS skip). Surface "Nothing transcribable detected" via the
                # status overlay and skip the paste, but still emit an empty
                # result so continuous-mode / key listener re-arming runs.
                self.statusSignal.emit('no_speech')
                self.resultSignal.emit('')
                return

            self.statusSignal.emit('idle')
            try:
                from dict_diag import dd
                dd('result_thread.emit', result)
            except Exception:
                pass
            self.resultSignal.emit(result)

        except TranscriptionAPIError as e:
            traceback.print_exc()
            ConfigManager.console_print(f'Transcription API failure: {e.reason}')
            if audio_data is not None and audio_data.size > 0:
                self._persist_failed_recording(audio_data, e.reason)
            self.statusSignal.emit('error')
            # Emit empty result so the existing post-completion flow runs
            # (key listener restart, continuous-mode re-arm).
            self.resultSignal.emit('')
        except Exception as e:
            traceback.print_exc()
            self.statusSignal.emit('error')
            self.resultSignal.emit('')
        finally:
            self.stop_recording()
            # Make sure no streaming session is left dangling — _commit_*
            # already clears self._stream_session on success/failure, but the
            # exception paths above may have left it set.
            self._cancel_streaming_session()
            # Release the warm-capture consumer (no-op if None or in legacy
            # path). The InputStream itself stays open — that's the point.
            if consumer is not None and self._audio_capture_service is not None:
                self._audio_capture_service.detach_consumer(consumer)

    def _persist_failed_recording(self, audio_data, reason):
        """Save audio to failed/<timestamp>.wav and append an entry to failed_log.txt."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        failed_dir = os.path.join(project_root, 'failed')
        try:
            os.makedirs(failed_dir, exist_ok=True)
        except Exception as e:
            ConfigManager.console_print(f'Could not create failed/ dir: {e}')
            return

        now = datetime.datetime.now()
        ts_human = now.strftime('%Y-%m-%d %H:%M:%S')
        fname = now.strftime('%Y-%m-%d_%H-%M-%S_%f') + '.wav'
        audio_path = os.path.join(failed_dir, fname)

        try:
            sf.write(audio_path, audio_data, self.sample_rate or 16000)
        except Exception as e:
            ConfigManager.console_print(f'Could not save failed audio: {e}')
            return

        rel_path = os.path.relpath(audio_path, project_root).replace(os.sep, '/')
        log_path = os.path.join(project_root, 'failed_log.txt')
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f'[{ts_human}]\n  AUDIO:    {rel_path}\n  ERROR:    {reason}\n\n')
        except Exception as e:
            ConfigManager.console_print(f'Could not write failed_log.txt: {e}')

        self.failedSignal.emit(audio_path, reason)

    # ---- v0.3.2 PC: streaming-during-recording helpers ----

    def _acquire_warm_session_or_open(self) -> None:
        """v0.4 PC: zero-latency session acquisition.

        Tries the ElevenLabsSessionPool first — if a warm session is
        available, hand-off is instant. Otherwise falls back to the legacy
        in-record open (blocks up to 2s on the WS handshake) for safety.

        Sets self._stream_session to a ready Session, or None.
        """
        self._stream_session = None
        self._stream_failed_reason = None

        # Pool fast-path — only attempted when the pool is enabled. Returns
        # an already-ready Session with keepalive running.
        pool = self._session_pool
        if pool is not None and pool.is_enabled():
            warm = pool.acquire()
            if warm is not None:
                self._stream_session = warm
                ConfigManager.console_print(
                    'Acquired warm streaming session from pool (0ms WS handshake)'
                )
                return
            ConfigManager.console_print(
                'Session pool had no warm session; falling back to in-record open'
            )

        # Legacy slow-path: open synchronously in this thread. Same code as
        # before the pool existed.
        self._open_streaming_session_if_enabled()

    def _open_streaming_session_if_enabled(self) -> None:
        """Open an ElevenLabs RT session if streaming mode is configured.

        Caller (run) calls this before _record_audio so frames can flow as
        soon as capture starts. Failure to open is silently tolerated — we
        fall back to the burst path via transcribe(). Sets self._stream_session
        to a ready Session, or None.
        """
        self._stream_session = None
        self._stream_failed_reason = None

        stt_engine = ConfigManager.get_config_value('model_options', 'stt_engine') or 'groq'
        if stt_engine != 'elevenlabs':
            return
        streaming_enabled = ConfigManager.get_config_value('model_options', 'stt_streaming_mode')
        if streaming_enabled is False:  # default True
            return
        api_key = os.getenv('ELEVENLABS_API_KEY') or ConfigManager.get_config_value('model_options', 'elevenlabs_api_key')
        if not api_key:
            ConfigManager.console_print('ElevenLabs key not set; streaming session not opened (will use burst on stop)')
            return

        try:
            from engine.stt.elevenlabs_rt import Session
            from transcription import _build_elevenlabs_keyterms
            keyterms = _build_elevenlabs_keyterms()
        except Exception as e:
            ConfigManager.console_print(f'Streaming session prep failed: {e}; will use burst on stop')
            return

        from threading import Event as _Event
        ready_event = _Event()

        def on_ready(ok: bool) -> None:
            if ok:
                ready_event.set()
            else:
                # Mirrors mobile v0.3.6: session.onFailure fires onReadyCallback(false)
                # for mid-recording deaths too. Mark a reason so the eventual
                # stop path falls through to burst/Groq instead of trying to
                # commit a dead session.
                self._stream_failed_reason = 'ElevenLabs streaming session ended'
                ready_event.set()  # unblock the caller

        session = Session(api_key, keyterms)
        session.start(on_ready=on_ready)
        # Wait briefly for session_started. If it doesn't come, we still let
        # the recording proceed — _record_audio will treat the session as dead
        # and we'll fall back at stop time.
        if not ready_event.wait(timeout=2.0):
            ConfigManager.console_print('Streaming session not ready within 2s; falling back to burst on stop')
            session.cancel()
            return
        if self._stream_failed_reason:
            ConfigManager.console_print('Streaming session failed to open; falling back to burst on stop')
            session.cancel()
            return

        self._stream_session = session
        ConfigManager.console_print(f'Streaming session opened (keyterms={len(keyterms)})')

    def _feed_streaming_session(self, pcm_frame: np.ndarray) -> None:
        """Forward one captured PCM frame to the streaming session, if any."""
        session = self._stream_session
        if session is None:
            return
        if not session.is_ready():
            # Session died during recording — note the reason once and stop
            # trying to feed it. Burst fallback will pick up at stop.
            if not self._stream_failed_reason:
                self._stream_failed_reason = 'ElevenLabs streaming session died mid-recording'
                ConfigManager.console_print('Streaming session died mid-recording; switching to burst-on-stop')
            self._stream_session = None
            return
        try:
            session.send_audio_chunk(pcm_frame.tobytes(), commit=False)
        except Exception as e:
            ConfigManager.console_print(f'Streaming send failed: {e}; switching to burst-on-stop')
            self._stream_failed_reason = f'send failed: {e}'
            self._stream_session = None

    def _commit_streaming_session(self, full_audio: np.ndarray) -> str | None:
        """Commit the streaming session and wait for the final transcript.

        Returns the finalised text string when streaming delivered usable
        output. Returns None when streaming wasn't active or it failed —
        caller should fall through to the burst path on `full_audio`.
        """
        session = self._stream_session
        if session is None:
            return None
        self._stream_session = None

        from threading import Event as _Event
        done_event = _Event()
        result_holder: dict = {}

        def on_result(r: dict) -> None:
            result_holder.update(r)
            done_event.set()

        try:
            session.commit(on_result)
        except Exception as e:
            ConfigManager.console_print(f'Streaming commit raised: {e}; falling back to burst')
            return None

        # Mobile uses a 5s in-session watchdog plus a 7s service-level safety
        # net. Mirror that here: wait up to 7s for the final transcript.
        if not done_event.wait(timeout=7.0):
            ConfigManager.console_print('Streaming commit timed out (>7s); falling back to burst')
            try:
                session.cancel()
            except Exception:
                pass
            return None

        if result_holder.get('error'):
            ConfigManager.console_print(f'Streaming commit failed: {result_holder["error"]}; falling back to burst')
            return None

        text = (result_holder.get('text') or '').strip()
        if not text:
            ConfigManager.console_print('Streaming commit returned empty text; falling back to burst')
            return None

        ConfigManager.console_print(f'Streaming delivered {len(text)} chars; skipping burst STT')
        return text

    def _cancel_streaming_session(self) -> None:
        """Abort streaming without committing (user cancelled / empty audio)."""
        session = self._stream_session
        if session is not None:
            try:
                session.cancel()
            except Exception:
                pass
        self._stream_session = None

    # ---- recording loop ----

    def _record_audio_from_consumer(self, consumer):
        """Drain frames from an AudioCaptureService consumer.

        This is the warm-path counterpart to `_record_audio()`. The InputStream
        is already open (since app startup); we only consume frames from the
        consumer's queue. The first frames returned are the pre-roll snapshot
        (last ~200 ms of audio captured BEFORE the hotkey landed), which is
        what catches the leading phoneme when the user starts speaking on the
        keypress.

        Same VAD / silence-detection / dual-write-to-streaming-session logic
        as `_record_audio()` — only the source of frames differs.
        """
        recording_options = ConfigManager.get_config_section('recording_options')
        self.sample_rate = consumer.sample_rate
        frame_size = consumer.frame_size
        frame_duration_ms = frame_size * 1000.0 / consumer.sample_rate
        silence_duration_ms = recording_options.get('silence_duration') or 900
        silence_frames = int(silence_duration_ms / frame_duration_ms)

        # 150 ms VAD-warmup skip — same as legacy path. Doesn't drop the
        # audio itself, just defers VAD's silence-counter so the keypress
        # click doesn't accidentally trigger "speech detected".
        initial_frames_to_skip = int(0.15 * consumer.sample_rate / frame_size)

        recording_mode = recording_options.get('recording_mode') or 'continuous'
        vad = None
        if recording_mode in ('voice_activity_detection', 'continuous'):
            vad = webrtcvad.Vad(2)
            speech_detected = False
            silent_frame_count = 0

        recording = []

        # Tight poll interval — when stop_recording() sets is_recording=False,
        # the next iteration must exit promptly so the user-perceived stop is
        # immediate. 50 ms is fine; with 30 ms frames there's almost always a
        # frame waiting and the timeout rarely fires.
        while self.is_running and self.is_recording:
            frame = consumer.get_frame(timeout=0.05)
            if frame is None:
                if consumer.is_closed():
                    ConfigManager.console_print(
                        'Capture consumer closed mid-recording; stopping'
                    )
                    break
                continue  # timeout — loop, check is_recording

            if self.is_paused:
                continue  # discard frame; ring keeps filling in the background

            recording.extend(frame)

            # Dual-write — capture stays in `recording` for the burst fallback /
            # failed-recording persistence AND streams to ElevenLabs RT in real
            # time when the session is open.
            self._feed_streaming_session(frame)

            if initial_frames_to_skip > 0:
                initial_frames_to_skip -= 1
                continue

            if vad:
                if vad.is_speech(frame.tobytes(), self.sample_rate):
                    silent_frame_count = 0
                    if not speech_detected:
                        ConfigManager.console_print("Speech detected.")
                        speech_detected = True
                else:
                    silent_frame_count += 1

                if speech_detected and silent_frame_count > silence_frames:
                    break

        audio_data = np.array(recording, dtype=np.int16)
        duration = len(audio_data) / self.sample_rate

        ConfigManager.console_print(
            f'Recording finished (warm-capture). Size: {audio_data.size} samples, '
            f'Duration: {duration:.2f} seconds'
        )

        min_duration_ms = recording_options.get('min_duration') or 100
        if (duration * 1000) < min_duration_ms:
            ConfigManager.console_print('Discarded due to being too short.')
            return None

        return audio_data

    def _record_audio(self):
        """
        Record audio from the microphone and save it to a temporary file.

        :return: numpy array of audio data, or None if the recording is too short
        """
        recording_options = ConfigManager.get_config_section('recording_options')
        self.sample_rate = recording_options.get('sample_rate') or 16000
        frame_duration_ms = 30  # 30ms frame duration for WebRTC VAD
        frame_size = int(self.sample_rate * (frame_duration_ms / 1000.0))
        silence_duration_ms = recording_options.get('silence_duration') or 900
        silence_frames = int(silence_duration_ms / frame_duration_ms)

        # 150ms delay before starting VAD to avoid mistaking the sound of key pressing for voice
        initial_frames_to_skip = int(0.15 * self.sample_rate / frame_size)

        # Create VAD only for recording modes that use it
        recording_mode = recording_options.get('recording_mode') or 'continuous'
        vad = None
        if recording_mode in ('voice_activity_detection', 'continuous'):
            vad = webrtcvad.Vad(2)  # VAD aggressiveness: 0 to 3, 3 being the most aggressive
            speech_detected = False
            silent_frame_count = 0

        audio_buffer = deque(maxlen=frame_size)
        recording = []

        data_ready = Event()

        def audio_callback(indata, frames, time, status):
            if status:
                ConfigManager.console_print(f"Audio callback status: {status}")
            audio_buffer.extend(indata[:, 0])
            data_ready.set()

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16',
                            blocksize=frame_size, device=recording_options.get('sound_device'),
                            callback=audio_callback):
            while self.is_running and self.is_recording:
                data_ready.wait()
                data_ready.clear()

                if len(audio_buffer) < frame_size:
                    continue

                # PAUSED: discard the buffered frames so the recording skips
                # the paused span. The InputStream callback keeps firing in
                # the background; we just don't carry the audio forward.
                if self.is_paused:
                    audio_buffer.clear()
                    continue

                # Save frame
                frame = np.array(list(audio_buffer), dtype=np.int16)
                audio_buffer.clear()
                recording.extend(frame)

                # v0.3.2: dual-write — capture stays in `recording` (for the
                # burst fallback / failed-recording persistence) AND streams
                # to ElevenLabs RT in real time when the session is open.
                self._feed_streaming_session(frame)

                # Avoid trying to detect voice in initial frames
                if initial_frames_to_skip > 0:
                    initial_frames_to_skip -= 1
                    continue

                if vad:
                    if vad.is_speech(frame.tobytes(), self.sample_rate):
                        silent_frame_count = 0
                        if not speech_detected:
                            ConfigManager.console_print("Speech detected.")
                            speech_detected = True
                    else:
                        silent_frame_count += 1

                    if speech_detected and silent_frame_count > silence_frames:
                        break

        audio_data = np.array(recording, dtype=np.int16)
        duration = len(audio_data) / self.sample_rate

        ConfigManager.console_print(f'Recording finished. Size: {audio_data.size} samples, Duration: {duration:.2f} seconds')

        min_duration_ms = recording_options.get('min_duration') or 100

        if (duration * 1000) < min_duration_ms:
            ConfigManager.console_print(f'Discarded due to being too short.')
            return None

        return audio_data
