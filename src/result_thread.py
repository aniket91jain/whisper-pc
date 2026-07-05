import datetime
import os
import threading
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


def prune_recordings_archive(project_root, retention_days=7, max_files=2000):
    """Trim the save-first recordings/ archive at startup.

    Deletes .wav files older than `retention_days`, then trims what remains to
    the newest `max_files`. This is the ONE place we hard-delete audio: the
    archive is a self-cleaning safety cache, so aged files are meant to expire.
    Safe no-op if the folder doesn't exist. Never touches failed/ or any log."""
    rec_dir = os.path.join(project_root, 'recordings')
    if not os.path.isdir(rec_dir):
        return
    cutoff = time.time() - retention_days * 86400
    survivors = []
    removed = 0
    try:
        entries = list(os.scandir(rec_dir))
    except OSError:
        return
    for entry in entries:
        if not entry.name.lower().endswith('.wav'):
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            try:
                os.remove(entry.path)
                removed += 1
            except OSError:
                pass
        else:
            survivors.append((mtime, entry.path))
    # Count cap: keep the newest max_files, delete the rest.
    if len(survivors) > max_files:
        survivors.sort(reverse=True)  # newest first
        for _mtime, path in survivors[max_files:]:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    if removed:
        ConfigManager.console_print(
            f'Pruned {removed} old recording(s) from the archive '
            f'(kept files ≤{retention_days} days old, newest {max_files}).'
        )


# Phase-timing diagnostics file. pythonw.exe has no console, so console_print()
# output goes to the void — when a transcription hangs, there's no way to tell
# which phase wedged. This sink writes one short line per phase boundary to a
# dedicated log so the next stuck attempt is debuggable from the on-disk file.
_PHASE_DIAG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'transcribe_diag.log',
)
# Rotate at ~512 KB so the file doesn't grow unbounded; truncate to keep the
# tail (the bit we'd actually want to read after a hang).
_PHASE_DIAG_MAX_BYTES = 512 * 1024


class _PhaseDiagWriter:
    """Per-ResultThread instance writer. Stamps each phase with monotonic
    elapsed-since-construction so the gaps between phases are obvious at a
    glance. Failures are swallowed — diagnostics must never break dictation."""

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        # Single header line per run, so an audit reader can see where one
        # dictation starts and the previous one ends.
        try:
            self._truncate_if_huge()
            with open(_PHASE_DIAG_PATH, 'a', encoding='utf-8') as f:
                f.write(
                    f'--- {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]} '
                    f'thread_id={id(self):x} ---\n'
                )
        except Exception:
            pass

    def mark(self, phase: str, **fields) -> None:
        try:
            elapsed_ms = int((time.monotonic() - self._t0) * 1000)
            extra = ''
            if fields:
                extra = ' ' + ' '.join(f'{k}={v!r}' for k, v in fields.items())
            with open(_PHASE_DIAG_PATH, 'a', encoding='utf-8') as f:
                f.write(f'+{elapsed_ms:>7}ms  {phase}{extra}\n')
        except Exception:
            pass

    @staticmethod
    def _truncate_if_huge() -> None:
        try:
            if os.path.exists(_PHASE_DIAG_PATH) and \
                    os.path.getsize(_PHASE_DIAG_PATH) > _PHASE_DIAG_MAX_BYTES:
                # Keep the last quarter — the bit most likely to contain the
                # hang we want to debug.
                with open(_PHASE_DIAG_PATH, 'rb') as f:
                    f.seek(-_PHASE_DIAG_MAX_BYTES // 4, os.SEEK_END)
                    tail = f.read()
                with open(_PHASE_DIAG_PATH, 'wb') as f:
                    f.write(b'--- [truncated] ---\n')
                    f.write(tail)
        except Exception:
            pass


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
    # Engine label (e.g. 'elevenlabs-stream', 'groq+llama') of the just-finished
    # dictation, emitted just before the result so the GUI can flash a brief
    # "transcribed by X" toast next to the recording bubble.
    engineSignal = pyqtSignal(str)

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
        # Cancel flag — set by cancel() when the user hits the × badge on the
        # bubble. Checked at every yield point in run() so the captured audio
        # is discarded, no transcription is committed, no paste fires, and no
        # failed-audio is persisted. The thread is allowed to wind down on its
        # own — main.py disconnects the result/failed signals so any late
        # emissions are ignored. Non-blocking by design (a hard wait() on a
        # stuck network call would freeze the GUI).
        self.is_cancelled = False
        self.sample_rate = None
        # Save-first archive: every finished recording is written to recordings/
        # BEFORE transcription (see _archive_recording). These hold the abs/rel
        # path of the current recording so _persist_failed_recording can point
        # the failed log at the already-saved file instead of writing a 2nd copy.
        self._archive_abs = None
        self._archive_rel = None
        # Tier 2 (2026-07-06): frames captured in the tiny window on resume
        # before the fresh streaming session is assigned are buffered here and
        # flushed into that session the instant it appears — so a resume never
        # forces a Groq burst (ElevenLabs catches up at faster-than-real-time).
        # Reset per-dictation in run().
        self._resume_pending = []
        self.mutex = QMutex()
        # v0.3.2 PC: streaming-during-recording session. Open before recording
        # starts (when stt_engine=elevenlabs AND stt_streaming_mode=true), fed
        # per-frame in _record_audio, committed in run() after stop. None when
        # streaming is disabled or for the Groq path.
        self._stream_session = None
        self._stream_failed_reason: str | None = None
        # v0.4 PC: pause-segmented streaming. A long pause kills ElevenLabs'
        # WebSocket (server idle/session-time/queue limits — see
        # engine/stt/elevenlabs_rt.py). Rather than hold one socket open across
        # the pause (which always fails on hour-long gaps), each spoken span is
        # its own short-lived session: commit-on-pause finalizes the span and
        # stashes its text here; resume opens a fresh session; stop stitches the
        # spans together. `_streaming_in_use` records that this dictation took
        # the streaming path at all (so resume knows to re-open). Any span that
        # fails to stream sets `_stream_failed_reason`, which makes stop fall
        # back to a single Groq burst over the full (speech-only) audio buffer.
        self._stream_segments: list[str] = []
        self._streaming_in_use = False
        self._pause_commit_thread: threading.Thread | None = None
        self._stream_lock = threading.Lock()
        # Phase-diag writer for the current dictation (set in run()). pause/
        # resume stamp it so any future post-resume drop is visible in
        # transcribe_diag.log without needing to reproduce under a console.
        self._diag = None

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
        discarded so the resulting recording skips over the paused span.

        v0.4: also finalizes the current ElevenLabs streaming span. The span's
        text is committed and stashed so a long pause (which would otherwise
        kill the idle WebSocket) cannot lose it; resume opens a fresh session.
        """
        self.mutex.lock()
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            self.mutex.unlock()
            if self._diag is not None:
                self._diag.mark('pause')
            self.statusSignal.emit('paused')
            self._rotate_streaming_session_on_pause()
        else:
            self.mutex.unlock()

    def resume_recording(self):
        """Resume capture after a pause.

        v0.4.4: reopen the streaming session BEFORE clearing is_paused, so
        _stream_session is already assigned by the time the capture loop resumes
        feeding frames. This closes the "resume gap" that forced a Groq burst on
        almost every pause: previously is_paused was cleared first (v0.4.1), so
        every frame captured during the new connection's handshake (up to ~1.5s)
        found _stream_session = None and tripped _feed_streaming_session's
        "no live stream" path, marking the whole dictation for a full Groq burst.

        The reopen is now NON-BLOCKING — _reopen_streaming_session_on_resume
        assigns a session that buffers frames internally until its WS handshake
        completes (Session._pending), and only takes a pool session if one is
        instantly ready (no join). So doing it while still paused costs <1ms and
        drops no audio (the capture loop discards frames while paused anyway).

        This does NOT reintroduce the v0.4.1 data-loss bug (where a *blocking*
        reopen-first stalled capture and dropped post-resume audio): is_paused is
        cleared unconditionally right after the reopen — even if it raised — so
        capture always resumes into `recording` (lossless), and the reopen no
        longer blocks. If the reopen genuinely fails, _stream_failed_reason makes
        stop burst the full buffer, which still holds every post-resume frame.
        """
        self.mutex.lock()
        if not (self.is_recording and self.is_paused):
            self.mutex.unlock()
            return
        self.mutex.unlock()
        # Reopen while still paused (non-blocking) so the session is live-or-
        # connecting before capture feeds it — no frame ever sees a None session.
        try:
            self._reopen_streaming_session_on_resume()
        except Exception as e:
            ConfigManager.console_print(
                f'Resume reopen raised ({e}); capture continues into the buffer, '
                f'will burst full audio on stop'
            )
            self._stream_failed_reason = (
                self._stream_failed_reason or f'resume reopen error: {e}'
            )
        # Clear is_paused LAST and unconditionally so capture always resumes.
        self.mutex.lock()
        self.is_paused = False
        self.mutex.unlock()
        if self._diag is not None:
            self._diag.mark('resume')
        self.statusSignal.emit('recording')

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

    def cancel(self):
        """User pressed × on the bubble — abort recording and any in-flight
        STT call. Discards captured audio and emits no result.

        Non-blocking: this returns immediately so the GUI thread is free to
        hide the bubble even if the worker is wedged inside a slow Groq
        upload. The worker thread itself is left running and checks the
        is_cancelled flag at every yield point. Main wires this to disconnect
        the resultSignal/failedSignal so any late emit from the dying thread
        does nothing.

        Side effects:
          - Sets is_cancelled, clears is_recording/is_paused.
          - Hard-aborts any open ElevenLabs streaming session (closes the WS
            so it stops uploading audio bytes the user no longer wants sent).
        """
        self.mutex.lock()
        self.is_cancelled = True
        self.is_recording = False
        self.is_paused = False
        self.mutex.unlock()
        # Closing the WS here is the actually-effective abort: it short-
        # circuits both the streaming-commit path (forces an error so the
        # 7s wait returns immediately) and the burst path (the WS is the
        # transport). For the Groq path there's nothing comparable — the
        # openai SDK doesn't expose mid-call cancellation, so that path
        # finishes naturally and the result is discarded by the is_cancelled
        # check in run().
        self._cancel_streaming_session()
        self.statusSignal.emit('cancel')

    def run(self):
        """Main execution method for the thread."""
        audio_data = None
        consumer = None
        diag = _PhaseDiagWriter()
        # Expose the writer to pause_recording / resume_recording (which run on
        # the GUI thread) so pause/resume transitions land in the same diag log.
        self._diag = diag
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
            #
            # Reset the per-dictation pause-segment state here (NOT inside
            # _acquire_warm_session_or_open — that also runs on resume, which
            # must preserve segments + any failure from earlier spans).
            self._stream_segments = []
            self._streaming_in_use = False
            self._pause_commit_thread = None
            self._stream_failed_reason = None
            self._resume_pending = []
            self._acquire_warm_session_or_open()

            self.statusSignal.emit('recording')
            ConfigManager.console_print('Recording...')
            diag.mark('record_start', warm=use_warm_capture,
                      has_stream=self._stream_session is not None)
            if consumer is not None:
                audio_data = self._record_audio_from_consumer(consumer)
            else:
                audio_data = self._record_audio()
            audio_samples = int(audio_data.size) if audio_data is not None else 0
            diag.mark('record_end', samples=audio_samples)

            if self.is_cancelled:
                ConfigManager.console_print('Cancelled during recording; discarding audio')
                diag.mark('cancelled', stage='post_record')
                self._cancel_streaming_session()
                return

            if not self.is_running:
                self._cancel_streaming_session()
                return

            if audio_data is None:
                self._cancel_streaming_session()
                self.statusSignal.emit('idle')
                return

            # Save-first: persist the raw recording to the rolling archive
            # BEFORE any transcription is attempted. This is the safety net that
            # makes EVERY downstream failure (empty result, network drop, queue
            # overflow, resume-gap burst, even an app crash) recoverable — the
            # audio is on disk before anything can go wrong. Pruned by age/count
            # at startup (prune_recordings_archive).
            self._archive_abs, self._archive_rel = self._archive_recording(audio_data)

            self.statusSignal.emit('transcribing')
            ConfigManager.console_print('Transcribing...')
            diag.mark('transcribe_start')

            start_time = time.time()
            diag.mark('stream_commit_start')
            streaming_text = self._commit_streaming_session(audio_data)
            diag.mark('stream_commit_end',
                      delivered=streaming_text is not None,
                      stream_failed_reason=self._stream_failed_reason)

            if self.is_cancelled:
                ConfigManager.console_print('Cancelled during transcription; discarding result')
                diag.mark('cancelled', stage='post_stream_commit')
                return

            recovery_burst = False
            if streaming_text is not None:
                # ElevenLabs streaming delivered text; just polish.
                diag.mark('polish_streaming_start')
                result = transcribe_streaming_result(streaming_text)
                diag.mark('polish_streaming_end')
            else:
                # No streaming, or streaming failed. Force Groq when streaming
                # was attempted-and-failed so we don't re-hit ElevenLabs burst
                # on the same audio that just killed the streaming session.
                # (Mirrors mobile v0.3.4 forceGroq path.)
                streaming_failed = self._stream_failed_reason is not None
                self._stream_failed_reason = None
                # A "recovery burst" is a burst forced *because the live stream
                # already failed* (e.g. the post-pause resume-gap safety net).
                # We know speech was being captured, so an empty result here is
                # a recovery FAILURE, not genuine silence — see the empty-result
                # handling below, which saves it for Retry instead of discarding.
                recovery_burst = streaming_failed
                diag.mark('transcribe_burst_start', force_groq=streaming_failed)
                result = transcribe(audio_data, self.local_model, force_groq=streaming_failed)
                diag.mark('transcribe_burst_end',
                          result_chars=len(result) if result else 0)
            end_time = time.time()

            transcription_time = end_time - start_time
            ConfigManager.console_print(f'Transcription completed in {transcription_time:.2f} seconds. Post-processed line: {result}')

            if self.is_cancelled:
                ConfigManager.console_print('Cancelled before paste; discarding result')
                diag.mark('cancelled', stage='post_transcribe')
                return

            if not self.is_running:
                return

            if not result.strip():
                # A recovery burst (live stream failed → forced Groq over the
                # full buffer) coming back empty is NOT genuine silence: the
                # stream had already been carrying real speech before it broke
                # (e.g. a long unpaused pause + resume gap on a multi-minute
                # dictation). Whisper/Groq can return empty or a filtered
                # hallucination when a long silence dominates the buffer, and
                # silently discarding it loses the whole dictation with no
                # recovery. Treat it as a FAILURE so the audio is saved to
                # failed/ and shows up with a Retry button in transcript history.
                substantial = audio_data is not None and audio_data.size > 16000  # >~1s
                if recovery_burst and substantial and not self.is_cancelled:
                    reason = 'recovery burst returned empty (post-pause stream gap)'
                    diag.mark('recovery_burst_empty', samples=int(audio_data.size))
                    ConfigManager.console_print(
                        'Recovery burst returned empty on a substantial recording; '
                        'saving audio for Retry instead of discarding.'
                    )
                    self._persist_failed_recording(audio_data, reason)
                    self.statusSignal.emit('error')
                    self.resultSignal.emit('')
                    return
                # Genuine silence / hallucination filter / RMS skip on a normal
                # dictation. Surface "Nothing transcribable detected" via the
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
            # Surface which backend produced this text (toast). Emit before the
            # result so the hint is up as the paste lands. consume_last_engine()
            # reads the thread-local set inside transcribe()/the streaming path.
            try:
                from transcription import consume_last_engine
                engine = consume_last_engine()
                if engine:
                    self.engineSignal.emit(engine)
            except Exception:
                pass
            self.resultSignal.emit(result)

        except TranscriptionAPIError as e:
            traceback.print_exc()
            ConfigManager.console_print(f'Transcription API failure: {e.reason}')
            diag.mark('api_error', reason=e.reason)
            # Don't persist or surface anything if the user already cancelled.
            # The exception may have arisen as a side effect of cancel() closing
            # the streaming WS; we don't want a stale failed/ entry for that.
            if audio_data is not None and audio_data.size > 0 and not self.is_cancelled:
                self._persist_failed_recording(audio_data, e.reason)
            if not self.is_cancelled:
                self.statusSignal.emit('error')
                # Emit empty result so the existing post-completion flow runs
                # (key listener restart, continuous-mode re-arm).
                self.resultSignal.emit('')
        except Exception as e:
            traceback.print_exc()
            diag.mark('exception', cls=type(e).__name__, msg=str(e)[:200])
            if not self.is_cancelled:
                self.statusSignal.emit('error')
                self.resultSignal.emit('')
        finally:
            diag.mark('finally', cancelled=self.is_cancelled)
            self.stop_recording()
            # Make sure no streaming session is left dangling — _commit_*
            # already clears self._stream_session on success/failure, but the
            # exception paths above may have left it set.
            self._cancel_streaming_session()
            # Release the warm-capture consumer (no-op if None or in legacy
            # path). The InputStream itself stays open — that's the point.
            if consumer is not None and self._audio_capture_service is not None:
                self._audio_capture_service.detach_consumer(consumer)

    def _archive_recording(self, audio_data):
        """Save every finished recording to recordings/ BEFORE transcription.

        This is the save-first safety net: the raw audio lands on disk before
        anything can fail, so no dictation is ever lost to a transcription
        error, empty result, network drop, or crash. Returns (abs_path,
        rel_path), or (None, None) if the write fails. Files are pruned by age
        and count at startup — see prune_recordings_archive."""
        if audio_data is None or getattr(audio_data, 'size', 0) <= 0:
            return None, None
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rec_dir = os.path.join(project_root, 'recordings')
        try:
            os.makedirs(rec_dir, exist_ok=True)
            now = datetime.datetime.now()
            fname = now.strftime('%Y-%m-%d_%H-%M-%S_%f') + '.wav'
            abs_path = os.path.join(rec_dir, fname)
            sf.write(abs_path, audio_data, self.sample_rate or 16000)
            rel_path = os.path.relpath(abs_path, project_root).replace(os.sep, '/')
            ConfigManager.console_print(f'Archived recording → {rel_path}')
            return abs_path, rel_path
        except Exception as e:
            ConfigManager.console_print(f'Could not archive recording: {e}')
            return None, None

    def _persist_failed_recording(self, audio_data, reason):
        """Record a failed transcription in failed_log.txt.

        The raw audio is already on disk in recordings/ (save-first, see
        _archive_recording), so normally we just append a log entry pointing at
        that archived file — no second copy — and Retry reads it straight from
        the archive. Only if the archive write failed do we fall back to a
        dedicated failed/ copy here, so a failure is NEVER left with no audio."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        now = datetime.datetime.now()
        ts_human = now.strftime('%Y-%m-%d %H:%M:%S')

        rel_path = self._archive_rel
        if not (rel_path and self._archive_abs and os.path.isfile(self._archive_abs)):
            # Archive missing (write failed) — fall back to a failed/ copy so we
            # still preserve the audio for Retry.
            failed_dir = os.path.join(project_root, 'failed')
            rel_path = None
            try:
                os.makedirs(failed_dir, exist_ok=True)
                fname = now.strftime('%Y-%m-%d_%H-%M-%S_%f') + '.wav'
                audio_path = os.path.join(failed_dir, fname)
                sf.write(audio_path, audio_data, self.sample_rate or 16000)
                rel_path = os.path.relpath(audio_path, project_root).replace(os.sep, '/')
            except Exception as e:
                ConfigManager.console_print(f'Could not save failed audio: {e}')

        log_path = os.path.join(project_root, 'failed_log.txt')
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                audio_line = rel_path if rel_path else '(audio not saved)'
                f.write(f'[{ts_human}]\n  AUDIO:    {audio_line}\n  ERROR:    {reason}\n\n')
        except Exception as e:
            ConfigManager.console_print(f'Could not write failed_log.txt: {e}')

        if rel_path:
            abs_out = os.path.join(project_root, rel_path.replace('/', os.sep))
            self.failedSignal.emit(abs_out, reason)

    # ---- v0.3.2 PC: streaming-during-recording helpers ----

    def _acquire_warm_session_or_open(self) -> None:
        """v0.4 PC: zero-latency session acquisition.

        Tries the ElevenLabsSessionPool first — if a warm session is
        available, hand-off is instant. Otherwise falls back to the legacy
        in-record open (blocks up to 2s on the WS handshake) for safety.

        Sets self._stream_session to a ready Session, or None.

        Note: per-dictation reset of _stream_failed_reason / _stream_segments
        lives in run(); this method ALSO runs on resume and must not wipe text
        or failures accumulated from earlier spans of the same dictation.
        """
        self._stream_session = None

        # Pool fast-path — only attempted when the pool is enabled. Returns
        # an already-ready Session with keepalive running.
        pool = self._session_pool
        if pool is not None and pool.is_enabled():
            warm = pool.acquire()
            if warm is not None:
                self._stream_session = warm
                self._streaming_in_use = True
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

        def on_ready(ok: bool) -> None:
            # Mirrors mobile v0.3.6: session.onFailure fires onReadyCallback(false)
            # for failed opens and mid-recording deaths. Mark a reason so the
            # stop path bursts the full audio instead of committing a dead
            # session. A *successful* open needs no action here — the session is
            # already assigned below and has been buffering frames until ready.
            if not ok:
                self._stream_failed_reason = (
                    self._stream_failed_reason or 'ElevenLabs streaming session failed to open'
                )

        session = Session(api_key, keyterms)
        session.start(on_ready=on_ready)
        # Assign the session IMMEDIATELY — do NOT block on the WS handshake.
        # Session.send_audio_chunk() buffers any frames that arrive before
        # session_started (Session._pending) and flushes them in order the
        # instant the socket is ready, so streaming into a still-connecting
        # session loses nothing. The old 2s ready-wait is exactly what left
        # _stream_session = None during a resume reopen, so every frame captured
        # in that window hit the "no live stream" path and forced a Groq burst.
        # Assigning up-front keeps dictation on ElevenLabs straight through
        # pause/resume; a genuine open failure is handled via on_ready(False)
        # above (sets _stream_failed_reason → stop bursts the full buffer).
        self._stream_session = session
        self._streaming_in_use = True
        ConfigManager.console_print(
            f'Streaming session opening (keyterms={len(keyterms)}); '
            f'frames buffer until session_started'
        )

    # Cap on how many frames we buffer across a resume gap before giving up and
    # bursting. A fresh ElevenLabs session is normally assigned in well under a
    # second; ~250 frames (~7.5s at 30ms) is a generous ceiling that still
    # protects against a session that never opens.
    _RESUME_STREAM_BUFFER_MAX_FRAMES = 250

    def _feed_streaming_session(self, pcm_frame: np.ndarray) -> None:
        """Forward one captured PCM frame to the streaming session, if any.

        Tier 2 (2026-07-06): a None session no longer condemns the whole
        dictation to a Groq burst. Two cases are now handled gracefully:

          * Pause-boundary straggler — a frame that was mid-flight in the
            capture loop when pause() nulled the session. is_paused is already
            True; the pre-pause span was already committed and the frame is
            safe in `recording`. Harmless: drop it from the stream, do NOT fail.

          * Resume gap — a frame captured just after resume, before the fresh
            session is assigned. Buffer it (bounded) and flush the buffer into
            the session the instant it appears, so we stay on ElevenLabs and it
            catches up at faster-than-real-time. Only if the buffer overflows
            (session never opened) do we fall back to a burst.
        """
        session = self._stream_session
        if session is None:
            if self.is_paused:
                # Pause-boundary straggler — not a failure.
                if self._diag is not None:
                    self._diag.mark('pause_straggler_ignored')
                return
            if self._streaming_in_use and self._stream_failed_reason is None:
                if len(self._resume_pending) < self._RESUME_STREAM_BUFFER_MAX_FRAMES:
                    self._resume_pending.append(pcm_frame)
                else:
                    self._stream_failed_reason = (
                        'resume session did not open in time; bursting full audio'
                    )
                    self._resume_pending = []
                    ConfigManager.console_print(
                        'Resume session never opened; will burst full audio on stop'
                    )
            return
        if session.is_dead():
            # The socket has actually closed/failed (server reject, network
            # drop, failed open). A still-CONNECTING session is NOT dead — it
            # returns False from is_ready() but buffers frames until
            # session_started, so we must not treat "not ready yet" as death or
            # we'd bail to a Groq burst on every resume reopen. Only a truly
            # closed socket forces the burst fallback.
            if not self._stream_failed_reason:
                self._stream_failed_reason = 'ElevenLabs streaming session died mid-recording'
                ConfigManager.console_print('Streaming session died mid-recording; switching to burst-on-stop')
            self._stream_session = None
            return
        try:
            # Flush any frames buffered during a resume gap first, in order, so
            # the fresh session hears the post-resume audio from its true start.
            # The session buffers them in _pending until session_started, then
            # streams them faster-than-real-time to catch up.
            if self._resume_pending:
                pending = self._resume_pending
                self._resume_pending = []
                ConfigManager.console_print(
                    f'Flushing {len(pending)} buffered resume-gap frame(s) into fresh session'
                )
                for f in pending:
                    session.send_audio_chunk(f.tobytes(), commit=False)
            session.send_audio_chunk(pcm_frame.tobytes(), commit=False)
        except Exception as e:
            ConfigManager.console_print(f'Streaming send failed: {e}; switching to burst-on-stop')
            self._stream_failed_reason = f'send failed: {e}'
            self._stream_session = None

    def _commit_session_blocking(self, session, timeout: float = 7.0) -> tuple[str | None, str | None]:
        """Commit ONE streaming session and block (≤ timeout) for its text.

        Returns (text, error): `text` is the finalised span text (None if empty
        or on failure); `error` is a human reason string when the commit failed.
        Used for both the per-pause span commit and the final-span commit.
        """
        from threading import Event as _Event
        done_event = _Event()
        result_holder: dict = {}

        def on_result(r: dict) -> None:
            result_holder.update(r)
            done_event.set()

        try:
            session.commit(on_result)
        except Exception as e:
            return None, f'commit raised: {e}'

        # ElevenLabs RT finalises in <1s healthy; 7s catches real hangs.
        if not done_event.wait(timeout=timeout):
            try:
                session.cancel()
            except Exception:
                pass
            return None, f'commit timed out (>{timeout:.0f}s)'

        if result_holder.get('error'):
            return None, result_holder['error']

        text = (result_holder.get('text') or '').strip()
        return (text or None), None

    def _rotate_streaming_session_on_pause(self) -> None:
        """Finalize the current streaming span when the user pauses.

        Commits the open session on a daemon thread (so the GUI never blocks),
        stashing the span text in _stream_segments. The pre-pause socket is
        then done — a long pause can't lose its text or trip the server idle/
        session/queue limits. Resume opens a fresh socket. If the commit fails,
        mark _stream_failed_reason so stop bursts the full audio instead.
        """
        with self._stream_lock:
            session = self._stream_session
            self._stream_session = None
            if session is None:
                return

            def _do_commit(sess=session):
                text, error = self._commit_session_blocking(sess, timeout=7.0)
                if error:
                    ConfigManager.console_print(f'Pause-commit failed: {error}; will burst on stop')
                    self._stream_failed_reason = self._stream_failed_reason or error
                    return
                if text:
                    with self._stream_lock:
                        self._stream_segments.append(text)
                    ConfigManager.console_print(
                        f'Pause-commit stashed span ({len(text)} chars; '
                        f'{len(self._stream_segments)} span(s) so far)'
                    )

            self._pause_commit_thread = threading.Thread(
                target=_do_commit, name='elevenlabs-pause-commit', daemon=True,
            )
            self._pause_commit_thread.start()

    def _join_pause_commit(self, timeout: float = 8.0) -> None:
        """Wait for an in-flight pause-commit to finish (if any)."""
        t = self._pause_commit_thread
        if t is not None:
            t.join(timeout=timeout)
            self._pause_commit_thread = None

    def _reopen_streaming_session_on_resume(self) -> None:
        """Open a fresh streaming session for the span after a pause.

        Re-acquires (the pool usually has a warm spare, so this is ~instant). If
        streaming was in use but re-acquire yields nothing, mark a failure so
        stop bursts the full audio — post-resume speech is still captured in the
        recording buffer, so nothing is lost.

        MUST be non-blocking: it runs while is_paused is still True (the caller
        clears is_paused immediately after) and the capture loop discards frames
        while paused, so any blocking here would drop post-resume audio. So we
        take a pool session ONLY if one is instantly ready (try_acquire_nowait,
        no join), otherwise open a fresh session inline — which assigns up-front
        and buffers frames until its handshake completes (see
        _open_streaming_session_if_enabled). Either way capture streams straight
        onto ElevenLabs across the pause instead of falling into the burst gap.

        v0.4.1: no longer joins the pause-commit here. That join only existed to
        keep span order, but stop's _commit_streaming_session already joins
        before it reads _stream_segments, so ordering still holds.
        """
        if not self._streaming_in_use:
            return
        pool = self._session_pool
        if pool is not None and pool.is_enabled():
            warm = pool.try_acquire_nowait()
            if warm is not None:
                self._stream_session = warm
                self._streaming_in_use = True
                ConfigManager.console_print('Resume: took instantly-ready warm session from pool')
                return
        # No instant warm session — open a fresh one inline (non-blocking).
        self._open_streaming_session_if_enabled()
        if self._stream_session is None and not self._stream_failed_reason:
            self._stream_failed_reason = 'could not reopen streaming session on resume'
            ConfigManager.console_print(
                'Resume could not reopen streaming session; will burst full audio on stop'
            )

    def _commit_streaming_session(self, full_audio: np.ndarray) -> str | None:
        """Commit the final span and stitch all spans into the result.

        Returns the finalised (stitched) text when streaming was used and every
        span succeeded. Returns None when streaming wasn't used or any span
        failed — caller falls through to a single burst over `full_audio`
        (which holds every spoken span; paused gaps were already dropped). The
        all-or-burst rule avoids both gaps and double-transcription.
        """
        # A pause may have left a commit in flight — let it land so its span is
        # included and ordered before the final one.
        self._join_pause_commit()

        with self._stream_lock:
            session = self._stream_session
            self._stream_session = None

        final_text = None
        if session is not None:
            final_text, error = self._commit_session_blocking(session, timeout=7.0)
            if error:
                ConfigManager.console_print(f'Final-span commit failed: {error}; falling back to burst')
                self._stream_failed_reason = self._stream_failed_reason or error

        if not self._streaming_in_use:
            return None  # Groq path — no streamed text to deliver
        if self._stream_failed_reason:
            return None  # some span failed → burst the whole audio instead

        with self._stream_lock:
            parts = list(self._stream_segments)
        if final_text:
            parts.append(final_text)
        combined = ' '.join(p for p in parts if p).strip()
        if not combined:
            ConfigManager.console_print('Streaming produced no text; falling back to burst')
            return None
        ConfigManager.console_print(
            f'Streaming delivered {len(combined)} chars across {len(parts)} span(s); skipping burst STT'
        )
        return combined

    def _cancel_streaming_session(self) -> None:
        """Abort streaming without committing (user cancelled / empty audio)."""
        with self._stream_lock:
            session = self._stream_session
            self._stream_session = None
            self._stream_segments = []
        if session is not None:
            try:
                session.cancel()
            except Exception:
                pass

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
