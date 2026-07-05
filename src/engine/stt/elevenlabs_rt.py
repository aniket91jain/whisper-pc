"""ElevenLabs Scribe v2 Realtime client for Whisper PC (v0.3).

Ports the mobile `ElevenLabsRtSession.kt` to Python. Two usage patterns:

  1. Single-shot record-then-burst (v0.2-style): collect full PCM buffer, then
     send it all in a burst with commit=true on the last chunk. Use
     `transcribe_burst()`.

  2. Streaming-during-recording (v0.3-style): open the WS at recording start,
     forward each captured frame as it arrives via `Session.send_audio_chunk()`,
     finalize with `Session.commit()` when the user stops. Use the `Session`
     class.

Both paths handle keyterms, idle-WS keepalive (every 10s during pause),
and graceful close on commit or error.

Protocol confirmed by direct probe (see whisper-engine-sandbox):
  wss://api.elevenlabs.io/v1/speech-to-text/realtime
    ?model_id=scribe_v2_realtime
    &audio_format=pcm_16000
    &commit_strategy=manual
    &keyterms=Plaud&keyterms=OwnerRez&...    (repeated, ≤50, ≤20 chars each)
  Auth header: xi-api-key
  Send (per ~100ms chunk):
    {"message_type": "input_audio_chunk", "audio_base_64": "...", "commit": bool}
  Receive:
    {"message_type": "session_started", "session_id": "...", "config": {...}}
    {"message_type": "partial_transcript", "text": "..."}    (repeated; ignored)
    {"message_type": "committed_transcript", "text": "..."}  (final, fires once)
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Callable, List, Optional
from urllib.parse import urlencode

import websocket  # websocket-client


_LOG = logging.getLogger(__name__)


WS_HOST = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
MODEL_ID = "scribe_v2_realtime"
SAMPLE_RATE = 16000
MAX_KEYTERMS = 50
MAX_KEYTERM_LEN = 20
KEEPALIVE_INTERVAL_S = 10  # ElevenLabs RT idle timeout is ~15s; ping every 10s
COMMIT_TIMEOUT_S = 5.0      # max wait for committed_transcript after commit sent
                            # (bench: ElevenLabs RT finalises in 248-870ms healthy;
                            #  5s catches real hangs without false-positive fallback)
# transcribe_burst pacing. ElevenLabs Realtime has a bounded server-side input
# queue; dumping a long pre-recorded buffer in a tight loop overflows it
# (queue_overflow close — observed in failed_log.txt on multi-minute audio).
# Pace the burst to at most this multiple of real-time so the server's queue
# can't run away. 4× finishes a 30s clip in ~7.5s while staying well under the
# overflow point. (Long buffers are routed to Groq upstream and never reach
# this path — see transcription.transcribe.)
BURST_PACE_FACTOR = 4.0


def _build_url(keyterms: List[str]) -> str:
    params: list[tuple[str, str]] = [
        ("model_id", MODEL_ID),
        ("audio_format", f"pcm_{SAMPLE_RATE}"),
        ("commit_strategy", "manual"),
        # no_verbatim hints to the server to drop disfluencies; documented in
        # the ElevenLabs SDK source. One bench-audio comparison showed near-
        # identical output, but the user has decided to keep it on for
        # potentially-helpful effect on heavier-disfluency inputs we haven't
        # tested. Cost is zero.
        ("no_verbatim", "true"),
        # Pin to English so Hindi loanwords come back Romanized instead of in
        # Devanagari. Local devanagari_translit is the belt-and-suspenders
        # safety net for any Devanagari that slips through the language pin.
        ("language_code", "eng"),
    ]
    for term in keyterms:
        term = term.strip()
        if not term or len(term) > MAX_KEYTERM_LEN:
            continue
        params.append(("keyterms", term))
        if len([p for p in params if p[0] == "keyterms"]) >= MAX_KEYTERMS:
            break
    return f"{WS_HOST}?{urlencode(params)}"


class Session:
    """Persistent ElevenLabs Realtime session — open WS, stream chunks, commit.

    Thread-safe via internal Lock. Designed for the sounddevice callback
    pattern in `src/result_thread.py:_record_audio` where each ~30ms PCM
    frame is forwarded into `send_audio_chunk()`.

    Lifecycle:
      session = Session(api_key, keyterms)
      session.start(on_ready=lambda ok: ...)
      session.send_audio_chunk(pcm_bytes)   # repeatedly, during recording
      session.commit(on_result=lambda r: ...)  # user released; await final
    """

    def __init__(self, api_key: str, keyterms: List[str]):
        self.api_key = api_key
        self.keyterms = keyterms or []
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._commit_sent = False
        self._pending: list[bytes] = []
        self._session_id: Optional[str] = None
        self._on_result: Optional[Callable[[dict], None]] = None
        self._on_ready: Optional[Callable[[bool], None]] = None
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: Optional[threading.Thread] = None
        self._t_start = 0.0
        self._t_commit = 0.0

        # PC mirror of mobile v0.3.5: Scribe v2 RT emits `committed_transcript`
        # per utterance segment, not per session. A 2-3s mid-dictation pause
        # triggers a server-initiated finalise even under commit_strategy=manual.
        # Pre-fix, we'd close() on every committed_transcript and drop the
        # segment text on the floor when no callback was waiting yet (server-
        # initiated). Now we accumulate into _segments and only close on the
        # user-initiated commit.
        self._segments: list[str] = []
        self._segments_lock = threading.Lock()

    # ---- public API ----

    def start(self, on_ready: Callable[[bool], None] = lambda ok: None) -> None:
        if not self.api_key:
            on_ready(False)
            return
        self._on_ready = on_ready
        self._t_start = time.monotonic()
        url = _build_url(self.keyterms)
        headers = [f"xi-api-key: {self.api_key}"]
        self._ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws_thread = threading.Thread(
            target=self._ws.run_forever,
            name="elevenlabs-rt-session",
            daemon=True,
        )
        self._ws_thread.start()

    def send_audio_chunk(self, pcm: bytes, commit: bool = False) -> None:
        """Forward a PCM16 chunk to the session. Buffered if WS isn't ready yet."""
        if self._closed.is_set():
            return
        with self._lock:
            if commit:
                if self._commit_sent:
                    return
                self._commit_sent = True
                self._t_commit = time.monotonic()
            if not self._ready.is_set():
                if pcm:
                    self._pending.append(pcm)
                return
            self._emit(pcm, commit=commit)

    def commit(self, on_result: Callable[[dict], None]) -> None:
        """Mark the session complete and await `committed_transcript`.

        Callback fires (on the WS thread) with a dict: {"text": str|None, "error": str|None}

        v0.3.1 bug fix: arms a 5s watchdog. If the server doesn't respond,
        the callback fires with an error so the caller's fallback (burst)
        path can take over instead of hanging indefinitely. Mirrors the
        commitWatchdog in mobile's ElevenLabsRtSession.kt.

        PC mirror of mobile v0.3.6: if the WS is already dead (server
        segmented mid-recording and closed) but we have stashed segment text,
        deliver it immediately instead of waiting for the watchdog.
        """
        if self._closed.is_set() or not self._ready.is_set():
            stashed = self._joined_segments()
            if stashed:
                _LOG.info(f"commit: WS dead, delivering {len(stashed)} chars of stashed segment text")
                try:
                    on_result({"text": stashed, "error": None})
                except Exception as e:
                    _LOG.warning(f"on_result raised on stashed-text delivery: {e}")
                return

        self._on_result = on_result
        self._t_commit = time.monotonic()
        threading.Timer(COMMIT_TIMEOUT_S, self._on_commit_timeout).start()
        self.send_audio_chunk(b"", commit=True)

    def _joined_segments(self) -> str:
        """Return all accumulated segment text joined with single spaces."""
        with self._segments_lock:
            return " ".join(s for s in self._segments if s)

    def _on_commit_timeout(self) -> None:
        with self._lock:
            cb = self._on_result
            self._on_result = None
        if cb is None:
            return

        # PC mirror of mobile v0.3.5: prefer stashed segment text over failure
        # when the user committed after the server had already given us some.
        stashed = self._joined_segments()
        try:
            if stashed:
                _LOG.warning(f"commit watchdog fired; delivering {len(stashed)} chars of stashed segments instead of failing")
                cb({"text": stashed, "error": None})
            else:
                _LOG.warning(f"commit watchdog fired after {COMMIT_TIMEOUT_S}s; firing failure")
                cb({
                    "text": None,
                    "error": f"ElevenLabs commit timed out (no committed_transcript in {COMMIT_TIMEOUT_S:.0f}s)",
                })
        except Exception as e:
            _LOG.warning(f"commit timeout callback raised: {e}")
        self.cancel()

    def cancel(self) -> None:
        """Abort without committing — user pressed cancel mid-dictation."""
        self._closed.set()
        self._stop_keepalive()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    def start_keepalive(self) -> None:
        """Send empty input_audio_chunk every 10s to survive idle pauses."""
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            return
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, name="elevenlabs-rt-keepalive", daemon=True,
        )
        self._keepalive_thread.start()

    def is_ready(self) -> bool:
        return self._ready.is_set() and not self._closed.is_set()

    def is_dead(self) -> bool:
        """True once the WS has actually closed/failed. Distinct from "not yet
        ready": a still-connecting session is neither ready nor dead — it buffers
        frames in _pending and flushes them on session_started. Callers feeding a
        live recording use this to tell a doomed socket (fall back to burst) from
        one that just hasn't finished its handshake (keep feeding; it buffers)."""
        return self._closed.is_set()

    def session_id(self) -> Optional[str]:
        return self._session_id

    # ---- internals ----

    def _stop_keepalive(self) -> None:
        self._keepalive_stop.set()

    def _keepalive_loop(self) -> None:
        while not self._keepalive_stop.wait(KEEPALIVE_INTERVAL_S):
            if self._closed.is_set() or self._commit_sent:
                return
            if not self._ready.is_set():
                continue
            try:
                self._emit(b"", commit=False)
            except Exception:
                pass

    def _emit(self, pcm: bytes, commit: bool) -> None:
        if not self._ws:
            return
        msg = {
            "message_type": "input_audio_chunk",
            "audio_base_64": (base64.b64encode(pcm).decode("ascii") if pcm else ""),
            "commit": commit,
        }
        try:
            self._ws.send(json.dumps(msg))
        except Exception as e:
            _LOG.warning(f"WS send failed: {e}")

    def _flush_pending(self) -> None:
        with self._lock:
            for chunk in self._pending:
                self._emit(chunk, commit=False)
            self._pending.clear()
            if self._commit_sent:
                self._emit(b"", commit=True)

    def _on_open(self, ws) -> None:
        _LOG.info(f"ElevenLabs RT WS opened at +{(time.monotonic() - self._t_start):.2f}s")

    def _on_message(self, ws, message) -> None:
        try:
            payload = json.loads(message)
        except Exception:
            _LOG.warning(f"Failed to parse RT message: {message[:200]}")
            return
        mtype = payload.get("message_type", "")
        if mtype == "session_started":
            self._session_id = payload.get("session_id")
            self._ready.set()
            _LOG.info(f"RT session_started id={self._session_id}")
            self._flush_pending()
            if self._on_ready:
                self._on_ready(True)
        elif mtype == "partial_transcript":
            pass  # ignored on PC v0.3 (no live UI rendering)
        elif mtype == "committed_transcript":
            segment_text = (payload.get("text") or "").strip()
            eoa_to_final_ms = (
                int((time.monotonic() - self._t_commit) * 1000)
                if self._t_commit else -1
            )

            # Append to the segment buffer first. Then atomically take the
            # callback to discriminate user vs server initiated.
            with self._segments_lock:
                if segment_text:
                    self._segments.append(segment_text)
                combined = " ".join(s for s in self._segments if s)

            with self._lock:
                cb = self._on_result
                self._on_result = None
            server_initiated = cb is None
            _LOG.info(
                f"RT committed_transcript chars={len(segment_text)} "
                f"eoa→final={eoa_to_final_ms}ms serverInitiated={server_initiated} "
                f"total_segments={len(self._segments)}"
            )

            if server_initiated:
                # PC mirror of mobile v0.3.5: don't close on server-initiated
                # commits. The WS may stay alive for more segments; if it
                # doesn't, _on_close will deliver the stashed text.
                return

            # User-initiated commit: deliver the accumulated text and tear down.
            try:
                cb({"text": combined, "error": None})
            except Exception as e:
                _LOG.warning(f"on_result raised: {e}")
            self._closed.set()
            self._stop_keepalive()
            try:
                ws.close()
            except Exception:
                pass
        elif mtype in ("input_error", "error"):
            err = f"{payload.get('error', 'unknown')}: {payload.get('message', message)[:200]}"
            _LOG.warning(f"RT error: {err}")
            with self._lock:
                cb = self._on_result
                self._on_result = None
            if cb is not None:
                cb({"text": None, "error": err})
            self._closed.set()
            self._stop_keepalive()
            try:
                ws.close()
            except Exception:
                pass
        # session_ended, vad events, etc — ignored.

    def _on_error(self, ws, error) -> None:
        _LOG.warning(f"RT WS error: {error}")
        self._ready.clear()
        if self._on_ready:
            self._on_ready(False)
            self._on_ready = None
        # PC mirror of mobile v0.3.5: prefer stashed segments over hard fail.
        if self._on_result:
            stashed = self._joined_segments()
            if stashed:
                _LOG.info(f"WS error but delivering {len(stashed)} chars of stashed segments")
                self._on_result({"text": stashed, "error": None})
            else:
                self._on_result({"text": None, "error": f"WS error: {error}"})
            self._on_result = None
        self._closed.set()
        self._stop_keepalive()

    def _on_close(self, ws, close_code, close_msg) -> None:
        _LOG.info(f"RT WS closed code={close_code} reason={close_msg}")
        self._ready.clear()
        if self._on_result and not self._closed.is_set():
            # PC mirror of mobile v0.3.5: if we have stashed segments from
            # prior server-initiated commits, deliver them on close rather
            # than fail. This handles the case where server commits a segment
            # then immediately closes the WS.
            stashed = self._joined_segments()
            if stashed:
                _LOG.info(f"WS closed (code={close_code}) but delivering {len(stashed)} chars of stashed segments")
                self._on_result({"text": stashed, "error": None})
            else:
                self._on_result({
                    "text": None,
                    "error": f"WS closed before committed_transcript (code={close_code})",
                })
            self._on_result = None
        self._closed.set()
        self._stop_keepalive()


def transcribe_burst(pcm_bytes: bytes, api_key: str, keyterms: List[str], timeout_s: float = 60.0) -> dict:
    """Single-shot record-then-burst transcription.

    Convenience for callers that already have the full PCM buffer (v0.2-style
    burst pattern). Opens a session, sends everything as fast as the network
    allows with commit=true on the last chunk, blocks until committed_transcript.

    Returns {"text": str|None, "error": str|None}.
    """
    if not pcm_bytes:
        return {"text": "", "error": None}
    if not api_key:
        return {"text": None, "error": "ElevenLabs API key not set"}

    session = Session(api_key, keyterms)
    result_holder: dict = {"text": None, "error": None}
    done = threading.Event()
    ready = threading.Event()

    def on_ready(ok: bool) -> None:
        if ok:
            ready.set()

    def on_result(r: dict) -> None:
        result_holder.update(r)
        done.set()

    session.start(on_ready=on_ready)
    if not ready.wait(timeout=10.0):
        session.cancel()
        return {"text": None, "error": "Timeout waiting for session_started"}

    # Burst the audio in 100ms chunks. Server will buffer + process.
    # Send commit=false on all chunks here; we'll send the explicit commit
    # message via session.commit() after the loop so the callback is wired
    # BEFORE the commit hits the server. Otherwise we race the server's
    # committed_transcript against our callback registration, and with the
    # new segment accumulator that race can stash the result and leave the
    # caller waiting forever.
    chunk_size = SAMPLE_RATE * 2 // 10  # 100ms of PCM16 mono
    offset = 0
    total = len(pcm_bytes)
    # Pace to BURST_PACE_FACTOR × real-time so we never outrun the server's
    # input queue (a tight unpaced loop overflows it → queue_overflow).
    t_burst = time.monotonic()
    sent_audio_s = 0.0
    while offset < total:
        end = min(offset + chunk_size, total)
        chunk = pcm_bytes[offset:end]
        offset = end
        session.send_audio_chunk(chunk, commit=False)
        sent_audio_s += (len(chunk) / 2) / SAMPLE_RATE
        behind = (sent_audio_s / BURST_PACE_FACTOR) - (time.monotonic() - t_burst)
        if behind > 0:
            time.sleep(behind)

    session.commit(on_result)

    if not done.wait(timeout=timeout_s):
        session.cancel()
        return {"text": None, "error": "Timeout waiting for committed_transcript"}

    return result_holder
