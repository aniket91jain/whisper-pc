"""Auto-delete ElevenLabs server-side transcription history.

Python equivalent of mobile's `HistoryDeleteWorker.kt`. ElevenLabs Zero
Retention Mode is Enterprise-only; on standard tiers the server records
every transcript. Closest practical privacy posture is delete-after-the-fact:
list everything in the account history and DELETE each ID.

Two scheduling modes:

  1. `start_periodic()` — background thread, wakes every 15 minutes and runs
     a sweep. Belt-and-suspenders catch-all.

  2. `schedule_one_shot(api_key)` — fire-and-forget thread, sleeps 10 seconds
     (so the server commits the transcript first), then runs one sweep.

Both call the same `_sweep()` function which:
  GET /v1/speech-to-text/transcripts → list of IDs
  for each id: DELETE /v1/speech-to-text/transcripts/{id}

API errors are logged and skipped; the next sweep retries them.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import requests


_LOG = logging.getLogger(__name__)


ELEVENLABS_BASE = "https://api.elevenlabs.io"
PERIODIC_INTERVAL_S = 15 * 60  # 15 minutes


_periodic_thread: Optional[threading.Thread] = None
_periodic_stop = threading.Event()


def _sweep(api_key: str) -> None:
    if not api_key:
        return
    headers = {"xi-api-key": api_key}
    try:
        r = requests.get(
            f"{ELEVENLABS_BASE}/v1/speech-to-text/transcripts",
            headers=headers,
            params={"page_size": 100},
            timeout=15,
        )
        if r.status_code >= 400:
            _LOG.warning(f"history list HTTP {r.status_code}: {r.text[:200]}")
            return
        transcripts = r.json().get("transcripts") or []
    except Exception as e:
        _LOG.warning(f"history list failed: {e}")
        return

    deleted = 0
    failed = 0
    for entry in transcripts:
        tid = (entry.get("id") or "").strip()
        if not tid:
            continue
        try:
            d = requests.delete(
                f"{ELEVENLABS_BASE}/v1/speech-to-text/transcripts/{tid}",
                headers=headers,
                timeout=15,
            )
            if d.status_code < 400:
                deleted += 1
            else:
                failed += 1
                _LOG.warning(f"delete {tid} HTTP {d.status_code}")
        except Exception as e:
            failed += 1
            _LOG.warning(f"delete {tid} failed: {e}")

    _LOG.info(f"ElevenLabs history sweep: deleted={deleted} failed={failed} total={len(transcripts)}")


def schedule_one_shot(api_key: str, delay_s: float = 10.0) -> None:
    """Fire a one-shot sweep after `delay_s` seconds. Daemon thread, no return.

    Called after every dictation completes — gives the server time to write
    the transcript to its history table before we try to delete it.
    """
    if not api_key:
        return
    def _run() -> None:
        time.sleep(delay_s)
        _sweep(api_key)
    threading.Thread(target=_run, name="el-rt-history-sweep-oneshot", daemon=True).start()


def start_periodic(api_key_getter) -> None:
    """Start a background thread that runs a sweep every 15 minutes.

    Idempotent — calling twice is harmless.

    `api_key_getter` is a callable so the worker re-reads the key on every
    iteration (handles the case where the user rotates the key at runtime).
    """
    global _periodic_thread
    if _periodic_thread is not None and _periodic_thread.is_alive():
        return
    _periodic_stop.clear()

    def _loop() -> None:
        # Skip the first immediate fire — there's nothing on the server yet
        # at app launch, and the one-shot post-dictation worker covers fresh
        # entries.
        while not _periodic_stop.wait(PERIODIC_INTERVAL_S):
            try:
                key = api_key_getter() or ""
                if key:
                    _sweep(key)
            except Exception as e:
                _LOG.warning(f"periodic sweep error: {e}")

    _periodic_thread = threading.Thread(
        target=_loop, name="el-rt-history-sweep-periodic", daemon=True,
    )
    _periodic_thread.start()


def stop_periodic() -> None:
    _periodic_stop.set()
