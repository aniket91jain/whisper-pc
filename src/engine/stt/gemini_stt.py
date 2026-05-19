"""Cloud STT fallback via Gemini's multimodal `generateContent` endpoint.

Used when Groq Whisper is blocked by Cloudflare's WAF — typically VPN exit IPs
on the blocklist. Gemini lives on `generativelanguage.googleapis.com` (Google
infra, not Cloudflare), so VPN-on requests go through.

Endpoint shape is plain stateless REST POST — no session, no resumption, no
`store` parameter exists for this endpoint, so the ZDR caveats for Live /
Interactions APIs do not apply to this code path. Privacy is governed by the
tier of the Google Cloud project that owns the API key (free vs paid; with or
without ZDR approval).

Mirrors `whisper-mobile`'s `GeminiSttClient.kt`.
"""

import base64
import io
import json
import os
from typing import Optional

import numpy as np
import requests
import soundfile as sf

from utils import ConfigManager


_ENDPOINT_BASE = 'https://generativelanguage.googleapis.com/v1beta/models'
# Flash supports audio input; Flash-Lite does not as of Gemini 2.5.
_DEFAULT_MODEL = 'gemini-2.5-flash'

_TRANSCRIBE_PROMPT = (
    'Transcribe the spoken English in this audio verbatim. '
    'Output only the transcript, with normal capitalization and punctuation. '
    'Do not add commentary, do not describe the audio, do not refuse. '
    'If the audio is silent or unintelligible, output __EMPTY__.'
)

_EMPTY_SENTINEL = '__EMPTY__'


class GeminiSttError(Exception):
    """Raised when the Gemini STT call fails. Caller logs + falls back."""


def is_configured() -> bool:
    """True iff a Gemini API key is set via env. Used by the pipeline to
    decide whether to even attempt the Gemini fallback."""
    return bool(_api_key())


def _api_key() -> Optional[str]:
    return os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')


def transcribe(audio_data: np.ndarray, sample_rate: int = 16000,
               model: str = _DEFAULT_MODEL, timeout_sec: float = 45.0) -> str:
    """Transcribe an audio array via Gemini.

    @param audio_data  PCM16 mono numpy array (same shape transcribe_api uses).
    @param sample_rate Sample rate of the audio array.
    @param model       Gemini model id; Flash is the only audio-capable
                       2.5-tier as of 2026-05.
    @param timeout_sec HTTP request timeout. 45s covers ~3-min dictation +
                       Gemini's first-token latency on cold connection.

    Returns the transcript string. Raises GeminiSttError on any failure
    (missing key, network, HTTP non-2xx, empty response, EMPTY sentinel).
    """
    api_key = _api_key()
    if not api_key:
        raise GeminiSttError('Gemini API key not set (GEMINI_API_KEY env var)')

    byte_io = io.BytesIO()
    sf.write(byte_io, audio_data, sample_rate, format='wav')
    wav_bytes = byte_io.getvalue()
    audio_b64 = base64.b64encode(wav_bytes).decode('ascii')

    body = {
        'contents': [{
            'role': 'user',
            'parts': [
                {'text': _TRANSCRIBE_PROMPT},
                {'inline_data': {
                    'mime_type': 'audio/wav',
                    'data': audio_b64,
                }},
            ],
        }],
        'generationConfig': {
            'temperature': 0,
            # 4 K output tokens covers ~3000 English words — generous for a
            # single dictation. Bump if long-form dictation hits the cap.
            'maxOutputTokens': 4096,
        },
    }

    url = f'{_ENDPOINT_BASE}/{model}:generateContent?key={api_key}'
    ConfigManager.console_print(
        f'Gemini STT request: model={model} wavBytes={len(wav_bytes)}'
    )
    try:
        resp = requests.post(
            url,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(body),
            timeout=timeout_sec,
        )
    except requests.RequestException as e:
        raise GeminiSttError(f'Gemini network failure: {e}') from e

    if resp.status_code != 200:
        body_text = (resp.text or '')[:200]
        raise GeminiSttError(f'Gemini HTTP {resp.status_code}: {body_text}')

    try:
        payload = resp.json()
    except ValueError as e:
        raise GeminiSttError(f'Gemini bad JSON: {e}') from e

    if 'error' in payload:
        msg = payload['error'].get('message', 'Unknown error')
        raise GeminiSttError(f'Gemini API error: {msg}')

    candidates = payload.get('candidates') or []
    if not candidates:
        raise GeminiSttError('Gemini returned no candidates')

    parts = (candidates[0].get('content') or {}).get('parts') or []
    text = ''.join(p.get('text', '') for p in parts).strip()
    if not text:
        raise GeminiSttError('Gemini returned empty transcript')
    if text == _EMPTY_SENTINEL:
        raise GeminiSttError('Audio silent or unintelligible')
    return text
