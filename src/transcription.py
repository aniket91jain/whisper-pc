import io
import os
import re
import threading
import time
from typing import Optional, Tuple
import numpy as np
import soundfile as sf
from openai import OpenAI

from utils import ConfigManager
from engine.polish.post_llm_repair import apply as apply_post_llm_repair
from engine.polish.proper_nouns_renderer import substitute as substitute_proper_nouns
from engine.net import blocked_cache
from engine.stt import gemini_stt
from engine.polish.spoken_punctuation import normalize as _normalize_spoken_symbols
from notifications import fire_dict_addition


# Cache of OpenAI SDK clients keyed by base_url. The SDK uses httpx with
# connection pooling internally, so sharing the client across calls reuses
# the TLS connection and skips the handshake (~200-400ms) on every polish
# or transcribe call after the first.
_OPENAI_CLIENTS: dict = {}

# Max audio length (seconds) the ElevenLabs record-then-burst fallback will
# handle. Longer audio is routed to Groq's file endpoint instead, because
# bursting a long buffer into the Realtime streaming socket overflows its
# server-side queue (queue_overflow). The live streaming path is unaffected.
# Tunable: lower this (and/or BURST_PACE_FACTOR in elevenlabs_rt.py) if
# queue_overflow ever recurs on shorter audio; raise it if Groq rerouting of
# medium clips proves unnecessary.
ELEVENLABS_BURST_MAX_S = 45.0


# Thread-local record of the engine that produced the most recent transcript
# on THIS thread. Set by _write_log_entry (the single chokepoint every ok path
# funnels through) and consumed by result_thread right after transcribe()
# returns, so the live "transcribed by X" toast knows which backend ran without
# threading the label back through every transcribe() return signature.
_LAST_ENGINE = threading.local()


def consume_last_engine() -> str:
    """Return the engine label recorded on this thread by the last successful
    transcription, then clear it. Empty string if none recorded."""
    value = getattr(_LAST_ENGINE, 'value', '') or ''
    _LAST_ENGINE.value = ''
    return value


def _write_log_entry(raw: str, polished: str, engine: str) -> None:
    """Append a single dictation entry to transcript_log.txt.

    Every transcription path goes through this so the user can audit which
    backend produced each entry (engine field). Engine values are stable
    identifiers — `groq+llama`, `groq+llama→gemini-fallback`,
    `groq+llama→local-whisper-fallback`, `elevenlabs-stream`,
    `elevenlabs-burst`. Failures fail silently — logging is informational,
    not load-bearing.
    """
    # Record for the live toast before touching the file — even if the log
    # write fails, the toast should still name the engine that just ran.
    _LAST_ENGINE.value = engine
    try:
        import datetime
        log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'transcript_log.txt',
        )
        with open(log_path, 'a', encoding='utf-8') as f:
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(
                f'[{ts}]\n'
                f'  ENGINE:  {engine}\n'
                f'  RAW:     {(raw or "").strip()}\n'
                f'  POLISHED: {polished}\n\n'
            )
    except Exception:
        pass


def get_openai_client(base_url: str = 'https://api.groq.com/openai/v1'):
    """Return a cached OpenAI SDK client for the given base URL.

    The cache is per-base_url so different endpoints (Groq vs OpenAI vs
    a local proxy) don't share a client. API key is read from env at
    first-use; rotating the key mid-session requires an app restart.
    """
    if base_url not in _OPENAI_CLIENTS:
        api_key = os.getenv('GROQ_API_KEY') or os.getenv('OPENAI_API_KEY')
        _OPENAI_CLIENTS[base_url] = OpenAI(api_key=api_key, base_url=base_url)
    return _OPENAI_CLIENTS[base_url]


def prewarm_groq_connection():
    """Best-effort HTTPS pre-warm to skip the TLS handshake on first dictation.

    Calls client.models.list() against the Groq endpoint — cheap, exercises
    auth (surfacing a bad API key at startup instead of mid-dictation), and
    primes DNS + TLS + httpx connection pool. Failures are swallowed; pre-
    warm is optional, not a startup blocker. Intended to be fired on a
    daemon thread at app initialization.
    """
    try:
        client = get_openai_client('https://api.groq.com/openai/v1')
        client.models.list()
        ConfigManager.console_print('Groq connection pre-warmed.')
    except Exception as e:
        ConfigManager.console_print(f'Groq pre-warm failed (non-fatal): {e}')


class TranscriptionAPIError(Exception):
    """Raised when the remote transcription API call fails (no internet,
    timeout, empty response). Caught by result_thread.run() so the captured
    audio can be persisted for a Retry."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class GroqBlockedError(TranscriptionAPIError):
    """Raised when Groq's Cloudflare WAF returns 403 — VPN exit IP, rate-
    limited residential IP, etc. Distinct from generic API errors so the
    retry/fallback layer can short-circuit to Gemini instead of paying
    another guaranteed 403."""

# Phrases Whisper is known to hallucinate on silence or near-silence.
# Matched case-insensitively against the stripped transcription.
_WHISPER_HALLUCINATIONS = frozenset({
    'thank you',
    'thank you.',
    'thank you for watching',
    'thank you for watching.',
    'thank you for watching!',
    'thanks for watching.',
    'thanks for watching!',
    'société radio-canada',
    'société radio canada',
    '[ silence ]',
    '[silence]',
    'subtitles by the amara.org community',
    "sous-titres réalisés para la communauté d'amara.org",
})

# Distinctive hallucination strings that are safe to substring-match — these
# never appear in real user speech, so dropping anything that contains them is
# fine. Generic phrases like "thank you" stay exact-match only (the trailing-
# strip below handles the common "real text + trailing thanks" case).
_WHISPER_HALLUCINATION_SUBSTRINGS = (
    'subtitles by the amara.org community',
    "sous-titres réalisés para la communauté d'amara.org",
    'société radio-canada',
    'société radio canada',
    '[silence]',
    '[ silence ]',
)

# Whisper hallucinates in random non-English scripts when fed silence/noise
# (Cyrillic, Arabic, CJK, Devanagari, Korean, Japanese, Hebrew, Thai), and in
# Turkish-specific Latin letters (ş ğ İ ı). Config sets language=en, so any
# character from these ranges in the output is almost certainly a hallucination.
# Common European accented letters (é, ü, ç, ñ, etc.) are intentionally NOT
# included — they appear in legitimate proper nouns and loanwords.
_NON_ENGLISH_SCRIPT_RE = re.compile(
    '['
    'Ѐ-ӿ'   # Cyrillic
    'Ԁ-ԯ'   # Cyrillic Supplement
    '֐-׿'   # Hebrew
    '؀-ۿ'   # Arabic
    '܀-ݏ'   # Syriac
    'ऀ-ॿ'   # Devanagari
    'ঀ-৿'   # Bengali
    '฀-๿'   # Thai
    '぀-ゟ'   # Hiragana
    '゠-ヿ'   # Katakana
    '㐀-䶿'   # CJK Unified Ideographs Extension A
    '一-鿿'   # CJK Unified Ideographs
    '가-힯'   # Hangul
    'şŞğĞıİ'  # Turkish-specific: ş Ş ğ Ğ ı İ
    ']'
)


# Spoken-punctuation normalization moved to engine/polish/spoken_punctuation.py
# (2026-06) so it can be unit-tested without importing this module's heavy
# numpy/openai stack, and shared with the ElevenLabs RegexPolish path. The
# `_normalize_spoken_symbols` name is re-exported (imported at the top of this
# file) for the existing call site in post_process_transcription.


# Matches two bare alphanumeric tokens separated by commas/hyphens (± spaces) or plain spaces.
_ALNUM_MERGE_RE = re.compile(r'([A-Za-z0-9]+)([,\-]\s*|\s+)([A-Za-z0-9]+)')


def _is_code_token(token: str, strict: bool = False) -> bool:
    """True if the token looks like a code/identifier component rather than an English word.

    strict=True (space-only separator) requires a digit or single char.
    strict=False (comma/hyphen separator) also accepts short all-caps strings.
    """
    if any(c.isdigit() for c in token):
        return True
    if len(token) == 1:
        return True
    if not strict and token.isupper() and len(token) <= 4:
        return True
    return False


def _merge_adjacent_alphanumeric(text: str) -> str:
    """Merge adjacent alphanumeric tokens separated by commas, hyphens, or spaces when both
    look like code/identifier components rather than natural-language words.

    Applied iteratively until no more merges are possible, so chains like
    "A, B, C, 1, 2, 3" fully collapse to "ABC123".
    """
    def replacer(m: re.Match) -> str:
        left, sep, right = m.group(1), m.group(2), m.group(3)
        # Space-only separators use stricter criteria to avoid merging e.g. "5 PM"
        strict = ',' not in sep and '-' not in sep
        if _is_code_token(left, strict) and _is_code_token(right, strict):
            return left + right
        return m.group(0)

    prev = None
    while prev != text:
        prev = text
        text = _ALNUM_MERGE_RE.sub(replacer, text)
    return text


def _word_overlap_ratio(source: str, candidate: str) -> float:
    """Fraction of candidate words that appear in source (case-insensitive)."""
    source_words = set(re.findall(r'\b[a-zA-Z]+\b', source.lower()))
    candidate_words = re.findall(r'\b[a-zA-Z]+\b', candidate.lower())
    if not candidate_words:
        return 1.0
    return sum(1 for w in candidate_words if w in source_words) / len(candidate_words)


def create_local_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("faster-whisper is not installed. Install it or switch to API mode (model_options.use_api: true).")

    ConfigManager.console_print('Creating local model...')
    local_model_options = ConfigManager.get_config_section('model_options')['local']
    compute_type = local_model_options['compute_type']
    model_path = local_model_options.get('model_path')

    if compute_type == 'int8':
        device = 'cpu'
        ConfigManager.console_print('Using int8 quantization, forcing CPU usage.')
    else:
        device = local_model_options['device']

    try:
        if model_path:
            ConfigManager.console_print(f'Loading model from: {model_path}')
            model = WhisperModel(model_path, device=device, compute_type=compute_type, download_root=None)
        else:
            model = WhisperModel(local_model_options['model'], device=device, compute_type=compute_type)
    except Exception as e:
        ConfigManager.console_print(f'Error initializing WhisperModel: {e}')
        ConfigManager.console_print('Falling back to CPU.')
        model = WhisperModel(
            model_path or local_model_options['model'],
            device='cpu',
            compute_type=compute_type,
            download_root=None if model_path else None,
        )

    ConfigManager.console_print('Local model created.')
    return model


def transcribe_local(audio_data, local_model=None):
    if not local_model:
        local_model = create_local_model()
    model_options = ConfigManager.get_config_section('model_options')

    audio_data_float = audio_data.astype(np.float32) / 32768.0

    # v0.3.4: language/temperature pinned to safe defaults — used to be
    # user-tunable but no one actually changes them, and auto-detect on
    # language is correct for the user's Hindi/English code-switching.
    # initial_prompt is the active vocabulary hint (auto-grows from the
    # "spelled" feature) so still read from config.
    response = local_model.transcribe(
        audio=audio_data_float,
        language=None,  # auto-detect (was: model_options['common']['language'])
        initial_prompt=ConfigManager.get_config_value('model_options', 'common', 'initial_prompt'),
        condition_on_previous_text=model_options['local']['condition_on_previous_text'],
        temperature=0.0,  # was: model_options['common']['temperature']
        vad_filter=model_options['local']['vad_filter'],
    )
    return ''.join([segment.text for segment in list(response[0])])


_TRANSIENT_API_ERROR_SIGNALS = (
    'connection', 'timeout', 'dns', 'unreachable', 'getaddrinfo',
    'network', 'temporary failure', 'reset', 'stream', 'broken pipe',
)


def _is_transient_api_error(exc: Exception) -> bool:
    """Decide whether an API exception is worth retrying. Mirrors the
    Mobile TranscriberClient.isTransientError heuristic — network-layer
    failures retry; 4xx/auth do not."""
    signal = (type(exc).__name__ + ' ' + (str(exc) or repr(exc))).lower()
    return any(k in signal for k in _TRANSIENT_API_ERROR_SIGNALS)


def _stt_call_budget_sec(audio_data) -> float:
    """Audio-length-aware per-call timeout budget for the STT request.
    Mirrors Mobile's TranscriberClient.callBudgetMs.

        5s clip   → 10s budget
        30s clip  → 15s budget
        60s clip  → 20s budget
        5min clip → 60s budget
        15min clip→ 180s budget (cap)
    """
    sample_rate = ConfigManager.get_config_value('recording_options', 'sample_rate') or 16000
    duration_sec = len(audio_data) / sample_rate
    return min(180.0, 10.0 + duration_sec / 3.0)


def transcribe_api(audio_data):
    """Single Groq/OpenAI STT call. Raises TranscriptionAPIError on
    failure. Caller [transcribe_api_with_retry] handles retry + local
    fallback."""
    # v0.3.4: base_url + model pinned to Groq production values. The legacy
    # user-tunable config (model_options.api.{model,base_url}) was removed
    # because it defaulted to "whisper-1" against api.openai.com which has
    # been wrong for a year. Engine choice is now made via stt_engine.
    base_url = 'https://api.groq.com/openai/v1'

    try:
        client = get_openai_client(base_url)

        byte_io = io.BytesIO()
        sample_rate = ConfigManager.get_config_section('recording_options').get('sample_rate') or 16000
        sf.write(byte_io, audio_data, sample_rate, format='wav')
        byte_io.seek(0)

        response = client.audio.transcriptions.create(
            model='whisper-large-v3-turbo',
            file=('audio.wav', byte_io, 'audio/wav'),
            language=None,  # auto-detect
            prompt=ConfigManager.get_config_value('model_options', 'common', 'initial_prompt'),
            temperature=0.0,
            timeout=_stt_call_budget_sec(audio_data),
        )
    except Exception as e:
        cls = type(e).__name__
        msg = str(e) or repr(e)
        status = getattr(e, 'status_code', None)
        # Cloudflare WAF returns 403 to VPN exit IPs (and some rate-limited
        # residential IPs) before the request even reaches Groq. The openai
        # SDK surfaces this as PermissionDeniedError or a generic APIStatus
        # Error with status_code=403. Cache the verdict so the next call
        # skips Groq entirely + raise the distinct subclass.
        if status == 403 or 'PermissionDenied' in cls or 'Forbidden' in msg:
            blocked_cache.mark_blocked()
            raise GroqBlockedError(f'Groq returned 403 (WAF block — likely VPN exit IP)') from e
        if _is_transient_api_error(e):
            raise TranscriptionAPIError(f'No internet connection ({cls})') from e
        raise TranscriptionAPIError(f'API error ({cls}): {msg}') from e

    # A successful response means the network is fine and Groq accepted us;
    # clear any stale "blocked" verdict so we don't keep routing to Gemini
    # after the user toggled VPN off.
    blocked_cache.clear_blocked()
    text = response.text or ''
    if not text.strip():
        raise TranscriptionAPIError('Empty response from transcription API')
    return text


def transcribe_api_with_retry(audio_data, max_attempts: int = 2) -> Tuple[str, bool]:
    """Run the API STT pipeline. Returns `(transcript, used_gemini_fallback)`.

    Pre-flight: if a VPN adapter is up *or* a recent Groq probe returned 403,
    skip Groq entirely and route to Gemini's audio endpoint. Saves a
    guaranteed-403 round-trip + sidesteps the watchdog stall.

    Otherwise, up to N attempts against Groq. Transient failures retry;
    GroqBlockedError immediately falls through to Gemini (no retry). Mirrors
    Mobile's TranscriberClient.transcribeOneWithRetry + DictationPipeline.run
    routing logic.
    """
    on_vpn = blocked_cache.is_on_vpn()
    cached_block = blocked_cache.is_likely_blocked()
    if (on_vpn or cached_block) and gemini_stt.is_configured():
        reason = blocked_cache.reason_label(on_vpn, cached_block)
        ConfigManager.console_print(
            f'STT: routing to Gemini pre-flight (reason={reason})'
        )
        return _gemini_transcribe(audio_data), True

    last_exc: Optional[TranscriptionAPIError] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return transcribe_api(audio_data), False
        except GroqBlockedError as e:
            if gemini_stt.is_configured():
                ConfigManager.console_print(
                    f'STT: Groq returned 403 ({e.reason}); routing to Gemini'
                )
                return _gemini_transcribe(audio_data), True
            # No Gemini key → propagate the block error so the caller can
            # decide (e.g. fall back to local Whisper).
            raise
        except TranscriptionAPIError as e:
            last_exc = e
            if attempt >= max_attempts:
                break
            if not _is_transient_api_error(e.__cause__ or e):
                break
            ConfigManager.console_print(
                f'STT attempt {attempt} failed ({e.reason}); retrying after warm.'
            )
            try:
                from threading import Thread
                Thread(target=prewarm_groq_connection, daemon=True).start()
            except Exception:
                pass
    raise last_exc or TranscriptionAPIError('Unknown STT failure')


def _gemini_transcribe(audio_data) -> str:
    """Shared Gemini-STT entry — wraps GeminiSttError in TranscriptionAPIError
    so the caller's single except branch handles both providers."""
    sample_rate = ConfigManager.get_config_value('recording_options', 'sample_rate') or 16000
    try:
        return gemini_stt.transcribe(audio_data, sample_rate)
    except gemini_stt.GeminiSttError as e:
        raise TranscriptionAPIError(f'Gemini STT fallback failed: {e}') from e


def llm_polish(transcription, engine: str = 'groq+llama'):
    config = ConfigManager.get_config_section('llm_polish')
    if not config.get('enabled') or not transcription.strip():
        return transcription

    api_key = os.getenv('GROQ_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not api_key:
        ConfigManager.console_print('LLM polish skipped: GROQ_API_KEY not set.')
        return transcription

    system_prompt = config.get('system_prompt')
    if not system_prompt:
        ConfigManager.console_print('LLM polish skipped: no system_prompt configured.')
        return transcription

    system_prompt = substitute_proper_nouns(system_prompt, config.get('proper_nouns'))

    try:
        client = get_openai_client(config.get('base_url') or 'https://api.groq.com/openai/v1')
        # gpt-oss-* family is a reasoning model on Groq; without reasoning_effort
        # it burns tokens on chain-of-thought and runs slower. 'low' is enough
        # for the mechanical polish task — verified empirically (Section 7 of
        # whisper-polish-deep-dive.md): output tokens dropped ~60% with no
        # quality loss. Llama models on Groq don't accept the param, so pass
        # it only for gpt-oss-*.
        extra_kwargs = {}
        model_name = config.get('model') or ''
        if model_name.startswith('openai/gpt-oss'):
            extra_kwargs['reasoning_effort'] = config.get('reasoning_effort') or 'low'
        # Polish payloads are tiny JSON; 20s is plenty for a healthy Groq
        # response. Past that, PostLlmRepair's reject path will paste the
        # cleaned raw transcript rather than waiting indefinitely.
        response = client.chat.completions.create(
            model=config['model'],
            max_tokens=config.get('max_tokens') or 1024,
            temperature=config.get('temperature', 0.2),
            timeout=20.0,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f'[TRANSCRIPT]\n{transcription}\n[/TRANSCRIPT]'},
            ],
            **extra_kwargs,
        )
        polished = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        ConfigManager.console_print(f'LLM polish: raw="{transcription.strip()}" → polished="{polished}"')

        # Detect silent token-budget truncation. With reasoning models the
        # hidden reasoning tokens share max_tokens with the visible output, so
        # long dictations can hit the cap and return mid-sentence. finish_reason
        # 'length' means the response was cut short by max_tokens.
        if finish_reason == 'length':
            ConfigManager.console_print(
                f'LLM polish TRUNCATED by max_tokens (finish_reason=length). '
                f'raw_chars={len(transcription)} polished_chars={len(polished or "")}'
            )
            try:
                from dict_diag import dd
                dd('polish.truncated_by_max_tokens', polished,
                   raw_chars=len(transcription),
                   polished_chars=len(polished or ''))
            except Exception:
                pass

        # Engine label is passed in from `post_process_transcription`; the
        # Groq + Llama-polish chain is the default. Other engines (Gemini
        # fallback, local-Whisper fallback, ElevenLabs) write their own log
        # entries from their own code paths.
        _write_log_entry(transcription, polished, engine)

        # Delegate post-LLM safety checks to the shared repair module so
        # Mobile + PC stay in lockstep. The module strips wrapper tags,
        # <thinking>/<reasoning>/<analysis> blocks, surrounding quotes, and
        # leading preambles; recognises the __EMPTY__ sentinel from the v3b
        # prompt; and rejects via Levenshtein-ratio + word-overlap + bad-
        # first-token checks. On rejection it returns the raw transcript.
        repair = apply_post_llm_repair(transcription, polished)

        # Auto-add from spelling: when the polish prompt's SPELLING
        # (dictionary add) rule fires on an explicit "spelled" trigger, it
        # emits a <<DICT_ADD: ...>> tail marker that PostLlmRepair extracts
        # into repair.dict_additions. Persist those words (gated by the
        # master kill-switch) so future dictations pick them up via the STT
        # vocabulary hint and the polish PROPER NOUNS active-correction list.
        # Runs even when polish was rejected -- the marker is independent
        # signal that the trigger word fired and the word was captured.
        if repair.dict_additions and config.get('enable_dict_autoadd_from_spelling', True):
            try:
                _persist_dict_additions(repair.dict_additions)
            except Exception as e:
                ConfigManager.console_print(f'Dict auto-add failed: {e}')

        if repair.polish_rejected:
            ConfigManager.console_print(
                f'LLM polish rejected ({repair.rejection_reason}); returning raw.'
            )
            try:
                from dict_diag import dd
                dd('polish.rejected_return_raw', transcription,
                   reason=repair.rejection_reason,
                   similarity=repair.similarity,
                   overlap=repair.overlap)
            except Exception:
                pass
            return transcription

        try:
            from dict_diag import dd
            dd('polish.final_text', repair.final_text,
               similarity=repair.similarity,
               overlap=repair.overlap)
        except Exception:
            pass
        return repair.final_text
    except Exception as e:
        ConfigManager.console_print(f'LLM polish error (returning raw transcription): {e}')
        return transcription


def _persist_dict_additions(words):
    """Append the captured spelled words to the STT vocabulary hint and the
    polish proper-nouns 'people' list (default category), de-duplicating
    case-insensitively against existing entries. Persist to disk and fire the
    notification event for the tray balloon.
    """
    if not words:
        return

    current_hint = ConfigManager.get_config_value('model_options', 'common', 'initial_prompt') or ''
    hint_seen = {w.strip().lower() for w in current_hint.split(',') if w.strip()}

    pn_section = ConfigManager.get_config_value('llm_polish', 'proper_nouns') or {}
    if not isinstance(pn_section, dict):
        pn_section = {}
    people = list(pn_section.get('people') or [])
    people_seen = {
        (e.get('word') or '').strip().lower()
        for e in people
        if isinstance(e, dict)
    }

    added_to_hint = []
    added_to_people = []
    for word in words:
        w = (word or '').strip()
        if not w:
            continue
        lower = w.lower()
        if lower not in hint_seen:
            added_to_hint.append(w)
            hint_seen.add(lower)
        if lower not in people_seen:
            added_to_people.append({'word': w, 'misheard': []})
            people_seen.add(lower)

    if not added_to_hint and not added_to_people:
        return  # everything was already in the dictionary

    if added_to_hint:
        new_hint = current_hint.rstrip()
        if new_hint and not new_hint.endswith(','):
            new_hint += ', '
        elif new_hint:
            new_hint += ' '
        new_hint += ', '.join(added_to_hint)
        ConfigManager.set_config_value(new_hint, 'model_options', 'common', 'initial_prompt')

    if added_to_people:
        people.extend(added_to_people)
        pn_section['people'] = people
        ConfigManager.set_config_value(pn_section, 'llm_polish', 'proper_nouns')

    ConfigManager.save_config()

    visible_added = added_to_hint or [e['word'] for e in added_to_people]
    ConfigManager.console_print(f"Auto-added to dictionary: {', '.join(visible_added)}")
    fire_dict_addition(visible_added)


def post_process_transcription(transcription, skip_polish: bool = False,
                               engine: str = 'groq+llama'):
    transcription = transcription.strip()

    transcription = _normalize_spoken_symbols(transcription)

    # LLM polish runs on the raw stripped transcript, before whitespace/case
    # tweaks. Skipped when the STT layer used the Gemini fallback — Groq
    # polish would also 403 (same Cloudflare WAF block) and Gemini transcripts
    # arrive already-punctuated, so the polish round-trip would be pure cost.
    if not skip_polish:
        transcription = llm_polish(transcription, engine=engine)
    else:
        # Paths that skip LLM polish (Gemini fallback, local-Whisper fallback)
        # still log so the user can audit engine usage from transcript_log.txt.
        # POLISHED is identical to RAW for these because no LLM step ran.
        _write_log_entry(transcription, transcription, engine)
    transcription = _merge_adjacent_alphanumeric(transcription)

    # v0.3.4: remove_trailing_period and remove_capitalization were removed
    # from the Settings UI; they're niche transforms and the cleaner code path
    # is to just default-off. add_trailing_space stays user-facing.
    post_processing = ConfigManager.get_config_section('post_processing')
    if post_processing.get('add_trailing_space', True):
        transcription += ' '

    return transcription


def _transcribe_via_elevenlabs(audio_data):
    """v0.3 PC path: ElevenLabs Scribe v2 Realtime + client-side RegexPolish.

    No LLM polish. ElevenLabs handles capitalisation, punctuation, numbers,
    fillers, Hindi natively. RegexPolish (in engine/polish/regex_polish.py)
    handles the proper-noun mishears, spoken-punctuation, NATO collapse,
    spelling capture, scratch-that, and email shorthand.

    Spelled-out proper nouns ("Aniket spelled A-N-I-K-E-T") get appended
    to the same proper_nouns config the LLM path uses, so the keyterms
    list grows automatically over time.

    Input: numpy int16 PCM array (16kHz mono).
    Output: final polished text string.
    """
    from engine.stt import elevenlabs_rt
    from engine.polish import regex_polish

    # Empty / silent guard — mirror the heuristics from the Groq path.
    rms = float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))
    if rms < 30:
        ConfigManager.console_print(f'Audio signal absent (RMS={rms:.0f}), skipping transcription.')
        return ''
    sample_rate = ConfigManager.get_config_value('recording_options', 'sample_rate') or 16000
    duration = len(audio_data) / sample_rate
    if duration < 0.8 and rms < 200:
        ConfigManager.console_print(
            f'Audio too short and quiet ({duration:.2f}s, RMS={rms:.0f}); skipping transcription.'
        )
        return ''

    api_key = os.getenv('ELEVENLABS_API_KEY') or ConfigManager.get_config_value('model_options', 'elevenlabs_api_key')
    if not api_key:
        ConfigManager.console_print('ElevenLabs API key not set; cannot use ElevenLabs RT engine.')
        raise TranscriptionAPIError('ElevenLabs API key not set')

    # Build keyterms list from the structured proper_nouns config (mirrors
    # how the mobile app does it). Max 50 × 20 chars per ElevenLabs limits.
    keyterms = _build_elevenlabs_keyterms()

    # v0.3 PC initial port: record-then-burst (same as v0.2 mobile). True
    # streaming-during-recording follows once the burst path is dogfooded.
    pcm_bytes = _audio_data_to_pcm16_bytes(audio_data)
    t0 = time.monotonic()
    result = elevenlabs_rt.transcribe_burst(pcm_bytes, api_key, keyterms)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if result.get('error'):
        ConfigManager.console_print(f'ElevenLabs RT failed: {result["error"]}')
        # v0.3.1: classify the failure so the caller (result_thread) can show
        # a short user-facing chip ("No internet", "Network slow", etc.)
        category = _classify_elevenlabs_failure(result['error'])
        raise TranscriptionAPIError(f"ElevenLabs RT: {result['error']} [{category}]")

    raw_text = result.get('text') or ''
    ConfigManager.console_print(f'ElevenLabs RT in {elapsed_ms}ms: "{raw_text.strip()}"')
    return _finalize_elevenlabs_transcript(raw_text, api_key)


def _finalize_elevenlabs_transcript(raw_text: str, api_key: str,
                                     engine: str = 'elevenlabs-burst') -> str:
    """Post-STT polish + persistence for the ElevenLabs path.

    Shared between record-then-burst (`_transcribe_via_elevenlabs`, engine =
    'elevenlabs-burst') and the streaming-during-recording path
    (`transcribe_streaming_result`, engine = 'elevenlabs-stream'). Runs
    RegexPolish, persists any auto-added proper nouns, writes the
    transcript-log entry tagged with the engine, schedules the server-side
    history-delete sweep, and returns the polished final text.
    """
    from engine.polish import regex_polish
    from engine.polish import devanagari_translit

    # Devanagari → Latin transliteration safety net. ElevenLabs is pinned to
    # language_code=eng which should romanize Hindi loanwords at the source;
    # this pass catches any Devanagari that slips through (e.g. mid-sentence
    # code-switch the language pin missed). No-op when the transcript has no
    # Devanagari at all (the dominant case).
    raw_text = devanagari_translit.transliterate(raw_text)

    polish_result = regex_polish.apply(raw_text, toggles=regex_polish.Toggles.from_config())

    if polish_result.dict_additions:
        autoadd_enabled = ConfigManager.get_config_value('llm_polish', 'enable_dict_autoadd_from_spelling')
        if autoadd_enabled is not False:  # default True
            try:
                _persist_dict_additions(polish_result.dict_additions)
            except Exception as e:
                ConfigManager.console_print(f'dict_add persistence failed: {e}')

    # Log the entry with engine info so the user can audit which backend ran.
    # Without this the ElevenLabs path was leaving transcript_log.txt empty.
    _write_log_entry(raw_text, polish_result.final_text, engine)

    try:
        from engine.retention import history_delete_worker
        history_delete_worker.schedule_one_shot(api_key)
    except Exception:
        pass

    return polish_result.final_text


def transcribe_streaming_result(raw_text: str) -> str:
    """Pipeline entry point for the streaming-during-recording path.

    The Session in `result_thread._record_audio` already delivered the
    finalised text from ElevenLabs RT. All we need to do is run the same
    post-STT polish + persistence that the burst path runs. No STT call here.

    Returns the polished text (empty if input was empty).
    """
    if not raw_text or not raw_text.strip():
        return ''
    api_key = os.getenv('ELEVENLABS_API_KEY') or ConfigManager.get_config_value('model_options', 'elevenlabs_api_key') or ''
    return _finalize_elevenlabs_transcript(raw_text, api_key, engine='elevenlabs-stream')


def _classify_elevenlabs_failure(reason: str) -> str:
    """Map a raw failure reason to a 2-3 word user-facing label.

    Mirrors mobile's classifyFailureReason() in WhisperAccessibilityService.kt.
    """
    if not reason:
        return "Transcription failed"
    # Network up at all? socket.gethostbyname is a cheap-ish probe.
    try:
        import socket
        socket.gethostbyname("api.elevenlabs.io")
        online = True
    except Exception:
        online = False
    if not online:
        return "No internet"
    lower = reason.lower()
    if "timed out" in lower or "timeout" in lower:
        return "Network slow"
    if "ws closed" in lower or "ws failure" in lower or "connection closed" in lower:
        return "Connection dropped"
    if "http 5" in lower or "internal_error" in lower or "service unavailable" in lower:
        return "Server error"
    if "http 4" in lower or "rate limit" in lower or "quota" in lower:
        return "API limit / auth"
    if "api key" in lower:
        return "Key not set"
    if "blocked" in lower or "vpn" in lower:
        return "Blocked (VPN?)"
    if "empty" in lower:
        return "No speech detected"
    return "Transcription failed"


def _audio_data_to_pcm16_bytes(audio_data) -> bytes:
    """Convert numpy int16 array to raw PCM16 LE bytes."""
    if audio_data.dtype != np.int16:
        # sounddevice can produce float32; convert to int16
        clipped = np.clip(audio_data, -1.0, 1.0)
        audio_data = (clipped * 32767).astype(np.int16)
    return audio_data.tobytes()


def _build_elevenlabs_keyterms() -> list[str]:
    """Flatten the structured proper_nouns config into a keyterms list.

    Order: locations, people, products. Filter to ≤20 chars per ElevenLabs's
    documented limit; cap total at 50.
    """
    pn = ConfigManager.get_config_value('llm_polish', 'proper_nouns') or {}
    terms: list[str] = []
    for category in ('locations', 'people', 'products'):
        entries = pn.get(category) or []
        for e in entries:
            w = (e.get('word') if isinstance(e, dict) else str(e)).strip()
            if w and len(w) <= 20:
                terms.append(w)
    # Dedup preserving order, cap at 50.
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= 50:
            break
    return out


def transcribe(audio_data, local_model=None, force_groq: bool = False):
    if audio_data is None:
        return ''

    # v0.3 routing: check the STT engine pref. If set to elevenlabs, route to
    # the Realtime + RegexPolish path. Default stays groq so v0.1/v0.2 behaviour
    # is preserved until the user opts in.
    # v0.3.2 PC: force_groq=True lets callers (specifically result_thread when
    # the streaming session has just failed) skip the ElevenLabs branch and go
    # straight to Groq. Without this, a failed streaming session falls back to
    # ElevenLabs *burst* on the same audio — same provider, same failure mode.
    stt_engine = ConfigManager.get_config_value('model_options', 'stt_engine') or 'groq'
    if not force_groq and stt_engine == 'elevenlabs':
        # ElevenLabs Realtime is a streaming endpoint; bursting a long
        # pre-recorded buffer floods its bounded server-side queue
        # (queue_overflow — see failed_log.txt on multi-minute audio). The live
        # streaming path (result_thread) is fine because it sends in real time,
        # but this record-then-burst fallback must cap length and hand long
        # audio to Groq's file endpoint (robust at any length) instead.
        sample_rate = ConfigManager.get_config_value('recording_options', 'sample_rate') or 16000
        duration = len(audio_data) / sample_rate
        if duration <= ELEVENLABS_BURST_MAX_S:
            return _transcribe_via_elevenlabs(audio_data)
        ConfigManager.console_print(
            f'Audio {duration:.0f}s exceeds ElevenLabs burst cap ({ELEVENLABS_BURST_MAX_S:.0f}s); '
            f'using Groq to avoid queue_overflow'
        )
        # fall through to the Groq path below (local_model fallback preserved)

    # Skip STT only on a completely dead signal (muted mic, no input device).
    # Threshold is intentionally very low — only catches zero/near-zero input, not quiet speech.
    rms = float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))
    if rms < 30:
        ConfigManager.console_print(f'Audio signal absent (RMS={rms:.0f}), skipping transcription.')
        return ''

    # Short + quiet clips are the prime hallucination zone: Whisper invents
    # words from <1s of low-energy audio. Drop before the API call.
    sample_rate = ConfigManager.get_config_value('recording_options', 'sample_rate') or 16000
    duration = len(audio_data) / sample_rate
    if duration < 0.8 and rms < 200:
        ConfigManager.console_print(
            f'Audio too short and quiet ({duration:.2f}s, RMS={rms:.0f}); skipping transcription.'
        )
        return ''

    used_gemini_fallback = False
    used_local_fallback = False
    if ConfigManager.get_config_value('model_options', 'use_api'):
        try:
            transcription, used_gemini_fallback = transcribe_api_with_retry(audio_data)
        except TranscriptionAPIError as e:
            # Fall back to the local faster-whisper model when both API
            # attempts fail with a transient error AND a local model was
            # eagerly loaded for fallback (see main.py / model_options.
            # enable_local_fallback). User-visible feedback is left to
            # result_thread which sees the resulting transcription.
            allow_fallback = ConfigManager.get_config_value('model_options', 'enable_local_fallback')
            if local_model is not None and allow_fallback and _is_transient_api_error(e.__cause__ or e):
                ConfigManager.console_print(
                    f'STT API exhausted retries ({e.reason}); falling back to local Whisper.'
                )
                transcription = transcribe_local(audio_data, local_model)
                used_local_fallback = True
            else:
                raise
    else:
        transcription = transcribe_local(audio_data, local_model)
        used_local_fallback = True

    ConfigManager.console_print(f'Whisper output: "{transcription.strip()}"')

    stripped_lower = transcription.strip().lower()

    # Discard known Whisper hallucinations produced on silence.
    if stripped_lower in _WHISPER_HALLUCINATIONS:
        ConfigManager.console_print(f'Whisper hallucination discarded (exact): "{transcription.strip()}"')
        return ''

    # Distinctive hallucination phrases (Amara, Société Radio-Canada, [silence])
    # never appear in real speech — drop on substring match.
    for needle in _WHISPER_HALLUCINATION_SUBSTRINGS:
        if needle in stripped_lower:
            ConfigManager.console_print(f'Whisper hallucination discarded (substring "{needle}"): "{transcription.strip()}"')
            return ''

    # v0.3.4: the non-English-script hallucination check used to be gated on
    # an explicit language='en' config (so Hindi-English users wouldn't trip
    # it). Now the user is on auto-detect; we drop the check entirely. A real
    # non-English transcription will just pass through, which is what they
    # want for Hindi-English code switching anyway.

    # Strip trailing "Thank you" appended by Whisper at the end of real transcriptions.
    stripped = re.sub(r'[,]?\s*\bthank you[.!]?\s*$', '', transcription, flags=re.IGNORECASE).strip()
    if stripped != transcription.strip():
        ConfigManager.console_print(f'Trailing thank-you stripped: "{transcription.strip()}" → "{stripped}"')
        if not stripped:
            return ''
        transcription = stripped

    # Engine label resolution for the log entry: Gemini and local-Whisper
    # fallbacks each replace the STT half but keep the same Llama-polish stage
    # *unless* skip_polish fires. We arrow-chain the names so an audit reader
    # can see the actual path (e.g. `groq→gemini` vs straight `groq+llama`).
    if used_local_fallback:
        engine_label = 'groq→local-whisper-fallback'
    elif used_gemini_fallback:
        engine_label = 'groq→gemini-fallback'
    else:
        engine_label = 'groq+llama'
    result = post_process_transcription(
        transcription,
        skip_polish=used_gemini_fallback,
        engine=engine_label,
    )
    try:
        from dict_diag import dd
        dd('transcribe.return', result, used_gemini=used_gemini_fallback,
           engine=engine_label)
    except Exception:
        pass
    return result
