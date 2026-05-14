import io
import os
import re
from typing import Optional
import numpy as np
import soundfile as sf
from openai import OpenAI

from utils import ConfigManager
from engine.polish.post_llm_repair import apply as apply_post_llm_repair
from engine.polish.proper_nouns_renderer import substitute as substitute_proper_nouns
from notifications import fire_dict_addition


# Cache of OpenAI SDK clients keyed by base_url. The SDK uses httpx with
# connection pooling internally, so sharing the client across calls reuses
# the TLS connection and skips the handshake (~200-400ms) on every polish
# or transcribe call after the first.
_OPENAI_CLIENTS: dict = {}


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


# Spoken-punctuation dictation table. Each tuple is
#   (phrase_regex, symbol, side, safe_inline)
#
# `side` controls inline-replacement spacing:
#   'L' = opening punctuation ([({") — keep leading whitespace, drop trailing
#   'R' = closing/sentence-ending punctuation (.,;:!?]}) — drop leading, keep trailing
#   'B' = bidirectional inline glyph (/ \ @ # = * - ...) — drop both spaces
#
# `safe_inline` gates the inline + end-of-utterance patterns. False means the
# spoken word is too often legitimate content (e.g. "period of rest") to risk
# replacing in those positions; the LLM SYMBOLS rule handles those cases.
# The Whisper double-render pattern (symbol on both sides of the word) fires
# regardless — it's near-zero false-positive because Whisper would never
# spontaneously produce ". period." or ", colon," around legitimate content.
#
# Order matters: longer/more-specific phrases first so "exclamation mark"
# wins over a hypothetical "exclamation", and "open parenthesis" wins over
# "open paren".
_SPOKEN_PUNCT = [
    (r"new\s+paragraph",                            "[blank line]", "B", True),
    (r"new\s+line",                                 "[newline]",    "B", True),
    (r"exclamation\s+(?:mark|point)",               "!",            "R", True),
    (r"question\s+mark",                            "?",            "R", True),
    (r"open\s+parenthesis|open\s+paren",            "(",            "L", True),
    (r"close\s+parenthesis|close\s+paren",          ")",            "R", True),
    (r"open\s+bracket",                             "[",            "L", True),
    (r"close\s+bracket",                            "]",            "R", True),
    (r"open\s+curly(?:\s+brace)?|open\s+brace",     "{",            "L", True),
    (r"close\s+curly(?:\s+brace)?|close\s+brace",   "}",            "R", True),
    (r"end\s+quote",                                '"',            "R", True),
    (r"semi[\s-]?colon",                            ";",            "R", True),
    (r"forward\s+slash",                            "/",            "B", True),
    (r"back[\s-]?slash",                            "\\",           "B", True),
    (r"at\s+(?:sign|symbol)",                       "@",            "B", True),
    (r"hash\s+(?:sign|tag)|hashtag",                "#",            "B", True),
    (r"equals\s+sign",                              "=",            "B", True),
    (r"ellipsis",                                   "...",          "R", True),
    (r"asterisk",                                   "*",            "B", True),
    (r"hyphen",                                     "-",            "B", True),
    (r"comma",                                      ",",            "R", True),
    (r"full[\s-]?stop",                             ".",            "R", True),
    # Risky inline matches — these spoken words are commonly legitimate content
    # ("period of rest", "made a dash", "colon cancer"). Only fire on a Whisper
    # double-render, which Whisper would never produce around real content.
    (r"period",                                     ".",            "R", False),
    (r"colon",                                      ":",            "R", False),
    (r"dash",                                       "-",            "B", False),
    (r"slash",                                      "/",            "B", False),
    (r"hash",                                       "#",            "B", False),
    (r"equals",                                     "=",            "B", False),
    (r"quote",                                      '"',            "L", False),
    (r"star",                                       "*",            "B", False),
]


def _normalize_spoken_symbols(text: str) -> str:
    """Convert spoken punctuation commands to their symbols when surrounding
    context indicates the word is a dictation command rather than content.

    Three patterns fire for safe phrases:
      1. Whisper double-render: ``<sym><phrase><sym>`` → ``<sym>``
         (model inserted both the symbol and the spoken word around it).
      2. Inline mid-sentence: ``<word> <phrase> <word>`` → ``<word><sym><word>``
         (clear command position, between content words).
      3. End-of-utterance: ``<word> <phrase>[. ! ?]?$`` → ``<word><sym>``
         (final word, optionally followed by Whisper's auto terminator).

    For risky phrases (period, colon, dash, etc. — commonly content words)
    only pattern 1 fires; the LLM SYMBOLS rule cleans up the rest.

    Patterns require word boundaries plus ``\\s+`` separators around the
    phrase, so embedded usages ("uncommon", "full-stopping") are safe.
    """
    # Use callable replacements throughout: replacement strings interpret \1,
    # \\, and similar backreferences, which collides with symbols like \ or
    # tokens like [newline].
    for phrase, sym, side, safe in _SPOKEN_PUNCT:
        sym_esc = re.escape(sym)
        phrase_grouped = f"(?:{phrase})"

        # 1. Whisper double-render. Always safe — Whisper does not surround
        #    legitimate content with redundant punctuation.
        text = re.sub(
            rf'{sym_esc}\s*\b{phrase_grouped}\b\s*{sym_esc}',
            lambda m, s=sym: s,
            text,
            flags=re.IGNORECASE,
        )

        if not safe:
            continue

        if side == "L":
            # Inline: keep leading space, drop trailing: "x open paren y" → "x (y"
            text = re.sub(
                rf'(?<=\w)(\s+)\b{phrase_grouped}\b\s+(?=\w)',
                lambda m, s=sym: m.group(1) + s,
                text,
                flags=re.IGNORECASE,
            )
            # Start-of-utterance: "open bracket 5 ..." → "[5 ..."
            text = re.sub(
                rf'^\s*\b{phrase_grouped}\b\s+(?=\w)',
                lambda m, s=sym: s,
                text,
                flags=re.IGNORECASE,
            )
        elif side == "R":
            # Drop leading space, keep trailing space: "x comma y" → "x, y"
            text = re.sub(
                rf'(?<=\w)\s+\b{phrase_grouped}\b(?=\s+\w)',
                lambda m, s=sym: s,
                text,
                flags=re.IGNORECASE,
            )
            # End-of-utterance, optional Whisper-inserted terminator absorbed.
            text = re.sub(
                rf'(?<=\w)\s+\b{phrase_grouped}\b\s*[.!?]?\s*$',
                lambda m, s=sym: s,
                text,
                flags=re.IGNORECASE,
            )
        else:  # 'B'
            # Drop both spaces: "x hash y" → "x#y"
            text = re.sub(
                rf'(?<=\w)\s+\b{phrase_grouped}\b\s+(?=\w)',
                lambda m, s=sym: s,
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                rf'(?<=\w)\s+\b{phrase_grouped}\b\s*$',
                lambda m, s=sym: s,
                text,
                flags=re.IGNORECASE,
            )

    return text


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

    response = local_model.transcribe(
        audio=audio_data_float,
        language=model_options['common']['language'],
        initial_prompt=model_options['common']['initial_prompt'],
        condition_on_previous_text=model_options['local']['condition_on_previous_text'],
        temperature=model_options['common']['temperature'],
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
    model_options = ConfigManager.get_config_section('model_options')
    base_url = model_options['api']['base_url'] or 'https://api.openai.com/v1'

    try:
        client = get_openai_client(base_url)

        byte_io = io.BytesIO()
        sample_rate = ConfigManager.get_config_section('recording_options').get('sample_rate') or 16000
        sf.write(byte_io, audio_data, sample_rate, format='wav')
        byte_io.seek(0)

        response = client.audio.transcriptions.create(
            model=model_options['api']['model'],
            file=('audio.wav', byte_io, 'audio/wav'),
            language=model_options['common']['language'],
            prompt=model_options['common']['initial_prompt'],
            temperature=model_options['common']['temperature'],
            timeout=_stt_call_budget_sec(audio_data),
        )
    except Exception as e:
        cls = type(e).__name__
        msg = str(e) or repr(e)
        if _is_transient_api_error(e):
            raise TranscriptionAPIError(f'No internet connection ({cls})') from e
        raise TranscriptionAPIError(f'API error ({cls}): {msg}') from e

    text = response.text or ''
    if not text.strip():
        raise TranscriptionAPIError('Empty response from transcription API')
    return text


def transcribe_api_with_retry(audio_data, max_attempts: int = 2):
    """Up-to-N STT attempts; only transient failures retry. Between
    attempts, fire prewarm_groq_connection on a daemon thread so the next
    try benefits from a freshly-established TLS session. Mirrors Mobile's
    TranscriberClient.transcribeOneWithRetry."""
    last_exc: Optional[TranscriptionAPIError] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return transcribe_api(audio_data)
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


def llm_polish(transcription):
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
        ConfigManager.console_print(f'LLM polish: raw="{transcription.strip()}" → polished="{polished}"')

        try:
            import datetime
            log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'transcript_log.txt')
            with open(log_path, 'a', encoding='utf-8') as f:
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f'[{ts}]\n  RAW:     {transcription.strip()}\n  POLISHED: {polished}\n\n')
        except Exception:
            pass

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
            return transcription

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


def post_process_transcription(transcription):
    transcription = transcription.strip()

    transcription = _normalize_spoken_symbols(transcription)

    # LLM polish runs on the raw stripped transcript, before whitespace/case tweaks
    transcription = llm_polish(transcription)
    transcription = _merge_adjacent_alphanumeric(transcription)

    post_processing = ConfigManager.get_config_section('post_processing')
    if post_processing['remove_trailing_period'] and transcription.endswith('.'):
        transcription = transcription[:-1]
    if post_processing['add_trailing_space']:
        transcription += ' '
    if post_processing['remove_capitalization']:
        transcription = transcription.lower()

    return transcription


def transcribe(audio_data, local_model=None):
    if audio_data is None:
        return ''

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

    if ConfigManager.get_config_value('model_options', 'use_api'):
        try:
            transcription = transcribe_api_with_retry(audio_data)
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
            else:
                raise
    else:
        transcription = transcribe_local(audio_data, local_model)

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

    # Non-English script in the output (Cyrillic, Arabic, CJK, Turkish-specific
    # Latin, etc.) when language is configured as English — Whisper hallucinated.
    language = ConfigManager.get_config_value('model_options', 'common', 'language')
    if language == 'en' and _NON_ENGLISH_SCRIPT_RE.search(transcription):
        ConfigManager.console_print(f'Whisper hallucination discarded (non-English script): "{transcription.strip()}"')
        return ''

    # Strip trailing "Thank you" appended by Whisper at the end of real transcriptions.
    stripped = re.sub(r'[,]?\s*\bthank you[.!]?\s*$', '', transcription, flags=re.IGNORECASE).strip()
    if stripped != transcription.strip():
        ConfigManager.console_print(f'Trailing thank-you stripped: "{transcription.strip()}" → "{stripped}"')
        if not stripped:
            return ''
        transcription = stripped

    return post_process_transcription(transcription)
