import io
import os
import re
import numpy as np
import soundfile as sf
from openai import OpenAI

from utils import ConfigManager


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


# Whisper double-insert: model inserts punctuation comma AND keeps spoken word "comma"
_SPOKEN_COMMA_DOUBLE = re.compile(r',\s*comma\s*,', re.IGNORECASE)
# Inline spoken command: "word comma word" with no surrounding commas
_SPOKEN_COMMA_INLINE = re.compile(r'(?<=\w)\s+comma(?=\s+\w)', re.IGNORECASE)


def _normalize_spoken_symbols(text: str) -> str:
    text = _SPOKEN_COMMA_DOUBLE.sub(',', text)
    text = _SPOKEN_COMMA_INLINE.sub(',', text)
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


def transcribe_api(audio_data):
    model_options = ConfigManager.get_config_section('model_options')
    # Prefer GROQ_API_KEY; fall back to OPENAI_API_KEY for vanilla OpenAI usage
    api_key = os.getenv('GROQ_API_KEY') or os.getenv('OPENAI_API_KEY') or None

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=model_options['api']['base_url'] or 'https://api.openai.com/v1',
        )

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
        )
    except Exception as e:
        cls = type(e).__name__
        msg = str(e) or repr(e)
        signal = (cls + ' ' + msg).lower()
        if any(k in signal for k in ('connection', 'timeout', 'dns', 'unreachable', 'getaddrinfo', 'network', 'temporary failure')):
            raise TranscriptionAPIError(f'No internet connection ({cls})') from e
        raise TranscriptionAPIError(f'API error ({cls}): {msg}') from e

    text = response.text or ''
    if not text.strip():
        raise TranscriptionAPIError('Empty response from transcription API')
    return text


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

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=config.get('base_url') or 'https://api.groq.com/openai/v1',
        )
        response = client.chat.completions.create(
            model=config['model'],
            max_tokens=config.get('max_tokens') or 1024,
            temperature=config.get('temperature', 0.2),
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f'[TRANSCRIPT]\n{transcription}\n[/TRANSCRIPT]'},
            ],
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

        # Strip accidental [TRANSCRIPT] / [/TRANSCRIPT] tags the model may echo
        polished = polished.replace('[TRANSCRIPT]', '').replace('[/TRANSCRIPT]', '').strip()

        # Safety net: intercept old sentinel if model still produces it
        if polished == 'NOTHING TO POLISH':
            ConfigManager.console_print('LLM polish: sentinel intercepted — returning raw.')
            return transcription

        # Safety net: if LLM wiped all content but input was non-empty, fall back to raw
        if not polished and transcription.strip():
            ConfigManager.console_print('LLM polish: empty output on non-empty input — returning raw.')
            return transcription

        # Safety net: catch wholesale hallucination (< 30% of output words appear in input)
        if len(transcription.strip()) > 20:
            overlap = _word_overlap_ratio(transcription, polished)
            if overlap < 0.30:
                ConfigManager.console_print(f'LLM polish: low word overlap ({overlap:.0%}) — likely hallucination, returning raw.')
                return transcription

        # Length guard as final catch-all
        if len(transcription.strip()) > 10 and len(polished) > len(transcription.strip()) * 3:
            ConfigManager.console_print('LLM polish: output 3x longer than input — returning raw.')
            return transcription

        return polished
    except Exception as e:
        ConfigManager.console_print(f'LLM polish error (returning raw transcription): {e}')
        return transcription


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

    if ConfigManager.get_config_value('model_options', 'use_api'):
        transcription = transcribe_api(audio_data)
    else:
        transcription = transcribe_local(audio_data, local_model)

    ConfigManager.console_print(f'Whisper output: "{transcription.strip()}"')

    # Discard known Whisper hallucinations produced on silence.
    if transcription.strip().lower() in _WHISPER_HALLUCINATIONS:
        ConfigManager.console_print(f'Whisper hallucination discarded: "{transcription.strip()}"')
        return ''

    # Strip trailing "Thank you" appended by Whisper at the end of real transcriptions.
    stripped = re.sub(r'[,]?\s*\bthank you[.!]?\s*$', '', transcription, flags=re.IGNORECASE).strip()
    if stripped != transcription.strip():
        ConfigManager.console_print(f'Trailing thank-you stripped: "{transcription.strip()}" → "{stripped}"')
        if not stripped:
            return ''
        transcription = stripped

    return post_process_transcription(transcription)
