import io
import os
import numpy as np
import soundfile as sf
from openai import OpenAI

from utils import ConfigManager


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
    return response.text


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
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': transcription},
            ],
        )
        polished = response.choices[0].message.content
        ConfigManager.console_print(f'LLM polish: raw="{transcription.strip()}" → polished="{polished}"')

        # Guard: if output is 3x+ longer than input, the model hallucinated — return raw
        if len(transcription.strip()) > 10 and len(polished) > len(transcription.strip()) * 3:
            ConfigManager.console_print(f'LLM polish hallucination detected ({len(polished)} chars out vs {len(transcription.strip())} in), returning raw transcription')
            return transcription

        return polished
    except Exception as e:
        ConfigManager.console_print(f'LLM polish error (returning raw transcription): {e}')
        return transcription


def post_process_transcription(transcription):
    transcription = transcription.strip()

    # LLM polish runs on the raw stripped transcript, before whitespace/case tweaks
    transcription = llm_polish(transcription)

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

    if ConfigManager.get_config_value('model_options', 'use_api'):
        transcription = transcribe_api(audio_data)
    else:
        transcription = transcribe_local(audio_data, local_model)

    return post_process_transcription(transcription)
