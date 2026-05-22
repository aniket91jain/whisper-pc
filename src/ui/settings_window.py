"""Settings window — hand-built tabs, hidden defaults, contextual show/hide.

Rewritten 2026-05-22 from the auto-schema-render approach (which dumped every
config_schema.yaml entry into a generic tab) to a curated UI organized around
how users actually think about Whisper PC:

    Engine                 ← what's transcribing
    Recording              ← capturing audio
    Output                 ← getting text into the focused app
    Polish                 ← cleaning the transcript (engine-specific content)
    Custom Vocabulary      ← proper-nouns list (locations/people/products)
    Advanced               ← collapsed defaults for power users

Setting visibility is now contextual:
  * ElevenLabs API key only appears when stt_engine = elevenlabs
  * Groq API key only appears when stt_engine = groq
  * Streaming-mode toggle only appears for elevenlabs
  * Regex-polish toggles only appear for elevenlabs
  * LLM-polish row only appears for groq
  * Writing-key-press-delay only appears when input_method = pynput

Schema entries that aren't in this UI are NOT removed — they're still read by
the underlying code (transcription.py uses common.language / common.temperature
/ api.model etc, main.py uses misc.noise_on_completion, utils.py uses
misc.print_to_terminal). The rewrite just stops exposing every knob to avoid
the cluttered every-setting-flat layout users complained about.

The Custom Vocabulary tab (`CustomVocabularyTab`) is preserved verbatim — it
already had the locations/people/products three-category structure the user
asked us to keep.
"""

import os
import sys
from typing import Callable

from dotenv import set_key, load_dotenv
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSizePolicy,
    QSpacerItem, QStyle, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui.base_window import BaseWindow
from ui.custom_vocabulary_tab import CustomVocabularyTab
from utils import ConfigManager

load_dotenv()


class SettingsWindow(BaseWindow):
    """Reorganized Whisper PC settings — 6 tabs, contextual visibility."""

    settings_closed = pyqtSignal()
    settings_saved = pyqtSignal()

    def __init__(self):
        super().__init__('Settings', 760, 680)

        # We don't auto-iterate config_schema anymore. Each setting widget is
        # registered explicitly in _register so save/reset/visibility can be
        # driven by a single source of truth instead of findChild() calls.
        self._registry: list[dict] = []

        # Visibility groups — controls in these groups are toggled by
        # contextual rules below (engine type, input method, etc.).
        self._engine_groq_widgets: list[QWidget] = []
        self._engine_elevenlabs_widgets: list[QWidget] = []
        self._input_pynput_widgets: list[QWidget] = []

        self._apply_stylesheet()
        self._build_ui()
        self._apply_contextual_visibility()

    def _apply_stylesheet(self) -> None:
        """Soft, modern QSS — gentle borders on group boxes, breathable tab
        headers, larger buttons. Keeps the OS theme but tightens spacing."""
        self.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid palette(mid);
                border-radius: 6px;
                top: -1px;
            }
            QTabBar::tab {
                padding: 8px 18px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                background: palette(window);
            }
            QTabBar::tab:selected {
                background: palette(base);
                border: 1px solid palette(mid);
                border-bottom: 1px solid palette(base);
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid palette(mid);
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 16px;
                background-color: palette(base);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                background-color: palette(base);
            }
            QPushButton {
                padding: 7px 18px;
                border-radius: 5px;
                min-width: 100px;
            }
            QLineEdit, QComboBox {
                padding: 5px 8px;
                border-radius: 4px;
            }
            QToolButton {
                padding: 2px;
            }
        """)

    # ------- UI scaffolding -------

    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self._build_engine_tab()
        self._build_recording_tab()
        self._build_output_tab()
        self._build_polish_tab()

        self.custom_vocab_tab = CustomVocabularyTab()
        self.tabs.addTab(self.custom_vocab_tab, 'Custom Vocabulary')

        self._build_advanced_tab()
        self._build_buttons()

    def _new_tab(self, title: str, intro: str = '') -> QFormLayout:
        """Create a new tab and return its form layout.

        If `intro` is given, an italic-gray one-liner is placed at the top of
        the tab to set the user's expectation for the controls below.
        """
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(24, 20, 24, 12)
        outer.setSpacing(12)

        if intro:
            intro_label = QLabel(intro)
            intro_label.setWordWrap(True)
            intro_label.setStyleSheet(
                'color: palette(mid); font-style: italic; padding-bottom: 4px;'
            )
            outer.addWidget(intro_label)

        form_container = QWidget()
        form = QFormLayout(form_container)
        form.setHorizontalSpacing(22)
        form.setVerticalSpacing(12)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        outer.addWidget(form_container)
        outer.addStretch(1)

        self.tabs.addTab(tab, title)
        return form

    def _build_buttons(self) -> None:
        row = QHBoxLayout()
        reset_btn = QPushButton('Reset to saved')
        reset_btn.clicked.connect(self._reset_settings)
        save_btn = QPushButton('Save')
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save_settings)
        row.addStretch(1)
        row.addWidget(reset_btn)
        row.addWidget(save_btn)
        self.main_layout.addLayout(row)

    # ------- Tab builders -------

    def _build_engine_tab(self) -> None:
        form = self._new_tab(
            'Engine',
            'Which speech-to-text provider runs your dictations. '
            'API keys for the selected engine appear below.',
        )

        stt_engine = self._add_combo(
            form, 'STT engine',
            options=['groq', 'elevenlabs'],
            options_labels={'groq': 'Groq (Whisper) + LLM polish',
                            'elevenlabs': 'ElevenLabs Scribe v2 RT'},
            path=('model_options', 'stt_engine'),
            default='groq',
            description=(
                'Groq routes audio through Whisper-large-v3-turbo then a Llama '
                'polish step. ElevenLabs uses Scribe v2 Realtime with built-in '
                'punctuation/capitalisation and a fast client-side regex polish.'
            ),
        )
        # When the user changes engine, update contextual visibility live.
        stt_engine.currentTextChanged.connect(
            lambda _: self._apply_contextual_visibility())

        # ---- Groq-specific rows ----
        groq_key = self._add_secret(
            form, 'Groq API key',
            path=('model_options', 'api', 'api_key'),
            env_var='GROQ_API_KEY',
            description=(
                'Required when STT engine is Groq. Stored in .env as '
                'GROQ_API_KEY, not in config.yaml (so it never gets committed '
                'to git). The legacy OPENAI_API_KEY env var is also honoured.'
            ),
        )
        self._mark_groq_only(groq_key, form.labelForField(groq_key))

        # ---- ElevenLabs-specific rows ----
        el_key = self._add_secret(
            form, 'ElevenLabs API key',
            path=('model_options', 'elevenlabs_api_key'),
            env_var='ELEVENLABS_API_KEY',
            description=(
                'Required when STT engine is ElevenLabs. Stored in .env, not '
                'in config.yaml. The .env value takes precedence at runtime.'
            ),
        )
        self._mark_elevenlabs_only(el_key, form.labelForField(el_key))

        streaming = self._add_checkbox(
            form, 'Stream during recording',
            path=('model_options', 'stt_streaming_mode'),
            default=True,
            description=(
                'PCM is sent to ElevenLabs while you are still talking, so the '
                'final commit response arrives within ~400ms of you releasing '
                'the hotkey instead of waiting for a full burst upload after.'
            ),
        )
        self._mark_elevenlabs_only(streaming, form.labelForField(streaming))

    def _build_recording_tab(self) -> None:
        form = self._new_tab(
            'Recording',
            'Hotkeys and microphone capture.',
        )

        self._add_line(
            form, 'Activation key',
            path=('recording_options', 'activation_key'),
            default='ctrl+shift+space',
            description=(
                'Keyboard shortcut to start/stop recording. Join keys with '
                '+ (e.g. "alt+z", "ctrl+shift+space").'
            ),
        )

        self._add_line(
            form, 'History hotkey',
            path=('recording_options', 'history_key'),
            default='alt+shift+z',
            description=(
                'Toggles the Transcript History popup (Ditto-style). Leave '
                'blank to disable.'
            ),
        )

        self._add_combo(
            form, 'Recording mode',
            options=['continuous', 'voice_activity_detection',
                     'press_to_toggle', 'hold_to_record'],
            options_labels={
                'continuous': 'Continuous (auto-restart on pause)',
                'voice_activity_detection': 'VAD (stop on long pause)',
                'press_to_toggle': 'Press to toggle',
                'hold_to_record': 'Hold to record',
            },
            path=('recording_options', 'recording_mode'),
            default='continuous',
            description=(
                'continuous = recording restarts automatically after a pause; '
                'press the hotkey again to stop. press_to_toggle = single tap '
                'starts, another tap ends. hold_to_record = release to end. '
                'voice_activity_detection = stops on long silence.'
            ),
        )

        self._add_line(
            form, 'Microphone (device index)',
            path=('recording_options', 'sound_device'),
            default='',
            placeholder='leave blank for default',
            description=(
                'Numeric index of the sound device. Run '
                '`python -m sounddevice` to list devices. Blank = default.'
            ),
        )

    def _build_output_tab(self) -> None:
        form = self._new_tab(
            'Output',
            'How transcribed text reaches the focused application.',
        )

        input_method = self._add_combo(
            form, 'Input method',
            options=['clipboard', 'pynput', 'ydotool', 'dotool'],
            options_labels={
                'clipboard': 'Clipboard (recommended, instant paste)',
                'pynput': 'Pynput (types char-by-char)',
                'ydotool': 'ydotool (Linux)',
                'dotool': 'dotool (Linux)',
            },
            path=('post_processing', 'input_method'),
            default='clipboard',
            description=(
                'How the transcribed text reaches the focused field. '
                'Clipboard copies + sends Ctrl+V (instant for any length); '
                'pynput types each character with a delay between keystrokes '
                '(visible char-by-char on long dictations).'
            ),
        )
        input_method.currentTextChanged.connect(
            lambda _: self._apply_contextual_visibility())

        self._add_checkbox(
            form, 'Add trailing space',
            path=('post_processing', 'add_trailing_space'),
            default=True,
            description=(
                'Append a space to the end of every transcript so the next '
                'thing you dictate or type runs cleanly into it.'
            ),
        )

        self._add_checkbox(
            form, 'Add leading space if needed',
            path=('post_processing', 'add_leading_space_if_needed'),
            default=True,
            description=(
                'When pasting into a field whose cursor is immediately after '
                'a non-whitespace character, prepend a space so the new text '
                'does not concatenate with the word before it.'
            ),
        )

        self._add_checkbox(
            form, 'Lowercase first letter mid-sentence',
            path=('post_processing', 'lowercase_first_letter_mid_sentence'),
            default=True,
            description=(
                'When the cursor is mid-sentence (preceding text doesn\'t end '
                'in a terminator), lowercase the first letter of the new text '
                'so it joins the existing sentence naturally. Skips "I" and '
                'likely acronyms.'
            ),
        )

        # Only shown when input_method = pynput
        wkpd = self._add_float(
            form, 'Writing key press delay (s)',
            path=('post_processing', 'writing_key_press_delay'),
            default=0.005,
            description=(
                'Delay between keystrokes when using pynput input method. '
                'Lower = faster, higher = more compatible with apps that lose '
                'characters under fast typing. Has no effect in clipboard mode.'
            ),
        )
        self._mark_pynput_only(wkpd, form.labelForField(wkpd))

    def _build_polish_tab(self) -> None:
        form = self._new_tab(
            'Polish',
            'How the raw transcript is cleaned. Different controls show up '
            'depending on which engine you picked on the Engine tab.',
        )

        # ---- LLM polish (Groq path) ----
        llm_box = QGroupBox('LLM polish (Groq path)')
        llm_form = QFormLayout(llm_box)
        llm_form.setHorizontalSpacing(20)
        llm_form.setVerticalSpacing(10)

        self._add_checkbox(
            llm_form, 'Enable LLM polish',
            path=('llm_polish', 'enabled'),
            default=False,
            description=(
                'Pass Groq Whisper transcripts through an LLM (default Groq '
                'gpt-oss-120b) to clean fillers, fix grammar, expand voice '
                'commands. No effect when STT engine = ElevenLabs (regex '
                'polish runs instead).'
            ),
        )

        self._add_combo(
            llm_form, 'Polish model',
            options=['openai/gpt-oss-120b', 'openai/gpt-oss-20b',
                     'llama-3.3-70b-versatile', 'llama-3.1-8b-instant'],
            path=('llm_polish', 'model'),
            default='openai/gpt-oss-120b',
            description=(
                'Groq model used for polish. gpt-oss-120b is the recommended '
                'default; gpt-oss-20b is faster but slightly lower quality.'
            ),
        )

        self._add_checkbox(
            llm_form, 'Auto-add from "spelled" trigger',
            path=('llm_polish', 'enable_dict_autoadd_from_spelling'),
            default=True,
            description=(
                'When the user says "<word> spelled X-X-X" during dictation, '
                'add that word to Whisper STT bias and to the Custom '
                'Vocabulary People list so polish corrects future mishears.'
            ),
        )

        self._mark_groq_only(llm_box, None)
        form.addRow(llm_box)

        # ---- Regex polish (ElevenLabs path) ----
        regex_box = QGroupBox('Regex polish (ElevenLabs path)')
        regex_form = QFormLayout(regex_box)
        regex_form.setHorizontalSpacing(20)
        regex_form.setVerticalSpacing(10)

        for key, label, desc in [
            ('proper_nouns', 'Fix proper noun mishears',
             'Catch common Whisper mishears for names in your Custom '
             'Vocabulary list (e.g. "owner res" → "OwnerRez").'),
            ('spoken_punctuation', 'Spoken punctuation',
             'Translate spoken punctuation commands to symbols ("comma" → ,, '
             '"new paragraph" → blank line).'),
            ('alphanumeric', 'Collapse NATO + digits',
             'Join NATO phonetic codes and digits into single tokens '
             '("alpha bravo 1 2 3" → "AB123").'),
            ('email', 'Email shorthand',
             'Convert spoken email shapes to addresses '
             '("aniket at gmail dot com" → "aniket@gmail.com").'),
            ('spelling', 'Spelling capture',
             'Detect "<word> spelled X-X-X" patterns; use the spelled letters '
             'as the canonical word AND auto-add to vocabulary.'),
            ('scratch', '"Scratch that" voice command',
             'Honor "scratch that" / "delete that" — replace the preceding '
             'phrase with whatever was said next.'),
        ]:
            self._add_checkbox(
                regex_form, label,
                path=('regex_polish', key),
                default=True,
                description=desc,
            )

        self._mark_elevenlabs_only(regex_box, None)
        form.addRow(regex_box)

    def _build_advanced_tab(self) -> None:
        form = self._new_tab(
            'Advanced',
            'Power-user settings. Defaults are sensible — only change these '
            'if you have a specific reason.',
        )

        # Recording boundary knobs
        self._add_int(
            form, 'Silence duration (ms)',
            path=('recording_options', 'silence_duration'),
            default=900,
            description=(
                'How long a pause must be (in milliseconds) before VAD/'
                'continuous mode considers the user done.'
            ),
        )
        self._add_int(
            form, 'Minimum recording duration (ms)',
            path=('recording_options', 'min_duration'),
            default=100,
            description=(
                'Recordings shorter than this are discarded. Helps suppress '
                'accidental tap-and-release fires.'
            ),
        )
        self._add_int(
            form, 'Sample rate (Hz)',
            path=('recording_options', 'sample_rate'),
            default=16000,
            description=(
                'PCM sample rate. 16000 is required for both Groq Whisper and '
                'ElevenLabs Scribe v2 RT.'
            ),
        )

        # LLM polish low-level knobs
        polish_box = QGroupBox('LLM polish tuning (Groq path)')
        polish_form = QFormLayout(polish_box)
        polish_form.setHorizontalSpacing(20)
        polish_form.setVerticalSpacing(10)
        self._add_int(
            polish_form, 'Max tokens',
            path=('llm_polish', 'max_tokens'),
            default=1024,
            description=(
                'Maximum tokens for the LLM polish response. Higher = handles '
                'longer dictations without truncation but slightly slower.'
            ),
        )
        self._add_float(
            polish_form, 'Temperature',
            path=('llm_polish', 'temperature'),
            default=0.2,
            description=(
                'Sampling temperature for the polish LLM. Lower = more '
                'deterministic. 0.2 is a good default.'
            ),
        )
        self._add_combo(
            polish_form, 'Reasoning effort',
            options=['low', 'medium', 'high'],
            path=('llm_polish', 'reasoning_effort'),
            default='low',
            description=(
                'reasoning_effort for gpt-oss-* models on Groq. "low" '
                'suppresses chain-of-thought tokens and keeps polish fast.'
            ),
        )
        form.addRow(polish_box)

        # Local Whisper (rarely used)
        local_box = QGroupBox('Local Whisper fallback (rarely used)')
        local_form = QFormLayout(local_box)
        local_form.setHorizontalSpacing(20)
        local_form.setVerticalSpacing(10)
        self._add_checkbox(
            local_form, 'Use local model instead of API',
            path=('model_options', 'use_api'),
            default=True,
            inverted=True,
            description=(
                'When checked, runs a local faster-whisper model instead of '
                'calling Groq / ElevenLabs. Slow + lower quality on a CPU — '
                'only useful for fully-offline use.'
            ),
        )
        self._add_checkbox(
            local_form, 'Eager-load local model as fallback',
            path=('model_options', 'enable_local_fallback'),
            default=True,
            description=(
                'Even with API mode, pre-load the local model into memory so '
                'it can stand in if the API fails after retries. Adds 1-3 GB '
                'RAM at startup. Uncheck to save memory.'
            ),
        )
        self._add_combo(
            local_form, 'Local model size',
            options=['base', 'base.en', 'tiny', 'tiny.en', 'small',
                     'small.en', 'medium', 'medium.en', 'large', 'large-v1',
                     'large-v2', 'large-v3'],
            path=('model_options', 'local', 'model'),
            default='base',
            description=(
                'faster-whisper model. Larger = more accurate but slower and '
                'more memory.'
            ),
        )
        form.addRow(local_box)

    # ------- Widget factories -------

    def _add_checkbox(self, form: QFormLayout, label: str, *, path: tuple,
                      default: bool, description: str, inverted: bool = False) -> QCheckBox:
        widget = QCheckBox()
        value = self._read_path(path, default)
        widget.setChecked(value if not inverted else not value)
        self._register(widget, path, kind='bool', default=default, inverted=inverted)
        form.addRow(self._labelled(label, description), widget)
        return widget

    def _add_combo(self, form: QFormLayout, label: str, *, options: list,
                   path: tuple, default: str, description: str,
                   options_labels: dict | None = None) -> QComboBox:
        widget = QComboBox()
        if options_labels:
            for opt in options:
                widget.addItem(options_labels.get(opt, opt), userData=opt)
        else:
            widget.addItems(options)
        value = self._read_path(path, default)
        # Find the index matching the userData (or fall back to the string)
        for i in range(widget.count()):
            data = widget.itemData(i)
            if data == value or widget.itemText(i) == value:
                widget.setCurrentIndex(i)
                break
        self._register(widget, path, kind='combo', default=default)
        form.addRow(self._labelled(label, description), widget)
        return widget

    def _add_line(self, form: QFormLayout, label: str, *, path: tuple,
                  default: str, description: str,
                  placeholder: str = '') -> QLineEdit:
        widget = QLineEdit(str(self._read_path(path, default) or ''))
        if placeholder:
            widget.setPlaceholderText(placeholder)
        self._register(widget, path, kind='str', default=default)
        form.addRow(self._labelled(label, description), widget)
        return widget

    def _add_int(self, form: QFormLayout, label: str, *, path: tuple,
                 default: int, description: str) -> QLineEdit:
        widget = QLineEdit(str(self._read_path(path, default) or default))
        self._register(widget, path, kind='int', default=default)
        form.addRow(self._labelled(label, description), widget)
        return widget

    def _add_float(self, form: QFormLayout, label: str, *, path: tuple,
                   default: float, description: str) -> QLineEdit:
        widget = QLineEdit(str(self._read_path(path, default) or default))
        self._register(widget, path, kind='float', default=default)
        form.addRow(self._labelled(label, description), widget)
        return widget

    def _add_secret(self, form: QFormLayout, label: str, *, path: tuple,
                    env_var: str, description: str) -> QLineEdit:
        widget = QLineEdit()
        widget.setEchoMode(QLineEdit.Password)
        # .env takes precedence over config.yaml for API keys.
        env_value = os.getenv(env_var) or ''
        config_value = self._read_path(path, '') or ''
        widget.setText(env_value or config_value)
        self._register(widget, path, kind='secret', default='', env_var=env_var)
        form.addRow(self._labelled(label, description), widget)
        return widget

    def _labelled(self, text: str, description: str) -> QWidget:
        """Return a horizontal widget containing the label + a help button."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        lbl = QLabel(text + ':')
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(lbl)
        help_btn = QToolButton()
        help_btn.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxQuestion))
        help_btn.setAutoRaise(True)
        help_btn.setToolTip(description)
        help_btn.setCursor(Qt.PointingHandCursor)
        help_btn.clicked.connect(
            lambda: QMessageBox.information(self, 'About this setting', description))
        layout.addWidget(help_btn)
        return container

    # ------- Visibility groups -------

    def _mark_groq_only(self, widget: QWidget, label_widget: QWidget | None) -> None:
        self._engine_groq_widgets.append(widget)
        if label_widget is not None:
            self._engine_groq_widgets.append(label_widget)

    def _mark_elevenlabs_only(self, widget: QWidget, label_widget: QWidget | None) -> None:
        self._engine_elevenlabs_widgets.append(widget)
        if label_widget is not None:
            self._engine_elevenlabs_widgets.append(label_widget)

    def _mark_pynput_only(self, widget: QWidget, label_widget: QWidget | None) -> None:
        self._input_pynput_widgets.append(widget)
        if label_widget is not None:
            self._input_pynput_widgets.append(label_widget)

    def _apply_contextual_visibility(self) -> None:
        engine = self._current_widget_value(('model_options', 'stt_engine'),
                                            default='groq')
        for w in self._engine_groq_widgets:
            w.setVisible(engine == 'groq')
        for w in self._engine_elevenlabs_widgets:
            w.setVisible(engine == 'elevenlabs')

        input_method = self._current_widget_value(
            ('post_processing', 'input_method'), default='clipboard')
        for w in self._input_pynput_widgets:
            w.setVisible(input_method == 'pynput')

    # ------- Registry / config glue -------

    def _register(self, widget: QWidget, path: tuple, *, kind: str,
                  default, inverted: bool = False,
                  env_var: str | None = None) -> None:
        self._registry.append({
            'widget': widget,
            'path': path,
            'kind': kind,
            'default': default,
            'inverted': inverted,
            'env_var': env_var,
        })

    def _read_path(self, path: tuple, default):
        # Explicit None check — `or default` would incorrectly fall back when
        # the saved value is False, 0, or empty string (all legitimate).
        value = ConfigManager.get_config_value(*path)
        return default if value is None else value

    def _current_widget_value(self, path: tuple, default):
        """Return the live UI value for `path` (not the saved config). Used
        by visibility logic so toggles react before Save is clicked."""
        for entry in self._registry:
            if entry['path'] == path:
                return self._extract_widget_value(entry)
        return self._read_path(path, default)

    def _extract_widget_value(self, entry: dict):
        widget = entry['widget']
        kind = entry['kind']
        if kind == 'bool':
            v = widget.isChecked()
            return (not v) if entry['inverted'] else v
        if kind == 'combo':
            data = widget.currentData()
            return data if data is not None else widget.currentText()
        if kind == 'secret':
            return widget.text()
        if kind == 'int':
            text = widget.text().strip()
            return int(text) if text else None
        if kind == 'float':
            text = widget.text().strip()
            return float(text) if text else None
        # str
        return widget.text() if widget.text() else None

    # ------- Save / Reset -------

    def _save_settings(self) -> None:
        try:
            for entry in self._registry:
                value = self._extract_widget_value(entry)
                env_var = entry['env_var']
                if env_var:
                    # Secrets land in .env, not config.yaml.
                    set_key('.env', env_var, value or '')
                    os.environ[env_var] = value or ''
                    # Don't persist secret to config.yaml.
                    ConfigManager.set_config_value(None, *entry['path'])
                else:
                    ConfigManager.set_config_value(value, *entry['path'])

            self.custom_vocab_tab.save_to_config()
            ConfigManager.save_config()
        except ValueError as e:
            QMessageBox.warning(
                self, 'Invalid value',
                f'One of the numeric fields has a non-numeric value: {e}.\n'
                f'Fix it and try Save again.')
            return

        QMessageBox.information(
            self, 'Settings saved',
            'Settings have been saved. The application will now restart.')
        self.settings_saved.emit()
        self.close()

    def _reset_settings(self) -> None:
        ConfigManager.reload_config()
        for entry in self._registry:
            value = self._read_path(entry['path'], entry['default'])
            self._set_widget_value(entry, value)
        self.custom_vocab_tab.load_from_config()
        self._apply_contextual_visibility()

    def _set_widget_value(self, entry: dict, value) -> None:
        widget = entry['widget']
        kind = entry['kind']
        if kind == 'bool':
            widget.setChecked((not value) if entry['inverted'] else bool(value))
        elif kind == 'combo':
            for i in range(widget.count()):
                data = widget.itemData(i)
                if data == value or widget.itemText(i) == value:
                    widget.setCurrentIndex(i)
                    return
        elif kind == 'secret':
            env_value = os.getenv(entry['env_var'] or '') or ''
            widget.setText(env_value or (value if isinstance(value, str) else ''))
        elif kind in ('int', 'float'):
            widget.setText(str(value) if value is not None else '')
        else:
            widget.setText(str(value) if value is not None else '')

    # ------- Window lifecycle -------

    def closeEvent(self, event) -> None:
        reply = QMessageBox.question(
            self, 'Close without saving?',
            'Discard any unsaved changes?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            ConfigManager.reload_config()
            self.custom_vocab_tab.load_from_config()
            self.settings_closed.emit()
            super().closeEvent(event)
        else:
            event.ignore()
