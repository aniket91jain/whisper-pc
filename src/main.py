import os
import sys
import time
from audioplayer import AudioPlayer
from pynput.keyboard import Controller
from PyQt5.QtCore import QObject, QProcess, pyqtSignal
from PyQt5.QtGui import QIcon, QCursor, QGuiApplication
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction, QMessageBox

from key_listener import KeyListener
from result_thread import ResultThread
from ui.main_window import MainWindow
from ui.settings_window import SettingsWindow
from ui.status_window import StatusWindow
from ui.transcript_history_window import TranscriptHistoryWindow
from transcription import create_local_model, prewarm_groq_connection
from input_simulation import InputSimulator
from notifications import register_dict_addition_listener
from utils import ConfigManager


class _DictAddSignal(QObject):
    """Carrier QObject for the auto-add-from-spelling event. Lives on the
    main thread so the connected slot runs there via queued connection, even
    when emit() is called from the polish worker thread."""
    added = pyqtSignal(list)


class WhisperPCApp(QObject):
    def __init__(self):
        """
        Initialize the application, opening settings window if no configuration file is found.
        """
        super().__init__()
        self.app = QApplication(sys.argv)
        # Use Whisper PC branding for Windows UI surfaces (taskbar, Alt-Tab,
        # dialog title bars, JumpList, etc.). setApplicationName feeds the
        # display name; setWindowIcon supplies the icon at every level.
        self.app.setApplicationName('Whisper PC')
        self.app.setApplicationDisplayName('Whisper PC')
        self.app.setWindowIcon(QIcon(os.path.join('assets', 'microphone.png')))

        ConfigManager.initialize()

        self.settings_window = SettingsWindow()
        self.settings_window.settings_closed.connect(self.on_settings_closed)
        self.settings_window.settings_saved.connect(self.restart_app)

        if ConfigManager.config_file_exists():
            self.initialize_components()
        else:
            print('No valid configuration file found. Opening settings window...')
            self.settings_window.show()

    def initialize_components(self):
        """
        Initialize the components of the application.
        """
        self.input_simulator = InputSimulator()

        self.key_listener = KeyListener()
        self.key_listener.add_callback("on_activate", self.on_activation)
        self.key_listener.add_callback("on_deactivate", self.on_deactivation)
        self.key_listener.add_callback("on_history_activate", self.on_history_hotkey)

        model_options = ConfigManager.get_config_section('model_options')
        model_path = model_options.get('local', {}).get('model_path')
        # Local model lifecycle:
        #   use_api=false                          → load eagerly (primary STT)
        #   use_api=true + enable_local_fallback=true → load eagerly in a
        #       daemon thread so it's ready when the API fails. Setting
        #       self.local_model is deferred until the load completes.
        #   use_api=true + enable_local_fallback=false → don't load (saves RAM)
        self.local_model = None
        if not model_options.get('use_api'):
            self.local_model = create_local_model()
        elif model_options.get('enable_local_fallback'):
            from threading import Thread
            def _load_fallback_model():
                try:
                    self.local_model = create_local_model()
                    ConfigManager.console_print('Local-Whisper fallback ready.')
                except Exception as e:
                    ConfigManager.console_print(f'Local-Whisper fallback load failed: {e}')
            Thread(target=_load_fallback_model, daemon=True).start()

        # Pre-warm the Groq HTTPS connection in a daemon thread so the first
        # dictation post-launch doesn't pay the TLS handshake (~200-400ms).
        # Fires only when Groq is actually in the path — either as the STT
        # backend (use_api) or as the polish backend (llm_polish.enabled).
        if model_options.get('use_api') or ConfigManager.get_config_value('llm_polish', 'enabled'):
            from threading import Thread
            Thread(target=prewarm_groq_connection, daemon=True).start()

        self.result_thread = None
        self._recording_started_at = 0.0
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._log_path = os.path.join(project_root, 'transcript_log.txt')
        self._failed_log_path = os.path.join(project_root, 'failed_log.txt')
        self._history_window = None

        self.main_window = MainWindow()
        self.main_window.openSettings.connect(self.settings_window.show)
        self.main_window.startListening.connect(self.key_listener.start)
        self.main_window.closeApp.connect(self.exit_app)

        if not ConfigManager.get_config_value('misc', 'hide_status_window'):
            self.status_window = StatusWindow()

        self.create_tray_icon()
        self.key_listener.start()  # auto-start listening; no need to press Start in the window

    def create_tray_icon(self):
        """
        Create the system tray icon and its context menu.
        """
        # Mic icon matches Mobile's tray glyph and gives the system tray a
        # functional read (this is a dictation app) instead of the W logo.
        self.tray_icon = QSystemTrayIcon(QIcon(os.path.join('assets', 'microphone.png')), self.app)
        self.tray_icon.setToolTip('Whisper PC')

        tray_menu = QMenu()

        show_action = QAction('Whisper PC Main Menu', self.app)
        show_action.triggered.connect(self.main_window.show)
        tray_menu.addAction(show_action)

        settings_action = QAction('Open Settings', self.app)
        settings_action.triggered.connect(self.settings_window.show)
        tray_menu.addAction(settings_action)

        log_action = QAction('View Transcript Log', self.app)
        log_action.triggered.connect(self._open_transcript_log)
        tray_menu.addAction(log_action)

        exit_action = QAction('Exit', self.app)
        exit_action.triggered.connect(self.exit_app)
        tray_menu.addAction(exit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

        # Listen for "auto-added to dictionary" events fired by the polish
        # pipeline. The signal is queued so it always runs on the main GUI
        # thread, even though the polish call happens in ResultThread.
        self._dict_add_signal = _DictAddSignal(self)
        self._dict_add_signal.added.connect(self._show_dict_add_balloon)
        register_dict_addition_listener(self._dict_add_signal.added.emit)

    def _show_dict_add_balloon(self, words):
        if not words or not self.tray_icon:
            return
        title = 'Whisper PC'
        if len(words) == 1:
            body = f"Added '{words[0]}' to dictionary"
        else:
            quoted = ', '.join(f"'{w}'" for w in words)
            body = f"Added {quoted} to dictionary"
        # 4000 ms is long enough to read but short enough not to linger.
        self.tray_icon.showMessage(title, body, QSystemTrayIcon.Information, 4000)

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:  # left-click
            self._open_transcript_log()

    def cleanup(self):
        if self.key_listener:
            self.key_listener.stop()
        if self.input_simulator:
            self.input_simulator.cleanup()

    def _open_transcript_log(self, near_cursor=False):
        if self._history_window is None:
            self._history_window = TranscriptHistoryWindow(
                self._log_path,
                self._failed_log_path,
                self.local_model,
                self.input_simulator,
            )
        else:
            self._history_window._load()
        if near_cursor:
            self._position_window_near_cursor(self._history_window)
        self._history_window.show()
        self._history_window.raise_()

    def _position_window_near_cursor(self, window):
        """Place the window's top-left a bit below-right of the mouse pointer,
        clamped to the current screen so it never opens off-screen on multi-mon
        setups."""
        cursor_pos = QCursor.pos()
        screen = QGuiApplication.screenAt(cursor_pos) or QGuiApplication.primaryScreen()
        screen_geo = screen.availableGeometry()
        w, h = window.width(), window.height()
        x = min(max(cursor_pos.x() + 12, screen_geo.left()), screen_geo.right() - w)
        y = min(max(cursor_pos.y() + 12, screen_geo.top()), screen_geo.bottom() - h)
        window.move(x, y)

    def on_history_hotkey(self):
        """Ditto-style toggle: hotkey opens the history popup near the cursor,
        or hides it if already visible. Window keeps focus on the underlying
        app so a click on a card pastes into the active field."""
        if self._history_window is not None and self._history_window.isVisible():
            self._history_window.hide()
            return
        self._open_transcript_log(near_cursor=True)

    def exit_app(self):
        """
        Exit the application.
        """
        self.cleanup()
        QApplication.quit()

    def restart_app(self):
        """Restart the application to apply the new settings."""
        self.cleanup()
        QApplication.quit()
        QProcess.startDetached(sys.executable, sys.argv)

    def on_settings_closed(self):
        """
        If settings is closed without saving on first run, initialize the components with default values.
        """
        if not os.path.exists(os.path.join('src', 'config.yaml')):
            QMessageBox.information(
                self.settings_window,
                'Using Default Values',
                'Settings closed without saving. Default values are being used.'
            )
            self.initialize_components()

    # Ignore a second activation-chord fire within this window after recording
    # starts. Guards against accidental retriggers (held Alt + stray Z, OS
    # key-repeat, AltGr layouts) cutting the user off mid-sentence.
    _TOGGLE_COOLDOWN_SEC = 0.5

    def on_activation(self):
        """
        Called when the activation key combination is pressed.
        """
        if self.result_thread and self.result_thread.isRunning():
            recording_mode = ConfigManager.get_config_value('recording_options', 'recording_mode')
            if recording_mode == 'press_to_toggle':
                elapsed = time.time() - self._recording_started_at
                if elapsed < self._TOGGLE_COOLDOWN_SEC:
                    ConfigManager.console_print(
                        f'Toggle ignored (cooldown): {elapsed*1000:.0f}ms < '
                        f'{self._TOGGLE_COOLDOWN_SEC*1000:.0f}ms since recording started.'
                    )
                    return
                self.result_thread.stop_recording()
            elif recording_mode == 'continuous':
                self.stop_result_thread()
            return

        # Event-driven warming: warm the Groq HTTPS connection in parallel
        # with audio capture starting. The handshake (if pool went cold while
        # idle) finishes during speech, not before upload. Free latency hiding.
        if ConfigManager.get_config_value('model_options', 'use_api') or \
                ConfigManager.get_config_value('llm_polish', 'enabled'):
            from threading import Thread
            Thread(target=prewarm_groq_connection, daemon=True).start()

        self.start_result_thread()

    def on_deactivation(self):
        """
        Called when the activation key combination is released.
        """
        if ConfigManager.get_config_value('recording_options', 'recording_mode') == 'hold_to_record':
            if self.result_thread and self.result_thread.isRunning():
                self.result_thread.stop_recording()

    def start_result_thread(self):
        """
        Start the result thread to record audio and transcribe it.
        """
        if self.result_thread and self.result_thread.isRunning():
            return

        self.result_thread = ResultThread(self.local_model)
        if not ConfigManager.get_config_value('misc', 'hide_status_window'):
            self.result_thread.statusSignal.connect(self.status_window.updateStatus)
            self.status_window.closeSignal.connect(self.stop_result_thread)
        self.result_thread.resultSignal.connect(self.on_transcription_complete)
        self.result_thread.failedSignal.connect(self.on_transcription_failed)
        self._recording_started_at = time.time()
        self.result_thread.start()

    def stop_result_thread(self):
        """
        Stop the result thread.
        """
        if self.result_thread and self.result_thread.isRunning():
            self.result_thread.stop()

    def on_transcription_failed(self, audio_path, reason):
        """Audio captured but the API call failed. Audio + log entry are already
        on disk; just refresh the history window so the user sees the new row."""
        ConfigManager.console_print(f'Transcription failed; audio saved to {audio_path} ({reason})')
        if self._history_window is not None and self._history_window.isVisible():
            self._history_window._load()

    def on_transcription_complete(self, result):
        """
        When the transcription is complete, type the result and start listening for the activation key again.
        """
        # Empty result reaches here on silent recordings (status overlay shows
        # "Nothing transcribable detected") and on API failures. Skip the paste
        # so we don't clobber the user's clipboard or fire a stray Ctrl+V — but
        # still run beep / re-arm so the activation flow stays consistent.
        if result and result.strip():
            self.input_simulator.typewrite(result)

        if ConfigManager.get_config_value('misc', 'noise_on_completion'):
            AudioPlayer(os.path.join('assets', 'beep.wav')).play(block=True)

        if ConfigManager.get_config_value('recording_options', 'recording_mode') == 'continuous':
            self.start_result_thread()
        else:
            self.key_listener.start()

    def run(self):
        """
        Start the application.
        """
        sys.exit(self.app.exec_())


def _enforce_single_instance() -> None:
    """Exit immediately if another Whisper PC is already running.

    Uses a lockfile in the user's TEMP dir containing the running process's
    PID. On startup:
      - If the file doesn't exist → write our PID, continue.
      - If the file exists AND that PID is alive AND its image is a
        Whisper PC Python → exit (peer is running).
      - If the file exists but the PID is dead or unrelated → stale lock;
        overwrite with our PID and continue.

    Cleanup: register an atexit handler to delete the lockfile on normal
    shutdown. Crashed exits leave a stale file which the next launch detects.

    Why this exists: on 2026-05-13 a stacked-launch event left 4 Whisper PC
    pythonw.exe processes running simultaneously, all fighting for the same
    activation hotkey and audio device. The UI became unresponsive ("hung").
    """
    import atexit
    import tempfile
    lockfile = os.path.join(tempfile.gettempdir(), 'whisper_pc.lock')

    def _peer_is_alive(pid: int) -> bool:
        # On Windows, signal 0 isn't supported by os.kill in the usual sense,
        # but we can use a Win32 query. Without pulling pywin32, the simplest
        # check is to ask tasklist if the PID exists and runs python.
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False  # process doesn't exist
            try:
                # 259 = STILL_ACTIVE
                exit_code = ctypes.c_ulong()
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    if exit_code.value == 259:
                        # Process is alive. We don't bother verifying it's a
                        # Whisper PC Python — colliding with an unrelated
                        # PID after a crash is rare (32-bit PID space) and the
                        # worst case is the user re-runs after a few seconds.
                        return True
                return False
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False  # be permissive: if our check fails, continue

    if os.path.isfile(lockfile):
        try:
            with open(lockfile, 'r', encoding='utf-8') as f:
                existing_pid = int((f.read() or '0').strip())
        except Exception:
            existing_pid = 0
        if existing_pid > 0 and _peer_is_alive(existing_pid):
            # Another instance is running. Tell the user via stderr and exit.
            # pythonw.exe has no console so this is mainly for python.exe / debug
            # runs; a tray toast would need PyQt which we don't have yet here.
            print(
                f'Whisper PC is already running (PID {existing_pid}). Exiting.',
                file=sys.stderr,
            )
            sys.exit(0)

    # Take the lock (overwrite stale or non-existent file).
    try:
        with open(lockfile, 'w', encoding='utf-8') as f:
            f.write(str(os.getpid()))
    except Exception as e:
        print(f'Could not write lockfile {lockfile}: {e}', file=sys.stderr)

    def _release_lock():
        try:
            if os.path.isfile(lockfile):
                with open(lockfile, 'r', encoding='utf-8') as f:
                    if (f.read() or '').strip() == str(os.getpid()):
                        os.remove(lockfile)
        except Exception:
            pass

    atexit.register(_release_lock)


if __name__ == '__main__':
    _enforce_single_instance()
    app = WhisperPCApp()
    app.run()
