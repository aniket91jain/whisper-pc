import subprocess
import os
import signal
import time
import ctypes
import ctypes.wintypes
import pyperclip
from pynput.keyboard import Controller as PynputController, Key

from utils import ConfigManager


class _GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ('cbSize', ctypes.wintypes.DWORD),
        ('flags', ctypes.wintypes.DWORD),
        ('hwndActive', ctypes.wintypes.HWND),
        ('hwndFocus', ctypes.wintypes.HWND),
        ('hwndCapture', ctypes.wintypes.HWND),
        ('hwndMenuOwner', ctypes.wintypes.HWND),
        ('hwndMoveSize', ctypes.wintypes.HWND),
        ('hwndCaret', ctypes.wintypes.HWND),
        ('rcCaret', ctypes.wintypes.RECT),
    ]

def run_command_or_exit_on_failure(command):
    """
    Run a shell command and exit if it fails.

    Args:
        command (list): The command to run as a list of strings.
    """
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        exit(1)

class InputSimulator:
    """
    A class to simulate keyboard input using various methods.
    """

    # Window class names that accept keyboard text input.
    # Includes both child-level focused controls (Edit, RichEdit, Scintilla,
    # Chrome_RenderWidgetHostHWND) AND top-level window classes as a fallback —
    # Chrome's render widget lives on a separate thread so GetGUIThreadInfo on
    # the browser thread returns hwndFocus=0, leaving only the top-level handle.
    _TEXT_INPUT_CLASSES = {
        'Edit',                          # Standard Win32 text boxes
        'RichEdit20W', 'RichEdit20A',    # Rich text editors (WordPad, etc.)
        'RICHEDIT50W',                   # Word and newer rich text controls
        'Scintilla',                     # Code editors (Notepad++, etc.)
        'Chrome_RenderWidgetHostHWND',   # Chrome / Edge / Electron (child control)
        'Chrome_WidgetWin_1',            # Chrome / Edge / Electron (top-level) ← main fix
        'MozillaWindowClass',            # Firefox
        'Notepad',                       # Windows Notepad
        'ConsoleWindowClass',            # Windows Terminal / cmd
        '_WwG',                          # Word document body
        'EXCEL7',                        # Excel cell-edit pane
        'paneClassDC',                   # PowerPoint slide pane
    }

    # Native classes whose cursor position can be queried via Win32 messages
    # (EM_GETSEL / WM_GETTEXT). Anything outside this set falls back to the
    # clipboard-probe path for the smart leading-space logic.
    _NATIVE_TEXT_CLASSES = {
        'Edit',
        'RichEdit20W', 'RichEdit20A', 'RICHEDIT50W',
        'Scintilla',
        'Notepad',
    }

    # Top-level classes for Office apps that activate ribbon KeyTip mode on
    # bare Alt press. When the activation chord includes Alt (e.g. alt+z),
    # Word/Excel/PPT respond to the Alt-down by parking the document caret
    # and routing focus to a NetUIHWND ribbon control. We detect that state
    # at paste-time and send Esc to dismiss it.
    _OFFICE_TOPLEVEL_CLASSES = {
        'OpusApp',          # Word
        'XLMAIN',           # Excel
        'PPTFrameClass',    # PowerPoint
    }

    # Office "document body" focus classes. If the focused control belongs
    # to one of these, the ribbon is NOT active and Esc would be wasted.
    _OFFICE_BODY_CLASSES = {
        '_WwG',             # Word
        'EXCEL7',           # Excel
        'paneClassDC',      # PowerPoint
    }

    def __init__(self):
        """
        Initialize the InputSimulator with the specified configuration.
        """
        self.input_method = ConfigManager.get_config_value('post_processing', 'input_method')
        self.dotool_process = None

        if self.input_method in ('pynput', 'clipboard'):
            self.keyboard = PynputController()
        elif self.input_method == 'dotool':
            self._initialize_dotool()

    def _initialize_dotool(self):
        """
        Initialize the dotool process for input simulation.
        """
        self.dotool_process = subprocess.Popen("dotool", stdin=subprocess.PIPE, text=True)
        assert self.dotool_process.stdin is not None

    def _terminate_dotool(self):
        """
        Terminate the dotool process if it's running.
        """
        if self.dotool_process:
            os.kill(self.dotool_process.pid, signal.SIGINT)
            self.dotool_process = None

    def typewrite(self, text):
        """
        Simulate typing the given text with the specified interval between keystrokes.

        Args:
            text (str): The text to type.
        """
        interval = ConfigManager.get_config_value('post_processing', 'writing_key_press_delay')
        if self.input_method == 'clipboard':
            self._typewrite_clipboard(text)
        elif self.input_method == 'pynput':
            self._typewrite_pynput(text, interval)
        elif self.input_method == 'ydotool':
            self._typewrite_ydotool(text, interval)
        elif self.input_method == 'dotool':
            self._typewrite_dotool(text, interval)

    def _foreground_focus(self):
        """Return (focus_hwnd, class_name) for the currently focused control.

        Returns the focus regardless of whether the class is a known text
        input. Caller decides whether to attempt paste vs. apply smart
        features. Returns None only when there is no foreground window or
        no focused control at all.
        """
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return None
            thread_id = user32.GetWindowThreadProcessId(hwnd, None)
            info = _GUITHREADINFO()
            info.cbSize = ctypes.sizeof(_GUITHREADINFO)
            if not user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)):
                return None
            focus_hwnd = info.hwndFocus or hwnd
            buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(focus_hwnd, buf, 256)
            return (focus_hwnd, buf.value)
        except Exception:
            return None

    def _focused_text_control(self):
        """Return (hwnd, class) only when class is a known text-input control."""
        focus = self._foreground_focus()
        if focus is not None and focus[1] in self._TEXT_INPUT_CLASSES:
            return focus
        return None

    def _is_text_input_focused(self):
        """Return True if the foreground window's focused control is a known text input."""
        return self._focused_text_control() is not None

    def _text_before_cursor(self, focus_hwnd, class_name, max_chars=8):
        """Return up to max_chars characters before the cursor.

        Returns:
            ''   - cursor is at position 0; no preceding chars.
            str  - up to max_chars preceding chars, in normal order.
            None - couldn't determine (active selection, ambiguous probe, error).
        """
        if class_name in self._NATIVE_TEXT_CLASSES:
            return self._text_before_cursor_native(focus_hwnd, max_chars)
        return self._text_before_cursor_probe()

    @staticmethod
    def _text_before_cursor_native(focus_hwnd, max_chars):
        """Win32 fast path: EM_GETSEL + WM_GETTEXT against the focused control."""
        try:
            user32 = ctypes.windll.user32
            EM_GETSEL = 0x00B0
            WM_GETTEXTLENGTH = 0x000E
            WM_GETTEXT = 0x000D

            start = ctypes.wintypes.DWORD(0)
            end = ctypes.wintypes.DWORD(0)
            user32.SendMessageW(focus_hwnd, EM_GETSEL,
                                ctypes.byref(start), ctypes.byref(end))
            if start.value != end.value:
                # Live selection — paste will overwrite; don't apply adjustments.
                return None
            if start.value == 0:
                return ''

            length = user32.SendMessageW(focus_hwnd, WM_GETTEXTLENGTH, 0, 0)
            if length <= 0 or start.value > length:
                return ''
            # Read just enough to cover positions 0..start (no offset arg in WM_GETTEXT).
            buf = ctypes.create_unicode_buffer(start.value + 1)
            user32.SendMessageW(focus_hwnd, WM_GETTEXT, start.value + 1, buf)
            text = buf.value
            cutoff = min(start.value, len(text))
            return text[max(0, cutoff - max_chars):cutoff]
        except Exception:
            return None

    def _text_before_cursor_probe(self):
        """Clipboard probe fallback: select up to 2 chars left, copy, restore.

        Two chars is enough to disambiguate the most common sentence-boundary
        case ('. ' vs ' '): a single trailing space looks the same in both,
        but the second char from the cursor reveals which.
        """
        try:
            saved = pyperclip.paste()
        except Exception:
            saved = ''
        try:
            # Multi-char so it can never be confused with the preceding 1-2 chars.
            sentinel = '__WW_PROBE_SENTINEL__'
            try:
                pyperclip.copy(sentinel)
            except Exception:
                return None
            time.sleep(0.02)
            with self.keyboard.pressed(Key.shift):
                self.keyboard.press(Key.left)
                self.keyboard.release(Key.left)
                self.keyboard.press(Key.left)
                self.keyboard.release(Key.left)
            time.sleep(0.02)
            with self.keyboard.pressed(Key.ctrl):
                self.keyboard.press('c')
                self.keyboard.release('c')
            time.sleep(0.04)
            try:
                probe = pyperclip.paste()
            except Exception:
                probe = sentinel
            # Collapse the selection back to the original cursor position.
            self.keyboard.press(Key.right)
            self.keyboard.release(Key.right)

            if probe == sentinel:
                # Clipboard untouched - nothing was selected (cursor at start).
                return ''
            if 1 <= len(probe) <= 2:
                return probe
            # Apps that "smart-copy" the current line on empty selection
            # produce ambiguous results - bail out conservatively.
            return None
        finally:
            try:
                pyperclip.copy(saved)
            except Exception:
                pass

    @staticmethod
    def _decide_text_adjustments(context):
        """Given preceding-text context, decide (prepend_space, lowercase_first).

        prepend_space: True if there's a non-whitespace char immediately
            before the cursor, so the new text would otherwise run on.

        lowercase_first: True if the cursor is mid-sentence (preceded by
            content that does NOT end in a sentence terminator or newline).
            False at sentence boundaries (start of doc/line, after .!?).
        """
        if context is None:
            return (False, False)
        if context == '':
            return (False, False)

        last = context[-1]
        prepend_space = last not in (' ', '\t', '\n', '\r')

        stripped = context.rstrip(' \t')
        if not stripped:
            lowercase_first = False
        elif stripped[-1] in ('\n', '\r'):
            lowercase_first = False
        elif stripped[-1] in '.!?':
            lowercase_first = False
        else:
            lowercase_first = True

        return (prepend_space, lowercase_first)

    @staticmethod
    def _lowercase_first_word(text):
        """Lowercase the first letter of text, but skip likely proper nouns
        and the English pronoun 'I'."""
        if not text or not text[0].isupper():
            return text
        # Find the end of the first alphabetic run.
        i = 0
        while i < len(text) and text[i].isalpha():
            i += 1
        first_word = text[:i]
        if first_word == 'I':
            return text  # English pronoun; keep capitalised.
        # If the first two letters are both upper, treat as acronym (JJ, USA).
        if len(first_word) >= 2 and first_word[1].isupper():
            return text
        return text[0].lower() + text[1:]

    # Cached SendInput plumbing (lazily built once on first use).
    _SI_TYPES = None

    @classmethod
    def _sendinput_types(cls):
        if cls._SI_TYPES is not None:
            return cls._SI_TYPES
        ULONG_PTR = ctypes.c_void_p

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [('dx', ctypes.c_long), ('dy', ctypes.c_long),
                        ('mouseData', ctypes.c_ulong), ('dwFlags', ctypes.c_ulong),
                        ('time', ctypes.c_ulong), ('dwExtraInfo', ULONG_PTR)]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [('wVk', ctypes.wintypes.WORD), ('wScan', ctypes.wintypes.WORD),
                        ('dwFlags', ctypes.wintypes.DWORD), ('time', ctypes.wintypes.DWORD),
                        ('dwExtraInfo', ULONG_PTR)]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [('uMsg', ctypes.wintypes.DWORD),
                        ('wParamL', ctypes.wintypes.WORD), ('wParamH', ctypes.wintypes.WORD)]

        class _INPUT_UNION(ctypes.Union):
            _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT), ('hi', HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [('type', ctypes.wintypes.DWORD), ('union', _INPUT_UNION)]

        cls._SI_TYPES = (INPUT, _INPUT_UNION, KEYBDINPUT)
        return cls._SI_TYPES

    @classmethod
    def _send_ctrl_v(cls):
        """Send Ctrl+V via Win32 SendInput, after lifting any sticky modifiers.

        Using SendInput directly (instead of pynput's Controller) sidesteps a
        Word-specific failure mode: Word treats Alt+Ctrl+V as "Paste Special"
        rather than paste, so if the user is still holding the activation
        chord (alt+z) — or the OS thinks they are — pynput's Ctrl+V opens the
        Paste Special dialog and inserts nothing. We pre-emptively send key-up
        events for Alt/Shift before pressing Ctrl+V to guarantee a clean
        accelerator. Releasing an already-up key is a no-op, so this is safe
        regardless of actual hand state.
        """
        INPUT, _INPUT_UNION, KEYBDINPUT = cls._sendinput_types()
        INPUT_KEYBOARD = 1
        KEYEVENTF_KEYUP = 0x0002

        VK_LMENU = 0xA4   # Left Alt
        VK_RMENU = 0xA5   # Right Alt
        VK_LSHIFT = 0xA0
        VK_RSHIFT = 0xA1
        VK_CONTROL = 0x11
        VK_V = 0x56

        def ki(vk, up=False):
            flags = KEYEVENTF_KEYUP if up else 0
            return INPUT(INPUT_KEYBOARD, _INPUT_UNION(ki=KEYBDINPUT(
                wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=None)))

        # Defensive: lift any modifier the user might still be holding from the
        # activation chord. Without this, Alt+Ctrl+V in Word = "Paste Special",
        # not paste.
        cleanup = [
            ki(VK_LMENU, up=True),
            ki(VK_RMENU, up=True),
            ki(VK_LSHIFT, up=True),
            ki(VK_RSHIFT, up=True),
        ]
        n = len(cleanup)
        ctypes.windll.user32.SendInput(n, (INPUT * n)(*cleanup), ctypes.sizeof(INPUT))
        time.sleep(0.02)

        paste = [
            ki(VK_CONTROL),
            ki(VK_V),
            ki(VK_V, up=True),
            ki(VK_CONTROL, up=True),
        ]
        n = len(paste)
        ctypes.windll.user32.SendInput(n, (INPUT * n)(*paste), ctypes.sizeof(INPUT))

    @classmethod
    def _send_esc(cls):
        """Send a clean Esc keystroke via SendInput, after lifting any sticky
        Alt the activation chord may have left virtually held. Used to dismiss
        Office's ribbon KeyTip mode before pasting."""
        INPUT, _INPUT_UNION, KEYBDINPUT = cls._sendinput_types()
        INPUT_KEYBOARD = 1
        KEYEVENTF_KEYUP = 0x0002
        VK_LMENU = 0xA4
        VK_RMENU = 0xA5
        VK_ESCAPE = 0x1B

        def ki(vk, up=False):
            flags = KEYEVENTF_KEYUP if up else 0
            return INPUT(INPUT_KEYBOARD, _INPUT_UNION(ki=KEYBDINPUT(
                wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=None)))

        cleanup = [ki(VK_LMENU, up=True), ki(VK_RMENU, up=True)]
        n = len(cleanup)
        ctypes.windll.user32.SendInput(n, (INPUT * n)(*cleanup), ctypes.sizeof(INPUT))
        seq = [ki(VK_ESCAPE), ki(VK_ESCAPE, up=True)]
        n = len(seq)
        ctypes.windll.user32.SendInput(n, (INPUT * n)(*seq), ctypes.sizeof(INPUT))

    def _dismiss_office_ribbon_if_active(self):
        """Office (Word/Excel/PPT) activates ribbon KeyTip mode on bare Alt
        press. With activation_key=alt+z, the user's Alt-down is observed by
        Office before pynput's listener fires, so by the time we paste, focus
        sits on a NetUIHWND ribbon control instead of the document body —
        Ctrl+V then either misses the doc or fires the wrong accelerator.

        Detect the state (foreground is Office, focus is NOT a body class)
        and send Esc to dismiss. Esc in the document body is a no-op, so the
        check is defensive — if the user happens to be paste-targeting Word
        with the doc body already focused, we don't disturb it.
        """
        try:
            user32 = ctypes.windll.user32
            top_hwnd = user32.GetForegroundWindow()
            if not top_hwnd:
                return
            top_buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(top_hwnd, top_buf, 256)
            if top_buf.value not in self._OFFICE_TOPLEVEL_CLASSES:
                return
            focus = self._foreground_focus()
            if focus is not None and focus[1] in self._OFFICE_BODY_CLASSES:
                return
            ConfigManager.console_print(
                f'Office ribbon detected (top={top_buf.value!r}, '
                f'focus={focus[1] if focus else None!r}); sending Esc.'
            )
            self._send_esc()
            time.sleep(0.05)
        except Exception:
            pass

    def _typewrite_clipboard(self, text):
        # If foreground is an Office app whose ribbon got activated by the
        # Alt in the activation chord, dismiss it before we read the focus
        # so the rest of the function sees the document body, not a ribbon
        # control. No-op for non-Office apps.
        self._dismiss_office_ribbon_if_active()

        focus = self._foreground_focus()
        if focus is None:
            # No foreground window / focus at all - leave in clipboard for manual paste.
            pyperclip.copy(text)
            return

        focus_hwnd, class_name = focus
        is_known_text = class_name in self._TEXT_INPUT_CLASSES
        ConfigManager.console_print(
            f'Paste target: hwnd=0x{focus_hwnd:X} class={class_name!r} '
            f'known_text={is_known_text}'
        )

        if is_known_text:
            want_space = ConfigManager.get_config_value(
                'post_processing', 'add_leading_space_if_needed')
            want_lower = ConfigManager.get_config_value(
                'post_processing', 'lowercase_first_letter_mid_sentence')
            if want_space or want_lower:
                context = self._text_before_cursor(focus_hwnd, class_name)
                prepend_space, lowercase_first = self._decide_text_adjustments(context)
                if want_space and prepend_space:
                    text = ' ' + text
                if want_lower and lowercase_first:
                    text = self._lowercase_first_word(text)

        # Save whatever the user had copied, paste transcription, then restore.
        # Ctrl+V is the only truly instantaneous path (browsers process WM_CHAR one at a time).
        # We paste even for unknown-class focus targets (Word's _WwG, modern Notepad,
        # etc.) — Ctrl+V is universally "paste" and is harmless if the focus turns
        # out to not accept text.
        try:
            saved = pyperclip.paste()
        except Exception:
            saved = ''
        pyperclip.copy(text)
        time.sleep(0.05)
        # SendInput with explicit Alt/Shift release first — pynput's Ctrl+V
        # was opening Word's Paste Special dialog when the activation chord's
        # Alt was still virtually down. Direct SendInput also routes through
        # Word's accelerator handling more reliably.
        self._send_ctrl_v()
        # Wait long enough for Word's clipboard reader to consume the data
        # before we restore the previous clipboard contents. 100ms wasn't
        # enough on Word with its rich-paste pipeline; 300ms covers it.
        time.sleep(0.3)
        try:
            pyperclip.copy(saved)
        except Exception:
            pass

    def _typewrite_pynput(self, text, interval):
        """
        Simulate typing using pynput.

        Args:
            text (str): The text to type.
            interval (float): The interval between keystrokes in seconds.
        """
        for char in text:
            self.keyboard.press(char)
            self.keyboard.release(char)
            time.sleep(interval)

    def _typewrite_ydotool(self, text, interval):
        """
        Simulate typing using ydotool.

        Args:
            text (str): The text to type.
            interval (float): The interval between keystrokes in seconds.
        """
        cmd = "ydotool"
        run_command_or_exit_on_failure([
            cmd,
            "type",
            "--key-delay",
            str(interval * 1000),
            "--",
            text,
        ])

    def _typewrite_dotool(self, text, interval):
        """
        Simulate typing using dotool.

        Args:
            text (str): The text to type.
            interval (float): The interval between keystrokes in seconds.
        """
        assert self.dotool_process and self.dotool_process.stdin
        self.dotool_process.stdin.write(f"typedelay {interval * 1000}\n")
        self.dotool_process.stdin.write(f"type {text}\n")
        self.dotool_process.stdin.flush()

    def cleanup(self):
        """
        Perform cleanup operations, such as terminating the dotool process.
        """
        if self.input_method == 'dotool':
            self._terminate_dotool()
