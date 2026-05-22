"""Dictation-pipeline diagnostic logger.

Writes one line per stage to ww_dict_diag.log next to transcript_log.txt.
Used to localize bugs where the typed text disagrees with the polished text
in transcript_log.txt — captures the exact value at each handoff so we can
see which stage mutated/truncated the content.

Remove once the single-character-paste bug and polish-truncation bug are
both verified resolved.
"""

import os
from datetime import datetime as _datetime


_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'ww_dict_diag.log',
)


def dd(stage, value=None, **extras):
    """Append a diagnostic line. Long values are middle-truncated so the log
    stays scannable while still showing the head + tail of any cutoff."""
    try:
        ts = _datetime.now().isoformat(timespec='milliseconds')
        if value is None:
            v = '<None>'
        else:
            v = repr(value)
            if len(v) > 600:
                v = f'{v[:300]} ...[{len(v) - 600}c omitted]... {v[-300:]}'
        line = f'[{ts}] {stage}  len={len(value) if isinstance(value, str) else "n/a"}  value={v}'
        for k, val in extras.items():
            line += f'  {k}={val!r}'
        with open(_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass
