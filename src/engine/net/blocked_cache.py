"""5-minute cache of the "Groq is blocked" verdict, plus a Windows-friendly
VPN-active detector. Mirrors the Mobile `HttpClientHolder.isLikelyBlocked`
+ `NetDiagnostics.isOnVpn` pair so the PC's STT path can short-circuit to
Gemini under the same conditions.

A VPN-active detection here is just a *hint*: if true, we skip Groq even
without first paying a 403 round-trip. The cached verdict catches the case
where the network is non-VPN but Groq is still WAF-blocked (rate-limited
residential IP, corporate proxy, etc.).
"""

from __future__ import annotations

import re
import threading
import time
from typing import Optional


_BLOCKED_TTL_SEC = 5 * 60.0

_lock = threading.Lock()
_last_blocked_at: float = 0.0


def mark_blocked() -> None:
    """Called by the STT layer when a 403 is observed. Subsequent calls
    within `_BLOCKED_TTL_SEC` will short-circuit to Gemini without paying
    another guaranteed-403 round-trip to Groq."""
    global _last_blocked_at
    with _lock:
        _last_blocked_at = time.time()


def clear_blocked() -> None:
    """Reset the verdict — called when a Groq probe or call succeeds, so
    toggling VPN off recovers Groq routing without waiting for TTL."""
    global _last_blocked_at
    with _lock:
        _last_blocked_at = 0.0


def is_likely_blocked() -> bool:
    """True iff a 403 was observed within the last `_BLOCKED_TTL_SEC`."""
    with _lock:
        ts = _last_blocked_at
    if ts == 0.0:
        return False
    return (time.time() - ts) < _BLOCKED_TTL_SEC


# Adapter-name patterns that strongly suggest a VPN tunnel adapter on Windows.
# Names are matched case-insensitively. Order isn't significant.
#
# Why a regex over psutil and not a routing-table check: Windows doesn't
# expose a cheap "active network is a VPN" API like Android does. The
# default-route adapter is reliable but requires `wmic` or pywin32 round-
# trips that add ~80-200ms to every dictation. Adapter-presence detection
# via psutil is sub-millisecond and good enough — if a known VPN adapter is
# *up*, the user is almost certainly routing through it. False positives
# (user has the adapter installed but no active connection) just mean we
# route to Gemini unnecessarily on one dictation; the cached-block + Groq
# success path corrects on the next call.
_VPN_ADAPTER_PATTERNS = [
    r'\btun\b', r'\btap\b',
    r'wireguard', r'wintun',
    r'nordlynx', r'nordvpn',
    r'openvpn',
    r'mullvad',
    r'expressvpn',
    r'protonvpn',
    r'cisco anyconnect', r'anyconnect',
    r'fortinet', r'forticlient',
    r'pulse secure',
    r'globalprotect',
]

_VPN_REGEX = re.compile('|'.join(_VPN_ADAPTER_PATTERNS), re.IGNORECASE)


def is_on_vpn() -> bool:
    """True iff a known-VPN network adapter is currently up. Best-effort —
    psutil enumerates adapters; we look for known VPN driver names in the
    up-and-running set. Returns False on any error so the path silently
    falls back to the cached-block detection.

    Caller should treat this as a *hint to prefer Gemini*, not a hard
    declaration that Groq will fail.
    """
    try:
        import psutil  # imported lazily — keeps cold-start cheap when feature unused
    except ImportError:
        return False
    try:
        stats = psutil.net_if_stats()
    except Exception:
        return False
    for name, st in stats.items():
        if not st.isup:
            continue
        if _VPN_REGEX.search(name):
            return True
    return False


def reason_label(on_vpn: bool, cached_block: bool) -> Optional[str]:
    """Short greppable label for logs / NET_CAPS phase tags. None if neither
    signal fired."""
    if on_vpn and cached_block:
        return 'vpn-and-cached-block'
    if on_vpn:
        return 'vpn-preflight'
    if cached_block:
        return 'cached-block'
    return None
