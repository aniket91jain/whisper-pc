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
    """True iff the network adapter currently carrying the default route is
    a known VPN adapter (NordLynx, WireGuard, OpenVPN, etc.).

    Strict variant: a mere "VPN adapter is up" check would false-positive
    when a VPN client is installed but disconnected — NordLynx in particular
    stays `isup=True` with stale state after a Disconnect. We instead ask
    the OS which adapter owns the outbound path via a UDP socket connect
    (no packets sent — `connect()` on UDP just picks a route) and check
    that adapter's name. This matches what `route print` would show as
    the default-route interface, without spawning a process.

    Returns False on any error (no internet, psutil not installed) so the
    pipeline falls back to the cached-block detection.
    """
    import socket
    try:
        import psutil  # lazy — keeps cold-start cheap when feature unused
    except ImportError:
        return False

    # Step 1: find the local IP that would be used to reach the public internet.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Public anycast IP; no packets are sent — UDP connect() just
            # binds to the local interface that the OS would route via.
            s.connect(('8.8.8.8', 80))
            outbound_ip = s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return False  # no internet / interface down

    # Sanity: 0.0.0.0 means no route picked; treat as not-VPN.
    if not outbound_ip or outbound_ip == '0.0.0.0':
        return False

    # Step 2: find the adapter that owns that IP, check its name.
    try:
        addrs = psutil.net_if_addrs()
    except Exception:
        return False
    for name, entries in addrs.items():
        for entry in entries:
            if entry.family == socket.AF_INET and entry.address == outbound_ip:
                return bool(_VPN_REGEX.search(name))
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
