"""Tiny event bus for cross-layer notifications.

Polish / persistence code does not import PyQt directly; instead it fires
events here, and the main app (or any other listener) subscribes. This keeps
the polish pipeline thread-agnostic — the subscriber is responsible for
marshalling to the right thread (e.g. via a queued Qt signal) before touching
UI.
"""

from typing import Callable, List, Optional

_dict_addition_listener: Optional[Callable[[List[str]], None]] = None


def register_dict_addition_listener(fn: Callable[[List[str]], None]) -> None:
    """Register a callback invoked when the polish auto-add persists new words.

    The callback may be invoked from any thread. If it touches UI it must
    marshal to the main thread itself (a queued Qt signal works).
    """
    global _dict_addition_listener
    _dict_addition_listener = fn


def fire_dict_addition(words: List[str]) -> None:
    """Notify the registered listener about new dictionary words. No-op if
    no listener is registered or the words list is empty."""
    if not words:
        return
    listener = _dict_addition_listener
    if listener is None:
        return
    try:
        listener(list(words))
    except Exception:
        # Listener errors must never break the polish pipeline.
        pass
