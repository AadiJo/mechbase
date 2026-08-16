from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass
from time import monotonic
from typing import Generic, TypeVar

Key = TypeVar("Key", bound=Hashable)
Value = TypeVar("Value")


@dataclass(frozen=True)
class _Entry(Generic[Value]):
    expires_at: float
    value: Value


class TTLCache(Generic[Key, Value]):
    """Small in-process cache for immutable MCP retrieval responses."""

    def __init__(self, ttl_seconds: float, max_entries: int = 256) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._entries: OrderedDict[Key, _Entry[Value]] = OrderedDict()

    def get(self, key: Key) -> Value | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= monotonic():
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return entry.value

    def put(self, key: Key, value: Value) -> None:
        self._entries[key] = _Entry(monotonic() + self._ttl_seconds, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
