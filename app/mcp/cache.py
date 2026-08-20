from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from threading import Condition, Lock
from time import monotonic
from typing import Generic, TypeVar, cast

Key = TypeVar("Key", bound=Hashable)
Value = TypeVar("Value")
_MISSING = object()


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
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._inflight: set[Key] = set()

    def get(self, key: Key) -> Value | None:
        with self._lock:
            value = self._get_locked(key)
            return None if value is _MISSING else cast(Value, value)

    def put(self, key: Key, value: Value) -> None:
        with self._lock:
            self._put_locked(key, value)

    def get_or_compute(self, key: Key, compute: Callable[[], Value]) -> Value:
        """Return one computed value when concurrent callers miss the same key."""
        with self._condition:
            while True:
                cached = self._get_locked(key)
                if cached is not _MISSING:
                    return cast(Value, cached)
                if key not in self._inflight:
                    self._inflight.add(key)
                    break
                self._condition.wait()

        try:
            value = compute()
        except BaseException:
            with self._condition:
                self._inflight.remove(key)
                self._condition.notify_all()
            raise

        with self._condition:
            self._put_locked(key, value)
            self._inflight.remove(key)
            self._condition.notify_all()
        return value

    def _get_locked(self, key: Key) -> Value | object:
        entry = self._entries.get(key)
        if entry is None:
            return _MISSING
        if entry.expires_at <= monotonic():
            self._entries.pop(key, None)
            return _MISSING
        self._entries.move_to_end(key)
        return entry.value

    def _put_locked(self, key: Key, value: Value) -> None:
        self._entries[key] = _Entry(monotonic() + self._ttl_seconds, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
