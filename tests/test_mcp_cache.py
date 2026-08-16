from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

from app.mcp.cache import TTLCache


def test_get_or_compute_coalesces_concurrent_cache_misses() -> None:
    cache: TTLCache[str, str] = TTLCache(ttl_seconds=60)
    callers = 8
    barrier = Barrier(callers)
    computation_started = Event()
    release_computation = Event()
    call_count = 0
    call_count_lock = Lock()

    def compute() -> str:
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        computation_started.set()
        assert release_computation.wait(timeout=2)
        return "catalog"

    def read() -> str:
        barrier.wait(timeout=2)
        return cache.get_or_compute("sources", compute)

    with ThreadPoolExecutor(max_workers=callers) as executor:
        futures = [executor.submit(read) for _ in range(callers)]
        assert computation_started.wait(timeout=2)
        release_computation.set()
        results = [future.result(timeout=2) for future in futures]

    assert results == ["catalog"] * callers
    assert call_count == 1


def test_get_or_compute_caches_none_values() -> None:
    cache: TTLCache[str, str | None] = TTLCache(ttl_seconds=60)
    call_count = 0

    def compute() -> None:
        nonlocal call_count
        call_count += 1

    assert cache.get_or_compute("missing", compute) is None
    assert cache.get_or_compute("missing", compute) is None
    assert call_count == 1
