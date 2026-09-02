from __future__ import annotations

import pytest

from leds import _cache


@pytest.fixture(autouse=True)
def _cold_shared_caches():
    """Start every test with empty process-wide caches.

    The caches in :mod:`leds._cache` are deliberately shared by all sessions
    in a worker, which also means they outlive a test. Clearing keeps tests
    independent of each other's data and of their execution order.
    """
    _cache.clear_all()
    yield
    _cache.clear_all()
