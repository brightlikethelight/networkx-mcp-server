"""Tests for GraphCache and GraphDict classes.

Covers LRU eviction, TTL expiration, thread safety, memory limits,
cache statistics, GraphDict backward compatibility, and shutdown.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from networkx_mcp.graph_cache import CachedGraph, GraphCache, GraphDict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(n: int = 3) -> nx.Graph:
    """Return a small path graph for testing."""
    return nx.path_graph(n)


@pytest.fixture()
def cache():
    """Fresh GraphCache with small limits for fast tests."""
    c = GraphCache(max_size=5, ttl_seconds=10, max_memory_mb=500, cleanup_interval=9999)
    yield c
    c.shutdown()


@pytest.fixture()
def tiny_cache():
    """GraphCache with max_size=3 for eviction tests."""
    c = GraphCache(max_size=3, ttl_seconds=10, max_memory_mb=500, cleanup_interval=9999)
    yield c
    c.shutdown()


@pytest.fixture()
def gdict(cache):
    """GraphDict backed by the cache fixture."""
    return GraphDict(cache)


# ===================================================================
# Basic put / get / delete
# ===================================================================


class TestBasicOperations:
    def test_put_and_get(self, cache):
        g = _make_graph()
        cache.put("g1", g)
        result = cache.get("g1")
        assert result is g

    def test_get_missing_returns_none(self, cache):
        assert cache.get("nonexistent") is None

    def test_put_overwrites_existing(self, cache):
        g1 = _make_graph(3)
        g2 = _make_graph(5)
        cache.put("g", g1)
        cache.put("g", g2)
        assert cache.get("g") is g2
        assert len(cache.list_graphs()) == 1

    def test_delete_existing(self, cache):
        cache.put("g", _make_graph())
        assert cache.delete("g") is True
        assert cache.get("g") is None

    def test_delete_missing_returns_false(self, cache):
        assert cache.delete("nope") is False

    def test_clear(self, cache):
        for i in range(4):
            cache.put(f"g{i}", _make_graph())
        cache.clear()
        assert cache.list_graphs() == []

    def test_clear_increments_evictions(self, cache):
        for i in range(3):
            cache.put(f"g{i}", _make_graph())
        cache.clear()
        assert cache.evictions == 3

    def test_list_graphs(self, cache):
        cache.put("a", _make_graph())
        cache.put("b", _make_graph())
        keys = cache.list_graphs()
        assert set(keys) == {"a", "b"}


# ===================================================================
# LRU eviction
# ===================================================================


class TestLRUEviction:
    def test_evicts_oldest_when_full(self, tiny_cache):
        """Inserting a 4th item into a max_size=3 cache evicts the first."""
        tiny_cache.put("a", _make_graph())
        tiny_cache.put("b", _make_graph())
        tiny_cache.put("c", _make_graph())
        tiny_cache.put("d", _make_graph())

        assert tiny_cache.get("a") is None  # evicted
        assert tiny_cache.get("d") is not None
        assert tiny_cache.evictions == 1

    def test_access_refreshes_lru_order(self, tiny_cache):
        """Accessing 'a' moves it to end, so 'b' becomes the LRU victim."""
        tiny_cache.put("a", _make_graph())
        tiny_cache.put("b", _make_graph())
        tiny_cache.put("c", _make_graph())

        # Touch 'a' to move it to most-recently-used
        tiny_cache.get("a")

        # Insert 'd' — should evict 'b', not 'a'
        tiny_cache.put("d", _make_graph())

        assert tiny_cache.get("b") is None
        assert tiny_cache.get("a") is not None
        assert tiny_cache.get("c") is not None

    def test_multiple_evictions(self, tiny_cache):
        """Inserting 6 items into a max_size=3 cache evicts first 3."""
        for i in range(6):
            tiny_cache.put(f"g{i}", _make_graph())

        assert len(tiny_cache.list_graphs()) == 3
        assert tiny_cache.evictions == 3
        # Only the last 3 survive
        for i in range(3, 6):
            assert tiny_cache.get(f"g{i}") is not None


# ===================================================================
# TTL expiration
# ===================================================================


class TestTTLExpiration:
    def test_expired_graph_returns_none(self, cache):
        cache.put("g", _make_graph())

        # Fast-forward time past TTL
        with patch("networkx_mcp.graph_cache.time") as mock_time:
            # The put already happened with real time; mock only affects get
            mock_time.time.return_value = time.time() + cache.ttl_seconds + 1
            result = cache.get("g")

        assert result is None

    def test_expired_graph_counts_as_miss_and_eviction(self, cache):
        cache.put("g", _make_graph())

        with patch("networkx_mcp.graph_cache.time") as mock_time:
            mock_time.time.return_value = time.time() + cache.ttl_seconds + 1
            cache.get("g")

        assert cache.misses == 1
        assert cache.evictions == 1

    def test_not_expired_within_ttl(self, cache):
        cache.put("g", _make_graph())

        with patch("networkx_mcp.graph_cache.time") as mock_time:
            mock_time.time.return_value = time.time() + cache.ttl_seconds - 1
            mock_time.sleep = time.sleep  # keep sleep working
            result = cache.get("g")

        assert result is not None

    def test_cleanup_removes_expired(self):
        """The _cleanup method should purge expired entries."""
        c = GraphCache(
            max_size=10, ttl_seconds=5, max_memory_mb=500, cleanup_interval=9999
        )
        c.put("old", _make_graph())

        with patch("networkx_mcp.graph_cache.time") as mock_time:
            mock_time.time.return_value = time.time() + 6
            c._cleanup()

        assert c.get("old") is None
        assert c.evictions >= 1
        c.shutdown()


# ===================================================================
# Thread safety
# ===================================================================


class TestThreadSafety:
    def test_concurrent_puts(self, cache):
        """Many threads putting concurrently should not corrupt the cache."""

        def writer(thread_id):
            for i in range(20):
                cache.put(f"t{thread_id}_g{i}", _make_graph())

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # No thread should still be alive
        for t in threads:
            assert not t.is_alive()

        # Cache should not exceed max_size
        assert len(cache.list_graphs()) <= cache.max_size

    def test_concurrent_reads_and_writes(self, cache):
        """Mixed readers and writers should not raise."""
        cache.put("shared", _make_graph())
        results = []

        def reader():
            for _ in range(50):
                r = cache.get("shared")
                results.append(r)

        def writer():
            for i in range(50):
                cache.put(f"w{i}", _make_graph())

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        for t in threads:
            assert not t.is_alive()

    def test_concurrent_deletes(self, cache):
        """Concurrent deletes on overlapping keys should not raise."""
        for i in range(10):
            cache.put(f"g{i}", _make_graph())

        def deleter(start):
            for i in range(start, start + 10):
                cache.delete(f"g{i}")

        threads = [threading.Thread(target=deleter, args=(0,)) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(cache.list_graphs()) == 0


# ===================================================================
# Memory limit enforcement
# ===================================================================


class TestMemoryLimit:
    def test_evicts_when_memory_exceeded(self):
        """When psutil reports memory over limit, put() should evict entries."""
        c = GraphCache(
            max_size=10, ttl_seconds=3600, max_memory_mb=100, cleanup_interval=9999
        )

        c.put("g0", _make_graph())
        c.put("g1", _make_graph())
        c.put("g2", _make_graph())

        # Simulate memory over limit, then back under after evictions
        mem_values = iter([150.0, 150.0, 50.0])

        with patch.object(
            c, "_get_memory_usage_mb", side_effect=lambda: next(mem_values)
        ):
            c.put("g3", _make_graph())

        # At least some entries were evicted
        assert c.evictions >= 1
        c.shutdown()

    def test_no_eviction_when_memory_ok(self):
        """No memory-based eviction when usage is below threshold."""
        c = GraphCache(
            max_size=10, ttl_seconds=3600, max_memory_mb=500, cleanup_interval=9999
        )

        with patch.object(c, "_get_memory_usage_mb", return_value=100.0):
            for i in range(5):
                c.put(f"g{i}", _make_graph())

        assert c.evictions == 0
        assert len(c.list_graphs()) == 5
        c.shutdown()

    def test_get_memory_usage_without_psutil(self):
        """Without psutil, _get_memory_usage_mb returns 0."""
        c = GraphCache(
            max_size=5, ttl_seconds=10, max_memory_mb=500, cleanup_interval=9999
        )

        with patch("networkx_mcp.graph_cache.HAS_PSUTIL", False):
            assert c._get_memory_usage_mb() == 0

        c.shutdown()

    def test_get_memory_usage_with_psutil(self):
        """With psutil available, memory is read from process info."""
        c = GraphCache(
            max_size=5, ttl_seconds=10, max_memory_mb=500, cleanup_interval=9999
        )

        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 200 * 1024 * 1024  # 200 MB

        with (
            patch("networkx_mcp.graph_cache.HAS_PSUTIL", True),
            patch("networkx_mcp.graph_cache.psutil") as mock_psutil,
        ):
            mock_psutil.Process.return_value = mock_process
            usage = c._get_memory_usage_mb()

        assert abs(usage - 200.0) < 0.1
        c.shutdown()


# ===================================================================
# Cache statistics
# ===================================================================


class TestCacheStatistics:
    def test_hits_and_misses(self, cache):
        cache.put("a", _make_graph())
        cache.get("a")  # hit
        cache.get("a")  # hit
        cache.get("z")  # miss

        assert cache.hits == 2
        assert cache.misses == 1

    def test_hit_rate_calculation(self, cache):
        cache.put("a", _make_graph())
        cache.get("a")  # hit
        cache.get("b")  # miss

        stats = cache.get_stats()
        assert stats["hit_rate"] == pytest.approx(0.5)

    def test_hit_rate_zero_requests(self, cache):
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0

    def test_eviction_counter(self, tiny_cache):
        for i in range(5):
            tiny_cache.put(f"g{i}", _make_graph())
        # max_size=3, so 2 evictions
        assert tiny_cache.evictions == 2

    def test_stats_snapshot(self, cache):
        cache.put("a", _make_graph())
        cache.put("b", _make_graph())
        cache.get("a")
        cache.get("missing")

        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["max_size"] == cache.max_size
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["evictions"] == 0

    def test_eviction_on_ttl_expiry_counted(self, cache):
        cache.put("g", _make_graph())

        with patch("networkx_mcp.graph_cache.time") as mock_time:
            mock_time.time.return_value = time.time() + cache.ttl_seconds + 1
            cache.get("g")

        stats = cache.get_stats()
        assert stats["evictions"] == 1
        assert stats["misses"] == 1


# ===================================================================
# GraphDict backward compatibility
# ===================================================================


class TestGraphDict:
    def test_setitem_and_getitem(self, gdict):
        g = _make_graph()
        gdict["k"] = g
        assert gdict["k"] is g

    def test_getitem_missing_raises_keyerror(self, gdict):
        with pytest.raises(KeyError):
            _ = gdict["missing"]

    def test_delitem(self, gdict):
        gdict["k"] = _make_graph()
        del gdict["k"]
        with pytest.raises(KeyError):
            _ = gdict["k"]

    def test_delitem_missing_raises_keyerror(self, gdict):
        with pytest.raises(KeyError):
            del gdict["missing"]

    def test_contains(self, gdict):
        gdict["k"] = _make_graph()
        assert "k" in gdict
        assert "missing" not in gdict

    def test_contains_non_string_key(self, gdict):
        assert 42 not in gdict

    def test_len(self, gdict):
        assert len(gdict) == 0
        gdict["a"] = _make_graph()
        gdict["b"] = _make_graph()
        assert len(gdict) == 2

    def test_iter(self, gdict):
        gdict["x"] = _make_graph()
        gdict["y"] = _make_graph()
        assert set(gdict) == {"x", "y"}

    def test_keys(self, gdict):
        gdict["a"] = _make_graph()
        gdict["b"] = _make_graph()
        assert set(gdict.keys()) == {"a", "b"}

    def test_get_existing(self, gdict):
        g = _make_graph()
        gdict["k"] = g
        assert gdict.get("k") is g

    def test_get_missing_returns_default(self, gdict):
        assert gdict.get("missing") is None
        assert gdict.get("missing", 42) == 42

    def test_clear(self, gdict, cache):
        gdict["a"] = _make_graph()
        gdict["b"] = _make_graph()
        gdict.clear()
        assert len(gdict) == 0

    def test_shutdown_delegates(self, cache):
        gd = GraphDict(cache)
        with patch.object(cache, "shutdown") as mock_shutdown:
            gd.shutdown()
            mock_shutdown.assert_called_once()


# ===================================================================
# CachedGraph dataclass
# ===================================================================


class TestCachedGraph:
    def test_touch_updates_access(self):
        cg = CachedGraph(graph=_make_graph(), created_at=1.0, last_accessed=1.0)
        before = cg.last_accessed
        cg.touch()
        assert cg.last_accessed >= before
        assert cg.access_count == 1

    def test_touch_increments_count(self):
        cg = CachedGraph(graph=_make_graph(), created_at=1.0, last_accessed=1.0)
        cg.touch()
        cg.touch()
        cg.touch()
        assert cg.access_count == 3


# ===================================================================
# Shutdown
# ===================================================================


class TestShutdown:
    def test_shutdown_sets_flag(self):
        c = GraphCache(
            max_size=5, ttl_seconds=10, max_memory_mb=500, cleanup_interval=9999
        )
        assert c._shutdown is False
        c.shutdown()
        assert c._shutdown is True

    def test_shutdown_joins_cleanup_thread(self):
        c = GraphCache(
            max_size=5, ttl_seconds=10, max_memory_mb=500, cleanup_interval=9999
        )
        c.shutdown()
        # After shutdown the daemon thread should have been joined (or is dead)
        assert not c._cleanup_thread.is_alive() or c._cleanup_thread.daemon

    def test_double_shutdown_is_safe(self):
        c = GraphCache(
            max_size=5, ttl_seconds=10, max_memory_mb=500, cleanup_interval=9999
        )
        c.shutdown()
        c.shutdown()  # should not raise


# ===================================================================
# GraphCache.has() — stat-free existence check
# ===================================================================


class TestGraphCacheHas:
    def test_has_does_not_increment_hits(self, cache):
        cache.put("g", _make_graph())
        initial_hits = cache.hits
        assert cache.has("g") is True
        assert cache.hits == initial_hits

    def test_has_returns_false_for_missing(self, cache):
        assert cache.has("nonexistent") is False

    def test_has_evicts_expired(self):
        cache = GraphCache(
            max_size=10, ttl_seconds=0.001, max_memory_mb=500, cleanup_interval=9999
        )
        cache.put("g", _make_graph())
        time.sleep(0.01)
        assert cache.has("g") is False
        cache.shutdown()

    def test_contains_does_not_pollute_hits(self, cache):
        gd = GraphDict(cache)
        gd["g"] = _make_graph()
        initial_hits = cache.hits
        assert "g" in gd
        assert cache.hits == initial_hits
