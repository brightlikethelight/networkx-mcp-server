"""Comprehensive tests for graph_cache.py module - Target: 95%+ coverage."""

import threading
import time
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from networkx_mcp.graph_cache import CachedGraph, GraphCache, GraphDict, get_graph_cache


class TestCachedGraph:
    """Test CachedGraph dataclass."""

    def test_cached_graph_creation(self):
        """Test creating a CachedGraph instance."""
        graph = nx.Graph()
        graph.add_edge("A", "B")

        cached = CachedGraph(
            graph=graph,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=0,
        )

        assert cached.graph == graph
        assert cached.access_count == 0
        assert isinstance(cached.created_at, float)
        assert isinstance(cached.last_accessed, float)

    def test_touch(self):
        """Test touch method updates access time and count."""
        graph = nx.Graph()
        cached = CachedGraph(
            graph=graph,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=0,
        )

        initial_time = cached.last_accessed
        initial_count = cached.access_count

        time.sleep(0.01)  # Small delay
        cached.touch()

        assert cached.last_accessed > initial_time
        assert cached.access_count == initial_count + 1


class TestGraphCache:
    """Test GraphCache class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache = GraphCache(max_size=3, ttl_seconds=10, max_memory_mb=100)

    def test_init(self):
        """Test cache initialization."""
        cache = GraphCache(max_size=5, ttl_seconds=30, max_memory_mb=10)

        assert cache.max_size == 5
        assert cache.ttl_seconds == 30
        assert cache.max_memory_mb == 10
        assert len(cache._cache) == 0
        assert cache._lock is not None
        assert cache._cleanup_thread is not None
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.evictions == 0

    def test_put_and_get(self):
        """Test basic put and get operations."""
        graph = nx.Graph()
        graph.add_edge("A", "B")

        # Put graph
        self.cache.put("test_graph", graph)

        # Get graph
        retrieved = self.cache.get("test_graph")
        assert retrieved == graph
        assert self.cache.hits == 1

        # Get non-existent
        assert self.cache.get("non_existent") is None
        assert self.cache.misses == 1

    def test_lru_eviction(self):
        """Test LRU eviction when max_size is reached."""
        # Add graphs up to max_size
        for i in range(3):
            graph = nx.Graph()
            graph.add_node(f"node_{i}")
            self.cache.put(f"graph_{i}", graph)

        # Access graph_0 to make it recently used
        self.cache.get("graph_0")

        # Add one more - should evict graph_1 (least recently used)
        new_graph = nx.Graph()
        new_graph.add_node("new")
        self.cache.put("graph_new", new_graph)

        assert self.cache.get("graph_0") is not None  # Still present
        assert self.cache.get("graph_1") is None  # Evicted
        assert self.cache.get("graph_2") is not None  # Still present
        assert self.cache.get("graph_new") is not None  # New one present
        assert self.cache.evictions > 0

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = GraphCache(max_size=10, ttl_seconds=0.1)  # 100ms TTL

        graph = nx.Graph()
        cache.put("temp_graph", graph)

        # Should exist immediately
        assert cache.get("temp_graph") is not None

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired
        assert cache.get("temp_graph") is None
        assert cache.evictions > 0

    def test_delete(self):
        """Test delete operation."""
        graph = nx.Graph()
        self.cache.put("to_delete", graph)

        assert self.cache.get("to_delete") is not None

        # Delete
        result = self.cache.delete("to_delete")
        assert result is True
        assert self.cache.get("to_delete") is None

        # Delete non-existent
        result = self.cache.delete("non_existent")
        assert result is False

    def test_clear(self):
        """Test clear operation."""
        # Add multiple graphs
        for i in range(3):
            graph = nx.Graph()
            self.cache.put(f"graph_{i}", graph)

        assert len(self.cache._cache) == 3

        # Clear
        self.cache.clear()
        assert len(self.cache._cache) == 0
        assert self.cache.evictions >= 3

    def test_list_graphs(self):
        """Test list_graphs operation."""
        graphs = {}
        for i in range(3):
            graph = nx.Graph()
            graph_id = f"graph_{i}"
            graphs[graph_id] = graph
            self.cache.put(graph_id, graph)

        keys = self.cache.list_graphs()
        assert len(keys) == 3
        for key in keys:
            assert key in graphs

    def test_get_stats(self):
        """Test get_stats operation."""
        # Add some graphs
        for i in range(2):
            graph = nx.Graph()
            self.cache.put(f"graph_{i}", graph)

        # Access one graph multiple times
        self.cache.get("graph_0")
        self.cache.get("graph_0")

        # Some misses
        self.cache.get("non_existent")

        stats = self.cache.get_stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 3
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3  # 2 hits out of 3 requests
        assert "memory_mb" in stats
        assert stats["evictions"] >= 0

    @patch("networkx_mcp.graph_cache.psutil.Process")
    def test_memory_limit_eviction(self, mock_process_class):
        """Test eviction based on memory limit."""
        # Mock memory usage to be over limit
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 600 * 1024 * 1024  # 600 MB
        mock_process_class.return_value = mock_process

        cache = GraphCache(max_size=100, ttl_seconds=3600, max_memory_mb=500)

        # Add graphs - should trigger memory eviction
        for i in range(10):
            graph = nx.complete_graph(10)
            cache.put(f"large_{i}", graph)

        # Should have evicted some graphs due to memory limit
        assert cache.evictions > 0

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = GraphCache(max_size=10, ttl_seconds=0.1, cleanup_interval=10)

        # Add graphs
        for i in range(3):
            graph = nx.Graph()
            cache.put(f"graph_{i}", graph)

        # Wait for expiration
        time.sleep(0.2)

        # Force cleanup
        cache._cleanup()

        assert len(cache._cache) == 0
        assert cache.evictions >= 3

    def test_thread_safety(self):
        """Test thread-safe operations."""
        results = []
        errors = []

        def add_graphs(start, count):
            try:
                for i in range(start, start + count):
                    graph = nx.Graph()
                    graph.add_node(i)
                    self.cache.put(f"graph_{i}", graph)
                    results.append(f"added_{i}")
            except Exception as e:
                errors.append(e)

        def get_graphs(start, count):
            try:
                for i in range(start, start + count):
                    graph = self.cache.get(f"graph_{i}")
                    if graph:
                        results.append(f"got_{i}")
            except Exception as e:
                errors.append(e)

        # Create threads
        threads = []
        threads.append(threading.Thread(target=add_graphs, args=(0, 5)))
        threads.append(threading.Thread(target=add_graphs, args=(5, 5)))
        threads.append(threading.Thread(target=get_graphs, args=(0, 10)))

        # Start threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Check no errors occurred
        assert len(errors) == 0
        assert len(results) > 0

    def test_is_expired(self):
        """Test _is_expired method."""
        graph = nx.Graph()

        # Create expired cached graph
        old_cached = CachedGraph(
            graph=graph,
            created_at=time.time() - 100,  # Created 100 seconds ago
            last_accessed=time.time(),
            access_count=0,
        )

        # Create fresh cached graph
        new_cached = CachedGraph(
            graph=graph,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=0,
        )

        cache = GraphCache(ttl_seconds=10)

        assert cache._is_expired(old_cached) is True
        assert cache._is_expired(new_cached) is False

    def test_evict_lru(self):
        """Test _evict_lru method."""
        # Add multiple graphs
        for i in range(3):
            graph = nx.Graph()
            self.cache.put(f"graph_{i}", graph)

        initial_evictions = self.cache.evictions
        initial_size = len(self.cache._cache)

        # Evict LRU
        self.cache._evict_lru()

        assert len(self.cache._cache) == initial_size - 1
        assert self.cache.evictions == initial_evictions + 1

    @patch("networkx_mcp.graph_cache.psutil.Process")
    def test_get_memory_usage_mb(self, mock_process_class):
        """Test _get_memory_usage_mb method."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100 MB
        mock_process_class.return_value = mock_process

        memory_mb = self.cache._get_memory_usage_mb()
        assert memory_mb == 100

    @patch("networkx_mcp.graph_cache.psutil.Process")
    def test_get_memory_usage_mb_error(self, mock_process_class):
        """Test _get_memory_usage_mb with error."""
        mock_process_class.side_effect = Exception("Memory error")

        memory_mb = self.cache._get_memory_usage_mb()
        assert memory_mb == 0  # Returns 0 on error

    def test_update_existing(self):
        """Test updating existing cached graph."""
        graph1 = nx.Graph()
        graph1.add_edge("A", "B")

        self.cache.put("test", graph1)

        # Update with new graph
        graph2 = nx.Graph()
        graph2.add_edge("C", "D")

        self.cache.put("test", graph2)

        retrieved = self.cache.get("test")
        assert retrieved == graph2
        assert retrieved != graph1

    def test_access_updates_lru(self):
        """Test that accessing a graph updates its LRU position."""
        # Add graphs
        for i in range(3):
            graph = nx.Graph()
            self.cache.put(f"graph_{i}", graph)

        # Access first graph to move it to end
        self.cache.get("graph_0")

        # Check it's at the end (most recently used)
        keys = list(self.cache._cache.keys())
        assert keys[-1] == "graph_0"


class TestGraphDict:
    """Test GraphDict compatibility layer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache = GraphCache(max_size=10, ttl_seconds=100)
        self.graph_dict = GraphDict(self.cache)

    def test_setitem_getitem(self):
        """Test dict-like set and get."""
        graph = nx.Graph()
        graph.add_edge("A", "B")

        # Set
        self.graph_dict["test"] = graph

        # Get
        retrieved = self.graph_dict["test"]
        assert retrieved == graph

        # Get non-existent raises KeyError
        with pytest.raises(KeyError):
            _ = self.graph_dict["non_existent"]

    def test_delitem(self):
        """Test dict-like delete."""
        graph = nx.Graph()
        self.graph_dict["test"] = graph

        # Delete existing
        del self.graph_dict["test"]

        # Should raise KeyError now
        with pytest.raises(KeyError):
            _ = self.graph_dict["test"]

        # Delete non-existent raises KeyError
        with pytest.raises(KeyError):
            del self.graph_dict["non_existent"]

    def test_contains(self):
        """Test dict-like contains."""
        graph = nx.Graph()
        self.graph_dict["test"] = graph

        assert "test" in self.graph_dict
        assert "non_existent" not in self.graph_dict

        # Non-string key returns False
        assert 123 not in self.graph_dict
        assert None not in self.graph_dict

    def test_len(self):
        """Test dict-like len."""
        assert len(self.graph_dict) == 0

        for i in range(3):
            self.graph_dict[f"graph_{i}"] = nx.Graph()

        assert len(self.graph_dict) == 3

    def test_iter(self):
        """Test dict-like iteration."""
        graphs = {}
        for i in range(3):
            graph = nx.Graph()
            key = f"graph_{i}"
            graphs[key] = graph
            self.graph_dict[key] = graph

        # Iterate over keys
        for key in self.graph_dict:
            assert key in graphs

    def test_keys(self):
        """Test dict-like keys method."""
        for i in range(3):
            self.graph_dict[f"graph_{i}"] = nx.Graph()

        keys = self.graph_dict.keys()
        assert len(keys) == 3
        assert "graph_0" in keys
        assert "graph_1" in keys
        assert "graph_2" in keys

    def test_get(self):
        """Test dict-like get method."""
        graph = nx.Graph()
        self.graph_dict["test"] = graph

        # Get existing
        retrieved = self.graph_dict.get("test")
        assert retrieved == graph

        # Get non-existent with default None
        result = self.graph_dict.get("non_existent")
        assert result is None

        # Get non-existent with custom default
        default_graph = nx.Graph()
        result = self.graph_dict.get("non_existent", default_graph)
        assert result == default_graph

    def test_clear(self):
        """Test dict-like clear method."""
        for i in range(3):
            self.graph_dict[f"graph_{i}"] = nx.Graph()

        assert len(self.graph_dict) == 3

        self.graph_dict.clear()
        assert len(self.graph_dict) == 0


class TestGlobalCache:
    """Test global cache instance."""

    def test_get_graph_cache(self):
        """Test get_graph_cache returns GraphCache instance."""
        cache = get_graph_cache()
        assert isinstance(cache, GraphCache)

        # Should return same instance
        cache2 = get_graph_cache()
        assert cache is cache2

    def test_global_graphs_dict(self):
        """Test global graphs dict."""
        from networkx_mcp.graph_cache import graphs

        assert isinstance(graphs, GraphDict)

        # Test basic operations
        graph = nx.Graph()
        graphs["test_global"] = graph
        assert graphs["test_global"] == graph

        # Clean up
        del graphs["test_global"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
