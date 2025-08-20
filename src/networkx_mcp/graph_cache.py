"""Graph cache with automatic cleanup to prevent memory leaks.

This module provides a thread-safe graph cache with:
- LRU (Least Recently Used) eviction
- TTL (Time To Live) expiration
- Maximum size limits
- Memory usage monitoring
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx
import psutil

logger = logging.getLogger(__name__)


@dataclass
class CachedGraph:
    """Container for a cached graph with metadata."""

    graph: nx.Graph
    created_at: float
    last_accessed: float
    access_count: int = 0

    def touch(self) -> None:
        """Update last access time and increment counter."""
        self.last_accessed = time.time()
        self.access_count += 1


class GraphCache:
    """Thread-safe graph cache with automatic cleanup.

    Features:
    - LRU eviction when max_size is reached
    - TTL expiration for old graphs
    - Memory-based eviction when threshold exceeded
    - Thread-safe operations
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: float = 3600,  # 1 hour default
        max_memory_mb: int = 500,
        cleanup_interval: float = 300,  # 5 minutes
    ):
        """Initialize the graph cache.

        Args:
            max_size: Maximum number of graphs to cache
            ttl_seconds: Time to live for cached graphs in seconds
            max_memory_mb: Maximum memory usage in MB before eviction
            cleanup_interval: Interval between cleanup runs in seconds
        """
        self._cache: OrderedDict[str, CachedGraph] = OrderedDict()
        self._lock = threading.RLock()
        self._shutdown = False
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_mb = max_memory_mb
        self.cleanup_interval = cleanup_interval

        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        logger.info(
            f"GraphCache initialized: max_size={max_size}, "
            f"ttl={ttl_seconds}s, max_memory={max_memory_mb}MB"
        )

    def get(self, key: str) -> Optional[nx.Graph]:
        """Get a graph from cache.

        Args:
            key: Graph identifier

        Returns:
            Graph if found and valid, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            cached = self._cache[key]

            # Check TTL
            if self._is_expired(cached):
                del self._cache[key]
                self.evictions += 1
                self.misses += 1
                logger.debug(f"Graph {key} expired (TTL)")
                return None

            # Update LRU position
            self._cache.move_to_end(key)
            cached.touch()
            self.hits += 1

            return cached.graph

    def put(self, key: str, graph: nx.Graph) -> None:
        """Add or update a graph in cache.

        Args:
            key: Graph identifier
            graph: NetworkX graph to cache
        """
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]

            # Check size limit
            if len(self._cache) >= self.max_size:
                self._evict_lru()

            # Check memory limit
            if self._get_memory_usage_mb() > self.max_memory_mb:
                self._evict_until_memory_ok()

            # Add to cache
            self._cache[key] = CachedGraph(
                graph=graph, created_at=time.time(), last_accessed=time.time()
            )

            logger.debug(f"Cached graph {key} ({graph.number_of_nodes()} nodes)")

    def delete(self, key: str) -> bool:
        """Remove a graph from cache.

        Args:
            key: Graph identifier

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Deleted graph {key} from cache")
                return True
            return False

    def clear(self) -> None:
        """Clear all cached graphs."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self.evictions += count
            logger.info(f"Cleared {count} graphs from cache")

    def list_graphs(self) -> List[str]:
        """Get list of cached graph keys."""
        with self._lock:
            return list(self._cache.keys())

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "memory_mb": self._get_memory_usage_mb(),
            }

    def _is_expired(self, cached: CachedGraph) -> bool:
        """Check if a cached graph has expired."""
        age = time.time() - cached.created_at
        return age > self.ttl_seconds

    def _evict_lru(self) -> None:
        """Evict least recently used graph."""
        if self._cache:
            key, _ = self._cache.popitem(last=False)
            self.evictions += 1
            logger.debug(f"Evicted LRU graph {key}")

    def _evict_until_memory_ok(self) -> None:
        """Evict graphs until memory usage is acceptable."""
        while self._cache and self._get_memory_usage_mb() > self.max_memory_mb:
            self._evict_lru()

    def _get_memory_usage_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0

    def _cleanup_loop(self) -> None:
        """Background thread for periodic cleanup."""
        while not self._shutdown:
            try:
                time.sleep(self.cleanup_interval)
                if not self._shutdown:
                    self._cleanup()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                if self._shutdown:
                    break

    def _cleanup(self) -> None:
        """Clean up expired graphs."""
        with self._lock:
            expired_keys = []

            for key, cached in self._cache.items():
                if self._is_expired(cached):
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                self.evictions += 1

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired graphs")

            # Check memory limit
            if self._get_memory_usage_mb() > self.max_memory_mb:
                self._evict_until_memory_ok()

    def shutdown(self) -> None:
        """Shutdown the cache and stop background thread."""
        self._shutdown = True
        if hasattr(self, "_cleanup_thread") and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)


# Global cache instance (replaces the simple dict)
_graph_cache = GraphCache(max_size=100, ttl_seconds=3600, max_memory_mb=500)


def get_graph_cache() -> GraphCache:
    """Get the global graph cache instance."""
    return _graph_cache


# Compatibility layer for existing code
class GraphDict(dict):
    """Dict-like interface for graph cache (backward compatibility)."""

    def __init__(self, cache: GraphCache):
        super().__init__()
        self._cache = cache

    def __getitem__(self, key: str) -> nx.Graph:
        graph = self._cache.get(key)
        if graph is None:
            raise KeyError(key)
        return graph

    def __setitem__(self, key: str, graph: nx.Graph) -> None:
        self._cache.put(key, graph)

    def __delitem__(self, key: str) -> None:
        if not self._cache.delete(key):
            raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self._cache.get(key) is not None

    def __len__(self) -> int:
        return len(self._cache.list_graphs())

    def __iter__(self):
        return iter(self._cache.list_graphs())

    def keys(self):
        return self._cache.list_graphs()

    def get(self, key: str, default=None):
        graph = self._cache.get(key)
        return graph if graph is not None else default

    def clear(self) -> None:
        self._cache.clear()

    def shutdown(self) -> None:
        """Shutdown the underlying cache."""
        self._cache.shutdown()


# Create backward-compatible graphs dict
graphs = GraphDict(_graph_cache)
