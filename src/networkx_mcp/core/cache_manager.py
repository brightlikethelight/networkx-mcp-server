"""
Intelligent caching system with memory limits and eviction policies.

This module provides a sophisticated caching layer for graph computations
with automatic memory management, TTL support, and intelligent eviction.
"""

import asyncio
import hashlib
import logging
import pickle
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Cache eviction policies"""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live based
    SIZE = "size"  # Size-based (evict largest)
    ADAPTIVE = "adaptive"  # Adaptive policy based on access patterns


@dataclass
class CacheEntry:
    """Individual cache entry with metadata"""

    key: str
    value: Any
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    computation_time: float = 0.0  # Time taken to compute
    priority: int = 0  # Higher priority entries are kept longer
    tags: Set[str] = field(default_factory=set)

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False

        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def touch(self):
        """Update last access time and increment count"""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics"""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_computation_time_saved: float = 0.0
    current_size_bytes: int = 0
    max_size_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class AdaptiveEvictionStrategy:
    """Adaptive eviction strategy based on access patterns"""

    def __init__(self):
        self.access_history = defaultdict(list)
        self.eviction_scores = {}

    def update_access(self, key: str):
        """Update access history for adaptive scoring"""
        self.access_history[key].append(time.time())

        # Keep only recent history (last hour)
        cutoff = time.time() - 3600
        self.access_history[key] = [t for t in self.access_history[key] if t > cutoff]

    def calculate_eviction_score(self, entry: CacheEntry) -> float:
        """
        Calculate eviction score (lower = more likely to evict).

        Considers:
        - Recency of access
        - Frequency of access
        - Entry size
        - Computation cost
        - Priority
        """
        now = datetime.now()

        # Recency score (exponential decay)
        recency = (now - entry.last_accessed).total_seconds()
        recency_score = 1.0 / (1.0 + recency / 3600)  # Decay over 1 hour

        # Frequency score (logarithmic)
        frequency_score = min(1.0, entry.access_count / 10.0)

        # Size penalty (prefer keeping smaller entries)
        size_penalty = 1.0 / (1.0 + entry.size_bytes / (1024 * 1024))  # MB scale

        # Computation cost bonus (keep expensive computations)
        computation_bonus = min(1.0, entry.computation_time / 10.0)

        # Priority bonus
        priority_bonus = entry.priority / 10.0

        # Weighted combination
        score = (
            recency_score * 0.3
            + frequency_score * 0.3
            + size_penalty * 0.1
            + computation_bonus * 0.2
            + priority_bonus * 0.1
        )

        return score

    def select_eviction_candidates(
        self, entries: List[CacheEntry], required_space: int
    ) -> List[str]:
        """Select entries to evict based on adaptive scoring"""
        # Calculate scores
        scored_entries = [
            (entry, self.calculate_eviction_score(entry)) for entry in entries
        ]

        # Sort by score (ascending - lower scores evicted first)
        scored_entries.sort(key=lambda x: x[1])

        # Select entries until we have enough space
        candidates = []
        freed_space = 0

        for entry, score in scored_entries:
            if freed_space >= required_space:
                break
            candidates.append(entry.key)
            freed_space += entry.size_bytes

        return candidates


class CacheManager:
    """
    Intelligent cache manager for graph computations.

    Features:
    - Multiple eviction policies
    - Memory limit enforcement
    - TTL support
    - Tagged entries for group invalidation
    - Computation time tracking
    - Adaptive eviction based on access patterns
    """

    def __init__(
        self,
        max_size_mb: int = 256,
        eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE,
        default_ttl: Optional[int] = 3600,
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl

        # Storage
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.size_bytes = 0

        # Statistics
        self.stats = CacheStats(max_size_bytes=self.max_size_bytes)

        # Eviction strategies
        self.adaptive_strategy = AdaptiveEvictionStrategy()

        # Tag index for fast invalidation
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)

        # Background cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None

        logger.info(
            f"Initialized cache manager with {max_size_mb}MB limit, "
            f"policy: {eviction_policy.value}"
        )

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value"""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            return 1000

    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from function arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    async def get(
        self, key: str, compute_func: Optional[Callable] = None, *args, **kwargs
    ) -> Optional[Any]:
        """
        Get value from cache or compute if missing.

        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            *args, **kwargs: Arguments for compute function

        Returns:
            Cached or computed value
        """
        # Check if entry exists and is valid
        if key in self.entries:
            entry = self.entries[key]

            # Check expiration
            if entry.is_expired():
                await self.invalidate(key)
            else:
                # Cache hit
                entry.touch()
                self.entries.move_to_end(key)  # LRU update
                self.stats.hits += 1
                self.stats.total_computation_time_saved += entry.computation_time

                if self.eviction_policy == EvictionPolicy.ADAPTIVE:
                    self.adaptive_strategy.update_access(key)

                logger.debug(f"Cache hit: {key}")
                return entry.value

        # Cache miss
        self.stats.misses += 1
        logger.debug(f"Cache miss: {key}")

        # Compute if function provided
        if compute_func:
            start_time = time.time()
            value = (
                await compute_func(*args, **kwargs)
                if asyncio.iscoroutinefunction(compute_func)
                else compute_func(*args, **kwargs)
            )
            computation_time = time.time() - start_time

            # Store in cache
            await self.set(key, value, computation_time=computation_time)

            return value

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        priority: int = 0,
        tags: Optional[Set[str]] = None,
        computation_time: float = 0.0,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            priority: Entry priority
            tags: Tags for group invalidation
            computation_time: Time taken to compute value

        Returns:
            True if successfully cached
        """
        # Estimate size
        size_bytes = self._estimate_size(value)

        # Check if we need to evict entries
        if self.size_bytes + size_bytes > self.max_size_bytes:
            required_space = (self.size_bytes + size_bytes) - self.max_size_bytes
            await self._evict(required_space)

            # Check if we have enough space after eviction
            if self.size_bytes + size_bytes > self.max_size_bytes:
                logger.warning(f"Cannot cache entry {key}: insufficient space")
                return False

        # Create entry
        now = datetime.now()
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes,
            created_at=now,
            last_accessed=now,
            ttl_seconds=ttl or self.default_ttl,
            priority=priority,
            tags=tags or set(),
            computation_time=computation_time,
        )

        # Remove old entry if exists
        if key in self.entries:
            await self.invalidate(key)

        # Add new entry
        self.entries[key] = entry
        self.size_bytes += size_bytes

        # Update tag index
        for tag in entry.tags:
            self.tag_index[tag].add(key)

        # Update stats
        self.stats.current_size_bytes = self.size_bytes
        self.stats.entry_count = len(self.entries)

        logger.debug(f"Cached {key} ({size_bytes} bytes)")
        return True

    async def invalidate(self, key: str) -> bool:
        """Remove entry from cache"""
        if key not in self.entries:
            return False

        entry = self.entries[key]

        # Remove from storage
        del self.entries[key]
        self.size_bytes -= entry.size_bytes

        # Update tag index
        for tag in entry.tags:
            self.tag_index[tag].discard(key)
            if not self.tag_index[tag]:
                del self.tag_index[tag]

        # Update stats
        self.stats.current_size_bytes = self.size_bytes
        self.stats.entry_count = len(self.entries)

        logger.debug(f"Invalidated cache entry: {key}")
        return True

    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a specific tag"""
        if tag not in self.tag_index:
            return 0

        keys = list(self.tag_index[tag])
        count = 0

        for key in keys:
            if await self.invalidate(key):
                count += 1

        logger.info(f"Invalidated {count} entries with tag: {tag}")
        return count

    async def _evict(self, required_space: int):
        """Evict entries based on policy"""
        logger.debug(f"Evicting entries to free {required_space} bytes")

        candidates = []

        if self.eviction_policy == EvictionPolicy.LRU:
            # Evict least recently used
            for key, entry in self.entries.items():
                candidates.append((key, entry.last_accessed))
            candidates.sort(key=lambda x: x[1])

        elif self.eviction_policy == EvictionPolicy.LFU:
            # Evict least frequently used
            for key, entry in self.entries.items():
                candidates.append((key, entry.access_count))
            candidates.sort(key=lambda x: x[1])

        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Evict oldest
            for key, entry in self.entries.items():
                candidates.append((key, entry.created_at))
            candidates.sort(key=lambda x: x[1])

        elif self.eviction_policy == EvictionPolicy.SIZE:
            # Evict largest
            for key, entry in self.entries.items():
                candidates.append((key, entry.size_bytes))
            candidates.sort(key=lambda x: x[1], reverse=True)

        elif self.eviction_policy == EvictionPolicy.TTL:
            # Evict expired first, then oldest
            expired = []
            valid = []

            for key, entry in self.entries.items():
                if entry.is_expired():
                    expired.append(key)
                else:
                    valid.append((key, entry.created_at))

            candidates = expired + [k for k, _ in sorted(valid, key=lambda x: x[1])]

        elif self.eviction_policy == EvictionPolicy.ADAPTIVE:
            # Use adaptive strategy
            entries = list(self.entries.values())
            candidate_keys = self.adaptive_strategy.select_eviction_candidates(
                entries, required_space
            )
            candidates = candidate_keys

        # Evict until we have enough space
        freed_space = 0
        eviction_count = 0

        for item in candidates:
            if freed_space >= required_space:
                break

            # Extract key based on policy
            if isinstance(item, str):
                key = item
            elif isinstance(item, tuple):
                key = item[0]
            else:
                continue

            if key in self.entries:
                freed_space += self.entries[key].size_bytes
                await self.invalidate(key)
                eviction_count += 1

        self.stats.evictions += eviction_count
        logger.info(f"Evicted {eviction_count} entries, freed {freed_space} bytes")

    async def cleanup(self):
        """Remove expired entries"""
        expired_keys = []

        for key, entry in self.entries.items():
            if entry.is_expired():
                expired_keys.append(key)

        for key in expired_keys:
            await self.invalidate(key)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired entries")

    async def start_background_cleanup(self, interval: int = 60):
        """Start background cleanup task"""

        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval)
                await self.cleanup()

        self.cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info(f"Started background cleanup with {interval}s interval")

    def stop_background_cleanup(self):
        """Stop background cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            self.cleanup_task = None

    def clear(self):
        """Clear all cache entries"""
        self.entries.clear()
        self.size_bytes = 0
        self.tag_index.clear()
        self.stats.entry_count = 0
        self.stats.current_size_bytes = 0
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "hit_rate": self.stats.hit_rate,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
            "entry_count": self.stats.entry_count,
            "size_mb": self.stats.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.stats.max_size_bytes / (1024 * 1024),
            "utilization": self.stats.current_size_bytes / self.stats.max_size_bytes,
            "computation_time_saved": self.stats.total_computation_time_saved,
            "eviction_policy": self.eviction_policy.value,
            "tag_count": len(self.tag_index),
        }

    def cache_decorator(
        self,
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None,
        tags: Optional[Set[str]] = None,
    ):
        """Decorator for caching function results"""

        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, *args, **kwargs)

                # Get or compute
                return await self.get(cache_key, func, *args, **kwargs)

            return wrapper

        return decorator


# Example usage
async def test_cache_manager():
    """Test cache manager functionality"""
    cache = CacheManager(max_size_mb=10, eviction_policy=EvictionPolicy.ADAPTIVE)

    # Start background cleanup
    await cache.start_background_cleanup()

    # Test caching
    import networkx as nx

    async def expensive_computation(graph):
        await asyncio.sleep(0.1)  # Simulate computation
        return nx.pagerank(graph)

    # Create test graph
    G = nx.karate_club_graph()

    # First call - cache miss
    result1 = await cache.get("pagerank_karate", expensive_computation, G)

    # Second call - cache hit
    result2 = await cache.get("pagerank_karate", expensive_computation, G)

    # Verify both results match (cache consistency)
    assert result1 == result2, "Cache should return same result"

    # Test with tags
    await cache.set("test_entry", {"data": "value"}, tags={"graph_analysis", "test"})

    # Invalidate by tag
    await cache.invalidate_by_tag("test")

    # Get statistics
    stats = cache.get_stats()
    logger.info(f"Cache stats: {stats}")

    # Stop background cleanup
    cache.stop_background_cleanup()


if __name__ == "__main__":
    asyncio.run(test_cache_manager())
