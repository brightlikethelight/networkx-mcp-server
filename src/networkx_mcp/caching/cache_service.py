"""Caching service for NetworkX MCP Server.

This module provides a comprehensive caching abstraction with support for
multiple backends (memory, Redis) and advanced features like TTL, patterns,
and cache statistics.
"""

import asyncio
import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any

from ..core.base import Component, ComponentStatus
from ..core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime
    expires_at: datetime | None = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


@dataclass
class CacheStats:
    """Cache statistics."""

    total_keys: int = 0
    total_size_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired_keys: int = 0
    average_key_size: float = 0.0
    hit_rate: float = 0.0

    def update_hit_rate(self) -> None:
        """Update hit rate calculation."""
        total_requests = self.hits + self.misses
        self.hit_rate = self.hits / total_requests if total_requests > 0 else 0.0


class CacheBackend(ABC):
    """Abstract cache backend."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value by key."""

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value with optional TTL."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all keys."""

    @abstractmethod
    async def keys(self, pattern: str | None = None) -> list[str]:
        """Get all keys matching pattern."""

    @abstractmethod
    async def size(self) -> int:
        """Get cache size."""


class MemoryCache(CacheBackend, Component):
    """In-memory cache backend."""

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        Component.__init__(self, "MemoryCache")
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize memory cache."""
        await self._set_status(ComponentStatus.READY)

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())

        logger.info(f"Memory cache initialized (max_size={self.max_size})")

    async def shutdown(self) -> None:
        """Shutdown memory cache."""
        await self._set_status(ComponentStatus.SHUTTING_DOWN)

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._cache.clear()
        await self._set_status(ComponentStatus.SHUTDOWN)
        logger.info("Memory cache shutdown")

    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            if entry.is_expired():
                del self._cache[key]
                return None

            entry.update_access()
            return entry.value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value with optional TTL."""
        async with self._lock:
            # Calculate expiration
            expires_at = None
            if ttl is not None:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            elif self.default_ttl > 0:
                expires_at = datetime.utcnow() + timedelta(seconds=self.default_ttl)

            # Calculate size (rough estimate)
            size_bytes = len(json.dumps(value, default=str).encode("utf-8"))

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                size_bytes=size_bytes,
            )

            # Evict if necessary
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()

            self._cache[key] = entry

    async def delete(self, key: str) -> bool:
        """Delete key."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False

            if entry.is_expired():
                del self._cache[key]
                return False

            return True

    async def clear(self) -> None:
        """Clear all keys."""
        async with self._lock:
            self._cache.clear()

    async def keys(self, pattern: str | None = None) -> list[str]:
        """Get all keys matching pattern."""
        async with self._lock:
            if pattern is None:
                return list(self._cache.keys())

            regex = re.compile(pattern.replace("*", ".*"))
            return [key for key in self._cache.keys() if regex.match(key)]

    async def size(self) -> int:
        """Get cache size."""
        async with self._lock:
            return len(self._cache)

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        del self._cache[lru_key]

    async def _cleanup_expired(self) -> None:
        """Periodically clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                async with self._lock:
                    expired_keys = [
                        key for key, entry in self._cache.items() if entry.is_expired()
                    ]

                    for key in expired_keys:
                        del self._cache[key]

                    if expired_keys:
                        logger.debug(
                            f"Cleaned up {len(expired_keys)} expired cache entries"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        async with self._lock:
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            return {
                "healthy": self.status == ComponentStatus.READY,
                "entries": len(self._cache),
                "max_size": self.max_size,
                "total_size_bytes": total_size,
                "cleanup_task_running": self._cleanup_task is not None
                and not self._cleanup_task.done(),
            }


class RedisCache(CacheBackend, Component):
    """Redis cache backend."""

    def __init__(self, redis_url: str | None = None, prefix: str = "cache"):
        Component.__init__(self, "RedisCache")
        self.redis_url = redis_url or get_config().redis.url
        self.prefix = prefix
        self.redis_client = None

    async def initialize(self) -> None:
        """Initialize Redis cache."""
        await self._set_status(ComponentStatus.INITIALIZING)

        if not self.redis_url:
            raise ValueError("Redis URL not configured")

        try:
            import redis.asyncio as redis

            self.redis_client = redis.from_url(self.redis_url)

            # Test connection
            await self.redis_client.ping()

            await self._set_status(ComponentStatus.READY)
            logger.info("Redis cache initialized")
        except ImportError:
            raise RuntimeError("redis package not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown Redis cache."""
        await self._set_status(ComponentStatus.SHUTTING_DOWN)
        if self.redis_client:
            await self.redis_client.close()
        await self._set_status(ComponentStatus.SHUTDOWN)
        logger.info("Redis cache shutdown")

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        if not self.redis_client:
            return None

        redis_key = self._make_key(key)
        data = await self.redis_client.get(redis_key)

        if data is None:
            return None

        try:
            return json.loads(data)
        except json.JSONDecodeError:
            logger.error(f"Failed to deserialize cached value for key: {key}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value with optional TTL."""
        if not self.redis_client:
            return

        redis_key = self._make_key(key)
        data = json.dumps(value, default=str)

        if ttl:
            await self.redis_client.setex(redis_key, ttl, data)
        else:
            await self.redis_client.set(redis_key, data)

    async def delete(self, key: str) -> bool:
        """Delete key."""
        if not self.redis_client:
            return False

        redis_key = self._make_key(key)
        result = await self.redis_client.delete(redis_key)
        return result > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self.redis_client:
            return False

        redis_key = self._make_key(key)
        return await self.redis_client.exists(redis_key) > 0

    async def clear(self) -> None:
        """Clear all keys with prefix."""
        if not self.redis_client:
            return

        pattern = f"{self.prefix}:*"
        keys = await self.redis_client.keys(pattern)
        if keys:
            await self.redis_client.delete(*keys)

    async def keys(self, pattern: str | None = None) -> list[str]:
        """Get all keys matching pattern."""
        if not self.redis_client:
            return []

        if pattern:
            redis_pattern = f"{self.prefix}:{pattern}"
        else:
            redis_pattern = f"{self.prefix}:*"

        redis_keys = await self.redis_client.keys(redis_pattern)

        # Remove prefix from keys
        prefix_len = len(self.prefix) + 1
        return [key[prefix_len:] for key in redis_keys]

    async def size(self) -> int:
        """Get cache size."""
        if not self.redis_client:
            return 0

        pattern = f"{self.prefix}:*"
        keys = await self.redis_client.keys(pattern)
        return len(keys)

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        try:
            if self.redis_client:
                await self.redis_client.ping()
                cache_size = await self.size()
                return {
                    "healthy": True,
                    "redis_url": self.redis_url,
                    "cache_size": cache_size,
                }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

        return {"healthy": False, "error": "Redis client not initialized"}


class CacheService(Component):
    """Main cache service with pluggable backends."""

    def __init__(self, backend: CacheBackend | None = None):
        super().__init__("CacheService")
        self.backend = backend or MemoryCache()
        self.stats = CacheStats()
        self._stats_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize cache service."""
        await self._set_status(ComponentStatus.INITIALIZING)

        if isinstance(self.backend, Component):
            await self.backend.initialize()

        await self._set_status(ComponentStatus.READY)
        logger.info(f"Cache service initialized with {type(self.backend).__name__}")

    async def shutdown(self) -> None:
        """Shutdown cache service."""
        await self._set_status(ComponentStatus.SHUTTING_DOWN)

        if isinstance(self.backend, Component):
            await self.backend.shutdown()

        await self._set_status(ComponentStatus.SHUTDOWN)
        logger.info("Cache service shutdown")

    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        value = await self.backend.get(key)

        async with self._stats_lock:
            if value is not None:
                self.stats.hits += 1
            else:
                self.stats.misses += 1
            self.stats.update_hit_rate()

        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value with optional TTL."""
        await self.backend.set(key, value, ttl)

        async with self._stats_lock:
            self.stats.total_keys = await self.backend.size()

    async def delete(self, key: str) -> bool:
        """Delete key."""
        result = await self.backend.delete(key)

        if result:
            async with self._stats_lock:
                self.stats.total_keys = await self.backend.size()

        return result

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        keys = await self.backend.keys(pattern)
        deleted_count = 0

        for key in keys:
            if await self.backend.delete(key):
                deleted_count += 1

        async with self._stats_lock:
            self.stats.total_keys = await self.backend.size()

        return deleted_count

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self.backend.exists(key)

    async def clear(self) -> None:
        """Clear all keys."""
        await self.backend.clear()

        async with self._stats_lock:
            self.stats.total_keys = 0

    async def keys(self, pattern: str | None = None) -> list[str]:
        """Get all keys matching pattern."""
        return await self.backend.keys(pattern)

    async def size(self) -> int:
        """Get cache size."""
        return await self.backend.size()

    async def get_multi(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values."""
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result

    async def set_multi(self, items: dict[str, Any], ttl: int | None = None) -> None:
        """Set multiple values."""
        for key, value in items.items():
            await self.set(key, value, ttl)

    def make_key(self, *parts: str | int) -> str:
        """Create cache key from parts."""
        key_parts = [str(part) for part in parts]
        return ":".join(key_parts)

    def make_hash_key(self, data: Any) -> str:
        """Create cache key from data hash."""
        json_data = json.dumps(data, sort_keys=True, default=str)
        hash_object = hashlib.md5(json_data.encode())
        return hash_object.hexdigest()

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        async with self._stats_lock:
            self.stats.total_keys = await self.backend.size()
            return self.stats

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        backend_health = {}
        if hasattr(self.backend, "health_check"):
            backend_health = await self.backend.health_check()

        return {
            "healthy": self.status == ComponentStatus.READY,
            "backend": type(self.backend).__name__,
            "backend_health": backend_health,
            "stats": {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate": self.stats.hit_rate,
                "total_keys": self.stats.total_keys,
            },
        }


# Cache decorators
def cached(ttl: int | None = None, key_func: Callable | None = None):
    """Decorator to cache function results."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache service from args (assumes it's injected)
            cache_service = None
            for arg in args:
                if isinstance(arg, CacheService):
                    cache_service = arg
                    break
                elif hasattr(arg, "cache_service") and isinstance(
                    arg.cache_service, CacheService
                ):
                    cache_service = arg.cache_service
                    break

            if cache_service is None:
                # No cache available, execute function directly
                return await func(*args, **kwargs)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args[1:])  # Skip self/cls
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                cache_key = cache_service.make_key(*key_parts)

            # Try to get from cache
            cached_result = await cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_service.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


def cache_invalidate(pattern: str):
    """Decorator to invalidate cache entries matching pattern."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            # Find cache service
            cache_service = None
            for arg in args:
                if isinstance(arg, CacheService):
                    cache_service = arg
                    break
                elif hasattr(arg, "cache_service") and isinstance(
                    arg.cache_service, CacheService
                ):
                    cache_service = arg.cache_service
                    break

            if cache_service:
                await cache_service.delete_pattern(pattern)

            return result

        return wrapper

    return decorator
