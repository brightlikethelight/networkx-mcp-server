"""Production Redis backend with compression and transactions."""

import asyncio
import json
import pickle
import zlib
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import networkx as nx

try:
    import redis.asyncio as redis
except ImportError:
    import aioredis as redis  # Fallback for older versions

from ..security.validator import SecurityValidator
from .base import (GraphNotFoundError, StorageBackend, StorageError,
                   StorageQuotaExceededError, Transaction, TransactionError)


class RedisTransaction(Transaction):
    """Redis transaction implementation."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._committed = False
        self._rolled_back = False

    async def commit(self) -> None:
        """Execute all commands in the transaction."""
        if self._committed:
            msg = "Transaction already committed"
            raise TransactionError(msg)
        if self._rolled_back:
            msg = "Transaction already rolled back"
            raise TransactionError(msg)

        try:
            await self.pipeline.execute()
            self._committed = True
        except Exception as e:
            msg = f"Failed to commit transaction: {e}"
            raise TransactionError(msg) from e

    async def rollback(self) -> None:
        """Discard all commands in the transaction."""
        if self._committed:
            msg = "Cannot rollback committed transaction"
            raise TransactionError(msg)
        if self._rolled_back:
            msg = "Transaction already rolled back"
            raise TransactionError(msg)

        await self.pipeline.reset()
        self._rolled_back = True


class RedisBackend(StorageBackend):
    """Production Redis backend with compression and metadata."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_graph_size_mb: int = 100,
        compression_level: int = 6,
        key_prefix: str = "networkx_mcp",
    ):
        self.redis_url = redis_url
        self.max_size_bytes = max_graph_size_mb * 1024 * 1024
        self.compression_level = compression_level
        self.key_prefix = key_prefix
        self.pool = None
        self._client = None

    async def initialize(self) -> None:
        """Create connection pool."""
        self.pool = redis.ConnectionPool.from_url(
            self.redis_url,
            max_connections=50,
            decode_responses=False,  # We handle encoding ourselves
            socket_keepalive=True,
            socket_keepalive_options={
                1: 1,  # TCP_KEEPIDLE
                2: 30,  # TCP_KEEPINTVL
                3: 5,  # TCP_KEEPCNT
            },
        )
        self._client = redis.Redis(connection_pool=self.pool)

        # Test connection
        await self._client.ping()

    async def close(self) -> None:
        """Close Redis connections."""
        if self._client:
            await self._client.close()
        if self.pool:
            await self.pool.disconnect()

    @asynccontextmanager
    async def transaction(self):
        """Create a Redis transaction."""
        async with self._client.pipeline(transaction=True) as pipe:
            try:
                tx = RedisTransaction(pipe)
                yield tx
                if not tx._committed and not tx._rolled_back:
                    await tx.commit()
            except Exception:
                if not tx._rolled_back:
                    await tx.rollback()
                raise

    def _make_key(self, *parts: str) -> str:
        """Create a namespaced Redis key."""
        safe_parts = [self.key_prefix]
        for part in parts:
            # Validate parts to prevent injection
            if ":" in part or "\n" in part or "\r" in part:
                msg = f"Invalid key part: {part}"
                raise ValueError(msg)
            safe_parts.append(part)
        return ":".join(safe_parts)

    async def save_graph(
        self,
        user_id: str,
        graph_id: str,
        graph: nx.Graph,
        metadata: dict[str, Any] | None = None,
        tx: Transaction | None = None,
    ) -> bool:
        """Save graph with compression and metadata."""
        # Validate inputs
        user_id = SecurityValidator.validate_user_id(user_id)
        graph_id = SecurityValidator.validate_graph_id(graph_id)

        # Serialize graph
        try:
            graph_data = pickle.dumps(graph, protocol=5)
        except Exception as e:
            msg = f"Failed to serialize graph: {e}"
            raise StorageError(msg) from e

        # Compress
        compressed = zlib.compress(graph_data, level=self.compression_level)

        # Check size limit
        if len(compressed) > self.max_size_bytes:
            msg = (
                f"Graph exceeds size limit: {len(compressed) / 1024 / 1024:.1f}MB "
                f"(max {self.max_size_bytes / 1024 / 1024:.1f}MB)"
            )
            raise StorageQuotaExceededError(msg)

        # Prepare metadata
        now = datetime.now(UTC).isoformat()
        graph_metadata = {
            "user_id": user_id,
            "graph_id": graph_id,
            "created_at": now,
            "updated_at": now,
            "size_bytes": len(compressed),
            "original_size_bytes": len(graph_data),
            "compression_ratio": len(graph_data) / len(compressed) if compressed else 1,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "graph_type": type(graph).__name__,
            "is_directed": graph.is_directed(),
            "is_multigraph": graph.is_multigraph(),
        }

        # Add custom metadata
        if metadata:
            graph_metadata["custom"] = SecurityValidator.sanitize_attributes(metadata)

        # Keys
        data_key = self._make_key("graph_data", user_id, graph_id)
        meta_key = self._make_key("graph_meta", user_id, graph_id)
        user_graphs_key = self._make_key("user_graphs", user_id)
        user_stats_key = self._make_key("user_stats", user_id)

        # Use transaction
        client = tx.pipeline if tx else self._client

        # Save atomically
        if not tx:
            async with self._client.pipeline(transaction=True) as pipe:
                # Check if exists (for update vs create)
                exists = await self._client.exists(data_key)

                # Save data and metadata
                await pipe.set(data_key, compressed)
                await pipe.set(meta_key, json.dumps(graph_metadata))

                # Add to user's graph list
                if not exists:
                    await pipe.sadd(user_graphs_key, graph_id)
                    await pipe.hincrby(user_stats_key, "graph_count", 1)

                # Update storage stats
                await pipe.hincrby(user_stats_key, "total_bytes", len(compressed))

                await pipe.execute()
        else:
            # Within existing transaction
            await client.set(data_key, compressed)
            await client.set(meta_key, json.dumps(graph_metadata))
            await client.sadd(user_graphs_key, graph_id)
            await client.hincrby(user_stats_key, "graph_count", 1)
            await client.hincrby(user_stats_key, "total_bytes", len(compressed))

        return True

    async def load_graph(
        self, user_id: str, graph_id: str, tx: Transaction | None = None
    ) -> nx.Graph | None:
        """Load graph from storage."""
        # Validate inputs
        user_id = SecurityValidator.validate_user_id(user_id)
        graph_id = SecurityValidator.validate_graph_id(graph_id)

        # Get data
        data_key = self._make_key("graph_data", user_id, graph_id)

        client = tx.pipeline if tx else self._client
        compressed = await client.get(data_key)

        if compressed is None:
            return None

        # Decompress
        try:
            graph_data = zlib.decompress(compressed)
        except Exception as e:
            msg = f"Failed to decompress graph: {e}"
            raise StorageError(msg) from e

        # Deserialize
        try:
            graph = pickle.loads(graph_data)  # nosec B301 - controlled storage context
        except Exception as e:
            msg = f"Failed to deserialize graph: {e}"
            raise StorageError(msg) from e

        # Update access time
        if not tx:
            meta_key = self._make_key("graph_meta", user_id, graph_id)
            await self._client.hset(
                meta_key, "last_accessed_at", datetime.now(UTC).isoformat()
            )

        return graph

    async def delete_graph(
        self, user_id: str, graph_id: str, tx: Transaction | None = None
    ) -> bool:
        """Delete graph from storage."""
        # Validate inputs
        user_id = SecurityValidator.validate_user_id(user_id)
        graph_id = SecurityValidator.validate_graph_id(graph_id)

        # Keys
        data_key = self._make_key("graph_data", user_id, graph_id)
        meta_key = self._make_key("graph_meta", user_id, graph_id)
        user_graphs_key = self._make_key("user_graphs", user_id)
        user_stats_key = self._make_key("user_stats", user_id)

        client = tx.pipeline if tx else self._client

        if not tx:
            async with self._client.pipeline(transaction=True) as pipe:
                # Get size for stats update
                metadata = await self._client.get(meta_key)
                if metadata:
                    meta_dict = json.loads(metadata)
                    size_bytes = meta_dict.get("size_bytes", 0)
                else:
                    size_bytes = 0

                # Delete atomically
                deleted = await pipe.delete(data_key, meta_key)
                await pipe.srem(user_graphs_key, graph_id)

                if deleted > 0:
                    await pipe.hincrby(user_stats_key, "graph_count", -1)
                    await pipe.hincrby(user_stats_key, "total_bytes", -size_bytes)

                result = await pipe.execute()
                return result[0] > 0  # True if anything was deleted
        else:
            # Within transaction
            await client.delete(data_key, meta_key)
            await client.srem(user_graphs_key, graph_id)
            await client.hincrby(user_stats_key, "graph_count", -1)
            return True

    async def list_graphs(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        tx: Transaction | None = None,
    ) -> list[dict[str, Any]]:
        """List user's graphs with metadata."""
        # Validate inputs
        user_id = SecurityValidator.validate_user_id(user_id)
        limit = min(max(1, limit), 1000)  # Cap at 1000
        offset = max(0, offset)

        # Get graph IDs
        user_graphs_key = self._make_key("user_graphs", user_id)
        client = tx.pipeline if tx else self._client

        graph_ids = await client.smembers(user_graphs_key)

        if not graph_ids:
            return []

        # Sort for consistent pagination
        sorted_ids = sorted(graph_ids)
        paginated_ids = sorted_ids[offset : offset + limit]

        # Get metadata for each graph
        graphs = []
        for graph_id_bytes in paginated_ids:
            graph_id = (
                graph_id_bytes.decode("utf-8")
                if isinstance(graph_id_bytes, bytes)
                else graph_id_bytes
            )
            meta_key = self._make_key("graph_meta", user_id, graph_id)

            metadata = await client.get(meta_key)
            if metadata:
                meta_dict = json.loads(metadata)
                graphs.append(meta_dict)

        # Sort by updated time (newest first)
        graphs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        return graphs

    async def get_graph_metadata(
        self, user_id: str, graph_id: str, tx: Transaction | None = None
    ) -> dict[str, Any] | None:
        """Get graph metadata without loading the full graph."""
        # Validate inputs
        user_id = SecurityValidator.validate_user_id(user_id)
        graph_id = SecurityValidator.validate_graph_id(graph_id)

        meta_key = self._make_key("graph_meta", user_id, graph_id)
        client = tx.pipeline if tx else self._client

        metadata = await client.get(meta_key)
        if metadata:
            return json.loads(metadata)
        return None

    async def update_graph_metadata(
        self,
        user_id: str,
        graph_id: str,
        metadata: dict[str, Any],
        tx: Transaction | None = None,
    ) -> bool:
        """Update graph metadata."""
        # Validate inputs
        user_id = SecurityValidator.validate_user_id(user_id)
        graph_id = SecurityValidator.validate_graph_id(graph_id)

        # Get existing metadata
        existing = await self.get_graph_metadata(user_id, graph_id, tx)
        if not existing:
            msg = f"Graph '{graph_id}' not found"
            raise GraphNotFoundError(msg)

        # Update metadata
        existing["updated_at"] = datetime.now(UTC).isoformat()
        existing["custom"] = SecurityValidator.sanitize_attributes(
            metadata.get("custom", {})
        )

        # Save
        meta_key = self._make_key("graph_meta", user_id, graph_id)
        client = tx.pipeline if tx else self._client

        await client.set(meta_key, json.dumps(existing))
        return True

    async def get_storage_stats(self, user_id: str) -> dict[str, Any]:
        """Get storage usage statistics for a user."""
        # Validate input
        user_id = SecurityValidator.validate_user_id(user_id)

        user_stats_key = self._make_key("user_stats", user_id)
        stats = await self._client.hgetall(user_stats_key)

        # Convert and provide defaults
        return {
            "user_id": user_id,
            "graph_count": int(stats.get(b"graph_count", 0)),
            "total_bytes": int(stats.get(b"total_bytes", 0)),
            "total_mb": round(int(stats.get(b"total_bytes", 0)) / 1024 / 1024, 2),
            "quota_mb": self.max_size_bytes / 1024 / 1024,
            "usage_percent": (
                round(int(stats.get(b"total_bytes", 0)) / self.max_size_bytes * 100, 2)
                if self.max_size_bytes > 0
                else 0
            ),
        }

    async def check_health(self) -> dict[str, Any]:
        """Check Redis backend health."""
        try:
            # Ping Redis
            start = asyncio.get_event_loop().time()
            await self._client.ping()
            latency = (asyncio.get_event_loop().time() - start) * 1000

            # Get Redis info
            info = await self._client.info()

            return {
                "status": "healthy",
                "backend": "redis",
                "latency_ms": round(latency, 2),
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
                "uptime_days": round(info.get("uptime_in_seconds", 0) / 86400, 2),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "redis",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def cleanup_expired(self, days: int = 30) -> int:
        """Clean up graphs not accessed for specified days."""
        # This would be called by a scheduled job
        # Implementation depends on requirements
