"""Graph repository for data persistence.

This module provides the data access layer for graph metadata and data
with support for multiple storage backends.
"""

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..core.base import Component, ComponentStatus, Repository
from ..core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class GraphMetadata:
    """Metadata for a graph."""

    graph_id: str
    graph_type: str
    description: str
    created_at: datetime
    updated_at: datetime | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.updated_at is None:
            self.updated_at = self.created_at


class GraphRepository(Repository[GraphMetadata], Component):
    """Repository for graph metadata and data."""

    def __init__(self, storage_backend: Optional["StorageBackend"] = None):
        Component.__init__(self, "GraphRepository")
        self.storage_backend = storage_backend or FileStorageBackend()
        self._metadata_cache: dict[str, GraphMetadata] = {}
        self._cache_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the repository."""
        await self._set_status(ComponentStatus.INITIALIZING)
        await self.storage_backend.initialize()
        await self._load_metadata_cache()
        await self._set_status(ComponentStatus.READY)
        logger.info("Graph repository initialized")

    async def shutdown(self) -> None:
        """Shutdown the repository."""
        await self._set_status(ComponentStatus.SHUTTING_DOWN)
        await self.storage_backend.shutdown()
        self._metadata_cache.clear()
        await self._set_status(ComponentStatus.SHUTDOWN)
        logger.info("Graph repository shutdown")

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        backend_health = await self.storage_backend.health_check()
        return {
            "healthy": self.status == ComponentStatus.READY
            and backend_health["healthy"],
            "storage_backend": backend_health,
            "cached_graphs": len(self._metadata_cache),
        }

    async def get(self, graph_id: str) -> GraphMetadata | None:
        """Get graph metadata by ID."""
        async with self._cache_lock:
            if graph_id in self._metadata_cache:
                return self._metadata_cache[graph_id]

        # Load from storage
        metadata = await self.storage_backend.get_metadata(graph_id)
        if metadata:
            async with self._cache_lock:
                self._metadata_cache[graph_id] = metadata

        return metadata

    async def list(self, **filters) -> list[GraphMetadata]:
        """List all graph metadata with optional filters."""
        # For now, return all cached metadata
        # In a real implementation, this would query the storage backend
        async with self._cache_lock:
            all_metadata = list(self._metadata_cache.values())

        # Apply filters
        if filters:
            filtered_metadata = []
            for metadata in all_metadata:
                match = True
                for key, value in filters.items():
                    if not hasattr(metadata, key) or getattr(metadata, key) != value:
                        match = False
                        break
                if match:
                    filtered_metadata.append(metadata)
            return filtered_metadata

        return all_metadata

    async def create(self, metadata: GraphMetadata) -> GraphMetadata:
        """Create new graph metadata."""
        # Check if already exists
        existing = await self.get(metadata.graph_id)
        if existing:
            raise ValueError(f"Graph {metadata.graph_id} already exists")

        # Save to storage
        await self.storage_backend.save_metadata(metadata)

        # Update cache
        async with self._cache_lock:
            self._metadata_cache[metadata.graph_id] = metadata

        logger.debug(f"Created metadata for graph {metadata.graph_id}")
        return metadata

    async def update(
        self, graph_id: str, metadata: GraphMetadata
    ) -> GraphMetadata | None:
        """Update existing graph metadata."""
        existing = await self.get(graph_id)
        if not existing:
            return None

        # Update timestamp
        metadata.updated_at = datetime.utcnow()

        # Save to storage
        await self.storage_backend.save_metadata(metadata)

        # Update cache
        async with self._cache_lock:
            self._metadata_cache[graph_id] = metadata

        logger.debug(f"Updated metadata for graph {graph_id}")
        return metadata

    async def delete(self, graph_id: str) -> bool:
        """Delete graph metadata and data."""
        # Check if exists
        existing = await self.get(graph_id)
        if not existing:
            return False

        # Delete from storage
        await self.storage_backend.delete_metadata(graph_id)
        await self.storage_backend.delete_graph_data(graph_id)

        # Remove from cache
        async with self._cache_lock:
            self._metadata_cache.pop(graph_id, None)

        logger.debug(f"Deleted graph {graph_id}")
        return True

    async def exists(self, graph_id: str) -> bool:
        """Check if graph exists."""
        return await self.get(graph_id) is not None

    async def get_graph_data(self, graph_id: str) -> dict[str, Any] | None:
        """Get graph data (nodes and edges)."""
        return await self.storage_backend.get_graph_data(graph_id)

    async def save_graph_data(self, graph_id: str, graph_data: dict[str, Any]) -> None:
        """Save graph data (nodes and edges)."""
        await self.storage_backend.save_graph_data(graph_id, graph_data)

        # Update metadata timestamp
        metadata = await self.get(graph_id)
        if metadata:
            metadata.updated_at = datetime.utcnow()
            await self.update(graph_id, metadata)

    async def _load_metadata_cache(self) -> None:
        """Load all metadata into cache."""
        all_metadata = await self.storage_backend.list_metadata()
        async with self._cache_lock:
            for metadata in all_metadata:
                self._metadata_cache[metadata.graph_id] = metadata

        logger.debug(f"Loaded {len(all_metadata)} graph metadata into cache")


class StorageBackend(Component):
    """Abstract storage backend."""

    async def get_metadata(self, graph_id: str) -> GraphMetadata | None:
        """Get metadata for a graph."""
        raise NotImplementedError

    async def save_metadata(self, metadata: GraphMetadata) -> None:
        """Save metadata for a graph."""
        raise NotImplementedError

    async def delete_metadata(self, graph_id: str) -> None:
        """Delete metadata for a graph."""
        raise NotImplementedError

    async def list_metadata(self) -> list[GraphMetadata]:
        """List all metadata."""
        raise NotImplementedError

    async def get_graph_data(self, graph_id: str) -> dict[str, Any] | None:
        """Get graph data."""
        raise NotImplementedError

    async def save_graph_data(self, graph_id: str, data: dict[str, Any]) -> None:
        """Save graph data."""
        raise NotImplementedError

    async def delete_graph_data(self, graph_id: str) -> None:
        """Delete graph data."""
        raise NotImplementedError


class FileStorageBackend(StorageBackend):
    """File-based storage backend."""

    def __init__(self, data_dir: str | None = None):
        super().__init__("FileStorageBackend")
        self.data_dir = Path(data_dir or "./data/graphs")
        self.metadata_dir = self.data_dir / "metadata"
        self.graph_data_dir = self.data_dir / "data"

    async def initialize(self) -> None:
        """Initialize file storage."""
        await self._set_status(ComponentStatus.INITIALIZING)

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.graph_data_dir.mkdir(parents=True, exist_ok=True)

        await self._set_status(ComponentStatus.READY)
        logger.debug(f"File storage initialized at {self.data_dir}")

    async def shutdown(self) -> None:
        """Shutdown file storage."""
        await self._set_status(ComponentStatus.SHUTDOWN)

    async def health_check(self) -> dict[str, Any]:
        """Check file storage health."""
        return {
            "healthy": self.status == ComponentStatus.READY,
            "data_dir": str(self.data_dir),
            "writable": self.data_dir.exists() and os.access(self.data_dir, os.W_OK),
        }

    async def get_metadata(self, graph_id: str) -> GraphMetadata | None:
        """Get metadata from file."""
        file_path = self.metadata_dir / f"{graph_id}.json"
        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                data = json.load(f)

            # Convert datetime strings back to datetime objects
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            if data.get("updated_at"):
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])

            return GraphMetadata(**data)
        except Exception as e:
            logger.error(f"Failed to load metadata for {graph_id}: {e}")
            return None

    async def save_metadata(self, metadata: GraphMetadata) -> None:
        """Save metadata to file."""
        file_path = self.metadata_dir / f"{metadata.graph_id}.json"

        try:
            data = asdict(metadata)
            # Convert datetime objects to strings
            data["created_at"] = metadata.created_at.isoformat()
            if metadata.updated_at:
                data["updated_at"] = metadata.updated_at.isoformat()

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata for {metadata.graph_id}: {e}")
            raise

    async def delete_metadata(self, graph_id: str) -> None:
        """Delete metadata file."""
        file_path = self.metadata_dir / f"{graph_id}.json"
        if file_path.exists():
            file_path.unlink()

    async def list_metadata(self) -> list[GraphMetadata]:
        """List all metadata files."""
        metadata_list = []

        for file_path in self.metadata_dir.glob("*.json"):
            graph_id = file_path.stem
            metadata = await self.get_metadata(graph_id)
            if metadata:
                metadata_list.append(metadata)

        return metadata_list

    async def get_graph_data(self, graph_id: str) -> dict[str, Any] | None:
        """Get graph data from file."""
        file_path = self.graph_data_dir / f"{graph_id}.json"
        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load graph data for {graph_id}: {e}")
            return None

    async def save_graph_data(self, graph_id: str, data: dict[str, Any]) -> None:
        """Save graph data to file."""
        file_path = self.graph_data_dir / f"{graph_id}.json"

        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save graph data for {graph_id}: {e}")
            raise

    async def delete_graph_data(self, graph_id: str) -> None:
        """Delete graph data file."""
        file_path = self.graph_data_dir / f"{graph_id}.json"
        if file_path.exists():
            file_path.unlink()


class RedisStorageBackend(StorageBackend):
    """Redis-based storage backend."""

    def __init__(self, redis_url: str | None = None):
        super().__init__("RedisStorageBackend")
        self.redis_url = redis_url or get_config().redis.url
        self.redis_client = None
        self.prefix = get_config().redis.prefix

    async def initialize(self) -> None:
        """Initialize Redis storage."""
        await self._set_status(ComponentStatus.INITIALIZING)

        if not self.redis_url:
            raise ValueError("Redis URL not configured")

        try:
            import redis.asyncio as redis

            self.redis_client = redis.from_url(self.redis_url)

            # Test connection
            await self.redis_client.ping()

            await self._set_status(ComponentStatus.READY)
            logger.debug("Redis storage initialized")
        except ImportError:
            raise RuntimeError("redis package not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Redis storage: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown Redis storage."""
        await self._set_status(ComponentStatus.SHUTTING_DOWN)
        if self.redis_client:
            await self.redis_client.close()
        await self._set_status(ComponentStatus.SHUTDOWN)

    async def health_check(self) -> dict[str, Any]:
        """Check Redis storage health."""
        try:
            if self.redis_client:
                await self.redis_client.ping()
                return {"healthy": True, "redis_url": self.redis_url}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

        return {"healthy": False, "error": "Redis client not initialized"}

    async def get_metadata(self, graph_id: str) -> GraphMetadata | None:
        """Get metadata from Redis."""
        if not self.redis_client:
            return None

        key = f"{self.prefix}:metadata:{graph_id}"
        data = await self.redis_client.get(key)

        if not data:
            return None

        try:
            metadata_dict = json.loads(data)
            metadata_dict["created_at"] = datetime.fromisoformat(
                metadata_dict["created_at"]
            )
            if metadata_dict.get("updated_at"):
                metadata_dict["updated_at"] = datetime.fromisoformat(
                    metadata_dict["updated_at"]
                )

            return GraphMetadata(**metadata_dict)
        except Exception as e:
            logger.error(f"Failed to deserialize metadata for {graph_id}: {e}")
            return None

    async def save_metadata(self, metadata: GraphMetadata) -> None:
        """Save metadata to Redis."""
        if not self.redis_client:
            return

        key = f"{self.prefix}:metadata:{metadata.graph_id}"

        data = asdict(metadata)
        data["created_at"] = metadata.created_at.isoformat()
        if metadata.updated_at:
            data["updated_at"] = metadata.updated_at.isoformat()

        await self.redis_client.set(key, json.dumps(data))

    async def delete_metadata(self, graph_id: str) -> None:
        """Delete metadata from Redis."""
        if not self.redis_client:
            return

        key = f"{self.prefix}:metadata:{graph_id}"
        await self.redis_client.delete(key)

    async def list_metadata(self) -> list[GraphMetadata]:
        """List all metadata from Redis."""
        if not self.redis_client:
            return []

        pattern = f"{self.prefix}:metadata:*"
        keys = await self.redis_client.keys(pattern)

        metadata_list = []
        for key in keys:
            graph_id = key.split(":")[-1]
            metadata = await self.get_metadata(graph_id)
            if metadata:
                metadata_list.append(metadata)

        return metadata_list

    async def get_graph_data(self, graph_id: str) -> dict[str, Any] | None:
        """Get graph data from Redis."""
        if not self.redis_client:
            return None

        key = f"{self.prefix}:data:{graph_id}"
        data = await self.redis_client.get(key)

        if not data:
            return None

        try:
            return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize graph data for {graph_id}: {e}")
            return None

    async def save_graph_data(self, graph_id: str, data: dict[str, Any]) -> None:
        """Save graph data to Redis."""
        if not self.redis_client:
            return

        key = f"{self.prefix}:data:{graph_id}"
        await self.redis_client.set(key, json.dumps(data))

    async def delete_graph_data(self, graph_id: str) -> None:
        """Delete graph data from Redis."""
        if not self.redis_client:
            return

        key = f"{self.prefix}:data:{graph_id}"
        await self.redis_client.delete(key)
