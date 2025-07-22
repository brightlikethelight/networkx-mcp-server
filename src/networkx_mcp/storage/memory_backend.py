"""In-memory storage backend for development and testing."""

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import networkx as nx

from .base import (
    GraphNotFoundError,
    StorageBackend,
    StorageError,
    StorageQuotaExceededError,
    Transaction,
    TransactionError,
)


class MemoryTransaction(Transaction):
    """In-memory transaction implementation."""

    def __init__(self, backend: "MemoryBackend") -> None:
        self.backend = backend
        self._operations = []
        self._committed = False
        self._rolled_back = False

    def add_operation(self, op_type: str, *args, **kwargs) -> None:
        """Add an operation to the transaction."""
        self._operations.append((op_type, args, kwargs))

    async def commit(self) -> None:
        """Execute all operations in the transaction."""
        if self._committed:
            raise TransactionError("Transaction already committed")
        if self._rolled_back:
            raise TransactionError("Transaction already rolled back")

        # In-memory operations are immediate, so just mark as committed
        self._committed = True

    async def rollback(self) -> None:
        """Rollback is no-op for in-memory since we apply operations immediately."""
        if self._committed:
            raise TransactionError("Cannot rollback committed transaction")
        if self._rolled_back:
            raise TransactionError("Transaction already rolled back")

        self._rolled_back = True


class MemoryBackend(StorageBackend):
    """In-memory storage backend for development."""

    def __init__(self, max_graph_size_mb: int = 100) -> None:
        self.max_size_bytes = max_graph_size_mb * 1024 * 1024
        # Storage structure: {user_id: {graph_id: {"graph": nx.Graph, "metadata": dict[str, Any]}}}
        self._storage = defaultdict(dict[str, Any])
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize in-memory storage."""
        self._initialized = True

    async def close(self) -> None:
        """Close in-memory storage."""
        self._initialized = False

    @asynccontextmanager
    async def transaction(self) -> None:
        """Context manager for atomic operations."""
        tx = MemoryTransaction(self)
        try:
            yield tx
            await tx.commit()
        except Exception:
            await tx.rollback()
            raise

    async def save_graph(
        self,
        user_id: str,
        graph_id: str,
        graph: nx.Graph,
        metadata: dict[str, Any] | None = None,
        tx: Transaction | None = None,
    ) -> bool:
        """Save graph with metadata."""
        if not self._initialized:
            raise StorageError("Storage not initialized")

        async with self._lock:
            # Estimate graph size (rough approximation)
            estimated_size = (
                graph.number_of_nodes() * 100  # Assume 100 bytes per node
                + graph.number_of_edges() * 200  # Assume 200 bytes per edge
            )

            if estimated_size > self.max_size_bytes:
                raise StorageQuotaExceededError(
                    f"Graph size ({estimated_size} bytes) exceeds limit ({self.max_size_bytes} bytes)"
                )

            # Deep copy to prevent external modifications
            graph_copy = graph.copy()

            # Prepare metadata
            if metadata is None:
                metadata = {}

            metadata.update(
                {
                    "created_at": metadata.get(
                        "created_at", datetime.now(UTC).isoformat()
                    ),
                    "updated_at": datetime.now(UTC).isoformat(),
                    "num_nodes": graph.number_of_nodes(),
                    "num_edges": graph.number_of_edges(),
                }
            )

            # Store the graph
            self._storage[user_id][graph_id] = {
                "graph": graph_copy,
                "metadata": metadata.copy(),
            }

            return True

    async def load_graph(
        self, user_id: str, graph_id: str, tx: Transaction | None = None
    ) -> nx.Graph | None:
        """Load graph from storage."""
        if not self._initialized:
            raise StorageError("Storage not initialized")

        async with self._lock:
            user_data = self._storage.get(user_id, {})
            graph_data = user_data.get(graph_id)

            if graph_data is None:
                return None

            # Return a copy to prevent external modifications
            return graph_data["graph"].copy()

    async def delete_graph(
        self, user_id: str, graph_id: str, tx: Transaction | None = None
    ) -> bool:
        """Delete graph from storage."""
        if not self._initialized:
            raise StorageError("Storage not initialized")

        async with self._lock:
            user_data = self._storage.get(user_id, {})
            if graph_id in user_data:
                del user_data[graph_id]
                return True
            return False

    async def list_graphs(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        tx: Transaction | None = None,
    ) -> list[dict[str, Any]]:
        """List user's graphs with metadata."""
        if not self._initialized:
            raise StorageError("Storage not initialized")

        async with self._lock:
            user_data = self._storage.get(user_id, {})

            # Get all graphs sorted by updated_at
            graphs = []
            for graph_id, data in user_data.items():
                metadata = data["metadata"].copy()
                metadata["graph_id"] = graph_id
                graphs.append(metadata)

            # Sort by updated_at descending
            graphs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

            # Apply pagination
            return graphs[offset : offset + limit]

    async def get_graph_metadata(
        self, user_id: str, graph_id: str, tx: Transaction | None = None
    ) -> dict[str, Any] | None:
        """Get graph metadata without loading the full graph."""
        if not self._initialized:
            raise StorageError("Storage not initialized")

        async with self._lock:
            user_data = self._storage.get(user_id, {})
            graph_data = user_data.get(graph_id)

            if graph_data is None:
                return None

            metadata = graph_data["metadata"].copy()
            metadata["graph_id"] = graph_id
            return metadata

    async def update_graph_metadata(
        self,
        user_id: str,
        graph_id: str,
        metadata: dict[str, Any],
        tx: Transaction | None = None,
    ) -> bool:
        """Update graph metadata."""
        if not self._initialized:
            raise StorageError("Storage not initialized")

        async with self._lock:
            user_data = self._storage.get(user_id, {})
            graph_data = user_data.get(graph_id)

            if graph_data is None:
                raise GraphNotFoundError(f"Graph {graph_id} not found")

            # Update metadata
            graph_data["metadata"].update(metadata)
            graph_data["metadata"]["updated_at"] = datetime.now(UTC).isoformat()

            return True

    async def get_storage_stats(self, user_id: str) -> dict[str, Any]:
        """Get storage usage statistics for a user."""
        if not self._initialized:
            raise StorageError("Storage not initialized")

        async with self._lock:
            user_data = self._storage.get(user_id, {})

            total_nodes = 0
            total_edges = 0
            total_graphs = len(user_data)

            for graph_data in user_data.values():
                graph = graph_data["graph"]
                total_nodes += graph.number_of_nodes()
                total_edges += graph.number_of_edges()

            # Estimate storage usage
            estimated_bytes = total_nodes * 100 + total_edges * 200

            return {
                "total_graphs": total_graphs,
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "estimated_bytes": estimated_bytes,
                "estimated_mb": estimated_bytes / (1024 * 1024),
                "quota_mb": self.max_size_bytes / (1024 * 1024),
                "usage_percent": (estimated_bytes / self.max_size_bytes) * 100,
            }

    async def check_health(self) -> dict[str, Any]:
        """Check storage backend health."""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "type": "memory",
            "total_users": len(self._storage),
            "total_graphs": sum(len(graphs) for graphs in self._storage.values()),
            "backend": "MemoryBackend",
            "persistent": False,
            "message": "In-memory storage is active (data will be lost on restart)",
        }
