"""Base storage abstraction with transaction support."""

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import networkx as nx


class Transaction(ABC):
    """Abstract transaction for atomic operations."""

    @abstractmethod
    async def commit(self) -> None:
        """Commit all operations in this transaction."""

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback all operations in this transaction."""


class StorageBackend(ABC):
    """Abstract storage backend with transaction support."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize storage connections."""

    @abstractmethod
    async def close(self) -> None:
        """Close storage connections."""

    @abstractmethod
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Transaction, None]:
        """Context manager for atomic operations."""
        # This is an abstract method, implementations should yield a Transaction
        raise NotImplementedError("Subclasses must implement transaction()")
        yield  # type: ignore[unreachable]

    @abstractmethod
    async def save_graph(
        self,
        user_id: str,
        graph_id: str,
        graph: nx.Graph,
        metadata: Optional[Dict[str, Any]] = None,
        tx: Optional[Transaction] = None,
    ) -> bool:
        """Save graph with metadata."""

    @abstractmethod
    async def load_graph(
        self, user_id: str, graph_id: str, tx: Optional[Transaction] = None
    ) -> Optional[nx.Graph]:
        """Load graph from storage."""

    @abstractmethod
    async def delete_graph(
        self, user_id: str, graph_id: str, tx: Optional[Transaction] = None
    ) -> bool:
        """Delete graph from storage."""

    @abstractmethod
    async def list_graphs(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        tx: Optional[Transaction] = None,
    ) -> List[Dict[str, Any]]:
        """List user's graphs with metadata."""

    @abstractmethod
    async def get_graph_metadata(
        self, user_id: str, graph_id: str, tx: Optional[Transaction] = None
    ) -> Optional[Dict[str, Any]]:
        """Get graph metadata without loading the full graph."""

    @abstractmethod
    async def update_graph_metadata(
        self,
        user_id: str,
        graph_id: str,
        metadata: Dict[str, Any],
        tx: Optional[Transaction] = None,
    ) -> bool:
        """Update graph metadata."""

    @abstractmethod
    async def get_storage_stats(self, user_id: str) -> Dict[str, Any]:
        """Get storage usage statistics for a user."""

    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Check storage backend health."""


class StorageError(Exception):
    """Base exception for storage errors."""


class GraphNotFoundError(StorageError):
    """Raised when requested graph doesn't exist."""


class StorageQuotaExceededError(StorageError):
    """Raised when user exceeds storage quota."""


class TransactionError(StorageError):
    """Raised when transaction operations fail."""
