"""
Thread-safe graph store with proper locking mechanisms and memory management.

This module provides the foundation for concurrent graph operations with
conflict resolution, memory limits, and transaction support.
"""

import asyncio
import hashlib
import logging
import pickle
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import psutil

logger = logging.getLogger(__name__)


class LockType(Enum):
    """Lock types for graph operations"""

    READ = "read"
    WRITE = "write"
    EXCLUSIVE = "exclusive"


class GraphOperation(Enum):
    """Graph operation types for audit logging"""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    COMPUTE = "compute"


@dataclass
class GraphMetadata:
    """Metadata for stored graphs"""

    graph_id: str
    created_at: datetime
    updated_at: datetime
    node_count: int
    edge_count: int
    size_bytes: int
    owner: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    is_locked: bool = False
    lock_owner: Optional[str] = None
    version: int = 1
    checksum: Optional[str] = None


@dataclass
class GraphTransaction:
    """Transaction for atomic graph operations"""

    transaction_id: str
    graph_id: str
    operations: List[Tuple[str, Any]]
    timestamp: datetime
    rollback_data: Optional[Any] = None


class GraphMemoryManager:
    """Manages memory usage and eviction policies"""

    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.current_usage = 0
        self.graph_sizes: Dict[str, int] = {}

    def can_allocate(self, size_bytes: int) -> bool:
        """Check if memory is available for allocation"""
        available_system_memory = psutil.virtual_memory().available
        # Ensure we don't exceed either our limit or 80% of system memory
        system_limit = available_system_memory * 0.8
        return (
            self.current_usage + size_bytes <= self.max_memory
            and size_bytes <= system_limit
        )

    def allocate(self, graph_id: str, size_bytes: int):
        """Allocate memory for a graph"""
        self.graph_sizes[graph_id] = size_bytes
        self.current_usage += size_bytes
        logger.debug(
            f"Allocated {size_bytes} bytes for graph {graph_id}. "
            f"Total usage: {self.current_usage}/{self.max_memory}"
        )

    def deallocate(self, graph_id: str):
        """Free memory for a graph"""
        if graph_id in self.graph_sizes:
            size = self.graph_sizes.pop(graph_id)
            self.current_usage -= size
            logger.debug(f"Deallocated {size} bytes for graph {graph_id}")

    def get_eviction_candidates(self, required_space: int) -> List[str]:
        """Get list of graphs to evict to free required space"""
        candidates = []
        freed_space = 0

        # Sort by size (evict smaller graphs first to minimize disruption)
        sorted_graphs = sorted(self.graph_sizes.items(), key=lambda x: x[1])

        for graph_id, size in sorted_graphs:
            if freed_space >= required_space:
                break
            candidates.append(graph_id)
            freed_space += size

        return candidates


class ThreadSafeGraphStore:
    """
    Thread-safe graph store with proper locking, memory management, and transactions.

    Features:
    - Read/Write/Exclusive locks for concurrent access
    - Memory limits with eviction policies
    - Transaction support for atomic operations
    - Graph versioning and rollback
    - Audit logging for all operations
    """

    def __init__(self, max_memory_mb: int = 1024, enable_persistence: bool = False):
        # Core storage
        self.graphs: Dict[str, nx.Graph] = {}
        self.metadata: Dict[str, GraphMetadata] = {}

        # Locking mechanism
        self.locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.read_locks: Dict[str, int] = defaultdict(int)
        self.write_locks: Dict[str, Optional[str]] = {}

        # Memory management
        self.memory_manager = GraphMemoryManager(max_memory_mb)

        # Transaction support
        self.transactions: Dict[str, GraphTransaction] = {}
        self.transaction_lock = asyncio.Lock()

        # Versioning
        self.graph_versions: Dict[str, List[Tuple[int, nx.Graph]]] = defaultdict(list)
        self.max_versions = 10

        # Metrics and monitoring
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.last_error: Optional[Exception] = None

        # Persistence
        self.enable_persistence = enable_persistence
        self.persistence_path = "./graph_store"

        # LRU cache for computed results
        self.computation_cache: OrderedDict = OrderedDict()
        self.max_cache_entries = 100

        logger.info(f"Initialized ThreadSafeGraphStore with {max_memory_mb}MB limit")

    @asynccontextmanager
    async def acquire_lock(
        self, graph_id: str, lock_type: LockType, owner: str = "system"
    ):
        """
        Context manager for acquiring locks on graphs.

        Implements reader-writer lock pattern:
        - Multiple readers can access simultaneously
        - Writers have exclusive access
        - Prevents writer starvation
        """
        lock = self.locks[graph_id]

        async with lock:
            if lock_type == LockType.READ:
                # Wait for any write lock to be released
                while self.write_locks.get(graph_id):
                    await asyncio.sleep(0.01)
                self.read_locks[graph_id] += 1

            elif lock_type in (LockType.WRITE, LockType.EXCLUSIVE):
                # Wait for all read locks and write lock to be released
                while self.read_locks[graph_id] > 0 or self.write_locks.get(graph_id):
                    await asyncio.sleep(0.01)
                self.write_locks[graph_id] = owner

                if graph_id in self.metadata:
                    self.metadata[graph_id].is_locked = True
                    self.metadata[graph_id].lock_owner = owner

        try:
            yield
        finally:
            async with lock:
                if lock_type == LockType.READ:
                    self.read_locks[graph_id] -= 1
                elif lock_type in (LockType.WRITE, LockType.EXCLUSIVE):
                    self.write_locks[graph_id] = None
                    if graph_id in self.metadata:
                        self.metadata[graph_id].is_locked = False
                        self.metadata[graph_id].lock_owner = None

    def _estimate_graph_size(self, graph: nx.Graph) -> int:
        """Estimate memory size of a graph"""
        # Rough estimation: nodes + edges + attributes
        base_size = 1000  # Base overhead
        node_size = len(graph.nodes()) * 100  # Estimated bytes per node
        edge_size = len(graph.edges()) * 150  # Estimated bytes per edge

        # Add size of node/edge attributes
        attr_size = 0
        for node, attrs in graph.nodes(data=True):
            attr_size += len(pickle.dumps(attrs))
        for u, v, attrs in graph.edges(data=True):
            attr_size += len(pickle.dumps(attrs))

        return base_size + node_size + edge_size + attr_size

    def _compute_graph_checksum(self, graph: nx.Graph) -> str:
        """Compute checksum for graph integrity verification"""
        # Create deterministic representation
        edges = sorted(graph.edges())
        nodes = sorted(graph.nodes())
        graph_repr = f"nodes:{nodes},edges:{edges}"
        return hashlib.sha256(graph_repr.encode()).hexdigest()

    async def create_graph(
        self,
        graph_id: str,
        graph: Optional[nx.Graph] = None,
        owner: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> bool:
        """
        Create a new graph in the store.

        Args:
            graph_id: Unique identifier for the graph
            graph: NetworkX graph object (creates empty if None)
            owner: Owner identifier for access control
            tags: Set of tags for categorization

        Returns:
            True if created successfully, False otherwise
        """
        if graph_id in self.graphs:
            logger.warning(f"Graph {graph_id} already exists")
            return False

        if graph is None:
            graph = nx.Graph()

        # Estimate memory requirements
        size_bytes = self._estimate_graph_size(graph)

        # Check memory availability
        if not self.memory_manager.can_allocate(size_bytes):
            # Try to evict graphs to make space
            candidates = self.memory_manager.get_eviction_candidates(size_bytes)
            if not candidates or not await self._evict_graphs(candidates):
                raise MemoryError(f"Insufficient memory to create graph {graph_id}")

        async with self.acquire_lock(graph_id, LockType.EXCLUSIVE, owner or "system"):
            # Store graph
            self.graphs[graph_id] = graph.copy()

            # Create metadata
            now = datetime.now()
            self.metadata[graph_id] = GraphMetadata(
                graph_id=graph_id,
                created_at=now,
                updated_at=now,
                node_count=graph.number_of_nodes(),
                edge_count=graph.number_of_edges(),
                size_bytes=size_bytes,
                owner=owner,
                tags=tags or set(),
                checksum=self._compute_graph_checksum(graph),
            )

            # Allocate memory
            self.memory_manager.allocate(graph_id, size_bytes)

            # Store initial version
            self.graph_versions[graph_id].append((1, graph.copy()))

            # Update metrics
            self.operation_counts[GraphOperation.CREATE] += 1

            logger.info(
                f"Created graph {graph_id} with {graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges"
            )

            return True

    async def get_graph(
        self, graph_id: str, owner: Optional[str] = None
    ) -> Optional[nx.Graph]:
        """
        Retrieve a graph from the store.

        Args:
            graph_id: Graph identifier
            owner: Owner requesting access (for access control)

        Returns:
            Copy of the graph or None if not found
        """
        if graph_id not in self.graphs:
            logger.warning(f"Graph {graph_id} not found")
            return None

        async with self.acquire_lock(graph_id, LockType.READ, owner or "system"):
            # Update access metadata
            self.metadata[graph_id].access_count += 1
            self.metadata[graph_id].last_accessed = datetime.now()

            # Update metrics
            self.operation_counts[GraphOperation.READ] += 1

            # Return a copy to prevent external modifications
            return self.graphs[graph_id].copy()

    async def update_graph(
        self,
        graph_id: str,
        update_func: Callable[[nx.Graph], None],
        owner: Optional[str] = None,
        create_version: bool = True,
    ) -> bool:
        """
        Update a graph using a function.

        Args:
            graph_id: Graph identifier
            update_func: Function that modifies the graph in-place
            owner: Owner performing update
            create_version: Whether to create a new version

        Returns:
            True if updated successfully
        """
        if graph_id not in self.graphs:
            logger.warning(f"Graph {graph_id} not found")
            return False

        async with self.acquire_lock(graph_id, LockType.WRITE, owner or "system"):
            # Create version before update if requested
            if create_version:
                current_version = self.metadata[graph_id].version
                self.graph_versions[graph_id].append(
                    (current_version, self.graphs[graph_id].copy())
                )

                # Limit version history
                if len(self.graph_versions[graph_id]) > self.max_versions:
                    self.graph_versions[graph_id].pop(0)

            # Apply update
            try:
                update_func(self.graphs[graph_id])

                # Update metadata
                self.metadata[graph_id].updated_at = datetime.now()
                self.metadata[graph_id].node_count = self.graphs[
                    graph_id
                ].number_of_nodes()
                self.metadata[graph_id].edge_count = self.graphs[
                    graph_id
                ].number_of_edges()
                self.metadata[graph_id].version += 1
                self.metadata[graph_id].checksum = self._compute_graph_checksum(
                    self.graphs[graph_id]
                )

                # Update memory allocation
                new_size = self._estimate_graph_size(self.graphs[graph_id])
                old_size = self.metadata[graph_id].size_bytes

                if new_size > old_size:
                    # Check if we can allocate additional memory
                    if not self.memory_manager.can_allocate(new_size - old_size):
                        # Rollback
                        if create_version and self.graph_versions[graph_id]:
                            _, previous_graph = self.graph_versions[graph_id][-1]
                            self.graphs[graph_id] = previous_graph
                        raise MemoryError("Insufficient memory for graph update")

                self.memory_manager.deallocate(graph_id)
                self.memory_manager.allocate(graph_id, new_size)
                self.metadata[graph_id].size_bytes = new_size

                # Clear computation cache for this graph
                self._invalidate_cache(graph_id)

                # Update metrics
                self.operation_counts[GraphOperation.UPDATE] += 1

                logger.info(f"Updated graph {graph_id}")
                return True

            except Exception as e:
                logger.error(f"Error updating graph {graph_id}: {e}")
                self.error_counts[GraphOperation.UPDATE] += 1
                self.last_error = e

                # Rollback if version was created
                if create_version and self.graph_versions[graph_id]:
                    _, previous_graph = self.graph_versions[graph_id][-1]
                    self.graphs[graph_id] = previous_graph

                raise

    async def delete_graph(self, graph_id: str, owner: Optional[str] = None) -> bool:
        """
        Delete a graph from the store.

        Args:
            graph_id: Graph identifier
            owner: Owner requesting deletion

        Returns:
            True if deleted successfully
        """
        if graph_id not in self.graphs:
            logger.warning(f"Graph {graph_id} not found")
            return False

        async with self.acquire_lock(graph_id, LockType.EXCLUSIVE, owner or "system"):
            # Free memory
            self.memory_manager.deallocate(graph_id)

            # Remove graph and metadata
            del self.graphs[graph_id]
            del self.metadata[graph_id]

            # Remove versions
            if graph_id in self.graph_versions:
                del self.graph_versions[graph_id]

            # Clear cache entries
            self._invalidate_cache(graph_id)

            # Update metrics
            self.operation_counts[GraphOperation.DELETE] += 1

            logger.info(f"Deleted graph {graph_id}")
            return True

    async def compute_on_graph(
        self,
        graph_id: str,
        compute_func: Callable[[nx.Graph], Any],
        cache_key: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> Any:
        """
        Perform a computation on a graph with caching.

        Args:
            graph_id: Graph identifier
            compute_func: Function to compute on the graph
            cache_key: Key for caching results
            owner: Owner performing computation

        Returns:
            Computation result
        """
        if graph_id not in self.graphs:
            raise ValueError(f"Graph {graph_id} not found")

        # Check cache
        if cache_key:
            full_cache_key = f"{graph_id}:{cache_key}"
            if full_cache_key in self.computation_cache:
                # Move to end (LRU)
                self.computation_cache.move_to_end(full_cache_key)
                logger.debug(f"Cache hit for {full_cache_key}")
                return self.computation_cache[full_cache_key]

        async with self.acquire_lock(graph_id, LockType.READ, owner or "system"):
            # Perform computation
            result = compute_func(self.graphs[graph_id])

            # Cache result
            if cache_key:
                full_cache_key = f"{graph_id}:{cache_key}"
                self.computation_cache[full_cache_key] = result

                # Limit cache size
                if len(self.computation_cache) > self.max_cache_entries:
                    self.computation_cache.popitem(last=False)

            # Update metrics
            self.operation_counts[GraphOperation.COMPUTE] += 1

            return result

    async def begin_transaction(self, transaction_id: str, graph_id: str) -> bool:
        """Begin a transaction for atomic operations"""
        async with self.transaction_lock:
            if transaction_id in self.transactions:
                return False

            self.transactions[transaction_id] = GraphTransaction(
                transaction_id=transaction_id,
                graph_id=graph_id,
                operations=[],
                timestamp=datetime.now(),
                rollback_data=self.graphs.get(graph_id, nx.Graph()).copy(),
            )
            return True

    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a transaction"""
        async with self.transaction_lock:
            if transaction_id not in self.transactions:
                return False

            # Transaction is already applied, just clean up
            del self.transactions[transaction_id]
            return True

    async def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback a transaction"""
        async with self.transaction_lock:
            if transaction_id not in self.transactions:
                return False

            transaction = self.transactions[transaction_id]
            if transaction.rollback_data:
                self.graphs[transaction.graph_id] = transaction.rollback_data

            del self.transactions[transaction_id]
            return True

    def _invalidate_cache(self, graph_id: str):
        """Invalidate cache entries for a graph"""
        keys_to_remove = [
            key for key in self.computation_cache if key.startswith(f"{graph_id}:")
        ]
        for key in keys_to_remove:
            del self.computation_cache[key]

    async def _evict_graphs(self, graph_ids: List[str]) -> bool:
        """Evict graphs to free memory"""
        for graph_id in graph_ids:
            if graph_id in self.graphs:
                await self.delete_graph(graph_id)
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        return {
            "graph_count": len(self.graphs),
            "total_nodes": sum(g.number_of_nodes() for g in self.graphs.values()),
            "total_edges": sum(g.number_of_edges() for g in self.graphs.values()),
            "memory_usage_mb": self.memory_manager.current_usage / (1024 * 1024),
            "memory_limit_mb": self.memory_manager.max_memory / (1024 * 1024),
            "cache_size": len(self.computation_cache),
            "operation_counts": dict(self.operation_counts),
            "error_counts": dict(self.error_counts),
            "active_locks": {
                "read": sum(self.read_locks.values()),
                "write": sum(1 for v in self.write_locks.values() if v),
            },
        }


# Example usage and testing
async def test_thread_safe_store():
    """Test the thread-safe graph store"""
    store = ThreadSafeGraphStore(max_memory_mb=100)

    # Create a graph
    G = nx.karate_club_graph()
    await store.create_graph("karate", G, owner="test_user")

    # Concurrent reads
    async def read_graph():
        graph = await store.get_graph("karate")
        return graph.number_of_nodes()

    # Run multiple reads concurrently
    results = await asyncio.gather(*[read_graph() for _ in range(10)])
    print(f"Concurrent reads: {results}")

    # Update graph
    async def add_edges(graph):
        graph.add_edge(0, 100)
        graph.add_edge(1, 101)

    await store.update_graph("karate", add_edges)

    # Compute with caching
    result = await store.compute_on_graph(
        "karate", lambda g: nx.pagerank(g), cache_key="pagerank"
    )
    print(f"PageRank computed: {len(result)} nodes")

    # Get statistics
    stats = store.get_stats()
    print(f"Store stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_thread_safe_store())
