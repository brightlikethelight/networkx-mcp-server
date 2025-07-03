"""Graph service module for NetworkX MCP Server.

This module provides the core graph management service with proper
dependency injection and modular architecture.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

import networkx as nx

from ..caching.cache_service import CacheService
from ..core.base import Service
from ..core.config import get_config
from ..events.graph_events import GraphEventPublisher
from ..repositories.graph_repository import GraphMetadata, GraphRepository
from ..validators.graph_validator import GraphValidator

logger = logging.getLogger(__name__)


class GraphService(Service):
    """Service for managing graph operations."""

    def __init__(
        self,
        repository: GraphRepository,
        validator: GraphValidator,
        event_publisher: GraphEventPublisher,
        cache_service: CacheService | None = None,
    ):
        super().__init__("GraphService")
        self.repository = repository
        self.validator = validator
        self.event_publisher = event_publisher
        self.cache_service = cache_service
        self._config = get_config()

        # In-memory graph storage for active graphs
        self._active_graphs: dict[str, nx.Graph] = {}
        self._graph_locks: dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        """Initialize the graph service."""
        await super().initialize()
        logger.info("Graph service initialized")

    async def shutdown(self) -> None:
        """Shutdown the graph service."""
        await self._set_status(self.ComponentStatus.SHUTTING_DOWN)

        # Save all active graphs
        for graph_id in list(self._active_graphs.keys()):
            await self._persist_graph(graph_id)

        self._active_graphs.clear()
        self._graph_locks.clear()

        await self._set_status(self.ComponentStatus.SHUTDOWN)
        logger.info("Graph service shutdown complete")

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        return {
            "healthy": self.status == self.ComponentStatus.READY,
            "active_graphs": len(self._active_graphs),
            "total_nodes": sum(
                g.number_of_nodes() for g in self._active_graphs.values()
            ),
            "total_edges": sum(
                g.number_of_edges() for g in self._active_graphs.values()
            ),
            "memory_usage_mb": self._estimate_memory_usage(),
        }

    async def create_graph(
        self,
        graph_id: str,
        graph_type: str = "Graph",
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new graph."""
        async with self.track_request():
            # Validate input
            validation = await self.validator.validate_graph_creation(
                {
                    "graph_id": graph_id,
                    "graph_type": graph_type,
                    "description": description,
                }
            )

            if not validation.valid:
                raise ValueError(f"Invalid graph creation request: {validation.errors}")

            # Check if graph already exists
            if graph_id in self._active_graphs:
                raise ValueError(f"Graph '{graph_id}' already exists")

            # Create graph instance
            graph_class = getattr(nx, graph_type)
            graph = graph_class()

            # Create metadata
            graph_metadata = GraphMetadata(
                graph_id=graph_id,
                graph_type=graph_type,
                description=description or "",
                created_at=datetime.utcnow(),
                metadata=metadata or {},
            )

            # Store in repository
            await self.repository.create(graph_metadata)

            # Add to active graphs
            self._active_graphs[graph_id] = graph
            self._graph_locks[graph_id] = asyncio.Lock()

            # Publish event
            await self.event_publisher.publish_graph_created(graph_id, graph_type)

            logger.info(f"Created graph '{graph_id}' of type {graph_type}")

            return {
                "success": True,
                "graph_id": graph_id,
                "graph_type": graph_type,
                "message": f"Graph '{graph_id}' created successfully",
            }

    async def delete_graph(self, graph_id: str) -> dict[str, Any]:
        """Delete a graph."""
        async with self.track_request():
            if graph_id not in self._active_graphs:
                # Try to load from repository
                if not await self.repository.exists(graph_id):
                    raise ValueError(f"Graph '{graph_id}' not found")
                await self._load_graph(graph_id)

            async with self._get_graph_lock(graph_id):
                # Remove from active graphs
                if graph_id in self._active_graphs:
                    del self._active_graphs[graph_id]

                # Remove lock
                if graph_id in self._graph_locks:
                    del self._graph_locks[graph_id]

                # Delete from repository
                await self.repository.delete(graph_id)

                # Clear cache
                if self.cache_service:
                    await self.cache_service.delete_pattern(f"graph:{graph_id}:*")

                # Publish event
                await self.event_publisher.publish_graph_deleted(graph_id)

                logger.info(f"Deleted graph '{graph_id}'")

                return {
                    "success": True,
                    "graph_id": graph_id,
                    "message": f"Graph '{graph_id}' deleted successfully",
                }

    async def get_graph(self, graph_id: str, load_if_needed: bool = True) -> nx.Graph:
        """Get a graph by ID."""
        if graph_id not in self._active_graphs:
            if load_if_needed:
                await self._load_graph(graph_id)
            else:
                raise ValueError(f"Graph '{graph_id}' not loaded")

        return self._active_graphs[graph_id]

    async def list_graphs(self) -> list[dict[str, Any]]:
        """List all graphs with metadata."""
        async with self.track_request():
            graphs = await self.repository.list()

            result = []
            for metadata in graphs:
                graph_info = {
                    "graph_id": metadata.graph_id,
                    "graph_type": metadata.graph_type,
                    "description": metadata.description,
                    "created_at": metadata.created_at.isoformat(),
                    "metadata": metadata.metadata,
                    "is_loaded": metadata.graph_id in self._active_graphs,
                }

                # Add runtime statistics if loaded
                if metadata.graph_id in self._active_graphs:
                    graph = self._active_graphs[metadata.graph_id]
                    graph_info.update(
                        {
                            "num_nodes": graph.number_of_nodes(),
                            "num_edges": graph.number_of_edges(),
                            "is_directed": graph.is_directed(),
                            "is_multigraph": graph.is_multigraph(),
                        }
                    )

                result.append(graph_info)

            return result

    async def add_nodes(
        self, graph_id: str, nodes: list[str | dict[str, Any]]
    ) -> dict[str, Any]:
        """Add nodes to a graph."""
        async with self.track_request():
            graph = await self.get_graph(graph_id)

            async with self._get_graph_lock(graph_id):
                added_count = 0

                for node in nodes:
                    if isinstance(node, dict):
                        node_id = node.get("id")
                        attributes = node.get("attributes", {})
                        graph.add_node(node_id, **attributes)
                    else:
                        graph.add_node(node)
                    added_count += 1

                # Invalidate cache
                await self._invalidate_graph_cache(graph_id)

                # Publish event
                await self.event_publisher.publish_nodes_added(graph_id, len(nodes))

                logger.debug(f"Added {added_count} nodes to graph '{graph_id}'")

                return {
                    "success": True,
                    "graph_id": graph_id,
                    "nodes_added": added_count,
                    "total_nodes": graph.number_of_nodes(),
                }

    async def remove_nodes(self, graph_id: str, nodes: list[str]) -> dict[str, Any]:
        """Remove nodes from a graph."""
        async with self.track_request():
            graph = await self.get_graph(graph_id)

            async with self._get_graph_lock(graph_id):
                removed_count = 0
                removed_edges = 0

                for node in nodes:
                    if graph.has_node(node):
                        # Count edges that will be removed
                        removed_edges += graph.degree(node)
                        graph.remove_node(node)
                        removed_count += 1

                # Invalidate cache
                await self._invalidate_graph_cache(graph_id)

                # Publish event
                await self.event_publisher.publish_nodes_removed(
                    graph_id, removed_count
                )

                logger.debug(f"Removed {removed_count} nodes from graph '{graph_id}'")

                return {
                    "success": True,
                    "graph_id": graph_id,
                    "nodes_removed": removed_count,
                    "edges_removed": removed_edges,
                    "total_nodes": graph.number_of_nodes(),
                }

    async def add_edges(
        self, graph_id: str, edges: list[tuple | dict[str, Any]]
    ) -> dict[str, Any]:
        """Add edges to a graph."""
        async with self.track_request():
            graph = await self.get_graph(graph_id)

            async with self._get_graph_lock(graph_id):
                added_count = 0

                for edge in edges:
                    if isinstance(edge, dict):
                        source = edge["source"]
                        target = edge["target"]
                        attributes = edge.get("attributes", {})
                        graph.add_edge(source, target, **attributes)
                    elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
                        if len(edge) == 2:
                            graph.add_edge(edge[0], edge[1])
                        else:
                            graph.add_edge(edge[0], edge[1], **edge[2])
                    else:
                        continue

                    added_count += 1

                # Invalidate cache
                await self._invalidate_graph_cache(graph_id)

                # Publish event
                await self.event_publisher.publish_edges_added(graph_id, added_count)

                logger.debug(f"Added {added_count} edges to graph '{graph_id}'")

                return {
                    "success": True,
                    "graph_id": graph_id,
                    "edges_added": added_count,
                    "total_edges": graph.number_of_edges(),
                }

    async def get_graph_info(self, graph_id: str) -> dict[str, Any]:
        """Get comprehensive information about a graph."""
        async with self.track_request():
            # Try cache first
            cache_key = f"graph:{graph_id}:info"
            if self.cache_service:
                cached = await self.cache_service.get(cache_key)
                if cached:
                    return cached

            graph = await self.get_graph(graph_id)
            metadata = await self.repository.get(graph_id)

            info = {
                "graph_id": graph_id,
                "graph_type": type(graph).__name__,
                "description": metadata.description if metadata else "",
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "is_directed": graph.is_directed(),
                "is_multigraph": graph.is_multigraph(),
                "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
                "created_at": metadata.created_at.isoformat() if metadata else None,
            }

            # Add connectivity information for small graphs
            if graph.number_of_nodes() <= 1000:
                if graph.is_directed():
                    info["is_strongly_connected"] = nx.is_strongly_connected(graph)
                    info["is_weakly_connected"] = nx.is_weakly_connected(graph)
                else:
                    info["is_connected"] = nx.is_connected(graph)
                    if info["is_connected"]:
                        info["diameter"] = nx.diameter(graph)

            # Cache result
            if self.cache_service:
                await self.cache_service.set(cache_key, info, ttl=300)  # 5 minutes

            return info

    async def _load_graph(self, graph_id: str) -> None:
        """Load a graph from repository."""
        if graph_id in self._active_graphs:
            return

        metadata = await self.repository.get(graph_id)
        if not metadata:
            raise ValueError(f"Graph '{graph_id}' not found")

        # Create graph instance
        graph_class = getattr(nx, metadata.graph_type)
        graph = graph_class()

        # Load graph data from repository
        graph_data = await self.repository.get_graph_data(graph_id)
        if graph_data:
            # Reconstruct graph from saved data
            for node, attrs in graph_data.get("nodes", []):
                graph.add_node(node, **attrs)

            for edge_data in graph_data.get("edges", []):
                if len(edge_data) == 3:
                    source, target, attrs = edge_data
                    graph.add_edge(source, target, **attrs)
                else:
                    source, target = edge_data
                    graph.add_edge(source, target)

        self._active_graphs[graph_id] = graph
        self._graph_locks[graph_id] = asyncio.Lock()

        logger.debug(f"Loaded graph '{graph_id}' from repository")

    async def _persist_graph(self, graph_id: str) -> None:
        """Persist a graph to repository."""
        if graph_id not in self._active_graphs:
            return

        graph = self._active_graphs[graph_id]

        # Serialize graph data
        graph_data = {
            "nodes": [(node, attrs) for node, attrs in graph.nodes(data=True)],
            "edges": [(u, v, attrs) for u, v, attrs in graph.edges(data=True)],
        }

        await self.repository.save_graph_data(graph_id, graph_data)
        logger.debug(f"Persisted graph '{graph_id}' to repository")

    def _get_graph_lock(self, graph_id: str) -> asyncio.Lock:
        """Get lock for a graph."""
        if graph_id not in self._graph_locks:
            self._graph_locks[graph_id] = asyncio.Lock()
        return self._graph_locks[graph_id]

    async def _invalidate_graph_cache(self, graph_id: str) -> None:
        """Invalidate cache entries for a graph."""
        if self.cache_service:
            await self.cache_service.delete_pattern(f"graph:{graph_id}:*")

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of active graphs in MB."""
        # Rough estimation: 100 bytes per node + 50 bytes per edge
        total_bytes = 0
        for graph in self._active_graphs.values():
            total_bytes += graph.number_of_nodes() * 100
            total_bytes += graph.number_of_edges() * 50

        return total_bytes / (1024 * 1024)  # Convert to MB
