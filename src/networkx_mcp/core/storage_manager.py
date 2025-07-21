"""Storage manager for handling graph persistence."""

import asyncio
import logging
from typing import Optional

from ..storage import StorageBackend, StorageFactory
from .graph_operations import GraphManager

logger = logging.getLogger(__name__, Any)


class StorageManager:
    """Manages graph persistence using storage backends."""

    def __init__(self, graph_manager: GraphManager, user_id: str = "default") -> None:
        self.graph_manager = graph_manager
        self.user_id = user_id
        self.storage_backend: Optional[StorageBackend] = None
        self._sync_task: Optional[asyncio.Task] = None
        self._sync_interval = 30  # seconds

    async def initialize(self, backend_type: Optional[str] = None, **kwargs) -> None:
        """Initialize storage backend."""
        self.storage_backend = await StorageFactory.create_backend(
            backend_type=backend_type, **kwargs
        )

        # Load existing graphs from storage
        await self.load_all_graphs()

        # Start background sync task
        self._sync_task = asyncio.create_task(self._background_sync())

        logger.info(
            f"StorageManager initialized with {type(self.storage_backend).__name__}"
        )

    async def close(self) -> None:
        """Close storage connections and stop sync."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        if self.storage_backend:
            await self.storage_backend.close()

    async def save_graph(self, graph_id: str, immediate: bool = False) -> None:
        """Save a graph to storage."""
        if not self.storage_backend:
            return

        if graph_id not in self.graph_manager.graphs:
            logger.warning(f"Graph {graph_id} not found in GraphManager")
            return

        graph = self.graph_manager.graphs[graph_id]
        metadata = self.graph_manager.metadata.get(graph_id, {})

        try:
            await self.storage_backend.save_graph(
                user_id=self.user_id, graph_id=graph_id, graph=graph, metadata=metadata
            )
            logger.debug(f"Saved graph {graph_id} to storage")
        except Exception as e:
            logger.error(f"Failed to save graph {graph_id}: {e}")

    async def load_graph(self, graph_id: str) -> bool:
        """Load a graph from storage."""
        if not self.storage_backend:
            return False

        try:
            graph = await self.storage_backend.load_graph(
                user_id=self.user_id, graph_id=graph_id
            )

            if graph:
                # Get metadata
                metadata = await self.storage_backend.get_graph_metadata(
                    user_id=self.user_id, graph_id=graph_id
                )

                # Add to GraphManager
                self.graph_manager.graphs[graph_id] = graph
                if metadata:
                    self.graph_manager.metadata[graph_id] = metadata

                logger.info(f"Loaded graph {graph_id} from storage")
                return True

        except Exception as e:
            logger.error(f"Failed to load graph {graph_id}: {e}")

        return False

    async def delete_graph(self, graph_id: str) -> None:
        """Delete a graph from storage."""
        if not self.storage_backend:
            return

        try:
            await self.storage_backend.delete_graph(
                user_id=self.user_id, graph_id=graph_id
            )
            logger.debug(f"Deleted graph {graph_id} from storage")
        except Exception as e:
            logger.error(f"Failed to delete graph {graph_id} from storage: {e}")

    async def load_all_graphs(self) -> None:
        """Load all graphs from storage into GraphManager."""
        if not self.storage_backend:
            return

        try:
            graphs = await self.storage_backend.list_graphs(
                user_id=self.user_id,
                limit=1000,  # Load up to 1000 graphs
            )

            for graph_info in graphs:
                graph_id = graph_info.get("graph_id")
                if graph_id and graph_id not in self.graph_manager.graphs:
                    await self.load_graph(graph_id)

            logger.info(f"Loaded {len(graphs)} graphs from storage")

        except Exception as e:
            logger.error(f"Failed to load graphs from storage: {e}")

    async def sync_all_graphs(self) -> None:
        """Sync all in-memory graphs to storage."""
        if not self.storage_backend:
            return

        for graph_id in list[Any](self.graph_manager.graphs.keys()):
            await self.save_graph(graph_id)

        logger.debug(f"Synced {len(self.graph_manager.graphs)} graphs to storage")

    async def _background_sync(self) -> None:
        """Background task to periodically sync graphs to storage."""
        while True:
            try:
                await asyncio.sleep(self._sync_interval)
                await self.sync_all_graphs()
            except asyncio.CancelledError:
                # Final sync before shutdown
                await self.sync_all_graphs()
                raise
            except Exception as e:
                logger.error(f"Background sync error: {e}")

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        if not self.storage_backend:
            return {"backend": "none", "initialized": False}

        try:
            stats = await self.storage_backend.get_storage_stats(self.user_id)
            health = await self.storage_backend.check_health()

            return {
                **stats,
                **health,
                "sync_interval": self._sync_interval,
                "in_memory_graphs": len(self.graph_manager.graphs),
            }
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e), "backend": type(self.storage_backend).__name__}
