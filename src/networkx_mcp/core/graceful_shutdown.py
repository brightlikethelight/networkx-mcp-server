#!/usr/bin/env python3
"""Graceful shutdown handling for NetworkX MCP Server.

Implements production-ready shutdown procedures based on testing results.
"""

import asyncio
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Set

from ..config.production import production_config
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class ShutdownState:
    """Track shutdown state and active operations."""

    accepting_connections: bool = True
    active_requests: int = 0
    active_connections: Set[str] = None
    shutdown_started: bool = False
    shutdown_complete: bool = False

    def __post_init__(self):
        if self.active_connections is None:
            self.active_connections = set()


class GracefulShutdownHandler:
    """Handles graceful shutdown for the MCP server."""

    def __init__(self, graph_manager=None, storage_backend=None):
        self.graph_manager = graph_manager
        self.storage_backend = storage_backend
        self.state = ShutdownState()
        self.shutdown_callbacks: list[Callable] = []
        self.shutdown_timeout = production_config.SHUTDOWN_TIMEOUT
        self._lock = threading.Lock()

        # Setup signal handlers
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.shutdown())

        # Handle SIGTERM (Kubernetes sends this)
        signal.signal(signal.SIGTERM, signal_handler)

        # Handle SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, signal_handler)

    def register_shutdown_callback(self, callback: Callable):
        """Register a callback to be called during shutdown."""
        self.shutdown_callbacks.append(callback)

    def track_request_start(self, request_id: str):
        """Track when a request starts."""
        with self._lock:
            self.state.active_requests += 1
            logger.debug(
                f"Request started: {request_id}, active: {self.state.active_requests}"
            )

    def track_request_end(self, request_id: str):
        """Track when a request ends."""
        with self._lock:
            self.state.active_requests = max(0, self.state.active_requests - 1)
            logger.debug(
                f"Request ended: {request_id}, active: {self.state.active_requests}"
            )

    def track_connection_start(self, connection_id: str):
        """Track when a connection starts."""
        with self._lock:
            self.state.active_connections.add(connection_id)
            logger.debug(
                f"Connection started: {connection_id}, total: {len(self.state.active_connections)}"
            )

    def track_connection_end(self, connection_id: str):
        """Track when a connection ends."""
        with self._lock:
            self.state.active_connections.discard(connection_id)
            logger.debug(
                f"Connection ended: {connection_id}, total: {len(self.state.active_connections)}"
            )

    def is_accepting_connections(self) -> bool:
        """Check if server is accepting new connections."""
        return self.state.accepting_connections and not self.state.shutdown_started

    def get_active_request_count(self) -> int:
        """Get current number of active requests."""
        return self.state.active_requests

    def get_active_connection_count(self) -> int:
        """Get current number of active connections."""
        return len(self.state.active_connections)

    async def shutdown(self):
        """Perform graceful shutdown sequence."""
        if self.state.shutdown_started:
            logger.warning("Shutdown already in progress")
            return

        logger.info("Starting graceful shutdown sequence")
        start_time = time.time()

        try:
            # Phase 1: Stop accepting new connections
            logger.info("Phase 1: Stopping new connections")
            self.state.shutdown_started = True
            self.state.accepting_connections = False

            # Phase 2: Wait for active requests to complete
            logger.info(
                f"Phase 2: Waiting for {self.state.active_requests} active requests"
            )
            await self._wait_for_requests_completion()

            # Phase 3: Close existing connections gracefully
            logger.info(
                f"Phase 3: Closing {len(self.state.active_connections)} connections"
            )
            await self._close_connections()

            # Phase 4: Save state if configured
            if production_config.SAVE_STATE_ON_SHUTDOWN:
                logger.info("Phase 4: Saving application state")
                await self._save_application_state()

            # Phase 5: Run shutdown callbacks
            logger.info("Phase 5: Running shutdown callbacks")
            await self._run_shutdown_callbacks()

            # Phase 6: Cleanup resources
            logger.info("Phase 6: Cleaning up resources")
            await self._cleanup_resources()

            self.state.shutdown_complete = True
            elapsed = time.time() - start_time
            logger.info(f"Graceful shutdown completed in {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Error during shutdown after {elapsed:.2f}s: {e}", exc_info=True
            )
            raise

    async def _wait_for_requests_completion(self):
        """Wait for active requests to complete."""
        wait_time = 0
        check_interval = 0.1

        while self.state.active_requests > 0 and wait_time < self.shutdown_timeout:
            logger.debug(
                f"Waiting for {self.state.active_requests} requests, elapsed: {wait_time:.1f}s"
            )
            await asyncio.sleep(check_interval)
            wait_time += check_interval

        if self.state.active_requests > 0:
            logger.warning(
                f"Shutdown timeout reached with {self.state.active_requests} active requests"
            )
        else:
            logger.info("All requests completed successfully")

    async def _close_connections(self):
        """Close active connections gracefully."""
        if not self.state.active_connections:
            return

        # Give connections time to close naturally
        close_timeout = min(5.0, self.shutdown_timeout / 4)
        start_time = time.time()

        while (
            self.state.active_connections and (time.time() - start_time) < close_timeout
        ):
            await asyncio.sleep(0.1)

        remaining = len(self.state.active_connections)
        if remaining > 0:
            logger.warning(f"Force closing {remaining} remaining connections")
            # In a real implementation, you'd force-close connections here
            self.state.active_connections.clear()

    async def _save_application_state(self):
        """Save application state to persistent storage."""
        try:
            if self.storage_backend and hasattr(self.storage_backend, "save_all"):
                logger.info("Saving to storage backend")
                await self.storage_backend.save_all()

            if self.graph_manager and hasattr(self.graph_manager, "save_all_graphs"):
                logger.info("Saving graph state")
                await self.graph_manager.save_all_graphs()

            logger.info("Application state saved successfully")

        except Exception as e:
            logger.error(f"Failed to save application state: {e}", exc_info=True)
            # Don't re-raise - continue with shutdown

    async def _run_shutdown_callbacks(self):
        """Run registered shutdown callbacks."""
        for i, callback in enumerate(self.shutdown_callbacks):
            try:
                logger.debug(
                    f"Running shutdown callback {i + 1}/{len(self.shutdown_callbacks)}"
                )
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Shutdown callback {i + 1} failed: {e}", exc_info=True)
                # Continue with other callbacks

    async def _cleanup_resources(self):
        """Clean up system resources."""
        try:
            # Close storage connections
            if self.storage_backend and hasattr(self.storage_backend, "close"):
                await self.storage_backend.close()

            # Clear graph data to free memory
            if self.graph_manager and hasattr(self.graph_manager, "clear_all"):
                self.graph_manager.clear_all()

            logger.info("Resource cleanup completed")

        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}", exc_info=True)

    def force_shutdown(self):
        """Force immediate shutdown (emergency only)."""
        logger.warning("FORCE SHUTDOWN - This may cause data loss!")
        self.state.shutdown_complete = True
        sys.exit(1)


class ShutdownContextManager:
    """Context manager for tracking operations during shutdown."""

    def __init__(self, shutdown_handler: GracefulShutdownHandler, operation_id: str):
        self.shutdown_handler = shutdown_handler
        self.operation_id = operation_id

    async def __aenter__(self):
        if not self.shutdown_handler.is_accepting_connections():
            raise RuntimeError("Server is shutting down - not accepting new requests")

        self.shutdown_handler.track_request_start(self.operation_id)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.shutdown_handler.track_request_end(self.operation_id)


# Global shutdown handler instance
_shutdown_handler: Optional[GracefulShutdownHandler] = None


def get_shutdown_handler() -> GracefulShutdownHandler:
    """Get or create the global shutdown handler."""
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = GracefulShutdownHandler()
    return _shutdown_handler


def initialize_shutdown_handler(
    graph_manager=None, storage_backend=None
) -> GracefulShutdownHandler:
    """Initialize the shutdown handler with components."""
    global _shutdown_handler
    _shutdown_handler = GracefulShutdownHandler(graph_manager, storage_backend)
    return _shutdown_handler


async def shutdown_context(operation_id: str):
    """Create a shutdown-aware context for operations."""
    handler = get_shutdown_handler()
    return ShutdownContextManager(handler, operation_id)
