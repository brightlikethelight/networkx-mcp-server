"""Storage backend factory for selecting appropriate storage implementation."""

import logging
import os
from typing import Optional

from .base import StorageBackend
from .memory_backend import MemoryBackend
from .redis_backend import RedisBackend

logger = logging.getLogger(__name__)


class StorageFactory:
    """Factory for creating storage backends based on configuration."""

    @staticmethod
    async def create_backend(
        backend_type: Optional[str] = None, **kwargs
    ) -> StorageBackend:
        """
        Create a storage backend based on configuration.

        Args:
            backend_type: Explicit backend type ('redis', 'memory', or None for auto)
            **kwargs: Additional arguments passed to backend constructor

        Returns:
            Initialized storage backend

        The selection logic:
        1. If backend_type is specified, use that
        2. If REDIS_URL environment variable exists, use Redis
        3. Otherwise, use in-memory storage
        """
        # Determine backend type
        if backend_type is None:
            # Auto-detect based on environment
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                backend_type = "redis"
                logger.info(
                    "Auto-detected Redis backend from REDIS_URL environment variable"
                )
            else:
                backend_type = "memory"
                logger.info("No REDIS_URL found, using in-memory backend")

        # Create backend instance
        if backend_type == "redis":
            redis_url = kwargs.pop(
                "redis_url", os.getenv("REDIS_URL", "redis://localhost:6379")
            )
            max_graph_size_mb = kwargs.pop(
                "max_graph_size_mb", int(os.getenv("MAX_GRAPH_SIZE_MB", "100"))
            )
            compression_level = kwargs.pop(
                "compression_level", int(os.getenv("COMPRESSION_LEVEL", "6"))
            )

            backend = RedisBackend(
                redis_url=redis_url,
                max_graph_size_mb=max_graph_size_mb,
                compression_level=compression_level,
                **kwargs,
            )
            logger.info(f"Created Redis backend: {redis_url}")

        elif backend_type == "memory":
            max_graph_size_mb = kwargs.pop(
                "max_graph_size_mb", int(os.getenv("MAX_GRAPH_SIZE_MB", "100"))
            )

            backend = MemoryBackend(max_graph_size_mb=max_graph_size_mb, **kwargs)
            logger.warning("Using in-memory backend - data will be lost on restart!")

        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

        # Initialize the backend
        try:
            await backend.initialize()
            logger.info(f"Successfully initialized {backend_type} backend")
        except Exception as e:
            logger.error(f"Failed to initialize {backend_type} backend: {e}")
            if backend_type == "redis":
                # Fallback to memory if Redis fails
                logger.warning("Falling back to in-memory backend")
                backend = MemoryBackend(
                    max_graph_size_mb=kwargs.get("max_graph_size_mb", 100)
                )
                await backend.initialize()
            else:
                raise

        # Check health
        health = await backend.check_health()
        logger.info(f"Storage backend health: {health}")

        return backend


async def get_default_backend() -> StorageBackend:
    """Get the default storage backend based on environment configuration."""
    return await StorageFactory.create_backend()


# Global backend instance for singleton pattern
_storage_backend: Optional[StorageBackend] = None


def get_storage_backend() -> StorageBackend:
    """
    Get the global storage backend instance.

    Returns:
        Storage backend instance

    Note:
        This function creates a synchronous memory backend if no async backend
        has been initialized. For production use, call get_default_backend()
        during application startup.
    """
    global _storage_backend

    # Return existing backend if available
    if _storage_backend is not None:
        return _storage_backend

    # Create a simple memory backend for immediate use
    # This is mainly for health checks and testing
    from .memory_backend import MemoryBackend

    _storage_backend = MemoryBackend()

    # Initialize synchronously (memory backend doesn't need async init)
    try:
        import asyncio

        # Try to run initialization in current event loop
        try:
            asyncio.get_running_loop()
            # If we're in an event loop, we can't await here
            # Just return the uninitialized backend
            logger.warning(
                "Returning uninitialized memory backend - call initialize() manually"
            )
        except RuntimeError:
            # No event loop running, we can create one
            asyncio.run(_storage_backend.initialize())
    except Exception as e:
        logger.error(f"Failed to initialize storage backend: {e}")

    return _storage_backend


def set_storage_backend(backend: StorageBackend) -> None:
    """Set the global storage backend instance."""
    global _storage_backend
    _storage_backend = backend
    logger.info(f"Set global storage backend: {backend.__class__.__name__}")
