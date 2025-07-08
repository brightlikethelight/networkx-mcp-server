"""Storage module for NetworkX MCP Server."""

from .base import (
    GraphNotFoundError,
    StorageBackend,
    StorageError,
    StorageQuotaExceededError,
    Transaction,
    TransactionError,
)
from .factory import StorageFactory, get_default_backend
from .memory_backend import MemoryBackend
from .redis_backend import RedisBackend

__all__ = [
    # Base classes and exceptions
    "StorageBackend",
    "Transaction",
    "StorageError",
    "GraphNotFoundError",
    "StorageQuotaExceededError",
    "TransactionError",
    # Implementations
    "MemoryBackend",
    "RedisBackend",
    # Factory
    "StorageFactory",
    "get_default_backend",
]