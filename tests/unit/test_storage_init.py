"""Tests for storage/__init__.py - Target: 100% coverage (12 lines)."""


class TestStorageInit:
    """Test the storage package imports."""

    def test_module_imports(self):
        """Test that all expected classes can be imported from storage."""
        from networkx_mcp import storage

        # Check that the module exists
        assert storage is not None

        # Check __all__ exports
        assert hasattr(storage, "__all__")
        assert isinstance(storage.__all__, list)

        # Expected exports from __all__
        expected_exports = [
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

        for export in expected_exports:
            assert export in storage.__all__

    def test_direct_imports(self):
        """Test direct imports of storage classes and functions."""
        from networkx_mcp.storage import (
            GraphNotFoundError,
            # Implementations
            MemoryBackend,
            RedisBackend,
            # Base classes and exceptions
            StorageBackend,
            StorageError,
            # Factory
            StorageFactory,
            StorageQuotaExceededError,
            Transaction,
            TransactionError,
            get_default_backend,
        )

        # Verify all imports are successful
        assert StorageBackend is not None
        assert Transaction is not None
        assert StorageError is not None
        assert GraphNotFoundError is not None
        assert StorageQuotaExceededError is not None
        assert TransactionError is not None
        assert MemoryBackend is not None
        assert RedisBackend is not None
        assert StorageFactory is not None
        assert get_default_backend is not None

        # Verify types
        assert isinstance(StorageBackend, type)
        assert isinstance(Transaction, type)
        assert isinstance(StorageError, type)
        assert isinstance(MemoryBackend, type)
        assert callable(get_default_backend)

    def test_exception_hierarchy(self):
        """Test that storage exceptions are properly structured."""
        from networkx_mcp.storage import (
            GraphNotFoundError,
            StorageError,
            StorageQuotaExceededError,
            TransactionError,
        )

        # All should be exception classes
        assert issubclass(StorageError, Exception)
        assert issubclass(GraphNotFoundError, Exception)
        assert issubclass(StorageQuotaExceededError, Exception)
        assert issubclass(TransactionError, Exception)

    def test_factory_functionality(self):
        """Test basic functionality of storage factory."""
        from networkx_mcp.storage import StorageFactory, get_default_backend

        # StorageFactory should have create_backend method
        assert hasattr(StorageFactory, "create_backend")

        # get_default_backend is an async function
        import inspect

        assert inspect.iscoroutinefunction(get_default_backend)
