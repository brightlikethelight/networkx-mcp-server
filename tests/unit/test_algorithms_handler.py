"""Tests for handlers/algorithms.py - Target: 100% coverage."""


class TestAlgorithmsHandler:
    """Test the algorithms handler module."""

    def test_module_imports(self):
        """Test that the module can be imported."""
        import networkx_mcp.handlers.algorithms as algorithms

        # Module should exist
        assert algorithms is not None

        # Check __all__ exports
        assert hasattr(algorithms, "__all__")
        assert isinstance(algorithms.__all__, list)

        # Expected exports
        expected_exports = [
            "shortest_path",
            "graphs",
            "mcp",
            "AlgorithmsHandler",
            "algorithms_handler",
        ]

        for export in expected_exports:
            assert export in algorithms.__all__

    def test_algorithms_handler_class(self):
        """Test AlgorithmsHandler class."""
        from networkx_mcp.handlers.algorithms import AlgorithmsHandler

        # Class should exist
        assert AlgorithmsHandler is not None
        assert isinstance(AlgorithmsHandler, type)

        # Create instance
        handler = AlgorithmsHandler()
        assert handler is not None

        # Check attributes
        assert hasattr(handler, "graphs")
        assert hasattr(handler, "mcp")
        assert handler.graphs is not None
        assert handler.mcp is not None

        # Check docstring
        assert AlgorithmsHandler.__doc__ is not None
        assert "Handler for graph algorithms" in AlgorithmsHandler.__doc__

    def test_singleton_handler_instance(self):
        """Test the pre-created algorithms_handler instance."""
        from networkx_mcp.handlers.algorithms import (
            AlgorithmsHandler,
            algorithms_handler,
        )

        # Should be an instance of AlgorithmsHandler
        assert isinstance(algorithms_handler, AlgorithmsHandler)

        # Should have expected attributes
        assert hasattr(algorithms_handler, "graphs")
        assert hasattr(algorithms_handler, "mcp")

    def test_imported_functions(self):
        """Test that algorithm functions are imported."""
        from networkx_mcp.handlers.algorithms import graphs, mcp, shortest_path

        # Functions should be imported
        assert shortest_path is not None
        assert callable(shortest_path)

        # Objects should be imported
        assert graphs is not None
        assert mcp is not None

    def test_direct_imports(self):
        """Test direct imports work correctly."""
        from networkx_mcp.handlers.algorithms import (
            AlgorithmsHandler,
            algorithms_handler,
            graphs,
            mcp,
            shortest_path,
        )

        # All imports should succeed
        assert AlgorithmsHandler is not None
        assert algorithms_handler is not None
        assert shortest_path is not None
        assert graphs is not None
        assert mcp is not None

        # Handler instance should use same objects
        assert algorithms_handler.graphs is graphs
        assert algorithms_handler.mcp is mcp
