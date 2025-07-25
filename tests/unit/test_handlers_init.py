"""Tests for handlers/__init__.py - Target: 100% coverage (4 lines)."""


class TestHandlersInit:
    """Test the handlers package imports."""

    def test_module_imports(self):
        """Test that all expected classes can be imported from handlers."""
        from networkx_mcp import handlers

        # Check that the module exists
        assert handlers is not None

        # Check __all__ exports
        assert hasattr(handlers, "__all__")
        assert isinstance(handlers.__all__, list)

        # Expected exports from __all__
        expected_exports = [
            "GraphOpsHandler",
            "AlgorithmsHandler",
            "graph_ops_handler",
            "algorithms_handler",
        ]

        for export in expected_exports:
            assert export in handlers.__all__

    def test_direct_imports(self):
        """Test direct imports of handler classes."""
        from networkx_mcp.handlers import (
            AlgorithmsHandler,
            GraphOpsHandler,
            algorithms_handler,
            graph_ops_handler,
        )

        # Verify classes are imported
        assert GraphOpsHandler is not None
        assert AlgorithmsHandler is not None

        # Verify instances are imported
        assert graph_ops_handler is not None
        assert algorithms_handler is not None

        # Verify instances are of correct type
        assert isinstance(graph_ops_handler, GraphOpsHandler)
        assert isinstance(algorithms_handler, AlgorithmsHandler)

    def test_handler_instances_have_attributes(self):
        """Test that handler instances have expected attributes."""
        from networkx_mcp.handlers import algorithms_handler, graph_ops_handler

        # GraphOpsHandler should have graphs and mcp attributes
        assert hasattr(graph_ops_handler, "graphs")
        assert hasattr(graph_ops_handler, "mcp")

        # AlgorithmsHandler should also have these attributes
        assert hasattr(algorithms_handler, "graphs")
        assert hasattr(algorithms_handler, "mcp")
