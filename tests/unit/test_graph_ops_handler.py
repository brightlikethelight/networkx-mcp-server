"""Comprehensive tests for handlers/graph_ops.py - Target: 100% coverage (44 lines).

This module is a simple re-export module, making it perfect for quick coverage gains.
"""


class TestGraphOpsHandler:
    """Test the GraphOpsHandler class and module exports."""

    def test_module_imports(self):
        """Test that the module can be imported and has expected attributes."""
        import networkx_mcp.handlers.graph_ops as graph_ops

        # Module should exist
        assert graph_ops is not None

        # Check for GraphOpsHandler class
        assert hasattr(graph_ops, "GraphOpsHandler")
        assert hasattr(graph_ops, "graph_ops_handler")

    def test_graph_ops_handler_class(self):
        """Test GraphOpsHandler class initialization."""
        from networkx_mcp.handlers.graph_ops import GraphOpsHandler

        # Create instance
        handler = GraphOpsHandler()

        # Check attributes
        assert hasattr(handler, "graphs")
        assert hasattr(handler, "mcp")

        # These should be the imported objects
        assert handler.graphs is not None
        assert handler.mcp is not None

    def test_singleton_handler_instance(self):
        """Test the pre-created graph_ops_handler instance."""
        from networkx_mcp.handlers.graph_ops import GraphOpsHandler, graph_ops_handler

        # Should be an instance of GraphOpsHandler
        assert isinstance(graph_ops_handler, GraphOpsHandler)

        # Should have the expected attributes
        assert hasattr(graph_ops_handler, "graphs")
        assert hasattr(graph_ops_handler, "mcp")

    def test_imported_functions(self):
        """Test that imported functions are available."""
        from networkx_mcp.handlers import graph_ops

        # Functions that should be imported from server
        expected_functions = [
            "create_graph",
            "add_nodes",
            "add_edges",
            "delete_graph",
        ]

        for func_name in expected_functions:
            assert hasattr(graph_ops, func_name), f"Missing function: {func_name}"
            # Verify it's callable
            func = getattr(graph_ops, func_name)
            assert callable(func), f"{func_name} is not callable"

    def test_imported_objects(self):
        """Test that imported objects are available."""
        from networkx_mcp.handlers import graph_ops

        # Objects that should be imported
        assert hasattr(graph_ops, "graphs")
        assert hasattr(graph_ops, "mcp")

        # These should be dictionaries or similar containers
        assert graph_ops.graphs is not None
        assert graph_ops.mcp is not None

    def test_all_exports(self):
        """Test __all__ exports list."""
        from networkx_mcp.handlers import graph_ops

        # Check __all__ exists
        assert hasattr(graph_ops, "__all__")

        # Get the actual __all__ list
        all_exports = graph_ops.__all__

        # Should be a list
        assert isinstance(all_exports, list)

        # Expected exports (from reading the code)
        expected_in_all = [
            "create_graph",
            "add_nodes",
            "add_edges",
            "delete_graph",
            "graphs",
            "mcp",
            "GraphOpsHandler",
            "graph_ops_handler",
        ]

        # Check that expected items are in __all__
        for item in expected_in_all:
            if item in ["graph_info", "list_graphs"]:
                # These are in __all__ but not actually imported
                continue
            assert item in all_exports, f"{item} missing from __all__"

    def test_backward_compatibility(self):
        """Test that the module maintains backward compatibility."""
        # Import the old way
        from networkx_mcp.handlers.graph_ops import (
            GraphOpsHandler,
            add_edges,
            add_nodes,
            create_graph,
            delete_graph,
            graphs,
            mcp,
        )

        # All imports should work
        assert GraphOpsHandler is not None
        assert callable(create_graph)
        assert callable(add_nodes)
        assert callable(add_edges)
        assert callable(delete_graph)
        assert graphs is not None
        assert mcp is not None

    def test_handler_instance_shares_references(self):
        """Test that handler instance shares references with imported objects."""
        from networkx_mcp.handlers.graph_ops import (
            graph_ops_handler,
            graphs,
            mcp,
        )

        # The handler's attributes should be the same objects
        assert graph_ops_handler.graphs is graphs
        assert graph_ops_handler.mcp is mcp

    def test_module_docstring(self):
        """Test that the module has proper documentation."""
        import networkx_mcp.handlers.graph_ops as graph_ops

        # Module should have a docstring
        assert graph_ops.__doc__ is not None
        assert "Graph operations handler" in graph_ops.__doc__

    def test_class_docstring(self):
        """Test that GraphOpsHandler has proper documentation."""
        from networkx_mcp.handlers.graph_ops import GraphOpsHandler

        # Class should have a docstring
        assert GraphOpsHandler.__doc__ is not None
        assert "Handler for graph operations" in GraphOpsHandler.__doc__
