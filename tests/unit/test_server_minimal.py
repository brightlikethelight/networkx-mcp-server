"""Unit tests for the minimal server module."""

import networkx as nx

from networkx_mcp.server import NetworkXMCPServer, graphs


class TestServerBasics:
    """Test basic server functionality."""

    def test_import(self):
        """Test that server module can be imported."""
        import networkx_mcp.server

        assert networkx_mcp.server is not None

    def test_server_wrapper_class(self):
        """Test NetworkXMCPServer wrapper."""
        server = NetworkXMCPServer()
        assert hasattr(server, "mcp")
        assert hasattr(server, "graphs")
        assert server.graphs is graphs

    def test_graphs_storage(self):
        """Test graphs storage is accessible."""
        # graphs should be a dictionary
        assert isinstance(graphs, dict)

        # We can add graphs to it
        test_graph = nx.Graph()
        test_graph.add_edge(1, 2)
        graphs["test"] = test_graph

        assert "test" in graphs
        assert isinstance(graphs["test"], nx.Graph)

        # Clean up
        if "test" in graphs:
            del graphs["test"]


class TestMCPCompat:
    """Test MCP compatibility layer usage."""

    def test_mcp_instance_exists(self):
        """Test that mcp instance is created."""
        from networkx_mcp.server import mcp

        assert mcp is not None
        assert hasattr(mcp, "tool")  # Should have tool decorator
