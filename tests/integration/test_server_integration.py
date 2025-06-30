"""Integration tests for the MCP server."""

import pytest
from networkx_mcp.server import mcp


class TestMCPServerIntegration:
    """Test MCP server integration."""

    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test server initializes correctly."""
        server = mcp

        # Server should have tools registered
        assert hasattr(server, "tools")
        assert len(server.tools) > 30  # Should have 39+ tools

    def test_create_and_analyze_graph(self, graph_manager):
        """Test creating a graph and running analysis."""
        # Create a graph
        result = graph_manager.create_graph(graph_id="test_graph", graph_type="Graph")
        graph_id = result["graph_id"]

        assert graph_id == "test_graph"

        # Add nodes
        nodes = list(range(5))
        graph_manager.add_nodes_from(graph_id, nodes)

        # Add edges
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        graph_manager.add_edges_from(graph_id, edges)

        # Get graph info
        info = graph_manager.get_graph_info(graph_id)

        assert info["num_nodes"] == 5
        assert info["num_edges"] == 5
        assert info["is_directed"] is False

        # Run basic analysis
        from networkx_mcp.core.algorithms import GraphAlgorithms

        algorithms = GraphAlgorithms()

        # Check if graph has cycles
        graph = graph_manager.get_graph(graph_id)
        cycle_info = algorithms.cycles_detection(graph)
        assert cycle_info["has_cycle"] is True

        # Calculate centrality
        centrality = algorithms.centrality_measures(graph, ["degree"])
        assert "degree_centrality" in centrality
        assert len(centrality["degree_centrality"]) == 5

    def test_graph_persistence(self, graph_manager):
        """Test graph persistence operations."""
        # Create a graph
        graph_id = "persistence_test"
        graph_manager.create_graph(graph_id, "DiGraph")

        # Add some data
        graph_manager.add_nodes_from(graph_id, ["A", "B", "C"])
        graph_manager.add_edges_from(graph_id, [("A", "B"), ("B", "C")])

        # List graphs
        graphs = graph_manager.list_graphs()
        graph_ids = [g["graph_id"] for g in graphs]
        assert graph_id in graph_ids

        # Delete graph
        result = graph_manager.delete_graph(graph_id)
        assert result["deleted"] is True

        # Verify deletion
        graphs = graph_manager.list_graphs()
        graph_ids = [g["graph_id"] for g in graphs]
        assert graph_id not in graph_ids

    def test_error_handling(self, graph_manager):
        """Test error handling."""
        # Try to get non-existent graph
        with pytest.raises(KeyError):
            graph_manager.get_graph("non_existent")

        # Try to add edge to non-existent graph
        with pytest.raises(KeyError):
            graph_manager.add_edges_from("non_existent", [(1, 2)])

    @pytest.mark.performance
    def test_large_graph_performance(self, graph_manager):
        """Test performance with larger graphs."""
        import time

        graph_id = "large_graph"
        graph_manager.create_graph(graph_id, "Graph")

        # Measure node addition time
        start = time.time()
        nodes = list(range(1000))
        graph_manager.add_nodes_from(graph_id, nodes)
        node_time = time.time() - start

        # Measure edge addition time
        start = time.time()
        edges = [(i, (i + 1) % 1000) for i in range(1000)]
        graph_manager.add_edges_from(graph_id, edges)
        edge_time = time.time() - start

        # Performance assertions
        assert node_time < 1.0  # Should add 1000 nodes in under 1 second
        assert edge_time < 1.0  # Should add 1000 edges in under 1 second

        # Cleanup
        graph_manager.delete_graph(graph_id)
