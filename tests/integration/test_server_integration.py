"""Integration tests for the MCP server."""

import json

import pytest

from networkx_mcp.server import mcp


class TestMCPServerIntegration:
    """Test MCP server integration."""

    async def test_server_initialization(self):
        """Test server initializes correctly."""
        server = mcp

        # Server should have tools registered
        assert hasattr(server, "tools")
        assert len(server.tools) > 30  # Should have 39+ tools

    def test_create_and_analyze_graph(self, graph_manager):
        """Test creating a graph and running analysis."""
        # Create a graph
        graph_id = graph_manager.create_graph(
            graph_type="undirected", 
            graph_id="test_graph"
        )
        
        assert graph_id == "test_graph"

        # Add nodes
        for i in range(5):
            graph_manager.add_nodes(graph_id, [i])

        # Add edges
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        graph_manager.add_edges(graph_id, edges)

        # Get graph info
        info = graph_manager.get_graph_info(graph_id)
        
        assert info["nodes"] == 5
        assert info["edges"] == 5
        assert info["is_directed"] is False

        # Run basic analysis
        from networkx_mcp.core.algorithms import GraphAlgorithms
        algorithms = GraphAlgorithms()
        
        # Check if graph is cyclic
        graph = graph_manager.get_graph(graph_id)
        assert algorithms.is_cyclic(graph) is True
        
        # Calculate centrality
        centrality = algorithms.centrality_analysis(graph, ["degree"])
        assert "degree" in centrality
        assert len(centrality["degree"]) == 5

    def test_graph_persistence(self, graph_manager):
        """Test graph persistence operations."""
        # Create a graph
        graph_id = "persistence_test"
        graph_manager.create_graph("directed", graph_id)
        
        # Add some data
        graph_manager.add_nodes(graph_id, ["A", "B", "C"])
        graph_manager.add_edges(graph_id, [("A", "B"), ("B", "C")])
        
        # List graphs
        graphs = graph_manager.list_graphs()
        assert graph_id in graphs
        
        # Delete graph
        result = graph_manager.delete_graph(graph_id)
        assert result["success"] is True
        
        # Verify deletion
        graphs = graph_manager.list_graphs()
        assert graph_id not in graphs

    def test_error_handling(self, graph_manager):
        """Test error handling."""
        # Try to get non-existent graph
        with pytest.raises(ValueError):
            graph_manager.get_graph("non_existent")
        
        # Try to add edge to non-existent graph
        with pytest.raises(ValueError):
            graph_manager.add_edges("non_existent", [(1, 2)])

    @pytest.mark.performance
    def test_large_graph_performance(self, graph_manager):
        """Test performance with larger graphs."""
        import time
        
        graph_id = "large_graph"
        graph_manager.create_graph("undirected", graph_id)
        
        # Measure node addition time
        start = time.time()
        nodes = list(range(1000))
        graph_manager.add_nodes(graph_id, nodes)
        node_time = time.time() - start
        
        # Measure edge addition time
        start = time.time()
        edges = [(i, (i + 1) % 1000) for i in range(1000)]
        graph_manager.add_edges(graph_id, edges)
        edge_time = time.time() - start
        
        # Performance assertions
        assert node_time < 1.0  # Should add 1000 nodes in under 1 second
        assert edge_time < 1.0  # Should add 1000 edges in under 1 second
        
        # Cleanup
        graph_manager.delete_graph(graph_id)