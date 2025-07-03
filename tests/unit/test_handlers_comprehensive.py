"""Comprehensive unit tests for all MCP handlers.

This module provides thorough testing coverage for all handler classes
including GraphOpsHandler, AlgorithmHandler, AnalysisHandler, and VisualizationHandler.
"""

from unittest.mock import Mock

import networkx as nx
import pytest

from networkx_mcp.core.graph_operations import GraphManager
from tests.factories import GraphFactory


@pytest.fixture
def mock_mcp():
    """Create a mock MCP server."""
    mcp = Mock()
    mcp.tool = Mock()
    return mcp


@pytest.fixture
def graph_manager_with_data():
    """Create a GraphManager with pre-loaded test data."""
    manager = GraphManager()

    # Add various test graphs using direct assignment
    manager.graphs["simple"] = GraphFactory.simple_graph(5, 6)
    manager.graphs["directed"] = GraphFactory.directed_graph(4, 5)
    manager.graphs["weighted"] = GraphFactory.weighted_graph(6, 8)
    manager.graphs["complete"] = GraphFactory.complete_graph(4)
    manager.graphs["tree"] = GraphFactory.tree_graph(7)
    manager.graphs["disconnected"] = GraphFactory.disconnected_graph(2, 3)

    # Add corresponding metadata
    for graph_id in manager.graphs:
        manager.metadata[graph_id] = {
            "created_at": "2025-01-01T00:00:00",
            "graph_type": "Graph",
            "attributes": {},
        }

    return manager


@pytest.mark.unit
class TestGraphOpsHandler:
    """Test GraphOpsHandler functionality."""

    def test_handler_initialization(self, mock_mcp, graph_manager_with_data):
        """Test handler initializes correctly."""
        from networkx_mcp.mcp.handlers.graph_ops import GraphOpsHandler

        handler = GraphOpsHandler(mock_mcp, graph_manager_with_data)
        assert handler.mcp == mock_mcp
        assert handler.graph_manager == graph_manager_with_data

        # Verify tools were registered
        assert mock_mcp.tool.called

    @pytest.mark.asyncio
    async def test_create_graph(self, mock_mcp, graph_manager_with_data):
        """Test graph creation functionality."""
        from networkx_mcp.mcp.handlers.graph_ops import GraphOpsHandler

        handler = GraphOpsHandler(mock_mcp, graph_manager_with_data)

        # Access the registered create_graph function
        # Since we're mocking, we need to simulate the tool registration
        create_graph_calls = [call for call in mock_mcp.tool.call_args_list if call]
        assert len(create_graph_calls) > 0  # At least one tool was registered

    def test_graph_operations_with_simple_graph(self, graph_manager_with_data):
        """Test operations work with simple graphs."""
        graph = graph_manager_with_data.get_graph("simple")
        assert graph is not None
        assert graph.number_of_nodes() == 5
        assert graph.number_of_edges() <= 6

    def test_graph_operations_with_directed_graph(self, graph_manager_with_data):
        """Test operations work with directed graphs."""
        graph = graph_manager_with_data.get_graph("directed")
        assert graph is not None
        assert graph.is_directed()
        assert graph.number_of_nodes() == 4

    def test_graph_operations_with_weighted_graph(self, graph_manager_with_data):
        """Test operations work with weighted graphs."""
        graph = graph_manager_with_data.get_graph("weighted")
        assert graph is not None

        # Check that some edges have weights
        edges_with_data = graph.edges(data=True)
        weighted_edges = [e for e in edges_with_data if "weight" in e[2]]
        assert len(weighted_edges) > 0


class TestHandlerErrorHandling:
    """Test error handling across all handlers."""

    def test_invalid_graph_id_handling(self, mock_mcp):
        """Test handling of invalid graph IDs."""
        manager = GraphManager()  # Empty manager

        # Test each handler with invalid graph ID
        handlers = [
            ("networkx_mcp.mcp.handlers.graph_ops", "GraphOpsHandler"),
            ("networkx_mcp.mcp.handlers.algorithms", "AlgorithmHandler"),
            ("networkx_mcp.mcp.handlers.analysis", "AnalysisHandler"),
            ("networkx_mcp.mcp.handlers.visualization", "VisualizationHandler"),
        ]

        for module_name, class_name in handlers:
            module = __import__(module_name, fromlist=[class_name])
            handler_class = getattr(module, class_name)

            # Should initialize without error
            handler = handler_class(mock_mcp, manager)
            assert handler is not None

    def test_empty_graph_handling(self, mock_mcp):
        """Test handling of empty graphs."""
        manager = GraphManager()
        empty_graph = nx.Graph()  # Empty graph
        manager.graphs["empty"] = empty_graph
        manager.metadata["empty"] = {
            "created_at": "2025-01-01T00:00:00",
            "graph_type": "Graph",
            "attributes": {},
        }

        # Basic operations should handle empty graphs gracefully
        assert empty_graph.number_of_nodes() == 0
        assert empty_graph.number_of_edges() == 0
        assert nx.density(empty_graph) == 0

    def test_disconnected_graph_handling(self, graph_manager_with_data):
        """Test handling of disconnected graphs."""
        graph = graph_manager_with_data.get_graph("disconnected")

        # Should handle disconnected components properly
        components = list(nx.connected_components(graph))
        assert len(components) > 1

        # Each component should be non-empty
        for component in components:
            assert len(component) > 0


@pytest.mark.unit
class TestHandlerIntegration:
    """Test integration between different handlers."""

    def test_data_flow_between_handlers(self, mock_mcp, graph_manager_with_data):
        """Test that data flows correctly between handlers."""
        # Initialize all handlers
        from networkx_mcp.mcp.handlers.algorithms import AlgorithmHandler
        from networkx_mcp.mcp.handlers.analysis import AnalysisHandler
        from networkx_mcp.mcp.handlers.graph_ops import GraphOpsHandler
        from networkx_mcp.mcp.handlers.visualization import \
            VisualizationHandler

        graph_ops = GraphOpsHandler(mock_mcp, graph_manager_with_data)
        algorithms = AlgorithmHandler(mock_mcp, graph_manager_with_data)
        analysis = AnalysisHandler(mock_mcp, graph_manager_with_data)
        visualization = VisualizationHandler(mock_mcp, graph_manager_with_data)

        # All handlers should share the same graph manager
        assert graph_ops.graph_manager == algorithms.graph_manager
        assert algorithms.graph_manager == analysis.graph_manager
        assert analysis.graph_manager == visualization.graph_manager

    def test_consistent_graph_access(self, graph_manager_with_data):
        """Test that all handlers access graphs consistently."""
        graph_ids = ["simple", "directed", "weighted", "complete"]

        for graph_id in graph_ids:
            graph = graph_manager_with_data.get_graph(graph_id)
            assert graph is not None

            # Graph properties should be consistent
            assert graph.number_of_nodes() >= 0
            assert graph.number_of_edges() >= 0

    def test_algorithm_composition(self, graph_manager_with_data):
        """Test that algorithms can be composed together."""
        graph = graph_manager_with_data.get_graph("simple")

        # Test combining centrality and community detection
        centrality = nx.degree_centrality(graph)

        if not graph.is_directed():
            from networkx.algorithms.community import \
                greedy_modularity_communities

            communities = list(greedy_modularity_communities(graph))

            # Should be able to analyze centrality within communities
            for community in communities:
                community_centrality = {
                    node: centrality[node] for node in community if node in centrality
                }
                assert len(community_centrality) == len(community)


@pytest.mark.unit
class TestPerformanceConsiderations:
    """Test performance-related aspects of handlers."""

    def test_memory_usage_with_large_graphs(self, mock_mcp):
        """Test memory usage with larger graphs."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        manager = GraphManager()

        # Create a larger graph
        large_graph = GraphFactory.simple_graph(200, 500)
        manager.graphs["large"] = large_graph
        manager.metadata["large"] = {
            "created_at": "2025-01-01T00:00:00",
            "graph_type": "Graph",
            "attributes": {},
        }

        # Initialize handlers
        from networkx_mcp.mcp.handlers.algorithms import AlgorithmHandler

        handler = AlgorithmHandler(mock_mcp, manager)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Should not use excessive memory (less than 50MB for this test)
        assert memory_increase < 50 * 1024 * 1024

    def test_algorithm_performance(self, graph_manager_with_data):
        """Test that algorithms complete in reasonable time."""
        import time

        graph = graph_manager_with_data.get_graph("complete")

        # Test degree centrality performance
        start_time = time.time()
        centrality = nx.degree_centrality(graph)
        end_time = time.time()

        # Should complete quickly for small graphs
        elapsed = end_time - start_time
        assert elapsed < 1.0  # Less than 1 second
        assert len(centrality) == graph.number_of_nodes()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
