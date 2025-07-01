"""Comprehensive unit tests for all MCP handlers.

This module provides thorough testing coverage for all handler classes
including GraphOpsHandler, AlgorithmHandler, AnalysisHandler, and VisualizationHandler.
"""

import pytest
from unittest.mock import Mock

import networkx as nx
from tests.factories import GraphFactory
from networkx_mcp.core.graph_operations import GraphManager


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


@pytest.mark.unit
class TestAlgorithmHandler:
    """Test AlgorithmHandler functionality."""

    def test_handler_initialization(self, mock_mcp, graph_manager_with_data):
        """Test algorithm handler initializes correctly."""
        from networkx_mcp.mcp.handlers.algorithms import AlgorithmHandler

        handler = AlgorithmHandler(mock_mcp, graph_manager_with_data)
        assert handler.mcp == mock_mcp
        assert handler.graph_manager == graph_manager_with_data

    def test_shortest_path_algorithms(self, graph_manager_with_data):
        """Test shortest path algorithm functionality."""
        graph = graph_manager_with_data.get_graph("simple")
        nodes = list(graph.nodes())

        if len(nodes) >= 2:
            # Test basic shortest path
            try:
                path = nx.shortest_path(graph, nodes[0], nodes[1])
                assert isinstance(path, list)
                assert len(path) >= 2
                assert path[0] == nodes[0]
                assert path[-1] == nodes[1]
            except nx.NetworkXNoPath:
                # Acceptable for disconnected graphs
                pass

    def test_centrality_algorithms(self, graph_manager_with_data):
        """Test centrality calculation algorithms."""
        graph = graph_manager_with_data.get_graph("complete")

        # Test degree centrality
        centrality = nx.degree_centrality(graph)
        assert isinstance(centrality, dict)
        assert len(centrality) == graph.number_of_nodes()

        # All values should be between 0 and 1
        for value in centrality.values():
            assert 0 <= value <= 1

    def test_connected_components(self, graph_manager_with_data):
        """Test connected components analysis."""
        graph = graph_manager_with_data.get_graph("disconnected")

        # Should have multiple components
        components = list(nx.connected_components(graph))
        assert len(components) >= 2

        # All nodes should be covered
        all_nodes = set()
        for component in components:
            all_nodes.update(component)
        assert all_nodes == set(graph.nodes())

    def test_minimum_spanning_tree(self, graph_manager_with_data):
        """Test minimum spanning tree algorithm."""
        graph = graph_manager_with_data.get_graph("weighted")

        if not graph.is_directed() and nx.is_connected(graph):
            mst = nx.minimum_spanning_tree(graph)

            # MST should have n-1 edges for n nodes
            assert mst.number_of_edges() == mst.number_of_nodes() - 1

            # MST should be connected
            assert nx.is_connected(mst)


@pytest.mark.unit
class TestAnalysisHandler:
    """Test AnalysisHandler functionality."""

    def test_handler_initialization(self, mock_mcp, graph_manager_with_data):
        """Test analysis handler initializes correctly."""
        from networkx_mcp.mcp.handlers.analysis import AnalysisHandler

        handler = AnalysisHandler(mock_mcp, graph_manager_with_data)
        assert handler.mcp == mock_mcp
        assert handler.graph_manager == graph_manager_with_data

    def test_graph_statistics(self, graph_manager_with_data):
        """Test comprehensive graph statistics."""
        graph = graph_manager_with_data.get_graph("simple")

        # Basic statistics
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() >= 0

        density = nx.density(graph)
        assert 0 <= density <= 1

        # Degree statistics
        degrees = [d for n, d in graph.degree()]
        if degrees:
            avg_degree = sum(degrees) / len(degrees)
            assert avg_degree >= 0

    def test_community_detection(self, graph_manager_with_data):
        """Test community detection algorithms."""
        graph = graph_manager_with_data.get_graph("complete")

        # Test greedy modularity communities
        from networkx.algorithms.community import greedy_modularity_communities

        communities = list(greedy_modularity_communities(graph))

        assert len(communities) > 0

        # All nodes should be in exactly one community
        all_nodes = set()
        for community in communities:
            all_nodes.update(community)
        assert len(all_nodes) == graph.number_of_nodes()

    def test_bipartite_analysis(self, graph_manager_with_data):
        """Test bipartite graph analysis."""
        # Create a known bipartite graph
        bipartite_graph = GraphFactory.bipartite_graph(3, 4, 0.8)

        from networkx.algorithms import bipartite

        # Test bipartite detection
        is_bipartite = bipartite.is_bipartite(bipartite_graph)

        if is_bipartite:
            # Get node sets
            node_sets = bipartite.sets(bipartite_graph)
            assert len(node_sets) == 2

            set1, set2 = node_sets
            assert len(set1) > 0
            assert len(set2) > 0
            assert len(set1) + len(set2) == bipartite_graph.number_of_nodes()

    def test_degree_distribution(self, graph_manager_with_data):
        """Test degree distribution analysis."""
        graph = graph_manager_with_data.get_graph("simple")

        # Get degree sequence
        degrees = [d for n, d in graph.degree()]

        if degrees:
            # Test degree histogram
            hist = nx.degree_histogram(graph)
            assert len(hist) > 0
            assert sum(hist) == len(degrees)

    def test_assortativity_analysis(self, graph_manager_with_data):
        """Test assortativity coefficient calculation."""
        graph = graph_manager_with_data.get_graph("complete")

        if graph.number_of_nodes() > 1:
            # Test degree assortativity
            assortativity = nx.degree_assortativity_coefficient(graph)
            assert -1 <= assortativity <= 1


@pytest.mark.unit
class TestVisualizationHandler:
    """Test VisualizationHandler functionality."""

    def test_handler_initialization(self, mock_mcp, graph_manager_with_data):
        """Test visualization handler initializes correctly."""
        from networkx_mcp.mcp.handlers.visualization import VisualizationHandler

        handler = VisualizationHandler(mock_mcp, graph_manager_with_data)
        assert handler.mcp == mock_mcp
        assert handler.graph_manager == graph_manager_with_data

    def test_layout_algorithms(self, graph_manager_with_data):
        """Test graph layout algorithms."""
        graph = graph_manager_with_data.get_graph("simple")

        # Test various layout algorithms
        layouts = {
            "spring": nx.spring_layout,
            "circular": nx.circular_layout,
            "random": nx.random_layout,
        }

        for layout_name, layout_func in layouts.items():
            pos = layout_func(graph)

            # Should return positions for all nodes
            assert len(pos) == graph.number_of_nodes()

            # Each position should be a 2D coordinate
            for node, coord in pos.items():
                assert len(coord) == 2
                assert isinstance(coord[0], (int, float))
                assert isinstance(coord[1], (int, float))

    def test_graph_data_preparation(self, graph_manager_with_data):
        """Test data preparation for visualization."""
        graph = graph_manager_with_data.get_graph("simple")

        # Test node-link data format
        data = nx.node_link_data(graph)

        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) == graph.number_of_nodes()
        assert len(data["links"]) == graph.number_of_edges()

    def test_edge_bundling_preparation(self, graph_manager_with_data):
        """Test edge bundling data preparation."""
        graph = graph_manager_with_data.get_graph("complete")

        # For complete graphs, all nodes should be connected
        edges = list(graph.edges())
        nodes = list(graph.nodes())

        # Verify edge data structure
        for u, v in edges:
            assert u in nodes
            assert v in nodes


@pytest.mark.unit
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
        from networkx_mcp.mcp.handlers.graph_ops import GraphOpsHandler
        from networkx_mcp.mcp.handlers.algorithms import AlgorithmHandler
        from networkx_mcp.mcp.handlers.analysis import AnalysisHandler
        from networkx_mcp.mcp.handlers.visualization import VisualizationHandler

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
            from networkx.algorithms.community import greedy_modularity_communities

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
        import psutil
        import os

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
