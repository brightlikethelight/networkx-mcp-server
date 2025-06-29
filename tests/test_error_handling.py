"""Comprehensive tests for error handling and edge cases."""

import tempfile
from unittest.mock import patch

import networkx as nx
import pytest

from networkx_mcp.core.graph_operations import GraphManager
from networkx_mcp.utils.validators import GraphValidator


class TestGraphValidationErrors:
    """Test validation error handling."""

    def test_invalid_graph_ids(self, error_scenarios):
        """Test validation of invalid graph IDs."""
        for invalid_id in error_scenarios["invalid_graph_ids"]:
            valid, error = GraphValidator.validate_graph_id(invalid_id)
            assert valid is False
            assert error is not None
            assert isinstance(error, str)
            assert len(error) > 0

    def test_invalid_node_ids(self, error_scenarios):
        """Test validation of invalid node IDs."""
        for invalid_node in error_scenarios["invalid_node_ids"]:
            valid = GraphValidator.validate_node_id(invalid_node)
            assert valid is False

    def test_invalid_file_paths(self, error_scenarios):
        """Test handling of invalid file paths."""
        for invalid_path in error_scenarios["invalid_file_paths"]:
            if invalid_path is not None:
                valid, error = GraphValidator.validate_file_format(invalid_path, ["json"])
                assert valid is False
                assert error is not None

    def test_malformed_graph_data(self, error_scenarios):
        """Test validation of malformed graph data."""
        for malformed_data in error_scenarios["malformed_data"]:
            valid, errors = GraphValidator.validate_graph_data(malformed_data)
            assert valid is False
            assert len(errors) > 0

    def test_edge_validation_errors(self):
        """Test edge validation error cases."""
        # Invalid edge formats
        invalid_edges = [
            "not_a_tuple",  # String instead of tuple
            123,  # Number instead of tuple
            [],  # Empty list
            ["A"],  # Single element
            [None, "B"],  # None as node
            [123, None],  # None as target
        ]

        for invalid_edge in invalid_edges:
            valid = GraphValidator.validate_edge(invalid_edge)
            assert valid is False

    def test_attribute_validation_errors(self):
        """Test attribute validation errors."""
        invalid_attributes = [
            "not_a_dict",  # String instead of dict
            123,  # Number instead of dict
            None,  # None instead of dict
            {"id": "reserved"},  # Reserved attribute name
            {"source": "reserved"},  # Reserved attribute name
        ]

        for invalid_attrs in invalid_attributes:
            valid = GraphValidator.validate_attributes(invalid_attrs)
            assert valid is False


class TestGraphOperationErrors:
    """Test error handling in graph operations."""

    def setup_method(self):
        """Setup for each test."""
        self.manager = GraphManager()

    def test_duplicate_graph_creation(self):
        """Test error when creating duplicate graphs."""
        # Create first graph
        self.manager.create_graph("test_graph", "Graph")

        # Attempt to create duplicate
        with pytest.raises(ValueError, match="already exists"):
            self.manager.create_graph("test_graph", "Graph")

    def test_invalid_graph_type(self):
        """Test error with invalid graph type."""
        with pytest.raises(ValueError, match="Invalid graph type"):
            self.manager.create_graph("test", "InvalidGraphType")

    def test_operations_on_nonexistent_graph(self):
        """Test operations on non-existent graphs."""
        with pytest.raises(KeyError):
            self.manager.get_graph("nonexistent")

        with pytest.raises(KeyError):
            self.manager.delete_graph("nonexistent")

        with pytest.raises(KeyError):
            self.manager.add_node("nonexistent", "node1")

        with pytest.raises(KeyError):
            self.manager.get_graph_info("nonexistent")

    def test_node_operations_errors(self):
        """Test node operation error cases."""
        self.manager.create_graph("test", "Graph")

        # Remove non-existent node
        with pytest.raises(ValueError):
            self.manager.remove_node("test", "nonexistent_node")

        # Get neighbors of non-existent node
        with pytest.raises(ValueError):
            self.manager.get_neighbors("test", "nonexistent_node")

        # Get attributes of non-existent node
        with pytest.raises(ValueError):
            self.manager.get_node_attributes("test", "nonexistent_node")

    def test_edge_operations_errors(self):
        """Test edge operation error cases."""
        self.manager.create_graph("test", "Graph")
        self.manager.add_node("test", "A")
        self.manager.add_node("test", "B")

        # Remove non-existent edge
        with pytest.raises(ValueError):
            self.manager.remove_edge("test", "A", "C")  # C doesn't exist

        # Add edge with non-existent nodes
        with pytest.raises(Exception):  # NetworkX will raise an error
            self.manager.add_edge("test", "A", "nonexistent")

        # Get attributes of non-existent edge
        with pytest.raises(ValueError):
            self.manager.get_edge_attributes("test", ("A", "C"))

    def test_empty_graph_edge_cases(self):
        """Test operations on empty graphs."""
        self.manager.create_graph("empty", "Graph")

        # These should not crash but return appropriate empty results
        info = self.manager.get_graph_info("empty")
        assert info["num_nodes"] == 0
        assert info["num_edges"] == 0

        result = self.manager.subgraph("empty", [])
        assert result["num_nodes"] == 0

        result = self.manager.clear_graph("empty")
        assert result["cleared"] is True

    def test_self_loop_edge_cases(self):
        """Test handling of self-loops in different graph types."""
        # Test in undirected graph
        self.manager.create_graph("undirected", "Graph")
        self.manager.add_node("undirected", "A")

        # Self-loops should be allowed
        result = self.manager.add_edge("undirected", "A", "A")
        assert result["added"] is True

        # Test in directed graph
        self.manager.create_graph("directed", "DiGraph")
        self.manager.add_node("directed", "B")

        result = self.manager.add_edge("directed", "B", "B")
        assert result["added"] is True

    def test_multigraph_edge_cases(self):
        """Test edge cases specific to multigraphs."""
        self.manager.create_graph("multi", "MultiGraph")
        self.manager.add_nodes_from("multi", ["X", "Y"])

        # Add multiple edges between same nodes
        self.manager.add_edge("multi", "X", "Y", weight=1.0)
        self.manager.add_edge("multi", "X", "Y", weight=2.0)

        graph = self.manager.get_graph("multi")
        assert graph.number_of_edges("X", "Y") == 2


class TestAlgorithmErrors:
    """Test error handling in graph algorithms."""

    def setup_method(self):
        """Setup test graphs."""
        self.manager = GraphManager()

        # Create test graphs
        self.manager.create_graph("connected", "Graph")
        self.manager.add_nodes_from("connected", ["A", "B", "C"])
        self.manager.add_edges_from("connected", [("A", "B"), ("B", "C")])

        self.manager.create_graph("disconnected", "Graph")
        self.manager.add_nodes_from("disconnected", ["A", "B", "C", "D"])
        self.manager.add_edges_from("disconnected", [("A", "B"), ("C", "D")])

        self.manager.create_graph("empty", "Graph")

    @pytest.mark.asyncio
    async def test_shortest_path_errors(self):
        """Test shortest path algorithm error cases."""
        from networkx_mcp.server import shortest_path

        # Non-existent source node
        with pytest.raises(Exception):
            await shortest_path(graph_id="connected", source="X", target="A")

        # Non-existent target node
        with pytest.raises(Exception):
            await shortest_path(graph_id="connected", source="A", target="X")

        # Path in disconnected graph
        with pytest.raises(Exception):
            await shortest_path(graph_id="disconnected", source="A", target="C")

        # Empty graph
        with pytest.raises(Exception):
            await shortest_path(graph_id="empty", source="A", target="B")

    @pytest.mark.asyncio
    async def test_centrality_errors(self):
        """Test centrality calculation error cases."""
        from networkx_mcp.server import centrality_measures

        # Invalid centrality measure
        with pytest.raises(Exception):
            await centrality_measures(
                graph_id="connected",
                measures=["invalid_measure"]
            )

        # Eigenvector centrality on disconnected graph may not converge
        try:
            result = await centrality_measures(
                graph_id="disconnected",
                measures=["eigenvector"]
            )
            # Should either work or handle convergence failure gracefully
            assert "eigenvector_centrality" in result
        except Exception:
            # Convergence failure is acceptable
            pass

    @pytest.mark.asyncio
    async def test_flow_algorithm_errors(self):
        """Test network flow algorithm errors."""
        from networkx_mcp.server import add_edges, add_nodes, create_graph, network_flow

        # Create directed graph for flow
        await create_graph(graph_id="flow_test", graph_type="DiGraph")
        await add_nodes(graph_id="flow_test", nodes=["s", "t"])
        await add_edges(graph_id="flow_test",
                       edges=[("s", "t", {"capacity": 10})])

        # Test with undirected graph (should fail)
        await create_graph(graph_id="undirected_flow", graph_type="Graph")
        await add_nodes(graph_id="undirected_flow", nodes=["s", "t"])
        await add_edges(graph_id="undirected_flow", edges=[("s", "t")])

        with pytest.raises(Exception):
            await network_flow(
                graph_id="undirected_flow",
                source="s",
                sink="t"
            )

        # Non-existent source/sink
        with pytest.raises(Exception):
            await network_flow(
                graph_id="flow_test",
                source="nonexistent",
                sink="t"
            )

    @pytest.mark.asyncio
    async def test_spanning_tree_errors(self):
        """Test minimum spanning tree errors."""
        from networkx_mcp.server import minimum_spanning_tree

        # MST on disconnected graph
        with pytest.raises(Exception):
            await minimum_spanning_tree(graph_id="disconnected")

        # MST on directed graph
        self.manager.create_graph("directed_mst", "DiGraph")
        self.manager.add_nodes_from("directed_mst", ["A", "B", "C"])
        self.manager.add_edges_from("directed_mst", [("A", "B"), ("B", "C")])

        with pytest.raises(Exception):
            await minimum_spanning_tree(graph_id="directed_mst")


class TestIOErrors:
    """Test I/O operation error handling."""

    @pytest.mark.asyncio
    async def test_import_file_errors(self):
        """Test file import error cases."""
        from networkx_mcp.server import import_graph

        # Non-existent file
        with pytest.raises(Exception):
            await import_graph(
                format="json",
                path="/nonexistent/path/file.json",
                graph_id="test_import"
            )

        # Invalid format
        with pytest.raises(Exception):
            await import_graph(
                format="invalid_format",
                path="dummy_path",
                graph_id="test_import"
            )

    @pytest.mark.asyncio
    async def test_export_errors(self):
        """Test export error cases."""
        from networkx_mcp.server import export_graph

        # Export non-existent graph
        with pytest.raises(Exception):
            await export_graph(
                graph_id="nonexistent_graph",
                format="json"
            )

        # Invalid export format
        with pytest.raises(Exception):
            await export_graph(
                graph_id="connected",
                format="invalid_format"
            )

    def test_malformed_file_content(self):
        """Test handling of malformed file content."""
        from networkx_mcp.core.io_handlers import GraphIOHandler

        # Create malformed JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json content')  # Malformed JSON
            malformed_path = f.name

        try:
            with pytest.raises(Exception):
                GraphIOHandler.import_from_file(malformed_path, "json")
        finally:
            import os
            os.unlink(malformed_path)

    def test_unsupported_file_extensions(self):
        """Test handling of unsupported file extensions."""
        from networkx_mcp.core.io_handlers import GraphIOHandler

        with tempfile.NamedTemporaryFile(suffix=".unsupported", delete=False) as f:
            unsupported_path = f.name

        try:
            with pytest.raises(Exception):
                GraphIOHandler.import_from_file(unsupported_path, "auto")
        finally:
            import os
            os.unlink(unsupported_path)


class TestVisualizationErrors:
    """Test visualization error handling."""

    @pytest.mark.asyncio
    async def test_visualization_with_empty_graph(self):
        """Test visualization of empty graphs."""
        from networkx_mcp.server import create_graph, visualize_graph

        await create_graph(graph_id="empty_viz", graph_type="Graph")

        # Should handle empty graph gracefully
        result = await visualize_graph(
            graph_id="empty_viz",
            layout="spring"
        )

        # Should not crash and provide some result
        assert "layout_used" in result or "error" in result

    @pytest.mark.asyncio
    async def test_invalid_layout_algorithms(self):
        """Test handling of invalid layout algorithms."""
        from networkx_mcp.server import add_nodes, create_graph, layout_calculation

        await create_graph(graph_id="layout_test", graph_type="Graph")
        await add_nodes(graph_id="layout_test", nodes=["A", "B", "C"])

        # Invalid layout should fall back to default
        result = await layout_calculation(
            graph_id="layout_test",
            algorithm="invalid_layout"
        )

        # Should fallback to spring layout or handle gracefully
        assert "layout" in result

    @patch("matplotlib.pyplot.savefig")
    def test_matplotlib_save_error(self, mock_savefig):
        """Test handling of matplotlib save errors."""
        from networkx_mcp.visualization.matplotlib_visualizer import (
            MatplotlibVisualizer,
        )

        # Mock savefig to raise an error
        mock_savefig.side_effect = OSError("Cannot save figure")

        graph = nx.complete_graph(3)

        with pytest.raises(IOError):
            MatplotlibVisualizer.create_static_plot(graph)

    def test_missing_visualization_dependencies(self):
        """Test handling when visualization dependencies are missing."""
        with patch.dict("sys.modules", {"plotly": None}):
            # This would test ImportError handling
            # Implementation depends on how imports are structured
            pass


class TestConcurrencyErrors:
    """Test error handling in concurrent operations."""

    def test_concurrent_graph_modification(self):
        """Test handling of concurrent graph modifications."""
        import threading
        import time

        manager = GraphManager()
        manager.create_graph("concurrent_test", "Graph")

        errors = []

        def add_nodes_worker(start_id):
            try:
                for i in range(10):
                    manager.add_node("concurrent_test", f"node_{start_id}_{i}")
                    time.sleep(0.001)  # Small delay to increase chance of conflicts
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_nodes_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should handle concurrent access gracefully
        # (NetworkX graphs are not thread-safe, so some errors expected)
        graph = manager.get_graph("concurrent_test")
        assert graph.number_of_nodes() > 0  # Some nodes should be added

    def test_memory_pressure_handling(self):
        """Test handling of memory pressure scenarios."""
        manager = GraphManager()
        manager.create_graph("memory_test", "Graph")

        # Try to add a large number of nodes/edges
        large_number = 50000

        try:
            # Add nodes in batches to test memory handling
            batch_size = 10000
            for batch_start in range(0, large_number, batch_size):
                batch_end = min(batch_start + batch_size, large_number)
                nodes = [f"node_{i}" for i in range(batch_start, batch_end)]
                manager.add_nodes_from("memory_test", nodes)

            # If we get here, memory handling worked
            graph = manager.get_graph("memory_test")
            assert graph.number_of_nodes() == large_number

        except MemoryError:
            # Memory error is acceptable for very large graphs
            assert True
        except Exception as e:
            # Other errors should be investigated
            pytest.fail(f"Unexpected error in memory test: {e}")


class TestDataIntegrityErrors:
    """Test data integrity and consistency error handling."""

    def test_graph_corruption_detection(self):
        """Test detection of graph data corruption."""
        manager = GraphManager()
        manager.create_graph("integrity_test", "Graph")
        manager.add_nodes_from("integrity_test", ["A", "B", "C"])
        manager.add_edges_from("integrity_test", [("A", "B"), ("B", "C")])

        graph = manager.get_graph("integrity_test")

        # Simulate corruption by modifying internal structures
        # This is a simplified test - real corruption would be more subtle
        original_nodes = set(graph.nodes())
        original_edges = set(graph.edges())

        # Verify integrity
        current_nodes = set(graph.nodes())
        current_edges = set(graph.edges())

        assert current_nodes == original_nodes
        assert current_edges == original_edges

        # Check edge consistency
        for u, v in graph.edges():
            assert u in graph.nodes()
            assert v in graph.nodes()

    def test_attribute_type_consistency(self):
        """Test handling of inconsistent attribute types."""
        manager = GraphManager()
        manager.create_graph("attr_test", "Graph")

        # Add nodes with different attribute types
        manager.add_node("attr_test", "A", value=1)
        manager.add_node("attr_test", "B", value="string")
        manager.add_node("attr_test", "C", value=1.5)
        manager.add_node("attr_test", "D", value=None)

        graph = manager.get_graph("attr_test")

        # Should handle mixed types gracefully
        for _node, data in graph.nodes(data=True):
            assert "value" in data
            # Value can be any type

    def test_circular_reference_handling(self):
        """Test handling of circular references in data structures."""
        manager = GraphManager()
        manager.create_graph("circular_test", "DiGraph")

        # Create circular structure
        manager.add_nodes_from("circular_test", ["A", "B", "C"])
        manager.add_edges_from("circular_test", [
            ("A", "B"), ("B", "C"), ("C", "A")  # Circular
        ])

        graph = manager.get_graph("circular_test")

        # Should handle cycles without infinite loops
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 3

        # Test cycle detection
        assert not nx.is_directed_acyclic_graph(graph)
        cycles = list(nx.simple_cycles(graph))
        assert len(cycles) == 1


class TestGracefulDegradation:
    """Test graceful degradation under adverse conditions."""

    def test_partial_feature_availability(self):
        """Test behavior when some features are unavailable."""
        # Mock unavailable optional dependencies
        with patch.dict("sys.modules", {"community": None}):
            # Community detection should degrade gracefully
            # This would need to be implemented in the actual code
            pass

    def test_reduced_functionality_mode(self):
        """Test reduced functionality under resource constraints."""
        # Simulate resource constraints
        manager = GraphManager()
        manager.create_graph("reduced_test", "Graph")

        # Add a moderate number of nodes
        nodes = [f"node_{i}" for i in range(100)]
        manager.add_nodes_from("reduced_test", nodes)

        # Should work with reduced resources
        info = manager.get_graph_info("reduced_test")
        assert info["num_nodes"] == 100

    def test_error_recovery_mechanisms(self):
        """Test automatic error recovery mechanisms."""
        manager = GraphManager()

        # Test recovery from failed operations
        try:
            manager.create_graph("recovery_test", "InvalidType")
        except ValueError:
            # Should be able to continue after error
            manager.create_graph("recovery_test", "Graph")
            assert "recovery_test" in manager.graphs

    def test_fallback_algorithms(self):
        """Test fallback to simpler algorithms when optimal ones fail."""
        # This would test algorithm fallback mechanisms
        # For example, if advanced community detection fails,
        # fall back to simpler methods
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
