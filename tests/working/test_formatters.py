"""Comprehensive tests for utils/formatters.py - Target: 100% coverage (97 lines).

This module is CRITICAL for user experience as ALL server responses flow through these formatters.
Testing must be thorough to prevent user-facing bugs.
"""

import json
from unittest.mock import Mock

import networkx as nx


class TestGraphFormatter:
    """Test the GraphFormatter class static methods."""

    def test_format_success_basic(self):
        """Test basic success response formatting."""
        from networkx_mcp.utils.formatters import GraphFormatter

        result = GraphFormatter.format_success({"test": "data"})

        assert result == {
            "success": True,
            "message": "Success",
            "data": {"test": "data"},
        }

        # Verify structure
        assert isinstance(result, dict)
        assert result["success"] is True
        assert "message" in result
        assert "data" in result

    def test_format_success_custom_message(self):
        """Test success response with custom message."""
        from networkx_mcp.utils.formatters import GraphFormatter

        result = GraphFormatter.format_success(
            {"nodes": 5}, message="Graph created successfully"
        )

        assert result == {
            "success": True,
            "message": "Graph created successfully",
            "data": {"nodes": 5},
        }

    def test_format_success_edge_cases(self):
        """Test success response with edge case data."""
        from networkx_mcp.utils.formatters import GraphFormatter

        # None data
        result = GraphFormatter.format_success(None)
        assert result["data"] is None
        assert result["success"] is True

        # Empty dict
        result = GraphFormatter.format_success({})
        assert result["data"] == {}

        # Complex nested data
        complex_data = {
            "graph": {"nodes": [1, 2, 3], "edges": [(1, 2), (2, 3)]},
            "metadata": {"created": "2024-01-01", "version": 1.0},
        }
        result = GraphFormatter.format_success(complex_data)
        assert result["data"] == complex_data

    def test_format_error_basic(self):
        """Test basic error response formatting."""
        from networkx_mcp.utils.formatters import GraphFormatter

        result = GraphFormatter.format_error("ValidationError", "Invalid graph ID")

        assert result == {
            "success": False,
            "error": "ValidationError",
            "message": "Invalid graph ID",
        }

        # Verify structure
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "error" in result
        assert "message" in result

    def test_format_error_edge_cases(self):
        """Test error response with edge cases."""
        from networkx_mcp.utils.formatters import GraphFormatter

        # Empty strings
        result = GraphFormatter.format_error("", "")
        assert result["error"] == ""
        assert result["message"] == ""
        assert result["success"] is False

        # Long error messages
        long_message = "A" * 1000
        result = GraphFormatter.format_error("SystemError", long_message)
        assert result["message"] == long_message

    def test_format_algorithm_result_basic(self):
        """Test algorithm result formatting without execution time."""
        from networkx_mcp.utils.formatters import GraphFormatter

        result = GraphFormatter.format_algorithm_result("shortest_path", [1, 2, 3])

        assert result == {"algorithm": "shortest_path", "result": [1, 2, 3]}

        # Should not have execution_time_ms when not provided
        assert "execution_time_ms" not in result

    def test_format_algorithm_result_with_timing(self):
        """Test algorithm result formatting with execution time."""
        from networkx_mcp.utils.formatters import GraphFormatter

        result = GraphFormatter.format_algorithm_result(
            "betweenness_centrality",
            {"node1": 0.5, "node2": 0.3},
            execution_time=123.45,
        )

        assert result == {
            "algorithm": "betweenness_centrality",
            "result": {"node1": 0.5, "node2": 0.3},
            "execution_time_ms": 123.45,
        }

    def test_format_algorithm_result_edge_cases(self):
        """Test algorithm result formatting edge cases."""
        from networkx_mcp.utils.formatters import GraphFormatter

        # None result
        result = GraphFormatter.format_algorithm_result("test_algo", None)
        assert result["result"] is None

        # Zero execution time
        result = GraphFormatter.format_algorithm_result("test_algo", "result", 0.0)
        assert result["execution_time_ms"] == 0.0

        # Negative execution time (edge case)
        result = GraphFormatter.format_algorithm_result("test_algo", "result", -1.0)
        assert result["execution_time_ms"] == -1.0

    def test_format_community_results_list(self):
        """Test community detection formatting with list input."""
        from networkx_mcp.utils.formatters import GraphFormatter

        communities = [[1, 2, 3], [4, 5], [6]]
        result = GraphFormatter.format_community_results(communities)

        assert result == {"communities": [[1, 2, 3], [4, 5], [6]], "num_communities": 3}

    def test_format_community_results_tuple(self):
        """Test community detection formatting with tuple input."""
        from networkx_mcp.utils.formatters import GraphFormatter

        communities = ({1, 2, 3}, {4, 5})
        result = GraphFormatter.format_community_results(communities)

        assert result["communities"] == communities
        assert result["num_communities"] == 2

    def test_format_community_results_edge_cases(self):
        """Test community detection formatting edge cases."""
        from networkx_mcp.utils.formatters import GraphFormatter

        # Empty communities
        result = GraphFormatter.format_community_results([])
        assert result["num_communities"] == 0

        # No __len__ method (like generators)
        mock_communities = Mock()
        del mock_communities.__len__  # Remove __len__ method
        result = GraphFormatter.format_community_results(mock_communities)
        assert result["num_communities"] == 0
        assert result["communities"] == mock_communities

        # None input
        result = GraphFormatter.format_community_results(None)
        assert result["num_communities"] == 0

    def test_format_visualization_data_complete(self):
        """Test visualization data formatting with complete graph."""
        from networkx_mcp.utils.formatters import GraphFormatter

        # Create test graph
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=1.0, color="red")
        graph.add_edge(2, 3, weight=2.0, color="blue")

        layout = {1: (0, 0), 2: (1, 1), 3: (2, 0)}
        options = {"node_color": "red", "edge_color": "black"}

        result = GraphFormatter.format_visualization_data(graph, layout, options)

        # Check structure
        assert "nodes" in result
        assert "edges" in result
        assert "layout" in result
        assert "options" in result

        # Check nodes
        assert len(result["nodes"]) == 3
        node_1 = next(n for n in result["nodes"] if n["id"] == "1")
        assert node_1 == {"id": "1", "label": "1", "x": 0, "y": 0}

        # Check edges
        assert len(result["edges"]) == 2
        edge_12 = next(
            e for e in result["edges"] if e["source"] == "1" and e["target"] == "2"
        )
        assert edge_12["weight"] == 1.0
        assert edge_12["color"] == "red"

        # Check layout and options
        assert result["layout"] == layout
        assert result["options"] == options

    def test_format_visualization_data_no_layout(self):
        """Test visualization data formatting without layout."""
        from networkx_mcp.utils.formatters import GraphFormatter

        graph = nx.Graph()
        graph.add_edge("A", "B")

        result = GraphFormatter.format_visualization_data(graph, None)

        # Nodes should not have x, y coordinates
        node_a = next(n for n in result["nodes"] if n["id"] == "A")
        assert "x" not in node_a
        assert "y" not in node_a
        assert node_a == {"id": "A", "label": "A"}

    def test_format_visualization_data_empty_graph(self):
        """Test visualization data formatting with empty graph."""
        from networkx_mcp.utils.formatters import GraphFormatter

        graph = nx.Graph()

        result = GraphFormatter.format_visualization_data(graph, {})

        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["layout"] == {}
        assert result["options"] == {}

    def test_format_visualization_data_no_options(self):
        """Test visualization data formatting without options."""
        from networkx_mcp.utils.formatters import GraphFormatter

        graph = nx.Graph()
        graph.add_node(1)

        result = GraphFormatter.format_visualization_data(graph, {}, None)

        assert result["options"] == {}


class TestStandaloneFunctions:
    """Test standalone formatting functions."""

    def test_format_graph_summary_basic(self):
        """Test basic graph summary formatting."""
        from networkx_mcp.utils.formatters import format_graph_summary

        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3), (3, 4)])

        result = format_graph_summary(graph)

        expected = {
            "nodes": 4,
            "edges": 3,
            "directed": False,
            "multigraph": False,
            "connected": True,  # This graph is connected
            "density": 3 / (4 * 3 / 2),  # edges / max_possible_edges
        }

        assert result["nodes"] == expected["nodes"]
        assert result["edges"] == expected["edges"]
        assert result["directed"] == expected["directed"]
        assert result["multigraph"] == expected["multigraph"]
        assert abs(result["density"] - expected["density"]) < 0.0001

    def test_format_graph_summary_directed(self):
        """Test graph summary for directed graph."""
        from networkx_mcp.utils.formatters import format_graph_summary

        graph = nx.DiGraph()
        graph.add_edges_from([(1, 2), (2, 3)])

        result = format_graph_summary(graph)

        assert result["directed"] is True
        assert result["connected"] is None  # Not checked for directed graphs

    def test_format_graph_summary_multigraph(self):
        """Test graph summary for multigraph."""
        from networkx_mcp.utils.formatters import format_graph_summary

        graph = nx.MultiGraph()
        graph.add_edges_from([(1, 2), (1, 2), (2, 3)])  # Multiple edges 1-2

        result = format_graph_summary(graph)

        assert result["multigraph"] is True
        assert result["edges"] == 3

    def test_format_graph_summary_empty(self):
        """Test graph summary for empty graph."""
        from networkx_mcp.utils.formatters import format_graph_summary

        graph = nx.Graph()

        result = format_graph_summary(graph)

        assert result["nodes"] == 0
        assert result["edges"] == 0
        assert result["density"] == 0

    def test_format_graph_summary_single_node(self):
        """Test graph summary for single node graph."""
        from networkx_mcp.utils.formatters import format_graph_summary

        graph = nx.Graph()
        graph.add_node(1)

        result = format_graph_summary(graph)

        assert result["nodes"] == 1
        assert result["edges"] == 0
        assert result["density"] == 0  # No possible edges

    def test_format_graph_summary_disconnected(self):
        """Test graph summary for disconnected graph."""
        from networkx_mcp.utils.formatters import format_graph_summary

        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (3, 4)])  # Two disconnected components

        result = format_graph_summary(graph)

        # Mock the is_connected method to return False
        graph.is_connected = lambda: False
        result = format_graph_summary(graph)
        assert result["connected"] is False

    def test_format_json_output_pretty(self):
        """Test JSON formatting with pretty printing."""
        from networkx_mcp.utils.formatters import format_json_output

        data = {"nodes": [1, 2, 3], "edges": [(1, 2), (2, 3)]}

        result = format_json_output(data, pretty=True)

        # Should be formatted with indentation
        assert isinstance(result, str)
        assert "  " in result  # Indentation
        assert "\n" in result  # Newlines

        # Should be valid JSON
        parsed = json.loads(result)
        # Note: JSON converts tuples to lists, which is expected
        assert parsed["nodes"] == [1, 2, 3]
        assert parsed["edges"] == [[1, 2], [2, 3]]  # Tuples become lists

    def test_format_json_output_compact(self):
        """Test JSON formatting without pretty printing."""
        from networkx_mcp.utils.formatters import format_json_output

        data = {"test": "value", "number": 42}

        result = format_json_output(data, pretty=False)

        # Should be compact (no unnecessary whitespace)
        assert isinstance(result, str)
        assert "  " not in result  # No indentation
        assert "\n" not in result  # No newlines

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed == data  # Simple data types should match exactly

    def test_format_json_output_special_types(self):
        """Test JSON formatting with special Python types."""
        from networkx_mcp.utils.formatters import format_json_output

        # Include types that need str() conversion
        data = {"set": {1, 2, 3}, "complex": complex(1, 2), "bytes": b"test"}

        result = format_json_output(data)

        # Should convert to strings without errors
        assert isinstance(result, str)
        parsed = json.loads(result)

        # Check that non-JSON types were converted to strings
        assert isinstance(parsed["set"], str)
        assert isinstance(parsed["complex"], str)
        assert isinstance(parsed["bytes"], str)

    def test_format_error_response_basic(self):
        """Test error response formatting."""
        from networkx_mcp.utils.formatters import format_error_response

        error = ValueError("Invalid input")
        result = format_error_response(error)

        assert result == {"error": "Invalid input", "type": "ValueError", "context": ""}

    def test_format_error_response_with_context(self):
        """Test error response formatting with context."""
        from networkx_mcp.utils.formatters import format_error_response

        error = FileNotFoundError("Graph file not found")
        result = format_error_response(error, context="Loading graph data")

        assert result == {
            "error": "Graph file not found",
            "type": "FileNotFoundError",
            "context": "Loading graph data",
        }

    def test_format_error_response_edge_cases(self):
        """Test error response formatting edge cases."""
        from networkx_mcp.utils.formatters import format_error_response

        # Empty error message
        error = Exception("")
        result = format_error_response(error)
        assert result["error"] == ""
        assert result["type"] == "Exception"

        # Custom exception class
        class CustomError(Exception):
            pass

        error = CustomError("Custom message")
        result = format_error_response(error, "Custom context")
        assert result["type"] == "CustomError"
        assert result["error"] == "Custom message"
        assert result["context"] == "Custom context"
