"""Comprehensive tests for utils/validators.py - Target: 90%+ coverage (603 lines).

This module is CRITICAL for security - all input validation flows through here.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import networkx as nx
import pytest

from networkx_mcp.utils.validators import GraphValidator


class TestGraphValidatorBasics:
    """Test basic validation methods."""

    def test_validate_graph_type(self):
        """Test graph type validation."""
        # Valid types
        assert GraphValidator.validate_graph_type("Graph") is True
        assert GraphValidator.validate_graph_type("DiGraph") is True
        assert GraphValidator.validate_graph_type("MultiGraph") is True
        assert GraphValidator.validate_graph_type("MultiDiGraph") is True

        # Invalid types
        assert GraphValidator.validate_graph_type("InvalidGraph") is False
        assert GraphValidator.validate_graph_type("graph") is False  # Case sensitive
        assert GraphValidator.validate_graph_type("") is False
        assert GraphValidator.validate_graph_type(None) is False

    def test_validate_node_id(self):
        """Test node ID validation."""
        # Valid node IDs
        assert GraphValidator.validate_node_id("node1") is True
        assert GraphValidator.validate_node_id(42) is True
        assert GraphValidator.validate_node_id((1, 2)) is True
        assert GraphValidator.validate_node_id("A") is True
        assert GraphValidator.validate_node_id(0) is True

        # Invalid node IDs
        assert GraphValidator.validate_node_id(None) is False
        assert GraphValidator.validate_node_id("") is False
        assert GraphValidator.validate_node_id([1, 2]) is False
        assert GraphValidator.validate_node_id({"id": 1}) is False
        assert GraphValidator.validate_node_id(set()) is False

    def test_validate_edge(self):
        """Test edge validation."""
        # Valid edges
        assert GraphValidator.validate_edge(("A", "B")) is True
        assert GraphValidator.validate_edge([1, 2]) is True
        assert GraphValidator.validate_edge((1, 2, {"weight": 1.5})) is True
        assert GraphValidator.validate_edge(["node1", "node2"]) is True

        # Invalid edges
        assert GraphValidator.validate_edge(None) is False
        assert GraphValidator.validate_edge("edge") is False
        assert GraphValidator.validate_edge([1]) is False  # Too short
        assert GraphValidator.validate_edge([]) is False  # Empty
        assert GraphValidator.validate_edge([None, "B"]) is False  # Invalid node
        assert GraphValidator.validate_edge(["A", None]) is False  # Invalid node
        assert GraphValidator.validate_edge(["", "B"]) is False  # Empty node ID

    def test_validate_attributes(self):
        """Test attributes validation."""
        # Valid attributes
        assert GraphValidator.validate_attributes({}) is True
        assert GraphValidator.validate_attributes({"color": "red"}) is True
        assert (
            GraphValidator.validate_attributes({"weight": 1.5, "label": "edge1"})
            is True
        )

        # Invalid attributes
        assert GraphValidator.validate_attributes(None) is False
        assert GraphValidator.validate_attributes("attrs") is False
        assert GraphValidator.validate_attributes([]) is False
        assert (
            GraphValidator.validate_attributes({1: "value"}) is False
        )  # Non-string key
        assert GraphValidator.validate_attributes({"id": "value"}) is False  # Reserved
        assert GraphValidator.validate_attributes({"source": "A"}) is False  # Reserved
        assert GraphValidator.validate_attributes({"target": "B"}) is False  # Reserved
        assert GraphValidator.validate_attributes({"key": "K"}) is False  # Reserved


class TestGraphValidatorGraphOperations:
    """Test graph-specific validation methods."""

    def test_validate_weight(self):
        """Test weight attribute validation."""
        # Graph with weight attribute
        g = nx.Graph()
        g.add_edge(1, 2, weight=1.5)
        g.add_edge(2, 3, weight=2.0)
        assert GraphValidator.validate_weight(g, "weight") is True
        assert GraphValidator.validate_weight(g, "nonexistent") is False

        # Empty graph
        empty_g = nx.Graph()
        assert GraphValidator.validate_weight(empty_g, "weight") is True  # Empty is OK

        # Graph without weight
        no_weight = nx.Graph()
        no_weight.add_edge(1, 2)
        assert GraphValidator.validate_weight(no_weight, "weight") is False

    def test_validate_path_exists(self):
        """Test path existence validation."""
        g = nx.Graph()
        g.add_edges_from([(1, 2), (2, 3), (4, 5)])  # Two components

        # Path exists
        assert GraphValidator.validate_path_exists(g, 1, 3) is True
        assert GraphValidator.validate_path_exists(g, 4, 5) is True

        # No path (different components)
        assert GraphValidator.validate_path_exists(g, 1, 4) is False

        # Node not in graph
        assert GraphValidator.validate_path_exists(g, 1, 99) is False
        assert GraphValidator.validate_path_exists(g, 99, 1) is False

    def test_validate_graph_connectivity(self):
        """Test graph connectivity validation."""
        # Connected graph
        connected = nx.Graph()
        connected.add_edges_from([(1, 2), (2, 3), (3, 1)])
        result = GraphValidator.validate_graph_connectivity(connected)
        assert result["valid"] is True
        assert result["is_connected"] is True
        assert result["num_nodes"] == 3
        assert result["num_edges"] == 3

        # Disconnected graph
        disconnected = nx.Graph()
        disconnected.add_edges_from([(1, 2), (3, 4)])
        result = GraphValidator.validate_graph_connectivity(
            disconnected, require_connected=True
        )
        assert result["valid"] is False
        assert result["is_connected"] is False
        assert "not connected" in result["message"]

        # Empty graph
        empty = nx.Graph()
        result = GraphValidator.validate_graph_connectivity(empty)
        assert result["valid"] is True
        assert result["message"] == "Graph is empty"

        # Directed graph - weakly connected
        digraph = nx.DiGraph()
        digraph.add_edges_from([(1, 2), (2, 3)])
        result = GraphValidator.validate_graph_connectivity(digraph)
        assert result["is_weakly_connected"] is True
        assert result["is_strongly_connected"] is False

        # Directed graph - strongly connected
        strong = nx.DiGraph()
        strong.add_edges_from([(1, 2), (2, 3), (3, 1)])
        result = GraphValidator.validate_graph_connectivity(strong)
        assert result["is_strongly_connected"] is True


class TestGraphValidatorAlgorithms:
    """Test algorithm validation methods."""

    def test_validate_algorithm_input_shortest_path(self):
        """Test shortest path algorithm validation."""
        g = nx.Graph()
        g.add_edges_from([(1, 2, {"weight": 1}), (2, 3, {"weight": 2})])

        # Valid inputs
        result = GraphValidator.validate_algorithm_input(
            "dijkstra", g, {"source": 1, "target": 3, "weight": "weight"}
        )
        assert result["valid"] is True
        assert len(result["errors"]) == 0

        # Invalid source
        result = GraphValidator.validate_algorithm_input(
            "bellman-ford", g, {"source": 99}
        )
        assert result["valid"] is False
        assert "Source node '99' not in graph" in result["errors"]

        # Invalid target
        result = GraphValidator.validate_algorithm_input(
            "floyd-warshall", g, {"target": 99}
        )
        assert result["valid"] is False
        assert "Target node '99' not in graph" in result["errors"]

        # Missing weight attribute
        result = GraphValidator.validate_algorithm_input(
            "dijkstra", g, {"source": 1, "weight": "cost"}
        )
        assert "Weight attribute 'cost' not found in edges" in result["errors"]

    def test_validate_algorithm_input_flow(self):
        """Test flow algorithm validation."""
        # Directed graph for flow
        digraph = nx.DiGraph()
        digraph.add_edges_from([(1, 2), (2, 3)])

        result = GraphValidator.validate_algorithm_input(
            "maximum_flow", digraph, {"source": 1, "sink": 3}
        )
        assert result["valid"] is True

        # Undirected graph (invalid for flow)
        undirected = nx.Graph()
        undirected.add_edges_from([(1, 2)])
        result = GraphValidator.validate_algorithm_input("minimum_cut", undirected, {})
        assert result["valid"] is False
        assert "Flow algorithms require directed graphs" in result["errors"]

        # Invalid source/sink
        result = GraphValidator.validate_algorithm_input(
            "maximum_flow", digraph, {"source": 99, "sink": 88}
        )
        assert result["valid"] is False
        assert "Source node '99' not in graph" in result["errors"]
        assert "Sink node '88' not in graph" in result["errors"]

    def test_validate_algorithm_input_spanning_tree(self):
        """Test spanning tree algorithm validation."""
        # Connected undirected graph
        connected = nx.Graph()
        connected.add_edges_from([(1, 2), (2, 3), (3, 1)])
        result = GraphValidator.validate_algorithm_input(
            "minimum_spanning_tree", connected, {}
        )
        assert result["valid"] is True

        # Directed graph (invalid)
        digraph = nx.DiGraph()
        digraph.add_edges_from([(1, 2)])
        result = GraphValidator.validate_algorithm_input(
            "maximum_spanning_tree", digraph, {}
        )
        assert result["valid"] is False
        assert "Spanning tree algorithms require undirected graphs" in result["errors"]

        # Disconnected graph (invalid)
        disconnected = nx.Graph()
        disconnected.add_edges_from([(1, 2), (3, 4)])
        result = GraphValidator.validate_algorithm_input(
            "minimum_spanning_tree", disconnected, {}
        )
        assert result["valid"] is False
        assert "Graph is not connected" in result["errors"]

    def test_validate_algorithm_input_coloring(self):
        """Test graph coloring validation."""
        # Undirected graph (preferred)
        undirected = nx.Graph()
        undirected.add_edges_from([(1, 2)])
        result = GraphValidator.validate_algorithm_input(
            "graph_coloring", undirected, {}
        )
        assert result["valid"] is True

        # Directed graph (warning)
        digraph = nx.DiGraph()
        digraph.add_edges_from([(1, 2)])
        result = GraphValidator.validate_algorithm_input("graph_coloring", digraph, {})
        assert (
            "Graph coloring typically applied to undirected graphs" in result["errors"]
        )

    def test_validate_centrality_measure(self):
        """Test centrality measure validation."""
        # Valid measures
        assert GraphValidator.validate_centrality_measure("degree") is True
        assert GraphValidator.validate_centrality_measure("betweenness") is True
        assert GraphValidator.validate_centrality_measure("closeness") is True
        assert GraphValidator.validate_centrality_measure("eigenvector") is True
        assert GraphValidator.validate_centrality_measure("pagerank") is True
        assert GraphValidator.validate_centrality_measure("katz") is True
        assert GraphValidator.validate_centrality_measure("hits") is True
        assert GraphValidator.validate_centrality_measure("harmonic") is True

        # Invalid measures
        assert GraphValidator.validate_centrality_measure("invalid") is False
        assert GraphValidator.validate_centrality_measure("") is False
        assert GraphValidator.validate_centrality_measure(None) is False

    def test_validate_layout_algorithm(self):
        """Test layout algorithm validation."""
        # Valid algorithms
        assert GraphValidator.validate_layout_algorithm("spring") is True
        assert GraphValidator.validate_layout_algorithm("circular") is True
        assert GraphValidator.validate_layout_algorithm("random") is True
        assert GraphValidator.validate_layout_algorithm("shell") is True
        assert GraphValidator.validate_layout_algorithm("spectral") is True
        assert GraphValidator.validate_layout_algorithm("kamada_kawai") is True
        assert GraphValidator.validate_layout_algorithm("planar") is True
        assert GraphValidator.validate_layout_algorithm("fruchterman_reingold") is True
        assert GraphValidator.validate_layout_algorithm("bipartite") is True

        # Invalid algorithms
        assert GraphValidator.validate_layout_algorithm("invalid") is False
        assert GraphValidator.validate_layout_algorithm("") is False


class TestGraphValidatorFileOperations:
    """Test file-related validation methods."""

    def test_validate_file_format(self):
        """Test file format validation."""
        # Export formats
        assert GraphValidator.validate_file_format("json", "export") is True
        assert GraphValidator.validate_file_format("graphml", "export") is True
        assert GraphValidator.validate_file_format("gexf", "export") is True
        assert GraphValidator.validate_file_format("pickle", "export") is True
        assert (
            GraphValidator.validate_file_format("YAML", "export") is True
        )  # Upper case

        # Import formats
        assert GraphValidator.validate_file_format("json", "import") is True
        assert GraphValidator.validate_file_format("edgelist", "import") is True

        # Invalid formats
        assert GraphValidator.validate_file_format("invalid", "export") is False
        assert GraphValidator.validate_file_format("", "export") is False

    def test_validate_file_format_backward_compatibility(self):
        """Test backward compatibility signature."""
        # This tests the dual signature support
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            f.write(b'{"test": "data"}')
            f.flush()

            # New signature: (filepath, [formats]) -> Tuple[bool, str]
            result = GraphValidator.validate_file_format(f.name, ["json", "yaml"])
            assert isinstance(result, tuple)
            assert result[0] is True
            assert result[1] is None

    def test_validate_file_path_format(self):
        """Test file path format validation."""
        # Create test file
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            f.write(b"<graphml></graphml>")
            temp_path = f.name

        try:
            # Valid file
            valid, error = GraphValidator.validate_file_path_format(
                temp_path, ["graphml", "xml"]
            )
            assert valid is True
            assert error is None

            # Wrong format
            valid, error = GraphValidator.validate_file_path_format(temp_path, ["json"])
            assert valid is False
            assert "Unexpected file format" in error

            # None path
            valid, error = GraphValidator.validate_file_path_format(None)
            assert valid is False
            assert "cannot be None" in error

            # Empty path
            valid, error = GraphValidator.validate_file_path_format("")
            assert valid is False
            assert "cannot be empty" in error

            # Path traversal
            valid, error = GraphValidator.validate_file_path_format(
                "../../../etc/passwd"
            )
            assert valid is False
            assert "Path traversal" in error

            # URL path
            valid, error = GraphValidator.validate_file_path_format(
                "http://example.com/file"
            )
            assert valid is False
            assert "URL paths are not allowed" in error

            # Non-existent file
            valid, error = GraphValidator.validate_file_path_format(
                "/nonexistent/file.json"
            )
            assert valid is False
            assert "File not found" in error
        finally:
            Path(temp_path).unlink()

    def test_validate_file_path_format_empty_file(self):
        """Test validation of empty files."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            # Empty file
            temp_path = f.name

        try:
            valid, error = GraphValidator.validate_file_path_format(temp_path)
            assert valid is False
            assert "File is empty" in error
        finally:
            Path(temp_path).unlink()

    @patch("networkx_mcp.utils.validators.logger")
    def test_validate_file_path_format_large_file(self, mock_logger):
        """Test large file warning."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            # Write small content
            f.write(b'{"test": "data"}')
            f.flush()
            temp_path = f.name

        try:
            # Mock Path.stat to simulate large file
            with patch("pathlib.Path.stat") as mock_stat:
                # Mock a large file size (600MB)
                mock_stat.return_value = Mock(st_size=600 * 1024 * 1024, st_mode=0o444)

                # The exists and is_file checks
                with (
                    patch("pathlib.Path.exists", return_value=True),
                    patch("pathlib.Path.is_file", return_value=True),
                ):
                    valid, error = GraphValidator.validate_file_path_format(temp_path)
                    assert valid is True
                    assert error is None

                    # Check warning was logged
                    mock_logger.warning.assert_called()
                    call_args = mock_logger.warning.call_args[0][0]
                    assert "Large file detected" in call_args
        finally:
            Path(temp_path).unlink()


class TestGraphValidatorDataValidation:
    """Test data structure validation methods."""

    def test_validate_graph_id(self):
        """Test graph ID validation."""
        # Valid IDs
        assert GraphValidator.validate_graph_id("my_graph")[0] is True
        assert GraphValidator.validate_graph_id("graph-123")[0] is True
        assert GraphValidator.validate_graph_id("G1")[0] is True
        assert GraphValidator.validate_graph_id("a" * 255)[0] is True  # Max length

        # Invalid IDs
        valid, error = GraphValidator.validate_graph_id("")
        assert valid is False
        assert "cannot be empty" in error

        valid, error = GraphValidator.validate_graph_id(None)
        assert valid is False
        assert "cannot be empty" in error  # None is treated as empty

        valid, error = GraphValidator.validate_graph_id("a" * 256)
        assert valid is False
        assert "between 1 and 255 characters" in error

        valid, error = GraphValidator.validate_graph_id("graph@123")
        assert valid is False
        assert "alphanumeric characters" in error

        # Reserved names
        valid, error = GraphValidator.validate_graph_id("graph")
        assert valid is False
        assert "reserved" in error

        valid, error = GraphValidator.validate_graph_id("CON")  # Windows reserved
        assert valid is False
        assert "reserved" in error

    def test_sanitize_graph_data(self):
        """Test graph data sanitization."""
        # Basic sanitization
        data = {
            "nodes": [1, "node2", {"id": 3, "label": "Node 3"}],
            "edges": [
                (1, 2),
                ["node2", 3],
                {"source": 1, "target": "node2", "weight": 1.5},
            ],
        }

        sanitized = GraphValidator.sanitize_graph_data(data)

        # Check nodes are sanitized
        assert len(sanitized["nodes"]) == 3
        assert sanitized["nodes"][0] == {"id": "1"}
        assert sanitized["nodes"][1] == {"id": "node2"}
        assert sanitized["nodes"][2]["id"] == "3"
        assert sanitized["nodes"][2]["label"] == "Node 3"

        # Check edges are sanitized
        assert len(sanitized["edges"]) == 3
        assert sanitized["edges"][0] == {"source": "1", "target": "2"}
        assert sanitized["edges"][2]["weight"] == 1.5

        # Graph attributes
        data_with_attrs = {"graph": {"name": "test", "directed": True}}
        sanitized = GraphValidator.sanitize_graph_data(data_with_attrs)
        assert sanitized["graph"] == {"name": "test", "directed": True}

        # Empty data
        assert GraphValidator.sanitize_graph_data({}) == {}

    def test_validate_graph_data(self):
        """Test graph data structure validation."""
        # Valid data
        valid_data = {
            "nodes": [1, 2, {"id": 3}],
            "edges": [[1, 2], {"source": 2, "target": 3}],
        }
        valid, errors = GraphValidator.validate_graph_data(valid_data)
        assert valid is True
        assert len(errors) == 0

        # Invalid nodes
        invalid_nodes = {"nodes": "not_a_list"}
        valid, errors = GraphValidator.validate_graph_data(invalid_nodes)
        assert valid is False
        assert "'nodes' must be a List[Any]" in errors

        # Node missing ID
        missing_id = {"nodes": [{"label": "A"}]}
        valid, errors = GraphValidator.validate_graph_data(missing_id)
        assert valid is False
        assert "missing 'id' field" in errors[0]

        # Invalid node ID
        invalid_id = {"nodes": [None]}
        valid, errors = GraphValidator.validate_graph_data(invalid_id)
        assert valid is False
        assert "Invalid node ID" in errors[0]

        # Invalid edges
        invalid_edges = {"edges": "not_a_list"}
        valid, errors = GraphValidator.validate_graph_data(invalid_edges)
        assert valid is False
        assert "'edges' must be a List[Any]" in errors

        # Edge missing fields
        missing_source = {"edges": [{"target": 2}]}
        valid, errors = GraphValidator.validate_graph_data(missing_source)
        assert valid is False
        assert "missing 'source' field" in errors[0]

        # Edge too short
        short_edge = {"edges": [[1]]}
        valid, errors = GraphValidator.validate_graph_data(short_edge)
        assert valid is False
        assert "must have at least 2 elements" in errors[0]

        # Edge too long
        long_edge = {"edges": [[1, 2, 3, 4]]}
        valid, errors = GraphValidator.validate_graph_data(long_edge)
        assert valid is False
        assert "too many elements" in errors[0]

        # Invalid graph metadata
        invalid_meta = {"graph": "not_a_dict"}
        valid, errors = GraphValidator.validate_graph_data(invalid_meta)
        assert valid is False
        assert "'graph' metadata must be a dictionary" in errors

        # Adjacency matrix
        valid_matrix = {"adjacency_matrix": [[0, 1], [1, 0]]}
        valid, errors = GraphValidator.validate_graph_data(valid_matrix)
        assert valid is True

        # Invalid adjacency matrix (not square)
        invalid_matrix = {"adjacency_matrix": [[0, 1], [1, 0, 1]]}
        valid, errors = GraphValidator.validate_graph_data(invalid_matrix)
        assert valid is False
        assert "incorrect length" in errors[0]

        # Not a dictionary
        valid, errors = GraphValidator.validate_graph_data("not_a_dict")
        assert valid is False
        assert "Graph data must be a dictionary" in errors

    def test_validate_import_data(self):
        """Test import data validation."""
        # File formats requiring path
        valid, error = GraphValidator.validate_import_data("graphml", path=None)
        assert valid is False
        assert "requires a file path" in error

        # Data formats
        valid, error = GraphValidator.validate_import_data("json", data={"nodes": []})
        assert valid is True
        assert error is None

        # Adjacency format
        valid, error = GraphValidator.validate_import_data(
            "adjacency", data={"matrix": [[0, 1], [1, 0]]}
        )
        assert valid is True

        # Invalid adjacency data
        valid, error = GraphValidator.validate_import_data(
            "adjacency", data={"wrong": "format"}
        )
        assert valid is False
        assert "requires Dict[str, Any] with 'matrix' key" in error

        # Unknown format
        valid, error = GraphValidator.validate_import_data("unknown_format")
        assert valid is False
        assert "Unknown import format" in error

        # Format requiring either data or path
        valid, error = GraphValidator.validate_import_data("json", data=None, path=None)
        assert valid is False
        assert "requires either data or path" in error

        # Test with mock file
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            f.write(b'{"nodes": []}')
            f.flush()
            valid, error = GraphValidator.validate_import_data("json", path=f.name)
            assert valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
