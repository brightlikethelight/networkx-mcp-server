"""Comprehensive tests for GraphIOHandler - Target: 90%+ coverage of io_handlers.py

This test suite aims to provide extensive coverage of the GraphIOHandler class
which currently has 0% coverage and 520 missing statements.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import yaml

from networkx_mcp.core.io_handlers import GraphIOHandler


class TestGraphIOHandler:
    """Comprehensive test suite for GraphIOHandler class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test graph
        self.simple_graph = nx.Graph()
        self.simple_graph.add_node(1, label="Node1", weight=1.5)
        self.simple_graph.add_node(2, label="Node2", weight=2.0)
        self.simple_graph.add_edge(1, 2, weight=0.8, relation="connects")

        # Create directed test graph
        self.directed_graph = nx.DiGraph()
        self.directed_graph.add_nodes_from([1, 2, 3])
        self.directed_graph.add_edges_from([(1, 2), (2, 3), (3, 1)])

        # Create multi graph
        self.multi_graph = nx.MultiGraph()
        self.multi_graph.add_edge(1, 2, key="a", weight=1.0)
        self.multi_graph.add_edge(1, 2, key="b", weight=2.0)

    # ===== FORMAT DETECTION TESTS =====

    def test_detect_format_graphml(self):
        """Test format detection for GraphML files."""
        assert GraphIOHandler.detect_format("test.graphml") == "graphml"
        assert GraphIOHandler.detect_format("test.xml") == "graphml"

    def test_detect_format_json(self):
        """Test format detection for JSON files."""
        assert GraphIOHandler.detect_format("test.json") == "json"

    def test_detect_format_csv(self):
        """Test format detection for CSV files."""
        assert GraphIOHandler.detect_format("test.csv") == "csv"

    def test_detect_format_yaml(self):
        """Test format detection for YAML files."""
        assert GraphIOHandler.detect_format("test.yaml") == "yaml"
        assert GraphIOHandler.detect_format("test.yml") == "yaml"

    def test_detect_format_edgelist(self):
        """Test format detection for edge list files."""
        assert GraphIOHandler.detect_format("test.edges") == "edgelist"
        assert GraphIOHandler.detect_format("test.txt") == "edgelist"

    def test_detect_format_pickle(self):
        """Test format detection for pickle files."""
        assert GraphIOHandler.detect_format("test.pickle") == "pickle"
        assert GraphIOHandler.detect_format("test.pkl") == "pickle"

    def test_detect_format_unknown(self):
        """Test format detection for unknown extensions."""
        assert GraphIOHandler.detect_format("test.unknown") == "edgelist"  # default

    def test_detect_format_with_path_object(self):
        """Test format detection with Path objects."""
        path = Path("test.json")
        assert GraphIOHandler.detect_format(path) == "json"

    # ===== JSON EXPORT/IMPORT TESTS =====

    def test_export_json_pretty_print(self):
        """Test JSON export with pretty printing."""
        result = GraphIOHandler._export_json(self.simple_graph, pretty_print=True)

        assert isinstance(result, dict)
        assert "graph" in result
        assert "nodes" in result["graph"]
        assert "edges" in result["graph"]
        assert len(result["graph"]["nodes"]) == 2
        assert len(result["graph"]["edges"]) == 1

    def test_export_json_compact(self):
        """Test JSON export without pretty printing."""
        result = GraphIOHandler._export_json(self.simple_graph, pretty_print=False)

        assert isinstance(result, dict)
        assert "graph" in result

    def test_export_json_directed_graph(self):
        """Test JSON export of directed graph."""
        result = GraphIOHandler._export_json(self.directed_graph)

        assert result["graph"]["directed"] is True
        assert len(result["graph"]["nodes"]) == 3
        assert len(result["graph"]["edges"]) == 3

    def test_import_json_from_data(self):
        """Test JSON import from data dictionary."""
        json_data = GraphIOHandler._export_json(self.simple_graph)
        imported_graph = GraphIOHandler._import_json(data=json_data)

        assert imported_graph.number_of_nodes() == 2
        assert imported_graph.number_of_edges() == 1
        assert 1 in imported_graph.nodes()
        assert 2 in imported_graph.nodes()

    def test_import_json_from_file(self):
        """Test JSON import from file."""
        json_data = GraphIOHandler._export_json(self.simple_graph)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            imported_graph = GraphIOHandler._import_json(data=None, path=temp_path)
            assert imported_graph.number_of_nodes() == 2
            assert imported_graph.number_of_edges() == 1
        finally:
            Path(temp_path).unlink()

    def test_import_json_invalid_data(self):
        """Test JSON import with invalid data."""
        with pytest.raises((ValueError, KeyError)):
            GraphIOHandler._import_json(data={"invalid": "format"})

    # ===== YAML EXPORT/IMPORT TESTS =====

    def test_export_yaml(self):
        """Test YAML export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            result = GraphIOHandler._export_yaml(self.simple_graph, temp_path)
            assert isinstance(result, str)

            # Verify file was created and contains valid YAML
            with open(temp_path, "r") as f:
                yaml_data = yaml.safe_load(f)
                assert "graph" in yaml_data
        finally:
            Path(temp_path).unlink()

    def test_import_yaml_from_file(self):
        """Test YAML import from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            # Export then import
            GraphIOHandler._export_yaml(self.simple_graph, temp_path)
            imported_graph = GraphIOHandler._import_yaml(data=None, path=temp_path)

            assert imported_graph.number_of_nodes() == 2
            assert imported_graph.number_of_edges() == 1
        finally:
            Path(temp_path).unlink()

    def test_import_yaml_from_string(self):
        """Test YAML import from string data."""
        yaml_data = """
        graph:
          directed: false
          nodes:
            - id: 1
              label: "Node1"
            - id: 2
              label: "Node2"
          edges:
            - source: 1
              target: 2
              weight: 1.0
        """
        imported_graph = GraphIOHandler._import_yaml(data=yaml_data, path=None)

        assert imported_graph.number_of_nodes() == 2
        assert imported_graph.number_of_edges() == 1

    # ===== CSV EXPORT/IMPORT TESTS =====

    def test_export_csv_basic(self):
        """Test basic CSV export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            result = GraphIOHandler._export_csv(self.simple_graph, temp_path)
            assert isinstance(result, str)

            # Verify file was created
            assert Path(temp_path).exists()

            # Check content
            with open(temp_path, "r") as f:
                content = f.read()
                assert "source" in content or "Source" in content
                assert "target" in content or "Target" in content
        finally:
            Path(temp_path).unlink()

    def test_export_csv_with_custom_columns(self):
        """Test CSV export with custom column names."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            GraphIOHandler._export_csv(
                self.simple_graph,
                temp_path,
                source_col="from",
                target_col="to",
                weight_col="edge_weight",
            )

            with open(temp_path, "r") as f:
                content = f.read()
                assert "from" in content
                assert "to" in content
                assert "edge_weight" in content
        finally:
            Path(temp_path).unlink()

    def test_import_csv_basic(self):
        """Test basic CSV import."""
        # Create test CSV content
        csv_content = "source,target,weight\\n1,2,0.5\\n2,3,1.0\\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            imported_graph = GraphIOHandler._import_csv(temp_path)
            assert imported_graph.number_of_nodes() >= 2
            assert imported_graph.number_of_edges() >= 1
        finally:
            Path(temp_path).unlink()

    def test_csv_to_edge_list(self):
        """Test CSV to edge list conversion."""
        # Create test CSV
        csv_content = "from,to,weight\\n1,2,0.8\\n2,3,1.2\\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            edge_list = GraphIOHandler.csv_to_edge_list(
                temp_path, source_col="from", target_col="to", weight_col="weight"
            )

            assert len(edge_list) == 2
            assert len(edge_list[0]) == 3  # (source, target, weight)
            assert edge_list[0] == ("1", "2", 0.8) or edge_list[0] == (1, 2, 0.8)
        finally:
            Path(temp_path).unlink()

    def test_csv_to_edge_list_no_weight(self):
        """Test CSV to edge list conversion without weight column."""
        csv_content = "source,target\\nA,B\\nB,C\\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            edge_list = GraphIOHandler.csv_to_edge_list(
                temp_path, source_col="source", target_col="target"
            )

            assert len(edge_list) == 2
            assert len(edge_list[0]) == 2  # (source, target)
        finally:
            Path(temp_path).unlink()

    # ===== DATAFRAME CONVERSION TESTS =====

    def test_dataframe_to_graph_basic(self):
        """Test basic dataframe to graph conversion."""
        df = pd.DataFrame(
            {"source": [1, 2, 3], "target": [2, 3, 1], "weight": [0.5, 1.0, 1.5]}
        )

        graph = GraphIOHandler.dataframe_to_graph(df, "source", "target")

        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 3

    def test_dataframe_to_graph_with_edge_attributes(self):
        """Test dataframe to graph with edge attributes."""
        df = pd.DataFrame(
            {
                "from": ["A", "B"],
                "to": ["B", "C"],
                "weight": [1.0, 2.0],
                "type": ["friend", "family"],
            }
        )

        graph = GraphIOHandler.dataframe_to_graph(
            df, "from", "to", edge_attr=["weight", "type"]
        )

        assert graph.number_of_edges() == 2
        # Check if edge attributes are preserved
        edge_data = graph.get_edge_data("A", "B")
        assert edge_data is not None

    def test_dataframe_to_graph_with_node_attributes(self):
        """Test dataframe to graph with node attributes."""
        edge_df = pd.DataFrame({"source": [1, 2], "target": [2, 3]})

        node_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "label": ["Node1", "Node2", "Node3"],
                "size": [10, 20, 30],
            }
        )

        graph = GraphIOHandler.dataframe_to_graph(
            edge_df, "source", "target", node_attr_df=node_df, node_key="id"
        )

        assert graph.number_of_nodes() == 3
        assert graph.nodes[1]["label"] == "Node1"
        assert graph.nodes[2]["size"] == 20

    def test_dataframe_to_graph_directed(self):
        """Test dataframe to directed graph conversion."""
        df = pd.DataFrame({"source": [1, 2], "target": [2, 1]})

        graph = GraphIOHandler.dataframe_to_graph(
            df, "source", "target", create_using=nx.DiGraph()
        )

        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_edges() == 2

    # ===== ADJACENCY MATRIX TESTS =====

    def test_adjacency_to_edge_list_basic(self):
        """Test adjacency matrix to edge list conversion."""
        matrix = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

        edge_list = GraphIOHandler.adjacency_to_edge_list(matrix)

        assert len(edge_list) >= 2  # At least some edges
        # Check edge format
        if edge_list:
            assert len(edge_list[0]) == 3  # (source, target, weight)

    def test_adjacency_to_edge_list_with_labels(self):
        """Test adjacency matrix with custom node labels."""
        matrix = [[0, 1], [1, 0]]
        labels = ["A", "B"]

        edge_list = GraphIOHandler.adjacency_to_edge_list(matrix, node_labels=labels)

        assert len(edge_list) == 2  # Undirected: A-B appears twice
        # Check that labels are used
        found_A = any("A" in str(edge[0]) or "A" in str(edge[1]) for edge in edge_list)
        assert found_A

    def test_adjacency_to_edge_list_directed(self):
        """Test adjacency matrix for directed graph."""
        matrix = [[0, 1], [0, 0]]  # A -> B only

        edge_list = GraphIOHandler.adjacency_to_edge_list(matrix, directed=True)

        assert len(edge_list) == 1  # Only one directed edge

    def test_adjacency_to_edge_list_with_threshold(self):
        """Test adjacency matrix with weight threshold."""
        matrix = [[0, 0.3, 0.8], [0.3, 0, 0.2], [0.8, 0.2, 0]]

        # Only edges with weight > 0.5
        edge_list = GraphIOHandler.adjacency_to_edge_list(matrix, threshold=0.5)

        # Should only include the 0.8 weight edge
        assert len(edge_list) == 2  # Undirected edge appears twice

    def test_adjacency_to_edge_list_numpy_array(self):
        """Test adjacency matrix with numpy array."""
        matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        edge_list = GraphIOHandler.adjacency_to_edge_list(matrix)

        assert len(edge_list) >= 2

    # ===== HIGH-LEVEL EXPORT/IMPORT TESTS =====

    def test_export_graph_json_format(self):
        """Test high-level graph export to JSON."""
        result = GraphIOHandler.export_graph(self.simple_graph, "json")

        assert isinstance(result, dict)
        assert "graph" in result

    def test_export_graph_to_file(self):
        """Test exporting graph to file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            GraphIOHandler.export_graph(self.simple_graph, "json", path=temp_path)
            assert Path(temp_path).exists()
        finally:
            Path(temp_path).unlink()

    def test_export_graph_unsupported_format(self):
        """Test export with unsupported format."""
        with pytest.raises(ValueError):
            GraphIOHandler.export_graph(self.simple_graph, "unsupported_format")

    def test_import_graph_auto_detect(self):
        """Test graph import with auto-detection."""
        # First export to get valid data
        json_data = GraphIOHandler.export_graph(self.simple_graph, "json")

        # Then import with auto-detection
        imported_graph = GraphIOHandler.import_graph(
            input_format=None, data=json_data, auto_detect=True
        )

        assert imported_graph.number_of_nodes() == 2
        assert imported_graph.number_of_edges() == 1

    def test_import_graph_from_file_path(self):
        """Test importing graph from file path."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Export to file first
            GraphIOHandler.export_graph(self.simple_graph, "json", path=temp_path)

            # Import from file path
            imported_graph = GraphIOHandler.import_graph(path=temp_path)

            assert imported_graph.number_of_nodes() == 2
            assert imported_graph.number_of_edges() == 1
        finally:
            Path(temp_path).unlink()

    def test_import_graph_explicit_format(self):
        """Test importing graph with explicit format specification."""
        json_data = GraphIOHandler.export_graph(self.simple_graph, "json")

        imported_graph = GraphIOHandler.import_graph(
            input_format="json", data=json_data
        )

        assert imported_graph.number_of_nodes() == 2

    def test_import_graph_invalid_format(self):
        """Test import with invalid format."""
        with pytest.raises(ValueError):
            GraphIOHandler.import_graph(input_format="invalid_format", data={})

    def test_import_graph_no_data_no_path(self):
        """Test import with neither data nor path provided."""
        with pytest.raises(ValueError):
            GraphIOHandler.import_graph()

    # ===== STREAMING EXPORT TESTS =====

    def test_export_for_streaming_json(self):
        """Test streaming export for JSON format."""
        from io import StringIO

        output_stream = StringIO()
        bytes_written = GraphIOHandler.export_for_streaming(
            self.simple_graph, "json", output_stream, chunk_size=100
        )

        assert bytes_written > 0

        # Check that valid JSON was written
        output_stream.seek(0)
        content = output_stream.getvalue()
        assert len(content) > 0
        # Should be valid JSON or JSON lines

    def test_export_for_streaming_csv(self):
        """Test streaming export for CSV format."""
        from io import StringIO

        output_stream = StringIO()
        bytes_written = GraphIOHandler.export_for_streaming(
            self.simple_graph, "csv", output_stream
        )

        assert bytes_written > 0

        output_stream.seek(0)
        content = output_stream.getvalue()
        assert "source" in content or "target" in content

    def test_export_for_streaming_large_graph(self):
        """Test streaming export with large graph."""
        from io import StringIO

        # Create larger test graph
        large_graph = nx.erdos_renyi_graph(100, 0.1)

        output_stream = StringIO()
        bytes_written = GraphIOHandler.export_for_streaming(
            large_graph, "json", output_stream, chunk_size=10
        )

        assert bytes_written > 0

    def test_export_for_streaming_unsupported_format(self):
        """Test streaming export with unsupported format."""
        from io import StringIO

        output_stream = StringIO()

        with pytest.raises(ValueError):
            GraphIOHandler.export_for_streaming(
                self.simple_graph, "unsupported", output_stream
            )

    # ===== FORMAT-SPECIFIC IMPORT/EXPORT TESTS =====

    @patch("networkx.write_graphml")
    def test_export_graphml(self, mock_write):
        """Test GraphML export."""
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            temp_path = f.name

        try:
            result = GraphIOHandler._export_graphml(self.simple_graph, temp_path)
            mock_write.assert_called_once()
            assert isinstance(result, str)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch("networkx.read_graphml")
    def test_import_graphml(self, mock_read):
        """Test GraphML import."""
        mock_read.return_value = self.simple_graph

        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            temp_path = f.name

        try:
            imported_graph = GraphIOHandler._import_graphml(temp_path)
            mock_read.assert_called_once_with(temp_path)
            assert imported_graph.number_of_nodes() == 2
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch("networkx.write_gexf")
    def test_export_gexf(self, mock_write):
        """Test GEXF export."""
        with tempfile.NamedTemporaryFile(suffix=".gexf", delete=False) as f:
            temp_path = f.name

        try:
            result = GraphIOHandler._export_gexf(self.simple_graph, temp_path)
            mock_write.assert_called_once()
            assert isinstance(result, str)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch("networkx.read_gexf")
    def test_import_gexf(self, mock_read):
        """Test GEXF import."""
        mock_read.return_value = self.simple_graph

        with tempfile.NamedTemporaryFile(suffix=".gexf", delete=False) as f:
            temp_path = f.name

        try:
            imported_graph = GraphIOHandler._import_gexf(temp_path)
            mock_read.assert_called_once_with(temp_path)
            assert imported_graph.number_of_nodes() == 2
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_export_edgelist_to_file(self):
        """Test edge list export to file."""
        with tempfile.NamedTemporaryFile(suffix=".edges", delete=False) as f:
            temp_path = f.name

        try:
            result = GraphIOHandler._export_edgelist(self.simple_graph, temp_path)
            assert isinstance(result, str)
            assert Path(temp_path).exists()

            # Check file content
            with open(temp_path, "r") as f:
                content = f.read()
                assert "1" in content and "2" in content  # Nodes should be in edge list
        finally:
            Path(temp_path).unlink()

    def test_export_edgelist_to_memory(self):
        """Test edge list export to memory (no path)."""
        result = GraphIOHandler._export_edgelist(self.simple_graph, path=None)

        assert isinstance(result, list)
        assert len(result) == 1  # One edge
        assert isinstance(result[0], dict)

    @patch("networkx.read_edgelist")
    def test_import_edgelist(self, mock_read):
        """Test edge list import."""
        mock_read.return_value = self.simple_graph

        with tempfile.NamedTemporaryFile(suffix=".edges", delete=False) as f:
            temp_path = f.name

        try:
            imported_graph = GraphIOHandler._import_edgelist(temp_path)
            mock_read.assert_called_once()
            assert imported_graph.number_of_nodes() == 2
        finally:
            Path(temp_path).unlink(missing_ok=True)

    # ===== ERROR HANDLING TESTS =====

    def test_export_graph_invalid_path(self):
        """Test export with invalid path."""
        with pytest.raises((OSError, FileNotFoundError, PermissionError)):
            GraphIOHandler.export_graph(
                self.simple_graph, "json", path="/invalid/path/that/does/not/exist.json"
            )

    def test_import_graph_nonexistent_file(self):
        """Test import from non-existent file."""
        with pytest.raises(FileNotFoundError):
            GraphIOHandler.import_graph(path="/does/not/exist.json")

    def test_csv_to_edge_list_invalid_columns(self):
        """Test CSV edge list with invalid column names."""
        csv_content = "col1,col2\\n1,2\\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            with pytest.raises((KeyError, ValueError)):
                GraphIOHandler.csv_to_edge_list(
                    temp_path, source_col="nonexistent", target_col="also_nonexistent"
                )
        finally:
            Path(temp_path).unlink()

    def test_dataframe_to_graph_missing_columns(self):
        """Test dataframe conversion with missing columns."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        with pytest.raises(KeyError):
            GraphIOHandler.dataframe_to_graph(
                df, "source", "target"
            )  # Columns don't exist

    def test_adjacency_to_edge_list_invalid_matrix(self):
        """Test adjacency conversion with invalid matrix."""
        # Non-square matrix
        invalid_matrix = [[1, 2, 3], [4, 5]]

        with pytest.raises((ValueError, IndexError)):
            GraphIOHandler.adjacency_to_edge_list(invalid_matrix)
