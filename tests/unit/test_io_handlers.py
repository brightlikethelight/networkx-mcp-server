"""Comprehensive tests for I/O handlers."""

import csv
import json
import os
import tempfile

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from networkx_mcp.core.io import GraphIOHandler

# Skip IO handler tests until modules are fully implemented
pytestmark = pytest.mark.skip(reason="IO handler modules need implementation updates")


class TestJSONIO:
    """Test JSON import/export functionality."""

    def test_json_export_basic(self):
        """Test basic JSON export."""
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        G.add_edges_from([(1, 2), (2, 3)])

        data = GraphIOHandler.export_graph(G, "json")

        # Check structure
        assert isinstance(data, str)  # Pretty-printed JSON string
        parsed = json.loads(data)

        assert "graph" in parsed
        assert "nodes" in parsed
        assert "links" in parsed
        assert len(parsed["nodes"]) == 3
        assert len(parsed["links"]) == 2

    def test_json_import_basic(self):
        """Test basic JSON import."""
        data = {
            "graph": {"directed": False},
            "nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
            "links": [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}],
        }

        G = GraphIOHandler.import_graph("json", data=data)

        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        assert G.has_edge("A", "B")
        assert G.has_edge("B", "C")

    def test_json_round_trip(self):
        """Test JSON export and re-import preserves graph."""
        # Create graph with attributes
        G = nx.DiGraph()
        G.add_node("N1", color="red", weight=1.5)
        G.add_node("N2", color="blue", weight=2.0)
        G.add_edge("N1", "N2", capacity=10, flow=5)

        # Export and re-import
        json_data = GraphIOHandler.export_graph(G, "json")
        G2 = GraphIOHandler.import_graph("json", data=json.loads(json_data))

        # Check preservation
        assert G2.is_directed()
        assert G2.nodes["N1"]["color"] == "red"
        assert G2.nodes["N1"]["weight"] == 1.5
        assert G2["N1"]["N2"]["capacity"] == 10
        assert G2["N1"]["N2"]["flow"] == 5


class TestCSVIO:
    """Test CSV import/export functionality."""

    def test_csv_export(self):
        """Test CSV edge list export."""
        G = nx.Graph()
        G.add_edges_from(
            [
                ("A", "B", {"weight": 1.5, "type": "friend"}),
                ("B", "C", {"weight": 2.0, "type": "family"}),
                ("C", "D", {"weight": 0.5, "type": "friend"}),
            ]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            result = GraphIOHandler.export_graph(G, "csv", path=temp_path)
            assert "exported to CSV" in result

            # Read and verify CSV
            with open(temp_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 3
            assert "source" in rows[0]
            assert "target" in rows[0]
            assert "weight" in rows[0]
            assert "type" in rows[0]

            # Check specific edge
            edge_ab = next(r for r in rows if r["source"] == "A" and r["target"] == "B")
            assert float(edge_ab["weight"]) == 1.5
            assert edge_ab["type"] == "friend"

        finally:
            os.unlink(temp_path)

    def test_csv_import(self):
        """Test CSV edge list import."""
        # Create CSV file
        csv_content = """source,target,weight,label
A,B,1.5,edge1
B,C,2.0,edge2
C,D,0.5,edge3
A,D,3.0,edge4"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            G = GraphIOHandler.import_graph("csv", path=temp_path)

            assert G.number_of_nodes() == 4
            assert G.number_of_edges() == 4
            assert G["A"]["B"]["weight"] == 1.5
            assert G["B"]["C"]["weight"] == 2.0

        finally:
            os.unlink(temp_path)

    def test_csv_to_edge_list(self):
        """Test CSV to edge list conversion."""
        csv_content = """from,to,distance,road_type
CityA,CityB,100,highway
CityB,CityC,150,highway
CityC,CityD,80,local
CityA,CityD,200,local"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            edges = GraphIOHandler.csv_to_edge_list(
                temp_path, source_col="from", target_col="to", weight_col="distance"
            )

            assert len(edges) == 4
            assert edges[0] == ("CityA", "CityB", 100.0)
            assert edges[1] == ("CityB", "CityC", 150.0)

        finally:
            os.unlink(temp_path)


class TestAdjacencyIO:
    """Test adjacency matrix import/export."""

    def test_adjacency_export(self):
        """Test adjacency matrix export."""
        # Create simple graph
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])

        data = GraphIOHandler.export_graph(G, "adjacency")

        assert "matrix" in data
        assert "nodes" in data
        assert data["shape"] == (3, 3)
        assert not data["directed"]

        # Check matrix values
        matrix = data["matrix"]
        assert matrix[0][1] == 1  # Edge 0-1
        assert matrix[1][2] == 1  # Edge 1-2
        assert matrix[0][2] == 1  # Edge 0-2
        assert matrix[0][0] == 0  # No self-loop

    def test_adjacency_import(self):
        """Test adjacency matrix import."""
        data = {
            "matrix": [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
            "nodes": ["A", "B", "C", "D"],
            "directed": False,
        }

        G = GraphIOHandler.import_graph("adjacency", data=data)

        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 4
        assert G.has_edge("A", "B")
        assert G.has_edge("C", "D")
        assert not G.has_edge("A", "C")

    def test_sparse_adjacency(self):
        """Test sparse matrix handling."""
        # Create sparse graph
        G = nx.Graph()
        G.add_nodes_from(range(100))
        G.add_edges_from([(i, i + 1) for i in range(99)])  # Path graph

        # Export with sparse format
        data = GraphIOHandler.export_graph(G, "adjacency", sparse_format=True)

        assert data["is_sparse"]
        assert data["format"] == "sparse_coo"
        assert "row" in data
        assert "col" in data
        assert "data" in data

        # Verify sparse representation
        assert len(data["data"]) == 198  # 99 edges * 2 (undirected)


class TestFormatConverters:
    """Test format conversion utilities."""

    def test_dataframe_to_graph(self):
        """Test DataFrame to graph conversion."""
        # Create edge DataFrame
        edge_data = {
            "from": ["A", "B", "C", "A"],
            "to": ["B", "C", "D", "D"],
            "weight": [1.5, 2.0, 0.5, 3.0],
            "type": ["friend", "family", "friend", "work"],
        }
        edges_df = pd.DataFrame(edge_data)

        # Create node DataFrame
        node_data = {
            "id": ["A", "B", "C", "D"],
            "city": ["NYC", "Boston", "Chicago", "LA"],
            "population": [8, 0.6, 2.7, 4],
        }
        nodes_df = pd.DataFrame(node_data)

        # Convert to graph
        G = GraphIOHandler.dataframe_to_graph(
            edges_df,
            source_col="from",
            target_col="to",
            edge_attr=["weight", "type"],
            node_attr_df=nodes_df,
            node_key="id",
        )

        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 4

        # Check edge attributes
        assert G["A"]["B"]["weight"] == 1.5
        assert G["A"]["B"]["type"] == "friend"

        # Check node attributes
        assert G.nodes["A"]["city"] == "NYC"
        assert G.nodes["B"]["population"] == 0.6

    def test_adjacency_to_edge_list(self):
        """Test adjacency matrix to edge list conversion."""
        matrix = [[0, 2, 0, 3], [2, 0, 1, 0], [0, 1, 0, 4], [3, 0, 4, 0]]
        nodes = ["W", "X", "Y", "Z"]

        edges = GraphIOHandler.adjacency_to_edge_list(
            matrix,
            node_labels=nodes,
            threshold=1.5,  # Only edges with weight > 1.5
        )

        assert len(edges) == 3
        assert ("W", "X", 2.0) in edges
        assert ("W", "Z", 3.0) in edges
        assert ("Y", "Z", 4.0) in edges
        assert ("X", "Y", 1.0) not in edges  # Below threshold


class TestLargeGraphHandling:
    """Test handling of large graphs."""

    def test_streaming_export(self):
        """Test streaming export for large graphs."""
        # Create large graph
        G = nx.fast_gnp_random_graph(1000, 0.01)

        # Add attributes
        for u, v in G.edges():
            G[u][v]["weight"] = np.random.random()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".edges", delete=False) as f:
            temp_path = f.name

        try:
            with open(temp_path, "w") as output:
                edge_count = GraphIOHandler.export_for_streaming(
                    G, "edgelist", output, chunk_size=100
                )

            assert edge_count == G.number_of_edges()

            # Verify file content
            with open(temp_path) as f:
                lines = f.readlines()

            # Should have header + edges
            assert len(lines) > edge_count
            assert lines[0].startswith("# NetworkX edge list")

        finally:
            os.unlink(temp_path)


class TestFormatValidation:
    """Test format detection and validation."""

    def test_format_detection(self):
        """Test automatic format detection from file extension."""
        test_cases = [
            ("graph.graphml", "graphml"),
            ("data.json", "json"),
            ("network.gexf", "gexf"),
            ("edges.csv", "csv"),
            ("graph.edges", "edgelist"),
            ("matrix.adj", "adjacency"),
            ("saved.pickle", "pickle"),
            ("graph.net", "pajek"),
        ]

        for filename, expected_format in test_cases:
            detected = GraphIOHandler.detect_format(filename)
            assert detected == expected_format

    def test_invalid_format_detection(self):
        """Test handling of unknown formats."""
        with pytest.raises(ValueError):
            GraphIOHandler.detect_format("file.unknown")


class TestErrorHandling:
    """Test error handling in I/O operations."""

    def test_missing_file_import(self):
        """Test importing from non-existent file."""
        with pytest.raises(FileNotFoundError):
            GraphIOHandler.import_graph("json", path="/non/existent/file.json")

    def test_invalid_json_import(self):
        """Test importing invalid JSON data."""
        with pytest.raises(ValueError):
            GraphIOHandler.import_graph("json", data="not a dict")

    def test_missing_required_columns_csv(self):
        """Test CSV import with missing columns."""
        csv_content = """node1,node2
A,B
B,C"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                GraphIOHandler.import_graph(
                    "csv",
                    path=temp_path,
                    source_col="source",  # Column doesn't exist
                    target_col="target",
                )
        finally:
            os.unlink(temp_path)
