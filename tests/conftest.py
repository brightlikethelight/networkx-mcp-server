"""Shared pytest fixtures for NetworkX MCP Server tests."""

import csv
import json
import tempfile

import networkx as nx
import numpy as np
import pytest

from networkx_mcp.core.graph_operations import GraphManager
from networkx_mcp.server import mcp


@pytest.fixture
def graph_manager():
    """Provide a fresh GraphManager instance for each test."""
    return GraphManager()


@pytest.fixture
def sample_graphs():
    """Provide various test graphs for testing different scenarios."""
    graphs = {}

    # Small simple graph
    graphs["simple"] = nx.Graph()
    graphs["simple"].add_edges_from([
        ("A", "B", {"weight": 1}),
        ("B", "C", {"weight": 2}),
        ("C", "D", {"weight": 1}),
        ("D", "A", {"weight": 3})
    ])

    # Directed graph
    graphs["directed"] = nx.DiGraph()
    graphs["directed"].add_edges_from([
        ("start", "middle", {"capacity": 10}),
        ("middle", "end", {"capacity": 5}),
        ("start", "end", {"capacity": 15})
    ])

    # Weighted graph with attributes
    graphs["weighted"] = nx.Graph()
    for i in range(6):
        graphs["weighted"].add_node(i, color=["red", "blue", "green"][i % 3])
    graphs["weighted"].add_weighted_edges_from([
        (0, 1, 2.5), (1, 2, 1.8), (2, 3, 3.2),
        (3, 4, 1.1), (4, 5, 2.9), (5, 0, 1.7)
    ])

    # Bipartite graph
    graphs["bipartite"] = nx.Graph()
    graphs["bipartite"].add_nodes_from([1, 2, 3, 4], bipartite=0)
    graphs["bipartite"].add_nodes_from(["a", "b", "c"], bipartite=1)
    graphs["bipartite"].add_edges_from([
        (1, "a"), (1, "b"), (2, "b"), (2, "c"),
        (3, "a"), (3, "c"), (4, "b")
    ])

    # Large graph for performance testing
    np.random.seed(42)
    graphs["large"] = nx.erdos_renyi_graph(1000, 0.01)

    # Complete graph
    graphs["complete"] = nx.complete_graph(5)

    # Tree graph
    graphs["tree"] = nx.balanced_tree(3, 3)

    # Multi-component graph
    graphs["multi_component"] = nx.Graph()
    # Component 1
    graphs["multi_component"].add_edges_from([
        ("A1", "A2"), ("A2", "A3"), ("A3", "A1")
    ])
    # Component 2
    graphs["multi_component"].add_edges_from([
        ("B1", "B2"), ("B2", "B3")
    ])
    # Isolated node
    graphs["multi_component"].add_node("C1")

    # Karate club graph (famous test graph)
    graphs["karate"] = nx.karate_club_graph()

    # Scale-free graph
    graphs["scale_free"] = nx.barabasi_albert_graph(100, 3)

    return graphs


@pytest.fixture
def sample_graph_data():
    """Provide sample graph data in various formats."""
    data = {}

    # Node-link format
    data["node_link"] = {
        "nodes": [
            {"id": "A", "color": "red"},
            {"id": "B", "color": "blue"},
            {"id": "C", "color": "green"}
        ],
        "links": [
            {"source": "A", "target": "B", "weight": 1.5},
            {"source": "B", "target": "C", "weight": 2.0}
        ]
    }

    # Adjacency format
    data["adjacency"] = {
        "A": {"B": {"weight": 1.5}},
        "B": {"C": {"weight": 2.0}},
        "C": {}
    }

    # Edge list format
    data["edge_list"] = [
        {"source": "X", "target": "Y", "weight": 0.8},
        {"source": "Y", "target": "Z", "weight": 1.2},
        {"source": "Z", "target": "X", "weight": 0.5}
    ]

    return data


@pytest.fixture
def temp_files():
    """Provide temporary files for testing I/O operations."""
    files = {}

    # Create temporary CSV file
    csv_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["source", "target", "weight"])
    csv_writer.writerow(["node1", "node2", 1.5])
    csv_writer.writerow(["node2", "node3", 2.0])
    csv_writer.writerow(["node3", "node1", 1.0])
    csv_file.close()
    files["csv"] = csv_file.name

    # Create temporary JSON file
    json_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({
        "nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "links": [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}]
    }, json_file)
    json_file.close()
    files["json"] = json_file.name

    # Create temporary GraphML file
    graphml_file = tempfile.NamedTemporaryFile(mode="w", suffix=".graphml", delete=False)
    graphml_content = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>
  <graph id="G" edgedefault="undirected">
    <node id="n0"/>
    <node id="n1"/>
    <edge id="e0" source="n0" target="n1">
      <data key="weight">1.0</data>
    </edge>
  </graph>
</graphml>"""
    graphml_file.write(graphml_content)
    graphml_file.close()
    files["graphml"] = graphml_file.name

    yield files

    # Cleanup
    import os
    for filepath in files.values():
        try:
            os.unlink(filepath)
        except OSError:
            pass


@pytest.fixture
def mock_mcp_server():
    """Provide a mock MCP server for testing tool functionality."""
    # Return the actual server instance for testing
    return mcp


@pytest.fixture
def performance_thresholds():
    """Define performance thresholds for various operations."""
    return {
        "create_graph": 0.001,  # seconds
        "add_nodes_1000": 0.1,
        "add_edges_1000": 0.1,
        "shortest_path_100": 0.05,
        "centrality_100": 0.1,
        "community_detection_100": 0.5,
        "visualization_100": 2.0,
        "memory_per_node": 1000,  # bytes
        "memory_per_edge": 200,   # bytes
    }


@pytest.fixture
def error_scenarios():
    """Provide common error scenarios for testing."""
    scenarios = {
        "invalid_graph_ids": [
            "",  # Empty
            "graph with spaces",
            "graph@invalid",
            "a" * 256,  # Too long
            123,  # Not string
            None,
            [],
        ],
        "invalid_node_ids": [
            None,
            {},
            [],
            lambda x: x,  # Function
        ],
        "invalid_file_paths": [
            "nonexistent.json",
            "/invalid/path/file.txt",
            "",
            None,
        ],
        "malformed_data": [
            {"nodes": "not_a_list"},
            {"edges": [{"source": "A"}]},  # Missing target
            {"invalid": "structure"},
            None,
            "not_a_dict",
        ]
    }
    return scenarios


@pytest.fixture
def algorithm_test_cases():
    """Provide test cases for algorithm validation."""
    cases = {
        "centrality_measures": [
            "degree", "betweenness", "closeness",
            "eigenvector", "pagerank"
        ],
        "shortest_path_methods": [
            "dijkstra", "bellman-ford"
        ],
        "community_algorithms": [
            "louvain", "label_propagation", "greedy_modularity"
        ],
        "layout_algorithms": [
            "spring", "circular", "random", "shell",
            "spectral", "kamada_kawai", "planar"
        ],
        "graph_generators": [
            "erdos_renyi", "barabasi_albert", "watts_strogatz",
            "complete", "cycle", "path", "star"
        ]
    }
    return cases


@pytest.fixture(scope="session")
def benchmark_data():
    """Create benchmark data for performance comparison."""
    data = {}

    # Graph size benchmarks
    sizes = [10, 50, 100, 500, 1000]
    for size in sizes:
        graph = nx.erdos_renyi_graph(size, 0.1, seed=42)
        data[f"erdos_renyi_{size}"] = graph

    # Different graph types
    data["complete_100"] = nx.complete_graph(100)
    data["path_1000"] = nx.path_graph(1000)
    data["star_1000"] = nx.star_graph(1000)
    data["grid_30x30"] = nx.grid_2d_graph(30, 30)

    return data


@pytest.fixture
def mcp_tool_params():
    """Provide test parameters for all 39 MCP tools."""
    return {
        # Core operations
        "create_graph": {
            "graph_id": "test_graph",
            "graph_type": "Graph",
            "params": {"name": "Test Graph"}
        },
        "add_nodes": {
            "graph_id": "test_graph",
            "nodes": ["A", "B", "C"]
        },
        "add_edges": {
            "graph_id": "test_graph",
            "edges": [("A", "B"), ("B", "C")]
        },
        "shortest_path": {
            "graph_id": "test_graph",
            "source": "A",
            "target": "C"
        },
        "centrality_measures": {
            "graph_id": "test_graph",
            "measures": ["degree", "betweenness"]
        },
        # Add more tool parameters as needed
    }


@pytest.fixture(autouse=True)
def cleanup_graphs(graph_manager):
    """Automatically cleanup graphs after each test."""
    yield
    # Clear all graphs after test
    graph_manager.graphs.clear()


@pytest.fixture
def validation_graphs():
    """Provide graphs specifically for validation testing."""
    graphs = {}

    # Empty graph
    graphs["empty"] = nx.Graph()

    # Single node
    graphs["single_node"] = nx.Graph()
    graphs["single_node"].add_node("alone")

    # Disconnected graph
    graphs["disconnected"] = nx.Graph()
    graphs["disconnected"].add_edges_from([("A", "B"), ("C", "D")])

    # Graph with self-loops
    graphs["self_loops"] = nx.DiGraph()
    graphs["self_loops"].add_edges_from([("A", "A"), ("A", "B"), ("B", "B")])

    # MultiGraph
    graphs["multigraph"] = nx.MultiGraph()
    graphs["multigraph"].add_edges_from([("X", "Y"), ("X", "Y"), ("Y", "Z")])

    return graphs


@pytest.fixture
def visualization_test_data():
    """Provide data for visualization testing."""
    return {
        "small_graph": nx.karate_club_graph(),
        "node_colors": {i: "red" if i < 17 else "blue" for i in range(34)},
        "node_sizes": {i: (i + 1) * 10 for i in range(34)},
        "edge_weights": {edge: np.random.random() for edge in nx.karate_club_graph().edges()},
        "layout_options": {
            "spring": {"k": 1, "iterations": 50},
            "circular": {},
            "random": {"seed": 42}
        }
    }
