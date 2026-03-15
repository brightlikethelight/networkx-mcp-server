"""Tests for the GraphManager class in core/graph_operations.py."""

import pytest

from networkx_mcp.core.graph_operations import GraphManager
from networkx_mcp.errors import (
    GraphAlreadyExistsError,
    GraphNotFoundError,
    ValidationError,
)


class TestGraphCreation:
    def setup_method(self):
        self.gm = GraphManager()

    def test_create_undirected_graph(self):
        result = self.gm.create_graph("g1")
        assert result["created"] is True
        assert result["graph_id"] == "g1"
        assert result["graph_type"] == "Graph"

    def test_create_directed_graph(self):
        result = self.gm.create_graph("g1", graph_type="DiGraph")
        assert result["graph_type"] == "DiGraph"
        assert self.gm.graphs["g1"].is_directed()

    def test_create_multigraph(self):
        result = self.gm.create_graph("g1", graph_type="MultiGraph")
        assert result["graph_type"] == "MultiGraph"
        assert self.gm.graphs["g1"].is_multigraph()

    def test_create_multidigraph(self):
        result = self.gm.create_graph("g1", graph_type="MultiDiGraph")
        assert result["graph_type"] == "MultiDiGraph"
        assert self.gm.graphs["g1"].is_directed()
        assert self.gm.graphs["g1"].is_multigraph()

    def test_create_duplicate_raises(self):
        self.gm.create_graph("g1")
        with pytest.raises(GraphAlreadyExistsError):
            self.gm.create_graph("g1")

    def test_create_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            self.gm.create_graph("g1", graph_type="BadType")

    def test_metadata_stored(self):
        self.gm.create_graph("g1")
        assert "g1" in self.gm.metadata
        assert "created_at" in self.gm.metadata["g1"]


class TestGraphAccess:
    def setup_method(self):
        self.gm = GraphManager()
        self.gm.create_graph("g1")

    def test_get_graph(self):
        g = self.gm.get_graph("g1")
        assert g is not None

    def test_get_nonexistent_raises(self):
        with pytest.raises(GraphNotFoundError):
            self.gm.get_graph("nonexistent")

    def test_delete_graph(self):
        result = self.gm.delete_graph("g1")
        assert result["deleted"] is True
        assert "g1" not in self.gm.graphs

    def test_delete_nonexistent_raises(self):
        with pytest.raises(GraphNotFoundError):
            self.gm.delete_graph("nonexistent")

    def test_list_graphs(self):
        self.gm.create_graph("g2", graph_type="DiGraph")
        result = self.gm.list_graphs()
        assert len(result) == 2
        ids = {r["graph_id"] for r in result}
        assert ids == {"g1", "g2"}

    def test_list_graphs_with_info(self):
        result = self.gm.list_graphs_with_info()
        assert "g1" in result
        assert "graph" in result["g1"]
        assert "metadata" in result["g1"]


class TestNodeOperations:
    def setup_method(self):
        self.gm = GraphManager()
        self.gm.create_graph("g1")

    def test_add_node(self):
        result = self.gm.add_node("g1", "A", color="red")
        assert result["added"] is True
        assert result["node_id"] == "A"
        g = self.gm.get_graph("g1")
        assert "A" in g.nodes
        assert g.nodes["A"]["color"] == "red"

    def test_add_nodes_from(self):
        result = self.gm.add_nodes_from("g1", [1, 2, 3])
        assert result["nodes_added"] == 3
        assert result["total_nodes"] == 3

    def test_remove_node(self):
        self.gm.add_node("g1", "A")
        result = self.gm.remove_node("g1", "A")
        assert result["removed"] is True

    def test_remove_nonexistent_node_raises(self):
        with pytest.raises(ValueError, match="not in graph"):
            self.gm.remove_node("g1", "nonexistent")

    def test_get_neighbors(self):
        self.gm.add_nodes_from("g1", [1, 2, 3])
        self.gm.add_edges_from("g1", [(1, 2), (1, 3)])
        neighbors = self.gm.get_neighbors("g1", 1)
        assert set(neighbors) == {2, 3}

    def test_get_neighbors_nonexistent_node(self):
        with pytest.raises(ValueError, match="not in graph"):
            self.gm.get_neighbors("g1", "nonexistent")


class TestEdgeOperations:
    def setup_method(self):
        self.gm = GraphManager()
        self.gm.create_graph("g1")
        self.gm.add_nodes_from("g1", [1, 2, 3, 4])

    def test_add_edge(self):
        result = self.gm.add_edge("g1", 1, 2, weight=5.0)
        assert result["added"] is True
        g = self.gm.get_graph("g1")
        assert g.has_edge(1, 2)
        assert g[1][2]["weight"] == 5.0

    def test_add_edges_from(self):
        result = self.gm.add_edges_from("g1", [(1, 2), (2, 3), (3, 4)])
        assert result["edges_added"] == 3
        assert result["total_edges"] == 3

    def test_remove_edge(self):
        self.gm.add_edge("g1", 1, 2)
        result = self.gm.remove_edge("g1", 1, 2)
        assert result["removed"] is True

    def test_remove_nonexistent_edge_raises(self):
        with pytest.raises(ValueError, match="not in graph"):
            self.gm.remove_edge("g1", 1, 2)


class TestGraphInfo:
    def setup_method(self):
        self.gm = GraphManager()
        self.gm.create_graph("g1")
        self.gm.add_nodes_from("g1", [1, 2, 3])
        self.gm.add_edges_from("g1", [(1, 2), (2, 3)])

    def test_get_graph_info(self):
        info = self.gm.get_graph_info("g1")
        assert info["num_nodes"] == 3
        assert info["num_edges"] == 2
        assert info["is_directed"] is False
        assert info["is_multigraph"] is False
        assert "density" in info
        assert "degree_stats" in info

    def test_graph_info_empty_graph(self):
        self.gm.create_graph("empty")
        info = self.gm.get_graph_info("empty")
        assert info["num_nodes"] == 0
        assert "degree_stats" not in info


class TestAttributeOperations:
    def setup_method(self):
        self.gm = GraphManager()
        self.gm.create_graph("g1")
        self.gm.add_nodes_from("g1", [1, 2])
        self.gm.add_edge("g1", 1, 2)

    def test_get_node_attributes_single(self):
        self.gm.set_node_attributes("g1", {1: {"color": "red"}})
        attrs = self.gm.get_node_attributes("g1", node_id=1)
        assert attrs["color"] == "red"

    def test_get_node_attributes_by_name(self):
        self.gm.set_node_attributes("g1", {1: "red", 2: "blue"}, name="color")
        attrs = self.gm.get_node_attributes("g1", attribute="color")
        assert attrs[1] == "red"
        assert attrs[2] == "blue"

    def test_get_all_node_attributes(self):
        attrs = self.gm.get_node_attributes("g1")
        assert isinstance(attrs, dict)

    def test_get_node_attributes_nonexistent_node(self):
        with pytest.raises(ValueError, match="not in graph"):
            self.gm.get_node_attributes("g1", node_id=99)

    def test_get_edge_attributes_single(self):
        self.gm.set_edge_attributes("g1", {(1, 2): {"weight": 5}})
        attrs = self.gm.get_edge_attributes("g1", edge=(1, 2))
        assert attrs["weight"] == 5

    def test_get_edge_attributes_by_name(self):
        self.gm.set_edge_attributes("g1", {(1, 2): 5}, name="weight")
        attrs = self.gm.get_edge_attributes("g1", attribute="weight")
        assert attrs[(1, 2)] == 5

    def test_get_all_edge_attributes(self):
        attrs = self.gm.get_edge_attributes("g1")
        assert isinstance(attrs, dict)

    def test_get_edge_attributes_nonexistent_edge(self):
        with pytest.raises(ValueError, match="not in graph"):
            self.gm.get_edge_attributes("g1", edge=(1, 99))

    def test_set_node_attributes(self):
        result = self.gm.set_node_attributes("g1", {1: {"color": "red"}})
        assert result["success"] is True
        assert result["nodes_updated"] == 1

    def test_set_edge_attributes(self):
        result = self.gm.set_edge_attributes("g1", {(1, 2): {"weight": 5}})
        assert result["success"] is True
        assert result["edges_updated"] == 1


class TestSubgraphAndClear:
    def setup_method(self):
        self.gm = GraphManager()
        self.gm.create_graph("g1")
        self.gm.add_nodes_from("g1", [1, 2, 3, 4])
        self.gm.add_edges_from("g1", [(1, 2), (2, 3), (3, 4)])

    def test_subgraph_copy(self):
        result = self.gm.subgraph("g1", [1, 2, 3])
        assert result["num_nodes"] == 3
        assert result["num_edges"] == 2

    def test_subgraph_view(self):
        result = self.gm.subgraph("g1", [1, 2], create_copy=False)
        # Returns a graph view, not a dict
        assert result.number_of_nodes() == 2

    def test_clear_graph(self):
        result = self.gm.clear_graph("g1")
        assert result["cleared"] is True
        assert result["num_nodes"] == 0
        g = self.gm.get_graph("g1")
        assert g.number_of_nodes() == 0
