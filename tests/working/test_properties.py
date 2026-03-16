"""Property-based tests using Hypothesis for graph invariant verification."""

import pytest
from hypothesis import given, settings, strategies as st

from networkx_mcp.graph_cache import graphs
from networkx_mcp.handlers import (
    handle_add_edges,
    handle_add_nodes,
    handle_betweenness_centrality,
    handle_community_detection,
    handle_create_graph,
    handle_get_info,
    handle_shortest_path,
)


@pytest.fixture(autouse=True)
def _clean_graphs():
    graphs.clear()
    yield
    graphs.clear()


class TestGraphInvariants:
    @given(nodes=st.lists(st.integers(0, 100), min_size=1, max_size=50, unique=True))
    @settings(max_examples=20)
    def test_add_n_nodes_reports_n(self, nodes):
        """Adding N nodes results in get_info reporting N nodes."""
        graphs.clear()
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": nodes})
        info = handle_get_info({"graph": "g"})
        assert info["nodes"] == len(nodes)

    @given(nodes=st.lists(st.integers(0, 50), min_size=3, max_size=20, unique=True))
    @settings(max_examples=20)
    def test_centrality_values_in_range(self, nodes):
        """Betweenness centrality values are always in [0, 1]."""
        graphs.clear()
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": nodes})
        edges = [[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1)]
        handle_add_edges({"graph": "g", "edges": edges})
        result = handle_betweenness_centrality({"graph": "g"})
        for val in result["centrality"].values():
            assert 0.0 <= val <= 1.0

    @given(nodes=st.lists(st.integers(0, 30), min_size=4, max_size=15, unique=True))
    @settings(max_examples=15)
    def test_communities_partition_all_nodes(self, nodes):
        """Community detection returns a partition covering all nodes."""
        graphs.clear()
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": nodes})
        edges = [[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1)]
        handle_add_edges({"graph": "g", "edges": edges})
        result = handle_community_detection({"graph": "g"})
        all_community_nodes = set()
        for comm in result["communities"]:
            all_community_nodes.update(comm)
        assert all_community_nodes == set(nodes)

    @given(nodes=st.lists(st.integers(0, 30), min_size=3, max_size=10, unique=True))
    @settings(max_examples=15)
    def test_shortest_path_is_valid_walk(self, nodes):
        """Shortest path is always a valid walk in the graph."""
        graphs.clear()
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": nodes})
        edges = [[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1)]
        handle_add_edges({"graph": "g", "edges": edges})
        result = handle_shortest_path(
            {
                "graph": "g",
                "source": nodes[0],
                "target": nodes[-1],
            }
        )
        path = result["path"]
        assert path[0] == nodes[0]
        assert path[-1] == nodes[-1]
        g = graphs["g"]
        for i in range(len(path) - 1):
            assert g.has_edge(path[i], path[i + 1])
