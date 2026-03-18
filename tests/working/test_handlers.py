"""Tests for handlers.py — direct handler calls and error paths."""

from unittest.mock import patch

import pytest

from networkx_mcp.errors import (
    EdgeNotFoundError,
    GraphAlreadyExistsError,
    GraphNotFoundError,
    GraphOperationError,
    MCPError,
    NodeNotFoundError,
    ResourceLimitExceededError,
)
from networkx_mcp.graph_cache import graphs
from networkx_mcp.handlers import (
    handle_add_edges,
    handle_add_nodes,
    handle_centrality_measures,
    handle_clustering_coefficients,
    handle_create_graph,
    handle_cycles_detection,
    handle_delete_graph,
    handle_export_json,
    handle_get_edge_attributes,
    handle_get_info,
    handle_get_neighbors,
    handle_get_node_attributes,
    handle_graph_coloring,
    handle_graph_statistics,
    handle_import_csv,
    handle_list_graphs,
    handle_matching,
    handle_maximum_flow,
    handle_merge_graphs,
    handle_minimum_spanning_tree,
    handle_remove_edges,
    handle_remove_nodes,
    handle_set_edge_attributes,
    handle_set_node_attributes,
    handle_shortest_path,
    handle_subgraph,
    handle_topological_sort,
    handle_degree_centrality,
    handle_betweenness_centrality,
    handle_connected_components,
    handle_pagerank,
    handle_community_detection,
    handle_visualize_graph,
    handle_trigger_workflow,
    handle_get_workflow_status,
    handle_cancel_workflow,
    handle_rerun_failed_jobs,
    handle_get_dora_metrics,
    handle_analyze_workflow_failures,
    make_health_handler,
)


@pytest.fixture(autouse=True)
def _clean_graphs():
    graphs.clear()
    yield
    graphs.clear()


# ═══════════════════════════════════════════════════════════════════════
# Sync graph management handlers
# ═══════════════════════════════════════════════════════════════════════


class TestSyncHandlers:
    def test_create_graph(self):
        result = handle_create_graph({"name": "g1"})
        assert result["created"] == "g1"
        assert result["type"] == "undirected"

    def test_create_graph_already_exists(self):
        handle_create_graph({"name": "g1"})
        with pytest.raises(GraphAlreadyExistsError):
            handle_create_graph({"name": "g1"})

    def test_create_directed_graph(self):
        result = handle_create_graph({"name": "g2", "directed": True})
        assert result["type"] == "directed"

    def test_add_nodes(self):
        handle_create_graph({"name": "g"})
        result = handle_add_nodes({"graph": "g", "nodes": [1, 2, 3]})
        assert result["added"] == 3
        assert result["total"] == 3

    def test_add_nodes_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_add_nodes({"graph": "nope", "nodes": [1]})

    def test_add_edges(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": [1, 2, 3]})
        result = handle_add_edges({"graph": "g", "edges": [[1, 2], [2, 3]]})
        assert result["added"] == 2
        assert result["total"] == 2

    def test_add_edges_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_add_edges({"graph": "nope", "edges": [[1, 2]]})

    def test_get_info(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": ["a", "b"]})
        result = handle_get_info({"graph": "g"})
        assert result["nodes"] == 2
        assert result["edges"] == 0
        assert result["directed"] is False

    def test_get_info_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_get_info({"graph": "nope"})

    def test_delete_graph_success(self):
        handle_create_graph({"name": "g"})
        result = handle_delete_graph({"graph": "g"})
        assert result["deleted"] == "g"

    def test_delete_graph_missing(self):
        with pytest.raises(GraphNotFoundError):
            handle_delete_graph({"graph": "nope"})

    def test_shortest_path(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": [1, 2, 3]})
        handle_add_edges({"graph": "g", "edges": [[1, 2], [2, 3]]})
        result = handle_shortest_path({"graph": "g", "source": 1, "target": 3})
        assert result["path"] == [1, 2, 3]
        assert result["length"] == 2

    def test_shortest_path_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_shortest_path({"graph": "nope", "source": 1, "target": 2})


# ═══════════════════════════════════════════════════════════════════════
# Algorithm handlers
# ═══════════════════════════════════════════════════════════════════════


class TestAlgorithmHandlers:
    @pytest.fixture(autouse=True)
    def _setup_graph(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": [1, 2, 3, 4, 5]})
        handle_add_edges(
            {"graph": "g", "edges": [[1, 2], [2, 3], [3, 4], [4, 5], [1, 5]]}
        )

    def test_degree_centrality(self):
        result = handle_degree_centrality({"graph": "g"})
        assert "centrality" in result

    def test_betweenness_centrality(self):
        result = handle_betweenness_centrality({"graph": "g"})
        assert "centrality" in result

    def test_connected_components(self):
        result = handle_connected_components({"graph": "g"})
        assert result["num_components"] == 1

    def test_pagerank(self):
        result = handle_pagerank({"graph": "g"})
        assert "pagerank" in result

    def test_community_detection(self):
        result = handle_community_detection({"graph": "g"})
        assert "communities" in result
        assert result["num_communities"] >= 1


# ═══════════════════════════════════════════════════════════════════════
# Advanced algorithm handlers
# ═══════════════════════════════════════════════════════════════════════


class TestAdvancedAlgorithmHandlers:
    """Tests for the 5 advanced algorithm handlers wired in 4e6e025."""

    @pytest.fixture(autouse=True)
    def _ring_graph(self):
        """Create a 5-node ring graph for algorithm tests."""
        handle_create_graph({"name": "ring"})
        handle_add_nodes({"graph": "ring", "nodes": [1, 2, 3, 4, 5]})
        handle_add_edges(
            {"graph": "ring", "edges": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1]]}
        )

    # ── clustering_coefficients ──────────────────────────────────────

    def test_clustering_coefficients(self):
        result = handle_clustering_coefficients({"graph": "ring"})
        assert "node_clustering" in result
        assert "average_clustering" in result
        assert isinstance(result["average_clustering"], float)

    def test_clustering_coefficients_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_clustering_coefficients({"graph": "nope"})

    # ── graph_statistics ─────────────────────────────────────────────

    def test_graph_statistics(self):
        result = handle_graph_statistics({"graph": "ring"})
        assert result["num_nodes"] == 5
        assert result["density"] > 0
        # Verify numpy conversion — all degree_stats values must be float
        for v in result["degree_stats"].values():
            assert isinstance(v, float)

    def test_graph_statistics_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_graph_statistics({"graph": "nope"})

    # ── minimum_spanning_tree ────────────────────────────────────────

    def test_minimum_spanning_tree_kruskal(self):
        result = handle_minimum_spanning_tree({"graph": "ring"})
        assert "edges" in result
        assert result["algorithm"] == "kruskal"
        assert result["num_edges"] == 4  # MST of 5-node graph has 4 edges

    def test_minimum_spanning_tree_prim(self):
        result = handle_minimum_spanning_tree({"graph": "ring", "algorithm": "prim"})
        assert result["algorithm"] == "prim"
        assert result["num_edges"] == 4

    def test_minimum_spanning_tree_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_minimum_spanning_tree({"graph": "nope"})

    # ── cycles_detection ─────────────────────────────────────────────

    def test_cycles_detection(self):
        result = handle_cycles_detection({"graph": "ring"})
        assert result["has_cycle"] is True

    def test_cycles_detection_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_cycles_detection({"graph": "nope"})

    # ── graph_coloring ───────────────────────────────────────────────

    def test_graph_coloring(self):
        result = handle_graph_coloring({"graph": "ring"})
        assert result["num_colors"] >= 1
        assert "coloring" in result

    def test_graph_coloring_strategy(self):
        result = handle_graph_coloring({"graph": "ring", "strategy": "smallest_last"})
        assert result["num_colors"] >= 1

    def test_graph_coloring_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_graph_coloring({"graph": "nope"})


# ═══════════════════════════════════════════════════════════════════════
# I/O handlers
# ═══════════════════════════════════════════════════════════════════════


class TestIOHandlers:
    def test_import_csv(self):
        result = handle_import_csv(
            {"graph": "csv_g", "csv_data": "a,b\nc,d\n", "directed": False}
        )
        assert result["nodes"] == 4
        assert result["edges"] == 2

    def test_import_csv_already_exists(self):
        handle_create_graph({"name": "csv_g"})
        with pytest.raises(GraphAlreadyExistsError):
            handle_import_csv(
                {"graph": "csv_g", "csv_data": "a,b\n", "directed": False}
            )

    def test_export_json(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": [1, 2]})
        handle_add_edges({"graph": "g", "edges": [[1, 2]]})
        result = handle_export_json({"graph": "g"})
        assert result["format"] == "node-link"
        assert result["nodes"] == 2

    def test_visualize_graph(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": [1, 2]})
        handle_add_edges({"graph": "g", "edges": [[1, 2]]})
        result = handle_visualize_graph({"graph": "g"})
        assert "visualization" in result
        assert result["format"] == "png"


# ═══════════════════════════════════════════════════════════════════════
# Async CI/CD handler ImportError paths
# ═══════════════════════════════════════════════════════════════════════


class TestCICDHandlerImportErrors:
    """Test that async CI/CD handlers raise MCPError on missing module."""

    @pytest.mark.asyncio
    async def test_trigger_workflow_import_error(self):
        with patch.dict("sys.modules", {"networkx_mcp.tools": None}):
            with pytest.raises(MCPError):
                await handle_trigger_workflow({"workflow": "ci.yml"})

    @pytest.mark.asyncio
    async def test_get_workflow_status_import_error(self):
        with patch.dict("sys.modules", {"networkx_mcp.tools": None}):
            with pytest.raises(MCPError):
                await handle_get_workflow_status({})

    @pytest.mark.asyncio
    async def test_cancel_workflow_import_error(self):
        with patch.dict("sys.modules", {"networkx_mcp.tools": None}):
            with pytest.raises(MCPError):
                await handle_cancel_workflow({"run_id": "123"})

    @pytest.mark.asyncio
    async def test_rerun_failed_jobs_import_error(self):
        with patch.dict("sys.modules", {"networkx_mcp.tools": None}):
            with pytest.raises(MCPError):
                await handle_rerun_failed_jobs({"run_id": "123"})

    @pytest.mark.asyncio
    async def test_get_dora_metrics_import_error(self):
        with patch.dict("sys.modules", {"networkx_mcp.tools": None}):
            with pytest.raises(MCPError):
                await handle_get_dora_metrics({})

    @pytest.mark.asyncio
    async def test_analyze_workflow_failures_import_error(self):
        with patch.dict("sys.modules", {"networkx_mcp.tools": None}):
            with pytest.raises(MCPError):
                await handle_analyze_workflow_failures({"run_id": "123"})


# ═══════════════════════════════════════════════════════════════════════
# make_health_handler factory
# ═══════════════════════════════════════════════════════════════════════


class TestMakeHealthHandler:
    def test_returns_callable(self):
        handler = make_health_handler(None)
        assert callable(handler)

    def test_with_monitor(self):
        from unittest.mock import MagicMock

        monitor = MagicMock()
        monitor.get_health_status.return_value = {"status": "healthy"}
        handler = make_health_handler(monitor)
        result = handler({})
        assert result["status"] == "healthy"
        monitor.get_health_status.assert_called_once()

    def test_without_monitor(self):
        handler = make_health_handler(None)
        result = handler({})
        assert result["status"] == "monitoring_disabled"


# ═══════════════════════════════════════════════════════════════════════
# B1: list_graphs
# ═══════════════════════════════════════════════════════════════════════


class TestListGraphs:
    def test_empty(self):
        result = handle_list_graphs({})
        assert result["graphs"] == []
        assert result["total"] == 0

    def test_with_graphs(self):
        handle_create_graph({"name": "a"})
        handle_create_graph({"name": "b", "directed": True})
        handle_add_nodes({"graph": "a", "nodes": [1, 2]})
        result = handle_list_graphs({})
        assert result["total"] == 2
        names = {g["name"] for g in result["graphs"]}
        assert names == {"a", "b"}
        a_entry = next(g for g in result["graphs"] if g["name"] == "a")
        assert a_entry["nodes"] == 2
        assert a_entry["directed"] is False


# ═══════════════════════════════════════════════════════════════════════
# B2: remove_nodes / remove_edges
# ═══════════════════════════════════════════════════════════════════════


class TestRemoveOperations:
    @pytest.fixture(autouse=True)
    def _setup(self):
        handle_create_graph({"name": "r"})
        handle_add_nodes({"graph": "r", "nodes": [1, 2, 3, 4]})
        handle_add_edges({"graph": "r", "edges": [[1, 2], [2, 3], [3, 4]]})

    def test_remove_nodes(self):
        result = handle_remove_nodes({"graph": "r", "nodes": [4]})
        assert result["removed"] == 1
        assert result["total_nodes"] == 3

    def test_remove_nodes_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_remove_nodes({"graph": "nope", "nodes": [1]})

    def test_remove_nonexistent_node(self):
        # NetworkX silently ignores nonexistent nodes in remove_nodes_from
        result = handle_remove_nodes({"graph": "r", "nodes": [99]})
        assert result["total_nodes"] == 4

    def test_remove_edges(self):
        result = handle_remove_edges({"graph": "r", "edges": [[1, 2]]})
        assert result["removed"] == 1
        assert result["total_edges"] == 2

    def test_remove_edges_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_remove_edges({"graph": "nope", "edges": [[1, 2]]})


# ═══════════════════════════════════════════════════════════════════════
# B3: centrality_measures / matching / maximum_flow
# ═══════════════════════════════════════════════════════════════════════


class TestNewAlgorithmHandlers:
    @pytest.fixture(autouse=True)
    def _setup(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": [1, 2, 3, 4]})
        handle_add_edges({"graph": "g", "edges": [[1, 2], [2, 3], [3, 4], [1, 4]]})

    # ── centrality_measures ──────────────────────────────────────────

    def test_centrality_measures_default(self):
        result = handle_centrality_measures({"graph": "g"})
        assert "degree_centrality" in result
        assert "betweenness_centrality" in result

    def test_centrality_measures_subset(self):
        result = handle_centrality_measures({"graph": "g", "measures": ["closeness"]})
        assert "closeness_centrality" in result
        assert "degree_centrality" not in result

    def test_centrality_measures_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_centrality_measures({"graph": "nope"})

    # ── matching ─────────────────────────────────────────────────────

    def test_matching(self):
        result = handle_matching({"graph": "g"})
        assert "matching" in result
        assert result["matching_size"] >= 1

    def test_matching_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_matching({"graph": "nope"})

    # ── maximum_flow ─────────────────────────────────────────────────

    def test_maximum_flow(self):
        handle_create_graph({"name": "flow", "directed": True})
        handle_add_edges(
            {"graph": "flow", "edges": [["s", "a"], ["a", "t"], ["s", "t"]]}
        )
        # Add capacity attributes
        from networkx_mcp.graph_cache import graphs as g_cache

        for u, v in g_cache["flow"].edges():
            g_cache["flow"][u][v]["capacity"] = 10
        result = handle_maximum_flow({"graph": "flow", "source": "s", "sink": "t"})
        assert result["flow_value"] == 20
        assert result["source"] == "s"
        assert result["sink"] == "t"

    def test_maximum_flow_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_maximum_flow({"graph": "nope", "source": "s", "sink": "t"})

    def test_maximum_flow_undirected_raises(self):
        with pytest.raises(GraphOperationError):
            handle_maximum_flow({"graph": "g", "source": 1, "sink": 4})


# ═══════════════════════════════════════════════════════════════════════
# Bulk operation limits
# ═══════════════════════════════════════════════════════════════════════


class TestBulkLimits:
    def test_add_nodes_exceeds_limit(self):
        handle_create_graph({"name": "g"})
        nodes = list(range(100_001))
        with pytest.raises(ResourceLimitExceededError):
            handle_add_nodes({"graph": "g", "nodes": nodes})

    def test_add_edges_exceeds_limit(self):
        handle_create_graph({"name": "g"})
        edges = [[i, i + 1] for i in range(500_001)]
        with pytest.raises(ResourceLimitExceededError):
            handle_add_edges({"graph": "g", "edges": edges})

    def test_visualize_large_graph_rejected(self):
        handle_create_graph({"name": "g"})
        import networkx as nx

        graphs["g"] = nx.path_graph(10_001)
        with pytest.raises(ResourceLimitExceededError):
            handle_visualize_graph({"graph": "g"})

    def test_visualize_small_graph_succeeds(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": [1, 2, 3]})
        handle_add_edges({"graph": "g", "edges": [[1, 2], [2, 3]]})
        result = handle_visualize_graph({"graph": "g"})
        assert "visualization" in result


# ═══════════════════════════════════════════════════════════════════════
# get_neighbors
# ═══════════════════════════════════════════════════════════════════════


class TestGetNeighbors:
    @pytest.fixture(autouse=True)
    def _setup(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": ["a", "b", "c"]})
        handle_add_edges({"graph": "g", "edges": [["a", "b"], ["a", "c"]]})

    def test_happy_path(self):
        result = handle_get_neighbors({"graph": "g", "node": "a"})
        assert set(result["neighbors"]) == {"b", "c"}
        assert result["count"] == 2

    def test_leaf_node(self):
        result = handle_get_neighbors({"graph": "g", "node": "b"})
        assert result["neighbors"] == ["a"]

    def test_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_get_neighbors({"graph": "nope", "node": "a"})

    def test_missing_node(self):
        with pytest.raises(NodeNotFoundError):
            handle_get_neighbors({"graph": "g", "node": "z"})


# ═══════════════════════════════════════════════════════════════════════
# set/get node attributes
# ═══════════════════════════════════════════════════════════════════════


class TestNodeAttributes:
    @pytest.fixture(autouse=True)
    def _setup(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": ["a", "b"]})

    def test_set_and_get(self):
        handle_set_node_attributes(
            {"graph": "g", "attributes": {"a": {"color": "red", "weight": 5}}}
        )
        result = handle_get_node_attributes({"graph": "g", "node": "a"})
        assert result["attributes"]["color"] == "red"
        assert result["attributes"]["weight"] == 5

    def test_set_multiple_nodes(self):
        result = handle_set_node_attributes(
            {"graph": "g", "attributes": {"a": {"x": 1}, "b": {"x": 2}}}
        )
        assert result["updated"] == 2

    def test_get_empty_attributes(self):
        result = handle_get_node_attributes({"graph": "g", "node": "a"})
        assert result["attributes"] == {}

    def test_set_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_set_node_attributes({"graph": "nope", "attributes": {"a": {"x": 1}}})

    def test_set_missing_node(self):
        with pytest.raises(NodeNotFoundError):
            handle_set_node_attributes({"graph": "g", "attributes": {"z": {"x": 1}}})

    def test_get_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_get_node_attributes({"graph": "nope", "node": "a"})

    def test_get_missing_node(self):
        with pytest.raises(NodeNotFoundError):
            handle_get_node_attributes({"graph": "g", "node": "z"})


# ═══════════════════════════════════════════════════════════════════════
# set/get edge attributes
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeAttributes:
    @pytest.fixture(autouse=True)
    def _setup(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": ["a", "b", "c"]})
        handle_add_edges({"graph": "g", "edges": [["a", "b"], ["b", "c"]]})

    def test_set_and_get(self):
        handle_set_edge_attributes(
            {
                "graph": "g",
                "attributes": [
                    {"source": "a", "target": "b", "attr": "weight", "value": 3.5}
                ],
            }
        )
        result = handle_get_edge_attributes(
            {"graph": "g", "source": "a", "target": "b"}
        )
        assert result["attributes"]["weight"] == 3.5

    def test_set_multiple_edges(self):
        result = handle_set_edge_attributes(
            {
                "graph": "g",
                "attributes": [
                    {"source": "a", "target": "b", "attr": "w", "value": 1},
                    {"source": "b", "target": "c", "attr": "w", "value": 2},
                ],
            }
        )
        assert result["updated"] == 2

    def test_get_empty_attributes(self):
        result = handle_get_edge_attributes(
            {"graph": "g", "source": "a", "target": "b"}
        )
        assert result["attributes"] == {}

    def test_set_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_set_edge_attributes(
                {
                    "graph": "nope",
                    "attributes": [
                        {"source": "a", "target": "b", "attr": "w", "value": 1}
                    ],
                }
            )

    def test_set_missing_edge(self):
        with pytest.raises(EdgeNotFoundError):
            handle_set_edge_attributes(
                {
                    "graph": "g",
                    "attributes": [
                        {"source": "a", "target": "c", "attr": "w", "value": 1}
                    ],
                }
            )

    def test_get_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_get_edge_attributes({"graph": "nope", "source": "a", "target": "b"})

    def test_get_missing_edge(self):
        with pytest.raises(EdgeNotFoundError):
            handle_get_edge_attributes({"graph": "g", "source": "a", "target": "c"})


# ═══════════════════════════════════════════════════════════════════════
# topological_sort
# ═══════════════════════════════════════════════════════════════════════


class TestTopologicalSort:
    def test_happy_path(self):
        handle_create_graph({"name": "dag", "directed": True})
        handle_add_nodes({"graph": "dag", "nodes": ["a", "b", "c", "d"]})
        handle_add_edges(
            {"graph": "dag", "edges": [["a", "b"], ["a", "c"], ["b", "d"], ["c", "d"]]}
        )
        result = handle_topological_sort({"graph": "dag"})
        assert result["graph"] == "dag"
        assert result["count"] == 4
        order = result["order"]
        # "a" must come before "b" and "c"; "b" and "c" must come before "d"
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cyclic_graph_error(self):
        handle_create_graph({"name": "cyc", "directed": True})
        handle_add_nodes({"graph": "cyc", "nodes": ["a", "b", "c"]})
        handle_add_edges(
            {"graph": "cyc", "edges": [["a", "b"], ["b", "c"], ["c", "a"]]}
        )
        with pytest.raises(GraphOperationError):
            handle_topological_sort({"graph": "cyc"})

    def test_undirected_graph_error(self):
        handle_create_graph({"name": "und"})
        handle_add_nodes({"graph": "und", "nodes": [1, 2]})
        with pytest.raises(GraphOperationError):
            handle_topological_sort({"graph": "und"})

    def test_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_topological_sort({"graph": "nope"})


# ═══════════════════════════════════════════════════════════════════════
# subgraph
# ═══════════════════════════════════════════════════════════════════════


class TestSubgraph:
    @pytest.fixture(autouse=True)
    def _setup(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": ["a", "b", "c", "d"]})
        handle_add_edges(
            {"graph": "g", "edges": [["a", "b"], ["b", "c"], ["c", "d"], ["a", "c"]]}
        )

    def test_happy_path(self):
        result = handle_subgraph(
            {"graph": "g", "nodes": ["a", "b", "c"], "new_graph": "sub"}
        )
        assert result["source"] == "g"
        assert result["new_graph"] == "sub"
        assert result["nodes"] == 3
        # edges a-b, b-c, a-c should all be included
        assert result["edges"] == 3

    def test_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            handle_subgraph({"graph": "nope", "nodes": ["a"], "new_graph": "sub"})

    def test_missing_node_error(self):
        with pytest.raises(NodeNotFoundError):
            handle_subgraph({"graph": "g", "nodes": ["a", "z"], "new_graph": "sub"})

    def test_subgraph_target_exists_error(self):
        handle_create_graph({"name": "existing"})
        with pytest.raises(GraphAlreadyExistsError):
            handle_subgraph(
                {"graph": "g", "nodes": ["a", "b"], "new_graph": "existing"}
            )

    def test_edges_included(self):
        """Verify that edges between selected nodes are preserved."""
        handle_subgraph({"graph": "g", "nodes": ["b", "c", "d"], "new_graph": "sub2"})
        # edges b-c and c-d should be included; a-b and a-c should not
        from networkx_mcp.graph_cache import graphs as g_cache

        sub = g_cache["sub2"]
        assert sub.has_edge("b", "c")
        assert sub.has_edge("c", "d")
        assert not sub.has_node("a")


# ═══════════════════════════════════════════════════════════════════════
# merge_graphs
# ═══════════════════════════════════════════════════════════════════════


class TestMergeGraphs:
    def test_same_type(self):
        handle_create_graph({"name": "a"})
        handle_add_nodes({"graph": "a", "nodes": [1, 2]})
        handle_add_edges({"graph": "a", "edges": [[1, 2]]})
        handle_create_graph({"name": "b"})
        handle_add_nodes({"graph": "b", "nodes": [3, 4]})
        handle_add_edges({"graph": "b", "edges": [[3, 4]]})
        result = handle_merge_graphs(
            {"graph_a": "a", "graph_b": "b", "new_graph": "merged"}
        )
        assert result["new_graph"] == "merged"
        assert result["nodes"] == 4
        assert result["edges"] == 2
        assert result["source_graphs"] == ["a", "b"]

    def test_merge_target_exists_error(self):
        handle_create_graph({"name": "a"})
        handle_create_graph({"name": "b"})
        handle_create_graph({"name": "existing"})
        with pytest.raises(GraphAlreadyExistsError):
            handle_merge_graphs(
                {"graph_a": "a", "graph_b": "b", "new_graph": "existing"}
            )

    def test_different_types_error(self):
        handle_create_graph({"name": "u"})  # undirected
        handle_create_graph({"name": "d", "directed": True})  # directed
        with pytest.raises(GraphOperationError):
            handle_merge_graphs({"graph_a": "u", "graph_b": "d", "new_graph": "m"})

    def test_overlapping_nodes(self):
        handle_create_graph({"name": "a"})
        handle_add_nodes({"graph": "a", "nodes": [1, 2, 3]})
        handle_add_edges({"graph": "a", "edges": [[1, 2]]})
        handle_create_graph({"name": "b"})
        handle_add_nodes({"graph": "b", "nodes": [2, 3, 4]})
        handle_add_edges({"graph": "b", "edges": [[3, 4]]})
        result = handle_merge_graphs(
            {"graph_a": "a", "graph_b": "b", "new_graph": "merged"}
        )
        # nodes 1,2,3,4 — union
        assert result["nodes"] == 4
        # edges 1-2, 3-4
        assert result["edges"] == 2

    def test_disjoint_graphs(self):
        handle_create_graph({"name": "x", "directed": True})
        handle_add_nodes({"graph": "x", "nodes": ["a", "b"]})
        handle_add_edges({"graph": "x", "edges": [["a", "b"]]})
        handle_create_graph({"name": "y", "directed": True})
        handle_add_nodes({"graph": "y", "nodes": ["c", "d"]})
        handle_add_edges({"graph": "y", "edges": [["c", "d"]]})
        result = handle_merge_graphs(
            {"graph_a": "x", "graph_b": "y", "new_graph": "xy"}
        )
        assert result["nodes"] == 4
        assert result["edges"] == 2
        assert result["source_graphs"] == ["x", "y"]


# ═══════════════════════════════════════════════════════════════════════
# Algorithm size guards (MAX_ALGORITHM_NODES = 50_000)
# ═══════════════════════════════════════════════════════════════════════


class TestAlgorithmSizeGuards:
    """Verify MAX_ALGORITHM_NODES (50k) rejects oversized graphs."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import networkx as nx

        graphs["big"] = nx.path_graph(50_001)
        yield
        graphs.pop("big", None)

    def test_degree_centrality_rejects(self):
        with pytest.raises(ResourceLimitExceededError):
            handle_degree_centrality({"graph": "big"})

    def test_betweenness_centrality_rejects(self):
        with pytest.raises(ResourceLimitExceededError):
            handle_betweenness_centrality({"graph": "big"})

    def test_connected_components_rejects(self):
        with pytest.raises(ResourceLimitExceededError):
            handle_connected_components({"graph": "big"})

    def test_pagerank_rejects(self):
        with pytest.raises(ResourceLimitExceededError):
            handle_pagerank({"graph": "big"})

    def test_community_detection_rejects(self):
        with pytest.raises(ResourceLimitExceededError):
            handle_community_detection({"graph": "big"})

    def test_clustering_coefficients_rejects(self):
        with pytest.raises(ResourceLimitExceededError):
            handle_clustering_coefficients({"graph": "big"})

    def test_graph_statistics_rejects(self):
        with pytest.raises(ResourceLimitExceededError):
            handle_graph_statistics({"graph": "big"})

    def test_graph_coloring_rejects(self):
        with pytest.raises(ResourceLimitExceededError):
            handle_graph_coloring({"graph": "big"})

    def test_centrality_measures_rejects(self):
        with pytest.raises(ResourceLimitExceededError):
            handle_centrality_measures({"graph": "big"})

    def test_matching_rejects(self):
        with pytest.raises(ResourceLimitExceededError):
            handle_matching({"graph": "big"})


# ═══════════════════════════════════════════════════════════════════════
# CSV edge limit
# ═══════════════════════════════════════════════════════════════════════


class TestCSVEdgeLimit:
    def test_import_csv_edge_limit(self):
        csv_data = "\n".join(f"{i},{i + 1}" for i in range(500_001))
        with pytest.raises(ValueError, match="too many edges"):
            handle_import_csv(
                {"graph": "csv_big", "csv_data": csv_data, "directed": False}
            )
