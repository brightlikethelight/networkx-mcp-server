"""Tests for handlers.py — direct handler calls and error paths."""

from unittest.mock import patch

import pytest

from networkx_mcp.errors import MCPError
from networkx_mcp.graph_cache import graphs
from networkx_mcp.handlers import (
    handle_add_edges,
    handle_add_nodes,
    handle_create_graph,
    handle_delete_graph,
    handle_export_json,
    handle_get_info,
    handle_import_csv,
    handle_shortest_path,
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

    def test_create_directed_graph(self):
        result = handle_create_graph({"name": "g2", "directed": True})
        assert result["type"] == "directed"

    def test_add_nodes(self):
        handle_create_graph({"name": "g"})
        result = handle_add_nodes({"graph": "g", "nodes": [1, 2, 3]})
        assert result["added"] == 3
        assert result["total"] == 3

    def test_add_nodes_missing_graph(self):
        with pytest.raises(ValueError, match="not found"):
            handle_add_nodes({"graph": "nope", "nodes": [1]})

    def test_add_edges(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": [1, 2, 3]})
        result = handle_add_edges({"graph": "g", "edges": [[1, 2], [2, 3]]})
        assert result["added"] == 2
        assert result["total"] == 2

    def test_add_edges_missing_graph(self):
        with pytest.raises(ValueError, match="not found"):
            handle_add_edges({"graph": "nope", "edges": [[1, 2]]})

    def test_get_info(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": ["a", "b"]})
        result = handle_get_info({"graph": "g"})
        assert result["nodes"] == 2
        assert result["edges"] == 0
        assert result["directed"] is False

    def test_get_info_missing_graph(self):
        with pytest.raises(ValueError, match="not found"):
            handle_get_info({"graph": "nope"})

    def test_delete_graph_success(self):
        handle_create_graph({"name": "g"})
        result = handle_delete_graph({"graph": "g"})
        assert result["success"] is True
        assert result["deleted"] is True

    def test_delete_graph_missing(self):
        result = handle_delete_graph({"graph": "nope"})
        assert result["success"] is False
        assert "error" in result

    def test_shortest_path(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": [1, 2, 3]})
        handle_add_edges({"graph": "g", "edges": [[1, 2], [2, 3]]})
        result = handle_shortest_path({"graph": "g", "source": 1, "target": 3})
        assert result["path"] == [1, 2, 3]
        assert result["length"] == 2

    def test_shortest_path_missing_graph(self):
        with pytest.raises(ValueError, match="not found"):
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
# I/O handlers
# ═══════════════════════════════════════════════════════════════════════


class TestIOHandlers:
    def test_import_csv(self):
        result = handle_import_csv(
            {"graph": "csv_g", "csv_data": "a,b\nc,d\n", "directed": False}
        )
        assert result["nodes"] == 4
        assert result["edges"] == 2

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
# Bulk operation limits
# ═══════════════════════════════════════════════════════════════════════


class TestBulkLimits:
    def test_add_nodes_exceeds_limit(self):
        handle_create_graph({"name": "g"})
        nodes = list(range(100_001))
        with pytest.raises(ValueError, match="Too many nodes"):
            handle_add_nodes({"graph": "g", "nodes": nodes})

    def test_add_edges_exceeds_limit(self):
        handle_create_graph({"name": "g"})
        edges = [[i, i + 1] for i in range(500_001)]
        with pytest.raises(ValueError, match="Too many edges"):
            handle_add_edges({"graph": "g", "edges": edges})

    def test_visualize_large_graph_rejected(self):
        handle_create_graph({"name": "g"})
        import networkx as nx

        graphs["g"] = nx.path_graph(10_001)
        with pytest.raises(ValueError, match="too large for visualization"):
            handle_visualize_graph({"graph": "g"})

    def test_visualize_small_graph_succeeds(self):
        handle_create_graph({"name": "g"})
        handle_add_nodes({"graph": "g", "nodes": [1, 2, 3]})
        handle_add_edges({"graph": "g", "edges": [[1, 2], [2, 3]]})
        result = handle_visualize_graph({"graph": "g"})
        assert "visualization" in result
