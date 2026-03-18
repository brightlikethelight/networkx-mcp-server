"""
Comprehensive tests for NetworkXMCPServer.

Covers initialization, MCP lifecycle, tool dispatch, error handling,
module-level exports, academic tools, and authentication flow.
"""

import json
import os

import networkx as nx
import pytest

# Suppress auth warning during import
os.environ["NETWORKX_MCP_SUPPRESS_AUTH_WARNING"] = "1"

from networkx_mcp.server import (
    NetworkXMCPServer,
    add_edges,
    add_nodes,
    betweenness_centrality,
    community_detection,
    connected_components,
    create_graph,
    degree_centrality,
    delete_graph,
    export_json,
    get_graph_info,
    graphs,
    import_csv,
    mcp,
    pagerank,
    shortest_path,
    visualize_graph,
)
from networkx_mcp.auth import APIKeyManager, AuthMiddleware
from networkx_mcp.errors import ErrorCodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_request(req_id=1):
    """Build an MCP initialize request."""
    return {"jsonrpc": "2.0", "id": req_id, "method": "initialize", "params": {}}


def _tools_list_request(req_id=2):
    return {"jsonrpc": "2.0", "id": req_id, "method": "tools/list", "params": {}}


def _tool_call(name, arguments, req_id=3):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


async def _init_server(server):
    """Send initialize + initialized to a server, return initialize response."""
    resp = await server.handle_request(_init_request())
    await server.handle_request(
        {"jsonrpc": "2.0", "method": "initialized", "params": {}}
    )
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_graphs():
    """Clear global graph state before and after every test."""
    graphs.clear()
    yield
    graphs.clear()


@pytest.fixture
def server():
    """Return a fresh, non-auth NetworkXMCPServer."""
    return NetworkXMCPServer(auth_required=False)


@pytest.fixture
def auth_server(tmp_path):
    """Return an auth-enabled server with a generated API key and its raw key."""
    key_file = tmp_path / "api_keys.json"
    mgr = APIKeyManager(storage_path=key_file)
    raw_key = mgr.generate_key("test-key", permissions={"read", "write"})

    srv = NetworkXMCPServer(auth_required=True)
    # Patch in our temp key manager so tests don't pollute ~/.networkx-mcp
    srv.key_manager = mgr
    srv.auth = AuthMiddleware(mgr, required=True)
    return srv, raw_key


@pytest.fixture
def read_only_server(tmp_path):
    """Return an auth-enabled server with a read-only API key."""
    key_file = tmp_path / "api_keys.json"
    mgr = APIKeyManager(storage_path=key_file)
    raw_key = mgr.generate_key("readonly-key", permissions={"read"})

    srv = NetworkXMCPServer(auth_required=True)
    srv.key_manager = mgr
    srv.auth = AuthMiddleware(mgr, required=True)
    return srv, raw_key


# ===========================================================================
# 1. Server Initialization
# ===========================================================================


class TestServerInitialization:
    def test_default_not_initialized(self, server):
        assert server.initialized is False

    def test_auth_disabled_no_warning_in_test(self, server):
        """When PYTEST_CURRENT_TEST is set, no RuntimeWarning fires."""
        assert server.auth is None
        assert server.auth_required is False

    def test_mcp_self_reference(self, server):
        assert server.mcp is server

    def test_graphs_reference(self, server):
        assert server.graphs is graphs

    def test_monitoring_disabled_by_default(self, server):
        assert server.monitoring_enabled is False
        assert server.monitor is None


# ===========================================================================
# 2. MCP Lifecycle via handle_request
# ===========================================================================


class TestMCPLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_returns_protocol_version(self, server):
        resp = await server.handle_request(_init_request())
        result = resp["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert "tools" in result["capabilities"]
        assert result["serverInfo"]["name"] == "networkx-mcp-server"

    @pytest.mark.asyncio
    async def test_initialize_returns_correct_version(self, server):
        """Version in initialize response matches __version__, not hardcoded."""
        from networkx_mcp.__version__ import __version__

        resp = await server.handle_request(_init_request())
        assert resp["result"]["serverInfo"]["version"] == __version__

    @pytest.mark.asyncio
    async def test_initialize_marks_server_initialized(self, server):
        await server.handle_request(_init_request())
        assert server.initialized is True

    @pytest.mark.asyncio
    async def test_initialized_notification_returns_none(self, server):
        await server.handle_request(_init_request())
        resp = await server.handle_request(
            {"jsonrpc": "2.0", "method": "initialized", "params": {}}
        )
        assert resp is None

    @pytest.mark.asyncio
    async def test_initialized_with_id_returns_result(self, server):
        await server.handle_request(_init_request())
        resp = await server.handle_request(
            {"jsonrpc": "2.0", "id": 99, "method": "initialized", "params": {}}
        )
        assert resp is not None
        assert resp["id"] == 99

    @pytest.mark.asyncio
    async def test_tools_list_before_init_returns_error(self, server):
        resp = await server.handle_request(_tools_list_request())
        assert "error" in resp
        assert resp["error"]["code"] == ErrorCodes.SERVER_NOT_INITIALIZED

    @pytest.mark.asyncio
    async def test_tools_call_before_init_returns_error(self, server):
        resp = await server.handle_request(_tool_call("create_graph", {"name": "g"}))
        assert "error" in resp
        assert resp["error"]["code"] == ErrorCodes.SERVER_NOT_INITIALIZED

    @pytest.mark.asyncio
    async def test_tools_list_after_init_returns_tools(self, server):
        await _init_server(server)
        resp = await server.handle_request(_tools_list_request())
        tools = resp["result"]["tools"]
        assert isinstance(tools, list)
        tool_names = {t["name"] for t in tools}
        assert "create_graph" in tool_names
        assert "add_nodes" in tool_names
        assert "shortest_path" in tool_names
        assert "delete_graph" in tool_names

    @pytest.mark.asyncio
    async def test_unknown_method_returns_error(self, server):
        await _init_server(server)
        resp = await server.handle_request(
            {"jsonrpc": "2.0", "id": 5, "method": "nonexistent/method", "params": {}}
        )
        assert resp["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_resources_list(self, server):
        await _init_server(server)
        resp = await server.handle_request(
            {"jsonrpc": "2.0", "id": 6, "method": "resources/list", "params": {}}
        )
        assert resp["result"]["resources"] == []

    @pytest.mark.asyncio
    async def test_resources_read_invalid_uri(self, server):
        await _init_server(server)
        resp = await server.handle_request(
            {"jsonrpc": "2.0", "id": 7, "method": "resources/read", "params": {}}
        )
        assert resp["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_resources_list_with_graphs(self, server):
        """resources/list returns entries for stored graphs."""
        await _init_server(server)
        # Create a graph first
        await server.handle_request(_tool_call("create_graph", {"name": "res_g"}))
        await server.handle_request(
            _tool_call("add_nodes", {"graph": "res_g", "nodes": ["a", "b"]})
        )
        await server.handle_request(
            _tool_call("add_edges", {"graph": "res_g", "edges": [["a", "b"]]})
        )
        resp = await server.handle_request(
            {"jsonrpc": "2.0", "id": 20, "method": "resources/list", "params": {}}
        )
        resources = resp["result"]["resources"]
        assert len(resources) == 1
        entry = resources[0]
        assert entry["uri"] == "graph://res_g"
        assert entry["name"] == "res_g"
        assert "description" in entry
        assert entry["mimeType"] == "application/json"

    @pytest.mark.asyncio
    async def test_resources_read_valid_graph(self, server):
        """resources/read returns node-link JSON for an existing graph."""
        await _init_server(server)
        await server.handle_request(_tool_call("create_graph", {"name": "res_r"}))
        await server.handle_request(
            _tool_call("add_nodes", {"graph": "res_r", "nodes": ["x", "y"]})
        )
        await server.handle_request(
            _tool_call("add_edges", {"graph": "res_r", "edges": [["x", "y"]]})
        )
        resp = await server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 21,
                "method": "resources/read",
                "params": {"uri": "graph://res_r"},
            }
        )
        contents = resp["result"]["contents"]
        assert len(contents) == 1
        data = json.loads(contents[0]["text"])
        assert "nodes" in data
        assert "links" in data

    @pytest.mark.asyncio
    async def test_resources_read_missing_graph(self, server):
        """resources/read returns -32602 for a missing graph."""
        await _init_server(server)
        resp = await server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 22,
                "method": "resources/read",
                "params": {"uri": "graph://does_not_exist"},
            }
        )
        assert resp["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_prompts_list(self, server):
        await _init_server(server)
        resp = await server.handle_request(
            {"jsonrpc": "2.0", "id": 8, "method": "prompts/list", "params": {}}
        )
        assert resp["result"]["prompts"] == []

    @pytest.mark.asyncio
    async def test_prompts_get_not_implemented(self, server):
        await _init_server(server)
        resp = await server.handle_request(
            {"jsonrpc": "2.0", "id": 9, "method": "prompts/get", "params": {}}
        )
        assert resp["error"]["code"] == -32601


# ===========================================================================
# 3. _call_tool Dispatch (via handle_request tools/call)
# ===========================================================================


class TestToolDispatchBasic:
    """Tests for core graph CRUD tools."""

    @pytest.mark.asyncio
    async def test_create_graph_undirected(self, server):
        await _init_server(server)
        resp = await server.handle_request(_tool_call("create_graph", {"name": "g1"}))
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["created"] == "g1"
        assert content["type"] == "undirected"
        assert "g1" in graphs

    @pytest.mark.asyncio
    async def test_create_graph_directed(self, server):
        await _init_server(server)
        resp = await server.handle_request(
            _tool_call("create_graph", {"name": "dg", "directed": True})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["type"] == "directed"
        assert graphs["dg"].is_directed()

    @pytest.mark.asyncio
    async def test_add_nodes(self, server):
        await _init_server(server)
        await server.handle_request(_tool_call("create_graph", {"name": "g"}))
        resp = await server.handle_request(
            _tool_call("add_nodes", {"graph": "g", "nodes": ["a", "b", "c"]})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["added"] == 3
        assert content["total"] == 3

    @pytest.mark.asyncio
    async def test_add_edges(self, server):
        await _init_server(server)
        await server.handle_request(_tool_call("create_graph", {"name": "g"}))
        await server.handle_request(
            _tool_call("add_nodes", {"graph": "g", "nodes": ["a", "b", "c"]})
        )
        resp = await server.handle_request(
            _tool_call("add_edges", {"graph": "g", "edges": [["a", "b"], ["b", "c"]]})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["added"] == 2
        assert content["total"] == 2

    @pytest.mark.asyncio
    async def test_get_info(self, server):
        await _init_server(server)
        await server.handle_request(_tool_call("create_graph", {"name": "g"}))
        await server.handle_request(
            _tool_call("add_nodes", {"graph": "g", "nodes": [1, 2, 3]})
        )
        await server.handle_request(
            _tool_call("add_edges", {"graph": "g", "edges": [[1, 2], [2, 3]]})
        )
        resp = await server.handle_request(_tool_call("get_info", {"graph": "g"}))
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["nodes"] == 3
        assert content["edges"] == 2
        assert content["directed"] is False

    @pytest.mark.asyncio
    async def test_shortest_path(self, server):
        await _init_server(server)
        await server.handle_request(_tool_call("create_graph", {"name": "g"}))
        await server.handle_request(
            _tool_call("add_edges", {"graph": "g", "edges": [[1, 2], [2, 3], [3, 4]]})
        )
        resp = await server.handle_request(
            _tool_call("shortest_path", {"graph": "g", "source": 1, "target": 4})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["path"] == [1, 2, 3, 4]
        assert content["length"] == 3

    @pytest.mark.asyncio
    async def test_delete_graph(self, server):
        await _init_server(server)
        await server.handle_request(_tool_call("create_graph", {"name": "g"}))
        assert "g" in graphs
        resp = await server.handle_request(_tool_call("delete_graph", {"graph": "g"}))
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["deleted"] == "g"
        assert "g" not in graphs


class TestToolDispatchAnalytics:
    """Tests for analytics tools: centrality, components, pagerank, community."""

    async def _create_triangle_graph(self, server):
        """Helper: create a triangle graph with an extra node for richer metrics."""
        await _init_server(server)
        await server.handle_request(_tool_call("create_graph", {"name": "t"}))
        await server.handle_request(
            _tool_call(
                "add_edges",
                {
                    "graph": "t",
                    "edges": [[1, 2], [2, 3], [3, 1], [3, 4]],
                },
            )
        )

    @pytest.mark.asyncio
    async def test_degree_centrality(self, server):
        await self._create_triangle_graph(server)
        resp = await server.handle_request(
            _tool_call("degree_centrality", {"graph": "t"})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert "centrality" in content
        assert "most_central" in content

    @pytest.mark.asyncio
    async def test_betweenness_centrality(self, server):
        await self._create_triangle_graph(server)
        resp = await server.handle_request(
            _tool_call("betweenness_centrality", {"graph": "t"})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert "centrality" in content

    @pytest.mark.asyncio
    async def test_connected_components(self, server):
        await self._create_triangle_graph(server)
        resp = await server.handle_request(
            _tool_call("connected_components", {"graph": "t"})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["num_components"] == 1
        assert content["component_sizes"] == [4]

    @pytest.mark.asyncio
    async def test_pagerank(self, server):
        await self._create_triangle_graph(server)
        resp = await server.handle_request(_tool_call("pagerank", {"graph": "t"}))
        content = json.loads(resp["result"]["content"][0]["text"])
        assert "pagerank" in content
        assert "highest_rank" in content

    @pytest.mark.asyncio
    async def test_community_detection(self, server):
        await self._create_triangle_graph(server)
        resp = await server.handle_request(
            _tool_call("community_detection", {"graph": "t"})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert "communities" in content
        assert content["num_communities"] >= 1
        assert content["method"] == "louvain"


class TestToolDispatchIO:
    """Tests for import/export and visualization tools."""

    @pytest.mark.asyncio
    async def test_visualize_graph(self, server):
        await _init_server(server)
        await server.handle_request(_tool_call("create_graph", {"name": "v"}))
        await server.handle_request(
            _tool_call("add_edges", {"graph": "v", "edges": [[1, 2], [2, 3]]})
        )
        resp = await server.handle_request(
            _tool_call("visualize_graph", {"graph": "v", "layout": "circular"})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert "visualization" in content
        assert content["format"] == "png"
        assert content["layout"] == "circular"
        assert content["visualization"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_import_csv(self, server):
        await _init_server(server)
        csv_data = "a,b\nb,c\nc,d"
        resp = await server.handle_request(
            _tool_call("import_csv", {"graph": "csv_g", "csv_data": csv_data})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["imported"] == "csv_g"
        assert content["nodes"] == 4
        assert content["edges"] == 3
        assert "csv_g" in graphs

    @pytest.mark.asyncio
    async def test_import_csv_directed(self, server):
        await _init_server(server)
        csv_data = "x,y\ny,z"
        resp = await server.handle_request(
            _tool_call(
                "import_csv",
                {
                    "graph": "dcsv",
                    "csv_data": csv_data,
                    "directed": True,
                },
            )
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["type"] == "directed"
        assert graphs["dcsv"].is_directed()

    @pytest.mark.asyncio
    async def test_export_json(self, server):
        await _init_server(server)
        await server.handle_request(_tool_call("create_graph", {"name": "ej"}))
        await server.handle_request(
            _tool_call("add_edges", {"graph": "ej", "edges": [[1, 2]]})
        )
        resp = await server.handle_request(_tool_call("export_json", {"graph": "ej"}))
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["format"] == "node-link"
        assert content["nodes"] == 2
        assert content["edges"] == 1
        assert "graph_data" in content

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, server):
        await _init_server(server)
        resp = await server.handle_request(_tool_call("totally_fake_tool", {}))
        assert "error" in resp
        assert resp["error"]["code"] == -32601
        assert "Unknown tool" in resp["error"]["message"]


# ===========================================================================
# 4. Error Handling in _call_tool
# ===========================================================================


class TestToolErrorHandling:
    @pytest.mark.asyncio
    async def test_missing_graph_returns_error(self, server):
        await _init_server(server)
        resp = await server.handle_request(
            _tool_call("add_nodes", {"graph": "nonexistent", "nodes": [1]})
        )
        assert "error" in resp
        assert resp["error"]["code"] == ErrorCodes.GRAPH_NOT_FOUND
        assert "not found" in resp["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_missing_required_param_returns_error(self, server):
        """Omitting 'name' from create_graph triggers KeyError handling."""
        await _init_server(server)
        resp = await server.handle_request(_tool_call("create_graph", {}))
        assert "error" in resp
        assert resp["error"]["code"] == ErrorCodes.INVALID_PARAMS

    @pytest.mark.asyncio
    async def test_networkx_error_shortest_path_no_path(self, server):
        """Disconnected graph -> NetworkXNoPath now correctly caught as ALGORITHM_ERROR."""
        await _init_server(server)
        await server.handle_request(_tool_call("create_graph", {"name": "dis"}))
        await server.handle_request(
            _tool_call("add_nodes", {"graph": "dis", "nodes": [1, 2]})
        )
        resp = await server.handle_request(
            _tool_call("shortest_path", {"graph": "dis", "source": 1, "target": 2})
        )
        assert "error" in resp
        assert resp["error"]["code"] == ErrorCodes.ALGORITHM_ERROR

    @pytest.mark.asyncio
    async def test_add_edges_missing_graph(self, server):
        await _init_server(server)
        resp = await server.handle_request(
            _tool_call("add_edges", {"graph": "no_such", "edges": [[1, 2]]})
        )
        assert "error" in resp
        assert "not found" in resp["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_get_info_missing_graph(self, server):
        await _init_server(server)
        resp = await server.handle_request(_tool_call("get_info", {"graph": "missing"}))
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_delete_nonexistent_graph(self, server):
        await _init_server(server)
        resp = await server.handle_request(
            _tool_call("delete_graph", {"graph": "nope"})
        )
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_error_response_is_jsonrpc_compliant(self, server):
        """Errors from _call_tool appear in resp['error'], not resp['result']['error']."""
        await _init_server(server)
        resp = await server.handle_request(
            _tool_call("add_nodes", {"graph": "no_such", "nodes": [1]})
        )
        # Must be a top-level error, not nested in result
        assert "error" in resp
        assert "result" not in resp
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] is not None

    @pytest.mark.asyncio
    async def test_cicd_tool_unavailable_returns_error(self, server):
        """CI/CD tools return proper JSON-RPC error when tools module unavailable."""
        import sys
        from unittest.mock import patch

        await _init_server(server)
        # Temporarily make the tools module unimportable
        with patch.dict(sys.modules, {"networkx_mcp.tools": None}):
            resp = await server.handle_request(
                _tool_call("trigger_workflow", {"workflow": "ci.yml"})
            )
        assert "error" in resp
        assert resp["error"]["code"] == ErrorCodes.METHOD_NOT_FOUND
        assert "CI/CD" in resp["error"]["message"]

    @pytest.mark.asyncio
    async def test_graph_not_found_no_key_leak(self, server):
        """Error message for missing graph does NOT enumerate all graph names."""
        await _init_server(server)
        # Create a graph to ensure there are keys to potentially leak
        await server.handle_request(
            _tool_call("create_graph", {"name": "secret_graph"})
        )
        resp = await server.handle_request(
            _tool_call("add_nodes", {"graph": "no_such", "nodes": [1]})
        )
        assert "error" in resp
        assert "secret_graph" not in resp["error"]["message"]


# ===========================================================================
# 5. handle_message (alias for handle_request)
# ===========================================================================


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_handle_message_delegates_to_handle_request(self, server):
        resp = await server.handle_message(_init_request())
        assert resp["result"]["protocolVersion"] == "2024-11-05"

    @pytest.mark.asyncio
    async def test_handle_message_notification(self, server):
        await server.handle_message(_init_request())
        resp = await server.handle_message(
            {"jsonrpc": "2.0", "method": "initialized", "params": {}}
        )
        assert resp is None

    @pytest.mark.asyncio
    async def test_handle_message_tools_call(self, server):
        await _init_server(server)
        resp = await server.handle_message(
            _tool_call("create_graph", {"name": "msg_g"})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["created"] == "msg_g"


# ===========================================================================
# 6. Academic Tools Dispatch
# ===========================================================================


class TestAcademicTools:
    """Academic tools operate on graphs with node metadata (authors, year, etc.)."""

    async def _create_citation_graph(self, server):
        """Build a small citation-style directed graph with metadata in global graphs."""
        await _init_server(server)
        g = nx.DiGraph()
        g.add_node(
            "doi1",
            title="Paper A",
            authors=["Alice Smith", "Bob Jones"],
            year=2020,
            citations=50,
        )
        g.add_node(
            "doi2",
            title="Paper B",
            authors=["Bob Jones", "Carol White"],
            year=2021,
            citations=30,
        )
        g.add_node(
            "doi3", title="Paper C", authors=["Alice Smith"], year=2022, citations=10
        )
        g.add_edge("doi1", "doi2")
        g.add_edge("doi1", "doi3")
        g.add_edge("doi2", "doi3")
        graphs["cite"] = g

    @pytest.mark.asyncio
    async def test_analyze_author_impact(self, server):
        await self._create_citation_graph(server)
        resp = await server.handle_request(
            _tool_call(
                "analyze_author_impact",
                {
                    "graph": "cite",
                    "author_name": "Alice Smith",
                },
            )
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["author"] == "Alice Smith"
        assert content["papers_found"] == 2
        assert content["h_index"] >= 1
        assert content["total_citations"] == 60

    @pytest.mark.asyncio
    async def test_analyze_author_impact_not_found(self, server):
        await self._create_citation_graph(server)
        resp = await server.handle_request(
            _tool_call(
                "analyze_author_impact",
                {
                    "graph": "cite",
                    "author_name": "Nobody",
                },
            )
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["papers_found"] == 0

    @pytest.mark.asyncio
    async def test_find_collaboration_patterns(self, server):
        await self._create_citation_graph(server)
        resp = await server.handle_request(
            _tool_call("find_collaboration_patterns", {"graph": "cite"})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert "coauthorship_network" in content
        # With co-author data present, we should see nodes in the coauthorship network
        assert content["coauthorship_network"]["nodes"] >= 1

    @pytest.mark.asyncio
    async def test_find_collaboration_patterns_no_authors(self, server):
        """Graph with no author metadata falls back to structural analysis."""
        await _init_server(server)
        await server.handle_request(_tool_call("create_graph", {"name": "bare"}))
        await server.handle_request(
            _tool_call("add_edges", {"graph": "bare", "edges": [[1, 2], [2, 3]]})
        )
        resp = await server.handle_request(
            _tool_call("find_collaboration_patterns", {"graph": "bare"})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["coauthorship_network"]["nodes"] == 0
        assert "note" in content

    @pytest.mark.asyncio
    async def test_detect_research_trends_with_years(self, server):
        await self._create_citation_graph(server)
        resp = await server.handle_request(
            _tool_call("detect_research_trends", {"graph": "cite", "time_window": 3})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["years_analyzed"] >= 2
        assert len(content["publication_trend"]) >= 2

    @pytest.mark.asyncio
    async def test_detect_research_trends_no_years(self, server):
        """Graph without year metadata returns no_temporal_data."""
        await _init_server(server)
        await server.handle_request(_tool_call("create_graph", {"name": "ny"}))
        await server.handle_request(
            _tool_call("add_edges", {"graph": "ny", "edges": [[1, 2]]})
        )
        resp = await server.handle_request(
            _tool_call("detect_research_trends", {"graph": "ny"})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["trend"] == "no_temporal_data"

    @pytest.mark.asyncio
    async def test_export_bibtex(self, server):
        await self._create_citation_graph(server)
        resp = await server.handle_request(
            _tool_call("export_bibtex", {"graph": "cite"})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["format"] == "bibtex"
        assert content["entries"] == 3
        assert isinstance(content["bibtex_data"], str)

    @pytest.mark.asyncio
    async def test_recommend_papers(self, server):
        await self._create_citation_graph(server)
        resp = await server.handle_request(
            _tool_call(
                "recommend_papers",
                {
                    "graph": "cite",
                    "seed_doi": "doi1",
                    "max_recommendations": 5,
                },
            )
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["seed_paper"] == "doi1"
        assert "recommendations" in content
        assert isinstance(content["recommendations"], list)

    @pytest.mark.asyncio
    async def test_recommend_papers_seed_not_in_graph(self, server):
        await self._create_citation_graph(server)
        resp = await server.handle_request(
            _tool_call(
                "recommend_papers",
                {
                    "graph": "cite",
                    "seed_doi": "nonexistent_doi",
                },
            )
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["total_found"] == 0
        assert "note" in content

    @pytest.mark.asyncio
    async def test_recommend_papers_missing_seed_param(self, server):
        """Omitting seed_doi entirely triggers ValueError."""
        await self._create_citation_graph(server)
        resp = await server.handle_request(
            _tool_call("recommend_papers", {"graph": "cite"})
        )
        assert "error" in resp
        assert (
            "seed_doi" in resp["error"]["message"]
            or "seed_paper" in resp["error"]["message"]
        )

    @pytest.mark.asyncio
    async def test_academic_tool_missing_graph(self, server):
        """Academic tools raise ValueError for missing graph."""
        await _init_server(server)
        resp = await server.handle_request(
            _tool_call(
                "analyze_author_impact",
                {
                    "graph": "nonexistent",
                    "author_name": "X",
                },
            )
        )
        assert "error" in resp


# ===========================================================================
# 7. Module-Level Exports
# ===========================================================================


class TestModuleLevelExports:
    def test_mcp_exists(self):
        assert mcp is not None
        assert isinstance(mcp, NetworkXMCPServer)

    def test_graphs_exists(self):
        assert graphs is not None

    def test_wrapper_create_graph(self):
        with pytest.warns(DeprecationWarning, match="create_graph"):
            result = create_graph("wrap_g")
        assert result["created"] is True
        assert "wrap_g" in graphs

    def test_wrapper_add_nodes(self):
        with pytest.warns(DeprecationWarning):
            create_graph("wn")
            result = add_nodes("wn", [10, 20])
        assert result["success"] is True
        assert result["total"] == 2

    def test_wrapper_add_edges(self):
        with pytest.warns(DeprecationWarning):
            create_graph("we")
            add_nodes("we", [1, 2, 3])
            result = add_edges("we", [[1, 2], [2, 3]])
        assert result["success"] is True
        assert result["edges_added"] == 2

    def test_wrapper_get_graph_info(self):
        with pytest.warns(DeprecationWarning):
            create_graph("wi")
            add_nodes("wi", [1, 2])
            result = get_graph_info("wi")
        assert result["num_nodes"] == 2

    def test_wrapper_shortest_path(self):
        with pytest.warns(DeprecationWarning):
            create_graph("wsp")
            add_edges("wsp", [[1, 2], [2, 3]])
            result = shortest_path("wsp", 1, 3)
        assert result["success"] is True
        assert result["path"] == [1, 2, 3]

    def test_wrapper_degree_centrality(self):
        with pytest.warns(DeprecationWarning):
            create_graph("wdc")
            add_edges("wdc", [[1, 2], [2, 3]])
            result = degree_centrality("wdc")
        assert "centrality" in result

    def test_wrapper_betweenness_centrality(self):
        with pytest.warns(DeprecationWarning):
            create_graph("wbc")
            add_edges("wbc", [[1, 2], [2, 3], [3, 4]])
            result = betweenness_centrality("wbc")
        assert "centrality" in result

    def test_wrapper_connected_components(self):
        with pytest.warns(DeprecationWarning):
            create_graph("wcc")
            add_edges("wcc", [[1, 2], [3, 4]])
            result = connected_components("wcc")
        assert result["num_components"] == 2

    def test_wrapper_pagerank(self):
        with pytest.warns(DeprecationWarning):
            create_graph("wpr")
            add_edges("wpr", [[1, 2], [2, 3]])
            result = pagerank("wpr")
        assert "pagerank" in result

    def test_wrapper_community_detection(self):
        with pytest.warns(DeprecationWarning):
            create_graph("wcm")
            add_edges("wcm", [[1, 2], [2, 3], [3, 1], [4, 5], [5, 6], [6, 4]])
            result = community_detection("wcm")
        assert result["num_communities"] >= 1

    def test_wrapper_visualize_graph(self):
        with pytest.warns(DeprecationWarning):
            create_graph("wviz")
            add_edges("wviz", [[1, 2]])
            result = visualize_graph("wviz", "spring")
        assert result["format"] == "png"
        assert result["image"].startswith("data:image/png;base64,")

    def test_wrapper_import_csv(self):
        with pytest.warns(DeprecationWarning, match="import_csv"):
            result = import_csv("wcsv", "a,b\nb,c")
        assert result["imported"] == "wcsv"
        assert "wcsv" in graphs

    def test_wrapper_export_json(self):
        with pytest.warns(DeprecationWarning):
            create_graph("wej")
            add_edges("wej", [[1, 2]])
            result = export_json("wej")
        assert result["format"] == "node-link"

    def test_wrapper_delete_graph(self):
        with pytest.warns(DeprecationWarning):
            create_graph("wdel")
        assert "wdel" in graphs
        with pytest.warns(DeprecationWarning, match="delete_graph"):
            result = delete_graph("wdel")
        assert result["success"] is True
        assert "wdel" not in graphs

    def test_wrapper_delete_nonexistent(self):
        with pytest.warns(DeprecationWarning, match="delete_graph"):
            result = delete_graph("doesnt_exist")
        assert result["success"] is False


# ===========================================================================
# 8. Authentication Flow
# ===========================================================================


class TestAuthenticationFlow:
    @pytest.mark.asyncio
    async def test_missing_key_returns_error(self, auth_server):
        srv, _key = auth_server
        await _init_server(srv)
        resp = await srv.handle_request(_tool_call("create_graph", {"name": "g"}))
        assert "error" in resp
        assert resp["error"]["code"] == -32603
        assert "API key required" in resp["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_key_returns_error(self, auth_server):
        srv, _key = auth_server
        await _init_server(srv)
        resp = await srv.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "create_graph",
                    "arguments": {"name": "g"},
                    "api_key": "nxmcp_invalid_key_here",
                },
            }
        )
        assert "error" in resp
        assert "Invalid API key" in resp["error"]["message"]

    @pytest.mark.asyncio
    async def test_valid_key_allows_access(self, auth_server):
        srv, raw_key = auth_server
        await _init_server(srv)
        resp = await srv.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {
                    "name": "create_graph",
                    "arguments": {"name": "auth_g"},
                    "api_key": raw_key,
                },
            }
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["created"] == "auth_g"
        assert "auth_g" in graphs

    @pytest.mark.asyncio
    async def test_valid_key_read_tool(self, auth_server):
        srv, raw_key = auth_server
        await _init_server(srv)
        # Create graph first
        await srv.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/call",
                "params": {
                    "name": "create_graph",
                    "arguments": {"name": "rg"},
                    "api_key": raw_key,
                },
            }
        )
        # Read-only tool (get_info)
        resp = await srv.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 13,
                "method": "tools/call",
                "params": {
                    "name": "get_info",
                    "arguments": {"graph": "rg"},
                    "api_key": raw_key,
                },
            }
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["nodes"] == 0

    @pytest.mark.asyncio
    async def test_read_only_key_blocked_on_write(self, read_only_server):
        srv, ro_key = read_only_server
        await _init_server(srv)
        resp = await srv.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 14,
                "method": "tools/call",
                "params": {
                    "name": "create_graph",
                    "arguments": {"name": "ronly"},
                    "api_key": ro_key,
                },
            }
        )
        assert "error" in resp
        assert "Permission denied" in resp["error"]["message"]
        assert "ronly" not in graphs

    @pytest.mark.asyncio
    async def test_read_only_key_allowed_read(self, read_only_server):
        """Read-only key can call non-write tools."""
        srv, ro_key = read_only_server
        await _init_server(srv)
        # Pre-populate a graph bypassing auth
        graphs["pre"] = nx.Graph()
        graphs["pre"].add_edge(1, 2)
        resp = await srv.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 15,
                "method": "tools/call",
                "params": {
                    "name": "get_info",
                    "arguments": {"graph": "pre"},
                    "api_key": ro_key,
                },
            }
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["nodes"] == 2

    @pytest.mark.asyncio
    async def test_auth_not_required_for_initialize(self, auth_server):
        """initialize and initialized bypass auth entirely."""
        srv, _key = auth_server
        resp = await srv.handle_request(_init_request())
        assert "result" in resp
        assert resp["result"]["protocolVersion"] == "2024-11-05"

    @pytest.mark.asyncio
    async def test_api_key_stripped_from_params(self, auth_server):
        """After auth, api_key is removed from params to avoid exposure."""
        srv, raw_key = auth_server
        await _init_server(srv)
        req = {
            "jsonrpc": "2.0",
            "id": 16,
            "method": "tools/list",
            "params": {"api_key": raw_key},
        }
        resp = await srv.handle_request(req)
        # Should succeed (tools/list needs init + auth)
        assert "result" in resp
        assert "tools" in resp["result"]
        # The api_key should have been deleted from params
        assert "api_key" not in req["params"]


# ===========================================================================
# 9. JSON-RPC 2.0 Validation
# ===========================================================================


class TestJsonRpcValidation:
    @pytest.mark.asyncio
    async def test_missing_jsonrpc_field(self, server):
        resp = await server.handle_request(
            {"id": 1, "method": "initialize", "params": {}}
        )
        assert "error" in resp
        assert resp["error"]["code"] == ErrorCodes.INVALID_REQUEST

    @pytest.mark.asyncio
    async def test_wrong_jsonrpc_version(self, server):
        resp = await server.handle_request(
            {"jsonrpc": "1.0", "id": 1, "method": "initialize", "params": {}}
        )
        assert "error" in resp
        assert resp["error"]["code"] == ErrorCodes.INVALID_REQUEST


# ===========================================================================
# 10. Graph ID Validation (Security)
# ===========================================================================


class TestGraphIdValidation:
    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self, server):
        await _init_server(server)
        resp = await server.handle_request(
            _tool_call("create_graph", {"name": "../../../etc/passwd"})
        )
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_special_chars_rejected(self, server):
        await _init_server(server)
        resp = await server.handle_request(
            _tool_call("create_graph", {"name": "graph;rm -rf /"})
        )
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_empty_graph_name_rejected(self, server):
        await _init_server(server)
        resp = await server.handle_request(_tool_call("create_graph", {"name": ""}))
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_valid_graph_name_accepted(self, server):
        await _init_server(server)
        resp = await server.handle_request(
            _tool_call("create_graph", {"name": "my-graph_123"})
        )
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["created"] == "my-graph_123"

    @pytest.mark.asyncio
    async def test_graph_id_validation_on_read_tools(self, server):
        """validate_graph_id is called for read tools too, not just create."""
        await _init_server(server)
        resp = await server.handle_request(
            _tool_call("add_nodes", {"graph": "../../bad", "nodes": [1]})
        )
        assert "error" in resp


# ===========================================================================
# 11. main() entry point
# ===========================================================================


class TestMainEntryPoint:
    """Cover server.py main() — lines 503-561."""

    def test_main_development_mode(self, monkeypatch):
        """Default dev mode: no auth, no monitoring, server created and run."""
        monkeypatch.delenv("NETWORKX_MCP_AUTH", raising=False)
        monkeypatch.delenv("NETWORKX_MCP_MONITORING", raising=False)
        monkeypatch.delenv("NETWORKX_MCP_ENV", raising=False)

        from unittest.mock import patch

        with patch("networkx_mcp.server.asyncio.run") as mock_run:
            from networkx_mcp.server import main

            main()
            mock_run.assert_called_once()

    def test_main_auth_enabled(self, monkeypatch):
        """NETWORKX_MCP_AUTH=true creates server with auth."""
        monkeypatch.setenv("NETWORKX_MCP_AUTH", "true")
        monkeypatch.delenv("NETWORKX_MCP_MONITORING", raising=False)

        from unittest.mock import patch

        with patch("networkx_mcp.server.asyncio.run") as mock_run:
            from networkx_mcp.server import main

            main()
            mock_run.assert_called_once()

    def test_main_production_no_auth_blocked(self, monkeypatch):
        """Production mode without auth raises RuntimeError."""
        monkeypatch.setenv("NETWORKX_MCP_ENV", "production")
        monkeypatch.delenv("NETWORKX_MCP_AUTH", raising=False)
        monkeypatch.delenv("NETWORKX_MCP_INSECURE_CONFIRM", raising=False)

        from networkx_mcp.server import main

        with pytest.raises(RuntimeError, match="Authentication disabled in production"):
            main()

    def test_main_production_insecure_confirm(self, monkeypatch):
        """Production + INSECURE_CONFIRM bypasses the safety check."""
        monkeypatch.setenv("NETWORKX_MCP_ENV", "production")
        monkeypatch.delenv("NETWORKX_MCP_AUTH", raising=False)
        monkeypatch.setenv("NETWORKX_MCP_INSECURE_CONFIRM", "true")

        from unittest.mock import patch

        with patch("networkx_mcp.server.asyncio.run") as mock_run:
            from networkx_mcp.server import main

            main()
            mock_run.assert_called_once()

    def test_main_monitoring_enabled(self, monkeypatch):
        """NETWORKX_MCP_MONITORING=true enables monitoring."""
        monkeypatch.delenv("NETWORKX_MCP_AUTH", raising=False)
        monkeypatch.delenv("NETWORKX_MCP_ENV", raising=False)
        monkeypatch.setenv("NETWORKX_MCP_MONITORING", "true")

        from unittest.mock import patch

        with patch("networkx_mcp.server.asyncio.run") as mock_run:
            from networkx_mcp.server import main

            main()
            mock_run.assert_called_once()


# ===========================================================================
# Server run loop tests
# ===========================================================================


class TestServerRunLoop:
    """Tests for NetworkXMCPServer.run() — the stdin/stdout JSON-RPC loop."""

    @pytest.mark.asyncio
    async def test_run_exits_on_empty_stdin(self, server):
        """EOF (empty string from readline) causes the loop to exit."""
        from unittest.mock import AsyncMock, patch

        # run_in_executor returns "" to simulate EOF
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value="")
            await server.run()
        # If we reach here, the loop exited cleanly on EOF

    @pytest.mark.asyncio
    async def test_run_handles_json_parse_error(self, server):
        """Malformed JSON line produces a parse-error response on stderr."""
        from unittest.mock import AsyncMock, patch

        call_count = 0

        async def fake_executor(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "{not valid json}\n"
            return ""  # EOF on second call

        with (
            patch("asyncio.get_event_loop") as mock_loop,
            patch("builtins.print") as mock_print,
        ):
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=fake_executor
            )
            await server.run()

            # The error response should have been printed to stderr
            mock_print.assert_called()
            error_call = mock_print.call_args
            parsed = json.loads(error_call[0][0])
            assert parsed["error"]["code"] == ErrorCodes.PARSE_ERROR

    @pytest.mark.asyncio
    async def test_run_handles_ioerror(self, server):
        """IOError during readline breaks the loop."""
        from unittest.mock import AsyncMock, patch

        async def raise_ioerror(*args):
            raise IOError("stdin pipe broken")

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=raise_ioerror
            )
            await server.run()
        # Loop should have broken out on IOError without propagating

    @pytest.mark.asyncio
    async def test_run_processes_valid_request(self, server):
        """A valid JSON-RPC initialize request gets a response on stdout."""
        from unittest.mock import AsyncMock, patch

        init_req = json.dumps(_init_request()) + "\n"
        call_count = 0

        async def fake_executor(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return init_req
            return ""  # EOF

        with (
            patch("asyncio.get_event_loop") as mock_loop,
            patch("builtins.print") as mock_print,
        ):
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=fake_executor
            )
            await server.run()

            # The initialize response should have been printed to stdout
            mock_print.assert_called()
            # Find the stdout call (no file= kwarg or file=sys.stdout)
            for call in mock_print.call_args_list:
                kwargs = call[1]
                if "file" not in kwargs or kwargs.get("file") is None:
                    parsed = json.loads(call[0][0])
                    assert "result" in parsed
                    assert parsed["result"]["protocolVersion"] == "2024-11-05"
                    break
            else:
                # If all calls had file=sys.stderr, fail
                pytest.fail("No stdout response found for valid initialize request")
