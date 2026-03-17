"""
Integration tests for NetworkX MCP Server.

Tests end-to-end flows through handle_request (in-process),
subprocess smoke tests via stdio transport, and JSON-RPC protocol compliance.
"""

import asyncio
import json
import os
import sys

import pytest

os.environ["NETWORKX_MCP_SUPPRESS_AUTH_WARNING"] = "1"

from networkx_mcp.server import NetworkXMCPServer, graphs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_request(req_id=1):
    return {"jsonrpc": "2.0", "id": req_id, "method": "initialize", "params": {}}


def _tool_call(name, arguments, req_id=3):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def _resources_list(req_id=10):
    return {"jsonrpc": "2.0", "id": req_id, "method": "resources/list", "params": {}}


def _resources_read(uri, req_id=11):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "resources/read",
        "params": {"uri": uri},
    }


async def _init_server(srv):
    await srv.handle_request(_init_request())
    await srv.handle_request({"jsonrpc": "2.0", "method": "initialized", "params": {}})


def _parse_tool_text(resp):
    """Extract the parsed JSON from a successful tool call response."""
    return json.loads(resp["result"]["content"][0]["text"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def server():
    srv = NetworkXMCPServer()
    graphs.clear()
    yield srv
    graphs.clear()


# ===========================================================================
# In-process integration tests
# ===========================================================================


@pytest.mark.integration
async def test_full_graph_lifecycle(server):
    """create -> add nodes -> add edges -> get_info -> degree_centrality -> delete"""
    await _init_server(server)

    # Create
    resp = await server.handle_request(
        _tool_call("create_graph", {"name": "lifecycle", "directed": False})
    )
    assert "error" not in resp
    data = _parse_tool_text(resp)
    assert data["created"] == "lifecycle"

    # Add nodes
    resp = await server.handle_request(
        _tool_call("add_nodes", {"graph": "lifecycle", "nodes": ["a", "b", "c", "d"]})
    )
    assert "error" not in resp

    # Add edges
    resp = await server.handle_request(
        _tool_call(
            "add_edges",
            {"graph": "lifecycle", "edges": [["a", "b"], ["b", "c"], ["c", "d"]]},
        )
    )
    assert "error" not in resp

    # Get info
    resp = await server.handle_request(_tool_call("get_info", {"graph": "lifecycle"}))
    info = _parse_tool_text(resp)
    assert info["nodes"] == 4
    assert info["edges"] == 3

    # Degree centrality
    resp = await server.handle_request(
        _tool_call("degree_centrality", {"graph": "lifecycle"})
    )
    centrality = _parse_tool_text(resp)
    assert "centrality" in centrality
    assert len(centrality["centrality"]) == 4

    # Delete
    resp = await server.handle_request(
        _tool_call("delete_graph", {"graph": "lifecycle"})
    )
    assert "error" not in resp
    assert "lifecycle" not in graphs


@pytest.mark.integration
async def test_multi_graph_isolation(server):
    """Two graphs with different data must not cross-contaminate."""
    await _init_server(server)

    # Create graph A with 3 nodes
    await server.handle_request(
        _tool_call("create_graph", {"name": "alpha", "directed": False})
    )
    await server.handle_request(
        _tool_call("add_nodes", {"graph": "alpha", "nodes": [1, 2, 3]})
    )

    # Create graph B with 5 nodes
    await server.handle_request(
        _tool_call("create_graph", {"name": "beta", "directed": True})
    )
    await server.handle_request(
        _tool_call("add_nodes", {"graph": "beta", "nodes": [10, 20, 30, 40, 50]})
    )

    # Verify counts are independent
    resp_a = await server.handle_request(_tool_call("get_info", {"graph": "alpha"}))
    resp_b = await server.handle_request(_tool_call("get_info", {"graph": "beta"}))
    info_a = _parse_tool_text(resp_a)
    info_b = _parse_tool_text(resp_b)

    assert info_a["nodes"] == 3
    assert info_b["nodes"] == 5
    assert info_a["directed"] is False
    assert info_b["directed"] is True


@pytest.mark.integration
async def test_algorithm_pipeline(server):
    """Run degree_centrality -> betweenness -> community_detection -> pagerank."""
    await _init_server(server)

    # Build a small connected graph
    await server.handle_request(
        _tool_call("create_graph", {"name": "algo", "directed": False})
    )
    await server.handle_request(
        _tool_call("add_nodes", {"graph": "algo", "nodes": [1, 2, 3, 4, 5]})
    )
    await server.handle_request(
        _tool_call(
            "add_edges",
            {
                "graph": "algo",
                "edges": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1], [1, 3]],
            },
        )
    )

    # Degree centrality
    resp = await server.handle_request(
        _tool_call("degree_centrality", {"graph": "algo"})
    )
    dc = _parse_tool_text(resp)
    assert "centrality" in dc
    assert len(dc["centrality"]) == 5

    # Betweenness centrality
    resp = await server.handle_request(
        _tool_call("betweenness_centrality", {"graph": "algo"})
    )
    bc = _parse_tool_text(resp)
    assert "centrality" in bc

    # Community detection
    resp = await server.handle_request(
        _tool_call("community_detection", {"graph": "algo"})
    )
    cd = _parse_tool_text(resp)
    assert "communities" in cd
    assert cd["num_communities"] >= 1

    # PageRank
    resp = await server.handle_request(_tool_call("pagerank", {"graph": "algo"}))
    pr = _parse_tool_text(resp)
    assert "pagerank" in pr
    assert len(pr["pagerank"]) == 5


@pytest.mark.integration
async def test_resources_api_lists_graphs(server):
    """resources/list should expose created graphs; resources/read returns data."""
    await _init_server(server)

    await server.handle_request(
        _tool_call("create_graph", {"name": "res_test", "directed": False})
    )
    await server.handle_request(
        _tool_call("add_nodes", {"graph": "res_test", "nodes": ["x", "y"]})
    )
    await server.handle_request(
        _tool_call("add_edges", {"graph": "res_test", "edges": [["x", "y"]]})
    )

    # resources/list
    resp = await server.handle_request(_resources_list())
    resources = resp["result"]["resources"]
    uris = [r["uri"] for r in resources]
    assert "graph://res_test" in uris

    matching = [r for r in resources if r["uri"] == "graph://res_test"][0]
    assert "2 nodes" in matching["description"]
    assert "1 edges" in matching["description"]

    # resources/read
    resp = await server.handle_request(_resources_read("graph://res_test"))
    contents = resp["result"]["contents"]
    assert len(contents) == 1
    graph_data = json.loads(contents[0]["text"])
    assert "nodes" in graph_data
    assert "links" in graph_data


@pytest.mark.integration
async def test_csv_import_then_analyze(server):
    """Import CSV data, then run shortest_path and degree_centrality."""
    await _init_server(server)

    csv_data = "source,target\nA,B\nB,C\nC,D\nA,D"
    resp = await server.handle_request(
        _tool_call(
            "import_csv",
            {"graph": "csv_graph", "csv_data": csv_data, "directed": False},
        )
    )
    assert "error" not in resp

    # Shortest path A -> C
    resp = await server.handle_request(
        _tool_call(
            "shortest_path", {"graph": "csv_graph", "source": "A", "target": "C"}
        )
    )
    sp = _parse_tool_text(resp)
    assert "path" in sp
    assert sp["path"][0] == "A"
    assert sp["path"][-1] == "C"

    # Degree centrality
    resp = await server.handle_request(
        _tool_call("degree_centrality", {"graph": "csv_graph"})
    )
    dc = _parse_tool_text(resp)
    assert "centrality" in dc
    assert len(dc["centrality"]) == 4


@pytest.mark.integration
async def test_error_recovery(server):
    """Invalid request must not break subsequent valid requests."""
    await _init_server(server)

    # Invalid: operate on non-existent graph
    resp = await server.handle_request(
        _tool_call("get_info", {"graph": "no_such_graph"})
    )
    assert "error" in resp

    # Valid: create and query a real graph
    await server.handle_request(
        _tool_call("create_graph", {"name": "recovery", "directed": False})
    )
    await server.handle_request(
        _tool_call("add_nodes", {"graph": "recovery", "nodes": [1, 2]})
    )
    resp = await server.handle_request(_tool_call("get_info", {"graph": "recovery"}))
    assert "error" not in resp
    info = _parse_tool_text(resp)
    assert info["nodes"] == 2


@pytest.mark.integration
async def test_concurrent_tool_calls(server):
    """Multiple tool calls via asyncio.gather must not interfere."""
    await _init_server(server)

    # Create 3 separate graphs
    for name in ["g1", "g2", "g3"]:
        await server.handle_request(
            _tool_call("create_graph", {"name": name, "directed": False})
        )
        await server.handle_request(
            _tool_call("add_nodes", {"graph": name, "nodes": [1, 2, 3]})
        )
        await server.handle_request(
            _tool_call("add_edges", {"graph": name, "edges": [[1, 2], [2, 3]]})
        )

    # Concurrent get_graph_info on all three
    results = await asyncio.gather(
        server.handle_request(_tool_call("get_info", {"graph": "g1"}, req_id=10)),
        server.handle_request(_tool_call("get_info", {"graph": "g2"}, req_id=11)),
        server.handle_request(_tool_call("get_info", {"graph": "g3"}, req_id=12)),
    )

    for resp in results:
        assert "error" not in resp
        info = _parse_tool_text(resp)
        assert info["nodes"] == 3
        assert info["edges"] == 2


# ===========================================================================
# Subprocess smoke tests
# ===========================================================================


async def _spawn_server():
    """Spawn the MCP server as a subprocess, return the process handle."""
    env = os.environ.copy()
    env["NETWORKX_MCP_SUPPRESS_AUTH_WARNING"] = "1"
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "networkx_mcp",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    return proc


async def _send_and_recv(proc, request, timeout=5.0):
    """Send a JSON-RPC request line and read one response line."""
    line = json.dumps(request) + "\n"
    proc.stdin.write(line.encode())
    await proc.stdin.drain()
    raw = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
    return json.loads(raw.decode().strip())


@pytest.mark.integration
async def test_server_process_starts_and_responds():
    """Spawn server subprocess, send initialize, verify response, terminate."""
    proc = await _spawn_server()
    try:
        resp = await _send_and_recv(proc, _init_request(req_id=1))
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "result" in resp
        assert resp["result"]["serverInfo"]["name"] == "networkx-mcp-server"
    finally:
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            proc.terminate()
            await proc.wait()


@pytest.mark.integration
async def test_server_process_handles_tool_call():
    """Spawn, initialize, create graph, get_info, terminate."""
    proc = await _spawn_server()
    try:
        # Initialize
        await _send_and_recv(proc, _init_request(req_id=1))
        # Send initialized notification (no id -> no response expected on stdout)
        notif = (
            json.dumps({"jsonrpc": "2.0", "method": "initialized", "params": {}}) + "\n"
        )
        proc.stdin.write(notif.encode())
        await proc.stdin.drain()

        # Create graph
        resp = await _send_and_recv(
            proc,
            _tool_call("create_graph", {"name": "sub_g", "directed": False}, req_id=2),
        )
        assert "error" not in resp
        data = json.loads(resp["result"]["content"][0]["text"])
        assert data["created"] == "sub_g"

        # Get info
        resp = await _send_and_recv(
            proc,
            _tool_call("get_info", {"graph": "sub_g"}, req_id=3),
        )
        assert "error" not in resp
        info = json.loads(resp["result"]["content"][0]["text"])
        assert info["nodes"] == 0
        assert info["edges"] == 0
    finally:
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            proc.terminate()
            await proc.wait()


# ===========================================================================
# Protocol compliance tests
# ===========================================================================


@pytest.mark.integration
async def test_jsonrpc_version_required(server):
    """Request without 'jsonrpc': '2.0' must return error -32600."""
    resp = await server.handle_request({"id": 1, "method": "initialize", "params": {}})
    assert resp["error"]["code"] == -32600


@pytest.mark.integration
async def test_unknown_method_returns_error(server):
    """Unknown method must return error -32601."""
    await _init_server(server)

    resp = await server.handle_request(
        {"jsonrpc": "2.0", "id": 99, "method": "foo/bar", "params": {}}
    )
    assert resp["error"]["code"] == -32601
    assert "foo/bar" in resp["error"]["message"]


@pytest.mark.integration
async def test_notification_no_response(server):
    """Notification (no id) for 'initialized' must return None."""
    await _init_server(server)

    resp = await server.handle_request(
        {"jsonrpc": "2.0", "method": "initialized", "params": {}}
    )
    assert resp is None
