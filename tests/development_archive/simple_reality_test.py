#!/usr/bin/env python3
"""Simple brutal reality test without complex file operations."""

import json
import subprocess
import sys
import time


def test_server_reality():
    """Test the actual server functionality."""
    print("🔥 BRUTAL REALITY CHECK 🔥")

    # Start server
    process = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def send_request(method, params=None):
        request = {"jsonrpc": "2.0", "method": method, "id": 1}
        if params:
            request["params"] = params

        try:
            request_str = json.dumps(request) + "\n"
            process.stdin.write(request_str)
            process.stdin.flush()

            response_line = process.stdout.readline()
            return json.loads(response_line.strip()) if response_line else None
        except Exception as e:
            return {"error": str(e)}

    try:
        print("\n=== INITIALIZATION ===")
        init_response = send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "brutal-test", "version": "1.0.0"},
            },
        )
        print(
            f"✅ Init: {init_response['result'] if 'result' in init_response else 'FAILED'}"
        )

        print("\n=== TOOLS LIST ===")
        tools_response = send_request("tools/list")
        if tools_response and "result" in tools_response:
            tools = tools_response["result"]["tools"]
            print(f"✅ Found {len(tools)} tools: {[t['name'] for t in tools]}")
        else:
            print("❌ Failed to get tools list")
            return

        print("\n=== BASIC GRAPH OPERATIONS ===")

        # Create graph
        response = send_request(
            "tools/call",
            {
                "name": "create_graph",
                "arguments": {"name": "brutal_test", "directed": False},
            },
        )
        print(f"Create graph: {'✅' if 'result' in response else '❌'}")

        # Add nodes
        response = send_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {
                    "graph": "brutal_test",
                    "nodes": ["A", "B", "C", "D", "E"],
                },
            },
        )
        print(f"Add nodes: {'✅' if 'result' in response else '❌'}")

        # Add edges
        response = send_request(
            "tools/call",
            {
                "name": "add_edges",
                "arguments": {
                    "graph": "brutal_test",
                    "edges": [
                        ["A", "B"],
                        ["B", "C"],
                        ["C", "D"],
                        ["D", "E"],
                        ["E", "A"],
                    ],
                },
            },
        )
        print(f"Add edges: {'✅' if 'result' in response else '❌'}")

        # Get info
        response = send_request(
            "tools/call", {"name": "get_info", "arguments": {"graph": "brutal_test"}}
        )
        if "result" in response:
            data = json.loads(response["result"]["content"][0]["text"])
            print(f"✅ Graph info: {data['nodes']} nodes, {data['edges']} edges")
        else:
            print("❌ Get info failed")

        print("\n=== NETWORK ANALYSIS ===")

        # PageRank
        response = send_request(
            "tools/call",
            {"name": "pagerank", "arguments": {"graph": "brutal_test", "top_n": 3}},
        )
        print(f"PageRank: {'✅' if 'result' in response else '❌'}")

        # Degree centrality
        response = send_request(
            "tools/call",
            {
                "name": "degree_centrality",
                "arguments": {"graph": "brutal_test", "top_n": 3},
            },
        )
        print(f"Degree centrality: {'✅' if 'result' in response else '❌'}")

        # Shortest path
        response = send_request(
            "tools/call",
            {
                "name": "shortest_path",
                "arguments": {"graph": "brutal_test", "source": "A", "target": "C"},
            },
        )
        print(f"Shortest path: {'✅' if 'result' in response else '❌'}")

        print("\n=== STRESS TEST ===")

        # Large graph
        large_nodes = [f"node_{i}" for i in range(1000)]
        start_time = time.time()
        response = send_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {"graph": "brutal_test", "nodes": large_nodes},
            },
        )
        duration = time.time() - start_time
        print(
            f"Add 1000 nodes: {'✅' if 'result' in response else '❌'} ({duration:.3f}s)"
        )

        # PageRank on large graph
        start_time = time.time()
        response = send_request(
            "tools/call",
            {"name": "pagerank", "arguments": {"graph": "brutal_test", "top_n": 10}},
        )
        duration = time.time() - start_time
        print(
            f"PageRank on 1000+ nodes: {'✅' if 'result' in response else '❌'} ({duration:.3f}s)"
        )

        print("\n=== ERROR HANDLING ===")

        # Nonexistent graph
        response = send_request(
            "tools/call", {"name": "get_info", "arguments": {"graph": "nonexistent"}}
        )
        if "result" in response and response["result"].get("isError"):
            print("✅ Nonexistent graph error handled correctly")
        else:
            print("❌ Nonexistent graph error not handled")

        # Invalid tool
        response = send_request("tools/call", {"name": "invalid_tool", "arguments": {}})
        if "result" in response and response["result"].get("isError"):
            print("✅ Invalid tool error handled correctly")
        else:
            print("❌ Invalid tool error not handled")

        print("\n=== ACADEMIC TOOLS ===")

        # DOI resolution (may fail without internet)
        response = send_request(
            "tools/call",
            {"name": "resolve_doi", "arguments": {"doi": "10.1038/nature12373"}},
        )
        print(
            f"DOI resolution: {'✅' if 'result' in response and not response['result'].get('isError') else '⚠️ (network required)'}"
        )

        # Citation network (may fail without internet)
        response = send_request(
            "tools/call",
            {
                "name": "build_citation_network",
                "arguments": {
                    "graph_name": "citation_test",
                    "paper_ids": ["10.1038/nature12373"],
                },
            },
        )
        print(
            f"Citation network: {'✅' if 'result' in response and not response['result'].get('isError') else '⚠️ (network required)'}"
        )

        print("\n=== VISUALIZATION ===")

        # Visualization (requires matplotlib)
        response = send_request(
            "tools/call",
            {
                "name": "visualize_graph",
                "arguments": {"graph": "brutal_test", "layout": "spring"},
            },
        )
        if "result" in response and not response["result"].get("isError"):
            result_data = json.loads(response["result"]["content"][0]["text"])
            if "visualization" in result_data:
                print("✅ Visualization generates base64 image")
            else:
                print("❌ Visualization doesn't generate image")
        else:
            print("❌ Visualization failed")

    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")

    finally:
        process.terminate()
        process.wait()

        # Check stderr for errors
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"\n⚠️ Server errors:\n{stderr_output}")


if __name__ == "__main__":
    test_server_reality()
