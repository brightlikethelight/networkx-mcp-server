#!/usr/bin/env python3
"""Simple MCP client to test the NetworkX MCP server."""

import json
import subprocess
import sys


def send_request(process, request):
    """Send a request and get response."""
    request_str = json.dumps(request)
    process.stdin.write(request_str + "\n")
    process.stdin.flush()

    # Read response
    response_line = process.stdout.readline()
    if response_line:
        return json.loads(response_line.strip())
    return None


def main():
    # Start the server
    process = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        # Test 1: Initialize
        print("Testing initialize...")
        response = send_request(
            process,
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
                "id": 1,
            },
        )
        print(f"Initialize response: {response}")

        # Test 2: Initialized notification
        print("\nSending initialized...")
        send_request(process, {"jsonrpc": "2.0", "method": "initialized"})

        # Test 3: List tools
        print("\nTesting tools/list...")
        response = send_request(
            process, {"jsonrpc": "2.0", "method": "tools/list", "id": 2}
        )
        print(f"Tools count: {len(response.get('result', {}).get('tools', []))}")

        # Test 4: Create a graph
        print("\nTesting create_graph...")
        response = send_request(
            process,
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "create_graph",
                    "arguments": {"name": "test_graph", "directed": False},
                },
                "id": 3,
            },
        )
        print(f"Create graph response: {response}")

        # Test 5: Add nodes
        print("\nTesting add_nodes...")
        response = send_request(
            process,
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "add_nodes",
                    "arguments": {"graph": "test_graph", "nodes": ["A", "B", "C", "D"]},
                },
                "id": 4,
            },
        )
        print(f"Add nodes response: {response}")

        # Test 6: Add edges
        print("\nTesting add_edges...")
        response = send_request(
            process,
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "add_edges",
                    "arguments": {
                        "graph": "test_graph",
                        "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]],
                    },
                },
                "id": 5,
            },
        )
        print(f"Add edges response: {response}")

        # Test 7: Get graph info
        print("\nTesting get_graph_info...")
        response = send_request(
            process,
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "get_graph_info",
                    "arguments": {"graph": "test_graph"},
                },
                "id": 6,
            },
        )
        print(f"Graph info response: {response}")

    finally:
        # Cleanup
        process.terminate()
        process.wait()

        # Print any errors
        stderr = process.stderr.read()
        if stderr:
            print(f"\nServer errors:\n{stderr}")


if __name__ == "__main__":
    main()
