#!/usr/bin/env python3
"""Test monitoring functionality."""

import json
import os
import subprocess
import sys
import time


def test_monitoring():
    """Test monitoring features."""
    print("📊 TESTING MONITORING 📊")

    # Test server WITH monitoring
    print("\n=== TEST: SERVER WITH MONITORING ===")
    env = os.environ.copy()
    env["NETWORKX_MCP_MONITORING"] = "true"

    process = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
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

    time.sleep(0.5)

    # Initialize
    response = send_request(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "monitoring-test", "version": "1.0.0"},
        },
    )

    if "result" in response:
        print("✅ Initialize successful")

    # Check if health_status tool is available
    response = send_request("tools/list")
    if "result" in response:
        tools = response["result"]["tools"]
        tool_names = [tool["name"] for tool in tools]
        if "health_status" in tool_names:
            print("✅ Health status tool available")
        else:
            print("❌ Health status tool not found")
            print(f"Available tools: {tool_names}")

    # Create some activity to monitor
    send_request(
        "tools/call",
        {
            "name": "create_graph",
            "arguments": {"name": "monitoring_test", "directed": False},
        },
    )

    send_request(
        "tools/call",
        {
            "name": "add_nodes",
            "arguments": {"graph": "monitoring_test", "nodes": ["A", "B", "C"]},
        },
    )

    send_request(
        "tools/call",
        {
            "name": "add_edges",
            "arguments": {
                "graph": "monitoring_test",
                "edges": [["A", "B"], ["B", "C"]],
            },
        },
    )

    # Wait a moment for activity to register
    time.sleep(0.1)

    # Check health status
    response = send_request("tools/call", {"name": "health_status", "arguments": {}})

    if "result" in response and not response["result"].get("isError"):
        health_data = json.loads(response["result"]["content"][0]["text"])
        print("✅ Health status retrieved successfully!")
        print(f"   Status: {health_data.get('status', 'N/A')}")
        print(f"   Uptime: {health_data.get('uptime_human', 'N/A')}")

        metrics = health_data.get("metrics", {})
        if "requests" in metrics:
            req_metrics = metrics["requests"]
            print(
                f"   Requests: {req_metrics.get('total', 0)} total, {req_metrics.get('errors', 0)} errors"
            )

        if "system" in metrics:
            sys_metrics = metrics["system"]
            print(f"   Memory: {sys_metrics.get('memory_mb', 0)} MB")
            print(f"   CPU: {sys_metrics.get('cpu_percent', 0)}%")

        if "graphs" in metrics:
            graph_metrics = metrics["graphs"]
            print(
                f"   Graphs: {graph_metrics.get('count', 0)} graphs, {graph_metrics.get('total_nodes', 0)} nodes, {graph_metrics.get('total_edges', 0)} edges"
            )
    else:
        print(f"❌ Health status failed: {response}")

    process.terminate()
    process.wait()

    # Check stderr for monitoring messages
    stderr = process.stderr.read()
    if "monitoring enabled" in stderr:
        print("✅ Server correctly logged monitoring status")

    print("\n=== MONITORING TEST COMPLETE ===")


if __name__ == "__main__":
    test_monitoring()
