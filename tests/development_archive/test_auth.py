#!/usr/bin/env python3
"""Test authentication functionality."""

import json
import os
import subprocess
import sys
import time


def test_auth():
    """Test authentication features."""
    print("🔐 TESTING AUTHENTICATION 🔐")

    # First, generate an API key
    print("\n=== GENERATING API KEY ===")
    result = subprocess.run(
        [sys.executable, "-m", "networkx_mcp.auth", "generate", "test-client"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Failed to generate API key: {result.stderr}")
        return

    # Extract API key from output
    lines = result.stdout.strip().split("\n")
    api_key = None
    for i, line in enumerate(lines):
        if line.startswith("nxmcp_"):
            api_key = line.strip()
            break

    if not api_key:
        print("❌ Could not extract API key from output")
        print(f"Output: {result.stdout}")
        return

    print(f"✅ Generated API key: {api_key[:20]}...")

    # List API keys
    print("\n=== LISTING API KEYS ===")
    result = subprocess.run(
        [sys.executable, "-m", "networkx_mcp.auth", "list"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    # Test server without auth
    print("\n=== TEST 1: SERVER WITHOUT AUTH ===")
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

    time.sleep(0.5)

    # Initialize
    response = send_request(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "auth-test", "version": "1.0.0"},
        },
    )

    # Try creating a graph without auth
    response = send_request(
        "tools/call",
        {
            "name": "create_graph",
            "arguments": {"name": "test_graph", "directed": False},
        },
    )

    if "result" in response and not response["result"].get("isError"):
        print("✅ Graph created without authentication (as expected)")
    else:
        print(f"❌ Failed to create graph: {response}")

    process.terminate()
    process.wait()

    # Test server WITH auth
    print("\n=== TEST 2: SERVER WITH AUTH REQUIRED ===")
    env = os.environ.copy()
    env["NETWORKX_MCP_AUTH"] = "true"

    process = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    time.sleep(0.5)

    # Initialize (should work without auth)
    response = send_request(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "auth-test", "version": "1.0.0"},
        },
    )

    if "result" in response:
        print("✅ Initialize works without auth")

    # Try creating a graph without API key
    response = send_request(
        "tools/call",
        {
            "name": "create_graph",
            "arguments": {"name": "test_graph", "directed": False},
        },
    )

    if "error" in response:
        print(
            f"✅ Correctly rejected request without API key: {response['error']['message']}"
        )
    else:
        print("❌ Should have rejected request without API key")

    # Try with invalid API key
    response = send_request(
        "tools/call",
        {
            "name": "create_graph",
            "arguments": {"name": "test_graph", "directed": False},
            "api_key": "invalid_key",
        },
    )

    if "error" in response:
        print(f"✅ Correctly rejected invalid API key: {response['error']['message']}")
    else:
        print("❌ Should have rejected invalid API key")

    # Try with valid API key
    response = send_request(
        "tools/call",
        {
            "name": "create_graph",
            "arguments": {"name": "test_graph", "directed": False},
            "api_key": api_key,
        },
    )

    if "result" in response and not response["result"].get("isError"):
        print("✅ Graph created with valid API key")
    else:
        print(f"❌ Failed to create graph with valid API key: {response}")

    # Test read operation with API key
    response = send_request(
        "tools/call",
        {"name": "get_info", "arguments": {"graph": "test_graph"}, "api_key": api_key},
    )

    if "result" in response and not response["result"].get("isError"):
        print("✅ Read operation works with API key")
    else:
        print(f"❌ Read operation failed: {response}")

    process.terminate()
    process.wait()

    # Check stderr for auth messages
    stderr = process.stderr.read()
    if "authentication enabled" in stderr:
        print("✅ Server correctly logged authentication status")

    print("\n=== AUTHENTICATION TEST COMPLETE ===")


if __name__ == "__main__":
    test_auth()
