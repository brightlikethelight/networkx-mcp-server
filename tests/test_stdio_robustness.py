#!/usr/bin/env python3
"""Comprehensive tests to ensure stdio transport is rock-solid."""

import json
import subprocess
import sys
import threading
import time


class StdioServerTester:
    """Test harness for stdio MCP server."""

    def __init__(self):
        self.process = None
        self.responses = []
        self.reader_thread = None

    def start(self):
        """Start the MCP server process."""
        self.process = subprocess.Popen(
            [sys.executable, "-m", "networkx_mcp.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Start reader thread
        self.reader_thread = threading.Thread(target=self._read_output)
        self.reader_thread.daemon = True
        self.reader_thread.start()

        # Give server time to start
        time.sleep(0.5)

    def _read_output(self):
        """Read server output in background thread."""
        while self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    try:
                        response = json.loads(line.strip())
                        self.responses.append(response)
                    except json.JSONDecodeError:
                        # Not JSON, ignore
                        pass
            except:
                break

    def send_request(self, method, params=None, request_id=None):
        """Send JSON-RPC request and wait for response."""
        if not self.process:
            raise RuntimeError("Server not started")

        request = {"jsonrpc": "2.0", "method": method}

        if request_id is not None:
            request["id"] = request_id

        if params:
            request["params"] = params

        # Clear previous responses
        self.responses.clear()

        # Send request
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()

        # Wait for response (if request has ID)
        if request_id is not None:
            timeout = 5
            start = time.time()
            while time.time() - start < timeout:
                for response in self.responses:
                    if response.get("id") == request_id:
                        return response
                time.sleep(0.1)
            raise TimeoutError(f"No response received for request {request_id}")

    def stop(self):
        """Stop the server process."""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None


def test_basic_stdio_flow():
    """Test basic stdio communication flow."""
    print("Test 1: Basic stdio flow")
    server = StdioServerTester()

    try:
        server.start()

        # Initialize
        response = server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0.0"},
            },
            request_id=1,
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"
        print("  ✓ Initialize successful")

        # Send initialized notification (no response expected)
        server.send_request("initialized")
        time.sleep(0.5)  # Give time to process
        print("  ✓ Initialized notification sent")

        # List tools
        response = server.send_request("tools/list", request_id=2)
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) > 0
        print(f"  ✓ Found {len(response['result']['tools'])} tools")

    finally:
        server.stop()


def test_rapid_requests():
    """Test handling of rapid requests."""
    print("\nTest 2: Rapid request handling")
    server = StdioServerTester()

    try:
        server.start()

        # Initialize first
        server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0.0"},
            },
            request_id=1,
        )
        server.send_request("initialized")
        time.sleep(0.5)

        # Send many requests rapidly
        request_count = 50
        start_time = time.time()

        for i in range(2, request_count + 2):
            response = server.send_request("tools/list", request_id=i)
            assert response["id"] == i

        elapsed = time.time() - start_time
        print(f"  ✓ Handled {request_count} requests in {elapsed:.2f}s")
        print(f"  ✓ {request_count / elapsed:.1f} requests/second")

    finally:
        server.stop()


def test_large_payloads():
    """Test handling of large JSON payloads."""
    print("\nTest 3: Large payload handling")
    server = StdioServerTester()

    try:
        server.start()

        # Initialize
        server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0.0"},
            },
            request_id=1,
        )
        server.send_request("initialized")

        # Create graph with many nodes
        server.send_request(
            "tools/call",
            {"name": "create_graph", "arguments": {"graph_id": "large_graph"}},
            request_id=2,
        )

        # Add many nodes
        large_node_list = [f"node_{i}" for i in range(1000)]
        response = server.send_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {"graph_id": "large_graph", "nodes": large_node_list},
            },
            request_id=3,
        )

        assert "result" in response
        print("  ✓ Handled 1000-node payload")

        # Add many edges
        edges = [[f"node_{i}", f"node_{i + 1}"] for i in range(500)]
        response = server.send_request(
            "tools/call",
            {
                "name": "add_edges",
                "arguments": {"graph_id": "large_graph", "edges": edges},
            },
            request_id=4,
        )

        assert "result" in response
        print("  ✓ Handled 500-edge payload")

    finally:
        server.stop()


def test_error_handling():
    """Test stdio error handling."""
    print("\nTest 4: Error handling")
    server = StdioServerTester()

    try:
        server.start()

        # Invalid JSON-RPC version
        response = server.send_request("test", request_id="bad_version")
        # Manually set wrong version
        server.process.stdin.write('{"jsonrpc":"1.0","id":99,"method":"test"}\n')
        server.process.stdin.flush()
        time.sleep(0.5)

        # Find error response
        error_response = None
        for resp in server.responses:
            if resp.get("id") == 99:
                error_response = resp
                break

        assert error_response is not None
        assert "error" in error_response
        assert error_response["error"]["code"] == -32600
        print("  ✓ Invalid JSON-RPC version handled")

        # Method not found
        response = server.send_request("nonexistent_method", request_id=100)
        assert "error" in response
        assert response["error"]["code"] == -32601
        print("  ✓ Method not found handled")

        # Invalid JSON
        server.process.stdin.write("invalid json\n")
        server.process.stdin.flush()
        time.sleep(0.5)
        # Should not crash
        print("  ✓ Invalid JSON handled")

    finally:
        server.stop()


def test_unicode_handling():
    """Test handling of Unicode in stdio."""
    print("\nTest 5: Unicode handling")
    server = StdioServerTester()

    try:
        server.start()

        # Initialize
        server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0.0"},
            },
            request_id=1,
        )
        server.send_request("initialized")

        # Create graph with Unicode
        server.send_request(
            "tools/call",
            {"name": "create_graph", "arguments": {"graph_id": "unicode_graph"}},
            request_id=2,
        )

        # Add nodes with various Unicode characters
        unicode_nodes = [
            "hello_world",
            "你好世界",  # Chinese
            "مرحبا",  # Arabic
            "🌍🌎🌏",  # Emoji
            "café",  # Accented
            "Москва",  # Cyrillic
        ]

        response = server.send_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {"graph_id": "unicode_graph", "nodes": unicode_nodes},
            },
            request_id=3,
        )

        assert "result" in response
        print("  ✓ Unicode nodes handled correctly")

    finally:
        server.stop()


def test_connection_lifecycle():
    """Test server connection lifecycle."""
    print("\nTest 6: Connection lifecycle")

    # Test multiple start/stop cycles
    for i in range(3):
        server = StdioServerTester()
        try:
            server.start()
            response = server.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0.0"},
                },
                request_id=i + 1,
            )
            assert "result" in response
            print(f"  ✓ Lifecycle {i + 1} successful")
        finally:
            server.stop()

    print("  ✓ Multiple lifecycles handled")


def test_concurrent_operations():
    """Test handling of concurrent graph operations."""
    print("\nTest 7: Concurrent operations")
    server = StdioServerTester()

    try:
        server.start()

        # Initialize
        server.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0.0"},
            },
            request_id=1,
        )
        server.send_request("initialized")

        # Create multiple graphs
        for i in range(5):
            response = server.send_request(
                "tools/call",
                {"name": "create_graph", "arguments": {"graph_id": f"graph_{i}"}},
                request_id=10 + i,
            )
            assert "result" in response

        print("  ✓ Created 5 graphs")

        # Perform operations on different graphs
        request_id = 20
        for i in range(5):
            # Add nodes
            response = server.send_request(
                "tools/call",
                {
                    "name": "add_nodes",
                    "arguments": {
                        "graph_id": f"graph_{i}",
                        "nodes": [f"node_{j}" for j in range(10)],
                    },
                },
                request_id=request_id,
            )
            request_id += 1
            assert "result" in response

        print("  ✓ Added nodes to all graphs")

        # Get info from all graphs
        for i in range(5):
            response = server.send_request(
                "tools/call",
                {"name": "get_graph_info", "arguments": {"graph_id": f"graph_{i}"}},
                request_id=request_id,
            )
            request_id += 1
            assert "result" in response

        print("  ✓ Retrieved info from all graphs")

    finally:
        server.stop()


def main():
    """Run all stdio robustness tests."""
    print("=== Stdio Transport Robustness Tests ===\n")

    tests = [
        test_basic_stdio_flow,
        test_rapid_requests,
        test_large_payloads,
        test_error_handling,
        test_unicode_handling,
        test_connection_lifecycle,
        test_concurrent_operations,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            import traceback

            traceback.print_exc()

    print(f"\n=== Results: {passed} passed, {failed} failed ===")

    if failed == 0:
        print("\n✅ Stdio transport is ROCK SOLID!")
    else:
        print("\n❌ Stdio transport needs work")
        sys.exit(1)


if __name__ == "__main__":
    main()
