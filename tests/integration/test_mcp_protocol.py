#!/usr/bin/env python3
"""
Integration tests for MCP protocol compliance.
Tests the NetworkX MCP server against the actual MCP protocol specification.
"""

import json
import subprocess
import sys
import time

import pytest


class MCPProtocolTester:
    """Test MCP protocol compliance with the NetworkX server."""

    def __init__(self):
        self.process = None
        self.request_id = 0

    def start_server(self):
        """Start the MCP server process."""
        self.process = subprocess.Popen(
            [sys.executable, "-m", "networkx_mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        time.sleep(0.1)  # Give server time to start

    def stop_server(self):
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def send_request(self, method, params=None, notification=False):
        """Send a JSON-RPC request to the server."""
        if not self.process:
            raise RuntimeError("Server not started")

        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
        }

        if params is not None:
            request["params"] = params

        if not notification:
            request["id"] = self.request_id

        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str)
        self.process.stdin.flush()

        if notification:
            return None

        # Read response
        response_line = self.process.stdout.readline()
        if response_line:
            return json.loads(response_line.strip())
        return None

    def test_initialize_handshake(self):
        """Test the MCP initialization handshake."""
        # Test initialize request
        response = self.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"sampling": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        )

        assert response is not None
        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert "protocolVersion" in response["result"]
        assert "capabilities" in response["result"]
        assert "serverInfo" in response["result"]

        # Test initialized notification
        self.send_request("initialized", notification=True)

        return response["result"]

    def test_tools_list(self):
        """Test tools/list method."""
        response = self.send_request("tools/list")

        assert response is not None
        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert "tools" in response["result"]

        tools = response["result"]["tools"]
        assert isinstance(tools, list)
        assert len(tools) == 20  # Expected number of tools

        # Verify tool structure
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert isinstance(tool["inputSchema"], dict)
            assert "type" in tool["inputSchema"]
            assert tool["inputSchema"]["type"] == "object"

        return tools

    def test_tool_call(self, tool_name, arguments):
        """Test tools/call method."""
        response = self.send_request(
            "tools/call", {"name": tool_name, "arguments": arguments}
        )

        assert response is not None
        assert response["jsonrpc"] == "2.0"

        if "error" in response:
            return response["error"]

        assert "result" in response
        result = response["result"]
        assert "content" in result
        assert isinstance(result["content"], list)

        return result

    def test_error_handling(self):
        """Test error handling for invalid requests."""
        # Test invalid method
        response = self.send_request("invalid/method")
        assert response is not None
        assert "error" in response
        assert response["error"]["code"] == -32601  # Method not found

        # Test invalid tool call
        response = self.send_request(
            "tools/call", {"name": "nonexistent_tool", "arguments": {}}
        )
        assert response is not None
        assert "error" in response

        # Test malformed request (send raw text)
        self.process.stdin.write("invalid json\n")
        self.process.stdin.flush()

        response_line = self.process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            assert "error" in response
            assert response["error"]["code"] == -32700  # Parse error

    def test_complete_workflow(self):
        """Test a complete MCP workflow."""
        # 1. Initialize
        server_info = self.test_initialize_handshake()
        assert server_info["serverInfo"]["name"] == "networkx-minimal"

        # 2. List tools
        tools = self.test_tools_list()
        tool_names = [tool["name"] for tool in tools]

        # Verify expected tools exist
        expected_tools = [
            "create_graph",
            "add_nodes",
            "add_edges",
            "get_info",
            "degree_centrality",
            "pagerank",
            "shortest_path",
        ]
        for tool in expected_tools:
            assert tool in tool_names

        # 3. Create a graph
        result = self.test_tool_call(
            "create_graph", {"name": "test_graph", "directed": False}
        )
        assert result["content"][0]["type"] == "text"
        response_data = json.loads(result["content"][0]["text"])
        assert response_data["created"] == "test_graph"

        # 4. Add nodes
        result = self.test_tool_call(
            "add_nodes", {"graph": "test_graph", "nodes": ["A", "B", "C", "D"]}
        )
        response_data = json.loads(result["content"][0]["text"])
        assert response_data["added"] == 4

        # 5. Add edges
        result = self.test_tool_call(
            "add_edges",
            {
                "graph": "test_graph",
                "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]],
            },
        )
        response_data = json.loads(result["content"][0]["text"])
        assert response_data["added"] == 4

        # 6. Get graph info
        result = self.test_tool_call("get_info", {"graph": "test_graph"})
        response_data = json.loads(result["content"][0]["text"])
        assert response_data["nodes"] == 4
        assert response_data["edges"] == 4

        # 7. Calculate centrality
        result = self.test_tool_call(
            "degree_centrality", {"graph": "test_graph", "top_n": 2}
        )
        response_data = json.loads(result["content"][0]["text"])
        assert "centrality" in response_data
        assert "most_central" in response_data

        # 8. Test error case
        error_result = self.test_tool_call("get_info", {"graph": "nonexistent_graph"})
        # Server returns errors as content with isError flag
        assert "content" in error_result
        assert error_result.get("isError", False)

        return True


@pytest.fixture
def mcp_tester():
    """Fixture to provide MCP protocol tester."""
    tester = MCPProtocolTester()
    tester.start_server()
    yield tester
    tester.stop_server()


class TestMCPProtocol:
    """Test suite for MCP protocol compliance."""

    def test_initialize_handshake(self, mcp_tester):
        """Test MCP initialization handshake."""
        server_info = mcp_tester.test_initialize_handshake()
        assert server_info is not None
        assert "protocolVersion" in server_info
        assert "capabilities" in server_info
        assert "serverInfo" in server_info

    def test_tools_list(self, mcp_tester):
        """Test tools/list method."""
        # Initialize first
        mcp_tester.test_initialize_handshake()

        tools = mcp_tester.test_tools_list()
        assert len(tools) == 20

        # Verify specific tools exist
        tool_names = [tool["name"] for tool in tools]
        assert "create_graph" in tool_names
        assert "add_nodes" in tool_names
        assert "pagerank" in tool_names

    def test_basic_graph_operations(self, mcp_tester):
        """Test basic graph operations."""
        # Initialize
        mcp_tester.test_initialize_handshake()

        # Create graph
        result = mcp_tester.test_tool_call(
            "create_graph", {"name": "test_graph", "directed": False}
        )
        assert "content" in result

        # Add nodes
        result = mcp_tester.test_tool_call(
            "add_nodes", {"graph": "test_graph", "nodes": ["A", "B", "C"]}
        )
        assert "content" in result

        # Add edges
        result = mcp_tester.test_tool_call(
            "add_edges", {"graph": "test_graph", "edges": [["A", "B"], ["B", "C"]]}
        )
        assert "content" in result

        # Get info
        result = mcp_tester.test_tool_call("get_info", {"graph": "test_graph"})
        assert "content" in result
        response_data = json.loads(result["content"][0]["text"])
        assert response_data["nodes"] == 3
        assert response_data["edges"] == 2

    def test_network_analysis(self, mcp_tester):
        """Test network analysis tools."""
        # Initialize
        mcp_tester.test_initialize_handshake()

        # Create and populate graph
        mcp_tester.test_tool_call(
            "create_graph", {"name": "analysis_graph", "directed": False}
        )
        mcp_tester.test_tool_call(
            "add_nodes", {"graph": "analysis_graph", "nodes": ["A", "B", "C", "D"]}
        )
        mcp_tester.test_tool_call(
            "add_edges",
            {
                "graph": "analysis_graph",
                "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]],
            },
        )

        # Test degree centrality
        result = mcp_tester.test_tool_call(
            "degree_centrality", {"graph": "analysis_graph", "top_n": 3}
        )
        assert "content" in result
        response_data = json.loads(result["content"][0]["text"])
        assert "centrality" in response_data
        assert "most_central" in response_data

        # Test PageRank
        result = mcp_tester.test_tool_call(
            "pagerank", {"graph": "analysis_graph", "top_n": 3}
        )
        assert "content" in result
        response_data = json.loads(result["content"][0]["text"])
        assert "pagerank" in response_data
        assert "highest_rank" in response_data

        # Test shortest path
        result = mcp_tester.test_tool_call(
            "shortest_path", {"graph": "analysis_graph", "source": "A", "target": "C"}
        )
        assert "content" in result
        response_data = json.loads(result["content"][0]["text"])
        assert "path" in response_data
        assert "length" in response_data

    def test_error_handling(self, mcp_tester):
        """Test error handling."""
        # Initialize
        mcp_tester.test_initialize_handshake()

        # Test invalid tool
        response = mcp_tester.send_request(
            "tools/call", {"name": "nonexistent_tool", "arguments": {}}
        )
        # Server returns errors as content with isError flag
        assert "result" in response
        assert "content" in response["result"]
        assert response["result"].get("isError", False)

        # Test invalid method
        response = mcp_tester.send_request("invalid/method")
        assert "error" in response
        assert response["error"]["code"] == -32601

    def test_complete_workflow(self, mcp_tester):
        """Test complete MCP workflow."""
        result = mcp_tester.test_complete_workflow()
        assert result is True


class TestMCPPerformance:
    """Performance tests for MCP protocol."""

    def test_latency(self, mcp_tester):
        """Test request/response latency."""
        # Initialize
        mcp_tester.test_initialize_handshake()

        # Measure latency for different operations
        latencies = []

        for i in range(10):
            start_time = time.time()
            mcp_tester.test_tool_call(
                "create_graph", {"name": f"perf_graph_{i}", "directed": False}
            )
            latency = time.time() - start_time
            latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 0.1  # Should be under 100ms

        # Test with larger operations
        large_nodes = [f"node_{i}" for i in range(100)]
        start_time = time.time()
        mcp_tester.test_tool_call(
            "add_nodes", {"graph": "perf_graph_0", "nodes": large_nodes}
        )
        large_op_latency = time.time() - start_time

        assert large_op_latency < 1.0  # Should be under 1 second

    def test_throughput(self, mcp_tester):
        """Test request throughput."""
        # Initialize
        mcp_tester.test_initialize_handshake()

        # Create test graph
        mcp_tester.test_tool_call(
            "create_graph", {"name": "throughput_graph", "directed": False}
        )

        # Measure throughput
        start_time = time.time()
        num_requests = 50

        for i in range(num_requests):
            mcp_tester.test_tool_call(
                "add_nodes", {"graph": "throughput_graph", "nodes": [f"node_{i}"]}
            )

        total_time = time.time() - start_time
        throughput = num_requests / total_time

        assert throughput > 10  # Should handle at least 10 requests per second


def run_integration_tests():
    """Run all integration tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_integration_tests()
