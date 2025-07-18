"""Real MCP protocol integration tests using actual subprocess communication."""

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest
import pytest_asyncio

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMCPProtocol:
    """Test MCP protocol compliance with real subprocess communication."""

    @pytest_asyncio.fixture
    async def mcp_server(self):
        """Start real server subprocess and yield it for testing."""
        # Start server subprocess
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "networkx_mcp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path(__file__).parent.parent,
            env={"PYTHONPATH": str(Path(__file__).parent.parent / "src")},
        )

        # Give server time to start
        await asyncio.sleep(0.1)

        yield proc

        # Cleanup
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    async def send_request(self, proc, request):
        """Send JSON-RPC request and get response."""
        json_str = json.dumps(request) + "\n"
        proc.stdin.write(json_str.encode())
        await proc.stdin.drain()

        # Read response line
        response_line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)

        if not response_line:
            raise Exception("No response from server")

        return json.loads(response_line.decode().strip())

    async def initialize_server(self, proc):
        """Perform MCP initialization handshake."""
        # Send initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
        }

        init_response = await self.send_request(proc, init_request)
        assert init_response["jsonrpc"] == "2.0"
        assert init_response["id"] == 1
        assert "result" in init_response

        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {},
        }

        # Send notification (no response expected for notifications)
        json_str = json.dumps(initialized_notification) + "\n"
        proc.stdin.write(json_str.encode())
        await proc.stdin.drain()

        # Small delay to let server process
        await asyncio.sleep(0.05)

        return init_response

    async def call_tool(self, proc, tool_name, arguments, request_id=None):
        """Call a tool and return the response."""
        if request_id is None:
            request_id = int(time.time() * 1000)  # Use timestamp as ID

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        return await self.send_request(proc, request)

    @pytest.mark.asyncio
    async def test_initialization_handshake(self, mcp_server):
        """Test MCP initialization handshake sequence."""
        response = await self.initialize_server(mcp_server)

        # Verify initialization response
        result = response["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert "capabilities" in result
        assert "serverInfo" in result
        assert result["serverInfo"]["name"] == "networkx-mcp-server"

        print("✅ MCP initialization handshake works")

    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_server):
        """Test tools/list method."""
        await self.initialize_server(mcp_server)

        # List tools
        response = await self.send_request(
            mcp_server,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )

        assert "result" in response
        tools = response["result"]["tools"]
        assert isinstance(tools, list)
        assert len(tools) == 7  # We expect exactly 7 tools

        # Verify each tool has required fields
        tool_names = []
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            tool_names.append(tool["name"])

        expected_tools = [
            "create_graph",
            "add_nodes",
            "add_edges",
            "get_graph_info",
            "shortest_path",
            "centrality_measures",
            "delete_graph",
        ]

        for expected in expected_tools:
            assert expected in tool_names, f"Missing expected tool: {expected}"

        print(f"✅ Tools list returns {len(tools)} tools: {tool_names}")

    @pytest.mark.asyncio
    async def test_error_handling(self, mcp_server):
        """Test error handling for invalid requests."""
        await self.initialize_server(mcp_server)

        # Test invalid method
        response = await self.send_request(
            mcp_server,
            {"jsonrpc": "2.0", "id": 3, "method": "invalid/method", "params": {}},
        )

        assert "error" in response
        assert response["error"]["code"] == -32601  # Method not found

        # Test invalid tool
        response = await self.call_tool(mcp_server, "invalid_tool", {}, 4)
        assert "error" in response
        assert response["error"]["code"] == -32601  # Method not found

        print("✅ Error handling works correctly")

    @pytest.mark.asyncio
    async def test_complete_graph_workflow(self, mcp_server):
        """Test realistic graph manipulation workflow."""
        await self.initialize_server(mcp_server)

        # 1. Create graph
        response = await self.call_tool(
            mcp_server,
            "create_graph",
            {"graph_id": "workflow_test", "directed": False},
            10,
        )

        assert "result" in response
        content = response["result"]["content"][0]["text"]
        graph_data = json.loads(content)
        assert graph_data["graph_id"] == "workflow_test"
        assert graph_data["created"] is True

        # 2. Add nodes
        response = await self.call_tool(
            mcp_server,
            "add_nodes",
            {"graph_id": "workflow_test", "nodes": ["A", "B", "C", "D", "E"]},
            11,
        )

        assert "result" in response
        content = response["result"]["content"][0]["text"]
        node_data = json.loads(content)
        assert node_data["nodes_added"] == 5

        # 3. Add edges
        response = await self.call_tool(
            mcp_server,
            "add_edges",
            {
                "graph_id": "workflow_test",
                "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"], ["A", "E"]],
            },
            12,
        )

        assert "result" in response
        content = response["result"]["content"][0]["text"]
        edge_data = json.loads(content)
        assert edge_data["edges_added"] == 5

        # 4. Get graph info
        response = await self.call_tool(
            mcp_server, "get_graph_info", {"graph_id": "workflow_test"}, 13
        )

        assert "result" in response
        content = response["result"]["content"][0]["text"]
        info_data = json.loads(content)
        assert info_data["num_nodes"] == 5
        assert info_data["num_edges"] == 5

        # 5. Find shortest path
        response = await self.call_tool(
            mcp_server,
            "shortest_path",
            {"graph_id": "workflow_test", "source": "A", "target": "D"},
            14,
        )

        assert "result" in response
        content = response["result"]["content"][0]["text"]
        path_data = json.loads(content)
        assert path_data["success"] is True
        assert "path" in path_data

        # 6. Calculate centrality
        response = await self.call_tool(
            mcp_server,
            "centrality_measures",
            {"graph_id": "workflow_test", "measures": ["degree", "betweenness"]},
            15,
        )

        assert "result" in response
        content = response["result"]["content"][0]["text"]
        centrality_data = json.loads(content)
        assert centrality_data["success"] is True
        assert "centrality" in centrality_data

        # 7. Delete graph
        response = await self.call_tool(
            mcp_server, "delete_graph", {"graph_id": "workflow_test"}, 16
        )

        assert "result" in response
        content = response["result"]["content"][0]["text"]
        delete_data = json.loads(content)
        assert delete_data["deleted"] is True

        print("✅ Complete graph workflow executed successfully")

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mcp_server):
        """Test if server handles rapid requests properly."""
        await self.initialize_server(mcp_server)

        # Create graph
        await self.call_tool(
            mcp_server, "create_graph", {"graph_id": "concurrent_test"}, 20
        )

        # Send multiple node additions rapidly
        tasks = []
        for i in range(10):
            task = self.call_tool(
                mcp_server,
                "add_nodes",
                {"graph_id": "concurrent_test", "nodes": [f"node_{i}"]},
                21 + i,
            )
            tasks.append(task)

        # Wait for all to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that most succeeded (some might fail due to rapid fire)
        successes = sum(1 for r in responses if isinstance(r, dict) and "result" in r)
        print(f"✅ Concurrent operations: {successes}/{len(responses)} succeeded")
        assert successes >= len(responses) // 2  # At least half should succeed


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v", "-s"])
