"""Integration tests for MCP clients with NetworkX MCP server.

Tests the server with various MCP client implementations to ensure
compatibility with the official Python SDK and other clients.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import MCP SDK components
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import Resource, Tool

    MCP_SDK_AVAILABLE = True
except ImportError:
    MCP_SDK_AVAILABLE = False


@pytest.mark.skipif(not MCP_SDK_AVAILABLE, reason="MCP SDK not installed")
class TestPythonMCPSDK:
    """Test NetworkX MCP server with official Python MCP SDK."""

    @pytest.fixture
    async def mcp_client(self):
        """Create MCP client connected to NetworkX server."""
        # Path to our server module
        server_path = Path(__file__).parent.parent.parent / "src"

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "networkx_mcp", "--jsonrpc"],
            env={"PYTHONPATH": str(server_path)},
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    @pytest.mark.asyncio
    async def test_initialization(self, mcp_client):
        """Test MCP client initialization."""
        # Client should be initialized from fixture
        assert mcp_client is not None

    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_client):
        """Test listing available tools."""
        result = await mcp_client.list_tools()

        # Should have multiple tools
        assert len(result.tools) >= 15

        # Check for specific tools
        tool_names = [t.name for t in result.tools]
        assert "create_graph" in tool_names
        assert "add_nodes" in tool_names
        assert "add_edges" in tool_names
        assert "graph_info" in tool_names
        assert "shortest_path" in tool_names

    @pytest.mark.asyncio
    async def test_create_graph(self, mcp_client):
        """Test creating a graph."""
        result = await mcp_client.call_tool(
            "create_graph", {"name": "test_graph", "graph_type": "undirected"}
        )

        # Parse response
        content = json.loads(result.content[0].text)
        assert content["success"] is True
        assert content["name"] == "test_graph"
        assert content["type"] == "undirected"

    @pytest.mark.asyncio
    async def test_complete_workflow(self, mcp_client):
        """Test a complete graph workflow."""
        # 1. Create graph
        result = await mcp_client.call_tool(
            "create_graph", {"name": "workflow_graph", "graph_type": "directed"}
        )
        content = json.loads(result.content[0].text)
        assert content["success"] is True

        # 2. Add nodes
        result = await mcp_client.call_tool(
            "add_nodes",
            {"graph_name": "workflow_graph", "nodes": ["A", "B", "C", "D", "E"]},
        )
        content = json.loads(result.content[0].text)
        assert content["nodes_added"] == 5

        # 3. Add edges
        result = await mcp_client.call_tool(
            "add_edges",
            {
                "graph_name": "workflow_graph",
                "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"], ["A", "E"]],
            },
        )
        content = json.loads(result.content[0].text)
        assert content["edges_added"] == 5

        # 4. Get graph info
        result = await mcp_client.call_tool(
            "graph_info", {"graph_name": "workflow_graph"}
        )
        content = json.loads(result.content[0].text)
        assert content["nodes"] == 5
        assert content["edges"] == 5

        # 5. Find shortest path
        result = await mcp_client.call_tool(
            "shortest_path",
            {"graph_name": "workflow_graph", "source": "A", "target": "C"},
        )
        content = json.loads(result.content[0].text)
        assert content["path"] == ["A", "B", "C"]
        assert content["length"] == 2

    @pytest.mark.asyncio
    async def test_error_handling(self, mcp_client):
        """Test error handling."""
        # Try to get info for non-existent graph
        result = await mcp_client.call_tool(
            "graph_info", {"graph_name": "non_existent_graph"}
        )

        # Should return error
        assert result.isError is True
        assert "not found" in result.content[0].text.lower()


@pytest.mark.asyncio
class TestBatchOperations:
    """Test JSON-RPC batch operations."""

    async def send_batch_request(self, requests):
        """Send batch JSON-RPC request to server."""
        import subprocess

        server_path = Path(__file__).parent.parent.parent / "src"

        # Prepare batch request
        batch_json = json.dumps(requests)

        # Run server and send request
        proc = subprocess.Popen(
            [sys.executable, "-m", "networkx_mcp", "--jsonrpc"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": str(server_path)},
        )

        stdout, stderr = proc.communicate(input=batch_json, timeout=10)

        # Parse responses
        responses = []
        for line in stdout.strip().split("\n"):
            if line.startswith("[") or line.startswith("{"):
                responses.append(json.loads(line))

        return responses

    async def test_batch_requests(self):
        """Test batch JSON-RPC requests."""
        # Prepare batch request
        batch = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "batch-test", "version": "1.0"},
                },
            },
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "create_graph",
                    "arguments": {"name": "batch_graph", "graph_type": "undirected"},
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",  # Notification - no response expected
            },
        ]

        responses = await self.send_batch_request(batch)

        # Should get one batch response
        assert len(responses) == 1
        batch_response = responses[0]

        # Should be an array of 3 responses (no response for notification)
        assert isinstance(batch_response, list)
        assert len(batch_response) == 3

        # Check each response
        assert batch_response[0]["id"] == 1
        assert "result" in batch_response[0]

        assert batch_response[1]["id"] == 2
        assert "result" in batch_response[1]
        assert len(batch_response[1]["result"]["tools"]) >= 15

        assert batch_response[2]["id"] == 3
        assert "result" in batch_response[2]


class TestClaudeDesktopConfiguration:
    """Test Claude Desktop configuration format."""

    def test_generate_claude_config(self):
        """Generate Claude Desktop configuration."""
        config = {
            "networkx-mcp": {
                "command": sys.executable,
                "args": ["-m", "networkx_mcp", "--jsonrpc"],
                "env": {"PYTHONPATH": str(Path(__file__).parent.parent.parent / "src")},
            }
        }

        # Save configuration
        config_path = Path(__file__).parent / "claude_desktop_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Verify configuration
        assert config_path.exists()

        # Load and validate
        with open(config_path) as f:
            loaded = json.load(f)

        assert "networkx-mcp" in loaded
        assert loaded["networkx-mcp"]["command"] == sys.executable
        assert "--jsonrpc" in loaded["networkx-mcp"]["args"]


class TestConcurrentClients:
    """Test multiple concurrent MCP clients."""

    @pytest.mark.asyncio
    async def test_concurrent_clients(self):
        """Test multiple clients connecting concurrently."""

        async def create_client_and_test(client_id):
            """Create a client and perform operations."""
            server_path = Path(__file__).parent.parent.parent / "src"

            if not MCP_SDK_AVAILABLE:
                # Fallback to direct JSON-RPC testing
                return await self._test_with_jsonrpc(client_id, server_path)

            server_params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "networkx_mcp", "--jsonrpc"],
                env={"PYTHONPATH": str(server_path)},
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Create a graph
                    result = await session.call_tool(
                        "create_graph",
                        {"name": f"graph_{client_id}", "graph_type": "undirected"},
                    )

                    content = json.loads(result.content[0].text)
                    assert content["success"] is True

                    # Add some nodes
                    result = await session.call_tool(
                        "add_nodes",
                        {
                            "graph_name": f"graph_{client_id}",
                            "nodes": [f"node_{i}" for i in range(5)],
                        },
                    )

                    content = json.loads(result.content[0].text)
                    assert content["nodes_added"] == 5

                    return f"Client {client_id} completed"

        # Run 5 concurrent clients
        tasks = [create_client_and_test(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete successfully
        successful = [r for r in results if isinstance(r, str) and "completed" in r]
        assert len(successful) == 5

    async def _test_with_jsonrpc(self, client_id, server_path):
        """Fallback JSON-RPC testing when MCP SDK not available."""
        import subprocess

        # Simulate client operations via JSON-RPC
        requests = [
            {
                "jsonrpc": "2.0",
                "id": f"{client_id}_init",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": f"client_{client_id}", "version": "1.0"},
                },
            },
            {
                "jsonrpc": "2.0",
                "id": f"{client_id}_create",
                "method": "tools/call",
                "params": {
                    "name": "create_graph",
                    "arguments": {
                        "name": f"graph_{client_id}",
                        "graph_type": "undirected",
                    },
                },
            },
        ]

        for request in requests:
            proc = subprocess.Popen(
                [sys.executable, "-m", "networkx_mcp", "--jsonrpc"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "PYTHONPATH": str(server_path)},
            )

            stdout, _ = proc.communicate(input=json.dumps(request), timeout=5)

            # Verify response
            for line in stdout.strip().split("\n"):
                if line.startswith("{"):
                    response = json.loads(line)
                    assert "result" in response or "error" in response

        return f"Client {client_id} completed"


# Additional test utilities
def create_test_mcp_client():
    """Create a test MCP client for end-to-end testing."""
    if not MCP_SDK_AVAILABLE:

        class MockClient:
            async def call_tool(self, name, arguments):
                # Mock implementation
                return type(
                    "Result",
                    (),
                    {
                        "content": [
                            type(
                                "Content",
                                (),
                                {
                                    "text": json.dumps(
                                        {
                                            "success": True,
                                            "message": f"Mock {name} called",
                                        }
                                    )
                                },
                            )
                        ]
                    },
                )()

        return MockClient()

    # Real client implementation would go here
    return None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
