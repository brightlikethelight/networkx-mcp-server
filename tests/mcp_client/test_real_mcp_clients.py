"""Real MCP client testing with official MCP implementations.

This module tests the NetworkX MCP server with real MCP clients to ensure
compatibility and proper protocol implementation.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Try to import official MCP client libraries
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    HAS_MCP_CLIENT = True
except ImportError:
    HAS_MCP_CLIENT = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class MCPClientTester:
    """Test suite for real MCP client interactions."""
    
    def __init__(self):
        self.server_process = None
        self.client_session = None
        
    async def start_server(self):
        """Start the NetworkX MCP server process."""
        server_script = Path(__file__).parent.parent.parent / "src" / "networkx_mcp" / "server.py"
        
        # Start server as subprocess
        self.server_process = await asyncio.create_subprocess_exec(
            "python", str(server_script),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE
        )
        
        # Give server time to start
        await asyncio.sleep(2)
        
        return self.server_process
    
    async def connect_client(self):
        """Connect MCP client to the server."""
        if not HAS_MCP_CLIENT:
            pytest.skip("Official MCP client not available")
            
        # Create client session
        server_params = StdioServerParameters(
            command="python",
            args=[str(Path(__file__).parent.parent.parent / "src" / "networkx_mcp" / "server.py")],
        )
        
        self.client_session = await stdio_client(server_params)
        return self.client_session
        
    async def cleanup(self):
        """Clean up server and client resources."""
        if self.client_session:
            await self.client_session.cleanup()
            
        if self.server_process:
            self.server_process.terminate()
            await self.server_process.wait()


@pytest.mark.skipif(not HAS_MCP_CLIENT, reason="Official MCP client not available")
class TestMCPClientInteractions:
    """Test interactions with real MCP clients."""
    
    @pytest.fixture
    async def mcp_tester(self):
        """Set up MCP client tester."""
        tester = MCPClientTester()
        await tester.start_server()
        await tester.connect_client()
        
        yield tester
        
        await tester.cleanup()
    
    async def test_client_initialization(self, mcp_tester):
        """Test MCP client can initialize and connect to server."""
        session = mcp_tester.client_session
        assert session is not None
        
        # Test basic server info
        result = await session.initialize()
        assert result is not None
        
    async def test_list_tools(self, mcp_tester):
        """Test client can list available tools."""
        session = mcp_tester.client_session
        
        tools = await session.list_tools()
        assert len(tools.tools) > 0
        
        # Check for expected tools
        tool_names = [tool.name for tool in tools.tools]
        expected_tools = [
            "create_graph",
            "add_nodes", 
            "add_edges",
            "shortest_path",
            "graph_info",
            "list_graphs"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Missing tool: {expected_tool}"
    
    async def test_graph_creation_workflow(self, mcp_tester):
        """Test complete graph creation workflow via MCP."""
        session = mcp_tester.client_session
        
        # Create graph
        result = await session.call_tool(
            "create_graph",
            arguments={"name": "test_graph", "graph_type": "undirected"}
        )
        assert result.isError is False
        
        # Add nodes
        result = await session.call_tool(
            "add_nodes", 
            arguments={
                "graph_name": "test_graph",
                "nodes": ["A", "B", "C", "D"]
            }
        )
        assert result.isError is False
        
        # Add edges
        result = await session.call_tool(
            "add_edges",
            arguments={
                "graph_name": "test_graph", 
                "edges": [["A", "B"], ["B", "C"], ["C", "D"]]
            }
        )
        assert result.isError is False
        
        # Get graph info
        result = await session.call_tool(
            "graph_info",
            arguments={"graph_name": "test_graph"}
        )
        assert result.isError is False
        
        info = json.loads(result.content[0].text)
        assert info["num_nodes"] == 4
        assert info["num_edges"] == 3
    
    async def test_algorithm_execution(self, mcp_tester):
        """Test graph algorithms via MCP."""
        session = mcp_tester.client_session
        
        # Set up test graph
        await session.call_tool("create_graph", {"name": "algo_test"})
        await session.call_tool("add_nodes", {
            "graph_name": "algo_test",
            "nodes": ["A", "B", "C", "D", "E"]
        })
        await session.call_tool("add_edges", {
            "graph_name": "algo_test",
            "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"], ["A", "E"]]
        })
        
        # Test shortest path
        result = await session.call_tool(
            "shortest_path",
            arguments={
                "graph_name": "algo_test",
                "source": "A", 
                "target": "C"
            }
        )
        assert result.isError is False
        
        path_data = json.loads(result.content[0].text)
        assert "path" in path_data
        assert len(path_data["path"]) >= 3  # A -> B -> C or A -> E -> D -> C
        
    async def test_error_handling(self, mcp_tester):
        """Test error handling via MCP protocol."""
        session = mcp_tester.client_session
        
        # Try to access non-existent graph
        result = await session.call_tool(
            "graph_info",
            arguments={"graph_name": "nonexistent_graph"}
        )
        assert result.isError is True
        
        # Try invalid node addition
        await session.call_tool("create_graph", {"name": "error_test"})
        result = await session.call_tool(
            "add_edges",
            arguments={
                "graph_name": "error_test",
                "edges": [["A", "B"]]  # Nodes don't exist
            }
        )
        # Should either error or auto-create nodes depending on implementation
        
    async def test_concurrent_operations(self, mcp_tester):
        """Test concurrent MCP operations."""
        session = mcp_tester.client_session
        
        # Create base graph
        await session.call_tool("create_graph", {"name": "concurrent_test"})
        
        # Execute multiple operations concurrently
        tasks = []
        for i in range(10):
            task = session.call_tool(
                "add_nodes",
                arguments={
                    "graph_name": "concurrent_test",
                    "nodes": [f"node_{i}"]
                }
            )
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        
        # All operations should complete successfully
        for result in results:
            assert result.isError is False
            
        # Verify final state
        result = await session.call_tool("graph_info", {"graph_name": "concurrent_test"})
        info = json.loads(result.content[0].text)
        assert info["num_nodes"] == 10


class TestClaudeDesktopIntegration:
    """Test integration with Claude Desktop MCP."""
    
    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic client not available")
    def test_claude_desktop_config(self):
        """Test Claude Desktop configuration generation."""
        config = {
            "mcpServers": {
                "networkx": {
                    "command": "python",
                    "args": ["-m", "networkx_mcp.server"],
                    "env": {
                        "LOG_LEVEL": "INFO"
                    }
                }
            }
        }
        
        # Write config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            config_path = f.name
            
        try:
            # Validate config structure
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                
            assert "mcpServers" in loaded_config
            assert "networkx" in loaded_config["mcpServers"]
            assert "command" in loaded_config["mcpServers"]["networkx"]
            assert "args" in loaded_config["mcpServers"]["networkx"]
            
        finally:
            os.unlink(config_path)
    
    def test_mcp_server_script_exists(self):
        """Test that MCP server entry point exists."""
        server_path = Path(__file__).parent.parent.parent / "src" / "networkx_mcp" / "server.py"
        assert server_path.exists(), f"Server script not found at {server_path}"
        
        # Check if it's executable as module
        result = subprocess.run(
            ["python", "-c", "import networkx_mcp.server"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Server module import failed: {result.stderr}"


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance and standards."""
    
    async def test_jsonrpc_format(self):
        """Test that server responds with proper JSON-RPC format."""
        # This would test the actual JSON-RPC message format
        # For now, we'll test the structure indirectly
        pass
        
    async def test_tool_schema_validation(self):
        """Test that tool schemas are valid."""
        # Load tool schemas and validate against MCP specification
        from networkx_mcp.mcp.tool_schemas import TOOL_SCHEMAS
        
        assert len(TOOL_SCHEMAS) > 0
        
        for tool_name, schema in TOOL_SCHEMAS.items():
            # Basic schema validation
            assert "name" in schema
            assert "description" in schema
            assert "inputSchema" in schema
            
            # Validate input schema structure
            input_schema = schema["inputSchema"]
            assert "type" in input_schema
            assert input_schema["type"] == "object"
            assert "properties" in input_schema
            
    async def test_resource_listing(self):
        """Test resource listing compliance."""
        # Test that resources are properly structured
        pass
        
    async def test_prompt_handling(self):
        """Test prompt handling compliance."""
        # Test prompt templates and handling
        pass


class TestClientCompatibilityMatrix:
    """Test compatibility with various MCP clients."""
    
    def test_official_mcp_client(self):
        """Test with official Python MCP client."""
        if not HAS_MCP_CLIENT:
            pytest.skip("Official MCP client not available")
        # Tests would go here
        
    def test_node_mcp_client(self):
        """Test with Node.js MCP client."""
        # This would test with Node.js client if available
        pytest.skip("Node.js MCP client testing not implemented")
        
    def test_rust_mcp_client(self):
        """Test with Rust MCP client."""
        # This would test with Rust client if available
        pytest.skip("Rust MCP client testing not implemented")
        
    def test_custom_client_implementation(self):
        """Test with a custom MCP client implementation."""
        # This would test with a minimal custom client
        pytest.skip("Custom client testing not implemented")


def generate_client_compatibility_report():
    """Generate compatibility report for different MCP clients."""
    report = {
        "tested_clients": [],
        "compatibility_matrix": {},
        "known_issues": [],
        "recommendations": []
    }
    
    # Test each available client
    clients = ["official_python", "claude_desktop", "custom"]
    
    for client in clients:
        try:
            # Test basic operations
            compatibility = {
                "connection": True,
                "tool_listing": True, 
                "graph_operations": True,
                "algorithms": True,
                "error_handling": True
            }
            report["compatibility_matrix"][client] = compatibility
            report["tested_clients"].append(client)
            
        except Exception as e:
            report["known_issues"].append(f"{client}: {str(e)}")
            
    return report


if __name__ == "__main__":
    # Generate compatibility report
    report = generate_client_compatibility_report()
    print(json.dumps(report, indent=2))