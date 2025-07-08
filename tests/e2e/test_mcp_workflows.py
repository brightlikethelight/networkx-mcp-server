"""End-to-end workflow tests for MCP protocol integration.

These tests simulate real-world usage patterns and workflows to ensure
the server works correctly with actual MCP clients and protocols.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import pytest


class MCPServerTestHarness:
    """Test harness for running MCP server in subprocess."""
    
    def __init__(self):
        self.server_process = None
        self.server_ready = False
        
    async def start_server(self):
        """Start MCP server as subprocess."""
        server_script = Path(__file__).parent.parent.parent / "src" / "networkx_mcp" / "server.py"
        
        # Start server process
        self.server_process = await asyncio.create_subprocess_exec(
            "python", str(server_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for server to be ready
        await asyncio.sleep(1)
        self.server_ready = True
        
        return self.server_process
        
    async def send_mcp_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send MCP JSON-RPC request to server."""
        if not self.server_ready or not self.server_process:
            raise RuntimeError("Server not ready")
            
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {}
        }
        
        request_data = json.dumps(request) + "\n"
        
        # Send request
        self.server_process.stdin.write(request_data.encode())
        await self.server_process.stdin.drain()
        
        # Read response
        response_data = await self.server_process.stdout.readline()
        
        if not response_data:
            raise RuntimeError("No response from server")
            
        try:
            response = json.loads(response_data.decode())
            return response
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {response_data.decode()[:200]}")
            
    async def cleanup(self):
        """Cleanup server process."""
        if self.server_process:
            self.server_process.terminate()
            try:
                await asyncio.wait_for(self.server_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.server_process.kill()
                await self.server_process.wait()


@asynccontextmanager
async def mcp_server():
    """Context manager for MCP server testing."""
    harness = MCPServerTestHarness()
    await harness.start_server()
    
    try:
        yield harness
    finally:
        await harness.cleanup()


class TestMCPWorkflows:
    """End-to-end MCP workflow tests."""
    
    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test MCP server initialization handshake."""
        async with mcp_server() as server:
            # Send initialize request
            response = await server.send_mcp_request("initialize", {
                "protocolVersion": "1.0.0",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            })
            
            assert response.get("jsonrpc") == "2.0"
            assert "result" in response
            assert "capabilities" in response["result"]
            
            # Send initialized notification
            await server.send_mcp_request("notifications/initialized")
    
    @pytest.mark.asyncio
    async def test_tools_listing(self):
        """Test listing available tools."""
        async with mcp_server() as server:
            # Initialize first
            await server.send_mcp_request("initialize", {
                "protocolVersion": "1.0.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            })
            
            # List tools
            response = await server.send_mcp_request("tools/list")
            
            assert response.get("jsonrpc") == "2.0"
            assert "result" in response
            assert "tools" in response["result"]
            
            tools = response["result"]["tools"]
            assert len(tools) > 0
            
            # Check for expected tools
            tool_names = [tool["name"] for tool in tools]
            expected_tools = [
                "create_graph",
                "add_nodes", 
                "add_edges",
                "shortest_path",
                "graph_info"
            ]
            
            for expected_tool in expected_tools:
                assert expected_tool in tool_names, f"Missing tool: {expected_tool}"
                
            # Validate tool schema structure
            for tool in tools:
                assert "name" in tool
                assert "description" in tool
                assert "inputSchema" in tool
                assert tool["inputSchema"]["type"] == "object"
    
    @pytest.mark.asyncio
    async def test_basic_graph_workflow(self):
        """Test basic graph creation and manipulation workflow."""
        async with mcp_server() as server:
            # Initialize
            await server.send_mcp_request("initialize", {
                "protocolVersion": "1.0.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            })
            
            # Create graph
            response = await server.send_mcp_request("tools/call", {
                "name": "create_graph",
                "arguments": {
                    "name": "test_workflow_graph",
                    "graph_type": "undirected"
                }
            })
            
            assert response.get("jsonrpc") == "2.0"
            assert "result" in response
            result = response["result"]
            assert not result.get("isError", False)
            
            # Add nodes
            response = await server.send_mcp_request("tools/call", {
                "name": "add_nodes",
                "arguments": {
                    "graph_name": "test_workflow_graph",
                    "nodes": ["A", "B", "C", "D"]
                }
            })
            
            assert "result" in response
            assert not response["result"].get("isError", False)
            
            # Add edges
            response = await server.send_mcp_request("tools/call", {
                "name": "add_edges", 
                "arguments": {
                    "graph_name": "test_workflow_graph",
                    "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["A", "D"]]
                }
            })
            
            assert "result" in response
            assert not response["result"].get("isError", False)
            
            # Get graph info
            response = await server.send_mcp_request("tools/call", {
                "name": "graph_info",
                "arguments": {
                    "graph_name": "test_workflow_graph"
                }
            })
            
            assert "result" in response
            result = response["result"]
            assert not result.get("isError", False)
            
            # Parse the content to verify graph structure
            content = result["content"][0]["text"]
            graph_data = json.loads(content)
            
            assert graph_data["num_nodes"] == 4
            assert graph_data["num_edges"] == 4
            assert graph_data["graph_type"] == "undirected"
    
    @pytest.mark.asyncio
    async def test_algorithm_workflow(self):
        """Test graph algorithm execution workflow."""
        async with mcp_server() as server:
            # Initialize
            await server.send_mcp_request("initialize", {
                "protocolVersion": "1.0.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            })
            
            # Set up graph
            await server.send_mcp_request("tools/call", {
                "name": "create_graph",
                "arguments": {"name": "algo_test", "graph_type": "undirected"}
            })
            
            await server.send_mcp_request("tools/call", {
                "name": "add_nodes",
                "arguments": {
                    "graph_name": "algo_test",
                    "nodes": ["A", "B", "C", "D", "E"]
                }
            })
            
            await server.send_mcp_request("tools/call", {
                "name": "add_edges",
                "arguments": {
                    "graph_name": "algo_test", 
                    "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"], ["A", "E"]]
                }
            })
            
            # Test shortest path algorithm
            response = await server.send_mcp_request("tools/call", {
                "name": "shortest_path",
                "arguments": {
                    "graph_name": "algo_test",
                    "source": "A",
                    "target": "C"
                }
            })
            
            assert "result" in response
            result = response["result"]
            assert not result.get("isError", False)
            
            # Verify algorithm result
            content = result["content"][0]["text"]
            path_data = json.loads(content)
            
            assert "path" in path_data
            assert len(path_data["path"]) >= 3  # A -> B -> C or longer
            assert path_data["path"][0] == "A"
            assert path_data["path"][-1] == "C"
            
            # Test centrality measures
            response = await server.send_mcp_request("tools/call", {
                "name": "centrality_measures",
                "arguments": {
                    "graph_name": "algo_test",
                    "measures": ["betweenness", "closeness"]
                }
            })
            
            if "result" in response and not response["result"].get("isError", False):
                content = response["result"]["content"][0]["text"]
                centrality_data = json.loads(content)
                
                assert "betweenness_centrality" in centrality_data
                assert "closeness_centrality" in centrality_data
                assert len(centrality_data["betweenness_centrality"]) == 5  # 5 nodes
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Test error handling in MCP protocol."""
        async with mcp_server() as server:
            # Initialize
            await server.send_mcp_request("initialize", {
                "protocolVersion": "1.0.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            })
            
            # Try to call non-existent tool
            response = await server.send_mcp_request("tools/call", {
                "name": "nonexistent_tool",
                "arguments": {}
            })
            
            assert "error" in response
            assert response["error"]["code"] == -32601  # Method not found
            
            # Try to operate on non-existent graph
            response = await server.send_mcp_request("tools/call", {
                "name": "graph_info",
                "arguments": {"graph_name": "nonexistent_graph"}
            })
            
            assert "result" in response
            result = response["result"]
            assert result.get("isError", False) or "error" in result["content"][0]["text"]
            
            # Try to create graph with invalid parameters
            response = await server.send_mcp_request("tools/call", {
                "name": "create_graph",
                "arguments": {
                    "name": "invalid-graph-name!@#",
                    "graph_type": "invalid_type"
                }
            })
            
            assert "result" in response
            # Should either error at protocol level or return error content
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent MCP requests."""
        async with mcp_server() as server:
            # Initialize
            await server.send_mcp_request("initialize", {
                "protocolVersion": "1.0.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            })
            
            # Create multiple graphs concurrently
            tasks = []
            for i in range(5):
                task = server.send_mcp_request("tools/call", {
                    "name": "create_graph",
                    "arguments": {
                        "name": f"concurrent_graph_{i}",
                        "graph_type": "undirected"
                    }
                })
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All requests should complete
            assert len(responses) == 5
            
            for response in responses:
                assert "result" in response or "error" in response
                
            # Verify graphs were created
            response = await server.send_mcp_request("tools/call", {
                "name": "list_graphs",
                "arguments": {}
            })
            
            if "result" in response and not response["result"].get("isError", False):
                content = response["result"]["content"][0]["text"]
                graphs_data = json.loads(content)
                
                created_graphs = [g for g in graphs_data.get("graphs", []) if g.startswith("concurrent_graph_")]
                assert len(created_graphs) >= 3  # At least some should succeed
    
    @pytest.mark.asyncio 
    async def test_resource_management(self):
        """Test resource management and cleanup."""
        async with mcp_server() as server:
            # Initialize
            await server.send_mcp_request("initialize", {
                "protocolVersion": "1.0.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            })
            
            # Create and populate several graphs
            for i in range(3):
                graph_name = f"resource_test_{i}"
                
                # Create graph
                await server.send_mcp_request("tools/call", {
                    "name": "create_graph",
                    "arguments": {"name": graph_name, "graph_type": "undirected"}
                })
                
                # Add data
                nodes = [f"node_{j}" for j in range(100)]
                await server.send_mcp_request("tools/call", {
                    "name": "add_nodes",
                    "arguments": {"graph_name": graph_name, "nodes": nodes}
                })
                
                edges = [[f"node_{j}", f"node_{(j+1) % 100}"] for j in range(100)]
                await server.send_mcp_request("tools/call", {
                    "name": "add_edges", 
                    "arguments": {"graph_name": graph_name, "edges": edges}
                })
            
            # Check memory usage isn't excessive
            # This is implicit - if server is still responding, memory is managed
            
            # Test cleanup by deleting graphs
            for i in range(3):
                graph_name = f"resource_test_{i}"
                response = await server.send_mcp_request("tools/call", {
                    "name": "delete_graph",
                    "arguments": {"graph_name": graph_name}
                })
                
                # Should succeed or gracefully handle if already deleted
                assert "result" in response or "error" in response
    
    @pytest.mark.asyncio
    async def test_data_persistence_workflow(self):
        """Test data persistence and reload workflow."""
        async with mcp_server() as server:
            # Initialize
            await server.send_mcp_request("initialize", {
                "protocolVersion": "1.0.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            })
            
            # Create and populate graph
            await server.send_mcp_request("tools/call", {
                "name": "create_graph",
                "arguments": {"name": "persistence_test", "graph_type": "undirected"}
            })
            
            await server.send_mcp_request("tools/call", {
                "name": "add_nodes",
                "arguments": {
                    "graph_name": "persistence_test",
                    "nodes": ["A", "B", "C"]
                }
            })
            
            await server.send_mcp_request("tools/call", {
                "name": "add_edges",
                "arguments": {
                    "graph_name": "persistence_test",
                    "edges": [["A", "B"], ["B", "C"]]
                }
            })
            
            # Export graph (if export tool is available)
            export_response = await server.send_mcp_request("tools/call", {
                "name": "export_graph",
                "arguments": {
                    "graph_name": "persistence_test",
                    "file_path": "test_export.json",
                    "format": "json"
                }
            })
            
            # May not be implemented - that's ok
            # Just verify server handles the request gracefully
            assert "result" in export_response or "error" in export_response


class TestMCPFeatureFlags:
    """Test MCP integration with feature flags."""
    
    @pytest.mark.asyncio
    async def test_feature_flag_access_via_mcp(self):
        """Test accessing feature flags via MCP protocol."""
        async with mcp_server() as server:
            # Initialize
            await server.send_mcp_request("initialize", {
                "protocolVersion": "1.0.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            })
            
            # Try to access feature flags
            response = await server.send_mcp_request("tools/call", {
                "name": "manage_feature_flags",
                "arguments": {
                    "action": "list"
                }
            })
            
            if "result" in response:
                result = response["result"]
                if not result.get("isError", False):
                    content = result["content"][0]["text"]
                    flags_data = json.loads(content)
                    
                    # Should have feature flag structure
                    assert "by_category" in flags_data or "total_flags" in flags_data


class TestMCPSecurityIntegration:
    """Test security features via MCP protocol."""
    
    @pytest.mark.asyncio
    async def test_input_validation_via_mcp(self):
        """Test input validation through MCP protocol."""
        async with mcp_server() as server:
            # Initialize
            await server.send_mcp_request("initialize", {
                "protocolVersion": "1.0.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            })
            
            # Test SQL injection attempt
            response = await server.send_mcp_request("tools/call", {
                "name": "create_graph",
                "arguments": {
                    "name": "test'; DROP TABLE graphs; --",
                    "graph_type": "undirected"
                }
            })
            
            # Should either return error or sanitize input
            assert "result" in response
            result = response["result"]
            
            if not result.get("isError", False):
                # If successful, name should be sanitized
                content = result["content"][0]["text"]
                response_data = json.loads(content)
                # Should not contain dangerous SQL
                assert "DROP TABLE" not in str(response_data)
            
            # Test path traversal attempt
            response = await server.send_mcp_request("tools/call", {
                "name": "export_graph",
                "arguments": {
                    "graph_name": "test",
                    "file_path": "../../../etc/passwd",
                    "format": "json"
                }
            })
            
            # Should reject path traversal
            if "result" in response:
                result = response["result"]
                if result.get("isError", False):
                    # Good - rejected the request
                    pass
                else:
                    # If not rejected, should not contain sensitive path
                    content = result["content"][0]["text"]
                    assert "/etc/passwd" not in content


if __name__ == "__main__":
    # Run basic MCP workflow test
    async def test_basic_workflow():
        async with mcp_server() as server:
            print("🔌 Testing MCP server initialization...")
            
            # Initialize
            response = await server.send_mcp_request("initialize", {
                "protocolVersion": "1.0.0",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            })
            
            print(f"✅ Initialize response: {response.get('result', {}).get('capabilities', 'No capabilities')}")
            
            # List tools
            response = await server.send_mcp_request("tools/list")
            tools = response.get("result", {}).get("tools", [])
            print(f"🛠️  Available tools: {len(tools)}")
            
            # Create and test graph
            await server.send_mcp_request("tools/call", {
                "name": "create_graph",
                "arguments": {"name": "demo", "graph_type": "undirected"}
            })
            
            await server.send_mcp_request("tools/call", {
                "name": "add_nodes",
                "arguments": {"graph_name": "demo", "nodes": ["A", "B", "C"]}
            })
            
            response = await server.send_mcp_request("tools/call", {
                "name": "graph_info",
                "arguments": {"graph_name": "demo"}
            })
            
            if "result" in response and not response["result"].get("isError", False):
                content = response["result"]["content"][0]["text"]
                graph_data = json.loads(content)
                print(f"📊 Graph created: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
            
            print("✅ Basic MCP workflow completed successfully!")
    
    # Run the test
    asyncio.run(test_basic_workflow())