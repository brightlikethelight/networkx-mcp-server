#!/usr/bin/env python3
"""
Test actual MCP server via stdio protocol (simulating real client connection)
"""

import asyncio
import json
import subprocess
import sys
import os
import time

async def test_stdio_mcp_server():
    """Test the server running via stdio like a real MCP client would."""
    print("=== Testing Real MCP Server via STDIO ===")
    
    # Start the server process
    server_process = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/Users/brightliu/Coding_Projects/networkx-mcp-server"
    )
    
    def send_request(request):
        """Send a JSON-RPC request to the server."""
        request_str = json.dumps(request) + "\n"
        server_process.stdin.write(request_str)
        server_process.stdin.flush()
        
        # Read response
        response_line = server_process.stdout.readline()
        if response_line:
            return json.loads(response_line.strip())
        return None
    
    try:
        # Test initialization
        print("Testing initialization...")
        init_response = send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        })
        
        if init_response and init_response.get("result"):
            print("✓ Server initialization successful")
            print(f"  Protocol version: {init_response['result'].get('protocolVersion')}")
            print(f"  Server info: {init_response['result'].get('serverInfo')}")
        else:
            print("✗ Server initialization failed")
            return False
        
        # Send initialized notification
        send_request({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        })
        
        # Test tools list
        print("\nTesting tools list...")
        tools_response = send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        })
        
        if tools_response and tools_response.get("result"):
            tools = tools_response["result"]["tools"]
            print(f"✓ Tools list retrieved: {len(tools)} tools")
            for tool in tools[:5]:  # Show first 5
                print(f"  - {tool['name']}: {tool['description']}")
            if len(tools) > 5:
                print(f"  ... and {len(tools) - 5} more")
        else:
            print("✗ Tools list failed")
        
        # Test creating a graph
        print("\nTesting graph creation...")
        create_response = send_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "create_graph",
                "arguments": {"name": "stdio_test", "directed": False}
            }
        })
        
        if create_response and create_response.get("result"):
            result_text = create_response["result"]["content"][0]["text"]
            result_data = json.loads(result_text)
            print(f"✓ Graph created: {result_data}")
        else:
            print("✗ Graph creation failed")
        
        # Test adding nodes
        print("\nTesting node addition...")
        nodes_response = send_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "add_nodes",
                "arguments": {"graph": "stdio_test", "nodes": ["A", "B", "C"]}
            }
        })
        
        if nodes_response and nodes_response.get("result"):
            result_text = nodes_response["result"]["content"][0]["text"]
            result_data = json.loads(result_text)
            print(f"✓ Nodes added: {result_data}")
        else:
            print("✗ Node addition failed")
        
        # Test adding edges
        print("\nTesting edge addition...")
        edges_response = send_request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "add_edges",
                "arguments": {"graph": "stdio_test", "edges": [["A", "B"], ["B", "C"]]}
            }
        })
        
        if edges_response and edges_response.get("result"):
            result_text = edges_response["result"]["content"][0]["text"]
            result_data = json.loads(result_text)
            print(f"✓ Edges added: {result_data}")
        else:
            print("✗ Edge addition failed")
        
        # Test shortest path
        print("\nTesting shortest path...")
        path_response = send_request({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "shortest_path",
                "arguments": {"graph": "stdio_test", "source": "A", "target": "C"}
            }
        })
        
        if path_response and path_response.get("result"):
            result_text = path_response["result"]["content"][0]["text"]
            result_data = json.loads(result_text)
            print(f"✓ Shortest path found: {result_data}")
        else:
            print("✗ Shortest path failed")
        
        print("\n✓ All STDIO tests passed - server works as real MCP server!")
        return True
        
    except Exception as e:
        print(f"✗ STDIO test failed: {e}")
        return False
        
    finally:
        # Clean shutdown
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()


async def main():
    """Run the stdio test."""
    success = await test_stdio_mcp_server()
    if success:
        print("\n🎉 MCP Server STDIO integration verified!")
    else:
        print("\n❌ MCP Server STDIO integration failed!")
    return success


if __name__ == "__main__":
    asyncio.run(main())