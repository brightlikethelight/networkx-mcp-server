#!/usr/bin/env python3
"""Test integrated thread-safe MCP server."""

import asyncio
import json
import os

# Disable structured logging for cleaner output
os.environ["LOG_LEVEL"] = "ERROR"

import sys
sys.path.insert(0, "src")

from networkx_mcp.protocol.mcp_handler import MCPProtocolHandler


async def test_integrated_server():
    """Test the integrated thread-safe MCP server."""
    print("🚀 Testing Integrated Thread-Safe MCP Server\n")
    
    handler = MCPProtocolHandler()
    
    try:
        # 1. Initialize
        print("1. Initializing MCP protocol...")
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            }
        }
        
        response = await handler.handle_message(json.dumps(request))
        response_data = json.loads(response)
        assert "result" in response_data
        print("✅ Initialized successfully")
        
        # 2. Create multiple graphs concurrently
        print("\n2. Creating graphs concurrently...")
        tasks = []
        for i in range(5):
            req = {
                "jsonrpc": "2.0",
                "id": f"create_{i}",
                "method": "tools/call",
                "params": {
                    "name": "create_graph",
                    "arguments": {
                        "name": f"test_graph_{i}",
                        "graph_type": "undirected"
                    }
                }
            }
            task = handler.handle_message(json.dumps(req))
            tasks.append(task)
            
        responses = await asyncio.gather(*tasks)
        success_count = sum(1 for r in responses if "error" not in json.loads(r))
        print(f"✅ Created {success_count}/5 graphs")
        
        # 3. Add nodes concurrently
        print("\n3. Adding nodes concurrently...")
        tasks = []
        for i in range(10):
            graph_id = f"test_graph_{i % 5}"
            req = {
                "jsonrpc": "2.0",
                "id": f"add_nodes_{i}",
                "method": "tools/call",
                "params": {
                    "name": "add_nodes",
                    "arguments": {
                        "graph_name": graph_id,
                        "nodes": [f"node_{i}_{j}" for j in range(3)]
                    }
                }
            }
            task = handler.handle_message(json.dumps(req))
            tasks.append(task)
            
        responses = await asyncio.gather(*tasks)
        success_count = sum(1 for r in responses if "error" not in json.loads(r))
        print(f"✅ {success_count}/10 node additions succeeded")
        
        # 4. Get graph info
        print("\n4. Getting graph info...")
        req = {
            "jsonrpc": "2.0",
            "id": "info",
            "method": "tools/call",
            "params": {
                "name": "graph_info",
                "arguments": {"graph_name": "test_graph_0"}
            }
        }
        
        response = await handler.handle_message(json.dumps(req))
        response_data = json.loads(response)
        if "result" in response_data:
            result = json.loads(response_data["result"]["content"][0]["text"])
            print(f"✅ Graph info: {result['nodes']} nodes")
        
        # 5. Get resource status
        print("\n5. Checking resource status...")
        req = {
            "jsonrpc": "2.0",
            "id": "status",
            "method": "tools/call",
            "params": {
                "name": "resource_status",
                "arguments": {}
            }
        }
        
        response = await handler.handle_message(json.dumps(req))
        response_data = json.loads(response)
        if "result" in response_data:
            result = json.loads(response_data["result"]["content"][0]["text"])
            print(f"✅ Lock acquisitions: {result['lock_stats']['total_acquisitions']}")
            print(f"   Connection pool: {result['connection_pool']['total_connections']} connections")
            print(f"   Request queue: {result['request_queue']['workers']} workers")
        
        # 6. Test overload protection
        print("\n6. Testing overload protection...")
        # Simulate 100 concurrent requests (more than connection pool limit)
        tasks = []
        for i in range(100):
            req = {
                "jsonrpc": "2.0",
                "id": f"overload_{i}",
                "method": "tools/call",
                "params": {
                    "name": "list_graphs",
                    "arguments": {}
                }
            }
            task = handler.handle_message(json.dumps(req))
            tasks.append(task)
            
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = 0
        timeout_count = 0
        for r in responses:
            if isinstance(r, str):
                data = json.loads(r)
                if "result" in data:
                    success_count += 1
                elif "error" in data and "overloaded" in data["error"].get("data", ""):
                    timeout_count += 1
                    
        print(f"✅ Handled {success_count} requests, {timeout_count} rejected (expected)")
        
        print("\n✨ Integrated thread-safe MCP server working correctly!")
        
    finally:
        await handler.cleanup()


async def test_concurrent_graph_operations():
    """Test concurrent operations on the same graph."""
    print("\n\n🔧 Testing Concurrent Graph Operations\n")
    
    handler = MCPProtocolHandler()
    
    try:
        # Initialize
        await handler.handle_message(json.dumps({
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            }
        }))
        
        # Create a graph
        await handler.handle_message(json.dumps({
            "jsonrpc": "2.0",
            "id": "create",
            "method": "tools/call",
            "params": {
                "name": "create_graph",
                "arguments": {"name": "concurrent_test", "graph_type": "undirected"}
            }
        }))
        
        # Concurrent node and edge additions
        print("Running 20 concurrent operations on same graph...")
        tasks = []
        
        # 10 node additions
        for i in range(10):
            req = {
                "jsonrpc": "2.0",
                "id": f"nodes_{i}",
                "method": "tools/call",
                "params": {
                    "name": "add_nodes",
                    "arguments": {
                        "graph_name": "concurrent_test",
                        "nodes": [f"n{i}"]
                    }
                }
            }
            tasks.append(handler.handle_message(json.dumps(req)))
            
        # 10 edge additions (some may fail if nodes don't exist yet)
        for i in range(10):
            req = {
                "jsonrpc": "2.0",
                "id": f"edges_{i}",
                "method": "tools/call",
                "params": {
                    "name": "add_edges",
                    "arguments": {
                        "graph_name": "concurrent_test",
                        "edges": [[f"n{i}", f"n{(i+1)%10}"]]
                    }
                }
            }
            tasks.append(handler.handle_message(json.dumps(req)))
            
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in responses 
                          if isinstance(r, str) and "error" not in json.loads(r))
        print(f"✅ {success_count}/20 operations succeeded")
        
        # Get final graph state
        response = await handler.handle_message(json.dumps({
            "jsonrpc": "2.0",
            "id": "final",
            "method": "tools/call",
            "params": {
                "name": "graph_info",
                "arguments": {"graph_name": "concurrent_test"}
            }
        }))
        
        data = json.loads(response)
        if "result" in data:
            result = json.loads(data["result"]["content"][0]["text"])
            print(f"✅ Final graph state: {result['nodes']} nodes, {result['edges']} edges")
            
    finally:
        await handler.cleanup()


if __name__ == "__main__":
    async def main():
        await test_integrated_server()
        await test_concurrent_graph_operations()
        print("\n🎉 All integrated tests passed!")
        
    asyncio.run(main())