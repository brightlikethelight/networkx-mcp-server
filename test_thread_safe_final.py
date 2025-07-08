#!/usr/bin/env python3
"""Final test for thread-safe integration."""

import asyncio
import json
import os
import sys

# Disable structured logging
os.environ["LOG_LEVEL"] = "ERROR"

sys.path.insert(0, "src")

from networkx_mcp.protocol.mcp_handler import MCPProtocolHandler


async def test_thread_safe_integration():
    """Test thread-safe MCP integration."""
    print("🧪 Testing Thread-Safe MCP Integration\n")
    
    handler = MCPProtocolHandler()
    
    # 1. Initialize
    print("1. Initializing...")
    response = await handler.handle_message(json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    }))
    
    result = json.loads(response)
    if "result" in result:
        print(f"✅ Initialized: {result['result']['serverInfo']['name']}")
    else:
        print(f"❌ Init failed: {result}")
        return
    
    # 2. Create graph
    print("\n2. Creating graph...")
    response = await handler.handle_message(json.dumps({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "create_graph",
            "arguments": {"name": "test_graph", "graph_type": "undirected"}
        }
    }))
    
    result = json.loads(response)
    if "result" in result:
        content = json.loads(result["result"]["content"][0]["text"])
        print(f"✅ Created graph: {content['name']}")
    else:
        print(f"❌ Create failed: {result}")
        return
    
    # 3. Add nodes
    print("\n3. Adding nodes...")
    response = await handler.handle_message(json.dumps({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "add_nodes",
            "arguments": {"graph_name": "test_graph", "nodes": ["A", "B", "C", "D"]}
        }
    }))
    
    result = json.loads(response)
    if "result" in result:
        content = json.loads(result["result"]["content"][0]["text"])
        print(f"✅ Added {content['nodes_added']} nodes")
    else:
        print(f"❌ Add nodes failed: {result}")
    
    # 4. Add edges
    print("\n4. Adding edges...")
    response = await handler.handle_message(json.dumps({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "add_edges",
            "arguments": {
                "graph_name": "test_graph",
                "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]]
            }
        }
    }))
    
    result = json.loads(response)
    if "result" in result:
        content = json.loads(result["result"]["content"][0]["text"])
        print(f"✅ Added {content['edges_added']} edges")
    else:
        print(f"❌ Add edges failed: {result}")
    
    # 5. Test concurrent operations
    print("\n5. Testing concurrent operations...")
    tasks = []
    for i in range(10):
        req = {
            "jsonrpc": "2.0",
            "id": f"concurrent_{i}",
            "method": "tools/call",
            "params": {
                "name": "graph_info",
                "arguments": {"graph_name": "test_graph"}
            }
        }
        tasks.append(handler.handle_message(json.dumps(req)))
    
    responses = await asyncio.gather(*tasks)
    success_count = sum(1 for r in responses if "error" not in json.loads(r))
    print(f"✅ {success_count}/10 concurrent reads succeeded")
    
    # 6. Get resource status
    print("\n6. Getting resource status...")
    response = await handler.handle_message(json.dumps({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "resource_status",
            "arguments": {}
        }
    }))
    
    result = json.loads(response)
    if "result" in result:
        content = json.loads(result["result"]["content"][0]["text"])
        stats = content["lock_stats"]
        pool = content["connection_pool"]
        print(f"✅ Resource Status:")
        print(f"   Lock acquisitions: {stats['total_acquisitions']}")
        print(f"   Lock contention rate: {stats['contention_rate']*100:.1f}%")
        print(f"   Connection pool usage: {pool['total_connections']} connections")
    
    # Cleanup
    await handler.cleanup()
    
    print("\n✨ Thread-safe integration test complete!")


if __name__ == "__main__":
    asyncio.run(test_thread_safe_integration())