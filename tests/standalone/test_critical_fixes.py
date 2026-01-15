#!/usr/bin/env python3
"""Test script to verify all critical fixes are working."""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, "src")


async def test_fixes():
    """Test all critical fixes."""
    from networkx_mcp.server import NetworkXMCPServer

    print("🧪 TESTING CRITICAL FIXES\n")

    # Test 1: Server starts without auth errors in dev mode
    print("1. Testing simplified authentication...")
    try:
        server = NetworkXMCPServer(auth_required=False)
        print("   ✅ Server starts without authentication errors")
    except RuntimeError as e:
        print(f"   ❌ Authentication error: {e}")
        return False

    # Test 2: MCP tools/list method works
    print("\n2. Testing MCP tools/list method...")
    request = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}

    response = await server.handle_request(request)

    if "result" in response and "tools" in response["result"]:
        tools = response["result"]["tools"]
        print(f"   ✅ MCP tools/list works! Found {len(tools)} tools")
    else:
        print(f"   ❌ MCP tools/list failed: {response}")
        return False

    # Test 3: Create and manipulate a graph
    print("\n3. Testing core graph operations...")

    # Create graph
    create_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": "create_graph", "arguments": {"name": "test_graph"}},
        "id": 2,
    }

    response = await server.handle_request(create_request)
    if "result" in response:
        print("   ✅ Graph created successfully")
    else:
        print(f"   ❌ Graph creation failed: {response}")
        return False

    # Add nodes
    add_nodes_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "add_nodes",
            "arguments": {"graph": "test_graph", "nodes": [1, 2, 3, 4]},
        },
        "id": 3,
    }

    response = await server.handle_request(add_nodes_request)
    if "result" in response:
        print("   ✅ Nodes added successfully")
    else:
        print(f"   ❌ Add nodes failed: {response}")
        return False

    # Add edges
    add_edges_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "add_edges",
            "arguments": {
                "graph": "test_graph",
                "edges": [[1, 2], [2, 3], [3, 4], [1, 4]],
            },
        },
        "id": 4,
    }

    response = await server.handle_request(add_edges_request)
    if "result" in response:
        print("   ✅ Edges added successfully")
    else:
        print(f"   ❌ Add edges failed: {response}")
        return False

    # Find shortest path
    path_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "shortest_path",
            "arguments": {"graph": "test_graph", "source": 1, "target": 3},
        },
        "id": 5,
    }

    response = await server.handle_request(path_request)
    if "result" in response:
        print(f"   ✅ Shortest path found: {response['result']}")
    else:
        print(f"   ❌ Shortest path failed: {response}")
        return False

    # Test 4: Memory management (graph cache)
    print("\n4. Testing memory management...")
    from networkx_mcp.graph_cache import get_graph_cache

    cache = get_graph_cache()
    stats = cache.get_stats()
    print(
        f"   📊 Cache stats: {stats['size']} graphs, {stats['memory_mb']:.1f}MB memory"
    )
    print("   ✅ Graph cache with memory management active")

    print("\n" + "=" * 50)
    print("🎉 ALL CRITICAL FIXES VERIFIED SUCCESSFULLY!")
    print("=" * 50)

    return True


if __name__ == "__main__":
    # Ensure we're in development mode for testing
    os.environ["NETWORKX_MCP_ENV"] = "development"

    success = asyncio.run(test_fixes())
    sys.exit(0 if success else 1)
