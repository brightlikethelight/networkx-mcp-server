#!/usr/bin/env python3
"""
Test performance with different graph sizes
"""

import asyncio
import json
import sys
import os
import time
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from networkx_mcp.server import NetworkXMCPServer


async def test_performance():
    """Test performance with different graph sizes."""
    print("=== Testing Performance ===")
    
    server = NetworkXMCPServer()
    
    # Test small graph performance
    print("Testing small graph (100 nodes, 200 edges)...")
    start_time = time.time()
    
    # Create graph
    await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 1,
        "params": {
            "name": "create_graph",
            "arguments": {"name": "small_graph", "directed": False}
        }
    })
    
    # Add nodes
    nodes = list(range(100))
    await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 2,
        "params": {
            "name": "add_nodes",
            "arguments": {"graph": "small_graph", "nodes": nodes}
        }
    })
    
    # Add edges (create a sparse random-like graph)
    edges = []
    for i in range(0, 100, 2):
        for j in range(i+1, min(i+5, 100)):
            edges.append([i, j])
    
    await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 3,
        "params": {
            "name": "add_edges",
            "arguments": {"graph": "small_graph", "edges": edges}
        }
    })
    
    # Test shortest path
    path_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 4,
        "params": {
            "name": "shortest_path",
            "arguments": {"graph": "small_graph", "source": 0, "target": 50}
        }
    })
    
    # Test centrality
    centrality_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 5,
        "params": {
            "name": "degree_centrality",
            "arguments": {"graph": "small_graph"}
        }
    })
    
    small_time = time.time() - start_time
    print(f"✓ Small graph operations completed in {small_time:.2f}s")
    
    # Test medium graph performance
    print("Testing medium graph (1000 nodes, 2000 edges)...")
    start_time = time.time()
    
    # Create medium graph
    await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 6,
        "params": {
            "name": "create_graph",
            "arguments": {"name": "medium_graph", "directed": False}
        }
    })
    
    # Add nodes
    nodes = list(range(1000))
    await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 7,
        "params": {
            "name": "add_nodes",
            "arguments": {"graph": "medium_graph", "nodes": nodes}
        }
    })
    
    # Add edges (sparser for performance)
    edges = []
    for i in range(0, 1000, 5):
        for j in range(i+1, min(i+10, 1000)):
            edges.append([i, j])
            if len(edges) >= 2000:  # Cap at 2000 edges
                break
        if len(edges) >= 2000:
            break
    
    await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 8,
        "params": {
            "name": "add_edges",
            "arguments": {"graph": "medium_graph", "edges": edges}
        }
    })
    
    # Test algorithms on medium graph
    centrality_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 9,
        "params": {
            "name": "degree_centrality",
            "arguments": {"graph": "medium_graph"}
        }
    })
    
    medium_time = time.time() - start_time
    print(f"✓ Medium graph operations completed in {medium_time:.2f}s")
    
    # Memory usage test
    print("Testing memory usage...")
    gc.collect()
    
    # Check that graphs exist
    info_response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 10,
        "params": {
            "name": "get_info",
            "arguments": {"graph": "medium_graph"}
        }
    })
    
    if "result" in info_response:
        content = json.loads(info_response["result"]["content"][0]["text"])
        print(f"✓ Medium graph has {content['nodes']} nodes, {content['edges']} edges")
    
    # Performance summary
    if medium_time > small_time * 100:  # If medium is much slower than expected
        print("⚠ Performance may degrade with larger graphs")
    else:
        print("✓ Performance scales reasonably with graph size")
    
    print("✓ Performance testing completed")


if __name__ == "__main__":
    asyncio.run(test_performance())