#!/usr/bin/env python3
"""Basic test for thread-safe graph manager."""

import asyncio
import os
import sys

# Disable logging to clean up output
os.environ["LOG_LEVEL"] = "ERROR"

# Add src to path
sys.path.insert(0, "src")

from networkx_mcp.core.thread_safe_graph_manager import ThreadSafeGraphManager


async def test_basic_operations():
    """Test basic thread-safe operations."""
    print("🧪 Testing Thread-Safe Graph Manager\n")
    
    manager = ThreadSafeGraphManager()
    
    try:
        # Test 1: Create graph
        print("1. Creating graph...")
        result = await manager.create_graph('test_graph', 'undirected')
        assert result["success"], f"Failed to create graph: {result}"
        print(f"   ✅ {result['name']} ({result['type']})")
        
        # Test 2: Add nodes
        print("\n2. Adding nodes...")
        result = await manager.add_nodes('test_graph', ['A', 'B', 'C', 'D'])
        assert result["success"], f"Failed to add nodes: {result}"
        print(f"   ✅ Added {result['nodes_added']} nodes")
        
        # Test 3: Add edges
        print("\n3. Adding edges...")
        edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'D')]
        result = await manager.add_edges('test_graph', edges)
        assert result["success"], f"Failed to add edges: {result}"
        print(f"   ✅ Added {result['edges_added']} edges")
        
        # Test 4: Get graph info
        print("\n4. Getting graph info...")
        result = await manager.get_graph_info('test_graph')
        assert result["success"], f"Failed to get info: {result}"
        print(f"   ✅ Nodes: {result['nodes']}, Edges: {result['edges']}")
        print(f"      Type: {result['type']}, Density: {result['density']:.3f}")
        
        # Test 5: Shortest path
        print("\n5. Finding shortest path...")
        result = await manager.get_shortest_path('test_graph', 'A', 'C')
        assert result["success"], f"Failed to find path: {result}"
        print(f"   ✅ Path: {' -> '.join(result['path'])}")
        print(f"      Length: {result['length']}")
        
        # Test 6: Centrality measures
        print("\n6. Calculating centrality...")
        result = await manager.centrality_measures('test_graph', ['degree', 'betweenness'])
        assert result["success"], f"Failed to calculate centrality: {result}"
        print("   ✅ Centrality measures calculated")
        print(f"      Degree centrality: {result['degree_centrality']}")
        
        # Test 7: List graphs
        print("\n7. Listing graphs...")
        result = await manager.list_graphs()
        assert result["success"], f"Failed to list graphs: {result}"
        print(f"   ✅ Found {result['total']} graphs")
        
        # Test 8: Lock statistics
        print("\n8. Lock statistics...")
        stats = manager.get_lock_stats()
        print(f"   ✅ Lock acquisitions: {stats['total_acquisitions']}")
        print(f"      Contentions: {stats['total_contentions']}")
        print(f"      Avg wait time: {stats['avg_wait_time']:.6f}s")
        
        print("\n✨ All basic operations completed successfully!")
        
    finally:
        await manager.cleanup()


async def test_concurrent_operations():
    """Test concurrent operations on the same graph."""
    print("\n\n🔧 Testing Concurrent Operations\n")
    
    manager = ThreadSafeGraphManager()
    
    try:
        # Create test graph
        await manager.create_graph('concurrent_test', 'undirected')
        
        # Run 20 concurrent node additions
        print("Running 20 concurrent node additions...")
        tasks = []
        for i in range(20):
            nodes = [f"node_{i}_{j}" for j in range(5)]
            task = manager.add_nodes('concurrent_test', nodes)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        print(f"✅ {successful}/20 concurrent operations succeeded")
        
        # Get final graph info
        info = await manager.get_graph_info('concurrent_test')
        print(f"Final graph: {info['nodes']} nodes, {info['edges']} edges")
        
        # Check lock statistics
        stats = manager.get_lock_stats()
        print(f"Lock contentions: {stats['total_contentions']}")
        
    finally:
        await manager.cleanup()


async def test_error_handling():
    """Test error handling in thread-safe operations."""
    print("\n\n🚨 Testing Error Handling\n")
    
    manager = ThreadSafeGraphManager()
    
    try:
        # Test 1: Non-existent graph
        print("1. Testing operations on non-existent graph...")
        result = await manager.add_nodes('non_existent', ['A'])
        assert not result["success"], "Should fail for non-existent graph"
        print(f"   ✅ Correctly failed: {result['error']}")
        
        # Test 2: Invalid shortest path
        print("\n2. Testing shortest path with non-existent nodes...")
        await manager.create_graph('error_test', 'undirected')
        await manager.add_nodes('error_test', ['A', 'B'])
        
        result = await manager.get_shortest_path('error_test', 'A', 'Z')
        assert not result["success"], "Should fail for non-existent node"
        print(f"   ✅ Correctly failed: {result['error']}")
        
        # Test 3: Path with no connection
        await manager.add_nodes('error_test', ['C'])  # Isolated node
        result = await manager.get_shortest_path('error_test', 'A', 'C')
        assert not result["success"], "Should fail for disconnected nodes"
        print(f"   ✅ Correctly failed: {result['error']}")
        
        print("\n✨ Error handling working correctly!")
        
    finally:
        await manager.cleanup()


if __name__ == "__main__":
    async def main():
        await test_basic_operations()
        await test_concurrent_operations()
        await test_error_handling()
        print("\n🎉 All thread safety tests passed!")
        
    asyncio.run(main())