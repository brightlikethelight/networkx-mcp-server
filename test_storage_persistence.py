#!/usr/bin/env python3
"""Test storage persistence across server restarts."""

import asyncio
import sys
import os
sys.path.insert(0, '.')

from src.networkx_mcp.server_with_storage import (
    initialize_storage, shutdown_storage, create_graph, add_nodes, add_edges,
    graph_info, list_graphs, storage_status, graph_manager, storage_manager
)


async def test_storage_backend():
    """Test storage backend functionality."""
    print("=== Testing Storage Backend ===\n")
    
    # Initialize storage
    print("1. Initializing storage...")
    await initialize_storage()
    
    # Check storage status
    print("\n2. Checking storage status...")
    status = await storage_status()
    print(f"Storage status: {status}")
    print(f"Backend: {status.get('backend', 'unknown')}")
    print(f"Persistent: {status.get('persistent', False)}")
    
    # List existing graphs
    print("\n3. Listing existing graphs...")
    graphs = list_graphs()
    print(f"Found {graphs['count']} existing graphs:")
    for g in graphs['graphs']:
        print(f"  - {g['name']}: {g['nodes']} nodes, {g['edges']} edges")
    
    # Create test graphs
    print("\n4. Creating test graphs...")
    
    # Graph 1: Simple network
    result = await create_graph("persistence_test_1", "undirected")
    print(f"Created graph 1: {result}")
    
    await add_nodes("persistence_test_1", ["A", "B", "C", "D", "E"])
    await add_edges("persistence_test_1", [
        ["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"], ["E", "A"]
    ])
    
    info = graph_info("persistence_test_1")
    print(f"Graph 1 info: {info['nodes']} nodes, {info['edges']} edges")
    
    # Graph 2: Directed network
    result = await create_graph("persistence_test_2", "directed", {
        "nodes": [1, 2, 3, 4],
        "edges": [[1, 2], [2, 3], [3, 4], [4, 1], [2, 4]]
    })
    print(f"Created graph 2: {result}")
    
    # Graph 3: Weighted network
    result = await create_graph("persistence_test_weighted", "undirected")
    await add_nodes("persistence_test_weighted", ["X", "Y", "Z"])
    await add_edges("persistence_test_weighted", [
        ["X", "Y", {"weight": 1.5}],
        ["Y", "Z", {"weight": 2.0}],
        ["Z", "X", {"weight": 0.5}]
    ])
    print(f"Created weighted graph: {result}")
    
    # Force sync to storage
    print("\n5. Syncing to storage...")
    await storage_manager.sync_all_graphs()
    
    # List graphs again
    print("\n6. Current graphs:")
    graphs = list_graphs()
    for g in graphs['graphs']:
        print(f"  - {g['name']}: {g['nodes']} nodes, {g['edges']} edges")
    
    # Show storage stats
    stats = await storage_status()
    print(f"\nStorage stats: {stats.get('total_graphs', 0)} graphs, "
          f"{stats.get('estimated_mb', 0):.2f} MB used")
    
    print("\n7. Shutting down storage...")
    await shutdown_storage()
    
    print("\n✅ Storage test complete!")
    print("\nTo test persistence:")
    print("1. Note the graphs created above")
    print("2. Run this script again")
    print("3. Check if the graphs are loaded from storage")
    
    return True


async def verify_persistence():
    """Verify that graphs persist across restarts."""
    print("\n=== Verifying Storage Persistence ===\n")
    
    # Initialize storage
    print("1. Initializing storage (simulating server restart)...")
    await initialize_storage()
    
    # Check what was loaded
    print("\n2. Checking loaded graphs...")
    graphs = list_graphs()
    print(f"Found {graphs['count']} graphs after restart:")
    
    persistence_verified = False
    for g in graphs['graphs']:
        print(f"  - {g['name']}: {g['nodes']} nodes, {g['edges']} edges")
        if g['name'].startswith('persistence_test'):
            persistence_verified = True
    
    if persistence_verified:
        print("\n✅ PERSISTENCE VERIFIED! Graphs were loaded from storage.")
        
        # Verify graph contents
        print("\n3. Verifying graph contents...")
        
        # Check graph 1
        if "persistence_test_1" in graph_manager.graphs:
            info = graph_info("persistence_test_1")
            print(f"  - persistence_test_1: {info['nodes']} nodes, {info['edges']} edges")
            assert info['nodes'] == 5, "Wrong node count"
            assert info['edges'] == 5, "Wrong edge count"
            print("    ✓ Content verified")
        
        # Check weighted graph
        if "persistence_test_weighted" in graph_manager.graphs:
            graph = graph_manager.get_graph("persistence_test_weighted")
            edge_data = graph.get_edge_data("X", "Y")
            if edge_data and 'weight' in edge_data:
                print(f"  - Weight X->Y: {edge_data['weight']} ✓")
            else:
                print("  - Warning: Weight data not preserved")
                
    else:
        print("\n⚠️  No persistence test graphs found.")
        print("This is expected on first run or if using in-memory storage.")
    
    # Check storage backend
    status = await storage_status()
    print(f"\n4. Storage backend: {status.get('backend', 'unknown')}")
    if status.get('persistent', True):
        print("   ✓ Using persistent storage")
    else:
        print("   ⚠️  Using non-persistent storage (in-memory)")
    
    await shutdown_storage()
    
    return persistence_verified


async def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test storage persistence")
    parser.add_argument("--verify", action="store_true", 
                       help="Verify persistence instead of creating test data")
    parser.add_argument("--backend", choices=["redis", "memory"], 
                       help="Force specific backend")
    args = parser.parse_args()
    
    # Set backend if specified
    if args.backend:
        os.environ["STORAGE_BACKEND"] = args.backend
        print(f"Forcing {args.backend} backend\n")
    
    try:
        if args.verify:
            verified = await verify_persistence()
            if not verified:
                print("\nRun without --verify to create test graphs first.")
        else:
            await test_storage_backend()
            print("\nNow run with --verify flag to test persistence:")
            print(f"  python {sys.argv[0]} --verify")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())