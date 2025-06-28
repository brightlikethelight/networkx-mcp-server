#!/usr/bin/env python3
"""Debug the Redis persistence issue."""

import sys
import time
from pathlib import Path
import redis
import pickle

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_redis_persistence():
    """Debug what's actually happening with Redis persistence."""
    print("🔍 DEBUGGING REDIS PERSISTENCE ISSUE")
    print("=" * 50)
    
    # Connect to Redis
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
    
    # Check what's in Redis
    all_keys = redis_client.keys("*")
    print(f"📊 Total Redis keys: {len(all_keys)}")
    
    for key in all_keys:
        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
        data_type = redis_client.type(key)
        if data_type == b'string':
            data_size = redis_client.strlen(key)
            print(f"  {key_str}: {data_type.decode()} ({data_size} bytes)")
        else:
            print(f"  {key_str}: {data_type.decode()}")
    
    # Look specifically for our graph data
    graph_keys = [k for k in all_keys if b'graph' in k.lower()]
    print(f"\n📊 Graph-related keys: {len(graph_keys)}")
    
    for key in graph_keys:
        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
        print(f"\n🔍 Inspecting key: {key_str}")
        
        data_type = redis_client.type(key)
        if data_type == b'string':
            data = redis_client.get(key)
            print(f"  Size: {len(data)} bytes")
            
            # Try to deserialize if it looks like pickled data
            try:
                if data.startswith(b'\x80'):  # Pickle magic number
                    obj = pickle.loads(data)
                    if hasattr(obj, 'number_of_nodes'):
                        print(f"  Graph object: {type(obj).__name__}")
                        print(f"  Nodes: {obj.number_of_nodes()}")
                        print(f"  Edges: {obj.number_of_edges()}")
                        if obj.number_of_nodes() > 0:
                            print(f"  Sample nodes: {list(obj.nodes())[:5]}")
                        if obj.number_of_edges() > 0:
                            print(f"  Sample edges: {list(obj.edges())[:5]}")
                    else:
                        print(f"  Pickled object: {type(obj)} = {obj}")
                else:
                    # Try JSON
                    try:
                        import json
                        obj = json.loads(data.decode('utf-8'))
                        print(f"  JSON object: {obj}")
                    except:
                        print(f"  Raw data (first 100 chars): {data[:100]}")
            except Exception as e:
                print(f"  Failed to deserialize: {e}")
                print(f"  Raw data (first 50 bytes): {data[:50]}")
    
    # Test the persistence directly
    print(f"\n🧪 Testing direct persistence...")
    
    try:
        import security_patches
        import add_persistence
        from src.networkx_mcp.server import graph_manager
        
        # Set up
        security_patches.apply_critical_patches()
        storage = add_persistence.patch_graph_manager_with_persistence()
        
        print(f"✅ Persistence set up with backend: {type(storage).__name__}")
        
        # Create a simple test graph
        print("📝 Creating debug test graph...")
        result = graph_manager.create_graph("debug_test", "Graph")
        print(f"  Create result: {result}")
        
        # Add some nodes directly to the graph object
        graph = graph_manager.graphs.get("debug_test")
        if graph is not None:
            print(f"  Graph object: {type(graph)} with {graph.number_of_nodes()} nodes")
            
            # Add nodes using NetworkX directly
            graph.add_nodes_from(["A", "B", "C"])
            graph.add_edges_from([("A", "B"), ("B", "C")])
            
            print(f"  After direct addition: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            
            # Save to persistence manually
            if hasattr(graph_manager, '_storage'):
                save_result = graph_manager._storage.save_graph(
                    graph_manager._default_user,
                    "debug_test",
                    graph,
                    {"test": "direct_save"}
                )
                print(f"  Manual save result: {save_result}")
                
                # Check what was saved
                print("🔍 Checking what was saved to Redis...")
                time.sleep(1)  # Give Redis a moment
                
                saved_keys = redis_client.keys("*debug_test*")
                for key in saved_keys:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    data = redis_client.get(key)
                    print(f"  {key_str}: {len(data)} bytes")
                    
                    if data.startswith(b'\x80'):  # Pickle
                        try:
                            obj = pickle.loads(data)
                            if hasattr(obj, 'number_of_nodes'):
                                print(f"    Saved graph: {obj.number_of_nodes()} nodes, {obj.number_of_edges()} edges")
                        except Exception as e:
                            print(f"    Failed to load: {e}")
                
                # Now try to load it back
                print("🔍 Testing load from Redis...")
                loaded_graph = graph_manager._storage.load_graph(
                    graph_manager._default_user,
                    "debug_test"
                )
                
                if loaded_graph is not None:
                    print(f"  Loaded graph: {type(loaded_graph)} with {loaded_graph.number_of_nodes()} nodes, {loaded_graph.number_of_edges()} edges")
                    if loaded_graph.number_of_nodes() > 0:
                        print(f"  Nodes: {list(loaded_graph.nodes())}")
                    if loaded_graph.number_of_edges() > 0:
                        print(f"  Edges: {list(loaded_graph.edges())}")
                else:
                    print("  ❌ Failed to load graph!")
        else:
            print("  ❌ No graph object found!")
            
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_redis_persistence()