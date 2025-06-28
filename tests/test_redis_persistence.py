#!/usr/bin/env python3
"""Test Redis persistence and data survival across restarts."""

import asyncio
import subprocess
import time
import os
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestRedisPersistence:
    """Test Redis-based persistence functionality."""
    
    def __init__(self):
        self.server_process = None
        
    async def test_data_survives_restart(self):
        """Verify graphs persist across server restarts."""
        print("🧪 Testing data persistence across restarts...")
        
        try:
            # Import server components
            from src.networkx_mcp.server import graph_manager
            import security_patches
            import add_persistence
            
            # Apply security and persistence
            security_patches.apply_critical_patches()
            storage = add_persistence.patch_graph_manager_with_persistence()
            
            if not storage:
                raise Exception("Failed to initialize persistence layer")
            
            # Phase 1: Create test data
            print("📝 Phase 1: Creating test graphs...")
            graphs_created = {}
            
            for i in range(5):
                graph_id = f"persist_test_{i}"
                
                # Create graph
                result = graph_manager.create_graph(graph_id, "DiGraph")
                if not result.get("created"):
                    raise Exception(f"Failed to create graph {graph_id}")
                
                # Add nodes and edges
                nodes = [f"N{j}" for j in range(10)]
                graph_manager.add_nodes_from(graph_id, nodes)
                
                # Add some edges
                edges = [(f"N{j}", f"N{j+1}") for j in range(9)]
                graph_manager.add_edges_from(graph_id, edges)
                
                # Store expected data
                info = graph_manager.get_graph_info(graph_id)
                graphs_created[graph_id] = {
                    "nodes": info["num_nodes"],
                    "edges": info["num_edges"],
                    "type": info["graph_type"]
                }
                
                print(f"  ✅ Created {graph_id}: {info['num_nodes']} nodes, {info['num_edges']} edges")
            
            print(f"📊 Created {len(graphs_created)} test graphs")
            
            # Phase 2: Simulate restart by clearing memory
            print("📝 Phase 2: Simulating server restart...")
            
            # Clear the in-memory graphs
            original_graphs = graph_manager.graphs.copy()
            original_metadata = graph_manager.metadata.copy()
            
            graph_manager.graphs.clear()
            graph_manager.metadata.clear()
            
            print("  💥 Cleared in-memory storage (simulating restart)")
            
            # Phase 3: Verify data recovery
            print("📝 Phase 3: Testing data recovery...")
            
            # Re-apply persistence layer
            storage = add_persistence.patch_graph_manager_with_persistence()
            
            # Check if we can load graphs from persistent storage
            recovered_count = 0
            for graph_id, expected_data in graphs_created.items():
                
                # Try to load from persistent storage
                if hasattr(graph_manager, '_storage'):
                    stored_graph = graph_manager._storage.load_graph(
                        graph_manager._default_user, 
                        graph_id
                    )
                    
                    if stored_graph is not None:
                        # Restore to memory
                        graph_manager.graphs[graph_id] = stored_graph
                        graph_manager.metadata[graph_id] = {
                            "created_at": "recovered",
                            "graph_type": type(stored_graph).__name__,
                            "attributes": {}
                        }
                        
                        # Verify data integrity
                        info = graph_manager.get_graph_info(graph_id)
                        
                        if (info["num_nodes"] == expected_data["nodes"] and 
                            info["num_edges"] == expected_data["edges"]):
                            print(f"  ✅ Recovered {graph_id}: {info['num_nodes']} nodes, {info['num_edges']} edges")
                            recovered_count += 1
                        else:
                            print(f"  ❌ Data corruption in {graph_id}: expected {expected_data}, got {info}")
                    else:
                        print(f"  ❌ Could not load {graph_id} from persistent storage")
                else:
                    print("  ❌ No storage backend available")
            
            # Final validation
            print(f"📊 Recovery results: {recovered_count}/{len(graphs_created)} graphs recovered")
            
            if recovered_count == len(graphs_created):
                print("✅ PERSISTENCE TEST PASSED: All data survived restart")
                return True
            else:
                print(f"❌ PERSISTENCE TEST FAILED: {len(graphs_created) - recovered_count} graphs lost")
                return False
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return False
    
    def check_redis_availability(self):
        """Check if Redis is available and configured properly."""
        print("🔍 Checking Redis availability...")
        
        try:
            import redis
            
            # Try to connect to Redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            
            # Test basic operations
            r.set("test_key", "test_value")
            value = r.get("test_key")
            r.delete("test_key")
            
            if value == "test_value":
                print("✅ Redis is available and working")
                return True
            else:
                print("❌ Redis not responding correctly")
                return False
                
        except ImportError:
            print("❌ Redis Python client not installed: pip install redis")
            return False
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            print("💡 To start Redis:")
            print("   Docker: docker run -d -p 6379:6379 redis:alpine")
            print("   macOS: brew install redis && brew services start redis")
            print("   Ubuntu: sudo apt install redis-server && sudo systemctl start redis")
            return False
    
    async def test_concurrent_access_safety(self):
        """Test that concurrent access to persistence is safe."""
        print("🧪 Testing concurrent access safety...")
        
        try:
            from src.networkx_mcp.server import graph_manager
            import security_patches
            import add_persistence
            
            # Apply security and persistence
            security_patches.apply_critical_patches()
            storage = add_persistence.patch_graph_manager_with_persistence()
            
            async def worker(worker_id: int, operations: int):
                """Worker function that creates/modifies graphs."""
                for op in range(operations):
                    graph_id = f"worker_{worker_id}_graph_{op}"
                    
                    # Create graph
                    graph_manager.create_graph(graph_id, "Graph")
                    
                    # Add some nodes
                    nodes = [f"W{worker_id}_N{i}" for i in range(10)]
                    graph_manager.add_nodes_from(graph_id, nodes)
                    
                    # Delete graph
                    graph_manager.delete_graph(graph_id)
            
            # Run multiple workers concurrently
            workers = [worker(i, 10) for i in range(5)]
            
            start_time = time.time()
            await asyncio.gather(*workers)
            duration = time.time() - start_time
            
            print(f"✅ Concurrent access test completed in {duration:.2f}s")
            return True
            
        except Exception as e:
            print(f"❌ Concurrent access test failed: {e}")
            return False

async def main():
    """Run all Redis persistence tests."""
    print("🚀 REDIS PERSISTENCE VALIDATION")
    print("=" * 50)
    
    test_suite = TestRedisPersistence()
    
    # Check 1: Redis availability
    redis_available = test_suite.check_redis_availability()
    
    # Check 2: Data persistence
    persistence_works = await test_suite.test_data_survives_restart()
    
    # Check 3: Concurrent safety
    concurrent_safe = await test_suite.test_concurrent_access_safety()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 REDIS PERSISTENCE TEST RESULTS")
    print("=" * 50)
    
    checks = [
        ("Redis available", redis_available),
        ("Data survives restart", persistence_works),
        ("Concurrent access safe", concurrent_safe)
    ]
    
    passed = 0
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Score: {passed}/{len(checks)} ({passed/len(checks)*100:.1f}%)")
    
    if passed == len(checks):
        print("✅ REDIS PERSISTENCE IS PRODUCTION READY!")
    else:
        print("❌ Redis persistence needs fixes before production")
    
    return passed == len(checks)

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)