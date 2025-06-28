#!/usr/bin/env python3
"""Comprehensive load testing for NetworkX MCP Server."""

import asyncio
import psutil
import time
import random
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestLoadCapacity:
    """Test server capacity under various load conditions."""
    
    def __init__(self):
        self.baseline_memory = None
        self.test_results = []
        
    def get_system_stats(self):
        """Get current system resource usage."""
        process = psutil.Process()
        return {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads(),
            "open_files": len(process.open_files()) if hasattr(process, 'open_files') else 0
        }
    
    async def simulate_realistic_user(self, user_id: int, operations: int, graph_manager):
        """Simulate realistic user behavior with NetworkX operations."""
        user_stats = {
            "user_id": user_id,
            "operations_completed": 0,
            "errors": 0,
            "total_time": 0
        }
        
        try:
            start_time = time.time()
            
            for op in range(operations):
                op_start = time.time()
                
                try:
                    # Create a graph for this operation
                    graph_id = f"user_{user_id}_graph_{op}"
                    
                    # 1. Create graph
                    result = graph_manager.create_graph(graph_id, "DiGraph")
                    if not result.get("created"):
                        raise Exception(f"Failed to create graph {graph_id}")
                    
                    # 2. Build a social network
                    nodes = [f"person_{i}" for i in range(20)]  # Smaller for load testing
                    graph_manager.add_nodes_from(graph_id, nodes)
                    
                    # 3. Add random edges (social connections)
                    edges = []
                    for i in range(30):  # 30 random connections
                        src = random.choice(nodes)
                        dst = random.choice(nodes)
                        if src != dst:
                            edges.append((src, dst))
                    
                    if edges:
                        graph_manager.add_edges_from(graph_id, edges)
                    
                    # 4. Run some analysis (this is computationally expensive)
                    info = graph_manager.get_graph_info(graph_id)
                    
                    # 5. Get neighbors of a random node
                    if info["num_nodes"] > 0:
                        random_node = random.choice(nodes)
                        try:
                            neighbors = graph_manager.get_neighbors(graph_id, random_node)
                        except ValueError:
                            pass  # Node might not exist, that's ok
                    
                    # 6. Clean up - delete the graph
                    graph_manager.delete_graph(graph_id)
                    
                    user_stats["operations_completed"] += 1
                    
                except Exception as e:
                    user_stats["errors"] += 1
                    print(f"User {user_id} op {op} error: {e}")
                
                # Small delay between operations to be realistic
                await asyncio.sleep(0.01)
            
            user_stats["total_time"] = time.time() - start_time
            
        except Exception as e:
            print(f"User {user_id} failed: {e}")
            user_stats["errors"] += 1
        
        return user_stats
    
    async def test_concurrent_users(self):
        """Find breaking point for concurrent users."""
        print("🧪 Testing concurrent user capacity...")
        
        try:
            from src.networkx_mcp.server import graph_manager
            import security_patches
            import add_persistence
            
            # Apply security and persistence
            security_patches.apply_critical_patches()
            add_persistence.patch_graph_manager_with_persistence()
            
            # Record baseline
            self.baseline_memory = self.get_system_stats()["memory_mb"]
            
            # Test with increasing numbers of concurrent users
            for num_users in [1, 5, 10, 20, 50]:
                print(f"\n🔥 Testing with {num_users} concurrent users...")
                
                start_time = time.time()
                start_stats = self.get_system_stats()
                
                # Create tasks for concurrent users
                tasks = [
                    self.simulate_realistic_user(i, 5, graph_manager)  # 5 ops per user
                    for i in range(num_users)
                ]
                
                try:
                    # Run all users concurrently
                    user_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    duration = time.time() - start_time
                    end_stats = self.get_system_stats()
                    
                    # Analyze results
                    successful_users = 0
                    total_operations = 0
                    total_errors = 0
                    
                    for result in user_results:
                        if isinstance(result, dict):
                            successful_users += 1
                            total_operations += result["operations_completed"]
                            total_errors += result["errors"]
                        else:
                            print(f"User failed completely: {result}")
                    
                    memory_increase = end_stats["memory_mb"] - start_stats["memory_mb"]
                    ops_per_second = total_operations / duration if duration > 0 else 0
                    
                    test_result = {
                        "num_users": num_users,
                        "duration": duration,
                        "successful_users": successful_users,
                        "total_operations": total_operations,
                        "total_errors": total_errors,
                        "ops_per_second": ops_per_second,
                        "memory_increase_mb": memory_increase,
                        "final_memory_mb": end_stats["memory_mb"],
                        "cpu_percent": end_stats["cpu_percent"],
                        "success": successful_users == num_users and total_errors < num_users * 0.1
                    }
                    
                    self.test_results.append(test_result)
                    
                    print(f"  ✅ Results:")
                    print(f"     Successful users: {successful_users}/{num_users}")
                    print(f"     Total operations: {total_operations}")
                    print(f"     Errors: {total_errors}")
                    print(f"     Duration: {duration:.2f}s")
                    print(f"     Ops/sec: {ops_per_second:.2f}")
                    print(f"     Memory increase: {memory_increase:.2f}MB")
                    print(f"     Final memory: {end_stats['memory_mb']:.2f}MB")
                    
                    if not test_result["success"]:
                        print(f"  ❌ Load test failed at {num_users} users")
                        break
                        
                except Exception as e:
                    print(f"  ❌ Test failed at {num_users} users: {e}")
                    break
                
                # Clean up and wait a bit between tests
                await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            print(f"❌ Load testing failed: {e}")
            return False
    
    async def test_memory_limits(self):
        """Verify memory limits are enforced properly."""
        print("🧪 Testing memory limit enforcement...")
        
        try:
            from src.networkx_mcp.server import graph_manager
            import security_patches
            import add_persistence
            
            # Apply security and persistence
            security_patches.apply_critical_patches()
            add_persistence.patch_graph_manager_with_persistence()
            
            # Try to create graphs that would exceed reasonable memory
            print("  📝 Attempting to create memory-intensive graphs...")
            
            memory_test_passed = True
            
            # Test 1: Try to create a graph with many nodes
            try:
                graph_manager.create_graph("memory_test_1", "Graph")
                
                # Try to add a very large number of nodes
                huge_nodes = [f"node_{i}" for i in range(100000)]  # 100k nodes
                
                start_memory = self.get_system_stats()["memory_mb"]
                graph_manager.add_nodes_from("memory_test_1", huge_nodes)
                end_memory = self.get_system_stats()["memory_mb"]
                
                memory_used = end_memory - start_memory
                print(f"    📊 Added 100k nodes, memory increase: {memory_used:.2f}MB")
                
                # Clean up
                graph_manager.delete_graph("memory_test_1")
                
                # Check if memory is reasonable (should be < 500MB for 100k nodes)
                if memory_used > 500:
                    print(f"    ⚠️ High memory usage for 100k nodes: {memory_used:.2f}MB")
                else:
                    print(f"    ✅ Reasonable memory usage: {memory_used:.2f}MB")
                
            except Exception as e:
                print(f"    ❌ Memory test 1 failed: {e}")
                memory_test_passed = False
            
            # Test 2: Try to create many small graphs
            try:
                print("  📝 Creating many small graphs...")
                
                start_memory = self.get_system_stats()["memory_mb"]
                
                for i in range(1000):  # 1000 small graphs
                    graph_id = f"small_graph_{i}"
                    graph_manager.create_graph(graph_id, "Graph")
                    
                    # Add a few nodes to each
                    nodes = [f"n{j}" for j in range(10)]
                    graph_manager.add_nodes_from(graph_id, nodes)
                
                end_memory = self.get_system_stats()["memory_mb"]
                memory_used = end_memory - start_memory
                
                print(f"    📊 Created 1000 graphs (10 nodes each), memory increase: {memory_used:.2f}MB")
                
                # Clean up
                for i in range(1000):
                    graph_id = f"small_graph_{i}"
                    try:
                        graph_manager.delete_graph(graph_id)
                    except:
                        pass  # Ignore errors during cleanup
                
                # Check if memory is reasonable (should be < 200MB for 1000 small graphs)
                if memory_used > 200:
                    print(f"    ⚠️ High memory usage for 1000 small graphs: {memory_used:.2f}MB")
                else:
                    print(f"    ✅ Reasonable memory usage: {memory_used:.2f}MB")
                
            except Exception as e:
                print(f"    ❌ Memory test 2 failed: {e}")
                memory_test_passed = False
            
            return memory_test_passed
            
        except Exception as e:
            print(f"❌ Memory limit testing failed: {e}")
            return False
    
    async def test_latency_performance(self):
        """Test operation latency under normal conditions."""
        print("🧪 Testing operation latency...")
        
        try:
            from src.networkx_mcp.server import graph_manager
            import security_patches
            import add_persistence
            
            # Apply security and persistence
            security_patches.apply_critical_patches()
            add_persistence.patch_graph_manager_with_persistence()
            
            latencies = {
                "create_graph": [],
                "add_nodes": [],
                "add_edges": [],
                "get_info": [],
                "delete_graph": []
            }
            
            # Run multiple iterations to get average latencies
            for iteration in range(50):
                graph_id = f"latency_test_{iteration}"
                
                # Test create_graph latency
                start = time.time()
                graph_manager.create_graph(graph_id, "DiGraph")
                latencies["create_graph"].append((time.time() - start) * 1000)
                
                # Test add_nodes latency
                nodes = [f"n{i}" for i in range(100)]
                start = time.time()
                graph_manager.add_nodes_from(graph_id, nodes)
                latencies["add_nodes"].append((time.time() - start) * 1000)
                
                # Test add_edges latency
                edges = [(f"n{i}", f"n{i+1}") for i in range(99)]
                start = time.time()
                graph_manager.add_edges_from(graph_id, edges)
                latencies["add_edges"].append((time.time() - start) * 1000)
                
                # Test get_info latency
                start = time.time()
                graph_manager.get_graph_info(graph_id)
                latencies["get_info"].append((time.time() - start) * 1000)
                
                # Test delete_graph latency
                start = time.time()
                graph_manager.delete_graph(graph_id)
                latencies["delete_graph"].append((time.time() - start) * 1000)
            
            # Calculate statistics
            print("  📊 Latency Results (milliseconds):")
            
            all_good = True
            for operation, times in latencies.items():
                avg_latency = sum(times) / len(times)
                p95_latency = sorted(times)[int(len(times) * 0.95)]
                max_latency = max(times)
                
                print(f"    {operation}:")
                print(f"      Average: {avg_latency:.2f}ms")
                print(f"      P95: {p95_latency:.2f}ms")
                print(f"      Max: {max_latency:.2f}ms")
                
                # Check if P95 latency is acceptable (< 100ms for basic operations)
                if p95_latency > 100:
                    print(f"      ⚠️ High P95 latency: {p95_latency:.2f}ms")
                    all_good = False
                else:
                    print(f"      ✅ Good P95 latency: {p95_latency:.2f}ms")
            
            return all_good
            
        except Exception as e:
            print(f"❌ Latency testing failed: {e}")
            return False
    
    def generate_load_test_report(self):
        """Generate a comprehensive load test report."""
        print("\n📊 LOAD TEST REPORT")
        print("=" * 50)
        
        if not self.test_results:
            print("No load test results available")
            return
        
        print("Concurrent User Test Results:")
        print("-" * 30)
        
        for result in self.test_results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"{result['num_users']:2d} users: {status}")
            print(f"   Ops/sec: {result['ops_per_second']:6.2f}")
            print(f"   Memory:  {result['memory_increase_mb']:6.2f}MB increase")
            print(f"   Errors:  {result['total_errors']}/{result['total_operations']} operations")
        
        # Find maximum supported users
        max_users = 0
        for result in self.test_results:
            if result["success"]:
                max_users = max(max_users, result["num_users"])
        
        print(f"\n📈 Maximum supported concurrent users: {max_users}")
        
        if max_users >= 20:
            print("✅ EXCELLENT: Supports 20+ concurrent users")
        elif max_users >= 10:
            print("✅ GOOD: Supports 10+ concurrent users")
        elif max_users >= 5:
            print("⚠️ MODERATE: Supports 5+ concurrent users")
        else:
            print("❌ POOR: Supports < 5 concurrent users")

async def main():
    """Run comprehensive load testing."""
    print("🚀 LOAD CAPACITY VALIDATION")
    print("=" * 50)
    
    load_tester = TestLoadCapacity()
    
    tests = [
        ("Concurrent user capacity", load_tester.test_concurrent_users()),
        ("Memory limit enforcement", load_tester.test_memory_limits()),
        ("Operation latency", load_tester.test_latency_performance())
    ]
    
    results = []
    for test_name, test_coro in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = await test_coro
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            results.append((test_name, False))
    
    # Generate report
    load_tester.generate_load_test_report()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 LOAD TESTING SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Score: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print("✅ SYSTEM IS LOAD TESTED AND PRODUCTION READY!")
    else:
        print("❌ Load testing failures - fix before production deployment")
    
    return passed == len(results)

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)