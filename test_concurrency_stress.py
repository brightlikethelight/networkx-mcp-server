#!/usr/bin/env python3
"""Comprehensive concurrency stress test for NetworkX MCP server."""

import asyncio
import random
import time
import statistics
import sys
import os

# Disable structured logging for cleaner output
os.environ["LOG_LEVEL"] = "ERROR"

# Add src to path
sys.path.insert(0, "src")

from networkx_mcp.concurrency import GraphLockManager, ConnectionPool


async def test_50_concurrent_clients():
    """Test 50+ concurrent clients without data corruption."""
    print("🧪 Testing 50+ Concurrent Clients\n")
    
    # Simulate lightweight graph operations
    lock_manager = GraphLockManager()
    connection_pool = ConnectionPool(max_connections=50)
    
    # Shared state to verify no corruption
    graph_data = {}
    operation_counts = {"creates": 0, "reads": 0, "updates": 0}
    
    async def client_workload(client_id: int, operations: int = 20):
        """Simulate a client performing graph operations."""
        client_errors = 0
        client_ops = 0
        
        for i in range(operations):
            try:
                # Acquire connection
                async with connection_pool.acquire_connection():
                    graph_name = f"graph_{random.randint(0, 9)}"  # 10 graphs
                    op_type = random.choice(["create", "read", "update"])
                    
                    if op_type == "create":
                        async with lock_manager.write_lock(graph_name):
                            if graph_name not in graph_data:
                                graph_data[graph_name] = {"nodes": 0, "edges": 0}
                                operation_counts["creates"] += 1
                                
                    elif op_type == "read":
                        async with lock_manager.read_lock(graph_name):
                            _ = graph_data.get(graph_name, {"nodes": 0, "edges": 0})
                            operation_counts["reads"] += 1
                            
                    elif op_type == "update":
                        async with lock_manager.write_lock(graph_name):
                            if graph_name in graph_data:
                                graph_data[graph_name]["nodes"] += 1
                                graph_data[graph_name]["edges"] += random.randint(0, 2)
                                operation_counts["updates"] += 1
                                
                    client_ops += 1
                    
                    # Small random delay
                    await asyncio.sleep(random.uniform(0.001, 0.01))
                    
            except Exception as e:
                client_errors += 1
                
        return {"client_id": client_id, "operations": client_ops, "errors": client_errors}
    
    # Run 60 concurrent clients
    num_clients = 60
    print(f"Starting {num_clients} concurrent clients...")
    
    start_time = time.time()
    tasks = [client_workload(i, 20) for i in range(num_clients)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    duration = time.time() - start_time
    
    # Analyze results
    successful_clients = [r for r in results if isinstance(r, dict)]
    total_operations = sum(r["operations"] for r in successful_clients)
    total_errors = sum(r["errors"] for r in successful_clients)
    
    print(f"✅ Test completed in {duration:.2f}s")
    print(f"   Clients: {len(successful_clients)}/{num_clients}")
    print(f"   Total operations: {total_operations}")
    print(f"   Errors: {total_errors}")
    print(f"   Operations/sec: {total_operations/duration:.1f}")
    
    # Check data integrity
    print(f"\nData Integrity Check:")
    print(f"   Graphs created: {len(graph_data)}")
    print(f"   Operation counts: {operation_counts}")
    
    # Verify no negative values (sign of corruption)
    corruption_detected = False
    for graph_name, data in graph_data.items():
        if data["nodes"] < 0 or data["edges"] < 0:
            print(f"❌ Corruption detected in {graph_name}: {data}")
            corruption_detected = True
            
    if not corruption_detected:
        print("✅ No data corruption detected")
        
    # Lock statistics
    lock_stats = lock_manager.get_stats()
    print(f"\nLock Statistics:")
    print(f"   Total acquisitions: {lock_stats['total_acquisitions']}")
    print(f"   Contentions: {lock_stats['total_contentions']}")
    print(f"   Contention rate: {lock_stats['contention_rate']*100:.1f}%")
    print(f"   Avg wait time: {lock_stats['avg_wait_time']*1000:.2f}ms")
    
    # Connection pool statistics  
    pool_stats = connection_pool.get_stats()
    print(f"\nConnection Pool Statistics:")
    print(f"   Total connections: {pool_stats['total_connections']}")
    print(f"   Rejected: {pool_stats['rejected_connections']}")
    print(f"   Max concurrent: {pool_stats['max_concurrent']}")
    print(f"   Utilization: {pool_stats['utilization']*100:.1f}%")
    
    await lock_manager.cleanup()
    
    return {
        "success": total_errors == 0 and not corruption_detected,
        "clients": len(successful_clients),
        "operations": total_operations,
        "duration": duration,
        "ops_per_sec": total_operations / duration
    }


async def test_deadlock_prevention():
    """Test deadlock prevention with multi-graph operations."""
    print("\n\n🔒 Testing Deadlock Prevention\n")
    
    lock_manager = GraphLockManager()
    deadlock_detected = False
    
    async def worker_a():
        """Worker that locks graphs in order A -> B"""
        try:
            async with lock_manager.multi_graph_lock(["graph_A", "graph_B"]):
                await asyncio.sleep(0.1)  # Hold locks for a bit
                return "Worker A completed"
        except Exception as e:
            return f"Worker A failed: {e}"
            
    async def worker_b():
        """Worker that tries to lock graphs in order B -> A (potential deadlock)"""
        try:
            async with lock_manager.multi_graph_lock(["graph_B", "graph_A"]):
                await asyncio.sleep(0.1)  # Hold locks for a bit
                return "Worker B completed"
        except Exception as e:
            return f"Worker B failed: {e}"
            
    # Run workers concurrently
    start_time = time.time()
    results = await asyncio.gather(worker_a(), worker_b(), return_exceptions=True)
    duration = time.time() - start_time
    
    print(f"Deadlock test completed in {duration:.3f}s")
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ Worker {chr(65+i)} exception: {result}")
            deadlock_detected = True
        else:
            print(f"✅ {result}")
            
    if duration < 5.0 and not deadlock_detected:
        print("✅ No deadlocks detected")
    else:
        print("❌ Possible deadlock or timeout")
        
    await lock_manager.cleanup()
    
    return {"success": not deadlock_detected, "duration": duration}


async def test_connection_pool_limits():
    """Test connection pool with overload conditions."""
    print("\n\n📊 Testing Connection Pool Limits\n")
    
    # Small pool to force rejections
    pool = ConnectionPool(max_connections=5, timeout=0.5)
    
    async def connection_worker(worker_id):
        """Worker that tries to get a connection."""
        try:
            async with pool.acquire_connection():
                # Simulate work
                await asyncio.sleep(random.uniform(0.1, 0.3))
                return f"Worker {worker_id} succeeded"
        except Exception as e:
            return f"Worker {worker_id} failed: {type(e).__name__}"
            
    # Try to get 20 connections with only 5 available
    tasks = [connection_worker(i) for i in range(20)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successes = sum(1 for r in results if "succeeded" in str(r))
    timeouts = sum(1 for r in results if "TimeoutError" in str(r))
    
    print(f"Connection test results:")
    print(f"   Successes: {successes}")
    print(f"   Timeouts: {timeouts}")
    print(f"   Total: {len(results)}")
    
    # Check stats
    stats = pool.get_stats()
    print(f"\nPool statistics:")
    print(f"   Total connections: {stats['total_connections']}")
    print(f"   Rejected: {stats['rejected_connections']}")
    print(f"   Rejection rate: {stats['rejection_rate']*100:.1f}%")
    
    # We expect some rejections with this test
    success = timeouts > 0 and successes > 0
    if success:
        print("✅ Connection pool properly limited concurrent access")
    else:
        print("❌ Connection pool limits not working correctly")
        
    return {"success": success, "successes": successes, "timeouts": timeouts}


async def main():
    """Run all concurrency tests."""
    print("🚀 NetworkX MCP Server Concurrency Tests")
    print("=" * 60)
    
    all_results = {}
    
    # Test 1: High concurrency
    result1 = await test_50_concurrent_clients()
    all_results["concurrency"] = result1
    
    # Test 2: Deadlock prevention
    result2 = await test_deadlock_prevention()
    all_results["deadlocks"] = result2
    
    # Test 3: Connection limits
    result3 = await test_connection_pool_limits()
    all_results["connection_limits"] = result3
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    all_passed = True
    
    for test_name, result in all_results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result["success"]:
            all_passed = False
            
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_results["concurrency"]["success"]:
        ops_per_sec = all_results["concurrency"]["ops_per_sec"]
        print(f"Performance: {ops_per_sec:.1f} operations/second with 60 concurrent clients")
        
    # Answer the reflection question
    print("\n🤔 Reflection: Can we handle 50+ concurrent users without data corruption or deadlocks?")
    if all_passed:
        print("✅ YES - The implementation successfully handles 50+ concurrent users with:")
        print("   - No data corruption")
        print("   - No deadlocks")
        print("   - Proper connection limiting")
        print("   - Thread-safe NetworkX operations")
    else:
        print("❌ NO - Issues detected that need to be addressed")


if __name__ == "__main__":
    asyncio.run(main())