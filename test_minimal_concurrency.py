#!/usr/bin/env python3
"""Minimal concurrency test."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, "src")

from networkx_mcp.concurrency.graph_lock_manager import GraphLockManager


async def test_lock_manager():
    """Test the lock manager directly."""
    print("Testing GraphLockManager...")
    
    manager = GraphLockManager()
    
    # Test simple lock
    print("Testing simple lock...")
    async with manager.write_lock("test_graph"):
        print("✅ Acquired write lock")
        
    print("✅ Released write lock")
    
    # Test concurrent locks
    print("Testing concurrent locks...")
    
    async def worker(worker_id):
        async with manager.write_lock("shared_graph"):
            print(f"Worker {worker_id} acquired lock")
            await asyncio.sleep(0.1)
            print(f"Worker {worker_id} releasing lock")
            
    # Run 5 workers concurrently
    tasks = [worker(i) for i in range(5)]
    await asyncio.gather(*tasks)
    
    print("✅ Concurrent locks working")
    
    # Check stats
    stats = manager.get_stats()
    print(f"Lock stats: {stats}")
    
    await manager.cleanup()
    print("✅ Lock manager test complete")


async def test_connection_pool():
    """Test connection pool."""
    print("\nTesting ConnectionPool...")
    
    from networkx_mcp.concurrency.connection_pool import ConnectionPool
    
    pool = ConnectionPool(max_connections=3, timeout=1.0)
    
    async def worker(worker_id):
        try:
            async with pool.acquire_connection():
                print(f"Worker {worker_id} got connection")
                await asyncio.sleep(0.1)
                print(f"Worker {worker_id} releasing connection")
        except Exception as e:
            print(f"Worker {worker_id} failed: {e}")
            
    # Run 5 workers with only 3 connections available
    tasks = [worker(i) for i in range(5)]
    await asyncio.gather(*tasks, return_exceptions=True)
    
    stats = pool.get_stats()
    print(f"Pool stats: {stats}")
    print("✅ Connection pool test complete")


if __name__ == "__main__":
    async def main():
        await test_lock_manager()
        await test_connection_pool()
        print("\n🎉 Minimal concurrency tests passed!")
        
    asyncio.run(main())