#!/usr/bin/env python3
"""Simple lock test without imports that might cause initialization."""

import asyncio
from contextlib import asynccontextmanager


class SimpleLockManager:
    """Simplified lock manager for testing."""
    
    def __init__(self):
        self.locks = {}
        
    def get_lock(self, name):
        if name not in self.locks:
            self.locks[name] = asyncio.Lock()
        return self.locks[name]
        
    @asynccontextmanager
    async def lock(self, name):
        lock = self.get_lock(name)
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()


async def test_simple():
    """Test simple lock functionality."""
    print("Testing simple lock...")
    
    manager = SimpleLockManager()
    
    # Test 1: Basic lock
    async with manager.lock("test"):
        print("✅ Acquired lock")
        
    print("✅ Released lock")
    
    # Test 2: Concurrent locks
    results = []
    
    async def worker(i):
        async with manager.lock("shared"):
            results.append(f"Worker {i} start")
            await asyncio.sleep(0.01)
            results.append(f"Worker {i} end")
            
    tasks = [worker(i) for i in range(3)]
    await asyncio.gather(*tasks)
    
    print("✅ Concurrent access completed")
    print(f"   Results: {results}")
    
    # Verify serialization
    for i in range(0, len(results), 2):
        start = results[i]
        end = results[i + 1]
        worker_id = start.split()[1]
        assert end.split()[1] == worker_id, "Lock not working properly"
        
    print("✅ Lock serialization verified")


if __name__ == "__main__":
    asyncio.run(test_simple())