"""Thread-safe lock management for NetworkX graph operations.

NetworkX is not thread-safe, so we need to protect all graph operations
with locks to prevent data corruption in concurrent environments.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Optional
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LockStats:
    """Statistics for lock usage."""
    acquisitions: int = 0
    contentions: int = 0
    total_wait_time: float = 0.0
    max_wait_time: float = 0.0
    

class GraphLockManager:
    """Manages per-graph locks for thread-safe NetworkX operations."""
    
    def __init__(self, enable_stats: bool = True):
        """Initialize the lock manager.
        
        Args:
            enable_stats: Whether to collect lock statistics
        """
        self.locks: Dict[str, asyncio.Lock] = {}
        self.manager_lock = asyncio.Lock()  # Protects the locks dict
        self.enable_stats = enable_stats
        self.stats: Dict[str, LockStats] = {}
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
        
    async def get_lock(self, graph_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific graph.
        
        Args:
            graph_id: The graph identifier
            
        Returns:
            asyncio.Lock for the specified graph
        """
        async with self.manager_lock:
            if graph_id not in self.locks:
                self.locks[graph_id] = asyncio.Lock()
                if self.enable_stats:
                    self.stats[graph_id] = LockStats()
                logger.debug(f"Created new lock for graph: {graph_id}")
            
            # Periodic cleanup of unused locks
            if time.time() - self._last_cleanup > self._cleanup_interval:
                self._cleanup_unused_locks()
                
            return self.locks[graph_id]
    
    def _cleanup_unused_locks(self):
        """Remove locks for graphs that no longer exist."""
        # This would be called periodically to prevent memory leaks
        # In a real implementation, we'd check against actual graph existence
        self._last_cleanup = time.time()
        logger.debug("Performed lock cleanup")
            
    @asynccontextmanager
    async def read_lock(self, graph_id: str):
        """Async context manager for read operations.
        
        Note: Even read operations need locks since NetworkX is not thread-safe.
        
        Args:
            graph_id: The graph identifier
            
        Yields:
            None - The lock is held during the context
        """
        lock = await self.get_lock(graph_id)
        start_time = time.time()
        
        # Acquire async lock
        await lock.acquire()
        acquired = True
        
        wait_time = time.time() - start_time
        if self.enable_stats:
            self._update_stats(graph_id, wait_time, acquired)
            
        if wait_time > 0.1:  # Log slow acquisitions
            logger.warning(f"Slow lock acquisition for {graph_id}: {wait_time:.3f}s")
            
        try:
            yield
        finally:
            lock.release()
            logger.debug(f"Released read lock for {graph_id}")
            
    @asynccontextmanager
    async def write_lock(self, graph_id: str):
        """Async context manager for write operations.
        
        Args:
            graph_id: The graph identifier
            
        Yields:
            None - The lock is held during the context
        """
        lock = await self.get_lock(graph_id)
        start_time = time.time()
        
        # Acquire async lock
        await lock.acquire()
        acquired = True
        
        wait_time = time.time() - start_time
        if self.enable_stats:
            self._update_stats(graph_id, wait_time, acquired)
            
        if wait_time > 0.1:  # Log slow acquisitions
            logger.warning(f"Slow lock acquisition for {graph_id}: {wait_time:.3f}s")
            
        try:
            yield
        finally:
            lock.release()
            logger.debug(f"Released write lock for {graph_id}")
    
    @asynccontextmanager
    async def multi_graph_lock(self, graph_ids: list[str], operation: str = "write"):
        """Lock multiple graphs atomically to prevent deadlocks.
        
        Always acquires locks in sorted order to prevent deadlocks.
        
        Args:
            graph_ids: List of graph identifiers
            operation: "read" or "write"
            
        Yields:
            None - All locks are held during the context
        """
        # Sort graph IDs to ensure consistent lock ordering
        sorted_ids = sorted(set(graph_ids))
        locks = []
        
        try:
            # Acquire all locks in order
            for graph_id in sorted_ids:
                if operation == "read":
                    lock_ctx = self.read_lock(graph_id)
                else:
                    lock_ctx = self.write_lock(graph_id)
                    
                await lock_ctx.__aenter__()
                locks.append(lock_ctx)
                
            yield
            
        finally:
            # Release in reverse order
            for lock_ctx in reversed(locks):
                try:
                    await lock_ctx.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error releasing lock: {e}")
    
    def _update_stats(self, graph_id: str, wait_time: float, acquired: bool):
        """Update lock statistics.
        
        Args:
            graph_id: The graph identifier
            wait_time: Time waited for lock acquisition
            acquired: Whether lock was successfully acquired
        """
        if graph_id not in self.stats:
            self.stats[graph_id] = LockStats()
            
        stats = self.stats[graph_id]
        stats.acquisitions += 1
        
        if wait_time > 0.001:  # Consider it contention if wait > 1ms
            stats.contentions += 1
            
        stats.total_wait_time += wait_time
        stats.max_wait_time = max(stats.max_wait_time, wait_time)
    
    def get_stats(self, graph_id: Optional[str] = None) -> Dict:
        """Get lock statistics.
        
        Args:
            graph_id: Specific graph or None for all graphs
            
        Returns:
            Dictionary of lock statistics
        """
        if not self.enable_stats:
            return {"stats_enabled": False}
            
        if graph_id:
            stats = self.stats.get(graph_id, LockStats())
            return {
                "graph_id": graph_id,
                "acquisitions": stats.acquisitions,
                "contentions": stats.contentions,
                "contention_rate": stats.contentions / max(stats.acquisitions, 1),
                "avg_wait_time": stats.total_wait_time / max(stats.acquisitions, 1),
                "max_wait_time": stats.max_wait_time
            }
        else:
            # Aggregate statistics
            total_acquisitions = sum(s.acquisitions for s in self.stats.values())
            total_contentions = sum(s.contentions for s in self.stats.values())
            total_wait_time = sum(s.total_wait_time for s in self.stats.values())
            max_wait = max((s.max_wait_time for s in self.stats.values()), default=0)
            
            return {
                "total_graphs": len(self.stats),
                "total_acquisitions": total_acquisitions,
                "total_contentions": total_contentions,
                "contention_rate": total_contentions / max(total_acquisitions, 1),
                "avg_wait_time": total_wait_time / max(total_acquisitions, 1),
                "max_wait_time": max_wait,
                "per_graph": {
                    gid: self.get_stats(gid) for gid in self.stats
                }
            }
    
    async def reset_stats(self):
        """Reset all statistics."""
        async with self.manager_lock:
            self.stats.clear()
            
    async def cleanup(self):
        """Cleanup resources (call on shutdown)."""
        async with self.manager_lock:
            self.locks.clear()
            self.stats.clear()