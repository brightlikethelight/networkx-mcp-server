"""Connection pooling and request management for MCP server.

Provides connection limiting and request queuing to prevent server overload.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ConnectionStats:
    """Statistics for connection pool."""
    total_connections: int = 0
    active_connections: int = 0
    rejected_connections: int = 0
    total_wait_time: float = 0.0
    max_wait_time: float = 0.0
    max_concurrent: int = 0


@dataclass
class QueuedRequest:
    """A request waiting in the queue."""
    request_id: str
    priority: RequestPriority
    timestamp: float
    future: asyncio.Future
    

class ConnectionPool:
    """Manages concurrent connections with configurable limits."""
    
    def __init__(self, 
                 max_connections: int = 50,
                 timeout: float = 30.0,
                 enable_stats: bool = True):
        """Initialize connection pool.
        
        Args:
            max_connections: Maximum concurrent connections
            timeout: Connection acquisition timeout
            enable_stats: Whether to collect statistics
        """
        self.max_connections = max_connections
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_connections)
        self.active_connections = 0
        self.enable_stats = enable_stats
        self.stats = ConnectionStats()
        self._lock = asyncio.Lock()
        
    @asynccontextmanager
    async def acquire_connection(self):
        """Acquire a connection from the pool.
        
        Yields:
            None - Connection is acquired during context
            
        Raises:
            TimeoutError: If connection cannot be acquired within timeout
        """
        start_time = time.time()
        acquired = False
        
        try:
            # Try to acquire with timeout
            try:
                await asyncio.wait_for(
                    self.semaphore.acquire(), 
                    timeout=self.timeout
                )
                acquired = True
            except asyncio.TimeoutError:
                if self.enable_stats:
                    async with self._lock:
                        self.stats.rejected_connections += 1
                raise TimeoutError(
                    f"Could not acquire connection within {self.timeout}s. "
                    f"Pool size: {self.max_connections}, active: {self.active_connections}"
                )
            
            wait_time = time.time() - start_time
            
            async with self._lock:
                self.active_connections += 1
                
                if self.enable_stats:
                    self.stats.total_connections += 1
                    self.stats.total_wait_time += wait_time
                    self.stats.max_wait_time = max(self.stats.max_wait_time, wait_time)
                    self.stats.max_concurrent = max(self.stats.max_concurrent, self.active_connections)
                    
                if wait_time > 1.0:
                    logger.warning(
                        f"Slow connection acquisition: {wait_time:.2f}s. "
                        f"Active: {self.active_connections}/{self.max_connections}"
                    )
                    
            yield
            
        finally:
            if acquired:
                async with self._lock:
                    self.active_connections -= 1
                self.semaphore.release()
                
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.enable_stats:
            return {"stats_enabled": False}
            
        return {
            "max_connections": self.max_connections,
            "active_connections": self.active_connections,
            "total_connections": self.stats.total_connections,
            "rejected_connections": self.stats.rejected_connections,
            "rejection_rate": self.stats.rejected_connections / max(self.stats.total_connections, 1),
            "avg_wait_time": self.stats.total_wait_time / max(self.stats.total_connections, 1),
            "max_wait_time": self.stats.max_wait_time,
            "max_concurrent": self.stats.max_concurrent,
            "utilization": self.active_connections / self.max_connections
        }
        
    def reset_stats(self):
        """Reset statistics."""
        self.stats = ConnectionStats()
        

class RequestQueue:
    """Priority queue for request management with overload protection."""
    
    def __init__(self,
                 max_queue_size: int = 1000,
                 max_workers: int = 10,
                 request_timeout: float = 60.0):
        """Initialize request queue.
        
        Args:
            max_queue_size: Maximum queued requests
            max_workers: Number of worker tasks
            request_timeout: Timeout for queued requests
        """
        self.max_queue_size = max_queue_size
        self.max_workers = max_workers
        self.request_timeout = request_timeout
        
        # Priority queue implementation
        self.queues = {
            priority: asyncio.Queue(maxsize=max_queue_size // 4)
            for priority in RequestPriority
        }
        
        self.workers = []
        self.running = False
        self.pending_requests: Dict[str, QueuedRequest] = {}
        self._lock = asyncio.Lock()
        self._request_counter = 0
        
    async def start(self, process_func):
        """Start worker tasks.
        
        Args:
            process_func: Async function to process requests
        """
        self.running = True
        self.process_func = process_func
        
        # Start workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
            
        logger.info(f"Started {self.max_workers} request workers")
        
    async def stop(self):
        """Stop all workers gracefully."""
        self.running = False
        
        # Cancel all pending requests
        async with self._lock:
            for req in self.pending_requests.values():
                req.future.cancel()
            self.pending_requests.clear()
            
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("Request queue stopped")
        
    async def submit_request(self, 
                           request: Any,
                           priority: RequestPriority = RequestPriority.NORMAL) -> Any:
        """Submit a request to the queue.
        
        Args:
            request: The request to process
            priority: Request priority
            
        Returns:
            Result from processing the request
            
        Raises:
            Exception: If queue is full or request times out
        """
        queue = self.queues[priority]
        
        if queue.full():
            raise Exception(
                f"Server overloaded. Queue full for priority {priority.name}. "
                f"Current size: {queue.qsize()}/{queue.maxsize}"
            )
            
        # Create request wrapper
        async with self._lock:
            self._request_counter += 1
            request_id = f"req-{self._request_counter}"
            
        future = asyncio.Future()
        queued_request = QueuedRequest(
            request_id=request_id,
            priority=priority,
            timestamp=time.time(),
            future=future
        )
        
        # Add to pending
        async with self._lock:
            self.pending_requests[request_id] = queued_request
            
        # Queue the request
        await queue.put((request, queued_request))
        
        try:
            # Wait for result with timeout
            result = await asyncio.wait_for(future, timeout=self.request_timeout)
            return result
            
        except asyncio.TimeoutError:
            async with self._lock:
                self.pending_requests.pop(request_id, None)
            raise TimeoutError(
                f"Request {request_id} timed out after {self.request_timeout}s"
            )
        except asyncio.CancelledError:
            async with self._lock:
                self.pending_requests.pop(request_id, None)
            raise
            
    async def _worker(self, worker_id: str):
        """Worker task to process requests."""
        logger.debug(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Check queues in priority order
                request_found = False
                
                for priority in reversed(list(RequestPriority)):
                    queue = self.queues[priority]
                    
                    if not queue.empty():
                        try:
                            request, queued_request = await asyncio.wait_for(
                                queue.get(), 
                                timeout=0.1
                            )
                            request_found = True
                            
                            # Check if request is still valid
                            age = time.time() - queued_request.timestamp
                            if age > self.request_timeout:
                                logger.warning(
                                    f"Dropping expired request {queued_request.request_id} "
                                    f"(age: {age:.1f}s)"
                                )
                                queued_request.future.set_exception(
                                    TimeoutError("Request expired while queued")
                                )
                                continue
                                
                            # Process request
                            try:
                                result = await self.process_func(request)
                                queued_request.future.set_result(result)
                            except Exception as e:
                                queued_request.future.set_exception(e)
                            finally:
                                async with self._lock:
                                    self.pending_requests.pop(queued_request.request_id, None)
                                    
                            break
                            
                        except asyncio.TimeoutError:
                            continue
                            
                if not request_found:
                    # No requests, sleep briefly
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Back off on error
                
        logger.debug(f"Worker {worker_id} stopped")
        
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics.
        
        Returns:
            Dictionary of queue statistics
        """
        stats = {
            "total_pending": len(self.pending_requests),
            "max_queue_size": self.max_queue_size,
            "workers": self.max_workers,
            "queues": {}
        }
        
        for priority in RequestPriority:
            queue = self.queues[priority]
            stats["queues"][priority.name] = {
                "size": queue.qsize(),
                "maxsize": queue.maxsize,
                "full": queue.full()
            }
            
        return stats