"""Resource management and limits enforcement."""

import asyncio
import gc
import sys
import time
import psutil
import resource
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Callable
from collections import defaultdict
from functools import wraps
import networkx as nx


@dataclass
class ResourceLimits:
    """Configurable resource limits."""
    # Memory limits
    max_memory_mb: int = 1000
    max_graph_memory_mb: int = 100
    
    # CPU limits
    max_cpu_percent: float = 80.0
    
    # Graph size limits
    max_graphs_total: int = 10000
    max_graphs_per_user: int = 100
    max_graph_nodes: int = 1_000_000
    max_graph_edges: int = 10_000_000
    
    # Operation limits
    max_operation_time_s: int = 60
    max_concurrent_operations: int = 100
    
    # Rate limits (per minute)
    max_operations_per_minute: int = 1000
    max_operations_per_user_per_minute: int = 100


class ResourceExhausted(Exception):
    """Base exception for resource exhaustion."""
    pass


class MemoryLimitExceeded(ResourceExhausted):
    """Memory limit exceeded."""
    pass


class GraphTooLarge(ResourceExhausted):
    """Graph exceeds size limits."""
    pass


class OperationTimeout(ResourceExhausted):
    """Operation took too long."""
    pass


class RateLimitExceeded(ResourceExhausted):
    """Rate limit exceeded."""
    pass


class ConcurrencyLimitExceeded(ResourceExhausted):
    """Too many concurrent operations."""
    pass


class ResourceManager:
    """Enforce resource limits and track usage."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.metrics = defaultdict(float)
        self.operation_counts = defaultdict(int)
        self.operation_times = defaultdict(list)
        self._active_operations = 0
        self._operation_lock = asyncio.Lock()
        self._process = psutil.Process()
        self._last_gc = time.time()
        self._rate_limiter = RateLimiter(self.limits)
    
    async def check_memory(self) -> Dict[str, float]:
        """Check if we're within memory limits."""
        # Get current memory usage
        memory_info = self._process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Run GC if needed (every 60 seconds or if close to limit)
        now = time.time()
        if (now - self._last_gc > 60 or 
            memory_mb > self.limits.max_memory_mb * 0.9):
            gc.collect()
            self._last_gc = now
            # Re-measure after GC
            memory_info = self._process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
        
        # Check limit
        if memory_mb > self.limits.max_memory_mb:
            raise MemoryLimitExceeded(
                f"Memory usage {memory_mb:.1f}MB exceeds "
                f"limit {self.limits.max_memory_mb}MB"
            )
        
        # Update metrics
        self.metrics['memory_mb'] = memory_mb
        self.metrics['memory_percent'] = (
            memory_mb / self.limits.max_memory_mb * 100
        )
        
        return {
            'current_mb': memory_mb,
            'limit_mb': self.limits.max_memory_mb,
            'percent': self.metrics['memory_percent']
        }
    
    def check_graph_size(self, graph: nx.Graph) -> Dict[str, Any]:
        """Validate graph size and estimate memory usage."""
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        # Check node limit
        if num_nodes > self.limits.max_graph_nodes:
            raise GraphTooLarge(
                f"Graph has {num_nodes:,} nodes, "
                f"limit is {self.limits.max_graph_nodes:,}"
            )
        
        # Check edge limit
        if num_edges > self.limits.max_graph_edges:
            raise GraphTooLarge(
                f"Graph has {num_edges:,} edges, "
                f"limit is {self.limits.max_graph_edges:,}"
            )
        
        # Estimate memory usage (rough approximation)
        # Base graph structure
        size_bytes = sys.getsizeof(graph)
        
        # Nodes (ID + dict for attributes)
        size_bytes += num_nodes * (sys.getsizeof(0) + sys.getsizeof({}))
        
        # Edges (2 IDs + dict for attributes)
        size_bytes += num_edges * (2 * sys.getsizeof(0) + sys.getsizeof({}))
        
        # Sample actual data sizes
        if num_nodes > 0:
            sample_nodes = list(graph.nodes(data=True))[:100]
            avg_node_size = sum(
                sys.getsizeof(n) + sys.getsizeof(d) 
                for n, d in sample_nodes
            ) / len(sample_nodes)
            size_bytes += int(avg_node_size * num_nodes)
        
        if num_edges > 0:
            sample_edges = list(graph.edges(data=True))[:100]
            avg_edge_size = sum(
                sys.getsizeof(u) + sys.getsizeof(v) + sys.getsizeof(d)
                for u, v, d in sample_edges
            ) / len(sample_edges)
            size_bytes += int(avg_edge_size * num_edges)
        
        size_mb = size_bytes / 1024 / 1024
        
        # Check graph memory limit
        if size_mb > self.limits.max_graph_memory_mb:
            raise GraphTooLarge(
                f"Graph estimated size {size_mb:.1f}MB exceeds "
                f"limit {self.limits.max_graph_memory_mb}MB"
            )
        
        return {
            'nodes': num_nodes,
            'edges': num_edges,
            'estimated_mb': round(size_mb, 2),
            'density': graph.density() if num_nodes > 0 else 0
        }
    
    async def check_cpu(self) -> Dict[str, float]:
        """Check CPU usage."""
        cpu_percent = self._process.cpu_percent(interval=0.1)
        
        if cpu_percent > self.limits.max_cpu_percent:
            # Don't throw exception, just log warning
            self.metrics['cpu_warnings'] += 1
        
        self.metrics['cpu_percent'] = cpu_percent
        
        return {
            'current_percent': cpu_percent,
            'limit_percent': self.limits.max_cpu_percent
        }
    
    async def with_resource_limits(
        self,
        coro: Callable,
        operation: str,
        user_id: Optional[str] = None
    ):
        """Execute coroutine with resource limits enforced."""
        # Check rate limits
        self._rate_limiter.check_rate_limit(operation, user_id)
        
        # Check concurrent operations
        async with self._operation_lock:
            if self._active_operations >= self.limits.max_concurrent_operations:
                raise ConcurrencyLimitExceeded(
                    f"Too many concurrent operations "
                    f"({self._active_operations}/{self.limits.max_concurrent_operations})"
                )
            self._active_operations += 1
        
        start_time = time.time()
        
        try:
            # Check memory before
            await self.check_memory()
            
            # Execute with timeout
            result = await asyncio.wait_for(
                coro,
                timeout=self.limits.max_operation_time_s
            )
            
            # Track success
            elapsed = time.time() - start_time
            self.operation_counts[f'{operation}_success'] += 1
            self.operation_times[operation].append(elapsed)
            
            # Keep only last 100 times for percentile calculations
            if len(self.operation_times[operation]) > 100:
                self.operation_times[operation] = self.operation_times[operation][-100:]
            
            return result
            
        except asyncio.TimeoutError:
            self.operation_counts[f'{operation}_timeout'] += 1
            raise OperationTimeout(
                f"Operation '{operation}' timed out after "
                f"{self.limits.max_operation_time_s}s"
            )
        except Exception as e:
            self.operation_counts[f'{operation}_error'] += 1
            raise
        finally:
            # Decrement active operations
            async with self._operation_lock:
                self._active_operations -= 1
            
            # Check memory after
            try:
                await self.check_memory()
            except MemoryLimitExceeded:
                # Force GC if memory limit exceeded after operation
                gc.collect()
                # Re-check
                await self.check_memory()
    
    def resource_limited(self, operation: str):
        """Decorator for resource-limited operations."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract user_id if available
                user_id = kwargs.get('user_id')
                if not user_id and len(args) > 0:
                    # Assume first arg might be user_id
                    if isinstance(args[0], str) and len(args[0]) < 100:
                        user_id = args[0]
                
                return await self.with_resource_limits(
                    func(*args, **kwargs),
                    operation,
                    user_id
                )
            return wrapper
        return decorator
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics."""
        # Calculate operation percentiles
        percentiles = {}
        for op, times in self.operation_times.items():
            if times:
                times_sorted = sorted(times)
                percentiles[op] = {
                    'p50': times_sorted[len(times_sorted) // 2],
                    'p95': times_sorted[int(len(times_sorted) * 0.95)],
                    'p99': times_sorted[int(len(times_sorted) * 0.99)]
                }
        
        return {
            'memory': {
                'current_mb': self.metrics.get('memory_mb', 0),
                'limit_mb': self.limits.max_memory_mb,
                'percent': self.metrics.get('memory_percent', 0)
            },
            'cpu': {
                'current_percent': self.metrics.get('cpu_percent', 0),
                'limit_percent': self.limits.max_cpu_percent,
                'warnings': self.metrics.get('cpu_warnings', 0)
            },
            'operations': {
                'active': self._active_operations,
                'max_concurrent': self.limits.max_concurrent_operations,
                'counts': dict(self.operation_counts),
                'percentiles': percentiles
            },
            'limits': {
                'max_graph_nodes': self.limits.max_graph_nodes,
                'max_graph_edges': self.limits.max_graph_edges,
                'max_operation_time_s': self.limits.max_operation_time_s
            }
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Health check for resource manager."""
        try:
            memory = await self.check_memory()
            cpu = await self.check_cpu()
            
            # Determine health status
            if memory['percent'] > 90 or cpu['current_percent'] > 90:
                status = 'degraded'
            else:
                status = 'healthy'
            
            return {
                'status': status,
                'memory': memory,
                'cpu': cpu,
                'active_operations': self._active_operations
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.buckets = defaultdict(lambda: {'tokens': 0, 'last_update': time.time()})
        self._lock = asyncio.Lock()
    
    def check_rate_limit(self, operation: str, user_id: Optional[str] = None):
        """Check if operation is within rate limits."""
        now = time.time()
        
        # Global rate limit
        global_key = f"global:{operation}"
        self._check_bucket(
            global_key,
            self.limits.max_operations_per_minute,
            now
        )
        
        # Per-user rate limit
        if user_id:
            user_key = f"user:{user_id}:{operation}"
            self._check_bucket(
                user_key,
                self.limits.max_operations_per_user_per_minute,
                now
            )
    
    def _check_bucket(self, key: str, limit: int, now: float):
        """Check and update token bucket."""
        bucket = self.buckets[key]
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = now - bucket['last_update']
        tokens_to_add = time_elapsed * (limit / 60.0)  # Tokens per second
        
        # Update bucket
        bucket['tokens'] = min(limit, bucket['tokens'] + tokens_to_add)
        bucket['last_update'] = now
        
        # Check if we have tokens
        if bucket['tokens'] < 1:
            raise RateLimitExceeded(
                f"Rate limit exceeded for {key}. "
                f"Limit: {limit} per minute"
            )
        
        # Consume a token
        bucket['tokens'] -= 1


# Global instance
_resource_manager = None


def get_resource_manager(limits: Optional[ResourceLimits] = None) -> ResourceManager:
    """Get or create global resource manager."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager(limits)
    return _resource_manager