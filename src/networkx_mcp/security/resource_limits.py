"""Resource limits module to prevent DoS attacks."""

import asyncio
import functools
import logging
import os
import psutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Configurable resource limits."""
    
    # Memory limits
    max_memory_mb: int = int(os.environ.get("MAX_MEMORY_MB", "1024"))  # 1GB default
    max_graph_size_mb: int = int(os.environ.get("MAX_GRAPH_SIZE_MB", "100"))  # 100MB per graph
    memory_check_threshold: float = 0.8  # Warn at 80% memory usage
    
    # Time limits
    operation_timeout_seconds: int = int(os.environ.get("OPERATION_TIMEOUT", "30"))
    
    # Concurrency limits
    max_concurrent_requests: int = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "10"))
    
    # Graph size limits
    max_nodes_per_graph: int = int(os.environ.get("MAX_NODES_PER_GRAPH", "100000"))
    max_edges_per_graph: int = int(os.environ.get("MAX_EDGES_PER_GRAPH", "1000000"))
    
    # Rate limiting
    requests_per_minute: int = int(os.environ.get("REQUESTS_PER_MINUTE", "60"))
    

# Global instance
LIMITS = ResourceLimits()

# Request tracking
_active_requests = 0
_request_lock = threading.Lock()
_request_times = []
_request_times_lock = threading.Lock()

# Thread pool for CPU-bound operations
_executor = ThreadPoolExecutor(max_workers=LIMITS.max_concurrent_requests)


class ResourceLimitError(Exception):
    """Raised when a resource limit is exceeded."""
    pass


class MemoryLimitError(ResourceLimitError):
    """Raised when memory limit is exceeded."""
    pass


class TimeoutError(ResourceLimitError):
    """Raised when operation timeout is exceeded."""
    pass


class ConcurrencyLimitError(ResourceLimitError):
    """Raised when concurrent request limit is exceeded."""
    pass


class GraphSizeLimitError(ResourceLimitError):
    """Raised when graph size limit is exceeded."""
    pass


class RateLimitError(ResourceLimitError):
    """Raised when rate limit is exceeded."""
    pass


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_available_memory_mb() -> float:
    """Get available system memory in MB."""
    return psutil.virtual_memory().available / (1024 * 1024)


def estimate_graph_size_mb(graph: nx.Graph) -> float:
    """Estimate the memory size of a graph in MB."""
    # Rough estimation based on nodes and edges
    # Each node: ~100 bytes overhead + attributes
    # Each edge: ~200 bytes overhead + attributes
    
    node_size = 100 * graph.number_of_nodes()
    edge_size = 200 * graph.number_of_edges()
    
    # Add attribute sizes
    for node, attrs in graph.nodes(data=True):
        node_size += sys.getsizeof(attrs)
    
    for u, v, attrs in graph.edges(data=True):
        edge_size += sys.getsizeof(attrs)
    
    total_bytes = node_size + edge_size
    return total_bytes / (1024 * 1024)


def check_memory_limit() -> None:
    """Check if memory usage is within limits."""
    current_memory = get_memory_usage_mb()
    
    if current_memory > LIMITS.max_memory_mb:
        raise MemoryLimitError(
            f"Memory usage ({current_memory:.1f}MB) exceeds limit ({LIMITS.max_memory_mb}MB)"
        )
    
    # Check available system memory
    available_memory = get_available_memory_mb()
    if available_memory < 100:  # Less than 100MB available
        raise MemoryLimitError(
            f"System memory critically low ({available_memory:.1f}MB available)"
        )
    
    # Warn if approaching limit
    usage_ratio = current_memory / LIMITS.max_memory_mb
    if usage_ratio > LIMITS.memory_check_threshold:
        logger.warning(
            f"Memory usage high: {current_memory:.1f}MB / {LIMITS.max_memory_mb}MB "
            f"({usage_ratio*100:.1f}%)"
        )


def check_graph_size(graph: nx.Graph) -> None:
    """Check if graph size is within limits."""
    # Check node and edge counts
    if graph.number_of_nodes() > LIMITS.max_nodes_per_graph:
        raise GraphSizeLimitError(
            f"Graph has too many nodes ({graph.number_of_nodes()} > {LIMITS.max_nodes_per_graph})"
        )
    
    if graph.number_of_edges() > LIMITS.max_edges_per_graph:
        raise GraphSizeLimitError(
            f"Graph has too many edges ({graph.number_of_edges()} > {LIMITS.max_edges_per_graph})"
        )
    
    # Estimate memory size
    estimated_size = estimate_graph_size_mb(graph)
    if estimated_size > LIMITS.max_graph_size_mb:
        raise GraphSizeLimitError(
            f"Graph size ({estimated_size:.1f}MB) exceeds limit ({LIMITS.max_graph_size_mb}MB)"
        )


def check_operation_feasibility(graph: nx.Graph, operation: str) -> None:
    """Check if an operation is feasible given the graph size."""
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    
    # Define complexity limits for different operations
    complexity_limits = {
        "shortest_path": 10000,  # O(V + E)
        "all_pairs_shortest_path": 1000,  # O(V^3)
        "betweenness_centrality": 5000,  # O(VE)
        "clustering": 50000,  # O(V)
        "diameter": 1000,  # O(V^3)
        "pagerank": 100000,  # O(V + E)
    }
    
    limit = complexity_limits.get(operation, 10000)
    
    if node_count > limit:
        raise GraphSizeLimitError(
            f"Graph too large for {operation} operation ({node_count} nodes > {limit} limit)"
        )


def check_concurrent_requests() -> None:
    """Check if we can accept more concurrent requests."""
    global _active_requests
    
    with _request_lock:
        if _active_requests >= LIMITS.max_concurrent_requests:
            raise ConcurrencyLimitError(
                f"Too many concurrent requests ({_active_requests} >= {LIMITS.max_concurrent_requests})"
            )
        _active_requests += 1


def release_request_slot() -> None:
    """Release a concurrent request slot."""
    global _active_requests
    
    with _request_lock:
        _active_requests = max(0, _active_requests - 1)


def check_rate_limit() -> None:
    """Check if request rate is within limits."""
    global _request_times
    
    current_time = time.time()
    cutoff_time = current_time - 60  # Last minute
    
    with _request_times_lock:
        # Remove old entries
        _request_times = [t for t in _request_times if t > cutoff_time]
        
        # Check rate
        if len(_request_times) >= LIMITS.requests_per_minute:
            raise RateLimitError(
                f"Rate limit exceeded ({len(_request_times)} requests in last minute)"
            )
        
        # Add current request
        _request_times.append(current_time)


def timeout(seconds: Optional[int] = None):
    """Decorator to add timeout to functions."""
    if seconds is None:
        seconds = LIMITS.operation_timeout_seconds
    
    def decorator(func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            """Wrapper for synchronous functions."""
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=seconds)
            
            if thread.is_alive():
                # Thread is still running, timeout exceeded
                raise TimeoutError(f"Operation timed out after {seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Wrapper for asynchronous functions."""
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_resource_limits(func: Callable) -> Callable:
    """Decorator to enforce resource limits on a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check rate limit
        try:
            check_rate_limit()
        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            return {"error": "Rate limit exceeded. Please try again later."}
        
        # Check concurrent requests
        try:
            check_concurrent_requests()
        except ConcurrencyLimitError as e:
            logger.warning(f"Concurrency limit exceeded: {e}")
            return {"error": "Server busy. Please try again later."}
        
        try:
            # Check memory before operation
            check_memory_limit()
            
            # Execute with timeout
            @timeout()
            def execute():
                return func(*args, **kwargs)
            
            result = execute()
            
            # Check memory after operation
            check_memory_limit()
            
            return result
            
        except MemoryLimitError as e:
            logger.error(f"Memory limit exceeded: {e}")
            return {"error": "Operation would exceed memory limits"}
        except TimeoutError as e:
            logger.error(f"Operation timeout: {e}")
            return {"error": "Operation timed out"}
        except Exception as e:
            logger.error(f"Operation failed: {e}")
            raise
        finally:
            release_request_slot()
    
    return wrapper


def cleanup_if_needed():
    """Perform cleanup if memory usage is high."""
    usage_ratio = get_memory_usage_mb() / LIMITS.max_memory_mb
    
    if usage_ratio > 0.9:  # Above 90%
        logger.warning("Memory usage critical, forcing garbage collection")
        import gc
        gc.collect()
        
        # Log new usage
        new_usage = get_memory_usage_mb()
        logger.info(f"Memory after GC: {new_usage:.1f}MB")


# Resource monitoring thread
def resource_monitor():
    """Background thread to monitor resources."""
    while True:
        try:
            # Check memory periodically
            current_memory = get_memory_usage_mb()
            if current_memory > LIMITS.max_memory_mb * 0.9:
                logger.warning(f"Memory usage critical: {current_memory:.1f}MB")
                cleanup_if_needed()
            
            # Log stats every minute
            with _request_lock:
                active = _active_requests
            
            logger.debug(
                f"Resource stats - Memory: {current_memory:.1f}MB, "
                f"Active requests: {active}"
            )
            
        except Exception as e:
            logger.error(f"Resource monitor error: {e}")
        
        time.sleep(30)  # Check every 30 seconds


# Start monitoring thread
monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
monitor_thread.start()


def get_resource_status() -> dict[str, Any]:
    """Get current resource usage status."""
    with _request_lock:
        active_requests = _active_requests
    
    with _request_times_lock:
        recent_requests = len([t for t in _request_times if t > time.time() - 60])
    
    return {
        "memory": {
            "current_mb": get_memory_usage_mb(),
            "limit_mb": LIMITS.max_memory_mb,
            "available_mb": get_available_memory_mb(),
        },
        "requests": {
            "active": active_requests,
            "max_concurrent": LIMITS.max_concurrent_requests,
            "recent_per_minute": recent_requests,
            "limit_per_minute": LIMITS.requests_per_minute,
        },
        "limits": {
            "max_graph_size_mb": LIMITS.max_graph_size_mb,
            "max_nodes_per_graph": LIMITS.max_nodes_per_graph,
            "max_edges_per_graph": LIMITS.max_edges_per_graph,
            "operation_timeout_seconds": LIMITS.operation_timeout_seconds,
        }
    }