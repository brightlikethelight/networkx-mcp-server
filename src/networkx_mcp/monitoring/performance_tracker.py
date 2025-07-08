"""Advanced performance metrics collection and tracking."""

import functools
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance measurement."""
    operation: str
    duration_ms: float
    timestamp: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationStats:
    """Statistics for a specific operation."""
    total_calls: int = 0
    total_duration_ms: float = 0.0
    success_count: int = 0
    error_count: int = 0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    recent_calls: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def avg_duration_ms(self) -> float:
        """Average duration in milliseconds."""
        return self.total_duration_ms / max(self.total_calls, 1)
    
    @property
    def success_rate(self) -> float:
        """Success rate percentage."""
        return (self.success_count / max(self.total_calls, 1)) * 100
    
    @property
    def error_rate(self) -> float:
        """Error rate percentage."""
        return (self.error_count / max(self.total_calls, 1)) * 100


class PerformanceTracker:
    """Advanced performance tracking and metrics collection."""
    
    def __init__(self, max_operations: int = 10000):
        self.max_operations = max_operations
        self.stats: Dict[str, OperationStats] = defaultdict(OperationStats)
        self._lock = threading.RLock()
        
        # Real-time metrics
        self.current_operations: Dict[str, int] = defaultdict(int)
        self.hourly_stats: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_data, daemon=True)
        self._cleanup_thread.start()
    
    def track_operation(self, operation_name: str, duration_ms: float, 
                       success: bool = True, metadata: Optional[Dict[str, Any]] = None):
        """Record a completed operation."""
        with self._lock:
            metric = PerformanceMetric(
                operation=operation_name,
                duration_ms=duration_ms,
                timestamp=time.time(),
                success=success,
                metadata=metadata or {}
            )
            
            # Update operation stats
            stats = self.stats[operation_name]
            stats.total_calls += 1
            stats.total_duration_ms += duration_ms
            
            if success:
                stats.success_count += 1
            else:
                stats.error_count += 1
            
            stats.min_duration_ms = min(stats.min_duration_ms, duration_ms)
            stats.max_duration_ms = max(stats.max_duration_ms, duration_ms)
            stats.recent_calls.append(metric)
            
            # Store in hourly stats
            hour_key = datetime.fromtimestamp(metric.timestamp).strftime("%Y-%m-%d-%H")
            self.hourly_stats[hour_key].append(metric)
    
    def start_operation(self, operation_name: str) -> str:
        """Start tracking an operation, returns operation ID."""
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        with self._lock:
            self.current_operations[operation_id] = time.time()
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, 
                     metadata: Optional[Dict[str, Any]] = None):
        """End tracking an operation."""
        with self._lock:
            if operation_id in self.current_operations:
                start_time = self.current_operations.pop(operation_id)
                duration_ms = (time.time() - start_time) * 1000
                
                # Extract operation name from ID
                operation_name = operation_id.rsplit('_', 1)[0]
                
                self.track_operation(operation_name, duration_ms, success, metadata)
    
    def get_operation_stats(self, operation_name: str) -> Optional[OperationStats]:
        """Get statistics for a specific operation."""
        with self._lock:
            return self.stats.get(operation_name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all operation statistics."""
        with self._lock:
            result = {}
            for name, stats in self.stats.items():
                result[name] = {
                    "total_calls": stats.total_calls,
                    "avg_duration_ms": round(stats.avg_duration_ms, 2),
                    "min_duration_ms": round(stats.min_duration_ms, 2) if stats.min_duration_ms != float('inf') else 0,
                    "max_duration_ms": round(stats.max_duration_ms, 2),
                    "success_rate": round(stats.success_rate, 2),
                    "error_rate": round(stats.error_rate, 2),
                    "recent_avg_ms": self._calculate_recent_average(stats.recent_calls),
                }
            return result
    
    def get_current_load(self) -> Dict[str, Any]:
        """Get current system load metrics."""
        with self._lock:
            active_operations = len(self.current_operations)
            
            # Calculate recent activity (last 5 minutes)
            now = time.time()
            recent_cutoff = now - 300  # 5 minutes
            
            recent_operations = 0
            recent_errors = 0
            
            for stats in self.stats.values():
                for call in stats.recent_calls:
                    if call.timestamp >= recent_cutoff:
                        recent_operations += 1
                        if not call.success:
                            recent_errors += 1
            
            return {
                "active_operations": active_operations,
                "recent_operations_5min": recent_operations,
                "recent_errors_5min": recent_errors,
                "error_rate_5min": (recent_errors / max(recent_operations, 1)) * 100,
                "operations_per_minute": (recent_operations / 5) if recent_operations > 0 else 0
            }
    
    def get_top_operations(self, limit: int = 10, sort_by: str = "total_calls") -> List[Dict[str, Any]]:
        """Get top operations by specified metric."""
        with self._lock:
            all_stats = self.get_all_stats()
            
            # Sort by the specified metric
            if sort_by in ["total_calls", "avg_duration_ms", "error_rate"]:
                sorted_ops = sorted(
                    all_stats.items(),
                    key=lambda x: x[1][sort_by],
                    reverse=True
                )[:limit]
                
                return [
                    {"operation": name, **stats}
                    for name, stats in sorted_ops
                ]
            else:
                return list(all_stats.items())[:limit]
    
    def _calculate_recent_average(self, recent_calls: deque) -> float:
        """Calculate average duration for recent calls."""
        if not recent_calls:
            return 0.0
        
        # Use last 10 calls or all if fewer
        calls_to_analyze = list(recent_calls)[-10:]
        total_duration = sum(call.duration_ms for call in calls_to_analyze)
        return round(total_duration / len(calls_to_analyze), 2)
    
    def _cleanup_old_data(self):
        """Background thread to clean up old data."""
        while True:
            try:
                # Sleep for 1 hour
                time.sleep(3600)
                
                with self._lock:
                    # Remove hourly stats older than 24 hours
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    cutoff_key = cutoff_time.strftime("%Y-%m-%d-%H")
                    
                    keys_to_remove = [
                        key for key in self.hourly_stats.keys()
                        if key < cutoff_key
                    ]
                    
                    for key in keys_to_remove:
                        del self.hourly_stats[key]
                    
                    logger.debug(f"Cleaned up {len(keys_to_remove)} old hourly stat entries")
                    
            except Exception as e:
                logger.error(f"Error in performance tracker cleanup: {e}")


def track_performance(operation_name: Optional[str] = None, include_args: bool = False):
    """Decorator to automatically track function performance."""
    def decorator(func: Callable) -> Callable:
        # Use function name if no operation name provided
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = get_performance_tracker()
            operation_id = tracker.start_operation(op_name)
            
            success = True
            metadata = {}
            
            try:
                # Include arguments in metadata if requested
                if include_args:
                    metadata["arg_count"] = len(args)
                    metadata["kwarg_count"] = len(kwargs)
                    # Safely include some argument info
                    if args:
                        metadata["first_arg_type"] = type(args[0]).__name__
                
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                success = False
                metadata["error"] = str(e)
                metadata["error_type"] = type(e).__name__
                raise
                
            finally:
                tracker.end_operation(operation_id, success, metadata)
        
        return wrapper
    return decorator


# Global performance tracker instance
_performance_tracker: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance."""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker


def get_metrics_collector():
    """Get metrics collector for backward compatibility."""
    # Import here to avoid circular imports
    from .metrics import MetricsCollector
    return MetricsCollector()