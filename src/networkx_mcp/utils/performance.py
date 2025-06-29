"""Performance optimization utilities."""

import time
from functools import wraps
from typing import Any, Callable, Dict

import psutil


class PerformanceMonitor:
    """Monitor and optimize performance."""

    def __init__(self):
        self.metrics = {}

    def time_operation(self, operation_name: str):
        """Decorator to time operations."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                if operation_name not in self.metrics:
                    self.metrics[operation_name] = []
                self.metrics[operation_name].append(duration)

                return result

            return wrapper

        return decorator

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {}
        for operation, times in self.metrics.items():
            summary[operation] = {
                "count": len(times),
                "total_time": sum(times),
                "avg_time": sum(times) / len(times),
                "max_time": max(times),
                "min_time": min(times),
            }
        return summary
