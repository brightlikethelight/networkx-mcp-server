"""Performance optimization utilities."""

import time
from collections.abc import Callable
from functools import wraps
from typing import Any

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None


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

    def get_memory_usage(self) -> dict[str, Any]:
        """Get current memory usage."""
        if not HAS_PSUTIL:
            # Return dummy values when psutil is not available
            return {
                "rss_mb": 0.0,
                "vms_mb": 0.0,
                "percent": 0.0,
                "error": "psutil not available",
            }
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }

    def get_performance_summary(self) -> dict[str, Any]:
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
