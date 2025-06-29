"""Performance monitoring and operation counting utilities."""

import json
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class PerformanceMonitor:
    """Monitor performance metrics for graph operations."""

    def __init__(self, max_history: int = 1000):
        """Initialize performance monitor.

        Args:
            max_history: Maximum number of operations to keep in history
        """
        self.max_history = max_history
        self.operations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.start_time = time.time()

    def record_operation(self, operation: str, duration: float, metadata: Optional[Dict] = None):
        """Record an operation's performance metrics.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
            metadata: Optional additional metadata
        """
        record = {
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "duration_ms": round(duration * 1000, 2),
            "metadata": metadata or {}
        }
        self.operations[operation].append(record)

    def get_statistics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics.

        Args:
            operation: Specific operation to get stats for, or None for all

        Returns:
            Dictionary of performance statistics
        """
        if operation:
            if operation not in self.operations:
                return {"error": f"No data for operation '{operation}'"}

            durations = [op["duration_ms"] for op in self.operations[operation]]
            if not durations:
                return {"error": "No operations recorded"}

            return {
                "operation": operation,
                "count": len(durations),
                "mean_ms": round(sum(durations) / len(durations), 2),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "total_ms": sum(durations),
                "recent_operations": list(self.operations[operation])[-10:]
            }
        else:
            # Stats for all operations
            stats = {}
            for op_name, op_list in self.operations.items():
                if op_list:
                    durations = [op["duration_ms"] for op in op_list]
                    stats[op_name] = {
                        "count": len(durations),
                        "mean_ms": round(sum(durations) / len(durations), 2),
                        "total_ms": sum(durations)
                    }

            return {
                "uptime_seconds": round(time.time() - self.start_time, 2),
                "operations": stats,
                "total_operations": sum(len(ops) for ops in self.operations.values())
            }

    def get_slow_operations(self, threshold_ms: float = 1000) -> List[Dict[str, Any]]:
        """Get operations that exceeded a duration threshold.

        Args:
            threshold_ms: Threshold in milliseconds

        Returns:
            List of slow operations
        """
        slow_ops = []
        for op_name, op_list in self.operations.items():
            for op in op_list:
                if op["duration_ms"] > threshold_ms:
                    slow_ops.append({
                        "operation": op_name,
                        "timestamp": op["timestamp"],
                        "duration_ms": op["duration_ms"],
                        "metadata": op["metadata"]
                    })

        return sorted(slow_ops, key=lambda x: x["duration_ms"], reverse=True)

    def clear_history(self, operation: Optional[str] = None):
        """Clear performance history.

        Args:
            operation: Specific operation to clear, or None for all
        """
        if operation:
            if operation in self.operations:
                self.operations[operation].clear()
        else:
            self.operations.clear()


class OperationCounter:
    """Count operations and track usage patterns."""

    def __init__(self):
        """Initialize operation counter."""
        self.counts: Dict[str, int] = defaultdict(int)
        self.hourly_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.start_time = datetime.now(timezone.utc).replace(tzinfo=None)
        self.errors: Dict[str, int] = defaultdict(int)

    def increment(self, operation: str, count: int = 1):
        """Increment operation count.

        Args:
            operation: Name of the operation
            count: Number to increment by
        """
        self.counts[operation] += count
        current_hour = datetime.now(timezone.utc).replace(tzinfo=None).hour
        self.hourly_counts[operation][current_hour] += count

    def increment_error(self, operation: str, error_type: str):
        """Increment error count for an operation.

        Args:
            operation: Name of the operation
            error_type: Type of error
        """
        error_key = f"{operation}:{error_type}"
        self.errors[error_key] += 1

    def get_counts(self) -> Dict[str, Any]:
        """Get operation counts and statistics.

        Returns:
            Dictionary of operation counts and statistics
        """
        total_operations = sum(self.counts.values())
        total_errors = sum(self.errors.values())

        # Calculate operation distribution
        operation_distribution = {}
        if total_operations > 0:
            for op, count in self.counts.items():
                operation_distribution[op] = {
                    "count": count,
                    "percentage": round((count / total_operations) * 100, 2)
                }

        # Calculate hourly patterns
        hourly_patterns = {}
        for op, hours in self.hourly_counts.items():
            hourly_patterns[op] = dict(hours)

        # Error rate by operation
        error_rates = {}
        for error_key, count in self.errors.items():
            operation = error_key.split(":")[0]
            if operation not in error_rates:
                error_rates[operation] = {"errors": 0, "total": self.counts.get(operation, 0)}
            error_rates[operation]["errors"] += count

        for op, data in error_rates.items():
            if data["total"] > 0:
                data["error_rate"] = round((data["errors"] / data["total"]) * 100, 2)
            else:
                data["error_rate"] = 0

        return {
            "total_operations": total_operations,
            "total_errors": total_errors,
            "error_rate": round((total_errors / total_operations * 100), 2) if total_operations > 0 else 0,
            "uptime": str(datetime.now(timezone.utc).replace(tzinfo=None) - self.start_time),
            "operation_counts": dict(self.counts),
            "operation_distribution": operation_distribution,
            "hourly_patterns": hourly_patterns,
            "error_rates": error_rates,
            "top_operations": sorted(
                [(op, count) for op, count in self.counts.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def reset(self):
        """Reset all counters."""
        self.counts.clear()
        self.hourly_counts.clear()
        self.errors.clear()
        self.start_time = datetime.now(timezone.utc).replace(tzinfo=None)

    def export_stats(self, filepath: str):
        """Export statistics to a JSON file.

        Args:
            filepath: Path to save the statistics
        """
        stats = self.get_counts()
        stats["exported_at"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)


class MemoryMonitor:
    """Monitor memory usage of graphs."""

    @staticmethod
    def estimate_graph_memory(graph) -> Dict[str, Any]:
        """Estimate memory usage of a graph.

        Args:
            graph: NetworkX graph object

        Returns:
            Dictionary with memory estimates
        """
        import sys

        # Base object sizes
        node_size = 100  # Estimated bytes per node
        edge_size = 50   # Estimated bytes per edge

        # Calculate attribute sizes
        node_attr_size = 0
        for _node, attrs in graph.nodes(data=True):
            node_attr_size += sys.getsizeof(attrs)

        edge_attr_size = 0
        for _u, _v, attrs in graph.edges(data=True):
            edge_attr_size += sys.getsizeof(attrs)

        # Total estimates
        total_bytes = (
            graph.number_of_nodes() * node_size +
            graph.number_of_edges() * edge_size +
            node_attr_size +
            edge_attr_size
        )

        return {
            "total_bytes": total_bytes,
            "total_kb": round(total_bytes / 1024, 2),
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "node_memory_kb": round((graph.number_of_nodes() * node_size + node_attr_size) / 1024, 2),
            "edge_memory_kb": round((graph.number_of_edges() * edge_size + edge_attr_size) / 1024, 2),
            "avg_node_size_bytes": round((node_size + node_attr_size / graph.number_of_nodes()), 2) if graph.number_of_nodes() > 0 else 0,
            "avg_edge_size_bytes": round((edge_size + edge_attr_size / graph.number_of_edges()), 2) if graph.number_of_edges() > 0 else 0
        }
