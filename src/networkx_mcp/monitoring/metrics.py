"""Metrics collection and export for OpenTelemetry integration."""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from ..core.base import Component

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """A single metric value with timestamp."""

    value: int | float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """A time series of metric values."""

    name: str
    metric_type: str  # counter, gauge, histogram, summary
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector(Component):
    """Metrics collection service with OpenTelemetry integration."""

    def __init__(self, max_series: int = 10000):
        super().__init__("metrics_collector")
        self.max_series = max_series
        self.metrics: dict[str, MetricSeries] = {}
        self._lock = threading.Lock()

        # Built-in counters
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)

    def increment_counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric."""
        labels = labels or {}
        metric_key = self._get_metric_key(name, labels)

        with self._lock:
            self._counters[metric_key] += value
            self._record_metric(name, "counter", value, labels)

    def set_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Set a gauge metric value."""
        labels = labels or {}
        metric_key = self._get_metric_key(name, labels)

        with self._lock:
            self._gauges[metric_key] = value
            self._record_metric(name, "gauge", value, labels)

    def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a histogram observation."""
        labels = labels or {}
        metric_key = self._get_metric_key(name, labels)

        with self._lock:
            self._histograms[metric_key].append(value)
            # Keep only last 1000 observations
            if len(self._histograms[metric_key]) > 1000:
                self._histograms[metric_key] = self._histograms[metric_key][-1000:]

            self._record_metric(name, "histogram", value, labels)

    def time_operation(self, name: str, labels: dict[str, str] | None = None):
        """Context manager for timing operations."""
        return TimedOperation(self, name, labels)

    def _get_metric_key(self, name: str, labels: dict[str, str]) -> str:
        """Generate a unique key for a metric with labels."""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _record_metric(
        self, name: str, metric_type: str, value: float, labels: dict[str, str]
    ) -> None:
        """Record a metric value in the time series."""
        metric_key = self._get_metric_key(name, labels)

        if metric_key not in self.metrics:
            if len(self.metrics) >= self.max_series:
                logger.warning(f"Max metric series limit reached: {self.max_series}")
                return

            self.metrics[metric_key] = MetricSeries(
                name=name, metric_type=metric_type, labels=labels.copy()
            )

        metric_value = MetricValue(value=value, labels=labels.copy())
        self.metrics[metric_key].values.append(metric_value)

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get current counter value."""
        metric_key = self._get_metric_key(name, labels or {})
        with self._lock:
            return self._counters.get(metric_key, 0.0)

    def get_gauge(
        self, name: str, labels: dict[str, str] | None = None
    ) -> float | None:
        """Get current gauge value."""
        metric_key = self._get_metric_key(name, labels or {})
        with self._lock:
            return self._gauges.get(metric_key)

    def get_histogram_stats(
        self, name: str, labels: dict[str, str] | None = None
    ) -> dict[str, float]:
        """Get histogram statistics."""
        metric_key = self._get_metric_key(name, labels or {})

        with self._lock:
            values = self._histograms.get(metric_key, [])

            if not values:
                return {}

            sorted_values = sorted(values)
            count = len(values)

            stats = {
                "count": count,
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / count,
                "p50": sorted_values[int(count * 0.5)],
                "p90": sorted_values[int(count * 0.9)],
                "p95": sorted_values[int(count * 0.95)],
                "p99": sorted_values[int(count * 0.99)],
            }

            return stats

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all current metric values."""
        with self._lock:
            result = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: self.get_histogram_stats(name.split("{")[0], {})
                    for name in self._histograms.keys()
                },
            }

            return result

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self.metrics.clear()
            logger.info("All metrics reset")

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        with self._lock:
            # Export counters
            for metric_key, value in self._counters.items():
                lines.append(f"# TYPE {metric_key.split('{')[0]} counter")
                lines.append(f"{metric_key} {value}")

            # Export gauges
            for metric_key, value in self._gauges.items():
                lines.append(f"# TYPE {metric_key.split('{')[0]} gauge")
                lines.append(f"{metric_key} {value}")

            # Export histograms
            for metric_key, values in self._histograms.items():
                if not values:
                    continue

                base_name = metric_key.split("{")[0]
                stats = self.get_histogram_stats(base_name, {})

                lines.append(f"# TYPE {base_name} histogram")
                lines.append(f"{metric_key}_count {stats.get('count', 0)}")
                lines.append(f"{metric_key}_sum {stats.get('sum', 0)}")

                # Add percentile buckets
                for percentile in [50, 90, 95, 99]:
                    p_key = f"p{percentile}"
                    if p_key in stats:
                        lines.append(
                            f"{metric_key}_bucket{{le=\"{stats[p_key]}\"}} {stats['count']}"
                        )

        return "\n".join(lines)


class TimedOperation:
    """Context manager for timing operations."""

    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        labels: dict[str, str] | None = None,
    ):
        self.collector = collector
        self.name = name
        self.labels = labels or {}
        self.start_time: float | None = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.observe_histogram(
                f"{self.name}_duration_seconds", duration, self.labels
            )

            # Track success/failure
            status_labels = self.labels.copy()
            status_labels["status"] = "error" if exc_type else "success"
            self.collector.increment_counter(f"{self.name}_total", 1.0, status_labels)


class MetricsExporter(Component):
    """Service for exporting metrics to external systems."""

    def __init__(self, collector: MetricsCollector):
        super().__init__("metrics_exporter")
        self.collector = collector

    async def export_to_opentelemetry(self) -> bool:
        """Export metrics to OpenTelemetry collector."""
        try:
            # This would integrate with OpenTelemetry SDK
            # For now, just log the metrics
            metrics = self.collector.get_all_metrics()
            logger.info(f"Exporting {len(metrics)} metric series to OpenTelemetry")
            return True

        except Exception as e:
            logger.error(f"Failed to export metrics to OpenTelemetry: {e}")
            return False

    async def export_to_prometheus(self, endpoint: str) -> bool:
        """Export metrics in Prometheus format."""
        try:
            prometheus_data = self.collector.export_prometheus_format()
            logger.info(f"Exported {len(prometheus_data)} bytes to Prometheus format")
            # In real implementation, would POST to Prometheus pushgateway
            return True

        except Exception as e:
            logger.error(f"Failed to export metrics to Prometheus: {e}")
            return False

    async def export_metrics(self, format_type: str = "opentelemetry") -> bool:
        """Export metrics in specified format."""
        if format_type == "opentelemetry":
            return await self.export_to_opentelemetry()
        elif format_type == "prometheus":
            return await self.export_to_prometheus("/metrics")
        else:
            logger.error(f"Unsupported export format: {format_type}")
            return False
