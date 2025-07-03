"""Unit tests for monitoring module."""

import logging
import time
from unittest.mock import patch

from networkx_mcp.monitoring.health_checks import (HealthCheck,
                                                   HealthCheckResult,
                                                   HealthStatus,
                                                   SystemHealthMonitor)
from networkx_mcp.monitoring.logging import (ContextualLogger, LogContext,
                                             StructuredLogger, setup_logging)
from networkx_mcp.monitoring.metrics import (Counter, Gauge, Histogram,
                                             MetricsCollector)


class TestHealthChecks:
    """Test health check functionality."""

    def test_health_status_enum(self):
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    def test_health_check_result(self):
        """Test HealthCheckResult dataclass."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="All systems operational",
            details={"uptime": 3600},
        )
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All systems operational"
        assert result.details["uptime"] == 3600

    def test_health_check_registration(self):
        """Test health check registration."""
        health_check = HealthCheck()

        # Register a check
        def test_check():
            return HealthCheckResult(HealthStatus.HEALTHY, "OK")

        health_check.register_check("test", test_check)
        assert "test" in health_check._checks

    def test_run_health_checks(self):
        """Test running health checks."""
        health_check = HealthCheck()

        # Register multiple checks
        health_check.register_check(
            "good", lambda: HealthCheckResult(HealthStatus.HEALTHY, "Good")
        )
        health_check.register_check(
            "bad", lambda: HealthCheckResult(HealthStatus.UNHEALTHY, "Bad")
        )

        results = health_check.run_all_checks()
        assert "good" in results
        assert "bad" in results
        assert results["good"].status == HealthStatus.HEALTHY
        assert results["bad"].status == HealthStatus.UNHEALTHY

    def test_system_health_monitor(self):
        """Test SystemHealthMonitor."""
        monitor = SystemHealthMonitor()

        # Test memory check
        memory_result = monitor.check_memory()
        assert isinstance(memory_result, HealthCheckResult)
        assert memory_result.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]
        assert "memory_percent" in memory_result.details

        # Test CPU check
        cpu_result = monitor.check_cpu()
        assert isinstance(cpu_result, HealthCheckResult)
        assert "cpu_percent" in cpu_result.details

        # Test disk check
        disk_result = monitor.check_disk()
        assert isinstance(disk_result, HealthCheckResult)
        assert "disk_percent" in disk_result.details


class TestMetrics:
    """Test metrics collection."""

    def test_counter(self):
        """Test Counter metric."""
        counter = Counter("test_counter", "Test counter metric")

        # Test increment
        counter.inc()
        assert counter.value == 1

        counter.inc(5)
        assert counter.value == 6

        # Test labels
        counter.inc(labels={"status": "success"})
        assert counter._values[("status",)] == {"status": "success", "value": 1}

    def test_histogram(self):
        """Test Histogram metric."""
        histogram = Histogram("test_histogram", "Test histogram metric")

        # Test observations
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for v in values:
            histogram.observe(v)

        assert histogram.count == 5
        assert histogram.sum == sum(values)
        assert histogram._values == values

    def test_gauge(self):
        """Test Gauge metric."""
        gauge = Gauge("test_gauge", "Test gauge metric")

        # Test set
        gauge.set(42)
        assert gauge.value == 42

        # Test inc/dec
        gauge.inc(8)
        assert gauge.value == 50

        gauge.dec(10)
        assert gauge.value == 40

        # Test labels
        gauge.set(100, labels={"type": "memory"})
        assert gauge._values[("type",)] == {"type": "memory", "value": 100}

    def test_metrics_collector(self):
        """Test MetricsCollector."""
        collector = MetricsCollector()

        # Register metrics
        counter = collector.counter("requests", "Request count")
        histogram = collector.histogram("latency", "Request latency")
        gauge = collector.gauge("connections", "Active connections")

        # Use metrics
        counter.inc()
        histogram.observe(0.05)
        gauge.set(10)

        # Collect metrics
        metrics = collector.collect()
        assert "requests" in metrics
        assert "latency" in metrics
        assert "connections" in metrics

        assert metrics["requests"]["value"] == 1
        assert metrics["latency"]["count"] == 1
        assert metrics["connections"]["value"] == 10


class TestLogging:
    """Test structured logging."""

    def test_log_context(self):
        """Test LogContext."""
        context = LogContext()

        # Add context
        context.add("request_id", "123")
        context.add("user_id", "456")

        ctx = context.get()
        assert ctx["request_id"] == "123"
        assert ctx["user_id"] == "456"

        # Clear context
        context.clear()
        assert context.get() == {}

    def test_structured_logger(self):
        """Test StructuredLogger."""
        with patch("logging.Logger.info") as mock_info:
            logger = StructuredLogger("test")

            # Test basic logging
            logger.info("Test message", extra={"key": "value"})
            mock_info.assert_called_once()

            # Check that extra data was passed
            call_args = mock_info.call_args
            assert "extra" in call_args.kwargs

    def test_contextual_logger(self):
        """Test ContextualLogger with context."""
        context = LogContext()
        context.add("app", "test")

        with patch("logging.Logger.info") as mock_info:
            logger = ContextualLogger("test", context)
            logger.info("Test message")

            # Check context was included
            call_args = mock_info.call_args
            extra = call_args.kwargs.get("extra", {})
            assert extra.get("app") == "test"

    def test_setup_logging(self):
        """Test logging setup."""
        # Test default setup
        setup_logging()

        # Test with custom level
        setup_logging(level="DEBUG")

        # Test with custom format
        setup_logging(format="json")

        # Verify root logger was configured
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0


class TestMonitoringIntegration:
    """Test monitoring components integration."""

    def test_metrics_with_health_checks(self):
        """Test metrics collection during health checks."""
        metrics = MetricsCollector()
        health_counter = metrics.counter("health_checks", "Health check runs")

        health = HealthCheck()

        def monitored_check():
            health_counter.inc()
            return HealthCheckResult(HealthStatus.HEALTHY, "OK")

        health.register_check("monitored", monitored_check)
        health.run_all_checks()

        collected = metrics.collect()
        assert collected["health_checks"]["value"] == 1

    @patch("time.time")
    def test_histogram_timing(self, mock_time):
        """Test histogram for timing measurements."""
        metrics = MetricsCollector()
        latency = metrics.histogram("operation_latency", "Operation latency")

        # Simulate timing
        mock_time.side_effect = [1.0, 1.5]  # 0.5 second operation

        start = time.time()
        # Simulate some operation
        end = time.time()

        latency.observe(end - start)

        collected = metrics.collect()
        assert collected["operation_latency"]["count"] == 1
        assert collected["operation_latency"]["sum"] == 0.5
