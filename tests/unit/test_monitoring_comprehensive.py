"""Comprehensive tests for monitoring.py module."""

import json
import os
import threading
import time
import unittest
from http.client import HTTPConnection
from unittest.mock import Mock, patch

from networkx_mcp.monitoring_legacy import HealthMonitor, create_health_endpoint


class TestHealthMonitor(unittest.TestCase):
    """Test HealthMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = HealthMonitor()

    def test_init(self):
        """Test HealthMonitor initialization."""
        assert hasattr(self.monitor, "start_time")
        assert isinstance(self.monitor.start_time, float)
        assert self.monitor.request_count == 0
        assert self.monitor.error_count == 0
        assert isinstance(self.monitor.tool_usage, dict)
        assert hasattr(self.monitor, "process")

    def test_record_request_success(self):
        """Test recording successful requests."""
        initial_count = self.monitor.request_count
        initial_errors = self.monitor.error_count

        self.monitor.record_request("test_method", success=True)

        assert self.monitor.request_count == initial_count + 1
        assert self.monitor.error_count == initial_errors

    def test_record_request_failure(self):
        """Test recording failed requests."""
        initial_count = self.monitor.request_count
        initial_errors = self.monitor.error_count

        self.monitor.record_request("test_method", success=False)

        assert self.monitor.request_count == initial_count + 1
        assert self.monitor.error_count == initial_errors + 1

    def test_record_request_default_success(self):
        """Test that success defaults to True."""
        initial_errors = self.monitor.error_count

        self.monitor.record_request("test_method")  # No success parameter

        assert self.monitor.error_count == initial_errors

    def test_record_tool_usage(self):
        """Test recording tool usage."""
        assert "tools/call" not in self.monitor.tool_usage

        self.monitor.record_request("tools/call")
        assert self.monitor.tool_usage["tools/call"] == 1

        self.monitor.record_request("tools/call")
        assert self.monitor.tool_usage["tools/call"] == 2

    def test_record_non_tool_request(self):
        """Test that non-tool requests don't affect tool usage."""
        self.monitor.record_request("other_method")
        assert "other_method" not in self.monitor.tool_usage

    @patch("time.time")
    def test_get_health_status_structure(self, mock_time):
        """Test get_health_status returns correct structure."""
        # Mock time for consistent testing
        mock_time.return_value = 1000.0
        self.monitor.start_time = 900.0  # 100 seconds ago

        status = self.monitor.get_health_status()

        # Check top-level keys
        assert "status" in status
        assert "timestamp" in status
        assert "uptime_seconds" in status
        assert "uptime_human" in status
        assert "metrics" in status
        assert "version" in status

        # Check metrics structure
        metrics = status["metrics"]
        assert "requests" in metrics
        assert "system" in metrics
        assert "graphs" in metrics

        # Check requests metrics
        requests = metrics["requests"]
        assert "total" in requests
        assert "errors" in requests
        assert "success_rate" in requests

        # Check system metrics
        system = metrics["system"]
        assert "memory_mb" in system
        assert "cpu_percent" in system
        assert "pid" in system

    def test_get_health_status_uptime(self):
        """Test uptime calculation."""
        # Wait a small amount to ensure uptime > 0
        time.sleep(0.01)

        status = self.monitor.get_health_status()

        assert status["uptime_seconds"] > 0
        assert isinstance(status["uptime_human"], str)

    def test_get_health_status_success_rate_calculation(self):
        """Test success rate calculation."""
        # Record some requests
        self.monitor.record_request("test1", success=True)
        self.monitor.record_request("test2", success=True)
        self.monitor.record_request("test3", success=False)

        status = self.monitor.get_health_status()
        metrics = status["metrics"]["requests"]

        assert metrics["total"] == 3
        assert metrics["errors"] == 1
        # Success rate should be (3-1)/3 * 100 = 66.67%
        assert abs(metrics["success_rate"] - 66.67) < 0.01

    def test_get_health_status_zero_requests(self):
        """Test success rate with zero requests."""
        status = self.monitor.get_health_status()
        metrics = status["metrics"]["requests"]

        assert metrics["total"] == 0
        assert metrics["errors"] == 0
        assert metrics["success_rate"] == 0.0

    def test_format_uptime_days(self):
        """Test uptime formatting with days."""
        # 2 days, 3 hours, 30 minutes
        seconds = 2 * 86400 + 3 * 3600 + 30 * 60
        result = self.monitor._format_uptime(seconds)
        assert result == "2d 3h 30m"

    def test_format_uptime_hours(self):
        """Test uptime formatting with hours."""
        # 5 hours, 45 minutes
        seconds = 5 * 3600 + 45 * 60
        result = self.monitor._format_uptime(seconds)
        assert result == "5h 45m"

    def test_format_uptime_minutes(self):
        """Test uptime formatting with only minutes."""
        seconds = 30 * 60  # 30 minutes
        result = self.monitor._format_uptime(seconds)
        assert result == "30m"

    def test_format_uptime_less_than_minute(self):
        """Test uptime formatting with less than a minute."""
        seconds = 30  # 30 seconds
        result = self.monitor._format_uptime(seconds)
        assert result == "< 1m"

    def test_format_uptime_exact_day(self):
        """Test uptime formatting with exact day."""
        seconds = 86400  # Exactly 1 day
        result = self.monitor._format_uptime(seconds)
        assert result == "1d"

    def test_graphs_metrics_no_graphs(self):
        """Test graphs metrics when no graphs attribute."""
        status = self.monitor.get_health_status()
        graphs = status["metrics"]["graphs"]

        assert graphs["count"] == 0
        assert graphs["total_nodes"] == 0
        assert graphs["total_edges"] == 0

    def test_graphs_metrics_with_graphs(self):
        """Test graphs metrics with actual graphs."""
        # Mock some graphs
        mock_graph1 = Mock()
        mock_graph1.number_of_nodes.return_value = 5
        mock_graph1.number_of_edges.return_value = 8

        mock_graph2 = Mock()
        mock_graph2.number_of_nodes.return_value = 3
        mock_graph2.number_of_edges.return_value = 2

        self.monitor.graphs = {"graph1": mock_graph1, "graph2": mock_graph2}

        status = self.monitor.get_health_status()
        graphs = status["metrics"]["graphs"]

        assert graphs["count"] == 2
        assert graphs["total_nodes"] == 8
        assert graphs["total_edges"] == 10

    def test_system_metrics_types(self):
        """Test that system metrics are correct types."""
        status = self.monitor.get_health_status()
        system = status["metrics"]["system"]

        assert isinstance(system["memory_mb"], (int, float))
        assert isinstance(system["cpu_percent"], (int, float))
        assert isinstance(system["pid"], int)
        assert system["pid"] == os.getpid()

    def test_status_always_healthy(self):
        """Test that status is always 'healthy'."""
        status = self.monitor.get_health_status()
        assert status["status"] == "healthy"

    def test_version_present(self):
        """Test that version is present."""
        status = self.monitor.get_health_status()
        assert "version" in status
        assert isinstance(status["version"], str)


class TestHealthEndpoint(unittest.TestCase):
    """Test HTTP health endpoint functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = HealthMonitor()
        # Use a higher port to avoid conflicts
        self.port = 18080

    def tearDown(self):
        """Clean up after tests."""
        # Give time for any server to shut down
        time.sleep(0.1)

    def test_create_health_endpoint(self):
        """Test creating health endpoint server."""
        server = create_health_endpoint(self.monitor, port=self.port)

        # Server should be created
        assert server is not None

        # Should be serving on the correct port
        assert server.server_address[1] == self.port

        # Clean up
        server.shutdown()
        server.server_close()

    def test_health_endpoint_response(self):
        """Test health endpoint HTTP response."""
        server = create_health_endpoint(self.monitor, port=self.port)

        try:
            # Give server time to start
            time.sleep(0.1)

            # Make HTTP request
            conn = HTTPConnection(f"localhost:{self.port}")
            conn.request("GET", "/health")
            response = conn.getresponse()

            # Check response
            assert response.status == 200
            assert response.getheader("Content-Type") == "application/json"

            # Check response body
            data = response.read().decode()
            health_data = json.loads(data)

            # Should contain expected keys
            assert "status" in health_data
            assert "metrics" in health_data
            assert health_data["status"] == "healthy"

            conn.close()

        finally:
            server.shutdown()
            server.server_close()

    def test_health_endpoint_404(self):
        """Test health endpoint returns 404 for other paths."""
        server = create_health_endpoint(self.monitor, port=self.port)

        try:
            # Give server time to start
            time.sleep(0.1)

            # Make HTTP request to non-existent path
            conn = HTTPConnection(f"localhost:{self.port}")
            conn.request("GET", "/nonexistent")
            response = conn.getresponse()

            # Should return 404
            assert response.status == 404

            conn.close()

        finally:
            server.shutdown()
            server.server_close()

    def test_health_handler_log_suppression(self):
        """Test that HealthHandler suppresses log messages."""
        # This is a bit tricky to test directly, but we can at least
        # verify the log_message method exists and returns None
        from networkx_mcp.monitoring import create_health_endpoint

        # Import the handler class indirectly through the function
        server = create_health_endpoint(self.monitor, port=self.port)

        # Get the handler class
        handler_class = server.RequestHandlerClass

        # Create instance (this is a bit hacky but works for testing)
        handler = handler_class.__new__(handler_class)

        # Call log_message - should not raise exception and return None
        result = handler.log_message("test format", "arg1", "arg2")
        assert result is None

        server.shutdown()
        server.server_close()

    def test_server_threading(self):
        """Test that server runs in daemon thread."""
        # This test verifies the server starts without blocking
        start_time = time.time()

        server = create_health_endpoint(self.monitor, port=self.port)

        # Should return immediately (not block)
        elapsed = time.time() - start_time
        assert elapsed < 0.5  # Should be nearly instantaneous

        # Verify server is actually running
        time.sleep(0.1)

        try:
            conn = HTTPConnection(f"localhost:{self.port}")
            conn.request("GET", "/health")
            response = conn.getresponse()
            assert response.status == 200
            conn.close()
        finally:
            server.shutdown()
            server.server_close()


class TestIntegration(unittest.TestCase):
    """Integration tests for monitoring components."""

    def test_monitor_with_endpoint(self):
        """Test monitor and endpoint working together."""
        monitor = HealthMonitor()

        # Record some activity
        monitor.record_request("tools/call", success=True)
        monitor.record_request("other_method", success=False)

        # Create endpoint
        port = 18081
        server = create_health_endpoint(monitor, port=port)

        try:
            time.sleep(0.1)

            # Get health data via HTTP
            conn = HTTPConnection(f"localhost:{port}")
            conn.request("GET", "/health")
            response = conn.getresponse()

            data = json.loads(response.read().decode())

            # Verify the recorded activity is reflected
            assert data["metrics"]["requests"]["total"] == 2
            assert data["metrics"]["requests"]["errors"] == 1
            assert data["metrics"]["requests"]["success_rate"] == 50.0

            conn.close()

        finally:
            server.shutdown()
            server.server_close()

    def test_concurrent_monitoring(self):
        """Test monitoring with concurrent requests."""
        monitor = HealthMonitor()

        def worker():
            for i in range(10):
                monitor.record_request("test_method", success=(i % 3 != 0))

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check final counts
        status = monitor.get_health_status()
        metrics = status["metrics"]["requests"]

        assert metrics["total"] == 30  # 3 threads * 10 requests
        assert metrics["errors"] == 10  # Every 3rd request fails
        assert abs(metrics["success_rate"] - 66.67) < 0.01


if __name__ == "__main__":
    unittest.main()
