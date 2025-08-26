"""Comprehensive tests for monitoring.py - Target: 100% coverage (122 lines).

This module is critical for production operations and server health monitoring.
Testing requires careful mocking of system dependencies.
"""

import json
import os
from unittest.mock import Mock, patch


class TestHealthMonitor:
    """Test the HealthMonitor class."""

    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_init(self, mock_process_class):
        """Test HealthMonitor initialization."""
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        from networkx_mcp.monitoring_legacy import HealthMonitor

        monitor = HealthMonitor()

        # Check initialization
        assert monitor.request_count == 0
        assert monitor.error_count == 0
        assert monitor.tool_usage == {}
        assert isinstance(monitor.start_time, float)
        assert monitor.process == mock_process

        # Verify psutil.Process was called with current PID
        mock_process_class.assert_called_once_with(os.getpid())

    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_record_request_success(self, mock_process_class):
        """Test recording successful requests."""
        mock_process_class.return_value = Mock()

        from networkx_mcp.monitoring_legacy import HealthMonitor

        monitor = HealthMonitor()

        # Record successful requests
        monitor.record_request("graph/create", success=True)
        monitor.record_request("graph/list", success=True)

        assert monitor.request_count == 2
        assert monitor.error_count == 0

    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_record_request_error(self, mock_process_class):
        """Test recording failed requests."""
        mock_process_class.return_value = Mock()

        from networkx_mcp.monitoring_legacy import HealthMonitor

        monitor = HealthMonitor()

        # Record mixed requests
        monitor.record_request("graph/create", success=True)
        monitor.record_request("graph/invalid", success=False)
        monitor.record_request("graph/delete", success=False)

        assert monitor.request_count == 3
        assert monitor.error_count == 2

    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_record_request_tool_usage(self, mock_process_class):
        """Test tool usage tracking."""
        mock_process_class.return_value = Mock()

        from networkx_mcp.monitoring_legacy import HealthMonitor

        monitor = HealthMonitor()

        # Record tool calls
        monitor.record_request("tools/call", success=True)
        monitor.record_request("tools/call", success=True)
        monitor.record_request("graph/create", success=True)  # Not a tool call

        assert monitor.tool_usage == {"tools/call": 2}
        assert monitor.request_count == 3

    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_record_request_default_success(self, mock_process_class):
        """Test that requests default to successful."""
        mock_process_class.return_value = Mock()

        from networkx_mcp.monitoring_legacy import HealthMonitor

        monitor = HealthMonitor()

        # Don't specify success parameter (should default to True)
        monitor.record_request("graph/create")

        assert monitor.request_count == 1
        assert monitor.error_count == 0

    @patch("networkx_mcp.monitoring_legacy.datetime")
    @patch("networkx_mcp.monitoring_legacy.time")
    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_get_health_status_basic(
        self, mock_process_class, mock_time, mock_datetime
    ):
        """Test basic health status retrieval."""
        # Mock time
        mock_time.time.side_effect = [1000.0, 1100.0]  # start_time, current_time

        # Mock datetime
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

        # Mock process
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.cpu_percent.return_value = 15.5
        mock_process_class.return_value = mock_process

        from networkx_mcp.monitoring_legacy import HealthMonitor

        monitor = HealthMonitor()

        # Record some requests
        monitor.record_request("test", success=True)
        monitor.record_request("test", success=False)

        status = monitor.get_health_status()

        # Check structure
        assert "status" in status
        assert "timestamp" in status
        assert "uptime_seconds" in status
        assert "uptime_human" in status
        assert "metrics" in status
        assert "version" in status

        # Check values
        assert status["status"] == "healthy"
        assert status["timestamp"] == "2024-01-01T12:00:00"
        assert status["uptime_seconds"] == 100.0
        assert status["version"] == "1.0.0"

        # Check metrics structure
        metrics = status["metrics"]
        assert "requests" in metrics
        assert "system" in metrics
        assert "graphs" in metrics

        # Check request metrics
        requests = metrics["requests"]
        assert requests["total"] == 2
        assert requests["errors"] == 1
        assert requests["success_rate"] == 50.0  # 1/2 * 100

        # Check system metrics
        system = metrics["system"]
        assert system["memory_mb"] == 100.0
        assert system["cpu_percent"] == 15.5
        assert system["pid"] == os.getpid()

    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_get_health_status_no_requests(self, mock_process_class):
        """Test health status with no requests (avoid division by zero)."""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 50 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.cpu_percent.return_value = 5.0
        mock_process_class.return_value = mock_process

        from networkx_mcp.monitoring_legacy import HealthMonitor

        monitor = HealthMonitor()
        status = monitor.get_health_status()

        # Should handle division by zero
        requests = status["metrics"]["requests"]
        assert requests["total"] == 0
        assert requests["errors"] == 0
        assert requests["success_rate"] == 0.0  # 0/1 * 100 = 0

    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_get_health_status_with_graphs(self, mock_process_class):
        """Test health status with graph metrics."""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 50 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.cpu_percent.return_value = 0.0
        mock_process_class.return_value = mock_process

        from networkx_mcp.monitoring_legacy import HealthMonitor

        monitor = HealthMonitor()

        # Mock graphs attribute
        mock_graph1 = Mock()
        mock_graph1.number_of_nodes.return_value = 5
        mock_graph1.number_of_edges.return_value = 7

        mock_graph2 = Mock()
        mock_graph2.number_of_nodes.return_value = 3
        mock_graph2.number_of_edges.return_value = 2

        monitor.graphs = {"graph1": mock_graph1, "graph2": mock_graph2}

        status = monitor.get_health_status()

        graphs = status["metrics"]["graphs"]
        assert graphs["count"] == 2
        assert graphs["total_nodes"] == 8  # 5 + 3
        assert graphs["total_edges"] == 9  # 7 + 2

    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_format_uptime_various_periods(self, mock_process_class):
        """Test uptime formatting for various time periods."""
        mock_process_class.return_value = Mock()

        from networkx_mcp.monitoring_legacy import HealthMonitor

        monitor = HealthMonitor()

        # Test various time periods
        assert monitor._format_uptime(30) == "< 1m"  # Less than 1 minute
        assert monitor._format_uptime(60) == "1m"  # Exactly 1 minute
        assert monitor._format_uptime(90) == "1m"  # 1.5 minutes
        assert monitor._format_uptime(3600) == "1h"  # 1 hour
        assert monitor._format_uptime(3660) == "1h 1m"  # 1 hour 1 minute
        assert monitor._format_uptime(86400) == "1d"  # 1 day
        assert monitor._format_uptime(90061) == "1d 1h 1m"  # 1 day 1 hour 1 minute

    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_format_uptime_edge_cases(self, mock_process_class):
        """Test uptime formatting edge cases."""
        mock_process_class.return_value = Mock()

        from networkx_mcp.monitoring_legacy import HealthMonitor

        monitor = HealthMonitor()

        # Edge cases
        assert monitor._format_uptime(0) == "< 1m"
        assert monitor._format_uptime(59) == "< 1m"
        assert monitor._format_uptime(3599) == "59m"  # 59 minutes 59 seconds
        assert monitor._format_uptime(86399) == "23h 59m"  # 23 hours 59 minutes

    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_format_uptime_large_values(self, mock_process_class):
        """Test uptime formatting with large values."""
        mock_process_class.return_value = Mock()

        from networkx_mcp.monitoring_legacy import HealthMonitor

        monitor = HealthMonitor()

        # Large values
        week = 7 * 24 * 3600  # 1 week in seconds
        assert monitor._format_uptime(week) == "7d"
        assert monitor._format_uptime(week + 3661) == "7d 1h 1m"


class TestHealthEndpoint:
    """Test the HTTP health endpoint functionality."""

    def test_create_health_endpoint_import(self):
        """Test that create_health_endpoint can be imported."""
        from networkx_mcp.monitoring_legacy import create_health_endpoint

        assert callable(create_health_endpoint)

    @patch("threading.Thread")
    @patch("http.server.HTTPServer")
    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_create_health_endpoint_basic(
        self, mock_process_class, mock_http_server, mock_thread
    ):
        """Test basic health endpoint creation."""
        from networkx_mcp.monitoring_legacy import HealthMonitor, create_health_endpoint

        mock_process_class.return_value = Mock()
        mock_server = Mock()
        mock_http_server.return_value = mock_server
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        monitor = HealthMonitor()

        result = create_health_endpoint(monitor, port=8080)

        # Verify server creation
        mock_http_server.assert_called_once()
        args, kwargs = mock_http_server.call_args
        assert args[0] == ("", 8080)  # (host, port)

        # Verify thread creation and start
        mock_thread.assert_called_once()
        thread_args = mock_thread.call_args
        assert thread_args[1]["target"] == mock_server.serve_forever
        assert thread_args[1]["daemon"] is True
        mock_thread_instance.start.assert_called_once()

        assert result == mock_server

    @patch("http.server.HTTPServer")
    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_health_handler_health_endpoint(self, mock_process_class, mock_http_server):
        """Test the health endpoint HTTP handler."""
        from networkx_mcp.monitoring_legacy import HealthMonitor, create_health_endpoint

        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.cpu_percent.return_value = 10.0
        mock_process_class.return_value = mock_process

        monitor = HealthMonitor()

        # Get the handler class that would be created
        with patch("threading.Thread"):
            create_health_endpoint(monitor)

        # Get the handler class from the HTTPServer call
        handler_class = mock_http_server.call_args[0][1]

        # Create handler instance by bypassing normal initialization
        handler = handler_class.__new__(handler_class)
        handler.path = "/health"

        # Mock the response methods
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.wfile = Mock()

        # Test GET request to /health
        handler.do_GET()

        # Verify response
        handler.send_response.assert_called_with(200)
        handler.send_header.assert_called_with("Content-Type", "application/json")
        handler.end_headers.assert_called_once()

        # Verify JSON was written
        handler.wfile.write.assert_called_once()
        written_data = handler.wfile.write.call_args[0][0]

        # Should be valid JSON
        health_data = json.loads(written_data.decode())
        assert "status" in health_data
        assert "metrics" in health_data

    @patch("http.server.HTTPServer")
    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_health_handler_404(self, mock_process_class, mock_http_server):
        """Test the health endpoint returns 404 for other paths."""
        from networkx_mcp.monitoring_legacy import HealthMonitor, create_health_endpoint

        mock_process_class.return_value = Mock()
        monitor = HealthMonitor()

        with patch("threading.Thread"):
            create_health_endpoint(monitor)

        handler_class = mock_http_server.call_args[0][1]

        # Create handler instance by bypassing normal initialization
        handler = handler_class.__new__(handler_class)
        handler.path = "/invalid"

        handler.send_response = Mock()
        handler.end_headers = Mock()

        handler.do_GET()

        handler.send_response.assert_called_with(404)
        handler.end_headers.assert_called_once()

    @patch("http.server.HTTPServer")
    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_health_handler_log_suppression(self, mock_process_class, mock_http_server):
        """Test that health handler suppresses HTTP logging."""
        from networkx_mcp.monitoring_legacy import HealthMonitor, create_health_endpoint

        mock_process_class.return_value = Mock()
        monitor = HealthMonitor()

        with patch("threading.Thread"):
            create_health_endpoint(monitor)

        handler_class = mock_http_server.call_args[0][1]

        # Create handler instance by bypassing normal initialization
        handler = handler_class.__new__(handler_class)

        # Test log_message method does nothing (suppresses logging)
        result = handler.log_message("test format", "arg1", "arg2")
        assert result is None

    @patch("threading.Thread")
    @patch("http.server.HTTPServer")
    @patch("networkx_mcp.monitoring_legacy.psutil.Process")
    def test_create_health_endpoint_custom_port(
        self, mock_process_class, mock_http_server, mock_thread
    ):
        """Test health endpoint creation with custom port."""
        from networkx_mcp.monitoring_legacy import HealthMonitor, create_health_endpoint

        mock_process_class.return_value = Mock()
        mock_server = Mock()
        mock_http_server.return_value = mock_server

        monitor = HealthMonitor()

        create_health_endpoint(monitor, port=9090)

        # Verify custom port was used
        args, kwargs = mock_http_server.call_args
        assert args[0] == ("", 9090)
