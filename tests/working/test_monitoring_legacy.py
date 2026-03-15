"""Tests for monitoring_legacy.py HealthMonitor."""

from networkx_mcp.monitoring_legacy import HealthMonitor


class TestHealthMonitor:
    def setup_method(self):
        self.monitor = HealthMonitor()

    def test_initial_state(self):
        assert self.monitor.request_count == 0
        assert self.monitor.error_count == 0
        assert self.monitor.tool_usage == {}

    def test_record_request_success(self):
        self.monitor.record_request("tools/call")
        assert self.monitor.request_count == 1
        assert self.monitor.error_count == 0
        assert self.monitor.tool_usage["tools/call"] == 1

    def test_record_request_failure(self):
        self.monitor.record_request("tools/call", success=False)
        assert self.monitor.request_count == 1
        assert self.monitor.error_count == 1

    def test_record_non_tool_request(self):
        self.monitor.record_request("initialize")
        assert self.monitor.request_count == 1
        assert "initialize" not in self.monitor.tool_usage

    def test_get_health_status(self):
        self.monitor.record_request("tools/call")
        self.monitor.record_request("tools/call", success=False)
        status = self.monitor.get_health_status()

        assert status["status"] == "healthy"
        assert "timestamp" in status
        assert status["metrics"]["requests"]["total"] == 2
        assert status["metrics"]["requests"]["errors"] == 1
        assert status["metrics"]["requests"]["success_rate"] == 50.0
        assert "system" in status["metrics"]
        assert "graphs" in status["metrics"]

    def test_health_status_with_graphs(self):
        import networkx as nx

        self.monitor.graphs = {"g1": nx.path_graph(5)}
        status = self.monitor.get_health_status()
        assert status["metrics"]["graphs"]["count"] == 1
        assert status["metrics"]["graphs"]["total_nodes"] == 5
        assert status["metrics"]["graphs"]["total_edges"] == 4

    def test_format_uptime_less_than_minute(self):
        assert self.monitor._format_uptime(30) == "< 1m"

    def test_format_uptime_minutes(self):
        assert self.monitor._format_uptime(300) == "5m"

    def test_format_uptime_hours(self):
        assert self.monitor._format_uptime(7200) == "2h"

    def test_format_uptime_days(self):
        result = self.monitor._format_uptime(90061)
        assert "1d" in result
        assert "1h" in result
        assert "1m" in result
