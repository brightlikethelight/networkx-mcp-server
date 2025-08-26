"""Monitoring dashboard for NetworkX MCP Server.

Provides a web-based dashboard for real-time monitoring of the MCP server
with metrics visualization and alert management.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from aiohttp import web

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    web = None  # type: ignore

# Type checking imports
if TYPE_CHECKING and HAS_AIOHTTP:
    from aiohttp import web

from .mcp_health import mcp_health_monitor
from .webhooks import AlertSeverity, alert_manager

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Web-based monitoring dashboard for MCP server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):  # nosec B104
        """Initialize monitoring dashboard.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        if not HAS_AIOHTTP:
            logger.warning("aiohttp not installed - dashboard disabled")
            self.enabled = False
            return

        self.host = host
        self.port = port
        self.enabled = True
        self.app = web.Application()
        self._setup_routes()
        self._setup_static_files()

    def _setup_routes(self):
        """Set up dashboard API routes."""
        self.app.router.add_get("/", self.index_handler)
        self.app.router.add_get("/api/health", self.health_handler)
        self.app.router.add_get("/api/metrics", self.metrics_handler)
        self.app.router.add_get("/api/metrics/prometheus", self.prometheus_handler)
        self.app.router.add_post("/api/alerts/test", self.test_alert_handler)
        self.app.router.add_get("/api/alerts/history", self.alert_history_handler)
        self.app.router.add_get("/ws", self.websocket_handler)

    def _setup_static_files(self):
        """Set up static file serving."""
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            self.app.router.add_static("/static", static_dir)

    async def index_handler(self, request: web.Request) -> web.Response:
        """Serve dashboard HTML page.

        Args:
            request: HTTP request

        Returns:
            HTML response
        """
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>NetworkX MCP Server - Monitoring Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 30px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card h2 {
            font-size: 1.3rem;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        .metric-value {
            font-size: 1.2rem;
            font-weight: bold;
        }
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-healthy {
            background: #4caf50;
        }
        .status-degraded {
            background: #ff9800;
        }
        .status-critical {
            background: #f44336;
        }
        .chart-container {
            height: 200px;
            margin-top: 15px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            opacity: 0.7;
        }
        .alert-box {
            background: rgba(255, 193, 7, 0.2);
            border: 1px solid #ffc107;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            display: none;
        }
        .alert-box.show {
            display: block;
        }
        .btn {
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.3s ease;
        }
        .btn:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .live-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #4caf50;
            border-radius: 50%;
            animation: pulse 2s infinite;
            margin-right: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌐 NetworkX MCP Server</h1>
        <p class="subtitle">
            <span class="live-indicator"></span>
            Real-time Monitoring Dashboard
        </p>

        <div id="alert-box" class="alert-box">
            <strong>⚠️ Alert:</strong> <span id="alert-message"></span>
        </div>

        <div class="grid">
            <div class="card">
                <h2>📊 System Health</h2>
                <div class="metric">
                    <span class="metric-label">Status</span>
                    <span class="status-badge status-healthy" id="health-status">Healthy</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Uptime</span>
                    <span class="metric-value" id="uptime">0h 0m</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Active Sessions</span>
                    <span class="metric-value" id="active-sessions">0</span>
                </div>
            </div>

            <div class="card">
                <h2>🔧 Tool Metrics</h2>
                <div class="metric">
                    <span class="metric-label">Total Invocations</span>
                    <span class="metric-value" id="total-invocations">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Unique Tools</span>
                    <span class="metric-value" id="unique-tools">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Latency</span>
                    <span class="metric-value" id="avg-latency">0ms</span>
                </div>
            </div>

            <div class="card">
                <h2>🚀 Performance</h2>
                <div class="metric">
                    <span class="metric-label">Throughput</span>
                    <span class="metric-value" id="throughput">0/min</span>
                </div>
                <div class="metric">
                    <span class="metric-label">P95 Session Duration</span>
                    <span class="metric-value" id="p95-duration">0s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Error Rate</span>
                    <span class="metric-value" id="error-rate">0%</span>
                </div>
            </div>

            <div class="card">
                <h2>🔐 Security</h2>
                <div class="metric">
                    <span class="metric-label">Auth Enabled</span>
                    <span class="metric-value" id="auth-enabled">Yes</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Permission Denials</span>
                    <span class="metric-value" id="permission-denials">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Protocol Errors</span>
                    <span class="metric-value" id="protocol-errors">0</span>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>📈 Tool Usage Trends</h2>
            <div class="chart-container">
                <span>Real-time chart would appear here (requires Chart.js)</span>
            </div>
        </div>

        <div class="card">
            <h2>🔔 Alert Testing</h2>
            <p style="margin-bottom: 15px; opacity: 0.9;">Test webhook integration by sending a test alert:</p>
            <button class="btn" onclick="sendTestAlert()">Send Test Alert</button>
        </div>
    </div>

    <script>
        let ws = null;

        async function fetchMetrics() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Failed to fetch metrics:', error);
                showAlert('Failed to fetch metrics from server');
            }
        }

        function updateDashboard(data) {
            // Update health status
            const statusElement = document.getElementById('health-status');
            statusElement.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
            statusElement.className = 'status-badge status-' + data.status;

            // Update uptime
            const uptime = data.uptime_seconds;
            const hours = Math.floor(uptime / 3600);
            const minutes = Math.floor((uptime % 3600) / 60);
            document.getElementById('uptime').textContent = hours + 'h ' + minutes + 'm';

            // Update MCP metrics
            const mcp = data.mcp_metrics;
            document.getElementById('active-sessions').textContent = mcp.transport.active_sessions;
            document.getElementById('total-invocations').textContent = mcp.tools.total_invocations;
            document.getElementById('unique-tools').textContent = mcp.tools.unique_tools_used;
            document.getElementById('avg-latency').textContent = Math.round(mcp.tools.avg_tool_latency) + 'ms';
            document.getElementById('throughput').textContent = mcp.performance.tool_throughput.toFixed(1) + '/min';
            document.getElementById('p95-duration').textContent = mcp.performance.p95_session_duration.toFixed(1) + 's';

            // Calculate error rate
            let errorRate = 0;
            const errorRates = mcp.tools.tool_error_rates;
            if (Object.keys(errorRates).length > 0) {
                const rates = Object.values(errorRates);
                errorRate = (rates.reduce((a, b) => a + b, 0) / rates.length) * 100;
            }
            document.getElementById('error-rate').textContent = errorRate.toFixed(1) + '%';

            // Update security metrics
            document.getElementById('auth-enabled').textContent = mcp.security.auth_enabled ? 'Yes' : 'No';
            document.getElementById('permission-denials').textContent = mcp.security.permission_denials;
            document.getElementById('protocol-errors').textContent = mcp.security.protocol_errors;
        }

        function showAlert(message) {
            const alertBox = document.getElementById('alert-box');
            const alertMessage = document.getElementById('alert-message');
            alertMessage.textContent = message;
            alertBox.classList.add('show');
            setTimeout(() => {
                alertBox.classList.remove('show');
            }, 5000);
        }

        async function sendTestAlert() {
            try {
                const response = await fetch('/api/alerts/test', {
                    method: 'POST'
                });
                const data = await response.json();
                showAlert(data.message || 'Test alert sent successfully');
            } catch (error) {
                console.error('Failed to send test alert:', error);
                showAlert('Failed to send test alert');
            }
        }

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(protocol + '//' + window.location.host + '/ws');

            ws.onopen = () => {
                console.log('WebSocket connected');
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'metrics') {
                        updateDashboard(data.data);
                    } else if (data.type === 'alert') {
                        showAlert(data.message);
                    }
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected, reconnecting...');
                setTimeout(connectWebSocket, 5000);
            };
        }

        // Initial fetch and setup
        fetchMetrics();
        setInterval(fetchMetrics, 5000);  // Refresh every 5 seconds

        // Try to establish WebSocket connection for real-time updates
        // connectWebSocket();  // Uncomment when WebSocket is implemented
    </script>
</body>
</html>
        """
        return web.Response(text=html, content_type="text/html")

    async def health_handler(self, request: web.Request) -> web.Response:
        """Get health status API endpoint.

        Args:
            request: HTTP request

        Returns:
            JSON response with health status
        """
        health_data = await mcp_health_monitor.get_mcp_health_status()
        return web.json_response(health_data)

    async def metrics_handler(self, request: web.Request) -> web.Response:
        """Get metrics API endpoint.

        Args:
            request: HTTP request

        Returns:
            JSON response with metrics
        """
        # Get comprehensive metrics
        metrics = {
            "mcp_health": await mcp_health_monitor.get_mcp_health_status(),
            "alert_history": [
                {
                    "title": alert.title,
                    "message": alert.message,
                    "severity": alert.severity.value,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in alert_manager.alert_history[-10:]  # Last 10 alerts
            ],
        }
        return web.json_response(metrics)

    async def prometheus_handler(self, request: web.Request) -> web.Response:
        """Get Prometheus-formatted metrics.

        Args:
            request: HTTP request

        Returns:
            Prometheus text format response
        """
        prometheus_metrics = mcp_health_monitor.export_metrics_prometheus()
        return web.Response(text=prometheus_metrics, content_type="text/plain")

    async def test_alert_handler(self, request: web.Request) -> web.Response:
        """Send a test alert.

        Args:
            request: HTTP request

        Returns:
            JSON response confirming alert sent
        """
        await alert_manager.send_alert(
            title="Test Alert",
            message="This is a test alert from the monitoring dashboard",
            severity=AlertSeverity.INFO,
            tags={"source": "dashboard", "type": "test"},
        )
        return web.json_response({"success": True, "message": "Test alert sent"})

    async def alert_history_handler(self, request: web.Request) -> web.Response:
        """Get alert history.

        Args:
            request: HTTP request

        Returns:
            JSON response with alert history
        """
        history = [
            {
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp.isoformat(),
                "tags": alert.tags,
                "metadata": alert.metadata,
            }
            for alert in alert_manager.alert_history
        ]
        return web.json_response({"alerts": history})

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket handler for real-time updates.

        Args:
            request: HTTP request

        Returns:
            WebSocket response
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        try:
            # Send metrics every 2 seconds
            while True:
                health_data = await mcp_health_monitor.get_mcp_health_status()
                await ws.send_json({"type": "metrics", "data": health_data})
                await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await ws.close()

        return ws

    async def start(self):
        """Start the monitoring dashboard."""
        if not self.enabled:
            logger.warning("Dashboard not enabled (aiohttp not installed)")
            return

        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"Monitoring dashboard started at http://{self.host}:{self.port}")

    async def stop(self):
        """Stop the monitoring dashboard."""
        if self.enabled:
            await self.app.shutdown()
            await self.app.cleanup()


# Global dashboard instance
dashboard = MonitoringDashboard()
