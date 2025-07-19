"""
Simple health monitoring for NetworkX MCP server.
"""

import os
import time
from datetime import datetime
from typing import Any, Dict

import psutil


class HealthMonitor:
    """Monitor server health and performance."""

    def __init__(self):
        """Initialize health monitor."""
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.tool_usage = {}
        self.process = psutil.Process(os.getpid())

    def record_request(self, method: str, success: bool = True):
        """Record a request."""
        self.request_count += 1
        if not success:
            self.error_count += 1

        # Track tool usage
        if method == "tools/call":
            self.tool_usage[method] = self.tool_usage.get(method, 0) + 1

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        uptime = time.time() - self.start_time

        # Get system metrics
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent(interval=0.1)

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(uptime, 2),
            "uptime_human": self._format_uptime(uptime),
            "metrics": {
                "requests": {
                    "total": self.request_count,
                    "errors": self.error_count,
                    "success_rate": round(
                        (self.request_count - self.error_count)
                        / max(self.request_count, 1)
                        * 100,
                        2,
                    ),
                },
                "system": {
                    "memory_mb": round(memory_info.rss / 1024 / 1024, 2),
                    "cpu_percent": round(cpu_percent, 2),
                    "pid": os.getpid(),
                },
                "graphs": {
                    "count": len(getattr(self, "graphs", {})),
                    "total_nodes": sum(
                        g.number_of_nodes()
                        for g in getattr(self, "graphs", {}).values()
                    ),
                    "total_edges": sum(
                        g.number_of_edges()
                        for g in getattr(self, "graphs", {}).values()
                    ),
                },
            },
            "version": "1.0.0",
        }

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")

        return " ".join(parts) if parts else "< 1m"


# Optional HTTP health endpoint
def create_health_endpoint(monitor: HealthMonitor, port: int = 8080):
    """Create a simple HTTP health endpoint."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()

                health_data = monitor.get_health_status()
                self.wfile.write(json.dumps(health_data, indent=2).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress logging

    server = HTTPServer(("", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    return server
