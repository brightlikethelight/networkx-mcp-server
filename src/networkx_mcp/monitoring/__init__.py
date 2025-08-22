"""Monitoring module for NetworkX MCP Server.

Provides comprehensive monitoring, alerting, and observability features.
"""

from .dashboard import MonitoringDashboard, dashboard
from .mcp_health import MCPHealthMonitor, MCPMetric, mcp_health_monitor
from .webhooks import (
    Alert,
    AlertManager,
    AlertSeverity,
    WebhookClient,
    WebhookProvider,
    alert_manager,
)

__all__ = [
    # Dashboard
    "MonitoringDashboard",
    "dashboard",
    # MCP Health
    "MCPHealthMonitor",
    "MCPMetric",
    "mcp_health_monitor",
    # Webhooks
    "Alert",
    "AlertManager",
    "AlertSeverity",
    "WebhookClient",
    "WebhookProvider",
    "alert_manager",
]
