"""Monitoring module for NetworkX MCP Server.

Provides comprehensive monitoring, alerting, and observability features.
"""

from .ci_dashboard import CIDashboard, ci_dashboard
from .dashboard import MonitoringDashboard, dashboard
from .dora_metrics import (
    DORAMetricsCollector,
    dora_collector,
    generate_dora_report,
    get_dora_metrics,
)
from .mcp_health import MCPHealthMonitor, MCPMetric, mcp_health_monitor
from .sentry_integration import SentryIntegration, init_sentry, sentry
from .webhooks import (
    Alert,
    AlertManager,
    AlertSeverity,
    WebhookClient,
    WebhookProvider,
    alert_manager,
)

__all__ = [
    # CI/CD Dashboard
    "CIDashboard",
    "ci_dashboard",
    # Dashboard
    "MonitoringDashboard",
    "dashboard",
    # DORA Metrics
    "DORAMetricsCollector",
    "dora_collector",
    "get_dora_metrics",
    "generate_dora_report",
    # MCP Health
    "MCPHealthMonitor",
    "MCPMetric",
    "mcp_health_monitor",
    # Sentry
    "SentryIntegration",
    "init_sentry",
    "sentry",
    # Webhooks
    "Alert",
    "AlertManager",
    "AlertSeverity",
    "WebhookClient",
    "WebhookProvider",
    "alert_manager",
]
