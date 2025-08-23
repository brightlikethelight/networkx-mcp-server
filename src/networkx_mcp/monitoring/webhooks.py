"""Webhook integration for monitoring alerts.

Provides flexible webhook notifications for monitoring events with support for
multiple webhook providers (Slack, Discord, generic HTTP).
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List
from urllib.parse import urlparse

try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class WebhookProvider(Enum):
    """Supported webhook providers."""

    GENERIC = "generic"
    SLACK = "slack"
    DISCORD = "discord"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"


@dataclass
class Alert:
    """Alert data structure."""

    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    source: str = "networkx-mcp-server"
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None


class WebhookClient:
    """Client for sending webhook notifications."""

    def __init__(self, webhook_url: str, provider: WebhookProvider = None):
        """Initialize webhook client.

        Args:
            webhook_url: Webhook endpoint URL
            provider: Provider type (auto-detected if None)
        """
        if not HAS_AIOHTTP:
            logger.warning("aiohttp not installed - webhook notifications disabled")
            self.enabled = False
            return

        self.webhook_url = webhook_url
        self.provider = provider or self._detect_provider(webhook_url)
        self.enabled = True
        self._session = None  # Will be aiohttp.ClientSession when created

    def _detect_provider(self, url: str) -> WebhookProvider:
        """Auto-detect webhook provider from URL.

        Args:
            url: Webhook URL

        Returns:
            Detected provider type
        """
        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        if "slack.com" in hostname or "hooks.slack.com" in hostname:
            return WebhookProvider.SLACK
        elif "discord.com" in hostname or "discordapp.com" in hostname:
            return WebhookProvider.DISCORD
        elif "webhook.office.com" in hostname:
            return WebhookProvider.TEAMS
        elif "pagerduty.com" in hostname:
            return WebhookProvider.PAGERDUTY
        else:
            return WebhookProvider.GENERIC

    async def _get_session(self):
        """Get or create aiohttp session.

        Returns:
            Active aiohttp session or None if aiohttp not available
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        try:
            payload = self._format_payload(alert)
            session = await self._get_session()

            async with session.post(
                self.webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status >= 200 and response.status < 300:
                    logger.info(f"Alert sent successfully: {alert.title}")
                    return True
                else:
                    logger.error(
                        f"Failed to send alert: {response.status} - {await response.text()}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return False

    def _format_payload(self, alert: Alert) -> Dict[str, Any]:
        """Format alert payload for specific provider.

        Args:
            alert: Alert to format

        Returns:
            Provider-specific payload
        """
        if self.provider == WebhookProvider.SLACK:
            return self._format_slack_payload(alert)
        elif self.provider == WebhookProvider.DISCORD:
            return self._format_discord_payload(alert)
        elif self.provider == WebhookProvider.TEAMS:
            return self._format_teams_payload(alert)
        elif self.provider == WebhookProvider.PAGERDUTY:
            return self._format_pagerduty_payload(alert)
        else:
            return self._format_generic_payload(alert)

    def _format_slack_payload(self, alert: Alert) -> Dict[str, Any]:
        """Format Slack webhook payload.

        Args:
            alert: Alert to format

        Returns:
            Slack-formatted payload
        """
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#990000",
        }.get(alert.severity, "#808080")

        fields = []
        if alert.tags:
            for key, value in alert.tags.items():
                fields.append({"title": key, "value": value, "short": True})

        return {
            "attachments": [
                {
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": fields,
                    "footer": alert.source,
                    "ts": int(alert.timestamp.timestamp()),
                }
            ]
        }

    def _format_discord_payload(self, alert: Alert) -> Dict[str, Any]:
        """Format Discord webhook payload.

        Args:
            alert: Alert to format

        Returns:
            Discord-formatted payload
        """
        color = {
            AlertSeverity.INFO: 0x36A64F,
            AlertSeverity.WARNING: 0xFF9900,
            AlertSeverity.ERROR: 0xFF0000,
            AlertSeverity.CRITICAL: 0x990000,
        }.get(alert.severity, 0x808080)

        fields = []
        if alert.tags:
            for key, value in alert.tags.items():
                fields.append({"name": key, "value": value, "inline": True})

        return {
            "embeds": [
                {
                    "title": alert.title,
                    "description": alert.message,
                    "color": color,
                    "fields": fields,
                    "footer": {"text": alert.source},
                    "timestamp": alert.timestamp.isoformat(),
                }
            ]
        }

    def _format_teams_payload(self, alert: Alert) -> Dict[str, Any]:
        """Format Microsoft Teams webhook payload.

        Args:
            alert: Alert to format

        Returns:
            Teams-formatted payload
        """
        theme_color = {
            AlertSeverity.INFO: "36a64f",
            AlertSeverity.WARNING: "ff9900",
            AlertSeverity.ERROR: "ff0000",
            AlertSeverity.CRITICAL: "990000",
        }.get(alert.severity, "808080")

        facts = []
        if alert.tags:
            for key, value in alert.tags.items():
                facts.append({"name": key, "value": value})

        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": theme_color,
            "summary": alert.title,
            "sections": [
                {
                    "activityTitle": alert.title,
                    "activitySubtitle": alert.source,
                    "facts": facts,
                    "text": alert.message,
                }
            ],
        }

    def _format_pagerduty_payload(self, alert: Alert) -> Dict[str, Any]:
        """Format PagerDuty webhook payload.

        Args:
            alert: Alert to format

        Returns:
            PagerDuty-formatted payload
        """
        severity = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical",
        }.get(alert.severity, "error")

        return {
            "routing_key": self.webhook_url.split("/")[-1],  # Extract integration key
            "event_action": "trigger",
            "payload": {
                "summary": alert.title,
                "source": alert.source,
                "severity": severity,
                "timestamp": alert.timestamp.isoformat(),
                "custom_details": {
                    "message": alert.message,
                    "tags": alert.tags or {},
                    "metadata": alert.metadata or {},
                },
            },
        }

    def _format_generic_payload(self, alert: Alert) -> Dict[str, Any]:
        """Format generic webhook payload.

        Args:
            alert: Alert to format

        Returns:
            Generic JSON payload
        """
        return {
            "title": alert.title,
            "message": alert.message,
            "severity": alert.severity.value,
            "timestamp": alert.timestamp.isoformat(),
            "source": alert.source,
            "tags": alert.tags or {},
            "metadata": alert.metadata or {},
        }

    async def close(self):
        """Close the webhook client and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()


class AlertManager:
    """Manages alerts and webhook notifications."""

    def __init__(self):
        """Initialize alert manager."""
        self.webhooks: List[WebhookClient] = []
        self.alert_history: List[Alert] = []
        self.alert_thresholds = {
            "error_rate": 0.1,  # 10% error rate
            "response_time": 5000,  # 5 seconds
            "memory_usage": 0.9,  # 90% memory
            "cpu_usage": 0.9,  # 90% CPU
        }

    def add_webhook(self, webhook_url: str, provider: WebhookProvider = None):
        """Add a webhook endpoint.

        Args:
            webhook_url: Webhook URL
            provider: Provider type (auto-detected if None)
        """
        client = WebhookClient(webhook_url, provider)
        self.webhooks.append(client)
        logger.info(f"Added webhook: {client.provider.value}")

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        tags: Dict[str, str] = None,
        metadata: Dict[str, Any] = None,
    ):
        """Send alert to all configured webhooks.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            tags: Optional tags
            metadata: Optional metadata
        """
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            timestamp=datetime.utcnow(),
            tags=tags,
            metadata=metadata,
        )

        self.alert_history.append(alert)

        # Send to all webhooks in parallel
        if self.webhooks:
            tasks = [webhook.send_alert(alert) for webhook in self.webhooks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = sum(1 for r in results if r is True)
            logger.info(f"Alert sent to {success_count}/{len(self.webhooks)} webhooks")

    async def check_thresholds(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and send alerts.

        Args:
            metrics: Current metrics to check
        """
        alerts_to_send = []

        # Check error rate
        if "error_rate" in metrics:
            if metrics["error_rate"] > self.alert_thresholds["error_rate"]:
                alerts_to_send.append(
                    (
                        "High Error Rate",
                        f"Error rate is {metrics['error_rate']:.1%} (threshold: {self.alert_thresholds['error_rate']:.1%})",
                        AlertSeverity.ERROR,
                    )
                )

        # Check response time
        if "avg_response_time" in metrics:
            if metrics["avg_response_time"] > self.alert_thresholds["response_time"]:
                alerts_to_send.append(
                    (
                        "Slow Response Time",
                        f"Average response time is {metrics['avg_response_time']}ms (threshold: {self.alert_thresholds['response_time']}ms)",
                        AlertSeverity.WARNING,
                    )
                )

        # Check memory usage
        if "memory_percent" in metrics:
            if metrics["memory_percent"] > self.alert_thresholds["memory_usage"]:
                alerts_to_send.append(
                    (
                        "High Memory Usage",
                        f"Memory usage is {metrics['memory_percent']:.1%} (threshold: {self.alert_thresholds['memory_usage']:.1%})",
                        AlertSeverity.WARNING,
                    )
                )

        # Check CPU usage
        if "cpu_percent" in metrics:
            if metrics["cpu_percent"] > self.alert_thresholds["cpu_usage"]:
                alerts_to_send.append(
                    (
                        "High CPU Usage",
                        f"CPU usage is {metrics['cpu_percent']:.1%} (threshold: {self.alert_thresholds['cpu_usage']:.1%})",
                        AlertSeverity.WARNING,
                    )
                )

        # Send alerts
        for title, message, severity in alerts_to_send:
            await self.send_alert(
                title=title,
                message=message,
                severity=severity,
                tags={"server": "networkx-mcp"},
                metadata=metrics,
            )

    async def close(self):
        """Close all webhook clients."""
        for webhook in self.webhooks:
            await webhook.close()


# Global alert manager instance
alert_manager = AlertManager()
