"""
Webhook Monitoring Service for NetworkX MCP Server CI/CD

This module provides comprehensive webhook monitoring capabilities for CI/CD pipelines,
including alert management, notification dispatching, and failure analysis.
"""

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Severity(Enum):
    """Alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Supported alert channels"""

    SLACK = "slack"
    DISCORD = "discord"
    TEAMS = "teams"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    CUSTOM = "custom"


@dataclass
class Alert:
    """Represents a CI/CD alert"""

    id: str
    severity: Severity
    title: str
    message: str
    workflow: str
    branch: str
    commit: str
    actor: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_patterns: List[str] = field(default_factory=list)
    failed_jobs: int = 0
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "id": self.id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "workflow": self.workflow,
            "branch": self.branch,
            "commit": self.commit,
            "actor": self.actor,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "error_patterns": self.error_patterns,
            "failed_jobs": self.failed_jobs,
            "url": self.url,
        }


@dataclass
class WebhookConfig:
    """Webhook configuration"""

    channel: AlertChannel
    url: str
    enabled: bool = True
    min_severity: Severity = Severity.MEDIUM
    rate_limit: int = 10  # Max alerts per hour
    retry_count: int = 3
    timeout: int = 30
    headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> List["WebhookConfig"]:
        """Load webhook configs from YAML file"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        configs = []
        for webhook in data.get("webhooks", []):
            configs.append(
                cls(
                    channel=AlertChannel(webhook["channel"]),
                    url=webhook["url"],
                    enabled=webhook.get("enabled", True),
                    min_severity=Severity(webhook.get("min_severity", "medium")),
                    rate_limit=webhook.get("rate_limit", 10),
                    retry_count=webhook.get("retry_count", 3),
                    timeout=webhook.get("timeout", 30),
                    headers=webhook.get("headers", {}),
                )
            )

        return configs


class WebhookMonitor:
    """Main webhook monitoring service"""

    def __init__(self, configs: List[WebhookConfig]):
        self.configs = {config.channel: config for config in configs}
        self.alert_history: List[Alert] = []
        self.rate_limiters: Dict[AlertChannel, List[datetime]] = defaultdict(list)
        self.session: Optional[aiohttp.ClientSession] = None
        self.formatters: Dict[AlertChannel, Callable] = {
            AlertChannel.SLACK: self._format_slack,
            AlertChannel.DISCORD: self._format_discord,
            AlertChannel.TEAMS: self._format_teams,
            AlertChannel.PAGERDUTY: self._format_pagerduty,
            AlertChannel.CUSTOM: self._format_custom,
        }

    async def start(self):
        """Start the monitoring service"""
        self.session = aiohttp.ClientSession()
        logger.info("Webhook monitoring service started")

    async def stop(self):
        """Stop the monitoring service"""
        if self.session:
            await self.session.close()
        logger.info("Webhook monitoring service stopped")

    async def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """Send alert to all configured channels"""
        results = {}

        for channel, config in self.configs.items():
            if not config.enabled:
                continue

            if alert.severity.value < config.min_severity.value:
                continue

            if not self._check_rate_limit(channel, config):
                logger.warning(f"Rate limit exceeded for {channel.value}")
                results[channel.value] = False
                continue

            try:
                success = await self._send_to_channel(channel, config, alert)
                results[channel.value] = success
            except Exception as e:
                logger.error(f"Failed to send alert to {channel.value}: {e}")
                results[channel.value] = False

        self.alert_history.append(alert)
        return results

    def _check_rate_limit(self, channel: AlertChannel, config: WebhookConfig) -> bool:
        """Check if we're within rate limits"""
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=1)

        # Clean old entries
        self.rate_limiters[channel] = [
            t for t in self.rate_limiters[channel] if t > cutoff
        ]

        # Check limit
        if len(self.rate_limiters[channel]) >= config.rate_limit:
            return False

        self.rate_limiters[channel].append(now)
        return True

    async def _send_to_channel(
        self, channel: AlertChannel, config: WebhookConfig, alert: Alert
    ) -> bool:
        """Send alert to specific channel with retries"""
        formatter = self.formatters.get(channel, self._format_custom)
        payload = formatter(alert)

        for attempt in range(config.retry_count):
            try:
                async with self.session.post(
                    config.url,
                    json=payload,
                    headers=config.headers,
                    timeout=aiohttp.ClientTimeout(total=config.timeout),
                ) as response:
                    if response.status < 300:
                        logger.info(f"Alert sent to {channel.value}: {alert.id}")
                        return True

                    logger.warning(
                        f"Failed to send alert to {channel.value}: "
                        f"HTTP {response.status}"
                    )

            except asyncio.TimeoutError:
                logger.warning(f"Timeout sending alert to {channel.value}")
            except Exception as e:
                logger.error(f"Error sending alert to {channel.value}: {e}")

            if attempt < config.retry_count - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff

        return False

    def _format_slack(self, alert: Alert) -> Dict[str, Any]:
        """Format alert for Slack"""
        color = {
            Severity.CRITICAL: "#FF0000",
            Severity.HIGH: "#FFA500",
            Severity.MEDIUM: "#FFFF00",
            Severity.LOW: "#00FF00",
        }[alert.severity]

        return {
            "attachments": [
                {
                    "color": color,
                    "title": alert.title,
                    "title_link": alert.url,
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True,
                        },
                        {"title": "Workflow", "value": alert.workflow, "short": True},
                        {"title": "Branch", "value": alert.branch, "short": True},
                        {"title": "Commit", "value": alert.commit, "short": True},
                        {"title": "Actor", "value": alert.actor, "short": True},
                        {
                            "title": "Failed Jobs",
                            "value": str(alert.failed_jobs),
                            "short": True,
                        },
                    ],
                    "footer": "NetworkX MCP Server CI/CD",
                    "ts": int(alert.timestamp.timestamp()),
                }
            ]
        }

    def _format_discord(self, alert: Alert) -> Dict[str, Any]:
        """Format alert for Discord"""
        color = {
            Severity.CRITICAL: 0xFF0000,
            Severity.HIGH: 0xFFA500,
            Severity.MEDIUM: 0xFFFF00,
            Severity.LOW: 0x00FF00,
        }[alert.severity]

        return {
            "embeds": [
                {
                    "title": alert.title,
                    "url": alert.url,
                    "color": color,
                    "description": alert.message,
                    "fields": [
                        {
                            "name": "Severity",
                            "value": alert.severity.value.upper(),
                            "inline": True,
                        },
                        {"name": "Workflow", "value": alert.workflow, "inline": True},
                        {"name": "Branch", "value": alert.branch, "inline": True},
                        {
                            "name": "Commit",
                            "value": f"`{alert.commit}`",
                            "inline": True,
                        },
                        {"name": "Actor", "value": alert.actor, "inline": True},
                        {
                            "name": "Failed Jobs",
                            "value": str(alert.failed_jobs),
                            "inline": True,
                        },
                    ],
                    "footer": {"text": "NetworkX MCP Server CI/CD"},
                    "timestamp": alert.timestamp.isoformat(),
                }
            ]
        }

    def _format_teams(self, alert: Alert) -> Dict[str, Any]:
        """Format alert for Microsoft Teams"""
        theme_color = {
            Severity.CRITICAL: "FF0000",
            Severity.HIGH: "FFA500",
            Severity.MEDIUM: "FFFF00",
            Severity.LOW: "00FF00",
        }[alert.severity]

        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": theme_color,
            "summary": alert.title,
            "sections": [
                {
                    "activityTitle": alert.title,
                    "text": alert.message,
                    "facts": [
                        {"name": "Severity", "value": alert.severity.value.upper()},
                        {"name": "Workflow", "value": alert.workflow},
                        {"name": "Branch", "value": alert.branch},
                        {"name": "Commit", "value": alert.commit},
                        {"name": "Actor", "value": alert.actor},
                        {"name": "Failed Jobs", "value": str(alert.failed_jobs)},
                    ],
                }
            ],
            "potentialAction": [
                {
                    "@type": "OpenUri",
                    "name": "View Workflow",
                    "targets": [{"os": "default", "uri": alert.url}],
                }
            ],
        }

    def _format_pagerduty(self, alert: Alert) -> Dict[str, Any]:
        """Format alert for PagerDuty"""
        return {
            "routing_key": "",  # Would be configured
            "event_action": "trigger",
            "payload": {
                "summary": alert.title,
                "severity": alert.severity.value,
                "source": "NetworkX MCP CI/CD",
                "component": alert.workflow,
                "group": alert.branch,
                "class": "ci/cd failure",
                "custom_details": {
                    "message": alert.message,
                    "commit": alert.commit,
                    "actor": alert.actor,
                    "failed_jobs": alert.failed_jobs,
                    "error_patterns": alert.error_patterns,
                },
            },
            "links": [{"href": alert.url, "text": "View Workflow"}],
        }

    def _format_custom(self, alert: Alert) -> Dict[str, Any]:
        """Format alert for custom webhook"""
        return alert.to_dict()

    async def analyze_failures(self, window_hours: int = 24) -> Dict[str, Any]:
        """Analyze recent failures for patterns"""
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent_alerts = [a for a in self.alert_history if a.timestamp > cutoff]

        if not recent_alerts:
            return {"status": "healthy", "alerts": 0}

        # Analyze patterns
        workflow_failures = defaultdict(int)
        branch_failures = defaultdict(int)
        actor_failures = defaultdict(int)
        error_patterns = defaultdict(int)
        severity_counts = defaultdict(int)

        for alert in recent_alerts:
            workflow_failures[alert.workflow] += 1
            branch_failures[alert.branch] += 1
            actor_failures[alert.actor] += 1
            severity_counts[alert.severity.value] += 1

            for pattern in alert.error_patterns:
                error_patterns[pattern] += 1

        # Determine health status
        total_alerts = len(recent_alerts)
        critical_count = severity_counts.get("critical", 0)
        high_count = severity_counts.get("high", 0)

        if critical_count > 0 or high_count > 5:
            status = "critical"
        elif high_count > 2 or total_alerts > 10:
            status = "degraded"
        elif total_alerts > 5:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "alerts": total_alerts,
            "window_hours": window_hours,
            "top_failing_workflows": dict(
                sorted(workflow_failures.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "top_failing_branches": dict(
                sorted(branch_failures.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "severity_breakdown": dict(severity_counts),
            "common_error_patterns": dict(
                sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }

    def generate_alert_id(self, workflow: str, branch: str, commit: str) -> str:
        """Generate unique alert ID"""
        data = f"{workflow}-{branch}-{commit}-{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]


class AlertAggregator:
    """Aggregates and deduplicates alerts"""

    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self.pending_alerts: Dict[str, List[Alert]] = defaultdict(list)
        self.last_sent: Dict[str, datetime] = {}

    def should_send(self, alert: Alert) -> bool:
        """Check if alert should be sent or aggregated"""
        key = f"{alert.workflow}-{alert.branch}"

        # Check if we've sent a similar alert recently
        if key in self.last_sent:
            time_since_last = (datetime.utcnow() - self.last_sent[key]).total_seconds()
            if time_since_last < self.window_seconds:
                self.pending_alerts[key].append(alert)
                return False

        self.last_sent[key] = datetime.utcnow()
        return True

    def get_aggregated_alert(self, key: str) -> Optional[Alert]:
        """Get aggregated alert for a key"""
        if key not in self.pending_alerts:
            return None

        alerts = self.pending_alerts[key]
        if not alerts:
            return None

        # Create aggregated alert
        first_alert = alerts[0]
        aggregated = Alert(
            id=first_alert.id,
            severity=max(
                (a.severity for a in alerts), key=lambda s: list(Severity).index(s)
            ),
            title=f"{first_alert.title} ({len(alerts)} occurrences)",
            message="Multiple failures detected:\n"
            + "\n".join(set(a.message for a in alerts)),
            workflow=first_alert.workflow,
            branch=first_alert.branch,
            commit=first_alert.commit,
            actor=first_alert.actor,
            timestamp=first_alert.timestamp,
            metadata={"aggregated_count": len(alerts)},
            error_patterns=list(set(p for a in alerts for p in a.error_patterns)),
            failed_jobs=sum(a.failed_jobs for a in alerts),
            url=first_alert.url,
        )

        # Clear pending alerts
        del self.pending_alerts[key]

        return aggregated


# Example usage
async def main():
    """Example webhook monitoring usage"""

    # Load configurations
    configs = [
        WebhookConfig(
            channel=AlertChannel.SLACK,
            url="https://hooks.slack.com/services/xxx",
            min_severity=Severity.MEDIUM,
        ),
        WebhookConfig(
            channel=AlertChannel.DISCORD,
            url="https://discord.com/api/webhooks/xxx",
            min_severity=Severity.HIGH,
        ),
    ]

    # Initialize monitor
    monitor = WebhookMonitor(configs)
    await monitor.start()

    try:
        # Create sample alert
        alert = Alert(
            id=monitor.generate_alert_id("CI", "main", "abc123"),
            severity=Severity.HIGH,
            title="CI Pipeline Failed",
            message="Tests failed on main branch",
            workflow="CI",
            branch="main",
            commit="abc123",
            actor="developer",
            timestamp=datetime.utcnow(),
            error_patterns=["test_failure", "type_error"],
            failed_jobs=3,
            url="https://github.com/org/repo/actions/runs/123",
        )

        # Send alert
        results = await monitor.send_alert(alert)
        print(f"Alert sent: {results}")

        # Analyze failures
        analysis = await monitor.analyze_failures(24)
        print(f"Failure analysis: {json.dumps(analysis, indent=2)}")

    finally:
        await monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())
