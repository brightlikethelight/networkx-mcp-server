"""Intelligent alerting system with context-aware notifications and alert correlation.

Provides advanced alerting capabilities including:
- Alert correlation and deduplication
- Context-aware severity adjustment
- Intelligent noise reduction
- Pattern recognition for recurring issues
- Automated alert enrichment
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .webhooks import Alert, AlertSeverity, alert_manager

logger = logging.getLogger(__name__)


@dataclass
class AlertContext:
    """Enhanced alert with additional context."""

    alert: Alert
    correlation_id: str
    fingerprint: str
    occurrence_count: int = 1
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    related_alerts: List[str] = field(default_factory=list)
    auto_resolved: bool = False
    resolution_time: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertPattern:
    """Pattern for identifying and correlating alerts."""

    def __init__(self, name: str, pattern: str, tags: List[str] = None):
        """Initialize alert pattern.

        Args:
            name: Pattern name
            pattern: Regular expression pattern
            tags: Tags to apply when pattern matches
        """
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.tags = tags or []

    def matches(self, alert: Alert) -> bool:
        """Check if alert matches this pattern.

        Args:
            alert: Alert to check

        Returns:
            True if alert matches pattern
        """
        text = f"{alert.title} {alert.message}"
        return bool(self.pattern.search(text))


class IntelligentAlertManager:
    """Advanced alert manager with correlation and noise reduction."""

    def __init__(
        self,
        correlation_window: int = 300,  # 5 minutes
        dedup_threshold: int = 3,
        auto_resolve_timeout: int = 3600,  # 1 hour
    ):
        """Initialize intelligent alert manager.

        Args:
            correlation_window: Time window for correlating alerts (seconds)
            dedup_threshold: Number of similar alerts before deduplication
            auto_resolve_timeout: Time before auto-resolving alerts (seconds)
        """
        self.correlation_window = correlation_window
        self.dedup_threshold = dedup_threshold
        self.auto_resolve_timeout = auto_resolve_timeout

        # Alert storage and tracking
        self.active_alerts: Dict[str, AlertContext] = {}
        self.alert_history: List[AlertContext] = []
        self.correlation_groups: Dict[str, List[str]] = defaultdict(list)

        # Alert patterns for categorization
        self.patterns = self._initialize_patterns()

        # Statistics
        self.stats = {
            "total_alerts": 0,
            "deduplicated": 0,
            "correlated": 0,
            "auto_resolved": 0,
            "by_severity": defaultdict(int),
            "by_pattern": defaultdict(int),
        }

    def _initialize_patterns(self) -> List[AlertPattern]:
        """Initialize common alert patterns.

        Returns:
            List of alert patterns
        """
        return [
            AlertPattern(
                "test_failure",
                r"test.*fail|fail.*test|pytest.*error|assertion.*error",
                ["testing", "ci"],
            ),
            AlertPattern(
                "dependency_issue",
                r"import.*error|module.*not found|package.*not.*install|dependency.*fail",
                ["dependencies", "build"],
            ),
            AlertPattern(
                "timeout",
                r"timeout|timed out|deadline.*exceed|took too long",
                ["performance", "timeout"],
            ),
            AlertPattern(
                "network_error",
                r"connection.*error|network.*fail|unable to connect|connection refused",
                ["network", "connectivity"],
            ),
            AlertPattern(
                "memory_issue",
                r"out of memory|memory.*exceed|heap.*space|oom|memory leak",
                ["resources", "memory"],
            ),
            AlertPattern(
                "permission_error",
                r"permission.*denied|access.*denied|unauthorized|forbidden",
                ["security", "permissions"],
            ),
            AlertPattern(
                "deployment_failure",
                r"deploy.*fail|rollback|deployment.*error|release.*fail",
                ["deployment", "release"],
            ),
            AlertPattern(
                "performance_regression",
                r"performance.*degrad|slow.*response|latency.*increase|throughput.*decrease",
                ["performance", "regression"],
            ),
        ]

    def _generate_fingerprint(self, alert: Alert) -> str:
        """Generate unique fingerprint for an alert.

        Args:
            alert: Alert to fingerprint

        Returns:
            Alert fingerprint hash
        """
        # Create fingerprint from key alert attributes
        key_parts = [
            alert.title.lower(),
            str(alert.severity.value),
            # Extract key error patterns from message
            re.sub(r"\d+", "N", alert.message)[:100],  # Replace numbers with N
        ]

        fingerprint_str = "|".join(key_parts)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

    def _generate_correlation_id(self, alert: Alert) -> str:
        """Generate correlation ID for grouping related alerts.

        Args:
            alert: Alert to correlate

        Returns:
            Correlation ID
        """
        # Use broader pattern for correlation
        correlation_parts = []

        # Add pattern-based correlation
        for pattern in self.patterns:
            if pattern.matches(alert):
                correlation_parts.append(pattern.name)
                break

        # Add severity-based grouping for critical alerts
        if alert.severity == AlertSeverity.CRITICAL:
            correlation_parts.append("critical")

        # Add time-based correlation (5-minute buckets)
        time_bucket = int(datetime.now().timestamp() / self.correlation_window)
        correlation_parts.append(str(time_bucket))

        return "_".join(correlation_parts) if correlation_parts else "uncorrelated"

    async def process_alert(self, alert: Alert) -> AlertContext:
        """Process an alert with intelligent handling.

        Args:
            alert: Alert to process

        Returns:
            Processed alert context
        """
        # Generate identifiers
        fingerprint = self._generate_fingerprint(alert)
        correlation_id = self._generate_correlation_id(alert)

        # Update statistics
        self.stats["total_alerts"] += 1
        self.stats["by_severity"][alert.severity.value] += 1

        # Check for duplicate/similar alerts
        if fingerprint in self.active_alerts:
            # Update existing alert
            existing = self.active_alerts[fingerprint]
            existing.occurrence_count += 1
            existing.last_seen = datetime.now()

            # Check deduplication threshold
            if existing.occurrence_count >= self.dedup_threshold:
                self.stats["deduplicated"] += 1
                logger.info(
                    f"Alert deduplicated: {alert.title} "
                    f"(occurred {existing.occurrence_count} times)"
                )

                # Escalate severity if recurring critical issue
                if (
                    existing.occurrence_count > 5
                    and alert.severity != AlertSeverity.CRITICAL
                ):
                    existing.alert.severity = AlertSeverity.CRITICAL
                    logger.warning(f"Alert escalated to CRITICAL: {alert.title}")

            return existing

        # Create new alert context
        alert_ctx = AlertContext(
            alert=alert,
            correlation_id=correlation_id,
            fingerprint=fingerprint,
        )

        # Apply pattern tags
        for pattern in self.patterns:
            if pattern.matches(alert):
                alert_ctx.tags.update(pattern.tags)
                self.stats["by_pattern"][pattern.name] += 1

        # Add to correlation group
        if correlation_id != "uncorrelated":
            self.correlation_groups[correlation_id].append(fingerprint)
            related = self.correlation_groups[correlation_id]
            if len(related) > 1:
                alert_ctx.related_alerts = [fp for fp in related if fp != fingerprint]
                self.stats["correlated"] += 1
                logger.info(f"Alert correlated with {len(related) - 1} related alerts")

        # Store active alert
        self.active_alerts[fingerprint] = alert_ctx

        # Send notification (with intelligent filtering)
        if await self._should_notify(alert_ctx):
            await self._send_enriched_notification(alert_ctx)

        # Schedule auto-resolution
        asyncio.create_task(self._auto_resolve_alert(fingerprint))

        return alert_ctx

    async def _should_notify(self, alert_ctx: AlertContext) -> bool:
        """Determine if notification should be sent.

        Args:
            alert_ctx: Alert context

        Returns:
            True if notification should be sent
        """
        # Always notify for critical alerts
        if alert_ctx.alert.severity == AlertSeverity.CRITICAL:
            return True

        # Suppress if similar alert recently sent
        if alert_ctx.occurrence_count > 1 and alert_ctx.occurrence_count < 5:
            return False

        # Suppress low-priority alerts during high activity
        active_critical = sum(
            1
            for a in self.active_alerts.values()
            if a.alert.severity == AlertSeverity.CRITICAL
        )
        if active_critical > 0 and alert_ctx.alert.severity == AlertSeverity.INFO:
            return False

        # Notify for escalated or correlated alerts
        if alert_ctx.occurrence_count >= 5 or len(alert_ctx.related_alerts) > 2:
            return True

        return True

    async def _send_enriched_notification(self, alert_ctx: AlertContext) -> None:
        """Send enriched notification with context.

        Args:
            alert_ctx: Alert context
        """
        # Enrich alert message
        enriched_message = alert_ctx.alert.message

        # Add occurrence information
        if alert_ctx.occurrence_count > 1:
            enriched_message += f"\n\n📊 Occurred {alert_ctx.occurrence_count} times"

        # Add correlation information
        if alert_ctx.related_alerts:
            enriched_message += (
                f"\n🔗 Related to {len(alert_ctx.related_alerts)} other alerts"
            )

        # Add tags
        if alert_ctx.tags:
            enriched_message += f"\n🏷️ Tags: {', '.join(alert_ctx.tags)}"

        # Add recommended actions
        actions = self._get_recommended_actions(alert_ctx)
        if actions:
            enriched_message += "\n\n📋 Recommended Actions:"
            for action in actions:
                enriched_message += f"\n• {action}"

        # Create enriched alert
        enriched_alert = Alert(
            title=alert_ctx.alert.title,
            message=enriched_message,
            severity=alert_ctx.alert.severity,
            source=alert_ctx.alert.source,
            timestamp=alert_ctx.alert.timestamp,
            metadata={
                **alert_ctx.alert.metadata,
                "correlation_id": alert_ctx.correlation_id,
                "fingerprint": alert_ctx.fingerprint,
            },
        )

        # Send via standard alert manager
        await alert_manager.send_alert(enriched_alert)

    def _get_recommended_actions(self, alert_ctx: AlertContext) -> List[str]:
        """Get recommended actions for an alert.

        Args:
            alert_ctx: Alert context

        Returns:
            List of recommended actions
        """
        actions = []

        # Pattern-specific recommendations
        if "testing" in alert_ctx.tags:
            actions.append("Review recent code changes")
            actions.append("Run tests locally to reproduce")
            if alert_ctx.occurrence_count > 3:
                actions.append("Consider marking test as flaky")

        if "dependencies" in alert_ctx.tags:
            actions.append("Check dependency versions")
            actions.append("Clear cache and reinstall dependencies")
            actions.append("Review recent dependency updates")

        if "timeout" in alert_ctx.tags:
            actions.append("Increase timeout limits")
            actions.append("Check for performance regression")
            actions.append("Review resource utilization")

        if "network" in alert_ctx.tags:
            actions.append("Check network connectivity")
            actions.append("Verify external service status")
            actions.append("Review firewall/proxy settings")

        if "deployment" in alert_ctx.tags:
            actions.append("Check deployment logs")
            actions.append("Verify configuration changes")
            if alert_ctx.alert.severity == AlertSeverity.CRITICAL:
                actions.append("Consider rollback to previous version")

        # Severity-based recommendations
        if alert_ctx.alert.severity == AlertSeverity.CRITICAL:
            actions.insert(0, "IMMEDIATE ACTION REQUIRED")
            actions.append("Notify on-call engineer")

        # Recurrence-based recommendations
        if alert_ctx.occurrence_count > 5:
            actions.append("Create incident ticket for recurring issue")
            actions.append("Schedule root cause analysis")

        return actions

    async def _auto_resolve_alert(self, fingerprint: str) -> None:
        """Auto-resolve alert after timeout.

        Args:
            fingerprint: Alert fingerprint
        """
        await asyncio.sleep(self.auto_resolve_timeout)

        if fingerprint in self.active_alerts:
            alert_ctx = self.active_alerts[fingerprint]

            # Check if alert is still active (no recent occurrences)
            time_since_last = (datetime.now() - alert_ctx.last_seen).total_seconds()

            if time_since_last >= self.auto_resolve_timeout:
                alert_ctx.auto_resolved = True
                alert_ctx.resolution_time = datetime.now()

                # Move to history
                self.alert_history.append(alert_ctx)
                del self.active_alerts[fingerprint]

                self.stats["auto_resolved"] += 1
                logger.info(f"Alert auto-resolved: {alert_ctx.alert.title}")

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert activity.

        Returns:
            Alert activity summary
        """
        return {
            "active_alerts": len(self.active_alerts),
            "total_processed": self.stats["total_alerts"],
            "deduplicated": self.stats["deduplicated"],
            "correlated": self.stats["correlated"],
            "auto_resolved": self.stats["auto_resolved"],
            "by_severity": dict(self.stats["by_severity"]),
            "by_pattern": dict(self.stats["by_pattern"]),
            "correlation_groups": len(self.correlation_groups),
            "top_alerts": [
                {
                    "title": ctx.alert.title,
                    "occurrences": ctx.occurrence_count,
                    "severity": ctx.alert.severity.value,
                }
                for ctx in sorted(
                    self.active_alerts.values(),
                    key=lambda x: x.occurrence_count,
                    reverse=True,
                )[:5]
            ],
        }


# Create singleton intelligent alert manager
intelligent_alerts = IntelligentAlertManager()
