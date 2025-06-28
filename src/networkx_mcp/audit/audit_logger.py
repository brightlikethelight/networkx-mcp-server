"""Comprehensive audit logging for compliance and security."""

import asyncio
import json
import uuid

from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any
from typing import Dict
from typing import Optional

from ..storage.base import StorageBackend
from ..storage.base import Transaction


# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="anonymous")
client_ip_var: ContextVar[str] = ContextVar("client_ip", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")


class AuditEvent:
    """Structured audit event."""

    def __init__(
        self,
        action: str,
        user_id: str,
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        self.event_id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.action = action
        self.user_id = user_id
        self.status = status
        self.details = details or {}
        self.error = error

        # Get context
        self.request_id = request_id_var.get() or str(uuid.uuid4())
        self.client_ip = client_ip_var.get() or "unknown"
        self.session_id = session_id_var.get() or "unknown"

        # System info
        self.server_version = "0.1.0"  # TODO: Get from config
        self.server_node = "server-1"  # TODO: Get from environment

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "user_id": self.user_id,
            "status": self.status,
            "details": self.details,
            "error": self.error,
            "request_id": self.request_id,
            "client_ip": self.client_ip,
            "session_id": self.session_id,
            "server_version": self.server_version,
            "server_node": self.server_node
        }

    def to_json(self) -> str:
        """Convert to JSON for storage."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


class SecurityAlert:
    """Security alert for suspicious activity."""

    SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

    def __init__(
        self,
        alert_type: str,
        severity: str,
        user_id: str,
        description: str,
        evidence: Dict[str, Any]
    ):
        self.alert_id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.alert_type = alert_type
        self.severity = severity
        self.user_id = user_id
        self.description = description
        self.evidence = evidence
        self.status = "new"
        self.assigned_to = None
        self.resolution = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "user_id": self.user_id,
            "description": self.description,
            "evidence": self.evidence,
            "status": self.status,
            "assigned_to": self.assigned_to,
            "resolution": self.resolution
        }


class AuditLogger:
    """Comprehensive audit logging for compliance and security monitoring."""

    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self._alert_handlers = []
        self._metrics = defaultdict(int)
        self._user_activity = defaultdict(lambda: defaultdict(int))
        self._lock = asyncio.Lock()

    async def log_event(
        self,
        action: str,
        user_id: str,
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        tx: Optional[Transaction] = None
    ) -> str:
        """Log an audit event."""
        # Create event
        event = AuditEvent(
            action=action,
            user_id=user_id,
            status=status,
            details=details,
            error=error
        )

        # Store event
        await self._store_event(event, tx)

        # Update metrics
        async with self._lock:
            self._metrics[f"action_{action}_{status}"] += 1
            self._user_activity[user_id][action] += 1

        # Check for suspicious activity
        if await self._is_suspicious(event):
            await self._raise_security_alert(event)

        return event.event_id

    async def _store_event(self, event: AuditEvent, tx: Optional[Transaction] = None):
        """Store audit event."""
        # Store in time-based buckets for efficient querying
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        hour_str = datetime.now(timezone.utc).strftime("%H")

        # Keys for different access patterns
        keys = [
            f"audit:events:{date_str}:{hour_str}",  # Time-based
            f"audit:user:{event.user_id}:{date_str}",  # User-based
            f"audit:action:{event.action}:{date_str}",  # Action-based
        ]

        # Store in all indexes
        event_json = event.to_json()

        if hasattr(self.storage, "_client"):  # Redis backend
            client = tx.pipeline if tx else self.storage._client
            for key in keys:
                await client.zadd(
                    key,
                    {event_json: datetime.now(timezone.utc).timestamp()}
                )
                # Set TTL (90 days)
                await client.expire(key, 90 * 24 * 3600)
        else:
            # Fallback for other storage backends
            # Store as metadata
            await self.storage.update_graph_metadata(
                "system",
                f"audit_{event.event_id}",
                event.to_dict(),
                tx
            )

    async def _is_suspicious(self, event: AuditEvent) -> bool:
        """Detect suspicious activity patterns."""
        user_id = event.user_id
        action = event.action

        # Get recent activity
        recent_activity = self._user_activity[user_id]

        suspicious_patterns = [
            # Rapid deletion
            (action == "delete_graph" and
             recent_activity.get("delete_graph", 0) > 10),

            # Mass data export
            (action == "export_graph" and
             recent_activity.get("export_graph", 0) > 20),

            # Failed auth attempts
            (action == "auth_failed" and
             recent_activity.get("auth_failed", 0) > 5),

            # Privilege escalation attempts
            (action in ["modify_permissions", "access_admin"] and
             event.status == "failed"),

            # Large graph operations
            (action in ["import_graph", "create_graph"] and
             event.details.get("size_mb", 0) > 50),

            # Unusual file paths
            (action in ["import_graph", "export_graph"] and
             any(pattern in str(event.details.get("filepath", ""))
                 for pattern in ["../", "/etc/", "/root/", "/proc/"])),

            # Rate anomalies
            (sum(recent_activity.values()) > 1000),  # 1000 actions recently

            # Time anomalies (activity at unusual hours)
            (datetime.now(timezone.utc).hour in [1, 2, 3, 4] and
             sum(recent_activity.values()) > 100),
        ]

        return any(suspicious_patterns)

    async def _raise_security_alert(self, event: AuditEvent):
        """Raise security alert for suspicious activity."""
        # Determine alert type and severity
        if event.action == "auth_failed":
            alert_type = "brute_force_attempt"
            severity = "high" if self._user_activity[event.user_id]["auth_failed"] > 10 else "medium"
        elif "delete" in event.action:
            alert_type = "mass_deletion"
            severity = "high"
        elif "export" in event.action:
            alert_type = "data_exfiltration"
            severity = "critical" if event.details.get("size_mb", 0) > 100 else "high"
        elif "../" in str(event.details.get("filepath", "")):
            alert_type = "path_traversal_attempt"
            severity = "critical"
        else:
            alert_type = "anomalous_activity"
            severity = "medium"

        # Create alert
        alert = SecurityAlert(
            alert_type=alert_type,
            severity=severity,
            user_id=event.user_id,
            description=f"Suspicious {event.action} activity detected",
            evidence={
                "event": event.to_dict(),
                "recent_activity": dict(self._user_activity[event.user_id]),
                "total_actions": sum(self._user_activity[event.user_id].values())
            }
        )

        # Store alert
        await self._store_alert(alert)

        # Notify handlers
        for handler in self._alert_handlers:
            try:
                await handler(alert)
            except Exception:
                # Log but don't fail
                pass

    async def _store_alert(self, alert: SecurityAlert):
        """Store security alert."""
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

        if hasattr(self.storage, "_client"):  # Redis backend
            # Store in sorted set by severity
            severity_score = {
                "low": 1, "medium": 2, "high": 3, "critical": 4
            }.get(alert.severity, 2)

            await self.storage._client.zadd(
                f"audit:alerts:{date_str}",
                {json.dumps(alert.to_dict()): severity_score}
            )

            # Also store by user
            await self.storage._client.zadd(
                f"audit:alerts:user:{alert.user_id}",
                {json.dumps(alert.to_dict()): datetime.now(timezone.utc).timestamp()}
            )

    def add_alert_handler(self, handler: callable):
        """Add handler for security alerts."""
        self._alert_handlers.append(handler)

    async def get_user_activity_summary(
        self,
        user_id: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get user activity summary."""
        # Get from cache
        recent = dict(self._user_activity.get(user_id, {}))

        # Get from storage for longer period
        if hasattr(self.storage, "_client"):
            now = datetime.now(timezone.utc)
            stored_activity = defaultdict(int)

            for h in range(hours):
                check_time = now - timedelta(hours=h)
                date_str = check_time.strftime("%Y%m%d")

                # Get events
                key = f"audit:user:{user_id}:{date_str}"
                events = await self.storage._client.zrange(key, 0, -1)

                for event_json in events:
                    try:
                        event_data = json.loads(event_json)
                        stored_activity[event_data["action"]] += 1
                    except:
                        continue
        else:
            stored_activity = recent

        # Calculate risk score
        risk_score = self._calculate_risk_score(stored_activity)

        return {
            "user_id": user_id,
            "period_hours": hours,
            "activity": dict(stored_activity),
            "total_actions": sum(stored_activity.values()),
            "risk_score": risk_score,
            "risk_level": self._risk_level(risk_score)
        }

    def _calculate_risk_score(self, activity: Dict[str, int]) -> float:
        """Calculate user risk score (0-100)."""
        score = 0.0

        # High-risk actions
        high_risk = {
            "delete_graph": 5.0,
            "export_graph": 3.0,
            "auth_failed": 10.0,
            "modify_permissions": 8.0,
            "import_graph": 2.0
        }

        for action, weight in high_risk.items():
            count = activity.get(action, 0)
            if count > 0:
                # Exponential increase for repeated actions
                score += weight * (1.5 ** min(count, 10))

        # Volume-based risk
        total = sum(activity.values())
        if total > 1000:
            score += 20
        elif total > 500:
            score += 10
        elif total > 200:
            score += 5

        return min(score, 100.0)

    def _risk_level(self, score: float) -> str:
        """Convert risk score to level."""
        if score >= 80:
            return "critical"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "medium"
        elif score >= 20:
            return "low"
        else:
            return "minimal"

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide audit metrics."""
        return {
            "total_events": sum(self._metrics.values()),
            "events_by_action": dict(self._metrics),
            "unique_users": len(self._user_activity),
            "high_risk_users": [
                user_id for user_id in self._user_activity
                if self._calculate_risk_score(self._user_activity[user_id]) >= 60
            ]
        }

    async def cleanup_old_events(self, days: int = 90):
        """Clean up old audit events."""
        # This would be called by a scheduled job
        # Redis handles TTL automatically
        pass


# Decorators for audit logging
def audit_log(action: str):
    """Decorator to automatically log actions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user_id
            user_id = kwargs.get("user_id", "anonymous")
            if not user_id and args:
                user_id = args[0] if isinstance(args[0], str) else "anonymous"

            # Get audit logger from somewhere (dependency injection)
            # This is a simplified example
            audit_logger = kwargs.get("_audit_logger")

            try:
                result = await func(*args, **kwargs)

                if audit_logger:
                    await audit_logger.log_event(
                        action=action,
                        user_id=user_id,
                        status="success",
                        details={"args": str(args)[:200], "kwargs": str(kwargs)[:200]}
                    )

                return result

            except Exception as e:
                if audit_logger:
                    await audit_logger.log_event(
                        action=action,
                        user_id=user_id,
                        status="failed",
                        error=str(e),
                        details={"args": str(args)[:200], "kwargs": str(kwargs)[:200]}
                    )
                raise

        return wrapper
    return decorator
