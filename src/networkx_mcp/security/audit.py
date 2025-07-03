"""Security audit logging and monitoring.

This module provides comprehensive audit logging, security event monitoring,
and compliance reporting for the NetworkX MCP Server.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""

    # Authentication events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_CREATED = "auth.token.created"
    AUTH_TOKEN_REVOKED = "auth.token.revoked"

    # Authorization events
    AUTHZ_PERMISSION_GRANTED = "authz.permission.granted"
    AUTHZ_PERMISSION_DENIED = "authz.permission.denied"
    AUTHZ_ROLE_ASSIGNED = "authz.role.assigned"
    AUTHZ_ROLE_REMOVED = "authz.role.removed"

    # Data access events
    DATA_ACCESS_READ = "data.access.read"
    DATA_ACCESS_WRITE = "data.access.write"
    DATA_ACCESS_DELETE = "data.access.delete"
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"

    # Graph operation events
    GRAPH_CREATED = "graph.created"
    GRAPH_MODIFIED = "graph.modified"
    GRAPH_DELETED = "graph.deleted"
    GRAPH_ANALYZED = "graph.analyzed"

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    SYSTEM_CONFIG_CHANGED = "system.config.changed"

    # Security events
    SECURITY_THREAT_DETECTED = "security.threat.detected"
    SECURITY_RATE_LIMIT_EXCEEDED = "security.rate_limit.exceeded"
    SECURITY_SUSPICIOUS_ACTIVITY = "security.suspicious.activity"
    SECURITY_POLICY_VIOLATION = "security.policy.violation"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents a single audit event."""

    event_id: str
    event_type: AuditEventType
    timestamp: float
    severity: AuditSeverity

    # Who/What
    user_id: str | None = None
    client_id: str | None = None
    session_id: str | None = None
    source_ip: str | None = None
    user_agent: str | None = None

    # Where/When
    component: str | None = None
    operation: str | None = None
    resource: str | None = None

    # What happened
    description: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    # Result
    success: bool = True
    error_message: str | None = None

    # Context
    request_id: str | None = None
    correlation_id: str | None = None

    # Compliance
    regulation_tags: list[str] = field(default_factory=list)
    retention_policy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
        }


class AuditStorage:
    """Abstract base class for audit storage backends."""

    async def store_event(self, event: AuditEvent) -> bool:
        """Store an audit event."""
        raise NotImplementedError

    async def query_events(
        self, filters: dict[str, Any], limit: int = 100
    ) -> list[AuditEvent]:
        """Query audit events with filters."""
        raise NotImplementedError

    async def get_event_count(self, filters: dict[str, Any]) -> int:
        """Get count of events matching filters."""
        raise NotImplementedError


class FileAuditStorage(AuditStorage):
    """File-based audit storage."""

    def __init__(self, log_file: str | Path):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    async def store_event(self, event: AuditEvent) -> bool:
        """Store event to log file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                json.dump(event.to_dict(), f)
                f.write("\n")
            return True
        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")
            return False

    async def query_events(
        self, filters: dict[str, Any], limit: int = 100
    ) -> list[AuditEvent]:
        """Query events from log file."""
        events = []
        try:
            if not self.log_file.exists():
                return events

            with open(self.log_file, encoding="utf-8") as f:
                for line in f:
                    if len(events) >= limit:
                        break
                    try:
                        data = json.loads(line.strip())
                        if self._matches_filters(data, filters):
                            event = self._dict_to_event(data)
                            events.append(event)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")

        return events[-limit:]  # Return most recent events

    async def get_event_count(self, filters: dict[str, Any]) -> int:
        """Get count of matching events."""
        count = 0
        try:
            if not self.log_file.exists():
                return 0

            with open(self.log_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if self._matches_filters(data, filters):
                            count += 1
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Failed to count audit events: {e}")

        return count

    def _matches_filters(self, data: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if event data matches filters."""
        for key, value in filters.items():
            if key not in data:
                return False
            if data[key] != value:
                return False
        return True

    def _dict_to_event(self, data: dict[str, Any]) -> AuditEvent:
        """Convert dictionary back to AuditEvent."""
        # Convert enum values back
        data["event_type"] = AuditEventType(data["event_type"])
        data["severity"] = AuditSeverity(data["severity"])

        return AuditEvent(**data)


class MemoryAuditStorage(AuditStorage):
    """In-memory audit storage for testing/development."""

    def __init__(self, max_events: int = 10000):
        self.events: list[AuditEvent] = []
        self.max_events = max_events

    async def store_event(self, event: AuditEvent) -> bool:
        """Store event in memory."""
        self.events.append(event)

        # Rotate if too many events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

        return True

    async def query_events(
        self, filters: dict[str, Any], limit: int = 100
    ) -> list[AuditEvent]:
        """Query events from memory."""
        matching_events = []

        for event in reversed(self.events):  # Most recent first
            if len(matching_events) >= limit:
                break

            if self._event_matches_filters(event, filters):
                matching_events.append(event)

        return matching_events

    async def get_event_count(self, filters: dict[str, Any]) -> int:
        """Get count of matching events."""
        count = 0
        for event in self.events:
            if self._event_matches_filters(event, filters):
                count += 1
        return count

    def _event_matches_filters(
        self, event: AuditEvent, filters: dict[str, Any]
    ) -> bool:
        """Check if event matches filters."""
        for key, value in filters.items():
            event_value = getattr(event, key, None)
            if event_value != value:
                return False
        return True


class AuditLogger:
    """Main audit logging system."""

    def __init__(
        self, storage: AuditStorage, buffer_size: int = 100, flush_interval: float = 5.0
    ):
        self.storage = storage
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self.event_buffer: list[AuditEvent] = []
        self.stats = {"events_logged": 0, "events_failed": 0, "last_flush": time.time()}

        # Start background flush task
        self._flush_task = None
        self._running = False

    async def start(self):
        """Start the audit logger."""
        if not self._running:
            self._running = True
            self._flush_task = asyncio.create_task(self._periodic_flush())

    async def stop(self):
        """Stop the audit logger and flush remaining events."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining events
        await self._flush_events()

    def generate_event_id(self) -> str:
        """Generate unique event ID."""
        return hashlib.md5(f"{time.time()}{id(self)}".encode()).hexdigest()[:16]

    async def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        **kwargs,
    ) -> str:
        """Log an audit event."""
        event_id = self.generate_event_id()

        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=time.time(),
            severity=severity,
            **kwargs,
        )

        # Add to buffer
        self.event_buffer.append(event)

        # Flush if buffer is full
        if len(self.event_buffer) >= self.buffer_size:
            await self._flush_events()

        logger.debug(f"Audit event logged: {event_type.value} ({event_id})")
        return event_id

    async def log_authentication_event(
        self,
        success: bool,
        user_id: str,
        source_ip: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Log authentication event."""
        event_type = (
            AuditEventType.AUTH_LOGIN_SUCCESS
            if success
            else AuditEventType.AUTH_LOGIN_FAILURE
        )
        severity = AuditSeverity.MEDIUM if success else AuditSeverity.HIGH

        await self.log_event(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            source_ip=source_ip,
            success=success,
            details=details or {},
            regulation_tags=["authentication", "access_control"],
        )

    async def log_authorization_event(
        self,
        granted: bool,
        user_id: str,
        operation: str,
        resource: str,
        details: dict[str, Any] | None = None,
    ):
        """Log authorization event."""
        event_type = (
            AuditEventType.AUTHZ_PERMISSION_GRANTED
            if granted
            else AuditEventType.AUTHZ_PERMISSION_DENIED
        )
        severity = AuditSeverity.LOW if granted else AuditSeverity.HIGH

        await self.log_event(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            operation=operation,
            resource=resource,
            success=granted,
            details=details or {},
            regulation_tags=["authorization", "access_control"],
        )

    async def log_data_access_event(
        self,
        operation: str,
        resource: str,
        user_id: str,
        success: bool = True,
        details: dict[str, Any] | None = None,
    ):
        """Log data access event."""
        event_type_map = {
            "read": AuditEventType.DATA_ACCESS_READ,
            "write": AuditEventType.DATA_ACCESS_WRITE,
            "delete": AuditEventType.DATA_ACCESS_DELETE,
            "export": AuditEventType.DATA_EXPORT,
            "import": AuditEventType.DATA_IMPORT,
        }

        event_type = event_type_map.get(
            operation.lower(), AuditEventType.DATA_ACCESS_READ
        )
        severity = AuditSeverity.MEDIUM if operation in ["read"] else AuditSeverity.HIGH

        await self.log_event(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            operation=operation,
            resource=resource,
            success=success,
            details=details or {},
            regulation_tags=["data_access", "privacy"],
        )

    async def log_security_event(
        self,
        threat_type: str,
        severity: AuditSeverity,
        source_ip: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Log security threat event."""
        await self.log_event(
            event_type=AuditEventType.SECURITY_THREAT_DETECTED,
            severity=severity,
            source_ip=source_ip,
            description=f"Security threat detected: {threat_type}",
            details=details or {},
            regulation_tags=["security", "incident_response"],
        )

    async def _flush_events(self):
        """Flush buffered events to storage."""
        if not self.event_buffer:
            return

        events_to_flush = self.event_buffer.copy()
        self.event_buffer.clear()

        success_count = 0
        for event in events_to_flush:
            try:
                success = await self.storage.store_event(event)
                if success:
                    success_count += 1
                else:
                    self.stats["events_failed"] += 1
            except Exception as e:
                logger.error(f"Failed to store audit event {event.event_id}: {e}")
                self.stats["events_failed"] += 1

        self.stats["events_logged"] += success_count
        self.stats["last_flush"] = time.time()

        logger.debug(f"Flushed {success_count}/{len(events_to_flush)} audit events")

    async def _periodic_flush(self):
        """Periodically flush events."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                if self.event_buffer:
                    await self._flush_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")

    async def query_events(
        self, filters: dict[str, Any] | None = None, limit: int = 100
    ) -> list[AuditEvent]:
        """Query audit events."""
        return await self.storage.query_events(filters or {}, limit)

    async def get_stats(self) -> dict[str, Any]:
        """Get audit logger statistics."""
        return {
            **self.stats,
            "buffered_events": len(self.event_buffer),
            "buffer_size": self.buffer_size,
            "flush_interval": self.flush_interval,
            "is_running": self._running,
        }


class ComplianceReporter:
    """Generate compliance reports from audit logs."""

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    async def generate_access_report(
        self, start_time: float, end_time: float
    ) -> dict[str, Any]:
        """Generate access control compliance report."""
        filters = {"timestamp": {"gte": start_time, "lte": end_time}}

        # Get all events in time range
        events = await self.audit_logger.query_events(filters, limit=10000)

        # Categorize events
        auth_events = [e for e in events if e.event_type.value.startswith("auth")]
        authz_events = [e for e in events if e.event_type.value.startswith("authz")]
        data_events = [e for e in events if e.event_type.value.startswith("data")]

        return {
            "report_type": "access_control",
            "period": {
                "start": start_time,
                "end": end_time,
                "duration_hours": (end_time - start_time) / 3600,
            },
            "summary": {
                "total_events": len(events),
                "authentication_events": len(auth_events),
                "authorization_events": len(authz_events),
                "data_access_events": len(data_events),
            },
            "authentication": {
                "successful_logins": len(
                    [
                        e
                        for e in auth_events
                        if e.event_type == AuditEventType.AUTH_LOGIN_SUCCESS
                    ]
                ),
                "failed_logins": len(
                    [
                        e
                        for e in auth_events
                        if e.event_type == AuditEventType.AUTH_LOGIN_FAILURE
                    ]
                ),
                "unique_users": len(set(e.user_id for e in auth_events if e.user_id)),
            },
            "authorization": {
                "permissions_granted": len([e for e in authz_events if e.success]),
                "permissions_denied": len([e for e in authz_events if not e.success]),
            },
            "data_access": {
                "read_operations": len(
                    [
                        e
                        for e in data_events
                        if e.event_type == AuditEventType.DATA_ACCESS_READ
                    ]
                ),
                "write_operations": len(
                    [
                        e
                        for e in data_events
                        if e.event_type == AuditEventType.DATA_ACCESS_WRITE
                    ]
                ),
                "delete_operations": len(
                    [
                        e
                        for e in data_events
                        if e.event_type == AuditEventType.DATA_ACCESS_DELETE
                    ]
                ),
            },
        }

    async def generate_security_report(
        self, start_time: float, end_time: float
    ) -> dict[str, Any]:
        """Generate security incident compliance report."""
        filters = {"timestamp": {"gte": start_time, "lte": end_time}}

        events = await self.audit_logger.query_events(filters, limit=10000)
        security_events = [
            e for e in events if e.event_type.value.startswith("security")
        ]

        return {
            "report_type": "security_incidents",
            "period": {"start": start_time, "end": end_time},
            "summary": {
                "total_security_events": len(security_events),
                "critical_events": len(
                    [e for e in security_events if e.severity == AuditSeverity.CRITICAL]
                ),
                "high_events": len(
                    [e for e in security_events if e.severity == AuditSeverity.HIGH]
                ),
            },
            "incidents": [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp,
                    "type": e.event_type.value,
                    "severity": e.severity.value,
                    "description": e.description,
                    "source_ip": e.source_ip,
                }
                for e in security_events
                if e.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]
            ],
        }


class SecurityEventLogger:
    """Specialized logger for security events."""

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    async def log_failed_login(self, user_id: str, source_ip: str, reason: str):
        """Log failed login attempt."""
        await self.audit_logger.log_security_event(
            threat_type="failed_login",
            severity=AuditSeverity.MEDIUM,
            source_ip=source_ip,
            details={"user_id": user_id, "reason": reason, "action": "login_failed"},
        )

    async def log_suspicious_activity(
        self, activity_type: str, details: dict[str, Any]
    ):
        """Log suspicious activity."""
        await self.audit_logger.log_security_event(
            threat_type=activity_type, severity=AuditSeverity.HIGH, details=details
        )

    async def log_access_violation(self, user_id: str, resource: str, action: str):
        """Log access violation."""
        await self.audit_logger.log_security_event(
            threat_type="access_violation",
            severity=AuditSeverity.HIGH,
            details={"user_id": user_id, "resource": resource, "action": action},
        )
