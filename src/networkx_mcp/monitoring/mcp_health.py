"""MCP-specific health monitoring and observability.

This module provides comprehensive monitoring capabilities specifically designed
for MCP (Model Context Protocol) servers, including transport metrics, tool
invocation patterns, and session management.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class MCPMetric:
    """Structured MCP metric for dashboard export."""

    timestamp: float
    metric_name: str
    value: float
    tags: Dict[str, str]
    tool_name: str = ""
    session_id: str = ""
    transport_type: str = "stdio"


class MCPHealthMonitor:
    """MCP-specific health monitoring with protocol-aware metrics."""

    def __init__(self):
        """Initialize MCP health monitor."""
        self.start_time = time.time()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.tool_invocations: deque = deque(maxlen=10000)
        self.transport_metrics = defaultdict(lambda: {"count": 0, "errors": 0})
        self.permission_denials = 0
        self.protocol_errors = 0
        self.tool_error_counts = defaultdict(int)
        self.tool_success_counts = defaultdict(int)
        self.session_durations: deque = deque(maxlen=1000)

    async def get_mcp_health_status(self) -> Dict[str, Any]:
        """Get comprehensive MCP health status including transport metrics.

        Returns:
            Dict containing MCP-specific health metrics
        """
        uptime = time.time() - self.start_time

        health_status = {
            "status": self._determine_health_status(),
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": uptime,
            "mcp_metrics": {
                "transport": {
                    "active_sessions": len(self.active_sessions),
                    "protocol_version": "2024-11-05",
                    "supported_transports": ["stdio", "http+sse", "streamable_http"],
                    "session_cleanup_rate": self._calculate_cleanup_rate(),
                    "transport_breakdown": dict(self.transport_metrics),
                },
                "tools": {
                    "total_invocations": len(self.tool_invocations),
                    "unique_tools_used": len(
                        set(t.get("tool") for t in self.tool_invocations)
                    ),
                    "tool_error_rates": self._calculate_tool_error_rates(),
                    "most_used_tools": self._get_most_used_tools(),
                    "avg_tool_latency": self._calculate_avg_tool_latency(),
                },
                "security": {
                    "auth_enabled": True,  # Configurable
                    "active_tokens": len(self.active_sessions),
                    "permission_denials": self.permission_denials,
                    "protocol_errors": self.protocol_errors,
                },
                "performance": {
                    "avg_session_duration": self._calculate_avg_session_duration(),
                    "p95_session_duration": self._calculate_percentile_duration(95),
                    "tool_throughput": self._calculate_tool_throughput(),
                },
            },
        }

        return health_status

    def record_session_start(
        self, session_id: str, transport_type: str = "stdio", metadata: Dict = None
    ):
        """Record the start of a new MCP session.

        Args:
            session_id: Unique session identifier
            transport_type: Type of transport (stdio, http+sse, etc.)
            metadata: Additional session metadata
        """
        self.active_sessions[session_id] = {
            "start_time": time.time(),
            "transport_type": transport_type,
            "tool_invocations": 0,
            "metadata": metadata or {},
        }
        self.transport_metrics[transport_type]["count"] += 1
        logger.info(f"Session started: {session_id} ({transport_type})")

    def record_session_end(self, session_id: str):
        """Record the end of an MCP session.

        Args:
            session_id: Session identifier to end
        """
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            duration = time.time() - session["start_time"]
            self.session_durations.append(duration)
            del self.active_sessions[session_id]
            logger.info(f"Session ended: {session_id} (duration: {duration:.2f}s)")

    def record_tool_invocation(
        self,
        tool_name: str,
        session_id: str,
        duration: float,
        success: bool,
        metadata: Dict = None,
    ):
        """Record a tool invocation with detailed metrics.

        Args:
            tool_name: Name of the tool invoked
            session_id: Session that invoked the tool
            duration: Execution duration in seconds
            success: Whether the invocation succeeded
            metadata: Additional invocation metadata
        """
        invocation = {
            "timestamp": time.time(),
            "tool": tool_name,
            "session_id": session_id,
            "duration": duration,
            "success": success,
            "metadata": metadata or {},
        }
        self.tool_invocations.append(invocation)

        if success:
            self.tool_success_counts[tool_name] += 1
        else:
            self.tool_error_counts[tool_name] += 1

        # Update session tool count
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["tool_invocations"] += 1

    def record_permission_denial(self, session_id: str, tool_name: str, reason: str):
        """Record a permission denial event.

        Args:
            session_id: Session that was denied
            tool_name: Tool that was denied
            reason: Reason for denial
        """
        self.permission_denials += 1
        logger.warning(
            f"Permission denied - Session: {session_id}, Tool: {tool_name}, Reason: {reason}"
        )

    def record_protocol_error(self, error_type: str, details: str):
        """Record a protocol-level error.

        Args:
            error_type: Type of protocol error
            details: Error details
        """
        self.protocol_errors += 1
        logger.error(f"Protocol error - Type: {error_type}, Details: {details}")

    def _determine_health_status(self) -> str:
        """Determine overall health status based on metrics.

        Returns:
            Health status string (healthy, degraded, critical)
        """
        # Calculate error rate
        total_tools = sum(self.tool_success_counts.values()) + sum(
            self.tool_error_counts.values()
        )
        if total_tools > 0:
            error_rate = sum(self.tool_error_counts.values()) / total_tools
            if error_rate > 0.1:  # >10% error rate
                return "critical"
            elif error_rate > 0.05:  # >5% error rate
                return "degraded"

        # Check protocol errors
        if self.protocol_errors > 10:
            return "degraded"

        # Check permission denials
        if self.permission_denials > 50:
            return "degraded"

        return "healthy"

    def _calculate_cleanup_rate(self) -> float:
        """Calculate session cleanup rate.

        Returns:
            Sessions cleaned up per hour
        """
        if not self.session_durations:
            return 0.0

        # Count sessions ended in last hour
        recent_sessions = sum(1 for _ in self.session_durations)
        return recent_sessions  # Simplified for now

    def _calculate_tool_error_rates(self) -> Dict[str, float]:
        """Calculate error rates for each tool.

        Returns:
            Dict mapping tool names to error rates
        """
        rates = {}
        for tool in set(
            list(self.tool_success_counts.keys()) + list(self.tool_error_counts.keys())
        ):
            success = self.tool_success_counts.get(tool, 0)
            errors = self.tool_error_counts.get(tool, 0)
            total = success + errors
            if total > 0:
                rates[tool] = errors / total
        return rates

    def _get_most_used_tools(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most frequently used tools.

        Args:
            limit: Number of tools to return

        Returns:
            List of tool usage statistics
        """
        tool_counts = defaultdict(int)
        for invocation in self.tool_invocations:
            tool_counts[invocation["tool"]] += 1

        sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]

        return [{"tool": tool, "invocations": count} for tool, count in sorted_tools]

    def _calculate_avg_tool_latency(self) -> float:
        """Calculate average tool invocation latency.

        Returns:
            Average latency in milliseconds
        """
        if not self.tool_invocations:
            return 0.0

        total_duration = sum(t["duration"] for t in self.tool_invocations)
        return (total_duration / len(self.tool_invocations)) * 1000  # Convert to ms

    def _calculate_avg_session_duration(self) -> float:
        """Calculate average session duration.

        Returns:
            Average duration in seconds
        """
        if not self.session_durations:
            # Include active sessions
            if self.active_sessions:
                current_time = time.time()
                durations = [
                    current_time - s["start_time"]
                    for s in self.active_sessions.values()
                ]
                return sum(durations) / len(durations)
            return 0.0

        return sum(self.session_durations) / len(self.session_durations)

    def _calculate_percentile_duration(self, percentile: int) -> float:
        """Calculate percentile session duration.

        Args:
            percentile: Percentile to calculate (0-100)

        Returns:
            Duration at the given percentile
        """
        if not self.session_durations:
            return 0.0

        sorted_durations = sorted(self.session_durations)
        index = int((percentile / 100) * len(sorted_durations))
        return sorted_durations[min(index, len(sorted_durations) - 1)]

    def _calculate_tool_throughput(self) -> float:
        """Calculate tool invocations per minute.

        Returns:
            Tool invocations per minute
        """
        if not self.tool_invocations:
            return 0.0

        # Look at last 5 minutes of invocations
        five_min_ago = time.time() - 300
        recent_invocations = sum(
            1 for t in self.tool_invocations if t["timestamp"] > five_min_ago
        )

        return recent_invocations / 5  # Per minute

    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []
        timestamp = int(time.time() * 1000)

        # Active sessions gauge
        lines.append(
            f'mcp_active_sessions{{server="networkx"}} {len(self.active_sessions)} {timestamp}'
        )

        # Tool invocation counters
        for tool, count in self.tool_success_counts.items():
            lines.append(
                f'mcp_tool_invocations_total{{tool="{tool}",status="success"}} {count} {timestamp}'
            )
        for tool, count in self.tool_error_counts.items():
            lines.append(
                f'mcp_tool_invocations_total{{tool="{tool}",status="error"}} {count} {timestamp}'
            )

        # Permission denials counter
        lines.append(
            f'mcp_permission_denials_total{{server="networkx"}} {self.permission_denials} {timestamp}'
        )

        # Protocol errors counter
        lines.append(
            f'mcp_protocol_errors_total{{server="networkx"}} {self.protocol_errors} {timestamp}'
        )

        # Average tool latency
        avg_latency = self._calculate_avg_tool_latency()
        lines.append(
            f'mcp_tool_latency_milliseconds{{server="networkx"}} {avg_latency} {timestamp}'
        )

        # Session duration histogram (simplified)
        if self.session_durations:
            for percentile in [50, 95, 99]:
                duration = self._calculate_percentile_duration(percentile)
                lines.append(
                    f'mcp_session_duration_seconds{{quantile="{percentile/100}"}} {duration} {timestamp}'
                )

        return "\n".join(lines)


# Global instance for easy access
mcp_health_monitor = MCPHealthMonitor()
