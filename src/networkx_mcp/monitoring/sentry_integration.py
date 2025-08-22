"""Sentry error tracking integration for NetworkX MCP Server.

Provides comprehensive error tracking, performance monitoring, and
distributed tracing capabilities through Sentry.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Check if Sentry SDK is available
try:
    import sentry_sdk
    from sentry_sdk import capture_exception, capture_message, set_context, set_tag
    from sentry_sdk.integrations.aiohttp import AioHttpIntegration
    from sentry_sdk.integrations.asyncio import AsyncioIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration

    HAS_SENTRY = True
except ImportError:
    HAS_SENTRY = False
    sentry_sdk = None

    def capture_exception(e):
        """No-op when Sentry not available."""
        pass

    def capture_message(m, level=None):
        """No-op when Sentry not available."""
        pass

    def set_context(name, context):
        """No-op when Sentry not available."""
        pass

    def set_tag(key, value):
        """No-op when Sentry not available."""
        pass


class SentryIntegration:
    """Manages Sentry error tracking and performance monitoring."""

    def __init__(
        self,
        dsn: Optional[str] = None,
        environment: str = "production",
        release: Optional[str] = None,
        sample_rate: float = 1.0,
        traces_sample_rate: float = 0.1,
        profiles_sample_rate: float = 0.1,
        debug: bool = False,
    ):
        """Initialize Sentry integration.

        Args:
            dsn: Sentry DSN (Data Source Name). If None, uses SENTRY_DSN env var
            environment: Environment name (production, staging, development)
            release: Release version. If None, auto-detected from package
            sample_rate: Error sampling rate (0.0 to 1.0)
            traces_sample_rate: Transaction sampling rate for performance monitoring
            profiles_sample_rate: Profiling sample rate
            debug: Enable debug mode
        """
        self.enabled = False

        if not HAS_SENTRY:
            logger.warning("sentry-sdk not installed - error tracking disabled")
            return

        # Get DSN from parameter or environment
        dsn = dsn or os.getenv("SENTRY_DSN")
        if not dsn:
            logger.info("Sentry DSN not configured - error tracking disabled")
            return

        # Auto-detect release version if not provided
        if not release:
            try:
                from networkx_mcp import __version__

                release = f"networkx-mcp-server@{__version__}"
            except ImportError:
                release = "networkx-mcp-server@unknown"

        try:
            # Initialize Sentry SDK
            sentry_sdk.init(
                dsn=dsn,
                environment=environment,
                release=release,
                sample_rate=sample_rate,
                traces_sample_rate=traces_sample_rate,
                profiles_sample_rate=profiles_sample_rate,
                debug=debug,
                integrations=[
                    LoggingIntegration(
                        level=logging.INFO,  # Capture info and above
                        event_level=logging.ERROR,  # Send errors as events
                    ),
                    AsyncioIntegration(),
                    AioHttpIntegration(),
                ],
                # Additional options
                attach_stacktrace=True,
                send_default_pii=False,  # Don't send personally identifiable information
                before_send=self._before_send,
                before_send_transaction=self._before_send_transaction,
            )

            self.enabled = True
            logger.info(f"Sentry initialized: {environment} ({release})")

            # Set initial tags
            set_tag("server.type", "mcp")
            set_tag("server.name", "networkx")
            set_tag("mcp.version", "2024-11-05")

        except Exception as e:
            logger.error(f"Failed to initialize Sentry: {e}")
            self.enabled = False

    def _before_send(
        self, event: Dict[str, Any], hint: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process event before sending to Sentry.

        Args:
            event: Event data
            hint: Additional context

        Returns:
            Modified event or None to drop it
        """
        # Filter out sensitive information
        if "request" in event:
            request = event["request"]
            # Remove authorization headers
            if "headers" in request:
                request["headers"] = {
                    k: v
                    for k, v in request["headers"].items()
                    if k.lower() not in ["authorization", "api-key", "x-api-key"]
                }

        # Add custom fingerprinting for better grouping
        if "exception" in event:
            exception = event["exception"]
            if "values" in exception:
                for exc_value in exception["values"]:
                    if exc_value.get("type") == "ValidationError":
                        # Group all validation errors together
                        event["fingerprint"] = ["validation-error", "{{ default }}"]
                    elif exc_value.get("type") == "PermissionError":
                        # Group permission errors by tool
                        event["fingerprint"] = ["permission-error", "{{ transaction }}"]

        return event

    def _before_send_transaction(
        self, event: Dict[str, Any], hint: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process transaction before sending to Sentry.

        Args:
            event: Transaction data
            hint: Additional context

        Returns:
            Modified transaction or None to drop it
        """
        # Add custom measurements
        if "measurements" not in event:
            event["measurements"] = {}

        # Filter out health check transactions
        if event.get("transaction") == "/health":
            return None  # Don't send health checks

        return event

    def capture_mcp_error(
        self,
        error: Exception,
        tool_name: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ):
        """Capture MCP-specific error with context.

        Args:
            error: Exception to capture
            tool_name: Name of the tool that failed
            session_id: MCP session ID
            request_id: Request ID
            additional_context: Additional context data
        """
        if not self.enabled:
            return

        try:
            # Set MCP-specific context
            set_context(
                "mcp",
                {
                    "tool": tool_name,
                    "session_id": session_id,
                    "request_id": request_id,
                    **(additional_context or {}),
                },
            )

            # Set tags for filtering
            if tool_name:
                set_tag("mcp.tool", tool_name)
            if session_id:
                set_tag("mcp.session_id", session_id)

            # Capture the exception
            capture_exception(error)

        except Exception as e:
            logger.error(f"Failed to capture error in Sentry: {e}")

    def capture_performance_metric(
        self,
        operation: str,
        duration_ms: float,
        tool_name: Optional[str] = None,
        success: bool = True,
    ):
        """Capture performance metric.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            tool_name: Tool name if applicable
            success: Whether operation succeeded
        """
        if not self.enabled or not HAS_SENTRY:
            return

        try:
            with sentry_sdk.start_transaction(
                op=f"mcp.{operation}",
                name=f"MCP {operation}",
            ) as transaction:
                transaction.set_tag("mcp.tool", tool_name or "unknown")
                transaction.set_tag("success", str(success))
                transaction.set_measurement("duration", duration_ms, "millisecond")

        except Exception as e:
            logger.error(f"Failed to capture performance metric: {e}")

    @contextmanager
    def monitor_operation(self, operation: str, **tags):
        """Context manager to monitor an operation.

        Args:
            operation: Operation name
            **tags: Additional tags to set

        Example:
            with sentry.monitor_operation("tool.create_graph", tool="create_graph"):
                # Perform operation
                pass
        """
        if not self.enabled or not HAS_SENTRY:
            yield
            return

        transaction = sentry_sdk.start_transaction(
            op=f"mcp.{operation}",
            name=f"MCP {operation}",
        )

        try:
            with transaction:
                for key, value in tags.items():
                    transaction.set_tag(key, value)
                yield transaction
        except Exception:
            transaction.set_status("internal_error")
            raise
        else:
            transaction.set_status("ok")

    def add_breadcrumb(
        self,
        message: str,
        category: str = "mcp",
        level: str = "info",
        data: Optional[Dict[str, Any]] = None,
    ):
        """Add a breadcrumb for debugging.

        Args:
            message: Breadcrumb message
            category: Category (e.g., "mcp", "auth", "tool")
            level: Level (debug, info, warning, error, critical)
            data: Additional data
        """
        if not self.enabled or not HAS_SENTRY:
            return

        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data or {},
        )

    def set_user_context(
        self,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        email: Optional[str] = None,
        ip_address: Optional[str] = None,
    ):
        """Set user context for error tracking.

        Args:
            user_id: User ID
            username: Username
            email: User email (be careful with PII)
            ip_address: IP address
        """
        if not self.enabled or not HAS_SENTRY:
            return

        sentry_sdk.set_user(
            {
                "id": user_id,
                "username": username,
                "email": email,
                "ip_address": ip_address,
            }
        )

    def capture_message(self, message: str, level: str = "info", **kwargs):
        """Capture a message event.

        Args:
            message: Message to capture
            level: Level (debug, info, warning, error, critical)
            **kwargs: Additional context
        """
        if not self.enabled:
            return

        capture_message(message, level=level)

    def flush(self, timeout: float = 2.0) -> bool:
        """Flush pending events to Sentry.

        Args:
            timeout: Timeout in seconds

        Returns:
            True if all events were sent
        """
        if not self.enabled or not HAS_SENTRY:
            return True

        try:
            return sentry_sdk.flush(timeout=timeout)
        except Exception as e:
            logger.error(f"Failed to flush Sentry events: {e}")
            return False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - flush events."""
        if self.enabled:
            self.flush()


# Global Sentry instance
sentry = SentryIntegration()


def init_sentry(**kwargs) -> SentryIntegration:
    """Initialize Sentry with custom configuration.

    Args:
        **kwargs: Configuration parameters for SentryIntegration

    Returns:
        Configured SentryIntegration instance
    """
    global sentry
    sentry = SentryIntegration(**kwargs)
    return sentry
