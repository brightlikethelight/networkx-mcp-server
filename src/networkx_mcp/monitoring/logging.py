"""Structured logging with correlation IDs and tracing integration."""

import asyncio
import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

from ..core.base import Component

# Default log format
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
JSON_FORMAT = "json"


@dataclass
class LogContext:
    """Context information for structured logging."""

    correlation_id: str
    trace_id: str | None = None
    span_id: str | None = None
    user_id: str | None = None
    request_id: str | None = None
    session_id: str | None = None
    operation: str | None = None
    component: str | None = None
    extra: dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class LogCorrelation:
    """Context manager for log correlation across async operations."""

    def __init__(self):
        self._local = asyncio.local()

    def get_context(self) -> LogContext | None:
        """Get the current log context."""
        return getattr(self._local, "log_context", None)

    def set_context(self, context: LogContext) -> None:
        """Set the current log context."""
        self._local.log_context = context

    def clear_context(self) -> None:
        """Clear the current log context."""
        self._local.log_context = None

    @contextmanager
    def context(self, **kwargs):
        """Context manager for temporary log context."""
        old_context = self.get_context()

        # Create new context or update existing
        if old_context:
            context_dict = asdict(old_context)
            context_dict.update(kwargs)
            new_context = LogContext(**context_dict)
        else:
            correlation_id = kwargs.pop("correlation_id", str(uuid.uuid4()))
            new_context = LogContext(correlation_id=correlation_id, **kwargs)

        try:
            self.set_context(new_context)
            yield new_context
        finally:
            self.set_context(old_context)


class StructuredFormatter(logging.Formatter):
    """Formatter for structured JSON logging."""

    def __init__(self, correlation: LogCorrelation):
        super().__init__()
        self.correlation = correlation

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log data
        log_data = {
            "timestamp": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add correlation context
        context = self.correlation.get_context()
        if context:
            log_data.update(
                {
                    "correlation_id": context.correlation_id,
                    "trace_id": context.trace_id,
                    "span_id": context.span_id,
                    "user_id": context.user_id,
                    "request_id": context.request_id,
                    "session_id": context.session_id,
                    "operation": context.operation,
                    "component": context.component,
                }
            )

            # Add extra fields
            if context.extra:
                log_data.update(context.extra)

        # Add any extra fields from the record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data, default=str)


class StructuredLogger(Component):
    """Structured logging service with correlation and tracing integration."""

    def __init__(
        self,
        name: str = "networkx_mcp",
        level: str = "INFO",
        format_type: str = "json",
        output_file: str | None = None,
    ):
        super().__init__("structured_logger")
        self.logger_name = name
        self.level = getattr(logging, level.upper())
        self.format_type = format_type
        self.output_file = output_file
        self.correlation = LogCorrelation()
        self.logger = logging.getLogger(name)

        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        # Clear existing handlers
        self.logger.handlers.clear()

        # Set level
        self.logger.setLevel(self.level)

        # Create handler
        if self.output_file:
            handler = logging.FileHandler(self.output_file)
        else:
            handler = logging.StreamHandler(sys.stdout)

        # Set formatter
        if self.format_type == JSON_FORMAT:
            formatter = StructuredFormatter(self.correlation)
        else:
            formatter = logging.Formatter(DEFAULT_FORMAT)

        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def get_logger(self, name: str | None = None) -> logging.Logger:
        """Get a logger instance."""
        if name:
            return logging.getLogger(f"{self.logger_name}.{name}")
        return self.logger

    def log(
        self,
        level: str,
        message: str,
        correlation_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        **extra_fields,
    ) -> None:
        """Log a message with structured data."""
        logger = self.get_logger()

        # Create log record
        record = logger.makeRecord(
            name=logger.name,
            level=getattr(logging, level.upper()),
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
        )

        # Add extra fields
        if extra_fields:
            record.extra_fields = extra_fields

        # Temporarily set correlation context if provided
        if correlation_id or trace_id or span_id:
            old_context = self.correlation.get_context()
            temp_context = LogContext(
                correlation_id=correlation_id or str(uuid.uuid4()),
                trace_id=trace_id,
                span_id=span_id,
            )
            self.correlation.set_context(temp_context)

            try:
                logger.handle(record)
            finally:
                self.correlation.set_context(old_context)
        else:
            logger.handle(record)

    def debug(self, message: str, **extra) -> None:
        """Log debug message."""
        self.log("DEBUG", message, **extra)

    def info(self, message: str, **extra) -> None:
        """Log info message."""
        self.log("INFO", message, **extra)

    def warning(self, message: str, **extra) -> None:
        """Log warning message."""
        self.log("WARNING", message, **extra)

    def error(self, message: str, **extra) -> None:
        """Log error message."""
        self.log("ERROR", message, **extra)

    def critical(self, message: str, **extra) -> None:
        """Log critical message."""
        self.log("CRITICAL", message, **extra)

    def start_operation(
        self, operation: str, correlation_id: str | None = None, **context_data
    ) -> LogContext:
        """Start a new operation with logging context."""
        correlation_id = correlation_id or str(uuid.uuid4())

        context = LogContext(
            correlation_id=correlation_id,
            operation=operation,
            component=self.logger_name,
            **context_data,
        )

        self.correlation.set_context(context)

        self.info(f"Started operation: {operation}", operation=operation)
        return context

    def end_operation(self, operation: str, success: bool = True, **extra) -> None:
        """End the current operation."""
        status = "completed" if success else "failed"
        self.info(
            f"Operation {status}: {operation}",
            operation=operation,
            status=status,
            **extra,
        )

        self.correlation.clear_context()

    def log_performance(self, operation: str, duration: float, **metrics) -> None:
        """Log performance metrics."""
        self.info(
            f"Performance: {operation}",
            operation=operation,
            duration_ms=duration * 1000,
            **metrics,
        )

    def log_error(
        self, error: Exception, operation: str | None = None, **context
    ) -> None:
        """Log an error with full context."""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "operation": operation,
            **context,
        }

        self.error(f"Error in {operation or 'operation'}: {error}", **error_data)

    def log_audit(
        self,
        action: str,
        resource: str,
        user_id: str | None = None,
        result: str = "success",
        **details,
    ) -> None:
        """Log audit event."""
        audit_data = {
            "audit": True,
            "action": action,
            "resource": resource,
            "user_id": user_id,
            "result": result,
            **details,
        }

        self.info(f"Audit: {action} on {resource}", **audit_data)

    def log_security(
        self, event_type: str, severity: str, description: str, **security_context
    ) -> None:
        """Log security event."""
        security_data = {
            "security": True,
            "event_type": event_type,
            "severity": severity,
            "description": description,
            **security_context,
        }

        level = "WARNING" if severity in ["medium", "high"] else "INFO"
        self.log(level, f"Security: {event_type} - {description}", **security_data)

    def configure_tracing_integration(self, tracing_service) -> None:
        """Configure integration with tracing service."""
        from .tracing import TracingService

        if isinstance(tracing_service, TracingService):
            # Monkey patch to automatically add trace context
            original_log = self.log

            def log_with_tracing(level, message, **extra):
                trace_info = tracing_service.get_trace_info()
                extra.update(trace_info)
                original_log(level, message, **extra)

            self.log = log_with_tracing

    async def initialize(self) -> None:
        """Initialize the structured logger."""
        await super().initialize()
        self.info("Structured logger initialized")

    async def cleanup(self) -> None:
        """Cleanup the structured logger."""
        self.info("Structured logger shutting down")

        # Flush all handlers
        for handler in self.logger.handlers:
            handler.flush()

        await super().cleanup()


# Global logger instance
_structured_logger: StructuredLogger | None = None


def get_logger(name: str | None = None) -> StructuredLogger:
    """Get the global structured logger instance."""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger()
    return _structured_logger


def with_logging_context(**context_data):
    """Decorator to add logging context to functions."""

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                logger = get_logger()
                correlation_id = str(uuid.uuid4())

                with logger.correlation.context(
                    correlation_id=correlation_id,
                    operation=func.__name__,
                    **context_data,
                ):
                    logger.info(f"Starting {func.__name__}")
                    try:
                        result = await func(*args, **kwargs)
                        logger.info(f"Completed {func.__name__}")
                        return result
                    except Exception as e:
                        logger.log_error(e, operation=func.__name__)
                        raise

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                logger = get_logger()
                correlation_id = str(uuid.uuid4())

                with logger.correlation.context(
                    correlation_id=correlation_id,
                    operation=func.__name__,
                    **context_data,
                ):
                    logger.info(f"Starting {func.__name__}")
                    try:
                        result = func(*args, **kwargs)
                        logger.info(f"Completed {func.__name__}")
                        return result
                    except Exception as e:
                        logger.log_error(e, operation=func.__name__)
                        raise

            return sync_wrapper

    return decorator


# Aliases for backward compatibility
ContextualLogger = StructuredLogger


def setup_logging(level="INFO", format_type="json", output_file=None):
    """Setup global structured logging."""
    global _structured_logger
    _structured_logger = StructuredLogger(
        level=level,
        format_type=format_type,
        output_file=output_file
    )
    return _structured_logger
