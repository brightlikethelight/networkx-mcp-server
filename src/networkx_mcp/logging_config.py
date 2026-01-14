"""Centralized logging configuration for NetworkX MCP Server.

This module provides structured logging with JSON output for production
environments and human-readable output for development.

Usage:
    from networkx_mcp.logging_config import configure_logging

    # Production (structured JSON)
    configure_logging(level="INFO", structured=True)

    # Development (human-readable)
    configure_logging(level="DEBUG", structured=False)

    # With file output
    configure_logging(level="INFO", log_file="/var/log/networkx-mcp.log")
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for production environments.

    Outputs logs in JSON format for easy parsing by log aggregators
    like Elasticsearch, Splunk, or CloudWatch.
    """

    def __init__(self, include_extra: bool = True) -> None:
        """Initialize the structured formatter.

        Args:
            include_extra: Whether to include extra fields from log records
        """
        super().__init__()
        self.include_extra = include_extra
        self._reserved_attrs = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            log_entry["exception_type"] = (
                record.exc_info[0].__name__ if record.exc_info[0] else None
            )

        # Add stack trace if present
        if record.stack_info:
            log_entry["stack_trace"] = record.stack_info

        # Add extra fields from the log record
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in self._reserved_attrs and not key.startswith("_"):
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)

        return json.dumps(log_entry, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for development environments.

    Provides clear, colorized output when running in a terminal.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True) -> None:
        """Initialize the human-readable formatter.

        Args:
            use_colors: Whether to use ANSI colors in output
        """
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as human-readable string.

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        formatted = super().format(record)

        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            formatted = f"{color}{formatted}{self.RESET}"

        return formatted


class MCPProtocolFormatter(logging.Formatter):
    """Formatter for MCP protocol logging.

    Ensures that MCP stdio protocol messages are not interfered with
    while still providing useful logging to stderr.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for MCP compatibility.

        All log output goes to stderr to avoid interfering with
        stdout-based MCP protocol communication.

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        level = record.levelname[0]  # D, I, W, E, C
        return f"[{timestamp}] {level} {record.name}: {record.getMessage()}"


def configure_logging(
    level: str = "INFO",
    structured: bool = False,
    log_file: Optional[str] = None,
    max_file_size_mb: int = 10,
    backup_count: int = 5,
    mcp_mode: bool = False,
) -> None:
    """Configure application-wide logging.

    Sets up logging for the networkx_mcp package with appropriate
    formatters and handlers based on the environment.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Use JSON structured logging (for production)
        log_file: Optional path to log file for persistent logging
        max_file_size_mb: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        mcp_mode: Use MCP-compatible logging (stderr only, minimal format)

    Example:
        # Development
        configure_logging(level="DEBUG", structured=False)

        # Production
        configure_logging(
            level="INFO",
            structured=True,
            log_file="/var/log/networkx-mcp/server.log"
        )

        # MCP server mode
        configure_logging(level="WARNING", mcp_mode=True)
    """
    # Get the root logger for networkx_mcp
    root_logger = logging.getLogger("networkx_mcp")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove any existing handlers
    root_logger.handlers.clear()

    # Select formatter based on mode
    if mcp_mode:
        formatter: logging.Formatter = MCPProtocolFormatter()
    elif structured:
        formatter = StructuredFormatter(include_extra=True)
    else:
        formatter = HumanReadableFormatter(use_colors=True)

    # Console handler (always stderr for MCP compatibility)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional, always uses structured format)
    if log_file:
        file_formatter = StructuredFormatter(include_extra=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    # Log configuration
    root_logger.debug(
        f"Logging configured: level={level}, structured={structured}, "
        f"mcp_mode={mcp_mode}, log_file={log_file}"
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Creates a child logger under the networkx_mcp namespace.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Operation completed", extra={"graph_id": "test"})
    """
    if not name.startswith("networkx_mcp"):
        name = f"networkx_mcp.{name}"
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding contextual information to logs.

    Adds extra fields to all log messages within the context.

    Usage:
        with LogContext(graph_id="test", operation="create"):
            logger.info("Creating graph")  # Includes graph_id and operation
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize log context with extra fields.

        Args:
            **kwargs: Key-value pairs to add to all log messages
        """
        self.extra = kwargs
        self._old_factory: Any = None

    def __enter__(self) -> "LogContext":
        """Enter the context and install the custom record factory."""
        self._old_factory = logging.getLogRecordFactory()

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = self._old_factory(*args, **kwargs)
            for key, value in self.extra.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit the context and restore the original record factory."""
        logging.setLogRecordFactory(self._old_factory)


# Convenience function for timing operations
class LogTimer:
    """Context manager for logging operation duration.

    Usage:
        with LogTimer(logger, "graph_creation"):
            create_graph()
        # Logs: "graph_creation completed in 123.45ms"
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        level: int = logging.DEBUG,
    ) -> None:
        """Initialize the timer.

        Args:
            logger: Logger to use
            operation: Name of the operation being timed
            level: Log level for the timing message
        """
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time: float = 0

    def __enter__(self) -> "LogTimer":
        """Start timing."""
        import time

        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Log the duration."""
        import time

        duration_ms = (time.perf_counter() - self.start_time) * 1000

        if exc_type is not None:
            self.logger.log(
                self.level,
                f"{self.operation} failed after {duration_ms:.2f}ms",
                extra={"operation": self.operation, "duration_ms": duration_ms},
            )
        else:
            self.logger.log(
                self.level,
                f"{self.operation} completed in {duration_ms:.2f}ms",
                extra={"operation": self.operation, "duration_ms": duration_ms},
            )
