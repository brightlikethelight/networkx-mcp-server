"""Custom log formatters for structured logging."""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from .correlation import get_correlation_id, get_request_context


class StructuredFormatter(logging.Formatter):
    """Base formatter that adds structured fields to log records."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add correlation ID if available
        correlation_id = get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        
        # Add request context if available
        context = get_request_context()
        if context:
            for key, value in context.items():
                setattr(record, f"context_{key}", value)
        
        # Add timestamp in ISO format
        record.timestamp = datetime.now(timezone.utc).isoformat()
        
        # Add process/thread info
        record.process_name = record.processName
        record.thread_name = record.threadName
        
        return super().format(record)


class JSONFormatter(StructuredFormatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        # Call parent to add structured fields
        super().format(record)
        
        # Build the log entry
        log_entry = {
            "timestamp": getattr(record, "timestamp", datetime.utcnow().isoformat() + "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": {
                "id": record.process,
                "name": getattr(record, "process_name", record.processName)
            },
            "thread": {
                "id": record.thread,
                "name": getattr(record, "thread_name", record.threadName)
            }
        }
        
        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = record.correlation_id
        
        # Add request context
        context = {}
        for key, value in record.__dict__.items():
            if key.startswith("context_"):
                context_key = key[8:]  # Remove "context_" prefix
                context[context_key] = value
        
        if context:
            log_entry["context"] = context
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields if requested
        if self.include_extra:
            extra = {}
            # Standard fields to exclude from extra
            standard_fields = {
                "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message", "exc_info",
                "exc_text", "stack_info", "timestamp", "correlation_id", "process_name",
                "thread_name"
            }
            
            for key, value in record.__dict__.items():
                if key not in standard_fields and not key.startswith("context_"):
                    try:
                        # Only include JSON-serializable values
                        json.dumps(value)
                        extra[key] = value
                    except (TypeError, ValueError):
                        extra[key] = str(value)
            
            if extra:
                log_entry["extra"] = extra
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class ColoredFormatter(StructuredFormatter):
    """Colored formatter for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    def __init__(self, include_correlation: bool = True, include_context: bool = True):
        super().__init__()
        self.include_correlation = include_correlation
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        # Call parent to add structured fields
        super().format(record)
        
        # Get color for log level
        level_color = self.COLORS.get(record.levelname, '')
        
        # Build the formatted message
        parts = []
        
        # Timestamp
        timestamp = getattr(record, "timestamp", datetime.now(timezone.utc).isoformat())
        parts.append(f"{self.DIM}{timestamp}{self.RESET}")
        
        # Log level (colored)
        parts.append(f"{level_color}{self.BOLD}{record.levelname:8s}{self.RESET}")
        
        # Logger name
        parts.append(f"{self.DIM}{record.name}{self.RESET}")
        
        # Correlation ID (if present and enabled)
        if self.include_correlation and hasattr(record, "correlation_id"):
            correlation_id = record.correlation_id[:8]  # Truncate for readability
            parts.append(f"{self.DIM}[{correlation_id}]{self.RESET}")
        
        # Context info (if present and enabled)
        if self.include_context:
            context_parts = []
            for key, value in record.__dict__.items():
                if key.startswith("context_"):
                    context_key = key[8:]  # Remove "context_" prefix
                    if context_key in ["operation", "user_id", "graph_id"]:
                        context_parts.append(f"{context_key}={value}")
            
            if context_parts:
                parts.append(f"{self.DIM}({', '.join(context_parts)}){self.RESET}")
        
        # Message
        parts.append(record.getMessage())
        
        # Exception (if present)
        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            parts.append(f"\n{exception_text}")
        
        return " ".join(parts)


class KeyValueFormatter(StructuredFormatter):
    """Key-value formatter for easy parsing."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Call parent to add structured fields
        super().format(record)
        
        # Build key-value pairs
        pairs = []
        
        # Standard fields
        pairs.append(f"timestamp={getattr(record, 'timestamp', datetime.utcnow().isoformat() + 'Z')}")
        pairs.append(f"level={record.levelname}")
        pairs.append(f"logger={record.name}")
        pairs.append(f"message=\"{record.getMessage()}\"")
        pairs.append(f"module={record.module}")
        pairs.append(f"function={record.funcName}")
        pairs.append(f"line={record.lineno}")
        
        # Correlation ID
        if hasattr(record, "correlation_id"):
            pairs.append(f"correlation_id={record.correlation_id}")
        
        # Context
        for key, value in record.__dict__.items():
            if key.startswith("context_"):
                context_key = key[8:]  # Remove "context_" prefix
                if isinstance(value, str):
                    pairs.append(f"{context_key}=\"{value}\"")
                else:
                    pairs.append(f"{context_key}={value}")
        
        # Exception
        if record.exc_info:
            exception_msg = str(record.exc_info[1]) if record.exc_info[1] else "Unknown"
            pairs.append(f"exception=\"{exception_msg}\"")
        
        return " ".join(pairs)


class CompactFormatter(StructuredFormatter):
    """Compact formatter for high-volume logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Call parent to add structured fields
        super().format(record)
        
        # Very compact format: timestamp level logger [correlation] message
        timestamp = getattr(record, "timestamp", datetime.now(timezone.utc).isoformat())
        timestamp_short = timestamp[:19]  # Remove milliseconds and timezone
        
        correlation_part = ""
        if hasattr(record, "correlation_id"):
            correlation_part = f"[{record.correlation_id[:8]}] "
        
        return f"{timestamp_short} {record.levelname[0]} {record.name} {correlation_part}{record.getMessage()}"