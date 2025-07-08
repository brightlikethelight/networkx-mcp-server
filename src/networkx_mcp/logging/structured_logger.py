"""Main structured logging implementation."""

import asyncio
import functools
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, Optional, Union

# Import functions locally to avoid circular imports
from .formatters import JSONFormatter, ColoredFormatter, CompactFormatter


class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(name)
    
    def _add_context(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """Add correlation ID and request context to log extra."""
        if extra is None:
            extra = {}
        
        # Add correlation ID
        from .correlation import get_correlation_id, get_request_context
        correlation_id = get_correlation_id()
        if correlation_id:
            extra["correlation_id"] = correlation_id
        
        # Add request context
        context = get_request_context()
        if context:
            for key, value in context.items():
                extra[f"context_{key}"] = value
        
        return extra
    
    def debug(self, msg: str, *args, extra: Dict[str, Any] = None, **kwargs):
        """Log debug message with structured context."""
        extra = self._add_context(extra)
        self.logger.debug(msg, *args, extra=extra, **kwargs)
    
    def info(self, msg: str, *args, extra: Dict[str, Any] = None, **kwargs):
        """Log info message with structured context."""
        extra = self._add_context(extra)
        self.logger.info(msg, *args, extra=extra, **kwargs)
    
    def warning(self, msg: str, *args, extra: Dict[str, Any] = None, **kwargs):
        """Log warning message with structured context."""
        extra = self._add_context(extra)
        self.logger.warning(msg, *args, extra=extra, **kwargs)
    
    def warn(self, msg: str, *args, extra: Dict[str, Any] = None, **kwargs):
        """Alias for warning."""
        self.warning(msg, *args, extra=extra, **kwargs)
    
    def error(self, msg: str, *args, extra: Dict[str, Any] = None, **kwargs):
        """Log error message with structured context."""
        extra = self._add_context(extra)
        self.logger.error(msg, *args, extra=extra, **kwargs)
    
    def critical(self, msg: str, *args, extra: Dict[str, Any] = None, **kwargs):
        """Log critical message with structured context."""
        extra = self._add_context(extra)
        self.logger.critical(msg, *args, extra=extra, **kwargs)
    
    def exception(self, msg: str, *args, extra: Dict[str, Any] = None, **kwargs):
        """Log exception with structured context."""
        extra = self._add_context(extra)
        self.logger.exception(msg, *args, extra=extra, **kwargs)
    
    def log_request(
        self,
        method: str,
        path: str,
        user_id: Optional[str] = None,
        **context_data
    ):
        """Log an incoming request."""
        extra = {
            "event_type": "request",
            "http_method": method,
            "http_path": path,
            "user_id": user_id,
            **context_data
        }
        
        # Add to request context for other logs
        from .correlation import set_request_context
        set_request_context("http_method", method)
        set_request_context("http_path", path)
        if user_id:
            set_request_context("user_id", user_id)
        
        self.info(f"Request started: {method} {path}", extra=extra)
    
    def log_response(
        self,
        status_code: int,
        duration_ms: float,
        response_size: Optional[int] = None,
        **context_data
    ):
        """Log a response."""
        extra = {
            "event_type": "response",
            "http_status": status_code,
            "duration_ms": duration_ms,
            "response_size": response_size,
            **context_data
        }
        
        level = "info"
        if status_code >= 500:
            level = "error"
        elif status_code >= 400:
            level = "warning"
        
        method = level
        getattr(self, method)(
            f"Request completed: {status_code} ({duration_ms:.2f}ms)",
            extra=extra
        )
    
    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **metrics
    ):
        """Log performance metrics."""
        extra = {
            "event_type": "performance",
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            **metrics
        }
        
        if success:
            self.info(f"Operation completed: {operation} ({duration_ms:.2f}ms)", extra=extra)
        else:
            self.warning(f"Operation failed: {operation} ({duration_ms:.2f}ms)", extra=extra)
    
    def log_operation_start(self, operation: str, **context_data):
        """Log the start of an operation."""
        extra = {
            "event_type": "operation_start",
            "operation": operation,
            **context_data
        }
        
        # Add to request context
        from .correlation import set_request_context
        set_request_context("operation", operation)
        set_request_context("operation_start_time", time.time())
        
        self.debug(f"Operation started: {operation}", extra=extra)
    
    def log_operation_end(self, operation: str, success: bool = True, **context_data):
        """Log the end of an operation."""
        # Calculate duration if start time is available
        from .correlation import get_request_context
        start_time = get_request_context("operation_start_time")
        duration_ms = None
        if start_time:
            duration_ms = (time.time() - start_time) * 1000
        
        extra = {
            "event_type": "operation_end",
            "operation": operation,
            "success": success,
            "duration_ms": duration_ms,
            **context_data
        }
        
        level = "debug" if success else "warning"
        status = "completed" if success else "failed"
        duration_str = f" ({duration_ms:.2f}ms)" if duration_ms else ""
        
        getattr(self, level)(f"Operation {status}: {operation}{duration_str}", extra=extra)
    
    def log_graph_operation(
        self,
        operation: str,
        graph_name: str,
        node_count: Optional[int] = None,
        edge_count: Optional[int] = None,
        **context_data
    ):
        """Log a graph-specific operation."""
        extra = {
            "event_type": "graph_operation",
            "operation": operation,
            "graph_name": graph_name,
            "node_count": node_count,
            "edge_count": edge_count,
            **context_data
        }
        
        # Add to request context
        from .correlation import set_request_context
        set_request_context("graph_name", graph_name)
        
        self.info(f"Graph operation: {operation} on {graph_name}", extra=extra)
    
    def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        success: bool = True,
        **context_data
    ):
        """Log a security-related event."""
        extra = {
            "event_type": "security",
            "security_event": event_type,
            "user_id": user_id,
            "source_ip": source_ip,
            "success": success,
            **context_data
        }
        
        level = "info" if success else "warning"
        status = "succeeded" if success else "failed"
        
        getattr(self, level)(f"Security event: {event_type} {status}", extra=extra)
    
    def log_error_with_context(
        self,
        error: Exception,
        operation: str,
        **context_data
    ):
        """Log an error with full context."""
        extra = {
            "event_type": "error",
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context_data
        }
        
        self.error(f"Error in {operation}: {error}", extra=extra, exc_info=True)


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


def configure_logging(
    level: Union[str, int] = "INFO",
    format_type: str = "colored",
    output_file: Optional[str] = None,
    include_correlation: bool = True,
    include_context: bool = True,
    json_include_extra: bool = True
) -> None:
    """Configure structured logging for the application."""
    
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Choose formatter
    if format_type == "json":
        formatter = JSONFormatter(include_extra=json_include_extra)
    elif format_type == "colored":
        formatter = ColoredFormatter(
            include_correlation=include_correlation,
            include_context=include_context
        )
    elif format_type == "compact":
        formatter = CompactFormatter()
    else:
        # Default to colored
        formatter = ColoredFormatter(
            include_correlation=include_correlation,
            include_context=include_context
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if output_file:
        # Always use JSON format for file output
        file_formatter = JSONFormatter(include_extra=json_include_extra)
        file_handler = logging.FileHandler(output_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    # Set logging level for specific libraries to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Log configuration
    logger = get_logger("networkx_mcp.logging")
    logger.info(
        "Structured logging configured",
        extra={
            "log_level": logging.getLevelName(level),
            "format_type": format_type,
            "output_file": output_file,
            "include_correlation": include_correlation,
            "include_context": include_context
        }
    )


# Convenience functions
def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current context."""
    from .correlation import set_correlation_id as _set_correlation_id
    _set_correlation_id(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get correlation ID from current context."""
    from .correlation import get_correlation_id as _get_correlation_id
    return _get_correlation_id()


def clear_correlation_id() -> None:
    """Clear correlation ID from current context."""
    from .correlation import clear_correlation_id as _clear_correlation_id
    _clear_correlation_id()


def log_performance(operation: str, duration_ms: float, success: bool = True, **metrics):
    """Log performance metrics using default logger."""
    logger = get_logger("networkx_mcp.performance")
    logger.log_performance(operation, duration_ms, success, **metrics)


def log_request(method: str, path: str, user_id: Optional[str] = None, **context_data):
    """Log request using default logger."""
    logger = get_logger("networkx_mcp.request")
    logger.log_request(method, path, user_id, **context_data)


def log_response(status_code: int, duration_ms: float, response_size: Optional[int] = None, **context_data):
    """Log response using default logger."""
    logger = get_logger("networkx_mcp.request")
    logger.log_response(status_code, duration_ms, response_size, **context_data)


def timed_operation(operation_name: str = None, log_args: bool = False):
    """Decorator to automatically log operation timing."""
    
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger("networkx_mcp.performance")
            
            # Log operation start
            start_time = time.time()
            extra_data = {}
            if log_args and args:
                extra_data["arg_count"] = len(args)
            if log_args and kwargs:
                extra_data["kwarg_count"] = len(kwargs)
            
            logger.log_operation_start(op_name, **extra_data)
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(op_name, duration_ms, success=True)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(op_name, duration_ms, success=False, error=str(e))
                logger.log_error_with_context(e, op_name)
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger("networkx_mcp.performance")
            
            # Log operation start
            start_time = time.time()
            extra_data = {}
            if log_args and args:
                extra_data["arg_count"] = len(args)
            if log_args and kwargs:
                extra_data["kwarg_count"] = len(kwargs)
            
            logger.log_operation_start(op_name, **extra_data)
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(op_name, duration_ms, success=True)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(op_name, duration_ms, success=False, error=str(e))
                logger.log_error_with_context(e, op_name)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator