#!/usr/bin/env python3
"""Production logging configuration with correlation IDs and performance tracking.

Based on production requirements for monitoring and observability.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict

from .correlation import CorrelationContext, get_correlation_id
from .structured_logger import StructuredLogger
from ..config.production import production_config


@dataclass
class RequestMetrics:
    """Track request-level metrics for production monitoring."""
    request_id: str
    method: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    user_id: Optional[str] = None
    memory_delta_mb: Optional[float] = None


class ProductionRequestLogger:
    """Enhanced request logger for production environments."""
    
    def __init__(self):
        self.logger = StructuredLogger("networkx_mcp.requests")
        self.metrics_logger = StructuredLogger("networkx_mcp.metrics")
        self._active_requests: Dict[str, RequestMetrics] = {}
        
    async def log_request_start(self, 
                              method: str, 
                              request_id: str = None,
                              params: Dict[str, Any] = None,
                              user_id: str = None) -> str:
        """Log the start of an MCP request."""
        if not request_id:
            request_id = str(uuid.uuid4())
            
        correlation_id = get_correlation_id() or request_id
        
        # Create metrics tracking
        metrics = RequestMetrics(
            request_id=request_id,
            method=method,
            start_time=time.time(),
            user_id=user_id
        )
        self._active_requests[request_id] = metrics
        
        # Log request start
        log_data = {
            "event": "mcp_request_start",
            "request_id": request_id,
            "correlation_id": correlation_id,
            "method": method,
            "user_id": user_id,
            "timestamp": time.time()
        }
        
        # Add sanitized params (remove sensitive data)
        if params:
            sanitized_params = self._sanitize_params(params)
            log_data["params"] = sanitized_params
            log_data["param_count"] = len(params)
            
        self.logger.info("MCP request started", extra=log_data)
        return request_id
    
    async def log_request_end(self, 
                            request_id: str,
                            success: bool = True,
                            result: Any = None,
                            error: Exception = None,
                            memory_usage_mb: float = None) -> Optional[float]:
        """Log the end of an MCP request and return duration."""
        if request_id not in self._active_requests:
            self.logger.warning(f"Request {request_id} not found in active requests")
            return None
            
        metrics = self._active_requests.pop(request_id)
        end_time = time.time()
        duration_ms = (end_time - metrics.start_time) * 1000
        
        # Update metrics
        metrics.end_time = end_time
        metrics.duration_ms = duration_ms
        metrics.success = success
        
        if error:
            metrics.error_type = type(error).__name__
            
        if memory_usage_mb:
            metrics.memory_delta_mb = memory_usage_mb
        
        # Log request completion
        log_data = {
            "event": "mcp_request_end",
            "request_id": request_id,
            "method": metrics.method,
            "duration_ms": round(duration_ms, 2),
            "success": success,
            "timestamp": end_time,
            "user_id": metrics.user_id
        }
        
        if error:
            log_data["error_type"] = type(error).__name__
            log_data["error_message"] = str(error)
            
        if result and success:
            log_data["result_type"] = type(result).__name__
            if hasattr(result, '__len__'):
                try:
                    log_data["result_size"] = len(result)
                except:
                    pass
                    
        if memory_usage_mb:
            log_data["memory_mb"] = round(memory_usage_mb, 1)
            
        # Choose log level based on performance
        if not success:
            self.logger.error("MCP request failed", extra=log_data)
        elif duration_ms > 2000:  # Over 2 seconds
            self.logger.warning("MCP request slow", extra=log_data)
        else:
            self.logger.info("MCP request completed", extra=log_data)
            
        # Log metrics separately for analysis
        await self._log_performance_metrics(metrics)
        
        return duration_ms
    
    async def _log_performance_metrics(self, metrics: RequestMetrics):
        """Log performance metrics for monitoring systems."""
        try:
            # Create metrics entry
            metrics_data = {
                "event": "mcp_performance_metric",
                "timestamp": time.time(),
                **asdict(metrics)
            }
            
            # Add performance classification
            if metrics.duration_ms:
                if metrics.duration_ms < 100:
                    metrics_data["performance_class"] = "excellent"
                elif metrics.duration_ms < 500:
                    metrics_data["performance_class"] = "good"
                elif metrics.duration_ms < 2000:
                    metrics_data["performance_class"] = "acceptable"
                else:
                    metrics_data["performance_class"] = "poor"
            
            self.metrics_logger.info("Performance metric", extra=metrics_data)
            
        except Exception as e:
            self.logger.error(f"Failed to log performance metrics: {e}")
    
    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from params before logging."""
        sanitized = {}
        
        sensitive_keys = {
            'password', 'token', 'secret', 'key', 'auth', 'credential',
            'admin_token', 'api_key', 'private_key'
        }
        
        for key, value in params.items():
            key_lower = key.lower()
            
            # Check if key contains sensitive terms
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, (dict, list)) and len(str(value)) > 1000:
                # Truncate large objects
                sanitized[key] = f"[LARGE_OBJECT: {type(value).__name__}]"
            elif isinstance(value, str) and len(value) > 500:
                # Truncate long strings
                sanitized[key] = value[:497] + "..."
            else:
                sanitized[key] = value
                
        return sanitized
    
    def get_active_request_count(self) -> int:
        """Get number of currently active requests."""
        return len(self._active_requests)
    
    def get_active_requests(self) -> Dict[str, RequestMetrics]:
        """Get copy of active requests for monitoring."""
        return self._active_requests.copy()


class CorrelationIDMiddleware:
    """Middleware to ensure correlation IDs for all requests."""
    
    def __init__(self, next_handler):
        self.next_handler = next_handler
        self.logger = StructuredLogger("networkx_mcp.middleware")
    
    async def __call__(self, request):
        """Process request with correlation ID context."""
        # Generate correlation ID if not present
        correlation_id = getattr(request, 'id', None) or str(uuid.uuid4())
        
        # Set up correlation context
        async with CorrelationContext(
            correlation_id=correlation_id,
            operation_name=getattr(request, 'method', 'unknown'),
            request_id=str(getattr(request, 'id', correlation_id))
        ):
            try:
                # Add request metadata to context
                self._add_request_context(request)
                
                # Process request
                result = await self.next_handler(request)
                
                return result
                
            except Exception as e:
                self.logger.error(
                    "Request processing failed",
                    extra={
                        "correlation_id": correlation_id,
                        "method": getattr(request, 'method', 'unknown'),
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    },
                    exc_info=True
                )
                raise
    
    def _add_request_context(self, request):
        """Add request-specific context information."""
        from .correlation import add_request_context
        
        context = {
            "method": getattr(request, 'method', 'unknown'),
            "request_type": type(request).__name__
        }
        
        # Add params info without sensitive data
        if hasattr(request, 'params') and request.params:
            context["param_keys"] = list(request.params.keys())
            context["param_count"] = len(request.params)
            
        add_request_context(context)


def configure_production_logging():
    """Configure logging for production environment."""
    root_logger = logging.getLogger()
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create production handler
    handler = logging.StreamHandler(sys.stdout)
    
    if production_config.LOG_FORMAT == "json":
        from .formatters import JSONFormatter
        formatter = JSONFormatter()
    else:
        # Structured text format for development
        formatter = logging.Formatter(
            '%(asctime)s %(name)s %(levelname)s [%(correlation_id)s] %(message)s'
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, production_config.LOG_LEVEL))
    
    # Configure specific loggers
    logging.getLogger("networkx_mcp").setLevel(getattr(logging, production_config.LOG_LEVEL))
    logging.getLogger("aiohttp").setLevel(logging.WARNING)  # Reduce noise
    
    # Disable request logging in development
    if not production_config.is_production:
        logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


@contextmanager
def request_logging_context(method: str, 
                          request_id: str = None,
                          params: Dict[str, Any] = None,
                          user_id: str = None):
    """Context manager for automatic request logging."""
    logger = ProductionRequestLogger()
    
    actual_request_id = await logger.log_request_start(
        method=method,
        request_id=request_id,
        params=params,
        user_id=user_id
    )
    
    start_memory = None
    if production_config.is_production:
        import psutil
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    try:
        yield actual_request_id
        
        # Calculate memory usage
        memory_delta = None
        if start_memory:
            import psutil
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            
        await logger.log_request_end(
            request_id=actual_request_id,
            success=True,
            memory_usage_mb=memory_delta
        )
        
    except Exception as e:
        memory_delta = None
        if start_memory:
            import psutil
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            
        await logger.log_request_end(
            request_id=actual_request_id,
            success=False,
            error=e,
            memory_usage_mb=memory_delta
        )
        raise


# Global production logger instance
_production_logger: Optional[ProductionRequestLogger] = None


def get_production_logger() -> ProductionRequestLogger:
    """Get or create the global production logger."""
    global _production_logger
    if _production_logger is None:
        _production_logger = ProductionRequestLogger()
    return _production_logger


# Configure production logging on import if in production
if production_config.is_production:
    configure_production_logging()