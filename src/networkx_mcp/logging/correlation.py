"""Correlation ID management for request tracing."""

import asyncio
import contextvars
import functools
import time
import uuid
from typing import Any, Callable, Dict, Optional

# Context variable to store correlation ID across async calls
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'correlation_id', default=None
)

# Context variable to store request metadata
request_context_var: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    'request_context', default={}
)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context."""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get the correlation ID from the current context."""
    return correlation_id_var.get()


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context."""
    correlation_id_var.set(None)


def set_request_context(key: str, value: Any) -> None:
    """Set a value in the request context."""
    context = request_context_var.get({}).copy()  # Make a copy to avoid mutation
    context[key] = value
    request_context_var.set(context)


def get_request_context(key: str = None) -> Any:
    """Get a value from the request context."""
    context = request_context_var.get({})
    if key is None:
        return context
    return context.get(key)


def clear_request_context() -> None:
    """Clear the request context."""
    request_context_var.set({})


class CorrelationContext:
    """Context manager for correlation ID and request tracking."""
    
    def __init__(
        self, 
        correlation_id: Optional[str] = None,
        operation_name: Optional[str] = None,
        **context_data
    ):
        self.correlation_id = correlation_id or generate_correlation_id()
        self.operation_name = operation_name
        self.context_data = context_data
        self.start_time = None
        self.previous_correlation_id = None
        self.previous_context = None
        
    def __enter__(self):
        # Store previous values
        self.previous_correlation_id = get_correlation_id()
        self.previous_context = get_request_context().copy() if get_request_context() else {}
        
        # Set new values
        set_correlation_id(self.correlation_id)
        clear_request_context()
        
        # Set operation context
        if self.operation_name:
            set_request_context("operation", self.operation_name)
        
        # Set additional context data
        for key, value in self.context_data.items():
            set_request_context(key, value)
        
        # Track timing
        self.start_time = time.time()
        set_request_context("start_time", self.start_time)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate duration
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            set_request_context("duration_ms", duration_ms)
        
        # Restore previous values
        if self.previous_correlation_id:
            set_correlation_id(self.previous_correlation_id)
        else:
            clear_correlation_id()
            
        if self.previous_context:
            request_context_var.set(self.previous_context)
        else:
            clear_request_context()
    
    async def __aenter__(self):
        return self.__enter__()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


def correlation_middleware(func: Callable) -> Callable:
    """Decorator to automatically set correlation ID for functions."""
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Check if correlation ID already exists
        existing_id = get_correlation_id()
        if existing_id:
            # Already in a correlation context, just call the function
            return func(*args, **kwargs)
        
        # Create new correlation context
        operation_name = f"{func.__module__}.{func.__name__}"
        with CorrelationContext(operation_name=operation_name):
            return func(*args, **kwargs)
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Check if correlation ID already exists
        existing_id = get_correlation_id()
        if existing_id:
            # Already in a correlation context, just call the function
            return await func(*args, **kwargs)
        
        # Create new correlation context
        operation_name = f"{func.__module__}.{func.__name__}"
        async with CorrelationContext(operation_name=operation_name):
            return await func(*args, **kwargs)
    
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def trace_request(operation_name: str = None):
    """Decorator for tracing request flow with correlation ID."""
    
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with CorrelationContext(operation_name=op_name):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with CorrelationContext(operation_name=op_name):
                return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_correlation_id(correlation_id: str):
    """Decorator to set a specific correlation ID."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with CorrelationContext(correlation_id=correlation_id):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with CorrelationContext(correlation_id=correlation_id):
                return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator