"""
Enterprise Rate Limiting and Resource Control

Comprehensive rate limiting system with:
- Token bucket algorithm for burst handling
- Per-user and per-operation limits
- Resource usage quotas
- Adaptive rate limiting
- Memory and execution time controls
"""

import time
import threading
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
from contextlib import contextmanager

from .config import get_config
from .monitoring import get_metrics_collector, get_audit_logger


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[float] = None
    reason: Optional[str] = None


class TokenBucket:
    """Token bucket for rate limiting with burst capability."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens per second to add
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        with self.lock:
            now = time.time()
            
            # Add tokens based on time elapsed
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_remaining(self) -> int:
        """Get number of remaining tokens."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            current_tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            return int(current_tokens)
    
    def time_to_refill(self, tokens: int) -> float:
        """Calculate time until enough tokens are available."""
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            
            needed = tokens - self.tokens
            return needed / self.refill_rate


class SlidingWindowCounter:
    """Sliding window counter for time-based rate limiting."""
    
    def __init__(self, window_size: int, max_requests: int):
        """
        Initialize sliding window counter.
        
        Args:
            window_size: Window size in seconds
            max_requests: Maximum requests in window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = threading.Lock()
    
    def add_request(self) -> bool:
        """
        Add a request and check if it's within limits.
        
        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        with self.lock:
            now = time.time()
            
            # Remove old requests outside the window
            while self.requests and self.requests[0] < now - self.window_size:
                self.requests.popleft()
            
            # Check if we're within limits
            if len(self.requests) >= self.max_requests:
                return False
            
            # Add current request
            self.requests.append(now)
            return True
    
    def get_remaining(self) -> int:
        """Get remaining requests in current window."""
        with self.lock:
            now = time.time()
            
            # Remove old requests
            while self.requests and self.requests[0] < now - self.window_size:
                self.requests.popleft()
            
            return max(0, self.max_requests - len(self.requests))
    
    def reset_time(self) -> float:
        """Get time when oldest request will expire."""
        with self.lock:
            if not self.requests:
                return time.time()
            return self.requests[0] + self.window_size


class ResourceMonitor:
    """Monitor and limit resource usage."""
    
    def __init__(self):
        self.config = get_config()
        self.active_operations = {}
        self.lock = threading.Lock()
    
    @contextmanager
    def track_operation(self, operation_id: str, user_id: str):
        """Context manager to track operation resource usage."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        with self.lock:
            self.active_operations[operation_id] = {
                'user_id': user_id,
                'start_time': start_time,
                'start_memory': start_memory
            }
        
        try:
            yield
        finally:
            with self.lock:
                self.active_operations.pop(operation_id, None)
    
    def check_memory_limit(self, user_id: str) -> bool:
        """Check if user is within memory limits."""
        current_memory = self._get_memory_usage()
        max_memory = self.config.rate_limit.max_memory_mb * 1024 * 1024
        
        return current_memory < max_memory
    
    def check_execution_time(self, operation_id: str) -> bool:
        """Check if operation is within time limits."""
        with self.lock:
            operation = self.active_operations.get(operation_id)
            if not operation:
                return True
            
            elapsed = time.time() - operation['start_time']
            return elapsed < self.config.rate_limit.max_execution_time
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except ImportError:
            # Fallback: estimate based on active operations
            return len(self.active_operations) * 10 * 1024 * 1024  # 10MB per operation estimate


class RateLimiter:
    """Main rate limiting coordinator."""
    
    def __init__(self):
        self.config = get_config()
        self.metrics = get_metrics_collector()
        self.audit = get_audit_logger()
        
        # Rate limiting storage
        self.user_buckets: Dict[str, TokenBucket] = {}
        self.user_windows: Dict[str, SlidingWindowCounter] = {}
        self.operation_counters: Dict[Tuple[str, str], SlidingWindowCounter] = {}
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Locks for thread safety
        self.bucket_lock = threading.Lock()
        self.window_lock = threading.Lock()
        self.operation_lock = threading.Lock()
    
    def check_rate_limit(self, user_id: str, operation: str = None) -> RateLimitResult:
        """
        Check if request is within rate limits.
        
        Args:
            user_id: User identifier
            operation: Optional operation name for operation-specific limits
            
        Returns:
            RateLimitResult with decision and metadata
        """
        if not self.config.rate_limit.enabled:
            return RateLimitResult(allowed=True, remaining=999999, reset_time=time.time())
        
        # Check per-minute rate limit (token bucket for burst handling)
        minute_result = self._check_minute_limit(user_id)
        if not minute_result.allowed:
            self.metrics.record_rate_limit_hit(user_id, 'per_minute')
            self.audit.log_security_event(
                'rate_limit_exceeded',
                'medium',
                {'limit_type': 'per_minute', 'user_id': user_id}
            )
            return minute_result
        
        # Check per-hour limit (sliding window)
        hour_result = self._check_hour_limit(user_id)
        if not hour_result.allowed:
            self.metrics.record_rate_limit_hit(user_id, 'per_hour')
            self.audit.log_security_event(
                'rate_limit_exceeded',
                'medium',
                {'limit_type': 'per_hour', 'user_id': user_id}
            )
            return hour_result
        
        # Check operation-specific limits
        if operation:
            op_result = self._check_operation_limit(user_id, operation)
            if not op_result.allowed:
                self.metrics.record_rate_limit_hit(user_id, f'operation_{operation}')
                return op_result
        
        # Check resource limits
        resource_result = self._check_resource_limits(user_id)
        if not resource_result.allowed:
            self.metrics.record_rate_limit_hit(user_id, 'resource_limit')
            return resource_result
        
        # All checks passed
        return RateLimitResult(
            allowed=True,
            remaining=min(minute_result.remaining, hour_result.remaining),
            reset_time=max(minute_result.reset_time, hour_result.reset_time)
        )
    
    def _check_minute_limit(self, user_id: str) -> RateLimitResult:
        """Check per-minute rate limit using token bucket."""
        with self.bucket_lock:
            if user_id not in self.user_buckets:
                self.user_buckets[user_id] = TokenBucket(
                    capacity=self.config.rate_limit.burst_size,
                    refill_rate=self.config.rate_limit.requests_per_minute / 60.0
                )
            
            bucket = self.user_buckets[user_id]
            
            if bucket.consume(1):
                return RateLimitResult(
                    allowed=True,
                    remaining=bucket.get_remaining(),
                    reset_time=time.time() + 60
                )
            else:
                retry_after = bucket.time_to_refill(1)
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=time.time() + retry_after,
                    retry_after=retry_after,
                    reason="Per-minute rate limit exceeded"
                )
    
    def _check_hour_limit(self, user_id: str) -> RateLimitResult:
        """Check per-hour rate limit using sliding window."""
        with self.window_lock:
            if user_id not in self.user_windows:
                self.user_windows[user_id] = SlidingWindowCounter(
                    window_size=3600,  # 1 hour
                    max_requests=self.config.rate_limit.requests_per_hour
                )
            
            window = self.user_windows[user_id]
            
            if window.add_request():
                return RateLimitResult(
                    allowed=True,
                    remaining=window.get_remaining(),
                    reset_time=window.reset_time()
                )
            else:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=window.reset_time(),
                    retry_after=window.reset_time() - time.time(),
                    reason="Per-hour rate limit exceeded"
                )
    
    def _check_operation_limit(self, user_id: str, operation: str) -> RateLimitResult:
        """Check operation-specific rate limits."""
        # Define operation-specific limits
        operation_limits = {
            'visualize_graph': 10,  # Expensive visualization operations
            'pagerank': 20,         # CPU-intensive algorithms
            'community_detection': 15,
            'betweenness_centrality': 10,
        }
        
        limit = operation_limits.get(operation, 60)  # Default to per-minute limit
        
        with self.operation_lock:
            key = (user_id, operation)
            if key not in self.operation_counters:
                self.operation_counters[key] = SlidingWindowCounter(
                    window_size=60,  # 1 minute window
                    max_requests=limit
                )
            
            counter = self.operation_counters[key]
            
            if counter.add_request():
                return RateLimitResult(
                    allowed=True,
                    remaining=counter.get_remaining(),
                    reset_time=counter.reset_time()
                )
            else:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=counter.reset_time(),
                    retry_after=counter.reset_time() - time.time(),
                    reason=f"Operation {operation} rate limit exceeded"
                )
    
    def _check_resource_limits(self, user_id: str) -> RateLimitResult:
        """Check resource usage limits."""
        # Check memory limit
        if not self.resource_monitor.check_memory_limit(user_id):
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=time.time() + 60,
                retry_after=60,
                reason="Memory limit exceeded"
            )
        
        return RateLimitResult(
            allowed=True,
            remaining=1000,  # Placeholder
            reset_time=time.time() + 60
        )
    
    @contextmanager
    def operation_tracking(self, user_id: str, operation: str):
        """Context manager for tracking operation resources."""
        operation_id = f"{user_id}_{operation}_{time.time()}"
        
        with self.resource_monitor.track_operation(operation_id, user_id):
            yield operation_id
            
        # Check if operation exceeded time limit
        if not self.resource_monitor.check_execution_time(operation_id):
            self.audit.log_security_event(
                'execution_timeout',
                'medium',
                {
                    'user_id': user_id,
                    'operation': operation,
                    'operation_id': operation_id
                }
            )
    
    def get_user_limits(self, user_id: str) -> Dict[str, Any]:
        """Get current rate limit status for user."""
        with self.bucket_lock, self.window_lock:
            minute_bucket = self.user_buckets.get(user_id)
            hour_window = self.user_windows.get(user_id)
            
            return {
                'requests_per_minute': {
                    'limit': self.config.rate_limit.requests_per_minute,
                    'remaining': minute_bucket.get_remaining() if minute_bucket else self.config.rate_limit.requests_per_minute,
                    'reset_time': time.time() + 60
                },
                'requests_per_hour': {
                    'limit': self.config.rate_limit.requests_per_hour,
                    'remaining': hour_window.get_remaining() if hour_window else self.config.rate_limit.requests_per_hour,
                    'reset_time': hour_window.reset_time() if hour_window else time.time() + 3600
                },
                'resource_limits': {
                    'max_memory_mb': self.config.rate_limit.max_memory_mb,
                    'max_execution_time': self.config.rate_limit.max_execution_time,
                    'max_graph_size': self.config.rate_limit.max_graph_size
                }
            }
    
    def reset_user_limits(self, user_id: str):
        """Reset rate limits for a user (admin function)."""
        with self.bucket_lock, self.window_lock, self.operation_lock:
            self.user_buckets.pop(user_id, None)
            self.user_windows.pop(user_id, None)
            
            # Remove operation-specific counters
            keys_to_remove = [key for key in self.operation_counters.keys() if key[0] == user_id]
            for key in keys_to_remove:
                self.operation_counters.pop(key, None)
        
        self.audit.log_security_event(
            'rate_limit_reset',
            'low',
            {'user_id': user_id, 'reset_by': 'admin'}
        )