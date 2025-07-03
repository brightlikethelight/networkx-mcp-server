"""Circuit breaker pattern for resilient service operations."""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..core.base import Component

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures to trigger open
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Request timeout in seconds

    # Sliding window configuration
    window_size: int = 100  # Size of sliding window
    minimum_requests: int = 10  # Minimum requests before considering failure rate


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    state_transitions: int = 0


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""



class CircuitBreaker(Component):
    """Circuit breaker implementation for resilient operations."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        super().__init__(f"circuit_breaker_{name}")
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = time.time()
        self._lock = asyncio.Lock()

        # Sliding window for tracking recent requests
        self.request_window = []

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function through the circuit breaker."""
        async with self._lock:
            # Check if we should reject the request
            if await self._should_reject_request():
                self.stats.rejected_requests += 1
                raise CircuitBreakerError(f"Circuit breaker '{self.name}' is open")

            # Update state if needed
            await self._update_state()

        # Execute the function
        start_time = time.time()

        try:
            # Apply timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=self.config.timeout
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, func, *args, **kwargs
                    ),
                    timeout=self.config.timeout,
                )

            # Record success
            await self._record_success()
            return result

        except Exception as e:
            # Record failure
            await self._record_failure(e)
            raise

    async def _should_reject_request(self) -> bool:
        """Check if request should be rejected."""
        if self.state == CircuitState.OPEN:
            return True

        if self.state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            return self.success_count >= self.config.success_threshold

        return False

    async def _update_state(self) -> None:
        """Update circuit breaker state based on current conditions."""
        now = time.time()

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if now - self.last_state_change >= self.config.recovery_timeout:
                await self._transition_to_half_open()

        elif self.state == CircuitState.HALF_OPEN:
            # Check if we should close (enough successes) or open (any failure)
            if self.success_count >= self.config.success_threshold:
                await self._transition_to_closed()

        elif self.state == CircuitState.CLOSED:
            # Check if we should open due to failures
            if await self._should_open():
                await self._transition_to_open()

    async def _should_open(self) -> bool:
        """Check if circuit should open due to failures."""
        # Simple threshold-based approach
        if self.failure_count >= self.config.failure_threshold:
            return True

        # Sliding window approach
        if len(self.request_window) >= self.config.minimum_requests:
            recent_failures = sum(1 for success in self.request_window if not success)
            failure_rate = recent_failures / len(self.request_window)

            # Open if failure rate is above 50% and we have enough failures
            if failure_rate > 0.5 and recent_failures >= self.config.failure_threshold:
                return True

        return False

    async def _transition_to_open(self) -> None:
        """Transition to open state."""
        logger.warning(f"Circuit breaker '{self.name}' opening due to failures")
        self.state = CircuitState.OPEN
        self.last_state_change = time.time()
        self.stats.state_transitions += 1
        self.success_count = 0

    async def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        logger.info(f"Circuit breaker '{self.name}' transitioning to half-open")
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = time.time()
        self.stats.state_transitions += 1
        self.success_count = 0
        self.failure_count = 0

    async def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        logger.info(f"Circuit breaker '{self.name}' closing - service recovered")
        self.state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self.stats.state_transitions += 1
        self.failure_count = 0
        self.success_count = 0

    async def _record_success(self) -> None:
        """Record a successful operation."""
        async with self._lock:
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.last_success_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0  # Reset failure count on success

            # Update sliding window
            self.request_window.append(True)
            if len(self.request_window) > self.config.window_size:
                self.request_window.pop(0)

    async def _record_failure(self, error: Exception) -> None:
        """Record a failed operation."""
        async with self._lock:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            self.stats.last_failure_time = time.time()

            if self.state == CircuitState.CLOSED:
                self.failure_count += 1
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state should open the circuit
                await self._transition_to_open()

            # Update sliding window
            self.request_window.append(False)
            if len(self.request_window) > self.config.window_size:
                self.request_window.pop(0)

            logger.warning(f"Circuit breaker '{self.name}' recorded failure: {error}")

    async def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        now = time.time()

        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_state_change": self.last_state_change,
            "seconds_since_state_change": now - self.last_state_change,
            "stats": {
                "total_requests": self.stats.total_requests,
                "successful_requests": self.stats.successful_requests,
                "failed_requests": self.stats.failed_requests,
                "rejected_requests": self.stats.rejected_requests,
                "state_transitions": self.stats.state_transitions,
                "success_rate": (
                    self.stats.successful_requests / self.stats.total_requests
                    if self.stats.total_requests > 0
                    else 0
                ),
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            },
        }

    async def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        async with self._lock:
            logger.info(f"Resetting circuit breaker '{self.name}'")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_state_change = time.time()
            self.request_window.clear()

    async def force_open(self) -> None:
        """Force circuit breaker to open state."""
        async with self._lock:
            logger.warning(f"Forcing circuit breaker '{self.name}' to open")
            await self._transition_to_open()


class CircuitBreakerRegistry(Component):
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        super().__init__("circuit_breaker_registry")
        self.breakers: dict[str, CircuitBreaker] = {}
        self.default_config = CircuitBreakerConfig()

    def get_breaker(
        self, name: str, config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config or self.default_config)
            logger.info(f"Created circuit breaker: {name}")

        return self.breakers[name]

    async def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers."""
        status = {}
        for name, breaker in self.breakers.items():
            status[name] = await breaker.get_status()
        return status

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            await breaker.reset()
        logger.info("Reset all circuit breakers")


class CircuitBreakerMiddleware:
    """Middleware for applying circuit breaker pattern to requests."""

    def __init__(self, registry: CircuitBreakerRegistry):
        self.registry = registry

    async def __call__(self, request, handler):
        """Apply circuit breaker to request handling."""
        # Determine circuit breaker name based on request
        breaker_name = self._get_breaker_name(request)

        if not breaker_name:
            # No circuit breaker needed for this request
            return await handler(request)

        # Get circuit breaker
        breaker = self.registry.get_breaker(breaker_name)

        # Execute through circuit breaker
        try:
            return await breaker.call(handler, request)
        except CircuitBreakerError:
            # Return 503 Service Unavailable when circuit is open
            from aiohttp import web

            return web.Response(
                status=503,
                text="Service temporarily unavailable",
                headers={
                    "Retry-After": "60",
                    "X-Circuit-Breaker": breaker_name,
                    "X-Circuit-State": breaker.state.value,
                },
            )

    def _get_breaker_name(self, request) -> str | None:
        """Determine circuit breaker name for request."""
        # Example logic - customize based on your needs
        path = getattr(request, "path", "/")

        if "/graph" in path:
            return "graph_operations"
        elif "/algorithm" in path:
            return "algorithm_execution"
        elif "/admin" in path:
            return "admin_operations"

        return None


# Decorator for circuit breaker protection
def circuit_breaker(name: str, config: CircuitBreakerConfig | None = None):
    """Decorator to protect functions with circuit breaker."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Get registry (would be injected in real app)
            from ..core.container import get_container

            container = get_container()
            registry = container.get(CircuitBreakerRegistry)

            breaker = registry.get_breaker(name, config)
            return await breaker.call(func, *args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            # For sync functions, create async wrapper
            return asyncio.run(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
