"""Rate limiting and throttling services."""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass

from ..core.base import Component
from ..monitoring.logging import with_logging_context

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration."""

    requests: int  # Number of requests
    window: int  # Time window in seconds
    burst: int | None = None  # Burst capacity (defaults to requests)

    def __post_init__(self):
        if self.burst is None:
            self.burst = self.requests


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    reset_time: float
    retry_after: int | None = None


class TokenBucket:
    """Token bucket algorithm implementation."""

    def __init__(self, rate_limit: RateLimit):
        self.rate_limit = rate_limit
        self.tokens = float(rate_limit.burst)
        self.last_update = time.time()
        self.max_tokens = float(rate_limit.burst)
        self.refill_rate = rate_limit.requests / rate_limit.window

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket."""
        now = time.time()

        # Refill tokens based on elapsed time
        elapsed = now - self.last_update
        self.tokens = min(self.max_tokens, self.tokens + (elapsed * self.refill_rate))
        self.last_update = now

        # Try to consume tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def remaining_tokens(self) -> int:
        """Get the number of remaining tokens."""
        now = time.time()
        elapsed = now - self.last_update
        tokens = min(self.max_tokens, self.tokens + (elapsed * self.refill_rate))
        return int(tokens)

    def reset_time(self) -> float:
        """Get the time when bucket will be full again."""
        if self.tokens >= self.max_tokens:
            return time.time()

        time_to_full = (self.max_tokens - self.tokens) / self.refill_rate
        return time.time() + time_to_full


class SlidingWindow:
    """Sliding window rate limiter implementation."""

    def __init__(self, rate_limit: RateLimit):
        self.rate_limit = rate_limit
        self.requests = deque()

    def is_allowed(self) -> tuple[bool, int]:
        """Check if request is allowed and return remaining count."""
        now = time.time()
        window_start = now - self.rate_limit.window

        # Remove old requests outside the window
        while self.requests and self.requests[0] < window_start:
            self.requests.popleft()

        # Check if we can allow the request
        if len(self.requests) < self.rate_limit.requests:
            self.requests.append(now)
            remaining = self.rate_limit.requests - len(self.requests)
            return True, remaining
        else:
            remaining = 0
            return False, remaining

    def reset_time(self) -> float:
        """Get the time when oldest request will expire."""
        if not self.requests:
            return time.time()
        return self.requests[0] + self.rate_limit.window


class RateLimiter(Component):
    """Service for API rate limiting and throttling."""

    def __init__(self, algorithm: str = "token_bucket"):
        super().__init__("rate_limiter")
        self.algorithm = algorithm
        self.limiters: dict[str, object] = {}
        self.rate_limits: dict[str, RateLimit] = {}

        # Default rate limits
        self.default_limits = {
            "global": RateLimit(requests=100, window=60),  # 100 req/min global
            "user": RateLimit(requests=60, window=60),  # 60 req/min per user
            "ip": RateLimit(requests=30, window=60),  # 30 req/min per IP
        }

    def configure_rate_limit(self, key: str, rate_limit: RateLimit):
        """Configure rate limit for a specific key pattern."""
        self.rate_limits[key] = rate_limit
        logger.info(
            f"Configured rate limit for {key}: {rate_limit.requests}/{rate_limit.window}s"
        )

    def _get_limiter(self, identifier: str, rate_limit: RateLimit) -> object:
        """Get or create a rate limiter for an identifier."""
        if identifier not in self.limiters:
            if self.algorithm == "token_bucket":
                self.limiters[identifier] = TokenBucket(rate_limit)
            elif self.algorithm == "sliding_window":
                self.limiters[identifier] = SlidingWindow(rate_limit)
            else:
                raise ValueError(f"Unknown rate limiting algorithm: {self.algorithm}")

        return self.limiters[identifier]

    @with_logging_context(component="rate_limiter")
    async def check_rate_limit(
        self, identifier: str, limit_type: str = "user", tokens: int = 1
    ) -> RateLimitResult:
        """Check if request is within rate limits."""

        # Get rate limit configuration
        rate_limit = self.rate_limits.get(
            limit_type, self.default_limits.get(limit_type)
        )
        if not rate_limit:
            # No rate limit configured, allow request
            return RateLimitResult(
                allowed=True, remaining=999999, reset_time=time.time() + 3600
            )

        # Get limiter for this identifier
        limiter = self._get_limiter(f"{limit_type}:{identifier}", rate_limit)

        if self.algorithm == "token_bucket":
            allowed = limiter.consume(tokens)
            remaining = limiter.remaining_tokens()
            reset_time = limiter.reset_time()

            retry_after = None
            if not allowed:
                # Calculate retry after time
                retry_after = int(reset_time - time.time()) + 1

        elif self.algorithm == "sliding_window":
            allowed, remaining = limiter.is_allowed()
            reset_time = limiter.reset_time()

            retry_after = None
            if not allowed:
                retry_after = int(reset_time - time.time()) + 1

        result = RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after,
        )

        if not allowed:
            logger.warning(f"Rate limit exceeded for {identifier} ({limit_type})")

        return result

    @with_logging_context(component="rate_limiter")
    async def check_multiple_limits(
        self, checks: list[tuple[str, str]]  # [(identifier, limit_type), ...]
    ) -> RateLimitResult:
        """Check multiple rate limits and return most restrictive result."""
        results = []

        for identifier, limit_type in checks:
            result = await self.check_rate_limit(identifier, limit_type)
            results.append(result)

        # Find most restrictive result
        most_restrictive = min(results, key=lambda r: r.remaining)

        # If any check failed, return failure
        if any(not r.allowed for r in results):
            failed_result = next(r for r in results if not r.allowed)
            return RateLimitResult(
                allowed=False,
                remaining=failed_result.remaining,
                reset_time=failed_result.reset_time,
                retry_after=failed_result.retry_after,
            )

        return most_restrictive

    async def cleanup_expired_limiters(self) -> int:
        """Clean up expired rate limiters to prevent memory leaks."""
        now = time.time()
        expired_keys = []

        for key, limiter in self.limiters.items():
            # Consider limiter expired if not used for 1 hour
            if hasattr(limiter, "last_update"):
                if now - limiter.last_update > 3600:
                    expired_keys.append(key)
            elif hasattr(limiter, "requests"):
                # For sliding window, check if any recent requests
                if not limiter.requests or (now - limiter.requests[-1]) > 3600:
                    expired_keys.append(key)

        for key in expired_keys:
            del self.limiters[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired rate limiters")

        return len(expired_keys)

    async def get_rate_limit_status(self, identifier: str, limit_type: str) -> dict:
        """Get current rate limit status for an identifier."""
        rate_limit = self.rate_limits.get(
            limit_type, self.default_limits.get(limit_type)
        )
        if not rate_limit:
            return {"configured": False}

        limiter_key = f"{limit_type}:{identifier}"
        if limiter_key not in self.limiters:
            return {
                "configured": True,
                "limit": rate_limit.requests,
                "window": rate_limit.window,
                "remaining": rate_limit.requests,
                "reset_time": time.time() + rate_limit.window,
            }

        limiter = self.limiters[limiter_key]

        if self.algorithm == "token_bucket":
            remaining = limiter.remaining_tokens()
            reset_time = limiter.reset_time()
        else:  # sliding_window
            _, remaining = limiter.is_allowed()
            # Undo the request we just added for status check
            if limiter.requests:
                limiter.requests.pop()
            reset_time = limiter.reset_time()

        return {
            "configured": True,
            "limit": rate_limit.requests,
            "window": rate_limit.window,
            "remaining": remaining,
            "reset_time": reset_time,
        }

    async def initialize(self) -> None:
        """Initialize the rate limiter."""
        await super().initialize()

        # Start cleanup task
        asyncio.create_task(self._cleanup_loop())

        logger.info(f"Rate limiter initialized with {self.algorithm} algorithm")

    async def _cleanup_loop(self):
        """Background task to clean up expired limiters."""
        while self.status.is_running():
            try:
                await self.cleanup_expired_limiters()
                await asyncio.sleep(300)  # Clean up every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rate limiter cleanup: {e}")
                await asyncio.sleep(60)


class RateLimitMiddleware:
    """Middleware for applying rate limits to requests."""

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter

    async def __call__(self, request, handler):
        """Apply rate limiting to incoming requests."""
        # Extract identifiers
        user_id = (
            getattr(request, "user", {}).get("user_id")
            if hasattr(request, "user")
            else None
        )
        client_ip = self._get_client_ip(request)

        # Prepare rate limit checks
        checks = [
            ("global", "global"),  # Global rate limit
        ]

        if client_ip:
            checks.append((client_ip, "ip"))

        if user_id:
            checks.append((user_id, "user"))

        # Check rate limits
        result = await self.rate_limiter.check_multiple_limits(checks)

        # Add rate limit headers to response
        def add_rate_limit_headers(response):
            response.headers["X-RateLimit-Remaining"] = str(result.remaining)
            response.headers["X-RateLimit-Reset"] = str(int(result.reset_time))

            if not result.allowed and result.retry_after:
                response.headers["Retry-After"] = str(result.retry_after)

            return response

        # If rate limit exceeded, return 429 response
        if not result.allowed:
            from aiohttp import web

            response = web.Response(
                status=429,
                text="Rate limit exceeded",
                headers={
                    "X-RateLimit-Remaining": str(result.remaining),
                    "X-RateLimit-Reset": str(int(result.reset_time)),
                    "Retry-After": str(result.retry_after or 60),
                },
            )
            return response

        # Process request and add headers to response
        response = await handler(request)
        return add_rate_limit_headers(response)

    def _get_client_ip(self, request) -> str | None:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fall back to remote address
        if hasattr(request, "remote"):
            return request.remote

        return None
