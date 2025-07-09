"""Security middleware for MCP request processing.

This module provides security middleware for request authentication,
authorization, rate limiting, and request/response filtering.
"""

import hashlib
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration for middleware."""

    enable_auth: bool = True
    enable_rate_limiting: bool = True
    enable_request_logging: bool = True
    enable_response_filtering: bool = True

    # Rate limiting settings
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    rate_limit_burst: int = 20

    # Auth settings
    auth_token_header: str = "Authorization"
    auth_token_prefix: str = "Bearer "

    # Request size limits
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_response_size: int = 50 * 1024 * 1024  # 50MB


@dataclass
class RequestContext:
    """Context for tracking request state through middleware."""

    request_id: str
    client_id: str | None = None
    user_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    source_ip: str | None = None
    user_agent: str | None = None
    auth_token: str | None = None
    is_authenticated: bool = False
    permissions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int, burst_size: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst_size = burst_size
        self.buckets: dict[str, deque] = defaultdict(deque)
        self.tokens: dict[str, int] = defaultdict(lambda: burst_size)
        self.last_refill: dict[str, float] = defaultdict(time.time)

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for given identifier."""
        now = time.time()

        # Refill tokens based on time elapsed
        last_refill = self.last_refill[identifier]
        time_elapsed = now - last_refill
        tokens_to_add = int(time_elapsed * (self.max_requests / self.window_seconds))

        if tokens_to_add > 0:
            self.tokens[identifier] = min(
                self.burst_size, self.tokens[identifier] + tokens_to_add
            )
            self.last_refill[identifier] = now

        # Check if we have tokens available
        if self.tokens[identifier] > 0:
            self.tokens[identifier] -= 1

            # Track request times for sliding window
            bucket = self.buckets[identifier]
            bucket.append(now)

            # Clean old entries
            while bucket and bucket[0] < now - self.window_seconds:
                bucket.popleft()

            return True

        return False

    def get_status(self, identifier: str) -> dict[str, Any]:
        """Get rate limit status for identifier."""
        now = time.time()
        bucket = self.buckets[identifier]

        # Clean old entries
        while bucket and bucket[0] < now - self.window_seconds:
            bucket.popleft()

        return {
            "requests_in_window": len(bucket),
            "tokens_remaining": self.tokens[identifier],
            "window_reset_time": now + self.window_seconds,
            "is_rate_limited": self.tokens[identifier] == 0,
        }


class AuthenticationMiddleware:
    """Handles request authentication."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.valid_tokens: dict[str, dict[str, Any]] = {}
        self.revoked_tokens: set = set()

    def add_token(
        self,
        token: str,
        user_id: str,
        permissions: list[str],
        expires_at: float | None = None,
    ):
        """Add a valid authentication token."""
        self.valid_tokens[token] = {
            "user_id": user_id,
            "permissions": permissions,
            "expires_at": expires_at,
            "created_at": time.time(),
        }

    def revoke_token(self, token: str):
        """Revoke an authentication token."""
        if token in self.valid_tokens:
            del self.valid_tokens[token]
        self.revoked_tokens.add(token)

    def authenticate_request(self, headers: dict[str, str]) -> dict[str, Any] | None:
        """Authenticate request based on headers."""
        if not self.config.enable_auth:
            return {"user_id": "anonymous", "permissions": ["read"]}

        auth_header = headers.get(self.config.auth_token_header)
        if not auth_header:
            return None

        if not auth_header.startswith(self.config.auth_token_prefix):
            return None

        token = auth_header[len(self.config.auth_token_prefix) :]

        if token in self.revoked_tokens:
            return None

        token_info = self.valid_tokens.get(token)
        if not token_info:
            return None

        # Check expiration
        if token_info.get("expires_at") and time.time() > token_info["expires_at"]:
            self.revoke_token(token)
            return None

        return {
            "user_id": token_info["user_id"],
            "permissions": token_info["permissions"],
            "token": token,
        }


class SecurityMiddleware:
    """Main security middleware orchestrator."""

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or SecurityConfig()
        self.rate_limiter = RateLimiter(
            self.config.rate_limit_requests,
            self.config.rate_limit_window,
            self.config.rate_limit_burst,
        )
        self.auth_middleware = AuthenticationMiddleware(self.config)
        self.request_log: list[dict[str, Any]] = []

    def generate_request_id(self) -> str:
        """Generate unique request ID."""
        return hashlib.sha256(f"{time.time()}{id(self)}".encode()).hexdigest()[:16]

    def extract_client_identifier(
        self, headers: dict[str, str], source_ip: str | None = None
    ) -> str:
        """Extract client identifier for rate limiting."""
        # Try to get authenticated user first
        auth_info = self.auth_middleware.authenticate_request(headers)
        if auth_info and auth_info.get("user_id") != "anonymous":
            return f"user:{auth_info['user_id']}"

        # Fall back to IP address
        if source_ip:
            return f"ip:{source_ip}"

        # Last resort - use a generic identifier
        return "anonymous"

    async def process_request(
        self,
        request_data: dict[str, Any],
        headers: dict[str, str],
        source_ip: str | None = None,
    ) -> RequestContext:
        """Process incoming request through security middleware."""
        context = RequestContext(
            request_id=self.generate_request_id(),
            source_ip=source_ip,
            user_agent=headers.get("User-Agent"),
            timestamp=time.time(),
        )

        try:
            # Size validation
            if self.config.max_request_size:
                request_size = len(str(request_data).encode("utf-8"))
                if request_size > self.config.max_request_size:
                    raise SecurityError(
                        f"Request size {request_size} exceeds limit {self.config.max_request_size}"
                    )

            # Authentication
            if self.config.enable_auth:
                auth_info = self.auth_middleware.authenticate_request(headers)
                if auth_info:
                    context.user_id = auth_info["user_id"]
                    context.permissions = auth_info["permissions"]
                    context.auth_token = auth_info.get("token")
                    context.is_authenticated = True
                else:
                    context.is_authenticated = False
                    logger.warning(
                        f"Unauthenticated request {context.request_id} from {source_ip}"
                    )

            # Rate limiting
            if self.config.enable_rate_limiting:
                client_id = self.extract_client_identifier(headers, source_ip)
                context.client_id = client_id

                if not self.rate_limiter.is_allowed(client_id):
                    rate_status = self.rate_limiter.get_status(client_id)
                    raise RateLimitError(
                        f"Rate limit exceeded for {client_id}", rate_status
                    )

            # Request logging
            if self.config.enable_request_logging:
                self.log_request(context, request_data)

            logger.debug(f"Request {context.request_id} processed successfully")
            return context

        except Exception as e:
            logger.error(
                f"Security middleware failed for request {context.request_id}: {e}"
            )
            raise

    def process_response(self, context: RequestContext, response_data: Any) -> Any:
        """Process outgoing response through security middleware."""
        try:
            # Response size validation
            if self.config.max_response_size:
                response_size = len(str(response_data).encode("utf-8"))
                if response_size > self.config.max_response_size:
                    logger.warning(
                        f"Response size {response_size} exceeds limit for request {context.request_id}"
                    )
                    return {
                        "error": "Response too large",
                        "request_id": context.request_id,
                    }

            # Response filtering (remove sensitive data)
            if self.config.enable_response_filtering:
                response_data = self.filter_response(response_data, context)

            # Log response
            if self.config.enable_request_logging:
                self.log_response(context, response_data)

            return response_data

        except Exception as e:
            logger.error(
                f"Response processing failed for request {context.request_id}: {e}"
            )
            return {
                "error": "Response processing failed",
                "request_id": context.request_id,
            }

    def filter_response(self, response_data: Any, context: RequestContext) -> Any:
        """Filter sensitive data from responses."""
        if not isinstance(response_data, dict):
            return response_data

        # Remove sensitive fields if user doesn't have admin permissions
        if "admin" not in context.permissions:
            sensitive_fields = ["internal_id", "debug_info", "stack_trace", "_private"]
            for field in sensitive_fields:
                response_data.pop(field, None)

        return response_data

    def log_request(self, context: RequestContext, request_data: dict[str, Any]):
        """Log request for security monitoring."""
        log_entry = {
            "request_id": context.request_id,
            "timestamp": context.timestamp,
            "user_id": context.user_id,
            "client_id": context.client_id,
            "source_ip": context.source_ip,
            "user_agent": context.user_agent,
            "authenticated": context.is_authenticated,
            "method": request_data.get("method"),
            "tool": (
                request_data.get("params", {}).get("name")
                if request_data.get("params")
                else None
            ),
        }

        self.request_log.append(log_entry)

        # Keep only last 1000 entries
        if len(self.request_log) > 1000:
            self.request_log = self.request_log[-1000:]

        logger.info(f"Request logged: {log_entry}")

    def log_response(self, context: RequestContext, response_data: Any):
        """Log response for security monitoring."""
        log_entry = {
            "request_id": context.request_id,
            "timestamp": time.time(),
            "response_type": type(response_data).__name__,
            "has_error": (
                "error" in str(response_data)
                if isinstance(response_data, dict)
                else False
            ),
            "processing_time": time.time() - context.timestamp,
        }

        logger.debug(f"Response logged: {log_entry}")

    def get_security_status(self) -> dict[str, Any]:
        """Get security middleware status."""
        return {
            "rate_limiter_active": self.config.enable_rate_limiting,
            "authentication_active": self.config.enable_auth,
            "request_logging_active": self.config.enable_request_logging,
            "total_tokens": len(self.auth_middleware.valid_tokens),
            "revoked_tokens": len(self.auth_middleware.revoked_tokens),
            "logged_requests": len(self.request_log),
            "config": {
                "rate_limit_requests": self.config.rate_limit_requests,
                "rate_limit_window": self.config.rate_limit_window,
                "max_request_size": self.config.max_request_size,
                "max_response_size": self.config.max_response_size,
            },
        }


class SecurityError(Exception):
    """Base security exception."""



class RateLimitError(SecurityError):
    """Rate limit exceeded exception."""

    def __init__(self, message: str, rate_status: dict[str, Any]):
        super().__init__(message)
        self.rate_status = rate_status


class AuthenticationError(SecurityError):
    """Authentication failed exception."""



class AuthorizationError(SecurityError):
    """Authorization failed exception."""



def require_permissions(*required_permissions: str):
    """Decorator to require specific permissions for a function."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context from kwargs
            context = kwargs.get("context")
            if not context or not isinstance(context, RequestContext):
                raise AuthorizationError("No security context provided")

            # Check permissions
            if not context.is_authenticated:
                raise AuthenticationError("Authentication required")

            missing_permissions = set(required_permissions) - set(context.permissions)
            if missing_permissions:
                raise AuthorizationError(f"Missing permissions: {missing_permissions}")

            return await func(*args, **kwargs)

        return wrapper

    return decorator


class CORSMiddleware:
    """CORS (Cross-Origin Resource Sharing) middleware."""

    def __init__(
        self,
        allowed_origins: list[str] = None,
        allowed_methods: list[str] = None,
        allowed_headers: list[str] = None,
    ):
        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "OPTIONS",
        ]
        self.allowed_headers = allowed_headers or ["*"]

    def add_cors_headers(
        self, headers: dict[str, str], origin: str | None = None
    ) -> dict[str, str]:
        """Add CORS headers to response."""
        cors_headers = headers.copy()

        # Check origin
        if "*" in self.allowed_origins or (origin and origin in self.allowed_origins):
            cors_headers["Access-Control-Allow-Origin"] = origin or "*"

        cors_headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        cors_headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        cors_headers["Access-Control-Max-Age"] = "86400"  # 24 hours

        return cors_headers
