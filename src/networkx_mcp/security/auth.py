"""Authentication and authorization services."""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from ..core.base import Component
# Monitoring module removed - using simple logging instead

logger = logging.getLogger(__name__)


@dataclass
class User:
    """User information."""

    user_id: str
    username: str
    email: str | None = None
    roles: set[str] = None
    permissions: set[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.roles is None:
            self.roles = set()
        if self.permissions is None:
            self.permissions = set()
        if self.metadata is None:
            self.metadata = {}

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions

    def has_any_role(self, roles: list[str]) -> bool:
        """Check if user has any of the specified roles."""
        return any(self.has_role(role) for role in roles)

    def has_all_permissions(self, permissions: list[str]) -> bool:
        """Check if user has all specified permissions."""
        return all(self.has_permission(perm) for perm in permissions)


@dataclass
class AuthToken:
    """Authentication token."""

    token: str
    user_id: str
    issued_at: float
    expires_at: float
    token_type: str = "bearer"
    scopes: set[str] = None

    def __post_init__(self):
        if self.scopes is None:
            self.scopes = set()

    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() > self.expires_at

    def has_scope(self, scope: str) -> bool:
        """Check if token has a specific scope."""
        return scope in self.scopes


class TokenValidator:
    """Service for validating authentication tokens."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode("utf-8")

    def create_token(
        self, user_id: str, expires_in: int = 3600, scopes: set[str] | None = None
    ) -> AuthToken:
        """Create a new authentication token."""
        now = time.time()
        expires_at = now + expires_in

        payload = {
            "user_id": user_id,
            "iat": now,
            "exp": expires_at,
            "scopes": list(scopes or set()),
        }

        # Simple HMAC-based token (in production, use JWT)
        payload_json = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            self.secret_key, payload_json.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        token = f"{payload_json}:{signature}"

        return AuthToken(
            token=token,
            user_id=user_id,
            issued_at=now,
            expires_at=expires_at,
            scopes=scopes or set(),
        )

    def validate_token(self, token: str) -> AuthToken | None:
        """Validate an authentication token."""
        try:
            # Split token and signature
            if ":" not in token:
                return None

            payload_json, signature = token.rsplit(":", 1)

            # Verify signature
            expected_signature = hmac.new(
                self.secret_key, payload_json.encode("utf-8"), hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                return None

            # Parse payload
            payload = json.loads(payload_json)

            auth_token = AuthToken(
                token=token,
                user_id=payload["user_id"],
                issued_at=payload["iat"],
                expires_at=payload["exp"],
                scopes=set(payload.get("scopes", [])),
            )

            # Check expiration
            if auth_token.is_expired():
                return None

            return auth_token

        except (json.JSONDecodeError, KeyError, ValueError):
            return None


class AuthService(Component):
    """Service for authentication and authorization."""

    def __init__(self, secret_key: str):
        super().__init__("auth_service")
        self.token_validator = TokenValidator(secret_key)
        self.users: dict[str, User] = {}
        self.user_credentials: dict[str, str] = {}  # username -> password_hash
        self.active_tokens: dict[str, AuthToken] = {}

        # Role-based permissions
        self.role_permissions: dict[str, set[str]] = {
            "admin": {
                "graph:read",
                "graph:write",
                "graph:delete",
                "algorithm:execute",
                "admin:manage",
                "system:monitor",
            },
            "user": {"graph:read", "graph:write", "algorithm:execute"},
            "readonly": {"graph:read"},
        }

    # @with_logging_context removed - monitoring module deleted
    async def register_user(
        self,
        username: str,
        password: str,
        email: str | None = None,
        roles: set[str] | None = None,
    ) -> User:
        """Register a new user."""
        user_id = f"user_{len(self.users) + 1}"

        # Hash password (in production, use bcrypt)
        password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()

        # Calculate permissions based on roles
        roles = roles or {"user"}
        permissions = set()
        for role in roles:
            permissions.update(self.role_permissions.get(role, set()))

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles,
            permissions=permissions,
        )

        self.users[user_id] = user
        self.user_credentials[username] = password_hash

        logger.info(f"Registered user: {username} with roles: {roles}")
        return user

    # @with_logging_context removed - monitoring module deleted
    async def authenticate_user(self, username: str, password: str) -> AuthToken | None:
        """Authenticate a user with username/password."""
        # Verify credentials
        password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        stored_hash = self.user_credentials.get(username)

        if not stored_hash or not hmac.compare_digest(password_hash, stored_hash):
            logger.warning(f"Failed authentication attempt for user: {username}")
            return None

        # Find user
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break

        if not user:
            return None

        # Create token
        token = self.token_validator.create_token(
            user_id=user.user_id, scopes=user.permissions
        )

        self.active_tokens[token.token] = token

        logger.info(f"User authenticated: {username}")
        return token

    # @with_logging_context removed - monitoring module deleted
    async def validate_token(self, token: str) -> AuthToken | None:
        """Validate an authentication token."""
        auth_token = self.token_validator.validate_token(token)

        if not auth_token:
            return None

        # Check if token is in active tokens
        if token not in self.active_tokens:
            return None

        return auth_token

    # @with_logging_context removed - monitoring module deleted
    async def get_user(self, user_id: str) -> User | None:
        """Get user by ID."""
        return self.users.get(user_id)

    # @with_logging_context removed - monitoring module deleted
    async def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has a specific permission."""
        user = await self.get_user(user_id)
        if not user:
            return False

        return user.has_permission(permission)

    # @with_logging_context removed - monitoring module deleted
    async def revoke_token(self, token: str) -> bool:
        """Revoke an authentication token."""
        if token in self.active_tokens:
            del self.active_tokens[token]
            logger.info("Token revoked")
            return True
        return False

    # @with_logging_context removed - monitoring module deleted
    async def cleanup_expired_tokens(self) -> int:
        """Clean up expired tokens."""
        now = time.time()
        expired_tokens = [
            token
            for token, auth_token in self.active_tokens.items()
            if auth_token.expires_at < now
        ]

        for token in expired_tokens:
            del self.active_tokens[token]

        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")

        return len(expired_tokens)

    async def initialize(self) -> None:
        """Initialize the auth service."""
        await super().initialize()

        # Create default admin user
        await self.register_user(
            "admin",
            "admin123",  # Change in production!
            email="admin@example.com",
            roles={"admin"},
        )

        # Start token cleanup task
        asyncio.create_task(self._token_cleanup_loop())

        logger.info("Auth service initialized")

    async def _token_cleanup_loop(self):
        """Background task to clean up expired tokens."""
        while self.status.is_running():
            try:
                await self.cleanup_expired_tokens()
                await asyncio.sleep(300)  # Clean up every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in token cleanup: {e}")
                await asyncio.sleep(60)


class AuthMiddleware:
    """Middleware for handling authentication."""

    def __init__(self, auth_service: AuthService):
        self.auth_service = auth_service

    async def __call__(self, request, handler):
        """Process authentication for incoming requests."""
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            # Allow unauthenticated requests to proceed
            request.user = None
            request.auth_token = None
            return await handler(request)

        token = auth_header[7:]  # Remove 'Bearer ' prefix

        # Validate token
        auth_token = await self.auth_service.validate_token(token)

        if auth_token:
            user = await self.auth_service.get_user(auth_token.user_id)
            request.user = user
            request.auth_token = auth_token
        else:
            request.user = None
            request.auth_token = None

        return await handler(request)


def require_auth(permission: str | None = None, roles: list[str] | None = None):
    """Decorator to require authentication and optionally specific permissions/roles."""

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                # Extract request from args (assuming it's the first argument)
                request = args[0] if args else None

                if not hasattr(request, "user") or not request.user:
                    raise PermissionError("Authentication required")

                user = request.user

                # Check permission
                if permission and not user.has_permission(permission):
                    raise PermissionError(f"Permission required: {permission}")

                # Check roles
                if roles and not user.has_any_role(roles):
                    raise PermissionError(f"Role required: one of {roles}")

                return await func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                request = args[0] if args else None

                if not hasattr(request, "user") or not request.user:
                    raise PermissionError("Authentication required")

                user = request.user

                if permission and not user.has_permission(permission):
                    raise PermissionError(f"Permission required: {permission}")

                if roles and not user.has_any_role(roles):
                    raise PermissionError(f"Role required: one of {roles}")

                return func(*args, **kwargs)

            return sync_wrapper

    return decorator
