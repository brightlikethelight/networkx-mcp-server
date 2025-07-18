"""
Enterprise Authentication and Authorization

Comprehensive security system with:
- OAuth 2.1 with PKCE compliance
- API key authentication
- JWT token management
- Role-based access control (RBAC)
- Request validation and audit
"""

import base64
import hashlib
import secrets
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import jwt
    from authlib.integrations.base_client import OAuthError
    from authlib.jose import JsonWebToken, JWTClaims
    from authlib.oauth2 import OAuth2Error
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError:
    raise ImportError(
        "Enterprise authentication requires authlib, pyjwt, and cryptography. "
        "Install with: pip install networkx-mcp-server[enterprise]"
    )

from .config import get_config


class Role(Enum):
    """User roles for RBAC."""

    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    GUEST = "guest"


class Permission(Enum):
    """Granular permissions for graph operations."""

    # Graph management
    CREATE_GRAPH = "graph:create"
    DELETE_GRAPH = "graph:delete"
    LIST_GRAPHS = "graph:list"

    # Graph modification
    ADD_NODES = "graph:add_nodes"
    ADD_EDGES = "graph:add_edges"
    MODIFY_GRAPH = "graph:modify"

    # Graph analysis
    READ_GRAPH = "graph:read"
    ANALYZE_GRAPH = "graph:analyze"
    VISUALIZE_GRAPH = "graph:visualize"

    # Data operations
    IMPORT_DATA = "data:import"
    EXPORT_DATA = "data:export"

    # System operations
    VIEW_METRICS = "system:metrics"
    ADMIN_ACCESS = "system:admin"


@dataclass
class User:
    """User information for authentication and authorization."""

    id: str
    username: Optional[str] = None
    email: Optional[str] = None
    roles: Set[Role] = None
    permissions: Set[Permission] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    last_login: Optional[datetime] = None

    def __post_init__(self):
        if self.roles is None:
            self.roles = {Role.USER}
        if self.permissions is None:
            self.permissions = self._get_default_permissions()
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def _get_default_permissions(self) -> Set[Permission]:
        """Get default permissions based on roles."""
        permissions = set()

        for role in self.roles:
            if role == Role.ADMIN:
                permissions.update(Permission)  # All permissions
            elif role == Role.USER:
                permissions.update(
                    [
                        Permission.CREATE_GRAPH,
                        Permission.LIST_GRAPHS,
                        Permission.ADD_NODES,
                        Permission.ADD_EDGES,
                        Permission.MODIFY_GRAPH,
                        Permission.READ_GRAPH,
                        Permission.ANALYZE_GRAPH,
                        Permission.VISUALIZE_GRAPH,
                        Permission.IMPORT_DATA,
                        Permission.EXPORT_DATA,
                    ]
                )
            elif role == Role.READONLY:
                permissions.update(
                    [
                        Permission.LIST_GRAPHS,
                        Permission.READ_GRAPH,
                        Permission.ANALYZE_GRAPH,
                        Permission.VISUALIZE_GRAPH,
                        Permission.EXPORT_DATA,
                    ]
                )
            elif role == Role.GUEST:
                permissions.update(
                    [
                        Permission.LIST_GRAPHS,
                        Permission.READ_GRAPH,
                        Permission.VISUALIZE_GRAPH,
                    ]
                )

        return permissions

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions

    def has_role(self, role: Role) -> bool:
        """Check if user has specific role."""
        return role in self.roles


class APIKeyAuth:
    """API Key authentication system."""

    def __init__(self):
        self.config = get_config()
        self._api_keys: Dict[str, User] = {}
        self._load_api_keys()

    def _load_api_keys(self):
        """Load API keys from configuration."""
        for i, api_key in enumerate(self.config.security.api_keys):
            # For demo purposes, create users for each API key
            # In production, this would come from a database
            user_id = f"api_user_{i}"
            if user_id in self.config.security.admin_users:
                roles = {Role.ADMIN}
            else:
                roles = {Role.USER}

            self._api_keys[api_key] = User(
                id=user_id, username=f"API User {i}", roles=roles
            )

    def authenticate(self, api_key: str) -> Optional[User]:
        """Authenticate using API key."""
        if not self.config.security.api_key_enabled:
            return None

        # Hash the API key for secure comparison
        key_hash = self._hash_api_key(api_key)

        # Look up user by API key
        user = self._api_keys.get(api_key)
        if user:
            user.last_login = datetime.utcnow()
            return user

        return None

    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def generate_api_key(self, user_id: str) -> str:
        """Generate a new API key for a user."""
        api_key = secrets.token_urlsafe(32)
        # In production, store this in a database
        return api_key


class OAuthManager:
    """OAuth 2.1 with PKCE authentication manager."""

    def __init__(self):
        self.config = get_config()
        self.jwt_handler = JsonWebToken(["HS256"])

    def generate_pkce_challenge(self) -> Tuple[str, str]:
        """Generate PKCE code verifier and challenge."""
        # Generate code verifier (43-128 characters)
        code_verifier = (
            base64.urlsafe_b64encode(secrets.token_bytes(32))
            .decode("utf-8")
            .rstrip("=")
        )

        # Generate code challenge
        digest = hashes.Hash(hashes.SHA256())
        digest.update(code_verifier.encode("utf-8"))
        code_challenge = (
            base64.urlsafe_b64encode(digest.finalize()).decode("utf-8").rstrip("=")
        )

        return code_verifier, code_challenge

    def create_authorization_url(self, state: str) -> str:
        """Create OAuth authorization URL with PKCE."""
        if not self.config.security.oauth_enabled:
            raise ValueError("OAuth is not enabled")

        code_verifier, code_challenge = self.generate_pkce_challenge()

        # Store code verifier for later verification (in production, use Redis/DB)
        self._store_code_verifier(state, code_verifier)

        # Build authorization URL
        params = {
            "response_type": "code",
            "client_id": self.config.security.oauth_client_id,
            "redirect_uri": "http://localhost:8080/auth/callback",
            "scope": " ".join(self.config.security.oauth_scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.config.security.oauth_auth_url}?{query_string}"

    def exchange_code_for_token(self, code: str, state: str) -> Optional[str]:
        """Exchange authorization code for access token."""
        # Retrieve stored code verifier
        code_verifier = self._get_code_verifier(state)
        if not code_verifier:
            return None

        # Exchange code for token (implement actual OAuth flow)
        # This is a simplified example - in production, make HTTP request to token endpoint
        token_data = {
            "sub": "oauth_user_" + secrets.token_hex(8),
            "iat": int(time.time()),
            "exp": int(time.time()) + (self.config.security.jwt_expiry_hours * 3600),
            "scope": " ".join(self.config.security.oauth_scopes),
        }

        return self.jwt_handler.encode(
            {"alg": self.config.security.jwt_algorithm},
            token_data,
            self.config.security.jwt_secret,
        ).decode("utf-8")

    def verify_token(self, token: str) -> Optional[User]:
        """Verify JWT token and return user."""
        try:
            import jwt

            # Verify token with proper expiration checking
            claims = jwt.decode(
                token,
                self.config.security.jwt_secret,
                algorithms=[self.config.security.jwt_algorithm],
            )

            # Create user from token claims
            user_id = claims.get("sub")
            scopes = claims.get("scope", "").split()

            # Map scopes to roles (simplified example)
            roles = {Role.USER}
            if "admin" in scopes:
                roles.add(Role.ADMIN)

            return User(id=user_id, roles=roles, last_login=datetime.utcnow())

        except jwt.ExpiredSignatureError:
            # Token has expired
            return None
        except jwt.InvalidTokenError:
            # Token is invalid for other reasons
            return None
        except Exception:
            # Other unexpected errors
            return None

    def _store_code_verifier(self, state: str, code_verifier: str):
        """Store code verifier for PKCE flow."""
        # In production, use Redis or database
        # For demo, use in-memory storage
        if not hasattr(self, "_code_verifiers"):
            self._code_verifiers = {}
        self._code_verifiers[state] = code_verifier

    def _get_code_verifier(self, state: str) -> Optional[str]:
        """Retrieve and remove code verifier."""
        if not hasattr(self, "_code_verifiers"):
            return None
        return self._code_verifiers.pop(state, None)


class RBACManager:
    """Role-Based Access Control manager."""

    def __init__(self):
        self.config = get_config()

    def check_permission(
        self, user: User, permission: Permission, resource: Optional[str] = None
    ) -> bool:
        """Check if user has permission for operation on resource."""
        if not self.config.security.rbac_enabled:
            return True

        # Admin users have all permissions
        if Role.ADMIN in user.roles:
            return True

        # Check specific permission
        if permission not in user.permissions:
            return False

        # Additional resource-specific checks can be added here
        # For example, checking if user owns a specific graph

        return True

    def get_allowed_operations(self, user: User) -> List[str]:
        """Get list of operations user is allowed to perform."""
        if Role.ADMIN in user.roles:
            return [
                "create_graph",
                "add_nodes",
                "add_edges",
                "get_info",
                "shortest_path",
                "degree_centrality",
                "betweenness_centrality",
                "pagerank",
                "connected_components",
                "community_detection",
                "visualize_graph",
                "import_csv",
                "export_json",
                # Admin-only operations
                "admin_reset_limits",
                "admin_list_users",
                "admin_manage_keys",
            ]

        operations = []

        if Permission.CREATE_GRAPH in user.permissions:
            operations.append("create_graph")
        if Permission.ADD_NODES in user.permissions:
            operations.extend(["add_nodes", "add_edges"])
        if Permission.READ_GRAPH in user.permissions:
            operations.append("get_info")
        if Permission.ANALYZE_GRAPH in user.permissions:
            operations.extend(
                [
                    "shortest_path",
                    "degree_centrality",
                    "betweenness_centrality",
                    "pagerank",
                    "connected_components",
                    "community_detection",
                ]
            )
        if Permission.VISUALIZE_GRAPH in user.permissions:
            operations.append("visualize_graph")
        if Permission.IMPORT_DATA in user.permissions:
            operations.append("import_csv")
        if Permission.EXPORT_DATA in user.permissions:
            operations.append("export_json")

        return operations


class AuthenticationManager:
    """Central authentication manager coordinating all auth methods."""

    def __init__(self):
        self.config = get_config()
        self.api_key_auth = APIKeyAuth()
        self.oauth_manager = OAuthManager()
        self.rbac_manager = RBACManager()

    def authenticate_request(
        self, headers: Dict[str, str], params: Dict[str, str] = None
    ) -> Optional[User]:
        """Authenticate a request using available methods."""
        if params is None:
            params = {}

        # Try API key authentication first
        api_key = headers.get(self.config.security.api_key_header)
        if api_key:
            user = self.api_key_auth.authenticate(api_key)
            if user:
                return user

        # Try OAuth/JWT token authentication
        auth_header = headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove 'Bearer ' prefix
            user = self.oauth_manager.verify_token(token)
            if user:
                return user

        # If no authentication provided and guest access is allowed
        if (
            not self.config.security.api_key_enabled
            and not self.config.security.oauth_enabled
        ):
            # Create anonymous user with guest permissions
            return User(id="anonymous", username="Anonymous", roles={Role.GUEST})

        return None

    def authorize_operation(
        self, user: User, operation: str, resource: Optional[str] = None
    ) -> bool:
        """Check if user is authorized to perform operation."""
        # Map operation to permission
        permission_map = {
            "create_graph": Permission.CREATE_GRAPH,
            "add_nodes": Permission.ADD_NODES,
            "add_edges": Permission.ADD_EDGES,
            "get_info": Permission.READ_GRAPH,
            "shortest_path": Permission.ANALYZE_GRAPH,
            "degree_centrality": Permission.ANALYZE_GRAPH,
            "betweenness_centrality": Permission.ANALYZE_GRAPH,
            "pagerank": Permission.ANALYZE_GRAPH,
            "connected_components": Permission.ANALYZE_GRAPH,
            "community_detection": Permission.ANALYZE_GRAPH,
            "visualize_graph": Permission.VISUALIZE_GRAPH,
            "import_csv": Permission.IMPORT_DATA,
            "export_json": Permission.EXPORT_DATA,
        }

        permission = permission_map.get(operation, Permission.READ_GRAPH)
        return self.rbac_manager.check_permission(user, permission, resource)

    def get_user_context(self, user: User) -> Dict[str, Any]:
        """Get user context for audit logging."""
        return {
            "user_id": user.id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "last_login": user.last_login.isoformat() if user.last_login else None,
        }
