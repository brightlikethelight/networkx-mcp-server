"""
Tests for enterprise authentication and authorization system.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Skip all tests if enterprise dependencies not available
pytest.importorskip("authlib")
pytest.importorskip("pydantic")

from networkx_mcp.enterprise.auth import (
    APIKeyAuth, OAuthManager, RBACManager, AuthenticationManager,
    User, Role, Permission
)
from networkx_mcp.enterprise.config import EnterpriseConfig, SecurityConfig


@pytest.fixture
def mock_config():
    """Mock enterprise configuration."""
    config = EnterpriseConfig()
    config.security.api_key_enabled = True
    config.security.api_keys = ["test_key_1", "test_key_2"]
    config.security.admin_users = ["api_user_0"]
    config.security.oauth_enabled = True
    config.security.oauth_client_id = "test_client"
    config.security.oauth_client_secret = "test_secret"
    config.security.jwt_secret = "test_jwt_secret"
    return config


@pytest.fixture
def api_key_auth(mock_config):
    """Create API key authentication instance."""
    with patch('networkx_mcp.enterprise.auth.get_config', return_value=mock_config):
        return APIKeyAuth()


@pytest.fixture
def oauth_manager(mock_config):
    """Create OAuth manager instance."""
    with patch('networkx_mcp.enterprise.auth.get_config', return_value=mock_config):
        return OAuthManager()


@pytest.fixture
def rbac_manager(mock_config):
    """Create RBAC manager instance."""
    with patch('networkx_mcp.enterprise.auth.get_config', return_value=mock_config):
        return RBACManager()


@pytest.fixture
def auth_manager(mock_config):
    """Create authentication manager instance."""
    with patch('networkx_mcp.enterprise.auth.get_config', return_value=mock_config):
        return AuthenticationManager()


class TestUser:
    """Test User model and functionality."""
    
    def test_user_creation_defaults(self):
        """Test user creation with default values."""
        user = User(id="test_user")
        
        assert user.id == "test_user"
        assert user.username is None
        assert user.email is None
        assert Role.USER in user.roles
        assert isinstance(user.created_at, datetime)
        assert user.last_login is None
        
        # Check default permissions for USER role
        assert Permission.CREATE_GRAPH in user.permissions
        assert Permission.READ_GRAPH in user.permissions
        assert Permission.ANALYZE_GRAPH in user.permissions
        assert Permission.ADMIN_ACCESS not in user.permissions
    
    def test_admin_user_permissions(self):
        """Test admin user has all permissions."""
        user = User(id="admin", roles={Role.ADMIN})
        
        # Admin should have all permissions
        for permission in Permission:
            assert user.has_permission(permission)
    
    def test_readonly_user_permissions(self):
        """Test readonly user has limited permissions."""
        user = User(id="readonly", roles={Role.READONLY})
        
        assert user.has_permission(Permission.READ_GRAPH)
        assert user.has_permission(Permission.ANALYZE_GRAPH)
        assert user.has_permission(Permission.VISUALIZE_GRAPH)
        assert not user.has_permission(Permission.CREATE_GRAPH)
        assert not user.has_permission(Permission.ADD_NODES)
        assert not user.has_permission(Permission.ADMIN_ACCESS)
    
    def test_guest_user_permissions(self):
        """Test guest user has minimal permissions."""
        user = User(id="guest", roles={Role.GUEST})
        
        assert user.has_permission(Permission.READ_GRAPH)
        assert user.has_permission(Permission.VISUALIZE_GRAPH)
        assert not user.has_permission(Permission.CREATE_GRAPH)
        assert not user.has_permission(Permission.ANALYZE_GRAPH)
        assert not user.has_permission(Permission.ADMIN_ACCESS)


class TestAPIKeyAuth:
    """Test API key authentication."""
    
    def test_valid_api_key_authentication(self, api_key_auth):
        """Test authentication with valid API key."""
        user = api_key_auth.authenticate("test_key_1")
        
        assert user is not None
        assert user.id == "api_user_0"
        assert user.username == "API User 0"
        assert Role.ADMIN in user.roles  # First user is admin
        assert isinstance(user.last_login, datetime)
    
    def test_invalid_api_key_authentication(self, api_key_auth):
        """Test authentication with invalid API key."""
        user = api_key_auth.authenticate("invalid_key")
        assert user is None
    
    def test_disabled_api_key_auth(self, mock_config):
        """Test authentication when API key auth is disabled."""
        mock_config.security.api_key_enabled = False
        
        with patch('networkx_mcp.enterprise.auth.get_config', return_value=mock_config):
            api_auth = APIKeyAuth()
            user = api_auth.authenticate("test_key_1")
            assert user is None
    
    def test_generate_api_key(self, api_key_auth):
        """Test API key generation."""
        api_key = api_key_auth.generate_api_key("test_user")
        
        assert api_key is not None
        assert len(api_key) > 20  # Should be a reasonable length
        assert isinstance(api_key, str)


class TestOAuthManager:
    """Test OAuth 2.1 with PKCE functionality."""
    
    def test_pkce_challenge_generation(self, oauth_manager):
        """Test PKCE code verifier and challenge generation."""
        code_verifier, code_challenge = oauth_manager.generate_pkce_challenge()
        
        assert len(code_verifier) >= 43  # RFC 7636 requirement
        assert len(code_challenge) > 0
        assert code_verifier != code_challenge
    
    def test_authorization_url_creation(self, oauth_manager):
        """Test OAuth authorization URL creation."""
        state = "test_state"
        auth_url = oauth_manager.create_authorization_url(state)
        
        assert "response_type=code" in auth_url
        assert "client_id=test_client" in auth_url
        assert "code_challenge=" in auth_url
        assert "code_challenge_method=S256" in auth_url
        assert f"state={state}" in auth_url
    
    def test_token_verification(self, oauth_manager):
        """Test JWT token verification."""
        # Create a test token
        import jwt
        payload = {
            'sub': 'test_user',
            'iat': int(time.time()),
            'exp': int(time.time()) + 3600,
            'scope': 'graph:read graph:write'
        }
        token = jwt.encode(payload, 'test_jwt_secret', algorithm='HS256')
        
        user = oauth_manager.verify_token(token)
        
        assert user is not None
        assert user.id == 'test_user'
        assert Role.USER in user.roles
    
    def test_invalid_token_verification(self, oauth_manager):
        """Test verification of invalid token."""
        invalid_token = "invalid.jwt.token"
        user = oauth_manager.verify_token(invalid_token)
        assert user is None
    
    def test_expired_token_verification(self, oauth_manager):
        """Test verification of expired token."""
        import jwt
        payload = {
            'sub': 'test_user',
            'iat': int(time.time()) - 7200,  # 2 hours ago
            'exp': int(time.time()) - 3600,  # 1 hour ago (expired)
            'scope': 'graph:read'
        }
        token = jwt.encode(payload, 'test_jwt_secret', algorithm='HS256')
        
        user = oauth_manager.verify_token(token)
        assert user is None


class TestRBACManager:
    """Test Role-Based Access Control."""
    
    def test_admin_has_all_permissions(self, rbac_manager):
        """Test admin user has all permissions."""
        admin_user = User(id="admin", roles={Role.ADMIN})
        
        # Admin should have access to all operations
        for permission in Permission:
            assert rbac_manager.check_permission(admin_user, permission)
    
    def test_user_permission_checks(self, rbac_manager):
        """Test regular user permission checks."""
        user = User(id="user", roles={Role.USER})
        
        assert rbac_manager.check_permission(user, Permission.CREATE_GRAPH)
        assert rbac_manager.check_permission(user, Permission.READ_GRAPH)
        assert rbac_manager.check_permission(user, Permission.ANALYZE_GRAPH)
        assert not rbac_manager.check_permission(user, Permission.ADMIN_ACCESS)
    
    def test_readonly_user_permissions(self, rbac_manager):
        """Test readonly user permissions."""
        readonly_user = User(id="readonly", roles={Role.READONLY})
        
        assert rbac_manager.check_permission(readonly_user, Permission.READ_GRAPH)
        assert rbac_manager.check_permission(readonly_user, Permission.VISUALIZE_GRAPH)
        assert not rbac_manager.check_permission(readonly_user, Permission.CREATE_GRAPH)
        assert not rbac_manager.check_permission(readonly_user, Permission.ADD_NODES)
    
    def test_get_allowed_operations(self, rbac_manager):
        """Test getting allowed operations for different user types."""
        admin_user = User(id="admin", roles={Role.ADMIN})
        regular_user = User(id="user", roles={Role.USER})
        readonly_user = User(id="readonly", roles={Role.READONLY})
        
        admin_ops = rbac_manager.get_allowed_operations(admin_user)
        user_ops = rbac_manager.get_allowed_operations(regular_user)
        readonly_ops = rbac_manager.get_allowed_operations(readonly_user)
        
        # Admin should have all operations
        assert "create_graph" in admin_ops
        assert "visualize_graph" in admin_ops
        assert len(admin_ops) > len(user_ops)
        
        # Regular user should have most operations
        assert "create_graph" in user_ops
        assert "add_nodes" in user_ops
        assert "visualize_graph" in user_ops
        
        # Readonly user should have limited operations
        assert "create_graph" not in readonly_ops
        assert "add_nodes" not in readonly_ops
        assert "get_info" in readonly_ops
        assert "visualize_graph" in readonly_ops
    
    def test_disabled_rbac(self, mock_config):
        """Test behavior when RBAC is disabled."""
        mock_config.security.rbac_enabled = False
        
        with patch('networkx_mcp.enterprise.auth.get_config', return_value=mock_config):
            rbac = RBACManager()
            
            # When RBAC is disabled, all permissions should be allowed
            user = User(id="guest", roles={Role.GUEST})
            assert rbac.check_permission(user, Permission.ADMIN_ACCESS)


class TestAuthenticationManager:
    """Test the main authentication manager."""
    
    def test_api_key_authentication(self, auth_manager):
        """Test authentication via API key."""
        headers = {"X-API-Key": "test_key_1"}
        user = auth_manager.authenticate_request(headers)
        
        assert user is not None
        assert user.id == "api_user_0"
        assert Role.ADMIN in user.roles
    
    def test_oauth_token_authentication(self, auth_manager):
        """Test authentication via OAuth token."""
        import jwt
        payload = {
            'sub': 'oauth_user',
            'iat': int(time.time()),
            'exp': int(time.time()) + 3600,
            'scope': 'graph:read graph:write'
        }
        token = jwt.encode(payload, 'test_jwt_secret', algorithm='HS256')
        
        headers = {"Authorization": f"Bearer {token}"}
        user = auth_manager.authenticate_request(headers)
        
        assert user is not None
        assert user.id == "oauth_user"
        assert Role.USER in user.roles
    
    def test_anonymous_access_when_auth_disabled(self, mock_config):
        """Test anonymous access when authentication is disabled."""
        mock_config.security.api_key_enabled = False
        mock_config.security.oauth_enabled = False
        
        with patch('networkx_mcp.enterprise.auth.get_config', return_value=mock_config):
            auth_mgr = AuthenticationManager()
            
            headers = {}
            user = auth_mgr.authenticate_request(headers)
            
            assert user is not None
            assert user.id == "anonymous"
            assert Role.GUEST in user.roles
    
    def test_no_authentication_provided(self, auth_manager):
        """Test when no authentication is provided."""
        headers = {}
        user = auth_manager.authenticate_request(headers)
        assert user is None
    
    def test_operation_authorization(self, auth_manager):
        """Test operation authorization."""
        admin_user = User(id="admin", roles={Role.ADMIN})
        regular_user = User(id="user", roles={Role.USER})
        guest_user = User(id="guest", roles={Role.GUEST})
        
        # Admin can do everything
        assert auth_manager.authorize_operation(admin_user, "create_graph")
        assert auth_manager.authorize_operation(admin_user, "visualize_graph")
        
        # Regular user can do most things
        assert auth_manager.authorize_operation(regular_user, "create_graph")
        assert auth_manager.authorize_operation(regular_user, "add_nodes")
        assert auth_manager.authorize_operation(regular_user, "visualize_graph")
        
        # Guest user has limited access
        assert not auth_manager.authorize_operation(guest_user, "create_graph")
        assert auth_manager.authorize_operation(guest_user, "get_info")  # Read operations
    
    def test_get_user_context(self, auth_manager):
        """Test getting user context for audit logging."""
        user = User(
            id="test_user",
            username="Test User",
            roles={Role.USER},
            last_login=datetime.utcnow()
        )
        
        context = auth_manager.get_user_context(user)
        
        assert context["user_id"] == "test_user"
        assert context["username"] == "Test User"
        assert "user" in context["roles"]
        assert context["last_login"] is not None
        assert isinstance(context["permissions"], list)


@pytest.mark.integration
class TestAuthenticationIntegration:
    """Integration tests for authentication system."""
    
    def test_full_authentication_flow(self, mock_config):
        """Test complete authentication flow."""
        with patch('networkx_mcp.enterprise.auth.get_config', return_value=mock_config):
            auth_manager = AuthenticationManager()
            
            # Test API key authentication
            headers = {"X-API-Key": "test_key_1"}
            user = auth_manager.authenticate_request(headers)
            
            assert user is not None
            assert auth_manager.authorize_operation(user, "create_graph")
            
            context = auth_manager.get_user_context(user)
            assert context["user_id"] == "api_user_0"
    
    def test_authentication_with_invalid_credentials(self, auth_manager):
        """Test authentication with various invalid credentials."""
        # Invalid API key
        headers = {"X-API-Key": "invalid_key"}
        user = auth_manager.authenticate_request(headers)
        assert user is None
        
        # Invalid Bearer token
        headers = {"Authorization": "Bearer invalid_token"}
        user = auth_manager.authenticate_request(headers)
        assert user is None
        
        # Malformed authorization header
        headers = {"Authorization": "NotBearer token"}
        user = auth_manager.authenticate_request(headers)
        assert user is None