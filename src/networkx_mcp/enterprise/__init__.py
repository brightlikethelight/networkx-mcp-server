"""
NetworkX MCP Server - Enterprise Edition

Production-ready features for enterprise deployment:
- OAuth 2.1 authentication with PKCE
- Role-based access control (RBAC)
- Rate limiting and resource quotas
- Audit logging and monitoring
- Configuration management
"""

__version__ = "1.1.0"

# Import enterprise components
try:
    from .auth import (
        AuthenticationManager,
        APIKeyAuth,
        OAuthManager,
        RBACManager,
    )
    from .config import EnterpriseConfig
    from .monitoring import MetricsCollector, AuditLogger
    from .rate_limiting import RateLimiter
    from .server import EnterpriseNetworkXServer
    
    ENTERPRISE_AVAILABLE = True
except ImportError as e:
    ENTERPRISE_AVAILABLE = False
    _missing_deps = str(e)

def check_enterprise_deps():
    """Check if enterprise dependencies are available."""
    if not ENTERPRISE_AVAILABLE:
        raise ImportError(
            f"Enterprise features require additional dependencies: {_missing_deps}\n"
            "Install with: pip install networkx-mcp-server[enterprise]"
        )
    return True

# Export main components
__all__ = [
    "AuthenticationManager",
    "APIKeyAuth", 
    "OAuthManager",
    "RBACManager",
    "EnterpriseConfig",
    "MetricsCollector",
    "AuditLogger",
    "RateLimiter",
    "EnterpriseNetworkXServer",
    "check_enterprise_deps",
    "ENTERPRISE_AVAILABLE",
]