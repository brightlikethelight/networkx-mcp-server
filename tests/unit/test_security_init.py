"""Tests for security/__init__.py - Target: 100% coverage."""


class TestSecurityInit:
    """Test the security package imports."""

    def test_module_imports(self):
        """Test that all expected classes can be imported from security."""
        from networkx_mcp import security

        # Check that the module exists
        assert security is not None

        # Check __all__ exports
        assert hasattr(security, "__all__")
        assert isinstance(security.__all__, list)

        # Expected exports from __all__
        expected_exports = [
            "AuthService",
            "AuthMiddleware",
            "TokenValidator",
            "RateLimiter",
            "RateLimitMiddleware",
            "RequestValidator",
            "SecurityValidator",
            "SecurityMiddleware",
            "CORSMiddleware",
            "AuditLogger",
            "SecurityEventLogger",
        ]

        for export in expected_exports:
            assert export in security.__all__

    def test_direct_imports(self):
        """Test direct imports of security classes."""
        from networkx_mcp.security import (
            AuditLogger,
            AuthMiddleware,
            AuthService,
            CORSMiddleware,
            RateLimiter,
            RateLimitMiddleware,
            RequestValidator,
            SecurityEventLogger,
            SecurityMiddleware,
            SecurityValidator,
            TokenValidator,
        )

        # Verify all imports are successful
        assert AuditLogger is not None
        assert SecurityEventLogger is not None
        assert AuthMiddleware is not None
        assert AuthService is not None
        assert TokenValidator is not None
        assert CORSMiddleware is not None
        assert SecurityMiddleware is not None
        assert RateLimiter is not None
        assert RateLimitMiddleware is not None
        assert RequestValidator is not None
        assert SecurityValidator is not None

        # Verify they are all classes
        all_classes = [
            AuditLogger,
            SecurityEventLogger,
            AuthMiddleware,
            AuthService,
            TokenValidator,
            CORSMiddleware,
            SecurityMiddleware,
            RateLimiter,
            RateLimitMiddleware,
            RequestValidator,
            SecurityValidator,
        ]

        for cls in all_classes:
            assert isinstance(cls, type), f"{cls.__name__} should be a class"

    def test_module_docstring(self):
        """Test that the module has proper documentation."""
        import networkx_mcp.security as security

        # Module should have a docstring
        assert security.__doc__ is not None
        assert "Security module for NetworkX MCP Server" in security.__doc__
        assert "Authentication and authorization" in security.__doc__
        assert "rate limiting" in security.__doc__
