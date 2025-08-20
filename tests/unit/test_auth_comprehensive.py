"""Comprehensive tests for auth.py module."""

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from networkx_mcp.auth import APIKeyManager, AuthMiddleware


class TestAPIKeyManager(unittest.TestCase):
    """Test APIKeyManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.temp_dir.name) / "test_keys.json"
        self.manager = APIKeyManager(storage_path=self.storage_path)

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_init_default_path(self):
        """Test APIKeyManager initialization with default path."""
        # This will use the actual home directory, so just check it doesn't crash
        manager = APIKeyManager()
        assert isinstance(manager.storage_path, Path)
        assert manager.storage_path.name == "api_keys.json"
        assert isinstance(manager.keys, dict)
        assert isinstance(manager.rate_limits, dict)

    def test_init_custom_path(self):
        """Test APIKeyManager initialization with custom path."""
        assert self.manager.storage_path == self.storage_path
        assert isinstance(self.manager.keys, dict)
        assert len(self.manager.keys) == 0

    def test_generate_key(self):
        """Test generating a new API key."""
        # Generate key with default permissions
        key = self.manager.generate_key("test_user")

        # Verify key format
        assert key.startswith("nxmcp_")
        assert len(key) > 10  # Should be reasonably long

        # Verify key is stored
        assert len(self.manager.keys) == 1

        # Verify metadata
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        key_data = self.manager.keys[key_hash]

        assert key_data["name"] == "test_user"
        assert key_data["active"] is True
        assert key_data["last_used"] is None
        assert key_data["request_count"] == 0
        assert set(key_data["permissions"]) == {"read", "write"}
        assert "created" in key_data

    def test_generate_key_custom_permissions(self):
        """Test generating key with custom permissions."""
        permissions = {"read", "admin"}
        key = self.manager.generate_key("admin_user", permissions)

        key_hash = hashlib.sha256(key.encode()).hexdigest()
        key_data = self.manager.keys[key_hash]

        assert set(key_data["permissions"]) == permissions

    def test_validate_key_valid(self):
        """Test validating a valid API key."""
        # Generate and then validate
        key = self.manager.generate_key("test_user")
        result = self.manager.validate_key(key)

        assert result is not None
        assert result["name"] == "test_user"
        assert result["active"] is True
        assert result["last_used"] is not None  # Should be updated
        assert result["request_count"] == 1  # Should be incremented

    def test_validate_key_invalid_format(self):
        """Test validating invalid key formats."""
        # Empty string
        assert self.manager.validate_key("") is None

        # Wrong prefix
        assert self.manager.validate_key("invalid_key") is None

        # None
        assert self.manager.validate_key(None) is None

    def test_validate_key_nonexistent(self):
        """Test validating a non-existent key."""
        fake_key = "nxmcp_" + "a" * 32
        result = self.manager.validate_key(fake_key)
        assert result is None

    def test_validate_key_inactive(self):
        """Test validating an inactive key."""
        # Generate and revoke key
        key = self.manager.generate_key("test_user")
        self.manager.revoke_key(key)

        # Should not validate
        result = self.manager.validate_key(key)
        assert result is None

    def test_revoke_key(self):
        """Test revoking an API key."""
        # Generate key
        key = self.manager.generate_key("test_user")

        # Revoke it
        result = self.manager.revoke_key(key)
        assert result is True

        # Key should be inactive
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        key_data = self.manager.keys[key_hash]
        assert key_data["active"] is False
        assert "revoked" in key_data

    def test_revoke_nonexistent_key(self):
        """Test revoking a non-existent key."""
        fake_key = "nxmcp_" + "a" * 32
        result = self.manager.revoke_key(fake_key)
        assert result is False

    def test_list_keys_empty(self):
        """Test listing keys when none exist."""
        keys = self.manager.list_keys()
        assert keys == []

    def test_list_keys_with_data(self):
        """Test listing keys with actual data."""
        # Generate some keys
        key1 = self.manager.generate_key("user1", {"read"})
        self.manager.generate_key(
            "user2", {"read", "write"}
        )  # Generate key2 for listing

        # Use one key to update stats
        self.manager.validate_key(key1)

        # List keys
        keys = self.manager.list_keys()
        assert len(keys) == 2

        # Check data is properly formatted and doesn't expose actual keys
        for key_info in keys:
            assert "name" in key_info
            assert "created" in key_info
            assert "active" in key_info
            assert "last_used" in key_info
            assert "request_count" in key_info
            assert "permissions" in key_info
            # Should not contain actual key
            assert not any(k.startswith("nxmcp_") for k in str(key_info))

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        key = self.manager.generate_key("test_user")

        # Should allow requests under limit
        for i in range(5):
            assert (
                self.manager.check_rate_limit(key, limit=10, window_minutes=60) is True
            )

        # Should deny when over limit
        for i in range(10):
            self.manager.check_rate_limit(key, limit=10, window_minutes=60)

        # Next request should be denied
        assert self.manager.check_rate_limit(key, limit=10, window_minutes=60) is False

    def test_rate_limiting_window_cleanup(self):
        """Test that old requests are cleaned up from rate limit window."""
        key = self.manager.generate_key("test_user")

        # Mock datetime to simulate passage of time
        with patch("networkx_mcp.auth.datetime") as mock_dt:
            # Start time
            start_time = datetime(2023, 1, 1, 12, 0, 0)
            mock_dt.now.return_value = start_time

            # Fill up the rate limit
            for i in range(10):
                self.manager.check_rate_limit(key, limit=10, window_minutes=60)

            # Should be at limit
            assert (
                self.manager.check_rate_limit(key, limit=10, window_minutes=60) is False
            )

            # Move time forward beyond window
            mock_dt.now.return_value = start_time + timedelta(minutes=61)

            # Should allow requests again
            assert (
                self.manager.check_rate_limit(key, limit=10, window_minutes=60) is True
            )

    def test_persistence_save_load(self):
        """Test that keys are properly saved and loaded."""
        # Generate key
        key = self.manager.generate_key("test_user", {"read", "admin"})

        # Create new manager with same storage path
        new_manager = APIKeyManager(storage_path=self.storage_path)

        # Should have loaded the key
        assert len(new_manager.keys) == 1

        # Should be able to validate the same key
        result = new_manager.validate_key(key)
        assert result is not None
        assert result["name"] == "test_user"

    def test_persistence_corrupted_file(self):
        """Test handling of corrupted storage file."""
        # Write invalid JSON to storage file
        with open(self.storage_path, "w") as f:
            f.write("invalid json")

        # Should handle gracefully
        manager = APIKeyManager(storage_path=self.storage_path)
        assert manager.keys == {}

    def test_persistence_invalid_data_type(self):
        """Test handling of invalid data type in storage file."""
        # Write valid JSON but wrong type
        with open(self.storage_path, "w") as f:
            json.dump(["not", "a", "dict"], f)

        # Should handle gracefully
        manager = APIKeyManager(storage_path=self.storage_path)
        assert manager.keys == {}


class TestAuthMiddleware(unittest.TestCase):
    """Test AuthMiddleware class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary manager
        self.temp_dir = tempfile.TemporaryDirectory()
        self.key_manager = APIKeyManager(
            storage_path=Path(self.temp_dir.name) / "test_keys.json"
        )
        self.middleware = AuthMiddleware(self.key_manager, required=True)

        # Generate test key
        self.test_key = self.key_manager.generate_key("test_user", {"read", "write"})

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_init_required(self):
        """Test middleware initialization with auth required."""
        middleware = AuthMiddleware(self.key_manager, required=True)
        assert middleware.key_manager == self.key_manager
        assert middleware.required is True

    def test_init_optional(self):
        """Test middleware initialization with auth optional."""
        middleware = AuthMiddleware(self.key_manager, required=False)
        assert middleware.required is False

    def test_authenticate_valid_key(self):
        """Test authentication with valid API key."""
        request = {"params": {"api_key": self.test_key}}

        result = self.middleware.authenticate(request)
        assert result is not None
        assert result["name"] == "test_user"
        assert "read" in result["permissions"]
        assert "write" in result["permissions"]

    def test_authenticate_api_key_variant(self):
        """Test authentication with apiKey parameter name."""
        request = {"params": {"apiKey": self.test_key}}

        result = self.middleware.authenticate(request)
        assert result is not None
        assert result["name"] == "test_user"

    def test_authenticate_no_key_required(self):
        """Test authentication fails when key required but not provided."""
        request = {"params": {}}

        with pytest.raises(ValueError, match="API key required"):
            self.middleware.authenticate(request)

    def test_authenticate_no_key_optional(self):
        """Test authentication succeeds when key not required and not provided."""
        middleware = AuthMiddleware(self.key_manager, required=False)
        request = {"params": {}}

        result = middleware.authenticate(request)
        assert result is not None
        assert result["name"] == "anonymous"
        assert result["permissions"] == ["read"]

    def test_authenticate_invalid_key(self):
        """Test authentication with invalid API key."""
        request = {"params": {"api_key": "nxmcp_invalid_key"}}

        with pytest.raises(ValueError, match="Invalid API key"):
            self.middleware.authenticate(request)

    def test_authenticate_rate_limited(self):
        """Test authentication fails when rate limited."""
        # Fill up rate limit
        for i in range(1001):  # Default limit is 1000
            try:
                self.key_manager.check_rate_limit(self.test_key)
            except Exception:
                break

        request = {"params": {"api_key": self.test_key}}

        with pytest.raises(ValueError, match="Rate limit exceeded"):
            self.middleware.authenticate(request)

    def test_check_permission_has_permission(self):
        """Test permission check when user has the permission."""
        auth_data = {"permissions": ["read", "write"]}

        assert self.middleware.check_permission(auth_data, "read") is True
        assert self.middleware.check_permission(auth_data, "write") is True

    def test_check_permission_no_permission(self):
        """Test permission check when user lacks the permission."""
        auth_data = {"permissions": ["read"]}

        assert self.middleware.check_permission(auth_data, "write") is False
        assert self.middleware.check_permission(auth_data, "admin") is False

    def test_check_permission_admin_override(self):
        """Test that admin permission grants all access."""
        auth_data = {"permissions": ["admin"]}

        assert self.middleware.check_permission(auth_data, "read") is True
        assert self.middleware.check_permission(auth_data, "write") is True
        assert self.middleware.check_permission(auth_data, "delete") is True

    def test_check_permission_no_permissions_key(self):
        """Test permission check when permissions key is missing."""
        auth_data = {}

        assert self.middleware.check_permission(auth_data, "read") is False


class TestAuthCLI(unittest.TestCase):
    """Test the CLI functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()

        # Patch the APIKeyManager to use our temp storage
        self.storage_patcher = patch("networkx_mcp.auth.APIKeyManager")
        mock_manager_class = self.storage_patcher.start()
        self.mock_manager = Mock()
        mock_manager_class.return_value = self.mock_manager

    def tearDown(self):
        """Clean up after tests."""
        self.storage_patcher.stop()
        self.temp_dir.cleanup()

    @patch("sys.argv", ["auth.py", "generate", "test_user"])
    @patch("builtins.print")
    def test_cli_generate_default_permissions(self, mock_print):
        """Test CLI generate command with default permissions."""
        from networkx_mcp.auth import main

        # Mock the generate_key method
        self.mock_manager.generate_key.return_value = "nxmcp_test_key_123"

        # Run CLI
        main()

        # Verify manager was called correctly
        self.mock_manager.generate_key.assert_called_once_with(
            "test_user", {"read", "write"}
        )

        # Verify output
        mock_print.assert_called()

    @patch(
        "sys.argv",
        [
            "auth.py",
            "generate",
            "admin_user",
            "--permissions",
            "read",
            "write",
            "admin",
        ],
    )
    @patch("builtins.print")
    def test_cli_generate_custom_permissions(self, mock_print):
        """Test CLI generate command with custom permissions."""
        from networkx_mcp.auth import main

        self.mock_manager.generate_key.return_value = "nxmcp_admin_key_123"

        main()

        self.mock_manager.generate_key.assert_called_once_with(
            "admin_user", {"read", "write", "admin"}
        )

    @patch("sys.argv", ["auth.py", "list"])
    @patch("builtins.print")
    def test_cli_list_no_keys(self, mock_print):
        """Test CLI list command with no keys."""
        from networkx_mcp.auth import main

        self.mock_manager.list_keys.return_value = []

        main()

        self.mock_manager.list_keys.assert_called_once()
        mock_print.assert_called_with("No API keys found")

    @patch("sys.argv", ["auth.py", "list"])
    @patch("builtins.print")
    def test_cli_list_with_keys(self, mock_print):
        """Test CLI list command with existing keys."""
        from networkx_mcp.auth import main

        self.mock_manager.list_keys.return_value = [
            {
                "name": "test_user",
                "created": "2023-01-01T12:00:00",
                "active": True,
                "request_count": 42,
            }
        ]

        main()

        self.mock_manager.list_keys.assert_called_once()
        # Should print header and key info
        assert mock_print.call_count >= 2

    @patch("sys.argv", ["auth.py", "revoke", "nxmcp_test_key"])
    @patch("builtins.print")
    def test_cli_revoke_success(self, mock_print):
        """Test CLI revoke command success."""
        from networkx_mcp.auth import main

        self.mock_manager.revoke_key.return_value = True

        main()

        self.mock_manager.revoke_key.assert_called_once_with("nxmcp_test_key")
        mock_print.assert_called_with("API key revoked successfully")

    @patch("sys.argv", ["auth.py", "revoke", "nxmcp_nonexistent"])
    @patch("builtins.print")
    def test_cli_revoke_not_found(self, mock_print):
        """Test CLI revoke command when key not found."""
        from networkx_mcp.auth import main

        self.mock_manager.revoke_key.return_value = False

        main()

        self.mock_manager.revoke_key.assert_called_once_with("nxmcp_nonexistent")
        mock_print.assert_called_with("API key not found")

    @patch("sys.argv", ["auth.py"])
    @patch("argparse.ArgumentParser.print_help")
    def test_cli_no_command(self, mock_help):
        """Test CLI with no command shows help."""
        from networkx_mcp.auth import main

        main()

        mock_help.assert_called_once()


if __name__ == "__main__":
    unittest.main()
