"""Tests for APIKeyManager and AuthMiddleware."""

import hashlib
import hmac
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from networkx_mcp.auth import APIKeyManager, AuthMiddleware


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(tmp_path: Path) -> APIKeyManager:
    """Fresh APIKeyManager backed by a temp directory."""
    return APIKeyManager(storage_path=tmp_path / "keys.json")


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------


class TestKeyGeneration:
    def test_key_starts_with_prefix(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("test")
        assert key.startswith("nxmcp_")

    def test_key_has_reasonable_length(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("test")
        # "nxmcp_" (6 chars) + base64url of 32 random bytes (~43 chars)
        assert len(key) >= 40

    def test_generated_keys_are_unique(self, manager: APIKeyManager) -> None:
        keys = {manager.generate_key(f"k{i}") for i in range(20)}
        assert len(keys) == 20

    def test_default_permissions(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("test")
        meta = manager.validate_key(key)
        assert set(meta["permissions"]) == {"read", "write"}

    def test_custom_permissions(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("test", permissions={"read", "admin"})
        meta = manager.validate_key(key)
        assert set(meta["permissions"]) == {"read", "admin"}

    def test_metadata_fields_on_creation(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("my-key")
        meta = manager.validate_key(key)
        assert meta["name"] == "my-key"
        assert meta["active"] is True
        assert "created" in meta
        # request_count is 1 because validate_key itself increments it
        assert meta["request_count"] == 1


# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------


class TestKeyValidation:
    def test_valid_key_returns_metadata(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("test")
        result = manager.validate_key(key)
        assert result is not None
        assert result["name"] == "test"

    def test_invalid_key_returns_none(self, manager: APIKeyManager) -> None:
        manager.generate_key("test")
        assert manager.validate_key("nxmcp_bogus") is None

    def test_empty_string_returns_none(self, manager: APIKeyManager) -> None:
        assert manager.validate_key("") is None

    def test_wrong_prefix_returns_none(self, manager: APIKeyManager) -> None:
        assert manager.validate_key("wrongprefix_abc123") is None

    def test_revoked_key_returns_none(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("test")
        manager.revoke_key(key)
        assert manager.validate_key(key) is None

    def test_validate_increments_request_count(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("test")
        manager.validate_key(key)
        manager.validate_key(key)
        meta = manager.validate_key(key)
        assert meta["request_count"] == 3

    def test_validate_updates_last_used(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("test")
        meta = manager.validate_key(key)
        assert meta["last_used"] is not None


# ---------------------------------------------------------------------------
# Constant-time comparison
# ---------------------------------------------------------------------------


class TestConstantTimeComparison:
    """Verify that validate_key iterates all stored keys via hmac.compare_digest,
    rather than short-circuiting on the first match."""

    def test_compare_digest_called_for_every_stored_key(
        self, manager: APIKeyManager
    ) -> None:
        keys = [manager.generate_key(f"k{i}") for i in range(5)]
        target = keys[0]  # match will happen on first iteration

        with patch(
            "networkx_mcp.auth.hmac.compare_digest", wraps=hmac.compare_digest
        ) as spy:
            manager.validate_key(target)
            # Must be called once per stored key, not just until match
            assert spy.call_count == len(keys)

    def test_compare_digest_called_for_invalid_key(
        self, manager: APIKeyManager
    ) -> None:
        for i in range(3):
            manager.generate_key(f"k{i}")

        with patch(
            "networkx_mcp.auth.hmac.compare_digest", wraps=hmac.compare_digest
        ) as spy:
            manager.validate_key("nxmcp_doesnotexist")
            assert spy.call_count == 3


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_under_limit_succeeds(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("test")
        for _ in range(5):
            assert manager.check_rate_limit(key, limit=10, window_minutes=60) is True

    def test_exceeding_limit_fails(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("test")
        limit = 3
        for _ in range(limit):
            assert manager.check_rate_limit(key, limit=limit, window_minutes=60) is True
        assert manager.check_rate_limit(key, limit=limit, window_minutes=60) is False

    def test_old_requests_expire(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("test")
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        # Inject requests from 2 hours ago
        old_time = datetime.now() - timedelta(hours=2)
        manager.rate_limits[key_hash] = [old_time] * 10

        # Should succeed because those requests are outside the 60-min window
        assert manager.check_rate_limit(key, limit=5, window_minutes=60) is True

    def test_rate_limits_are_per_key(self, manager: APIKeyManager) -> None:
        key_a = manager.generate_key("a")
        key_b = manager.generate_key("b")
        limit = 2

        for _ in range(limit):
            manager.check_rate_limit(key_a, limit=limit)

        # key_a is exhausted, key_b should still work
        assert manager.check_rate_limit(key_a, limit=limit) is False
        assert manager.check_rate_limit(key_b, limit=limit) is True


# ---------------------------------------------------------------------------
# Key revocation
# ---------------------------------------------------------------------------


class TestKeyRevocation:
    def test_revoke_existing_key(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("test")
        assert manager.revoke_key(key) is True

    def test_revoke_nonexistent_key(self, manager: APIKeyManager) -> None:
        assert manager.revoke_key("nxmcp_nonexistent") is False

    def test_revoked_key_marked_inactive(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("test")
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        manager.revoke_key(key)
        assert manager.keys[key_hash]["active"] is False
        assert "revoked" in manager.keys[key_hash]

    def test_revoke_is_persisted(self, tmp_path: Path) -> None:
        storage = tmp_path / "keys.json"
        mgr = APIKeyManager(storage_path=storage)
        key = mgr.generate_key("test")

        mgr.revoke_key(key)

        # Reload from disk
        mgr2 = APIKeyManager(storage_path=storage)
        assert mgr2.validate_key(key) is None


# ---------------------------------------------------------------------------
# Key listing
# ---------------------------------------------------------------------------


class TestKeyListing:
    def test_list_returns_all_keys(self, manager: APIKeyManager) -> None:
        manager.generate_key("alpha")
        manager.generate_key("beta")
        listed = manager.list_keys()
        assert len(listed) == 2

    def test_list_does_not_expose_hashes(self, manager: APIKeyManager) -> None:
        key = manager.generate_key("test")
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        listed = manager.list_keys()
        serialized = json.dumps(listed)
        assert key_hash not in serialized
        # Also ensure the raw key isn't present
        assert key not in serialized

    def test_list_contains_expected_fields(self, manager: APIKeyManager) -> None:
        manager.generate_key("test")
        entry = manager.list_keys()[0]
        expected_fields = {
            "name",
            "created",
            "active",
            "last_used",
            "request_count",
            "permissions",
        }
        assert set(entry.keys()) == expected_fields

    def test_list_empty_when_no_keys(self, manager: APIKeyManager) -> None:
        assert manager.list_keys() == []


# ---------------------------------------------------------------------------
# Storage persistence
# ---------------------------------------------------------------------------


class TestStoragePersistence:
    def test_keys_survive_reload(self, tmp_path: Path) -> None:
        storage = tmp_path / "keys.json"
        mgr1 = APIKeyManager(storage_path=storage)
        key = mgr1.generate_key("persistent")

        mgr2 = APIKeyManager(storage_path=storage)
        assert mgr2.validate_key(key) is not None
        assert mgr2.validate_key(key)["name"] == "persistent"

    def test_storage_file_created_on_generate(self, tmp_path: Path) -> None:
        storage = tmp_path / "keys.json"
        mgr = APIKeyManager(storage_path=storage)
        assert not storage.exists()
        mgr.generate_key("test")
        assert storage.exists()

    def test_corrupt_storage_returns_empty(self, tmp_path: Path) -> None:
        storage = tmp_path / "keys.json"
        storage.write_text("NOT VALID JSON {{{")
        mgr = APIKeyManager(storage_path=storage)
        assert mgr.keys == {}

    def test_non_dict_storage_returns_empty(self, tmp_path: Path) -> None:
        storage = tmp_path / "keys.json"
        storage.write_text('["a list, not a dict"]')
        mgr = APIKeyManager(storage_path=storage)
        assert mgr.keys == {}

    def test_nested_directory_creation(self, tmp_path: Path) -> None:
        storage = tmp_path / "a" / "b" / "c" / "keys.json"
        mgr = APIKeyManager(storage_path=storage)
        mgr.generate_key("deep")
        assert storage.exists()


# ---------------------------------------------------------------------------
# AuthMiddleware
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    @pytest.fixture
    def middleware(self, manager: APIKeyManager) -> AuthMiddleware:
        return AuthMiddleware(key_manager=manager, required=True)

    @pytest.fixture
    def optional_middleware(self, manager: APIKeyManager) -> AuthMiddleware:
        return AuthMiddleware(key_manager=manager, required=False)

    # --- authenticate ---

    def test_valid_key_authenticates(
        self, manager: APIKeyManager, middleware: AuthMiddleware
    ) -> None:
        key = manager.generate_key("test")
        result = middleware.authenticate({"params": {"api_key": key}})
        assert result is not None
        assert result["name"] == "test"

    def test_camel_case_key_field(
        self, manager: APIKeyManager, middleware: AuthMiddleware
    ) -> None:
        key = manager.generate_key("test")
        result = middleware.authenticate({"params": {"apiKey": key}})
        assert result is not None

    def test_missing_key_raises_when_required(self, middleware: AuthMiddleware) -> None:
        with pytest.raises(ValueError, match="API key required"):
            middleware.authenticate({"params": {}})

    def test_missing_key_allowed_when_not_required(
        self, optional_middleware: AuthMiddleware
    ) -> None:
        result = optional_middleware.authenticate({"params": {}})
        assert result is not None
        assert result["name"] == "anonymous"
        assert result["permissions"] == ["read"]

    def test_invalid_key_raises(self, middleware: AuthMiddleware) -> None:
        with pytest.raises(ValueError, match="Invalid API key"):
            middleware.authenticate({"params": {"api_key": "nxmcp_fake"}})

    def test_rate_limited_key_raises(
        self, manager: APIKeyManager, middleware: AuthMiddleware
    ) -> None:
        key = manager.generate_key("test")
        # Exhaust the rate limit
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        manager.rate_limits[key_hash] = [datetime.now()] * 1000

        with pytest.raises(ValueError, match="Rate limit exceeded"):
            middleware.authenticate({"params": {"api_key": key}})

    # --- check_permission ---

    def test_has_permission(self, middleware: AuthMiddleware) -> None:
        auth_data = {"permissions": ["read", "write"]}
        assert middleware.check_permission(auth_data, "read") is True
        assert middleware.check_permission(auth_data, "write") is True

    def test_lacks_permission(self, middleware: AuthMiddleware) -> None:
        auth_data = {"permissions": ["read"]}
        assert middleware.check_permission(auth_data, "write") is False

    def test_admin_grants_all(self, middleware: AuthMiddleware) -> None:
        auth_data = {"permissions": ["admin"]}
        assert middleware.check_permission(auth_data, "read") is True
        assert middleware.check_permission(auth_data, "write") is True
        assert middleware.check_permission(auth_data, "anything") is True

    def test_empty_permissions(self, middleware: AuthMiddleware) -> None:
        assert middleware.check_permission({}, "read") is False
        assert middleware.check_permission({"permissions": []}, "read") is False


# ---------------------------------------------------------------------------
# CLI entry point (main)
# ---------------------------------------------------------------------------


class TestAuthCLI:
    """Cover auth.py main() — lines 180-239."""

    @pytest.fixture
    def _patch_manager(self, tmp_path, monkeypatch):
        """Redirect APIKeyManager() in main() to use tmp_path storage."""
        storage = tmp_path / "keys.json"
        original_init = APIKeyManager.__init__

        def patched_init(self, storage_path=None):
            original_init(self, storage_path=storage)

        monkeypatch.setattr(APIKeyManager, "__init__", patched_init)
        return storage

    def test_cli_generate(self, _patch_manager, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv", ["auth", "generate", "test-key", "--permissions", "read"]
        )
        from networkx_mcp.auth import main

        main()
        captured = capsys.readouterr()
        assert "nxmcp_" in captured.out
        assert "test-key" in captured.out

    def test_cli_list_empty(self, _patch_manager, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["auth", "list"])
        from networkx_mcp.auth import main

        main()
        captured = capsys.readouterr()
        assert "No API keys found" in captured.out

    def test_cli_list_with_keys(self, _patch_manager, monkeypatch, capsys):
        # Pre-populate keys via the patched path
        mgr = APIKeyManager()
        mgr.generate_key("alpha")
        mgr.generate_key("beta")

        monkeypatch.setattr("sys.argv", ["auth", "list"])
        from networkx_mcp.auth import main

        main()
        captured = capsys.readouterr()
        assert "alpha" in captured.out
        assert "beta" in captured.out

    def test_cli_revoke(self, _patch_manager, monkeypatch, capsys):
        mgr = APIKeyManager()
        key = mgr.generate_key("revoke-me")

        monkeypatch.setattr("sys.argv", ["auth", "revoke", key])
        from networkx_mcp.auth import main

        main()
        captured = capsys.readouterr()
        assert "revoked successfully" in captured.out

    def test_cli_revoke_not_found(self, _patch_manager, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["auth", "revoke", "nxmcp_bogus"])
        from networkx_mcp.auth import main

        main()
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_cli_no_args(self, _patch_manager, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["auth"])
        from networkx_mcp.auth import main

        main()
        captured = capsys.readouterr()
        # Should print help/usage
        assert "usage" in captured.out.lower() or "Manage" in captured.out
