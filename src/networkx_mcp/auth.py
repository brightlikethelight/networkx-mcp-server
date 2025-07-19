"""
Simple API key authentication for NetworkX MCP server.
"""

import hashlib
import json
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set


class APIKeyManager:
    """Manage API keys for authentication."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize API key manager."""
        self.storage_path = (
            storage_path or Path.home() / ".networkx-mcp" / "api_keys.json"
        )
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.keys: Dict[str, Dict[str, any]] = self._load_keys()
        self.rate_limits: Dict[str, List[datetime]] = {}  # Track requests per key

    def _load_keys(self) -> Dict[str, Dict[str, any]]:
        """Load API keys from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_keys(self):
        """Save API keys to storage."""
        with open(self.storage_path, "w") as f:
            json.dump(self.keys, f, indent=2, default=str)

    def generate_key(self, name: str, permissions: Optional[Set[str]] = None) -> str:
        """Generate a new API key."""
        # Generate secure random key
        key = f"nxmcp_{secrets.token_urlsafe(32)}"

        # Hash the key for storage
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        # Store key metadata
        self.keys[key_hash] = {
            "name": name,
            "created": datetime.now().isoformat(),
            "permissions": list(permissions or {"read", "write"}),
            "active": True,
            "last_used": None,
            "request_count": 0,
        }

        self._save_keys()
        return key

    def validate_key(self, api_key: str) -> Optional[Dict[str, any]]:
        """Validate an API key and return its metadata."""
        if not api_key or not api_key.startswith("nxmcp_"):
            return None

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_data = self.keys.get(key_hash)

        if key_data and key_data.get("active", True):
            # Update usage stats
            key_data["last_used"] = datetime.now().isoformat()
            key_data["request_count"] = key_data.get("request_count", 0) + 1
            self._save_keys()
            return key_data

        return None

    def check_rate_limit(
        self, api_key: str, limit: int = 1000, window_minutes: int = 60
    ) -> bool:
        """Check if API key has exceeded rate limit."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)

        # Get or create request history
        if key_hash not in self.rate_limits:
            self.rate_limits[key_hash] = []

        # Remove old requests outside window
        self.rate_limits[key_hash] = [
            req_time
            for req_time in self.rate_limits[key_hash]
            if req_time > window_start
        ]

        # Check if under limit
        if len(self.rate_limits[key_hash]) >= limit:
            return False

        # Add current request
        self.rate_limits[key_hash].append(now)
        return True

    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if key_hash in self.keys:
            self.keys[key_hash]["active"] = False
            self.keys[key_hash]["revoked"] = datetime.now().isoformat()
            self._save_keys()
            return True
        return False

    def list_keys(self) -> List[Dict[str, any]]:
        """List all API keys (without exposing the actual keys)."""
        return [
            {
                "name": data["name"],
                "created": data["created"],
                "active": data.get("active", True),
                "last_used": data.get("last_used"),
                "request_count": data.get("request_count", 0),
                "permissions": data.get("permissions", []),
            }
            for data in self.keys.values()
        ]


class AuthMiddleware:
    """Authentication middleware for MCP server."""

    def __init__(self, key_manager: APIKeyManager, required: bool = True):
        """Initialize auth middleware."""
        self.key_manager = key_manager
        self.required = required

    def authenticate(self, request: dict) -> Optional[Dict[str, any]]:
        """Authenticate a request."""
        # Extract API key from request
        # Could be in params, headers, or a special auth field
        params = request.get("params", {})
        api_key = params.get("api_key") or params.get("apiKey")

        if not api_key and not self.required:
            # Allow unauthenticated requests if not required
            return {"name": "anonymous", "permissions": ["read"]}

        if not api_key:
            raise ValueError("API key required")

        # Validate key
        key_data = self.key_manager.validate_key(api_key)
        if not key_data:
            raise ValueError("Invalid API key")

        # Check rate limit
        if not self.key_manager.check_rate_limit(api_key):
            raise ValueError("Rate limit exceeded")

        return key_data

    def check_permission(self, auth_data: Dict[str, any], permission: str) -> bool:
        """Check if authenticated user has permission."""
        permissions = auth_data.get("permissions", [])
        return permission in permissions or "admin" in permissions


# CLI for managing API keys
def main():
    """CLI for managing API keys."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage NetworkX MCP API keys")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate key command
    gen_parser = subparsers.add_parser("generate", help="Generate a new API key")
    gen_parser.add_argument("name", help="Name for the API key")
    gen_parser.add_argument(
        "--permissions",
        nargs="+",
        default=["read", "write"],
        help="Permissions for the key",
    )

    # List keys command
    list_parser = subparsers.add_parser("list", help="List all API keys")

    # Revoke key command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke an API key")
    revoke_parser.add_argument("key", help="API key to revoke")

    args = parser.parse_args()

    manager = APIKeyManager()

    if args.command == "generate":
        key = manager.generate_key(args.name, set(args.permissions))
        print(f"Generated API key for '{args.name}':")
        print(f"\n{key}\n")
        print("⚠️  Save this key securely - it cannot be retrieved later!")

    elif args.command == "list":
        keys = manager.list_keys()
        if not keys:
            print("No API keys found")
        else:
            print(f"{'Name':<20} {'Created':<20} {'Active':<10} {'Requests':<10}")
            print("-" * 60)
            for key in keys:
                created = key["created"][:19] if key["created"] else "N/A"
                print(
                    f"{key['name']:<20} {created:<20} {str(key['active']):<10} {key['request_count']:<10}"
                )

    elif args.command == "revoke":
        if manager.revoke_key(args.key):
            print("API key revoked successfully")
        else:
            print("API key not found")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
