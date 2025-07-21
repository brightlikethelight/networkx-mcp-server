"""Feature flags - honest assessment of what actually works."""

from typing import Any

# Feature flags based on actual testing and verification
FEATURE_FLAGS = {
    # ✅ WORKING - Tested and verified
    "graph_operations": True,  # 7 core tools work with MCP protocol
    "stdio_transport": True,  # Works with Claude Desktop
    "networkx_integration": True,  # Core NetworkX functionality works
    "basic_algorithms": True,  # Shortest path, centrality measures work
    # ❌ NOT IMPLEMENTED - Planned for future versions
    "http_transport": False,  # No HTTP server implemented
    "persistent_storage": False,  # In-memory only, data lost on restart
    "multi_user": False,  # Single-user, no concurrency
    "authentication": False,  # No auth system implemented
    "authorization": False,  # No access controls
    "rate_limiting": False,  # No rate limiting
    "monitoring": False,  # No metrics or health checks
    "logging": False,  # Basic logging only, no structured logs
    "caching": False,  # No caching layer
    "clustering": False,  # No distributed operation
    "ml_algorithms": False,  # No ML features implemented
    "visualization": False,  # No visualization tools
    "import_export": False,  # No file import/export
    "backup_restore": False,  # No backup functionality
    "configuration": False,  # No runtime configuration
    "webhooks": False,  # No webhook support
    "batch_operations": False,  # No batch processing
    "async_operations": False,  # All operations are synchronous
    "data_validation": False,  # Minimal input validation
    "error_recovery": False,  # Basic error handling only
}


def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled.

    Args:
        feature_name: Name of the feature to check

    Returns:
        True if feature is enabled and actually works, False otherwise
    """
    return FEATURE_FLAGS.get(feature_name, False)


def get_enabled_features() -> list[str]:
    """Get list[Any] of all enabled (working) features."""
    return [name for name, enabled in FEATURE_FLAGS.items() if enabled]


def get_disabled_features() -> list[str]:
    """Get list[Any] of all disabled (not implemented) features."""
    return [name for name, enabled in FEATURE_FLAGS.items() if not enabled]


def get_feature_summary() -> dict[str, Any]:
    """Get summary of feature status."""
    enabled = get_enabled_features()
    disabled = get_disabled_features()

    return {
        "total_features": len(FEATURE_FLAGS),
        "enabled_count": len(enabled),
        "disabled_count": len(disabled),
        "enabled_percentage": round(len(enabled) / len(FEATURE_FLAGS) * 100, 1),
        "enabled_features": enabled,
        "disabled_features": disabled,
        "status": "Alpha - Core features working, most advanced features not implemented",
    }


# Version-specific feature roadmap
ROADMAP = {
    "0.1.0": {
        "target_features": [
            "graph_operations",
            "stdio_transport",
            "networkx_integration",
            "basic_algorithms",
        ],
        "status": "✅ Complete",
    },
    "0.2.0": {
        "target_features": ["http_transport", "authentication", "multi_user"],
        "status": "❌ Planned",
    },
    "0.3.0": {
        "target_features": ["persistent_storage", "import_export", "backup_restore"],
        "status": "❌ Planned",
    },
    "1.0.0": {
        "target_features": [
            "monitoring",
            "rate_limiting",
            "data_validation",
            "error_recovery",
        ],
        "status": "❌ Future",
    },
}

if __name__ == "__main__":
    # Print current feature status
    summary = get_feature_summary()
    print("🚀 NetworkX MCP Server - Feature Status")
    print("=" * 50)
    print(f"Status: {summary['status']}")
    print(
        f"Features: {summary['enabled_count']}/{summary['total_features']} ({summary['enabled_percentage']}%)"
    )
    print()
    print("✅ Working Features:")
    for feature in summary["enabled_features"]:
        print(f"  - {feature}")
    print()
    print("❌ Not Implemented:")
    for feature in summary["disabled_features"]:
        print(f"  - {feature}")
    print()
    print("📋 Roadmap:")
    for version, info in ROADMAP.items():
        print(
            f"  {version}: {info['status']} - {len(info['target_features'])} features"
        )
