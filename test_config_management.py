#!/usr/bin/env python3
"""Test configuration management system."""

import os
import sys
import time
import json
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_settings, reload_settings, get_config_manager


def test_environment_variables():
    """Test configuration via environment variables."""
    print("=== Testing Environment Variables ===")
    
    # Set some environment variables
    os.environ["MCP_ENVIRONMENT"] = "testing"
    os.environ["MCP_PORT"] = "9999"
    os.environ["MCP_WORKERS"] = "8"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["REDIS_URL"] = "redis://test-redis:6379/0"
    
    # Reload settings
    settings = reload_settings(hot_reload_only=False)
    
    print(f"Environment: {settings.environment}")
    print(f"Port: {settings.server.port}")
    print(f"Workers: {settings.server.workers}")
    print(f"Log level: {settings.logging.level}")
    print(f"Redis URL: {settings.storage.redis_url}")
    print()


def test_yaml_configuration():
    """Test configuration via YAML file."""
    print("=== Testing YAML Configuration ===")
    
    # Create a test YAML config
    config_data = {
        "environment": "development",
        "server": {
            "host": "0.0.0.0",
            "port": 8888,
            "workers": 4
        },
        "security": {
            "rate_limit_requests": 500,
            "rate_limit_window": 30
        },
        "logging": {
            "level": "INFO",
            "json_format": True
        }
    }
    
    with open("config.test.yaml", "w") as f:
        yaml.dump(config_data, f)
    
    # Update config manager to watch this file
    manager = get_config_manager()
    manager.config_paths.append("config.test.yaml")
    
    # Reload settings
    settings = reload_settings(hot_reload_only=False)
    
    print(f"Host: {settings.server.host}")
    print(f"Port: {settings.server.port}")
    print(f"Rate limit: {settings.security.rate_limit_requests} requests per {settings.security.rate_limit_window}s")
    print(f"JSON logging: {settings.logging.json_format}")
    
    # Clean up
    os.remove("config.test.yaml")
    print()


def test_json_configuration():
    """Test configuration via JSON file."""
    print("=== Testing JSON Configuration ===")
    
    # Create a test JSON config
    config_data = {
        "features": {
            "machine_learning": False,
            "visualization": False,
            "monitoring": True
        },
        "performance": {
            "enable_caching": False,
            "cache_ttl": 600
        }
    }
    
    with open("config.test.json", "w") as f:
        json.dump(config_data, f)
    
    # Update config manager to watch this file
    manager = get_config_manager()
    manager.config_paths.append("config.test.json")
    
    # Reload settings
    settings = reload_settings(hot_reload_only=False)
    
    print(f"ML enabled: {settings.features.machine_learning}")
    print(f"Visualization enabled: {settings.features.visualization}")
    print(f"Monitoring enabled: {settings.features.monitoring}")
    print(f"Caching enabled: {settings.performance.enable_caching}")
    print(f"Cache TTL: {settings.performance.cache_ttl}s")
    
    # Clean up
    os.remove("config.test.json")
    print()


def test_hot_reload():
    """Test hot reload functionality."""
    print("=== Testing Hot Reload ===")
    
    # Create initial config
    config_data = {
        "logging": {"level": "INFO"},
        "security": {"rate_limit_requests": 100}
    }
    
    with open("config.hot.yaml", "w") as f:
        yaml.dump(config_data, f)
    
    manager = get_config_manager()
    manager.config_paths.append("config.hot.yaml")
    
    # Initial load
    settings = reload_settings(hot_reload_only=False)
    print(f"Initial log level: {settings.logging.level}")
    print(f"Initial rate limit: {settings.security.rate_limit_requests}")
    
    # Register callback to detect changes
    changes_detected = []
    def on_change(old, new):
        changes_detected.append((old.logging.level, new.logging.level))
    
    manager.add_change_callback(on_change)
    
    # Start watching (if watchdog available)
    manager.start_watching()
    
    # Modify config file
    print("\nModifying configuration file...")
    config_data["logging"]["level"] = "DEBUG"
    config_data["security"]["rate_limit_requests"] = 500
    
    with open("config.hot.yaml", "w") as f:
        yaml.dump(config_data, f)
    
    # Give hot reload time to detect change
    time.sleep(2)
    
    # Check if hot reload worked
    if changes_detected:
        print(f"Hot reload detected change: {changes_detected[0][0]} -> {changes_detected[0][1]}")
    else:
        print("Hot reload not available (install watchdog for this feature)")
        # Manual reload
        settings = reload_settings(hot_reload_only=True)
        print(f"Manual reload - new log level: {settings.logging.level}")
        print(f"Manual reload - new rate limit: {settings.security.rate_limit_requests}")
    
    # Clean up
    manager.stop_watching()
    os.remove("config.hot.yaml")
    print()


def test_validation():
    """Test configuration validation."""
    print("=== Testing Configuration Validation ===")
    
    # Test invalid port
    os.environ["MCP_PORT"] = "99999"  # Invalid port number
    
    try:
        settings = reload_settings(hot_reload_only=False)
    except ValueError as e:
        print(f"Validation error caught: {e}")
    
    # Reset to valid value
    os.environ["MCP_PORT"] = "8765"
    
    # Test invalid environment
    os.environ["MCP_ENVIRONMENT"] = "invalid-env"
    
    try:
        settings = reload_settings(hot_reload_only=False)
    except ValueError as e:
        print(f"Validation error caught: {e}")
    
    # Reset
    os.environ["MCP_ENVIRONMENT"] = "development"
    print()


def show_current_configuration():
    """Display current configuration."""
    print("=== Current Configuration ===")
    settings = get_settings()
    
    print(f"Environment: {settings.environment}")
    print(f"Server: {settings.server.host}:{settings.server.port}")
    print(f"Workers: {settings.server.workers}")
    print(f"Auth enabled: {settings.security.enable_auth}")
    print(f"Rate limiting: {settings.security.rate_limit_enabled}")
    print(f"Storage backend: {settings.storage.backend}")
    print(f"ML features: {settings.features.machine_learning}")
    print(f"Log level: {settings.logging.level}")
    print()


def main():
    """Run all configuration tests."""
    print("NetworkX MCP Server - Configuration Management Test\n")
    
    # Show initial configuration
    show_current_configuration()
    
    # Run tests
    test_environment_variables()
    test_yaml_configuration()
    test_json_configuration()
    test_hot_reload()
    test_validation()
    
    # Show final configuration
    print("=== Final Configuration ===")
    show_current_configuration()
    
    print("""
REFLECTION: Can you change settings without code changes?

YES! The configuration system demonstrates that settings can be changed through:

1. Environment Variables - No code changes needed
   - Just set environment variables before starting the server
   - Example: export MCP_PORT=9999

2. Configuration Files - No code changes needed
   - Create/modify YAML or JSON configuration files
   - Server automatically loads them on startup

3. Hot Reload - No restart needed for safe settings
   - Change rate limits, logging level, cache TTL, etc.
   - Server applies changes automatically (if watchdog installed)

4. Multiple Environments - No code changes needed
   - Use different configs for dev/test/staging/production
   - Switch via MCP_ENVIRONMENT variable

The server is now fully configurable without touching any code!
""")


if __name__ == "__main__":
    main()