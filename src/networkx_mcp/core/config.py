"""Configuration management for NetworkX MCP Server.

This module provides a comprehensive configuration system with support for
environment variables, YAML/JSON files, and runtime configuration updates.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigFormat(Enum):
    """Supported configuration file formats."""

    JSON = "json"
    YAML = "yaml"
    YML = "yml"
    ENV = "env"


class Environment(Enum):
    """Application environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str = "localhost"
    port: int = 8765
    workers: int = 4
    max_connections: int = 1000
    request_timeout: int = 30
    keepalive_timeout: int = 5
    debug: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "text"  # text, json
    file: str | None = None
    max_size: str = "100MB"
    backup_count: int = 5
    rotate: bool = True


@dataclass
class PerformanceConfig:
    """Performance configuration."""

    max_nodes: int = 1000000
    max_edges: int = 10000000
    memory_limit_mb: int = 4096
    timeout_seconds: int = 300
    enable_caching: bool = True
    cache_size_mb: int = 512
    cache_ttl: int = 3600
    parallel_processing: bool = True
    use_cython: bool = True
    numpy_optimization: bool = True


@dataclass
class RedisConfig:
    """Redis configuration."""

    url: str | None = None
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    prefix: str = "networkx_mcp"
    pool_size: int = 10
    timeout: int = 5
    retry_attempts: int = 3
    ttl: int = 3600


@dataclass
class SecurityConfig:
    """Security configuration."""

    enable_auth: bool = True
    api_key_required: bool = True
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 1000
    rate_limit_window: int = 60
    audit_enabled: bool = False
    audit_log_file: str = "audit.log"
    max_request_size: int = 10485760  # 10MB


@dataclass
class FeaturesConfig:
    """Feature flags configuration."""

    machine_learning: bool = True
    visualization: bool = True
    gpu_acceleration: bool = False
    enterprise_features: bool = False
    monitoring: bool = True
    metrics_endpoint: str = "/metrics"
    health_endpoint: str = "/health"


@dataclass
class QualityConfig:
    """Quality assurance configuration."""

    coverage_threshold: float = 95.0
    complexity_threshold: int = 10
    duplication_threshold: float = 5.0
    security_threshold: float = 9.0
    type_coverage_threshold: float = 90.0


@dataclass
class AppConfig:
    """Main application configuration."""

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    version: str = "2.0.0"
    name: str = "NetworkX MCP Server"

    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    # Additional custom configuration
    custom: dict[str, Any] = field(default_factory=dict[str, Any])


class ConfigLoader(ABC):
    """Abstract base class for configuration loaders."""

    @abstractmethod
    def load(self, source: str) -> dict[str, Any]:
        """Load configuration from source."""

    @abstractmethod
    def supports(self, source: str) -> bool:
        """Check if this loader supports the source."""


class FileConfigLoader(ConfigLoader):
    """Load configuration from files."""

    def supports(self, source: str) -> bool:
        """Check if source is a file path."""
        return Path(source).exists()

    def load(self, source: str) -> dict[str, Any]:
        """Load configuration from file."""
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {source}")

        suffix = path.suffix.lower()
        content = path.read_text()

        if suffix == ".json":
            return json.loads(content)
        elif suffix in [".yaml", ".yml"]:
            return yaml.safe_load(content)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")


class EnvironmentConfigLoader(ConfigLoader):
    """Load configuration from environment variables."""

    def __init__(self, prefix: str = "MCP_") -> None:
        self.prefix = prefix

    def supports(self, source: str) -> bool:
        """Always supports environment loading."""
        return True

    def load(self, source: str = "") -> dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}

        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Remove prefix and convert to nested dict[str, Any]
                config_key = key[len(self.prefix) :].lower()
                self._set_nested_value(config, config_key, self._convert_value(value))

        return config

    def _set_nested_value(self, config: dict[str, Any], key: str, value: Any) -> None:
        """Set nested value in config dict[str, Any]."""
        parts = key.split("_")
        current = config

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Boolean values
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Integer values
        try:
            return int(value)
        except ValueError:
            pass

        # Float values
        try:
            return float(value)
        except ValueError:
            pass

        # JSON values
        if value.startswith("{") or value.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # String value
        return value


class ConfigManager:
    """Manages application configuration with multiple sources."""

    def __init__(self) -> None:
        self._config: AppConfig | None = None
        self._loaders: list[ConfigLoader] = [
            FileConfigLoader(),
            EnvironmentConfigLoader(),
        ]
        self._sources: list[str] = []

    def add_loader(self, loader: ConfigLoader) -> None:
        """Add a configuration loader."""
        self._loaders.append(loader)

    def load_from_file(self, file_path: str) -> "ConfigManager":
        """Load configuration from file."""
        self._sources.append(file_path)
        return self

    def load_from_env(self, prefix: str = "MCP_") -> "ConfigManager":
        """Load configuration from environment variables."""
        env_loader = EnvironmentConfigLoader(prefix)
        if env_loader not in self._loaders:
            self._loaders.append(env_loader)
        return self

    def load_from_dict(self, config_dict: dict[str, Any]) -> "ConfigManager":
        """Load configuration from dictionary."""
        self._config = self._dict_to_config(config_dict)
        return self

    def build(self) -> AppConfig:
        """Build final configuration from all sources."""
        merged_config = {}

        # Load from files first
        for source in self._sources:
            for loader in self._loaders:
                if loader.supports(source):
                    try:
                        config_data = loader.load(source)
                        merged_config = self._deep_merge(merged_config, config_data)
                        logger.debug(f"Loaded configuration from {source}")
                    except Exception as e:
                        logger.warning(f"Failed to load config from {source}: {e}")
                    break

        # Load from environment (highest priority)
        env_loader = EnvironmentConfigLoader()
        try:
            env_config = env_loader.load()
            merged_config = self._deep_merge(merged_config, env_config)
            logger.debug("Loaded configuration from environment")
        except Exception as e:
            logger.warning(f"Failed to load environment config: {e}")

        self._config = self._dict_to_config(merged_config)
        self._validate_config()

        logger.info(
            f"Configuration loaded for environment: {self._config.environment.value}"
        )
        return self._config

    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not built. Call build() first.")
        return self._config

    def reload(self) -> AppConfig:
        """Reload configuration from all sources."""
        logger.info("Reloading configuration")
        return self.build()

    def _dict_to_config(self, config_dict: dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig instance."""
        # Handle environment conversion
        if "environment" in config_dict:
            env_value = config_dict["environment"]
            if isinstance(env_value, str):
                config_dict["environment"] = Environment(env_value.lower())

        # Create nested configurations
        for field_info in fields(AppConfig):
            if field_info.name in config_dict and hasattr(
                field_info.type, "__dataclass_fields__"
            ):
                field_data = config_dict[field_info.name]
                if isinstance(field_data, dict[str, Any]):
                    config_dict[field_info.name] = field_info.type(**field_data)

        return AppConfig(**config_dict)

    def _deep_merge(
        self, base: dict[str, Any], update: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict[str, Any])
                and isinstance(value, dict[str, Any])
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_config(self) -> None:
        """Validate configuration values."""
        if not self._config:
            return

        # Validate server config
        if self._config.server.port < 1 or self._config.server.port > 65535:
            raise ValueError(f"Invalid port: {self._config.server.port}")

        if self._config.server.workers < 1:
            raise ValueError(f"Invalid worker count: {self._config.server.workers}")

        # Validate performance limits
        if self._config.performance.max_nodes < 1:
            raise ValueError(f"Invalid max_nodes: {self._config.performance.max_nodes}")

        # Validate Redis config if URL is provided
        if self._config.redis.url:
            try:
                from urllib.parse import urlparse

                parsed = urlparse(self._config.redis.url)
                if parsed.scheme not in ["redis", "rediss"]:
                    raise ValueError(f"Invalid Redis URL scheme: {parsed.scheme}")
            except ImportError:
                pass  # Skip validation if urllib not available

        logger.debug("Configuration validation passed")


# Global configuration instance
_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> AppConfig:
    """Get the current application configuration."""
    return get_config_manager().get_config()


def load_config(
    config_file: str | None = None,
    env_prefix: str = "MCP_",
    environment: Environment | None = None,
) -> AppConfig:
    """Load application configuration."""
    manager = get_config_manager()

    # Load from default files
    default_files = [
        "config.yaml",
        "config.yml",
        "config.json",
        f"config.{environment.value if environment else 'development'}.yaml",
    ]

    for file_path in default_files:
        if Path(file_path).exists():
            manager.load_from_file(file_path)

    # Load from specified file
    if config_file:
        manager.load_from_file(config_file)

    # Load from environment
    manager.load_from_env(env_prefix)

    return manager.build()


# Configuration decorators
def config_value(key: str, default: Any = None) -> Any:
    """Decorator to inject configuration values."""

    def decorator(func: Any) -> Any:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            config = get_config()
            value = _get_nested_value(config, key, default)
            return func(value, *args, **kwargs)

        return wrapper

    return decorator


def _get_nested_value(obj: Any, key: str, default: Any = None) -> Any:
    """Get nested value from object using dot notation."""
    parts = key.split(".")
    current = obj

    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return default

    return current
