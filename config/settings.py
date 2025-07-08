"""Comprehensive configuration management for NetworkX MCP Server.

This module provides a centralized configuration system that supports:
- Environment variables with defaults
- Configuration files (YAML/JSON)
- Runtime validation
- Hot reload for safe settings
- Schema validation and documentation

Configuration Priority (highest to lowest):
1. Environment variables
2. Configuration files 
3. Default values
"""

import os
import json
import yaml
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, Callable
from dataclasses import dataclass, field, fields, asdict

# Optional dependencies for hot reload
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object  # Dummy base class

# Optional JSON schema validation
try:
    import jsonschema
    from jsonschema import validate, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server network and connection settings."""
    host: str = "localhost"
    port: int = 8765
    workers: int = 4
    max_connections: int = 1000
    request_timeout: int = 30
    keepalive_timeout: int = 5
    debug: bool = False
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load from environment variables."""
        return cls(
            host=os.getenv("MCP_HOST", "localhost"),
            port=int(os.getenv("MCP_PORT", "8765")),
            workers=int(os.getenv("MCP_WORKERS", "4")),
            max_connections=int(os.getenv("MCP_MAX_CONNECTIONS", "1000")),
            request_timeout=int(os.getenv("MCP_REQUEST_TIMEOUT", "30")),
            keepalive_timeout=int(os.getenv("MCP_KEEPALIVE_TIMEOUT", "5")),
            debug=os.getenv("MCP_DEBUG", "false").lower() == "true"
        )


@dataclass
class SecurityConfig:
    """Security and authentication settings."""
    enable_auth: bool = False
    api_key_required: bool = False
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 1000
    rate_limit_window: int = 60
    audit_enabled: bool = False
    audit_log_file: str = "audit.log"
    max_request_size: int = 10485760  # 10MB
    
    # Input validation limits
    max_nodes_per_request: int = 1000
    max_edges_per_request: int = 10000
    safe_id_pattern: str = r"^[a-zA-Z0-9_-]{1,100}$"
    
    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Load from environment variables."""
        return cls(
            enable_auth=os.getenv("MCP_ENABLE_AUTH", "false").lower() == "true",
            api_key_required=os.getenv("MCP_API_KEY_REQUIRED", "false").lower() == "true",
            allowed_origins=os.getenv("MCP_ALLOWED_ORIGINS", "*").split(","),
            rate_limit_enabled=os.getenv("MCP_RATE_LIMIT_ENABLED", "true").lower() == "true",
            rate_limit_requests=int(os.getenv("MCP_RATE_LIMIT_REQUESTS", "1000")),
            rate_limit_window=int(os.getenv("MCP_RATE_LIMIT_WINDOW", "60")),
            audit_enabled=os.getenv("MCP_AUDIT_ENABLED", "false").lower() == "true",
            audit_log_file=os.getenv("MCP_AUDIT_LOG_FILE", "audit.log"),
            max_request_size=int(os.getenv("MCP_MAX_REQUEST_SIZE", "10485760")),
            max_nodes_per_request=int(os.getenv("MAX_NODES_PER_REQUEST", "1000")),
            max_edges_per_request=int(os.getenv("MAX_EDGES_PER_REQUEST", "10000")),
            safe_id_pattern=os.getenv("SAFE_ID_PATTERN", r"^[a-zA-Z0-9_-]{1,100}$")
        )


@dataclass
class PerformanceConfig:
    """Performance and resource management settings."""
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
    
    # Resource limits
    max_memory_mb: int = 1024
    operation_timeout: int = 30
    max_concurrent_requests: int = 10
    
    @classmethod
    def from_env(cls) -> "PerformanceConfig":
        """Load from environment variables."""
        return cls(
            max_nodes=int(os.getenv("MAX_NODES_PER_GRAPH", "1000000")),
            max_edges=int(os.getenv("MAX_EDGES_PER_GRAPH", "10000000")),
            memory_limit_mb=int(os.getenv("MEMORY_LIMIT_MB", "4096")),
            timeout_seconds=int(os.getenv("OPERATION_TIMEOUT", "300")),
            enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
            cache_size_mb=int(os.getenv("CACHE_SIZE_MB", "512")),
            cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
            parallel_processing=os.getenv("PARALLEL_PROCESSING", "true").lower() == "true",
            use_cython=os.getenv("USE_CYTHON", "true").lower() == "true",
            numpy_optimization=os.getenv("NUMPY_OPTIMIZATION", "true").lower() == "true",
            max_memory_mb=int(os.getenv("MAX_MEMORY_MB", "1024")),
            operation_timeout=int(os.getenv("OPERATION_TIMEOUT", "30")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        )


@dataclass
class StorageConfig:
    """Storage backend configuration."""
    backend: str = "auto"  # auto, redis, memory
    redis_url: Optional[str] = None
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_prefix: str = "networkx_mcp"
    redis_pool_size: int = 10
    redis_timeout: int = 5
    redis_retry_attempts: int = 3
    redis_ttl: int = 3600
    compression_level: int = 6
    max_graph_size_mb: int = 100
    
    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Load from environment variables."""
        return cls(
            backend=os.getenv("STORAGE_BACKEND", "auto"),
            redis_url=os.getenv("REDIS_URL"),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            redis_prefix=os.getenv("REDIS_PREFIX", "networkx_mcp"),
            redis_pool_size=int(os.getenv("REDIS_POOL_SIZE", "10")),
            redis_timeout=int(os.getenv("REDIS_TIMEOUT", "5")),
            redis_retry_attempts=int(os.getenv("REDIS_RETRY_ATTEMPTS", "3")),
            redis_ttl=int(os.getenv("REDIS_TTL", "3600")),
            compression_level=int(os.getenv("COMPRESSION_LEVEL", "6")),
            max_graph_size_mb=int(os.getenv("MAX_GRAPH_SIZE_MB", "100"))
        )


@dataclass
class FeaturesConfig:
    """Feature flags and optional components."""
    machine_learning: bool = True
    visualization: bool = True
    gpu_acceleration: bool = False
    enterprise_features: bool = False
    monitoring: bool = True
    metrics_endpoint: str = "/metrics"
    health_endpoint: str = "/health"
    api_v2_enabled: bool = True
    advanced_algorithms: bool = True
    caching_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> "FeaturesConfig":
        """Load from environment variables."""
        return cls(
            machine_learning=os.getenv("FEATURE_ML", "true").lower() == "true",
            visualization=os.getenv("FEATURE_VISUALIZATION", "true").lower() == "true",
            gpu_acceleration=os.getenv("FEATURE_GPU", "false").lower() == "true",
            enterprise_features=os.getenv("FEATURE_ENTERPRISE", "false").lower() == "true",
            monitoring=os.getenv("FEATURE_MONITORING", "true").lower() == "true",
            metrics_endpoint=os.getenv("METRICS_ENDPOINT", "/metrics"),
            health_endpoint=os.getenv("HEALTH_ENDPOINT", "/health"),
            api_v2_enabled=os.getenv("API_V2_ENABLED", "true").lower() == "true",
            advanced_algorithms=os.getenv("ADVANCED_ALGORITHMS", "true").lower() == "true",
            caching_enabled=os.getenv("CACHING_ENABLED", "true").lower() == "true"
        )


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    json_format: bool = False
    
    @classmethod
    def from_env(cls) -> "LoggingConfig":
        """Load from environment variables."""
        return cls(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file=os.getenv("LOG_FILE"),
            max_file_size=int(os.getenv("LOG_MAX_FILE_SIZE", "10485760")),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            json_format=os.getenv("LOG_JSON_FORMAT", "false").lower() == "true"
        )


@dataclass
class Settings:
    """Main settings container."""
    environment: str = "development"
    name: str = "networkx-mcp-server"
    version: str = "1.0.0"
    debug: bool = False
    
    # Component configurations
    server: ServerConfig = field(default_factory=ServerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Hot reload safe settings (can be changed at runtime)
    _hot_reload_safe = {
        "security.rate_limit_requests",
        "security.rate_limit_window", 
        "security.audit_enabled",
        "performance.cache_ttl",
        "performance.enable_caching",
        "features.monitoring",
        "logging.level",
        "logging.json_format"
    }
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            environment=os.getenv("MCP_ENVIRONMENT", "development"),
            name=os.getenv("MCP_NAME", "networkx-mcp-server"),
            version=os.getenv("MCP_VERSION", "1.0.0"),
            debug=os.getenv("MCP_DEBUG", "false").lower() == "true",
            server=ServerConfig.from_env(),
            security=SecurityConfig.from_env(),
            performance=PerformanceConfig.from_env(),
            storage=StorageConfig.from_env(),
            features=FeaturesConfig.from_env(),
            logging=LoggingConfig.from_env()
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        # Validate server config
        if not (1 <= self.server.port <= 65535):
            errors.append(f"Invalid server port: {self.server.port}")
        
        if self.server.workers < 1:
            errors.append(f"Invalid worker count: {self.server.workers}")
        
        # Validate security config
        if self.security.rate_limit_requests < 1:
            errors.append(f"Invalid rate limit: {self.security.rate_limit_requests}")
        
        if self.security.max_request_size < 1024:  # 1KB minimum
            errors.append(f"Request size too small: {self.security.max_request_size}")
        
        # Validate performance config
        if self.performance.memory_limit_mb < 64:  # 64MB minimum
            errors.append(f"Memory limit too small: {self.performance.memory_limit_mb}")
        
        if self.performance.timeout_seconds < 1:
            errors.append(f"Invalid timeout: {self.performance.timeout_seconds}")
        
        # Validate storage config
        if self.storage.redis_port < 1 or self.storage.redis_port > 65535:
            errors.append(f"Invalid Redis port: {self.storage.redis_port}")
        
        # Validate environment
        valid_environments = {"development", "testing", "staging", "production"}
        if self.environment not in valid_environments:
            errors.append(f"Invalid environment: {self.environment}. Must be one of {valid_environments}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)
    
    def is_hot_reload_safe(self, key: str) -> bool:
        """Check if a setting can be safely changed at runtime."""
        return key in self._hot_reload_safe


if WATCHDOG_AVAILABLE:
    class ConfigFileHandler(FileSystemEventHandler):
        """Handles configuration file changes for hot reload."""
        
        def __init__(self, config_manager: "ConfigManager"):
            self.config_manager = config_manager
            self.last_modified = {}
        
        def on_modified(self, event):
            """Handle file modification events."""
            if event.is_directory:
                return
            
            # Debounce rapid file changes
            now = time.time()
            if event.src_path in self.last_modified:
                if now - self.last_modified[event.src_path] < 1.0:  # 1 second debounce
                    return
            
            self.last_modified[event.src_path] = now
            
            if event.src_path in self.config_manager.watched_files:
                logger.info(f"Configuration file changed: {event.src_path}")
                try:
                    self.config_manager.reload()
                except Exception as e:
                    logger.error(f"Failed to reload configuration: {e}")
else:
    # Dummy class when watchdog is not available
    class ConfigFileHandler:
        def __init__(self, config_manager):
            pass


class ConfigManager:
    """Manages configuration loading, validation, and hot reload."""
    
    def __init__(self, config_paths: Optional[List[str]] = None):
        self.config_paths = config_paths or self._default_config_paths()
        self.settings: Optional[Settings] = None
        self.watched_files: List[str] = []
        self.observer: Optional[Observer] = None
        self._lock = threading.Lock()
        self._change_callbacks: List[Callable[[Settings, Settings], None]] = []
        
    def _default_config_paths(self) -> List[str]:
        """Get default configuration file paths."""
        env = os.getenv("MCP_ENVIRONMENT", "development")
        return [
            "config.yaml",
            "config.yml", 
            "config.json",
            f"config.{env}.yaml",
            f"config.{env}.yml",
            f"config.{env}.json",
            f"config/{env}.yaml",
            f"config/{env}.yml",
            f"config/{env}.json"
        ]
    
    def load(self) -> Settings:
        """Load configuration from all sources."""
        with self._lock:
            # Start with environment variables
            settings = Settings.from_env()
            
            # Override with configuration files
            config_data = {}
            for path in self.config_paths:
                if os.path.exists(path):
                    logger.info(f"Loading configuration from {path}")
                    try:
                        data = self._load_config_file(path)
                        config_data.update(data)
                        if path not in self.watched_files:
                            self.watched_files.append(path)
                    except Exception as e:
                        logger.error(f"Failed to load config file {path}: {e}")
            
            # Apply configuration file data
            if config_data:
                settings = self._merge_config(settings, config_data)
            
            # Validate configuration
            errors = settings.validate()
            if errors:
                error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
                raise ValueError(error_msg)
            
            self.settings = settings
            return settings
    
    def _load_config_file(self, path: str) -> Dict[str, Any]:
        """Load configuration from a file."""
        with open(path, 'r') as f:
            if path.endswith('.json'):
                return json.load(f)
            elif path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f) or {}
            else:
                raise ValueError(f"Unsupported config file format: {path}")
    
    def _merge_config(self, settings: Settings, config_data: Dict[str, Any]) -> Settings:
        """Merge configuration data into settings."""
        # Create a new settings object with updated values
        settings_dict = asdict(settings)
        
        # Deep merge configuration data
        def deep_update(base_dict: Dict, update_dict: Dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(settings_dict, config_data)
        
        # Reconstruct settings object
        return Settings(
            environment=settings_dict.get("environment", settings.environment),
            name=settings_dict.get("name", settings.name),
            version=settings_dict.get("version", settings.version),
            debug=settings_dict.get("debug", settings.debug),
            server=ServerConfig(**settings_dict.get("server", asdict(settings.server))),
            security=SecurityConfig(**settings_dict.get("security", asdict(settings.security))),
            performance=PerformanceConfig(**settings_dict.get("performance", asdict(settings.performance))),
            storage=StorageConfig(**settings_dict.get("storage", asdict(settings.storage))),
            features=FeaturesConfig(**settings_dict.get("features", asdict(settings.features))),
            logging=LoggingConfig(**settings_dict.get("logging", asdict(settings.logging)))
        )
    
    def reload(self, hot_reload_only: bool = True) -> Settings:
        """Reload configuration."""
        old_settings = self.settings
        new_settings = self.load()
        
        if hot_reload_only and old_settings:
            # Only apply hot reload safe changes
            safe_settings = self._apply_hot_reload_changes(old_settings, new_settings)
            self.settings = safe_settings
            
            # Notify change callbacks
            for callback in self._change_callbacks:
                try:
                    callback(old_settings, safe_settings)
                except Exception as e:
                    logger.error(f"Configuration change callback failed: {e}")
            
            return safe_settings
        else:
            self.settings = new_settings
            return new_settings
    
    def _apply_hot_reload_changes(self, old_settings: Settings, new_settings: Settings) -> Settings:
        """Apply only hot-reload safe changes."""
        # Start with old settings
        settings_dict = asdict(old_settings)
        new_dict = asdict(new_settings)
        
        # Apply only safe changes
        safe_keys = old_settings._hot_reload_safe
        for key in safe_keys:
            parts = key.split('.')
            old_value = settings_dict
            new_value = new_dict
            
            # Navigate to the nested value
            for part in parts[:-1]:
                old_value = old_value.get(part, {})
                new_value = new_value.get(part, {})
            
            final_key = parts[-1]
            if final_key in new_value and new_value[final_key] != old_value.get(final_key):
                old_value[final_key] = new_value[final_key]
                logger.info(f"Hot reloaded setting: {key} = {new_value[final_key]}")
        
        # Reconstruct settings
        return Settings(
            environment=settings_dict["environment"],
            name=settings_dict["name"],
            version=settings_dict["version"],
            debug=settings_dict["debug"],
            server=ServerConfig(**settings_dict["server"]),
            security=SecurityConfig(**settings_dict["security"]),
            performance=PerformanceConfig(**settings_dict["performance"]),
            storage=StorageConfig(**settings_dict["storage"]),
            features=FeaturesConfig(**settings_dict["features"]),
            logging=LoggingConfig(**settings_dict["logging"])
        )
    
    def start_watching(self):
        """Start watching configuration files for changes."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("Watchdog not available - hot reload disabled")
            return
            
        if not self.watched_files or self.observer:
            return
        
        self.observer = Observer()
        handler = ConfigFileHandler(self)
        
        # Watch directories containing config files
        watched_dirs = set()
        for file_path in self.watched_files:
            dir_path = os.path.dirname(os.path.abspath(file_path))
            if dir_path not in watched_dirs:
                self.observer.schedule(handler, dir_path, recursive=False)
                watched_dirs.add(dir_path)
        
        self.observer.start()
        logger.info(f"Started watching {len(watched_dirs)} directories for config changes")
    
    def stop_watching(self):
        """Stop watching configuration files."""
        if not WATCHDOG_AVAILABLE:
            return
            
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Stopped watching configuration files")
    
    def add_change_callback(self, callback: Callable[[Settings, Settings], None]):
        """Add a callback for configuration changes."""
        self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[Settings, Settings], None]):
        """Remove a configuration change callback."""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None
_settings: Optional[Settings] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_settings() -> Settings:
    """Get the current settings."""
    global _settings
    if _settings is None:
        manager = get_config_manager()
        _settings = manager.load()
        
        # Start watching for changes in non-test environments
        if _settings.environment != "testing":
            manager.start_watching()
    
    return _settings


def reload_settings(hot_reload_only: bool = True) -> Settings:
    """Reload settings from configuration sources."""
    global _settings
    manager = get_config_manager()
    _settings = manager.reload(hot_reload_only)
    return _settings


def configure_logging(settings: Settings):
    """Configure logging based on settings."""
    logging_config = settings.logging
    
    # Set log level
    log_level = getattr(logging, logging_config.level.upper(), logging.INFO)
    
    # Create formatter
    if logging_config.json_format:
        # For JSON logging, you might want to use a JSON formatter library
        formatter = logging.Formatter(logging_config.format)
    else:
        formatter = logging.Formatter(logging_config.format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if logging_config.file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            logging_config.file,
            maxBytes=logging_config.max_file_size,
            backupCount=logging_config.backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


# Configuration change callback to reconfigure logging
def _on_config_change(old_settings: Settings, new_settings: Settings):
    """Handle configuration changes."""
    if old_settings.logging != new_settings.logging:
        configure_logging(new_settings)
        logger.info("Logging configuration updated")


# Register the logging callback
get_config_manager().add_change_callback(_on_config_change)