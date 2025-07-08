"""Configuration validation and schema definitions."""

import re
import ipaddress
from typing import Any, Dict, List, Optional, Union
from jsonschema import Draft7Validator

# JSON Schema for configuration validation
CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "environment": {
            "type": "string",
            "enum": ["development", "testing", "staging", "production"],
            "description": "Application environment"
        },
        "name": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9-_]+$",
            "minLength": 1,
            "maxLength": 100,
            "description": "Application name"
        },
        "version": {
            "type": "string",
            "pattern": "^\\d+\\.\\d+\\.\\d+",
            "description": "Application version (semantic versioning)"
        },
        "debug": {
            "type": "boolean",
            "description": "Enable debug mode"
        },
        "server": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Server host address"
                },
                "port": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 65535,
                    "description": "Server port number"
                },
                "workers": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Number of worker processes"
                },
                "max_connections": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum concurrent connections"
                },
                "request_timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3600,
                    "description": "Request timeout in seconds"
                },
                "keepalive_timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                    "description": "Keep-alive timeout in seconds"
                },
                "debug": {
                    "type": "boolean",
                    "description": "Enable server debug mode"
                }
            },
            "additionalProperties": False
        },
        "security": {
            "type": "object",
            "properties": {
                "enable_auth": {
                    "type": "boolean",
                    "description": "Enable authentication"
                },
                "api_key_required": {
                    "type": "boolean",
                    "description": "Require API key for requests"
                },
                "allowed_origins": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "CORS allowed origins"
                },
                "rate_limit_enabled": {
                    "type": "boolean",
                    "description": "Enable rate limiting"
                },
                "rate_limit_requests": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of requests allowed per window"
                },
                "rate_limit_window": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Rate limit window in seconds"
                },
                "audit_enabled": {
                    "type": "boolean",
                    "description": "Enable audit logging"
                },
                "audit_log_file": {
                    "type": "string",
                    "description": "Audit log file path"
                },
                "max_request_size": {
                    "type": "integer",
                    "minimum": 1024,
                    "description": "Maximum request size in bytes"
                },
                "max_nodes_per_request": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100000,
                    "description": "Maximum nodes per request"
                },
                "max_edges_per_request": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000000,
                    "description": "Maximum edges per request"
                },
                "safe_id_pattern": {
                    "type": "string",
                    "description": "Regex pattern for safe identifiers"
                }
            },
            "additionalProperties": False
        },
        "performance": {
            "type": "object",
            "properties": {
                "max_nodes": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum nodes in a graph"
                },
                "max_edges": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum edges in a graph"
                },
                "memory_limit_mb": {
                    "type": "integer",
                    "minimum": 64,
                    "description": "Memory limit in MB"
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3600,
                    "description": "Operation timeout in seconds"
                },
                "enable_caching": {
                    "type": "boolean",
                    "description": "Enable caching"
                },
                "cache_size_mb": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Cache size in MB"
                },
                "cache_ttl": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Cache TTL in seconds"
                },
                "parallel_processing": {
                    "type": "boolean",
                    "description": "Enable parallel processing"
                },
                "use_cython": {
                    "type": "boolean",
                    "description": "Use Cython optimizations"
                },
                "numpy_optimization": {
                    "type": "boolean",
                    "description": "Use NumPy optimizations"
                },
                "max_memory_mb": {
                    "type": "integer",
                    "minimum": 64,
                    "description": "Maximum memory usage in MB"
                },
                "operation_timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Individual operation timeout"
                },
                "max_concurrent_requests": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum concurrent requests"
                }
            },
            "additionalProperties": False
        },
        "storage": {
            "type": "object",
            "properties": {
                "backend": {
                    "type": "string",
                    "enum": ["auto", "redis", "memory"],
                    "description": "Storage backend type"
                },
                "redis_url": {
                    "type": ["string", "null"],
                    "description": "Redis connection URL"
                },
                "redis_host": {
                    "type": "string",
                    "description": "Redis host"
                },
                "redis_port": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 65535,
                    "description": "Redis port"
                },
                "redis_db": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 15,
                    "description": "Redis database number"
                },
                "redis_prefix": {
                    "type": "string",
                    "description": "Redis key prefix"
                },
                "redis_pool_size": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Redis connection pool size"
                },
                "redis_timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Redis connection timeout"
                },
                "redis_retry_attempts": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Redis retry attempts"
                },
                "redis_ttl": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Default Redis TTL"
                },
                "compression_level": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 9,
                    "description": "Compression level (0-9)"
                },
                "max_graph_size_mb": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum graph size in MB"
                }
            },
            "additionalProperties": False
        },
        "features": {
            "type": "object",
            "properties": {
                "machine_learning": {
                    "type": "boolean",
                    "description": "Enable ML features"
                },
                "visualization": {
                    "type": "boolean",
                    "description": "Enable visualization"
                },
                "gpu_acceleration": {
                    "type": "boolean",
                    "description": "Enable GPU acceleration"
                },
                "enterprise_features": {
                    "type": "boolean",
                    "description": "Enable enterprise features"
                },
                "monitoring": {
                    "type": "boolean",
                    "description": "Enable monitoring"
                },
                "metrics_endpoint": {
                    "type": "string",
                    "pattern": "^/.*",
                    "description": "Metrics endpoint path"
                },
                "health_endpoint": {
                    "type": "string",
                    "pattern": "^/.*",
                    "description": "Health check endpoint path"
                },
                "api_v2_enabled": {
                    "type": "boolean",
                    "description": "Enable API v2"
                },
                "advanced_algorithms": {
                    "type": "boolean",
                    "description": "Enable advanced algorithms"
                },
                "caching_enabled": {
                    "type": "boolean",
                    "description": "Enable caching"
                }
            },
            "additionalProperties": False
        },
        "logging": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    "description": "Log level"
                },
                "format": {
                    "type": "string",
                    "description": "Log format string"
                },
                "file": {
                    "type": ["string", "null"],
                    "description": "Log file path"
                },
                "max_file_size": {
                    "type": "integer",
                    "minimum": 1024,
                    "description": "Maximum log file size"
                },
                "backup_count": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Number of backup log files"
                },
                "json_format": {
                    "type": "boolean",
                    "description": "Use JSON log format"
                }
            },
            "additionalProperties": False
        }
    },
    "additionalProperties": False
}


class ConfigValidator:
    """Advanced configuration validator with business logic."""
    
    def __init__(self):
        self.validator = Draft7Validator(CONFIG_SCHEMA)
    
    def validate_schema(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against JSON schema."""
        errors = []
        for error in self.validator.iter_errors(config):
            path = " -> ".join(str(p) for p in error.absolute_path)
            errors.append(f"{path}: {error.message}")
        return errors
    
    def validate_business_rules(self, config: Dict[str, Any]) -> List[str]:
        """Validate business logic and constraints."""
        errors = []
        
        # Server configuration validation
        server = config.get("server", {})
        if server:
            errors.extend(self._validate_server_config(server))
        
        # Security configuration validation
        security = config.get("security", {})
        if security:
            errors.extend(self._validate_security_config(security))
        
        # Performance configuration validation
        performance = config.get("performance", {})
        if performance:
            errors.extend(self._validate_performance_config(performance))
        
        # Storage configuration validation
        storage = config.get("storage", {})
        if storage:
            errors.extend(self._validate_storage_config(storage))
        
        # Cross-section validation
        errors.extend(self._validate_cross_sections(config))
        
        return errors
    
    def _validate_server_config(self, server: Dict[str, Any]) -> List[str]:
        """Validate server configuration."""
        errors = []
        
        # Validate host
        host = server.get("host")
        if host and not self._is_valid_host(host):
            errors.append(f"Invalid host address: {host}")
        
        # Validate worker/connection ratios
        workers = server.get("workers", 1)
        max_connections = server.get("max_connections", 1000)
        if max_connections < workers * 10:
            errors.append(f"max_connections ({max_connections}) should be at least 10x workers ({workers})")
        
        return errors
    
    def _validate_security_config(self, security: Dict[str, Any]) -> List[str]:
        """Validate security configuration."""
        errors = []
        
        # Validate regex pattern
        pattern = security.get("safe_id_pattern")
        if pattern:
            try:
                re.compile(pattern)
            except re.error as e:
                errors.append(f"Invalid regex pattern: {e}")
        
        # Validate rate limiting
        if security.get("rate_limit_enabled"):
            requests = security.get("rate_limit_requests", 0)
            window = security.get("rate_limit_window", 0)
            if requests > 0 and window > 0:
                rate = requests / window
                if rate > 1000:  # More than 1000 requests per second
                    errors.append(f"Rate limit too high: {rate:.1f} requests/second")
        
        # Validate resource limits
        max_nodes = security.get("max_nodes_per_request", 0)
        max_edges = security.get("max_edges_per_request", 0)
        if max_nodes > 0 and max_edges > 0:
            if max_edges > max_nodes * max_nodes:
                errors.append("max_edges_per_request exceeds theoretical maximum for complete graph")
        
        return errors
    
    def _validate_performance_config(self, performance: Dict[str, Any]) -> List[str]:
        """Validate performance configuration."""
        errors = []
        
        # Validate memory limits
        memory_limit = performance.get("memory_limit_mb", 0)
        cache_size = performance.get("cache_size_mb", 0)
        max_memory = performance.get("max_memory_mb", 0)
        
        if memory_limit > 0 and cache_size > 0:
            if cache_size > memory_limit * 0.8:
                errors.append("cache_size_mb should not exceed 80% of memory_limit_mb")
        
        if max_memory > 0 and memory_limit > 0:
            if max_memory > memory_limit:
                errors.append("max_memory_mb should not exceed memory_limit_mb")
        
        # Validate timeout relationships
        timeout = performance.get("timeout_seconds", 0)
        operation_timeout = performance.get("operation_timeout", 0)
        if timeout > 0 and operation_timeout > 0:
            if operation_timeout >= timeout:
                errors.append("operation_timeout should be less than timeout_seconds")
        
        return errors
    
    def _validate_storage_config(self, storage: Dict[str, Any]) -> List[str]:
        """Validate storage configuration."""
        errors = []
        
        # Validate Redis URL format
        redis_url = storage.get("redis_url")
        if redis_url:
            if not redis_url.startswith(("redis://", "rediss://")):
                errors.append("redis_url must start with redis:// or rediss://")
        
        # Validate compression level
        compression = storage.get("compression_level", 6)
        if not (0 <= compression <= 9):
            errors.append(f"compression_level must be between 0 and 9, got {compression}")
        
        return errors
    
    def _validate_cross_sections(self, config: Dict[str, Any]) -> List[str]:
        """Validate cross-section constraints."""
        errors = []
        
        # Environment-specific validations
        environment = config.get("environment", "development")
        
        if environment == "production":
            # Production environment checks
            if config.get("debug", False):
                errors.append("Debug mode should be disabled in production")
            
            security = config.get("security", {})
            if not security.get("rate_limit_enabled", True):
                errors.append("Rate limiting should be enabled in production")
            
            if not security.get("audit_enabled", False):
                errors.append("Audit logging should be enabled in production")
        
        # Feature dependency validation
        features = config.get("features", {})
        performance = config.get("performance", {})
        
        if features.get("gpu_acceleration") and not features.get("machine_learning"):
            errors.append("GPU acceleration requires machine learning features to be enabled")
        
        if features.get("caching_enabled") and not performance.get("enable_caching"):
            errors.append("Inconsistent caching configuration between features and performance")
        
        return errors
    
    def _is_valid_host(self, host: str) -> bool:
        """Validate host address."""
        if host in ("localhost", "0.0.0.0"):
            return True
        
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            # Could be a hostname
            if re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$", host):
                return True
        
        return False
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration completely."""
        errors = []
        
        # Schema validation
        errors.extend(self.validate_schema(config))
        
        # Business rule validation
        if not errors:  # Only validate business rules if schema is valid
            errors.extend(self.validate_business_rules(config))
        
        return errors


def validate_environment_requirements(config: Dict[str, Any]) -> List[str]:
    """Validate environment-specific requirements."""
    errors = []
    environment = config.get("environment", "development")
    
    production_requirements = {
        "security.enable_auth": True,
        "security.rate_limit_enabled": True,
        "security.audit_enabled": True,
        "features.monitoring": True,
        "debug": False
    }
    
    staging_requirements = {
        "security.rate_limit_enabled": True,
        "features.monitoring": True
    }
    
    requirements = {}
    if environment == "production":
        requirements = production_requirements
    elif environment == "staging":
        requirements = staging_requirements
    
    for key_path, expected_value in requirements.items():
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if key not in current:
                current = None
                break
            current = current[key]
        
        if current != expected_value:
            errors.append(f"{environment} environment requires {key_path} = {expected_value}")
    
    return errors