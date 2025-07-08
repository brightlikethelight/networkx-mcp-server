"""Configuration schema documentation and utilities."""

from typing import Dict, Any, List
from dataclasses import fields
import json

from .settings import (
    Settings, ServerConfig, SecurityConfig, PerformanceConfig, 
    StorageConfig, FeaturesConfig, LoggingConfig
)


def generate_schema_documentation() -> str:
    """Generate comprehensive schema documentation."""
    
    doc = """# NetworkX MCP Server Configuration Schema

## Overview

The NetworkX MCP Server uses a comprehensive configuration system that supports:
- Environment variables (highest priority)
- Configuration files in YAML/JSON format
- Default values (lowest priority)

## Configuration Structure

The configuration is organized into the following sections:

"""
    
    # Generate documentation for each config section
    config_classes = [
        ("Application", Settings),
        ("Server", ServerConfig),
        ("Security", SecurityConfig), 
        ("Performance", PerformanceConfig),
        ("Storage", StorageConfig),
        ("Features", FeaturesConfig),
        ("Logging", LoggingConfig),
    ]
    
    for section_name, config_class in config_classes:
        doc += f"### {section_name} Configuration\n\n"
        doc += _generate_class_documentation(config_class)
        doc += "\n"
    
    # Add environment variable documentation
    doc += generate_environment_variables_doc()
    
    # Add examples
    doc += generate_examples_doc()
    
    # Add hot reload documentation
    doc += generate_hot_reload_doc()
    
    return doc


def _generate_class_documentation(config_class) -> str:
    """Generate documentation for a configuration class."""
    doc = f"**{config_class.__name__}**\n\n"
    
    if config_class.__doc__:
        doc += f"{config_class.__doc__.strip()}\n\n"
    
    doc += "| Field | Type | Default | Description |\n"
    doc += "|-------|------|---------|-------------|\n"
    
    instance = config_class()
    
    for field in fields(config_class):
        field_name = field.name
        field_type = _format_type(field.type)
        default_value = getattr(instance, field_name)
        
        # Format default value
        if isinstance(default_value, str):
            default_value = f'"{default_value}"'
        elif isinstance(default_value, list):
            default_value = f"{default_value}"
        elif default_value is None:
            default_value = "null"
        else:
            default_value = str(default_value)
        
        # Extract description from docstring or field metadata
        description = _get_field_description(field)
        
        doc += f"| `{field_name}` | {field_type} | {default_value} | {description} |\n"
    
    doc += "\n"
    return doc


def _format_type(field_type) -> str:
    """Format type annotation for documentation."""
    if hasattr(field_type, '__origin__'):
        if field_type.__origin__ is list:
            return f"List[{_format_type(field_type.__args__[0])}]"
        elif field_type.__origin__ is Union:
            types = [_format_type(arg) for arg in field_type.__args__]
            return " | ".join(types)
    
    if hasattr(field_type, '__name__'):
        return field_type.__name__
    
    return str(field_type)


def _get_field_description(field) -> str:
    """Extract field description from metadata or infer from name."""
    # You could extend this to read from field metadata
    # For now, we'll use some basic descriptions
    descriptions = {
        'host': 'Server host address',
        'port': 'Server port number',
        'workers': 'Number of worker processes',
        'max_connections': 'Maximum concurrent connections',
        'request_timeout': 'Request timeout in seconds',
        'keepalive_timeout': 'Keep-alive timeout in seconds',
        'debug': 'Enable debug mode',
        'enable_auth': 'Enable authentication',
        'api_key_required': 'Require API key for requests',
        'allowed_origins': 'CORS allowed origins',
        'rate_limit_enabled': 'Enable rate limiting',
        'rate_limit_requests': 'Number of requests allowed per window',
        'rate_limit_window': 'Rate limit window in seconds',
        'audit_enabled': 'Enable audit logging',
        'audit_log_file': 'Audit log file path',
        'max_request_size': 'Maximum request size in bytes',
        'max_nodes_per_request': 'Maximum nodes per request',
        'max_edges_per_request': 'Maximum edges per request',
        'safe_id_pattern': 'Regex pattern for safe identifiers',
        'max_nodes': 'Maximum nodes in a graph',
        'max_edges': 'Maximum edges in a graph',
        'memory_limit_mb': 'Memory limit in MB',
        'timeout_seconds': 'Operation timeout in seconds',
        'enable_caching': 'Enable caching',
        'cache_size_mb': 'Cache size in MB',
        'cache_ttl': 'Cache TTL in seconds',
        'parallel_processing': 'Enable parallel processing',
        'use_cython': 'Use Cython optimizations',
        'numpy_optimization': 'Use NumPy optimizations',
        'max_memory_mb': 'Maximum memory usage in MB',
        'operation_timeout': 'Individual operation timeout',
        'max_concurrent_requests': 'Maximum concurrent requests',
        'backend': 'Storage backend type',
        'redis_url': 'Redis connection URL',
        'redis_host': 'Redis host',
        'redis_port': 'Redis port',
        'redis_db': 'Redis database number',
        'redis_prefix': 'Redis key prefix',
        'redis_pool_size': 'Redis connection pool size',
        'redis_timeout': 'Redis connection timeout',
        'redis_retry_attempts': 'Redis retry attempts',
        'redis_ttl': 'Default Redis TTL',
        'compression_level': 'Compression level (0-9)',
        'max_graph_size_mb': 'Maximum graph size in MB',
        'machine_learning': 'Enable ML features',
        'visualization': 'Enable visualization',
        'gpu_acceleration': 'Enable GPU acceleration',
        'enterprise_features': 'Enable enterprise features',
        'monitoring': 'Enable monitoring',
        'metrics_endpoint': 'Metrics endpoint path',
        'health_endpoint': 'Health check endpoint path',
        'api_v2_enabled': 'Enable API v2',
        'advanced_algorithms': 'Enable advanced algorithms',
        'caching_enabled': 'Enable caching',
        'level': 'Log level',
        'format': 'Log format string',
        'file': 'Log file path',
        'max_file_size': 'Maximum log file size',
        'backup_count': 'Number of backup log files',
        'json_format': 'Use JSON log format',
        'environment': 'Application environment',
        'name': 'Application name',
        'version': 'Application version',
    }
    
    return descriptions.get(field.name, 'Configuration option')


def generate_environment_variables_doc() -> str:
    """Generate environment variables documentation."""
    doc = """## Environment Variables

All configuration options can be set via environment variables. The variable names follow this pattern:

### Core Application Variables
- `MCP_ENVIRONMENT` - Application environment (development/testing/staging/production)
- `MCP_NAME` - Application name
- `MCP_VERSION` - Application version  
- `MCP_DEBUG` - Enable debug mode (true/false)

### Server Variables
- `MCP_HOST` - Server host address
- `MCP_PORT` - Server port number
- `MCP_WORKERS` - Number of worker processes
- `MCP_MAX_CONNECTIONS` - Maximum concurrent connections
- `MCP_REQUEST_TIMEOUT` - Request timeout in seconds
- `MCP_KEEPALIVE_TIMEOUT` - Keep-alive timeout in seconds

### Security Variables
- `MCP_ENABLE_AUTH` - Enable authentication (true/false)
- `MCP_API_KEY_REQUIRED` - Require API key (true/false)
- `MCP_ALLOWED_ORIGINS` - Comma-separated list of allowed origins
- `MCP_RATE_LIMIT_ENABLED` - Enable rate limiting (true/false)
- `MCP_RATE_LIMIT_REQUESTS` - Number of requests per window
- `MCP_RATE_LIMIT_WINDOW` - Rate limit window in seconds
- `MCP_AUDIT_ENABLED` - Enable audit logging (true/false)
- `MCP_AUDIT_LOG_FILE` - Audit log file path
- `MCP_MAX_REQUEST_SIZE` - Maximum request size in bytes
- `MAX_NODES_PER_REQUEST` - Maximum nodes per request
- `MAX_EDGES_PER_REQUEST` - Maximum edges per request
- `SAFE_ID_PATTERN` - Regex pattern for safe identifiers

### Performance Variables
- `MAX_NODES_PER_GRAPH` - Maximum nodes in a graph
- `MAX_EDGES_PER_GRAPH` - Maximum edges in a graph
- `MEMORY_LIMIT_MB` - Memory limit in MB
- `OPERATION_TIMEOUT` - Operation timeout in seconds
- `ENABLE_CACHING` - Enable caching (true/false)
- `CACHE_SIZE_MB` - Cache size in MB
- `CACHE_TTL` - Cache TTL in seconds
- `PARALLEL_PROCESSING` - Enable parallel processing (true/false)
- `USE_CYTHON` - Use Cython optimizations (true/false)
- `NUMPY_OPTIMIZATION` - Use NumPy optimizations (true/false)
- `MAX_MEMORY_MB` - Maximum memory usage in MB
- `MAX_CONCURRENT_REQUESTS` - Maximum concurrent requests

### Storage Variables
- `STORAGE_BACKEND` - Storage backend (auto/redis/memory)
- `REDIS_URL` - Redis connection URL
- `REDIS_HOST` - Redis host
- `REDIS_PORT` - Redis port
- `REDIS_DB` - Redis database number
- `REDIS_PREFIX` - Redis key prefix
- `REDIS_POOL_SIZE` - Redis connection pool size
- `REDIS_TIMEOUT` - Redis connection timeout
- `REDIS_RETRY_ATTEMPTS` - Redis retry attempts
- `REDIS_TTL` - Default Redis TTL
- `COMPRESSION_LEVEL` - Compression level (0-9)
- `MAX_GRAPH_SIZE_MB` - Maximum graph size in MB

### Feature Variables
- `FEATURE_ML` - Enable ML features (true/false)
- `FEATURE_VISUALIZATION` - Enable visualization (true/false)
- `FEATURE_GPU` - Enable GPU acceleration (true/false)
- `FEATURE_ENTERPRISE` - Enable enterprise features (true/false)
- `FEATURE_MONITORING` - Enable monitoring (true/false)
- `METRICS_ENDPOINT` - Metrics endpoint path
- `HEALTH_ENDPOINT` - Health check endpoint path
- `API_V2_ENABLED` - Enable API v2 (true/false)
- `ADVANCED_ALGORITHMS` - Enable advanced algorithms (true/false)
- `CACHING_ENABLED` - Enable caching (true/false)

### Logging Variables
- `LOG_LEVEL` - Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
- `LOG_FORMAT` - Log format string
- `LOG_FILE` - Log file path
- `LOG_MAX_FILE_SIZE` - Maximum log file size in bytes
- `LOG_BACKUP_COUNT` - Number of backup log files
- `LOG_JSON_FORMAT` - Use JSON log format (true/false)

"""
    return doc


def generate_examples_doc() -> str:
    """Generate configuration examples documentation."""
    doc = """## Configuration Examples

### Environment Variable Example
```bash
export MCP_ENVIRONMENT=production
export MCP_HOST=0.0.0.0
export MCP_PORT=8765
export MCP_WORKERS=8
export REDIS_URL=redis://redis-cluster:6379/0
export MCP_ENABLE_AUTH=true
export MCP_RATE_LIMIT_REQUESTS=1000
export LOG_LEVEL=INFO
```

### YAML Configuration Example
```yaml
environment: production
name: networkx-mcp-server
debug: false

server:
  host: 0.0.0.0
  port: 8765
  workers: 8
  max_connections: 1000

security:
  enable_auth: true
  rate_limit_enabled: true
  rate_limit_requests: 1000
  audit_enabled: true

performance:
  memory_limit_mb: 8192
  enable_caching: true
  parallel_processing: true

storage:
  backend: redis
  redis_host: redis-cluster
  compression_level: 6

features:
  machine_learning: true
  visualization: true
  monitoring: true

logging:
  level: INFO
  json_format: true
```

### JSON Configuration Example
```json
{
  "environment": "development",
  "name": "networkx-mcp-server",
  "debug": true,
  "server": {
    "host": "localhost",
    "port": 8765,
    "workers": 2
  },
  "security": {
    "enable_auth": false,
    "rate_limit_enabled": true
  },
  "storage": {
    "backend": "memory"
  },
  "logging": {
    "level": "DEBUG"
  }
}
```

### Docker Environment File (.env)
```bash
# Application
MCP_ENVIRONMENT=production
MCP_NAME=networkx-mcp-server
MCP_VERSION=1.0.0

# Server
MCP_HOST=0.0.0.0
MCP_PORT=8765
MCP_WORKERS=8

# Security
MCP_ENABLE_AUTH=true
MCP_API_KEY_REQUIRED=true
MCP_RATE_LIMIT_REQUESTS=1000

# Storage
REDIS_URL=redis://redis:6379/0

# Logging
LOG_LEVEL=INFO
LOG_JSON_FORMAT=true
```

"""
    return doc


def generate_hot_reload_doc() -> str:
    """Generate hot reload documentation."""
    doc = """## Hot Reload

The configuration system supports hot reloading for certain settings that can be safely changed at runtime without restarting the server.

### Hot Reload Safe Settings

The following settings can be changed at runtime:

- `security.rate_limit_requests` - Rate limit requests per window
- `security.rate_limit_window` - Rate limit window duration
- `security.audit_enabled` - Enable/disable audit logging
- `performance.cache_ttl` - Cache time-to-live
- `performance.enable_caching` - Enable/disable caching
- `features.monitoring` - Enable/disable monitoring
- `logging.level` - Log level
- `logging.json_format` - JSON log format

### How Hot Reload Works

1. **File Watching**: The server watches configuration files for changes
2. **Safe Updates**: Only hot-reload safe settings are applied
3. **Callbacks**: Registered callbacks are notified of changes
4. **Logging**: Configuration changes are logged for audit

### Triggering Hot Reload

Hot reload is triggered automatically when:
- Configuration files are modified
- The `reload_settings()` function is called programmatically

### Example: Changing Log Level at Runtime

1. Edit your configuration file:
   ```yaml
   logging:
     level: DEBUG  # Change from INFO to DEBUG
   ```

2. Save the file - changes are applied automatically

3. Check logs for confirmation:
   ```
   INFO - Hot reloaded setting: logging.level = DEBUG
   ```

### Programmatic Hot Reload

```python
from config import reload_settings

# Reload with hot-reload safety (default)
new_settings = reload_settings(hot_reload_only=True)

# Full reload (requires restart for unsafe settings)
new_settings = reload_settings(hot_reload_only=False)
```

### Configuration Change Callbacks

```python
from config import get_config_manager

def on_config_change(old_settings, new_settings):
    if old_settings.logging.level != new_settings.logging.level:
        print(f"Log level changed: {old_settings.logging.level} -> {new_settings.logging.level}")

manager = get_config_manager()
manager.add_change_callback(on_config_change)
```

### Non-Hot-Reload Settings

Settings that require a full restart:

- Server host/port configuration
- Worker process count
- Storage backend type
- Major feature toggles
- Security authentication settings

Attempting to change these settings will require a server restart to take effect.

"""
    return doc


def export_json_schema() -> Dict[str, Any]:
    """Export the JSON schema for configuration validation."""
    from .validation import CONFIG_SCHEMA
    return CONFIG_SCHEMA


def generate_config_template(environment: str = "development") -> str:
    """Generate a configuration template for the specified environment."""
    
    templates = {
        "development": """# Development Configuration Template
environment: development
name: networkx-mcp-server
version: 1.0.0
debug: true

server:
  host: localhost
  port: 8765
  workers: 2

security:
  enable_auth: false
  rate_limit_enabled: true
  rate_limit_requests: 100

performance:
  memory_limit_mb: 2048
  enable_caching: true

storage:
  backend: memory

features:
  machine_learning: true
  visualization: true
  monitoring: true

logging:
  level: DEBUG
  file: logs/dev.log
""",
        
        "production": """# Production Configuration Template
environment: production
name: networkx-mcp-server
version: 1.0.0
debug: false

server:
  host: 0.0.0.0
  port: 8765
  workers: 8

security:
  enable_auth: true
  api_key_required: true
  rate_limit_enabled: true
  rate_limit_requests: 1000
  audit_enabled: true

performance:
  memory_limit_mb: 8192
  enable_caching: true
  parallel_processing: true

storage:
  backend: redis
  # Set REDIS_URL environment variable

features:
  machine_learning: true
  visualization: true
  enterprise_features: true
  monitoring: true

logging:
  level: INFO
  file: /var/log/networkx-mcp/app.log
  json_format: true
""",
        
        "testing": """# Testing Configuration Template
environment: testing
name: networkx-mcp-server-test
version: 1.0.0
debug: false

server:
  host: localhost
  port: 8766
  workers: 1

security:
  enable_auth: false
  rate_limit_enabled: false

performance:
  memory_limit_mb: 512
  enable_caching: false
  parallel_processing: false

storage:
  backend: memory

features:
  machine_learning: false
  visualization: false
  monitoring: false

logging:
  level: WARNING
  file: null
"""
    }
    
    return templates.get(environment, templates["development"])


if __name__ == "__main__":
    # Generate and save documentation
    docs = generate_schema_documentation()
    with open("CONFIG_SCHEMA.md", "w") as f:
        f.write(docs)
    
    # Export JSON schema
    schema = export_json_schema()
    with open("config-schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    
    print("Configuration documentation generated:")
    print("- CONFIG_SCHEMA.md")
    print("- config-schema.json")