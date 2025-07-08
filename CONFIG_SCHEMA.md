# NetworkX MCP Server Configuration Schema

## Overview

The NetworkX MCP Server uses a comprehensive configuration system that supports:
- Environment variables (highest priority)
- Configuration files in YAML/JSON format
- Default values (lowest priority)

## Configuration Structure

The configuration is organized into the following sections:

### Application Configuration

**Settings**

Main settings container for the entire application.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `environment` | str | "development" | Application environment (development/testing/staging/production) |
| `name` | str | "networkx-mcp-server" | Application name |
| `version` | str | "1.0.0" | Application version |
| `debug` | bool | false | Enable debug mode |

### Server Configuration

**ServerConfig**

Server network and connection settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | str | "localhost" | Server host address |
| `port` | int | 8765 | Server port number |
| `workers` | int | 4 | Number of worker processes |
| `max_connections` | int | 1000 | Maximum concurrent connections |
| `request_timeout` | int | 30 | Request timeout in seconds |
| `keepalive_timeout` | int | 5 | Keep-alive timeout in seconds |
| `debug` | bool | false | Enable server debug mode |

### Security Configuration

**SecurityConfig**

Security and authentication settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_auth` | bool | false | Enable authentication |
| `api_key_required` | bool | false | Require API key for requests |
| `allowed_origins` | List[str] | ["*"] | CORS allowed origins |
| `rate_limit_enabled` | bool | true | Enable rate limiting |
| `rate_limit_requests` | int | 1000 | Number of requests allowed per window |
| `rate_limit_window` | int | 60 | Rate limit window in seconds |
| `audit_enabled` | bool | false | Enable audit logging |
| `audit_log_file` | str | "audit.log" | Audit log file path |
| `max_request_size` | int | 10485760 | Maximum request size in bytes |
| `max_nodes_per_request` | int | 1000 | Maximum nodes per request |
| `max_edges_per_request` | int | 10000 | Maximum edges per request |
| `safe_id_pattern` | str | "^[a-zA-Z0-9_-]{1,100}$" | Regex pattern for safe identifiers |

### Performance Configuration

**PerformanceConfig**

Performance and resource management settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_nodes` | int | 1000000 | Maximum nodes in a graph |
| `max_edges` | int | 10000000 | Maximum edges in a graph |
| `memory_limit_mb` | int | 4096 | Memory limit in MB |
| `timeout_seconds` | int | 300 | Operation timeout in seconds |
| `enable_caching` | bool | true | Enable caching |
| `cache_size_mb` | int | 512 | Cache size in MB |
| `cache_ttl` | int | 3600 | Cache TTL in seconds |
| `parallel_processing` | bool | true | Enable parallel processing |
| `use_cython` | bool | true | Use Cython optimizations |
| `numpy_optimization` | bool | true | Use NumPy optimizations |
| `max_memory_mb` | int | 1024 | Maximum memory usage in MB |
| `operation_timeout` | int | 30 | Individual operation timeout |
| `max_concurrent_requests` | int | 10 | Maximum concurrent requests |

### Storage Configuration

**StorageConfig**

Storage backend configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | str | "auto" | Storage backend type (auto/redis/memory) |
| `redis_url` | str | null | Redis connection URL |
| `redis_host` | str | "localhost" | Redis host |
| `redis_port` | int | 6379 | Redis port |
| `redis_db` | int | 0 | Redis database number |
| `redis_prefix` | str | "networkx_mcp" | Redis key prefix |
| `redis_pool_size` | int | 10 | Redis connection pool size |
| `redis_timeout` | int | 5 | Redis connection timeout |
| `redis_retry_attempts` | int | 3 | Redis retry attempts |
| `redis_ttl` | int | 3600 | Default Redis TTL |
| `compression_level` | int | 6 | Compression level (0-9) |
| `max_graph_size_mb` | int | 100 | Maximum graph size in MB |

### Features Configuration

**FeaturesConfig**

Feature flags and optional components.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `machine_learning` | bool | true | Enable ML features |
| `visualization` | bool | true | Enable visualization |
| `gpu_acceleration` | bool | false | Enable GPU acceleration |
| `enterprise_features` | bool | false | Enable enterprise features |
| `monitoring` | bool | true | Enable monitoring |
| `metrics_endpoint` | str | "/metrics" | Metrics endpoint path |
| `health_endpoint` | str | "/health" | Health check endpoint path |
| `api_v2_enabled` | bool | true | Enable API v2 |
| `advanced_algorithms` | bool | true | Enable advanced algorithms |
| `caching_enabled` | bool | true | Enable caching |

### Logging Configuration

**LoggingConfig**

Logging configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `level` | str | "INFO" | Log level |
| `format` | str | "%(asctime)s - %(name)s - %(levelname)s - %(message)s" | Log format string |
| `file` | str | null | Log file path |
| `max_file_size` | int | 10485760 | Maximum log file size |
| `backup_count` | int | 5 | Number of backup log files |
| `json_format` | bool | false | Use JSON log format |

## Environment Variables

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

## Configuration Examples

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

## Hot Reload

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

1. **File Watching**: The server watches configuration files for changes (requires watchdog library)
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

## Configuration Priority

When loading configuration, the system follows this priority order (highest to lowest):

1. **Environment Variables** - Always take precedence
2. **Configuration Files** - Override defaults
3. **Default Values** - Built-in fallbacks

## Configuration Validation

The configuration system performs comprehensive validation on startup:

### Schema Validation
- Type checking for all fields
- Range validation for numeric values
- Pattern matching for string fields
- Required field verification

### Business Rule Validation
- Cross-field dependencies
- Environment-specific requirements
- Resource limit consistency
- Security policy enforcement

### Validation Examples

```python
from config import get_settings

try:
    settings = get_settings()
except ValueError as e:
    print(f"Configuration error: {e}")
```

Common validation errors:
- Invalid port numbers (must be 1-65535)
- Memory limits too low (minimum 64MB)
- Invalid regex patterns
- Conflicting feature flags
- Production environment violations

## Using Configuration in Code

```python
from config import get_settings

# Get current settings
settings = get_settings()

# Access configuration values
host = settings.server.host
port = settings.server.port
redis_url = settings.storage.redis_url

# Check feature flags
if settings.features.machine_learning:
    # Enable ML features
    pass

# Use security settings
if settings.security.rate_limit_enabled:
    rate_limit = settings.security.rate_limit_requests
```

## Best Practices

1. **Environment Variables for Secrets**: Never store sensitive data in configuration files
2. **Environment-Specific Files**: Use separate configs for dev/staging/production
3. **Validation on Startup**: Always validate configuration before starting the server
4. **Hot Reload Judiciously**: Only use for settings that are safe to change
5. **Document Changes**: Keep configuration documentation up to date
6. **Version Control**: Track configuration templates but not actual configs with secrets

## Troubleshooting

### Configuration Not Loading
- Check file paths and permissions
- Verify YAML/JSON syntax
- Look for validation errors in logs

### Environment Variables Not Working
- Ensure variables are exported
- Check for typos in variable names
- Verify boolean values are "true"/"false"

### Hot Reload Not Working
- Install watchdog: `pip install watchdog`
- Check file system permissions
- Verify settings are hot-reload safe

### Validation Errors
- Read error messages carefully
- Check for conflicting settings
- Verify environment requirements