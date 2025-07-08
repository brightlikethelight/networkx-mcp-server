# NetworkX MCP Server Configuration

## Quick Start

The NetworkX MCP Server can be configured without any code changes using:
1. Environment variables
2. Configuration files (YAML/JSON)
3. Hot reload for runtime changes

## Configuration Methods

### 1. Environment Variables

Set any configuration option using environment variables:

```bash
export MCP_PORT=9000
export LOG_LEVEL=DEBUG
export REDIS_URL=redis://localhost:6379/0
python -m networkx_mcp.server
```

### 2. Configuration Files

Create a `config.yaml` in your project root:

```yaml
server:
  port: 9000
  workers: 8

logging:
  level: DEBUG

storage:
  backend: redis
```

Or use JSON format (`config.json`):

```json
{
  "server": {
    "port": 9000,
    "workers": 8
  },
  "logging": {
    "level": "DEBUG"
  }
}
```

### 3. Environment-Specific Configuration

The server automatically loads configuration based on the `MCP_ENVIRONMENT` variable:

```bash
export MCP_ENVIRONMENT=production
# Loads: config.production.yaml (or .json)
```

## Configuration Priority

Settings are loaded in this order (highest priority first):
1. Environment variables
2. Configuration files
3. Default values

## Hot Reload

Change these settings at runtime without restarting:
- `security.rate_limit_requests`
- `security.rate_limit_window`
- `security.audit_enabled`
- `performance.cache_ttl`
- `performance.enable_caching`
- `features.monitoring`
- `logging.level`
- `logging.json_format`

To enable hot reload, install watchdog:
```bash
pip install watchdog
```

## Examples

### Development Setup

```yaml
# config.development.yaml
environment: development
debug: true

server:
  host: localhost
  port: 8765

storage:
  backend: memory

logging:
  level: DEBUG
```

### Production Setup

```yaml
# config.production.yaml
environment: production
debug: false

server:
  host: 0.0.0.0
  port: 8765
  workers: 8

security:
  enable_auth: true
  rate_limit_enabled: true
  audit_enabled: true

storage:
  backend: redis

logging:
  level: INFO
  json_format: true
```

### Docker Setup

```bash
# .env file
MCP_ENVIRONMENT=production
MCP_PORT=8765
REDIS_URL=redis://redis:6379/0
LOG_LEVEL=INFO
```

## Validation

The configuration system validates all settings on startup:
- Port numbers (1-65535)
- Memory limits (minimum 64MB)
- Environment names (development/testing/staging/production)
- Cross-field dependencies

## Using Configuration in Code

```python
from config import get_settings

settings = get_settings()
print(f"Server running on {settings.server.host}:{settings.server.port}")
```

## Files in This Directory

- `settings.py` - Main configuration module
- `validation.py` - Configuration validation logic
- `schema.py` - Schema documentation generator
- `examples/` - Example configuration files
  - `development.yaml` - Development environment
  - `production.yaml` - Production environment
  - `testing.yaml` - Testing environment
- `.env.example` - Environment variable template

## See Also

- [CONFIG_SCHEMA.md](../CONFIG_SCHEMA.md) - Complete configuration reference
- [examples/](examples/) - Example configuration files