# Configuration

Configure NetworkX MCP Server for your specific needs, from development to enterprise production deployments.

## Configuration Methods

NetworkX MCP Server supports multiple configuration approaches:

<div class="grid cards" markdown>

- :material-cog:{ .lg .middle } __Environment Variables__

    ---

    Simple key-value configuration using environment variables. Perfect for containers and cloud deployments.

    [:octicons-arrow-right-24: Environment Variables](#environment-variables)

- :material-file-code:{ .lg .middle } __Configuration Files__

    ---

    Structured YAML/JSON configuration files for complex setups and version control.

    [:octicons-arrow-right-24: Configuration Files](#configuration-files)

- :material-console:{ .lg .middle } __Command Line Arguments__

    ---

    Override settings directly from the command line for quick testing and debugging.

    [:octicons-arrow-right-24: CLI Arguments](#command-line-arguments)

- :material-api:{ .lg .middle } __Runtime Configuration__

    ---

    Dynamic configuration changes through MCP tools and admin endpoints.

    [:octicons-arrow-right-24: Runtime Config](#runtime-configuration)

</div>

## Environment Variables

### Core Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVER_HOST` | `localhost` | Server bind address |
| `MCP_SERVER_PORT` | `8765` | Server port number |
| `MCP_SERVER_WORKERS` | `4` | Number of worker processes |
| `MCP_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `MCP_LOG_FORMAT` | `text` | Log format (text, json) |

### Performance Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_GRAPH_SIZE` | `1000000` | Maximum nodes per graph |
| `MAX_EDGE_COUNT` | `10000000` | Maximum edges per graph |
| `MAX_MEMORY_MB` | `4096` | Memory limit in MB |
| `ENABLE_CACHING` | `true` | Enable result caching |
| `CACHE_SIZE_MB` | `512` | Cache size limit |
| `CACHE_TTL` | `3600` | Cache TTL in seconds |

### Redis Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `None` | Redis connection URL |
| `REDIS_PREFIX` | `networkx_mcp` | Key prefix for Redis |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_TTL` | `3600` | Default TTL for Redis keys |
| `REDIS_MAX_CONNECTIONS` | `10` | Connection pool size |

### Security Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_AUTH` | `false` | Enable authentication |
| `API_KEY_REQUIRED` | `false` | Require API keys |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |
| `RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `RATE_LIMIT_REQUESTS` | `1000` | Requests per minute |
| `AUDIT_ENABLED` | `false` | Enable audit logging |

### Example .env File

```bash
# Server Configuration
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8765
MCP_SERVER_WORKERS=8
MCP_LOG_LEVEL=INFO

# Performance
MAX_GRAPH_SIZE=5000000
MAX_MEMORY_MB=8192
ENABLE_CACHING=true
CACHE_SIZE_MB=1024

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
REDIS_PREFIX=networkx_prod
REDIS_TTL=7200

# Security
ENABLE_AUTH=true
API_KEY_REQUIRED=true
RATE_LIMIT_REQUESTS=5000
AUDIT_ENABLED=true
```

---

## Configuration Files

### YAML Configuration

Create `config.yaml` for structured configuration:

```yaml
# NetworkX MCP Server Configuration
server:
  host: "0.0.0.0"
  port: 8765
  workers: 4

logging:
  level: "INFO"
  format: "json"
  file: "/var/log/networkx-mcp.log"
  rotate: true
  max_size: "100MB"
  backup_count: 5

performance:
  limits:
    max_nodes: 1000000
    max_edges: 10000000
    memory_mb: 4096
    timeout_seconds: 300

  caching:
    enabled: true
    backend: "redis"  # memory, redis, hybrid
    size_mb: 512
    ttl: 3600

  optimization:
    parallel_processing: true
    use_cython: true
    numpy_optimization: true

redis:
  url: "redis://localhost:6379/0"
  prefix: "networkx_mcp"
  pool_size: 10
  timeout: 5
  retry_attempts: 3

security:
  authentication:
    enabled: true
    method: "api_key"  # api_key, oauth2, jwt
    api_keys:
      - name: "admin"
        key: "your-secure-api-key"
        permissions: ["read", "write", "admin"]
      - name: "readonly"
        key: "readonly-key"
        permissions: ["read"]

  rate_limiting:
    enabled: true
    requests_per_minute: 1000
    burst_limit: 100

  cors:
    enabled: true
    allowed_origins: ["https://example.com"]
    allowed_methods: ["GET", "POST"]

  audit:
    enabled: true
    log_file: "/var/log/networkx-audit.log"
    include_payload: false

features:
  machine_learning:
    enabled: true
    gpu_acceleration: false

  visualization:
    enabled: true
    max_image_size: "10MB"
    formats: ["png", "svg", "html"]

  enterprise:
    monitoring: true
    metrics_endpoint: "/metrics"
    health_endpoint: "/health"
```

### JSON Configuration

Alternative JSON format:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8765,
    "workers": 4
  },
  "logging": {
    "level": "INFO",
    "format": "json"
  },
  "performance": {
    "limits": {
      "max_nodes": 1000000,
      "max_edges": 10000000,
      "memory_mb": 4096
    },
    "caching": {
      "enabled": true,
      "backend": "redis",
      "ttl": 3600
    }
  },
  "security": {
    "authentication": {
      "enabled": true,
      "method": "api_key"
    },
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 1000
    }
  }
}
```

---

## Command Line Arguments

Override configuration with command line arguments:

### Basic Usage

```bash
# Start with custom settings
networkx-mcp-server \
  --host 0.0.0.0 \
  --port 8765 \
  --workers 8 \
  --log-level DEBUG

# Load configuration file
networkx-mcp-server --config config.yaml

# Override specific settings
networkx-mcp-server \
  --config config.yaml \
  --max-memory 8192 \
  --enable-auth
```

### Available Arguments

| Argument | Environment Variable | Description |
|----------|---------------------|-------------|
| `--host` | `MCP_SERVER_HOST` | Server bind address |
| `--port` | `MCP_SERVER_PORT` | Server port |
| `--workers` | `MCP_SERVER_WORKERS` | Worker processes |
| `--config` | `MCP_CONFIG_FILE` | Configuration file path |
| `--log-level` | `MCP_LOG_LEVEL` | Logging level |
| `--max-memory` | `MAX_MEMORY_MB` | Memory limit |
| `--redis-url` | `REDIS_URL` | Redis connection |
| `--enable-auth` | `ENABLE_AUTH` | Enable authentication |
| `--disable-cache` | `ENABLE_CACHING` | Disable caching |

### Help and Version

```bash
# Show help
networkx-mcp-server --help

# Show version
networkx-mcp-server --version

# Validate configuration
networkx-mcp-server --config config.yaml --validate
```

---

## Runtime Configuration

### Configuration Tools

Use MCP tools to change settings at runtime:

```python
# Get current configuration
config = get_server_config()

# Update settings
update_server_config({
    "performance": {
        "max_nodes": 2000000,
        "cache_ttl": 7200
    }
})

# Reload configuration from file
reload_config(config_file="config.yaml")
```

### Admin Endpoints

Access configuration via HTTP endpoints:

```bash
# Get configuration (requires admin permissions)
curl -H "Authorization: Bearer admin-key" \
     http://localhost:8765/admin/config

# Update configuration
curl -X PUT \
     -H "Authorization: Bearer admin-key" \
     -H "Content-Type: application/json" \
     -d '{"performance": {"max_nodes": 2000000}}' \
     http://localhost:8765/admin/config

# Reload from file
curl -X POST \
     -H "Authorization: Bearer admin-key" \
     http://localhost:8765/admin/reload
```

---

## Environment-Specific Configurations

### Development

```yaml
# config/development.yaml
server:
  host: "localhost"
  port: 8765
  workers: 2

logging:
  level: "DEBUG"
  format: "text"

performance:
  limits:
    max_nodes: 100000
  caching:
    enabled: false

security:
  authentication:
    enabled: false
  rate_limiting:
    enabled: false
```

### Staging

```yaml
# config/staging.yaml
server:
  host: "0.0.0.0"
  port: 8765
  workers: 4

logging:
  level: "INFO"
  format: "json"

performance:
  limits:
    max_nodes: 500000
  caching:
    enabled: true
    backend: "memory"

security:
  authentication:
    enabled: true
  rate_limiting:
    enabled: true
    requests_per_minute: 500
```

### Production

```yaml
# config/production.yaml
server:
  host: "0.0.0.0"
  port: 8765
  workers: 8

logging:
  level: "WARNING"
  format: "json"
  file: "/var/log/networkx-mcp.log"

performance:
  limits:
    max_nodes: 10000000
    memory_mb: 16384
  caching:
    enabled: true
    backend: "redis"
    ttl: 3600

redis:
  url: "redis://redis-cluster:6379/0"
  pool_size: 20

security:
  authentication:
    enabled: true
    method: "jwt"
  rate_limiting:
    enabled: true
    requests_per_minute: 5000
  audit:
    enabled: true

features:
  monitoring: true
  metrics_endpoint: "/metrics"
```

---

## Configuration Validation

### Validation Rules

The server validates configuration on startup:

- __Required fields__: Must be present
- __Type checking__: Correct data types
- __Range validation__: Values within acceptable ranges
- __Dependency checks__: Related settings are compatible

### Validation Errors

Common validation errors:

```bash
# Invalid port range
Error: port must be between 1 and 65535

# Missing required Redis URL when caching enabled
Error: redis.url required when caching.backend is 'redis'

# Memory limit too low
Warning: max_memory_mb below recommended minimum (1024MB)

# Conflicting security settings
Error: authentication.enabled requires security.api_keys
```

### Manual Validation

```bash
# Validate configuration file
networkx-mcp-server --config config.yaml --validate --dry-run

# Check specific settings
networkx-mcp-server --validate-redis-connection
networkx-mcp-server --validate-auth-config
```

---

## Configuration Templates

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  networkx-mcp:
    image: networkx-mcp-server:latest
    ports:
      - "8765:8765"
    environment:
      - MCP_SERVER_HOST=0.0.0.0
      - REDIS_URL=redis://redis:6379
      - ENABLE_AUTH=true
      - LOG_LEVEL=INFO
    volumes:
      - ./config.yaml:/app/config.yaml
    depends_on:
      - redis

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### Kubernetes

```yaml
# k8s-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: networkx-mcp-config
data:
  config.yaml: |
    server:
      host: "0.0.0.0"
      port: 8765
    redis:
      url: "redis://redis-service:6379"
    security:
      authentication:
        enabled: true
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: networkx-mcp-server
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: networkx-mcp
        image: networkx-mcp-server:latest
        env:
        - name: MCP_CONFIG_FILE
          value: "/etc/config/config.yaml"
        volumeMounts:
        - name: config
          mountPath: /etc/config
      volumes:
      - name: config
        configMap:
          name: networkx-mcp-config
```

---

## Best Practices

### Security

!!! warning "Security Recommendations"

    - **Always enable authentication** in production
    - **Use strong API keys** with appropriate permissions
    - **Enable audit logging** for compliance
    - **Set appropriate rate limits** to prevent abuse
    - **Use TLS/SSL** for encrypted connections
    - **Regularly rotate API keys** and credentials

### Performance

!!! tip "Performance Tips"

    - **Use Redis caching** for better performance
    - **Set appropriate memory limits** to prevent OOM
    - **Tune worker count** based on CPU cores
    - **Monitor resource usage** regularly
    - **Use connection pooling** for databases
    - **Enable compression** for large responses

### Monitoring

!!! info "Monitoring Setup"

    - **Enable metrics endpoints** for monitoring
    - **Set up health checks** for load balancers
    - **Configure log aggregation** (ELK, Splunk)
    - **Monitor Redis performance** if using caching
    - **Set up alerts** for critical metrics
    - **Track API usage patterns** for optimization

---

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Server won't start | Port already in use | Change port or stop conflicting service |
| High memory usage | Large graphs without limits | Set `MAX_MEMORY_MB` and `MAX_GRAPH_SIZE` |
| Slow performance | No caching enabled | Enable Redis caching |
| Authentication errors | Invalid API keys | Check key format and permissions |
| Connection timeouts | Network issues | Verify network connectivity and firewall |

### Debugging Configuration

```bash
# Enable debug logging
export MCP_LOG_LEVEL=DEBUG

# Dump current configuration
networkx-mcp-server --dump-config

# Test specific features
networkx-mcp-server --test-redis
networkx-mcp-server --test-auth
```

### Configuration Hierarchy

Settings are applied in this order (later overrides earlier):

1. __Default values__ (built into application)
2. __Configuration file__ (YAML/JSON)
3. __Environment variables__
4. __Command line arguments__
5. __Runtime updates__ (via API)

---

## Next Steps

<div class="grid cards" markdown>

- [:material-docker: __Deployment__](enterprise/deployment.md)

    Deploy to production environments

- [:material-shield-check: __Security__](enterprise/security.md)

    Implement security best practices

- [:material-chart-line: __Monitoring__](enterprise/monitoring.md)

    Set up monitoring and alerting

- [:material-cog-box: __Optimization__](enterprise/performance.md)

    Optimize for your workload

</div>
