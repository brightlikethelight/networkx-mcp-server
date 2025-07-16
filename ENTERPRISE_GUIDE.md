# 🏢 NetworkX MCP Server - Enterprise Edition

**Production-ready graph analysis with enterprise security, monitoring, and scale.**

## 🚀 What is the Enterprise Edition?

The NetworkX MCP Server Enterprise Edition extends the community version with production-grade features designed for enterprise deployment:

- **🔐 Enterprise Security**: OAuth 2.1, API key authentication, RBAC
- **⚡ Rate Limiting**: Per-user and per-operation quotas
- **📊 Monitoring**: Prometheus metrics, structured audit logging
- **🛡️ Input Validation**: Comprehensive security validation
- **📈 Resource Control**: Memory and execution time limits
- **🔄 Production Ready**: Health checks, graceful shutdown, configuration management

## 📋 Quick Start

### Installation

```bash
# Install with enterprise features
pip install networkx-mcp-server[enterprise]
```

### Basic Configuration

Create a configuration file `enterprise_config.json`:

```json
{
  "enterprise_mode": true,
  "security": {
    "api_key_enabled": true,
    "api_keys": ["your-secure-api-key-here"],
    "rbac_enabled": true,
    "admin_users": ["admin_user_id"]
  },
  "rate_limit": {
    "enabled": true,
    "requests_per_minute": 60,
    "requests_per_hour": 1000,
    "max_graph_size": 100000,
    "max_memory_mb": 512
  },
  "monitoring": {
    "metrics_enabled": true,
    "audit_enabled": true,
    "log_level": "INFO"
  }
}
```

### Running the Enterprise Server

```bash
# Start enterprise server
networkx-mcp-enterprise

# Or with configuration file
NETWORKX_MCP_CONFIG_FILE=enterprise_config.json networkx-mcp-enterprise
```

### Claude Desktop Integration

Update your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "networkx-enterprise": {
      "command": "networkx-mcp-enterprise",
      "args": [],
      "env": {
        "NETWORKX_MCP_SECURITY_API_KEYS": "your-secure-api-key-here",
        "NETWORKX_MCP_RATE_LIMIT_ENABLED": "true",
        "NETWORKX_MCP_MONITORING_METRICS_ENABLED": "true"
      }
    }
  }
}
```

## 🔐 Security & Authentication

### API Key Authentication

The simplest way to secure your server:

```bash
# Set API key via environment variable
export NETWORKX_MCP_SECURITY_API_KEYS="key1,key2,key3"
export NETWORKX_MCP_SECURITY_ADMIN_USERS="api_user_0"  # First key is admin

# Start server
networkx-mcp-enterprise
```

Clients must include the API key in headers:
```
X-API-Key: your-secure-api-key-here
```

### OAuth 2.1 with PKCE

For enterprise SSO integration:

```json
{
  "security": {
    "oauth_enabled": true,
    "oauth_client_id": "your-oauth-client-id",
    "oauth_client_secret": "your-oauth-client-secret",
    "oauth_auth_url": "https://your-provider.com/oauth/authorize",
    "oauth_token_url": "https://your-provider.com/oauth/token",
    "oauth_scopes": ["graph:read", "graph:write"]
  }
}
```

### Role-Based Access Control (RBAC)

Users are automatically assigned roles based on their authentication:

- **Admin**: Full access to all operations
- **User**: Standard graph operations (create, analyze, visualize)
- **Readonly**: Read and visualization only
- **Guest**: Basic read access

Customize permissions by configuring admin users:

```json
{
  "security": {
    "admin_users": ["admin@company.com", "api_user_0"]
  }
}
```

## ⚡ Rate Limiting & Resource Control

### Request Rate Limits

Protect against abuse with configurable rate limits:

```json
{
  "rate_limit": {
    "enabled": true,
    "requests_per_minute": 60,    // Token bucket with burst handling
    "requests_per_hour": 1000,    // Sliding window limit
    "burst_size": 10,             // Burst capacity
    
    // Operation-specific limits
    "visualize_graph": 10,        // Heavy operations get lower limits
    "pagerank": 20,
    "community_detection": 15
  }
}
```

### Resource Quotas

Prevent resource exhaustion:

```json
{
  "rate_limit": {
    "max_graph_size": 100000,     // Maximum nodes per graph
    "max_memory_mb": 512,         // Memory limit per user
    "max_execution_time": 30      // Timeout for operations (seconds)
  }
}
```

### Rate Limit Headers

Clients receive rate limit information in responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642531200
```

## 📊 Monitoring & Observability

### Prometheus Metrics

Enterprise server exposes comprehensive metrics:

```bash
# Access metrics endpoint
curl http://localhost:8090/metrics
```

Key metrics:
- `networkx_mcp_requests_total` - Request count by operation and status
- `networkx_mcp_request_duration_seconds` - Request latency
- `networkx_mcp_auth_attempts_total` - Authentication attempts
- `networkx_mcp_rate_limit_hits_total` - Rate limit violations
- `networkx_mcp_memory_usage_bytes` - Memory usage
- `networkx_mcp_graphs_total` - Number of graphs in memory

### Structured Audit Logging

All operations are logged with correlation IDs:

```json
{
  "timestamp": "2025-07-15T20:30:45.123Z",
  "event_type": "operation",
  "user_id": "user123",
  "operation": "create_graph",
  "resource": "social_network",
  "success": true,
  "duration_ms": 150.5,
  "request_id": "req-abc-123",
  "ip_address": "192.168.1.100",
  "user_agent": "Claude Desktop/1.0"
}
```

### Health Checks

Monitor server health:

```bash
# Health check endpoint
curl http://localhost:8091/health

# Ready check (dependencies)
curl http://localhost:8091/ready
```

### Log Configuration

Configure logging format and level:

```json
{
  "monitoring": {
    "log_level": "INFO",           // DEBUG, INFO, WARNING, ERROR
    "log_format": "json",          // json or text
    "audit_enabled": true,
    "audit_log_file": "/var/log/networkx-mcp/audit.log"
  }
}
```

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install enterprise edition
RUN pip install networkx-mcp-server[enterprise]

# Copy configuration
COPY enterprise_config.json /app/config.json

# Set environment
ENV NETWORKX_MCP_CONFIG_FILE=/app/config.json
ENV NETWORKX_MCP_MONITORING_METRICS_PORT=8090
ENV NETWORKX_MCP_MONITORING_HEALTH_CHECK_PORT=8091

# Expose ports
EXPOSE 8080 8090 8091

# Run server
CMD ["networkx-mcp-enterprise"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: networkx-mcp-enterprise
spec:
  replicas: 3
  selector:
    matchLabels:
      app: networkx-mcp-enterprise
  template:
    metadata:
      labels:
        app: networkx-mcp-enterprise
    spec:
      containers:
      - name: networkx-mcp
        image: networkx-mcp-enterprise:1.1.0
        ports:
        - containerPort: 8080
        - containerPort: 8090  # Metrics
        - containerPort: 8091  # Health checks
        env:
        - name: NETWORKX_MCP_SECURITY_API_KEYS
          valueFrom:
            secretKeyRef:
              name: networkx-mcp-secrets
              key: api-keys
        livenessProbe:
          httpGet:
            path: /health
            port: 8091
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8091
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: networkx-mcp-service
spec:
  selector:
    app: networkx-mcp-enterprise
  ports:
  - name: mcp
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 8090
    targetPort: 8090
```

### Monitoring Stack

```yaml
# Prometheus scraping configuration
- job_name: 'networkx-mcp'
  static_configs:
  - targets: ['networkx-mcp-service:8090']
  metrics_path: /metrics
  scrape_interval: 15s
```

## 🔧 Configuration Reference

### Environment Variables

All configuration can be set via environment variables:

```bash
# Security
NETWORKX_MCP_SECURITY_API_KEY_ENABLED=true
NETWORKX_MCP_SECURITY_API_KEYS="key1,key2,key3"
NETWORKX_MCP_SECURITY_ADMIN_USERS="admin@company.com"
NETWORKX_MCP_SECURITY_RBAC_ENABLED=true

# OAuth
NETWORKX_MCP_SECURITY_OAUTH_ENABLED=true
NETWORKX_MCP_SECURITY_OAUTH_CLIENT_ID="your-client-id"
NETWORKX_MCP_SECURITY_OAUTH_CLIENT_SECRET="your-client-secret"

# Rate Limiting
NETWORKX_MCP_RATE_LIMIT_ENABLED=true
NETWORKX_MCP_RATE_LIMIT_REQUESTS_PER_MINUTE=60
NETWORKX_MCP_RATE_LIMIT_REQUESTS_PER_HOUR=1000
NETWORKX_MCP_RATE_LIMIT_MAX_GRAPH_SIZE=100000
NETWORKX_MCP_RATE_LIMIT_MAX_MEMORY_MB=512

# Monitoring
NETWORKX_MCP_MONITORING_METRICS_ENABLED=true
NETWORKX_MCP_MONITORING_METRICS_PORT=8090
NETWORKX_MCP_MONITORING_AUDIT_ENABLED=true
NETWORKX_MCP_MONITORING_LOG_LEVEL=INFO

# Server
NETWORKX_MCP_SERVER_HOST=localhost
NETWORKX_MCP_SERVER_PORT=8080
```

### Configuration File Format

```json
{
  "enterprise_mode": true,
  "development_mode": false,
  
  "security": {
    "api_key_enabled": true,
    "api_keys": ["secure-key-1", "secure-key-2"],
    "api_key_header": "X-API-Key",
    
    "oauth_enabled": false,
    "oauth_client_id": "",
    "oauth_client_secret": "",
    "oauth_auth_url": "",
    "oauth_token_url": "",
    "oauth_scopes": ["graph:read", "graph:write"],
    
    "jwt_secret": "auto-generated-if-empty",
    "jwt_algorithm": "HS256",
    "jwt_expiry_hours": 24,
    
    "rbac_enabled": true,
    "admin_users": []
  },
  
  "rate_limit": {
    "enabled": true,
    "requests_per_minute": 60,
    "requests_per_hour": 1000,
    "burst_size": 10,
    "max_graph_size": 100000,
    "max_memory_mb": 512,
    "max_execution_time": 30
  },
  
  "monitoring": {
    "metrics_enabled": true,
    "metrics_port": 8090,
    "metrics_path": "/metrics",
    
    "log_level": "INFO",
    "log_format": "json",
    "audit_enabled": true,
    "audit_log_file": null,
    
    "health_check_enabled": true,
    "health_check_port": 8091
  },
  
  "server": {
    "host": "localhost",
    "port": 8080,
    "workers": 1,
    "transport": "stdio",
    "cors_enabled": true,
    "cors_origins": ["*"],
    "async_enabled": true,
    "connection_pool_size": 100,
    "request_timeout": 60
  }
}
```

## 🚨 Security Best Practices

### 1. Secure API Keys

```bash
# Generate secure API keys
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Store in secure secrets management
export NETWORKX_MCP_SECURITY_API_KEYS="$(cat /etc/secrets/networkx-api-keys)"
```

### 2. Network Security

- Deploy behind reverse proxy (nginx, HAProxy)
- Use TLS/SSL for all connections
- Restrict network access to authorized clients
- Enable CORS only for trusted origins

### 3. Resource Limits

```json
{
  "rate_limit": {
    "max_graph_size": 10000,        // Limit for shared environments
    "max_memory_mb": 256,           // Conservative memory limit
    "max_execution_time": 10        // Shorter timeout for responsiveness
  }
}
```

### 4. Monitoring & Alerting

Set up alerts for:
- High error rates (`networkx_mcp_errors_total`)
- Authentication failures (`networkx_mcp_auth_attempts_total{result="failure"}`)
- Rate limit violations (`networkx_mcp_rate_limit_hits_total`)
- Memory usage spikes (`networkx_mcp_memory_usage_bytes`)

## 🔄 Migration from Community Edition

### 1. Update Installation

```bash
# Upgrade to enterprise edition
pip install --upgrade networkx-mcp-server[enterprise]
```

### 2. Add Authentication

```bash
# Generate API key
export NETWORKX_MCP_SECURITY_API_KEYS="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

# Update Claude Desktop config
# Add API key to environment variables
```

### 3. Enable Monitoring

```bash
# Enable metrics and audit logging
export NETWORKX_MCP_MONITORING_METRICS_ENABLED=true
export NETWORKX_MCP_MONITORING_AUDIT_ENABLED=true

# Start with enterprise server
networkx-mcp-enterprise
```

### 4. Gradual Rollout

1. **Phase 1**: Enable authentication with permissive rate limits
2. **Phase 2**: Tighten rate limits based on usage patterns
3. **Phase 3**: Enable full monitoring and alerting
4. **Phase 4**: Optimize resource limits for your workload

## 📚 Advanced Topics

### Custom Authentication Providers

Extend the authentication system:

```python
from networkx_mcp.enterprise.auth import AuthenticationManager

class CustomAuthProvider:
    def authenticate(self, token: str) -> Optional[User]:
        # Implement custom authentication logic
        pass

# Register custom provider
auth_manager.register_provider(CustomAuthProvider())
```

### Custom Rate Limiting

Implement domain-specific rate limiting:

```python
from networkx_mcp.enterprise.rate_limiting import RateLimiter

class GraphSizeRateLimiter:
    def check_limit(self, user_id: str, graph_size: int) -> bool:
        # Implement graph-size-based limiting
        pass
```

### Integration with Enterprise Tools

- **LDAP/Active Directory**: Custom OAuth provider
- **HashiCorp Vault**: Secure secret management
- **Prometheus/Grafana**: Monitoring dashboards
- **ELK Stack**: Centralized logging
- **Kubernetes**: Horizontal pod autoscaling

## 🆘 Troubleshooting

### Common Issues

**Authentication Failures**
```bash
# Check API key configuration
echo $NETWORKX_MCP_SECURITY_API_KEYS

# Verify user permissions
curl -H "X-API-Key: your-key" http://localhost:8080/user/info
```

**Rate Limiting**
```bash
# Check current limits
curl -H "X-API-Key: your-key" http://localhost:8080/user/limits

# Reset user limits (admin only)
curl -X POST -H "X-API-Key: admin-key" http://localhost:8080/admin/reset-limits/user123
```

**Performance Issues**
```bash
# Check metrics
curl http://localhost:8090/metrics | grep networkx_mcp_memory

# Analyze slow operations
grep "duration_ms" /var/log/networkx-mcp/audit.log | sort -k4 -nr | head -10
```

### Debug Mode

Enable debug logging:

```bash
export NETWORKX_MCP_MONITORING_LOG_LEVEL=DEBUG
export NETWORKX_MCP_DEVELOPMENT_MODE=true
networkx-mcp-enterprise
```

### Health Checks

Monitor server health:

```bash
# Overall health
curl http://localhost:8091/health

# Component status
curl http://localhost:8091/health/detailed
```

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/Bright-L01/networkx-mcp-server/wiki)
- **Issues**: [GitHub Issues](https://github.com/Bright-L01/networkx-mcp-server/issues)
- **Security**: Report security issues to security@networkx-mcp.com
- **Enterprise Support**: Contact enterprise@networkx-mcp.com

---

**Ready to deploy enterprise-grade graph analysis?**

```bash
pip install networkx-mcp-server[enterprise]
networkx-mcp-enterprise
```