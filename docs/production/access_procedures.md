# Production Access Procedures

## Overview

This document outlines the procedures for accessing and managing the NetworkX MCP Server in production environments. The server supports dual-mode transport (stdio and HTTP) with comprehensive monitoring and security controls.

## Production Endpoints

### Primary Production Endpoint
- **HTTPS/JSON-RPC**: `https://api.mcp.example.com`
- **Port**: 443 (HTTPS) / 8080 (HTTP redirect)
- **Protocol**: JSON-RPC 2.0 over HTTPS
- **Authentication**: Bearer token required

### Monitoring Endpoints
- **Health Check**: `https://api.mcp.example.com/health`
- **Readiness**: `https://api.mcp.example.com/ready`
- **Metrics**: `https://api.mcp.example.com:9090/metrics` (Prometheus)
- **Startup**: `https://api.mcp.example.com/startup`

### Internal Endpoints (Kubernetes)
- **Service**: `networkx-mcp-service.default.svc.cluster.local:8080`
- **Canary Service**: `networkx-mcp-canary-service.default.svc.cluster.local:8080`
- **Metrics**: `*.default.svc.cluster.local:9090`

## Transport Modes

### 1. HTTP/JSON-RPC Mode (Recommended for Production)

#### Client Connection
```bash
# Using curl for simple requests
curl -X POST https://api.mcp.example.com/jsonrpc \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "create_graph",
      "arguments": {"graph_id": "production_graph"}
    },
    "id": 1
  }'
```

#### Python Client Example
```python
import aiohttp
import asyncio

async def connect_to_mcp():
    headers = {
        "Authorization": "Bearer YOUR_TOKEN",
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "list_graphs",
                "arguments": {}
            },
            "id": 1
        }
        
        async with session.post(
            "https://api.mcp.example.com/jsonrpc",
            json=request
        ) as response:
            result = await response.json()
            print(f"Response: {result}")

# Run the client
asyncio.run(connect_to_mcp())
```

#### JavaScript/Node.js Client Example
```javascript
const axios = require('axios');

async function callMCPServer() {
  try {
    const response = await axios.post('https://api.mcp.example.com/jsonrpc', {
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'create_graph',
        arguments: { graph_id: 'js_graph' }
      },
      id: 1
    }, {
      headers: {
        'Authorization': 'Bearer YOUR_TOKEN',
        'Content-Type': 'application/json'
      }
    });
    
    console.log('MCP Response:', response.data);
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
}

callMCPServer();
```

### 2. Stdio Mode (Development/Local Use)

#### Direct Invocation
```bash
# Start server in stdio mode
python -m networkx_mcp

# Or with specific configuration
python -m networkx_mcp \
  --log-level INFO \
  --storage-backend redis \
  --redis-url redis://localhost:6379
```

#### Using MCP SDK
```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def connect_stdio():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "networkx_mcp"],
        env={"LOG_LEVEL": "DEBUG"}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            result = await session.call_tool(
                "create_graph",
                {"graph_id": "stdio_graph"}
            )
            print(f"Result: {result}")

asyncio.run(connect_stdio())
```

## Authentication & Authorization

### Bearer Token Authentication
Production requires authentication via Bearer tokens in the Authorization header:

```bash
Authorization: Bearer YOUR_TOKEN_HERE
```

### Token Management
1. **Obtain Token**: Contact ops team or use service account
2. **Token Scope**: Limited to specific graph operations
3. **Token Rotation**: Tokens rotate every 90 days
4. **Emergency Access**: Break-glass procedures available

### Service Account Access
```bash
# Example using service account
export MCP_SERVICE_TOKEN=$(kubectl get secret mcp-service-account \
  -o jsonpath='{.data.token}' | base64 -d)

curl -H "Authorization: Bearer $MCP_SERVICE_TOKEN" \
  https://api.mcp.example.com/health
```

## Production Limits & Quotas

### Connection Limits
- **Maximum Concurrent Connections**: 50 users
- **Rate Limiting**: 100 requests/minute per client
- **Timeout**: 30 seconds per request
- **Keep-alive**: 300 seconds

### Graph Limits
- **Maximum Graph Size**: 10,000 nodes
- **Maximum Memory per Graph**: 2GB
- **Maximum Storage**: 50GB per tenant
- **Algorithm Timeout**: 60 seconds

### Performance Expectations
Based on production testing:
- **50 concurrent users**: 95.2% success rate
- **10K node graphs**: ~120MB memory, <2s response time
- **Typical latency**: P95 < 2000ms, P99 < 5000ms

## Monitoring & Observability

### Health Monitoring
```bash
# Check overall health
curl https://api.mcp.example.com/health

# Check readiness
curl https://api.mcp.example.com/ready

# Check startup status
curl https://api.mcp.example.com/startup
```

### Metrics Collection
```bash
# Prometheus metrics
curl https://api.mcp.example.com:9090/metrics

# Key metrics to monitor:
# - mcp_requests_total
# - mcp_request_duration_seconds
# - mcp_active_connections
# - mcp_graph_operations_total
# - process_resident_memory_bytes
```

### Distributed Tracing
- **Jaeger UI**: `https://jaeger.monitoring.example.com`
- **Trace Headers**: X-Trace-ID automatically added
- **Sampling Rate**: 10% for performance

### Log Aggregation
```bash
# View logs via kubectl
kubectl logs -l app=networkx-mcp -n default --tail=100

# Structured logging in JSON format
kubectl logs deployment/networkx-mcp-server --tail=50 | jq '.'

# Follow logs in real-time
kubectl logs -f deployment/networkx-mcp-server
```

## Troubleshooting

### Common Issues

#### 1. Connection Refused
```bash
# Check service status
kubectl get pods -l app=networkx-mcp
kubectl get svc networkx-mcp-service

# Check ingress
kubectl get ingress networkx-mcp-ingress
```

#### 2. Authentication Failures
```bash
# Verify token
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://api.mcp.example.com/health

# Check token expiry
echo $MCP_TOKEN | base64 -d | jq '.exp'
```

#### 3. Performance Issues
```bash
# Check metrics
curl https://api.mcp.example.com:9090/metrics | grep mcp_request_duration

# Check resource usage
kubectl top pods -l app=networkx-mcp
```

#### 4. Memory/Resource Limits
```bash
# Check pod resource usage
kubectl describe pod -l app=networkx-mcp | grep -A 10 "Limits\|Requests"

# Check if hitting limits
kubectl get events --sort-by=.metadata.creationTimestamp | grep OOMKilled
```

### Debug Mode Access

#### Enable Debug Logging
```bash
# Temporarily enable debug logging
kubectl patch deployment networkx-mcp-server -p '{"spec":{"template":{"spec":{"containers":[{"name":"mcp-server","env":[{"name":"LOG_LEVEL","value":"DEBUG"}]}]}}}}'

# Rollback to INFO level
kubectl patch deployment networkx-mcp-server -p '{"spec":{"template":{"spec":{"containers":[{"name":"mcp-server","env":[{"name":"LOG_LEVEL","value":"INFO"}]}]}}}}'
```

#### Port Forwarding for Direct Access
```bash
# Forward health port
kubectl port-forward deployment/networkx-mcp-server 8080:8080

# Forward metrics port
kubectl port-forward deployment/networkx-mcp-server 9090:9090

# Test locally
curl http://localhost:8080/health
curl http://localhost:9090/metrics
```

## Emergency Procedures

### 1. Emergency Rollback
```bash
# Rollback to previous version
kubectl rollout undo deployment/networkx-mcp-server

# Check rollback status
kubectl rollout status deployment/networkx-mcp-server

# Verify health after rollback
curl https://api.mcp.example.com/health
```

### 2. Scale Down (Emergency Stop)
```bash
# Scale to zero replicas
kubectl scale deployment networkx-mcp-server --replicas=0

# Scale back up
kubectl scale deployment networkx-mcp-server --replicas=3
```

### 3. Drain Node (Maintenance)
```bash
# Safely drain a node
kubectl drain NODE_NAME --ignore-daemonsets --delete-emptydir-data

# Uncordon after maintenance
kubectl uncordon NODE_NAME
```

### 4. Database Emergency
```bash
# Check Redis connection
kubectl exec -it deployment/redis -- redis-cli ping

# Backup current data
kubectl exec deployment/redis -- redis-cli --rdb /tmp/backup.rdb

# Clear cache if needed (CAREFUL!)
kubectl exec deployment/redis -- redis-cli FLUSHDB
```

## Maintenance Windows

### Scheduled Maintenance
- **Window**: Sundays 02:00-04:00 UTC
- **Notification**: 24 hours advance notice
- **Process**: Rolling updates with zero downtime
- **Rollback**: Automatic if health checks fail

### Emergency Maintenance
- **Authorization**: On-call engineer + manager approval
- **Documentation**: All changes must be logged
- **Communication**: Immediate notification to stakeholders
- **Validation**: Full test suite after changes

## Access Control

### Production Access Levels

#### 1. Read-Only Access
- View logs and metrics
- Health check endpoints
- Basic kubectl get commands

#### 2. Operator Access
- Deployment management
- Configuration changes
- Scaling operations
- Log analysis

#### 3. Admin Access
- Full cluster access
- Security configurations
- Network policies
- Emergency procedures

### VPN and Network Access
- **VPN Required**: All production access requires VPN
- **IP Allowlist**: Specific IP ranges allowed
- **Audit Logging**: All access attempts logged
- **Session Recording**: Critical operations recorded

## Security Considerations

### Network Security
- **TLS 1.3**: All communications encrypted
- **Certificate Rotation**: Automatic via cert-manager
- **Network Policies**: Kubernetes network policies enforced
- **Firewall Rules**: Ingress/egress restrictions

### Application Security
- **Input Validation**: All inputs validated and sanitized
- **Rate Limiting**: Per-client request rate limits
- **RBAC**: Role-based access control
- **Security Headers**: OWASP recommended headers

### Data Security
- **Encryption at Rest**: Redis data encrypted
- **Encryption in Transit**: All network traffic encrypted
- **Data Retention**: Automatic cleanup of old data
- **Backup Encryption**: All backups encrypted

## Support Contacts

### Escalation Chain
1. **Level 1**: On-call Engineer (`oncall-mcp@example.com`)
2. **Level 2**: Senior SRE (`sre-team@example.com`)
3. **Level 3**: Engineering Manager (`eng-manager@example.com`)

### Communication Channels
- **Slack**: `#mcp-production` channel
- **PagerDuty**: `networkx-mcp-production` service
- **Email**: `mcp-operations@example.com`
- **Emergency**: `+1-555-ONCALL-1`

---

**Last Updated**: 2024-07-07  
**Document Owner**: SRE Team  
**Review Cycle**: Monthly