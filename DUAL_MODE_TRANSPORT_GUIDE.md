# Dual-Mode Transport Guide - NetworkX MCP Server

## 🚀 Overview

The NetworkX MCP Server supports dual-mode transport, allowing it to run in both **local (stdio)** and **remote (HTTP)** modes with the same JSON-RPC protocol and tools.

### ✅ Supported Transport Modes

| Mode | Transport | Use Case | Authentication | Session Management |
|------|-----------|----------|----------------|-------------------|
| **Local** | stdio | MCP clients, Claude Desktop | Token-based | Per-process |
| **Remote** | HTTP/SSE | Web apps, remote clients | OAuth 2.0 + CORS | Session-based |

---

## 🔧 Usage

### Local Mode (stdio)
```bash
# Default - stdio transport for MCP clients
python -m networkx_mcp

# Explicit stdio mode
python -m networkx_mcp --jsonrpc

# For Claude Desktop configuration
python -m networkx_mcp --jsonrpc
```

### Remote Mode (HTTP)
```bash
# HTTP server with authentication (production)
python -m networkx_mcp --http --port 3000

# HTTP server without authentication (development)
python -m networkx_mcp --http --port 3000 --no-auth

# Custom host and port
python -m networkx_mcp --http --host 0.0.0.0 --port 8080
```

---

## 📡 HTTP Mode Features

### Endpoints

| Endpoint | Method | Purpose | Authentication |
|----------|---------|---------|----------------|
| `/mcp/session` | POST | Create MCP session | Required |
| `/mcp` | POST | JSON-RPC requests | Required + Session |
| `/mcp` | GET | SSE connection | Required + Session |
| `/health` | GET | Health check | None |
| `/info` | GET | Server information | None |

### Session Management
```javascript
// 1. Create session
const sessionResp = await fetch('http://localhost:3000/mcp/session', {
    method: 'POST',
    headers: {
        'Authorization': 'Bearer your-token',
        'Content-Type': 'application/json'
    }
});
const { session_id } = await sessionResp.json();

// 2. Use session for requests
const response = await fetch('http://localhost:3000/mcp', {
    method: 'POST',
    headers: {
        'X-Session-ID': session_id,
        'Authorization': 'Bearer your-token',
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/list'
    })
});
```

### Server-Sent Events (SSE)
```javascript
// Connect to SSE stream
const eventSource = new EventSource(
    `http://localhost:3000/mcp?session=${session_id}`,
    {
        headers: {
            'Authorization': 'Bearer your-token',
            'X-Session-ID': session_id
        }
    }
);

eventSource.addEventListener('heartbeat', (event) => {
    console.log('Heartbeat:', JSON.parse(event.data));
});

eventSource.addEventListener('connected', (event) => {
    console.log('Connected:', JSON.parse(event.data));
});
```

---

## 🔐 Authentication

### OAuth 2.0 Configuration
```bash
# Environment variables for OAuth
export OAUTH_PROVIDER_URL="https://your-oauth-provider.com"
export OAUTH_CLIENT_ID="your-client-id"
export OAUTH_CLIENT_SECRET="your-client-secret" 
export OAUTH_JWKS_URL="https://your-oauth-provider.com/.well-known/jwks.json"
```

### Development Tokens
For development, these tokens are accepted:
- `dev-token-123` - Development user with full access
- `test-token-456` - Test user with read/write access
- Production `AUTH_TOKEN` from config - Admin access

### Required Scopes
- `mcp:read` - Read access to MCP operations
- `mcp:write` - Write access to graph operations

---

## 🛡️ Security Features

### CORS Protection
```python
# Allowed origins (configurable)
allowed_origins = {
    "http://localhost:3000",
    "https://your-app.com"
}
```

### DNS Rebinding Protection
- Origin header validation
- Configurable allowed origins
- Automatic localhost allowance in development

### Rate Limiting
- Session-based rate limiting
- Configurable request limits
- Connection limits based on testing

### Session Security
- Session timeout (1 hour default)
- Automatic cleanup of expired sessions
- Session-based request tracking

---

## 📊 Monitoring & Health Checks

### Health Check Response
```json
{
  "status": "healthy",
  "transport": "http",
  "sessions": 5,
  "sse_connections": 3,
  "timestamp": 1703123456.789
}
```

### Server Info Response
```json
{
  "name": "NetworkX MCP Server",
  "version": "1.0.0",
  "transport": "http",
  "protocol_version": "2024-11-05",
  "capabilities": {
    "tools": true,
    "resources": false,
    "prompts": false
  },
  "limits": {
    "max_sessions": 45,
    "session_timeout": 3600,
    "max_graph_nodes": 10000
  }
}
```

---

## 🧪 Testing

### Run Dual-Mode Tests
```bash
# Test both stdio and HTTP modes
python test_dual_mode_server.py
```

### Manual Testing

#### Stdio Mode
```bash
# Start server
python -m networkx_mcp --jsonrpc

# Test with echo (in another terminal)
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | python -m networkx_mcp --jsonrpc
```

#### HTTP Mode
```bash
# Start server
python -m networkx_mcp --http --port 3000 --no-auth

# Test health check
curl http://localhost:3000/health

# Create session  
curl -X POST http://localhost:3000/mcp/session

# Test JSON-RPC (replace SESSION_ID)
curl -X POST http://localhost:3000/mcp \
  -H "X-Session-ID: SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

---

## ⚙️ Configuration

### Environment Variables
```bash
# Server configuration
export ENVIRONMENT="production"
export LOG_LEVEL="INFO"
export LOG_FORMAT="json"

# Transport limits (based on testing)
export MAX_CONCURRENT_CONNECTIONS="45"
export MAX_GRAPH_SIZE_NODES="10000"
export REQUEST_TIMEOUT="20"

# Authentication
export AUTH_TOKEN="your-production-token"
export ENABLE_AUTH="true"

# OAuth (optional)
export OAUTH_PROVIDER_URL="https://oauth.example.com"
export OAUTH_CLIENT_ID="your-client-id"
export OAUTH_CLIENT_SECRET="your-client-secret"
```

### Production Deployment
```yaml
# Docker Compose example
version: '3.8'
services:
  networkx-mcp-http:
    build: .
    command: ["python", "-m", "networkx_mcp", "--http", "--port", "3000"]
    ports:
      - "3000:3000"
    environment:
      - ENVIRONMENT=production
      - ENABLE_AUTH=true
      - AUTH_TOKEN=${AUTH_TOKEN}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## 🔄 Client Examples

### Python Client (HTTP)
```python
import aiohttp
import asyncio

async def mcp_client_example():
    async with aiohttp.ClientSession() as session:
        # Create MCP session
        async with session.post(
            'http://localhost:3000/mcp/session',
            headers={'Authorization': 'Bearer dev-token-123'}
        ) as resp:
            session_data = await resp.json()
            session_id = session_data['session_id']
        
        # Initialize MCP
        async with session.post(
            'http://localhost:3000/mcp',
            headers={
                'X-Session-ID': session_id,
                'Authorization': 'Bearer dev-token-123'
            },
            json={
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'initialize',
                'params': {
                    'protocolVersion': '2024-11-05',
                    'capabilities': {},
                    'clientInfo': {'name': 'python-client', 'version': '1.0'}
                }
            }
        ) as resp:
            init_result = await resp.json()
            print(f"Initialized: {init_result}")
        
        # Call tools
        async with session.post(
            'http://localhost:3000/mcp',
            headers={
                'X-Session-ID': session_id,
                'Authorization': 'Bearer dev-token-123'
            },
            json={
                'jsonrpc': '2.0',
                'id': 2,
                'method': 'tools/call',
                'params': {
                    'name': 'create_graph',
                    'arguments': {'name': 'test_graph', 'graph_type': 'undirected'}
                }
            }
        ) as resp:
            result = await resp.json()
            print(f"Graph created: {result}")

# Run the example
asyncio.run(mcp_client_example())
```

### JavaScript Client (HTTP)
```javascript
class MCPHttpClient {
    constructor(baseUrl, authToken) {
        this.baseUrl = baseUrl;
        this.authToken = authToken;
        this.sessionId = null;
    }
    
    async createSession() {
        const response = await fetch(`${this.baseUrl}/mcp/session`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.authToken}`,
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        this.sessionId = data.session_id;
        return this.sessionId;
    }
    
    async call(method, params = {}, id = null) {
        if (!this.sessionId) {
            throw new Error('Session not created');
        }
        
        const response = await fetch(`${this.baseUrl}/mcp`, {
            method: 'POST',
            headers: {
                'X-Session-ID': this.sessionId,
                'Authorization': `Bearer ${this.authToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                jsonrpc: '2.0',
                id: id || Date.now(),
                method,
                params
            })
        });
        
        return await response.json();
    }
    
    async initialize() {
        return await this.call('initialize', {
            protocolVersion: '2024-11-05',
            capabilities: {},
            clientInfo: { name: 'js-client', version: '1.0' }
        });
    }
    
    async listTools() {
        return await this.call('tools/list');
    }
    
    async callTool(name, arguments) {
        return await this.call('tools/call', { name, arguments });
    }
}

// Usage example
async function example() {
    const client = new MCPHttpClient('http://localhost:3000', 'dev-token-123');
    
    await client.createSession();
    await client.initialize();
    
    const tools = await client.listTools();
    console.log('Available tools:', tools.result.tools.map(t => t.name));
    
    const result = await client.callTool('create_graph', {
        name: 'my_graph',
        graph_type: 'undirected'
    });
    console.log('Graph created:', result);
}
```

---

## 🎯 Reflection: Can the server run in both local (stdio) and remote (HTTP) modes?

**✅ YES** - The NetworkX MCP Server successfully supports dual-mode transport:

### Local Mode (stdio)
- **Purpose**: Local MCP clients, Claude Desktop integration
- **Protocol**: JSON-RPC 2.0 over stdin/stdout
- **Session**: Per-process, stateless between requests
- **Authentication**: Simple token validation
- **Performance**: Optimized for single-user, low latency

### Remote Mode (HTTP)
- **Purpose**: Web applications, remote clients, multi-user access
- **Protocol**: JSON-RPC 2.0 over HTTP + Server-Sent Events
- **Session**: HTTP session management with timeouts
- **Authentication**: OAuth 2.0 + CORS protection
- **Performance**: Optimized for concurrent users (45 max per instance)

### Shared Features
- **Same JSON-RPC protocol** across both modes
- **Identical tools and capabilities**
- **Same performance limits** (10K nodes, 2GB memory)
- **Graceful shutdown** and health monitoring
- **Production-ready** configuration

### Use Cases
- **Local**: `python -m networkx_mcp --jsonrpc` for Claude Desktop
- **Remote**: `python -m networkx_mcp --http` for web applications
- **Development**: `--no-auth` flag for testing
- **Production**: OAuth 2.0 with proper CORS configuration

**Conclusion**: The dual-mode design provides maximum flexibility while maintaining protocol compatibility and security across different deployment scenarios.

---

## 📚 Additional Resources

- [MCP Protocol Specification](https://docs.anthropic.com/claude/reference/mcp)
- [Production Deployment Guide](./PRODUCTION_DEPLOYMENT_SUMMARY.md)
- [Security Configuration](./SECURITY.md)
- [Performance Testing Results](./MCP_CLIENT_COMPATIBILITY.md)

---

*Updated: December 2024 - Version 1.0.0*