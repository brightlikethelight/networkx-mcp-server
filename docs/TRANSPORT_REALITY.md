# Transport Reality Check

This document provides an honest assessment of transport support in NetworkX MCP Server.

## Current Status (v0.1.0)

### ✅ What Works: Stdio Transport

The server implements **stdio transport only**. This means:
- Communication via standard input/output
- Line-delimited JSON-RPC 2.0 messages
- Single client connection at a time
- Perfect for local CLI usage and Docker containers

### ❌ What Doesn't Exist: HTTP Transport

Despite any lingering references, **HTTP transport is NOT implemented**. 

Our reality check (`test_http_reality.py`) confirmed:
- No `--transport` command line flag
- No HTTP server code
- No HTTP endpoint handlers
- No aiohttp in requirements

## Why Stdio-Only is Fine

1. **It Actually Works** - Unlike broken HTTP implementations, our stdio transport is solid
2. **MCP Standard** - Most MCP servers start with stdio
3. **Docker Compatible** - Works perfectly in containers
4. **Simple & Secure** - No network exposure, no auth complexity

## Future Transport Options

### v0.2.0 - HTTP Transport (Planned)
When we do implement HTTP, we'll need:
- Proper async HTTP server (aiohttp or FastAPI)
- JSON-RPC over HTTP POST
- Session management
- CORS handling
- Authentication headers
- Origin validation (security)

### Alternative Transports (Research Phase)
- **WebSocket** - For bidirectional communication
- **Server-Sent Events (SSE)** - For server push
- **gRPC** - For performance-critical applications

## Testing Transport

To verify which transports actually work:

```bash
# Run the reality check
python test_http_reality.py

# Test stdio (the only working transport)
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python -m networkx_mcp.server
```

## Implementation Honesty

We removed all misleading references to "dual-mode transport" because:
- False advertising hurts trust
- Broken features are worse than no features
- Clear limitations help users make informed decisions

## Contributing HTTP Transport

If you want to help implement HTTP transport:

1. Start with `src/networkx_mcp/transport/http_server.py`
2. Use aiohttp for async HTTP handling
3. Implement proper JSON-RPC routing
4. Add comprehensive tests
5. Update documentation honestly

Remember: **Working code > Aspirational features**