# 🎯 NetworkX MCP Server - Exact Commands to Run

## ✅ WORKING COMMANDS

### 1. **Test Everything Works** (Run These First!)
```bash
# Quick health check - verifies installation
python quickstart.py

# Full validation - tests all components
python validate_server.py

# Test stdio server can start
python test_stdio_server.py
```

### 2. **Start the Server** (Choose One Based on Your Needs)

#### Option A: Stdio Transport (RECOMMENDED - No Port Issues!)
```bash
python -m networkx_mcp.server
```
or
```bash
python -m networkx_mcp.server stdio
```

**Use this for:**
- Command-line tools
- Scripts that communicate via stdin/stdout
- Claude Desktop integration
- When you don't need HTTP/web access

#### Option B: SSE Transport (For Web Apps)
```bash
# First, kill any process on the port
lsof -ti:8765 | xargs kill -9

# Then start the server
python -m networkx_mcp.server sse 8765
```

**Use this for:**
- Web applications
- HTTP API access
- When you need network connectivity

### 3. **If You Get Errors**

#### "Address already in use" (SSE only)
```bash
# Kill the process on that port
lsof -ti:8765 | xargs kill -9

# Or use a different port
python -m networkx_mcp.server sse 8766
```

#### Module import errors
```bash
# Reinstall in development mode
pip install -e .
```

## 📌 QUICK REFERENCE

**Stdio (No Network):**
```bash
python -m networkx_mcp.server
```

**SSE (HTTP Server):**
```bash
python -m networkx_mcp.server sse 8765
```

**With Custom Port:**
```bash
python -m networkx_mcp.server sse 9000
```

## ⚠️ IMPORTANT NOTES

1. The "RuntimeWarning about sys.modules" is harmless - ignore it
2. Stdio transport doesn't use any network ports (no conflicts!)
3. SSE transport requires a free port (default: 8765)
4. The server is fully functional with all 39 MCP tools available

## 🚀 RECOMMENDED WORKFLOW

```bash
# 1. First time setup check
python quickstart.py

# 2. Start server (stdio mode - no port issues!)
python -m networkx_mcp.server

# Server is now running and ready to process graph operations!
```
