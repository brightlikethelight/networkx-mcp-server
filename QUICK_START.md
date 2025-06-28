# 🚀 NetworkX MCP Server Quick Start

## Starting the Server

The NetworkX MCP Server supports multiple transport methods for different use cases.

### Basic Usage

#### 1. **Stdio Transport** (Default - for command-line tools)
```bash
# Start with stdio transport (default)
python -m networkx_mcp.server

# Or explicitly specify stdio
python -m networkx_mcp.server stdio
```

#### 2. **SSE Transport** (For web applications)
```bash
# Start with SSE transport on default port 8765
python -m networkx_mcp.server sse

# Start with custom port
python -m networkx_mcp.server sse 9000

# Using environment variable
MCP_PORT=9000 python -m networkx_mcp.server sse
```

#### 3. **Streamable HTTP Transport**
```bash
python -m networkx_mcp.server streamable-http 8765
```

### Transport Methods Explained

- **stdio**: Communicates via standard input/output (best for CLI tools and scripts)
- **sse**: Server-Sent Events over HTTP (best for web applications)  
- **streamable-http**: HTTP with streaming support

### If Port is Already in Use (SSE/HTTP only)

If you get an "Address already in use" error:

1. **Kill the existing process:**
   ```bash
   lsof -ti:8765 | xargs kill -9
   ```

2. **Or use a different port:**
   ```bash
   python -m networkx_mcp.server sse 8766
   ```

## Testing the Installation

1. **Quick health check:**
   ```bash
   python quickstart.py
   ```

2. **Full validation:**
   ```bash
   python validate_server.py
   ```

3. **Test stdio server:**
   ```bash
   python test_stdio_server.py
   ```

4. **Test SSE server:**
   ```bash
   python test_server_startup.py
   ```

## Server Information

- **Default Transport:** stdio (no network required)
- **Default Port (SSE/HTTP):** 8765
- **Host:** 0.0.0.0 (accessible from all network interfaces)
- **Protocol:** MCP (Model Context Protocol)

## Example Usage

### With Claude Desktop (stdio)
```bash
# In your Claude Desktop config, add:
# command: "python -m networkx_mcp.server"
```

### With Web Application (SSE)
```bash
# Start server
python -m networkx_mcp.server sse 8765

# Connect from your app to:
# http://localhost:8765/sse
```

## Troubleshooting

### RuntimeWarning about sys.modules
This is a harmless warning that can be ignored. The server will still function correctly.

### Port Already in Use (SSE/HTTP only)
Use `lsof -ti:PORT | xargs kill -9` to kill processes on that port

### Module Import Errors
Run `pip install -e .` to ensure the package is properly installed