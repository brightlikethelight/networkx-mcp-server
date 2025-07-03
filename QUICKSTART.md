# 🚀 NetworkX MCP Server - Quick Start

## CRITICAL FIX: Making it Run NOW

### Option 1: Minimal Server (Guaranteed to Work)

```bash
# 1. Install minimal dependencies
pip install networkx mcp

# 2. Run the minimal server
python -m networkx_mcp --minimal
```

### Option 2: Full Installation

```bash
# 1. Install without conflicts
pip install -e .

# 2. Try running the full server (will fall back to minimal if needed)
python -m networkx_mcp
```

### Option 3: With FastMCP (if you have Pydantic v2)

```bash
# Install with FastMCP support
pip install -e ".[fastmcp]"

# Run the server
python -m networkx_mcp
```

## Testing the Server

```bash
# Run the test script
python test_minimal_server.py
```

## What's Working

The minimal server provides these core features:
- ✅ Create/delete graphs
- ✅ Add/remove nodes and edges
- ✅ Basic graph info and statistics
- ✅ Shortest path algorithms
- ✅ Node degree analysis

## Troubleshooting

### Pydantic Conflict
If you see Pydantic version conflicts:
```bash
# Use the minimal server
python -m networkx_mcp --minimal
```

### Import Errors
The compatibility layer handles different MCP versions automatically.

### Can't Connect
Make sure you're using the correct MCP client configuration for stdio transport.

## Next Steps

Once the minimal server is running, you can:
1. Add more algorithms
2. Enable the full feature set
3. Configure enterprise features

---

**The server is designed to ALWAYS work** - if the full version fails, it automatically falls back to the minimal version.