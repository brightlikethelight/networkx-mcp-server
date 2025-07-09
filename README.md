# NetworkX MCP Server

A **truly minimal** MCP server providing NetworkX graph operations to AI assistants.

> **⚠️ Architecture Fix**: v0.1.0-alpha.2 reduces memory from 118MB to 54MB by removing forced pandas/scipy imports. See [ADR-001](docs/ADR-001-remove-heavyweight-dependencies.md) for details.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-orange.svg)](https://networkx.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Current Status: Alpha (v0.1.0-alpha.2)

**This is a minimal, working implementation - not production software.**

### 🎯 Memory Footprint (Honest Numbers)

| Version | Memory | Modules | What You Get |
|---------|--------|---------|---------------|
| **Minimal** (default) | **54MB** | ~600 | Core graph operations only |
| **With Excel** | **89MB** | ~800 | + pandas for Excel/CSV I/O |
| **Full** | **118MB** | ~900 | + scipy, matplotlib |

<details>
<summary>Why these numbers?</summary>

- Python interpreter: 16MB
- NetworkX library: 20MB  
- Our server code: 18MB
- **Total: 54MB** (not 20MB - let's be honest)

The original v0.1.0 forced everyone to load pandas (+35MB) even for basic operations. We fixed this architectural disaster.
</details>

### ✅ What Works
- **7 graph tools** via MCP protocol (tested and verified)
- **Stdio transport** for local operation  
- **Claude Desktop integration** (works out of the box)
- **Core NetworkX algorithms** (shortest path, centrality measures)
- **Basic graph operations** (create, add nodes/edges, delete)

### ❌ What Doesn't Work Yet
- HTTP transport (doesn't exist - stdio only)
- Persistent storage (exists but not integrated into server) 
- Multi-user support (stdio limitation)
- Authentication/authorization (not applicable for stdio)

### ⚠️ Current Limitations (REAL TESTED LIMITS)
- **Server Memory**: 54MB minimum (not 20MB - NetworkX needs ~20MB alone)
- **Graph Capacity**: 10,000 nodes tested (performance degrades beyond this)
- **Graph Memory**: ~0.2KB per node (~2MB for 10K nodes)
- **Speed**: Graph creation ~935ms for 10K nodes (MCP protocol overhead)
- **Concurrency**: Single-user only (one graph operation at a time)
- **Transport**: Stdio only (no remote access)
- **Storage**: In-memory only (data lost on restart)
- **Security**: Local process isolation only

> **Performance Reality**: Claims were 5x inflated. See `docs/PERFORMANCE_REALITY_CHECK.md` for actual benchmarks.
> **Memory Reality**: Was using 118MB while claiming "minimal". Now actually 54MB. See [MEMORY_BLOAT_ANALYSIS.md](MEMORY_BLOAT_ANALYSIS.md).

## Installation

### Option 1: Minimal (Recommended) - 54MB
```bash
pip install networkx-mcp
```
Only NetworkX, no data science bloat. Perfect for 90% of use cases.

### Option 2: With Excel/CSV Support - 89MB
```bash
pip install networkx-mcp[excel]
```
Adds pandas for data I/O. Only install if you actually need it.

### Option 3: Everything - 118MB  
```bash
pip install networkx-mcp[full]
```
Includes pandas, scipy, matplotlib. The old bloated default.

### From Source
```bash
git clone https://github.com/your-username/networkx-mcp-server
cd networkx-mcp-server
pip install -e .  # Minimal by default
```

## Usage with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "networkx": {
      "command": "python",
      "args": ["-m", "networkx_mcp"],
      "cwd": "/path/to/networkx-mcp-server",
      "env": {
        "PYTHONPATH": "/path/to/networkx-mcp-server/src"
      }
    }
  }
}
```

## Available Tools

| Tool | Description | Status |
|------|-------------|--------|
| `create_graph` | Create new graph (directed/undirected) | ✅ Working |
| `add_nodes` | Add nodes to graph | ✅ Working |
| `add_edges` | Add edges between nodes | ✅ Working |
| `get_graph_info` | Get graph statistics and metadata | ✅ Working |
| `shortest_path` | Find shortest path between nodes | ✅ Working |
| `centrality_measures` | Calculate network centrality | ✅ Working |
| `delete_graph` | Remove graph from memory | ✅ Working |

## Performance Characteristics

**Measured with real subprocess benchmarks and memory profiling:**

| Metric | Tested Result | Performance |
|--------|---------------|-------------|
| **Max Tested Nodes** | 10,000 | Good |
| **Max Tested Edges** | 10,000 | Good |
| **Memory Usage** | ~0.2KB/node, ~234 bytes/edge | Reasonable |
| **Basic Operations** | 10-25ms | Acceptable |
| **Graph Creation** | 935ms | Slow ⚠️ |
| **Shortest Path** | 10.5ms (100 nodes) | Fast |
| **Centrality** | 11.1ms (100 nodes) | Fast |

*See [benchmarks/REAL_PERFORMANCE_REPORT.md](benchmarks/REAL_PERFORMANCE_REPORT.md) for detailed results.*

### Performance Guidance
- **Small graphs (1-1K nodes)**: Fast (0.01-0.13s)
- **Medium graphs (1K-5K nodes)**: Good (0.13-0.59s)  
- **Large graphs (5K-10K nodes)**: Acceptable (0.59-1.19s)
- **Very large graphs (10K+ nodes)**: Untested

## Docker Support

### Quick Start with Docker
```bash
# Build the image
docker build -t networkx-mcp:0.1.0 .

# Run with JSON-RPC input
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | \
  docker run -i networkx-mcp:0.1.0

# Run test suite
./test_docker.sh
```

See [docs/DOCKER_USAGE.md](docs/DOCKER_USAGE.md) for detailed Docker instructions.

## Testing

```bash
# Run honest tests that verify actual functionality
python -m pytest tests/test_brutally_honest_tools.py -v

# Test with real MCP protocol
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | python -m networkx_mcp
```

## Limitations & Roadmap

### v0.1.0 (Current) - Minimal Working Server
- ✅ Basic graph operations
- ✅ Stdio transport only
- ✅ Single-user, in-memory

### v0.2.0 (Planned) - Network Transport
- ❌ HTTP transport
- ❌ Basic authentication
- ❌ Multi-client support

### v0.3.0 (Planned) - Persistence  
- ❌ File-based storage
- ❌ Session management
- ❌ Graph import/export

### v1.0.0 (Future) - Production Ready
- ❌ Security features
- ❌ Monitoring & logging
- ❌ Performance optimizations

## Contributing

This project values **honesty over hype**. Before contributing:

1. Verify your feature actually works
2. Write tests that prove it works
3. Document limitations clearly
4. No marketing claims without evidence

## Security Warning

**⚠️ DO NOT USE IN PRODUCTION**

This is alpha software with:
- No authentication
- No input validation
- No rate limiting
- No access controls

Use only for local development and experimentation.

## License

MIT License - see [LICENSE](LICENSE) file.

## Honest Assessment

This project was built to demonstrate a working MCP server, not as production software. It prioritizes clarity and reliability over features. If you need production graph analysis, consider established solutions like Neo4j or AWS Neptune.