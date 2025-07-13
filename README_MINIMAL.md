# NetworkX MCP Server - Actually Minimal Edition

**150 lines. No BS. Just works.**

## What This Is

A truly minimal MCP (Model Context Protocol) server that wraps NetworkX for graph operations. Unlike the main implementation with 16,000+ lines across 68 files, this does the same job in 150 lines.

## What Works

- ✅ Create directed/undirected graphs
- ✅ Add nodes and edges  
- ✅ Find shortest paths
- ✅ Get graph information
- ✅ Proper error messages
- ✅ Tests that actually run
- ✅ Docker deployment

## What Doesn't Work

- ❌ Excel import (you don't need it)
- ❌ Visualization (broken anyway)
- ❌ Redis persistence (untested)
- ❌ 500-line validator classes
- ❌ Abstract factory patterns
- ❌ Your ego

## Installation

```bash
pip install networkx
```

That's it. No 50MB pandas dependency for basic graph operations.

## Usage

### Run the Server
```bash
python server_truly_minimal.py
```

### Run Tests
```bash
python test_minimal_server.py
```

### Deploy with Docker
```bash
docker build -f Dockerfile.minimal -t networkx-mcp-minimal .
docker run -it networkx-mcp-minimal
```

## Example MCP Interaction

```json
// Request
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "create_graph",
    "arguments": {"name": "my_graph", "directed": false}
  }
}

// Response  
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{
      "type": "text",
      "text": "{\"created\": \"my_graph\", \"type\": \"undirected\"}"
    }]
  }
}
```

## Memory Usage

- **This implementation**: ~30MB
- **Main implementation**: 54.6MB (was 118MB!)
- **Difference**: Your RAM thanks you

## Why This Exists

The main NetworkX MCP Server is a cautionary tale of what happens when architecture astronauts are left unsupervised:

- 900+ line "minimal" server
- 68 Python files for basic CRUD operations
- Tests that don't run
- Performance benchmarks showing negative memory usage
- Can't actually be deployed

This minimal version proves the same functionality can be achieved in 150 lines with no external dependencies beyond NetworkX.

## Philosophy

> "Make it work, make it right, make it fast" - Kent Beck

The main implementation never got past step 0: "Make it complicated."

This implementation:
1. **Works** ✓
2. **Is right** ✓ 
3. **Is fast enough** ✓

## Contributing

Don't. It's 150 lines. If you need more features, you're looking for a different project.

## License

MIT - Do whatever you want, but don't blame me.

---

**Remember**: The best code is no code. The second best is minimal code that actually works.