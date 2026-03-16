# NetworkX MCP Server - Quick Start

Python 3.11+ required.

## Install

```bash
# Standard install
pip install -e .

# With dev/test dependencies
pip install -e ".[dev]"
```

## Run the Server

```bash
python -m networkx_mcp
```

The server uses stdio transport (JSON-RPC 2.0 over stdin/stdout).

## Run Tests

```bash
pytest tests/working/ -v
```

## Claude Desktop Integration

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "networkx": {
      "command": "python",
      "args": ["-m", "networkx_mcp"]
    }
  }
}
```

## Features

- **Graph CRUD** -- create, read, update, delete graphs, nodes, and edges
- **Centrality algorithms** -- betweenness, closeness, eigenvector, degree
- **Community detection** -- Louvain, label propagation
- **PageRank**
- **Visualization** -- graph rendering and export
- **Citation network analysis**
- **CI/CD tools** -- DORA metrics, deployment tracking

## Troubleshooting

**Import errors:** Make sure you installed with `pip install -e .` from the project root.

**Can't connect from Claude Desktop:** Verify the `claude_desktop_config.json` path and that `python -m networkx_mcp` runs without errors in your terminal first.
