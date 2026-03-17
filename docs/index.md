# NetworkX MCP Server

Graph analysis via the Model Context Protocol. v3.0.0.

NetworkX MCP Server exposes [NetworkX](https://networkx.org/) through the
[Model Context Protocol](https://github.com/anthropics/mcp), giving Claude
Desktop and other MCP clients access to graph algorithms without writing Python.

## Capabilities

**46 tools** across 4 categories:

- **Core graph operations (30)** -- create/delete graphs, list graphs,
  add/remove nodes and edges, get/set node/edge attributes,
  get neighbors, graph statistics, shortest path,
  centrality (degree, betweenness, multi-centrality), PageRank,
  community detection, clustering coefficients, minimum spanning tree,
  cycle detection, graph coloring, matching, maximum flow,
  visualization (PNG), CSV import/export, JSON export
- **Academic / citation tools (7)** -- build citation networks from the CrossRef
  API, BibTeX export, author impact analysis, plus `resolve_doi`
- **CI/CD tools (6)** -- dependency graph analysis for build pipelines
- **Monitoring (1)** -- server health check

## Architecture

```
MCP Client (Claude Desktop, etc.)
    |
    | stdio (JSON-RPC 2.0)
    |
NetworkX MCP Server
    |
    +-- Graph Manager (in-memory storage)
    +-- NetworkX library
    +-- CrossRef API (citation tools)
```

All graphs live in memory and are lost on server restart. There is no
persistence layer, no Redis, and no HTTP transport.

## Quick start

Install and register with Claude Desktop:

```bash
pip install networkx-mcp-server
```

Add to your Claude Desktop config (`claude_desktop_config.json`):

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

Then ask Claude to create a graph:

> "Create a graph called 'demo', add edges Alice-Bob, Bob-Charlie,
> Alice-David, Charlie-Eve, then run community detection."

## Example tool calls

Create a graph and analyze it:

```
create_graph(name="social")

add_edges(graph="social", edges=[
    ["Alice", "Bob"], ["Bob", "Charlie"],
    ["Alice", "David"], ["Charlie", "Eve"]
])

get_info(graph="social")
# => {"nodes": 5, "edges": 4, "directed": false}

betweenness_centrality(graph="social")
# => {"centrality": {"Alice": 0.5, "Bob": 0.5, ...}, ...}

community_detection(graph="social")
# => {"communities": [["Alice", "Bob", "David"], ["Charlie", "Eve"]], ...}
```

## Test coverage

630 tests, ~86% line coverage.

```bash
pytest tests/ -v
```

## Limitations

- **In-memory only** -- graphs do not survive server restarts.
- **stdio transport only** -- no HTTP/SSE endpoint.
- **Single process** -- not designed for concurrent multi-client access.
- **NetworkX scale** -- practical for graphs up to ~100K nodes; larger graphs
  will be slow since NetworkX is pure Python.

## Links

- [Source code](https://github.com/Bright-L01/networkx-mcp-server)
- [NetworkX documentation](https://networkx.org/documentation/stable/)
- [MCP specification](https://spec.modelcontextprotocol.io/)
