# create_graph

**Category:** Core Operations

## Description

Create a new NetworkX graph with comprehensive initialization options.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | Unique identifier for the graph |
| `graph_type` | `str` | No | `undirected` | Type of graph - 'undirected', 'directed', 'multigraph', 'multidigraph' |
| `from_data` | `Optional[Dict[str, Any]]` | No | `-` | Optional initialization data: |
| `attributes` | `Optional[Dict[str, Any]]` | No | `-` | Additional graph attributes |

## Returns

**Type:** `Dict[str, Any]`

Graph creation status with metadata including: - graph_id: The created graph ID - graph_type: The type of graph created - num_nodes: Number of nodes - num_edges: Number of edges - created_at: Creation timestamp - initialization_method: How the graph was initialized - memory_estimate: Estimated memory usage

## Source

Located in: `src/networkx_mcp/server.py:66`
