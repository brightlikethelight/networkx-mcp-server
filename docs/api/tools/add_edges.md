# add_edges

**Category:** Core Operations

## Description

Add edges to a graph with support for bulk operations, weights, and attributes.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph to add edges to |
| `edges` | `List[Union[Tuple, Dict[str, Any]]]` | Yes | `-` | List of edges in any of these formats: |
| `edge_attributes` | `Optional[Dict[str, Any]]` | No | `-` | Optional default attributes to apply to all edges |

## Returns

**Type:** `Dict[str, Any]`

Operation status including: - edges_added: Number of edges added - total_edges: Total edges in graph after operation - weighted_edges: Number of edges with weights - multi_edges_created: Number of multi-edges (for multigraphs) - self_loops_added: Number of self-loops added - execution_time_ms: Time taken for the operation

## Source

Located in: `src/networkx_mcp/server.py:385`
