# get_graph_info

**Category:** Core Operations

## Description

Get comprehensive information about a graph.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph to analyze |

## Returns

**Type:** `Dict[str, Any]`

Detailed graph information including: - Basic properties: nodes, edges, density, type - Connectivity: is_connected, components count - Degree statistics: min, max, average, distribution - Memory usage estimation - Graph properties: has_self_loops, is_weighted, etc. - Performance characteristics

## Source

Located in: `src/networkx_mcp/server.py:217`
