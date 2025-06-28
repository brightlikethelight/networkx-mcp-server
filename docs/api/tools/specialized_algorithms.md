# specialized_algorithms

**Category:** Visualization

## Description

Run specialized graph algorithms.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `algorithm` | `str` | Yes | `-` | Algorithm - 'spanning_tree', 'coloring', 'max_clique', |
| `params` | `Optional[Dict[str, Any]]` | No | `-` | Algorithm-specific parameters |

## Returns

**Type:** `Dict[str, Any]`

Algorithm results

## Source

Located in: `src/networkx_mcp/server.py:2766`
