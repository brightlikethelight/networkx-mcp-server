# find_all_paths

**Category:** Graph Algorithms

## Description

Find all simple paths between two nodes with constraints.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `source` | `Union[str, int]` | Yes | `-` | Source node |
| `target` | `Union[str, int]` | Yes | `-` | Target node |
| `max_length` | `Optional[int]` | No | `-` | Maximum path length (None for no limit) |
| `max_paths` | `int` | No | `100` | Maximum number of paths to return (default 100) |

## Returns

**Type:** `Dict[str, Any]`

Path information including: - paths: List of all paths found - path_count: Total number of paths - length_distribution: Distribution of path lengths - shortest_path_length: Length of shortest path - longest_path_length: Length of longest path

## Source

Located in: `src/networkx_mcp/server.py:1621`
