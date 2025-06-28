# shortest_path

**Category:** Graph Algorithms

## Description

Find shortest path(s) in a graph with multiple algorithm support.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `source` | `Union[str, int]` | Yes | `-` | Source node |
| `target` | `Optional[Union[str, int]]` | No | `-` | Target node (optional - if not provided, finds paths to all nodes) |
| `weight` | `Optional[str]` | No | `-` | Edge weight attribute name (None for unweighted) |
| `method` | `str` | No | `dijkstra` | Algorithm - 'dijkstra', 'bellman-ford', 'floyd-warshall', 'astar' |
| `k_paths` | `Optional[int]` | No | `-` | Number of shortest paths to find (for k-shortest paths) |

## Returns

**Type:** `Dict[str, Any]`

Path information including: - path(s): Node sequence(s) - distance(s): Total path weight(s) - algorithm_used: Which algorithm was applied - computation_time: Time taken - path_count: Number of paths found

## Source

Located in: `src/networkx_mcp/server.py:514`
