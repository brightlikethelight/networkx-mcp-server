# flow_paths

**Category:** Graph Algorithms

## Description

Analyze flow paths in a directed graph including max flow and edge-disjoint paths.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph (must be directed) |
| `source` | `Union[str, int]` | Yes | `-` | Source node |
| `target` | `Union[str, int]` | Yes | `-` | Target (sink) node |
| `capacity` | `str` | No | `capacity` | Edge attribute name for capacity (default 'capacity') |
| `flow_type` | `str` | No | `maximum` | Type of flow analysis - 'maximum', 'edge_disjoint', 'node_disjoint' |

## Returns

**Type:** `Dict[str, Any]`

Flow analysis including: - flow_value: Maximum flow value - flow_dict: Flow on each edge - min_cut: Edges in minimum cut - min_cut_value: Value of minimum cut - disjoint_paths: Edge or node disjoint paths - path_capacities: Capacity of each path

## Source

Located in: `src/networkx_mcp/server.py:2026`
