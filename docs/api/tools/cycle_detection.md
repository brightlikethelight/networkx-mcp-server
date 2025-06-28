# cycle_detection

**Category:** Utilities

## Description

Detect cycles in a graph with detailed analysis.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `max_cycle_length` | `Optional[int]` | No | `-` | Maximum length of cycles to find (None for no limit) |
| `limit` | `int` | No | `100` | Maximum number of cycles to return |

## Returns

**Type:** `Dict[str, Any]`

Cycle analysis including: - has_cycles: Whether graph contains cycles - cycles: List of cycles found (limited by 'limit' parameter) - cycle_count: Number of cycles found - shortest_cycle: Shortest cycle found - cycle_basis: Minimum cycle basis (for undirected) - is_dag: Whether graph is a DAG (directed only)

## Source

Located in: `src/networkx_mcp/server.py:1880`
