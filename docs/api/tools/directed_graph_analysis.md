# directed_graph_analysis

**Category:** Utilities

## Description

Analyze directed graphs with specialized algorithms.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `analysis_type` | `str` | No | `dag_check` | Type - 'dag_check', 'scc', 'topological_sort', 'tournament', |
| `params` | `Optional[Dict[str, Any]]` | No | `-` | Analysis-specific parameters |

## Returns

**Type:** `Dict[str, Any]`

Directed graph analysis results

## Source

Located in: `src/networkx_mcp/server.py:2704`
