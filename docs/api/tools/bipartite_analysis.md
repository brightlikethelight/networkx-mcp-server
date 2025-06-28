# bipartite_analysis

**Category:** Advanced Analytics

## Description

Analyze bipartite graphs with specialized algorithms.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `analysis_type` | `str` | No | `check` | Type - 'check', 'projection', 'matching', 'clustering', 'communities' |
| `weight` | `Optional[str]` | No | `-` | Edge weight attribute (for weighted operations) |
| `params` | `Optional[Dict[str, Any]]` | No | `-` | Analysis-specific parameters |

## Returns

**Type:** `Dict[str, Any]`

Bipartite analysis results

## Source

Located in: `src/networkx_mcp/server.py:2650`
