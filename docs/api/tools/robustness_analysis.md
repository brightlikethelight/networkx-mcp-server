# robustness_analysis

**Category:** Advanced Analytics

## Description

Analyze network robustness and resilience.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `analysis_type` | `str` | No | `attack` | Type - 'attack', 'percolation', 'cascading', 'resilience' |
| `params` | `Optional[Dict[str, Any]]` | No | `-` | Analysis-specific parameters |

## Returns

**Type:** `Dict[str, Any]`

Robustness analysis results

## Source

Located in: `src/networkx_mcp/server.py:2885`
