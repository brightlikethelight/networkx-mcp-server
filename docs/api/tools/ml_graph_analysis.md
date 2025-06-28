# ml_graph_analysis

**Category:** Utilities

## Description

Machine learning-based graph analysis.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `analysis_type` | `str` | Yes | `-` | Type - 'embeddings', 'features', 'similarity', 'anomaly' |
| `params` | `Optional[Dict[str, Any]]` | No | `-` | Analysis-specific parameters |

## Returns

**Type:** `Dict[str, Any]`

ML analysis results

## Source

Located in: `src/networkx_mcp/server.py:2821`
