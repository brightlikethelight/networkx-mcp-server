# advanced_community_detection

**Category:** Advanced Analytics

## Description

Detect communities using advanced algorithms with auto-selection.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `algorithm` | `str` | No | `auto` | Algorithm - 'auto', 'louvain', 'girvan_newman', 'spectral', |
| `resolution` | `float` | No | `1.0` | Resolution parameter (for resolution-based methods) |
| `seed` | `Optional[int]` | No | `-` | Random seed for reproducibility |
| `params` | `Optional[Dict[str, Any]]` | No | `-` | Additional algorithm-specific parameters |

## Returns

**Type:** `Dict[str, Any]`

Community structure with quality metrics

## Source

Located in: `src/networkx_mcp/server.py:2477`
