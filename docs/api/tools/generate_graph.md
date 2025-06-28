# generate_graph

**Category:** Advanced Analytics

## Description

Generate synthetic graphs using various models.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_type` | `str` | Yes | `-` | Type - 'random', 'scale_free', 'small_world', 'regular', |
| `n` | `int` | Yes | `-` | Number of nodes |
| `graph_id` | `Optional[str]` | No | `-` | ID for the generated graph (auto-generated if not provided) |
| `params` | `Optional[Dict[str, Any]]` | No | `-` | Model-specific parameters (e.g., m for BA model, p for ER model) |

## Returns

**Type:** `Dict[str, Any]`

Generated graph information and statistics

## Source

Located in: `src/networkx_mcp/server.py:2578`
