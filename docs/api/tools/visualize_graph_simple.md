# visualize_graph_simple

**Category:** Visualization

## Description

Generate visualization data for a graph.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `layout` | `str` | No | `spring` | Layout algorithm |
| `output_format` | `str` | No | `pyvis` | Visualization format (pyvis, plotly, matplotlib) |
| `options` | `Optional[Dict[str, Any]]` | No | `-` | Visualization options |

## Returns

**Type:** `Dict[str, Any]`

Visualization data or file path

## Source

Located in: `src/networkx_mcp/server.py:1031`
