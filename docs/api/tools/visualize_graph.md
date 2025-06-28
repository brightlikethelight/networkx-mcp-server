# visualize_graph

**Category:** Visualization

## Description

Create graph visualizations with various backends.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `visualization_type` | `str` | No | `static` | Type - 'static', 'interactive', 'pyvis', 'specialized' |
| `layout` | `str` | No | `spring` | Layout algorithm - 'spring', 'circular', 'shell', 'kamada_kawai', 'hierarchical' |
| `format` | `str` | No | `png` | Output format - 'png', 'svg', 'html', 'json' |
| `params` | `Optional[Dict[str, Any]]` | No | `-` | Additional visualization parameters |

## Returns

**Type:** `Dict[str, Any]`

Visualization data in requested format

## Source

Located in: `src/networkx_mcp/server.py:2940`
