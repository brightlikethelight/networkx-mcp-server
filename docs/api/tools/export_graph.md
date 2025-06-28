# export_graph

**Category:** Data Integration

## Description

Export a graph to various formats.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `format` | `str` | Yes | `-` | Export format (json, graphml, gexf, etc.) |
| `path` | `Optional[str]` | No | `-` | Output file path (optional for some formats) |
| `options` | `Optional[Dict[str, Any]]` | No | `-` | Format-specific options |

## Returns

**Type:** `Dict[str, Any]`

Export status or data

## Source

Located in: `src/networkx_mcp/server.py:940`
