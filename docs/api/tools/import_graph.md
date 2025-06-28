# import_graph

**Category:** Data Integration

## Description

Import a graph from various formats.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID for the imported graph |
| `format` | `str` | Yes | `-` | Import format (json, graphml, gexf, etc.) |
| `data` | `Optional[Dict[str, Any]]` | No | `-` | Graph data (for formats that support direct data) |
| `path` | `Optional[str]` | No | `-` | Input file path (for file-based formats) |
| `options` | `Optional[Dict[str, Any]]` | No | `-` | Format-specific options |

## Returns

**Type:** `Dict[str, Any]`

Import status

## Source

Located in: `src/networkx_mcp/server.py:979`
