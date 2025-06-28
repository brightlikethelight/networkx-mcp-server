# import_from_source

**Category:** Data Integration

## Description

Import graph data from various sources with intelligent parsing.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `source_type` | `str` | Yes | `-` | Type - 'csv', 'json', 'database', 'api', 'excel' |
| `source_config` | `Dict[str, Any]` | Yes | `-` | Source-specific configuration |
| `graph_id` | `Optional[str]` | No | `-` | ID for the imported graph (auto-generated if not provided) |
| `params` | `Optional[Dict[str, Any]]` | No | `-` | Additional import parameters |

## Returns

**Type:** `Dict[str, Any]`

Import results with graph metadata

## Source

Located in: `src/networkx_mcp/server.py:3066`
