# generate_report

**Category:** Advanced Analytics

## Description

Generate analysis report in various formats.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `analysis_data` | `Dict[str, Any]` | Yes | `-` | Analysis results to include |
| `report_format` | `str` | No | `pdf` | Format - 'pdf', 'html' |
| `template` | `str` | No | `default` | Report template |
| `params` | `Optional[Dict[str, Any]]` | No | `-` | Additional parameters |

## Returns

**Type:** `Dict[str, Any]`

Generated report data

## Source

Located in: `src/networkx_mcp/server.py:3252`
