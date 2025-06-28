# create_dashboard

**Category:** Core Operations

## Description

Create interactive dashboard with multiple visualizations.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `visualizations` | `List[str]` | No | `-` | List of visualization types to include |
| `params` | `Optional[Dict[str, Any]]` | No | `-` | Additional parameters |

## Returns

**Type:** `Dict[str, Any]`

Dashboard data

## Source

Located in: `src/networkx_mcp/server.py:3362`
