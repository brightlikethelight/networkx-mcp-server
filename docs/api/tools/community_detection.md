# community_detection

**Category:** Advanced Analytics

## Description

Detect communities in a graph.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `method` | `str` | No | `louvain` | Community detection algorithm |

## Returns

**Type:** `Dict[str, Any]`

Detected communities

## Source

Located in: `src/networkx_mcp/server.py:882`
