# network_flow_analysis

**Category:** Advanced Analytics

## Description

Analyze network flow with multiple algorithms.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `source` | `Union[str, int]` | Yes | `-` | Source node |
| `sink` | `Union[str, int]` | Yes | `-` | Sink node |
| `capacity` | `str` | No | `capacity` | Edge attribute for capacities |
| `algorithm` | `str` | No | `auto` | Algorithm - 'auto', 'ford_fulkerson', 'edmonds_karp', 'dinic', 'preflow_push' |
| `flow_type` | `str` | No | `max_flow` | Analysis type - 'max_flow', 'min_cut', 'multi_commodity' |

## Returns

**Type:** `Dict[str, Any]`

Flow analysis results including flow value, flow dict, and cut sets

## Source

Located in: `src/networkx_mcp/server.py:2525`
