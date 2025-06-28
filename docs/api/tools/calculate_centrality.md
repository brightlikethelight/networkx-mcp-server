# calculate_centrality

**Category:** Graph Algorithms

## Description

Calculate various centrality measures for nodes in a graph.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `centrality_type` | `Union[str, List[str]]` | No | `degree` | Type(s) of centrality - 'degree', 'betweenness', 'closeness', |
| `top_n` | `Optional[int]` | No | `10` | Number of top central nodes to highlight |
| `include_statistics` | `bool` | No | `True` | Whether to include mean, std, min, max statistics |
| `weight` | `Optional[str]` | No | `-` | Edge attribute to use as weight (for weighted centralities) |

## Returns

**Type:** `Dict[str, Any]`

Centrality results including: - centrality_scores: Dict of node -> centrality value - top_nodes: List of top N most central nodes - statistics: Mean, std, min, max of centrality values - distribution: Histogram of centrality values - computation_time: Time taken for calculation

## Source

Located in: `src/networkx_mcp/server.py:678`
