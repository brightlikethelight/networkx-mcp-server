# clustering_analysis

**Category:** Graph Algorithms

## Description

Analyze clustering coefficients and triangles in a graph.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `include_triangles` | `bool` | No | `True` | Whether to count triangles |
| `nodes` | `Optional[List[Union[str, int]]]` | No | `-` | Specific nodes to analyze (None for all nodes) |

## Returns

**Type:** `Dict[str, Any]`

Clustering analysis including: - local_clustering: Clustering coefficient for each node - global_clustering: Average clustering coefficient - transitivity: Global clustering coefficient (fraction of triangles) - triangle_count: Number of triangles (if requested) - node_triangles: Triangles per node - clustering_distribution: Distribution of clustering values

## Source

Located in: `src/networkx_mcp/server.py:1366`
