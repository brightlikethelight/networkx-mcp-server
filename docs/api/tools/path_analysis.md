# path_analysis

**Category:** Graph Algorithms

## Description

Analyze path properties of a graph including diameter, radius, and eccentricity.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `sample_size` | `Optional[int]` | No | `1000` | Number of node pairs to sample for large graphs |

## Returns

**Type:** `Dict[str, Any]`

Path analysis including: - average_shortest_path_length: Average distance between all node pairs - diameter: Maximum eccentricity - radius: Minimum eccentricity - eccentricity: Dict of node eccentricities - center: Nodes with eccentricity equal to radius - periphery: Nodes with eccentricity equal to diameter - path_length_distribution: Distribution of shortest path lengths

## Source

Located in: `src/networkx_mcp/server.py:1732`
