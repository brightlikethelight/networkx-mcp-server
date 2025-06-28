# connected_components

**Category:** Graph Algorithms

## Description

Find connected components in a graph.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `component_type` | `str` | No | `weakly` | For directed graphs - 'weakly' or 'strongly' connected |
| `return_sizes` | `bool` | No | `True` | Whether to return component size distribution |
| `largest_only` | `bool` | No | `False` | Return only the largest component |

## Returns

**Type:** `Dict[str, Any]`

Component analysis including: - components: List of components (each component is a list of nodes) - num_components: Total number of components - component_sizes: Size of each component - largest_component_size: Size of the largest component - is_connected: Whether graph is fully connected - size_distribution: Distribution of component sizes

## Source

Located in: `src/networkx_mcp/server.py:1482`
