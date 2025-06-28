# graph_metrics

**Category:** Utilities

## Description

Calculate comprehensive graph metrics and statistics.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph to analyze |
| `include_distributions` | `bool` | No | `True` | Whether to include degree/distance distributions |

## Returns

**Type:** `Dict[str, Any]`

Comprehensive metrics including: - Basic metrics: density, sparsity, order, size - Degree statistics: distribution, assortativity, power law analysis - Distance metrics: diameter, radius, average path length - Connectivity: components, articulation points, bridges - Structural metrics: clustering, transitivity, reciprocity - Performance metrics for analysis

## Source

Located in: `src/networkx_mcp/server.py:1099`
