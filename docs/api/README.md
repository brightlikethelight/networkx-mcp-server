# NetworkX MCP Server API Reference

Complete documentation for all 39 graph analysis tools available in the NetworkX MCP Server.

## Overview

- **Total Tools:** 39
- **Categories:** 6
- **Protocol:** Model Context Protocol (MCP)
- **Engine:** NetworkX

## Tools by Category

### Core Operations

- [`add_edges`](tools/add_edges.md) - Add edges to a graph with support for bulk operations, weights, and attributes.
- [`add_nodes`](tools/add_nodes.md) - Add nodes to a graph with support for bulk operations and attributes.
- [`clear_graph`](tools/clear_graph.md) - Clear all nodes and edges from a graph while preserving the graph instance.
- [`create_analysis_workflow`](tools/create_analysis_workflow.md) - Execute analysis workflow with chained operations.
- [`create_dashboard`](tools/create_dashboard.md) - Create interactive dashboard with multiple visualizations.
- [`create_graph`](tools/create_graph.md) - Create a new NetworkX graph with comprehensive initialization options.
- [`delete_graph`](tools/delete_graph.md) - Delete a graph by ID.
- [`get_graph_info`](tools/get_graph_info.md) - Get comprehensive information about a graph.
- [`list_graphs`](tools/list_graphs.md) - List all available graphs.  Returns: List of graphs with their metadata

### Graph Algorithms

- [`calculate_centrality`](tools/calculate_centrality.md) - Calculate various centrality measures for nodes in a graph.
- [`clustering_analysis`](tools/clustering_analysis.md) - Analyze clustering coefficients and triangles in a graph.
- [`connected_components`](tools/connected_components.md) - Find connected components in a graph.
- [`find_all_paths`](tools/find_all_paths.md) - Find all simple paths between two nodes with constraints.
- [`flow_paths`](tools/flow_paths.md) - Analyze flow paths in a directed graph including max flow and edge-disjoint paths.
- [`path_analysis`](tools/path_analysis.md) - Analyze path properties of a graph including diameter, radius, and eccentricity.
- [`shortest_path`](tools/shortest_path.md) - Find shortest path(s) in a graph with multiple algorithm support.

### Advanced Analytics

- [`advanced_community_detection`](tools/advanced_community_detection.md) - Detect communities using advanced algorithms with auto-selection.
- [`bipartite_analysis`](tools/bipartite_analysis.md) - Analyze bipartite graphs with specialized algorithms.
- [`community_detection`](tools/community_detection.md) - Detect communities in a graph.
- [`generate_graph`](tools/generate_graph.md) - Generate synthetic graphs using various models.
- [`generate_report`](tools/generate_report.md) - Generate analysis report in various formats.
- [`network_flow_analysis`](tools/network_flow_analysis.md) - Analyze network flow with multiple algorithms.
- [`robustness_analysis`](tools/robustness_analysis.md) - Analyze network robustness and resilience.

### Visualization

- [`specialized_algorithms`](tools/specialized_algorithms.md) - Run specialized graph algorithms.
- [`visualize_3d`](tools/visualize_3d.md) - Create 3D graph visualization.
- [`visualize_graph`](tools/visualize_graph.md) - Create graph visualizations with various backends.
- [`visualize_graph_simple`](tools/visualize_graph_simple.md) - Generate visualization data for a graph.

### Data Integration

- [`batch_graph_analysis`](tools/batch_graph_analysis.md) - Process multiple graphs with batch operations.
- [`export_graph`](tools/export_graph.md) - Export a graph to various formats.
- [`import_from_source`](tools/import_from_source.md) - Import graph data from various sources with intelligent parsing.
- [`import_graph`](tools/import_graph.md) - Import a graph from various formats.

### Utilities

- [`cycle_detection`](tools/cycle_detection.md) - Detect cycles in a graph with detailed analysis.
- [`directed_graph_analysis`](tools/directed_graph_analysis.md) - Analyze directed graphs with specialized algorithms.
- [`graph_metrics`](tools/graph_metrics.md) - Calculate comprehensive graph metrics and statistics.
- [`graph_statistics`](tools/graph_statistics.md) - Calculate comprehensive graph statistics.
- [`ml_graph_analysis`](tools/ml_graph_analysis.md) - Machine learning-based graph analysis.
- [`monitoring_stats`](tools/monitoring_stats.md) - Get server monitoring statistics and performance metrics.  Returns: Server statistics including: - O...
- [`setup_monitoring`](tools/setup_monitoring.md) - Setup monitoring and alerts for a graph.
- [`subgraph_extraction`](tools/subgraph_extraction.md) - Extract subgraphs based on various criteria.

## Quick Start

```python
from mcp import Client
import asyncio

async def example():
    # Connect to server
    client = Client()
    await client.connect('localhost:8765')

    # Create a graph
    result = await client.call_tool('create_graph', {
        'graph_id': 'my_graph',
        'graph_type': 'undirected'
    })

    # Add some nodes
    await client.call_tool('add_nodes', {
        'graph_id': 'my_graph',
        'nodes': ['A', 'B', 'C', 'D']
    })

    # Add edges
    await client.call_tool('add_edges', {
        'graph_id': 'my_graph',
        'edges': [['A', 'B'], ['B', 'C'], ['C', 'D']]
    })

    # Analyze
    centrality = await client.call_tool('centrality_measures', {
        'graph_id': 'my_graph',
        'centrality_type': 'betweenness'
    })

    print(centrality)

asyncio.run(example())
```

## Need Help?

- [Getting Started Guide](../getting-started.md)
- [Examples](../examples/)
- [Contributing](../CONTRIBUTING.md)
- [Issue Tracker](https://github.com/yourusername/networkx-mcp-server/issues)
