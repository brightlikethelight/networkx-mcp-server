# NetworkX MCP Server API Documentation

This document provides detailed API documentation for all available tools in the NetworkX MCP Server, including Phase 1 core functionality, Phase 2 advanced analytics, and Phase 3 visualization & integration features.

## Table of Contents

### Phase 1: Core Functionality
1. [Graph Management](#graph-management)
2. [Core Algorithms](#core-algorithms)
3. [Advanced Path Analysis](#advanced-path-analysis)
4. [Graph Analysis](#graph-analysis)
5. [Monitoring](#monitoring)
6. [I/O Operations](#io-operations)

### Phase 2: Advanced Analytics
7. [Advanced Community Detection](#advanced-community-detection)
8. [Network Flow Analysis](#network-flow-analysis)
9. [Graph Generation](#graph-generation)
10. [Bipartite Analysis](#bipartite-analysis)
11. [Directed Graph Analysis](#directed-graph-analysis)
12. [Specialized Algorithms](#specialized-algorithms)
13. [Machine Learning Integration](#machine-learning-integration)
14. [Robustness Analysis](#robustness-analysis)

### Phase 3: Visualization & Integration
15. [Graph Visualization](#graph-visualization)
16. [3D Visualization](#3d-visualization)
17. [Interactive Dashboards](#interactive-dashboards)
18. [Data Import Pipeline](#data-import-pipeline)
19. [Batch Graph Analysis](#batch-graph-analysis)
20. [Analysis Workflows](#analysis-workflows)
21. [Report Generation](#report-generation)
22. [Monitoring & Alerts](#monitoring-alerts)

## Graph Management

### create_graph

Create a new graph with specified type and optional initialization data.

**Parameters:**
- `graph_id` (string, required): Unique identifier for the graph
- `graph_type` (string, optional): Type of graph - "undirected", "directed", "multi_undirected", "multi_directed" (default: "undirected")
- `from_data` (object, optional): Initialize graph from data
  - `edge_list` (array): List of edges as [source, target] or [source, target, weight]
  - `adjacency_matrix` (array): 2D adjacency matrix
  - `weighted` (boolean): Whether edges have weights (default: false)
- `**attributes` (object, optional): Additional graph attributes

**Returns:**
```json
{
  "graph_id": "my_graph",
  "graph_type": "Graph",
  "created": true,
  "num_nodes": 10,
  "num_edges": 15,
  "metadata": {
    "created_at": "2024-01-01T12:00:00",
    "attributes": {...}
  }
}
```

**Example:**
```python
await mcp.call_tool("create_graph", {
    "graph_id": "social_network",
    "graph_type": "undirected",
    "from_data": {
        "edge_list": [
            ["Alice", "Bob", 0.8],
            ["Bob", "Charlie", 0.6]
        ],
        "weighted": true
    },
    "name": "Friend Network",
    "description": "Social connections"
})
```

### add_nodes

Add nodes to a graph with optional attributes.

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `nodes` (array, required): List of nodes to add
  - Can be strings/numbers for simple nodes
  - Can be objects with `id` field and additional attributes

**Returns:**
```json
{
  "nodes_added": 5,
  "total_nodes": 15,
  "nodes_already_exist": ["node1", "node2"],
  "execution_time_ms": 12.5
}
```

**Example:**
```python
# Simple nodes
await mcp.call_tool("add_nodes", {
    "graph_id": "my_graph",
    "nodes": ["A", "B", "C", "D"]
})

# Nodes with attributes
await mcp.call_tool("add_nodes", {
    "graph_id": "social_network",
    "nodes": [
        {"id": "Alice", "age": 30, "city": "NYC"},
        {"id": "Bob", "age": 25, "city": "Boston"}
    ]
})
```

### add_edges

Add edges to a graph with optional attributes.

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `edges` (array, required): List of edges to add
  - Each edge must have `source` and `target` fields
  - Can include additional attributes like `weight`

**Returns:**
```json
{
  "edges_added": 10,
  "total_edges": 25,
  "nodes_added": 2,
  "invalid_edges": [],
  "execution_time_ms": 15.3
}
```

**Example:**
```python
await mcp.call_tool("add_edges", {
    "graph_id": "transport_network",
    "edges": [
        {"source": "Station_A", "target": "Station_B", "distance": 5.2, "time": 10},
        {"source": "Station_B", "target": "Station_C", "distance": 3.8, "time": 7}
    ]
})
```

### get_graph_info

Get comprehensive information about a graph.

**Parameters:**
- `graph_id` (string, required): Graph identifier

**Returns:**
```json
{
  "graph_id": "my_graph",
  "graph_type": "Graph",
  "num_nodes": 150,
  "num_edges": 320,
  "density": 0.0286,
  "is_directed": false,
  "is_multigraph": false,
  "degree_stats": {
    "average": 4.27,
    "min": 1,
    "max": 15,
    "std": 2.81
  },
  "metadata": {
    "created_at": "2024-01-01T12:00:00",
    "last_modified": "2024-01-01T13:30:00",
    "attributes": {...}
  }
}
```

## Core Algorithms

### shortest_path

Find shortest path(s) between nodes using various algorithms.

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `source` (string/number, required): Source node
- `target` (string/number, required): Target node
- `weight` (string, optional): Edge attribute to use as weight (default: None for unweighted)
- `method` (string, optional): Algorithm to use - "dijkstra", "bellman-ford", "bidirectional" (default: auto-select)
- `k_paths` (integer, optional): Find k shortest paths (default: 1)

**Returns:**
```json
{
  "path": ["A", "B", "C", "D"],
  "length": 3,
  "weight": 12.5,
  "method": "dijkstra",
  "k_shortest_paths": [
    {"path": ["A", "B", "C", "D"], "length": 3, "weight": 12.5},
    {"path": ["A", "E", "F", "D"], "length": 3, "weight": 13.2}
  ],
  "execution_time_ms": 5.2
}
```

**Example:**
```python
# Unweighted shortest path
await mcp.call_tool("shortest_path", {
    "graph_id": "social_network",
    "source": "Alice",
    "target": "Charlie"
})

# Weighted shortest path with k-paths
await mcp.call_tool("shortest_path", {
    "graph_id": "transport_network",
    "source": "Station_A",
    "target": "Station_Z",
    "weight": "distance",
    "k_paths": 3
})
```

### calculate_centrality

Calculate various centrality measures for nodes.

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `centrality_type` (string/array, required): Type(s) of centrality - "degree", "betweenness", "closeness", "eigenvector", "pagerank"
- `weight` (string, optional): Edge attribute for weighted centrality
- `top_n` (integer, optional): Return only top N nodes
- `normalized` (boolean, optional): Normalize values (default: true)
- `include_statistics` (boolean, optional): Include summary statistics

**Returns:**
```json
{
  "degree_centrality": {
    "Alice": 0.8,
    "Bob": 0.6,
    "Charlie": 0.4
  },
  "betweenness_centrality": {
    "Bob": 0.5,
    "Diana": 0.3,
    "Alice": 0.2
  },
  "statistics": {
    "degree": {"mean": 0.45, "std": 0.21, "min": 0.1, "max": 0.8},
    "betweenness": {"mean": 0.25, "std": 0.15, "min": 0.0, "max": 0.5}
  },
  "execution_time_ms": 25.7
}
```

### clustering_analysis

Analyze clustering coefficients and triangles in the graph.

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `weight` (string, optional): Edge weight for weighted clustering
- `nodes` (array, optional): Specific nodes to analyze (default: all)
- `include_triangles` (boolean, optional): Count triangles (default: false)

**Returns:**
```json
{
  "average_clustering": 0.456,
  "clustering_coefficients": {
    "Alice": 0.667,
    "Bob": 0.5,
    "Charlie": 0.333
  },
  "triangles": 15,
  "transitivity": 0.423,
  "statistics": {
    "min": 0.0,
    "max": 1.0,
    "std": 0.287
  },
  "execution_time_ms": 18.3
}
```

### connected_components

Find connected components in the graph.

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `component_type` (string, optional): "weak" or "strong" for directed graphs (default: "weak")

**Returns:**
```json
{
  "num_components": 3,
  "largest_component_size": 120,
  "component_sizes": [120, 25, 5],
  "is_connected": false,
  "connected_components": [
    ["A", "B", "C", ...],
    ["X", "Y", "Z", ...],
    ["P", "Q"]
  ],
  "execution_time_ms": 8.9
}
```

## Advanced Path Analysis

### find_all_paths

Find all simple paths between two nodes up to a specified length.

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `source` (string/number, required): Source node
- `target` (string/number, required): Target node
- `max_length` (integer, optional): Maximum path length (default: no limit)
- `max_paths` (integer, optional): Maximum number of paths to return (default: 100)

**Returns:**
```json
{
  "num_paths": 5,
  "paths": [
    {"path": ["A", "B", "C"], "length": 2},
    {"path": ["A", "D", "C"], "length": 2},
    {"path": ["A", "B", "E", "C"], "length": 3}
  ],
  "shortest_length": 2,
  "longest_length": 3,
  "truncated": false,
  "execution_time_ms": 12.1
}
```

### path_analysis

Comprehensive path analysis for the entire graph.

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `sample_size` (integer, optional): Number of node pairs to sample (default: 1000)

**Returns:**
```json
{
  "average_shortest_path_length": 3.45,
  "diameter": 8,
  "radius": 4,
  "center_nodes": ["Node_5", "Node_12"],
  "peripheral_nodes": ["Node_1", "Node_99"],
  "path_length_distribution": {
    "1": 150,
    "2": 420,
    "3": 380,
    "4": 210
  },
  "is_connected": true,
  "execution_time_ms": 156.8
}
```

### cycle_detection

Find all cycles in the graph up to a specified length.

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `max_cycle_length` (integer, optional): Maximum cycle length (default: no limit)
- `node` (string/number, optional): Find cycles containing specific node

**Returns:**
```json
{
  "num_cycles": 8,
  "cycles": [
    {"cycle": ["A", "B", "C", "A"], "length": 3},
    {"cycle": ["X", "Y", "Z", "W", "X"], "length": 4}
  ],
  "has_cycles": true,
  "shortest_cycle_length": 3,
  "execution_time_ms": 23.4
}
```

### flow_paths

Analyze flow paths including maximum flow and edge-disjoint paths.

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `source` (string/number, required): Source node
- `target` (string/number, required): Target node
- `capacity` (string, optional): Edge attribute for capacity (default: unit capacity)
- `flow_type` (string, optional): "maximum", "edge_disjoint", "node_disjoint", "all" (default: "maximum")

**Returns:**
```json
{
  "maximum_flow": 25.5,
  "flow_dict": {
    "A": {"B": 10, "C": 15.5},
    "B": {"D": 10},
    "C": {"D": 15.5}
  },
  "edge_disjoint_paths": [
    ["A", "B", "D"],
    ["A", "C", "D"]
  ],
  "num_edge_disjoint": 2,
  "minimum_cut": [["A"], ["B", "C", "D"]],
  "execution_time_ms": 34.2
}
```

## Graph Analysis

### graph_metrics

Calculate comprehensive graph metrics and statistics.

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `include_distributions` (boolean, optional): Include degree and other distributions (default: false)

**Returns:**
```json
{
  "basic_metrics": {
    "num_nodes": 150,
    "num_edges": 450,
    "density": 0.0402,
    "is_directed": false,
    "is_weighted": true,
    "is_connected": true,
    "num_components": 1
  },
  "degree_statistics": {
    "mean": 6.0,
    "std": 2.35,
    "min": 1,
    "max": 15,
    "median": 5.5
  },
  "weight_statistics": {
    "mean": 12.5,
    "std": 5.8,
    "min": 0.5,
    "max": 50.0
  },
  "structural_metrics": {
    "diameter": 12,
    "radius": 6,
    "average_clustering": 0.234,
    "transitivity": 0.219,
    "assortativity": 0.123
  },
  "degree_distribution": {
    "1": 5,
    "2": 12,
    "3": 25,
    "4": 30
  },
  "execution_time_ms": 89.7
}
```

### subgraph_extraction

Extract subgraphs based on various criteria.

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `method` (string, required): Extraction method - "k_hop", "induced", "edge", "condition"
- `nodes` (array, conditional): Nodes for induced/edge subgraph
- `center_node` (string/number, conditional): Center node for k-hop
- `k_hop` (integer, conditional): Number of hops for k-hop method
- `condition` (string, conditional): Condition string for filtering (e.g., "degree > 5")
- `create_new` (boolean, optional): Create as new graph (default: false)
- `new_graph_id` (string, conditional): ID for new graph if create_new is true

**Returns:**
```json
{
  "num_nodes": 45,
  "num_edges": 120,
  "extracted_nodes": ["A", "B", "C", ...],
  "method": "k_hop",
  "created_new_graph": true,
  "new_graph_id": "subgraph_1",
  "execution_time_ms": 15.6
}
```

**Examples:**
```python
# K-hop neighborhood
await mcp.call_tool("subgraph_extraction", {
    "graph_id": "social_network",
    "method": "k_hop",
    "center_node": "Alice",
    "k_hop": 2,
    "create_new": true,
    "new_graph_id": "alice_friends"
})

# Condition-based extraction
await mcp.call_tool("subgraph_extraction", {
    "graph_id": "network",
    "method": "condition",
    "condition": "city = 'NYC' and age > 25",
    "create_new": true,
    "new_graph_id": "nyc_adults"
})
```

## Monitoring

### monitoring_stats

Get comprehensive server performance and operation statistics.

**Parameters:** None

**Returns:**
```json
{
  "performance": {
    "operations": {
      "create_graph": {
        "count": 25,
        "mean_ms": 3.2,
        "std_ms": 1.1,
        "min_ms": 1.5,
        "max_ms": 8.9,
        "total_ms": 80.0
      },
      "shortest_path": {
        "count": 150,
        "mean_ms": 15.7,
        "std_ms": 8.3,
        "min_ms": 2.1,
        "max_ms": 125.6,
        "total_ms": 2355.0
      }
    },
    "total_operations": 500,
    "total_time_ms": 8765.4
  },
  "operations": {
    "total_operations": 500,
    "successful_operations": 498,
    "failed_operations": 2,
    "error_rate": 0.4,
    "operations_by_type": {
      "create_graph": 25,
      "add_nodes": 100,
      "add_edges": 125,
      "shortest_path": 150
    },
    "errors_by_type": {
      "ValidationError": 1,
      "GraphNotFoundError": 1
    },
    "uptime": "2h 15m 30s"
  },
  "memory": {
    "current_usage_mb": 256.7,
    "peak_usage_mb": 512.3,
    "available_mb": 3839.3,
    "graph_memory": {
      "social_network": 45.2,
      "transport_network": 78.9
    }
  },
  "graphs": {
    "total_graphs": 5,
    "graphs_by_type": {
      "Graph": 3,
      "DiGraph": 2
    },
    "total_nodes": 1250,
    "total_edges": 3890
  }
}
```

## I/O Operations

While I/O operations are not directly exposed as MCP tools, they can be accessed through the GraphIOHandler class in code. The server supports the following formats:

### Supported Import/Export Formats

1. **JSON**: Node-link format
2. **CSV**: Edge list format
3. **YAML**: Structured format
4. **GraphML**: XML-based format
5. **GEXF**: Graph Exchange XML Format
6. **Edge List**: Simple text format
7. **Adjacency Matrix**: Matrix representation
8. **Pickle**: Binary Python format
9. **Pajek**: Network analysis format
10. **DOT**: Graphviz format

### Format Auto-detection

The server can automatically detect file formats based on extensions:
- `.json` → JSON
- `.csv` → CSV
- `.yaml`, `.yml` → YAML
- `.graphml` → GraphML
- `.gexf` → GEXF
- `.edges`, `.edgelist` → Edge List
- `.adj` → Adjacency Matrix
- `.pickle`, `.pkl` → Pickle
- `.net`, `.pajek` → Pajek
- `.dot`, `.gv` → DOT

### Streaming Support

For large graphs, the server supports streaming I/O operations:
- Chunked reading for large files
- Progressive writing with configurable chunk size
- Memory-efficient processing

## Error Handling

All tools return structured error responses:

```json
{
  "error": true,
  "error_type": "GraphNotFoundError",
  "message": "Graph 'unknown_graph' not found",
  "details": {
    "available_graphs": ["graph1", "graph2"]
  }
}
```

Common error types:
- `GraphNotFoundError`: Specified graph doesn't exist
- `NodeNotFoundError`: Specified node doesn't exist
- `ValidationError`: Invalid input parameters
- `FormatError`: Invalid file format or data
- `MemoryError`: Insufficient memory for operation

## Performance Considerations

1. **Large Graphs**: Use streaming I/O for graphs with >100k nodes/edges
2. **Path Finding**: Consider using `k_paths` parameter to limit results
3. **Centrality**: Use `top_n` to get only most important nodes
4. **Sampling**: Many analysis tools support sampling for large graphs
5. **Monitoring**: Regularly check `monitoring_stats` to track performance

## Best Practices

1. **Graph IDs**: Use descriptive, unique identifiers
2. **Validation**: Always validate input data before bulk operations
3. **Error Handling**: Implement proper error handling for all operations
4. **Performance**: Monitor operation times and adjust parameters accordingly
5. **Memory**: For large graphs, consider using subgraph extraction
6. **Formats**: Choose appropriate I/O format based on use case:
   - JSON: Human-readable, good for small-medium graphs
   - CSV: Simple edge lists, good for data exchange
   - GraphML/GEXF: Preserve all attributes, good for complex graphs
   - Pickle: Fastest for Python-to-Python transfer

## Phase 2: Advanced Analytics

### Advanced Community Detection

Detect communities using advanced algorithms with automatic algorithm selection based on graph characteristics.

**Tool:** `advanced_community_detection`

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `algorithm` (string, optional): Algorithm selection:
  - `"auto"` (default): Automatically selects best algorithm based on graph size
  - `"louvain"`: Louvain method for modularity optimization
  - `"girvan_newman"`: Edge betweenness-based hierarchical clustering
  - `"spectral"`: Spectral clustering using graph Laplacian
  - `"label_propagation"`: Fast label propagation algorithm
  - `"modularity_max"`: Greedy modularity maximization
- `resolution` (float, optional): Resolution parameter for Louvain (default: 1.0)
- `seed` (integer, optional): Random seed for reproducible results
- Additional algorithm-specific parameters

**Returns:**
```json
{
  "communities": {
    "0": ["Alice", "Bob", "Charlie"],
    "1": ["Diana", "Eve", "Frank"],
    "2": ["Grace", "Henry"]
  },
  "num_communities": 3,
  "modularity": 0.687,
  "algorithm_used": "louvain",
  "quality_metrics": {
    "modularity": 0.687,
    "coverage": 0.95,
    "performance": 0.82
  },
  "community_sizes": [3, 3, 2],
  "execution_time_ms": 45.3
}
```

**Example:**
```python
# Auto-select best algorithm
result = await mcp.call_tool("advanced_community_detection", {
    "graph_id": "social_network",
    "algorithm": "auto",
    "resolution": 1.5
})

# Hierarchical community detection
result = await mcp.call_tool("advanced_community_detection", {
    "graph_id": "network",
    "algorithm": "girvan_newman",
    "num_communities": 5  # Stop at 5 communities
})
```

### Network Flow Analysis

Analyze network flow with multiple algorithms including Ford-Fulkerson, Edmonds-Karp, and Dinic's algorithm.

**Tool:** `network_flow_analysis`

**Parameters:**
- `graph_id` (string, required): Graph identifier (must be directed)
- `source` (string/number, required): Source node
- `sink` (string/number, required): Sink/target node
- `capacity` (string, optional): Edge attribute for capacities (default: "capacity")
- `algorithm` (string, optional): Algorithm selection:
  - `"auto"` (default): Auto-selects based on graph size
  - `"ford_fulkerson"`: Classic Ford-Fulkerson method
  - `"edmonds_karp"`: BFS-based Ford-Fulkerson
  - `"dinic"`: Dinic's algorithm for large graphs
  - `"preflow_push"`: Push-relabel algorithm
- `flow_type` (string, optional): Analysis type:
  - `"max_flow"` (default): Maximum flow computation
  - `"min_cut"`: Minimum cut analysis

**Returns:**
```json
{
  "flow_type": "max_flow",
  "max_flow_value": 23.5,
  "flow_dict": {
    "A": {"B": 10, "C": 13.5},
    "B": {"D": 8, "E": 2},
    "C": {"E": 11.5, "F": 2}
  },
  "algorithm_used": "edmonds_karp",
  "saturated_edges": [["B", "D"], ["E", "sink"]],
  "execution_time_ms": 67.8
}
```

### Graph Generation

Generate synthetic graphs using various theoretical models.

**Tool:** `generate_graph`

**Parameters:**
- `graph_type` (string, required): Type of graph to generate:
  - `"random"`: Erdős-Rényi random graph
  - `"scale_free"`: Barabási-Albert preferential attachment
  - `"small_world"`: Watts-Strogatz small-world model
  - `"regular"`: Regular graphs (k-regular, circulant)
  - `"tree"`: Various tree structures
  - `"geometric"`: Geometric random graphs
  - `"social"`: Social network models
- `n` (integer, required): Number of nodes
- `graph_id` (string, optional): ID for generated graph (auto-generated if not provided)
- Model-specific parameters:
  - Random: `p` (edge probability) or `m` (number of edges)
  - Scale-free: `m` (edges to attach), `power` (exponent)
  - Small-world: `k` (neighbors), `p` (rewiring probability)
  - Regular: `k` (degree)
  - Tree: `branching_factor`, `tree_type`
  - Geometric: `radius`, `dimensions`
  - Social: `communities`, `p_in`, `p_out`

**Returns:**
```json
{
  "graph_id": "scale_free_network",
  "graph_type": "scale_free",
  "num_nodes": 100,
  "num_edges": 294,
  "parameters_used": {
    "m": 3,
    "seed": 42
  },
  "properties": {
    "power_law_exponent": 2.87,
    "clustering_coefficient": 0.037,
    "average_degree": 5.88
  },
  "execution_time_ms": 23.4
}
```

### Bipartite Analysis

Specialized analysis for bipartite graphs including projections, matching, and community detection.

**Tool:** `bipartite_analysis`

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `analysis_type` (string, required): Type of analysis:
  - `"check"`: Check if graph is bipartite
  - `"projection"`: Create weighted projection
  - `"matching"`: Find maximum matching
  - `"clustering"`: Bipartite clustering coefficients
  - `"communities"`: Bipartite community detection
- `weight` (string, optional): Edge weight attribute
- `nodes` (array, conditional): Node set for projection
- Analysis-specific parameters

**Returns vary by analysis type:**

For `"matching"`:
```json
{
  "analysis_type": "matching",
  "matching_size": 45,
  "matching_edges": [["user1", "item5"], ["user2", "item3"], ...],
  "is_perfect_matching": false,
  "total_weight": 156.7,
  "unmatched_nodes": ["user15", "item22"],
  "execution_time_ms": 34.5
}
```

### Directed Graph Analysis

Specialized algorithms for directed graphs including DAG analysis, SCCs, and hierarchy metrics.

**Tool:** `directed_graph_analysis`

**Parameters:**
- `graph_id` (string, required): Graph identifier (must be directed)
- `analysis_type` (string, required): Type of analysis:
  - `"dag_check"`: Check DAG properties and find longest paths
  - `"scc"`: Find strongly connected components
  - `"topological_sort"`: Topological ordering (DAGs only)
  - `"tournament"`: Tournament graph analysis
  - `"bow_tie"`: Web graph bow-tie decomposition
  - `"hierarchy"`: Hierarchy and flow hierarchy metrics
  - `"temporal"`: Temporal network analysis
- Analysis-specific parameters (e.g., `algorithm` for SCC)

**Returns vary by analysis type:**

For `"bow_tie"`:
```json
{
  "analysis_type": "bow_tie",
  "components": {
    "scc": {"size": 1250, "fraction": 0.25},
    "in": {"size": 2000, "fraction": 0.40},
    "out": {"size": 1500, "fraction": 0.30},
    "tendrils": {"size": 200, "fraction": 0.04},
    "disconnected": {"size": 50, "fraction": 0.01}
  },
  "largest_scc_diameter": 15,
  "execution_time_ms": 234.6
}
```

### Specialized Algorithms

Run specialized graph algorithms for optimization and analysis.

**Tool:** `specialized_algorithms`

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `algorithm` (string, required): Algorithm to run:
  - `"spanning_tree"`: MST with Kruskal/Prim/Borůvka
  - `"coloring"`: Graph coloring strategies
  - `"max_clique"`: Maximum clique finding
  - `"matching"`: Various matching algorithms
  - `"vertex_cover"`: Minimum vertex cover
  - `"dominating_set"`: Minimum dominating set
  - `"link_prediction"`: Predict missing links
- Algorithm-specific parameters

**Returns vary by algorithm:**

For `"coloring"`:
```json
{
  "algorithm": "coloring",
  "coloring": {
    "A": 0, "B": 1, "C": 0, "D": 2, "E": 1
  },
  "num_colors_used": 3,
  "strategy": "dsatur",
  "is_valid_coloring": true,
  "lower_bound": 3,
  "upper_bound": 5,
  "color_distribution": {"0": 2, "1": 2, "2": 1},
  "execution_time_ms": 12.3
}
```

### Machine Learning Integration

Machine learning-based graph analysis including embeddings, features, and anomaly detection.

**Tool:** `ml_graph_analysis`

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `analysis_type` (string, required): Type of ML analysis:
  - `"embeddings"`: Generate node embeddings
  - `"features"`: Extract graph features for ML
  - `"similarity"`: Compare two graphs
  - `"anomaly"`: Detect anomalous nodes/edges
- Analysis-specific parameters:
  - Embeddings: `method` (node2vec/deepwalk/spectral), `dimensions`, walk parameters
  - Features: `feature_types` (basic/spectral/graphlet)
  - Similarity: `graph2_id`, `metrics` to compute
  - Anomaly: `method`, `contamination` rate

**Returns vary by analysis type:**

For `"embeddings"`:
```json
{
  "analysis_type": "embeddings",
  "method": "node2vec",
  "dimensions": 64,
  "num_nodes": 150,
  "embedding_stats": {
    "mean": 0.002,
    "std": 0.156,
    "sparsity": 0.0
  },
  "sample_embeddings": {
    "nodeA": [0.234, -0.156, 0.087, ...],
    "nodeB": [-0.123, 0.456, -0.234, ...]
  },
  "parameters": {
    "walk_length": 80,
    "num_walks": 10,
    "p": 1.0,
    "q": 1.0
  },
  "execution_time_ms": 567.8
}
```

### Robustness Analysis

Analyze network robustness and resilience to failures and attacks.

**Tool:** `robustness_analysis`

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `analysis_type` (string, required): Type of robustness analysis:
  - `"attack"`: Simulate node/edge removal attacks
  - `"percolation"`: Percolation threshold analysis
  - `"cascading"`: Cascading failure simulation
  - `"resilience"`: Comprehensive resilience metrics
- Analysis-specific parameters:
  - Attack: `attack_type` (random/targeted), `fraction` to remove
  - Percolation: `percolation_type` (site/bond), probability range
  - Cascading: `initial_failures`, `failure_model`
  - Resilience: `resilience_metrics` to compute

**Returns vary by analysis type:**

For `"attack"`:
```json
{
  "analysis_type": "attack",
  "attack_type": "targeted_degree",
  "fraction_removed": 0.2,
  "initial_metrics": {
    "is_connected": true,
    "largest_component_size": 500,
    "global_efficiency": 0.234
  },
  "final_metrics": {
    "is_connected": false,
    "largest_component_size": 245,
    "global_efficiency": 0.087
  },
  "robustness_index": 0.156,
  "critical_fraction": 0.15,
  "removal_sequence": [
    {
      "step": 1,
      "removed_node": "hub1",
      "fraction_removed": 0.002,
      "largest_component_size": 498
    }
  ],
  "execution_time_ms": 234.5
}
```

## Advanced Features

### Algorithm Auto-Selection

Many Phase 2 tools support automatic algorithm selection based on graph characteristics:

- **Community Detection**: Selects based on graph size and density
- **Network Flow**: Chooses algorithm based on number of edges
- **Robustness Analysis**: Adapts sampling strategies for large graphs

### Performance Optimization

Phase 2 tools include several optimizations:

- **Sampling**: Large graph methods use intelligent sampling
- **Approximation**: Fast approximation algorithms available
- **Parallel Processing**: Some algorithms support parallel execution
- **Incremental Updates**: Support for dynamic graph updates

### Integration Examples

**Complete Network Analysis Pipeline:**
```python
# 1. Generate a scale-free network
await mcp.call_tool("generate_graph", {
    "graph_type": "scale_free",
    "n": 1000,
    "m": 3,
    "graph_id": "test_network"
})

# 2. Detect communities
communities = await mcp.call_tool("advanced_community_detection", {
    "graph_id": "test_network",
    "algorithm": "auto"
})

# 3. Generate embeddings for ML
embeddings = await mcp.call_tool("ml_graph_analysis", {
    "graph_id": "test_network",
    "analysis_type": "embeddings",
    "method": "node2vec",
    "dimensions": 32
})

# 4. Test robustness
robustness = await mcp.call_tool("robustness_analysis", {
    "graph_id": "test_network",
    "analysis_type": "attack",
    "attack_type": "targeted_degree",
    "fraction": 0.1
})
```

## Phase 3: Visualization & Integration

### Graph Visualization

Create static or interactive graph visualizations using multiple backend engines.

**Tool:** `visualize_graph`

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `visualization_type` (string, optional): Type of visualization:
  - `"static"` (default): Static image using Matplotlib
  - `"interactive"`: Interactive HTML using PyVis
  - `"plotly"`: Interactive plot using Plotly
  - `"specialized"`: Special visualizations (heatmap, chord, etc.)
- `layout` (string, optional): Layout algorithm:
  - For static: `"spring"` (default), `"circular"`, `"shell"`, `"kamada_kawai"`, `"hierarchical"`
  - For interactive: `"force_atlas"`, `"barnes_hut"`, `"hierarchical"`, `"random"`
- `format` (string, optional): Output format:
  - For static: `"png"` (default), `"svg"`, `"pdf"`
  - For interactive: `"html"` (default), `"json"`
- Styling parameters:
  - `node_size`: Size of nodes (number, dict, or attribute name)
  - `node_color`: Node colors (string, dict, or attribute name)
  - `edge_width`: Edge widths (number, dict, or attribute name)
  - `edge_color`: Edge colors (string, dict, or attribute name)
  - `show_labels`: Show node labels (boolean, default: true)
  - `title`: Visualization title
  - `figsize`: Figure size tuple for static plots
  - `physics`: Enable physics simulation for interactive (boolean)

**Returns:**
```json
{
  "visualization_type": "interactive",
  "format": "html",
  "data": "<html>...</html>",
  "layout_used": "force_atlas",
  "num_nodes": 150,
  "num_edges": 450,
  "file_path": "/tmp/graph_viz_123.html",
  "execution_time_ms": 234.5
}
```

**Examples:**
```python
# Static visualization with custom styling
result = await mcp.call_tool("visualize_graph", {
    "graph_id": "social_network",
    "visualization_type": "static",
    "layout": "kamada_kawai",
    "node_size": "degree",  # Size by degree
    "node_color": "community",  # Color by community
    "edge_width": 2,
    "format": "png",
    "title": "Social Network Structure"
})

# Interactive physics-based visualization
result = await mcp.call_tool("visualize_graph", {
    "graph_id": "network",
    "visualization_type": "interactive",
    "layout": "force_atlas",
    "physics": true,
    "node_size": 300,
    "edge_color": "gray"
})

# Specialized visualization - heatmap
result = await mcp.call_tool("visualize_graph", {
    "graph_id": "correlation_network",
    "visualization_type": "specialized",
    "plot_type": "heatmap",
    "colormap": "viridis"
})
```

### 3D Visualization

Generate three-dimensional graph visualizations with Plotly.

**Tool:** `visualize_3d`

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `layout` (string, optional): 3D layout algorithm:
  - `"spring_3d"` (default): 3D force-directed layout
  - `"random_3d"`: Random 3D positions
  - `"sphere"`: Nodes on sphere surface
  - `"cube"`: Nodes in cube arrangement
- `node_color` (string/dict, optional): Node colors (attribute name or mapping)
- `node_size` (number/dict, optional): Node sizes (default: 5)
- `edge_color` (string/dict, optional): Edge colors
- `show_labels` (boolean, optional): Show node labels (default: false)
- `camera` (object, optional): Camera position settings
- `animation` (object, optional): Animation settings for temporal networks

**Returns:**
```json
{
  "format": "html",
  "data": "<html>...</html>",
  "layout_used": "spring_3d",
  "num_nodes": 100,
  "num_edges": 250,
  "bounds": {
    "x": [-10.5, 10.3],
    "y": [-8.2, 9.7],
    "z": [-7.5, 8.8]
  },
  "file_path": "/tmp/graph_3d_456.html",
  "execution_time_ms": 156.7
}
```

**Example:**
```python
# 3D molecular network
result = await mcp.call_tool("visualize_3d", {
    "graph_id": "molecule",
    "layout": "spring_3d",
    "node_color": "element",
    "node_size": {"C": 8, "H": 4, "O": 10},
    "edge_color": "bond_type",
    "show_labels": true,
    "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}}
})
```

### Interactive Dashboards

Create comprehensive dashboards combining multiple visualizations.

**Tool:** `create_dashboard`

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `components` (array, required): List of dashboard components:
  - `type`: Component type ("graph", "metrics", "distribution", "timeline")
  - `config`: Component-specific configuration
- `layout` (string, optional): Dashboard layout ("grid", "tabs", "vertical")
- `title` (string, optional): Dashboard title
- `refresh_interval` (integer, optional): Auto-refresh interval in seconds

**Returns:**
```json
{
  "dashboard_id": "dashboard_789",
  "url": "http://localhost:8050/dashboard_789",
  "components_created": 4,
  "layout": "grid",
  "execution_time_ms": 456.7
}
```

### Data Import Pipeline

Import graphs from various data sources with intelligent parsing.

**Tool:** `import_from_source`

**Parameters:**
- `source_type` (string, required): Type of data source:
  - `"csv"`: CSV files with edge lists or adjacency
  - `"json"`: JSON in various formats
  - `"database"`: SQL database queries
  - `"api"`: REST API endpoints
  - `"excel"`: Multi-sheet Excel files
  - `"stream"`: Real-time data streams
- `path` (string, conditional): File path or connection string
- `graph_id` (string, optional): ID for imported graph (auto-generated if not provided)
- Source-specific parameters:
  - CSV: `edge_columns`, `delimiter`, `type_inference`
  - JSON: `format_type` ("node_link", "adjacency", "tree")
  - Database: `query`, `db_type` ("sqlite", "postgresql", "mysql")
  - API: `endpoints`, `rate_limit`, `pagination`
  - Excel: `sheet_mapping`

**Returns:**
```json
{
  "graph_id": "imported_network",
  "source_type": "csv",
  "num_nodes": 1250,
  "num_edges": 5680,
  "attributes_detected": ["weight", "timestamp", "category"],
  "data_types": {
    "source": "string",
    "target": "string",
    "weight": "float64",
    "timestamp": "datetime64"
  },
  "processing_time": 2.34,
  "warnings": [],
  "execution_time_ms": 2340.5
}
```

**Examples:**
```python
# Import from CSV with type inference
result = await mcp.call_tool("import_from_source", {
    "source_type": "csv",
    "path": "network_data.csv",
    "graph_id": "imported_network",
    "type_inference": true,
    "edge_columns": ["from_user", "to_user"]
})

# Import from database
result = await mcp.call_tool("import_from_source", {
    "source_type": "database",
    "path": "postgresql://user:pass@host/db",
    "query": "SELECT source, target, weight FROM edges WHERE active = true",
    "db_type": "postgresql",
    "graph_id": "db_network"
})

# Import from REST API
result = await mcp.call_tool("import_from_source", {
    "source_type": "api",
    "path": "https://api.example.com",
    "endpoints": [
        {
            "path": "/nodes",
            "type": "nodes",
            "id_field": "id",
            "pagination_type": "page"
        },
        {
            "path": "/edges",
            "type": "edges",
            "source_field": "from",
            "target_field": "to"
        }
    ],
    "rate_limit": 1.0,
    "graph_id": "api_network"
})
```

### Batch Graph Analysis

Process multiple graphs in parallel with comprehensive operations.

**Tool:** `batch_graph_analysis`

**Parameters:**
- `graph_ids` (array, required): List of graph identifiers to process
- `operations` (array, required): List of operations to perform on each graph:
  - `name`: Operation name for results
  - `type`: Operation type ("metrics", "centrality", "community", etc.)
  - `params`: Operation-specific parameters
- `parallel` (boolean, optional): Use parallel processing (default: true)
- `batch_size` (integer, optional): Batch size for processing (default: 10)
- `output_format` (string, optional): Results format ("summary", "detailed")

**Returns:**
```json
{
  "results": {
    "graph1": {
      "metrics": {"density": 0.045, "clustering": 0.234},
      "centrality": {"top_nodes": ["A", "B", "C"]},
      "communities": {"num_communities": 5, "modularity": 0.67}
    },
    "graph2": {
      "metrics": {"density": 0.032, "clustering": 0.189},
      "centrality": {"top_nodes": ["X", "Y", "Z"]},
      "communities": {"num_communities": 3, "modularity": 0.71}
    }
  },
  "summary": {
    "total_graphs": 2,
    "successful": 2,
    "failed": 0,
    "total_time": 5.67,
    "avg_time_per_graph": 2.84
  },
  "batch_config": {
    "parallel": true,
    "batch_size": 10,
    "max_workers": 4
  },
  "execution_time_ms": 5670.3
}
```

**Example:**
```python
# Batch analysis of multiple networks
result = await mcp.call_tool("batch_graph_analysis", {
    "graph_ids": ["network1", "network2", "network3"],
    "operations": [
        {
            "name": "basic_metrics",
            "type": "metrics",
            "params": {"include_distributions": true}
        },
        {
            "name": "pagerank",
            "type": "centrality",
            "params": {"centrality_type": "pagerank", "top_n": 10}
        },
        {
            "name": "communities",
            "type": "community",
            "params": {"algorithm": "louvain"}
        }
    ],
    "parallel": true,
    "output_format": "summary"
})
```

### Analysis Workflows

Create and execute complex analysis workflows with caching and conditional logic.

**Tool:** `create_analysis_workflow`

**Parameters:**
- `graph_id` (string, required): Graph identifier
- `workflow` (array, required): List of workflow steps:
  - `name`: Step name
  - `operation`: Operation to perform
  - `params`: Operation parameters
  - `modifies_graph`: Whether step modifies the graph
  - `stop_condition`: Optional condition to stop workflow
- `cache_intermediate` (boolean, optional): Cache intermediate results (default: true)
- `save_results` (boolean, optional): Save workflow results

**Returns:**
```json
{
  "workflow_results": {
    "preprocess": {"nodes_removed": 45, "edges_removed": 120},
    "centrality": {"top_node": "hub1", "max_centrality": 0.89},
    "communities": {"num_communities": 8, "modularity": 0.72}
  },
  "final_graph": {
    "num_nodes": 455,
    "num_edges": 1230
  },
  "steps_completed": 3,
  "total_time": 8.9,
  "cache_hits": 1,
  "workflow_id": "workflow_abc123",
  "execution_time_ms": 8900.5
}
```

**Example:**
```python
# Complex analysis workflow
result = await mcp.call_tool("create_analysis_workflow", {
    "graph_id": "raw_network",
    "workflow": [
        {
            "name": "filter_nodes",
            "operation": "filter",
            "params": {"min_degree": 5},
            "modifies_graph": true
        },
        {
            "name": "find_core",
            "operation": "k_core",
            "params": {"k": 3},
            "modifies_graph": true
        },
        {
            "name": "analyze_centrality",
            "operation": "centrality",
            "params": {"type": "betweenness", "top_n": 20}
        },
        {
            "name": "detect_communities",
            "operation": "community",
            "params": {"algorithm": "louvain", "resolution": 1.2},
            "stop_condition": {
                "field": "num_communities",
                "operator": "gt",
                "value": 10
            }
        }
    ],
    "cache_intermediate": true
})
```

### Report Generation

Generate comprehensive PDF or HTML reports with visualizations and analysis results.

**Tool:** `generate_report`

**Parameters:**
- `graph_id` (string, required): Graph identifier or analysis results ID
- `format` (string, optional): Output format ("pdf", "html") (default: "pdf")
- `template` (string, optional): Report template ("default", "academic", "business")
- `sections` (array, optional): Sections to include:
  - `"summary"`: Executive summary
  - `"metrics"`: Key metrics and statistics
  - `"visualizations"`: Graph visualizations
  - `"centrality"`: Centrality analysis
  - `"communities"`: Community structure
  - `"recommendations"`: Analysis recommendations
- `include_visualizations` (boolean, optional): Include graph plots (default: true)
- `metadata` (object, optional): Additional metadata for report

**Returns:**
```json
{
  "format": "pdf",
  "file_path": "/tmp/network_report_20240115.pdf",
  "file_size_mb": 2.5,
  "pages": 12,
  "sections_included": ["summary", "metrics", "visualizations", "communities"],
  "report_id": "report_xyz789",
  "generation_time_ms": 3456.7
}
```

**Example:**
```python
# Generate comprehensive PDF report
result = await mcp.call_tool("generate_report", {
    "graph_id": "analyzed_network",
    "format": "pdf",
    "template": "academic",
    "sections": [
        "summary",
        "metrics",
        "visualizations",
        "centrality",
        "communities",
        "recommendations"
    ],
    "include_visualizations": true,
    "metadata": {
        "title": "Social Network Analysis Report",
        "author": "Data Science Team",
        "date": "2024-01-15"
    }
})
```

### Monitoring & Alerts

Set up real-time monitoring and alerting for graph metrics.

**Tool:** `setup_monitoring`

**Parameters:**
- `graph_id` (string, required): Graph identifier to monitor
- `alert_rules` (array, required): List of alert rules:
  - `name`: Rule name
  - `type`: Alert type ("threshold", "anomaly", "pattern")
  - `metric`: Metric to monitor
  - Rule-specific parameters:
    - Threshold: `threshold`, `operator` ("gt", "lt", "eq")
    - Anomaly: `sensitivity` (standard deviations)
    - Pattern: `pattern` type ("component_split", "density_drop")
  - `severity`: Alert severity ("low", "medium", "high")
- `check_interval` (integer, optional): Check interval in seconds (default: 60)
- `notification_webhook` (string, optional): Webhook URL for notifications

**Returns:**
```json
{
  "monitoring_id": "monitor_123",
  "graph_id": "production_network",
  "rules_configured": 3,
  "status": "active",
  "current_metrics": {
    "density": 0.045,
    "avg_degree": 5.6,
    "num_components": 1
  },
  "alerts_triggered": [],
  "next_check": "2024-01-15T12:35:00Z",
  "execution_time_ms": 123.4
}
```

**Example:**
```python
# Setup comprehensive monitoring
result = await mcp.call_tool("setup_monitoring", {
    "graph_id": "production_network",
    "alert_rules": [
        {
            "name": "high_density_alert",
            "type": "threshold",
            "metric": "density",
            "threshold": 0.8,
            "operator": "gt",
            "severity": "medium"
        },
        {
            "name": "component_split_alert",
            "type": "pattern",
            "pattern": "component_split",
            "severity": "high"
        },
        {
            "name": "degree_anomaly",
            "type": "anomaly",
            "metric": "avg_degree",
            "sensitivity": 2.5,
            "severity": "medium"
        }
    ],
    "check_interval": 300,
    "notification_webhook": "https://alerts.example.com/webhook"
})
```

## Phase 3 Advanced Features

### Visualization Intelligence

The visualization tools include intelligent features:

- **Layout Auto-Selection**: Chooses optimal layout based on graph properties
- **Adaptive Styling**: Automatically adjusts visual parameters for clarity
- **Performance Optimization**: Uses sampling for very large graphs
- **Multi-Format Export**: Supports various output formats for different use cases

### Data Pipeline Intelligence

Data import pipelines include:

- **Type Inference**: Automatically detects data types in CSV/Excel
- **Format Detection**: Auto-detects JSON structure and graph format
- **Schema Mapping**: Intelligent mapping of database schemas to graph structure
- **Error Recovery**: Handles malformed data with detailed error reporting

### Enterprise Capabilities

- **Scalability**: Batch processing supports graphs with millions of nodes
- **Caching**: Intelligent caching for workflow intermediate results
- **Versioning**: Built-in graph versioning system
- **Scheduling**: Schedule periodic analysis runs
- **Audit Trail**: Complete logging of all operations

### Integration Patterns

**Complete Analysis Pipeline with Visualization:**
```python
# 1. Import data from multiple sources
csv_result = await mcp.call_tool("import_from_source", {
    "source_type": "csv",
    "path": "edges.csv",
    "graph_id": "combined_network"
})

# 2. Run analysis workflow
workflow_result = await mcp.call_tool("create_analysis_workflow", {
    "graph_id": "combined_network",
    "workflow": [
        {"name": "clean", "operation": "remove_isolates"},
        {"name": "analyze", "operation": "full_analysis"}
    ]
})

# 3. Generate visualizations
viz_result = await mcp.call_tool("visualize_graph", {
    "graph_id": "combined_network",
    "visualization_type": "interactive",
    "layout": "force_atlas",
    "node_color": "community"
})

# 4. Create report
report = await mcp.call_tool("generate_report", {
    "graph_id": "combined_network",
    "format": "pdf",
    "include_visualizations": true
})

# 5. Setup monitoring
monitoring = await mcp.call_tool("setup_monitoring", {
    "graph_id": "combined_network",
    "alert_rules": [{"name": "density_check", "type": "threshold", "metric": "density", "threshold": 0.9}]
})
```
