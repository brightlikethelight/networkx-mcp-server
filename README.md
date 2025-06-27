# NetworkX MCP Server

A comprehensive Model Context Protocol (MCP) server for NetworkX graph operations, providing powerful graph analysis capabilities through a standardized interface. Built with FastMCP, this server enables AI assistants and other MCP clients to create, manipulate, analyze, and visualize complex networks and graphs with advanced features including performance monitoring, bulk operations, and streaming support.

## Features

### Core Capabilities
- **Graph Management**: Create and manage multiple graph types (Graph, DiGraph, MultiGraph, MultiDiGraph)
- **Bulk Operations**: Efficient bulk addition of nodes and edges with detailed statistics
- **Advanced I/O**: Support for 10+ formats with auto-detection and streaming capabilities
- **Comprehensive Algorithms**: 15+ graph algorithms including paths, centrality, clustering, and community detection
- **Performance Monitoring**: Built-in tracking of operation times, counts, and memory usage
- **CLI Interface**: Interactive command-line interface for testing and debugging
- **Input Validation**: Comprehensive validation for all operations

### Graph Operations
- **Graph Management**: Create, delete, list, and clear graphs with initialization from data
- **Node Operations**: Add, remove, update nodes with bulk operations and validation
- **Edge Operations**: Add, remove, update edges with bulk operations and detailed statistics
- **Subgraph Extraction**: Extract subgraphs using k-hop, induced, edge, or condition-based methods
- **Attributes**: Get and set node/edge attributes dynamically with validation

### Algorithms
- **Shortest Paths**: Dijkstra, Bellman-Ford, bidirectional search with k-shortest paths support
- **Centrality Measures**: Degree, betweenness, closeness, eigenvector, PageRank with top-N filtering
- **Community Detection**: Advanced algorithms with auto-selection (Louvain, Girvan-Newman, spectral, label propagation)
- **Clustering Analysis**: Clustering coefficients, triangles, and local clustering
- **Path Analysis**: All simple paths, path statistics, reachability analysis
- **Flow Algorithms**: Maximum flow, minimum cut, edge-disjoint paths with multiple algorithm support
- **Cycle Detection**: Find all cycles up to specified length
- **Connected Components**: Weak and strong components with detailed analysis
- **Graph Metrics**: Comprehensive statistics including distributions and structural properties

### Advanced Analytics (Phase 2)
- **Network Flow**: Ford-Fulkerson, Edmonds-Karp, Dinic's algorithm with auto-selection
- **Graph Generators**: Random, scale-free (BA), small-world (WS), regular, trees, geometric, social networks
- **Bipartite Analysis**: Projections, maximum matching, clustering, specialized community detection
- **Directed Graph Analysis**: DAG properties, SCCs, tournament graphs, bow-tie structure, hierarchy metrics
- **Specialized Algorithms**: Spanning trees, graph coloring, maximum clique, vertex cover, dominating set
- **Machine Learning**: Node embeddings (Node2Vec, DeepWalk, spectral), graph features, anomaly detection
- **Robustness Analysis**: Attack simulation, percolation analysis, cascading failures, resilience metrics

### Visualization & Integration (Phase 3)
- **Visualization Engines**: Multiple backends for different use cases
  - Matplotlib: Static plots with various layouts (spring, circular, hierarchical, etc.)
  - Plotly: Interactive 2D/3D visualizations with hover details and animations
  - PyVis: Physics-based interactive networks using vis.js
  - Specialized: Heatmaps, chord diagrams, Sankey flows, dendrograms
- **Data Integration Pipelines**: Intelligent data import from multiple sources
  - CSV: Auto-detection of edge columns, type inference
  - JSON: Support for node-link, adjacency, tree, and custom formats
  - Database: SQL queries to graph with PostgreSQL, MySQL, SQLite support
  - API: REST API integration with pagination and rate limiting
  - Excel: Multi-sheet processing with automatic mapping
  - Streaming: Real-time data ingestion with windowing
- **Enterprise Features**: Production-grade capabilities
  - Batch Processing: Parallel analysis of multiple graphs
  - Workflow Orchestration: Chain operations with caching
  - Report Generation: Automated PDF/HTML reports with visualizations
  - Monitoring & Alerts: Anomaly detection and threshold-based alerting
  - Scheduling: Periodic analysis execution
  - Versioning: Graph version control with metadata

### I/O Support
- **Import/Export Formats**: JSON, CSV, YAML, GraphML, GEXF, edge list, adjacency matrix, pickle, Pajek, DOT
- **Format Auto-detection**: Automatic format detection from file extensions
- **Streaming Support**: Efficient handling of large graphs with chunked I/O
- **Format Converters**: CSV to edge list, DataFrame to graph, adjacency to edge list
- **Pandas Integration**: Import/export graphs as DataFrames with full attribute support
- **Visualization**: Generate layout data for PyVis, Plotly, and Matplotlib

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/networkx-mcp-server.git
cd networkx-mcp-server

# Install with pip (recommended)
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Starting the Server

```bash
# Using the command line entry point
networkx-mcp

# Or using Python module
python -m networkx_mcp.server

# Or use the CLI for interactive testing
python -m networkx_mcp.cli
```

The server will start on `http://localhost:8000` by default.

### Basic Usage Example

```python
# Example MCP client usage (pseudo-code)
client = MCPClient("http://localhost:8000")

# Create a graph with initialization from data
await client.call_tool("create_graph", {
    "graph_id": "social_network",
    "graph_type": "undirected",
    "from_data": {
        "edge_list": [["Alice", "Bob"], ["Bob", "Charlie"], ["Charlie", "Diana"]],
        "weighted": false
    }
})

# Add nodes with bulk operation
await client.call_tool("add_nodes", {
    "graph_id": "social_network",
    "nodes": [
        {"id": "Eve", "age": 30, "city": "NYC"},
        {"id": "Frank", "age": 25, "city": "Boston"},
        {"id": "Grace", "age": 35, "city": "Chicago"}
    ]
})

# Add edges with bulk operation and attributes
await client.call_tool("add_edges", {
    "graph_id": "social_network",
    "edges": [
        {"source": "Diana", "target": "Eve", "weight": 0.8, "since": 2020},
        {"source": "Eve", "target": "Frank", "weight": 0.6, "since": 2021},
        {"source": "Frank", "target": "Grace", "weight": 0.9, "since": 2019}
    ]
})

# Find shortest path with k-paths
result = await client.call_tool("shortest_path", {
    "graph_id": "social_network",
    "source": "Alice",
    "target": "Grace",
    "k_paths": 3  # Find top 3 shortest paths
})
# Result: {
#   "path": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"],
#   "length": 6,
#   "k_shortest_paths": [...],
#   "method": "dijkstra"
# }

# Calculate multiple centrality measures
centrality = await client.call_tool("calculate_centrality", {
    "graph_id": "social_network",
    "centrality_type": ["degree", "betweenness", "pagerank"],
    "top_n": 5
})

# Get comprehensive graph metrics
metrics = await client.call_tool("graph_metrics", {
    "graph_id": "social_network",
    "include_distributions": true
})

# Phase 2: Advanced Analytics Examples

# Generate a scale-free network
await client.call_tool("generate_graph", {
    "graph_type": "scale_free",
    "n": 100,
    "graph_id": "scale_free_network",
    "m": 3,  # Edges to attach from new node
    "seed": 42
})

# Advanced community detection with auto-selection
communities = await client.call_tool("advanced_community_detection", {
    "graph_id": "social_network",
    "algorithm": "auto",  # Auto-selects based on graph size
    "resolution": 1.0
})

# Network flow analysis
flow_result = await client.call_tool("network_flow_analysis", {
    "graph_id": "transport_network",
    "source": "Station_A",
    "sink": "Station_Z",
    "capacity": "capacity",
    "algorithm": "auto",  # Auto-selects optimal algorithm
    "flow_type": "max_flow"
})

# ML-based node embeddings
embeddings = await client.call_tool("ml_graph_analysis", {
    "graph_id": "social_network",
    "analysis_type": "embeddings",
    "method": "node2vec",
    "dimensions": 64,
    "walk_length": 80,
    "num_walks": 10
})

# Network robustness analysis
robustness = await client.call_tool("robustness_analysis", {
    "graph_id": "infrastructure_network",
    "analysis_type": "attack",
    "attack_type": "targeted_degree",  # Target high-degree nodes
    "fraction": 0.2  # Remove 20% of nodes
})

# Bipartite graph analysis
bipartite_result = await client.call_tool("bipartite_analysis", {
    "graph_id": "user_item_network",
    "analysis_type": "matching",
    "weight": "preference"  # Weighted matching
})

# Phase 3: Visualization & Integration Examples

# Create interactive visualization
viz_result = await client.call_tool("visualize_graph", {
    "graph_id": "social_network",
    "visualization_type": "interactive",
    "layout": "force_atlas",
    "node_size": "degree",
    "node_color": "community",
    "physics": True
})

# Generate 3D visualization
viz_3d = await client.call_tool("visualize_3d", {
    "graph_id": "molecule_network",
    "layout": "spring_3d",
    "node_color": "element_type",
    "edge_color": "bond_type",
    "show_labels": True
})

# Import data from multiple sources
import_result = await client.call_tool("import_from_source", {
    "source_type": "csv",
    "path": "network_data.csv",
    "graph_id": "imported_network",
    "type_inference": True,
    "edge_columns": ["source", "target"]
})

# Batch analysis of multiple graphs
batch_result = await client.call_tool("batch_graph_analysis", {
    "graph_ids": ["network1", "network2", "network3"],
    "operations": [
        {"name": "metrics", "type": "metrics"},
        {"name": "centrality", "type": "centrality", "params": {"type": "pagerank"}},
        {"name": "communities", "type": "community"}
    ],
    "parallel": True
})

# Create analysis workflow
workflow = await client.call_tool("create_analysis_workflow", {
    "graph_id": "social_network",
    "workflow": [
        {"name": "preprocess", "operation": "filter_nodes", "params": {"min_degree": 5}},
        {"name": "analyze", "operation": "centrality", "params": {"type": "betweenness"}},
        {"name": "detect", "operation": "community", "params": {"algorithm": "louvain"}}
    ],
    "cache_intermediate": True
})

# Generate comprehensive report
report = await client.call_tool("generate_report", {
    "graph_id": "analysis_results",
    "format": "pdf",
    "include_visualizations": True,
    "sections": ["summary", "metrics", "centrality", "communities", "recommendations"]
})

# Setup monitoring and alerts
monitoring = await client.call_tool("setup_monitoring", {
    "graph_id": "production_network",
    "alert_rules": [
        {
            "name": "high_density",
            "type": "threshold",
            "metric": "density",
            "threshold": 0.8,
            "operator": "gt"
        },
        {
            "name": "component_split",
            "type": "pattern",
            "pattern": "component_split",
            "severity": "high"
        }
    ]
})
```

## Available Tools

### Graph Management
- `create_graph`: Create a new graph with optional initialization from data
- `add_nodes`: Add nodes with bulk operation support and validation
- `add_edges`: Add edges with bulk operation support and detailed statistics
- `get_graph_info`: Get comprehensive graph information including metrics

### Core Algorithms
- `shortest_path`: Find shortest path(s) with multiple algorithms and k-paths support
- `calculate_centrality`: Calculate various centrality measures with top-N filtering
- `clustering_analysis`: Analyze clustering coefficients and triangles
- `connected_components`: Find weak/strong components with detailed analysis

### Advanced Path Analysis
- `find_all_paths`: Find all simple paths up to specified length
- `path_analysis`: Comprehensive path statistics and analysis
- `cycle_detection`: Find all cycles up to specified length
- `flow_paths`: Maximum flow, minimum cut, and edge-disjoint paths

### Graph Analysis
- `graph_metrics`: Comprehensive graph metrics with distributions
- `subgraph_extraction`: Extract subgraphs using various methods
- `community_detection`: Detect communities (when available)

### Phase 2: Advanced Analytics

#### Community & Network Flow
- `advanced_community_detection`: Advanced community detection with auto-algorithm selection
- `network_flow_analysis`: Network flow analysis with Ford-Fulkerson, Edmonds-Karp, Dinic algorithms

#### Graph Generation & Structure
- `generate_graph`: Generate synthetic graphs (random, scale-free, small-world, regular, tree, geometric, social)
- `bipartite_analysis`: Analyze bipartite graphs (check, projection, matching, clustering, communities)
- `directed_graph_analysis`: Directed graph analysis (DAG check, SCCs, topological sort, tournament, bow-tie, hierarchy)

#### Specialized Algorithms & ML
- `specialized_algorithms`: Run specialized algorithms (spanning_tree, coloring, max_clique, matching, vertex_cover, dominating_set, link_prediction)
- `ml_graph_analysis`: Machine learning analysis (embeddings, features, similarity, anomaly detection)
- `robustness_analysis`: Network robustness analysis (attack simulation, percolation, cascading failures, resilience)

### Phase 3: Visualization & Integration

#### Visualization
- `visualize_graph`: Create static or interactive graph visualizations with multiple backends
- `visualize_3d`: Generate 3D graph visualizations with Plotly
- `create_dashboard`: Build interactive dashboards combining multiple visualizations

#### Data Integration
- `import_from_source`: Import graphs from various sources (CSV, JSON, database, API, Excel)

#### Enterprise Operations
- `batch_graph_analysis`: Process multiple graphs in parallel with comprehensive operations
- `create_analysis_workflow`: Chain operations with caching and conditional execution
- `generate_report`: Create PDF/HTML reports with visualizations and analysis results
- `setup_monitoring`: Configure real-time monitoring and alerting for graph metrics

### Monitoring
- `monitoring_stats`: Get server performance and operation statistics

### I/O Operations (via direct GraphIOHandler usage)
- Import formats: JSON, CSV, YAML, GraphML, GEXF, edge list, adjacency matrix, pickle, Pajek, DOT
- Export formats: Same as import, with pretty printing and streaming support
- Format converters: CSV to edge list, DataFrame to graph, adjacency to edge list

## CLI Interface

The NetworkX MCP server includes a comprehensive CLI for testing and interactive use:

```bash
# Start interactive CLI
python -m networkx_mcp.cli

# Run benchmark with 1000 nodes
python -m networkx_mcp.cli --benchmark 1000

# Run demo (social, transport, or citation)
python -m networkx_mcp.cli --demo social
```

### CLI Commands

**Graph Management:**
- `create <graph_id> [type]` - Create a new graph
- `list` - List all graphs
- `info [graph_id]` - Show graph information
- `select <graph_id>` - Select active graph
- `delete <graph_id>` - Delete a graph
- `clear <graph_id>` - Clear graph contents

**Graph Building:**
- `add nodes <n1> <n2> ...` - Add nodes
- `add edge <source> <target>` - Add edge
- `import <format> <file>` - Import graph from file
- `export <format> <file>` - Export graph to file

**Analysis:**
- `analyze centrality [type]` - Calculate centrality
- `analyze path <src> <dst>` - Find shortest path
- `analyze components` - Find connected components
- `analyze metrics` - Calculate graph metrics
- `analyze communities` - Detect communities

**Other:**
- `monitor` - Show performance statistics
- `benchmark <size>` - Run performance benchmark
- `demo <example>` - Run demo (social/transport/citation)
- `help` - Show help
- `exit` - Exit CLI

## Examples

The repository includes three comprehensive examples demonstrating real-world use cases:

### 1. Social Network Analysis (`examples/social_network_analysis.py`)
Demonstrates friend network analysis including:
- Influence analysis using various centrality measures
- Community detection to find friend groups
- Connection path analysis between people
- Subgroup extraction (e.g., NYC residents)
- Network visualization

### 2. Transportation Network (`examples/transportation_network.py`)
Shows transportation system optimization:
- Multi-modal route optimization (metro, bus, road)
- Flow capacity analysis for passenger movement
- Network resilience testing (station closure impact)
- Mode-specific analysis (metro-only network)
- Bottleneck identification

### 3. Citation Network (`examples/citation_network.py`)
Analyzes academic paper citations:
- Paper influence metrics (PageRank, in-degree)
- Research trend identification through citation paths
- Cross-field paper identification (high betweenness)
- Collaboration pattern analysis
- Future impact prediction

Run examples:
```bash
python examples/social_network_analysis.py
python examples/transportation_network.py
python examples/citation_network.py
```

## Performance Monitoring

The server includes built-in performance monitoring accessible via the `monitoring_stats` tool:

```python
stats = await client.call_tool("monitoring_stats", {})
```

Returns comprehensive statistics:
- **Performance Metrics:**
  - Operation counts and average execution times
  - Total time spent per operation type
  - Success/failure rates
- **Memory Usage:**
  - Current and peak memory usage
  - Memory usage trends
- **Operation Tracking:**
  - Total operations executed
  - Error rates and types
  - Uptime information

Example output:
```json
{
  "performance": {
    "operations": {
      "create_graph": {"count": 10, "avg_ms": 2.3, "total_ms": 23},
      "add_nodes": {"count": 50, "avg_ms": 5.1, "total_ms": 255},
      "shortest_path": {"count": 25, "avg_ms": 15.7, "total_ms": 392.5}
    }
  },
  "operations": {
    "total_operations": 500,
    "successful_operations": 498,
    "failed_operations": 2,
    "error_rate": 0.4,
    "uptime": "2h 15m 30s"
  },
  "memory": {
    "current_usage_mb": 125.3,
    "peak_usage_mb": 256.7,
    "graph_count": 5
  }
}
```

## MCP Resources

The server provides MCP resources for accessing graph data:

- `graph://{graph_id}`: Get graph data in JSON format
- `graphs://list`: List all available graphs

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=networkx_mcp

# Run specific test file
pytest tests/test_graph_operations.py

# Run performance tests
pytest tests/ -k "performance"
```

### Code Quality

```bash
# Format code
black src tests

# Lint code
ruff src tests

# Type checking
mypy src
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Architecture

```
networkx-mcp-server/
├── src/networkx_mcp/
│   ├── __init__.py
│   ├── server.py          # Main MCP server with comprehensive tools
│   ├── cli.py            # Interactive CLI interface
│   ├── core/             # Core functionality
│   │   ├── __init__.py
│   │   ├── graph_operations.py   # Graph management and operations
│   │   ├── algorithms.py         # Graph algorithms implementation
│   │   └── io_handlers.py        # I/O with 10+ formats
│   ├── advanced/         # Phase 2: Advanced analytics
│   │   ├── __init__.py
│   │   ├── community.py          # Advanced community detection
│   │   ├── flow.py               # Network flow algorithms
│   │   ├── generators.py         # Graph generators
│   │   ├── bipartite.py          # Bipartite graph analysis
│   │   ├── directed.py           # Directed graph analysis
│   │   ├── specialized.py        # Specialized algorithms
│   │   ├── ml_integration.py     # Machine learning integration
│   │   ├── robustness.py         # Robustness analysis
│   │   └── enterprise.py         # Enterprise features
│   ├── visualization/    # Phase 3: Visualization engines
│   │   ├── __init__.py
│   │   ├── matplotlib_visualizer.py  # Static visualizations
│   │   ├── plotly_visualizer.py      # Interactive 2D/3D
│   │   ├── pyvis_visualizer.py       # Physics-based networks
│   │   └── specialized_viz.py        # Specialized visualizations
│   ├── integration/      # Phase 3: Data integration
│   │   ├── __init__.py
│   │   └── data_pipelines.py     # Multi-source data pipelines
│   ├── utils/            # Utilities
│   │   ├── __init__.py
│   │   ├── validators.py         # Comprehensive input validation
│   │   ├── formatters.py         # Output formatting utilities
│   │   └── monitoring.py         # Performance monitoring
│   └── schemas/          # Data schemas
│       ├── __init__.py
│       └── graph_schemas.py      # Graph data structures
├── tests/                # Comprehensive test suite
│   ├── test_graph_operations.py  # Including performance tests
│   ├── test_algorithms.py        # Algorithm correctness and benchmarks
│   └── test_io_handlers.py       # I/O format testing
├── examples/             # Real-world usage examples
│   ├── social_network_analysis.py
│   ├── transportation_network.py
│   └── citation_network.py
├── pyproject.toml        # Project configuration
├── requirements.txt      # Dependencies
└── README.md            # This file
```

### Key Components

1. **Server (server.py)**
   - FastMCP-based server implementation
   - Comprehensive logging and error handling
   - Performance monitoring integration
   - 30+ MCP tools for graph operations, analytics, visualization, and integration

2. **Core Modules**
   - **graph_operations.py**: GraphManager class for multi-graph management
   - **algorithms.py**: GraphAlgorithms class with static methods for all algorithms
   - **io_handlers.py**: GraphIOHandler with format detection and streaming support

3. **Utilities**
   - **validators.py**: Input validation for graph IDs, formats, and data
   - **monitoring.py**: PerformanceMonitor, OperationCounter, MemoryMonitor
   - **formatters.py**: Output formatting and data structure conversion

4. **CLI (cli.py)**
   - Rich-based interactive interface
   - Command completion and help
   - Benchmarking and demo modes
   - Real-time graph manipulation

### Design Principles

1. **Modular Architecture**: Clear separation of concerns
2. **Performance First**: Built-in monitoring and optimization
3. **Comprehensive Validation**: All inputs validated before processing
4. **Error Recovery**: Graceful error handling with informative messages
5. **Extensibility**: Easy to add new algorithms and formats
6. **Testing**: Comprehensive test coverage including edge cases

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [FastMCP](https://github.com/FastMCP/FastMCP) framework
- Powered by [NetworkX](https://networkx.org/) library
- Implements the [Model Context Protocol](https://modelcontextprotocol.io/) specification

## Support

For issues, questions, or contributions, please visit:
- GitHub Issues: [https://github.com/yourusername/networkx-mcp-server/issues](https://github.com/yourusername/networkx-mcp-server/issues)
- Documentation: [https://github.com/yourusername/networkx-mcp-server/wiki](https://github.com/yourusername/networkx-mcp-server/wiki)