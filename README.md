# NetworkX MCP Server

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Bright-L01/networkx-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/Bright-L01/networkx-mcp-server/actions)

A production-ready MCP (Model Context Protocol) server providing comprehensive graph analysis tools powered by NetworkX. Perfect for analyzing relationships, networks, and complex data structures.

## 🎯 What is NetworkX MCP Server?

NetworkX MCP Server exposes NetworkX's powerful graph analysis capabilities through the Model Context Protocol, allowing AI assistants and other tools to perform sophisticated graph operations. Whether you're analyzing social networks, transportation systems, or dependency graphs, this server provides the tools you need.

## ✨ Key Features

- **39+ Graph Tools**: Comprehensive suite for graph creation, analysis, and visualization
- **Multiple Graph Types**: Support for directed, undirected, multi-graphs, and weighted graphs
- **Advanced Algorithms**: Shortest paths, centrality measures, community detection, and more
- **Rich Visualizations**: Generate interactive visualizations with matplotlib, Plotly, or pyvis
- **Enterprise Features**: Redis persistence, monitoring, audit logging, and security controls
- **Easy Integration**: Works with any MCP-compatible client

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install networkx-mcp-server

# With visualization support
pip install networkx-mcp-server[visualization]

# Full installation with all features
pip install networkx-mcp-server[all]
```

### Running the Server

```bash
# Start the server
networkx-mcp-server

# Or with Python
python -m networkx_mcp.server
```

### Basic Usage Example

Once the server is running, you can use any MCP client to interact with it:

```python
# Example: Create and analyze a simple graph
create_graph(graph_id="my_network", graph_type="Graph")
add_nodes(graph_id="my_network", nodes=["A", "B", "C", "D"])
add_edges(graph_id="my_network", edges=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])

# Analyze the graph
info = graph_info(graph_id="my_network")
# Returns: {"num_nodes": 4, "num_edges": 4, "density": 0.67, ...}

# Find shortest path
path = shortest_path(graph_id="my_network", source="A", target="C")
# Returns: {"path": ["A", "B", "C"], "length": 2}

# Calculate centrality
centrality = centrality_measures(graph_id="my_network", measures=["degree", "betweenness"])
# Returns node importance metrics
```

## 📚 Available Tools

### Core Graph Operations
- `create_graph` - Create new graphs
- `add_nodes` / `remove_nodes` - Manage nodes
- `add_edges` / `remove_edges` - Manage edges
- `get_node_attributes` / `set_node_attributes` - Node properties
- `get_edge_attributes` / `set_edge_attributes` - Edge properties

### Graph Algorithms
- `shortest_path` - Find shortest paths between nodes
- `all_shortest_paths` - Find all shortest paths
- `centrality_measures` - Calculate node importance (degree, betweenness, closeness, eigenvector)
- `clustering_coefficient` - Measure graph clustering
- `connected_components` - Find connected subgraphs
- `minimum_spanning_tree` - Find minimum weight tree

### Advanced Analytics
- `community_detection` - Identify communities/clusters
- `link_prediction` - Predict future connections
- `node_classification` - Classify nodes using features
- `network_flow` - Analyze flow capacity
- `graph_embedding` - Generate node embeddings

### Visualization
- `visualize_graph` - Create graph visualizations
- `visualize_communities` - Highlight community structure
- `visualize_paths` - Show paths in graphs
- `visualize_flow` - Display network flow

### Import/Export
- `import_graph` - Load from JSON, CSV, GraphML, etc.
- `export_graph` - Save in various formats
- `batch_import` - Import multiple graphs
- `data_pipeline` - Process data into graphs

## 🔧 Configuration

### Environment Variables

```bash
# Server configuration
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8765

# Redis persistence (optional)
REDIS_URL=redis://localhost:6379
REDIS_PREFIX=networkx_mcp

# Security
ENABLE_SECURITY=true
MAX_GRAPH_SIZE=10000
```

### Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  networkx-mcp:
    image: networkx-mcp-server:latest
    ports:
      - "8765:8765"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## 📖 Examples

### Social Network Analysis

```python
# Create a social network
create_graph(graph_id="social", graph_type="Graph")
add_edges(graph_id="social", edges=[
    ("Alice", "Bob"), ("Bob", "Charlie"), 
    ("Charlie", "David"), ("Alice", "David"),
    ("Bob", "Eve"), ("Eve", "Frank")
])

# Find influential people
centrality = centrality_measures(
    graph_id="social", 
    measures=["betweenness", "eigenvector"]
)
# Identify key connectors and influential nodes

# Detect communities
communities = community_detection(graph_id="social", algorithm="louvain")
# Find friend groups
```

### Transportation Network

```python
# Create a weighted directed graph for routes
create_graph(graph_id="routes", graph_type="DiGraph")
add_edges(graph_id="routes", edges=[
    ("CityA", "CityB", {"distance": 100, "time": 2}),
    ("CityB", "CityC", {"distance": 150, "time": 3}),
    ("CityA", "CityC", {"distance": 300, "time": 4})
])

# Find optimal route
path = shortest_path(
    graph_id="routes", 
    source="CityA", 
    target="CityC",
    weight="time"  # Optimize for time instead of distance
)
```

## 🛡️ Security Features

- **Input validation**: All inputs are validated and sanitized
- **Graph size limits**: Configurable limits prevent resource exhaustion
- **Access control**: Optional authentication and authorization
- **Audit logging**: Track all operations for compliance
- **Secure file operations**: Sandboxed file access

## 🔌 Integration

NetworkX MCP Server works with any MCP-compatible client:

- **Claude Desktop**: Direct integration with Anthropic's Claude
- **MCP CLI**: Command-line interface for testing
- **Custom Clients**: Build your own using the MCP SDK

## 📊 Performance

- Handles graphs with millions of nodes and edges
- Redis persistence for fast graph retrieval
- Efficient algorithms with NetworkX's optimized implementations
- Configurable memory and CPU limits

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Setup development environment
git clone https://github.com/Bright-L01/networkx-mcp-server
cd networkx-mcp-server
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black . && ruff check .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [NetworkX](https://networkx.org/), the powerful Python graph library
- Implements the [Model Context Protocol](https://github.com/anthropics/mcp) specification
- Inspired by the need for accessible graph analysis tools

## 📞 Support

- **Documentation**: [Full API Documentation](https://networkx-mcp-server.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/Bright-L01/networkx-mcp-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Bright-L01/networkx-mcp-server/discussions)

---

Made with ❤️ by Bright Liu