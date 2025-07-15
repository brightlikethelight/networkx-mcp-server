# NetworkX MCP Server

**The first NetworkX integration for Model Context Protocol** - Bringing graph analysis directly into your AI conversations.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0%2B-orange.svg)](https://networkx.org/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 What is this?

NetworkX MCP Server enables Large Language Models (like Claude) to perform graph analysis operations directly within conversations. No more context switching between tools - analyze networks, find communities, calculate centrality, and visualize graphs all through natural language.

### 🎯 Key Features

- **13 Essential Graph Operations**: From basic graph creation to advanced algorithms like PageRank and community detection
- **Visualization**: Generate graph visualizations on-demand with multiple layout options
- **Import/Export**: Load graphs from CSV, export to JSON
- **Zero Setup**: Works immediately with Claude Desktop or any MCP-compatible client
- **First of Its Kind**: The first NetworkX server in the MCP ecosystem

## 🌟 Why NetworkX MCP Server?

- **Natural Language Graph Analysis**: Describe what you want to analyze in plain English
- **No Database Required**: Unlike graph database integrations, this works with in-memory graphs
- **Instant Insights**: Get centrality metrics, find communities, and discover patterns immediately
- **Visual Understanding**: See your graphs, don't just analyze them

## 📊 Available Operations

### Core Operations
- `create_graph` - Create directed or undirected graphs
- `add_nodes` - Add nodes to your graph
- `add_edges` - Connect nodes with edges
- `get_info` - Get basic graph statistics
- `shortest_path` - Find optimal paths between nodes

### Analysis Operations
- `degree_centrality` - Find the most connected nodes
- `betweenness_centrality` - Identify bridges and key connectors
- `pagerank` - Google's PageRank algorithm for node importance
- `connected_components` - Find isolated subgraphs
- `community_detection` - Discover natural groupings

### Visualization & I/O
- `visualize_graph` - Create PNG visualizations with multiple layouts
- `import_csv` - Load graphs from edge lists
- `export_json` - Export graphs in standard formats

## 🚦 Quick Start

### Installation

```bash
pip install networkx-mcp-server
```

### Configuration for Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "networkx": {
      "command": "networkx-mcp",
      "args": []
    }
  }
}
```

### Basic Usage Example

```
Human: Create a social network graph and analyze it

Claude: I'll help you create and analyze a social network graph.

[Creates graph "social_network"]
[Adds nodes: Alice, Bob, Charlie, David, Eve]
[Adds edges: Alice-Bob, Alice-Charlie, Bob-Charlie, Bob-David, Charlie-Eve]

Now let me analyze this network:

[Calculates degree centrality]
Most connected: Bob and Charlie (degree centrality: 0.75)

[Calculates betweenness centrality]
Key bridges: Charlie (betweenness: 0.58) - connects different parts of the network

[Detects communities]
Found 2 communities: [Alice, Bob, Charlie] and [David, Eve]

[Visualizes the graph]
Here's what your social network looks like: [displays graph image]
```

## 🎓 Example Use Cases

### 1. Social Network Analysis
- Identify influencers and key connectors
- Find communities and cliques
- Analyze information flow patterns

### 2. Transportation Planning
- Find shortest routes between locations
- Identify critical intersections
- Analyze network resilience

### 3. Knowledge Graphs
- Map concept relationships
- Find learning paths
- Identify prerequisite chains

See the [demos/](demos/) folder for complete examples.

## 📈 Performance

- **Memory**: ~70MB (including Python, NetworkX, and visualization)
- **Graph Size**: Tested up to 10,000 nodes
- **Operations**: Most complete in milliseconds
- **Visualization**: 1-2 seconds for complex graphs

## 🛠️ Development

### Running from Source

```bash
# Clone the repository
git clone https://github.com/Bright-L01/networkx-mcp-server
cd networkx-mcp-server

# Install dependencies
pip install -e .

# Run the server
python -m networkx_mcp.server_minimal
```

### Running Tests

```bash
pytest tests/working/
```

## 📚 Documentation

- [API Reference](docs/api.md) - Detailed operation descriptions
- [Examples](demos/) - Real-world use cases
- [Contributing](CONTRIBUTING.md) - How to contribute

## 🤝 Contributing

We welcome contributions! This is the first NetworkX MCP server, and there's lots of room for improvement:

- Add more graph algorithms
- Improve visualization options
- Add graph file format support
- Optimize performance
- Write more examples

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [NetworkX](https://networkx.org/) - The amazing graph library that powers this server
- [Anthropic](https://anthropic.com/) - For creating the Model Context Protocol
- The MCP community - For inspiration and examples

---

**Built with ❤️ for the AI and Graph Analysis communities**