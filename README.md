# NetworkX MCP Server

**Academic-focused graph analysis in your AI conversations** - The first and only NetworkX MCP server specialized for academic research and citation analysis.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0%2B-orange.svg)](https://networkx.org/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic](https://img.shields.io/badge/Academic-Research-red.svg)](https://crossref.org/)

## 🎓 What is this?

NetworkX MCP Server enables Large Language Models (like Claude) to perform sophisticated **academic research and citation analysis** directly within conversations. Built specifically for researchers, academics, and students who need to analyze citation networks, calculate author impact metrics, and discover literature patterns.

**Stop switching between VOSviewer, CitNetExplorer, and manual analysis. Start doing academic research in your AI conversations.**

### 🎯 Key Features

#### 🔬 Academic Research Tools
- **Citation Network Analysis**: Build citation networks from DOIs using CrossRef API
- **Author Impact Metrics**: Calculate h-index, total citations, and academic influence
- **Literature Discovery**: Automated paper recommendations based on citation patterns
- **Collaboration Analysis**: Map co-authorship networks and identify key researchers
- **Research Trend Detection**: Analyze publication and citation trends over time

#### 📊 Core Graph Operations
- **20+ Graph Functions**: From basic operations to advanced algorithms like PageRank
- **BibTeX Export**: Export citation networks in academic-standard BibTeX format
- **CrossRef Integration**: Access 156+ million academic papers via DOI resolution
- **Visualization**: Generate publication-ready network visualizations
- **First of Its Kind**: The only academic-focused NetworkX MCP server

## 🌟 Why NetworkX MCP Server for Academic Research?

- **Built for Researchers**: Designed specifically for academic workflows and citation analysis
- **Real-time Literature Discovery**: Find related papers and collaboration opportunities instantly
- **Reproducible Research**: Python-based, version-controlled, and shareable analysis workflows
- **Academic Data Integration**: Direct access to CrossRef's 156+ million paper database
- **No Enterprise Complexity**: Focus on research, not IT infrastructure
- **Cost-Effective**: Free alternative to expensive commercial citation analysis tools

## 📦 Installation

```bash
pip install networkx-mcp-server
```

## 🚀 Setup

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "networkx": {
      "command": "python",
      "args": ["-m", "networkx_mcp.server"]
    }
  }
}
```


## 📊 Available Operations

### 🔬 Academic Research Functions
- `resolve_doi` - Resolve DOI to publication metadata using CrossRef API
- `build_citation_network` - Build citation networks from seed DOIs
- `analyze_author_impact` - Calculate h-index and impact metrics for authors
- `find_collaboration_patterns` - Analyze co-authorship networks
- `detect_research_trends` - Identify publication and citation trends over time
- `recommend_papers` - Get paper recommendations based on citation patterns
- `export_bibtex` - Export citation networks in BibTeX format

### 📊 Core Graph Operations
- `create_graph` - Create directed or undirected graphs
- `add_nodes` - Add nodes to your graph
- `add_edges` - Connect nodes with edges
- `get_info` - Get basic graph statistics
- `shortest_path` - Find optimal paths between nodes

### 🔍 Analysis Operations
- `degree_centrality` - Find the most connected nodes
- `betweenness_centrality` - Identify bridges and key connectors
- `pagerank` - Google's PageRank algorithm for node importance
- `connected_components` - Find isolated subgraphs
- `community_detection` - Discover natural groupings

### 🎨 Visualization & I/O
- `visualize_graph` - Create PNG visualizations with multiple layouts
- `import_csv` - Load graphs from edge lists
- `export_json` - Export graphs in standard formats

## 🚦 Quick Start

### Community Edition

```bash
# Install community edition
pip install networkx-mcp-server
```

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


### Academic Research Example

```
Human: Analyze citation patterns for the paper "Attention Is All You Need"

Claude: I'll help you analyze citation patterns for that influential paper.

[Resolves DOI: 10.5555/3295222.3295349]
Found paper: "Attention Is All You Need" by Vaswani et al. (2017)
Citations: 82,892 | Journal: NIPS

[Builds citation network from seed DOI]
Built citation network with 847 nodes and 2,341 edges from 2-hop analysis

[Analyzes author impact]
Ashish Vaswani: h-index 45, total citations 127,436
Most impactful paper: "Attention Is All You Need" (82,892 citations)

[Finds collaboration patterns]
Key collaborators: Noam Shazeer (Google), Niki Parmar (Google)
Research cluster: Google Brain team with 47 collaborations

[Detects research trends]
Trend: MASSIVE INCREASE in attention mechanism research post-2017
2017: 12 papers → 2023: 3,847 papers (320x growth)

[Recommends related papers]
Top recommendations based on co-citation patterns:
1. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
2. "GPT-2: Language Models are Unsupervised Multitask Learners" (2019)
3. "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)

[Exports BibTeX]
Generated BibTeX file with 847 entries ready for LaTeX integration
```

## 🎓 Academic Use Cases

### 1. Literature Review & Meta-Analysis
- Automatically expand citation networks from key papers
- Identify research gaps and emerging trends
- Calculate field-wide impact metrics
- Generate comprehensive BibTeX databases

### 2. Collaboration Network Analysis
- Map research collaborations within and across institutions
- Identify key researchers and potential collaborators
- Analyze interdisciplinary connections
- Study research community evolution

### 3. Citation Pattern Analysis
- Track knowledge diffusion through citation networks
- Identify influential papers and breakthrough research
- Analyze citation bias and self-citation patterns
- Study geographic and institutional citation patterns

### 4. Research Trend Detection
- Identify emerging research areas and hot topics
- Analyze publication volume and citation trends
- Track research lifecycle from emergence to maturity
- Predict future research directions

### 5. Academic Impact Assessment
- Calculate comprehensive author impact metrics
- Compare researchers across different career stages
- Analyze journal and conference impact patterns
- Study citation half-life and research longevity

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
git clone https://github.com/brightlikethelight/networkx-mcp-server
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