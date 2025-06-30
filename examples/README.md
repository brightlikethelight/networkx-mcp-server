# NetworkX MCP Server Examples

This directory contains example scripts demonstrating various use cases for the NetworkX MCP Server.

## Available Examples

### 🚀 basic_usage.py
Simple examples showing core functionality - graph creation, node/edge management, and basic algorithms.

### 🌐 social_network_analysis.py
Demonstrates analyzing social networks - finding influencers, detecting communities, and understanding network structure.

### 🚗 transportation_network.py
Shows how to model and analyze transportation systems - finding optimal routes, analyzing traffic flow, and network resilience.

### 📚 citation_network.py
Academic citation network analysis - identifying influential papers, research trends, and collaboration patterns.

### 🔬 advanced_network_analysis.py
Advanced techniques including machine learning integration, temporal analysis, and large-scale graph processing.

### 🎨 visualization_and_integration_demo.py
Creating beautiful and interactive visualizations using matplotlib, Plotly, and pyvis backends.

## Running the Examples

```bash
# Run any example
python examples/basic_usage.py

# Or from the examples directory
cd examples
python social_network_analysis.py
```

## Requirements

Most examples work with the base installation:
```bash
pip install networkx-mcp-server
```

For visualization examples:
```bash
pip install networkx-mcp-server[visualization]
```

For all features:
```bash
pip install networkx-mcp-server[all]
```