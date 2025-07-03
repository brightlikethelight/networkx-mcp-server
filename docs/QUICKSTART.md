# Quick Start Guide

Get NetworkX MCP Server running in just 5 minutes! This guide will walk you through installation, basic configuration, and your first graph analysis.

## Prerequisites

Before you start, make sure you have:

- **Python 3.11+** (recommended: 3.12)
- **pip** or **uv** for package management
- An **MCP-compatible client** (Claude Desktop, MCP CLI, etc.)

!!! tip "Recommended Setup"
    
    For the best experience, we recommend using **Python 3.12** with **uv** for faster dependency resolution:
    
    ```bash
    # Install uv (fast Python package manager)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

## Installation

Choose your installation method based on your needs:

=== "Basic Installation"

    ```bash
    pip install networkx-mcp-server
    ```

=== "With Visualization"

    ```bash
    pip install networkx-mcp-server[visualization]
    ```

=== "With Machine Learning"

    ```bash
    pip install networkx-mcp-server[ml]
    ```

=== "Full Installation"

    ```bash
    pip install networkx-mcp-server[all]
    ```

=== "Using uv (Recommended)"

    ```bash
    uv pip install networkx-mcp-server[all]
    ```

## Starting the Server

### Method 1: Command Line

```bash
# Start with default settings
networkx-mcp-server

# Custom configuration
networkx-mcp-server --host 0.0.0.0 --port 8765
```

### Method 2: Python Module

```bash
python -m networkx_mcp.server
```

### Method 3: Docker

```bash
docker run -p 8765:8765 networkx-mcp-server:latest
```

You should see output like:

```
🕸️  NetworkX MCP Server v2.0.0
🚀 Starting server on localhost:8765
✅ Server ready! Connect your MCP client to stdio.
```

## Your First Graph

Let's create and analyze a simple social network:

### Step 1: Create a Graph

```python
# Create a new graph
create_graph(graph_id="friends", graph_type="Graph")
```

**Expected output:**
```json
{
  "success": true,
  "graph_id": "friends",
  "message": "Graph 'friends' created successfully"
}
```

### Step 2: Add Some Data

```python
# Add people as nodes
add_nodes("friends", nodes=["Alice", "Bob", "Charlie", "David", "Eve"])

# Add friendships as edges
add_edges("friends", edges=[
    ("Alice", "Bob"),
    ("Bob", "Charlie"), 
    ("Alice", "David"),
    ("Charlie", "David"),
    ("David", "Eve")
])
```

### Step 3: Analyze the Network

```python
# Get basic information
info = graph_statistics("friends")
print(info)
```

**Output:**
```json
{
  "basic": {
    "num_nodes": 5,
    "num_edges": 5,
    "density": 0.5,
    "is_connected": true
  },
  "degree": {
    "average": 2.0,
    "max": 3,
    "min": 1
  },
  "clustering": {
    "average_clustering": 0.33,
    "transitivity": 0.4
  }
}
```

### Step 4: Find Important People

```python
# Calculate centrality measures
centrality = centrality_measures("friends", measures=["degree", "betweenness"])
print(centrality)
```

**Output:**
```json
{
  "Alice": {"degree": 0.5, "betweenness": 0.25},
  "Bob": {"degree": 0.5, "betweenness": 0.25}, 
  "Charlie": {"degree": 0.5, "betweenness": 0.25},
  "David": {"degree": 0.75, "betweenness": 0.5},
  "Eve": {"degree": 0.25, "betweenness": 0.0}
}
```

!!! success "Analysis Results"
    
    **David** has the highest centrality scores, making him the most influential person in this network!

### Step 5: Visualize the Network

```python
# Create an interactive visualization
visualize_graph("friends", 
               layout="spring",
               save_path="friends_network.html",
               node_size_by="degree",
               node_color_by="betweenness")
```

This creates an interactive HTML file you can open in your browser.

## Common Patterns

Here are some common analysis patterns you'll use frequently:

### Pattern 1: Network Analysis Workflow

```python
# 1. Create and populate graph
create_graph("network", graph_type="Graph")
import_graph("network", file_path="data.csv", format="edgelist")

# 2. Basic analysis
stats = graph_statistics("network")
components = connected_components("network")

# 3. Find important nodes
centrality = centrality_measures("network", ["pagerank", "betweenness"])

# 4. Detect communities
communities = community_detection("network", algorithm="louvain")

# 5. Visualize results
visualize_communities("network", communities, save_path="communities.html")
```

### Pattern 2: Path Analysis

```python
# Find shortest paths
path = shortest_path("network", source="A", target="Z")
print(f"Shortest path: {path['path']}")
print(f"Distance: {path['length']}")

# Find all paths between nodes
all_paths = all_shortest_paths("network", source="A", target="Z")
print(f"Found {len(all_paths)} shortest paths")
```

### Pattern 3: Structural Analysis

```python
# Analyze graph structure
clustering = clustering_coefficient("network")
diameter = graph_diameter("network") 
efficiency = global_efficiency("network")

# Check connectivity
is_connected = graph_statistics("network")["connectivity"]["is_connected"]
components = connected_components("network")
```

## Configuration

### Environment Variables

Create a `.env` file for configuration:

```bash
# Server settings
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8765

# Performance
MAX_GRAPH_SIZE=1000000
ENABLE_CACHING=true

# Optional: Redis for persistence
REDIS_URL=redis://localhost:6379
```

### Configuration File

Create `config.yaml` for advanced settings:

```yaml
server:
  host: "0.0.0.0"
  port: 8765
  
performance:
  max_nodes: 1000000
  cache_ttl: 3600
  
logging:
  level: "INFO"
  format: "json"
```

## Connecting MCP Clients

### Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "networkx": {
      "command": "networkx-mcp-server",
      "args": []
    }
  }
}
```

### MCP CLI

```bash
# Install MCP CLI
pip install mcp-cli

# Connect to server
mcp-cli connect stdio networkx-mcp-server
```

### Custom Integration

```python
from mcp import Client
import asyncio

async def main():
    async with Client("networkx-mcp-server") as client:
        # Use the client
        result = await client.call_tool("create_graph", {
            "graph_id": "test",
            "graph_type": "Graph"
        })
        print(result)

asyncio.run(main())
```

## Next Steps

Now that you have the basics working, explore more advanced features:

<div class="grid cards" markdown>

-   [:material-book-open-page-variant: **User Guide**](user-guide/concepts.md)
    
    Learn core concepts and best practices

-   [:material-lightbulb-outline: **Examples**](examples/social-networks.md)
    
    Explore real-world use cases

-   [:material-api: **API Reference**](api/index.md)
    
    Complete tool documentation

-   [:material-cog: **Configuration**](configuration.md)
    
    Advanced server configuration

</div>

## Troubleshooting

### Common Issues

!!! question "Server won't start?"
    
    **Check Python version:**
    ```bash
    python --version  # Should be 3.11+
    ```
    
    **Verify installation:**
    ```bash
    pip show networkx-mcp-server
    ```

!!! question "Client can't connect?"
    
    **Check server is running:**
    ```bash
    # Server should show "Server ready!" message
    ```
    
    **Verify client configuration:**
    ```bash
    # Make sure client points to correct server command
    ```

!!! question "Performance issues?"
    
    **Enable Redis caching:**
    ```bash
    export REDIS_URL=redis://localhost:6379
    ```
    
    **Increase memory limits:**
    ```bash
    export MAX_MEMORY_MB=4096
    ```

### Getting Help

- **📖 Documentation**: Check the [User Guide](user-guide/concepts.md)
- **💬 Community**: Join [GitHub Discussions](https://github.com/brightliu/networkx-mcp-server/discussions)
- **🐛 Issues**: Report bugs on [GitHub Issues](https://github.com/brightliu/networkx-mcp-server/issues)
- **📧 Support**: Email [support@networkx-mcp-server.com](mailto:support@networkx-mcp-server.com)

!!! success "You're Ready!"
    
    Congratulations! You now have NetworkX MCP Server running and know the basics. 
    
    Ready for more advanced features? Check out our [Examples](examples/social-networks.md) or dive into the [API Reference](api/index.md).