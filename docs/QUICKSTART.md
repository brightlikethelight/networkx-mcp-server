# NetworkX MCP Server Quick Start Guide

This guide will help you get up and running with the NetworkX MCP Server in minutes.

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/networkx-mcp-server.git
cd networkx-mcp-server
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -e .
```

## Starting the Server

You have three options to interact with the NetworkX MCP Server:

### Option 1: MCP Server Mode (Production)
```bash
python -m networkx_mcp.server
```
This starts the MCP server on `http://localhost:8000` for use with MCP clients.

### Option 2: Interactive CLI (Development/Testing)
```bash
python -m networkx_mcp.cli
```
This provides an interactive command-line interface for testing and debugging.

### Option 3: Python Script (Direct Usage)
```python
# example_script.py
from networkx_mcp.server import app
import asyncio

async def main():
    # Your code here
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

## Your First Graph

Let's create a simple social network and perform some basic analysis.

### Using the CLI

1. **Start the CLI:**
```bash
python -m networkx_mcp.cli
```

2. **Create a graph:**
```
> create friends Graph
✓ Created graph 'friends' (type: Graph)
```

3. **Add some people:**
```
> add nodes Alice Bob Charlie Diana Eve
✓ Added 5 nodes
```

4. **Add friendships:**
```
> add edge Alice Bob
✓ Added edge Alice -> Bob
> add edge Bob Charlie
✓ Added edge Bob -> Charlie
> add edge Charlie Diana
✓ Added edge Charlie -> Diana
> add edge Diana Eve
✓ Added edge Diana -> Eve
> add edge Alice Diana
✓ Added edge Alice -> Diana
```

5. **Analyze the network:**
```
> analyze centrality betweenness
Betweenness Centrality
┏━━━━━━━━━┳━━━━━━━━┓
┃ Node    ┃ Score  ┃
┡━━━━━━━━━╇━━━━━━━━┩
│ Diana   │ 0.5000 │
│ Bob     │ 0.1667 │
│ Charlie │ 0.1667 │
│ Alice   │ 0.0000 │
│ Eve     │ 0.0000 │
└─────────┴────────┘
```

6. **Find shortest path:**
```
> analyze path Alice Eve
Shortest path: Alice → Diana → Eve
Length: 2
```

### Using MCP Client

```python
# example_client.py
import asyncio
from your_mcp_client import MCPClient  # Use your MCP client library

async def create_social_network():
    client = MCPClient("http://localhost:8000")
    
    # Create graph
    await client.call_tool("create_graph", {
        "graph_id": "friends",
        "graph_type": "undirected"
    })
    
    # Add people with attributes
    await client.call_tool("add_nodes", {
        "graph_id": "friends",
        "nodes": [
            {"id": "Alice", "age": 28, "city": "NYC"},
            {"id": "Bob", "age": 32, "city": "Boston"},
            {"id": "Charlie", "age": 25, "city": "NYC"},
            {"id": "Diana", "age": 30, "city": "Chicago"},
            {"id": "Eve", "age": 27, "city": "NYC"}
        ]
    })
    
    # Add friendships with relationship strength
    await client.call_tool("add_edges", {
        "graph_id": "friends",
        "edges": [
            {"source": "Alice", "target": "Bob", "strength": 0.9},
            {"source": "Bob", "target": "Charlie", "strength": 0.7},
            {"source": "Charlie", "target": "Diana", "strength": 0.6},
            {"source": "Diana", "target": "Eve", "strength": 0.8},
            {"source": "Alice", "target": "Diana", "strength": 0.5}
        ]
    })
    
    # Find most influential person
    result = await client.call_tool("calculate_centrality", {
        "graph_id": "friends",
        "centrality_type": ["degree", "betweenness", "closeness"],
        "top_n": 3
    })
    print("Most influential people:", result)
    
    # Find communities
    result = await client.call_tool("clustering_analysis", {
        "graph_id": "friends",
        "include_triangles": True
    })
    print("Network clustering:", result)

if __name__ == "__main__":
    asyncio.run(create_social_network())
```

## Common Use Cases

### 1. Transportation Route Optimization

```python
# Create transport network
await client.call_tool("create_graph", {
    "graph_id": "city_transport",
    "graph_type": "directed"
})

# Add stations
await client.call_tool("add_nodes", {
    "graph_id": "city_transport",
    "nodes": ["Central", "North", "South", "East", "West"]
})

# Add routes with travel times
await client.call_tool("add_edges", {
    "graph_id": "city_transport",
    "edges": [
        {"source": "Central", "target": "North", "time": 5, "distance": 3.2},
        {"source": "Central", "target": "South", "time": 7, "distance": 4.1},
        {"source": "North", "target": "East", "time": 10, "distance": 6.5},
        {"source": "South", "target": "West", "time": 8, "distance": 5.0}
    ]
})

# Find fastest route
result = await client.call_tool("shortest_path", {
    "graph_id": "city_transport",
    "source": "Central",
    "target": "East",
    "weight": "time",
    "k_paths": 3  # Get top 3 fastest routes
})
```

### 2. Citation Network Analysis

```python
# Create citation network
await client.call_tool("create_graph", {
    "graph_id": "papers",
    "graph_type": "directed",
    "from_data": {
        "edge_list": [
            ["Paper_A", "Paper_B"],  # Paper_A cites Paper_B
            ["Paper_A", "Paper_C"],
            ["Paper_B", "Paper_D"],
            ["Paper_C", "Paper_D"],
            ["Paper_E", "Paper_D"]
        ]
    }
})

# Find most cited papers
result = await client.call_tool("calculate_centrality", {
    "graph_id": "papers",
    "centrality_type": "pagerank",
    "top_n": 5
})
```

### 3. Network Resilience Analysis

```python
# Analyze network resilience
result = await client.call_tool("connected_components", {
    "graph_id": "infrastructure",
    "component_type": "strong"
})

# Extract critical nodes subgraph
result = await client.call_tool("subgraph_extraction", {
    "graph_id": "infrastructure",
    "method": "condition",
    "condition": "criticality > 0.8",
    "create_new": True,
    "new_graph_id": "critical_infrastructure"
})
```

## Working with Large Graphs

### 1. Initialize from File

```python
# For large graphs, initialize from file
await client.call_tool("create_graph", {
    "graph_id": "large_network",
    "graph_type": "undirected",
    "from_data": {
        "file_path": "network_data.csv",
        "format": "csv",
        "source_col": "from",
        "target_col": "to",
        "weight_col": "weight"
    }
})
```

### 2. Use Sampling for Analysis

```python
# Sample-based analysis for large graphs
result = await client.call_tool("path_analysis", {
    "graph_id": "large_network",
    "sample_size": 1000  # Analyze 1000 random node pairs
})
```

### 3. Extract Subgraphs for Detailed Analysis

```python
# Extract 2-hop neighborhood
result = await client.call_tool("subgraph_extraction", {
    "graph_id": "large_network",
    "method": "k_hop",
    "center_node": "important_node",
    "k_hop": 2,
    "create_new": True,
    "new_graph_id": "local_network"
})
```

## Performance Tips

1. **Monitor Performance:**
```python
stats = await client.call_tool("monitoring_stats", {})
print(f"Average operation time: {stats['performance']['mean_ms']}ms")
```

2. **Use Bulk Operations:**
```python
# Good - single bulk operation
await client.call_tool("add_edges", {
    "graph_id": "network",
    "edges": [edge1, edge2, edge3, ...]  # Add all at once
})

# Avoid - multiple individual operations
for edge in edges:
    await client.call_tool("add_edge", {...})  # Slower
```

3. **Specify Weights When Needed:**
```python
# Unweighted (faster)
result = await client.call_tool("shortest_path", {
    "graph_id": "network",
    "source": "A",
    "target": "B"
})

# Weighted (more accurate but slower)
result = await client.call_tool("shortest_path", {
    "graph_id": "network",
    "source": "A",
    "target": "B",
    "weight": "distance"
})
```

## Running Examples

The repository includes three comprehensive examples:

```bash
# Social network analysis
python examples/social_network_analysis.py

# Transportation optimization
python examples/transportation_network.py

# Citation network analysis
python examples/citation_network.py
```

## CLI Quick Reference

```bash
# Create and manage graphs
create <id> [type]     # Create new graph
list                   # List all graphs
info [id]             # Show graph details
select <id>           # Select active graph
delete <id>           # Delete graph

# Build graphs
add nodes <n1> <n2>... # Add nodes
add edge <src> <tgt>   # Add edge
import <fmt> <file>    # Import from file
export <fmt> <file>    # Export to file

# Analyze
analyze centrality     # Centrality measures
analyze path <s> <t>   # Find paths
analyze components     # Connected components
analyze metrics        # Graph statistics

# Utilities
monitor               # Performance stats
benchmark <size>      # Run benchmark
demo <type>          # Run demo
help                 # Show help
exit                 # Exit CLI
```

## Troubleshooting

### Common Issues

1. **"Graph not found" error:**
   - Ensure the graph_id exists using the `list` command
   - Graph IDs are case-sensitive

2. **"Node not found" error:**
   - Check node exists in the graph
   - Node IDs must match exactly (including type)

3. **Performance issues:**
   - Use `monitoring_stats` to identify slow operations
   - Consider using subgraph extraction for large graphs
   - Enable sampling for statistical operations

4. **Memory issues:**
   - Use streaming I/O for large files
   - Process graphs in chunks
   - Clear unused graphs with `delete` command

### Getting Help

- Check the [API Documentation](API.md) for detailed parameter information
- Run `help` in the CLI for command reference
- Check the examples folder for working code samples

## Next Steps

1. Explore the [example scripts](../examples/) for real-world use cases
2. Read the [API documentation](API.md) for detailed tool descriptions
3. Try the interactive CLI with your own data
4. Build custom analysis workflows using the MCP client

Happy graphing!