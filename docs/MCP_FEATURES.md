# NetworkX MCP Server - Enhanced Features Documentation

## Overview

NetworkX MCP Server now implements the complete Model Context Protocol specification with Tools, Resources, and Prompts. This document explains how to use these features.

## 🔧 Tools

Tools are functions that LLMs can call to perform specific actions. NetworkX MCP Server provides 39+ tools for graph manipulation and analysis.

### Example Tools:
```python
# Create a graph
create_graph(graph_id="social_network", graph_type="Graph")

# Add nodes and edges
add_nodes(graph_id="social_network", nodes=["Alice", "Bob", "Charlie"])
add_edges(graph_id="social_network", edges=[("Alice", "Bob"), ("Bob", "Charlie")])

# Analyze the graph
shortest_path(graph_id="social_network", source="Alice", target="Charlie")
centrality_measures(graph_id="social_network", measures=["degree", "betweenness"])
```

## 📊 Resources (NEW!)

Resources provide read-only access to graph data, similar to GET endpoints in REST APIs. They allow LLMs to retrieve information without performing operations.

### Available Resources:

#### 1. Graph Catalog
**URI**: `graph://catalog`  
**Description**: Lists all available graphs with their metadata  
**Returns**: JSON array of graph information

```json
[
  {
    "id": "social_network",
    "type": "Graph",
    "nodes": 150,
    "edges": 320,
    "directed": false,
    "multigraph": false
  }
]
```

#### 2. Graph Data
**URI**: `graph://data/{graph_id}`  
**Description**: Complete graph data in node-link format  
**Returns**: JSON graph representation

```json
{
  "directed": false,
  "multigraph": false,
  "nodes": [
    {"id": "Alice"},
    {"id": "Bob"},
    {"id": "Charlie"}
  ],
  "links": [
    {"source": "Alice", "target": "Bob"},
    {"source": "Bob", "target": "Charlie"}
  ]
}
```

#### 3. Graph Statistics
**URI**: `graph://stats/{graph_id}`  
**Description**: Detailed statistics about a graph  
**Returns**: JSON with comprehensive metrics

```json
{
  "basic": {
    "nodes": 150,
    "edges": 320,
    "density": 0.0286,
    "is_directed": false,
    "is_multigraph": false
  },
  "connectivity": {
    "is_connected": true,
    "number_connected_components": 1
  },
  "degree": {
    "average_degree": 4.27,
    "max_degree": 15,
    "min_degree": 1
  },
  "clustering": {
    "average_clustering": 0.234,
    "transitivity": 0.189
  }
}
```

#### 4. Algorithm Results Cache
**URI**: `graph://results/{graph_id}/{algorithm}`  
**Description**: Cached results from previous algorithm runs  
**Returns**: JSON with algorithm-specific results

#### 5. Visualization Data
**URI**: `graph://viz/{graph_id}`  
**Description**: Graph data optimized for visualization with node positions  
**Returns**: JSON with nodes and edges including layout positions

```json
{
  "nodes": [
    {"id": "Alice", "x": 0.5, "y": 0.7, "label": "Alice"},
    {"id": "Bob", "x": 0.3, "y": 0.4, "label": "Bob"}
  ],
  "edges": [
    {"source": "Alice", "target": "Bob"}
  ]
}
```

## 💡 Prompts (NEW!)

Prompts are pre-defined templates that help users leverage tools and resources effectively. They provide structured workflows for common graph analysis tasks.

### Available Prompts:

#### 1. Social Network Analysis
**Name**: `analyze_social_network`  
**Description**: Complete workflow for analyzing social networks  
**Parameters**: 
- `graph_id`: The social network to analyze

**Workflow**:
1. Get basic network information
2. Identify influential nodes using centrality measures
3. Detect communities
4. Analyze network properties
5. Create visualizations

#### 2. Path Finding
**Name**: `find_optimal_path`  
**Description**: Find and analyze paths between nodes  
**Parameters**:
- `graph_id`: The graph to search
- `source`: Starting node
- `target`: Destination node

**Features**:
- Shortest path (weighted and unweighted)
- All shortest paths
- K-shortest paths for redundancy
- Path robustness analysis
- Path visualization

#### 3. Graph Generation
**Name**: `generate_test_graph`  
**Description**: Generate various types of test graphs  
**Parameters**:
- `graph_type`: Type of graph (scale_free, small_world, random, etc.)
- `num_nodes`: Number of nodes

**Supported Types**:
- Scale-free (Barabási-Albert)
- Small-world (Watts-Strogatz)
- Random (Erdős-Rényi)
- Complete graphs
- Bipartite graphs

#### 4. Performance Benchmarking
**Name**: `benchmark_algorithms`  
**Description**: Benchmark algorithm performance  
**Parameters**:
- `graph_id`: Graph to test on

**Tests**:
- Shortest path algorithms comparison
- Centrality measures performance
- Community detection speed
- Memory usage analysis

#### 5. Machine Learning
**Name**: `ml_graph_analysis`  
**Description**: Apply ML techniques to graphs  
**Parameters**:
- `graph_id`: Graph to analyze
- `task`: ML task (node_classification, link_prediction, etc.)

**Tasks**:
- Node classification
- Link prediction
- Graph embedding
- Anomaly detection

#### 6. Visualization Workflow
**Name**: `create_visualization`  
**Description**: Create customized graph visualizations  
**Parameters**:
- `graph_id`: Graph to visualize
- `viz_type`: Visualization type

**Options**:
- Static (matplotlib)
- Interactive (plotly)
- 3D (plotly 3D)
- Web-based (pyvis)

## 🚀 Usage Examples

### Using Resources in Claude Desktop

```python
# Get catalog of all graphs
resource("graph://catalog")

# Get specific graph data
resource("graph://data/social_network")

# Get graph statistics
resource("graph://stats/social_network")

# Get visualization-ready data
resource("graph://viz/social_network")
```

### Using Prompts in Claude Desktop

```python
# Start a social network analysis
prompt("analyze_social_network", graph_id="my_network")

# Find optimal paths
prompt("find_optimal_path", 
       graph_id="transport_network",
       source="Station_A", 
       target="Station_Z")

# Generate test graph
prompt("generate_test_graph", 
       graph_type="scale_free",
       num_nodes=1000)
```

### Combining Features

Resources and Prompts work together with Tools:

1. Use a **Prompt** to get a workflow template
2. Check **Resources** for current graph state
3. Execute **Tools** to perform operations
4. Monitor results via **Resources**

Example workflow:
```python
# 1. Get workflow from prompt
prompt("analyze_social_network", graph_id="company_network")

# 2. Check current state
resource("graph://stats/company_network")

# 3. Execute analysis tools
centrality_measures(graph_id="company_network", measures=["all"])
community_detection(graph_id="company_network", algorithm="louvain")

# 4. Retrieve results
resource("graph://results/company_network/centrality")
resource("graph://results/company_network/communities")
```

## 🔌 Integration with MCP Clients

### Claude Desktop Configuration

Add to your Claude Desktop configuration:

```json
{
  "servers": {
    "networkx": {
      "command": "networkx-mcp-server",
      "args": [],
      "env": {
        "REDIS_URL": "redis://localhost:6379"
      }
    }
  }
}
```

### Accessing Features

Once connected, you can:
- List available tools with `list_tools()`
- Browse resources with `list_resources()`
- View prompts with `list_prompts()`

## 🛠️ Advanced Configuration

### Environment Variables

```bash
# Enable resource caching
ENABLE_RESOURCE_CACHE=true
CACHE_TTL_SECONDS=300

# Configure resource limits
MAX_RESOURCE_SIZE_MB=10
MAX_GRAPH_SIZE_FOR_RESOURCES=10000

# Enable prompt customization
CUSTOM_PROMPTS_DIR=/path/to/prompts
```

### Custom Resources

You can add custom resources by extending the GraphResources class:

```python
@mcp.resource("graph://custom/{graph_id}")
async def custom_resource(graph_id: str) -> ResourceContent:
    # Your custom logic here
    return TextResourceContent(
        uri=f"graph://custom/{graph_id}",
        mimeType="application/json",
        text=json.dumps(custom_data)
    )
```

### Custom Prompts

Add custom prompts for your specific use cases:

```python
@mcp.prompt()
async def custom_analysis(graph_id: str) -> List[TextContent]:
    return [TextContent(
        type="text",
        text="Your custom workflow here..."
    )]
```

## 📚 Best Practices

1. **Use Resources for Read Operations**
   - Checking graph state
   - Retrieving analysis results
   - Getting visualization data

2. **Use Tools for Write Operations**
   - Creating/modifying graphs
   - Running algorithms
   - Generating visualizations

3. **Use Prompts for Workflows**
   - Complex multi-step analyses
   - Guided exploration
   - Learning tool usage

4. **Performance Tips**
   - Resources are cached by default
   - Use specific resource URIs to minimize data transfer
   - Batch operations when possible

## 🔍 Troubleshooting

### Resource Not Found
If a resource returns 404, check:
- Graph ID exists (`list_graphs()`)
- Resource URI is correct
- Graph has required data

### Prompt Parameters
Prompts accept optional parameters. View available parameters:
```python
prompt_info("analyze_social_network")
```

### Performance Issues
For large graphs:
- Use pagination in resources
- Enable caching
- Consider graph sampling

## 🚀 Future Enhancements

Coming soon:
- Real-time resource updates via WebSocket
- Resource subscriptions
- Prompt chaining
- Custom resource transformers
- GraphQL resource endpoint