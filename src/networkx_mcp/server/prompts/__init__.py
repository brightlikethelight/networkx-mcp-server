"""MCP Prompts for NetworkX Server.

Prompts provide pre-defined templates for common graph analysis workflows,
helping users leverage tools and resources effectively.
"""


try:
    from fastmcp import FastMCP
    from mcp.types import Prompt, PromptArgument, TextContent
except ImportError:
    # Mock imports for when MCP is not available
    from networkx_mcp.mcp_mock import MockMCP

    Prompt = MockMCP.types.Prompt
    PromptArgument = MockMCP.types.PromptArgument
    TextContent = MockMCP.types.TextContent


class GraphPrompts:
    """MCP Prompts for graph analysis workflows."""

    def __init__(self, mcp: FastMCP):
        """Initialize prompts with MCP server."""
        self.mcp = mcp
        self._register_prompts()

    def _register_prompts(self):
        """Register all available prompts."""

        # Social Network Analysis Prompt
        @self.mcp.prompt()
        async def analyze_social_network(
            graph_id: str = "social_network",
        ) -> list[TextContent]:
            """Complete workflow for analyzing a social network."""
            return [
                TextContent(
                    type="text",
                    text=f"""Let's analyze the social network '{graph_id}'.

Step 1: First, let's get basic information about the network:
- Use get_graph_info to understand the network structure
- Check if it's connected using connected_components

Step 2: Identify influential nodes:
- Calculate centrality measures (degree, betweenness, eigenvector, closeness)
- Find nodes with highest centrality scores

Step 3: Detect communities:
- Use community_detection with algorithm='louvain'
- Analyze community structure and sizes

Step 4: Analyze network properties:
- Calculate clustering coefficient
- Find shortest paths between key nodes
- Check for small-world properties

Step 5: Visualize the network:
- Use visualize_graph with community coloring
- Highlight influential nodes

Would you like me to execute this analysis workflow?""",
                )
            ]

        # Path Finding Prompt
        @self.mcp.prompt()
        async def find_optimal_path(
            graph_id: str = "network", source: str = "A", target: str = "Z"
        ) -> list[TextContent]:
            """Find optimal paths in a network."""
            return [
                TextContent(
                    type="text",
                    text=f"""Let's find optimal paths in '{graph_id}' from '{source}' to '{target}'.

Analysis Steps:
1. Find shortest path (unweighted)
2. Find shortest path by weight (if edges have weights)
3. Find all shortest paths
4. Find k-shortest paths for redundancy
5. Analyze path robustness
6. Visualize the paths

Commands to execute:
- shortest_path(graph_id="{graph_id}", source="{source}", target="{target}")
- shortest_path(graph_id="{graph_id}", source="{source}", target="{target}", weight="weight")
- find_all_paths(graph_id="{graph_id}", source="{source}", target="{target}", max_length=10)
- visualize_paths(graph_id="{graph_id}", paths=<results>)

Shall I proceed with the path analysis?""",
                )
            ]

        # Graph Generation Prompt
        @self.mcp.prompt()
        async def generate_test_graph(
            graph_type: str = "scale_free", num_nodes: int = 100
        ) -> list[TextContent]:
            """Generate test graphs for analysis."""
            return [
                TextContent(
                    type="text",
                    text=f"""Let's generate a {graph_type} graph with {num_nodes} nodes.

Available graph types:
1. **scale_free**: Power-law degree distribution (social networks)
   - Parameters: n={num_nodes}, m=3 (edges per new node)

2. **small_world**: High clustering, short paths (brain networks)
   - Parameters: n={num_nodes}, k=6, p=0.3

3. **random**: Erdős-Rényi random graph
   - Parameters: n={num_nodes}, p=0.1 (edge probability)

4. **complete**: All nodes connected
   - Parameters: n={num_nodes}

5. **bipartite**: Two-mode network
   - Parameters: n1={num_nodes//2}, n2={num_nodes//2}

Command:
generate_graph(
    graph_id="test_graph",
    graph_type="{graph_type}",
    num_nodes={num_nodes},
    <additional_params>
)

Would you like to generate this graph and analyze its properties?""",
                )
            ]

        # Performance Analysis Prompt
        @self.mcp.prompt()
        async def benchmark_algorithms(
            graph_id: str = "test_graph",
        ) -> list[TextContent]:
            """Benchmark algorithm performance on a graph."""
            return [
                TextContent(
                    type="text",
                    text=f"""Let's benchmark algorithm performance on '{graph_id}'.

Performance Tests:
1. **Shortest Path Algorithms**:
   - Dijkstra vs Bellman-Ford vs A*
   - Single-source vs all-pairs

2. **Centrality Measures**:
   - Degree vs Betweenness vs Eigenvector
   - Exact vs approximate algorithms

3. **Community Detection**:
   - Louvain vs Label Propagation vs Spectral
   - Quality vs speed trade-offs

4. **Graph Traversal**:
   - BFS vs DFS performance
   - Memory usage analysis

Metrics to collect:
- Execution time
- Memory usage
- Result quality
- Scalability

Would you like to run the performance benchmarks?""",
                )
            ]

        # Machine Learning Prompt
        @self.mcp.prompt()
        async def ml_graph_analysis(
            graph_id: str = "ml_graph", task: str = "node_classification"
        ) -> list[TextContent]:
            """Apply machine learning to graph analysis."""
            return [
                TextContent(
                    type="text",
                    text=f"""Let's apply machine learning to '{graph_id}' for {task}.

Available ML Tasks:
1. **Node Classification**:
   - Predict node labels/categories
   - Use: node features + graph structure

2. **Link Prediction**:
   - Predict future/missing edges
   - Use: node similarity + graph patterns

3. **Graph Embedding**:
   - Learn vector representations
   - Methods: Node2Vec, DeepWalk, GraphSAGE

4. **Anomaly Detection**:
   - Find unusual nodes/edges
   - Use: statistical + structural features

Workflow:
1. Extract graph features
2. Prepare training data
3. Apply ML algorithm
4. Evaluate results
5. Visualize predictions

Command:
ml_graph_analysis(
    graph_id="{graph_id}",
    task="{task}",
    algorithm="auto",
    test_size=0.2
)

Ready to start the ML analysis?""",
                )
            ]

        # Visualization Workflow Prompt
        @self.mcp.prompt()
        async def create_visualization(
            graph_id: str = "my_graph", viz_type: str = "interactive"
        ) -> list[TextContent]:
            """Create beautiful graph visualizations."""
            return [
                TextContent(
                    type="text",
                    text=f"""Let's create a {viz_type} visualization for '{graph_id}'.

Visualization Options:

1. **Simple Static** (matplotlib):
   - Quick overview plots
   - Good for small graphs (<100 nodes)
   - Export as PNG/PDF

2. **Interactive** (plotly):
   - Zoom, pan, hover details
   - Good for medium graphs (<1000 nodes)
   - Export as HTML

3. **3D Interactive** (plotly 3D):
   - Three-dimensional layout
   - Rotation and exploration
   - Best for spatial networks

4. **Web-based** (pyvis):
   - Physics simulation
   - Drag nodes, adjust layout
   - Best for exploration

Customization options:
- Node colors by: community, centrality, attributes
- Node sizes by: degree, importance
- Edge colors by: weight, type
- Layout: spring, circular, hierarchical, geographical

Command:
visualize_graph(
    graph_id="{graph_id}",
    backend="{viz_type}",
    node_color="community",
    node_size="degree",
    layout="spring"
)

Would you like to create this visualization?""",
                )
            ]


__all__ = ["GraphPrompts"]
