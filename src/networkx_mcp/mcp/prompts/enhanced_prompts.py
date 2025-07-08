"""Enhanced MCP Prompts with parameter substitution and discovery."""

import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class PromptArgument:
    """Represents a prompt argument."""
    name: str
    description: str
    required: bool = True
    default: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP format."""
        return {
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "default": self.default
        }


@dataclass
class PromptMessage:
    """Represents a prompt message."""
    role: str = "assistant"
    content: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP format."""
        return {
            "role": self.role,
            "content": {
                "type": "text",
                "text": self.content
            }
        }


class PromptTemplate:
    """Enhanced prompt template with parameter substitution."""
    
    def __init__(
        self,
        name: str,
        description: str,
        template: str,
        arguments: List[PromptArgument],
        examples: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.description = description
        self.template = template
        self.arguments = arguments
        self.examples = examples or {}
        
    def substitute_parameters(self, params: Dict[str, Any]) -> str:
        """Substitute parameters in template."""
        # Apply defaults for missing parameters
        for arg in self.arguments:
            if arg.name not in params and arg.default is not None:
                params[arg.name] = arg.default
        
        # Validate required parameters
        missing = []
        for arg in self.arguments:
            if arg.required and arg.name not in params:
                missing.append(arg.name)
        
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")
        
        # Perform substitution
        result = self.template
        for key, value in params.items():
            # Replace {{key}} with value
            pattern = r'\{\{' + re.escape(key) + r'\}\}'
            result = re.sub(pattern, str(value), result)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP discovery format."""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": [arg.to_dict() for arg in self.arguments],
            "examples": self.examples
        }


class EnhancedGraphPrompts:
    """Enhanced MCP Prompts for graph analysis workflows."""
    
    def __init__(self, server):
        """Initialize prompts with server."""
        self.server = server
        self.templates = {}
        self._register_prompts()
    
    def _register_prompts(self):
        """Register all available prompts."""
        
        # 1. Graph Analysis Workflow
        self.register_template(
            name="analyze_graph",
            description="Complete graph analysis workflow",
            template="""I'll help you analyze the graph '{{graph_id}}' comprehensively.

## Analysis Workflow

### 1. Basic Structure Analysis
```
# Get graph information
graph_info(graph_name="{{graph_id}}")

# Check connectivity
connected_components(graph_name="{{graph_id}}")
```

### 2. Centrality Analysis
```
# Calculate various centrality measures
centrality_measures(
    graph_name="{{graph_id}}",
    measures=["degree", "betweenness", "closeness", "eigenvector"]
)
```

### 3. Community Detection
```
# Detect communities
community_detection(
    graph_name="{{graph_id}}",
    method="{{community_method}}"
)
```

### 4. Statistical Properties
```
# Get clustering coefficients
clustering_coefficients(graph_name="{{graph_id}}")

# Access detailed statistics via resource
resources/read(uri="graph://stats/{{graph_id}}")
```

### 5. Path Analysis
```
# Find important paths
shortest_path(
    graph_name="{{graph_id}}",
    source="{{source_node}}",
    target="{{target_node}}"
)
```

Would you like me to execute this analysis step by step?""",
            arguments=[
                PromptArgument("graph_id", "Graph to analyze", required=True),
                PromptArgument("community_method", "Community detection method", required=False, default="louvain"),
                PromptArgument("source_node", "Source node for path analysis", required=False, default="auto"),
                PromptArgument("target_node", "Target node for path analysis", required=False, default="auto")
            ],
            examples={
                "basic": {"graph_id": "social_network"},
                "full": {
                    "graph_id": "social_network",
                    "community_method": "greedy_modularity",
                    "source_node": "Alice",
                    "target_node": "Bob"
                }
            }
        )
        
        # 2. Visualization Generation
        self.register_template(
            name="visualize_graph",
            description="Generate graph visualizations",
            template="""Let's create a {{viz_type}} visualization for '{{graph_id}}'.

## Visualization Steps

### 1. Prepare Data
```
# Get graph data
graph_data = resources/read(uri="graph://data/{{graph_id}}")

# Get layout positions
layout = compute_layout(
    graph_name="{{graph_id}}",
    algorithm="{{layout_algorithm}}"
)
```

### 2. Configure Visual Properties
```
# Node properties
node_size = "{{node_size_attr}}"  # Attribute or fixed size
node_color = "{{node_color_attr}}"  # Attribute or color

# Edge properties  
edge_width = "{{edge_width_attr}}"  # Attribute or fixed width
edge_color = "{{edge_color}}"
```

### 3. Generate Visualization
```
visualize_graph(
    graph_name="{{graph_id}}",
    layout="{{layout_algorithm}}",
    node_size="{{node_size_attr}}",
    node_color="{{node_color_attr}}",
    edge_width="{{edge_width_attr}}",
    with_labels={{with_labels}},
    title="{{title}}"
)
```

### 4. Export Options
- PNG: High-quality static image
- HTML: Interactive visualization
- JSON: Raw visualization data

Shall I generate this visualization for you?""",
            arguments=[
                PromptArgument("graph_id", "Graph to visualize", required=True),
                PromptArgument("viz_type", "Visualization type", required=False, default="interactive"),
                PromptArgument("layout_algorithm", "Layout algorithm", required=False, default="spring"),
                PromptArgument("node_size_attr", "Node size attribute", required=False, default="degree"),
                PromptArgument("node_color_attr", "Node color attribute", required=False, default="community"),
                PromptArgument("edge_width_attr", "Edge width attribute", required=False, default="weight"),
                PromptArgument("edge_color", "Edge color", required=False, default="gray"),
                PromptArgument("with_labels", "Show node labels", required=False, default="true"),
                PromptArgument("title", "Visualization title", required=False, default="Graph Visualization")
            ]
        )
        
        # 3. Performance Optimization
        self.register_template(
            name="optimize_graph_performance",
            description="Optimize graph for performance",
            template="""I'll help optimize the performance of graph '{{graph_id}}' for {{use_case}}.

## Performance Analysis

### 1. Current Statistics
```
# Get current stats
stats = resources/read(uri="graph://stats/{{graph_id}}")
```
Current size: {{current_nodes}} nodes, {{current_edges}} edges

### 2. Optimization Strategies

#### For {{use_case}}:
{{#if use_case == "pathfinding"}}
- **Index Creation**: Build shortest path indices
- **Graph Simplification**: Remove redundant edges
- **Weight Optimization**: Normalize edge weights
{{/if}}

{{#if use_case == "community_detection"}}
- **Edge Sampling**: Reduce edges while preserving structure
- **Node Aggregation**: Merge similar nodes
- **Modularity Preservation**: Keep community structure
{{/if}}

{{#if use_case == "visualization"}}
- **Node Sampling**: Show representative nodes
- **Edge Bundling**: Group similar edges
- **Level of Detail**: Progressive rendering
{{/if}}

### 3. Implementation
```
# Optimize graph
optimized_graph = optimize_graph(
    graph_name="{{graph_id}}",
    target_size={{target_size}},
    preserve_properties=["{{preserve_property}}"],
    method="{{optimization_method}}"
)
```

### 4. Validation
```
# Compare performance
benchmark_results = benchmark_algorithms(
    original="{{graph_id}}",
    optimized="{{graph_id}}_optimized",
    algorithms=["shortest_path", "pagerank", "community_detection"]
)
```

Ready to optimize your graph?""",
            arguments=[
                PromptArgument("graph_id", "Graph to optimize", required=True),
                PromptArgument("use_case", "Primary use case", required=True),
                PromptArgument("current_nodes", "Current node count", required=False, default="auto"),
                PromptArgument("current_edges", "Current edge count", required=False, default="auto"),
                PromptArgument("target_size", "Target size percentage", required=False, default="50"),
                PromptArgument("preserve_property", "Property to preserve", required=False, default="connectivity"),
                PromptArgument("optimization_method", "Optimization method", required=False, default="smart_sampling")
            ]
        )
        
        # 4. Data Import Workflow
        self.register_template(
            name="import_graph_data",
            description="Import graph from various sources",
            template="""I'll help you import graph data from {{source_type}} into NetworkX MCP.

## Import Workflow for {{source_type}}

{{#if source_type == "csv"}}
### CSV Import Steps:
1. **Prepare CSV Format**
   - Nodes: `node_id, attribute1, attribute2, ...`
   - Edges: `source, target, weight, type, ...`

2. **Import Commands**
```python
# Import nodes
import_nodes_csv(
    file_path="{{nodes_file}}",
    graph_name="{{graph_name}}",
    id_column="{{id_column}}"
)

# Import edges
import_edges_csv(
    file_path="{{edges_file}}",
    graph_name="{{graph_name}}",
    source_column="{{source_column}}",
    target_column="{{target_column}}"
)
```
{{/if}}

{{#if source_type == "json"}}
### JSON Import Steps:
1. **Expected JSON Structure**
```json
{
    "nodes": [
        {"id": "A", "label": "Node A", ...},
        {"id": "B", "label": "Node B", ...}
    ],
    "edges": [
        {"source": "A", "target": "B", "weight": 1.0, ...}
    ]
}
```

2. **Import Command**
```python
import_json(
    file_path="{{json_file}}",
    graph_name="{{graph_name}}",
    format="{{json_format}}"
)
```
{{/if}}

{{#if source_type == "database"}}
### Database Import Steps:
1. **Query Setup**
```sql
-- Nodes query
{{nodes_query}}

-- Edges query  
{{edges_query}}
```

2. **Import Command**
```python
import_from_database(
    connection_string="{{db_connection}}",
    graph_name="{{graph_name}}",
    nodes_query="{{nodes_query}}",
    edges_query="{{edges_query}}"
)
```
{{/if}}

### 3. Validation
```python
# Verify import
graph_info(graph_name="{{graph_name}}")

# Check data integrity
validate_graph(graph_name="{{graph_name}}")
```

Ready to import your data?""",
            arguments=[
                PromptArgument("source_type", "Data source type (csv/json/database)", required=True),
                PromptArgument("graph_name", "Name for imported graph", required=True),
                PromptArgument("nodes_file", "Nodes CSV file path", required=False),
                PromptArgument("edges_file", "Edges CSV file path", required=False),
                PromptArgument("id_column", "Node ID column name", required=False, default="id"),
                PromptArgument("source_column", "Source column name", required=False, default="source"),
                PromptArgument("target_column", "Target column name", required=False, default="target"),
                PromptArgument("json_file", "JSON file path", required=False),
                PromptArgument("json_format", "JSON format type", required=False, default="node_link"),
                PromptArgument("db_connection", "Database connection string", required=False),
                PromptArgument("nodes_query", "SQL query for nodes", required=False),
                PromptArgument("edges_query", "SQL query for edges", required=False)
            ]
        )
        
        # 5. Algorithm Comparison
        self.register_template(
            name="compare_algorithms",
            description="Compare different algorithms on a graph",
            template="""Let's compare {{algorithm_type}} algorithms on graph '{{graph_id}}'.

## Algorithm Comparison: {{algorithm_type}}

### 1. Algorithms to Compare
{{#if algorithm_type == "shortest_path"}}
- **Dijkstra**: Best for non-negative weights
- **Bellman-Ford**: Handles negative weights
- **A***: Faster with heuristic
- **Floyd-Warshall**: All-pairs paths
{{/if}}

{{#if algorithm_type == "centrality"}}
- **Degree Centrality**: Simple connectivity
- **Betweenness Centrality**: Flow importance
- **Closeness Centrality**: Distance to all nodes
- **Eigenvector Centrality**: Influence measure
- **PageRank**: Web-style importance
{{/if}}

{{#if algorithm_type == "community"}}
- **Louvain**: Fast modularity optimization
- **Label Propagation**: Simple and scalable
- **Greedy Modularity**: Hierarchical approach
- **Spectral**: Mathematical foundation
{{/if}}

### 2. Benchmark Setup
```python
# Prepare benchmark
algorithms = {{algorithms_list}}
metrics = ["execution_time", "memory_usage", "quality_score"]

# Run comparison
results = compare_algorithms(
    graph_name="{{graph_id}}",
    algorithm_type="{{algorithm_type}}",
    algorithms=algorithms,
    iterations={{iterations}},
    metrics=metrics
)
```

### 3. Results Visualization
```python
# Plot comparison
plot_algorithm_comparison(
    results=results,
    chart_type="{{chart_type}}",
    highlight_best=True
)
```

### 4. Recommendations
Based on your graph characteristics:
- Size: {{graph_size}} nodes
- Density: {{graph_density}}
- Use case: {{use_case}}

**Recommended algorithm**: {{recommended_algorithm}}

Execute the comparison?""",
            arguments=[
                PromptArgument("graph_id", "Graph to analyze", required=True),
                PromptArgument("algorithm_type", "Type of algorithms to compare", required=True),
                PromptArgument("algorithms_list", "Specific algorithms to test", required=False, default="auto"),
                PromptArgument("iterations", "Number of test iterations", required=False, default="10"),
                PromptArgument("chart_type", "Visualization type", required=False, default="bar"),
                PromptArgument("graph_size", "Graph size category", required=False, default="auto"),
                PromptArgument("graph_density", "Graph density", required=False, default="auto"),
                PromptArgument("use_case", "Primary use case", required=False, default="general"),
                PromptArgument("recommended_algorithm", "Recommended algorithm", required=False, default="auto")
            ]
        )
        
        # Register all templates with the server
        for template in self.templates.values():
            self._register_server_prompt(template)
    
    def register_template(self, name: str, description: str, template: str, 
                         arguments: List[PromptArgument], examples: Optional[Dict] = None):
        """Register a prompt template."""
        prompt_template = PromptTemplate(name, description, template, arguments, examples)
        self.templates[name] = prompt_template
    
    def _register_server_prompt(self, template: PromptTemplate):
        """Register prompt with the MCP server."""
        @self.server.prompt(
            name=template.name,
            description=template.description,
            arguments=[arg.to_dict() for arg in template.arguments]
        )
        def prompt_handler(**kwargs):
            """Handle prompt execution."""
            try:
                # Substitute parameters
                content = template.substitute_parameters(kwargs)
                
                # Return as message
                return [PromptMessage(content=content).to_dict()]
                
            except ValueError as e:
                # Return error message
                return [PromptMessage(
                    content=f"Error: {str(e)}\n\nRequired parameters: {', '.join([arg.name for arg in template.arguments if arg.required])}"
                ).to_dict()]
    
    def get_prompt(self, name: str, **kwargs) -> List[Dict[str, Any]]:
        """Get a prompt with substituted parameters."""
        if name not in self.templates:
            raise ValueError(f"Unknown prompt: {name}")
        
        template = self.templates[name]
        content = template.substitute_parameters(kwargs)
        
        return [PromptMessage(content=content).to_dict()]
    
    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all available prompts."""
        return [template.to_dict() for template in self.templates.values()]