# Graph Operations Tools

This section documents all MCP tools for basic graph operations including creation, modification, and management.

## Overview

Graph operations form the foundation of NetworkX MCP Server functionality. These tools allow you to:

- Create and delete graphs
- Add and remove nodes and edges  
- Manage node and edge attributes
- Query graph structure and properties
- Perform basic graph transformations

!!! info "Tool Categories"
    
    - **[Graph Management](#graph-management)**: Create, delete, list graphs
    - **[Node Operations](#node-operations)**: Add, remove, query nodes
    - **[Edge Operations](#edge-operations)**: Add, remove, query edges
    - **[Attribute Management](#attribute-management)**: Handle node/edge properties
    - **[Graph Queries](#graph-queries)**: Get information about graphs

---

## Graph Management

### create_graph

Create a new graph with specified type and properties.

=== "Tool Definition"

    ```json
    {
      "name": "create_graph",
      "description": "Create a new graph of specified type",
      "parameters": {
        "type": "object",
        "properties": {
          "graph_id": {
            "type": "string",
            "description": "Unique identifier for the graph"
          },
          "graph_type": {
            "type": "string", 
            "enum": ["Graph", "DiGraph", "MultiGraph", "MultiDiGraph"],
            "description": "Type of graph to create"
          },
          "description": {
            "type": "string",
            "description": "Optional description of the graph"
          }
        },
        "required": ["graph_id", "graph_type"]
      }
    }
    ```

=== "Usage Example"

    ```python
    # Create a simple undirected graph
    result = create_graph(
        graph_id="social_network",
        graph_type="Graph",
        description="Social network of friends"
    )
    ```

=== "Response"

    ```json
    {
      "success": true,
      "graph_id": "social_network",
      "graph_type": "Graph",
      "message": "Graph 'social_network' created successfully",
      "metadata": {
        "created_at": "2024-01-15T10:30:00Z",
        "description": "Social network of friends"
      }
    }
    ```

**Graph Types:**

| Type | Description | Use Case |
|------|-------------|----------|
| `Graph` | Undirected graph | Social networks, collaboration |
| `DiGraph` | Directed graph | Web links, workflows, dependencies |
| `MultiGraph` | Multiple edges allowed | Transportation networks |
| `MultiDiGraph` | Directed with multiple edges | Complex flow networks |

---

### delete_graph

Remove a graph and all its data.

=== "Tool Definition"

    ```json
    {
      "name": "delete_graph", 
      "description": "Delete an existing graph",
      "parameters": {
        "type": "object",
        "properties": {
          "graph_id": {
            "type": "string",
            "description": "ID of graph to delete"
          }
        },
        "required": ["graph_id"]
      }
    }
    ```

=== "Usage Example"

    ```python
    # Delete a graph
    result = delete_graph(graph_id="old_network")
    ```

=== "Response"

    ```json
    {
      "success": true,
      "graph_id": "old_network", 
      "message": "Graph 'old_network' deleted successfully"
    }
    ```

!!! warning "Permanent Deletion"
    This operation cannot be undone. All nodes, edges, and attributes will be permanently removed.

---

### list_graphs

Get information about all available graphs.

=== "Tool Definition"

    ```json
    {
      "name": "list_graphs",
      "description": "List all available graphs with metadata",
      "parameters": {
        "type": "object",
        "properties": {
          "include_stats": {
            "type": "boolean",
            "default": false,
            "description": "Include basic statistics for each graph"
          }
        }
      }
    }
    ```

=== "Usage Example"

    ```python
    # List all graphs with statistics
    result = list_graphs(include_stats=True)
    ```

=== "Response"

    ```json
    {
      "graphs": [
        {
          "graph_id": "social_network",
          "graph_type": "Graph", 
          "description": "Social network of friends",
          "created_at": "2024-01-15T10:30:00Z",
          "stats": {
            "num_nodes": 150,
            "num_edges": 300,
            "density": 0.027
          }
        }
      ],
      "total_count": 1
    }
    ```

---

## Node Operations

### add_nodes

Add one or more nodes to a graph.

=== "Tool Definition"

    ```json
    {
      "name": "add_nodes",
      "description": "Add nodes to a graph with optional attributes",
      "parameters": {
        "type": "object", 
        "properties": {
          "graph_id": {
            "type": "string",
            "description": "Target graph ID"
          },
          "nodes": {
            "type": "array",
            "description": "List of nodes to add",
            "items": {
              "oneOf": [
                {"type": "string"},
                {"type": "number"},
                {
                  "type": "object",
                  "properties": {
                    "id": {"type": ["string", "number"]},
                    "attributes": {"type": "object"}
                  }
                }
              ]
            }
          }
        },
        "required": ["graph_id", "nodes"]
      }
    }
    ```

=== "Usage Example"

    ```python
    # Add simple nodes
    add_nodes("social_network", nodes=["Alice", "Bob", "Charlie"])
    
    # Add nodes with attributes
    add_nodes("social_network", nodes=[
        {
            "id": "Alice",
            "attributes": {
                "age": 25,
                "city": "New York",
                "interests": ["music", "travel"]
            }
        },
        {
            "id": "Bob", 
            "attributes": {
                "age": 30,
                "city": "Boston"
            }
        }
    ])
    ```

=== "Response"

    ```json
    {
      "success": true,
      "graph_id": "social_network",
      "nodes_added": 3,
      "nodes": ["Alice", "Bob", "Charlie"],
      "message": "Added 3 nodes to graph 'social_network'"
    }
    ```

---

### remove_nodes

Remove nodes from a graph.

=== "Tool Definition"

    ```json
    {
      "name": "remove_nodes",
      "description": "Remove nodes and their connected edges from graph",
      "parameters": {
        "type": "object",
        "properties": {
          "graph_id": {
            "type": "string", 
            "description": "Target graph ID"
          },
          "nodes": {
            "type": "array",
            "items": {"type": ["string", "number"]},
            "description": "List of node IDs to remove"
          }
        },
        "required": ["graph_id", "nodes"]
      }
    }
    ```

=== "Usage Example"

    ```python
    # Remove specific nodes
    remove_nodes("social_network", nodes=["Alice", "Bob"])
    ```

=== "Response"

    ```json
    {
      "success": true,
      "graph_id": "social_network", 
      "nodes_removed": 2,
      "edges_removed": 3,
      "message": "Removed 2 nodes and 3 connected edges"
    }
    ```

!!! warning "Cascade Deletion"
    Removing nodes also removes all edges connected to those nodes.

---

### get_nodes

Retrieve nodes and their attributes from a graph.

=== "Tool Definition"

    ```json
    {
      "name": "get_nodes",
      "description": "Get nodes from graph with optional filtering",
      "parameters": {
        "type": "object",
        "properties": {
          "graph_id": {
            "type": "string",
            "description": "Source graph ID"
          },
          "node_ids": {
            "type": "array", 
            "items": {"type": ["string", "number"]},
            "description": "Specific nodes to retrieve (optional)"
          },
          "include_attributes": {
            "type": "boolean",
            "default": true,
            "description": "Include node attributes in response"
          },
          "filter_by": {
            "type": "object",
            "description": "Filter nodes by attribute values"
          }
        },
        "required": ["graph_id"]
      }
    }
    ```

=== "Usage Example"

    ```python
    # Get all nodes with attributes
    nodes = get_nodes("social_network", include_attributes=True)
    
    # Get specific nodes
    nodes = get_nodes("social_network", node_ids=["Alice", "Bob"])
    
    # Filter nodes by attributes
    nodes = get_nodes("social_network", filter_by={"city": "New York"})
    ```

=== "Response"

    ```json
    {
      "graph_id": "social_network",
      "nodes": [
        {
          "id": "Alice",
          "attributes": {
            "age": 25,
            "city": "New York",
            "interests": ["music", "travel"]
          }
        },
        {
          "id": "Bob",
          "attributes": {
            "age": 30, 
            "city": "Boston"
          }
        }
      ],
      "total_count": 2
    }
    ```

---

## Edge Operations

### add_edges

Add edges between nodes in a graph.

=== "Tool Definition"

    ```json
    {
      "name": "add_edges",
      "description": "Add edges to graph with optional attributes", 
      "parameters": {
        "type": "object",
        "properties": {
          "graph_id": {
            "type": "string",
            "description": "Target graph ID"
          },
          "edges": {
            "type": "array",
            "description": "List of edges to add",
            "items": {
              "oneOf": [
                {
                  "type": "array",
                  "items": {"type": ["string", "number"]},
                  "minItems": 2,
                  "maxItems": 2
                },
                {
                  "type": "object",
                  "properties": {
                    "source": {"type": ["string", "number"]},
                    "target": {"type": ["string", "number"]}, 
                    "attributes": {"type": "object"}
                  }
                }
              ]
            }
          }
        },
        "required": ["graph_id", "edges"]
      }
    }
    ```

=== "Usage Example"

    ```python
    # Add simple edges
    add_edges("social_network", edges=[
        ("Alice", "Bob"),
        ("Bob", "Charlie"),
        ("Alice", "Charlie")
    ])
    
    # Add edges with attributes
    add_edges("social_network", edges=[
        {
            "source": "Alice",
            "target": "Bob", 
            "attributes": {
                "relationship": "friend",
                "since": "2020-01-15",
                "strength": 0.8
            }
        }
    ])
    ```

=== "Response"

    ```json
    {
      "success": true,
      "graph_id": "social_network",
      "edges_added": 3,
      "message": "Added 3 edges to graph 'social_network'"
    }
    ```

---

### remove_edges

Remove specific edges from a graph.

=== "Tool Definition"

    ```json
    {
      "name": "remove_edges",
      "description": "Remove edges from graph",
      "parameters": {
        "type": "object",
        "properties": {
          "graph_id": {
            "type": "string",
            "description": "Target graph ID"
          },
          "edges": {
            "type": "array", 
            "items": {
              "type": "array",
              "items": {"type": ["string", "number"]},
              "minItems": 2,
              "maxItems": 2
            },
            "description": "List of edges to remove"
          }
        },
        "required": ["graph_id", "edges"]
      }
    }
    ```

=== "Usage Example"

    ```python
    # Remove specific edges
    remove_edges("social_network", edges=[
        ("Alice", "Bob"),
        ("Bob", "Charlie")
    ])
    ```

=== "Response"

    ```json
    {
      "success": true,
      "graph_id": "social_network",
      "edges_removed": 2,
      "message": "Removed 2 edges from graph 'social_network'"
    }
    ```

---

### get_edges

Retrieve edges and their attributes from a graph.

=== "Tool Definition"

    ```json
    {
      "name": "get_edges", 
      "description": "Get edges from graph with optional filtering",
      "parameters": {
        "type": "object",
        "properties": {
          "graph_id": {
            "type": "string",
            "description": "Source graph ID"
          },
          "edge_list": {
            "type": "array",
            "description": "Specific edges to retrieve (optional)"
          },
          "include_attributes": {
            "type": "boolean", 
            "default": true,
            "description": "Include edge attributes"
          },
          "filter_by": {
            "type": "object",
            "description": "Filter edges by attribute values"
          }
        },
        "required": ["graph_id"]
      }
    }
    ```

=== "Usage Example"

    ```python
    # Get all edges
    edges = get_edges("social_network")
    
    # Get specific edges
    edges = get_edges("social_network", edge_list=[("Alice", "Bob")])
    
    # Filter by attributes
    edges = get_edges("social_network", filter_by={"relationship": "friend"})
    ```

=== "Response"

    ```json
    {
      "graph_id": "social_network",
      "edges": [
        {
          "source": "Alice",
          "target": "Bob",
          "attributes": {
            "relationship": "friend",
            "since": "2020-01-15", 
            "strength": 0.8
          }
        }
      ],
      "total_count": 1
    }
    ```

---

## Attribute Management

### set_node_attributes

Set or update attributes for nodes.

=== "Tool Definition"

    ```json
    {
      "name": "set_node_attributes",
      "description": "Set attributes for nodes in graph",
      "parameters": {
        "type": "object",
        "properties": {
          "graph_id": {
            "type": "string",
            "description": "Target graph ID"
          },
          "attributes": {
            "type": "object",
            "description": "Node ID to attributes mapping"
          },
          "merge": {
            "type": "boolean",
            "default": true,
            "description": "Merge with existing attributes or replace"
          }
        },
        "required": ["graph_id", "attributes"]
      }
    }
    ```

=== "Usage Example"

    ```python
    # Set attributes for multiple nodes
    set_node_attributes("social_network", attributes={
        "Alice": {"age": 26, "status": "active"},
        "Bob": {"location": "Boston", "verified": True}
    })
    
    # Replace all attributes (merge=False)
    set_node_attributes("social_network", 
                       attributes={"Alice": {"age": 26}},
                       merge=False)
    ```

=== "Response"

    ```json
    {
      "success": true,
      "graph_id": "social_network",
      "nodes_updated": 2,
      "message": "Updated attributes for 2 nodes"
    }
    ```

---

### set_edge_attributes

Set or update attributes for edges.

=== "Tool Definition"

    ```json
    {
      "name": "set_edge_attributes", 
      "description": "Set attributes for edges in graph",
      "parameters": {
        "type": "object",
        "properties": {
          "graph_id": {
            "type": "string",
            "description": "Target graph ID"
          },
          "attributes": {
            "type": "object",
            "description": "Edge to attributes mapping"
          },
          "merge": {
            "type": "boolean",
            "default": true,
            "description": "Merge with existing attributes"
          }
        },
        "required": ["graph_id", "attributes"]
      }
    }
    ```

=== "Usage Example"

    ```python
    # Set edge attributes
    set_edge_attributes("social_network", attributes={
        ("Alice", "Bob"): {"weight": 0.9, "type": "close_friend"},
        ("Bob", "Charlie"): {"weight": 0.6, "type": "acquaintance"}
    })
    ```

=== "Response"

    ```json
    {
      "success": true,
      "graph_id": "social_network", 
      "edges_updated": 2,
      "message": "Updated attributes for 2 edges"
    }
    ```

---

## Graph Queries

### graph_info

Get basic information about a graph.

=== "Tool Definition"

    ```json
    {
      "name": "graph_info",
      "description": "Get basic information about a graph",
      "parameters": {
        "type": "object",
        "properties": {
          "graph_id": {
            "type": "string",
            "description": "Graph to query"
          }
        },
        "required": ["graph_id"]
      }
    }
    ```

=== "Usage Example"

    ```python
    # Get graph information
    info = graph_info("social_network")
    ```

=== "Response"

    ```json
    {
      "graph_id": "social_network",
      "graph_type": "Graph",
      "num_nodes": 150,
      "num_edges": 300,
      "is_directed": false,
      "is_multigraph": false,
      "description": "Social network of friends",
      "created_at": "2024-01-15T10:30:00Z"
    }
    ```

---

### graph_statistics

Get detailed statistics about a graph.

=== "Tool Definition"

    ```json
    {
      "name": "graph_statistics",
      "description": "Calculate comprehensive graph statistics", 
      "parameters": {
        "type": "object",
        "properties": {
          "graph_id": {
            "type": "string",
            "description": "Graph to analyze"
          },
          "include_advanced": {
            "type": "boolean",
            "default": false,
            "description": "Include computationally expensive metrics"
          }
        },
        "required": ["graph_id"]
      }
    }
    ```

=== "Usage Example"

    ```python
    # Get basic statistics
    stats = graph_statistics("social_network")
    
    # Include advanced metrics
    stats = graph_statistics("social_network", include_advanced=True)
    ```

=== "Response"

    ```json
    {
      "graph_id": "social_network",
      "basic": {
        "num_nodes": 150,
        "num_edges": 300,
        "density": 0.027,
        "is_connected": true
      },
      "degree": {
        "average": 4.0,
        "max": 25,
        "min": 1,
        "std": 3.2
      },
      "connectivity": {
        "num_connected_components": 1,
        "largest_component_size": 150
      },
      "clustering": {
        "average_clustering": 0.35,
        "transitivity": 0.42
      },
      "advanced": {
        "diameter": 6,
        "radius": 3,
        "efficiency": 0.78
      }
    }
    ```

---

## Error Handling

### Common Errors

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `GRAPH_NOT_FOUND` | Graph ID doesn't exist | Check graph ID or create graph |
| `GRAPH_ALREADY_EXISTS` | Graph ID already in use | Use different ID or delete existing |
| `NODE_NOT_FOUND` | Node doesn't exist in graph | Add node first or check node ID |
| `EDGE_NOT_FOUND` | Edge doesn't exist | Check source/target nodes exist |
| `INVALID_GRAPH_TYPE` | Unknown graph type | Use Graph, DiGraph, MultiGraph, or MultiDiGraph |
| `MEMORY_LIMIT_EXCEEDED` | Too many nodes/edges | Reduce graph size or increase limits |

### Example Error Response

```json
{
  "error": {
    "code": "NODE_NOT_FOUND",
    "message": "Node 'NonExistent' not found in graph 'social_network'",
    "details": {
      "graph_id": "social_network",
      "node_id": "NonExistent",
      "available_nodes": ["Alice", "Bob", "Charlie"]
    }
  }
}
```

---

## Performance Tips

!!! tip "Optimization Strategies"
    
    - **Batch operations**: Add multiple nodes/edges in single calls
    - **Use appropriate graph types**: Choose the simplest type for your use case
    - **Limit attribute complexity**: Avoid deeply nested attribute objects
    - **Cache frequently accessed data**: Use Resources for read-heavy workloads
    - **Delete unused graphs**: Free memory by removing old graphs

### Batch Operations Example

```python
#  Good: Batch operation
add_nodes("graph", nodes=["A", "B", "C", "D"])

# L Avoid: Multiple individual calls  
add_nodes("graph", nodes=["A"])
add_nodes("graph", nodes=["B"]) 
add_nodes("graph", nodes=["C"])
add_nodes("graph", nodes=["D"])
```

---

## Next Steps

<div class="grid cards" markdown>

-   [:material-math-integral: **Algorithms**](algorithms.md)
    
    Path finding and centrality analysis

-   [:material-chart-line: **Analysis**](analysis.md)
    
    Community detection and clustering

-   [:material-palette: **Visualization**](visualization.md)
    
    Create interactive graph visualizations

-   [:material-database-import: **Import/Export**](import-export.md)
    
    Load and save graph data

</div>