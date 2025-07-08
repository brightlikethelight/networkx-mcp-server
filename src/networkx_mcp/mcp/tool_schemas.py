"""MCP Tool Schemas for parameter validation."""

from typing import Dict, Any

# Graph operation schemas
CREATE_GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]{1,100}$",
            "description": "Unique identifier for the graph"
        },
        "graph_type": {
            "type": "string",
            "enum": ["undirected", "directed", "multi", "multi_directed"],
            "default": "undirected",
            "description": "Type of graph structure"
        },
        "data": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "integer"}
                        ]
                    },
                    "description": "Initial nodes to add"
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 3,
                        "items": [
                            {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                            {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                            {"type": "object"}
                        ]
                    },
                    "description": "Initial edges to add"
                }
            },
            "description": "Initial graph data"
        }
    },
    "required": ["name"],
    "additionalProperties": False
}

DELETE_GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "graph_name": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]{1,100}$",
            "description": "Graph identifier to delete"
        }
    },
    "required": ["graph_name"],
    "additionalProperties": False
}

ADD_NODES_SCHEMA = {
    "type": "object",
    "properties": {
        "graph_name": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]{1,100}$",
            "description": "Graph identifier"
        },
        "nodes": {
            "type": "array",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "integer"}
                ]
            },
            "minItems": 1,
            "maxItems": 1000,
            "description": "Nodes to add to the graph"
        }
    },
    "required": ["graph_name", "nodes"],
    "additionalProperties": False
}

ADD_EDGES_SCHEMA = {
    "type": "object",
    "properties": {
        "graph_name": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]{1,100}$",
            "description": "Graph identifier"
        },
        "edges": {
            "type": "array",
            "items": {
                "oneOf": [
                    {
                        "type": "array",
                        "items": [
                            {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                            {"oneOf": [{"type": "string"}, {"type": "integer"}]}
                        ],
                        "minItems": 2,
                        "maxItems": 2
                    },
                    {
                        "type": "array",
                        "items": [
                            {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                            {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                            {"type": "object"}
                        ],
                        "minItems": 3,
                        "maxItems": 3
                    }
                ]
            },
            "minItems": 1,
            "maxItems": 10000,
            "description": "Edges to add (with optional attributes)"
        }
    },
    "required": ["graph_name", "edges"],
    "additionalProperties": False
}

# Algorithm schemas
SHORTEST_PATH_SCHEMA = {
    "type": "object",
    "properties": {
        "graph_name": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]{1,100}$",
            "description": "Graph identifier"
        },
        "source": {
            "oneOf": [{"type": "string"}, {"type": "integer"}],
            "description": "Source node"
        },
        "target": {
            "oneOf": [{"type": "string"}, {"type": "integer"}],
            "description": "Target node"
        },
        "weight": {
            "type": "string",
            "description": "Edge attribute name to use as weight"
        },
        "method": {
            "type": "string",
            "enum": ["auto", "dijkstra", "bellman-ford"],
            "default": "auto",
            "description": "Algorithm to use"
        }
    },
    "required": ["graph_name", "source", "target"],
    "additionalProperties": False
}

CENTRALITY_MEASURES_SCHEMA = {
    "type": "object",
    "properties": {
        "graph_name": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]{1,100}$",
            "description": "Graph identifier"
        },
        "measures": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["degree", "betweenness", "closeness", "eigenvector", "pagerank"]
            },
            "default": ["degree"],
            "description": "Centrality measures to compute"
        },
        "normalized": {
            "type": "boolean",
            "default": True,
            "description": "Whether to normalize centrality values"
        }
    },
    "required": ["graph_name"],
    "additionalProperties": False
}

CONNECTED_COMPONENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "graph_name": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]{1,100}$",
            "description": "Graph identifier"
        },
        "component_type": {
            "type": "string",
            "enum": ["weak", "strong"],
            "default": "weak",
            "description": "Type of connectivity (for directed graphs)"
        }
    },
    "required": ["graph_name"],
    "additionalProperties": False
}

# Query schemas
GRAPH_INFO_SCHEMA = {
    "type": "object",
    "properties": {
        "graph_name": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]{1,100}$",
            "description": "Graph identifier"
        }
    },
    "required": ["graph_name"],
    "additionalProperties": False
}

LIST_GRAPHS_SCHEMA = {
    "type": "object",
    "properties": {
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1000,
            "default": 100,
            "description": "Maximum number of graphs to return"
        },
        "offset": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
            "description": "Number of graphs to skip"
        }
    },
    "additionalProperties": False
}

# Visualization schemas
VISUALIZE_GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "graph_name": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]{1,100}$",
            "description": "Graph identifier"
        },
        "layout": {
            "type": "string",
            "enum": ["spring", "circular", "random", "shell", "spectral", "kamada_kawai"],
            "default": "spring",
            "description": "Layout algorithm"
        },
        "node_color": {
            "type": "string",
            "default": "lightblue",
            "description": "Node color or attribute name"
        },
        "node_size": {
            "oneOf": [
                {"type": "integer", "minimum": 1, "maximum": 5000},
                {"type": "string"}
            ],
            "default": 300,
            "description": "Node size or attribute name"
        },
        "with_labels": {
            "type": "boolean",
            "default": True,
            "description": "Whether to show node labels"
        },
        "output_format": {
            "type": "string",
            "enum": ["matplotlib", "plotly", "json"],
            "default": "matplotlib",
            "description": "Visualization output format"
        }
    },
    "required": ["graph_name"],
    "additionalProperties": False
}

# Collect all schemas
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # Graph operations
    "create_graph": CREATE_GRAPH_SCHEMA,
    "delete_graph": DELETE_GRAPH_SCHEMA,
    "add_nodes": ADD_NODES_SCHEMA,
    "add_edges": ADD_EDGES_SCHEMA,
    
    # Algorithms
    "shortest_path": SHORTEST_PATH_SCHEMA,
    "centrality_measures": CENTRALITY_MEASURES_SCHEMA,
    "connected_components": CONNECTED_COMPONENTS_SCHEMA,
    
    # Queries
    "graph_info": GRAPH_INFO_SCHEMA,
    "list_graphs": LIST_GRAPHS_SCHEMA,
    
    # Visualization
    "visualize_graph": VISUALIZE_GRAPH_SCHEMA,
}

# Output schemas
OUTPUT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "create_graph": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "name": {"type": "string"},
            "type": {"type": "string"},
            "nodes": {"type": "integer"},
            "edges": {"type": "integer"},
            "error": {"type": "string"}
        }
    },
    
    "shortest_path": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "path": {
                "type": "array",
                "items": {"oneOf": [{"type": "string"}, {"type": "integer"}]}
            },
            "length": {"type": "number"},
            "weighted": {"type": "boolean"},
            "error": {"type": "string"}
        }
    },
    
    "centrality_measures": {
        "type": "object",
        "patternProperties": {
            "^.*_centrality$": {
                "type": "object",
                "additionalProperties": {"type": "number"}
            }
        }
    },
    
    "graph_info": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string"},
            "nodes": {"type": "integer"},
            "edges": {"type": "integer"},
            "is_directed": {"type": "boolean"},
            "is_multigraph": {"type": "boolean"},
            "density": {"type": "number"},
            "error": {"type": "string"}
        }
    }
}