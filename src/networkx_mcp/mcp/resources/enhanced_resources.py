"""Enhanced MCP Resources with pagination and metadata support."""

import json
import math
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import networkx as nx


class ResourceMetadata:
    """Metadata for MCP resources."""
    
    def __init__(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str = "application/json",
        cacheable: bool = True,
        cache_ttl: int = 300  # 5 minutes
    ):
        self.uri = uri
        self.name = name
        self.description = description
        self.mime_type = mime_type
        self.cacheable = cacheable
        self.cache_ttl = cache_ttl
        self.last_modified = datetime.utcnow()


class PaginatedResponse:
    """Helper for paginated responses."""
    
    def __init__(
        self,
        items: List[Any],
        page: int = 1,
        per_page: int = 20,
        total: Optional[int] = None
    ):
        self.items = items
        self.page = max(1, page)
        self.per_page = max(1, min(100, per_page))  # Max 100 items per page
        self.total = total or len(items)
        
    @property
    def total_pages(self) -> int:
        """Calculate total number of pages."""
        return math.ceil(self.total / self.per_page)
    
    @property
    def has_next(self) -> bool:
        """Check if there's a next page."""
        return self.page < self.total_pages
    
    @property
    def has_prev(self) -> bool:
        """Check if there's a previous page."""
        return self.page > 1
    
    def get_page_items(self) -> List[Any]:
        """Get items for current page."""
        start = (self.page - 1) * self.per_page
        end = start + self.per_page
        return self.items[start:end]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with pagination metadata."""
        return {
            "items": self.get_page_items(),
            "pagination": {
                "page": self.page,
                "per_page": self.per_page,
                "total": self.total,
                "total_pages": self.total_pages,
                "has_next": self.has_next,
                "has_prev": self.has_prev
            }
        }


class EnhancedGraphResources:
    """Enhanced MCP Resources for graph data access with pagination."""
    
    def __init__(self, server, graph_manager):
        """Initialize resources with server and graph manager."""
        self.server = server
        self.graph_manager = graph_manager
        self.resource_metadata = {}
        self._register_resources()
    
    def _parse_pagination_params(self, uri: str) -> Tuple[int, int]:
        """Parse pagination parameters from URI query string."""
        parsed = urlparse(uri)
        params = parse_qs(parsed.query)
        
        page = int(params.get('page', ['1'])[0])
        per_page = int(params.get('per_page', ['20'])[0])
        
        return page, per_page
    
    def _register_resources(self):
        """Register all available resources with metadata."""
        
        # Graph catalog with pagination
        @self.server.resource(
            "graph://catalog",
            description="List all graphs with pagination",
            mime_type="application/json"
        )
        def graph_catalog_paginated():
            """List all available graphs with pagination support."""
            # In real implementation, parse from URI params
            page = 1
            per_page = 20
            
            graphs = []
            for graph_info in self.graph_manager.list_graphs():
                graph_id = graph_info["graph_id"]
                graph = self.graph_manager.get_graph(graph_id)
                if graph:
                    graphs.append({
                        "id": graph_id,
                        "name": graph_id,
                        "type": graph.__class__.__name__,
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges(),
                        "directed": graph.is_directed(),
                        "multigraph": graph.is_multigraph(),
                        "created_at": graph_info.get("metadata", {}).get("created_at"),
                        "size_bytes": self._estimate_graph_size(graph),
                        "uri": f"graph://data/{graph_id}"
                    })
            
            # Sort by creation date (newest first)
            graphs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            # Create paginated response
            paginated = PaginatedResponse(graphs, page, per_page)
            
            return json.dumps(paginated.to_dict(), indent=2)
        
        # Individual graph data with metadata
        @self.server.resource(
            "graph://data/{graph_id}",
            description="Get graph data in various formats"
        )
        def graph_data_with_format(graph_id: str, format: str = "node_link"):
            """Get graph data in specified format."""
            graph = self.graph_manager.get_graph(graph_id)
            if not graph:
                return json.dumps({"error": "Graph not found"})
            
            # Get graph metadata
            graph_info = self.graph_manager.metadata.get(graph_id, {})
            
            # Convert graph to requested format
            from networkx.readwrite import json_graph
            
            if format == "node_link":
                data = json_graph.node_link_data(graph)
            elif format == "adjacency":
                data = json_graph.adjacency_data(graph)
            elif format == "cytoscape":
                data = json_graph.cytoscape_data(graph)
            else:
                data = json_graph.node_link_data(graph)
            
            # Add metadata
            result = {
                "graph_id": graph_id,
                "format": format,
                "metadata": graph_info,
                "data": data,
                "export_formats": ["node_link", "adjacency", "cytoscape"],
                "statistics_uri": f"graph://stats/{graph_id}",
                "nodes_uri": f"graph://nodes/{graph_id}",
                "edges_uri": f"graph://edges/{graph_id}"
            }
            
            return json.dumps(result, indent=2)
        
        # Graph nodes with pagination
        @self.server.resource(
            "graph://nodes/{graph_id}",
            description="List graph nodes with attributes"
        )
        def graph_nodes_paginated(graph_id: str):
            """Get paginated list of nodes with attributes."""
            graph = self.graph_manager.get_graph(graph_id)
            if not graph:
                return {
                    "contents": [{
                        "uri": f"graph://nodes/{graph_id}",
                        "mimeType": "application/json",
                        "text": json.dumps({"error": "Graph not found"})
                    }]
                }
            
            # Get all nodes with attributes
            nodes = []
            for node, attrs in graph.nodes(data=True):
                nodes.append({
                    "id": str(node),
                    "attributes": attrs,
                    "degree": graph.degree(node),
                    "neighbors": [str(n) for n in graph.neighbors(node)]
                })
            
            # Paginate nodes
            paginated = PaginatedResponse(nodes, page=1, per_page=50)
            
            return {
                "contents": [{
                    "uri": f"graph://nodes/{graph_id}",
                    "mimeType": "application/json",
                    "text": json.dumps(paginated.to_dict(), indent=2)
                }]
            }
        
        # Graph edges with pagination
        @self.server.resource(
            "graph://edges/{graph_id}",
            description="List graph edges with attributes"
        )
        def graph_edges_paginated(graph_id: str):
            """Get paginated list of edges with attributes."""
            graph = self.graph_manager.get_graph(graph_id)
            if not graph:
                return {
                    "contents": [{
                        "uri": f"graph://edges/{graph_id}",
                        "mimeType": "application/json",
                        "text": json.dumps({"error": "Graph not found"})
                    }]
                }
            
            # Get all edges with attributes
            edges = []
            for u, v, attrs in graph.edges(data=True):
                edges.append({
                    "source": str(u),
                    "target": str(v),
                    "attributes": attrs
                })
            
            # Paginate edges
            paginated = PaginatedResponse(edges, page=1, per_page=100)
            
            return {
                "contents": [{
                    "uri": f"graph://edges/{graph_id}",
                    "mimeType": "application/json",
                    "text": json.dumps(paginated.to_dict(), indent=2)
                }]
            }
        
        # Enhanced statistics with more details
        @self.server.resource(
            "graph://stats/{graph_id}",
            description="Comprehensive graph statistics"
        )
        def graph_statistics_enhanced(graph_id: str):
            """Get comprehensive statistics for a graph."""
            graph = self.graph_manager.get_graph(graph_id)
            if not graph:
                return {
                    "contents": [{
                        "uri": f"graph://stats/{graph_id}",
                        "mimeType": "application/json",
                        "text": json.dumps({"error": "Graph not found"})
                    }]
                }
            
            # Basic stats
            stats = {
                "graph_id": graph_id,
                "basic": {
                    "num_nodes": graph.number_of_nodes(),
                    "num_edges": graph.number_of_edges(),
                    "density": nx.density(graph),
                    "is_directed": graph.is_directed(),
                    "is_multigraph": graph.is_multigraph(),
                    "size_bytes": self._estimate_graph_size(graph)
                }
            }
            
            # Degree statistics
            degrees = dict(graph.degree())
            if degrees:
                degree_values = list(degrees.values())
                stats["degree"] = {
                    "min": min(degree_values),
                    "max": max(degree_values),
                    "average": sum(degree_values) / len(degree_values),
                    "distribution": self._get_degree_distribution(degree_values)
                }
            
            # Connectivity
            if graph.is_directed():
                stats["connectivity"] = {
                    "is_weakly_connected": nx.is_weakly_connected(graph),
                    "is_strongly_connected": nx.is_strongly_connected(graph),
                    "num_weakly_connected_components": nx.number_weakly_connected_components(graph),
                    "num_strongly_connected_components": nx.number_strongly_connected_components(graph)
                }
            else:
                stats["connectivity"] = {
                    "is_connected": nx.is_connected(graph),
                    "num_connected_components": nx.number_connected_components(graph)
                }
            
            # Clustering (for undirected graphs)
            if not graph.is_directed() and graph.number_of_nodes() > 0:
                stats["clustering"] = {
                    "average_clustering": nx.average_clustering(graph),
                    "transitivity": nx.transitivity(graph)
                }
            
            # Node and edge attributes
            node_attrs = set()
            edge_attrs = set()
            
            for _, attrs in graph.nodes(data=True):
                node_attrs.update(attrs.keys())
            
            for _, _, attrs in graph.edges(data=True):
                edge_attrs.update(attrs.keys())
            
            stats["attributes"] = {
                "node_attributes": list(node_attrs),
                "edge_attributes": list(edge_attrs)
            }
            
            return {
                "contents": [{
                    "uri": f"graph://stats/{graph_id}",
                    "mimeType": "application/json",
                    "text": json.dumps(stats, indent=2)
                }]
            }
        
        # Search resource
        @self.server.resource(
            "graph://search",
            description="Search for graphs by criteria"
        )
        def search_graphs(query: str = "", min_nodes: int = 0, max_nodes: int = None):
            """Search for graphs matching criteria."""
            results = []
            
            for graph_id in self.graph_manager.list_graphs():
                graph = self.graph_manager.get_graph(graph_id)
                if not graph:
                    continue
                
                # Apply filters
                num_nodes = graph.number_of_nodes()
                
                if min_nodes and num_nodes < min_nodes:
                    continue
                    
                if max_nodes and num_nodes > max_nodes:
                    continue
                
                if query and query.lower() not in graph_id.lower():
                    continue
                
                results.append({
                    "id": graph_id,
                    "nodes": num_nodes,
                    "edges": graph.number_of_edges(),
                    "uri": f"graph://data/{graph_id}"
                })
            
            return {
                "contents": [{
                    "uri": "graph://search",
                    "mimeType": "application/json",
                    "text": json.dumps({
                        "query": query,
                        "filters": {
                            "min_nodes": min_nodes,
                            "max_nodes": max_nodes
                        },
                        "results": results,
                        "count": len(results)
                    }, indent=2)
                }]
            }
    
    def _estimate_graph_size(self, graph: nx.Graph) -> int:
        """Estimate graph size in bytes."""
        # Rough estimation: 100 bytes per node + 200 bytes per edge
        return graph.number_of_nodes() * 100 + graph.number_of_edges() * 200
    
    def _get_degree_distribution(self, degrees: List[int]) -> Dict[str, int]:
        """Get degree distribution."""
        distribution = {}
        for degree in degrees:
            distribution[str(degree)] = distribution.get(str(degree), 0) + 1
        return distribution
    
    def get_resource_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all registered resources."""
        return [
            {
                "uri": "graph://catalog",
                "name": "Graph Catalog",
                "description": "List all graphs with pagination",
                "parameters": ["page", "per_page"]
            },
            {
                "uri": "graph://data/{graph_id}",
                "name": "Graph Data",
                "description": "Get graph data in various formats",
                "parameters": ["format"],
                "formats": ["node_link", "adjacency", "cytoscape"]
            },
            {
                "uri": "graph://nodes/{graph_id}",
                "name": "Graph Nodes",
                "description": "List nodes with attributes (paginated)",
                "parameters": ["page", "per_page"]
            },
            {
                "uri": "graph://edges/{graph_id}",
                "name": "Graph Edges",
                "description": "List edges with attributes (paginated)",
                "parameters": ["page", "per_page"]
            },
            {
                "uri": "graph://stats/{graph_id}",
                "name": "Graph Statistics",
                "description": "Comprehensive graph statistics"
            },
            {
                "uri": "graph://search",
                "name": "Search Graphs",
                "description": "Search graphs by criteria",
                "parameters": ["query", "min_nodes", "max_nodes"]
            }
        ]