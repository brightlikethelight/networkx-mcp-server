"""Thread-safe graph manager implementation using lock manager.

This module wraps the existing GraphManager with thread-safe operations
using the GraphLockManager to prevent data corruption in concurrent environments.
"""

import asyncio
import networkx as nx
from typing import Dict, List, Any, Optional, Union
import logging

from ..concurrency import GraphLockManager

logger = logging.getLogger(__name__)


class ThreadSafeGraphManager:
    """Thread-safe wrapper for NetworkX graph operations."""
    
    def __init__(self, max_graphs: int = 1000):
        """Initialize thread-safe graph manager.
        
        Args:
            max_graphs: Maximum number of graphs to manage
        """
        self.graphs: Dict[str, nx.Graph] = {}
        self.max_graphs = max_graphs
        self.lock_manager = GraphLockManager(enable_stats=True)
        
    async def create_graph(self, 
                          name: str, 
                          graph_type: str = "undirected",
                          **attrs) -> Dict[str, Any]:
        """Create a new graph (thread-safe).
        
        Args:
            name: Unique graph identifier
            graph_type: Type of graph (undirected, directed, etc.)
            **attrs: Additional graph attributes
            
        Returns:
            Dictionary with creation status
        """
        # Check limit without lock first (optimization)
        if len(self.graphs) >= self.max_graphs:
            return {
                "success": False,
                "error": f"Maximum number of graphs ({self.max_graphs}) reached"
            }
            
        async with self.lock_manager.write_lock("_manager"):
            # Double-check inside lock
            if name in self.graphs:
                return {
                    "success": False,
                    "error": f"Graph '{name}' already exists"
                }
                
            if len(self.graphs) >= self.max_graphs:
                return {
                    "success": False,
                    "error": f"Maximum number of graphs ({self.max_graphs}) reached"
                }
                
            # Create graph in thread pool
            graph = await asyncio.to_thread(self._create_graph_sync, graph_type, **attrs)
            self.graphs[name] = graph
            
            # metrics_collector.increment("graphs.created")
            
            return {
                "success": True,
                "name": name,
                "type": graph_type,
                "nodes": 0,
                "edges": 0
            }
            
    def _create_graph_sync(self, graph_type: str, **attrs) -> nx.Graph:
        """Synchronous graph creation (runs in thread pool)."""
        if graph_type == "directed":
            return nx.DiGraph(**attrs)
        elif graph_type == "multi":
            return nx.MultiGraph(**attrs)
        elif graph_type == "multi_directed":
            return nx.MultiDiGraph(**attrs)
        else:
            return nx.Graph(**attrs)
            
    async def add_nodes(self, 
                       graph_name: str,
                       nodes: List[Union[str, int]],
                       **attrs) -> Dict[str, Any]:
        """Add nodes to a graph (thread-safe).
        
        Args:
            graph_name: Graph identifier
            nodes: List of nodes to add
            **attrs: Node attributes
            
        Returns:
            Dictionary with operation status
        """
        async with self.lock_manager.write_lock(graph_name):
            if graph_name not in self.graphs:
                return {
                    "success": False,
                    "error": f"Graph '{graph_name}' not found"
                }
                
            graph = self.graphs[graph_name]
            
            # Add nodes in thread pool
            added = await asyncio.to_thread(
                self._add_nodes_sync, graph, nodes, **attrs
            )
            
            # metrics_collector.increment("nodes.added", added)
            
            return {
                "success": True,
                "nodes_added": added,
                "total_nodes": graph.number_of_nodes()
            }
            
    def _add_nodes_sync(self, graph: nx.Graph, nodes: List, **attrs) -> int:
        """Synchronous node addition (runs in thread pool)."""
        initial_count = graph.number_of_nodes()
        
        for node in nodes:
            graph.add_node(node, **attrs)
            
        return graph.number_of_nodes() - initial_count
        
    async def add_edges(self,
                       graph_name: str,
                       edges: List[tuple],
                       **attrs) -> Dict[str, Any]:
        """Add edges to a graph (thread-safe).
        
        Args:
            graph_name: Graph identifier
            edges: List of edges (tuples of nodes)
            **attrs: Edge attributes
            
        Returns:
            Dictionary with operation status
        """
        async with self.lock_manager.write_lock(graph_name):
            if graph_name not in self.graphs:
                return {
                    "success": False,
                    "error": f"Graph '{graph_name}' not found"
                }
                
            graph = self.graphs[graph_name]
            
            # Add edges in thread pool
            added = await asyncio.to_thread(
                self._add_edges_sync, graph, edges, **attrs
            )
            
            # metrics_collector.increment("edges.added", added)
            
            return {
                "success": True,
                "edges_added": added,
                "total_edges": graph.number_of_edges()
            }
            
    def _add_edges_sync(self, graph: nx.Graph, edges: List[tuple], **attrs) -> int:
        """Synchronous edge addition (runs in thread pool)."""
        initial_count = graph.number_of_edges()
        
        for edge in edges:
            if len(edge) >= 2:
                graph.add_edge(edge[0], edge[1], **attrs)
                
        return graph.number_of_edges() - initial_count
        
    async def get_shortest_path(self,
                               graph_name: str,
                               source: Union[str, int],
                               target: Union[str, int],
                               weight: Optional[str] = None) -> Dict[str, Any]:
        """Find shortest path between nodes (thread-safe).
        
        Args:
            graph_name: Graph identifier
            source: Source node
            target: Target node
            weight: Edge attribute to use as weight
            
        Returns:
            Dictionary with path information
        """
        async with self.lock_manager.read_lock(graph_name):
            if graph_name not in self.graphs:
                return {
                    "success": False,
                    "error": f"Graph '{graph_name}' not found"
                }
                
            graph = self.graphs[graph_name]
            
            # Run CPU-intensive operation in thread pool
            try:
                path = await asyncio.to_thread(
                    nx.shortest_path, graph, source, target, weight=weight
                )
                
                length = len(path) - 1
                if weight:
                    length = await asyncio.to_thread(
                        nx.shortest_path_length, graph, source, target, weight=weight
                    )
                    
                return {
                    "success": True,
                    "path": path,
                    "length": length,
                    "weighted": weight is not None
                }
                
            except nx.NetworkXNoPath:
                return {
                    "success": False,
                    "error": f"No path between {source} and {target}"
                }
            except nx.NodeNotFound as e:
                return {
                    "success": False,
                    "error": str(e)
                }
                
    async def get_graph_info(self, graph_name: str) -> Dict[str, Any]:
        """Get graph information (thread-safe).
        
        Args:
            graph_name: Graph identifier
            
        Returns:
            Dictionary with graph information
        """
        async with self.lock_manager.read_lock(graph_name):
            if graph_name not in self.graphs:
                return {
                    "success": False,
                    "error": f"Graph '{graph_name}' not found"
                }
                
            graph = self.graphs[graph_name]
            
            # Get info in thread pool
            info = await asyncio.to_thread(self._get_graph_info_sync, graph)
            info["name"] = graph_name
            info["success"] = True
            
            return info
            
    def _get_graph_info_sync(self, graph: nx.Graph) -> Dict[str, Any]:
        """Synchronous graph info retrieval (runs in thread pool)."""
        return {
            "type": graph.__class__.__name__,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "is_directed": graph.is_directed(),
            "is_multigraph": graph.is_multigraph(),
            "density": nx.density(graph) if graph.number_of_nodes() > 0 else 0.0
        }
        
    async def list_graphs(self, 
                         limit: int = 100,
                         offset: int = 0) -> Dict[str, Any]:
        """List all graphs (thread-safe).
        
        Args:
            limit: Maximum number of graphs to return
            offset: Number of graphs to skip
            
        Returns:
            Dictionary with graph list
        """
        async with self.lock_manager.read_lock("_manager"):
            graph_names = list(self.graphs.keys())
            
        # Get info for each graph
        graphs_info = []
        for name in graph_names[offset:offset + limit]:
            info = await self.get_graph_info(name)
            if info["success"]:
                graphs_info.append(info)
                
        return {
            "success": True,
            "graphs": graphs_info,
            "total": len(graph_names),
            "limit": limit,
            "offset": offset
        }
        
    async def delete_graph(self, graph_name: str) -> Dict[str, Any]:
        """Delete a graph (thread-safe).
        
        Args:
            graph_name: Graph identifier
            
        Returns:
            Dictionary with deletion status
        """
        async with self.lock_manager.write_lock("_manager"):
            if graph_name not in self.graphs:
                return {
                    "success": False,
                    "error": f"Graph '{graph_name}' not found"
                }
                
            # Get final stats
            graph = self.graphs[graph_name]
            final_nodes = graph.number_of_nodes()
            final_edges = graph.number_of_edges()
            
            # Delete graph
            del self.graphs[graph_name]
            
            # metrics_collector.increment("graphs.deleted")
            
            return {
                "success": True,
                "deleted": graph_name,
                "final_nodes": final_nodes,
                "final_edges": final_edges
            }
            
    async def centrality_measures(self,
                                 graph_name: str,
                                 measures: List[str] = None,
                                 normalized: bool = True) -> Dict[str, Any]:
        """Calculate centrality measures (thread-safe).
        
        Args:
            graph_name: Graph identifier
            measures: List of centrality measures to calculate
            normalized: Whether to normalize values
            
        Returns:
            Dictionary with centrality values
        """
        if measures is None:
            measures = ["degree"]
            
        async with self.lock_manager.read_lock(graph_name):
            if graph_name not in self.graphs:
                return {
                    "success": False,
                    "error": f"Graph '{graph_name}' not found"
                }
                
            graph = self.graphs[graph_name]
            
            # Calculate in thread pool
            results = await asyncio.to_thread(
                self._calculate_centrality_sync, graph, measures, normalized
            )
            
            return results
            
    def _calculate_centrality_sync(self, 
                                  graph: nx.Graph,
                                  measures: List[str],
                                  normalized: bool) -> Dict[str, Any]:
        """Synchronous centrality calculation (runs in thread pool)."""
        results = {"success": True}
        
        for measure in measures:
            if measure == "degree":
                results["degree_centrality"] = dict(
                    nx.degree_centrality(graph)
                )
            elif measure == "betweenness":
                results["betweenness_centrality"] = dict(
                    nx.betweenness_centrality(graph, normalized=normalized)
                )
            elif measure == "closeness":
                results["closeness_centrality"] = dict(
                    nx.closeness_centrality(graph)
                )
            elif measure == "eigenvector":
                try:
                    results["eigenvector_centrality"] = dict(
                        nx.eigenvector_centrality(graph, max_iter=1000)
                    )
                except:
                    results["eigenvector_centrality"] = "Failed to converge"
            elif measure == "pagerank":
                results["pagerank"] = dict(nx.pagerank(graph))
                
        return results
        
    async def multi_graph_operation(self,
                                   operation: str,
                                   graph_names: List[str],
                                   **params) -> Dict[str, Any]:
        """Perform operation on multiple graphs atomically.
        
        Args:
            operation: Operation to perform
            graph_names: List of graph identifiers
            **params: Operation parameters
            
        Returns:
            Dictionary with operation results
        """
        # Use multi-graph lock to prevent deadlocks
        async with self.lock_manager.multi_graph_lock(graph_names):
            results = {"success": True, "results": {}}
            
            for graph_name in graph_names:
                if graph_name not in self.graphs:
                    results["results"][graph_name] = {
                        "success": False,
                        "error": f"Graph '{graph_name}' not found"
                    }
                    continue
                    
                # Perform operation based on type
                if operation == "merge":
                    # Example: merge graphs
                    pass
                elif operation == "intersect":
                    # Example: find common nodes/edges
                    pass
                    
            return results
            
    def get_lock_stats(self) -> Dict[str, Any]:
        """Get lock manager statistics.
        
        Returns:
            Dictionary with lock statistics
        """
        return self.lock_manager.get_stats()
        
    async def cleanup(self):
        """Cleanup resources."""
        await self.lock_manager.cleanup()
        self.graphs.clear()