"""NetworkX MCP Server v2 - Modular implementation with Resources and Prompts."""

import logging
import os
import sys

try:
    from fastmcp import FastMCP
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False
    from networkx_mcp.mcp_mock import MockMCP
    FastMCP = MockMCP.FastMCP

from networkx_mcp.core.graph_operations import GraphManager
from networkx_mcp.server.prompts import GraphPrompts
from networkx_mcp.server.resources import GraphResources

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NetworkXMCPServer:
    """Enhanced NetworkX MCP Server with Resources and Prompts."""
    
    def __init__(self, name: str = "NetworkX Graph Analysis Server v2"):
        """Initialize the enhanced MCP server."""
        self.mcp = FastMCP(name)
        self.graph_manager = GraphManager()
        
        # Initialize resources and prompts
        self.resources = GraphResources(self.mcp, self.graph_manager)
        self.prompts = GraphPrompts(self.mcp)
        
        # Register core tools
        self._register_core_tools()
        
        logger.info(f"Initialized {name}")
        logger.info(f"MCP Features: Tools ✓, Resources ✓, Prompts ✓")
    
    def _register_core_tools(self):
        """Register core graph manipulation tools."""
        
        @self.mcp.tool()
        async def create_graph(
            graph_id: str,
            graph_type: str = "Graph",
            **kwargs
        ) -> str:
            """Create a new graph.
            
            Args:
                graph_id: Unique identifier for the graph
                graph_type: Type of graph (Graph, DiGraph, MultiGraph, MultiDiGraph)
                **kwargs: Additional graph attributes
            
            Returns:
                Success message with graph info
            """
            graph = self.graph_manager.create_graph(graph_id, graph_type, **kwargs)
            return f"Created {graph_type} with id '{graph_id}'"
        
        @self.mcp.tool()
        async def get_graph_info(graph_id: str) -> dict:
            """Get information about a graph.
            
            Args:
                graph_id: Graph identifier
            
            Returns:
                Dictionary with graph information
            """
            graph = self.graph_manager.get_graph(graph_id)
            if not graph:
                return {"error": f"Graph '{graph_id}' not found"}
            
            import networkx as nx
            
            info = {
                "id": graph_id,
                "type": graph.__class__.__name__,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "directed": graph.is_directed(),
                "multigraph": graph.is_multigraph(),
                "density": nx.density(graph)
            }
            
            # Add degree info
            if graph.number_of_nodes() > 0:
                degrees = [d for n, d in graph.degree()]
                info["average_degree"] = sum(degrees) / len(degrees)
                info["max_degree"] = max(degrees)
                info["min_degree"] = min(degrees)
            
            return info
        
        @self.mcp.tool()
        async def list_graphs() -> list:
            """List all available graphs.
            
            Returns:
                List of graph IDs
            """
            return self.graph_manager.list_graphs()
        
        @self.mcp.tool()
        async def add_nodes(graph_id: str, nodes: list) -> str:
            """Add nodes to a graph.
            
            Args:
                graph_id: Graph identifier
                nodes: List of node identifiers or (node, attributes) tuples
            
            Returns:
                Success message
            """
            graph = self.graph_manager.get_graph(graph_id)
            if not graph:
                return f"Error: Graph '{graph_id}' not found"
            
            # Handle both simple nodes and nodes with attributes
            for node in nodes:
                if isinstance(node, (list, tuple)) and len(node) == 2:
                    graph.add_node(node[0], **node[1])
                else:
                    graph.add_node(node)
            
            return f"Added {len(nodes)} nodes to graph '{graph_id}'"
        
        @self.mcp.tool()
        async def add_edges(graph_id: str, edges: list) -> str:
            """Add edges to a graph.
            
            Args:
                graph_id: Graph identifier
                edges: List of (source, target) or (source, target, attributes) tuples
            
            Returns:
                Success message
            """
            graph = self.graph_manager.get_graph(graph_id)
            if not graph:
                return f"Error: Graph '{graph_id}' not found"
            
            # Handle both simple edges and edges with attributes
            for edge in edges:
                if len(edge) == 2:
                    graph.add_edge(edge[0], edge[1])
                elif len(edge) == 3:
                    graph.add_edge(edge[0], edge[1], **edge[2])
            
            return f"Added {len(edges)} edges to graph '{graph_id}'"
        
        @self.mcp.tool()
        async def shortest_path(
            graph_id: str,
            source: str,
            target: str,
            weight: str = None
        ) -> dict:
            """Find shortest path between two nodes.
            
            Args:
                graph_id: Graph identifier
                source: Source node
                target: Target node
                weight: Edge attribute to use as weight (optional)
            
            Returns:
                Dictionary with path and length
            """
            graph = self.graph_manager.get_graph(graph_id)
            if not graph:
                return {"error": f"Graph '{graph_id}' not found"}
            
            import networkx as nx
            
            try:
                if weight:
                    path = nx.shortest_path(graph, source, target, weight=weight)
                    length = nx.shortest_path_length(graph, source, target, weight=weight)
                else:
                    path = nx.shortest_path(graph, source, target)
                    length = nx.shortest_path_length(graph, source, target)
                
                return {
                    "path": path,
                    "length": length,
                    "source": source,
                    "target": target,
                    "weighted": weight is not None
                }
            except nx.NetworkXNoPath:
                return {"error": f"No path from '{source}' to '{target}'"}
            except nx.NodeNotFound as e:
                return {"error": str(e)}
        
        @self.mcp.tool()
        async def visualize_graph(
            graph_id: str,
            backend: str = "matplotlib",
            layout: str = "spring",
            **kwargs
        ) -> str:
            """Visualize a graph.
            
            Args:
                graph_id: Graph identifier
                backend: Visualization backend (matplotlib, plotly, pyvis)
                layout: Layout algorithm
                **kwargs: Additional visualization parameters
            
            Returns:
                Success message with file path or HTML
            """
            graph = self.graph_manager.get_graph(graph_id)
            if not graph:
                return f"Error: Graph '{graph_id}' not found"
            
            # Import visualization modules
            if backend == "matplotlib":
                from networkx_mcp.visualization import MatplotlibVisualizer
                viz = MatplotlibVisualizer()
            elif backend == "plotly":
                from networkx_mcp.visualization import PlotlyVisualizer
                viz = PlotlyVisualizer()
            elif backend == "pyvis":
                from networkx_mcp.visualization import PyvisVisualizer
                viz = PyvisVisualizer()
            else:
                return f"Error: Unknown backend '{backend}'"
            
            result = viz.visualize(graph, layout=layout, **kwargs)
            return f"Visualization created: {result}"
    
    async def run(self):
        """Run the MCP server."""
        logger.info("Starting NetworkX MCP Server v2...")
        logger.info("Features: Tools, Resources, Prompts")
        logger.info(f"Tools: {len(self.mcp._tools)}")
        logger.info(f"Resources: {len(self.mcp._resources)}")
        logger.info(f"Prompts: {len(self.mcp._prompts)}")
        
        if HAS_FASTMCP:
            await self.mcp.run()
        else:
            logger.warning("Running with mock MCP implementation")
            await self.mcp.run()


async def main():
    """Main entry point."""
    server = NetworkXMCPServer()
    await server.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())