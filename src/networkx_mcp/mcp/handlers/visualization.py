"""Visualization Handler for NetworkX MCP Server.

This module handles graph visualization tools using various backends
including matplotlib, plotly, and pyvis.
"""

from typing import Any, Dict, Optional
import os


try:
    from fastmcp import FastMCP
except ImportError:
    from networkx_mcp.mcp_mock import MockMCP as FastMCP


class VisualizationHandler:
    """Handler for graph visualization operations."""

    def __init__(self, mcp: FastMCP, graph_manager):
        """Initialize the handler with MCP server and graph manager."""
        self.mcp = mcp
        self.graph_manager = graph_manager
        self._register_tools()

    def _register_tools(self):
        """Register all visualization tools."""

        @self.mcp.tool()
        async def visualize_graph(
            graph_id: str,
            backend: str = "matplotlib",
            layout: str = "spring",
            node_color: str = "lightblue",
            edge_color: str = "gray",
            node_size: int = 300,
            with_labels: bool = True,
            title: Optional[str] = None,
            output_path: Optional[str] = None,
            **kwargs
        ) -> Dict[str, Any]:
            """Visualize a graph using various backends.

            Args:
                graph_id: ID of the graph to visualize
                backend: Visualization backend ('matplotlib', 'plotly', 'pyvis')
                layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', etc.)
                node_color: Color for nodes (can be attribute name for coloring)
                edge_color: Color for edges
                node_size: Size of nodes (can be attribute name for sizing)
                with_labels: Whether to show node labels
                title: Title for the visualization
                output_path: Where to save the visualization
                **kwargs: Additional backend-specific parameters

            Returns:
                Dict with visualization details and file path
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                # Import appropriate visualizer
                if backend == "matplotlib":
                    from networkx_mcp.visualization import MatplotlibVisualizer

                    visualizer = MatplotlibVisualizer()
                elif backend == "plotly":
                    from networkx_mcp.visualization import PlotlyVisualizer

                    visualizer = PlotlyVisualizer()
                elif backend == "pyvis":
                    from networkx_mcp.visualization import PyvisVisualizer

                    visualizer = PyvisVisualizer()
                else:
                    return {"error": f"Unknown backend: {backend}"}

                # Prepare visualization parameters
                viz_params = {
                    "layout": layout,
                    "node_color": node_color,
                    "edge_color": edge_color,
                    "node_size": node_size,
                    "with_labels": with_labels,
                    "title": title or f"Graph: {graph_id}",
                }
                viz_params.update(kwargs)

                # Generate visualization
                if output_path:
                    viz_params["output_path"] = output_path
                else:
                    # Default output path
                    ext = "html" if backend in ["plotly", "pyvis"] else "png"
                    viz_params["output_path"] = f"graph_{graph_id}_{backend}.{ext}"

                result = visualizer.visualize(G, **viz_params)

                return {
                    "backend": backend,
                    "output_path": result,
                    "graph_id": graph_id,
                    "layout": layout,
                    "success": True,
                }

            except ImportError as e:
                return {"error": f"Backend '{backend}' not available: {str(e)}"}
            except Exception as e:
                return {"error": f"Visualization failed: {str(e)}"}

        @self.mcp.tool()
        async def visualize_subgraph(
            graph_id: str,
            center_node: Optional[str] = None,
            k_hop: int = 2,
            max_nodes: int = 50,
            backend: str = "matplotlib",
            layout: str = "spring",
            **kwargs
        ) -> Dict[str, Any]:
            """Visualize a subgraph around specific nodes.

            Args:
                graph_id: ID of the graph
                center_node: Center node for k-hop neighborhood
                k_hop: Number of hops from center
                max_nodes: Maximum nodes to include
                backend: Visualization backend
                layout: Layout algorithm
                **kwargs: Additional visualization parameters

            Returns:
                Dict with visualization details
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                import networkx as nx

                # Extract subgraph
                if center_node:
                    if center_node not in G:
                        return {"error": f"Node '{center_node}' not found"}

                    # Get k-hop neighborhood
                    nodes = {center_node}
                    for _ in range(k_hop):
                        neighbors = set()
                        for node in nodes:
                            neighbors.update(G.neighbors(node))
                        nodes.update(neighbors)
                        if len(nodes) > max_nodes:
                            break

                    subgraph = G.subgraph(list(nodes)[:max_nodes])
                else:
                    # Get largest component if no center specified
                    if G.is_directed():
                        components = nx.weakly_connected_components(G)
                    else:
                        components = nx.connected_components(G)

                    largest_component = max(components, key=len)
                    nodes = list(largest_component)[:max_nodes]
                    subgraph = G.subgraph(nodes)

                # Create temporary subgraph and visualize
                temp_graph_id = f"{graph_id}_subgraph"
                self.graph_manager.add_graph(temp_graph_id, subgraph)

                # Visualize subgraph
                result = await visualize_graph(
                    temp_graph_id,
                    backend=backend,
                    layout=layout,
                    title=f"Subgraph of {graph_id}",
                    **kwargs
                )

                # Clean up temporary graph
                self.graph_manager.delete_graph(temp_graph_id)

                result["subgraph_info"] = {
                    "nodes": subgraph.number_of_nodes(),
                    "edges": subgraph.number_of_edges(),
                    "center_node": center_node,
                    "k_hop": k_hop if center_node else None,
                }

                return result

            except Exception as e:
                return {"error": f"Subgraph visualization failed: {str(e)}"}

        @self.mcp.tool()
        async def visualize_communities(
            graph_id: str,
            method: str = "louvain",
            backend: str = "matplotlib",
            layout: str = "spring",
            **kwargs
        ) -> Dict[str, Any]:
            """Visualize graph with communities highlighted.

            Args:
                graph_id: ID of the graph
                method: Community detection method
                backend: Visualization backend
                layout: Layout algorithm
                **kwargs: Additional parameters

            Returns:
                Dict with visualization details
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                # Detect communities using the analysis handler
                from networkx_mcp.server.handlers.analysis import AnalysisHandler

                # Create temporary handler for community detection
                temp_handler = AnalysisHandler(self.mcp, self.graph_manager)

                # Get communities
                community_result = await temp_handler.community_detection(
                    graph_id, method=method
                )

                if "error" in community_result:
                    return community_result

                communities = community_result["communities"]

                # Create node color mapping based on communities
                node_colors = {}
                colors = [
                    "red",
                    "blue",
                    "green",
                    "yellow",
                    "purple",
                    "orange",
                    "cyan",
                    "magenta",
                    "brown",
                    "pink",
                ]

                for i, community in enumerate(communities):
                    color = colors[i % len(colors)]
                    for node in community:
                        node_colors[node] = color

                # Add community colors as node attributes
                import networkx as nx

                for node in G.nodes():
                    G.nodes[node]["community_color"] = node_colors.get(node, "gray")

                # Visualize with community colors
                result = await visualize_graph(
                    graph_id,
                    backend=backend,
                    layout=layout,
                    node_color="community_color",  # Use attribute for coloring
                    title=f"Communities in {graph_id} ({method})",
                    **kwargs
                )

                result["community_info"] = {
                    "method": method,
                    "num_communities": len(communities),
                    "modularity": community_result.get("modularity", 0),
                }

                return result

            except Exception as e:
                return {"error": f"Community visualization failed: {str(e)}"}

        @self.mcp.tool()
        async def visualize_path(
            graph_id: str,
            source: str,
            target: str,
            path_type: str = "shortest",
            backend: str = "matplotlib",
            layout: str = "spring",
            highlight_color: str = "red",
            **kwargs
        ) -> Dict[str, Any]:
            """Visualize a path in the graph.

            Args:
                graph_id: ID of the graph
                source: Source node
                target: Target node
                path_type: Type of path ('shortest', 'all_shortest')
                backend: Visualization backend
                layout: Layout algorithm
                highlight_color: Color for highlighting the path
                **kwargs: Additional parameters

            Returns:
                Dict with visualization details
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                import networkx as nx

                # Find path
                if path_type == "shortest":
                    try:
                        path = nx.shortest_path(G, source, target)
                        paths = [path]
                    except nx.NetworkXNoPath:
                        return {"error": f"No path from '{source}' to '{target}'"}
                elif path_type == "all_shortest":
                    try:
                        paths = list(nx.all_shortest_paths(G, source, target))
                    except nx.NetworkXNoPath:
                        return {"error": f"No path from '{source}' to '{target}'"}
                else:
                    return {"error": f"Unknown path type: {path_type}"}

                # Create edge colors
                path_edges = set()
                for path in paths:
                    for i in range(len(path) - 1):
                        path_edges.add((path[i], path[i + 1]))
                        if not G.is_directed():
                            path_edges.add((path[i + 1], path[i]))

                # Set edge attributes
                for edge in G.edges():
                    if edge in path_edges:
                        G.edges[edge]["highlight"] = highlight_color
                    else:
                        G.edges[edge]["highlight"] = "gray"

                # Set node attributes
                path_nodes = set()
                for path in paths:
                    path_nodes.update(path)

                for node in G.nodes():
                    if node in path_nodes:
                        G.nodes[node]["highlight"] = highlight_color
                    else:
                        G.nodes[node]["highlight"] = "lightgray"

                # Visualize
                result = await visualize_graph(
                    graph_id,
                    backend=backend,
                    layout=layout,
                    node_color="highlight",
                    edge_color="highlight",
                    title=f"Path from {source} to {target}",
                    **kwargs
                )

                result["path_info"] = {
                    "source": source,
                    "target": target,
                    "num_paths": len(paths),
                    "path_length": len(paths[0]) - 1 if paths else 0,
                    "paths": paths[:5],  # First 5 paths
                }

                return result

            except Exception as e:
                return {"error": f"Path visualization failed: {str(e)}"}

        @self.mcp.tool()
        async def export_visualization_data(
            graph_id: str, format: str = "json", include_positions: bool = True
        ) -> Dict[str, Any]:
            """Export graph data formatted for visualization tools.

            Args:
                graph_id: ID of the graph
                format: Export format ('json', 'd3', 'graphml')
                include_positions: Whether to include layout positions

            Returns:
                Dict with exported data
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                import networkx as nx
                import json

                # Calculate positions if requested
                if include_positions:
                    pos = nx.spring_layout(G)
                else:
                    pos = {}

                if format == "json" or format == "d3":
                    # D3.js compatible format
                    nodes = []
                    for node in G.nodes():
                        node_data = {
                            "id": str(node),
                            "label": str(node),
                            **G.nodes[node],
                        }
                        if node in pos:
                            node_data["x"] = float(pos[node][0])
                            node_data["y"] = float(pos[node][1])
                        nodes.append(node_data)

                    edges = []
                    for u, v, data in G.edges(data=True):
                        edge_data = {
                            "source": str(u),
                            "target": str(v),
                            **data,
                        }
                        edges.append(edge_data)

                    export_data = {
                        "nodes": nodes,
                        "links": edges,
                        "directed": G.is_directed(),
                        "multigraph": G.is_multigraph(),
                    }

                    # Save to file
                    output_path = f"graph_{graph_id}_visualization.json"
                    with open(output_path, "w") as f:
                        json.dump(export_data, f, indent=2)

                elif format == "graphml":
                    # GraphML format with positions
                    for node in G.nodes():
                        if node in pos:
                            G.nodes[node]["x"] = float(pos[node][0])
                            G.nodes[node]["y"] = float(pos[node][1])

                    output_path = f"graph_{graph_id}_visualization.graphml"
                    nx.write_graphml(G, output_path)

                else:
                    return {"error": f"Unknown format: {format}"}

                return {
                    "format": format,
                    "output_path": output_path,
                    "num_nodes": G.number_of_nodes(),
                    "num_edges": G.number_of_edges(),
                    "file_size": os.path.getsize(output_path),
                }

            except Exception as e:
                return {"error": f"Export failed: {str(e)}"}


__all__ = ["VisualizationHandler"]