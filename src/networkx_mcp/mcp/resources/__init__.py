"""MCP Resources for NetworkX Server.

Resources provide read-only access to graph data and analysis results.
They are similar to GET endpoints in REST APIs - no side effects.
"""

from ...compat.fastmcp_compat import FastMCPCompat as FastMCP

try:
    from mcp.types import Resource, ResourceContents, TextResourceContents
except ImportError:
    # Fallback for compatibility
    Resource = dict
    ResourceContents = dict
    TextResourceContents = dict


class GraphResources:
    """MCP Resources for graph data access."""

    def __init__(self, mcp: FastMCP, graph_manager):
        """Initialize resources with MCP server and graph manager."""
        self.mcp = mcp
        self.graph_manager = graph_manager
        self._register_resources()

    def _register_resources(self):
        """Register all available resources."""

        # Graph catalog resource
        @self.mcp.resource("graph://catalog")
        async def graph_catalog() -> ResourceContents:
            """List all available graphs with their metadata."""
            graphs = self.graph_manager.list_graphs()
            catalog = []

            for graph_id in graphs:
                graph = self.graph_manager.get_graph(graph_id)
                if graph:
                    catalog.append(
                        {
                            "id": graph_id,
                            "type": graph.__class__.__name__,
                            "nodes": graph.number_of_nodes(),
                            "edges": graph.number_of_edges(),
                            "directed": graph.is_directed(),
                            "multigraph": graph.is_multigraph(),
                        }
                    )

            return TextResourceContents(
                uri="graph://catalog", mimeType="application/json", text=str(catalog)
            )

        # Individual graph data resource
        @self.mcp.resource("graph://data/{graph_id}")
        async def graph_data(graph_id: str) -> ResourceContents:
            """Get complete graph data in JSON format."""
            graph = self.graph_manager.get_graph(graph_id)
            if not graph:
                return TextResourceContents(
                    uri=f"graph://data/{graph_id}",
                    mimeType="application/json",
                    text='{"error": "Graph not found"}',
                )

            from networkx.readwrite import json_graph

            data = json_graph.node_link_data(graph)

            return TextResourceContents(
                uri=f"graph://data/{graph_id}",
                mimeType="application/json",
                text=str(data),
            )

        # Graph statistics resource
        @self.mcp.resource("graph://stats/{graph_id}")
        async def graph_stats(graph_id: str) -> ResourceContents:
            """Get detailed statistics for a graph."""
            graph = self.graph_manager.get_graph(graph_id)
            if not graph:
                return TextResourceContents(
                    uri=f"graph://stats/{graph_id}",
                    mimeType="application/json",
                    text='{"error": "Graph not found"}',
                )

            import networkx as nx

            stats = {
                "basic": {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "density": nx.density(graph),
                    "is_directed": graph.is_directed(),
                    "is_multigraph": graph.is_multigraph(),
                },
                "connectivity": {
                    "is_connected": (
                        nx.is_connected(graph)
                        if not graph.is_directed()
                        else nx.is_weakly_connected(graph)
                    ),
                    "number_connected_components": (
                        nx.number_connected_components(graph)
                        if not graph.is_directed()
                        else nx.number_weakly_connected_components(graph)
                    ),
                },
                "degree": {
                    "average_degree": (
                        sum(d for n, d in graph.degree()) / graph.number_of_nodes()
                        if graph.number_of_nodes() > 0
                        else 0
                    ),
                    "max_degree": (
                        max(d for n, d in graph.degree())
                        if graph.number_of_nodes() > 0
                        else 0
                    ),
                    "min_degree": (
                        min(d for n, d in graph.degree())
                        if graph.number_of_nodes() > 0
                        else 0
                    ),
                },
            }

            # Add clustering coefficient for undirected graphs
            if not graph.is_directed():
                stats["clustering"] = {
                    "average_clustering": nx.average_clustering(graph),
                    "transitivity": nx.transitivity(graph),
                }

            return TextResourceContents(
                uri=f"graph://stats/{graph_id}",
                mimeType="application/json",
                text=str(stats),
            )

        # Algorithm results cache resource
        @self.mcp.resource("graph://results/{graph_id}/{algorithm}")
        async def algorithm_results(graph_id: str, algorithm: str) -> ResourceContents:
            """Get cached results from previous algorithm runs."""
            # This would integrate with a results cache in production
            # For now, return a placeholder
            return TextResourceContents(
                uri=f"graph://results/{graph_id}/{algorithm}",
                mimeType="application/json",
                text='{"status": "No cached results available"}',
            )

        # Visualization data resource
        @self.mcp.resource("graph://viz/{graph_id}")
        async def visualization_data(graph_id: str) -> ResourceContents:
            """Get graph data optimized for visualization."""
            graph = self.graph_manager.get_graph(graph_id)
            if not graph:
                return TextResourceContents(
                    uri=f"graph://viz/{graph_id}",
                    mimeType="application/json",
                    text='{"error": "Graph not found"}',
                )

            # Prepare visualization-friendly data
            import networkx as nx

            # Use spring layout for positions
            pos = nx.spring_layout(graph)

            viz_data = {
                "nodes": [
                    {
                        "id": str(node),
                        "x": pos[node][0],
                        "y": pos[node][1],
                        "label": str(node),
                        **graph.nodes[node],  # Include node attributes
                    }
                    for node in graph.nodes()
                ],
                "edges": [
                    {
                        "source": str(u),
                        "target": str(v),
                        **graph.edges[u, v],  # Include edge attributes
                    }
                    for u, v in graph.edges()
                ],
            }

            return TextResourceContents(
                uri=f"graph://viz/{graph_id}",
                mimeType="application/json",
                text=str(viz_data),
            )


__all__ = ["GraphResources"]
