"""Matplotlib-based graph visualization."""

import base64
import io

import matplotlib.pyplot as plt
import networkx as nx

from networkx_mcp.visualization.base import BaseVisualizer
from networkx_mcp.visualization.base import VisualizationResult
from networkx_mcp.visualization.base import calculate_layout


class MatplotlibVisualizer(BaseVisualizer):
    """Matplotlib backend for graph visualization."""

    def __init__(self):
        super().__init__("matplotlib", "png")
        self.default_options = {
            "node_size": 300,
            "node_color": "lightblue",
            "edge_color": "gray",
            "with_labels": True,
            "font_size": 10,
            "figure_size": (10, 8),
            "dpi": 100
        }

    async def render(self, graph: nx.Graph, layout: str = "spring", **options) -> VisualizationResult:
        """Render graph using matplotlib."""
        try:
            # Merge options
            opts = {**self.default_options, **options}

            # Calculate layout
            pos = calculate_layout(graph, layout, k=opts.get("k", 1), iterations=opts.get("iterations", 50))

            # Create figure
            fig, ax = plt.subplots(figsize=opts["figure_size"], dpi=opts["dpi"])

            # Draw graph
            nx.draw_networkx_nodes(
                graph, pos, ax=ax,
                node_size=opts["node_size"],
                node_color=opts["node_color"],
                alpha=0.8
            )

            nx.draw_networkx_edges(
                graph, pos, ax=ax,
                edge_color=opts["edge_color"],
                alpha=0.6,
                arrows=graph.is_directed(),
                arrowsize=20
            )

            if opts["with_labels"]:
                nx.draw_networkx_labels(
                    graph, pos, ax=ax,
                    font_size=opts["font_size"]
                )

            # Style the plot
            ax.set_title(f"Graph Visualization ({len(graph.nodes())} nodes, {len(graph.edges())} edges)")
            ax.axis("off")
            plt.tight_layout()

            # Convert to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)

            # Create HTML with embedded image
            html_output = f'<div class="graph-visualization"><img src="data:image/png;base64,{image_base64}" alt="Graph Visualization" style="max-width: 100%; height: auto;"><div class="graph-info"><p>Layout: {layout} | Nodes: {len(graph.nodes())} | Edges: {len(graph.edges())}</p></div></div>'

            return VisualizationResult(
                output=html_output,
                format="html",
                metadata={
                    "layout": layout,
                    "nodes": len(graph.nodes()),
                    "edges": len(graph.edges()),
                    "options": opts
                }
            )

        except Exception as e:
            return VisualizationResult(
                output="",
                format="html",
                metadata={},
                success=False,
                error=str(e)
            )

async def create_matplotlib_visualization(graph: nx.Graph, layout: str = "spring", **options) -> str:
    """Simple function interface for matplotlib visualization."""
    visualizer = MatplotlibVisualizer()
    result = await visualizer.render(graph, layout, **options)
    return result.output
