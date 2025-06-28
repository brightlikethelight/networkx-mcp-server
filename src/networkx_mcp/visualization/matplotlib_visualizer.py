"""Matplotlib-based static graph visualization."""

import base64
import logging

from io import BytesIO
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


logger = logging.getLogger(__name__)


class MatplotlibVisualizer:
    """Create high-quality static graph visualizations using matplotlib."""

    # Available layout algorithms
    LAYOUT_ALGORITHMS = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "shell": nx.shell_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "spectral": nx.spectral_layout,
        "random": nx.random_layout,
        "hierarchical": None  # Custom implementation
    }

    # Node shape mappings
    NODE_SHAPES = {
        "circle": "o",
        "square": "s",
        "triangle": "^",
        "diamond": "D",
        "hexagon": "h",
        "octagon": "8",
        "pentagon": "p",
        "star": "*"
    }

    @staticmethod
    def create_static_plot(
        graph: Union[nx.Graph, nx.DiGraph],
        layout: str = "spring",
        node_size: Union[int, Dict, str] = 300,
        node_color: Union[str, Dict, str] = "lightblue",
        node_shape: Union[str, Dict] = "circle",
        edge_width: Union[float, Dict, str] = 1.0,
        edge_color: Union[str, Dict, str] = "gray",
        edge_style: Union[str, Dict] = "solid",
        show_labels: bool = True,
        label_font_size: int = 10,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a static graph visualization with extensive customization options.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            The graph to visualize
        layout : str
            Layout algorithm to use
        node_size : int, dict, or attribute name
            Node sizes (can vary by node)
        node_color : str, dict, or attribute name
            Node colors (can vary by node)
        node_shape : str or dict
            Node shapes (can vary by node)
        edge_width : float, dict, or attribute name
            Edge widths (can vary by edge)
        edge_color : str, dict, or attribute name
            Edge colors (can vary by edge)
        edge_style : str or dict
            Edge styles ('solid', 'dashed', 'dotted')
        show_labels : bool
            Whether to show node labels
        label_font_size : int
            Font size for labels
        title : str
            Plot title
        figsize : tuple
            Figure size (width, height)
        dpi : int
            Dots per inch for resolution

        Returns:
        --------
        Dict containing plot data and metadata
        """
        plt.figure(figsize=figsize, dpi=dpi)

        # Get layout positions
        if layout == "hierarchical":
            pos = MatplotlibVisualizer._hierarchical_layout(graph)
        else:
            layout_func = MatplotlibVisualizer.LAYOUT_ALGORITHMS.get(layout, nx.spring_layout)
            pos = layout_func(graph, **kwargs.get("layout_params", {}))

        # Process node attributes
        node_sizes = MatplotlibVisualizer._get_node_attributes(
            graph, node_size, default=300
        )
        node_colors = MatplotlibVisualizer._get_node_attributes(
            graph, node_color, default="lightblue"
        )

        # Process edge attributes
        edge_widths = MatplotlibVisualizer._get_edge_attributes(
            graph, edge_width, default=1.0
        )
        edge_colors = MatplotlibVisualizer._get_edge_attributes(
            graph, edge_color, default="gray"
        )
        edge_styles = MatplotlibVisualizer._get_edge_attributes(
            graph, edge_style, default="solid"
        )

        # Draw edges with custom styles
        for (u, v), width, color, style in zip(
            graph.edges(), edge_widths, edge_colors, edge_styles
        ):
            if graph.is_directed():
                # Draw directed edges with arrows
                nx.draw_networkx_edges(
                    graph, pos,
                    edgelist=[(u, v)],
                    width=width,
                    edge_color=[color],
                    style=style,
                    arrows=True,
                    arrowsize=10,
                    arrowstyle="->",
                    alpha=0.7
                )
            else:
                # Draw undirected edges
                nx.draw_networkx_edges(
                    graph, pos,
                    edgelist=[(u, v)],
                    width=width,
                    edge_color=[color],
                    style=style,
                    alpha=0.7
                )

        # Draw nodes by shape groups
        if isinstance(node_shape, dict):
            shape_groups = {}
            for node in graph.nodes():
                shape = node_shape.get(node, "circle")
                if shape not in shape_groups:
                    shape_groups[shape] = []
                shape_groups[shape].append(node)

            for shape, nodes in shape_groups.items():
                shape_marker = MatplotlibVisualizer.NODE_SHAPES.get(shape, "o")
                nx.draw_networkx_nodes(
                    graph, pos,
                    nodelist=nodes,
                    node_size=[node_sizes[i] for i, n in enumerate(graph.nodes()) if n in nodes],
                    node_color=[node_colors[i] for i, n in enumerate(graph.nodes()) if n in nodes],
                    node_shape=shape_marker,
                    alpha=0.9
                )
        else:
            # Draw all nodes with same shape
            shape_marker = MatplotlibVisualizer.NODE_SHAPES.get(node_shape, "o")
            nx.draw_networkx_nodes(
                graph, pos,
                node_size=node_sizes,
                node_color=node_colors,
                node_shape=shape_marker,
                alpha=0.9
            )

        # Draw labels with smart placement
        if show_labels:
            if kwargs.get("smart_labels", True) and graph.number_of_nodes() > 20:
                # Smart label placement for dense graphs
                MatplotlibVisualizer._smart_label_placement(
                    graph, pos, label_font_size
                )
            else:
                nx.draw_networkx_labels(
                    graph, pos,
                    font_size=label_font_size,
                    font_color="black"
                )

        # Add title
        if title:
            plt.title(title, fontsize=16, fontweight="bold")

        # Add legend if using attribute-based coloring
        if isinstance(node_color, str) and node_color in graph.nodes[next(iter(graph.nodes()))]:
            MatplotlibVisualizer._add_color_legend(graph, node_color)

        # Remove axes
        plt.axis("off")
        plt.tight_layout()

        # Save to different formats
        results = {
            "layout_used": layout,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "formats": {}
        }

        # Save as PNG (base64)
        png_buffer = BytesIO()
        plt.savefig(png_buffer, format="png", dpi=dpi, bbox_inches="tight")
        png_buffer.seek(0)
        results["formats"]["png_base64"] = base64.b64encode(png_buffer.read()).decode()

        # Save as SVG
        if kwargs.get("include_svg", True):
            svg_buffer = BytesIO()
            plt.savefig(svg_buffer, format="svg", bbox_inches="tight")
            svg_buffer.seek(0)
            results["formats"]["svg"] = svg_buffer.read().decode()

        plt.close()

        return results

    @staticmethod
    def _hierarchical_layout(graph: nx.Graph) -> Dict[Any, Tuple[float, float]]:
        """Create hierarchical layout for tree-like structures."""
        if nx.is_directed_acyclic_graph(graph):
            # Use topological generations for DAGs
            generations = list(nx.topological_generations(graph))
            pos = {}

            for i, generation in enumerate(generations):
                for j, node in enumerate(generation):
                    x = (j - len(generation)/2) * 2 / max(1, len(generation))
                    y = -i
                    pos[node] = (x, y)

            return pos
        else:
            # Fall back to spring layout for non-DAGs
            return nx.spring_layout(graph)

    @staticmethod
    def _get_node_attributes(
        graph: nx.Graph,
        attr: Union[int, float, str, Dict],
        default: Any
    ) -> List[Any]:
        """Get node attributes as a list."""
        if isinstance(attr, dict):
            return [attr.get(node, default) for node in graph.nodes()]
        elif isinstance(attr, str):
            # Attribute name
            return [graph.nodes[node].get(attr, default) for node in graph.nodes()]
        else:
            # Single value
            return [attr] * graph.number_of_nodes()

    @staticmethod
    def _get_edge_attributes(
        graph: nx.Graph,
        attr: Union[int, float, str, Dict],
        default: Any
    ) -> List[Any]:
        """Get edge attributes as a list."""
        if isinstance(attr, dict):
            return [attr.get(edge, default) for edge in graph.edges()]
        elif isinstance(attr, str):
            # Attribute name
            return [graph.edges[edge].get(attr, default) for edge in graph.edges()]
        else:
            # Single value
            return [attr] * graph.number_of_edges()

    @staticmethod
    def _smart_label_placement(
        graph: nx.Graph,
        pos: Dict[Any, Tuple[float, float]],
        font_size: int
    ) -> Dict[Any, str]:
        """Smart label placement to avoid overlap."""
        # For now, show labels only for high-degree nodes
        degrees = dict(graph.degree())
        threshold = np.percentile(list(degrees.values()), 75)

        labels = {}
        for node, degree in degrees.items():
            if degree >= threshold:
                labels[node] = str(node)

        nx.draw_networkx_labels(
            graph, pos,
            labels=labels,
            font_size=font_size,
            font_color="black"
        )

        return labels

    @staticmethod
    def _add_color_legend(graph: nx.Graph, color_attr: str):
        """Add color legend for attribute-based coloring."""
        # Get unique values
        values = set()
        for node in graph.nodes():
            val = graph.nodes[node].get(color_attr)
            if val is not None:
                values.add(val)

        # Create color map
        cmap = plt.cm.get_cmap("viridis")
        colors = [cmap(i / len(values)) for i in range(len(values))]

        # Create legend
        patches = []
        for val, color in zip(sorted(values), colors):
            patches.append(mpatches.Patch(color=color, label=str(val)))

        plt.legend(handles=patches, title=color_attr, loc="best")

    @staticmethod
    def create_subplot_layout(
        graph: nx.Graph,
        views: List[Dict[str, Any]],
        figsize: Tuple[int, int] = (16, 8),
        dpi: int = 100
    ) -> Dict[str, Any]:
        """Create multiple graph views in subplots."""
        num_views = len(views)
        fig, axes = plt.subplots(1, num_views, figsize=figsize, dpi=dpi)

        if num_views == 1:
            axes = [axes]

        results = {"views": []}

        for ax, view_config in zip(axes, views):
            plt.sca(ax)

            # Apply view configuration
            layout = view_config.get("layout", "spring")

            # Get layout positions
            if layout == "hierarchical":
                pos = MatplotlibVisualizer._hierarchical_layout(graph)
            else:
                layout_func = MatplotlibVisualizer.LAYOUT_ALGORITHMS.get(
                    layout, nx.spring_layout
                )
                pos = layout_func(graph)

            # Draw the graph with view-specific settings
            nx.draw(
                graph, pos,
                node_color=view_config.get("node_color", "lightblue"),
                node_size=view_config.get("node_size", 300),
                edge_color=view_config.get("edge_color", "gray"),
                width=view_config.get("edge_width", 1.0),
                with_labels=view_config.get("show_labels", True),
                font_size=view_config.get("font_size", 10),
                ax=ax
            )

            ax.set_title(view_config.get("title", f"View {layout}"))
            ax.axis("off")

            results["views"].append({
                "layout": layout,
                "title": view_config.get("title", f"View {layout}")
            })

        plt.tight_layout()

        # Save result
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        buffer.seek(0)
        results["png_base64"] = base64.b64encode(buffer.read()).decode()

        plt.close()

        return results
