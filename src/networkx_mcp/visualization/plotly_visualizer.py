"""Plotly-based interactive graph visualization."""

import logging
from typing import Any

import networkx as nx
import numpy as np

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None

logger = logging.getLogger(__name__)


class PlotlyVisualizer:
    """Create interactive graph visualizations using Plotly."""

    def __init__(self) -> None:
        """Initialize PlotlyVisualizer."""
        if not HAS_PLOTLY:
            raise ImportError(
                "plotly is required for PlotlyVisualizer. Install with: pip install plotly"
            )

    @staticmethod
    def create_interactive_plot(graph: nx.Graph | nx.DiGraph, layout: str = "spring", node_size: int | dict[str, Any] | str = 10, node_color: str | dict[str, Any] | str = "degree", edge_width: float | dict[str, Any] | str = 1.0, edge_color: str | dict[str, Any] | str = "gray", hover_data: list[str] | None = None, title: str | None = None, height: int = 800, width: int = 1200, show_edge_labels: bool = False, **kwargs) -> dict[str, Any]:
        """
        Create an interactive graph visualization with zoom, pan, and hover.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            The graph to visualize
        layout : str
            Layout algorithm ('spring', 'circular', 'random', 'spectral')
        node_size : int, dict[str, Any], or attribute name
            Node sizes
        node_color : str, dict[str, Any], or attribute name
            Node colors (can be 'degree', 'betweenness', etc.)
        hover_data : list[Any]
            Node attributes to show on hover

        Returns:
        --------
        Dict containing the interactive plot and metadata
        """
        # Calculate layout
        if layout == "spring":
            pos = nx.spring_layout(
                graph, k=1 / np.sqrt(graph.number_of_nodes()), iterations=50
            )
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        elif layout == "spectral":
            pos = nx.spectral_layout(graph)
        else:
            pos = nx.random_layout(graph)

        # Create edge traces
        edge_traces = PlotlyVisualizer._create_edge_traces(
            graph, pos, edge_width, edge_color, show_edge_labels
        )

        # Create node trace
        node_trace = PlotlyVisualizer._create_node_trace(
            graph, pos, node_size, node_color, hover_data
        )

        # Create figure
        fig = go.Figure(
            data=[*edge_traces, node_trace],
            layout=go.Layout(
                title={
                    "text": title or "Interactive Network Graph",
                    "font": {"size": 16},
                },
                showlegend=False,
                hovermode="closest",
                margin={"b": 20, "l": 5, "r": 5, "t": 40},
                annotations=[],
                xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                height=height,
                width=width,
            ),
        )

        # Add interactivity options
        fig.update_layout(
            dragmode="pan",
            modebar_add=["select2d", "lasso2d"],
            modebar_remove=["pan2d", "zoomIn2d", "zoomOut2d"],
        )

        # Generate HTML
        html_str = fig.to_html(include_plotlyjs="cdn")

        return {
            "figure": fig.to_dict(),
            "html": html_str,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "layout_used": layout,
        }

    @staticmethod
    def create_3d_plot(graph: nx.Graph | nx.DiGraph, layout: str = "spring3d", node_color: str | dict[str, Any] | str = "degree", title: str | None = None, height: int = 800, width: int = 1200) -> dict[str, Any]:
        """Create 3D force-directed graph visualization."""
        # Calculate 3D layout
        if layout == "spring3d":
            pos = nx.spring_layout(graph, dim=3, k=1 / np.sqrt(graph.number_of_nodes()))
        else:
            # Random 3D positions
            pos = {
                node: (np.random.rand(), np.random.rand(), np.random.rand())
                for node in graph.nodes()
            }

        # Extract coordinates
        x_nodes = [pos[node][0] for node in graph.nodes()]
        y_nodes = [pos[node][1] for node in graph.nodes()]
        z_nodes = [pos[node][2] for node in graph.nodes()]

        # Create edge traces
        edge_traces = []
        for edge in graph.edges():
            x_edge = [pos[edge[0]][0], pos[edge[1]][0], None]
            y_edge = [pos[edge[0]][1], pos[edge[1]][1], None]
            z_edge = [pos[edge[0]][2], pos[edge[1]][2], None]

            edge_trace = go.Scatter3d(
                x=x_edge,
                y=y_edge,
                z=z_edge,
                mode="lines",
                line={"color": "rgba(125,125,125,0.5)", "width": 1},
                hoverinfo="none",
            )
            edge_traces.append(edge_trace)

        # Node colors
        if node_color == "degree":
            node_colors = list[Any](dict[str, Any](graph.degree()).values())
        elif node_color == "betweenness":
            node_colors = list[Any](nx.betweenness_centrality(graph).values())
        else:
            node_colors = node_color

        # Create node trace
        node_trace = go.Scatter3d(
            x=x_nodes,
            y=y_nodes,
            z=z_nodes,
            mode="markers+text",
            marker={
                "size": 10,
                "color": node_colors,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {
                    "title": node_color if isinstance(node_color, str) else "Value",
                    "tickmode": "linear",
                },
            },
            text=[str(node) for node in graph.nodes()],
            textposition="top center",
            hoverinfo="text",
            hovertext=[
                f"Node: {node}<br>Degree: {graph.degree(node)}"
                for node in graph.nodes()
            ],
        )

        # Create figure
        fig = go.Figure(data=[*edge_traces, node_trace])

        fig.update_layout(
            title=title or "3D Network Visualization",
            width=width,
            height=height,
            showlegend=False,
            scene={
                "xaxis": {"showgrid": False, "showticklabels": False, "title": ""},
                "yaxis": {"showgrid": False, "showticklabels": False, "title": ""},
                "zaxis": {"showgrid": False, "showticklabels": False, "title": ""},
                "bgcolor": "rgba(0,0,0,0)",
            },
            margin={"l": 0, "r": 0, "b": 0, "t": 30},
        )

        return {
            "figure": fig.to_dict(),
            "html": fig.to_html(include_plotlyjs="cdn"),
            "layout_type": "3d",
        }

    @staticmethod
    def create_animated_plot(graphs: list[nx.Graph], timestamps: list[str] | None = None, layout: str = "spring", title: str = "Temporal Network Animation", height: int = 800, width: int = 1200) -> dict[str, Any]:
        """Create animated visualization for temporal networks."""
        if not timestamps:
            timestamps = [f"T{i}" for i in range(len(graphs))]

        # Calculate consistent layout across all graphs
        # Use union of all graphs for consistent positioning
        union_graph = nx.Graph()
        for g in graphs:
            union_graph.add_nodes_from(g.nodes())
            union_graph.add_edges_from(g.edges())

        if layout == "spring":
            base_pos = nx.spring_layout(union_graph)
        else:
            base_pos = nx.circular_layout(union_graph)

        # Create frames
        frames = []
        for _i, (graph, timestamp) in enumerate(zip(graphs, timestamps, strict=False)):
            # Create traces for this frame
            edge_traces = PlotlyVisualizer._create_edge_traces(
                graph, base_pos, edge_width=1.0, edge_color="gray"
            )

            node_trace = PlotlyVisualizer._create_node_trace(
                graph, base_pos, node_size=10, node_color="degree"
            )

            frame = go.Frame(data=[*edge_traces, node_trace], name=str(timestamp))
            frames.append(frame)

        # Create initial figure
        fig = go.Figure(data=frames[0].data, frames=frames)

        # Add animation controls
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 300},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "steps": [
                        {
                            "label": str(timestamp),
                            "method": "animate",
                            "args": [
                                [str(timestamp)],
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 300},
                                },
                            ],
                        }
                        for timestamp in timestamps
                    ],
                }
            ],
        )

        return {
            "figure": fig.to_dict(),
            "html": fig.to_html(include_plotlyjs="cdn"),
            "num_frames": len(frames),
            "timestamps": timestamps,
        }

    @staticmethod
    def _create_edge_traces(graph: nx.Graph, pos: dict[str, Any], edge_width: float | str | dict[str, Any], edge_color: str | dict[str, Any], show_labels: bool = False) -> list[Any]:
        """Create edge traces for plotly."""
        if not HAS_PLOTLY:
            return []
        edge_traces = []

        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            # Get edge attributes
            if isinstance(edge_width, dict[str, Any]):
                width = edge_width.get(edge, 1.0)
            elif isinstance(edge_width, str):
                width = graph.edges[edge].get(edge_width, 1.0)
            else:
                width = edge_width

            if isinstance(edge_color, dict[str, Any]):
                color = edge_color.get(edge, "gray")
            elif isinstance(edge_color, str) and edge_color != "gray":
                color = graph.edges[edge].get(edge_color, "gray")
            else:
                color = edge_color

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line={"width": width, "color": color},
                hoverinfo="none",
            )
            edge_traces.append(edge_trace)

            # Add edge labels if requested
            if show_labels and "weight" in graph.edges[edge]:
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                edge_label = go.Scatter(
                    x=[mid_x],
                    y=[mid_y],
                    mode="text",
                    text=[str(graph.edges[edge]["weight"])],
                    textposition="middle center",
                    hoverinfo="none",
                )
                edge_traces.append(edge_label)

        return edge_traces

    @staticmethod
    def _create_node_trace(graph: nx.Graph, pos: dict[str, Any], node_size: int | str | dict[str, Any], node_color: str | dict[str, Any], hover_data: list[str] | None = None) -> Any:
        """Create node trace for plotly."""
        if not HAS_PLOTLY:
            return None
        x_nodes = []
        y_nodes = []

        for node in graph.nodes():
            x, y = pos[node]
            x_nodes.append(x)
            y_nodes.append(y)

        # Node sizes
        if isinstance(node_size, dict[str, Any]):
            sizes = [node_size.get(node, 10) for node in graph.nodes()]
        elif isinstance(node_size, str):
            sizes = [graph.nodes[node].get(node_size, 10) for node in graph.nodes()]
        else:
            sizes = [node_size] * graph.number_of_nodes()

        # Node colors
        if node_color == "degree":
            colors = list[Any](dict[str, Any](graph.degree()).values())
            colorbar_title = "Degree"
        elif node_color == "betweenness":
            colors = list[Any](nx.betweenness_centrality(graph).values())
            colorbar_title = "Betweenness"
        elif node_color == "closeness":
            colors = list[Any](nx.closeness_centrality(graph).values())
            colorbar_title = "Closeness"
        elif isinstance(node_color, dict[str, Any]):
            colors = [node_color.get(node, 0) for node in graph.nodes()]
            colorbar_title = "Value"
        elif isinstance(node_color, str):
            colors = [graph.nodes[node].get(node_color, 0) for node in graph.nodes()]
            colorbar_title = node_color
        else:
            colors = node_color
            colorbar_title = "Value"

        # Hover text
        hover_texts = []
        for node in graph.nodes():
            text = f"Node: {node}<br>Degree: {graph.degree(node)}"
            if hover_data:
                for attr in hover_data:
                    if attr in graph.nodes[node]:
                        text += f"<br>{attr}: {graph.nodes[node][attr]}"
            hover_texts.append(text)

        node_trace = go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode="markers+text",
            marker={
                "size": sizes,
                "color": colors,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": colorbar_title, "tickmode": "linear"},
                "line": {"width": 2, "color": "white"},
            },
            text=[str(node) for node in graph.nodes()],
            textposition="top center",
            hoverinfo="text",
            hovertext=hover_texts,
        )

        return node_trace

    @staticmethod
    def export_interactive_html(figure: dict[str, Any], filename: str, include_plotlyjs: str = "cdn", config: dict[str, Any] | None = None) -> str:
        """Export interactive plot as standalone HTML."""
        if config is None:
            config = {
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
            }

        # Convert dict[str, Any] back to figure if needed
        if isinstance(figure, dict[str, Any]):
            fig = go.Figure(figure)
        else:
            fig = figure

        # Generate HTML
        html_str = fig.to_html(
            include_plotlyjs=include_plotlyjs, config=config, div_id="networkx-graph"
        )

        # Save to file
        with open(filename, "w") as f:
            f.write(html_str)

        return filename
