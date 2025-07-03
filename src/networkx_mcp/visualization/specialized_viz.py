"""Specialized visualizations for network analysis."""

import base64
import logging
from io import BytesIO
from typing import Any

import networkx as nx
import numpy as np

# Optional imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.cluster.hierarchy import dendrogram, linkage

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None
    dendrogram = None
    linkage = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None
    make_subplots = None

logger = logging.getLogger(__name__)


class SpecializedVisualizations:
    """Create specialized network visualizations."""

    @staticmethod
    def heatmap_adjacency(
        graph: nx.Graph | nx.DiGraph,
        node_order: list | None = None,
        cmap: str = "viridis",
        figsize: tuple[int, int] = (10, 8),
        title: str = "Adjacency Matrix Heatmap",
        show_labels: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Create adjacency matrix heatmap visualization.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            The graph to visualize
        node_order : list, optional
            Order of nodes (useful for clustering visualization)
        cmap : str
            Colormap for the heatmap

        Returns:
        --------
        Dict containing the heatmap visualization
        """
        if not HAS_MATPLOTLIB:
            msg = (
                "Matplotlib is required for heatmap visualization. "
                "Install with: pip install matplotlib seaborn"
            )
            raise ImportError(msg)
        if not HAS_PLOTLY:
            msg = (
                "Plotly is required for interactive heatmap visualization. "
                "Install with: pip install plotly"
            )
            raise ImportError(msg)

        # Get adjacency matrix
        if node_order is None:
            node_order = list(graph.nodes())

        adj_matrix = nx.adjacency_matrix(graph, nodelist=node_order).toarray()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        if show_labels and len(node_order) <= 50:
            sns.heatmap(
                adj_matrix,
                xticklabels=node_order,
                yticklabels=node_order,
                cmap=cmap,
                square=True,
                cbar_kws={"label": "Edge Weight"},
                ax=ax,
            )
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
        else:
            sns.heatmap(
                adj_matrix,
                cmap=cmap,
                square=True,
                cbar_kws={"label": "Edge Weight"},
                ax=ax,
            )

        plt.title(title)
        plt.tight_layout()

        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        # Create interactive version with Plotly
        fig_plotly = go.Figure(
            data=go.Heatmap(
                z=adj_matrix,
                x=node_order,
                y=node_order,
                colorscale=cmap,
                hoverongaps=False,
                hovertemplate="Source: %{y}<br>Target: %{x}<br>Weight: %{z}<extra></extra>",
            )
        )

        fig_plotly.update_layout(
            title=title,
            xaxis={"title": "Target Node", "side": "bottom"},
            yaxis={"title": "Source Node", "autorange": "reversed"},
            width=800,
            height=800,
        )

        return {
            "static_png_base64": img_base64,
            "interactive_plotly": fig_plotly.to_dict(),
            "matrix_shape": adj_matrix.shape,
            "density": nx.density(graph),
        }

    @staticmethod
    def chord_diagram(
        graph: nx.Graph | nx.DiGraph,
        top_nodes: int | None = None,
        min_weight: float = 0,
        title: str = "Network Chord Diagram",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Create chord diagram for network relationships.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            The graph to visualize
        top_nodes : int, optional
            Show only top N nodes by degree
        min_weight : float
            Minimum edge weight to display

        Returns:
        --------
        Dict containing chord diagram visualization
        """
        if not HAS_PLOTLY:
            msg = "Plotly is required for chord diagram visualization. Install with: pip install plotly"
            raise ImportError(msg)

        # Filter nodes if needed
        if top_nodes and graph.number_of_nodes() > top_nodes:
            degrees = dict(graph.degree())
            top_nodes_list = sorted(
                degrees.keys(), key=lambda x: degrees[x], reverse=True
            )[:top_nodes]
            subgraph = graph.subgraph(top_nodes_list).copy()
        else:
            subgraph = graph.copy()

        nodes = list(subgraph.nodes())
        n = len(nodes)

        # Create adjacency matrix
        adj_matrix = nx.adjacency_matrix(subgraph, nodelist=nodes).toarray()

        # Filter by minimum weight
        adj_matrix[adj_matrix < min_weight] = 0

        # Create positions on circle
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)

        # Create traces for Plotly
        traces = []

        # Add chord paths
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] > 0 or adj_matrix[j, i] > 0:
                    weight = max(adj_matrix[i, j], adj_matrix[j, i])

                    # Create bezier curve path
                    t = np.linspace(0, 1, 50)

                    # Control point at center
                    cx, cy = 0, 0

                    # Bezier curve
                    path_x = (1 - t) ** 2 * x[i] + 2 * (1 - t) * t * cx + t**2 * x[j]
                    path_y = (1 - t) ** 2 * y[i] + 2 * (1 - t) * t * cy + t**2 * y[j]

                    trace = go.Scatter(
                        x=path_x,
                        y=path_y,
                        mode="lines",
                        line={
                            "color": f"rgba(100, 100, 100, {min(weight / adj_matrix.max(), 1)})",
                            "width": weight * 3 / adj_matrix.max(),
                        },
                        hoverinfo="text",
                        hovertext=f"{nodes[i]} - {nodes[j]}: {weight:.2f}",
                        showlegend=False,
                    )
                    traces.append(trace)

        # Add nodes
        node_trace = go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            marker={
                "size": 20,
                "color": "lightblue",
                "line": {"color": "darkblue", "width": 2},
            },
            text=nodes,
            textposition="outside",
            textfont={"size": 12},
            hoverinfo="text",
            hovertext=[f"{node}<br>Degree: {subgraph.degree(node)}" for node in nodes],
            showlegend=False,
        )
        traces.append(node_trace)

        # Create figure
        fig = go.Figure(data=traces)

        fig.update_layout(
            title=title,
            showlegend=False,
            xaxis={"showgrid": False, "showticklabels": False, "zeroline": False},
            yaxis={"showgrid": False, "showticklabels": False, "zeroline": False},
            width=800,
            height=800,
            plot_bgcolor="white",
        )

        return {
            "plotly_figure": fig.to_dict(),
            "html": fig.to_html(include_plotlyjs="cdn"),
            "num_nodes": n,
            "num_chords": sum(
                1
                for i in range(n)
                for j in range(i + 1, n)
                if adj_matrix[i, j] > 0 or adj_matrix[j, i] > 0
            ),
        }

    @staticmethod
    def sankey_diagram(
        graph: nx.DiGraph,
        source_nodes: list | None = None,
        target_nodes: list | None = None,
        flow_attribute: str = "weight",
        title: str = "Network Flow Sankey Diagram",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Create Sankey diagram for flow visualization.

        Parameters:
        -----------
        graph : nx.DiGraph
            Directed graph with flow data
        source_nodes : list, optional
            Source nodes (auto-detect if None)
        target_nodes : list, optional
            Target nodes (auto-detect if None)
        flow_attribute : str
            Edge attribute containing flow values

        Returns:
        --------
        Dict containing Sankey diagram
        """
        if not HAS_PLOTLY:
            msg = "Plotly is required for Sankey diagram visualization. Install with: pip install plotly"
            raise ImportError(msg)

        if not graph.is_directed():
            msg = "Sankey diagram requires a directed graph"
            raise ValueError(msg)

        # Auto-detect source and target nodes if not provided
        if source_nodes is None:
            source_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        if target_nodes is None:
            target_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        # Create node labels and indices
        all_nodes = list(graph.nodes())
        node_indices = {node: i for i, node in enumerate(all_nodes)}

        # Create link data
        sources = []
        targets = []
        values = []
        link_labels = []

        for edge in graph.edges(data=True):
            sources.append(node_indices[edge[0]])
            targets.append(node_indices[edge[1]])
            value = edge[2].get(flow_attribute, 1)
            values.append(value)
            link_labels.append(f"{edge[0]} → {edge[1]}: {value}")

        # Node colors
        node_colors = []
        for node in all_nodes:
            if node in source_nodes:
                node_colors.append("rgba(0, 150, 0, 0.8)")  # Green for sources
            elif node in target_nodes:
                node_colors.append("rgba(150, 0, 0, 0.8)")  # Red for targets
            else:
                node_colors.append("rgba(100, 100, 100, 0.8)")  # Gray for intermediate

        # Create Sankey diagram
        fig = go.Figure(
            data=[
                go.Sankey(
                    node={
                        "pad": 15,
                        "thickness": 20,
                        "line": {"color": "black", "width": 0.5},
                        "label": [str(node) for node in all_nodes],
                        "color": node_colors,
                    },
                    link={
                        "source": sources,
                        "target": targets,
                        "value": values,
                        "label": link_labels,
                        "color": "rgba(100, 100, 100, 0.3)",
                    },
                )
            ]
        )

        fig.update_layout(title=title, font_size=10, width=1000, height=600)

        return {
            "plotly_figure": fig.to_dict(),
            "html": fig.to_html(include_plotlyjs="cdn"),
            "num_sources": len(source_nodes),
            "num_targets": len(target_nodes),
            "total_flow": sum(values),
        }

    @staticmethod
    def dendrogram_clustering(
        graph: nx.Graph | nx.DiGraph,
        method: str = "average",
        metric: str = "euclidean",
        figsize: tuple[int, int] = (12, 8),
        title: str = "Hierarchical Clustering Dendrogram",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Create dendrogram for hierarchical clustering visualization.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            The graph to cluster
        method : str
            Linkage method ('single', 'complete', 'average', 'ward')
        metric : str
            Distance metric

        Returns:
        --------
        Dict containing dendrogram visualization
        """
        if not HAS_MATPLOTLIB:
            msg = "Matplotlib and scipy are required for dendrogram visualization. Install with: pip install matplotlib scipy"
            raise ImportError(msg)

        # Create feature matrix from graph structure
        nodes = list(graph.nodes())
        n = len(nodes)

        # Use multiple graph features for clustering
        features = []

        for node in nodes:
            node_features = []

            # Degree
            node_features.append(graph.degree(node))

            # Clustering coefficient
            if not graph.is_directed():
                node_features.append(nx.clustering(graph, node))
            else:
                node_features.append(0)

            # Average neighbor degree
            neighbor_degrees = [graph.degree(n) for n in graph.neighbors(node)]
            avg_neighbor_degree = np.mean(neighbor_degrees) if neighbor_degrees else 0
            node_features.append(avg_neighbor_degree)

            # Betweenness centrality (for small graphs)
            if n < 100:
                betweenness = nx.betweenness_centrality(graph)
                node_features.append(betweenness[node])
            else:
                node_features.append(0)

            features.append(node_features)

        features = np.array(features)

        # Normalize features
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Perform hierarchical clustering
        linkage_matrix = linkage(features_scaled, method=method, metric=metric)

        # Create dendrogram
        plt.figure(figsize=figsize)

        dendrogram(linkage_matrix, labels=nodes, leaf_rotation=90, leaf_font_size=10)

        plt.title(title)
        plt.xlabel("Nodes")
        plt.ylabel("Distance")
        plt.tight_layout()

        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        # Extract cluster information
        from scipy.cluster.hierarchy import fcluster

        # Get clusters at different thresholds
        thresholds = [0.5, 1.0, 1.5, 2.0]
        cluster_results = {}

        for t in thresholds:
            clusters = fcluster(linkage_matrix, t, criterion="distance")
            cluster_dict = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_dict:
                    cluster_dict[cluster_id] = []
                cluster_dict[cluster_id].append(nodes[i])
            cluster_results[f"threshold_{t}"] = {
                "num_clusters": len(cluster_dict),
                "clusters": cluster_dict,
            }

        return {
            "dendrogram_png_base64": img_base64,
            "linkage_method": method,
            "distance_metric": metric,
            "num_nodes": n,
            "cluster_results": cluster_results,
            "features_used": [
                "degree",
                "clustering",
                "avg_neighbor_degree",
                "betweenness",
            ],
        }

    @staticmethod
    def create_dashboard(
        graph: nx.Graph | nx.DiGraph,
        visualizations: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Create a dashboard with multiple visualizations.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            The graph to visualize
        visualizations : list
            List of visualizations to include

        Returns:
        --------
        Dict containing dashboard HTML
        """
        if not HAS_PLOTLY:
            msg = "Plotly is required for dashboard visualization. Install with: pip install plotly"
            raise ImportError(msg)

        if visualizations is None:
            visualizations = ["adjacency", "degree_dist", "centrality", "components"]

        # Create subplots
        rows = (len(visualizations) + 1) // 2
        fig = make_subplots(
            rows=rows,
            cols=2,
            subplot_titles=visualizations,
            specs=[[{"type": "heatmap"}, {"type": "bar"}] for _ in range(rows)],
        )

        for i, viz_type in enumerate(visualizations):
            row = i // 2 + 1
            col = i % 2 + 1

            if viz_type == "adjacency":
                # Small adjacency heatmap
                nodes = list(graph.nodes())[:20]  # Limit for visibility
                subgraph = graph.subgraph(nodes)
                adj_matrix = nx.adjacency_matrix(subgraph).toarray()

                fig.add_trace(
                    go.Heatmap(z=adj_matrix, colorscale="Viridis"), row=row, col=col
                )

            elif viz_type == "degree_dist":
                # Degree distribution
                degrees = [d for n, d in graph.degree()]
                fig.add_trace(go.Histogram(x=degrees, nbinsx=20), row=row, col=col)

            elif viz_type == "centrality":
                # Top centrality nodes
                centrality = nx.degree_centrality(graph)
                top_nodes = sorted(
                    centrality.items(), key=lambda x: x[1], reverse=True
                )[:10]

                fig.add_trace(
                    go.Bar(
                        x=[str(n[0]) for n in top_nodes], y=[n[1] for n in top_nodes]
                    ),
                    row=row,
                    col=col,
                )

            elif viz_type == "components":
                # Component sizes
                if graph.is_directed():
                    components = list(nx.weakly_connected_components(graph))
                else:
                    components = list(nx.connected_components(graph))

                sizes = sorted([len(c) for c in components], reverse=True)[:10]

                fig.add_trace(
                    go.Bar(
                        x=list(range(1, len(sizes) + 1)),
                        y=sizes,
                        text=[f"Size: {s}" for s in sizes],
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(
            title="Network Analysis Dashboard", height=400 * rows, showlegend=False
        )

        return {
            "dashboard_plotly": fig.to_dict(),
            "dashboard_html": fig.to_html(include_plotlyjs="cdn"),
            "visualizations_included": visualizations,
        }
