"""PyVis-based interactive graph visualization with physics simulation."""

import json
import logging
import os
import tempfile
from typing import Any

import networkx as nx

try:
    from pyvis.network import Network

    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False
    Network = None

logger = logging.getLogger(__name__)


class PyvisVisualizer:
    """Create interactive graph visualizations using PyVis with physics simulation."""

    def __init__(self) -> None:
        """Initialize PyvisVisualizer."""
        if not HAS_PYVIS:
            raise ImportError(
                "pyvis is required for PyvisVisualizer. Install with: pip install pyvis"
            )

    # Default physics configurations
    PHYSICS_CONFIGS = {
        "barnes_hut": {
            "gravitationalConstant": -2000,
            "centralGravity": 0.3,
            "springLength": 100,
            "springConstant": 0.05,
            "damping": 0.09,
            "avoidOverlap": 0.5,
        },
        "force_atlas": {
            "gravitationalConstant": -50,
            "centralGravity": 0.01,
            "springLength": 100,
            "springConstant": 0.08,
            "damping": 0.4,
        },
        "repulsion": {
            "centralGravity": 0.2,
            "springLength": 200,
            "springConstant": 0.05,
            "nodeDistance": 100,
            "damping": 0.09,
        },
    }

    @staticmethod
    def create_interactive_network(graph: nx.Graph | nx.DiGraph, height: str = "750px", width: str = "100%", bgcolor: str = "#ffffff", font_color: str = "#000000", physics: bool | str | dict[str, Any] = "barnes_hut", hierarchical: bool = False, **kwargs) -> dict[str, Any]:
        """
        Create an interactive network visualization with physics simulation.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            The graph to visualize
        height : str
            Height of the visualization
        width : str
            Width of the visualization
        bgcolor : str
            Background color
        font_color : str
            Default font color
        physics : bool, str, or dict[str, Any]
            Physics configuration ('barnes_hut', 'force_atlas', 'repulsion', or custom)
        hierarchical : bool
            Use hierarchical layout

        Returns:
        --------
        Dict containing the visualization HTML and metadata
        """
        # Initialize PyVis network
        net = Network(
            height=height,
            width=width,
            bgcolor=bgcolor,
            font_color=font_color,
            directed=graph.is_directed(),
            notebook=False,
        )

        # Configure physics
        if isinstance(physics, str) and physics in PyvisVisualizer.PHYSICS_CONFIGS:
            physics_config = PyvisVisualizer.PHYSICS_CONFIGS[physics]
        elif isinstance(physics, dict[str, Any]):
            physics_config = physics
        else:
            physics_config = physics

        if physics_config:
            net.barnes_hut(
                gravity=physics_config.get("gravitationalConstant", -2000),
                central_gravity=physics_config.get("centralGravity", 0.3),
                spring_length=physics_config.get("springLength", 100),
                spring_strength=physics_config.get("springConstant", 0.05),
                damping=physics_config.get("damping", 0.09),
            )

        # Configure hierarchical layout if requested
        if hierarchical:
            net.set_options(
                """
            var options = {
                "layout": {
                    "hierarchical": {
                        "enabled": true,
                        "levelSeparation": 150,
                        "nodeSpacing": 100,
                        "treeSpacing": 200,
                        "direction": "UD",
                        "sortMethod": "directed"
                    }
                }
            }
            """
            )

        # Add nodes with attributes
        PyvisVisualizer._add_nodes(net, graph, **kwargs)

        # Add edges with attributes
        PyvisVisualizer._add_edges(net, graph, **kwargs)

        # Set additional options
        if kwargs.get("show_buttons", True):
            net.show_buttons(filter_=["physics", "layout", "interaction"])

        # Generate HTML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            net.save_graph(f.name)
            temp_filename = f.name

        with open(temp_filename) as f:
            html_content = f.read()

        os.unlink(temp_filename)

        return {
            "html": html_content,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "physics_type": physics if isinstance(physics, str) else "custom",
            "hierarchical": hierarchical,
        }

    @staticmethod
    def create_community_visualization(graph: nx.Graph | nx.DiGraph, communities: dict[int, list[Any]], height: str = "750px", width: str = "100%", physics: str = "force_atlas") -> dict[str, Any]:
        """
        Visualize graph with community coloring.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            The graph to visualize
        communities : dict[str, Any]
            Community assignments {community_id: [node_list]}

        Returns:
        --------
        Dict containing the visualization
        """
        # Create network
        net = Network(
            height=height, width=width, directed=graph.is_directed(), notebook=False
        )

        # Configure physics for community visualization
        if physics == "force_atlas":
            net.force_atlas_2based(
                gravity=-50,
                central_gravity=0.01,
                spring_length=100,
                spring_strength=0.08,
                damping=0.4,
            )

        # Create color palette for communities
        colors = [
            "#e41a1c",
            "#377eb8",
            "#4daf4a",
            "#984ea3",
            "#ff7f00",
            "#ffff33",
            "#a65628",
            "#f781bf",
            "#999999",
            "#66c2a5",
        ]

        # Create node to community mapping
        node_to_community = {}
        for comm_id, nodes in communities.items():
            for node in nodes:
                node_to_community[node] = comm_id

        # Add nodes with community colors
        for node in graph.nodes():
            comm_id = node_to_community.get(node, -1)
            color = colors[comm_id % len(colors)] if comm_id >= 0 else "#cccccc"

            net.add_node(
                str(node),
                label=str(node),
                color=color,
                title=f"Node: {node}<br>Community: {comm_id}<br>Degree: {graph.degree(node)}",
                size=20 + graph.degree(node) * 2,
            )

        # Add edges
        for edge in graph.edges(data=True):
            net.add_edge(str(edge[0]), str(edge[1]), weight=edge[2].get("weight", 1.0))

        # Enable community grouping
        net.set_options(
            """
        var options = {
            "physics": {
                "enabled": true,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "springLength": 100,
                    "springConstant": 0.08
                }
            },
            "groups": {
                "useDefaultGroups": true
            }
        }
        """
        )

        # Generate HTML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            net.save_graph(f.name)
            temp_filename = f.name

        with open(temp_filename) as f:
            html_content = f.read()

        os.unlink(temp_filename)

        return {
            "html": html_content,
            "num_communities": len(communities),
            "community_sizes": {k: len(v) for k, v in communities.items()},
        }

    @staticmethod
    def create_hierarchical_tree(tree: nx.DiGraph, root: Any | None = None, height: str = "750px", width: str = "100%") -> dict[str, Any]:
        """Create hierarchical visualization for tree structures."""
        if not nx.is_tree(tree):
            msg = "Graph must be a tree"
            raise ValueError(msg)

        # Find root if not specified
        if root is None:
            # Find node with no predecessors
            roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
            if not roots:
                msg = "No root node found"
                raise ValueError(msg)
            root = roots[0]

        # Create network with hierarchical layout
        net = Network(
            height=height, width=width, directed=True, notebook=False, layout=True
        )

        # Calculate levels using BFS
        levels = {root: 0}
        queue = [root]

        while queue:
            current = queue.pop(0)
            for child in tree.successors(current):
                if child not in levels:
                    levels[child] = levels[current] + 1
                    queue.append(child)

        # Add nodes with level-based positioning
        for node in tree.nodes():
            level = levels.get(node, 0)
            net.add_node(
                str(node),
                label=str(node),
                level=level,
                title=f"Node: {node}<br>Level: {level}<br>Children: {tree.out_degree(node)}",
                size=25,
            )

        # Add edges
        for edge in tree.edges():
            net.add_edge(str(edge[0]), str(edge[1]))

        # Set hierarchical options
        net.set_options(
            """
        var options = {
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "levelSeparation": 150,
                    "nodeSpacing": 100,
                    "treeSpacing": 200,
                    "blockShifting": true,
                    "edgeMinimization": true,
                    "parentCentralization": true,
                    "direction": "UD",
                    "sortMethod": "directed"
                }
            },
            "physics": {
                "enabled": false
            }
        }
        """
        )

        # Generate HTML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            net.save_graph(f.name)
            temp_filename = f.name

        with open(temp_filename) as f:
            html_content = f.read()

        os.unlink(temp_filename)

        return {
            "html": html_content,
            "root": root,
            "num_levels": max(levels.values()) + 1,
            "tree_height": max(levels.values()),
        }

    @staticmethod
    def _add_nodes(net: Network, graph: nx.Graph, node_size_attr: str | None = None, node_color_attr: str | None = None, node_shape: str = "dot", **kwargs):
        """Add nodes to PyVis network with attributes."""
        # Default sizes and colors
        default_size = 25
        default_color = "#97c2fc"

        for node in graph.nodes(data=True):
            node_id = str(node[0])
            node_data = node[1]

            # Determine size
            if node_size_attr and node_size_attr in node_data:
                size = node_data[node_size_attr]
            else:
                size = default_size + graph.degree(node[0]) * 2

            # Determine color
            if node_color_attr and node_color_attr in node_data:
                color = node_data[node_color_attr]
            else:
                color = default_color

            # Create hover title
            title_parts = [f"Node: {node[0]}"]
            title_parts.append(f"Degree: {graph.degree(node[0])}")

            # Add additional attributes
            for key, value in node_data.items():
                if key not in [node_size_attr, node_color_attr]:
                    title_parts.append(f"{key}: {value}")

            title = "<br>".join(title_parts)

            # Add node
            net.add_node(
                node_id,
                label=node_id,
                title=title,
                size=size,
                color=color,
                shape=node_shape,
            )

    @staticmethod
    def _add_edges(net: Network, graph: nx.Graph, edge_width_attr: str | None = None, edge_color_attr: str | None = None, **kwargs):
        """Add edges to PyVis network with attributes."""
        default_width = 1
        default_color = "#848484"

        for edge in graph.edges(data=True):
            source = str(edge[0])
            target = str(edge[1])
            edge_data = edge[2]

            # Determine width
            if edge_width_attr and edge_width_attr in edge_data:
                width = edge_data[edge_width_attr]
            else:
                width = edge_data.get("weight", default_width)

            # Determine color
            if edge_color_attr and edge_color_attr in edge_data:
                color = edge_data[edge_color_attr]
            else:
                color = default_color

            # Create hover title
            title_parts = [f"Edge: {edge[0]} - {edge[1]}"]
            for key, value in edge_data.items():
                title_parts.append(f"{key}: {value}")

            title = "<br>".join(title_parts)

            # Add edge
            net.add_edge(source, target, width=width, color=color, title=title)

    @staticmethod
    def set_custom_options(net: Network, options: dict[str, Any]) -> Network:
        """Set custom vis.js options."""
        options_str = json.dumps(options)
        net.set_options(f"var options = {options_str}")
        return net
