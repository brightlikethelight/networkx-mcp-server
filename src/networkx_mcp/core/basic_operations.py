"""
Basic graph operations extracted from server.py.
These are compatibility functions for existing code.
"""

import base64
import csv
import io
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm

matplotlib.use("Agg")  # Use non-interactive backend


def create_graph(
    name: str, directed: bool = False, graphs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a graph - compatibility function."""
    if graphs is None:
        graphs = {}
    graphs[name] = nx.DiGraph() if directed else nx.Graph()
    return {
        "created": True,
        "graph_id": name,
        "metadata": {"attributes": {"directed": directed}},
    }


def add_nodes(
    graph_name: str,
    nodes: List[Union[str, int]],
    graphs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add nodes - compatibility function."""
    if graphs is None:
        graphs = {}
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]

    # Count only new nodes
    existing_nodes = set(graph.nodes())
    new_nodes = [node for node in nodes if node not in existing_nodes]

    graph.add_nodes_from(nodes)
    return {
        "success": True,
        "nodes_added": len(new_nodes),
        "total": graph.number_of_nodes(),
    }


def add_edges(
    graph_name: str,
    edges: List[List[Union[str, int]]],
    graphs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add edges - compatibility function."""
    if graphs is None:
        graphs = {}
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    edge_tuples = [(e[0], e[1]) for e in edges if len(e) >= 2]
    graph.add_edges_from(edge_tuples)
    return {
        "success": True,
        "edges_added": len(edge_tuples),
        "total": graph.number_of_edges(),
    }


def get_graph_info(
    graph_name: str, graphs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get graph info - compatibility function."""
    if graphs is None:
        graphs = {}
    if graph_name not in graphs:
        return {"success": False, "error": f"Graph '{graph_name}' not found"}
    graph = graphs[graph_name]
    return {
        "graph_id": graph_name,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "nodes": list(graph.nodes()),  # Return actual nodes list
        "edges": [[u, v] for u, v in graph.edges()],  # Return edges as list of lists
        "is_directed": graph.is_directed(),  # Use is_directed as expected by test
        "directed": graph.is_directed(),  # Keep for backward compatibility
        "metadata": {"attributes": {"directed": graph.is_directed()}},
    }


def shortest_path(
    graph_name: str,
    source: Union[str, int],
    target: Union[str, int],
    graphs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find shortest path - compatibility function."""
    if graphs is None:
        graphs = {}
    if graph_name not in graphs:
        return {"success": False, "error": f"Graph '{graph_name}' not found"}

    graph = graphs[graph_name]
    try:
        path = nx.shortest_path(graph, source, target)
        return {"success": True, "path": path, "length": len(path) - 1}
    except nx.NetworkXNoPath:
        return {
            "success": False,
            "error": f"No path found between {source} and {target}",
        }
    except nx.NodeNotFound as e:
        return {"success": False, "error": f"Node not found: {e}"}


def degree_centrality(
    graph_name: str, graphs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Calculate degree centrality - compatibility function."""
    if graphs is None:
        graphs = {}
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    centrality = nx.degree_centrality(graph)
    # Convert to serializable format and sort by centrality
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return {
        "centrality": dict(sorted_nodes[:10]),  # Top 10 nodes
        "most_central": sorted_nodes[0] if sorted_nodes else None,
    }


def betweenness_centrality(
    graph_name: str, graphs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Calculate betweenness centrality - compatibility function."""
    if graphs is None:
        graphs = {}
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    centrality = nx.betweenness_centrality(graph)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return {
        "centrality": dict(sorted_nodes[:10]),  # Top 10 nodes
        "most_central": sorted_nodes[0] if sorted_nodes else None,
    }


def connected_components(
    graph_name: str, graphs: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[int, List[int], List[List[Union[str, int]]]]]:
    """Find connected components - compatibility function."""
    if graphs is None:
        graphs = {}
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))

    # Convert sets to lists for JSON serialization
    components_list = [list(comp) for comp in components]
    components_list.sort(key=len, reverse=True)  # Largest first

    return {
        "num_components": len(components_list),
        "component_sizes": [len(comp) for comp in components_list],
        "largest_component": components_list[0] if components_list else [],
    }


def pagerank(
    graph_name: str, graphs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Calculate PageRank - compatibility function."""
    if graphs is None:
        graphs = {}
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    pr = nx.pagerank(graph)
    sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return {
        "pagerank": dict(sorted_nodes[:10]),  # Top 10 nodes
        "highest_rank": sorted_nodes[0] if sorted_nodes else None,
    }


def visualize_graph(
    graph_name: str, layout: str = "spring", graphs: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """Visualize graph and return as base64 image - compatibility function."""
    if graphs is None:
        graphs = {}
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]

    plt.figure(figsize=(10, 8))

    # Choose layout
    if layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.spring_layout(graph)

    # Draw the graph
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        arrows=True,
    )

    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    plt.close()

    # Convert to base64
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return {
        "image": f"data:image/png;base64,{image_base64}",
        "format": "png",
        "layout": layout,
    }


def import_csv(
    graph_name: str,
    csv_data: str,
    directed: bool = False,
    graphs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Union[str, int]]:
    """Import graph from CSV edge List[Any] - compatibility function."""
    if graphs is None:
        graphs = {}
    # Parse CSV data
    reader = csv.reader(io.StringIO(csv_data))
    edges = []

    # Skip header if it looks like one
    first_row = next(reader, None)
    if first_row and not (
        first_row[0].lower() in ["source", "from"]
        and first_row[1].lower() in ["target", "to"]
    ):
        # Not a header, add it as an edge
        if len(first_row) >= 2:
            edges.append((first_row[0].strip(), first_row[1].strip()))

    for row in reader:
        if len(row) >= 2:
            edges.append((row[0].strip(), row[1].strip()))

    # Create graph
    graph: Any = nx.DiGraph() if directed else nx.Graph()
    graph.add_edges_from(edges)
    graphs[graph_name] = graph

    return {
        "imported": graph_name,
        "type": "directed" if directed else "undirected",
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
    }


def export_json(
    graph_name: str, graphs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Export graph as JSON - compatibility function."""
    if graphs is None:
        graphs = {}
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]

    # Convert to node-link format
    data = nx.node_link_data(graph)

    return {
        "graph_data": data,
        "format": "node-link",
        "nodes": len(data["nodes"]),
        "edges": len(data["links"]),
    }


def community_detection(
    graph_name: str, graphs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Detect communities in the graph - compatibility function."""
    if graphs is None:
        graphs = {}
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]

    # Use Louvain method for community detection
    communities = nx_comm.louvain_communities(graph)

    # Convert to List[Any] format
    communities_list = [list(comm) for comm in communities]
    communities_list.sort(key=len, reverse=True)  # Largest first

    # Create node to community mapping
    node_community = {}
    for i, comm in enumerate(communities_list):
        for node in comm:
            node_community[node] = i

    return {
        "communities": communities_list,
        "num_communities": len(communities_list),
        "method": "louvain",
        "community_sizes": [len(comm) for comm in communities_list],
        "largest_community": communities_list[0] if communities_list else [],
        "node_community_map": dict(
            list(node_community.items())[:20]
        ),  # First 20 nodes
    }
