"""Tool handler functions for NetworkX MCP Server.

Each handler takes an `args` dict and returns a result dict.
Exceptions propagate to the dispatcher in server.py for uniform error handling.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import networkx as nx

from .graph_cache import graphs

# ── Underlying implementations ────────────────────────────────────────
from .core.algorithms import GraphAlgorithms
from .core.basic_operations import (
    betweenness_centrality as _betweenness_centrality,
    community_detection as _community_detection,
    connected_components as _connected_components,
    degree_centrality as _degree_centrality,
    export_json as _export_json,
    import_csv as _import_csv,
    pagerank as _pagerank,
    visualize_graph as _visualize_graph,
)
from .errors import ErrorCodes, MCPError, validate_graph_id

# ── Bulk operation limits (DoS protection) ────────────────────────────
MAX_NODES_PER_CALL = 100_000
MAX_EDGES_PER_CALL = 500_000
MAX_VISUALIZATION_NODES = 10_000


# ═══════════════════════════════════════════════════════════════════════
# Graph Management
# ═══════════════════════════════════════════════════════════════════════


def handle_create_graph(args: Dict[str, Any]) -> Dict[str, Any]:
    name = args["name"]
    directed = args.get("directed", False)
    graphs[name] = nx.DiGraph() if directed else nx.Graph()
    return {"created": name, "type": "directed" if directed else "undirected"}


def handle_add_nodes(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    nodes = args["nodes"]
    if len(nodes) > MAX_NODES_PER_CALL:
        raise ValueError(
            f"Too many nodes ({len(nodes)}). Maximum is {MAX_NODES_PER_CALL} per call."
        )
    graph = graphs[graph_name]
    graph.add_nodes_from(nodes)
    return {"added": len(nodes), "total": graph.number_of_nodes()}


def handle_add_edges(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    raw_edges = args["edges"]
    if len(raw_edges) > MAX_EDGES_PER_CALL:
        raise ValueError(
            f"Too many edges ({len(raw_edges)}). Maximum is {MAX_EDGES_PER_CALL} per call."
        )
    graph = graphs[graph_name]
    edges = [tuple(e) for e in raw_edges]
    graph.add_edges_from(edges)
    return {"added": len(edges), "total": graph.number_of_edges()}


def handle_get_info(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "directed": graph.is_directed(),
    }


def handle_list_graphs(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_list = []
    for name in graphs:
        g = graphs[name]
        graph_list.append(
            {
                "name": name,
                "nodes": g.number_of_nodes(),
                "edges": g.number_of_edges(),
                "directed": g.is_directed(),
            }
        )
    return {"graphs": graph_list, "total": len(graph_list)}


def handle_delete_graph(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    del graphs[graph_name]
    return {"deleted": graph_name}


def handle_remove_nodes(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    nodes = args["nodes"]
    graph = graphs[graph_name]
    graph.remove_nodes_from(nodes)
    return {"removed": len(nodes), "total_nodes": graph.number_of_nodes()}


def handle_remove_edges(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    raw_edges = args["edges"]
    graph = graphs[graph_name]
    edges = [tuple(e) for e in raw_edges]
    graph.remove_edges_from(edges)
    return {"removed": len(edges), "total_edges": graph.number_of_edges()}


def handle_shortest_path(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    path = nx.shortest_path(graph, args["source"], args["target"])
    return {"path": path, "length": len(path) - 1}


def handle_get_neighbors(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    node = args["node"]
    graph = graphs[graph_name]
    if node not in graph:
        raise ValueError(f"Node '{node}' not found in graph '{graph_name}'")
    neighbors = list(graph.neighbors(node))
    return {"node": node, "neighbors": neighbors, "count": len(neighbors)}


def handle_set_node_attributes(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    attributes = args["attributes"]  # {node: {attr: value}}
    for node, attrs in attributes.items():
        if node not in graph:
            raise ValueError(f"Node '{node}' not found")
        for key, val in attrs.items():
            graph.nodes[node][key] = val
    return {"updated": len(attributes)}


def handle_get_node_attributes(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    node = args["node"]
    graph = graphs[graph_name]
    if node not in graph:
        raise ValueError(f"Node '{node}' not found in graph '{graph_name}'")
    return {"node": node, "attributes": dict(graph.nodes[node])}


def handle_set_edge_attributes(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    attributes = args[
        "attributes"
    ]  # [{"source": s, "target": t, "attr": k, "value": v}]
    count = 0
    for entry in attributes:
        s, t = entry["source"], entry["target"]
        if not graph.has_edge(s, t):
            raise ValueError(f"Edge '{s}'->'{t}' not found")
        graph[s][t][entry["attr"]] = entry["value"]
        count += 1
    return {"updated": count}


def handle_get_edge_attributes(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    source, target = args["source"], args["target"]
    graph = graphs[graph_name]
    if not graph.has_edge(source, target):
        raise ValueError(f"Edge '{source}'->'{target}' not found")
    return {
        "source": source,
        "target": target,
        "attributes": dict(graph[source][target]),
    }


# ═══════════════════════════════════════════════════════════════════════
# Algorithms
# ═══════════════════════════════════════════════════════════════════════


def handle_degree_centrality(args: Dict[str, Any]) -> Dict[str, Any]:
    return _degree_centrality(args["graph"], graphs)


def handle_betweenness_centrality(args: Dict[str, Any]) -> Dict[str, Any]:
    return _betweenness_centrality(args["graph"], graphs)


def handle_connected_components(args: Dict[str, Any]) -> Dict[str, Any]:
    return _connected_components(args["graph"], graphs)


def handle_pagerank(args: Dict[str, Any]) -> Dict[str, Any]:
    return _pagerank(args["graph"], graphs)


def handle_community_detection(args: Dict[str, Any]) -> Dict[str, Any]:
    return _community_detection(args["graph"], graphs)


# ═══════════════════════════════════════════════════════════════════════
# Advanced Algorithms (via GraphAlgorithms)
# ═══════════════════════════════════════════════════════════════════════


def handle_clustering_coefficients(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    return GraphAlgorithms.clustering_coefficients(graphs[graph_name])


def handle_graph_statistics(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    result = GraphAlgorithms.graph_statistics(graphs[graph_name])
    # Convert numpy scalars to Python types for JSON serialization
    for key in ("degree_stats", "in_degree_stats", "out_degree_stats"):
        if key in result:
            result[key] = {k: float(v) for k, v in result[key].items()}
    return result


def handle_minimum_spanning_tree(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    weight = args.get("weight", "weight")
    algorithm = args.get("algorithm", "kruskal")
    return GraphAlgorithms.minimum_spanning_tree(graphs[graph_name], weight, algorithm)


def handle_cycles_detection(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    return GraphAlgorithms.cycles_detection(graphs[graph_name])


def handle_graph_coloring(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    strategy = args.get("strategy", "largest_first")
    return GraphAlgorithms.graph_coloring(graphs[graph_name], strategy)


def handle_centrality_measures(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    measures = args.get("measures")
    return GraphAlgorithms.centrality_measures(graphs[graph_name], measures)


def handle_matching(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    max_cardinality = args.get("max_cardinality", True)
    return GraphAlgorithms.matching(graphs[graph_name], max_cardinality)


def handle_maximum_flow(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    source = args["source"]
    sink = args["sink"]
    capacity = args.get("capacity", "capacity")
    return GraphAlgorithms.maximum_flow(graphs[graph_name], source, sink, capacity)


def handle_topological_sort(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    if not graph.is_directed():
        raise ValueError("Topological sort requires a directed graph")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph contains cycles; topological sort requires a DAG")
    order = list(nx.topological_sort(graph))
    return {"graph": graph_name, "order": order, "count": len(order)}


def handle_subgraph(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    nodes = args["nodes"]
    new_graph_name = args["new_graph"]
    validate_graph_id(new_graph_name)
    graph = graphs[graph_name]
    missing = [n for n in nodes if n not in graph]
    if missing:
        raise ValueError(f"Nodes not found in graph: {missing}")
    sub = graph.subgraph(nodes).copy()
    graphs[new_graph_name] = sub
    return {
        "source": graph_name,
        "new_graph": new_graph_name,
        "nodes": sub.number_of_nodes(),
        "edges": sub.number_of_edges(),
    }


def handle_merge_graphs(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_a_name = args["graph_a"]
    graph_b_name = args["graph_b"]
    new_graph_name = args["new_graph"]
    validate_graph_id(graph_a_name)
    validate_graph_id(graph_b_name)
    validate_graph_id(new_graph_name)
    if graph_a_name not in graphs:
        raise ValueError(f"Graph '{graph_a_name}' not found")
    if graph_b_name not in graphs:
        raise ValueError(f"Graph '{graph_b_name}' not found")
    ga = graphs[graph_a_name]
    gb = graphs[graph_b_name]
    if type(ga) is not type(gb):
        raise ValueError(
            f"Cannot merge different graph types: {type(ga).__name__} and {type(gb).__name__}"
        )
    merged = nx.compose(ga, gb)
    graphs[new_graph_name] = merged
    return {
        "new_graph": new_graph_name,
        "nodes": merged.number_of_nodes(),
        "edges": merged.number_of_edges(),
        "source_graphs": [graph_a_name, graph_b_name],
    }


# ═══════════════════════════════════════════════════════════════════════
# I/O & Visualization
# ═══════════════════════════════════════════════════════════════════════


def handle_visualize_graph(args: Dict[str, Any]) -> Dict[str, Any]:
    graph_name = args["graph"]
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    if graph.number_of_nodes() > MAX_VISUALIZATION_NODES:
        raise ValueError(
            f"Graph too large for visualization ({graph.number_of_nodes()} nodes). "
            f"Maximum is {MAX_VISUALIZATION_NODES}."
        )
    layout = args.get("layout", "spring")
    viz_result = _visualize_graph(graph_name, layout, graphs)
    # Key rename: basic_operations returns 'image', MCP API exposes 'visualization'
    return {
        "visualization": viz_result["image"],
        "format": viz_result["format"],
        "layout": viz_result["layout"],
    }


def handle_import_csv(args: Dict[str, Any]) -> Dict[str, Any]:
    return _import_csv(
        args["graph"], args["csv_data"], args.get("directed", False), graphs
    )


def handle_export_json(args: Dict[str, Any]) -> Dict[str, Any]:
    return _export_json(args["graph"], graphs)


# ═══════════════════════════════════════════════════════════════════════
# Academic / Citation
# ═══════════════════════════════════════════════════════════════════════


def handle_build_citation_network(args: Dict[str, Any]) -> Dict[str, Any]:
    from .academic import build_citation_network as _build_citation_network

    return _build_citation_network(
        args["graph"], args["seed_dois"], args.get("max_depth", 2), graphs
    )


def handle_analyze_author_impact(args: Dict[str, Any]) -> Dict[str, Any]:
    from .academic import analyze_author_impact as _analyze_author_impact

    return _analyze_author_impact(args["graph"], args["author_name"], graphs)


def handle_find_collaboration_patterns(args: Dict[str, Any]) -> Dict[str, Any]:
    from .academic import find_collaboration_patterns as _find_collaboration_patterns

    return _find_collaboration_patterns(args["graph"], graphs)


def handle_detect_research_trends(args: Dict[str, Any]) -> Dict[str, Any]:
    from .academic import detect_research_trends as _detect_research_trends

    return _detect_research_trends(args["graph"], args.get("time_window", 5), graphs)


def handle_export_bibtex(args: Dict[str, Any]) -> Dict[str, Any]:
    from .academic import export_bibtex as _export_bibtex

    return _export_bibtex(args["graph"], graphs)


def handle_recommend_papers(args: Dict[str, Any]) -> Dict[str, Any]:
    from .academic import recommend_papers as _recommend_papers

    # Backward-compatible parameter handling
    seed = args.get("seed_doi") or args.get("seed_paper")
    max_recs = args.get("max_recommendations") or args.get("top_n", 10)
    if not seed:
        raise ValueError("Missing required parameter: seed_doi or seed_paper")
    return _recommend_papers(args["graph"], seed, max_recs, graphs)


def handle_resolve_doi(args: Dict[str, Any]) -> Dict[str, Any]:
    from .academic import resolve_doi as _resolve_doi

    result, error = _resolve_doi(args["doi"])
    if result is None:
        error_msg = error or "Unknown error"
        raise ValueError(f"Could not resolve DOI: {args['doi']} - {error_msg}")
    return result


# ═══════════════════════════════════════════════════════════════════════
# CI/CD Control (async, lazy imports)
# ═══════════════════════════════════════════════════════════════════════


async def handle_trigger_workflow(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from .tools import mcp_trigger_workflow
    except ImportError:
        raise MCPError(ErrorCodes.METHOD_NOT_FOUND, "CI/CD tools not available")
    return await mcp_trigger_workflow(
        args["workflow"], args.get("branch", "main"), args.get("inputs")
    )


async def handle_get_workflow_status(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from .tools import mcp_get_workflow_status
    except ImportError:
        raise MCPError(ErrorCodes.METHOD_NOT_FOUND, "CI/CD tools not available")
    return await mcp_get_workflow_status(args.get("run_id"))


async def handle_cancel_workflow(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from .tools import mcp_cancel_workflow
    except ImportError:
        raise MCPError(ErrorCodes.METHOD_NOT_FOUND, "CI/CD tools not available")
    return await mcp_cancel_workflow(args["run_id"])


async def handle_rerun_failed_jobs(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from .tools import mcp_rerun_failed_jobs
    except ImportError:
        raise MCPError(ErrorCodes.METHOD_NOT_FOUND, "CI/CD tools not available")
    return await mcp_rerun_failed_jobs(args["run_id"])


async def handle_get_dora_metrics(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from .tools import mcp_get_dora_metrics
    except ImportError:
        raise MCPError(ErrorCodes.METHOD_NOT_FOUND, "CI/CD tools not available")
    return await mcp_get_dora_metrics()


async def handle_analyze_workflow_failures(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from .tools import mcp_analyze_failures
    except ImportError:
        raise MCPError(ErrorCodes.METHOD_NOT_FOUND, "CI/CD tools not available")
    return await mcp_analyze_failures(args["run_id"])


# ═══════════════════════════════════════════════════════════════════════
# Monitoring
# ═══════════════════════════════════════════════════════════════════════


def make_health_handler(monitor: Any) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a health_status handler bound to a specific monitor instance."""

    def handle_health_status(args: Dict[str, Any]) -> Dict[str, Any]:
        if monitor:
            return monitor.get_health_status()
        return {"status": "monitoring_disabled"}

    return handle_health_status
