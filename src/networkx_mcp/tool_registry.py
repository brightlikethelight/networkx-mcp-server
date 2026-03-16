"""Tool registry for NetworkX MCP Server.

Maps tool names to their schemas, handlers, and metadata. Single source of truth
for tool definitions — adding a new tool means one entry here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class ToolDef:
    """Definition of an MCP tool."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any]
    is_write: bool = False
    graph_param: Optional[str] = None  # param name holding graph ID, None if no graph


class ToolRegistry:
    """Registry of available MCP tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDef] = {}

    def register(self, tool_def: ToolDef) -> None:
        self._tools[tool_def.name] = tool_def

    def get(self, name: str) -> Optional[ToolDef]:
        return self._tools.get(name)

    def list_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas in MCP format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def write_tool_names(self) -> Set[str]:
        return {t.name for t in self._tools.values() if t.is_write}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def build_registry(
    monitoring_enabled: bool = False,
    monitor: Any = None,
) -> ToolRegistry:
    """Build and return a fully populated tool registry.

    Args:
        monitoring_enabled: Whether to include health_status tool.
        monitor: HealthMonitor instance (required if monitoring_enabled).
    """
    from . import handlers as h

    registry = ToolRegistry()

    # ── Graph Management ──────────────────────────────────────────────
    registry.register(
        ToolDef(
            name="create_graph",
            description="Create a new graph",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "directed": {"type": "boolean", "default": False},
                },
                "required": ["name"],
            },
            handler=h.handle_create_graph,
            is_write=True,
            graph_param="name",
        )
    )
    registry.register(
        ToolDef(
            name="add_nodes",
            description="Add nodes to a graph",
            input_schema={
                "type": "object",
                "properties": {
                    "graph": {"type": "string"},
                    "nodes": {
                        "type": "array",
                        "items": {"type": ["string", "number"]},
                    },
                },
                "required": ["graph", "nodes"],
            },
            handler=h.handle_add_nodes,
            is_write=True,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="add_edges",
            description="Add edges to a graph",
            input_schema={
                "type": "object",
                "properties": {
                    "graph": {"type": "string"},
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": ["string", "number"]},
                        },
                    },
                },
                "required": ["graph", "edges"],
            },
            handler=h.handle_add_edges,
            is_write=True,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="get_info",
            description="Get graph information",
            input_schema={
                "type": "object",
                "properties": {"graph": {"type": "string"}},
                "required": ["graph"],
            },
            handler=h.handle_get_info,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="delete_graph",
            description="Delete a graph from storage",
            input_schema={
                "type": "object",
                "properties": {"graph": {"type": "string"}},
                "required": ["graph"],
            },
            handler=h.handle_delete_graph,
            is_write=True,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="shortest_path",
            description="Find shortest path between nodes",
            input_schema={
                "type": "object",
                "properties": {
                    "graph": {"type": "string"},
                    "source": {"type": ["string", "number"]},
                    "target": {"type": ["string", "number"]},
                },
                "required": ["graph", "source", "target"],
            },
            handler=h.handle_shortest_path,
            graph_param="graph",
        )
    )

    # ── Algorithms ────────────────────────────────────────────────────
    for algo_name, algo_handler in [
        ("degree_centrality", h.handle_degree_centrality),
        ("betweenness_centrality", h.handle_betweenness_centrality),
        ("connected_components", h.handle_connected_components),
        ("pagerank", h.handle_pagerank),
        ("community_detection", h.handle_community_detection),
    ]:
        desc_map = {
            "degree_centrality": "Calculate degree centrality for all nodes",
            "betweenness_centrality": "Calculate betweenness centrality for all nodes",
            "connected_components": "Find connected components in the graph",
            "pagerank": "Calculate PageRank for all nodes",
            "community_detection": "Detect communities in the graph using Louvain method",
        }
        registry.register(
            ToolDef(
                name=algo_name,
                description=desc_map[algo_name],
                input_schema={
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"],
                },
                handler=algo_handler,
                graph_param="graph",
            )
        )

    # ── Advanced Algorithms ────────────────────────────────────────────
    registry.register(
        ToolDef(
            name="clustering_coefficients",
            description="Calculate clustering coefficients for all nodes",
            input_schema={
                "type": "object",
                "properties": {"graph": {"type": "string"}},
                "required": ["graph"],
            },
            handler=h.handle_clustering_coefficients,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="graph_statistics",
            description="Calculate comprehensive graph statistics (density, diameter, degree distribution)",
            input_schema={
                "type": "object",
                "properties": {"graph": {"type": "string"}},
                "required": ["graph"],
            },
            handler=h.handle_graph_statistics,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="minimum_spanning_tree",
            description="Find minimum spanning tree of an undirected graph",
            input_schema={
                "type": "object",
                "properties": {
                    "graph": {"type": "string"},
                    "weight": {"type": "string", "default": "weight"},
                    "algorithm": {
                        "type": "string",
                        "enum": ["kruskal", "prim"],
                        "default": "kruskal",
                    },
                },
                "required": ["graph"],
            },
            handler=h.handle_minimum_spanning_tree,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="cycles_detection",
            description="Detect cycles in a graph (cycle basis for undirected, DAG check for directed)",
            input_schema={
                "type": "object",
                "properties": {"graph": {"type": "string"}},
                "required": ["graph"],
            },
            handler=h.handle_cycles_detection,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="graph_coloring",
            description="Color graph vertices using greedy algorithm",
            input_schema={
                "type": "object",
                "properties": {
                    "graph": {"type": "string"},
                    "strategy": {
                        "type": "string",
                        "default": "largest_first",
                    },
                },
                "required": ["graph"],
            },
            handler=h.handle_graph_coloring,
            graph_param="graph",
        )
    )

    # ── I/O & Visualization ───────────────────────────────────────────
    registry.register(
        ToolDef(
            name="visualize_graph",
            description="Create a visualization of the graph",
            input_schema={
                "type": "object",
                "properties": {
                    "graph": {"type": "string"},
                    "layout": {
                        "type": "string",
                        "enum": ["spring", "circular", "kamada_kawai"],
                        "default": "spring",
                    },
                },
                "required": ["graph"],
            },
            handler=h.handle_visualize_graph,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="import_csv",
            description="Import graph from CSV edge list (format: source,target per line)",
            input_schema={
                "type": "object",
                "properties": {
                    "graph": {"type": "string"},
                    "csv_data": {"type": "string"},
                    "directed": {"type": "boolean", "default": False},
                },
                "required": ["graph", "csv_data"],
            },
            handler=h.handle_import_csv,
            is_write=True,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="export_json",
            description="Export graph as JSON in node-link format",
            input_schema={
                "type": "object",
                "properties": {"graph": {"type": "string"}},
                "required": ["graph"],
            },
            handler=h.handle_export_json,
            graph_param="graph",
        )
    )

    # ── Academic / Citation ───────────────────────────────────────────
    registry.register(
        ToolDef(
            name="build_citation_network",
            description="Build citation network from DOIs using CrossRef API",
            input_schema={
                "type": "object",
                "properties": {
                    "graph": {"type": "string"},
                    "seed_dois": {"type": "array", "items": {"type": "string"}},
                    "max_depth": {"type": "integer", "default": 2},
                },
                "required": ["graph", "seed_dois"],
            },
            handler=h.handle_build_citation_network,
            is_write=True,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="analyze_author_impact",
            description="Analyze author impact metrics including h-index",
            input_schema={
                "type": "object",
                "properties": {
                    "graph": {"type": "string"},
                    "author_name": {"type": "string"},
                },
                "required": ["graph", "author_name"],
            },
            handler=h.handle_analyze_author_impact,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="find_collaboration_patterns",
            description="Find collaboration patterns in citation network",
            input_schema={
                "type": "object",
                "properties": {"graph": {"type": "string"}},
                "required": ["graph"],
            },
            handler=h.handle_find_collaboration_patterns,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="detect_research_trends",
            description="Detect research trends over time",
            input_schema={
                "type": "object",
                "properties": {
                    "graph": {"type": "string"},
                    "time_window": {"type": "integer", "default": 5},
                },
                "required": ["graph"],
            },
            handler=h.handle_detect_research_trends,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="export_bibtex",
            description="Export citation network as BibTeX format",
            input_schema={
                "type": "object",
                "properties": {"graph": {"type": "string"}},
                "required": ["graph"],
            },
            handler=h.handle_export_bibtex,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="recommend_papers",
            description="Recommend papers based on citation network analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "graph": {"type": "string"},
                    "seed_doi": {"type": "string"},
                    "max_recommendations": {"type": "integer", "default": 10},
                },
                "required": ["graph", "seed_doi"],
            },
            handler=h.handle_recommend_papers,
            graph_param="graph",
        )
    )
    registry.register(
        ToolDef(
            name="resolve_doi",
            description="Resolve DOI to publication metadata using CrossRef API",
            input_schema={
                "type": "object",
                "properties": {"doi": {"type": "string"}},
                "required": ["doi"],
            },
            handler=h.handle_resolve_doi,
        )
    )

    # ── CI/CD Control ─────────────────────────────────────────────────
    try:
        import importlib.util

        if importlib.util.find_spec("networkx_mcp.tools") is not None:
            registry.register(
                ToolDef(
                    name="trigger_workflow",
                    description="Trigger a GitHub Actions workflow",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "workflow": {
                                "type": "string",
                                "description": "Workflow file name",
                            },
                            "branch": {"type": "string", "default": "main"},
                            "inputs": {
                                "type": "string",
                                "description": "JSON string of inputs",
                            },
                        },
                        "required": ["workflow"],
                    },
                    handler=h.handle_trigger_workflow,
                )
            )
            registry.register(
                ToolDef(
                    name="get_workflow_status",
                    description="Get CI/CD workflow status",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "run_id": {
                                "type": "string",
                                "description": "Optional run ID",
                            },
                        },
                    },
                    handler=h.handle_get_workflow_status,
                )
            )
            registry.register(
                ToolDef(
                    name="cancel_workflow",
                    description="Cancel a running workflow",
                    input_schema={
                        "type": "object",
                        "properties": {"run_id": {"type": "string"}},
                        "required": ["run_id"],
                    },
                    handler=h.handle_cancel_workflow,
                )
            )
            registry.register(
                ToolDef(
                    name="rerun_failed_jobs",
                    description="Rerun failed jobs in a workflow",
                    input_schema={
                        "type": "object",
                        "properties": {"run_id": {"type": "string"}},
                        "required": ["run_id"],
                    },
                    handler=h.handle_rerun_failed_jobs,
                )
            )
            registry.register(
                ToolDef(
                    name="get_dora_metrics",
                    description="Get DORA metrics for CI/CD performance",
                    input_schema={
                        "type": "object",
                        "properties": {},
                    },
                    handler=h.handle_get_dora_metrics,
                )
            )
            registry.register(
                ToolDef(
                    name="analyze_workflow_failures",
                    description="Analyze workflow failures with AI-powered insights",
                    input_schema={
                        "type": "object",
                        "properties": {"run_id": {"type": "string"}},
                        "required": ["run_id"],
                    },
                    handler=h.handle_analyze_workflow_failures,
                )
            )
    except ImportError:
        pass

    # ── Monitoring ────────────────────────────────────────────────────
    if monitoring_enabled and monitor:
        registry.register(
            ToolDef(
                name="health_status",
                description="Get server health and performance metrics",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                handler=h.make_health_handler(monitor),
            )
        )

    return registry
