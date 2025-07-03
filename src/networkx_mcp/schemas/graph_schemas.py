"""Pydantic schemas for graph data validation."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NodeSchema(BaseModel):
    """Schema for node data."""

    model_config = ConfigDict(extra="allow")

    id: str | int
    attributes: dict[str, Any] = Field(default_factory=dict)


class EdgeSchema(BaseModel):
    """Schema for edge data."""

    model_config = ConfigDict(extra="allow")

    source: str | int
    target: str | int
    attributes: dict[str, Any] = Field(default_factory=dict)


class GraphSchema(BaseModel):
    """Schema for graph data."""

    directed: bool = False
    multigraph: bool = False
    graph_attributes: dict[str, Any] = Field(default_factory=dict)
    nodes: list[NodeSchema] = Field(default_factory=list)
    edges: list[EdgeSchema] = Field(default_factory=list)


class CreateGraphRequest(BaseModel):
    """Request schema for creating a graph."""

    graph_id: str
    graph_type: Literal["Graph", "DiGraph", "MultiGraph", "MultiDiGraph"] = "Graph"
    attributes: dict[str, Any] = Field(default_factory=dict)


class AddNodeRequest(BaseModel):
    """Request schema for adding a node."""

    graph_id: str
    node_id: str | int
    attributes: dict[str, Any] = Field(default_factory=dict)


class AddNodesRequest(BaseModel):
    """Request schema for adding multiple nodes."""

    graph_id: str
    nodes: list[str | int | NodeSchema]


class AddEdgeRequest(BaseModel):
    """Request schema for adding an edge."""

    graph_id: str
    source: str | int
    target: str | int
    attributes: dict[str, Any] = Field(default_factory=dict)


class AddEdgesRequest(BaseModel):
    """Request schema for adding multiple edges."""

    graph_id: str
    edges: list[EdgeSchema]


class ShortestPathRequest(BaseModel):
    """Request schema for shortest path algorithms."""

    graph_id: str
    source: str | int
    target: str | int | None = None
    weight: str | None = None
    method: Literal["dijkstra", "bellman-ford"] = "dijkstra"


class CentralityRequest(BaseModel):
    """Request schema for centrality measures."""

    graph_id: str
    measures: list[
        Literal["degree", "betweenness", "closeness", "eigenvector", "pagerank"]
    ] = Field(default=["degree"])
    top_k: int | None = 10


class CommunityDetectionRequest(BaseModel):
    """Request schema for community detection."""

    graph_id: str
    method: Literal["louvain", "label_propagation", "greedy_modularity"] = "louvain"


class ExportGraphRequest(BaseModel):
    """Request schema for exporting a graph."""

    graph_id: str
    format: Literal[
        "json", "graphml", "gexf", "edgelist", "adjacency", "pickle", "dot", "pajek"
    ]
    path: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ImportGraphRequest(BaseModel):
    """Request schema for importing a graph."""

    format: Literal[
        "json", "graphml", "gexf", "edgelist", "adjacency", "pickle", "pajek"
    ]
    path: str | None = None
    data: dict[str, Any] | None = None
    graph_id: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data")
    @classmethod
    def validate_data_or_path(cls, v, info):
        """Ensure either data or path is provided."""
        if v is None and info.data.get("path") is None:
            msg = "Either 'data' or 'path' must be provided"
            raise ValueError(msg)
        return v


class LayoutRequest(BaseModel):
    """Request schema for graph layout calculation."""

    graph_id: str
    algorithm: Literal[
        "spring", "circular", "random", "shell", "spectral", "kamada_kawai", "planar"
    ] = "spring"
    options: dict[str, Any] = Field(default_factory=dict)


class SubgraphRequest(BaseModel):
    """Request schema for creating a subgraph."""

    graph_id: str
    nodes: list[str | int]
    create_copy: bool = True


class GraphAttributesRequest(BaseModel):
    """Request schema for getting/setting graph attributes."""

    graph_id: str
    node_id: str | int | None = None
    attribute: str | None = None
    values: dict[str, Any] | None = None


class AlgorithmResponse(BaseModel):
    """Generic response schema for algorithm results."""

    algorithm: str
    success: bool
    result: dict[str, Any]
    execution_time_ms: float | None = None
    error: str | None = None


class GraphInfoResponse(BaseModel):
    """Response schema for graph information."""

    graph_id: str
    graph_type: str
    num_nodes: int
    num_edges: int
    density: float
    is_directed: bool
    is_multigraph: bool
    metadata: dict[str, Any]
    degree_stats: dict[str, float] | None = None


class VisualizationData(BaseModel):
    """Schema for graph visualization data."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    layout: dict[str, list[float]] | None = None
    options: dict[str, Any] = Field(default_factory=dict)
