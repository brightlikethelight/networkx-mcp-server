"""Pydantic schemas for graph data validation."""

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NodeSchema(BaseModel):
    """Schema for node data."""

    model_config = ConfigDict(extra="allow")

    id: Union[str, int]
    attributes: dict[str, Any] = Field(default_factory=dict)


class EdgeSchema(BaseModel):
    """Schema for edge data."""

    model_config = ConfigDict(extra="allow")

    source: Union[str, int]
    target: Union[str, int]
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
    node_id: Union[str, int]
    attributes: dict[str, Any] = Field(default_factory=dict)


class AddNodesRequest(BaseModel):
    """Request schema for adding multiple nodes."""

    graph_id: str
    nodes: list[Union[str, int, NodeSchema]]


class AddEdgeRequest(BaseModel):
    """Request schema for adding an edge."""

    graph_id: str
    source: Union[str, int]
    target: Union[str, int]
    attributes: dict[str, Any] = Field(default_factory=dict)


class AddEdgesRequest(BaseModel):
    """Request schema for adding multiple edges."""

    graph_id: str
    edges: list[EdgeSchema]


class ShortestPathRequest(BaseModel):
    """Request schema for shortest path algorithms."""

    graph_id: str
    source: Union[str, int]
    target: Optional[Union[str, int]] = None
    weight: Optional[str] = None
    method: Literal["dijkstra", "bellman-ford"] = "dijkstra"


class CentralityRequest(BaseModel):
    """Request schema for centrality measures."""

    graph_id: str
    measures: list[
        Literal["degree", "betweenness", "closeness", "eigenvector", "pagerank"]
    ] = Field(default=["degree"])
    top_k: Optional[int] = 10


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
    path: Optional[str] = None
    options: dict[str, Any] = Field(default_factory=dict)


class ImportGraphRequest(BaseModel):
    """Request schema for importing a graph."""

    format: Literal[
        "json", "graphml", "gexf", "edgelist", "adjacency", "pickle", "pajek"
    ]
    path: Optional[str] = None
    data: Optional[dict[str, Any]] = None
    graph_id: Optional[str] = None
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
    nodes: list[Union[str, int]]
    create_copy: bool = True


class GraphAttributesRequest(BaseModel):
    """Request schema for getting/setting graph attributes."""

    graph_id: str
    node_id: Optional[Union[str, int]] = None
    attribute: Optional[str] = None
    values: Optional[dict[str, Any]] = None


class AlgorithmResponse(BaseModel):
    """Generic response schema for algorithm results."""

    algorithm: str
    success: bool
    result: dict[str, Any]
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None


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
    degree_stats: Optional[dict[str, float]] = None


class VisualizationData(BaseModel):
    """Schema for graph visualization data."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    layout: Optional[dict[str, list[float]]] = None
    options: dict[str, Any] = Field(default_factory=dict)
