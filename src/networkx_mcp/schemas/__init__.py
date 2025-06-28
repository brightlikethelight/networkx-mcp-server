"""Graph data schemas for NetworkX MCP server."""

from networkx_mcp.schemas.graph_schemas import AddEdgeRequest
from networkx_mcp.schemas.graph_schemas import AddEdgesRequest
from networkx_mcp.schemas.graph_schemas import AddNodeRequest
from networkx_mcp.schemas.graph_schemas import AddNodesRequest
from networkx_mcp.schemas.graph_schemas import AlgorithmResponse
from networkx_mcp.schemas.graph_schemas import CentralityRequest
from networkx_mcp.schemas.graph_schemas import CommunityDetectionRequest
from networkx_mcp.schemas.graph_schemas import CreateGraphRequest
from networkx_mcp.schemas.graph_schemas import EdgeSchema
from networkx_mcp.schemas.graph_schemas import ExportGraphRequest
from networkx_mcp.schemas.graph_schemas import GraphAttributesRequest
from networkx_mcp.schemas.graph_schemas import GraphInfoResponse
from networkx_mcp.schemas.graph_schemas import GraphSchema
from networkx_mcp.schemas.graph_schemas import ImportGraphRequest
from networkx_mcp.schemas.graph_schemas import LayoutRequest
from networkx_mcp.schemas.graph_schemas import NodeSchema
from networkx_mcp.schemas.graph_schemas import ShortestPathRequest
from networkx_mcp.schemas.graph_schemas import SubgraphRequest
from networkx_mcp.schemas.graph_schemas import VisualizationData


__all__ = [
    "AddEdgeRequest",
    "AddEdgesRequest",
    "AddNodeRequest",
    "AddNodesRequest",
    "AlgorithmResponse",
    "CentralityRequest",
    "CommunityDetectionRequest",
    "CreateGraphRequest",
    "EdgeSchema",
    "ExportGraphRequest",
    "GraphAttributesRequest",
    "GraphInfoResponse",
    "GraphSchema",
    "ImportGraphRequest",
    "LayoutRequest",
    "NodeSchema",
    "ShortestPathRequest",
    "SubgraphRequest",
    "VisualizationData",
]
