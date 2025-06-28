"""Community detection algorithms for NetworkX MCP Server."""

from networkx_mcp.advanced.community.base import CommunityDetector
from networkx_mcp.advanced.community.base import CommunityResult
from networkx_mcp.advanced.community.base import format_community_result
from networkx_mcp.advanced.community.base import validate_communities
from networkx_mcp.advanced.community.girvan_newman import GirvanNewmanDetector
from networkx_mcp.advanced.community.girvan_newman import girvan_newman_communities
from networkx_mcp.advanced.community.louvain import LouvainCommunityDetector
from networkx_mcp.advanced.community.louvain import louvain_communities


__all__ = [
    "CommunityDetector",
    "CommunityResult",
    "GirvanNewmanDetector",
    "LouvainCommunityDetector",
    "format_community_result",
    "girvan_newman_communities",
    "louvain_communities",
    "validate_communities"
]

# Factory function for easy access
def get_community_detector(algorithm: str, graph):
    """Get community detector by algorithm name."""
    detectors = {
        "louvain": LouvainCommunityDetector,
        "girvan_newman": GirvanNewmanDetector
    }

    if algorithm not in detectors:
        msg = f"Unknown algorithm: {algorithm}"
        raise ValueError(msg)

    return detectors[algorithm](graph)
