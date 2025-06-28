"""Community detection algorithms for NetworkX MCP Server."""

from .base import CommunityDetector, CommunityResult, validate_communities, format_community_result
from .louvain import LouvainCommunityDetector, louvain_communities
from .girvan_newman import GirvanNewmanDetector, girvan_newman_communities

__all__ = [
    "CommunityDetector",
    "CommunityResult", 
    "LouvainCommunityDetector",
    "GirvanNewmanDetector",
    "louvain_communities",
    "girvan_newman_communities",
    "validate_communities",
    "format_community_result"
]

# Factory function for easy access
def get_community_detector(algorithm: str, graph):
    """Get community detector by algorithm name."""
    detectors = {
        "louvain": LouvainCommunityDetector,
        "girvan_newman": GirvanNewmanDetector
    }
    
    if algorithm not in detectors:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return detectors[algorithm](graph)
