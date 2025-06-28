"""Louvain algorithm for community detection."""

import networkx as nx
from typing import Dict, List, Set, Any
from .base import CommunityDetector, CommunityResult, format_community_result

class LouvainCommunityDetector(CommunityDetector):
    """Louvain algorithm implementation for community detection."""
    
    async def detect_communities(self, resolution: float = 1.0, threshold: float = 1e-7, max_iter: int = 100) -> CommunityResult:
        """Detect communities using Louvain algorithm."""
        if not self.validate_graph():
            raise ValueError("Graph is not suitable for community detection")
        
        try:
            # Use NetworkX's Louvain implementation
            communities = nx.community.louvain_communities(
                self.graph, 
                resolution=resolution,
                threshold=threshold
            )
            
            # Calculate modularity
            modularity = nx.community.modularity(self.graph, communities)
            
            return CommunityResult(
                communities=list(communities),
                modularity=modularity,
                algorithm="louvain",
                parameters={
                    "resolution": resolution,
                    "threshold": threshold,
                    "max_iter": max_iter
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Louvain algorithm failed: {e}")

def louvain_communities(graph: nx.Graph, resolution: float = 1.0) -> List[Set[str]]:
    """Simple function interface for Louvain communities."""
    detector = LouvainCommunityDetector(graph)
    import asyncio
    result = asyncio.run(detector.detect_communities(resolution=resolution))
    return result.communities

def modularity_optimization(graph: nx.Graph, communities: List[Set[str]]) -> float:
    """Calculate modularity for given community partition."""
    return nx.community.modularity(graph, communities)
