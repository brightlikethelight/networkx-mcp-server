"""Louvain algorithm for community detection."""

import asyncio

import networkx as nx

from networkx_mcp.advanced.community.base import (CommunityDetector,
                                                  CommunityResult)


class LouvainCommunityDetector(CommunityDetector):
    """Louvain algorithm implementation for community detection."""

    async def detect_communities(
        self, resolution: float = 1.0, threshold: float = 1e-7, max_iter: int = 100
    ) -> CommunityResult:
        """Detect communities using Louvain algorithm."""
        if not self.validate_graph():
            msg = "Graph is not suitable for community detection"
            raise ValueError(msg)

        try:
            # Use NetworkX's Louvain implementation
            communities = nx.community.louvain_communities(
                self.graph, resolution=resolution, threshold=threshold
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
                    "max_iter": max_iter,
                },
            )

        except Exception as e:
            msg = f"Louvain algorithm failed: {e}"
            raise RuntimeError(msg) from e


def louvain_communities(graph: nx.Graph, resolution: float = 1.0) -> list[set[str]]:
    """Simple function interface for Louvain communities."""
    detector = LouvainCommunityDetector(graph)
    result = asyncio.run(detector.detect_communities(resolution=resolution))
    return result.communities


def modularity_optimization(graph: nx.Graph, communities: list[set[str]]) -> float:
    """Calculate modularity for given community partition."""
    return nx.community.modularity(graph, communities)
