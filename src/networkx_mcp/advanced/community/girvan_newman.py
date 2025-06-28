"""Girvan-Newman algorithm for community detection."""

import asyncio

from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import networkx as nx

from networkx_mcp.advanced.community.base import CommunityDetector
from networkx_mcp.advanced.community.base import CommunityResult


class GirvanNewmanDetector(CommunityDetector):
    """Girvan-Newman algorithm implementation."""

    async def detect_communities(self, k: Optional[int] = None, max_communities: int = 10) -> CommunityResult:
        """Detect communities using Girvan-Newman algorithm."""
        if not self.validate_graph():
            msg = "Graph is not suitable for community detection"
            raise ValueError(msg)

        try:
            # Use NetworkX's Girvan-Newman implementation
            communities_generator = nx.community.girvan_newman(self.graph)

            # Get the first k communities or stop at max_communities
            if k is not None:
                communities = []
                for i, community_set in enumerate(communities_generator):
                    if i >= k - 1:
                        communities = list(community_set)
                        break
            else:
                # Find optimal number of communities (up to max_communities)
                best_communities = None
                best_modularity = -1

                for i, community_set in enumerate(communities_generator):
                    if i >= max_communities:
                        break

                    communities_list = list(community_set)
                    modularity = nx.community.modularity(self.graph, communities_list)

                    if modularity > best_modularity:
                        best_modularity = modularity
                        best_communities = communities_list

                communities = best_communities if best_communities else [set(self.graph.nodes())]

            # Calculate final modularity
            modularity = nx.community.modularity(self.graph, communities)

            return CommunityResult(
                communities=communities,
                modularity=modularity,
                algorithm="girvan_newman",
                parameters={"k": k, "max_communities": max_communities}
            )

        except Exception as e:
            msg = f"Girvan-Newman algorithm failed: {e}"
            raise RuntimeError(msg) from e

def girvan_newman_communities(graph: nx.Graph, k: int = 2) -> List[Set[str]]:
    """Simple function interface for Girvan-Newman communities."""
    detector = GirvanNewmanDetector(graph)
    result = asyncio.run(detector.detect_communities(k=k))
    return result.communities

def edge_betweenness_centrality(graph: nx.Graph) -> Dict[tuple, float]:
    """Calculate edge betweenness centrality (used in Girvan-Newman)."""
    return nx.edge_betweenness_centrality(graph)
