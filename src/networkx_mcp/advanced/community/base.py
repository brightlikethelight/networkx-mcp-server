"""Base interfaces for community detection algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import networkx as nx

# Constants
MIN_NODES_FOR_COMMUNITY_DETECTION = 2


@dataclass
class CommunityResult:
    """Result from community detection algorithm."""

    communities: list[set[str]]
    modularity: float
    algorithm: str
    parameters: dict[str, Any]
    metadata: dict[str, Any] | None = None


class CommunityDetector(ABC):
    """Base class for community detection algorithms."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.name = self.__class__.__name__

    @abstractmethod
    async def detect_communities(self, **params) -> CommunityResult:
        """Detect communities in the graph."""

    def validate_graph(self) -> bool:
        """Validate graph is suitable for community detection."""
        if self.graph.number_of_nodes() < MIN_NODES_FOR_COMMUNITY_DETECTION:
            return False
        if self.graph.number_of_edges() == 0:
            return False
        return True


def validate_communities(communities: list[set[str]], graph: nx.Graph) -> bool:
    """Validate that communities are valid for the graph."""
    all_nodes = set()
    for community in communities:
        if not community:  # Empty community
            return False
        all_nodes.update(community)

    # Check all nodes are covered
    return all_nodes == set(graph.nodes())


def format_community_result(
    communities: list[set[str]], algorithm: str, modularity: float
) -> dict[str, Any]:
    """Format community detection result for API response."""
    return {
        "algorithm": algorithm,
        "num_communities": len(communities),
        "communities": [list(community) for community in communities],
        "modularity": modularity,
        "largest_community_size": (
            max(len(c) for c in communities) if communities else 0
        ),
        "smallest_community_size": (
            min(len(c) for c in communities) if communities else 0
        ),
    }
