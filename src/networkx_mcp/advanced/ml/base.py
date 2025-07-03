"""Base interfaces for machine learning on graphs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass
class MLResult:
    """Result from ML algorithm on graphs."""

    predictions: dict[str, Any] | list[Any]
    confidence: dict[str, float] | None = None
    model_info: dict[str, Any] | None = None
    features_used: list[str] | None = None


class GraphMLModel(ABC):
    """Base class for graph machine learning models."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.model = None
        self.is_trained = False

    @abstractmethod
    async def extract_features(self, nodes: list[str] | None = None) -> np.ndarray:
        """Extract features from graph nodes."""

    @abstractmethod
    async def train(self, labels: dict[str, Any], **params) -> bool:
        """Train the model."""

    @abstractmethod
    async def predict(self, nodes: list[str]) -> MLResult:
        """Make predictions for given nodes."""


def extract_node_features(
    graph: nx.Graph, feature_types: list[str] | None = None
) -> dict[str, np.ndarray]:
    """Extract standard node features from graph."""
    if feature_types is None:
        feature_types = ["degree", "clustering", "betweenness"]

    features = {}
    nodes = list(graph.nodes())

    if "degree" in feature_types:
        features["degree"] = np.array([graph.degree(node) for node in nodes])

    if "clustering" in feature_types:
        clustering = nx.clustering(graph)
        features["clustering"] = np.array([clustering[node] for node in nodes])

    if "betweenness" in feature_types:
        betweenness = nx.betweenness_centrality(graph)
        features["betweenness"] = np.array([betweenness[node] for node in nodes])

    return features
