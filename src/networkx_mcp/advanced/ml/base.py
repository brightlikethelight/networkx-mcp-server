"""Base interfaces for machine learning on graphs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import numpy as np


@dataclass
class MLResult:
    """Result from ML algorithm on graphs."""
    predictions: Union[Dict[str, Any], List[Any]]
    confidence: Optional[Dict[str, float]] = None
    model_info: Optional[Dict[str, Any]] = None
    features_used: Optional[List[str]] = None

class GraphMLModel(ABC):
    """Base class for graph machine learning models."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.model = None
        self.is_trained = False

    @abstractmethod
    async def extract_features(self, nodes: Optional[List[str]] = None) -> np.ndarray:
        """Extract features from graph nodes."""
        pass

    @abstractmethod
    async def train(self, labels: Dict[str, Any], **params) -> bool:
        """Train the model."""
        pass

    @abstractmethod
    async def predict(self, nodes: List[str]) -> MLResult:
        """Make predictions for given nodes."""
        pass

def extract_node_features(graph: nx.Graph, feature_types: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
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
