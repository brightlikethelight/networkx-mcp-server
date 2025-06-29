"""Link prediction algorithms."""

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from networkx_mcp.advanced.ml.base import GraphMLModel, MLResult


class LinkPredictor(GraphMLModel):
    """Link prediction using graph topology."""

    async def extract_features(
        self, node_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> np.ndarray:
        """Extract features for link prediction."""
        if node_pairs is None:
            # Generate all possible pairs
            nodes = list(self.graph.nodes())
            node_pairs = [
                (u, v)
                for u in nodes
                for v in nodes
                if u < v and not self.graph.has_edge(u, v)
            ]

        features = []
        for u, v in node_pairs:
            # Common neighbors
            common_neighbors = len(list(nx.common_neighbors(self.graph, u, v)))

            # Jaccard coefficient
            try:
                jaccard = next(iter(nx.jaccard_coefficient(self.graph, [(u, v)])))[2]
            except (StopIteration, nx.NetworkXError):
                jaccard = 0

            # Adamic-Adar index
            try:
                adamic_adar = next(iter(nx.adamic_adar_index(self.graph, [(u, v)])))[2]
            except (StopIteration, nx.NetworkXError):
                adamic_adar = 0

            features.append([common_neighbors, jaccard, adamic_adar])

        return np.array(features)

    async def train(
        self,
        positive_edges: List[Tuple[str, str]],
        negative_edges: List[Tuple[str, str]],
        **params,
    ) -> bool:
        """Train link predictor."""
        try:
            # Extract features for positive and negative examples
            pos_features = await self.extract_features(positive_edges)
            neg_features = await self.extract_features(negative_edges)

            # Store training data
            self.pos_features = pos_features
            self.neg_features = neg_features

            self.is_trained = True
            return True

        except Exception:
            return False

    async def predict(self, node_pairs: List[Tuple[str, str]]) -> MLResult:
        """Predict likelihood of links."""
        features = await self.extract_features(node_pairs)

        predictions = {}
        confidence = {}

        for i, (u, v) in enumerate(node_pairs):
            if i < len(features):
                # Simple scoring based on features
                score = np.sum(features[i])  # Sum of all features
                PREDICTION_THRESHOLD = 0.1
                predictions[(u, v)] = score > PREDICTION_THRESHOLD  # Threshold
                confidence[(u, v)] = min(1.0, score / 2.0)  # Normalize

        return MLResult(
            predictions=predictions,
            confidence=confidence,
            model_info={
                "type": "link_predictor",
                "features": ["common_neighbors", "jaccard", "adamic_adar"],
            },
            features_used=["common_neighbors", "jaccard", "adamic_adar"],
        )


async def predict_links(
    graph: nx.Graph, num_predictions: int = 10
) -> List[Tuple[str, str, float]]:
    """Simple function interface for link prediction."""
    predictor = LinkPredictor(graph)

    # Generate candidate pairs
    nodes = list(graph.nodes())
    candidates = [
        (u, v) for u in nodes for v in nodes if u < v and not graph.has_edge(u, v)
    ]

    MAX_CANDIDATES = 100
    if len(candidates) > MAX_CANDIDATES:  # Limit for performance
        candidates = candidates[:MAX_CANDIDATES]

    result = await predictor.predict(candidates)

    # Sort by confidence and return top predictions
    scored_links = [(u, v, conf) for (u, v), conf in result.confidence.items()]
    scored_links.sort(key=lambda x: x[2], reverse=True)

    return scored_links[:num_predictions]
