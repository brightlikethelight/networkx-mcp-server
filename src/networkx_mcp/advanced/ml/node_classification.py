"""Node classification algorithms."""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional
from .base import GraphMLModel, MLResult, extract_node_features

class NodeClassifier(GraphMLModel):
    """Node classification using graph features."""
    
    async def extract_features(self, nodes: Optional[List[str]] = None) -> np.ndarray:
        """Extract features for node classification."""
        if nodes is None:
            nodes = list(self.graph.nodes())
        
        # Extract multiple feature types
        all_features = extract_node_features(
            self.graph, 
            ["degree", "clustering", "betweenness", "closeness"]
        )
        
        # Combine features into matrix
        node_indices = {node: i for i, node in enumerate(self.graph.nodes())}
        target_indices = [node_indices[node] for node in nodes if node in node_indices]
        
        feature_matrix = np.column_stack([
            all_features["degree"][target_indices],
            all_features["clustering"][target_indices], 
            all_features["betweenness"][target_indices]
        ])
        
        return feature_matrix
    
    async def train(self, labels: Dict[str, Any], **params) -> bool:
        """Train node classifier."""
        try:
            # Simple implementation - in real world would use sklearn
            labeled_nodes = list(labels.keys())
            features = await self.extract_features(labeled_nodes)
            
            # Store training data (simplified)
            self.training_features = features
            self.training_labels = np.array(list(labels.values()))
            self.labeled_nodes = labeled_nodes
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    async def predict(self, nodes: List[str]) -> MLResult:
        """Predict labels for nodes."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        features = await self.extract_features(nodes)
        
        # Simple prediction (in practice would use trained model)
        # For demo: predict based on degree centrality
        predictions = {}
        confidence = {}
        
        for i, node in enumerate(nodes):
            if i < len(features):
                degree_feature = features[i][0]  # Degree is first feature
                # Simple rule-based prediction
                pred = "high_degree" if degree_feature > np.mean(features[:, 0]) else "low_degree"
                predictions[node] = pred
                confidence[node] = min(0.9, abs(degree_feature - np.mean(features[:, 0])) / np.std(features[:, 0]))
        
        return MLResult(
            predictions=predictions,
            confidence=confidence,
            model_info={"type": "node_classifier", "features": ["degree", "clustering", "betweenness"]},
            features_used=["degree", "clustering", "betweenness"]
        )

async def classify_nodes(graph: nx.Graph, labeled_nodes: Dict[str, str], target_nodes: List[str]) -> Dict[str, str]:
    """Simple function interface for node classification."""
    classifier = NodeClassifier(graph)
    await classifier.train(labeled_nodes)
    result = await classifier.predict(target_nodes)
    return result.predictions
