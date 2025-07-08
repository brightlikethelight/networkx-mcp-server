"""
Node classification tasks.

Split from ml_integration.py for better organization.
"""

import logging
import time
from typing import Any, Dict, List, Optional
import networkx as nx
import numpy as np

from ...features import feature_flag

logger = logging.getLogger(__name__)


class NodeClassification:
    """Node classification algorithms with feature flag protection."""
    
    @staticmethod
    @feature_flag("ml_node_classification", fallback=lambda *args, **kwargs: {
        "error": "ML node classification feature is disabled",
        "enabled": False,
        "suggestion": "Enable 'ml_node_classification' feature flag to use this functionality"
    })
    def classify_nodes(
        graph: nx.Graph,
        labels: Optional[Dict[Any, Any]] = None,
        method: str = "label_propagation",
        **params
    ) -> Dict[str, Any]:
        """
        Classify nodes based on graph structure and known labels.
        
        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        labels : dict
            Known node labels for semi-supervised learning
        method : str
            Classification method
            
        Returns:
        --------
        Dict containing classifications
        """
        start_time = time.time()
        
        # Simple label propagation
        if labels is None:
            # Unsupervised clustering as classification
            communities = list(nx.community.greedy_modularity_communities(graph))
            predicted_labels = {}
            for i, community in enumerate(communities):
                for node in community:
                    predicted_labels[node] = i
        else:
            # Semi-supervised label propagation
            predicted_labels = labels.copy()
            unlabeled = set(graph.nodes()) - set(labels.keys())
            
            # Simple propagation: assign most common neighbor label
            for node in unlabeled:
                neighbor_labels = [
                    predicted_labels.get(n) 
                    for n in graph.neighbors(node) 
                    if n in predicted_labels
                ]
                if neighbor_labels:
                    # Most common label
                    label_counts = {}
                    for label in neighbor_labels:
                        label_counts[label] = label_counts.get(label, 0) + 1
                    predicted_labels[node] = max(label_counts, key=label_counts.get)
        
        # Calculate statistics
        unique_labels = set(predicted_labels.values())
        label_distribution = {
            label: sum(1 for v in predicted_labels.values() if v == label)
            for label in unique_labels
        }
        
        return {
            "predicted_labels": predicted_labels,
            "num_classes": len(unique_labels),
            "label_distribution": label_distribution,
            "method": method,
            "num_labeled": len(labels) if labels else 0,
            "num_unlabeled": len(graph) - (len(labels) if labels else 0),
            "execution_time_ms": (time.time() - start_time) * 1000
        }
