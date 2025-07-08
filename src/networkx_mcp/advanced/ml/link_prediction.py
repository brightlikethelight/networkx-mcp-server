"""
Link prediction algorithms.

Split from ml_integration.py for better organization.
"""

import logging
import time
from typing import Any, Dict, List, Tuple
import networkx as nx
import numpy as np

from ...features import feature_flag, is_feature_enabled

logger = logging.getLogger(__name__)


class LinkPrediction:
    """Link prediction algorithms with feature flag protection."""
    
    @staticmethod
    @feature_flag("ml_link_prediction", fallback=lambda *args, **kwargs: {
        "error": "ML link prediction feature is disabled",
        "enabled": False,
        "suggestion": "Enable 'ml_link_prediction' feature flag to use this functionality"
    })
    def predict_links(
        graph: nx.Graph,
        method: str = "common_neighbors",
        top_k: int = 10,
        **params
    ) -> Dict[str, Any]:
        """
        Predict potential new links in the graph.
        
        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        method : str
            Prediction method
        top_k : int
            Number of top predictions to return
            
        Returns:
        --------
        Dict containing predictions and scores
        """
        start_time = time.time()
        
        # Simple common neighbors prediction
        predictions = []
        nodes = list(graph.nodes())
        
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                if not graph.has_edge(u, v):
                    # Calculate prediction score
                    common = len(list(nx.common_neighbors(graph, u, v)))
                    if common > 0:
                        predictions.append((u, v, common))
        
        # Sort by score
        predictions.sort(key=lambda x: x[2], reverse=True)
        top_predictions = predictions[:top_k]
        
        return {
            "predictions": [
                {"source": u, "target": v, "score": float(score)}
                for u, v, score in top_predictions
            ],
            "method": method,
            "top_k": top_k,
            "total_possible_edges": len(nodes) * (len(nodes) - 1) // 2 - graph.number_of_edges(),
            "execution_time_ms": (time.time() - start_time) * 1000
        }
