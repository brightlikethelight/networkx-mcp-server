"""Machine learning algorithms for graphs."""

from .base import GraphMLModel, MLResult, extract_node_features
from .node_classification import NodeClassifier, classify_nodes
from .link_prediction import LinkPredictor, predict_links

__all__ = [
    "GraphMLModel",
    "MLResult",
    "NodeClassifier", 
    "LinkPredictor",
    "classify_nodes",
    "predict_links",
    "extract_node_features"
]

def get_ml_model(model_type: str, graph):
    """Get ML model by type."""
    models = {
        "node_classifier": NodeClassifier,
        "link_predictor": LinkPredictor
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](graph)
