"""Machine learning algorithms for graphs."""

from networkx_mcp.advanced.ml.base import GraphMLModel
from networkx_mcp.advanced.ml.base import MLResult
from networkx_mcp.advanced.ml.base import extract_node_features
from networkx_mcp.advanced.ml.link_prediction import LinkPredictor
from networkx_mcp.advanced.ml.link_prediction import predict_links
from networkx_mcp.advanced.ml.node_classification import NodeClassifier
from networkx_mcp.advanced.ml.node_classification import classify_nodes


__all__ = [
    "GraphMLModel",
    "LinkPredictor",
    "MLResult",
    "NodeClassifier",
    "classify_nodes",
    "extract_node_features",
    "predict_links"
]

def get_ml_model(model_type: str, graph):
    """Get ML model by type."""
    models = {
        "node_classifier": NodeClassifier,
        "link_predictor": LinkPredictor
    }

    if model_type not in models:
        msg = f"Unknown model type: {model_type}"
        raise ValueError(msg)

    return models[model_type](graph)
