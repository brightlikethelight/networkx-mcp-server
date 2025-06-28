"""Base interfaces for graph visualization."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import networkx as nx

@dataclass
class VisualizationResult:
    """Result from visualization rendering."""
    output: str  # HTML, SVG, or file path
    format: str  # "html", "svg", "png", etc.
    metadata: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None

class BaseVisualizer(ABC):
    """Base class for all visualization backends."""
    
    def __init__(self, name: str, output_format: str):
        self.name = name
        self.output_format = output_format
        self.default_options = {}
    
    @abstractmethod
    async def render(self, graph: nx.Graph, layout: str = "spring", **options) -> VisualizationResult:
        """Render a graph visualization."""
        pass
    
    def get_supported_layouts(self) -> List[str]:
        """Get list of supported layout algorithms."""
        return ["spring", "circular", "random"]
    
    def validate_options(self, options: Dict[str, Any]) -> bool:
        """Validate visualization options."""
        return True

def calculate_layout(graph: nx.Graph, layout: str, **params) -> Dict[str, Tuple[float, float]]:
    """Calculate node positions for given layout algorithm."""
    layout_funcs = {
        "spring": lambda g, **p: nx.spring_layout(g, **p),
        "circular": lambda g, **p: nx.circular_layout(g, **p),
        "random": lambda g, **p: nx.random_layout(g, **p),
        "shell": lambda g, **p: nx.shell_layout(g, **p),
        "spectral": lambda g, **p: nx.spectral_layout(g, **p),
        "planar": lambda g, **p: nx.planar_layout(g, **p) if nx.is_planar(g) else nx.spring_layout(g, **p)
    }
    
    if layout not in layout_funcs:
        layout = "spring"  # fallback
    
    try:
        return layout_funcs[layout](graph, **params)
    except Exception:
        # Fallback to spring layout if specific layout fails
        return nx.spring_layout(graph)

def prepare_graph_data(graph: nx.Graph, pos: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """Prepare graph data for visualization."""
    nodes = []
    for node in graph.nodes():
        x, y = pos.get(node, (0, 0))
        node_data = {
            "id": str(node),
            "x": float(x),
            "y": float(y),
            "degree": graph.degree(node),
            **graph.nodes[node]  # Include node attributes
        }
        nodes.append(node_data)
    
    edges = []
    for source, target in graph.edges():
        edge_data = {
            "source": str(source),
            "target": str(target),
            **graph.edges[source, target]  # Include edge attributes
        }
        edges.append(edge_data)
    
    return {
        "nodes": nodes,
        "edges": edges,
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph()
    }
