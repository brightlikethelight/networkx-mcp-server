"""Graph visualization modules.

This package provides multiple visualization backends for graphs:
- matplotlib: Static high-quality plots
- plotly: Interactive web visualizations  
- pyvis: Network visualizations

Example usage:
    from networkx_mcp.visualization import MatplotlibVisualizer, PlotlyVisualizer
    
    # Use existing matplotlib visualizer
    viz = MatplotlibVisualizer()
    
    # Use existing plotly visualizer  
    plotly_viz = PlotlyVisualizer()
"""

# Import existing visualizers
from .matplotlib_visualizer import MatplotlibVisualizer
from .plotly_visualizer import PlotlyVisualizer
from .pyvis_visualizer import PyvisVisualizer
from .specialized_viz import SpecializedVisualizations

# Import new modular components
try:
    from .base import BaseVisualizer, VisualizationResult, calculate_layout, prepare_graph_data
    from .matplotlib_viz import create_matplotlib_visualization
    NEW_MODULES_AVAILABLE = True
except ImportError:
    NEW_MODULES_AVAILABLE = False

__all__ = [
    "MatplotlibVisualizer",
    "PlotlyVisualizer", 
    "PyvisVisualizer",
    "SpecializedVisualizations"
]

# Add new modular components if available
if NEW_MODULES_AVAILABLE:
    __all__.extend([
        "BaseVisualizer",
        "VisualizationResult",
        "create_matplotlib_visualization", 
        "calculate_layout",
        "prepare_graph_data"
    ])

# Factory function
def get_visualizer(backend: str = "matplotlib"):
    """Get visualizer by backend name."""
    visualizers = {
        "matplotlib": MatplotlibVisualizer,
        "plotly": PlotlyVisualizer,
        "pyvis": PyvisVisualizer
    }
    
    if backend not in visualizers:
        raise ValueError(f"Unknown visualization backend: {backend}")
    
    return visualizers[backend]()
