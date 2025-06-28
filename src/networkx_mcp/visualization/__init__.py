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
from networkx_mcp.visualization.matplotlib_visualizer import MatplotlibVisualizer
from networkx_mcp.visualization.plotly_visualizer import PlotlyVisualizer
from networkx_mcp.visualization.pyvis_visualizer import PyvisVisualizer
from networkx_mcp.visualization.specialized_viz import SpecializedVisualizations


# Import new modular components
try:
    from networkx_mcp.visualization.base import BaseVisualizer
    from networkx_mcp.visualization.base import VisualizationResult
    from networkx_mcp.visualization.base import calculate_layout
    from networkx_mcp.visualization.base import prepare_graph_data
    from networkx_mcp.visualization.matplotlib_viz import (
        create_matplotlib_visualization,
    )
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
        "calculate_layout",
        "create_matplotlib_visualization",
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
        msg = f"Unknown visualization backend: {backend}"
        raise ValueError(msg)

    return visualizers[backend]()
