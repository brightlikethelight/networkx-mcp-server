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
# Import helper functions
from networkx_mcp.visualization.base import calculate_layout, prepare_graph_data

try:
    from networkx_mcp.visualization.matplotlib_visualizer import MatplotlibVisualizer
except ImportError:
    MatplotlibVisualizer = None

try:
    from networkx_mcp.visualization.plotly_visualizer import PlotlyVisualizer
except ImportError:
    PlotlyVisualizer = None

try:
    from networkx_mcp.visualization.pyvis_visualizer import PyvisVisualizer
except ImportError:
    PyvisVisualizer = None

try:
    from networkx_mcp.visualization.specialized_viz import SpecializedVisualizations
except ImportError:
    SpecializedVisualizations = None

__all__ = [
    "calculate_layout",
    "prepare_graph_data",
]

if SpecializedVisualizations is not None:
    __all__.append("SpecializedVisualizations")

# Add available visualizers to __all__
if MatplotlibVisualizer is not None:
    __all__.append("MatplotlibVisualizer")
if PlotlyVisualizer is not None:
    __all__.append("PlotlyVisualizer")
if PyvisVisualizer is not None:
    __all__.append("PyvisVisualizer")


# Factory function
def get_visualizer(backend: str = "matplotlib"):
    """Get visualizer by backend name."""
    visualizers = {}

    if MatplotlibVisualizer is not None:
        visualizers["matplotlib"] = MatplotlibVisualizer
    if PlotlyVisualizer is not None:
        visualizers["plotly"] = PlotlyVisualizer
    if PyvisVisualizer is not None:
        visualizers["pyvis"] = PyvisVisualizer

    if backend not in visualizers:
        available = list(visualizers.keys())
        msg = f"Backend '{backend}' not available. Available backends: {available}"
        raise ValueError(msg)

    return visualizers[backend]()
