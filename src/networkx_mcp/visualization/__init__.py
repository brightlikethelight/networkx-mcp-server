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
from networkx_mcp.visualization.matplotlib_visualizer import \
    MatplotlibVisualizer
from networkx_mcp.visualization.plotly_visualizer import PlotlyVisualizer
from networkx_mcp.visualization.pyvis_visualizer import PyvisVisualizer
from networkx_mcp.visualization.specialized_viz import \
    SpecializedVisualizations

# Import helper functions
from networkx_mcp.visualization.base import calculate_layout, prepare_graph_data

__all__ = [
    "MatplotlibVisualizer",
    "PlotlyVisualizer",
    "PyvisVisualizer",
    "SpecializedVisualizations",
    "calculate_layout",
    "prepare_graph_data"
]

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
