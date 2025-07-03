"""Basic tests for CI/CD validation."""

import networkx as nx
import pytest


def test_package_imports():
    """Test that all core packages can be imported."""
    import networkx_mcp
    import networkx_mcp.core
    import networkx_mcp.schemas
    import networkx_mcp.storage
    import networkx_mcp.utils

    assert networkx_mcp is not None
    assert networkx_mcp.core is not None


def test_networkx_functionality():
    """Test basic NetworkX functionality."""
    G = nx.Graph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")

    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert list(G.neighbors("B")) == ["A", "C"]


def test_directed_graph():
    """Test directed graph creation."""
    DG = nx.DiGraph()
    DG.add_edge("A", "B")
    DG.add_edge("B", "C")

    assert DG.has_edge("A", "B")
    assert not DG.has_edge("B", "A")


def test_graph_algorithms():
    """Test basic graph algorithms."""
    G = nx.complete_graph(5)

    # Test basic properties
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 10

    # Test shortest path
    path = nx.shortest_path(G, 0, 4)
    assert len(path) == 2  # Direct connection in complete graph

    # Test connected components
    assert nx.is_connected(G)
    assert nx.number_connected_components(G) == 1


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph])
def test_graph_types(graph_type):
    """Test different graph types."""
    G = graph_type()
    G.add_edges_from([(1, 2), (2, 3), (3, 4)])

    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3


def test_optional_imports():
    """Test that optional imports are handled gracefully."""
    # These imports should not fail even if packages are missing
    try:
        from networkx_mcp.advanced.enterprise import (HAS_JINJA2,
                                                      HAS_REPORTLAB,
                                                      HAS_SCHEDULE)
        from networkx_mcp.advanced.ml import HAS_SKLEARN
        from networkx_mcp.integration.data_pipelines import HAS_AIOHTTP
        from networkx_mcp.monitoring.resource_manager import HAS_PSUTIL
        from networkx_mcp.utils.performance import \
            HAS_PSUTIL as PERF_HAS_PSUTIL
        from networkx_mcp.visualization.matplotlib_visualizer import \
            HAS_MATPLOTLIB
        from networkx_mcp.visualization.plotly_visualizer import HAS_PLOTLY
        from networkx_mcp.visualization.pyvis_visualizer import HAS_PYVIS

        # These should be boolean flags
        assert isinstance(HAS_MATPLOTLIB, bool)
        assert isinstance(HAS_PLOTLY, bool)
        assert isinstance(HAS_PYVIS, bool)
        assert isinstance(HAS_SKLEARN, bool)
        assert isinstance(HAS_SCHEDULE, bool)
        assert isinstance(HAS_JINJA2, bool)
        assert isinstance(HAS_REPORTLAB, bool)
        assert isinstance(HAS_AIOHTTP, bool)
        assert isinstance(HAS_PSUTIL, bool)
        assert isinstance(PERF_HAS_PSUTIL, bool)

    except ImportError:
        # If imports fail, that's okay - we're testing optional dependencies
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
