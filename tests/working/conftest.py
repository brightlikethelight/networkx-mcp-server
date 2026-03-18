"""
Test configuration with proper test isolation.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up clean test environment for each test."""
    # Clear any existing graphs
    from networkx_mcp.graph_cache import graphs

    graphs.clear()
    yield
    # Clean up after test
    graphs.clear()


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    from networkx_mcp.core.basic_operations import add_edges, add_nodes, create_graph
    from networkx_mcp.graph_cache import graphs

    # Create a simple test graph
    create_graph("test_graph", directed=False, graphs=graphs)
    add_nodes("test_graph", [1, 2, 3, 4, 5], graphs=graphs)
    add_edges("test_graph", [[1, 2], [2, 3], [3, 4], [4, 5]], graphs=graphs)

    return "test_graph"
