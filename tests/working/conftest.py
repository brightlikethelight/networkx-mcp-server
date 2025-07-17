"""
Test configuration that actually works.

This forces the use of the minimal server implementation
and sets up proper test isolation.
"""

import pytest
import os
import sys
from pathlib import Path

# Force minimal server for all tests
os.environ["USE_MINIMAL_SERVER"] = "true"

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up clean test environment for each test."""
    # Clear any existing graphs
    from networkx_mcp.server import graphs
    graphs.clear()
    yield
    # Clean up after test
    graphs.clear()

@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    from networkx_mcp.server import create_graph, add_nodes, add_edges
    
    # Create a simple test graph
    create_graph("test_graph", directed=False)
    add_nodes("test_graph", [1, 2, 3, 4, 5])
    add_edges("test_graph", [[1, 2], [2, 3], [3, 4], [4, 5]])
    
    return "test_graph"