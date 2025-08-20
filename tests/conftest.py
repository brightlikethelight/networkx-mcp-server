"""Pytest configuration and fixtures."""

import asyncio

import networkx as nx
import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_graph() -> nx.Graph:
    """Create a sample graph for testing."""
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])
    G.add_node(5)  # Isolated node
    return G


@pytest.fixture(scope="session", autouse=True)
def cleanup_background_threads():
    """Cleanup background threads after test session."""
    yield
    try:
        from networkx_mcp.graph_cache import graphs

        graphs.shutdown()
    except ImportError:
        pass
