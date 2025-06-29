"""Pytest configuration and fixtures."""

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import networkx as nx
import pytest

from networkx_mcp.storage.redis_backend import RedisBackend


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

    # Add some attributes
    nx.set_node_attributes(G, {1: {"color": "red"}, 2: {"color": "blue"}})
    nx.set_edge_attributes(G, {(1, 2): {"weight": 0.5}, (2, 3): {"weight": 1.5}})

    return G


@pytest.fixture
def directed_graph() -> nx.DiGraph:
    """Create a sample directed graph."""
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1), (3, 4)])
    return G


@pytest.fixture
def large_graph() -> nx.Graph:
    """Create a larger graph for performance testing."""
    return nx.karate_club_graph()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
async def mcp_server():
    """Create an MCP server instance."""
    # Since server.py doesn't export a class, we'll mock the functionality
    from networkx_mcp.core.graph_operations import GraphManager

    manager = GraphManager()
    yield manager


@pytest.fixture
def mock_redis_backend(mocker):
    """Mock Redis backend for testing."""
    mock = mocker.Mock(spec=RedisBackend)
    mock.is_connected = True
    mock.save_graph = mocker.AsyncMock(return_value=True)
    mock.load_graph = mocker.AsyncMock(return_value=None)
    mock.list_graphs = mocker.AsyncMock(return_value=[])
    return mock


@pytest.fixture
def graph_data():
    """Sample graph data in various formats."""
    return {
        "json": {
            "nodes": [{"id": 1}, {"id": 2}, {"id": 3}],
            "links": [{"source": 1, "target": 2}, {"source": 2, "target": 3}]
        },
        "edgelist": "1 2\n2 3\n3 1\n",
        "adjacency": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    }


@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        "small_graph_size": 100,
        "medium_graph_size": 1000,
        "large_graph_size": 10000,
        "timeout": 30
    }
