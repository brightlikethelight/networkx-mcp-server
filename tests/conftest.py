"""Pytest configuration and fixtures."""

import asyncio
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

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
def graph_manager():
    """Create a GraphManager instance for testing."""
    from networkx_mcp.core.graph_operations import GraphManager

    return GraphManager()


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
            "links": [{"source": 1, "target": 2}, {"source": 2, "target": 3}],
        },
        "edgelist": "1 2\n2 3\n3 1\n",
        "adjacency": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    }


@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        "small_graph_size": 100,
        "medium_graph_size": 1000,
        "large_graph_size": 10000,
        "timeout": 30,
    }


# Advanced Testing Fixtures for Comprehensive Coverage


@pytest.fixture(scope="session")
def test_graph_collection():
    """Collection of various graph types for comprehensive testing."""
    from tests.factories import DataGenerators, GraphFactory

    return {
        "simple": GraphFactory.simple_graph(10, 15),
        "directed": GraphFactory.directed_graph(10, 15),
        "weighted": GraphFactory.weighted_graph(10, 15),
        "complete": GraphFactory.complete_graph(8),
        "tree": GraphFactory.tree_graph(10),
        "bipartite": GraphFactory.bipartite_graph(5, 6, 0.5),
        "social_network": GraphFactory.social_network(20, 4, 0.3),
        "scale_free": GraphFactory.scale_free_graph(20),
        "disconnected": GraphFactory.disconnected_graph(3, 5),
        "with_attributes": GraphFactory.graph_with_attributes(8),
        "centrality_test_graphs": dict(DataGenerators.centrality_test_graphs()),
        "community_test_graphs": dict(DataGenerators.community_test_graphs()),
    }


@pytest.fixture
def graph_factory():
    """Provide GraphFactory for dynamic graph creation in tests."""
    from tests.factories import GraphFactory

    return GraphFactory


@pytest.fixture
def mcp_factory():
    """Provide MCPFactory for creating MCP protocol messages."""
    from tests.factories import MCPFactory

    return MCPFactory


@pytest.fixture
def security_test_data():
    """Provide SecurityTestData for security testing."""
    from tests.factories import SecurityTestData

    return SecurityTestData


@pytest.fixture
def data_generators():
    """Provide DataGenerators for test data creation."""
    from tests.factories import DataGenerators

    return DataGenerators


@pytest.fixture
def memory_tracker():
    """Track memory usage during tests."""
    import os

    import psutil

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    class MemoryTracker:
        def __init__(self):
            self.initial = initial_memory
            self.process = process

        def current_usage(self):
            return self.process.memory_info().rss

        def increase_since_start(self):
            return self.current_usage() - self.initial

        def increase_since_start_mb(self):
            return self.increase_since_start() / (1024 * 1024)

        def assert_reasonable_usage(self, max_increase_mb=100):
            increase = self.increase_since_start_mb()
            assert (
                increase < max_increase_mb
            ), f"Memory usage increased by {increase:.2f}MB, exceeding {max_increase_mb}MB limit"

    return MemoryTracker()


@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time

    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                raise ValueError("Timer not properly started/stopped")
            return self.end_time - self.start_time

        def assert_within_limit(self, max_seconds):
            elapsed = self.elapsed()
            assert (
                elapsed < max_seconds
            ), f"Operation took {elapsed:.3f}s, exceeding {max_seconds}s limit"

    return PerformanceTimer()


@pytest.fixture
async def mcp_server_instance():
    """Create a full MCP server instance for integration testing."""
    try:
        from networkx_mcp.server import NetworkXMCPServer

        server = NetworkXMCPServer()
        yield server
    except ImportError:
        # Fallback if server module structure changes
        from networkx_mcp.core.graph_operations import GraphManager

        yield GraphManager()


@pytest.fixture
def graph_algorithms():
    """Provide GraphAlgorithms instance for testing."""
    from networkx_mcp.core.algorithms import GraphAlgorithms

    return GraphAlgorithms()


@pytest.fixture
def mock_mcp_transport(mocker):
    """Mock MCP transport for testing protocol interactions."""
    transport = mocker.Mock()
    transport.send = mocker.AsyncMock()
    transport.receive = mocker.AsyncMock()
    transport.close = mocker.AsyncMock()
    return transport


@pytest.fixture
def hypothesis_settings():
    """Custom Hypothesis settings for property-based tests."""
    from hypothesis import HealthCheck, settings

    return settings(
        max_examples=50,  # Reduced for faster CI
        deadline=5000,  # 5 second deadline
        suppress_health_check=[HealthCheck.too_slow],
        derandomize=True,
    )


@pytest.fixture(scope="session")
def coverage_tracker():
    """Track code coverage during test execution."""
    try:
        import coverage

        cov = coverage.Coverage()
        cov.start()
        yield cov
        cov.stop()
        cov.save()
    except ImportError:
        # Coverage not available
        yield None


@pytest.fixture
def error_logger():
    """Logger for capturing and analyzing errors in tests."""
    import io
    import logging

    # Create a string buffer to capture log output
    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setLevel(logging.ERROR)

    # Get the root logger and add our handler
    logger = logging.getLogger()
    logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(logging.ERROR)

    class ErrorLogger:
        def get_errors(self):
            return log_buffer.getvalue()

        def has_errors(self):
            return len(self.get_errors()) > 0

        def clear(self):
            log_buffer.truncate(0)
            log_buffer.seek(0)

    error_logger = ErrorLogger()
    yield error_logger

    # Cleanup
    logger.removeHandler(handler)
    logger.setLevel(original_level)


@pytest.fixture
def concurrent_test_helper():
    """Helper for testing concurrent operations."""
    import queue
    import threading
    import time

    class ConcurrentTestHelper:
        def __init__(self):
            self.results = queue.Queue()
            self.errors = queue.Queue()

        def run_concurrent(self, func, n_threads=5, timeout=10):
            """Run a function concurrently in multiple threads."""
            threads = []

            def worker():
                try:
                    result = func()
                    self.results.put(result)
                except Exception as e:
                    self.errors.put(str(e))

            # Start threads
            for _ in range(n_threads):
                thread = threading.Thread(target=worker)
                threads.append(thread)
                thread.start()

            # Wait for completion
            start_time = time.time()
            for thread in threads:
                remaining_time = timeout - (time.time() - start_time)
                thread.join(timeout=max(0, remaining_time))

            # Collect results
            results = []
            while not self.results.empty():
                results.append(self.results.get())

            errors = []
            while not self.errors.empty():
                errors.append(self.errors.get())

            return results, errors

    return ConcurrentTestHelper()


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset any global state before each test."""
    # Clear any global caches or state
    yield
    # Cleanup after test
    import gc

    gc.collect()  # Force garbage collection


# Test categorization markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "redis: Tests requiring Redis")
    config.addinivalue_line("markers", "network: Tests requiring network access")


@pytest.fixture
def temp_files(temp_dir):
    """Create temporary test files with sample data."""
    import json

    import pandas as pd

    # Create sample data files
    files = {}

    # JSON file
    json_data = {
        "nodes": [
            {"id": "A", "label": "Node A", "value": 10},
            {"id": "B", "label": "Node B", "value": 20},
            {"id": "C", "label": "Node C", "value": 15},
        ],
        "edges": [
            {"source": "A", "target": "B", "weight": 1.5},
            {"source": "B", "target": "C", "weight": 2.0},
        ],
    }
    json_file = temp_dir / "graph.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f)
    files["json"] = str(json_file)

    # CSV file
    csv_data = [
        ["source", "target", "weight"],
        ["A", "B", "1.5"],
        ["B", "C", "2.0"],
        ["C", "A", "1.0"],
    ]
    csv_file = temp_dir / "graph.csv"
    with open(csv_file, "w") as f:
        for row in csv_data:
            f.write(",".join(row) + "\n")
    files["csv"] = str(csv_file)

    # Excel file
    try:
        excel_file = temp_dir / "graph.xlsx"
        df = pd.DataFrame(
            [
                {"source": "A", "target": "B", "weight": 1.5},
                {"source": "B", "target": "C", "weight": 2.0},
            ]
        )
        df.to_excel(excel_file, index=False)
        files["excel"] = str(excel_file)
    except ImportError:
        # Skip Excel if openpyxl not available
        pass

    return files


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for testing."""
    return {
        "create_graph": 0.1,  # seconds
        "add_nodes": 0.5,
        "add_edges": 0.5,
        "shortest_path": 1.0,
        "centrality": 2.0,
        "community_detection": 5.0,
    }


@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing error handling."""
    return {
        "invalid_graph_ids": [
            "",  # Empty string
            None,  # None value
            123,  # Number instead of string
            "a" * 256,  # Very long string
            "../../../etc/passwd",  # Path traversal attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "'; DROP TABLE graphs; --",  # SQL injection attempt
            "\x00\x01\x02",  # Binary data
            "CON",  # Windows reserved name
            "PRN",  # Windows reserved name
        ],
        "invalid_node_ids": [
            None,  # None value
            "",  # Empty string
            [],  # Empty list
            {},  # Empty dict
            {"id": "nested"},  # Dict instead of simple value
            [1, 2, 3],  # List instead of simple value
        ],
        "invalid_file_paths": [
            None,  # None value
            "",  # Empty string
            "/nonexistent/path/to/file.json",  # Non-existent path
            "relative/path.json",  # Relative path
            "/path/with spaces/file.json",  # Path with spaces
            "/path/to/file.invalid",  # Invalid extension
            "../../../etc/passwd",  # Path traversal
            "file://malicious.json",  # File URL
            "http://example.com/file.json",  # HTTP URL
        ],
    }
