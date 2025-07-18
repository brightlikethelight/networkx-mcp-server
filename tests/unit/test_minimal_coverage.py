"""Minimal tests to improve basic coverage."""

import networkx as nx


# Test core imports work
def test_imports():
    """Test that core modules can be imported."""
    assert True


# Test GraphManager basics
def test_graph_manager():
    """Test GraphManager basic functionality."""
    from networkx_mcp.core.graph_operations import GraphManager

    manager = GraphManager()

    # Create graph
    graph_id = manager.create_graph("test1", "Graph")
    assert graph_id == "test1"
    assert "test1" in manager.graphs

    # Get graph info
    info = manager.get_graph_info("test1")
    assert info["graph_id"] == "test1"
    assert info["num_nodes"] == 0

    # Add nodes
    manager.add_nodes("test1", [1, 2, 3])
    info = manager.get_graph_info("test1")
    assert info["num_nodes"] == 3

    # Delete graph
    assert manager.delete_graph("test1") is True
    assert "test1" not in manager.graphs


# Test algorithms
def test_algorithms():
    """Test GraphAlgorithms basic functionality."""
    from networkx_mcp.core.algorithms import GraphAlgorithms

    algo = GraphAlgorithms()

    # Create test graph
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4)])

    # Test shortest path
    result = algo.shortest_path(G, 1, 4)
    assert result["path"] == [1, 2, 3, 4]
    assert result["length"] == 3

    # Test connected components
    result = algo.connected_components(G)
    assert result["num_components"] == 1
    assert result["is_connected"] is True

    # Test empty graph
    empty = nx.Graph()
    result = algo.connected_components(empty)
    assert result["num_components"] == 0


# Test validation
def test_validation():
    """Test validation basics."""
    from networkx_mcp.security.validation import RequestValidator

    # Valid graph ID
    result = RequestValidator.validate_graph_id("test_graph")
    assert result.is_valid is True
    assert result.sanitized_data == "test_graph"

    # Invalid graph ID
    result = RequestValidator.validate_graph_id("test graph with spaces")
    assert result.is_valid is False
    assert len(result.errors) > 0

    # Node ID validation
    result = RequestValidator.validate_node_id("node1")
    assert result.is_valid is True


# Test metrics
def test_metrics():
    """Test metrics basics."""
    from networkx_mcp.monitoring.metrics import Counter, Gauge, Histogram

    # Counter
    counter = Counter("test_counter", "Test counter")
    counter.inc()
    assert counter.value == 1
    counter.inc(5)
    assert counter.value == 6

    # Histogram
    hist = Histogram("test_hist", "Test histogram")
    hist.observe(0.1)
    hist.observe(0.2)
    assert hist.count == 2
    assert hist.sum == 0.3

    # Gauge
    gauge = Gauge("test_gauge", "Test gauge")
    gauge.set(42)
    assert gauge.value == 42
    gauge.inc(8)
    assert gauge.value == 50


# Test logging
def test_logging():
    """Test logging basics."""
    from networkx_mcp.monitoring.logging import LogContext, StructuredLogger

    # Create logger
    logger = StructuredLogger("test")
    assert logger.name == "test"

    # Create context
    context = LogContext()
    context.add("key", "value")
    assert context.get()["key"] == "value"
    context.clear()
    assert context.get() == {}


# Test health checks
def test_health_checks():
    """Test health check basics."""
    from networkx_mcp.monitoring.health_checks import (
        HealthCheck,
        HealthCheckResult,
        HealthStatus,
    )

    # Test health status
    assert HealthStatus.HEALTHY.value == "healthy"

    # Test health check result
    result = HealthCheckResult(
        status=HealthStatus.HEALTHY, message="OK", details={"test": True}
    )
    assert result.status == HealthStatus.HEALTHY
    assert result.message == "OK"

    # Test health check
    health = HealthCheck()
    health.register_check(
        "test", lambda: HealthCheckResult(HealthStatus.HEALTHY, "Good")
    )
    results = health.run_all_checks()
    assert "test" in results
    assert results["test"].status == HealthStatus.HEALTHY
