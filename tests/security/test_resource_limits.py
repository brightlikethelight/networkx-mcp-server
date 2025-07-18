"""Test resource limits to prevent DoS attacks."""

import threading
import time
from unittest.mock import patch

import networkx as nx
import pytest

from src.networkx_mcp.security.resource_limits import (
    LIMITS,
    ConcurrencyLimitError,
    GraphSizeLimitError,
    MemoryLimitError,
    RateLimitError,
    TimeoutError,
    check_concurrent_requests,
    check_graph_size,
    check_memory_limit,
    check_operation_feasibility,
    check_rate_limit,
    estimate_graph_size_mb,
    get_resource_status,
    release_request_slot,
    timeout,
    with_resource_limits,
)


class TestResourceLimits:
    """Test resource limit enforcement."""

    def test_graph_size_limits(self):
        """Test graph size validation."""
        # Small graph should pass
        small_graph = nx.Graph()
        small_graph.add_nodes_from(range(100))
        small_graph.add_edges_from([(i, i + 1) for i in range(99)])

        # Should not raise
        check_graph_size(small_graph)

        # Large graph should fail
        large_graph = nx.Graph()
        large_graph.add_nodes_from(range(LIMITS.max_nodes_per_graph + 1))

        with pytest.raises(GraphSizeLimitError) as exc_info:
            check_graph_size(large_graph)
        assert "too many nodes" in str(exc_info.value)

    def test_memory_estimation(self):
        """Test graph memory size estimation."""
        # Create a graph with known size
        G = nx.Graph()
        G.add_nodes_from(range(1000))
        G.add_edges_from([(i, (i + 1) % 1000) for i in range(1000)])

        size_mb = estimate_graph_size_mb(G)

        # Should be reasonable size
        assert 0 < size_mb < 10  # Small graph should be < 10MB

        # Add attributes to increase size
        for node in G.nodes():
            G.nodes[node]["data"] = "x" * 1000  # 1KB per node

        new_size_mb = estimate_graph_size_mb(G)
        assert new_size_mb > size_mb  # Should be larger with attributes

    def test_operation_feasibility(self):
        """Test operation feasibility checks."""
        # Small graph - all operations should be allowed
        small_graph = nx.complete_graph(100)

        # Should not raise
        check_operation_feasibility(small_graph, "shortest_path")
        check_operation_feasibility(small_graph, "clustering")
        check_operation_feasibility(small_graph, "pagerank")

        # Large graph - expensive operations should be blocked
        large_graph = nx.Graph()
        large_graph.add_nodes_from(range(10000))

        with pytest.raises(GraphSizeLimitError) as exc_info:
            check_operation_feasibility(large_graph, "all_pairs_shortest_path")
        assert "too large for" in str(exc_info.value)

    def test_timeout_decorator(self):
        """Test operation timeout."""

        @timeout(seconds=1)
        def slow_function():
            time.sleep(2)
            return "done"

        with pytest.raises(TimeoutError) as exc_info:
            slow_function()
        assert "timed out after 1 seconds" in str(exc_info.value)

        @timeout(seconds=2)
        def fast_function():
            time.sleep(0.1)
            return "done"

        result = fast_function()
        assert result == "done"

    def test_concurrent_request_limits(self):
        """Test concurrent request limiting."""
        global _active_requests
        _active_requests = 0

        # Should allow up to max concurrent requests
        for i in range(LIMITS.max_concurrent_requests):
            check_concurrent_requests()

        # Next request should fail
        with pytest.raises(ConcurrencyLimitError):
            check_concurrent_requests()

        # Release a slot and try again
        release_request_slot()
        check_concurrent_requests()  # Should work now

        # Clean up
        _active_requests = 0

    def test_rate_limiting(self):
        """Test rate limiting."""
        global _request_times
        _request_times = []

        # Should allow requests up to limit
        for i in range(10):  # Well below rate limit
            check_rate_limit()
            time.sleep(0.01)

        # Simulate hitting rate limit
        current_time = time.time()
        _request_times = [current_time - i for i in range(LIMITS.requests_per_minute)]

        with pytest.raises(RateLimitError):
            check_rate_limit()

        # Clean up
        _request_times = []

    @patch("src.networkx_mcp.security.resource_limits.get_memory_usage_mb")
    def test_memory_limit_check(self, mock_memory):
        """Test memory limit checking."""
        # Under limit - should pass
        mock_memory.return_value = LIMITS.max_memory_mb * 0.5
        check_memory_limit()  # Should not raise

        # Over limit - should fail
        mock_memory.return_value = LIMITS.max_memory_mb * 1.1
        with pytest.raises(MemoryLimitError):
            check_memory_limit()

    def test_with_resource_limits_decorator(self):
        """Test the comprehensive resource limits decorator."""
        call_count = [0]

        @with_resource_limits
        def test_function(value):
            call_count[0] += 1
            if value == "fail":
                raise ValueError("Test error")
            return {"result": value}

        # Normal operation
        result = test_function("success")
        assert result["result"] == "success"
        assert call_count[0] == 1

        # With error
        result = test_function("fail")
        assert "error" in result
        assert call_count[0] == 2

    def test_resource_status(self):
        """Test resource status reporting."""
        status = get_resource_status()

        assert "memory" in status
        assert "current_mb" in status["memory"]
        assert "limit_mb" in status["memory"]

        assert "requests" in status
        assert "active" in status["requests"]
        assert "max_concurrent" in status["requests"]

        assert "limits" in status
        assert "max_graph_size_mb" in status["limits"]
        assert "operation_timeout_seconds" in status["limits"]


class TestDoSPrevention:
    """Test DoS prevention scenarios."""

    def test_large_graph_creation_blocked(self):
        """Test that extremely large graphs are blocked."""
        G = nx.Graph()

        # Try to create a graph that would use too much memory
        with pytest.raises(GraphSizeLimitError):
            # Add nodes that would exceed limits
            G.add_nodes_from(range(LIMITS.max_nodes_per_graph + 1))
            check_graph_size(G)

    def test_concurrent_attack_blocked(self):
        """Test that concurrent request flooding is blocked."""
        global _active_requests
        _active_requests = 0

        blocked_count = 0
        allowed_count = 0

        def make_request():
            nonlocal blocked_count, allowed_count
            try:
                check_concurrent_requests()
                allowed_count += 1
                time.sleep(0.1)  # Simulate work
                release_request_slot()
            except ConcurrencyLimitError:
                blocked_count += 1

        # Launch many concurrent threads
        threads = []
        for i in range(20):  # More than limit
            t = threading.Thread(target=make_request)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have blocked some requests
        assert allowed_count <= LIMITS.max_concurrent_requests
        assert blocked_count > 0

        # Clean up
        _active_requests = 0

    def test_memory_exhaustion_prevented(self):
        """Test that memory exhaustion attacks are prevented."""
        graphs = []

        with patch(
            "src.networkx_mcp.security.resource_limits.get_memory_usage_mb"
        ) as mock_memory:
            # Simulate increasing memory usage
            mock_memory.side_effect = lambda: len(graphs) * 100

            # Try to create many graphs
            for i in range(20):
                G = nx.complete_graph(100)
                graphs.append(G)

                if len(graphs) * 100 > LIMITS.max_memory_mb:
                    with pytest.raises(MemoryLimitError):
                        check_memory_limit()
                    break

            # Should have stopped before using all memory
            assert len(graphs) < 20


def test_server_survives_attack():
    """Integration test: server survives various attack attempts."""
    from src.networkx_mcp.server import add_edges, add_nodes, create_graph

    # Test 1: Try to create oversized graph
    result = create_graph(
        "huge_graph",
        "undirected",
        {
            "nodes": list(range(200000)),  # Way over limit
            "edges": [(i, i + 1) for i in range(199999)],
        },
    )
    assert "error" in result

    # Test 2: Try to add too many nodes
    create_graph("test_graph", "undirected")
    result = add_nodes("test_graph", list(range(50000)))  # Over limit
    assert "error" in result

    # Test 3: Try to add too many edges
    result = add_edges("test_graph", [[i, i + 1] for i in range(50000)])
    assert "error" in result

    # Server should still be responsive
    result = create_graph("normal_graph", "undirected")
    assert result.get("success") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
