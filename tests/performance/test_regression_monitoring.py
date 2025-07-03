"""Performance regression monitoring tests for NetworkX MCP Server.

These tests establish performance baselines and detect regressions in
critical operations using pytest-benchmark.
"""

import json
import os
import time

import networkx as nx
import psutil
import pytest

from networkx_mcp.core.graph_operations import GraphManager
from tests.factories import GraphFactory


@pytest.mark.performance
class TestGraphOperationPerformance:
    """Test performance of basic graph operations."""

    def test_graph_creation_performance(self, benchmark):
        """Benchmark graph creation performance."""

        def create_graph():
            return GraphFactory.simple_graph(100, 200)

        result = benchmark(create_graph)
        assert result.number_of_nodes() == 100
        assert result.number_of_edges() <= 200

    def test_graph_storage_performance(self, benchmark):
        """Benchmark graph storage performance."""
        manager = GraphManager()
        graph = GraphFactory.simple_graph(100, 200)

        def store_graph():
            manager.store_graph("perf_test", graph)
            return manager.get_graph("perf_test")

        result = benchmark(store_graph)
        assert result is not None
        assert result.number_of_nodes() == 100

    def test_large_graph_handling(self, benchmark):
        """Benchmark performance with larger graphs."""

        def create_large_graph():
            return GraphFactory.simple_graph(1000, 2000)

        result = benchmark(create_large_graph)
        assert result.number_of_nodes() == 1000
        # Verify it completes within reasonable time (handled by benchmark)

    def test_graph_serialization_performance(self, benchmark):
        """Benchmark graph serialization performance."""
        graph = GraphFactory.graph_with_attributes(50)

        def serialize_graph():
            data = nx.node_link_data(graph)
            return json.dumps(data)

        result = benchmark(serialize_graph)
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.performance
class TestAlgorithmPerformance:
    """Test performance of graph algorithms."""

    def test_shortest_path_performance(self, benchmark):
        """Benchmark shortest path calculation."""
        graph = GraphFactory.simple_graph(100, 300)

        def calculate_shortest_path():
            nodes = list(graph.nodes())
            if len(nodes) >= 2:
                return nx.shortest_path(graph, nodes[0], nodes[-1])
            return []

        result = benchmark(calculate_shortest_path)
        assert isinstance(result, list)

    def test_centrality_performance(self, benchmark):
        """Benchmark centrality calculation."""
        graph = GraphFactory.simple_graph(50, 100)

        def calculate_centrality():
            return nx.degree_centrality(graph)

        result = benchmark(calculate_centrality)
        assert isinstance(result, dict)
        assert len(result) == graph.number_of_nodes()

    def test_community_detection_performance(self, benchmark):
        """Benchmark community detection."""
        graph = GraphFactory.social_network(50, 4, 0.3)

        def detect_communities():
            from networkx.algorithms.community import \
                greedy_modularity_communities

            return list(greedy_modularity_communities(graph))

        result = benchmark(detect_communities)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_connected_components_performance(self, benchmark):
        """Benchmark connected components calculation."""
        graph = GraphFactory.disconnected_graph(5, 10)

        def find_components():
            return list(nx.connected_components(graph))

        result = benchmark(find_components)
        assert isinstance(result, list)
        assert len(result) >= 1


@pytest.mark.performance
class TestMemoryUsageMonitoring:
    """Monitor memory usage for potential leaks and excessive consumption."""

    def test_memory_usage_graph_creation(self):
        """Monitor memory usage during graph creation."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create multiple graphs
        graphs = []
        for i in range(20):
            graph = GraphFactory.simple_graph(50, 100)
            graphs.append(graph)

        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory

        # Should not use more than 50MB for 20 small graphs
        assert (
            memory_increase < 50 * 1024 * 1024
        ), f"Memory usage {memory_increase / 1024 / 1024:.2f}MB too high"

        # Cleanup
        del graphs

    def test_memory_usage_algorithm_operations(self):
        """Monitor memory usage during algorithm operations."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        graph = GraphFactory.simple_graph(200, 500)

        # Perform various operations
        nx.shortest_path_length(graph)
        nx.degree_centrality(graph)
        nx.clustering(graph)
        list(nx.connected_components(graph))

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Should not use excessive memory for algorithm operations
        assert (
            memory_increase < 100 * 1024 * 1024
        ), f"Algorithm memory usage {memory_increase / 1024 / 1024:.2f}MB too high"

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        process = psutil.Process(os.getpid())

        memory_measurements = []

        for i in range(10):
            # Create and destroy graphs repeatedly
            manager = GraphManager()
            graph = GraphFactory.simple_graph(100, 200)
            manager.store_graph(f"test_{i}", graph)
            retrieved = manager.get_graph(f"test_{i}")
            assert retrieved is not None

            # Measure memory after each iteration
            memory_measurements.append(process.memory_info().rss)

            # Clean up
            del manager, graph, retrieved

        # Memory should stabilize, not continuously increase
        if len(memory_measurements) >= 5:
            early_avg = sum(memory_measurements[:3]) / 3
            late_avg = sum(memory_measurements[-3:]) / 3

            # Allow for some increase but detect major leaks
            memory_increase = late_avg - early_avg
            max_allowed_increase = 20 * 1024 * 1024  # 20MB

            assert (
                memory_increase < max_allowed_increase
            ), f"Potential memory leak detected: {memory_increase / 1024 / 1024:.2f}MB increase"


@pytest.mark.performance
class TestScalabilityLimits:
    """Test performance scalability and identify limits."""

    @pytest.mark.parametrize(
        "n_nodes,n_edges",
        [
            (10, 20),
            (50, 100),
            (100, 300),
            (200, 600),
            (500, 1500),
        ],
    )
    def test_graph_size_scalability(self, n_nodes, n_edges, benchmark):
        """Test how performance scales with graph size."""

        def create_and_analyze():
            graph = GraphFactory.simple_graph(
                n_nodes, min(n_edges, n_nodes * (n_nodes - 1) // 2)
            )

            # Perform basic analysis
            nx.number_of_nodes(graph)
            nx.number_of_edges(graph)
            nx.density(graph)

            return graph

        result = benchmark(create_and_analyze)
        assert result.number_of_nodes() <= n_nodes

        # Performance should degrade gracefully
        # benchmark automatically tracks this

    def test_concurrent_operations_performance(self, benchmark):
        """Test performance under concurrent load."""
        import queue
        import threading

        manager = GraphManager()
        results_queue = queue.Queue()

        def worker():
            try:
                graph = GraphFactory.simple_graph(20, 40)
                manager.store_graph(
                    f"concurrent_{threading.current_thread().ident}", graph
                )
                retrieved = manager.get_graph(
                    f"concurrent_{threading.current_thread().ident}"
                )
                results_queue.put(retrieved is not None)
            except Exception:
                results_queue.put(False)

        def run_concurrent_test():
            threads = []
            for _ in range(5):  # 5 concurrent operations
                thread = threading.Thread(target=worker)
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # Collect results
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())

            return results

        results = benchmark(run_concurrent_test)
        assert len(results) == 5
        assert all(results), "All concurrent operations should succeed"


@pytest.mark.performance
class TestPerformanceRegression:
    """Test for performance regressions against baseline metrics."""

    PERFORMANCE_BASELINES = {
        "graph_creation_100_nodes": 0.01,  # seconds
        "shortest_path_100_nodes": 0.005,  # seconds
        "centrality_50_nodes": 0.01,  # seconds
        "community_detection_50_nodes": 0.1,  # seconds
    }

    def test_graph_creation_regression(self):
        """Test for regression in graph creation performance."""
        start_time = time.time()
        graph = GraphFactory.simple_graph(100, 200)
        end_time = time.time()

        elapsed = end_time - start_time
        baseline = self.PERFORMANCE_BASELINES["graph_creation_100_nodes"]

        # Allow 50% performance degradation before failing
        assert (
            elapsed < baseline * 1.5
        ), f"Graph creation regression: {elapsed:.4f}s > {baseline * 1.5:.4f}s"

    def test_shortest_path_regression(self):
        """Test for regression in shortest path performance."""
        graph = GraphFactory.simple_graph(100, 300)
        nodes = list(graph.nodes())

        if len(nodes) >= 2:
            start_time = time.time()
            nx.shortest_path(graph, nodes[0], nodes[-1])
            end_time = time.time()

            elapsed = end_time - start_time
            baseline = self.PERFORMANCE_BASELINES["shortest_path_100_nodes"]

            assert (
                elapsed < baseline * 1.5
            ), f"Shortest path regression: {elapsed:.4f}s > {baseline * 1.5:.4f}s"

    def test_centrality_regression(self):
        """Test for regression in centrality calculation performance."""
        graph = GraphFactory.simple_graph(50, 100)

        start_time = time.time()
        nx.degree_centrality(graph)
        end_time = time.time()

        elapsed = end_time - start_time
        baseline = self.PERFORMANCE_BASELINES["centrality_50_nodes"]

        assert (
            elapsed < baseline * 1.5
        ), f"Centrality regression: {elapsed:.4f}s > {baseline * 1.5:.4f}s"

    def test_community_detection_regression(self):
        """Test for regression in community detection performance."""
        graph = GraphFactory.social_network(50, 4, 0.3)

        start_time = time.time()
        from networkx.algorithms.community import greedy_modularity_communities

        list(greedy_modularity_communities(graph))
        end_time = time.time()

        elapsed = end_time - start_time
        baseline = self.PERFORMANCE_BASELINES["community_detection_50_nodes"]

        assert (
            elapsed < baseline * 1.5
        ), f"Community detection regression: {elapsed:.4f}s > {baseline * 1.5:.4f}s"


@pytest.mark.performance
class TestResourceUtilization:
    """Test CPU and memory utilization patterns."""

    def test_cpu_utilization_monitoring(self):
        """Monitor CPU utilization during intensive operations."""
        import time

        # Get initial CPU usage
        cpu_percent_start = psutil.cpu_percent(interval=0.1)

        # Perform CPU-intensive graph operations
        graph = GraphFactory.scale_free_graph(200)

        start_time = time.time()

        # Multiple operations that should use CPU
        nx.betweenness_centrality(graph)
        nx.closeness_centrality(graph)
        list(nx.connected_components(graph))

        end_time = time.time()
        cpu_percent_end = psutil.cpu_percent(interval=0.1)

        # Operations should complete in reasonable time
        elapsed = end_time - start_time
        assert elapsed < 10.0, f"Operations took too long: {elapsed:.2f}s"

        # CPU usage should be reasonable (not stuck at 100%)
        # Note: This is environment-dependent, so we use loose bounds
        assert cpu_percent_end < 95.0, f"CPU usage too high: {cpu_percent_end}%"

    def test_disk_io_monitoring(self):
        """Monitor disk I/O during graph serialization."""
        import os
        import tempfile

        graph = GraphFactory.graph_with_attributes(100)

        # Test file I/O performance
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_file = f.name

            start_time = time.time()

            # Serialize graph to file
            data = nx.node_link_data(graph)
            json.dump(data, f)

            end_time = time.time()

        # Read it back
        start_read = time.time()
        with open(temp_file) as f:
            loaded_data = json.load(f)
        end_read = time.time()

        # Cleanup
        os.unlink(temp_file)

        # I/O should be reasonably fast
        write_time = end_time - start_time
        read_time = end_read - start_read

        assert write_time < 1.0, f"File write too slow: {write_time:.3f}s"
        assert read_time < 1.0, f"File read too slow: {read_time:.3f}s"
        assert loaded_data is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
