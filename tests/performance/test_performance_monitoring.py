"""Performance Monitoring Tests for NetworkX MCP Server.

This module implements performance regression testing and benchmarking
to ensure algorithms scale appropriately and maintain performance standards.
"""

import pytest
import networkx as nx
import time
import psutil
import os
from typing import Dict, List, Callable
import statistics

from networkx_mcp.core.graph_operations import GraphManager


class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    @staticmethod
    def measure_time_and_memory(func: Callable, *args, **kwargs) -> Dict:
        """Measure execution time and memory usage of a function."""
        process = psutil.Process(os.getpid())
        
        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time execution
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_used': final_memory - initial_memory,
            'peak_memory': final_memory
        }
    
    @staticmethod
    def create_test_graphs() -> Dict[str, nx.Graph]:
        """Create test graphs of various sizes for benchmarking."""
        graphs = {}
        
        # Small graph
        graphs['small'] = nx.erdos_renyi_graph(50, 0.1, seed=42)
        
        # Medium graph
        graphs['medium'] = nx.erdos_renyi_graph(200, 0.05, seed=42)
        
        # Large graph
        graphs['large'] = nx.erdos_renyi_graph(500, 0.02, seed=42)
        
        # Dense small graph
        graphs['dense'] = nx.erdos_renyi_graph(100, 0.3, seed=42)
        
        # Scale-free graph
        graphs['scale_free'] = nx.barabasi_albert_graph(300, 3, seed=42)
        
        # Small world graph
        graphs['small_world'] = nx.watts_strogatz_graph(200, 6, 0.3, seed=42)
        
        return graphs


class TestBasicOperationPerformance:
    """Test performance of basic graph operations."""
    
    @pytest.fixture(scope="class")
    def test_graphs(self):
        return PerformanceMonitor.create_test_graphs()
    
    @pytest.fixture
    def graph_manager(self):
        return GraphManager()
    
    def test_graph_creation_performance(self, graph_manager):
        """Test performance of graph creation operations."""
        
        sizes = [10, 50, 100, 200]
        times = []
        
        for size in sizes:
            def create_graph():
                G = nx.erdos_renyi_graph(size, 0.1, seed=42)
                graph_manager.create_graph(f"perf_test_{size}")
                graph_manager.graphs[f"perf_test_{size}"] = G
                return G
            
            metrics = PerformanceMonitor.measure_time_and_memory(create_graph)
            times.append(metrics['execution_time'])
            
            # Clean up
            graph_manager.delete_graph(f"perf_test_{size}")
            
            # Performance assertions
            assert metrics['execution_time'] < 1.0  # Should be fast
            assert metrics['memory_used'] < 100  # Less than 100MB
        
        # Check that time scales reasonably with size
        # Should not be exponential growth
        for i in range(1, len(times)):
            growth_factor = times[i] / times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            # Growth should be roughly proportional to size, not exponential
            assert growth_factor < size_ratio ** 2

    def test_graph_storage_performance(self, test_graphs, graph_manager):
        """Test performance of graph storage and retrieval."""
        
        for name, graph in test_graphs.items():
            # Test storage
            def store_graph():
                graph_manager.create_graph(name)
                graph_manager.graphs[name] = graph
                
            store_metrics = PerformanceMonitor.measure_time_and_memory(store_graph)
            
            # Test retrieval
            retrieve_metrics = PerformanceMonitor.measure_time_and_memory(
                graph_manager.get_graph, name
            )
            
            # Performance assertions
            assert store_metrics['execution_time'] < 0.1
            assert retrieve_metrics['execution_time'] < 0.01  # Retrieval should be very fast
            
            # Verify correctness
            retrieved = retrieve_metrics['result']
            assert retrieved.number_of_nodes() == graph.number_of_nodes()
            assert retrieved.number_of_edges() == graph.number_of_edges()


class TestAlgorithmPerformance:
    """Test performance of graph algorithms."""
    
    @pytest.fixture(scope="class")
    def test_graphs(self):
        return PerformanceMonitor.create_test_graphs()
    
    @pytest.mark.benchmark
    def test_shortest_path_performance(self, test_graphs):
        """Test shortest path algorithm performance."""
        
        for name, graph in test_graphs.items():
            if graph.number_of_nodes() < 2:
                continue
                
            nodes = list(graph.nodes())
            source, target = nodes[0], nodes[-1]
            
            # Test single shortest path
            sp_metrics = PerformanceMonitor.measure_time_and_memory(
                nx.shortest_path, graph, source, target
            )
            
            # Performance thresholds based on graph size
            size = graph.number_of_nodes()
            max_time = 0.001 * size  # Linear scaling expectation
            
            assert sp_metrics['execution_time'] < max_time
            assert sp_metrics['memory_used'] < 50  # Reasonable memory usage
    
    @pytest.mark.benchmark
    def test_centrality_performance(self, test_graphs):
        """Test centrality calculation performance."""
        
        for name, graph in test_graphs.items():
            # Degree centrality (should be very fast)
            degree_metrics = PerformanceMonitor.measure_time_and_memory(
                nx.degree_centrality, graph
            )
            
            size = graph.number_of_nodes()
            assert degree_metrics['execution_time'] < 0.001 * size
            
            # Betweenness centrality (more expensive)
            if size <= 200:  # Only test on smaller graphs
                between_metrics = PerformanceMonitor.measure_time_and_memory(
                    nx.betweenness_centrality, graph
                )
                
                # Betweenness is O(n^3) worst case, but should be reasonable for small graphs
                max_time = 0.01 * (size ** 1.5)  # More lenient scaling
                assert between_metrics['execution_time'] < max_time
    
    @pytest.mark.benchmark
    def test_community_detection_performance(self, test_graphs):
        """Test community detection performance."""
        
        for name, graph in test_graphs.items():
            if graph.number_of_edges() == 0:
                continue
                
            # Greedy modularity communities
            comm_metrics = PerformanceMonitor.measure_time_and_memory(
                lambda g: list(nx.community.greedy_modularity_communities(g)), graph
            )
            
            size = graph.number_of_nodes()
            max_time = 0.1 * size  # Should scale reasonably
            
            assert comm_metrics['execution_time'] < max_time
            assert comm_metrics['memory_used'] < 100


class TestScalabilityLimits:
    """Test scalability limits and degradation patterns."""
    
    def test_memory_scaling(self):
        """Test memory usage scaling with graph size."""
        
        sizes = [100, 200, 500]
        memory_usage = []
        
        for size in sizes:
            def create_large_graph():
                G = nx.erdos_renyi_graph(size, 0.05, seed=42)
                # Add some attributes to increase memory usage
                for node in G.nodes():
                    G.nodes[node]['data'] = list(range(10))
                for edge in G.edges():
                    G.edges[edge]['weight'] = 1.0
                return G
            
            metrics = PerformanceMonitor.measure_time_and_memory(create_large_graph)
            memory_usage.append(metrics['peak_memory'])
            
            # Should not use excessive memory
            assert metrics['peak_memory'] < 500  # Less than 500MB
        
        # Memory should scale roughly linearly with size
        for i in range(1, len(memory_usage)):
            growth_factor = memory_usage[i] / memory_usage[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            # Should not grow faster than quadratically
            assert growth_factor < size_ratio ** 1.5
    
    def test_algorithm_timeout_handling(self):
        """Test that expensive algorithms can be interrupted."""
        
        # Create a graph that might cause expensive computations
        G = nx.erdos_renyi_graph(300, 0.1, seed=42)
        
        start_time = time.time()
        try:
            # Try an expensive operation with reasonable timeout
            result = nx.betweenness_centrality(G)
            elapsed = time.time() - start_time
            
            # Should complete in reasonable time or be interruptible
            assert elapsed < 10.0  # 10 second timeout
            assert len(result) == G.number_of_nodes()
            
        except Exception as e:
            elapsed = time.time() - start_time
            # Even if it fails, should fail quickly
            assert elapsed < 10.0
            assert "timeout" in str(e).lower() or "interrupt" in str(e).lower()


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    @pytest.fixture
    def baseline_graph(self):
        """Create a consistent baseline graph for regression testing."""
        return nx.barabasi_albert_graph(200, 3, seed=12345)
    
    def test_baseline_operations(self, baseline_graph):
        """Test baseline performance for key operations."""
        
        # These are baseline expectations that should not regress
        baselines = {
            'degree_centrality': 0.01,  # 10ms
            'clustering': 0.05,  # 50ms  
            'connected_components': 0.01,  # 10ms
        }
        
        # Degree centrality
        metrics = PerformanceMonitor.measure_time_and_memory(
            nx.degree_centrality, baseline_graph
        )
        assert metrics['execution_time'] < baselines['degree_centrality']
        
        # Clustering coefficient
        metrics = PerformanceMonitor.measure_time_and_memory(
            nx.average_clustering, baseline_graph
        )
        assert metrics['execution_time'] < baselines['clustering']
        
        # Connected components
        metrics = PerformanceMonitor.measure_time_and_memory(
            lambda g: list(nx.connected_components(g)), baseline_graph
        )
        assert metrics['execution_time'] < baselines['connected_components']
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # Perform many graph operations
        for i in range(100):
            G = nx.erdos_renyi_graph(50, 0.1, seed=i)
            nx.degree_centrality(G)
            nx.average_clustering(G)
            # Graph should be garbage collected here
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Should not have significant memory growth
        assert memory_growth < 50  # Less than 50MB growth


class TestConcurrencyPerformance:
    """Test performance under concurrent operations."""
    
    def test_concurrent_graph_access(self):
        """Test performance of concurrent graph operations."""
        import threading
        import queue
        
        manager = GraphManager()
        
        # Create test graph
        G = nx.erdos_renyi_graph(100, 0.1, seed=42)
        manager.create_graph("concurrent_test")
        manager.graphs["concurrent_test"] = G
        
        results = queue.Queue()
        
        def worker():
            start_time = time.time()
            for _ in range(10):
                graph = manager.get_graph("concurrent_test")
                if graph:
                    nx.degree_centrality(graph)
            end_time = time.time()
            results.put(end_time - start_time)
        
        # Run multiple threads
        threads = []
        for _ in range(4):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check that all operations completed in reasonable time
        times = []
        while not results.empty():
            times.append(results.get())
        
        assert len(times) == 4
        assert all(t < 1.0 for t in times)  # Each thread should complete in < 1s
        
        # Check that concurrent access didn't cause excessive slowdown
        avg_time = statistics.mean(times)
        assert avg_time < 0.5  # Average should be reasonable


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])