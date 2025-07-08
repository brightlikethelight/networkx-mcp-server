"""Load testing and performance testing for NetworkX MCP server.

This module implements comprehensive load testing scenarios to understand
system behavior under stress, including concurrent users, large graphs,
and resource monitoring.
"""

import asyncio
import gc
import json
import psutil
import resource
import statistics
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pytest

from networkx_mcp.server import (
    add_edges,
    add_nodes, 
    create_graph,
    delete_graph,
    graph_info,
    list_graphs,
    shortest_path
)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation_name: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    peak_memory_mb: float
    success: bool
    error: str = None


@dataclass 
class LoadTestResults:
    """Container for load test results."""
    test_name: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    total_duration_seconds: float
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    peak_memory_mb: float
    average_cpu_percent: float
    throughput_ops_per_second: float
    errors: List[str]


class PerformanceMonitor:
    """Monitor system performance during tests."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = None
        self.peak_memory = 0
        self.cpu_samples = []
        self.monitoring = False
        
    def start_monitoring(self):
        """Start performance monitoring."""
        tracemalloc.start()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.cpu_samples = []
        self.monitoring = True
        
    def sample_metrics(self) -> Dict[str, float]:
        """Sample current performance metrics."""
        if not self.monitoring:
            return {}
            
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        
        self.peak_memory = max(self.peak_memory, memory_mb)
        self.cpu_samples.append(cpu_percent)
        
        return {
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "peak_memory_mb": self.peak_memory
        }
        
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return summary stats."""
        if not self.monitoring:
            return {}
            
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        avg_cpu = statistics.mean(self.cpu_samples) if self.cpu_samples else 0
        
        self.monitoring = False
        
        return {
            "peak_memory_mb": self.peak_memory,
            "average_cpu_percent": avg_cpu,
            "tracemalloc_peak_mb": peak / 1024 / 1024
        }


class LoadTestRunner:
    """Run various load testing scenarios."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.results = []
        
    @contextmanager
    def performance_context(self, operation_name: str):
        """Context manager for measuring operation performance."""
        start_time = time.perf_counter()
        self.monitor.start_monitoring()
        
        try:
            yield
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        final_metrics = self.monitor.stop_monitoring()
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            duration_ms=duration_ms,
            memory_mb=final_metrics.get("peak_memory_mb", 0),
            cpu_percent=final_metrics.get("average_cpu_percent", 0),
            peak_memory_mb=final_metrics.get("peak_memory_mb", 0),
            success=success,
            error=error
        )
        
        self.results.append(metrics)
        
    def run_concurrent_users_test(self, num_users: int = 100) -> LoadTestResults:
        """Test with multiple concurrent users."""
        print(f"\n🚀 Running concurrent users test: {num_users} users")
        
        def user_workflow(user_id: int) -> List[PerformanceMetrics]:
            """Simulate a single user's workflow."""
            user_results = []
            graph_name = f"user_{user_id}_graph"
            
            try:
                # Create graph
                start = time.perf_counter()
                result = create_graph(name=graph_name, graph_type="undirected")
                duration = (time.perf_counter() - start) * 1000
                
                user_results.append(PerformanceMetrics(
                    operation_name="create_graph",
                    duration_ms=duration,
                    memory_mb=0,  # Individual metrics not tracked in concurrent test
                    cpu_percent=0,
                    peak_memory_mb=0,
                    success="error" not in result
                ))
                
                if "error" in result:
                    return user_results
                
                # Add nodes
                start = time.perf_counter()
                nodes = [f"node_{i}" for i in range(50)]  # 50 nodes per user
                result = add_nodes(graph_name=graph_name, nodes=nodes)
                duration = (time.perf_counter() - start) * 1000
                
                user_results.append(PerformanceMetrics(
                    operation_name="add_nodes",
                    duration_ms=duration,
                    memory_mb=0,
                    cpu_percent=0,
                    peak_memory_mb=0,
                    success="error" not in result
                ))
                
                # Add edges
                start = time.perf_counter()
                edges = []
                for i in range(0, 49, 2):  # Connect every other node
                    edges.append([f"node_{i}", f"node_{i+1}"])
                result = add_edges(graph_name=graph_name, edges=edges)
                duration = (time.perf_counter() - start) * 1000
                
                user_results.append(PerformanceMetrics(
                    operation_name="add_edges", 
                    duration_ms=duration,
                    memory_mb=0,
                    cpu_percent=0,
                    peak_memory_mb=0,
                    success="error" not in result
                ))
                
                # Run algorithm
                start = time.perf_counter()
                result = shortest_path(graph_name=graph_name, source="node_0", target="node_1")
                duration = (time.perf_counter() - start) * 1000
                
                user_results.append(PerformanceMetrics(
                    operation_name="shortest_path",
                    duration_ms=duration,
                    memory_mb=0,
                    cpu_percent=0,
                    peak_memory_mb=0,
                    success="error" not in result
                ))
                
            except Exception as e:
                user_results.append(PerformanceMetrics(
                    operation_name="user_workflow",
                    duration_ms=0,
                    memory_mb=0,
                    cpu_percent=0,
                    peak_memory_mb=0,
                    success=False,
                    error=str(e)
                ))
                
            return user_results
        
        # Monitor overall performance
        self.monitor.start_monitoring()
        start_time = time.perf_counter()
        
        # Run concurrent user workflows
        all_results = []
        with ThreadPoolExecutor(max_workers=min(num_users, 50)) as executor:
            futures = [executor.submit(user_workflow, i) for i in range(num_users)]
            
            for future in as_completed(futures):
                try:
                    user_results = future.result(timeout=60)  # 60 second timeout per user
                    all_results.extend(user_results)
                except Exception as e:
                    all_results.append(PerformanceMetrics(
                        operation_name="concurrent_user",
                        duration_ms=0,
                        memory_mb=0,
                        cpu_percent=0,
                        peak_memory_mb=0,
                        success=False,
                        error=str(e)
                    ))
        
        end_time = time.perf_counter()
        final_metrics = self.monitor.stop_monitoring()
        
        # Calculate summary statistics
        successful_ops = [r for r in all_results if r.success]
        failed_ops = [r for r in all_results if not r.success]
        
        response_times = [r.duration_ms for r in successful_ops]
        
        if response_times:
            avg_response = statistics.mean(response_times)
            p95_response = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_response = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        else:
            avg_response = p95_response = p99_response = 0
            
        total_duration = end_time - start_time
        throughput = len(successful_ops) / total_duration if total_duration > 0 else 0
        
        return LoadTestResults(
            test_name=f"concurrent_users_{num_users}",
            total_operations=len(all_results),
            successful_operations=len(successful_ops),
            failed_operations=len(failed_ops),
            total_duration_seconds=total_duration,
            average_response_time_ms=avg_response,
            p95_response_time_ms=p95_response,
            p99_response_time_ms=p99_response,
            peak_memory_mb=final_metrics.get("peak_memory_mb", 0),
            average_cpu_percent=final_metrics.get("average_cpu_percent", 0),
            throughput_ops_per_second=throughput,
            errors=[r.error for r in failed_ops if r.error]
        )
    
    def run_large_graph_test(self, num_nodes: int = 10000) -> LoadTestResults:
        """Test performance with large graphs."""
        print(f"\n📊 Running large graph test: {num_nodes} nodes")
        
        operations = []
        
        with self.performance_context("create_large_graph"):
            result = create_graph(name="large_graph", graph_type="undirected")
            assert "error" not in result
            
        with self.performance_context("add_large_nodes"):
            # Add nodes in batches to avoid memory issues
            batch_size = 1000
            for i in range(0, num_nodes, batch_size):
                batch_end = min(i + batch_size, num_nodes)
                nodes = [f"node_{j}" for j in range(i, batch_end)]
                result = add_nodes(graph_name="large_graph", nodes=nodes)
                if "error" in result:
                    print(f"Error adding nodes batch {i}-{batch_end}: {result['error']}")
                    break
                    
        with self.performance_context("add_large_edges"):
            # Create a sparse graph (each node connected to next few nodes)
            edges = []
            connections_per_node = min(5, num_nodes - 1)  # Each node connects to 5 others
            
            for i in range(num_nodes):
                for j in range(1, connections_per_node + 1):
                    target = (i + j) % num_nodes
                    if i < target:  # Avoid duplicate edges in undirected graph
                        edges.append([f"node_{i}", f"node_{target}"])
                        
                # Add edges in batches
                if len(edges) >= 5000:
                    result = add_edges(graph_name="large_graph", edges=edges)
                    if "error" in result:
                        print(f"Error adding edges: {result['error']}")
                        break
                    edges = []
                    
            # Add remaining edges
            if edges:
                result = add_edges(graph_name="large_graph", edges=edges)
                
        with self.performance_context("large_graph_algorithm"):
            # Test algorithm on large graph
            result = shortest_path(
                graph_name="large_graph", 
                source="node_0", 
                target="node_100"
            )
            
        with self.performance_context("large_graph_info"):
            # Get graph statistics
            result = graph_info(graph_name="large_graph")
            
        # Clean up
        try:
            delete_graph(graph_name="large_graph")
        except:
            pass
            
        # Calculate results
        successful_ops = [r for r in self.results if r.success]
        failed_ops = [r for r in self.results if not r.success]
        
        response_times = [r.duration_ms for r in successful_ops]
        total_duration = sum(response_times) / 1000  # Convert to seconds
        
        return LoadTestResults(
            test_name=f"large_graph_{num_nodes}_nodes",
            total_operations=len(self.results),
            successful_operations=len(successful_ops),
            failed_operations=len(failed_ops),
            total_duration_seconds=total_duration,
            average_response_time_ms=statistics.mean(response_times) if response_times else 0,
            p95_response_time_ms=statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0,
            p99_response_time_ms=statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else 0,
            peak_memory_mb=max(r.peak_memory_mb for r in self.results) if self.results else 0,
            average_cpu_percent=statistics.mean(r.cpu_percent for r in self.results) if self.results else 0,
            throughput_ops_per_second=len(successful_ops) / total_duration if total_duration > 0 else 0,
            errors=[r.error for r in failed_ops if r.error]
        )
    
    def run_memory_stress_test(self) -> LoadTestResults:
        """Test memory usage patterns and limits."""
        print("\n🧠 Running memory stress test")
        
        # Test creating many small graphs
        num_graphs = 100
        
        with self.performance_context("memory_stress_many_graphs"):
            for i in range(num_graphs):
                graph_name = f"stress_graph_{i}"
                result = create_graph(name=graph_name, graph_type="undirected")
                
                if "error" in result:
                    break
                    
                # Add some data to each graph
                nodes = [f"node_{j}" for j in range(50)]
                add_nodes(graph_name=graph_name, nodes=nodes)
                
                edges = [[f"node_{j}", f"node_{(j+1) % 50}"] for j in range(50)]
                add_edges(graph_name=graph_name, edges=edges)
        
        # Clean up graphs
        graphs_result = list_graphs()
        for graph_name in graphs_result.get("graphs", []):
            if graph_name.startswith("stress_graph_"):
                try:
                    delete_graph(graph_name=graph_name)
                except:
                    pass
        
        # Force garbage collection
        gc.collect()
        
        successful_ops = [r for r in self.results if r.success]
        failed_ops = [r for r in self.results if not r.success]
        
        return LoadTestResults(
            test_name="memory_stress_test",
            total_operations=len(self.results),
            successful_operations=len(successful_ops),
            failed_operations=len(failed_ops),
            total_duration_seconds=sum(r.duration_ms for r in self.results) / 1000,
            average_response_time_ms=statistics.mean(r.duration_ms for r in self.results) if self.results else 0,
            p95_response_time_ms=0,
            p99_response_time_ms=0,
            peak_memory_mb=max(r.peak_memory_mb for r in self.results) if self.results else 0,
            average_cpu_percent=statistics.mean(r.cpu_percent for r in self.results) if self.results else 0,
            throughput_ops_per_second=0,
            errors=[r.error for r in failed_ops if r.error]
        )


class TestLoadPerformance:
    """Load and performance test suite."""
    
    def setup_method(self):
        """Setup for each test."""
        self.runner = LoadTestRunner()
        
    def teardown_method(self):
        """Cleanup after each test."""
        # Clean up any remaining test graphs
        try:
            graphs_result = list_graphs()
            for graph_name in graphs_result.get("graphs", []):
                if any(prefix in graph_name for prefix in ["user_", "test_", "large_", "stress_"]):
                    try:
                        delete_graph(graph_name=graph_name)
                    except:
                        pass
        except:
            pass
    
    @pytest.mark.slow
    def test_concurrent_users_10(self):
        """Test with 10 concurrent users."""
        results = self.runner.run_concurrent_users_test(num_users=10)
        
        # Assertions for reasonable performance
        assert results.successful_operations > 0, "No successful operations"
        assert results.average_response_time_ms < 10000, f"Average response time too high: {results.average_response_time_ms}ms"
        assert results.throughput_ops_per_second > 0.1, f"Throughput too low: {results.throughput_ops_per_second} ops/sec"
        
        print(f"✅ 10 users: {results.successful_operations}/{results.total_operations} ops, "
              f"{results.average_response_time_ms:.1f}ms avg, "
              f"{results.throughput_ops_per_second:.1f} ops/sec")
    
    @pytest.mark.slow 
    def test_concurrent_users_50(self):
        """Test with 50 concurrent users."""
        results = self.runner.run_concurrent_users_test(num_users=50)
        
        assert results.successful_operations > 0, "No successful operations"
        assert results.average_response_time_ms < 20000, f"Average response time too high: {results.average_response_time_ms}ms"
        
        print(f"✅ 50 users: {results.successful_operations}/{results.total_operations} ops, "
              f"{results.average_response_time_ms:.1f}ms avg, "
              f"{results.throughput_ops_per_second:.1f} ops/sec")
    
    @pytest.mark.slow
    @pytest.mark.load_test
    def test_concurrent_users_100(self):
        """Test with 100 concurrent users - the main load test."""
        results = self.runner.run_concurrent_users_test(num_users=100)
        
        # More lenient assertions for high load
        assert results.successful_operations > results.total_operations * 0.5, "Less than 50% operations successful"
        
        print(f"🚀 100 users: {results.successful_operations}/{results.total_operations} ops, "
              f"{results.average_response_time_ms:.1f}ms avg, "
              f"{results.p95_response_time_ms:.1f}ms p95, "
              f"{results.throughput_ops_per_second:.1f} ops/sec, "
              f"{results.peak_memory_mb:.1f}MB peak")
              
        # Log any errors
        if results.errors:
            print(f"⚠️  Errors encountered: {len(results.errors)}")
            for error in results.errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
    
    @pytest.mark.slow
    def test_large_graph_1k_nodes(self):
        """Test with 1,000 node graph."""
        results = self.runner.run_large_graph_test(num_nodes=1000)
        
        assert results.successful_operations > 0, "No successful operations"
        
        print(f"📊 1K nodes: {results.successful_operations}/{results.total_operations} ops, "
              f"{results.peak_memory_mb:.1f}MB peak memory")
    
    @pytest.mark.slow
    def test_large_graph_5k_nodes(self):
        """Test with 5,000 node graph.""" 
        results = self.runner.run_large_graph_test(num_nodes=5000)
        
        assert results.successful_operations > 0, "No successful operations"
        
        print(f"📊 5K nodes: {results.successful_operations}/{results.total_operations} ops, "
              f"{results.peak_memory_mb:.1f}MB peak memory")
    
    @pytest.mark.slow
    @pytest.mark.load_test
    def test_large_graph_10k_nodes(self):
        """Test with 10,000 node graph - the main large graph test."""
        results = self.runner.run_large_graph_test(num_nodes=10000)
        
        # Should handle 10K nodes successfully
        assert results.successful_operations >= 3, "Should complete basic operations"  # create, add_nodes, add_edges
        
        print(f"📊 10K nodes: {results.successful_operations}/{results.total_operations} ops, "
              f"{results.average_response_time_ms:.1f}ms avg, "
              f"{results.peak_memory_mb:.1f}MB peak memory")
              
        # Check memory usage isn't excessive
        assert results.peak_memory_mb < 2048, f"Memory usage too high: {results.peak_memory_mb}MB"
    
    @pytest.mark.slow
    def test_memory_stress(self):
        """Test memory usage patterns."""
        results = self.runner.run_memory_stress_test()
        
        print(f"🧠 Memory stress: {results.peak_memory_mb:.1f}MB peak memory")
        
        # Memory should be reasonable
        assert results.peak_memory_mb < 1024, f"Memory usage too high: {results.peak_memory_mb}MB"


def run_comprehensive_load_tests():
    """Run comprehensive load tests and generate report."""
    print("🧪 Starting comprehensive load testing...")
    
    runner = LoadTestRunner()
    test_results = []
    
    # Test scenarios
    scenarios = [
        ("10_users", lambda: runner.run_concurrent_users_test(10)),
        ("50_users", lambda: runner.run_concurrent_users_test(50)), 
        ("100_users", lambda: runner.run_concurrent_users_test(100)),
        ("1k_nodes", lambda: runner.run_large_graph_test(1000)),
        ("5k_nodes", lambda: runner.run_large_graph_test(5000)),
        ("10k_nodes", lambda: runner.run_large_graph_test(10000)),
        ("memory_stress", lambda: runner.run_memory_stress_test())
    ]
    
    for scenario_name, test_func in scenarios:
        print(f"\n{'='*50}")
        print(f"Running scenario: {scenario_name}")
        print(f"{'='*50}")
        
        try:
            start_time = datetime.now()
            results = test_func()
            end_time = datetime.now()
            
            results.total_duration_seconds = (end_time - start_time).total_seconds()
            test_results.append(results)
            
            print(f"✅ {scenario_name} completed successfully")
            
        except Exception as e:
            print(f"❌ {scenario_name} failed: {str(e)}")
            
        # Clear results for next test
        runner.results = []
    
    # Generate summary report
    generate_performance_report(test_results)
    
    return test_results


def generate_performance_report(results: List[LoadTestResults]):
    """Generate comprehensive performance report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"performance_report_{timestamp}.json"
    
    # Performance summary
    summary = {
        "test_timestamp": timestamp,
        "system_info": {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        },
        "test_results": [],
        "performance_analysis": {},
        "bottlenecks_identified": [],
        "recommendations": []
    }
    
    # Add detailed results
    for result in results:
        summary["test_results"].append({
            "test_name": result.test_name,
            "total_operations": result.total_operations,
            "success_rate": result.successful_operations / result.total_operations if result.total_operations > 0 else 0,
            "average_response_time_ms": result.average_response_time_ms,
            "p95_response_time_ms": result.p95_response_time_ms,
            "throughput_ops_per_second": result.throughput_ops_per_second,
            "peak_memory_mb": result.peak_memory_mb,
            "average_cpu_percent": result.average_cpu_percent,
            "error_count": len(result.errors)
        })
    
    # Analyze performance patterns
    concurrent_tests = [r for r in results if "users" in r.test_name]
    large_graph_tests = [r for r in results if "nodes" in r.test_name]
    
    if concurrent_tests:
        throughputs = [r.throughput_ops_per_second for r in concurrent_tests]
        summary["performance_analysis"]["max_throughput"] = max(throughputs)
        summary["performance_analysis"]["concurrent_user_scaling"] = "linear" if throughputs[-1] > throughputs[0] * 0.5 else "degraded"
    
    if large_graph_tests:
        memory_usage = [r.peak_memory_mb for r in large_graph_tests] 
        summary["performance_analysis"]["memory_scaling"] = "linear" if len(set([m // 100 for m in memory_usage])) <= 2 else "exponential"
    
    # Identify bottlenecks
    for result in results:
        if result.peak_memory_mb > 500:
            summary["bottlenecks_identified"].append(f"High memory usage in {result.test_name}: {result.peak_memory_mb:.1f}MB")
        
        if result.average_response_time_ms > 5000:
            summary["bottlenecks_identified"].append(f"Slow response time in {result.test_name}: {result.average_response_time_ms:.1f}ms")
            
        if result.successful_operations / result.total_operations < 0.8:
            summary["bottlenecks_identified"].append(f"High failure rate in {result.test_name}: {100 - (result.successful_operations / result.total_operations * 100):.1f}% failed")
    
    # Add recommendations
    summary["recommendations"] = [
        "Monitor memory usage for graphs > 5K nodes",
        "Consider connection pooling for high concurrency",
        "Implement caching for frequently accessed graph data",
        "Add circuit breakers for resource protection",
        "Consider async processing for large operations"
    ]
    
    # Save report
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("🏆 PERFORMANCE TEST SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        success_rate = result.successful_operations / result.total_operations * 100 if result.total_operations > 0 else 0
        print(f"📋 {result.test_name:20s}: {success_rate:5.1f}% success, "
              f"{result.average_response_time_ms:6.1f}ms avg, "
              f"{result.peak_memory_mb:6.1f}MB peak")
    
    if summary["bottlenecks_identified"]:
        print(f"\n⚠️  BOTTLENECKS IDENTIFIED:")
        for bottleneck in summary["bottlenecks_identified"]:
            print(f"   - {bottleneck}")
    
    print(f"\n📊 Full report saved to: {report_file}")
    
    return summary


if __name__ == "__main__":
    import sys
    
    # Run comprehensive load tests
    results = run_comprehensive_load_tests()
    
    # Determine exit code based on results
    total_success_rate = sum(r.successful_operations for r in results) / sum(r.total_operations for r in results)
    
    if total_success_rate > 0.8:
        print("\n✅ Load tests PASSED - System performs well under load")
        sys.exit(0)
    else:
        print(f"\n❌ Load tests FAILED - Success rate {total_success_rate*100:.1f}% below threshold")
        sys.exit(1)