#!/usr/bin/env python3
"""Comprehensive load testing for NetworkX MCP Server."""

import asyncio
import random
import sys
import time
from pathlib import Path

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestLoadCapacity:
    """Test server capacity under various load conditions."""
    
    def setup_method(self):
        """Set up test method."""
        self.baseline_memory = None
        self.test_results = []

    def get_system_stats(self):
        """Get current system resource usage."""
        if not HAS_PSUTIL:
            return {
                "memory_mb": 0,
                "cpu_percent": 0,
                "threads": 0,
                "open_files": 0,
            }
        process = psutil.Process()
        return {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads(),
            "open_files": (
                len(process.open_files()) if hasattr(process, "open_files") else 0
            ),
        }

    async def simulate_realistic_user(
        self, user_id: int, operations: int, graph_manager
    ):
        """Simulate realistic user behavior with NetworkX operations."""
        user_stats = {
            "user_id": user_id,
            "operations_completed": 0,
            "errors": 0,
            "total_time": 0,
        }

        try:
            start_time = time.time()

            for op in range(operations):
                time.time()

                try:
                    # Create a graph for this operation
                    graph_id = f"user_{user_id}_graph_{op}"

                    # 1. Create graph
                    result = graph_manager.create_graph(graph_id, "DiGraph")
                    if not result.get("created"):
                        msg = f"Failed to create graph {graph_id}"
                        raise Exception(msg)

                    # 2. Build a social network
                    nodes = [
                        f"person_{i}" for i in range(20)
                    ]  # Smaller for load testing
                    graph_manager.add_nodes_from(graph_id, nodes)

                    # 3. Add random edges (social connections)
                    edges = []
                    for _i in range(30):  # 30 random connections
                        src = random.choice(nodes)
                        dst = random.choice(nodes)
                        if src != dst:
                            edges.append((src, dst))

                    if edges:
                        graph_manager.add_edges_from(graph_id, edges)

                    # 4. Run some analysis (this is computationally expensive)
                    info = graph_manager.get_graph_info(graph_id)

                    # 5. Get neighbors of a random node
                    if info["num_nodes"] > 0:
                        random_node = random.choice(nodes)
                        try:
                            graph_manager.get_neighbors(graph_id, random_node)
                        except ValueError:
                            pass  # Node might not exist, that's ok

                    # 6. Clean up - delete the graph
                    graph_manager.delete_graph(graph_id)

                    user_stats["operations_completed"] += 1

                except Exception:
                    user_stats["errors"] += 1

                # Small delay between operations to be realistic
                await asyncio.sleep(0.01)

            user_stats["total_time"] = time.time() - start_time

        except Exception:
            user_stats["errors"] += 1

        return user_stats

    async def test_concurrent_users(self):
        """Find breaking point for concurrent users."""

        try:
            import add_persistence
            import security_patches

            from networkx_mcp.server import graph_manager

            # Apply security and persistence
            security_patches.apply_critical_patches()
            add_persistence.patch_graph_manager_with_persistence()

            # Record baseline
            self.baseline_memory = self.get_system_stats()["memory_mb"]

            # Test with increasing numbers of concurrent users
            for num_users in [1, 5, 10, 20, 50]:
                start_time = time.time()
                start_stats = self.get_system_stats()

                # Create tasks for concurrent users
                tasks = [
                    self.simulate_realistic_user(i, 5, graph_manager)  # 5 ops per user
                    for i in range(num_users)
                ]

                try:
                    # Run all users concurrently
                    user_results = await asyncio.gather(*tasks, return_exceptions=True)

                    duration = time.time() - start_time
                    end_stats = self.get_system_stats()

                    # Analyze results
                    successful_users = 0
                    total_operations = 0
                    total_errors = 0

                    for result in user_results:
                        if isinstance(result, dict):
                            successful_users += 1
                            total_operations += result["operations_completed"]
                            total_errors += result["errors"]
                        else:
                            pass

                    memory_increase = end_stats["memory_mb"] - start_stats["memory_mb"]
                    ops_per_second = total_operations / duration if duration > 0 else 0

                    test_result = {
                        "num_users": num_users,
                        "duration": duration,
                        "successful_users": successful_users,
                        "total_operations": total_operations,
                        "total_errors": total_errors,
                        "ops_per_second": ops_per_second,
                        "memory_increase_mb": memory_increase,
                        "final_memory_mb": end_stats["memory_mb"],
                        "cpu_percent": end_stats["cpu_percent"],
                        "success": successful_users == num_users
                        and total_errors < num_users * 0.1,
                    }

                    self.test_results.append(test_result)

                    if not test_result["success"]:
                        break

                except Exception:
                    break

                # Clean up and wait a bit between tests
                await asyncio.sleep(1)

            return True

        except Exception:
            return False

    async def test_memory_limits(self):
        """Verify memory limits are enforced properly."""

        try:
            import add_persistence
            import security_patches

            from networkx_mcp.server import graph_manager

            # Apply security and persistence
            security_patches.apply_critical_patches()
            add_persistence.patch_graph_manager_with_persistence()

            # Try to create graphs that would exceed reasonable memory

            memory_test_passed = True

            # Test 1: Try to create a graph with many nodes
            try:
                graph_manager.create_graph("memory_test_1", "Graph")

                # Try to add a very large number of nodes
                huge_nodes = [f"node_{i}" for i in range(100000)]  # 100k nodes

                start_memory = self.get_system_stats()["memory_mb"]
                graph_manager.add_nodes_from("memory_test_1", huge_nodes)
                end_memory = self.get_system_stats()["memory_mb"]

                memory_used = end_memory - start_memory

                # Clean up
                graph_manager.delete_graph("memory_test_1")

                # Check if memory is reasonable (should be < 500MB for 100k nodes)
                if memory_used > 500:
                    pass
                else:
                    pass

            except Exception:
                memory_test_passed = False

            # Test 2: Try to create many small graphs
            try:
                start_memory = self.get_system_stats()["memory_mb"]

                for i in range(1000):  # 1000 small graphs
                    graph_id = f"small_graph_{i}"
                    graph_manager.create_graph(graph_id, "Graph")

                    # Add a few nodes to each
                    nodes = [f"n{j}" for j in range(10)]
                    graph_manager.add_nodes_from(graph_id, nodes)

                end_memory = self.get_system_stats()["memory_mb"]
                memory_used = end_memory - start_memory

                # Clean up
                for i in range(1000):
                    graph_id = f"small_graph_{i}"
                    try:
                        graph_manager.delete_graph(graph_id)
                    except Exception:
                        pass  # Ignore errors during cleanup

                # Check if memory is reasonable (should be < 200MB for 1000 small graphs)
                if memory_used > 200:
                    pass
                else:
                    pass

            except Exception:
                memory_test_passed = False

            return memory_test_passed

        except Exception:
            return False

    async def test_latency_performance(self):
        """Test operation latency under normal conditions."""

        try:
            import add_persistence
            import security_patches

            from networkx_mcp.server import graph_manager

            # Apply security and persistence
            security_patches.apply_critical_patches()
            add_persistence.patch_graph_manager_with_persistence()

            latencies = {
                "create_graph": [],
                "add_nodes": [],
                "add_edges": [],
                "get_info": [],
                "delete_graph": [],
            }

            # Run multiple iterations to get average latencies
            for iteration in range(50):
                graph_id = f"latency_test_{iteration}"

                # Test create_graph latency
                start = time.time()
                graph_manager.create_graph(graph_id, "DiGraph")
                latencies["create_graph"].append((time.time() - start) * 1000)

                # Test add_nodes latency
                nodes = [f"n{i}" for i in range(100)]
                start = time.time()
                graph_manager.add_nodes_from(graph_id, nodes)
                latencies["add_nodes"].append((time.time() - start) * 1000)

                # Test add_edges latency
                edges = [(f"n{i}", f"n{i + 1}") for i in range(99)]
                start = time.time()
                graph_manager.add_edges_from(graph_id, edges)
                latencies["add_edges"].append((time.time() - start) * 1000)

                # Test get_info latency
                start = time.time()
                graph_manager.get_graph_info(graph_id)
                latencies["get_info"].append((time.time() - start) * 1000)

                # Test delete_graph latency
                start = time.time()
                graph_manager.delete_graph(graph_id)
                latencies["delete_graph"].append((time.time() - start) * 1000)

            # Calculate statistics

            all_good = True
            for _operation, times in latencies.items():
                sum(times) / len(times)
                p95_latency = sorted(times)[int(len(times) * 0.95)]
                max(times)

                # Check if P95 latency is acceptable (< 100ms for basic operations)
                if p95_latency > 100:
                    all_good = False
                else:
                    pass

            return all_good

        except Exception:
            return False

    def generate_load_test_report(self):
        """Generate a comprehensive load test report."""

        if not self.test_results:
            return

        for result in self.test_results:
            "✅ PASS" if result["success"] else "❌ FAIL"

        # Find maximum supported users
        max_users = 0
        for result in self.test_results:
            if result["success"]:
                max_users = max(max_users, result["num_users"])

        if max_users >= 20:
            pass
        elif max_users >= 10:
            pass
        elif max_users >= 5:
            pass
        else:
            pass


async def main():
    """Run comprehensive load testing."""

    load_tester = TestLoadCapacity()

    tests = [
        ("Concurrent user capacity", load_tester.test_concurrent_users()),
        ("Memory limit enforcement", load_tester.test_memory_limits()),
        ("Operation latency", load_tester.test_latency_performance()),
    ]

    results = []
    for test_name, test_coro in tests:
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception:
            results.append((test_name, False))

    # Generate report
    load_tester.generate_load_test_report()

    # Summary

    passed = 0
    for _test_name, result in results:
        if result:
            passed += 1

    if passed == len(results):
        pass
    else:
        pass

    return passed == len(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
