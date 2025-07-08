#!/usr/bin/env python3
"""Comprehensive concurrency stress test for NetworkX MCP server."""

import asyncio
import random
import time
import statistics
from typing import List, Dict, Any
import logging

from networkx_mcp.core.thread_safe_graph_manager import ThreadSafeGraphManager
from networkx_mcp.concurrency import ConnectionPool, RequestQueue, RequestPriority

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConcurrencyStressTester:
    """Stress test for concurrent graph operations."""
    
    def __init__(self, num_clients: int = 100, num_graphs: int = 10):
        """Initialize stress tester.
        
        Args:
            num_clients: Number of concurrent clients
            num_graphs: Number of graphs to operate on
        """
        self.num_clients = num_clients
        self.num_graphs = num_graphs
        self.graph_manager = ThreadSafeGraphManager(max_graphs=num_graphs * 2)
        self.connection_pool = ConnectionPool(max_connections=50)
        self.request_queue = RequestQueue(max_queue_size=1000, max_workers=10)
        self.results = []
        self.errors = []
        
    async def setup(self):
        """Set up test environment."""
        # Start request queue
        await self.request_queue.start(self._process_request)
        
        # Create initial graphs
        for i in range(self.num_graphs):
            result = await self.graph_manager.create_graph(
                f"graph_{i}",
                "undirected" if i % 2 == 0 else "directed"
            )
            if not result["success"]:
                raise Exception(f"Failed to create graph: {result}")
                
        logger.info(f"Created {self.num_graphs} test graphs")
        
    async def teardown(self):
        """Clean up test environment."""
        await self.request_queue.stop()
        await self.graph_manager.cleanup()
        
    async def _process_request(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single request."""
        op_type = operation["type"]
        params = operation["params"]
        
        if op_type == "add_nodes":
            return await self.graph_manager.add_nodes(**params)
        elif op_type == "add_edges":
            return await self.graph_manager.add_edges(**params)
        elif op_type == "shortest_path":
            return await self.graph_manager.get_shortest_path(**params)
        elif op_type == "centrality":
            return await self.graph_manager.centrality_measures(**params)
        elif op_type == "info":
            return await self.graph_manager.get_graph_info(**params)
        else:
            raise ValueError(f"Unknown operation: {op_type}")
            
    async def client_workload(self, client_id: int, duration: float = 10.0):
        """Simulate a client performing random operations.
        
        Args:
            client_id: Unique client identifier
            duration: How long to run operations
        """
        start_time = time.time()
        operations = 0
        client_errors = 0
        
        operation_types = [
            ("add_nodes", RequestPriority.NORMAL, self._generate_add_nodes),
            ("add_edges", RequestPriority.NORMAL, self._generate_add_edges),
            ("shortest_path", RequestPriority.HIGH, self._generate_shortest_path),
            ("centrality", RequestPriority.LOW, self._generate_centrality),
            ("info", RequestPriority.NORMAL, self._generate_info)
        ]
        
        while time.time() - start_time < duration:
            # Random operation
            op_type, priority, generator = random.choice(operation_types)
            
            try:
                # Acquire connection
                async with self.connection_pool.acquire_connection():
                    # Generate operation
                    operation = generator()
                    
                    # Submit to queue
                    op_start = time.time()
                    result = await self.request_queue.submit_request(
                        operation, priority
                    )
                    op_time = time.time() - op_start
                    
                    operations += 1
                    
                    # Record result
                    self.results.append({
                        "client_id": client_id,
                        "operation": op_type,
                        "time": op_time,
                        "success": result.get("success", False)
                    })
                    
                    if not result.get("success", False):
                        logger.warning(f"Client {client_id} operation failed: {result}")
                        client_errors += 1
                        
            except Exception as e:
                client_errors += 1
                self.errors.append({
                    "client_id": client_id,
                    "operation": op_type,
                    "error": str(e)
                })
                logger.error(f"Client {client_id} error: {e}")
                
            # Small delay between operations
            await asyncio.sleep(random.uniform(0.01, 0.1))
            
        logger.info(
            f"Client {client_id} completed: {operations} operations, "
            f"{client_errors} errors"
        )
        
    def _generate_add_nodes(self) -> Dict[str, Any]:
        """Generate add_nodes operation."""
        graph_id = f"graph_{random.randint(0, self.num_graphs - 1)}"
        num_nodes = random.randint(1, 10)
        nodes = [f"node_{random.randint(0, 999999)}" for _ in range(num_nodes)]
        
        return {
            "type": "add_nodes",
            "params": {
                "graph_name": graph_id,
                "nodes": nodes
            }
        }
        
    def _generate_add_edges(self) -> Dict[str, Any]:
        """Generate add_edges operation."""
        graph_id = f"graph_{random.randint(0, self.num_graphs - 1)}"
        num_edges = random.randint(1, 5)
        edges = []
        
        for _ in range(num_edges):
            n1 = f"node_{random.randint(0, 99)}"
            n2 = f"node_{random.randint(0, 99)}"
            edges.append((n1, n2))
            
        return {
            "type": "add_edges",
            "params": {
                "graph_name": graph_id,
                "edges": edges
            }
        }
        
    def _generate_shortest_path(self) -> Dict[str, Any]:
        """Generate shortest_path operation."""
        graph_id = f"graph_{random.randint(0, self.num_graphs - 1)}"
        source = f"node_{random.randint(0, 99)}"
        target = f"node_{random.randint(0, 99)}"
        
        return {
            "type": "shortest_path",
            "params": {
                "graph_name": graph_id,
                "source": source,
                "target": target
            }
        }
        
    def _generate_centrality(self) -> Dict[str, Any]:
        """Generate centrality operation."""
        graph_id = f"graph_{random.randint(0, self.num_graphs - 1)}"
        measures = random.sample(["degree", "betweenness", "closeness"], k=2)
        
        return {
            "type": "centrality",
            "params": {
                "graph_name": graph_id,
                "measures": measures
            }
        }
        
    def _generate_info(self) -> Dict[str, Any]:
        """Generate info operation."""
        graph_id = f"graph_{random.randint(0, self.num_graphs - 1)}"
        
        return {
            "type": "info",
            "params": {
                "graph_name": graph_id
            }
        }
        
    async def run_stress_test(self, duration: float = 30.0):
        """Run the stress test.
        
        Args:
            duration: How long to run the test
        """
        logger.info(f"Starting stress test: {self.num_clients} clients, {duration}s duration")
        
        # Create client tasks
        tasks = []
        for i in range(self.num_clients):
            task = asyncio.create_task(self.client_workload(i, duration))
            tasks.append(task)
            
        # Wait for all clients to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        self._analyze_results()
        
    def _analyze_results(self):
        """Analyze test results."""
        print("\n" + "=" * 80)
        print("STRESS TEST RESULTS")
        print("=" * 80)
        
        # Overall statistics
        total_ops = len(self.results)
        successful_ops = sum(1 for r in self.results if r["success"])
        failed_ops = total_ops - successful_ops
        total_errors = len(self.errors)
        
        print(f"\nOverall Statistics:")
        print(f"  Total operations: {total_ops}")
        print(f"  Successful: {successful_ops} ({successful_ops/max(total_ops, 1)*100:.1f}%)")
        print(f"  Failed: {failed_ops} ({failed_ops/max(total_ops, 1)*100:.1f}%)")
        print(f"  Errors: {total_errors}")
        
        # Operation times
        if self.results:
            times = [r["time"] for r in self.results]
            print(f"\nOperation Times:")
            print(f"  Min: {min(times):.3f}s")
            print(f"  Max: {max(times):.3f}s")
            print(f"  Mean: {statistics.mean(times):.3f}s")
            print(f"  Median: {statistics.median(times):.3f}s")
            if len(times) > 1:
                print(f"  Std Dev: {statistics.stdev(times):.3f}s")
                
        # Per-operation breakdown
        op_stats = {}
        for result in self.results:
            op = result["operation"]
            if op not in op_stats:
                op_stats[op] = {"count": 0, "success": 0, "times": []}
            op_stats[op]["count"] += 1
            if result["success"]:
                op_stats[op]["success"] += 1
            op_stats[op]["times"].append(result["time"])
            
        print(f"\nPer-Operation Statistics:")
        for op, stats in op_stats.items():
            success_rate = stats["success"] / stats["count"] * 100
            avg_time = statistics.mean(stats["times"])
            print(f"  {op}:")
            print(f"    Count: {stats['count']}")
            print(f"    Success rate: {success_rate:.1f}%")
            print(f"    Avg time: {avg_time:.3f}s")
            
        # Connection pool stats
        pool_stats = self.connection_pool.get_stats()
        print(f"\nConnection Pool Statistics:")
        print(f"  Max connections: {pool_stats['max_connections']}")
        print(f"  Total connections: {pool_stats['total_connections']}")
        print(f"  Rejected: {pool_stats['rejected_connections']}")
        print(f"  Max concurrent: {pool_stats['max_concurrent']}")
        print(f"  Avg wait time: {pool_stats['avg_wait_time']:.3f}s")
        
        # Queue stats
        queue_stats = self.request_queue.get_queue_stats()
        print(f"\nRequest Queue Statistics:")
        print(f"  Workers: {queue_stats['workers']}")
        print(f"  Total pending: {queue_stats['total_pending']}")
        for priority, stats in queue_stats['queues'].items():
            print(f"  {priority}: {stats['size']}/{stats['maxsize']}")
            
        # Lock stats
        lock_stats = self.graph_manager.get_lock_stats()
        print(f"\nLock Manager Statistics:")
        print(f"  Total graphs: {lock_stats['total_graphs']}")
        print(f"  Total acquisitions: {lock_stats['total_acquisitions']}")
        print(f"  Contentions: {lock_stats['total_contentions']}")
        print(f"  Contention rate: {lock_stats['contention_rate']*100:.1f}%")
        print(f"  Avg wait time: {lock_stats['avg_wait_time']:.3f}s")
        
        # Error analysis
        if self.errors:
            print(f"\nError Analysis:")
            error_types = {}
            for error in self.errors:
                error_msg = error["error"]
                error_type = error_msg.split(":")[0]
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
            for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
                print(f"  {error_type}: {count}")
                
        print("\n" + "=" * 80)
        
    async def verify_data_integrity(self):
        """Verify no data corruption occurred."""
        print("\nVerifying Data Integrity...")
        
        issues = []
        
        # Check each graph
        for i in range(self.num_graphs):
            graph_name = f"graph_{i}"
            info = await self.graph_manager.get_graph_info(graph_name)
            
            if not info["success"]:
                issues.append(f"Graph {graph_name} missing or corrupted")
                continue
                
            # Basic sanity checks
            if info["nodes"] < 0 or info["edges"] < 0:
                issues.append(f"Graph {graph_name} has negative counts")
                
            if info["density"] < 0 or info["density"] > 1:
                issues.append(f"Graph {graph_name} has invalid density")
                
        if issues:
            print("❌ Data integrity issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ All graphs passed integrity checks")
            
        return len(issues) == 0


async def main():
    """Run the concurrency stress test."""
    print("🧪 NetworkX MCP Server Concurrency Stress Test\n")
    
    # Test configurations
    test_configs = [
        # (clients, graphs, duration, description)
        (10, 5, 5.0, "Light load"),
        (50, 10, 10.0, "Medium load"),
        (100, 10, 30.0, "Heavy load"),
    ]
    
    for num_clients, num_graphs, duration, description in test_configs:
        print(f"\n{'='*60}")
        print(f"Test: {description}")
        print(f"Clients: {num_clients}, Graphs: {num_graphs}, Duration: {duration}s")
        print(f"{'='*60}")
        
        tester = ConcurrencyStressTester(num_clients, num_graphs)
        
        try:
            await tester.setup()
            
            start_time = time.time()
            await tester.run_stress_test(duration)
            elapsed = time.time() - start_time
            
            print(f"\nTest completed in {elapsed:.1f}s")
            
            # Verify data integrity
            integrity_ok = await tester.verify_data_integrity()
            
            if integrity_ok:
                print(f"\n✅ Test '{description}' PASSED")
            else:
                print(f"\n❌ Test '{description}' FAILED - Data corruption detected")
                
        except Exception as e:
            print(f"\n❌ Test '{description}' FAILED with error: {e}")
            
        finally:
            await tester.teardown()
            
    print("\n✨ All stress tests completed!")


if __name__ == "__main__":
    asyncio.run(main())