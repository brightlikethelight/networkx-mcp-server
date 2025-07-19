"""Stress tests to find ACTUAL performance limits and characteristics."""

import asyncio
import json
import sys
import time
from pathlib import Path

import psutil
import pytest
import pytest_asyncio

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestStressLimits:
    """Find actual performance limits through systematic testing."""

    @pytest_asyncio.fixture
    async def mcp_server(self):
        """Start server and measure its baseline resource usage."""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "networkx_mcp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path(__file__).parent.parent,
            env={"PYTHONPATH": str(Path(__file__).parent.parent / "src")},
        )

        await asyncio.sleep(0.2)  # Let server fully start
        yield proc

        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    async def send_request(self, proc, request):
        """Send request with timeout handling."""
        json_str = json.dumps(request) + "\n"
        proc.stdin.write(json_str.encode())
        await proc.stdin.drain()

        response_line = await asyncio.wait_for(
            proc.stdout.readline(),
            timeout=10.0,  # Longer timeout for stress tests
        )

        return json.loads(response_line.decode().strip())

    async def initialize_server(self, proc):
        """Initialize server for testing."""
        init_response = await self.send_request(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
            },
        )

        # Send initialized notification
        json_str = (
            json.dumps({"jsonrpc": "2.0", "method": "initialized", "params": {}}) + "\n"
        )
        proc.stdin.write(json_str.encode())
        await proc.stdin.drain()
        await asyncio.sleep(0.05)

        return init_response

    async def call_tool(self, proc, tool_name, arguments, request_id):
        """Call tool with error handling."""
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        return await self.send_request(proc, request)

    def get_memory_usage(self):
        """Get current process memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    @pytest.mark.asyncio
    async def test_node_limits(self, mcp_server):
        """Find actual node limit through systematic testing."""
        await self.initialize_server(mcp_server)

        # Create test graph
        await self.call_tool(
            mcp_server, "create_graph", {"graph_id": "node_stress_test"}, 100
        )

        print("🔬 Testing node limits...")
        start_memory = self.get_memory_usage()

        # Test different batch sizes to find optimal approach
        batch_sizes = [10, 50, 100, 500, 1000]
        max_nodes = 0

        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            batch_count = 0

            try:
                # Add nodes in batches
                for batch in range(50):  # Try up to 50 batches
                    nodes = [f"n_{batch_size}_{batch}_{i}" for i in range(batch_size)]

                    start_time = time.time()
                    response = await self.call_tool(
                        mcp_server,
                        "add_nodes",
                        {"graph_id": "node_stress_test", "nodes": nodes},
                        101 + batch,
                    )
                    end_time = time.time()

                    # Check if operation succeeded
                    if "error" in response:
                        print(
                            f"    Failed at batch {batch}: {response['error']['message']}"
                        )
                        break

                    # Parse response
                    content = response["result"]["content"][0]["text"]
                    result_data = json.loads(content)

                    batch_count = batch + 1
                    current_nodes = batch_count * batch_size
                    response_time = (end_time - start_time) * 1000
                    current_memory = self.get_memory_usage()

                    # Stop if response time becomes too slow (>2 seconds)
                    if response_time > 2000:
                        print(
                            f"    Stopping at {current_nodes} nodes - response time too slow ({response_time:.1f}ms)"
                        )
                        break

                    # Stop if memory usage becomes excessive (>100MB growth)
                    memory_growth = current_memory - start_memory
                    if memory_growth > 100:
                        print(
                            f"    Stopping at {current_nodes} nodes - memory usage too high ({memory_growth:.1f}MB)"
                        )
                        break

                    if batch % 10 == 0:  # Progress update every 10 batches
                        print(
                            f"    Batch {batch}: {current_nodes} nodes, {response_time:.1f}ms, {memory_growth:.1f}MB"
                        )

                total_nodes = batch_count * batch_size
                max_nodes = max(max_nodes, total_nodes)
                print(f"  Batch size {batch_size}: Maximum {total_nodes} nodes")

            except Exception as e:
                print(f"  Batch size {batch_size} failed: {e}")
                continue

        print("📊 NODE LIMIT RESULTS:")
        print(f"  Maximum nodes achieved: {max_nodes}")
        print(f"  Memory usage: {self.get_memory_usage() - start_memory:.1f}MB")

        # Verify we can still operate on the graph
        try:
            info_response = await self.call_tool(
                mcp_server, "get_graph_info", {"graph_id": "node_stress_test"}, 200
            )

            if "result" in info_response:
                content = info_response["result"]["content"][0]["text"]
                info_data = json.loads(content)
                actual_nodes = info_data["num_nodes"]
                print(f"  Verified node count: {actual_nodes}")

        except Exception as e:
            print(f"  ⚠️  Graph info failed after stress test: {e}")

    @pytest.mark.asyncio
    async def test_edge_limits(self, mcp_server):
        """Find actual edge limit."""
        await self.initialize_server(mcp_server)

        # Create graph with moderate number of nodes
        await self.call_tool(
            mcp_server, "create_graph", {"graph_id": "edge_stress_test"}, 300
        )

        # Add nodes first (500 nodes for edge testing)
        nodes = [f"e_node_{i}" for i in range(500)]
        await self.call_tool(
            mcp_server,
            "add_nodes",
            {"graph_id": "edge_stress_test", "nodes": nodes},
            301,
        )

        print("🔬 Testing edge limits...")
        start_memory = self.get_memory_usage()

        max_edges = 0
        batch_size = 100

        try:
            for batch in range(200):  # Try up to 20k edges
                # Create random edges within the node set
                edges = []
                for i in range(batch_size):
                    src = f"e_node_{(batch * batch_size + i) % 500}"
                    dst = f"e_node_{(batch * batch_size + i + 1) % 500}"
                    edges.append([src, dst])

                start_time = time.time()
                response = await self.call_tool(
                    mcp_server,
                    "add_edges",
                    {"graph_id": "edge_stress_test", "edges": edges},
                    302 + batch,
                )
                end_time = time.time()

                if "error" in response:
                    print(f"Failed at batch {batch}: {response['error']['message']}")
                    break

                response_time = (end_time - start_time) * 1000
                current_edges = (batch + 1) * batch_size
                memory_growth = self.get_memory_usage() - start_memory

                if response_time > 3000 or memory_growth > 150:
                    print(f"Stopping at {current_edges} edges - performance degraded")
                    break

                max_edges = current_edges

                if batch % 20 == 0:
                    print(
                        f"  Batch {batch}: {current_edges} edges, {response_time:.1f}ms, {memory_growth:.1f}MB"
                    )

        except Exception as e:
            print(f"Edge test failed: {e}")

        print("📊 EDGE LIMIT RESULTS:")
        print(f"  Maximum edges achieved: {max_edges}")
        print(f"  Memory usage: {self.get_memory_usage() - start_memory:.1f}MB")

    @pytest.mark.asyncio
    async def test_algorithm_performance(self, mcp_server):
        """Test performance of graph algorithms on realistic graphs."""
        await self.initialize_server(mcp_server)

        # Create medium-sized graph for algorithm testing
        await self.call_tool(
            mcp_server, "create_graph", {"graph_id": "algorithm_test"}, 400
        )

        # Add 100 nodes
        nodes = [f"alg_node_{i}" for i in range(100)]
        await self.call_tool(
            mcp_server, "add_nodes", {"graph_id": "algorithm_test", "nodes": nodes}, 401
        )

        # Create a connected graph (path + some random connections)
        edges = []
        # Path edges
        for i in range(99):
            edges.append([f"alg_node_{i}", f"alg_node_{i + 1}"])
        # Random connections
        for i in range(0, 100, 10):
            for j in range(i + 5, min(i + 15, 100)):
                edges.append([f"alg_node_{i}", f"alg_node_{j}"])

        await self.call_tool(
            mcp_server, "add_edges", {"graph_id": "algorithm_test", "edges": edges}, 402
        )

        print("🔬 Testing algorithm performance...")

        # Test shortest path performance
        print("  Testing shortest path...")
        start_time = time.time()
        path_response = await self.call_tool(
            mcp_server,
            "shortest_path",
            {
                "graph_id": "algorithm_test",
                "source": "alg_node_0",
                "target": "alg_node_99",
            },
            403,
        )
        path_time = (time.time() - start_time) * 1000

        print(f"    Shortest path: {path_time:.1f}ms")

        # Test centrality calculation performance
        print("  Testing centrality measures...")
        start_time = time.time()
        centrality_response = await self.call_tool(
            mcp_server,
            "centrality_measures",
            {
                "graph_id": "algorithm_test",
                "measures": ["degree", "betweenness", "closeness"],
            },
            404,
        )
        centrality_time = (time.time() - start_time) * 1000

        print(f"    Centrality measures: {centrality_time:.1f}ms")

        print("📊 ALGORITHM PERFORMANCE RESULTS:")
        print(f"  Shortest path (100 nodes): {path_time:.1f}ms")
        print(f"  Centrality measures (100 nodes): {centrality_time:.1f}ms")

    @pytest.mark.asyncio
    async def test_memory_growth_pattern(self, mcp_server):
        """Test memory growth patterns with different operations."""
        await self.initialize_server(mcp_server)

        print("🔬 Testing memory growth patterns...")
        baseline_memory = self.get_memory_usage()

        # Test 1: Multiple small graphs
        print("  Testing multiple small graphs...")
        for i in range(50):
            await self.call_tool(
                mcp_server, "create_graph", {"graph_id": f"small_graph_{i}"}, 500 + i
            )

            # Add some nodes/edges to each
            nodes = [f"n_{j}" for j in range(10)]
            await self.call_tool(
                mcp_server,
                "add_nodes",
                {"graph_id": f"small_graph_{i}", "nodes": nodes},
                550 + i,
            )

        multi_graph_memory = self.get_memory_usage()
        print(f"    50 small graphs: {multi_graph_memory - baseline_memory:.1f}MB")

        # Test 2: Graph deletion and memory reclaim
        print("  Testing graph deletion...")
        for i in range(25):  # Delete half
            await self.call_tool(
                mcp_server, "delete_graph", {"graph_id": f"small_graph_{i}"}, 600 + i
            )

        after_deletion_memory = self.get_memory_usage()
        print(
            f"    After deleting 25 graphs: {after_deletion_memory - baseline_memory:.1f}MB"
        )

        print("📊 MEMORY PATTERN RESULTS:")
        print(f"  Baseline: {baseline_memory:.1f}MB")
        print(f"  50 small graphs: +{multi_graph_memory - baseline_memory:.1f}MB")
        print(f"  After 25 deletions: +{after_deletion_memory - baseline_memory:.1f}MB")
        print(f"  Memory reclaimed: {multi_graph_memory - after_deletion_memory:.1f}MB")

    @pytest.mark.asyncio
    async def test_response_time_degradation(self, mcp_server):
        """Test how response times degrade with graph size."""
        await self.initialize_server(mcp_server)

        await self.call_tool(
            mcp_server, "create_graph", {"graph_id": "response_time_test"}, 700
        )

        print("🔬 Testing response time degradation...")

        response_times = []
        node_counts = [10, 50, 100, 200, 500, 1000]

        for target_nodes in node_counts:
            try:
                # Add nodes to reach target count
                current_nodes = len(response_times) * 10 if response_times else 0
                new_nodes = [f"rt_node_{i}" for i in range(current_nodes, target_nodes)]

                if new_nodes:  # Only add if we have new nodes to add
                    start_time = time.time()
                    await self.call_tool(
                        mcp_server,
                        "add_nodes",
                        {"graph_id": "response_time_test", "nodes": new_nodes},
                        701 + len(response_times),
                    )
                    add_time = (time.time() - start_time) * 1000

                    # Test get_graph_info response time
                    start_time = time.time()
                    await self.call_tool(
                        mcp_server,
                        "get_graph_info",
                        {"graph_id": "response_time_test"},
                        750 + len(response_times),
                    )
                    info_time = (time.time() - start_time) * 1000

                    response_times.append((target_nodes, add_time, info_time))
                    print(
                        f"  {target_nodes} nodes: add={add_time:.1f}ms, info={info_time:.1f}ms"
                    )

            except Exception as e:
                print(f"  Failed at {target_nodes} nodes: {e}")
                break

        print("📊 RESPONSE TIME RESULTS:")
        for nodes, add_time, info_time in response_times:
            print(f"  {nodes:4d} nodes: add={add_time:6.1f}ms, info={info_time:6.1f}ms")


if __name__ == "__main__":
    # Run stress tests
    pytest.main([__file__, "-v", "-s"])
