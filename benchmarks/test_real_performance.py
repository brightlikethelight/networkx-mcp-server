#!/usr/bin/env python3
"""Real performance benchmarks for NetworkX MCP Server.

No fabricated results - actual testing with memory profiling and timing.
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

import psutil


class RealBenchmark:
    """Honest performance benchmarking for NetworkX MCP Server."""

    def __init__(self):
        self.results = {
            "node_limits": {},
            "edge_limits": {},
            "operation_times": {},
            "memory_usage": {},
            "algorithm_performance": {},
            "errors": [],
            "test_environment": self._get_environment(),
        }
        self.server_process = None

    def _get_environment(self) -> Dict[str, Any]:
        """Get test environment details."""
        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_count": os.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "networkx_version": self._get_networkx_version(),
        }

    def _get_networkx_version(self) -> str:
        """Get NetworkX version."""
        try:
            import networkx as nx

            return nx.__version__
        except ImportError:
            return "unknown"

    async def start_server(self) -> subprocess.Popen:
        """Start the MCP server."""
        self.server_process = subprocess.Popen(
            [sys.executable, "-m", "networkx_mcp.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Start reader thread
        self.reader_thread = threading.Thread(target=self._read_responses)
        self.reader_thread.daemon = True
        self.reader_thread.start()

        # Initialize server
        await self._initialize_server()
        return self.server_process

    def _read_responses(self):
        """Read server responses in background."""
        self.responses = []
        while self.server_process and self.server_process.poll() is None:
            try:
                line = self.server_process.stdout.readline()
                if line:
                    try:
                        response = json.loads(line.strip())
                        self.responses.append(response)
                    except json.JSONDecodeError:
                        pass
            except:
                break

    async def _initialize_server(self):
        """Initialize the MCP server."""
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "benchmark", "version": "1.0.0"},
            },
        }

        self.server_process.stdin.write(json.dumps(init_request) + "\n")
        self.server_process.stdin.flush()

        # Send initialized notification
        initialized = {"jsonrpc": "2.0", "method": "initialized"}
        self.server_process.stdin.write(json.dumps(initialized) + "\n")
        self.server_process.stdin.flush()

        # Wait for initialization
        await asyncio.sleep(0.5)

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any], request_id: int
    ) -> Dict[str, Any]:
        """Call a tool and measure response time."""
        start_time = time.time()

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        # Clear previous responses
        self.responses.clear()

        # Send request
        self.server_process.stdin.write(json.dumps(request) + "\n")
        self.server_process.stdin.flush()

        # Wait for response
        timeout = 30  # Generous timeout for large operations
        while time.time() - start_time < timeout:
            for response in self.responses:
                if response.get("id") == request_id:
                    response_time = time.time() - start_time
                    return {"response": response, "time_ms": response_time * 1000}
            await asyncio.sleep(0.01)

        raise TimeoutError(f"No response for {tool_name} after {timeout}s")

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.server_process:
            try:
                process = psutil.Process(self.server_process.pid)
                return process.memory_info().rss / (1024 * 1024)
            except:
                return 0.0
        return 0.0

    async def test_node_scaling(self):
        """Test node capacity scaling."""
        print("Testing node scaling limits...")

        await self.start_server()
        baseline_memory = self.get_memory_usage()

        # Test node counts
        node_counts = [100, 500, 1000, 2500, 5000, 10000]

        for count in node_counts:
            try:
                print(f"  Testing {count} nodes...")

                # Create graph
                result = await self.call_tool(
                    "create_graph", {"graph_id": f"test_{count}"}, count * 10
                )
                if "error" in result["response"]:
                    self.results["errors"].append(
                        f"Failed to create graph for {count} nodes: {result['response']['error']}"
                    )
                    break

                # Add nodes in batches to avoid timeout
                batch_size = min(100, count)
                start_time = time.time()
                start_memory = self.get_memory_usage()

                for i in range(0, count, batch_size):
                    batch_end = min(i + batch_size, count)
                    nodes = [f"node_{j}" for j in range(i, batch_end)]

                    result = await self.call_tool(
                        "add_nodes",
                        {"graph_id": f"test_{count}", "nodes": nodes},
                        count * 10 + i,
                    )

                    if "error" in result["response"]:
                        self.results["errors"].append(
                            f"Failed to add nodes {i}-{batch_end}: {result['response']['error']}"
                        )
                        break

                end_time = time.time()
                end_memory = self.get_memory_usage()

                self.results["node_limits"][count] = {
                    "success": True,
                    "total_time_s": end_time - start_time,
                    "memory_mb": end_memory - baseline_memory,
                    "memory_per_node_kb": (end_memory - start_memory) * 1024 / count
                    if count > 0
                    else 0,
                }

                print(
                    f"    ✓ {count} nodes: {end_time - start_time:.2f}s, {end_memory - start_memory:.1f}MB"
                )

            except Exception as e:
                self.results["node_limits"][count] = {"success": False, "error": str(e)}
                self.results["errors"].append(
                    f"Node scaling test failed at {count}: {e}"
                )
                print(f"    ✗ {count} nodes: {e}")
                break

        self.server_process.terminate()
        self.server_process.wait()

    async def test_edge_scaling(self):
        """Test edge capacity scaling."""
        print("Testing edge scaling limits...")

        await self.start_server()
        baseline_memory = self.get_memory_usage()

        # Create base graph with enough nodes
        await self.call_tool("create_graph", {"graph_id": "edge_test"}, 1)

        # Add 1000 nodes first
        nodes = [f"node_{i}" for i in range(1000)]
        await self.call_tool("add_nodes", {"graph_id": "edge_test", "nodes": nodes}, 2)

        # Test edge counts
        edge_counts = [100, 500, 1000, 2500, 5000, 10000]

        for count in edge_counts:
            try:
                print(f"  Testing {count} edges...")

                # Generate random edges
                import random

                edges = []
                for _ in range(count):
                    source = f"node_{random.randint(0, 999)}"
                    target = f"node_{random.randint(0, 999)}"
                    edges.append([source, target])

                start_time = time.time()
                start_memory = self.get_memory_usage()

                # Add edges in batches
                batch_size = min(100, count)
                for i in range(0, count, batch_size):
                    batch_edges = edges[i : i + batch_size]

                    result = await self.call_tool(
                        "add_edges",
                        {"graph_id": "edge_test", "edges": batch_edges},
                        1000 + i,
                    )

                    if "error" in result["response"]:
                        self.results["errors"].append(
                            f"Failed to add edges {i}-{i + len(batch_edges)}: {result['response']['error']}"
                        )
                        break

                end_time = time.time()
                end_memory = self.get_memory_usage()

                self.results["edge_limits"][count] = {
                    "success": True,
                    "total_time_s": end_time - start_time,
                    "memory_mb": end_memory - baseline_memory,
                    "memory_per_edge_bytes": (end_memory - start_memory)
                    * 1024
                    * 1024
                    / count
                    if count > 0
                    else 0,
                }

                print(
                    f"    ✓ {count} edges: {end_time - start_time:.2f}s, {end_memory - start_memory:.1f}MB"
                )

            except Exception as e:
                self.results["edge_limits"][count] = {"success": False, "error": str(e)}
                self.results["errors"].append(
                    f"Edge scaling test failed at {count}: {e}"
                )
                print(f"    ✗ {count} edges: {e}")
                break

        self.server_process.terminate()
        self.server_process.wait()

    async def test_algorithm_performance(self):
        """Test performance of graph algorithms."""
        print("Testing algorithm performance...")

        await self.start_server()

        # Create test graph
        await self.call_tool("create_graph", {"graph_id": "algo_test"}, 1)

        # Add nodes and edges for a connected graph
        nodes = [f"node_{i}" for i in range(100)]
        await self.call_tool("add_nodes", {"graph_id": "algo_test", "nodes": nodes}, 2)

        # Create a connected graph (ring topology)
        edges = [[f"node_{i}", f"node_{(i + 1) % 100}"] for i in range(100)]
        await self.call_tool("add_edges", {"graph_id": "algo_test", "edges": edges}, 3)

        # Test shortest path
        try:
            result = await self.call_tool(
                "shortest_path",
                {"graph_id": "algo_test", "source": "node_0", "target": "node_50"},
                4,
            )

            if "error" not in result["response"]:
                self.results["algorithm_performance"]["shortest_path"] = {
                    "time_ms": result["time_ms"],
                    "success": True,
                }
                print(f"    ✓ Shortest path: {result['time_ms']:.1f}ms")
            else:
                self.results["algorithm_performance"]["shortest_path"] = {
                    "success": False,
                    "error": result["response"]["error"],
                }
                print(f"    ✗ Shortest path failed: {result['response']['error']}")
        except Exception as e:
            self.results["algorithm_performance"]["shortest_path"] = {
                "success": False,
                "error": str(e),
            }
            print(f"    ✗ Shortest path exception: {e}")

        # Test centrality measures
        try:
            result = await self.call_tool(
                "centrality_measures",
                {"graph_id": "algo_test", "measures": ["degree", "betweenness"]},
                5,
            )

            if "error" not in result["response"]:
                self.results["algorithm_performance"]["centrality"] = {
                    "time_ms": result["time_ms"],
                    "success": True,
                }
                print(f"    ✓ Centrality measures: {result['time_ms']:.1f}ms")
            else:
                self.results["algorithm_performance"]["centrality"] = {
                    "success": False,
                    "error": result["response"]["error"],
                }
                print(f"    ✗ Centrality failed: {result['response']['error']}")
        except Exception as e:
            self.results["algorithm_performance"]["centrality"] = {
                "success": False,
                "error": str(e),
            }
            print(f"    ✗ Centrality exception: {e}")

        self.server_process.terminate()
        self.server_process.wait()

    async def test_operation_timing(self):
        """Test basic operation timing."""
        print("Testing basic operation timing...")

        await self.start_server()

        operations = [
            ("create_graph", {"graph_id": "timing_test"}),
            ("add_nodes", {"graph_id": "timing_test", "nodes": ["n1", "n2", "n3"]}),
            (
                "add_edges",
                {"graph_id": "timing_test", "edges": [["n1", "n2"], ["n2", "n3"]]},
            ),
            ("get_graph_info", {"graph_id": "timing_test"}),
        ]

        for i, (op_name, args) in enumerate(operations):
            try:
                result = await self.call_tool(op_name, args, 100 + i)

                if "error" not in result["response"]:
                    self.results["operation_times"][op_name] = {
                        "time_ms": result["time_ms"],
                        "success": True,
                    }
                    print(f"    ✓ {op_name}: {result['time_ms']:.1f}ms")
                else:
                    self.results["operation_times"][op_name] = {
                        "success": False,
                        "error": result["response"]["error"],
                    }
                    print(f"    ✗ {op_name} failed: {result['response']['error']}")

            except Exception as e:
                self.results["operation_times"][op_name] = {
                    "success": False,
                    "error": str(e),
                }
                print(f"    ✗ {op_name} exception: {e}")

        self.server_process.terminate()
        self.server_process.wait()

    def generate_report(self) -> str:
        """Generate honest performance report."""
        report = []
        report.append("# Real Performance Report - NetworkX MCP Server v0.1.0")
        report.append("")
        report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Testing Environment**: {self.results['test_environment']}")
        report.append(
            "**Testing Method**: Real subprocess communication with memory profiling"
        )
        report.append("")

        # Node scaling results
        report.append("## Node Scaling Results")
        report.append("")
        report.append("| Nodes | Time (s) | Memory (MB) | Memory/Node (KB) | Status |")
        report.append("|-------|----------|-------------|------------------|---------|")

        for count, data in self.results["node_limits"].items():
            if data["success"]:
                report.append(
                    f"| {count:,} | {data['total_time_s']:.2f} | {data['memory_mb']:.1f} | {data['memory_per_node_kb']:.1f} | ✓ |"
                )
            else:
                report.append(
                    f"| {count:,} | - | - | - | ✗ {data.get('error', 'Failed')} |"
                )

        report.append("")

        # Edge scaling results
        report.append("## Edge Scaling Results")
        report.append("")
        report.append(
            "| Edges | Time (s) | Memory (MB) | Memory/Edge (bytes) | Status |"
        )
        report.append(
            "|-------|----------|-------------|---------------------|---------|"
        )

        for count, data in self.results["edge_limits"].items():
            if data["success"]:
                report.append(
                    f"| {count:,} | {data['total_time_s']:.2f} | {data['memory_mb']:.1f} | {data['memory_per_edge_bytes']:.0f} | ✓ |"
                )
            else:
                report.append(
                    f"| {count:,} | - | - | - | ✗ {data.get('error', 'Failed')} |"
                )

        report.append("")

        # Operation timing
        report.append("## Basic Operation Performance")
        report.append("")
        report.append("| Operation | Time (ms) | Status |")
        report.append("|-----------|-----------|---------|")

        for op_name, data in self.results["operation_times"].items():
            if data["success"]:
                report.append(f"| {op_name} | {data['time_ms']:.1f} | ✓ |")
            else:
                report.append(f"| {op_name} | - | ✗ {data.get('error', 'Failed')} |")

        report.append("")

        # Algorithm performance
        report.append("## Algorithm Performance")
        report.append("")
        report.append("| Algorithm | Time (ms) | Status |")
        report.append("|-----------|-----------|---------|")

        for algo_name, data in self.results["algorithm_performance"].items():
            if data["success"]:
                report.append(f"| {algo_name} | {data['time_ms']:.1f} | ✓ |")
            else:
                report.append(f"| {algo_name} | - | ✗ {data.get('error', 'Failed')} |")

        report.append("")

        # Errors
        if self.results["errors"]:
            report.append("## Errors Encountered")
            report.append("")
            for error in self.results["errors"]:
                report.append(f"- {error}")
            report.append("")

        # Conclusions
        report.append("## Conclusions")
        report.append("")

        # Analyze results
        max_nodes = max(
            [k for k, v in self.results["node_limits"].items() if v["success"]],
            default=0,
        )
        max_edges = max(
            [k for k, v in self.results["edge_limits"].items() if v["success"]],
            default=0,
        )

        report.append(f"- **Maximum tested nodes**: {max_nodes:,}")
        report.append(f"- **Maximum tested edges**: {max_edges:,}")

        if max_nodes > 0:
            node_data = self.results["node_limits"][max_nodes]
            report.append(
                f"- **Memory per node**: ~{node_data['memory_per_node_kb']:.1f} KB"
            )

        if max_edges > 0:
            edge_data = self.results["edge_limits"][max_edges]
            report.append(
                f"- **Memory per edge**: ~{edge_data['memory_per_edge_bytes']:.0f} bytes"
            )

        if self.results["errors"]:
            report.append(f"- **Errors encountered**: {len(self.results['errors'])}")

        report.append("")
        report.append(
            "*This report contains actual measured performance data, not estimates.*"
        )

        return "\n".join(report)

    async def run_all_tests(self):
        """Run all performance tests."""
        print("=== Real Performance Benchmark ===")
        print(f"Environment: {self.results['test_environment']}")
        print()

        await self.test_operation_timing()
        await self.test_node_scaling()
        await self.test_edge_scaling()
        await self.test_algorithm_performance()

        print("\n=== Generating Report ===")
        report = self.generate_report()

        # Save report
        report_path = Path(__file__).parent / "REAL_PERFORMANCE_REPORT.md"
        with open(report_path, "w") as f:
            f.write(report)

        print(f"Report saved to: {report_path}")
        return report


async def main():
    """Run real performance benchmarks."""
    benchmark = RealBenchmark()
    try:
        await benchmark.run_all_tests()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        if benchmark.server_process:
            benchmark.server_process.terminate()
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback

        traceback.print_exc()
        if benchmark.server_process:
            benchmark.server_process.terminate()


if __name__ == "__main__":
    asyncio.run(main())
