#!/usr/bin/env python3
"""
BRUTAL REALITY CHECK: Test actual functionality vs claimed features
Tests the NetworkX MCP server comprehensively to identify what's actually working.
"""

import asyncio
import json
import os
import sys
import time
import traceback

import networkx as nx
import psutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from networkx_mcp.academic.citations import build_citation_network, resolve_doi
from networkx_mcp.core.basic_operations import (
    add_edges,
    add_nodes,
    betweenness_centrality,
    community_detection,
    connected_components,
    create_graph,
    degree_centrality,
    export_json,
    get_graph_info,
    import_csv,
    pagerank,
    shortest_path,
    visualize_graph,
)
from networkx_mcp.server import NetworkXMCPServer


class BrutalTester:
    """Comprehensive tester that doesn't hold back on finding issues."""

    def __init__(self):
        self.server = NetworkXMCPServer()
        self.test_results = {}
        self.errors = []
        self.warnings = []
        self.performance_data = {}

    def log_error(self, test_name: str, error: str):
        """Log an error with context."""
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ ERROR in {test_name}: {error}")

    def log_warning(self, test_name: str, warning: str):
        """Log a warning."""
        self.warnings.append(f"{test_name}: {warning}")
        print(f"⚠️  WARNING in {test_name}: {warning}")

    def log_success(self, test_name: str, message: str = ""):
        """Log a success."""
        print(f"✅ SUCCESS in {test_name}: {message}")

    def measure_performance(self, func_name: str, func, *args, **kwargs):
        """Measure performance of a function call."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            self.log_error(func_name, f"Function failed: {str(e)}")

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        self.performance_data[func_name] = {
            "duration": end_time - start_time,
            "memory_delta": end_memory - start_memory,
            "success": success,
        }

        return result

    def test_basic_tools(self):
        """Test all basic tools with various scenarios."""
        print("\n🔍 Testing Basic Tools...")

        # Test 1: Basic graph creation
        try:
            graphs = {}
            result = create_graph("test_graph", False, graphs)
            assert "test_graph" in graphs
            assert isinstance(graphs["test_graph"], nx.Graph)
            self.log_success("create_graph")
        except Exception as e:
            self.log_error("create_graph", str(e))

        # Test 2: Add nodes
        try:
            result = add_nodes("test_graph", [1, 2, 3, "A", "B"], graphs)
            assert graphs["test_graph"].number_of_nodes() == 5
            self.log_success("add_nodes")
        except Exception as e:
            self.log_error("add_nodes", str(e))

        # Test 3: Add edges
        try:
            result = add_edges("test_graph", [[1, 2], [2, 3], ["A", "B"]], graphs)
            assert graphs["test_graph"].number_of_edges() == 3
            self.log_success("add_edges")
        except Exception as e:
            self.log_error("add_edges", str(e))

        # Test 4: Graph info
        try:
            result = get_graph_info("test_graph", graphs)
            assert result["nodes"] == 5
            assert result["edges"] == 3
            assert not result["directed"]
            self.log_success("get_graph_info")
        except Exception as e:
            self.log_error("get_graph_info", str(e))

        # Test 5: Shortest path
        try:
            result = shortest_path("test_graph", 1, 3, graphs)
            assert result["path"] == [1, 2, 3]
            self.log_success("shortest_path")
        except Exception as e:
            self.log_error("shortest_path", str(e))

        # Test 6: Centrality measures
        try:
            result = degree_centrality("test_graph", graphs)
            assert "centrality" in result
            self.log_success("degree_centrality")
        except Exception as e:
            self.log_error("degree_centrality", str(e))

        try:
            result = betweenness_centrality("test_graph", graphs)
            assert "centrality" in result
            self.log_success("betweenness_centrality")
        except Exception as e:
            self.log_error("betweenness_centrality", str(e))

        # Test 7: Connected components
        try:
            result = connected_components("test_graph", graphs)
            assert "num_components" in result
            self.log_success("connected_components")
        except Exception as e:
            self.log_error("connected_components", str(e))

        # Test 8: PageRank
        try:
            result = pagerank("test_graph", graphs)
            assert "pagerank" in result
            self.log_success("pagerank")
        except Exception as e:
            self.log_error("pagerank", str(e))

        # Test 9: Community detection
        try:
            result = community_detection("test_graph", graphs)
            assert "num_communities" in result
            self.log_success("community_detection")
        except Exception as e:
            self.log_error("community_detection", str(e))

        # Test 10: Visualization
        try:
            result = visualize_graph("test_graph", "spring", graphs)
            assert "image" in result
            assert result["image"].startswith("data:image/png;base64,")
            self.log_success("visualize_graph")
        except Exception as e:
            self.log_error("visualize_graph", str(e))

        # Test 11: Export JSON
        try:
            result = export_json("test_graph", graphs)
            assert "graph_data" in result
            assert result["nodes"] == 5
            self.log_success("export_json")
        except Exception as e:
            self.log_error("export_json", str(e))

        # Test 12: Import CSV
        try:
            csv_data = "node1,node2\nnode2,node3\nnode3,node4"
            result = import_csv("csv_graph", csv_data, False, graphs)
            assert result["nodes"] == 4
            assert result["edges"] == 3
            self.log_success("import_csv")
        except Exception as e:
            self.log_error("import_csv", str(e))

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        print("\n🔍 Testing Edge Cases...")

        graphs = {}

        # Test nonexistent graph
        try:
            get_graph_info("nonexistent", graphs)
            self.log_error("nonexistent_graph", "Should have raised ValueError")
        except ValueError:
            self.log_success("nonexistent_graph", "Correctly raised ValueError")
        except Exception as e:
            self.log_error("nonexistent_graph", f"Wrong exception type: {type(e)}")

        # Test empty graph operations
        try:
            create_graph("empty", False, graphs)
            shortest_path("empty", 1, 2, graphs)
            self.log_error("empty_graph_path", "Should have raised exception")
        except Exception:
            self.log_success("empty_graph_path", "Correctly handled empty graph")

        # Test large node names
        try:
            create_graph("large_names", False, graphs)
            large_name = "x" * 1000
            add_nodes("large_names", [large_name], graphs)
            self.log_success("large_node_names")
        except Exception as e:
            self.log_error("large_node_names", str(e))

        # Test invalid CSV
        try:
            import_csv(
                "invalid_csv", "invalid,csv,data,too,many,columns", False, graphs
            )
            self.log_success("invalid_csv", "Handled gracefully")
        except Exception as e:
            self.log_warning(
                "invalid_csv", f"May not handle invalid CSV well: {str(e)}"
            )

        # Test self-loops
        try:
            create_graph("self_loop", False, graphs)
            add_nodes("self_loop", [1], graphs)
            add_edges("self_loop", [[1, 1]], graphs)
            self.log_success("self_loops")
        except Exception as e:
            self.log_error("self_loops", str(e))

    def test_performance_large_graphs(self):
        """Test performance with large graphs."""
        print("\n🔍 Testing Performance with Large Graphs...")

        graphs = {}

        # Test 1000 node graph
        try:
            start_time = time.time()
            create_graph("large_graph", False, graphs)

            # Add 1000 nodes
            nodes = list(range(1000))
            add_nodes("large_graph", nodes, graphs)

            # Add edges in a ring + random connections
            edges = []
            for i in range(1000):
                edges.append([i, (i + 1) % 1000])  # Ring
                if i % 10 == 0:  # Add some random connections
                    edges.append([i, (i + 100) % 1000])

            add_edges("large_graph", edges, graphs)

            # Test operations on large graph
            operations = [
                (
                    "large_graph_centrality",
                    lambda: degree_centrality("large_graph", graphs),
                ),
                (
                    "large_graph_components",
                    lambda: connected_components("large_graph", graphs),
                ),
                ("large_graph_pagerank", lambda: pagerank("large_graph", graphs)),
                ("large_graph_export", lambda: export_json("large_graph", graphs)),
            ]

            for name, op in operations:
                result = self.measure_performance(name, op)
                if result is not None:
                    self.log_success(
                        name, f"Took {self.performance_data[name]['duration']:.2f}s"
                    )
                    if self.performance_data[name]["duration"] > 30:
                        self.log_warning(
                            name,
                            f"Very slow: {self.performance_data[name]['duration']:.2f}s",
                        )

            total_time = time.time() - start_time
            self.log_success("large_graph_test", f"Total time: {total_time:.2f}s")

        except Exception as e:
            self.log_error("large_graph_test", str(e))

    def test_academic_features(self):
        """Test academic citation network features."""
        print("\n🔍 Testing Academic Features...")

        # Test DOI resolution
        try:
            # Use a known DOI
            doi = "10.1038/nature12373"
            result = resolve_doi(doi)
            if result:
                assert "doi" in result
                assert "title" in result
                self.log_success(
                    "resolve_doi", f"Resolved: {result.get('title', 'N/A')[:50]}..."
                )
            else:
                self.log_warning(
                    "resolve_doi", "Could not resolve DOI - may be network issue"
                )
        except Exception as e:
            self.log_error("resolve_doi", str(e))

        # Test citation network building
        try:
            graphs = {}
            # Use a well-known paper DOI
            seed_dois = ["10.1038/nature12373"]
            result = build_citation_network("citation_test", seed_dois, 1, graphs)

            if result and result.get("nodes", 0) > 0:
                self.log_success(
                    "build_citation_network",
                    f"Built network with {result['nodes']} nodes",
                )
            else:
                self.log_warning(
                    "build_citation_network", "Built empty network - may be API limits"
                )
        except Exception as e:
            self.log_error("build_citation_network", str(e))

    def test_memory_usage(self):
        """Test memory usage and potential leaks."""
        print("\n🔍 Testing Memory Usage...")

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Create and destroy many graphs
        graphs = {}
        for i in range(100):
            graph_name = f"temp_graph_{i}"
            create_graph(graph_name, False, graphs)
            add_nodes(graph_name, list(range(100)), graphs)
            add_edges(graph_name, [[j, (j + 1) % 100] for j in range(100)], graphs)

            # Delete every 10th graph
            if i % 10 == 0:
                del graphs[graph_name]

        # Force garbage collection
        import gc

        gc.collect()

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        self.log_success("memory_usage", f"Memory growth: {memory_growth:.2f}MB")
        if memory_growth > 100:
            self.log_warning(
                "memory_usage", f"High memory growth: {memory_growth:.2f}MB"
            )

    def test_server_protocol(self):
        """Test MCP protocol compliance."""
        print("\n🔍 Testing MCP Protocol...")

        server = NetworkXMCPServer()

        # Test initialize
        try:
            request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
            response = asyncio.run(server.handle_request(request))
            assert response["jsonrpc"] == "2.0"
            assert response["id"] == 1
            assert "result" in response
            self.log_success("mcp_initialize")
        except Exception as e:
            self.log_error("mcp_initialize", str(e))

        # Test tools/list
        try:
            request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            response = asyncio.run(server.handle_request(request))
            tools = response["result"]["tools"]
            assert len(tools) > 0
            self.log_success("mcp_tools_list", f"Found {len(tools)} tools")
        except Exception as e:
            self.log_error("mcp_tools_list", str(e))

        # Test tools/call
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "create_graph",
                    "arguments": {"name": "test_protocol", "directed": False},
                },
            }
            response = asyncio.run(server.handle_request(request))
            assert "result" in response
            self.log_success("mcp_tools_call")
        except Exception as e:
            self.log_error("mcp_tools_call", str(e))

        # Test invalid method
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "invalid_method",
                "params": {},
            }
            response = asyncio.run(server.handle_request(request))
            assert "error" in response
            self.log_success("mcp_invalid_method", "Correctly returned error")
        except Exception as e:
            self.log_error("mcp_invalid_method", str(e))

    def test_concurrent_operations(self):
        """Test concurrent operation handling."""
        print("\n🔍 Testing Concurrent Operations...")

        async def create_and_modify_graph(graph_name: str):
            """Create a graph and perform operations on it."""
            graphs = {}
            try:
                create_graph(graph_name, False, graphs)
                add_nodes(graph_name, list(range(50)), graphs)
                add_edges(graph_name, [[i, (i + 1) % 50] for i in range(50)], graphs)
                degree_centrality(graph_name, graphs)
                return True
            except Exception as e:
                self.log_error(f"concurrent_{graph_name}", str(e))
                return False

        async def run_concurrent_test():
            tasks = []
            for i in range(10):
                task = create_and_modify_graph(f"concurrent_graph_{i}")
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if r is True)
            return successful

        try:
            successful = asyncio.run(run_concurrent_test())
            self.log_success(
                "concurrent_operations", f"{successful}/10 operations successful"
            )
            if successful < 8:
                self.log_warning(
                    "concurrent_operations",
                    f"Only {successful}/10 operations successful",
                )
        except Exception as e:
            self.log_error("concurrent_operations", str(e))

    def test_visualization_robustness(self):
        """Test visualization with various graph types."""
        print("\n🔍 Testing Visualization Robustness...")

        graphs = {}

        # Test empty graph
        try:
            create_graph("empty_viz", False, graphs)
            result = visualize_graph("empty_viz", "spring", graphs)
            self.log_success("visualize_empty", "Handled empty graph")
        except Exception as e:
            self.log_error("visualize_empty", str(e))

        # Test single node
        try:
            create_graph("single_viz", False, graphs)
            add_nodes("single_viz", [1], graphs)
            result = visualize_graph("single_viz", "spring", graphs)
            self.log_success("visualize_single", "Handled single node")
        except Exception as e:
            self.log_error("visualize_single", str(e))

        # Test different layouts
        try:
            create_graph("layout_test", False, graphs)
            add_nodes("layout_test", list(range(10)), graphs)
            add_edges("layout_test", [[i, (i + 1) % 10] for i in range(10)], graphs)

            for layout in ["spring", "circular", "kamada_kawai"]:
                result = visualize_graph("layout_test", layout, graphs)
                if result and "image" in result:
                    self.log_success(f"visualize_{layout}", "Generated image")
                else:
                    self.log_error(f"visualize_{layout}", "Failed to generate image")
        except Exception as e:
            self.log_error("visualize_layouts", str(e))

    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n📊 BRUTAL REALITY CHECK REPORT")
        print("=" * 60)

        total_tests = (
            len(self.errors)
            + len(self.warnings)
            + len(
                [
                    k
                    for k in self.performance_data.keys()
                    if self.performance_data[k]["success"]
                ]
            )
        )
        failed_tests = len(self.errors)
        warning_tests = len(self.warnings)
        passed_tests = len(
            [
                k
                for k in self.performance_data.keys()
                if self.performance_data[k]["success"]
            ]
        )

        print("\n🎯 OVERALL RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ⚠️  Warnings: {warning_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   Success Rate: {(passed_tests / max(total_tests, 1)) * 100:.1f}%")

        if self.errors:
            print(f"\n❌ FAILURES ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")

        if self.performance_data:
            print("\n⏱️  PERFORMANCE DATA:")
            for func_name, data in self.performance_data.items():
                status = "✅" if data["success"] else "❌"
                print(
                    f"   {status} {func_name}: {data['duration']:.3f}s, {data['memory_delta']:.2f}MB"
                )

        print("\n🎭 HONEST ASSESSMENT:")
        if failed_tests == 0:
            print("   🎉 All tests passed! Server appears to be working correctly.")
        elif failed_tests < 3:
            print("   👍 Server is mostly working with minor issues.")
        elif failed_tests < 10:
            print("   ⚠️  Server has several issues but basic functionality works.")
        else:
            print(
                "   🚨 Server has significant issues and may not be production-ready."
            )

        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "warnings": warning_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests / max(total_tests, 1)) * 100,
            "errors": self.errors,
            "warnings": self.warnings,
            "performance_data": self.performance_data,
        }

    def run_all_tests(self):
        """Run all tests and generate report."""
        print("🔥 STARTING BRUTAL REALITY CHECK...")
        print("Testing NetworkX MCP Server comprehensively...")

        test_methods = [
            self.test_basic_tools,
            self.test_edge_cases,
            self.test_performance_large_graphs,
            self.test_academic_features,
            self.test_memory_usage,
            self.test_server_protocol,
            self.test_concurrent_operations,
            self.test_visualization_robustness,
        ]

        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.log_error(test_method.__name__, f"Test method crashed: {str(e)}")
                traceback.print_exc()

        return self.generate_report()


if __name__ == "__main__":
    tester = BrutalTester()
    report = tester.run_all_tests()

    # Save detailed report
    with open("brutal_reality_check_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n💾 Detailed report saved to: brutal_reality_check_report.json")
