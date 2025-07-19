#!/usr/bin/env python3
"""
Stress test and edge case testing for NetworkX MCP server.
This is where we find the real issues.
"""

import asyncio
import json
import os
import sys
import time
import traceback

import psutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from networkx_mcp.core.basic_operations import *
from networkx_mcp.server import NetworkXMCPServer


class StressAndEdgeTester:
    """Test edge cases and stress scenarios."""

    def __init__(self):
        self.server = NetworkXMCPServer()
        self.issues = []
        self.critical_issues = []
        self.warnings = []

    def log_issue(self, severity: str, test_name: str, description: str):
        """Log an issue with severity."""
        issue = f"{test_name}: {description}"
        if severity == "critical":
            self.critical_issues.append(issue)
            print(f"🚨 CRITICAL: {issue}")
        elif severity == "warning":
            self.warnings.append(issue)
            print(f"⚠️  WARNING: {issue}")
        else:
            self.issues.append(issue)
            print(f"❌ ISSUE: {issue}")

    def log_success(self, test_name: str, message: str = ""):
        """Log a success."""
        print(f"✅ {test_name}: {message}")

    def test_massive_graphs(self):
        """Test with truly massive graphs."""
        print("\n🔥 Testing Massive Graphs...")

        graphs = {}

        # Test 10,000 node graph
        try:
            start_time = time.time()
            create_graph("massive", False, graphs)

            # Add 10,000 nodes
            nodes = list(range(10000))
            add_nodes("massive", nodes, graphs)

            # Add many edges (dense graph)
            edges = []
            for i in range(0, 10000, 100):  # Connect every 100th node
                for j in range(i + 1, min(i + 100, 10000)):
                    edges.append([i, j])

            add_edges("massive", edges, graphs)

            creation_time = time.time() - start_time
            print(f"Created massive graph in {creation_time:.2f}s")

            # Test operations on massive graph
            start_time = time.time()
            centrality_result = degree_centrality("massive", graphs)
            centrality_time = time.time() - start_time

            if centrality_time > 60:  # More than 1 minute
                self.log_issue(
                    "critical",
                    "massive_graph_centrality",
                    f"Took {centrality_time:.2f}s - too slow for production",
                )
            elif centrality_time > 10:
                self.log_issue(
                    "warning",
                    "massive_graph_centrality",
                    f"Took {centrality_time:.2f}s - slow",
                )
            else:
                self.log_success(
                    "massive_graph_centrality", f"Took {centrality_time:.2f}s"
                )

        except MemoryError:
            self.log_issue(
                "critical",
                "massive_graph",
                "Out of memory - server crashes on large graphs",
            )
        except Exception as e:
            self.log_issue("critical", "massive_graph", f"Crashed: {str(e)}")

    def test_malformed_inputs(self):
        """Test malformed and malicious inputs."""
        print("\n🔥 Testing Malformed Inputs...")

        graphs = {}

        # Test extremely long node names
        try:
            create_graph("malformed", False, graphs)
            giant_name = "x" * 10000
            add_nodes("malformed", [giant_name], graphs)
            self.log_success("giant_node_name", "Handled gracefully")
        except Exception as e:
            self.log_issue("warning", "giant_node_name", f"Failed: {str(e)}")

        # Test invalid edge formats
        try:
            invalid_edges = [
                [1],  # Single element
                [1, 2, 3, 4, 5],  # Too many elements
                ["a", None],  # None value
                [{"key": "value"}, 2],  # Dict as node
            ]
            add_edges("malformed", invalid_edges, graphs)
            self.log_success("invalid_edges", "Handled gracefully")
        except Exception as e:
            self.log_issue(
                "warning",
                "invalid_edges",
                f"May not handle invalid edges well: {str(e)}",
            )

        # Test deeply nested structures
        try:
            nested_node = {"level1": {"level2": {"level3": {"level4": "deep"}}}}
            add_nodes("malformed", [nested_node], graphs)
            self.log_success("nested_structures", "Handled gracefully")
        except Exception as e:
            self.log_issue(
                "warning",
                "nested_structures",
                f"May not handle nested structures: {str(e)}",
            )

    def test_memory_leaks(self):
        """Test for memory leaks with repeated operations."""
        print("\n🔥 Testing Memory Leaks...")

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Create and destroy many graphs repeatedly
        for cycle in range(10):
            graphs = {}

            # Create 50 graphs per cycle
            for i in range(50):
                graph_name = f"cycle_{cycle}_graph_{i}"
                create_graph(graph_name, False, graphs)
                add_nodes(graph_name, list(range(100)), graphs)
                add_edges(graph_name, [[j, (j + 1) % 100] for j in range(100)], graphs)

                # Perform operations
                degree_centrality(graph_name, graphs)
                connected_components(graph_name, graphs)

                # Delete graph
                del graphs[graph_name]

            # Force garbage collection
            import gc

            gc.collect()

            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_growth = current_memory - initial_memory

            if memory_growth > 50:  # More than 50MB growth
                self.log_issue(
                    "critical",
                    "memory_leak",
                    f"Memory grew {memory_growth:.2f}MB after {cycle + 1} cycles",
                )
                break

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        total_growth = final_memory - initial_memory

        if total_growth > 100:
            self.log_issue(
                "critical", "memory_leak", f"Total memory growth: {total_growth:.2f}MB"
            )
        elif total_growth > 50:
            self.log_issue(
                "warning", "memory_leak", f"Total memory growth: {total_growth:.2f}MB"
            )
        else:
            self.log_success(
                "memory_leak", f"Total memory growth: {total_growth:.2f}MB"
            )

    def test_infinite_loops(self):
        """Test for potential infinite loops."""
        print("\n🔥 Testing Infinite Loop Prevention...")

        graphs = {}

        # Create graph with self-loops
        try:
            create_graph("self_loop", False, graphs)
            add_nodes("self_loop", [1, 2, 3], graphs)
            add_edges("self_loop", [[1, 1], [2, 2], [3, 3]], graphs)

            # Test operations that might loop
            start_time = time.time()
            result = shortest_path("self_loop", 1, 2, graphs)
            duration = time.time() - start_time

            if duration > 5:  # More than 5 seconds
                self.log_issue(
                    "critical",
                    "self_loop_path",
                    f"Took {duration:.2f}s - possible infinite loop",
                )
            else:
                self.log_success("self_loop_path", f"Handled in {duration:.2f}s")
        except Exception as e:
            self.log_success("self_loop_path", f"Correctly raised exception: {str(e)}")

    def test_concurrent_modifications(self):
        """Test concurrent modifications to same graph."""
        print("\n🔥 Testing Concurrent Modifications...")

        graphs = {}
        create_graph("concurrent", False, graphs)

        async def modify_graph(modifier_id: int):
            """Modify the graph concurrently."""
            try:
                for i in range(100):
                    node_id = f"{modifier_id}_{i}"
                    add_nodes("concurrent", [node_id], graphs)
                    if i > 0:
                        add_edges(
                            "concurrent", [[f"{modifier_id}_{i - 1}", node_id]], graphs
                        )
                return True
            except Exception as e:
                self.log_issue("warning", f"concurrent_modifier_{modifier_id}", str(e))
                return False

        async def run_concurrent_test():
            tasks = []
            for i in range(10):
                task = modify_graph(i)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if r is True)
            return successful

        try:
            successful = asyncio.run(run_concurrent_test())
            if successful < 5:
                self.log_issue(
                    "critical",
                    "concurrent_modifications",
                    f"Only {successful}/10 concurrent operations successful",
                )
            else:
                self.log_success(
                    "concurrent_modifications", f"{successful}/10 operations successful"
                )
        except Exception as e:
            self.log_issue("critical", "concurrent_modifications", f"Crashed: {str(e)}")

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        print("\n🔥 Testing Unicode and Special Characters...")

        graphs = {}

        # Test various unicode characters
        try:
            create_graph("unicode", False, graphs)
            unicode_nodes = [
                "普通话",  # Chinese
                "日本語",  # Japanese
                "العربية",  # Arabic
                "🚀💻📊",  # Emojis
                "node\nwith\nnewlines",  # Newlines
                "node\twith\ttabs",  # Tabs
                'node with "quotes"',  # Quotes
                "node with 'apostrophes'",  # Apostrophes
                "node/with/slashes",  # Slashes
                "node\\with\\backslashes",  # Backslashes
            ]

            add_nodes("unicode", unicode_nodes, graphs)

            # Test operations on unicode nodes
            degree_centrality("unicode", graphs)
            export_json("unicode", graphs)

            self.log_success(
                "unicode_handling", f"Handled {len(unicode_nodes)} unicode nodes"
            )
        except Exception as e:
            self.log_issue("warning", "unicode_handling", f"Failed: {str(e)}")

    def test_resource_exhaustion(self):
        """Test resource exhaustion scenarios."""
        print("\n🔥 Testing Resource Exhaustion...")

        # Test creating too many graphs
        graphs = {}
        try:
            for i in range(1000):
                create_graph(f"resource_test_{i}", False, graphs)

            self.log_success("many_graphs", f"Created {len(graphs)} graphs")
        except Exception as e:
            self.log_issue(
                "warning", "many_graphs", f"Failed at {len(graphs)} graphs: {str(e)}"
            )

        # Test very dense graph
        try:
            create_graph("dense", False, graphs)
            nodes = list(range(1000))
            add_nodes("dense", nodes, graphs)

            # Create complete graph (all nodes connected to all others)
            edges = []
            for i in range(1000):
                for j in range(i + 1, 1000):
                    edges.append([i, j])

            start_time = time.time()
            add_edges("dense", edges, graphs)
            duration = time.time() - start_time

            if duration > 30:
                self.log_issue(
                    "critical",
                    "dense_graph",
                    f"Took {duration:.2f}s to create dense graph",
                )
            else:
                self.log_success(
                    "dense_graph", f"Created dense graph in {duration:.2f}s"
                )
        except Exception as e:
            self.log_issue("critical", "dense_graph", f"Failed: {str(e)}")

    def test_api_failures(self):
        """Test handling of external API failures."""
        print("\n🔥 Testing API Failure Handling...")

        # Test with invalid DOI
        try:
            from networkx_mcp.academic.citations import resolve_doi

            result = resolve_doi("invalid_doi_123456789")
            if result is None:
                self.log_success("invalid_doi", "Correctly returned None")
            else:
                self.log_issue(
                    "warning", "invalid_doi", f"Returned unexpected result: {result}"
                )
        except Exception as e:
            self.log_issue("warning", "invalid_doi", f"Raised exception: {str(e)}")

        # Test with malformed DOI
        try:
            result = resolve_doi("not_a_doi_at_all")
            if result is None:
                self.log_success("malformed_doi", "Correctly returned None")
            else:
                self.log_issue(
                    "warning", "malformed_doi", "Returned unexpected result"
                )
        except Exception as e:
            self.log_issue("warning", "malformed_doi", f"Raised exception: {str(e)}")

    def generate_report(self):
        """Generate comprehensive stress test report."""
        print("\n🔥 STRESS TEST & EDGE CASE REPORT")
        print("=" * 60)

        total_issues = len(self.critical_issues) + len(self.issues) + len(self.warnings)

        print("\n📊 SUMMARY:")
        print(f"   🚨 Critical Issues: {len(self.critical_issues)}")
        print(f"   ❌ Issues: {len(self.issues)}")
        print(f"   ⚠️  Warnings: {len(self.warnings)}")
        print(f"   Total: {total_issues}")

        if self.critical_issues:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in self.critical_issues:
                print(f"   • {issue}")

        if self.issues:
            print("\n❌ ISSUES:")
            for issue in self.issues:
                print(f"   • {issue}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")

        print("\n🎯 PRODUCTION READINESS ASSESSMENT:")
        if len(self.critical_issues) > 0:
            print("   🚨 NOT PRODUCTION READY - Critical issues found")
        elif len(self.issues) > 3:
            print("   ⚠️  NEEDS WORK - Multiple issues found")
        elif len(self.warnings) > 5:
            print("   ⚠️  MOSTLY READY - Some warnings to address")
        else:
            print("   ✅ PRODUCTION READY - No critical issues")

        return {
            "critical_issues": self.critical_issues,
            "issues": self.issues,
            "warnings": self.warnings,
            "total_issues": total_issues,
        }

    def run_all_tests(self):
        """Run all stress tests."""
        print("🔥 STARTING STRESS & EDGE CASE TESTING...")

        test_methods = [
            self.test_massive_graphs,
            self.test_malformed_inputs,
            self.test_memory_leaks,
            self.test_infinite_loops,
            self.test_concurrent_modifications,
            self.test_unicode_and_special_characters,
            self.test_resource_exhaustion,
            self.test_api_failures,
        ]

        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.log_issue(
                    "critical", test_method.__name__, f"Test crashed: {str(e)}"
                )
                traceback.print_exc()

        return self.generate_report()


if __name__ == "__main__":
    tester = StressAndEdgeTester()
    report = tester.run_all_tests()

    # Save report
    with open("stress_test_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n💾 Detailed report saved to: stress_test_report.json")
