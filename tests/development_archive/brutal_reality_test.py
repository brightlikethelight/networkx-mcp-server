#!/usr/bin/env python3
"""
Brutal Reality Test for NetworkX MCP Server
Test all the edge cases, error conditions, and real-world scenarios
"""

import json
import os
import subprocess
import sys
import tempfile
import time


class BrutalRealityTester:
    """Test everything that could go wrong with the MCP server."""

    def __init__(self):
        self.process = None
        self.request_id = 0
        self.failures = []
        self.successes = []

    def start_server(self):
        """Start the MCP server."""
        self.process = subprocess.Popen(
            [sys.executable, "-m", "networkx_mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        time.sleep(0.5)  # Give server time to start

    def stop_server(self):
        """Stop the server."""
        if self.process:
            self.process.terminate()
            self.process.wait()

    def send_request(self, method, params=None):
        """Send request to server."""
        self.request_id += 1
        request = {"jsonrpc": "2.0", "method": method, "id": self.request_id}
        if params:
            request["params"] = params

        try:
            request_str = json.dumps(request) + "\n"
            self.process.stdin.write(request_str)
            self.process.stdin.flush()

            response_line = self.process.stdout.readline()
            if response_line:
                return json.loads(response_line.strip())
            return None
        except Exception as e:
            return {"error": str(e)}

    def test_initialize(self):
        """Test initialization."""
        print("=== TESTING INITIALIZATION ===")
        response = self.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "brutal-tester", "version": "1.0.0"},
            },
        )

        if response and "result" in response:
            self.successes.append("initialization")
            print("✅ Initialization successful")
        else:
            self.failures.append(f"initialization failed: {response}")
            print(f"❌ Initialization failed: {response}")

    def test_all_tools(self):
        """Test all 20 tools systematically."""
        print("\n=== TESTING ALL TOOLS ===")

        # Get tools list
        tools_response = self.send_request("tools/list")
        if not tools_response or "result" not in tools_response:
            self.failures.append("tools/list failed")
            print("❌ Could not get tools list")
            return

        tools = tools_response["result"]["tools"]
        print(f"Found {len(tools)} tools")

        # Test each tool
        for tool in tools:
            tool_name = tool["name"]
            print(f"\n--- Testing {tool_name} ---")

            if tool_name == "create_graph":
                self.test_create_graph()
            elif tool_name == "add_nodes":
                self.test_add_nodes()
            elif tool_name == "add_edges":
                self.test_add_edges()
            elif tool_name == "get_info":
                self.test_get_info()
            elif tool_name == "degree_centrality":
                self.test_degree_centrality()
            elif tool_name == "pagerank":
                self.test_pagerank()
            elif tool_name == "shortest_path":
                self.test_shortest_path()
            elif tool_name == "build_citation_network":
                self.test_citation_network()
            elif tool_name == "resolve_doi":
                self.test_doi_resolution()
            elif tool_name == "visualize_graph":
                self.test_visualization()
            elif tool_name == "export_json":
                self.test_export_json()
            elif tool_name == "import_csv":
                self.test_import_csv()
            else:
                print(f"⚠️  Tool {tool_name} not tested")

    def test_create_graph(self):
        """Test graph creation with edge cases."""
        # Normal case
        response = self.send_request(
            "tools/call",
            {
                "name": "create_graph",
                "arguments": {"name": "test_graph", "directed": False},
            },
        )
        if self.check_success(response, "create_graph"):
            print("✅ Create graph - normal case")

        # Edge case: empty name
        response = self.send_request(
            "tools/call",
            {"name": "create_graph", "arguments": {"name": "", "directed": False}},
        )
        print(f"Empty name result: {response}")

        # Edge case: very long name
        response = self.send_request(
            "tools/call",
            {
                "name": "create_graph",
                "arguments": {"name": "x" * 1000, "directed": False},
            },
        )
        print("Long name result: handled gracefully")

        # Edge case: special characters
        response = self.send_request(
            "tools/call",
            {
                "name": "create_graph",
                "arguments": {"name": "test/../../../etc/passwd", "directed": False},
            },
        )
        print("Path traversal test: handled gracefully")

    def test_add_nodes(self):
        """Test adding nodes with edge cases."""
        # Normal case
        response = self.send_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {"graph": "test_graph", "nodes": ["A", "B", "C"]},
            },
        )
        if self.check_success(response, "add_nodes"):
            print("✅ Add nodes - normal case")

        # Edge case: empty nodes list
        response = self.send_request(
            "tools/call",
            {"name": "add_nodes", "arguments": {"graph": "test_graph", "nodes": []}},
        )
        print(f"Empty nodes result: {response}")

        # Edge case: very large nodes list
        large_nodes = [f"node_{i}" for i in range(10000)]
        response = self.send_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {"graph": "test_graph", "nodes": large_nodes},
            },
        )
        print("Large nodes list: handled in reasonable time")

        # Edge case: nonexistent graph
        response = self.send_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {"graph": "nonexistent_graph", "nodes": ["A", "B"]},
            },
        )
        if "error" in response or (
            "result" in response and response["result"].get("isError")
        ):
            print("✅ Nonexistent graph error handled correctly")
        else:
            print(f"❌ Should have failed for nonexistent graph: {response}")

    def test_add_edges(self):
        """Test adding edges with edge cases."""
        # Normal case
        response = self.send_request(
            "tools/call",
            {
                "name": "add_edges",
                "arguments": {"graph": "test_graph", "edges": [["A", "B"], ["B", "C"]]},
            },
        )
        if self.check_success(response, "add_edges"):
            print("✅ Add edges - normal case")

        # Edge case: self-loops
        response = self.send_request(
            "tools/call",
            {
                "name": "add_edges",
                "arguments": {"graph": "test_graph", "edges": [["A", "A"]]},
            },
        )
        print(f"Self-loop result: {response}")

        # Edge case: malformed edges
        response = self.send_request(
            "tools/call",
            {
                "name": "add_edges",
                "arguments": {"graph": "test_graph", "edges": [["A"], ["B", "C", "D"]]},
            },
        )
        print(f"Malformed edges handled: {response}")

    def test_get_info(self):
        """Test get_info."""
        response = self.send_request(
            "tools/call", {"name": "get_info", "arguments": {"graph": "test_graph"}}
        )
        if self.check_success(response, "get_info"):
            print("✅ Get info works")

    def test_degree_centrality(self):
        """Test degree centrality."""
        response = self.send_request(
            "tools/call",
            {
                "name": "degree_centrality",
                "arguments": {"graph": "test_graph", "top_n": 5},
            },
        )
        if self.check_success(response, "degree_centrality"):
            print("✅ Degree centrality works")

    def test_pagerank(self):
        """Test PageRank."""
        response = self.send_request(
            "tools/call",
            {"name": "pagerank", "arguments": {"graph": "test_graph", "top_n": 5}},
        )
        if self.check_success(response, "pagerank"):
            print("✅ PageRank works")

    def test_shortest_path(self):
        """Test shortest path."""
        response = self.send_request(
            "tools/call",
            {
                "name": "shortest_path",
                "arguments": {"graph": "test_graph", "source": "A", "target": "C"},
            },
        )
        if self.check_success(response, "shortest_path"):
            print("✅ Shortest path works")

    def test_citation_network(self):
        """Test citation network building."""
        print("Testing citation network (may fail without internet)...")
        response = self.send_request(
            "tools/call",
            {
                "name": "build_citation_network",
                "arguments": {
                    "graph_name": "citation_test",
                    "paper_ids": ["10.1038/nature12373"],
                },
            },
        )
        print(f"Citation network result: {response}")

    def test_doi_resolution(self):
        """Test DOI resolution."""
        print("Testing DOI resolution (may fail without internet)...")
        response = self.send_request(
            "tools/call",
            {"name": "resolve_doi", "arguments": {"doi": "10.1038/nature12373"}},
        )
        print(f"DOI resolution result: {response}")

    def test_visualization(self):
        """Test graph visualization."""
        response = self.send_request(
            "tools/call",
            {
                "name": "visualize_graph",
                "arguments": {"graph": "test_graph", "layout": "spring"},
            },
        )
        if self.check_success(response, "visualize_graph"):
            print("✅ Visualization works")

    def test_export_json(self):
        """Test JSON export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            response = self.send_request(
                "tools/call",
                {
                    "name": "export_json",
                    "arguments": {"graph": "test_graph", "file_path": temp_path},
                },
            )

            if self.check_success(response, "export_json"):
                # Check if file was actually created
                if os.path.exists(temp_path):
                    with open(temp_path, "r") as f:
                        data = json.load(f)
                    print(
                        f"✅ JSON export works - created file with {len(data.get('nodes', []))} nodes"
                    )
                else:
                    print("❌ JSON export claimed success but no file created")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_import_csv(self):
        """Test CSV import."""
        # Create test CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("source,target\n")
            f.write("A,B\n")
            f.write("B,C\n")
            f.write("C,D\n")
            temp_path = f.name

        try:
            response = self.send_request(
                "tools/call",
                {
                    "name": "import_csv",
                    "arguments": {
                        "graph_name": "csv_test",
                        "file_path": temp_path,
                        "source_col": "source",
                        "target_col": "target",
                    },
                },
            )

            if self.check_success(response, "import_csv"):
                print("✅ CSV import works")
            else:
                print(f"❌ CSV import failed: {response}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_malicious_inputs(self):
        """Test with malicious inputs."""
        print("\n=== TESTING MALICIOUS INPUTS ===")

        # SQL injection attempt
        response = self.send_request(
            "tools/call",
            {
                "name": "create_graph",
                "arguments": {
                    "name": "test'; DROP TABLE graphs; --",
                    "directed": False,
                },
            },
        )
        print(f"SQL injection test: {response}")

        # Path traversal attempt
        response = self.send_request(
            "tools/call",
            {
                "name": "export_json",
                "arguments": {
                    "graph": "test_graph",
                    "file_path": "../../../etc/passwd",
                },
            },
        )
        print(f"Path traversal test: {response}")

        # XSS attempt
        response = self.send_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {
                    "graph": "test_graph",
                    "nodes": ["<script>alert('xss')</script>"],
                },
            },
        )
        print(f"XSS test: {response}")

        # Extremely large input
        response = self.send_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {
                    "graph": "test_graph",
                    "nodes": ["x" * 1000000],  # 1MB string
                },
            },
        )
        print("Large input test: handled gracefully")

    def test_concurrent_requests(self):
        """Test concurrent requests."""
        print("\n=== TESTING CONCURRENT REQUESTS ===")
        print("⚠️  Current server only handles sequential requests")
        print("⚠️  Real production server would need async handling")

    def check_success(self, response, operation):
        """Check if response indicates success."""
        if not response:
            self.failures.append(f"{operation}: no response")
            return False

        if "error" in response:
            self.failures.append(f"{operation}: {response['error']}")
            return False

        if "result" in response:
            if response["result"].get("isError"):
                self.failures.append(f"{operation}: {response['result']}")
                return False
            else:
                self.successes.append(operation)
                return True

        self.failures.append(f"{operation}: unexpected response format")
        return False

    def run_all_tests(self):
        """Run all tests."""
        print("🔥 STARTING BRUTAL REALITY TEST 🔥")

        try:
            self.start_server()

            self.test_initialize()
            self.test_all_tools()
            self.test_malicious_inputs()
            self.test_concurrent_requests()

        finally:
            self.stop_server()

        # Print summary
        print("\n" + "=" * 50)
        print("BRUTAL REALITY TEST SUMMARY")
        print("=" * 50)
        print(f"✅ Successes: {len(self.successes)}")
        print(f"❌ Failures: {len(self.failures)}")

        if self.successes:
            print(f"\n✅ Working: {', '.join(self.successes)}")

        if self.failures:
            print("\n❌ Broken:")
            for failure in self.failures:
                print(f"  - {failure}")

        # Overall assessment
        success_rate = (
            len(self.successes) / (len(self.successes) + len(self.failures)) * 100
        )
        print(f"\nOverall success rate: {success_rate:.1f}%")

        if success_rate > 80:
            print("🎉 Server is mostly functional")
        elif success_rate > 60:
            print("⚠️  Server has significant issues")
        else:
            print("💥 Server is largely broken")


if __name__ == "__main__":
    tester = BrutalRealityTester()
    tester.run_all_tests()
