#!/usr/bin/env python3
"""
Test the actual server running in stdio mode like a real MCP client would.
"""

import json
import subprocess
import sys
import time
from typing import Any, Dict, Optional


class STDIOServerTester:
    """Test the server running in stdio mode."""

    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.errors = []
        self.successes = []

    def start_server(self):
        """Start the server process."""
        try:
            # Start server in stdio mode
            cmd = [sys.executable, "-m", "networkx_mcp.server"]
            self.server_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            time.sleep(1)  # Give server time to start
            return True
        except Exception as e:
            self.errors.append(f"Failed to start server: {str(e)}")
            return False

    def send_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a request to the server and get response."""
        if not self.server_process:
            return None

        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self.server_process.stdin.write(request_json)
            self.server_process.stdin.flush()

            # Read response
            response_line = self.server_process.stdout.readline()
            if response_line:
                return json.loads(response_line.strip())
            else:
                return None
        except Exception as e:
            self.errors.append(f"Request failed: {str(e)}")
            return None

    def test_initialization(self):
        """Test MCP initialization sequence."""
        print("Testing MCP initialization...")

        # Initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        response = self.send_request(init_request)
        if response and response.get("result"):
            self.successes.append("MCP initialization successful")
            print("✅ MCP initialization successful")
            return True
        else:
            self.errors.append("MCP initialization failed")
            print("❌ MCP initialization failed")
            return False

    def test_tools_list(self):
        """Test tools listing."""
        print("Testing tools list...")

        request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

        response = self.send_request(request)
        if response and response.get("result", {}).get("tools"):
            tools = response["result"]["tools"]
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"   • {tool.get('name', 'Unknown')}")
            self.successes.append(f"Tools list successful: {len(tools)} tools")
            return True
        else:
            self.errors.append("Tools list failed")
            print("❌ Tools list failed")
            return False

    def test_graph_operations(self):
        """Test basic graph operations."""
        print("Testing graph operations...")

        operations = [
            # Create graph
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "create_graph",
                    "arguments": {"name": "test_graph", "directed": False},
                },
            },
            # Add nodes
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "add_nodes",
                    "arguments": {"graph": "test_graph", "nodes": [1, 2, 3, 4, 5]},
                },
            },
            # Add edges
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "add_edges",
                    "arguments": {
                        "graph": "test_graph",
                        "edges": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1]],
                    },
                },
            },
            # Get info
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "get_info", "arguments": {"graph": "test_graph"}},
            },
            # Calculate centrality
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "degree_centrality",
                    "arguments": {"graph": "test_graph"},
                },
            },
        ]

        for i, op in enumerate(operations):
            response = self.send_request(op)
            if response and "result" in response:
                print(f"✅ Operation {i + 1} successful")
                self.successes.append(f"Graph operation {i + 1} successful")
            else:
                print(f"❌ Operation {i + 1} failed")
                self.errors.append(f"Graph operation {i + 1} failed")
                return False

        return True

    def test_academic_features(self):
        """Test academic features."""
        print("Testing academic features...")

        operations = [
            # Resolve DOI
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {
                    "name": "resolve_doi",
                    "arguments": {"doi": "10.1038/nature12373"},
                },
            },
            # Build citation network
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {
                    "name": "build_citation_network",
                    "arguments": {
                        "graph": "citation_network",
                        "seed_dois": ["10.1038/nature12373"],
                        "max_depth": 1,
                    },
                },
            },
        ]

        for i, op in enumerate(operations):
            response = self.send_request(op)
            if response and "result" in response:
                print(f"✅ Academic operation {i + 1} successful")
                self.successes.append(f"Academic operation {i + 1} successful")
            else:
                print(f"❌ Academic operation {i + 1} failed")
                self.errors.append(f"Academic operation {i + 1} failed")
                # Don't fail the whole test for academic features (may be network dependent)

        return True

    def test_error_handling(self):
        """Test error handling."""
        print("Testing error handling...")

        # Test invalid graph name
        request = {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "get_info", "arguments": {"graph": "nonexistent_graph"}},
        }

        response = self.send_request(request)
        if response and response.get("result", {}).get("isError"):
            print("✅ Error handling works correctly")
            self.successes.append("Error handling works")
            return True
        else:
            print("❌ Error handling may not work correctly")
            self.errors.append("Error handling issues")
            return False

    def stop_server(self):
        """Stop the server process."""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None

    def run_tests(self):
        """Run all tests."""
        print("🔥 TESTING ACTUAL SERVER (STDIO MODE)")
        print("=" * 50)

        if not self.start_server():
            print("❌ Failed to start server")
            return False

        try:
            # Run test sequence
            tests = [
                self.test_initialization,
                self.test_tools_list,
                self.test_graph_operations,
                self.test_academic_features,
                self.test_error_handling,
            ]

            for test in tests:
                if not test():
                    print(f"Test {test.__name__} failed")
                    break
                time.sleep(0.1)  # Small delay between tests

            # Generate report
            print("\n📊 STDIO SERVER TEST REPORT")
            print("=" * 40)
            print(f"✅ Successes: {len(self.successes)}")
            print(f"❌ Errors: {len(self.errors)}")

            if self.errors:
                print("\nErrors:")
                for error in self.errors:
                    print(f"  • {error}")

            if self.successes:
                print("\nSuccesses:")
                for success in self.successes:
                    print(f"  • {success}")

            success_rate = (
                len(self.successes) / (len(self.successes) + len(self.errors)) * 100
            )
            print(f"\nSuccess Rate: {success_rate:.1f}%")

            return len(self.errors) == 0

        finally:
            self.stop_server()


if __name__ == "__main__":
    tester = STDIOServerTester()
    success = tester.run_tests()
    sys.exit(0 if success else 1)
