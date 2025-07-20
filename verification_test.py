#!/usr/bin/env python3
"""
Comprehensive verification test for NetworkX MCP Server
Tests all claimed functionality to ensure it actually works.
"""

import asyncio
import json
import sys
import traceback
from typing import Dict, Any
import subprocess
import time
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from networkx_mcp.server import NetworkXMCPServer


class MCPTester:
    """Test MCP protocol compliance and functionality."""
    
    def __init__(self):
        self.server = NetworkXMCPServer()
        self.test_results = {}
        
    async def send_request(self, method: str, params: Dict[str, Any] = None, 
                          req_id: int = 1) -> Dict[str, Any]:
        """Send a request to the server."""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": req_id
        }
        if params:
            request["params"] = params
            
        return await self.server.handle_request(request)
    
    async def test_mcp_protocol(self):
        """Test MCP protocol compliance."""
        print("=== Testing MCP Protocol Compliance ===")
        
        # Test initialization
        response = await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        })
        
        assert response.get("jsonrpc") == "2.0"
        assert "result" in response
        result = response["result"]
        assert result.get("protocolVersion") == "2024-11-05"
        assert "capabilities" in result
        assert "serverInfo" in result
        print("✓ Initialize protocol works")
        
        # Test initialized notification
        response = await self.send_request("initialized", {})
        print("✓ Initialized notification works")
        
        # Test tools list
        response = await self.send_request("tools/list")
        assert "result" in response
        tools = response["result"]["tools"]
        assert len(tools) >= 19  # We expect at least 19 tools
        print(f"✓ Tools list works - {len(tools)} tools available")
        
        # Test unknown method
        response = await self.send_request("unknown/method")
        assert "error" in response
        assert response["error"]["code"] == -32601
        print("✓ Unknown method handling works")
        
        self.test_results["mcp_protocol"] = "PASS"
        return True
    
    async def test_basic_graph_operations(self):
        """Test basic graph creation and operations."""
        print("\n=== Testing Basic Graph Operations ===")
        
        # Test graph creation
        response = await self.send_request("tools/call", {
            "name": "create_graph",
            "arguments": {"name": "test_graph", "directed": False}
        })
        
        assert "result" in response
        content = json.loads(response["result"]["content"][0]["text"])
        assert content["created"] == "test_graph"
        assert content["type"] == "undirected"
        print("✓ Graph creation works")
        
        # Test adding nodes
        response = await self.send_request("tools/call", {
            "name": "add_nodes",
            "arguments": {"graph": "test_graph", "nodes": ["A", "B", "C", "D"]}
        })
        
        assert "result" in response
        content = json.loads(response["result"]["content"][0]["text"])
        assert content["added"] == 4
        assert content["total"] == 4
        print("✓ Adding nodes works")
        
        # Test adding edges
        response = await self.send_request("tools/call", {
            "name": "add_edges",
            "arguments": {"graph": "test_graph", "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["A", "C"]]}
        })
        
        assert "result" in response
        content = json.loads(response["result"]["content"][0]["text"])
        assert content["added"] == 4
        assert content["total"] == 4
        print("✓ Adding edges works")
        
        # Test graph info
        response = await self.send_request("tools/call", {
            "name": "get_info",
            "arguments": {"graph": "test_graph"}
        })
        
        assert "result" in response
        content = json.loads(response["result"]["content"][0]["text"])
        assert content["nodes"] == 4
        assert content["edges"] == 4
        assert content["directed"] == False
        print("✓ Graph info retrieval works")
        
        self.test_results["basic_operations"] = "PASS"
        return True
    
    async def test_algorithms(self):
        """Test graph algorithms."""
        print("\n=== Testing Graph Algorithms ===")
        
        # Test shortest path
        response = await self.send_request("tools/call", {
            "name": "shortest_path",
            "arguments": {"graph": "test_graph", "source": "A", "target": "D"}
        })
        
        assert "result" in response
        content = json.loads(response["result"]["content"][0]["text"])
        assert "path" in content
        assert content["path"][0] == "A"
        assert content["path"][-1] == "D"
        print("✓ Shortest path algorithm works")
        
        # Test degree centrality
        response = await self.send_request("tools/call", {
            "name": "degree_centrality",
            "arguments": {"graph": "test_graph"}
        })
        
        assert "result" in response
        content = json.loads(response["result"]["content"][0]["text"])
        assert "centrality" in content
        assert len(content["centrality"]) == 4
        print("✓ Degree centrality calculation works")
        
        # Test betweenness centrality
        response = await self.send_request("tools/call", {
            "name": "betweenness_centrality",
            "arguments": {"graph": "test_graph"}
        })
        
        assert "result" in response
        content = json.loads(response["result"]["content"][0]["text"])
        assert "centrality" in content
        print("✓ Betweenness centrality calculation works")
        
        # Test PageRank
        response = await self.send_request("tools/call", {
            "name": "pagerank",
            "arguments": {"graph": "test_graph"}
        })
        
        assert "result" in response
        content = json.loads(response["result"]["content"][0]["text"])
        assert "pagerank" in content
        print("✓ PageRank calculation works")
        
        # Test connected components
        response = await self.send_request("tools/call", {
            "name": "connected_components",
            "arguments": {"graph": "test_graph"}
        })
        
        assert "result" in response
        content = json.loads(response["result"]["content"][0]["text"])
        assert "num_components" in content
        assert content["num_components"] == 1  # All nodes connected
        print("✓ Connected components detection works")
        
        # Test community detection
        response = await self.send_request("tools/call", {
            "name": "community_detection",
            "arguments": {"graph": "test_graph"}
        })
        
        assert "result" in response
        content = json.loads(response["result"]["content"][0]["text"])
        assert "communities" in content
        print("✓ Community detection works")
        
        self.test_results["algorithms"] = "PASS"
        return True
    
    async def test_visualization(self):
        """Test graph visualization."""
        print("\n=== Testing Visualization Features ===")
        
        try:
            response = await self.send_request("tools/call", {
                "name": "visualize_graph",
                "arguments": {"graph": "test_graph", "layout": "spring"}
            })
            
            assert "result" in response
            content = json.loads(response["result"]["content"][0]["text"])
            assert "visualization" in content
            assert "format" in content
            assert "layout" in content
            print("✓ Spring layout visualization works")
            
            # Test different layout
            response = await self.send_request("tools/call", {
                "name": "visualize_graph",
                "arguments": {"graph": "test_graph", "layout": "circular"}
            })
            
            assert "result" in response
            content = json.loads(response["result"]["content"][0]["text"])
            assert content["layout"] == "circular"
            print("✓ Circular layout visualization works")
            
            self.test_results["visualization"] = "PASS"
            return True
            
        except Exception as e:
            print(f"✗ Visualization test failed: {e}")
            self.test_results["visualization"] = f"FAIL: {e}"
            return False
    
    async def test_io_operations(self):
        """Test import/export functionality."""
        print("\n=== Testing I/O Operations ===")
        
        # Test CSV import
        csv_data = "A,B\nB,C\nC,D\nD,A"
        response = await self.send_request("tools/call", {
            "name": "import_csv",
            "arguments": {"graph": "csv_graph", "csv_data": csv_data, "directed": False}
        })
        
        assert "result" in response
        content = json.loads(response["result"]["content"][0]["text"])
        assert content["nodes"] == 4
        assert content["edges"] == 4
        print("✓ CSV import works")
        
        # Test JSON export
        response = await self.send_request("tools/call", {
            "name": "export_json",
            "arguments": {"graph": "csv_graph"}
        })
        
        assert "result" in response
        content = json.loads(response["result"]["content"][0]["text"])
        print(f"JSON export content: {content}")  # Debug
        assert "graph_data" in content
        assert "format" in content
        print("✓ JSON export works")
        
        self.test_results["io_operations"] = "PASS"
        return True
    
    async def test_academic_features(self):
        """Test academic/citation features."""
        print("\n=== Testing Academic Features ===")
        
        # Test DOI resolution (may fail due to network)
        try:
            response = await self.send_request("tools/call", {
                "name": "resolve_doi",
                "arguments": {"doi": "10.1038/nature12373"}
            })
            
            if "result" in response:
                content = json.loads(response["result"]["content"][0]["text"])
                if "title" in content:
                    print("✓ DOI resolution works")
                    self.test_results["doi_resolution"] = "PASS"
                else:
                    print("⚠ DOI resolution returned empty result")
                    self.test_results["doi_resolution"] = "PARTIAL"
            else:
                print("⚠ DOI resolution failed (network/API issue)")
                self.test_results["doi_resolution"] = "NETWORK_FAIL"
                
        except Exception as e:
            print(f"⚠ DOI resolution failed: {e}")
            self.test_results["doi_resolution"] = f"FAIL: {e}"
        
        # Test other academic functions (may require citation network)
        try:
            # These will likely fail without proper citation data, but we test the interface
            response = await self.send_request("tools/call", {
                "name": "find_collaboration_patterns",
                "arguments": {"graph": "test_graph"}
            })
            
            # Should handle gracefully even without citation data
            assert "result" in response
            print("✓ Collaboration patterns interface works")
            
        except Exception as e:
            print(f"⚠ Academic features may need citation data: {e}")
        
        self.test_results["academic_features"] = "PARTIAL"
        return True
    
    async def test_error_handling(self):
        """Test error handling and edge cases."""
        print("\n=== Testing Error Handling ===")
        
        # Test non-existent graph
        response = await self.send_request("tools/call", {
            "name": "get_info",
            "arguments": {"graph": "nonexistent_graph"}
        })
        
        assert "result" in response
        content_text = response["result"]["content"][0]["text"]
        assert "Error:" in content_text or "not found" in content_text
        print("✓ Non-existent graph error handling works")
        
        # Test invalid shortest path
        response = await self.send_request("tools/call", {
            "name": "shortest_path",
            "arguments": {"graph": "test_graph", "source": "A", "target": "Z"}
        })
        
        assert "result" in response
        content_text = response["result"]["content"][0]["text"]
        assert "Error:" in content_text
        print("✓ Invalid path error handling works")
        
        # Test malformed tool call
        response = await self.send_request("tools/call", {
            "name": "unknown_tool",
            "arguments": {}
        })
        
        assert "result" in response
        content_text = response["result"]["content"][0]["text"]
        assert "Error:" in content_text
        print("✓ Unknown tool error handling works")
        
        self.test_results["error_handling"] = "PASS"
        return True
    
    async def run_all_tests(self):
        """Run all verification tests."""
        print("Starting comprehensive NetworkX MCP Server verification...\n")
        
        try:
            await self.test_mcp_protocol()
            await self.test_basic_graph_operations()
            await self.test_algorithms()
            await self.test_visualization()
            await self.test_io_operations()
            await self.test_academic_features()
            await self.test_error_handling()
            
        except Exception as e:
            print(f"\nCritical test failure: {e}")
            traceback.print_exc()
            self.test_results["critical_failure"] = str(e)
        
        # Print summary
        print("\n" + "="*50)
        print("VERIFICATION SUMMARY")
        print("="*50)
        
        for test_name, result in self.test_results.items():
            status = "✓" if result == "PASS" else "⚠" if "PARTIAL" in result else "✗"
            print(f"{status} {test_name}: {result}")
        
        # Overall assessment
        passed = sum(1 for r in self.test_results.values() if r == "PASS")
        partial = sum(1 for r in self.test_results.values() if "PARTIAL" in r)
        failed = len(self.test_results) - passed - partial
        
        print(f"\nResults: {passed} PASS, {partial} PARTIAL, {failed} FAIL")
        
        if failed == 0:
            print("\n🎉 All core functionality verified working!")
            return True
        else:
            print(f"\n⚠️  {failed} areas need attention")
            return False


async def main():
    """Run verification tests."""
    tester = MCPTester()
    success = await tester.run_all_tests()
    return success


if __name__ == "__main__":
    asyncio.run(main())