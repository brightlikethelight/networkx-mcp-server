#!/usr/bin/env python3
"""
Simple tests that actually work for the minimal server.
No 500-line test files. Just verify it works.
"""

import json
import asyncio
import subprocess
import sys
from typing import Dict, Any

class TestMinimalServer:
    """Test the minimal server with real subprocess communication."""
    
    def __init__(self):
        self.process = None
        self.reader = None
        self.writer = None
        self.request_id = 0
    
    async def start_server(self):
        """Start the server as a subprocess."""
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "server_truly_minimal.py",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.reader = self.process.stdout
        self.writer = self.process.stdin
        
        # Start error reader task
        asyncio.create_task(self._read_stderr())
    
    async def _read_stderr(self):
        """Read and print stderr from server."""
        while True:
            line = await self.process.stderr.readline()
            if not line:
                break
            print(f"SERVER ERROR: {line.decode().strip()}")
    
    async def send_request(self, method: str, params: Dict[str, Any] = None, req_id: int = None) -> Dict:
        """Send a request and get response."""
        if req_id is None:
            self.request_id += 1
            req_id = self.request_id
            
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {}
        }
        
        # Send request
        self.writer.write((json.dumps(request) + "\n").encode())
        await self.writer.drain()
        
        # Read response
        response_line = await self.reader.readline()
        if not response_line:
            raise ValueError("No response from server")
        decoded = response_line.decode().strip()
        if not decoded:
            raise ValueError("Empty response from server")
        return json.loads(decoded)
    
    async def test_basic_operations(self):
        """Test basic graph operations."""
        print("Testing minimal NetworkX MCP server...")
        
        # Test 1: Initialize
        print("\n1. Testing initialize...")
        response = await self.send_request("initialize")
        assert response["result"]["protocolVersion"] == "2024-11-05"
        print("✓ Initialize works")
        
        # Test 2: List tools
        print("\n2. Testing tools/list...")
        response = await self.send_request("tools/list")
        tools = response["result"]["tools"]
        assert len(tools) == 5
        assert any(t["name"] == "create_graph" for t in tools)
        print(f"✓ Found {len(tools)} tools")
        
        # Test 3: Create graph
        print("\n3. Testing create_graph...")
        response = await self.send_request("tools/call", {
            "name": "create_graph",
            "arguments": {"name": "test_graph", "directed": False}
        })
        result = json.loads(response["result"]["content"][0]["text"])
        assert result["created"] == "test_graph"
        print("✓ Graph created")
        
        # Test 4: Add nodes
        print("\n4. Testing add_nodes...")
        response = await self.send_request("tools/call", {
            "name": "add_nodes",
            "arguments": {"graph": "test_graph", "nodes": [1, 2, 3, 4, 5]}
        })
        if "error" in response:
            raise ValueError(f"Server error: {response['error']}")
            
        result = json.loads(response["result"]["content"][0]["text"])
        assert result["total"] == 5
        print(f"✓ Added {result['added']} nodes")
        
        # Test 5: Add edges
        print("\n5. Testing add_edges...")
        response = await self.send_request("tools/call", {
            "name": "add_edges",
            "arguments": {"graph": "test_graph", "edges": [[1, 2], [2, 3], [3, 4], [4, 5]]}
        })
        result = json.loads(response["result"]["content"][0]["text"])
        assert result["total"] == 4
        print(f"✓ Added {result['added']} edges")
        
        # Test 6: Shortest path
        print("\n6. Testing shortest_path...")
        response = await self.send_request("tools/call", {
            "name": "shortest_path",
            "arguments": {"graph": "test_graph", "source": 1, "target": 5}
        })
        result = json.loads(response["result"]["content"][0]["text"])
        assert result["path"] == [1, 2, 3, 4, 5]
        assert result["length"] == 4
        print(f"✓ Found path: {result['path']}")
        
        # Test 7: Get info
        print("\n7. Testing get_info...")
        response = await self.send_request("tools/call", {
            "name": "get_info",
            "arguments": {"graph": "test_graph"}
        })
        result = json.loads(response["result"]["content"][0]["text"])
        assert result["nodes"] == 5
        assert result["edges"] == 4
        print(f"✓ Graph has {result['nodes']} nodes and {result['edges']} edges")
        
        # Test 8: Error handling
        print("\n8. Testing error handling...")
        response = await self.send_request("tools/call", {
            "name": "get_info",
            "arguments": {"graph": "nonexistent"}
        })
        assert response["result"]["isError"] == True
        assert "not found" in response["result"]["content"][0]["text"]
        print("✓ Error handling works")
        
        print("\n✅ All tests passed!")
        
    async def cleanup(self):
        """Clean up the server process."""
        if self.process:
            self.process.terminate()
            await self.process.wait()
    
    async def run_tests(self):
        """Run all tests."""
        try:
            await self.start_server()
            await self.test_basic_operations()
        finally:
            await self.cleanup()

# Run the tests
if __name__ == "__main__":
    tester = TestMinimalServer()
    asyncio.run(tester.run_tests())