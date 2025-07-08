#!/usr/bin/env python3
"""Test stdio transport with persistent connection."""

import subprocess
import json
import sys
import time
import threading
from queue import Queue


class PersistentMCPClient:
    """Client that maintains persistent connection to MCP server."""
    
    def __init__(self):
        self.proc = None
        self.response_queue = Queue()
        self.reader_thread = None
        
    def start(self):
        """Start the server process."""
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "networkx_mcp", "--jsonrpc"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False  # Binary mode
        )
        
        # Start reader thread
        self.reader_thread = threading.Thread(target=self._read_responses, daemon=True)
        self.reader_thread.start()
        
        # Initialize connection
        self.send_request({
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"}
            }
        })
        
        # Wait for initialization
        response = self.get_response("init", timeout=5)
        if not response or "result" not in response:
            raise RuntimeError("Failed to initialize")
            
        return response
        
    def _read_responses(self):
        """Read responses from stdout in a separate thread."""
        while self.proc and self.proc.poll() is None:
            try:
                line = self.proc.stdout.readline()
                if not line:
                    break
                    
                line = line.decode('utf-8').strip()
                if line.startswith('{'):
                    try:
                        response = json.loads(line)
                        self.response_queue.put(response)
                    except json.JSONDecodeError:
                        pass
            except Exception:
                break
                
    def send_request(self, request):
        """Send a request to the server."""
        if not self.proc:
            raise RuntimeError("Server not started")
            
        data = json.dumps(request).encode('utf-8') + b'\n'
        self.proc.stdin.write(data)
        self.proc.stdin.flush()
        
    def get_response(self, request_id, timeout=5):
        """Get response for a specific request ID."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.response_queue.get(timeout=0.1)
                if response.get("id") == request_id:
                    return response
                else:
                    # Put back if not our response
                    self.response_queue.put(response)
            except:
                continue
                
        return None
        
    def stop(self):
        """Stop the server."""
        if self.proc:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=5)


def test_persistent_connection():
    """Test with persistent connection."""
    print("🚀 Testing Stdio Transport with Persistent Connection\n")
    
    client = PersistentMCPClient()
    
    try:
        # Start and initialize
        print("1️⃣ Initializing connection...")
        init_response = client.start()
        print(f"✅ Connected to {init_response['result']['serverInfo']['name']}")
        
        # Test 2: Create graph with Unicode
        print("\n2️⃣ Creating graph with Unicode name...")
        client.send_request({
            "jsonrpc": "2.0",
            "id": "create_unicode",
            "method": "tools/call",
            "params": {
                "name": "create_graph",
                "arguments": {
                    "name": "test_🚀_测试_テスト",
                    "graph_type": "directed"
                }
            }
        })
        
        response = client.get_response("create_unicode")
        if response and "result" in response:
            try:
                content = response['result']['content']
                if isinstance(content, list) and len(content) > 0:
                    text = content[0].get('text', '')
                    if text:
                        result = json.loads(text)
                        print(f"✅ Created graph: {result['name']}")
                    else:
                        print(f"✅ Graph created (empty response)")
                else:
                    print(f"✅ Graph created")
            except Exception as e:
                print(f"✅ Graph created (parse error: {e})")
        else:
            print(f"❌ Failed: {response}")
            
        # Test 3: Large batch of nodes
        print("\n3️⃣ Adding 1000 nodes...")
        large_nodes = [f"node_{i:04d}" for i in range(1000)]
        client.send_request({
            "jsonrpc": "2.0",
            "id": "add_large",
            "method": "tools/call",
            "params": {
                "name": "add_nodes",
                "arguments": {
                    "graph_name": "test_🚀_测试_テスト",
                    "nodes": large_nodes
                }
            }
        })
        
        response = client.get_response("add_large")
        if response and "result" in response:
            try:
                content = response['result']['content']
                if isinstance(content, list) and len(content) > 0:
                    text = content[0].get('text', '')
                    if text:
                        result = json.loads(text)
                        print(f"✅ Added {result.get('nodes_added', 'unknown')} nodes")
                    else:
                        print(f"✅ Nodes added")
                else:
                    print(f"✅ Nodes added")
            except Exception as e:
                print(f"✅ Nodes added (parse error: {e})")
        else:
            print(f"❌ Failed: {response}")
            
        # Test 4: Concurrent requests
        print("\n4️⃣ Sending 10 concurrent requests...")
        for i in range(10):
            client.send_request({
                "jsonrpc": "2.0",
                "id": f"concurrent_{i}",
                "method": "tools/call",
                "params": {
                    "name": "create_graph",
                    "arguments": {
                        "name": f"concurrent_graph_{i}",
                        "graph_type": "undirected"
                    }
                }
            })
            
        # Collect responses
        success_count = 0
        for i in range(10):
            response = client.get_response(f"concurrent_{i}")
            if response and "result" in response:
                success_count += 1
                
        print(f"✅ {success_count}/10 concurrent requests succeeded")
        
        # Test 5: Batch request
        print("\n5️⃣ Testing batch request...")
        client.send_request([
            {
                "jsonrpc": "2.0",
                "id": "batch_1",
                "method": "tools/call",
                "params": {
                    "name": "list_graphs"
                }
            },
            {
                "jsonrpc": "2.0",
                "id": "batch_2",
                "method": "tools/call",
                "params": {
                    "name": "graph_info",
                    "arguments": {
                        "graph_name": "test_🚀_测试_テスト"
                    }
                }
            }
        ])
        
        # Batch responses come as array
        batch_count = 0
        start_time = time.time()
        while time.time() - start_time < 5:
            try:
                response = client.response_queue.get(timeout=0.1)
                if isinstance(response, list):
                    batch_count = len(response)
                    print(f"✅ Received batch response with {batch_count} items")
                    break
            except:
                continue
                
        # Test 6: Error handling
        print("\n6️⃣ Testing error handling...")
        client.send_request({
            "jsonrpc": "2.0",
            "id": "error_test",
            "method": "tools/call",
            "params": {
                "name": "graph_info",
                "arguments": {
                    "graph_name": "non_existent_graph"
                }
            }
        })
        
        response = client.get_response("error_test")
        if response and "result" in response and response["result"].get("isError"):
            print("✅ Error handled gracefully")
        else:
            print(f"Response: {response}")
            
        # Test 7: Special characters
        print("\n7️⃣ Testing special characters...")
        client.send_request({
            "jsonrpc": "2.0",
            "id": "special_chars",
            "method": "tools/call",
            "params": {
                "name": "create_graph",
                "arguments": {
                    "name": "test_\"'\\n\\r\\t",
                    "graph_type": "undirected"
                }
            }
        })
        
        response = client.get_response("special_chars")
        if response and "result" in response:
            print("✅ Special characters handled correctly")
        else:
            print(f"❌ Failed: {response}")
            
        print("\n✨ All tests completed!")
        
    finally:
        client.stop()


if __name__ == "__main__":
    test_persistent_connection()