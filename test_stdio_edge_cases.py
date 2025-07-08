#!/usr/bin/env python3
"""Test stdio transport edge cases."""

import asyncio
import json
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor


def send_request(request_data, timeout=5):
    """Send a request and get response."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp", "--jsonrpc"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False  # Use binary mode
    )
    
    try:
        # Send request
        if isinstance(request_data, str):
            input_data = request_data.encode('utf-8')
        else:
            input_data = json.dumps(request_data).encode('utf-8')
            
        stdout, stderr = proc.communicate(input=input_data + b'\n', timeout=timeout)
        
        # Extract JSON response
        for line in stdout.decode('utf-8').strip().split('\n'):
            if line.startswith('{'):
                return json.loads(line), stderr.decode('utf-8')
                
        return None, stderr.decode('utf-8')
        
    except subprocess.TimeoutExpired:
        proc.kill()
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


def test_edge_cases():
    """Test various edge cases."""
    print("🧪 Testing Stdio Transport Edge Cases\n")
    
    # Test 1: Valid initialization
    print("1️⃣ Testing valid initialization...")
    response, stderr = send_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    })
    
    if response and "result" in response:
        print("✅ Valid initialization handled correctly")
    else:
        print(f"❌ Failed: {response or 'No response'}")
        
    # Test 2: Malformed JSON
    print("\n2️⃣ Testing malformed JSON...")
    response, stderr = send_request('{"invalid": json}')
    
    if response and "error" in response and response["error"]["code"] == -32700:
        print("✅ Malformed JSON handled correctly")
    else:
        print(f"❌ Failed: {response or 'No response'}")
        
    # Test 3: Unicode handling
    print("\n3️⃣ Testing Unicode handling...")
    response, stderr = send_request({
        "jsonrpc": "2.0",
        "id": "unicode_test",
        "method": "tools/call",
        "params": {
            "name": "create_graph",
            "arguments": {
                "name": "test_graph_🚀_测试_テスト",
                "graph_type": "undirected"
            }
        }
    })
    
    if response and "result" in response:
        print("✅ Unicode handled correctly")
    else:
        print(f"❌ Failed: {response or 'No response'}")
        
    # Test 4: Large message
    print("\n4️⃣ Testing large message...")
    large_nodes = [f"node_{i}" for i in range(1000)]
    response, stderr = send_request({
        "jsonrpc": "2.0",
        "id": "large_test",
        "method": "tools/call",
        "params": {
            "name": "add_nodes",
            "arguments": {
                "graph_name": "test_graph_🚀_测试_テスト",
                "nodes": large_nodes
            }
        }
    })
    
    if response and "result" in response:
        print("✅ Large message handled correctly")
    else:
        print(f"❌ Failed: {response or 'No response'}")
        
    # Test 5: Binary data in JSON (should fail gracefully)
    print("\n5️⃣ Testing binary data handling...")
    response, stderr = send_request({
        "jsonrpc": "2.0",
        "id": "binary_test",
        "method": "echo",
        "params": {"data": "\x00\x01\x02\x03"}
    })
    
    if response and "error" in response:
        print("✅ Binary data rejected correctly")
    else:
        print(f"❌ Failed: {response or 'No response'}")
        
    # Test 6: Empty/whitespace messages
    print("\n6️⃣ Testing empty/whitespace messages...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp", "--jsonrpc"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False
    )
    
    # Send multiple empty lines and whitespace
    proc.stdin.write(b'\n\n   \n\t\n')
    proc.stdin.write(b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}\n')
    proc.stdin.close()
    
    stdout, stderr = proc.communicate(timeout=5)
    
    if b'"result"' in stdout:
        print("✅ Empty/whitespace messages ignored correctly")
    else:
        print("❌ Failed to handle empty messages")
        
    print("\n✨ Edge case testing complete!")


def test_concurrent_writes():
    """Test concurrent write safety."""
    print("\n⚡ Testing Concurrent Write Safety\n")
    
    # Start a persistent server process
    proc = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp", "--jsonrpc"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False
    )
    
    # Initialize
    init_req = json.dumps({
        "jsonrpc": "2.0",
        "id": "init",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    }).encode('utf-8')
    
    proc.stdin.write(init_req + b'\n')
    proc.stdin.flush()
    
    # Wait for initialization
    time.sleep(0.5)
    
    # Send multiple concurrent requests
    requests = []
    for i in range(10):
        req = json.dumps({
            "jsonrpc": "2.0",
            "id": f"concurrent_{i}",
            "method": "tools/call",
            "params": {
                "name": "create_graph",
                "arguments": {
                    "name": f"graph_{i}",
                    "graph_type": "undirected"
                }
            }
        }).encode('utf-8')
        requests.append(req)
    
    # Send all requests rapidly
    for req in requests:
        proc.stdin.write(req + b'\n')
    proc.stdin.flush()
    proc.stdin.close()
    
    # Read all responses
    stdout, stderr = proc.communicate(timeout=5)
    
    # Check responses
    responses = []
    for line in stdout.decode('utf-8').strip().split('\n'):
        if line.startswith('{'):
            try:
                responses.append(json.loads(line))
            except:
                print(f"Failed to parse: {line}")
    
    # Verify we got all responses
    success_count = sum(1 for r in responses if "result" in r)
    print(f"✅ Received {success_count}/{len(requests)+1} responses without corruption")
    
    proc.terminate()


if __name__ == "__main__":
    test_edge_cases()
    test_concurrent_writes()