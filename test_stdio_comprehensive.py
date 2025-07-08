#!/usr/bin/env python3
"""Comprehensive stdio transport test."""

import subprocess
import json
import sys
import os


def test_json_rpc_messages():
    """Test various JSON-RPC message patterns."""
    print("🧪 Comprehensive Stdio Transport Test\n")
    
    test_cases = [
        # Test 1: Basic flow
        {
            "name": "Basic Flow",
            "messages": [
                {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}},
                {"jsonrpc":"2.0","id":2,"method":"tools/list"},
                {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"create_graph","arguments":{"name":"test_graph","graph_type":"directed"}}},
            ]
        },
        
        # Test 2: Error handling
        {
            "name": "Error Handling",
            "messages": [
                {"jsonrpc":"2.0","id":"e1","method":"unknown_method"},
                {"jsonrpc":"2.0","id":"e2","method":"tools/call","params":{"name":"unknown_tool"}},
                {"jsonrpc":"2.0","id":"e3","method":"tools/call"},  # Missing params
            ]
        },
        
        # Test 3: Batch requests
        {
            "name": "Batch Requests",
            "batch": True,
            "messages": [
                {"jsonrpc":"2.0","id":"b1","method":"tools/call","params":{"name":"create_graph","arguments":{"name":"batch_graph_1"}}},
                {"jsonrpc":"2.0","id":"b2","method":"tools/call","params":{"name":"create_graph","arguments":{"name":"batch_graph_2"}}},
                {"jsonrpc":"2.0","id":"b3","method":"tools/call","params":{"name":"list_graphs"}},
            ]
        },
        
        # Test 4: Notifications (no response expected)
        {
            "name": "Notifications",
            "messages": [
                {"jsonrpc":"2.0","method":"notifications/initialized"},
                {"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"test","reason":"User cancelled"}},
            ],
            "expect_no_response": True
        },
        
        # Test 5: Special characters and Unicode
        {
            "name": "Special Characters",
            "messages": [
                {"jsonrpc":"2.0","id":"s1","method":"tools/call","params":{"name":"create_graph","arguments":{"name":"test_\\n\\r\\t"}}},
                {"jsonrpc":"2.0","id":"s2","method":"tools/call","params":{"name":"create_graph","arguments":{"name":"测试_🚀_тест"}}},
            ]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print("-" * 40)
        
        # Prepare input
        if test_case.get('batch'):
            input_data = json.dumps(test_case['messages'])
        else:
            input_data = '\n'.join(json.dumps(msg) for msg in test_case['messages'])
        
        # Run test
        proc = subprocess.Popen(
            [sys.executable, "-m", "networkx_mcp", "--jsonrpc"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = proc.communicate(input=input_data, timeout=5)
        
        # Parse responses
        responses = []
        for line in stdout.strip().split('\n'):
            if line.startswith('{') or line.startswith('['):
                try:
                    responses.append(json.loads(line))
                except:
                    print(f"Failed to parse: {line[:100]}...")
        
        # Check results
        if test_case.get('expect_no_response'):
            if not responses:
                print("✅ No responses (as expected for notifications)")
            else:
                print(f"❌ Unexpected responses: {len(responses)}")
        else:
            print(f"Sent: {len(test_case['messages'])} messages")
            print(f"Received: {len(responses)} responses")
            
            # Analyze responses
            for resp in responses:
                if isinstance(resp, list):
                    # Batch response
                    print(f"  Batch response with {len(resp)} items")
                    for item in resp:
                        status = "✅" if "result" in item else "❌"
                        print(f"    {status} ID: {item.get('id')}")
                else:
                    # Single response
                    status = "✅" if "result" in resp else "❌"
                    error = f" - {resp['error']['message']}" if "error" in resp else ""
                    print(f"  {status} ID: {resp.get('id')}{error}")


def test_edge_conditions():
    """Test edge conditions."""
    print("\n\n🔧 Edge Condition Tests:")
    print("-" * 40)
    
    # Test 1: Empty input
    print("\n1. Empty input:")
    proc = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp", "--jsonrpc"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = proc.communicate(input=b'', timeout=2)
    print("✅ Handled empty input gracefully" if proc.returncode == 0 else "❌ Failed on empty input")
    
    # Test 2: Invalid JSON
    print("\n2. Invalid JSON:")
    proc = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp", "--jsonrpc"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = proc.communicate(input='{"invalid": json}\n', timeout=2)
    if '"code":-32700' in stdout:
        print("✅ Parse error returned correctly")
    else:
        print("❌ Parse error not handled properly")
    
    # Test 3: Mixed valid/invalid
    print("\n3. Mixed valid/invalid messages:")
    input_data = '''{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
{"invalid": json}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
'''
    proc = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp", "--jsonrpc"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = proc.communicate(input=input_data, timeout=2)
    
    response_count = stdout.count('"jsonrpc"')
    print(f"✅ Processed {response_count} responses (including error)" if response_count >= 3 else f"❌ Only {response_count} responses")
    
    print("\n✨ All tests completed!")


if __name__ == "__main__":
    test_json_rpc_messages()
    test_edge_conditions()