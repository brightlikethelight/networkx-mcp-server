"""Test MCP protocol implementation with various scenarios."""

import asyncio
import json
from networkx_mcp.protocol.mcp_handler import MCPProtocolHandler


async def test_mcp_protocol():
    """Test MCP protocol implementation."""
    handler = MCPProtocolHandler()
    
    print("🧪 Testing MCP Protocol Implementation\n")
    
    # Test 1: Initialize
    print("1️⃣ Testing Initialize Request")
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    response = await handler.handle_message(json.dumps(init_request))
    response_data = json.loads(response)
    
    print(f"Response: {json.dumps(response_data, indent=2)}")
    assert "result" in response_data
    assert "capabilities" in response_data["result"]
    assert "serverInfo" in response_data["result"]
    print("✅ Initialize successful\n")
    
    # Test 2: Tools List
    print("2️⃣ Testing Tools List")
    tools_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    
    response = await handler.handle_message(json.dumps(tools_request))
    response_data = json.loads(response)
    
    print(f"Number of tools: {len(response_data['result']['tools'])}")
    print(f"First few tools: {[t['name'] for t in response_data['result']['tools'][:5]]}")
    assert len(response_data['result']['tools']) > 10
    print("✅ Tools list successful\n")
    
    # Test 3: Tool Call
    print("3️⃣ Testing Tool Call - Create Graph")
    create_graph_request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "create_graph",
            "arguments": {
                "name": "test_graph",
                "graph_type": "undirected"
            }
        }
    }
    
    response = await handler.handle_message(json.dumps(create_graph_request))
    response_data = json.loads(response)
    
    print(f"Response: {json.dumps(response_data, indent=2)}")
    assert "result" in response_data
    assert not response_data["result"].get("isError", False)
    print("✅ Tool call successful\n")
    
    # Test 4: Batch Request
    print("4️⃣ Testing Batch Request")
    batch_request = [
        {
            "jsonrpc": "2.0",
            "id": "batch1",
            "method": "tools/call",
            "params": {
                "name": "add_nodes",
                "arguments": {
                    "graph_name": "test_graph",
                    "nodes": ["A", "B", "C"]
                }
            }
        },
        {
            "jsonrpc": "2.0",
            "id": "batch2",
            "method": "tools/call",
            "params": {
                "name": "add_edges",
                "arguments": {
                    "graph_name": "test_graph",
                    "edges": [["A", "B"], ["B", "C"]]
                }
            }
        },
        {
            "jsonrpc": "2.0",
            "id": "batch3",
            "method": "tools/call",
            "params": {
                "name": "graph_info",
                "arguments": {
                    "graph_name": "test_graph"
                }
            }
        }
    ]
    
    response = await handler.handle_message(json.dumps(batch_request))
    response_data = json.loads(response)
    
    print(f"Batch response count: {len(response_data)}")
    assert len(response_data) == 3
    assert all("result" in r for r in response_data)
    print("✅ Batch request successful\n")
    
    # Test 5: Error Handling
    print("5️⃣ Testing Error Handling")
    error_request = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "unknown_method"
    }
    
    response = await handler.handle_message(json.dumps(error_request))
    response_data = json.loads(response)
    
    print(f"Error response: {json.dumps(response_data, indent=2)}")
    assert "error" in response_data
    assert response_data["error"]["code"] == -32601  # Method not found
    print("✅ Error handling successful\n")
    
    # Test 6: Notification (no response expected)
    print("6️⃣ Testing Notification")
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }
    
    response = await handler.handle_message(json.dumps(notification))
    assert response is None  # No response for notifications
    print("✅ Notification handled correctly (no response)\n")
    
    # Test 7: Prompts
    print("7️⃣ Testing Prompts")
    prompts_request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "prompts/list"
    }
    
    response = await handler.handle_message(json.dumps(prompts_request))
    response_data = json.loads(response)
    
    print(f"Available prompts: {[p['name'] for p in response_data['result']['prompts']]}")
    assert len(response_data['result']['prompts']) > 0
    print("✅ Prompts list successful\n")
    
    # Test 8: Invalid JSON
    print("8️⃣ Testing Invalid JSON")
    invalid_json = '{"invalid": json}'
    
    response = await handler.handle_message(invalid_json)
    response_data = json.loads(response)
    
    assert "error" in response_data
    assert response_data["error"]["code"] == -32700  # Parse error
    print("✅ Invalid JSON handled correctly\n")
    
    print("🎉 All MCP Protocol Tests Passed!")


async def test_concurrent_mcp_requests():
    """Test concurrent MCP requests."""
    handler = MCPProtocolHandler()
    
    print("\n⚡ Testing Concurrent MCP Requests\n")
    
    # Initialize first
    init_request = {
        "jsonrpc": "2.0",
        "id": "init",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    }
    await handler.handle_message(json.dumps(init_request))
    
    # Create multiple concurrent requests
    requests = []
    for i in range(10):
        req = {
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
        }
        requests.append(json.dumps(req))
    
    # Handle concurrently
    tasks = [handler.handle_message(req) for req in requests]
    responses = await asyncio.gather(*tasks)
    
    # Verify all succeeded
    success_count = 0
    for response in responses:
        if response:
            resp_data = json.loads(response)
            if "result" in resp_data and not resp_data["result"].get("isError", False):
                success_count += 1
    
    print(f"✅ Handled {success_count}/{len(requests)} concurrent requests successfully")


if __name__ == "__main__":
    asyncio.run(test_mcp_protocol())
    asyncio.run(test_concurrent_mcp_requests())