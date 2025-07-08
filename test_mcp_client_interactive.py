#!/usr/bin/env python3
"""Interactive test of MCP protocol with persistent connection."""

import asyncio
import json
from networkx_mcp.protocol.mcp_handler import MCPProtocolHandler


async def test_mcp_interactive():
    """Test MCP protocol with persistent connection."""
    handler = MCPProtocolHandler()
    
    print("🚀 NetworkX MCP Interactive Test\n")
    
    # 1. Initialize
    print("1. Initializing connection...")
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "demo-client",
                "version": "1.0.0"
            }
        }
    }
    
    response = await handler.handle_message(json.dumps(request))
    response_data = json.loads(response)
    print(f"   Server: {response_data['result']['serverInfo']['name']} v{response_data['result']['serverInfo']['version']}")
    print(f"   Protocol: {response_data['result']['protocolVersion']}")
    
    # 2. List tools
    print("\n2. Listing available tools...")
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    
    response = await handler.handle_message(json.dumps(request))
    response_data = json.loads(response)
    tools = response_data['result']['tools']
    print(f"   Found {len(tools)} tools:")
    for tool in tools[:5]:
        print(f"   - {tool['name']}: {tool['description']}")
    print("   ...")
    
    # 3. Create a graph
    print("\n3. Creating a social network graph...")
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "create_graph",
            "arguments": {
                "name": "social_network",
                "graph_type": "undirected"
            }
        }
    }
    
    response = await handler.handle_message(json.dumps(request))
    response_data = json.loads(response)
    result = json.loads(response_data['result']['content'][0]['text'])
    print(f"   Created: {result['name']} ({result['type']})")
    
    # 4. Add nodes
    print("\n4. Adding users to network...")
    request = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "add_nodes",
            "arguments": {
                "graph_name": "social_network",
                "nodes": ["Alice", "Bob", "Charlie", "David", "Eve"]
            }
        }
    }
    
    response = await handler.handle_message(json.dumps(request))
    response_data = json.loads(response)
    result = json.loads(response_data['result']['content'][0]['text'])
    print(f"   Added {result['nodes_added']} users")
    
    # 5. Add friendships
    print("\n5. Adding friendships...")
    request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "add_edges",
            "arguments": {
                "graph_name": "social_network",
                "edges": [
                    ["Alice", "Bob"],
                    ["Alice", "Charlie"],
                    ["Bob", "Charlie"],
                    ["Bob", "David"],
                    ["Charlie", "Eve"],
                    ["David", "Eve"]
                ]
            }
        }
    }
    
    response = await handler.handle_message(json.dumps(request))
    response_data = json.loads(response)
    result = json.loads(response_data['result']['content'][0]['text'])
    print(f"   Added {result['edges_added']} friendships")
    
    # 6. Analyze the network
    print("\n6. Analyzing social network...")
    request = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "centrality_measures",
            "arguments": {
                "graph_name": "social_network",
                "measures": ["degree", "betweenness"]
            }
        }
    }
    
    response = await handler.handle_message(json.dumps(request))
    response_data = json.loads(response)
    result = json.loads(response_data['result']['content'][0]['text'])
    
    print("   Degree centrality (popularity):")
    for user, score in sorted(result['degree_centrality'].items(), key=lambda x: -x[1]):
        print(f"   - {user}: {score:.2f}")
    
    print("\n   Betweenness centrality (influence):")
    for user, score in sorted(result['betweenness_centrality'].items(), key=lambda x: -x[1]):
        print(f"   - {user}: {score:.2f}")
    
    # 7. Find shortest path
    print("\n7. Finding connection between Alice and Eve...")
    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "shortest_path",
            "arguments": {
                "graph_name": "social_network",
                "source": "Alice",
                "target": "Eve"
            }
        }
    }
    
    response = await handler.handle_message(json.dumps(request))
    response_data = json.loads(response)
    result = json.loads(response_data['result']['content'][0]['text'])
    print(f"   Path: {' -> '.join(result['path'])}")
    print(f"   Degrees of separation: {result['length']}")
    
    # 8. Test batch request
    print("\n8. Testing batch request...")
    batch_request = [
        {
            "jsonrpc": "2.0",
            "id": "batch1",
            "method": "tools/call",
            "params": {
                "name": "graph_info",
                "arguments": {"graph_name": "social_network"}
            }
        },
        {
            "jsonrpc": "2.0",
            "id": "batch2",
            "method": "tools/call",
            "params": {
                "name": "connected_components",
                "arguments": {"graph_name": "social_network"}
            }
        }
    ]
    
    response = await handler.handle_message(json.dumps(batch_request))
    batch_responses = json.loads(response)
    
    for resp in batch_responses:
        if resp['id'] == 'batch1':
            result = json.loads(resp['result']['content'][0]['text'])
            print(f"   Graph info: {result['nodes']} nodes, {result['edges']} edges")
        elif resp['id'] == 'batch2':
            result = json.loads(resp['result']['content'][0]['text'])
            print(f"   Connected components: {result['num_components']} component(s)")
    
    print("\n✨ Interactive test complete!")


if __name__ == "__main__":
    asyncio.run(test_mcp_interactive())