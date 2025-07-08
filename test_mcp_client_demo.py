#!/usr/bin/env python3
"""Demonstrate MCP client interaction with NetworkX server."""

import json
import subprocess
import sys


def send_request(request):
    """Send JSON-RPC request to server and get response."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp", "--jsonrpc"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = proc.communicate(input=json.dumps(request))
    
    # Extract JSON response from stdout
    lines = stdout.strip().split('\n')
    for line in lines:
        if line.startswith('{'):
            return json.loads(line)
    
    # Debug print if no response found
    print(f"DEBUG: No JSON response found")
    print(f"STDOUT: {stdout}")
    print(f"STDERR: {stderr[:500]}...")
    
    return None


def main():
    """Run MCP client demo."""
    print("🚀 NetworkX MCP Client Demo\n")
    
    # 1. Initialize
    print("1. Initializing connection...")
    response = send_request({
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
    })
    print(f"   Server: {response['result']['serverInfo']['name']} v{response['result']['serverInfo']['version']}")
    print(f"   Protocol: {response['result']['protocolVersion']}")
    
    # 2. List tools
    print("\n2. Listing available tools...")
    response = send_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    })
    tools = response['result']['tools']
    print(f"   Found {len(tools)} tools:")
    for tool in tools[:5]:
        print(f"   - {tool['name']}: {tool['description']}")
    print("   ...")
    
    # 3. Create a graph
    print("\n3. Creating a social network graph...")
    response = send_request({
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
    })
    result = json.loads(response['result']['content'][0]['text'])
    print(f"   Created: {result['name']} ({result['type']})")
    
    # 4. Add nodes
    print("\n4. Adding users to network...")
    response = send_request({
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
    })
    result = json.loads(response['result']['content'][0]['text'])
    print(f"   Added {result['nodes_added']} users")
    
    # 5. Add friendships
    print("\n5. Adding friendships...")
    response = send_request({
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
    })
    result = json.loads(response['result']['content'][0]['text'])
    print(f"   Added {result['edges_added']} friendships")
    
    # 6. Analyze the network
    print("\n6. Analyzing social network...")
    response = send_request({
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
    })
    result = json.loads(response['result']['content'][0]['text'])
    
    print("   Degree centrality (popularity):")
    for user, score in sorted(result['degree_centrality'].items(), key=lambda x: -x[1]):
        print(f"   - {user}: {score:.2f}")
    
    print("\n   Betweenness centrality (influence):")
    for user, score in sorted(result['betweenness_centrality'].items(), key=lambda x: -x[1]):
        print(f"   - {user}: {score:.2f}")
    
    # 7. Find shortest path
    print("\n7. Finding connection between Alice and Eve...")
    response = send_request({
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
    })
    result = json.loads(response['result']['content'][0]['text'])
    print(f"   Path: {' -> '.join(result['path'])}")
    print(f"   Degrees of separation: {result['length']}")
    
    print("\n✨ Demo complete!")


if __name__ == "__main__":
    main()