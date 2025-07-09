#!/usr/bin/env python3
"""Example Python client for NetworkX MCP Server running in Docker."""

import subprocess
import json
import sys
from typing import Dict, Any, Optional


class DockerMCPClient:
    """Simple client for MCP server running in Docker."""
    
    def __init__(self, image_name: str = "networkx-mcp:0.1.0"):
        self.image_name = image_name
        self.process = None
        self.request_id = 0
    
    def start(self):
        """Start the Docker container."""
        self.process = subprocess.Popen(
            ['docker', 'run', '-i', '--rm', self.image_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    
    def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request and get response."""
        if not self.process:
            raise RuntimeError("Client not started. Call start() first.")
        
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method
        }
        
        if params:
            request["params"] = params
        
        # Send request
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        
        # Read response
        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from server")
        
        return json.loads(response_line)
    
    def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
        """Send a JSON-RPC notification (no response expected)."""
        if not self.process:
            raise RuntimeError("Client not started. Call start() first.")
        
        notification = {
            "jsonrpc": "2.0",
            "method": method
        }
        
        if params:
            notification["params"] = params
        
        self.process.stdin.write(json.dumps(notification) + '\n')
        self.process.stdin.flush()
    
    def stop(self):
        """Stop the Docker container."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None


def main():
    """Demonstrate MCP server usage with Docker."""
    client = DockerMCPClient()
    
    try:
        print("Starting NetworkX MCP Server in Docker...")
        client.start()
        
        # Initialize the server
        print("\n1. Initializing server...")
        response = client.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "docker-client", "version": "1.0.0"}
        })
        print(f"Response: {json.dumps(response, indent=2)}")
        
        # Send initialized notification
        print("\n2. Sending initialized notification...")
        client.send_notification("initialized")
        print("Notification sent (no response expected)")
        
        # List available tools
        print("\n3. Listing available tools...")
        response = client.send_request("tools/list")
        print(f"Available tools: {len(response['result']['tools'])}")
        for tool in response['result']['tools'][:3]:  # Show first 3 tools
            print(f"  - {tool['name']}: {tool['description']}")
        
        # Create a graph
        print("\n4. Creating a graph...")
        response = client.send_request("tools/call", {
            "name": "create_graph",
            "arguments": {"graph_id": "example_graph"}
        })
        print(f"Result: {response['result']['content'][0]['text']}")
        
        # Add nodes
        print("\n5. Adding nodes...")
        response = client.send_request("tools/call", {
            "name": "add_nodes",
            "arguments": {
                "graph_id": "example_graph",
                "nodes": ["A", "B", "C", "D"]
            }
        })
        print(f"Result: {response['result']['content'][0]['text']}")
        
        # Add edges
        print("\n6. Adding edges...")
        response = client.send_request("tools/call", {
            "name": "add_edges",
            "arguments": {
                "graph_id": "example_graph",
                "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["A", "D"]]
            }
        })
        print(f"Result: {response['result']['content'][0]['text']}")
        
        # Get graph info
        print("\n7. Getting graph info...")
        response = client.send_request("tools/call", {
            "name": "get_graph_info",
            "arguments": {"graph_id": "example_graph"}
        })
        info = json.loads(response['result']['content'][0]['text'])
        print(f"Graph info: {json.dumps(info, indent=2)}")
        
        # Find shortest path
        print("\n8. Finding shortest path from A to C...")
        response = client.send_request("tools/call", {
            "name": "shortest_path",
            "arguments": {
                "graph_id": "example_graph",
                "source": "A",
                "target": "C"
            }
        })
        result = json.loads(response['result']['content'][0]['text'])
        print(f"Shortest path: {' -> '.join(result['path'])}")
        print(f"Path length: {result['length']}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nStopping server...")
        client.stop()
        print("Done!")


if __name__ == "__main__":
    main()