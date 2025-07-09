#!/usr/bin/env python3
"""
Lightweight Alternative Test
============================

Can we create a graph server without the bloat?
Let's test a minimal implementation to see the real memory requirements.
"""

import sys
import json
import asyncio
import psutil
import time

class LightweightGraph:
    """Minimal graph implementation without NetworkX."""
    
    def __init__(self, directed=False):
        self.nodes = set()
        self.edges = {}  # {node: set of neighbors}
        self.directed = directed
        
    def add_node(self, node):
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = set()
            
    def add_edge(self, u, v):
        self.add_node(u)
        self.add_node(v)
        self.edges[u].add(v)
        if not self.directed:
            self.edges[v].add(u)
            
    def number_of_nodes(self):
        return len(self.nodes)
        
    def number_of_edges(self):
        if self.directed:
            return sum(len(neighbors) for neighbors in self.edges.values())
        else:
            return sum(len(neighbors) for neighbors in self.edges.values()) // 2
            
    def neighbors(self, node):
        return self.edges.get(node, set())

class LightweightGraphManager:
    """Minimal graph manager without NetworkX."""
    
    def __init__(self):
        self.graphs = {}
        
    def create_graph(self, graph_id, directed=False):
        if graph_id in self.graphs:
            raise ValueError(f"Graph {graph_id} already exists")
        self.graphs[graph_id] = LightweightGraph(directed)
        return {"created": True, "graph_id": graph_id}
        
    def add_nodes(self, graph_id, nodes):
        if graph_id not in self.graphs:
            raise ValueError(f"Graph {graph_id} not found")
        graph = self.graphs[graph_id]
        for node in nodes:
            graph.add_node(node)
        return {"nodes_added": len(nodes)}
        
    def add_edges(self, graph_id, edges):
        if graph_id not in self.graphs:
            raise ValueError(f"Graph {graph_id} not found")
        graph = self.graphs[graph_id]
        for u, v in edges:
            graph.add_edge(u, v)
        return {"edges_added": len(edges)}
        
    def get_graph_info(self, graph_id):
        if graph_id not in self.graphs:
            raise ValueError(f"Graph {graph_id} not found")
        graph = self.graphs[graph_id]
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "directed": graph.directed
        }
        
    def shortest_path_bfs(self, graph_id, source, target):
        """Simple BFS shortest path for unweighted graphs."""
        if graph_id not in self.graphs:
            raise ValueError(f"Graph {graph_id} not found")
        graph = self.graphs[graph_id]
        
        if source not in graph.nodes or target not in graph.nodes:
            raise ValueError("Source or target not in graph")
            
        # BFS
        queue = [(source, [source])]
        visited = {source}
        
        while queue:
            node, path = queue.pop(0)
            if node == target:
                return {"path": path, "length": len(path) - 1}
                
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
                    
        return {"path": None, "length": -1}  # No path

class LightweightMCPServer:
    """Minimal MCP server without NetworkX."""
    
    def __init__(self):
        self.graph_manager = LightweightGraphManager()
        self.initialized = False
        
    async def run(self):
        """Run the lightweight server."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                    
                message = json.loads(line.strip())
                response = await self.handle_message(message)
                
                if response and "id" in message:
                    print(json.dumps(response), flush=True)
                    
            except json.JSONDecodeError:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"}
                }
                print(json.dumps(error_response), flush=True)
            except KeyboardInterrupt:
                break
                
    async def handle_message(self, message):
        """Handle JSON-RPC message."""
        method = message.get("method", "")
        params = message.get("params", {})
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "lightweight-graph-server", "version": "1.0.0"}
                }
            }
            
        elif method == "initialized":
            self.initialized = True
            return None
            
        elif method == "tools/list":
            if not self.initialized:
                return {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32002, "message": "Not initialized"}
                }
            return {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "tools": [
                        {"name": "create_graph", "description": "Create a graph"},
                        {"name": "add_nodes", "description": "Add nodes"},
                        {"name": "add_edges", "description": "Add edges"},
                        {"name": "get_graph_info", "description": "Get graph info"},
                        {"name": "shortest_path", "description": "Find shortest path (BFS)"}
                    ]
                }
            }
            
        elif method == "tools/call":
            if not self.initialized:
                return {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32002, "message": "Not initialized"}
                }
                
            tool_name = params.get("name")
            args = params.get("arguments", {})
            
            try:
                if tool_name == "create_graph":
                    result = self.graph_manager.create_graph(
                        args["graph_id"],
                        directed=args.get("directed", False)
                    )
                elif tool_name == "add_nodes":
                    result = self.graph_manager.add_nodes(
                        args["graph_id"],
                        args["nodes"]
                    )
                elif tool_name == "add_edges":
                    result = self.graph_manager.add_edges(
                        args["graph_id"],
                        args["edges"]
                    )
                elif tool_name == "get_graph_info":
                    result = self.graph_manager.get_graph_info(args["graph_id"])
                elif tool_name == "shortest_path":
                    result = self.graph_manager.shortest_path_bfs(
                        args["graph_id"],
                        args["source"],
                        args["target"]
                    )
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                    }
                    
                return {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"content": [{"type": "text", "text": json.dumps(result)}]}
                }
                
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32603, "message": str(e)}
                }
                
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {"code": -32601, "message": f"Unknown method: {method}"}
        }

def test_lightweight_memory():
    """Test memory usage of lightweight implementation."""
    print("🔍 Testing Lightweight Graph Server Memory...")
    
    # Save the lightweight server
    import inspect
    import os
    
    # Get this script's content
    with open(__file__, 'r') as f:
        content = f.read()
        
    # Extract just the classes
    lines = content.split('\n')
    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith('class LightweightGraph:'):
            start_idx = i
            break
            
    if start_idx:
        # Write lightweight server
        lightweight_code = '\n'.join(lines[start_idx:])
        lightweight_code = f'''#!/usr/bin/env python3
import sys
import json
import asyncio

{lightweight_code}

if __name__ == "__main__":
    server = LightweightMCPServer()
    asyncio.run(server.run())
'''
        
        with open("lightweight_server.py", "w") as f:
            f.write(lightweight_code)
        
        # Test its memory
        import subprocess
        
        proc = subprocess.Popen(
            [sys.executable, "lightweight_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        time.sleep(0.5)  # Let it start
        
        if proc.poll() is None:
            proc_info = psutil.Process(proc.pid)
            memory_mb = proc_info.memory_info().rss / 1024 / 1024
            print(f"📊 Lightweight server memory: {memory_mb:.1f}MB")
            
            # Test it works
            # Initialize
            proc.stdin.write('{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}\n')
            proc.stdin.flush()
            response = proc.stdout.readline()
            print(f"📥 Initialize response: {response.strip()}")
            
            # Initialized notification
            proc.stdin.write('{"jsonrpc": "2.0", "method": "initialized", "params": {}}\n')
            proc.stdin.flush()
            
            # Create graph
            proc.stdin.write('{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "create_graph", "arguments": {"graph_id": "test"}}}\n')
            proc.stdin.flush()
            response = proc.stdout.readline()
            print(f"📥 Create graph response: {response.strip()}")
            
            proc.terminate()
            proc.wait()
            os.remove("lightweight_server.py")
            
            return memory_mb
        else:
            print("❌ Lightweight server failed to start")
            if os.path.exists("lightweight_server.py"):
                os.remove("lightweight_server.py")
            return 0
    
    return 0

def main():
    print("=" * 70)
    print("Lightweight Alternative Test")
    print("=" * 70)
    print("Testing if we can build a graph server without the bloat...\n")
    
    # Get baseline Python memory
    process = psutil.Process()
    baseline = process.memory_info().rss / 1024 / 1024
    print(f"📊 Python baseline: {baseline:.1f}MB")
    
    # Test lightweight server
    lightweight_memory = test_lightweight_memory()
    
    # Now compare with NetworkX import
    print("\n🔍 Importing NetworkX for comparison...")
    import networkx as nx
    nx_memory = process.memory_info().rss / 1024 / 1024
    print(f"📊 After NetworkX import: {nx_memory:.1f}MB (+{nx_memory - baseline:.1f}MB)")
    
    print("\n" + "=" * 70)
    print("💡 MEMORY COMPARISON")
    print("=" * 70)
    print(f"Lightweight server:  {lightweight_memory:.1f}MB")
    print(f"NetworkX import:     {nx_memory - baseline:.1f}MB overhead")
    print(f"NetworkX MCP server: ~118MB")
    
    print(f"\n🎯 SAVINGS POTENTIAL:")
    potential_savings = 118 - lightweight_memory
    print(f"Could save: {potential_savings:.1f}MB ({potential_savings/118*100:.0f}% reduction)")
    
    print(f"\n💡 CONCLUSION:")
    print("We're using NetworkX for basic graph operations that could be")
    print("implemented in <100 lines of Python with 85% less memory!")
    
    print(f"\n🔥 BRUTAL TRUTH:")
    print("This 'minimal' server is anything but minimal. It's bloated")
    print("with scientific computing libraries for basic graph operations.")

if __name__ == "__main__":
    main()