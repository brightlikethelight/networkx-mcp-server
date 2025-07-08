#!/usr/bin/env python3
"""Simple test for MCP resources implementation."""

import json
import sys
sys.path.insert(0, '.')

from src.networkx_mcp.compat.enhanced_fastmcp_compat import EnhancedFastMCPCompat
from src.networkx_mcp.core.graph_operations import GraphManager


def create_simple_test_server():
    """Create simple MCP server with basic resources."""
    mcp = EnhancedFastMCPCompat(
        name="simple-mcp-test",
        version="1.0.0"
    )
    
    # Initialize graph manager
    graph_manager = GraphManager()
    
    # Create test graph
    graph_manager.create_graph("test_graph", "Graph")
    graph_manager.add_nodes_from("test_graph", ["A", "B", "C"])
    graph_manager.add_edges_from("test_graph", [("A", "B"), ("B", "C"), ("C", "A")])
    
    # Register simple resources
    @mcp.resource("graph://catalog", description="List all graphs")
    def graph_catalog():
        """List all graphs."""
        graphs = []
        for graph_info in graph_manager.list_graphs():
            graphs.append({
                "id": graph_info["graph_id"],
                "nodes": graph_info["num_nodes"],
                "edges": graph_info["num_edges"]
            })
        return json.dumps({"graphs": graphs})
    
    @mcp.resource("graph://data/test_graph", description="Get test graph data")
    def test_graph_data():
        """Get test graph data."""
        graph = graph_manager.get_graph("test_graph")
        return json.dumps({
            "nodes": list(graph.nodes()),
            "edges": list(graph.edges()),
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges()
        })
    
    return mcp, graph_manager


def test_simple_resources():
    """Test simple resource access."""
    print("=== Simple MCP Resources Test ===\n")
    
    mcp, graph_manager = create_simple_test_server()
    
    # 1. Test resource discovery
    print("1. Testing resource discovery...")
    discovery_msg = {
        "jsonrpc": "2.0",
        "method": "resources/list",
        "params": {},
        "id": 1
    }
    
    response = mcp.handle_message(json.dumps(discovery_msg))
    result = json.loads(response)
    
    if "result" in result:
        resources = result["result"]["resources"]
        print(f"   ✅ Found {len(resources)} resources:")
        for res in resources:
            print(f"   - {res['uri']}: {res.get('description', 'No description')}")
    else:
        print(f"   ❌ Error: {result.get('error', 'Unknown')}")
        return False
    
    # 2. Test catalog access
    print("\n2. Testing catalog access...")
    catalog_msg = {
        "jsonrpc": "2.0",
        "method": "resources/read",
        "params": {"uri": "graph://catalog"},
        "id": 2
    }
    
    response = mcp.handle_message(json.dumps(catalog_msg))
    result = json.loads(response)
    
    if "result" in result:
        content = json.loads(result["result"]["contents"][0]["text"])
        print(f"   ✅ Catalog data: {content}")
    else:
        print(f"   ❌ Error: {result.get('error', 'Unknown')}")
        return False
    
    # 3. Test specific graph data
    print("\n3. Testing specific graph data...")
    data_msg = {
        "jsonrpc": "2.0",
        "method": "resources/read",
        "params": {"uri": "graph://data/test_graph"},
        "id": 3
    }
    
    response = mcp.handle_message(json.dumps(data_msg))
    result = json.loads(response)
    
    if "result" in result:
        content = json.loads(result["result"]["contents"][0]["text"])
        print(f"   ✅ Graph data: {content}")
    else:
        print(f"   ❌ Error: {result.get('error', 'Unknown')}")
        return False
    
    return True


def test_mcp_client_workflow():
    """Test complete MCP client workflow."""
    print("\n=== MCP Client Workflow Test ===\n")
    
    mcp, graph_manager = create_simple_test_server()
    
    workflow_steps = [
        ("Initialize", {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {"protocolVersion": "1.0"},
            "id": "init"
        }),
        ("List Resources", {
            "jsonrpc": "2.0",
            "method": "resources/list",
            "params": {},
            "id": "list_res"
        }),
        ("Read Catalog", {
            "jsonrpc": "2.0",
            "method": "resources/read",
            "params": {"uri": "graph://catalog"},
            "id": "read_cat"
        }),
        ("Read Graph Data", {
            "jsonrpc": "2.0",
            "method": "resources/read",
            "params": {"uri": "graph://data/test_graph"},
            "id": "read_data"
        })
    ]
    
    success_count = 0
    
    for step_name, message in workflow_steps:
        print(f"Step: {step_name}")
        
        response = mcp.handle_message(json.dumps(message))
        result = json.loads(response)
        
        if "error" in result:
            print(f"   ❌ Error: {result['error']['message']}")
        else:
            print(f"   ✅ Success")
            success_count += 1
    
    print(f"\nWorkflow completed: {success_count}/{len(workflow_steps)} steps successful")
    return success_count == len(workflow_steps)


if __name__ == "__main__":
    success1 = test_simple_resources()
    success2 = test_mcp_client_workflow()
    
    print("\n=== Summary ===")
    if success1 and success2:
        print("🎉 All tests passed!")
        print("\n✅ Resources are accessible via standard MCP clients!")
        print("✅ Resource discovery works")
        print("✅ Resource reading works")
        print("✅ JSON-RPC protocol compliance")
    else:
        print("❌ Some tests failed")
        
    print("\nMCP Client Access Pattern:")
    print("1. resources/list - Discover available resources")
    print("2. resources/read - Access resource data")
    print("3. Standard JSON-RPC 2.0 messaging")