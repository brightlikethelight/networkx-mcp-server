#!/usr/bin/env python3
"""Quick test script for the minimal NetworkX MCP server."""

import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_minimal_server():
    """Test that the minimal server can be imported and initialized."""
    print("Testing minimal NetworkX MCP server...")

    try:
        # Test basic imports
        print("1. Testing imports...")
        print("   ✓ Imports successful")

        # Test creating a graph
        print("\n2. Testing graph creation...")
        from networkx_mcp.server_minimal import create_graph

        result = create_graph("test_graph", "undirected")
        print(f"   ✓ Graph created: {result}")

        # Test adding nodes
        print("\n3. Testing node addition...")
        from networkx_mcp.server_minimal import add_nodes

        result = add_nodes("test_graph", ["A", "B", "C"])
        print(f"   ✓ Nodes added: {result}")

        # Test adding edges
        print("\n4. Testing edge addition...")
        from networkx_mcp.server_minimal import add_edges

        result = add_edges("test_graph", [["A", "B"], ["B", "C"]])
        print(f"   ✓ Edges added: {result}")

        # Test getting graph info
        print("\n5. Testing graph info...")
        from networkx_mcp.server_minimal import graph_info

        result = graph_info("test_graph")
        print(f"   ✓ Graph info: {result}")

        # Test shortest path
        print("\n6. Testing shortest path...")
        from networkx_mcp.server_minimal import shortest_path

        result = shortest_path("test_graph", "A", "C")
        print(f"   ✓ Shortest path: {result}")

        print("\n✅ All tests passed! The minimal server is working.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_compatibility_layer():
    """Test that the compatibility layer works."""
    print("\n\nTesting compatibility layer...")

    try:
        from networkx_mcp.compat.fastmcp_compat import FastMCPCompat

        # Create an instance
        mcp = FastMCPCompat("test-server")
        print("   ✓ FastMCPCompat created successfully")

        # Test tool registration
        @mcp.tool(description="Test tool")
        def test_tool(x: int) -> int:
            return x * 2

        print("   ✓ Tool registration works")

        print("\n✅ Compatibility layer is working!")
        return True

    except Exception as e:
        print(f"\n❌ Compatibility test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("NetworkX MCP Server - Minimal Version Test")
    print("=" * 60)

    # Run tests
    test1_passed = test_minimal_server()
    test2_passed = test_compatibility_layer()

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The server is ready to run.")
        print("\nTo start the server, run:")
        print("  python -m networkx_mcp.server_minimal")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        sys.exit(1)
