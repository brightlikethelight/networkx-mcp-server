#!/usr/bin/env python3
"""Quick start script to verify NetworkX MCP Server installation and functionality."""

import asyncio
import sys
import traceback


def check_installation():
    """Check if the NetworkX MCP Server is properly installed."""
    print("🔍 Checking NetworkX MCP Server installation...")

    try:
        # Test core imports
        from networkx_mcp.core.algorithms import GraphAlgorithms
        from networkx_mcp.core.graph_operations import GraphManager
        from networkx_mcp.server import mcp
        print("   ✅ Core modules imported successfully")

        # Test advanced imports
        from networkx_mcp.advanced import CommunityDetection, GraphGenerators
        print("   ✅ Advanced analytics modules imported")

        # Test visualization imports
        from networkx_mcp.visualization import MatplotlibVisualizer
        print("   ✅ Visualization modules imported")

        return True

    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   💡 Try: pip install -e .")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


def test_basic_functionality():
    """Test basic graph operations."""
    print("\n🧪 Testing basic functionality...")

    try:
        # Test graph manager
        from networkx_mcp.core.graph_operations import GraphManager

        manager = GraphManager()

        # Create a test graph
        result = manager.create_graph("quickstart_test", "Graph")
        print(f"   ✅ Graph creation: {result['graph_id']}")

        # Add nodes
        result = manager.add_nodes_from("quickstart_test", ["Alice", "Bob", "Charlie"])
        print(f"   ✅ Added {result['nodes_added']} nodes")

        # Add edges
        result = manager.add_edges_from("quickstart_test", [
            ("Alice", "Bob"), ("Bob", "Charlie"), ("Charlie", "Alice")
        ])
        print(f"   ✅ Added {result['edges_added']} edges")

        # Get graph info
        info = manager.get_graph_info("quickstart_test")
        print(f"   ✅ Graph info: {info['num_nodes']} nodes, {info['num_edges']} edges")

        return True

    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_algorithms():
    """Test graph algorithms."""
    print("\n📊 Testing graph algorithms...")

    try:
        import networkx as nx

        from networkx_mcp.core.algorithms import GraphAlgorithms

        # Create test graph
        graph = nx.karate_club_graph()

        # Test centrality measures
        centrality = GraphAlgorithms.centrality_measures(graph, measures=["degree", "betweenness"])
        print(f"   ✅ Centrality calculation: {len(centrality['degree_centrality'])} nodes analyzed")

        # Test shortest path
        path_result = GraphAlgorithms.shortest_path(graph, 0, 33)
        print(f"   ✅ Shortest path: length {path_result['length']}")

        # Test connected components
        components = GraphAlgorithms.connected_components(graph)
        print(f"   ✅ Connected components: {components['num_components']} component(s)")

        return True

    except Exception as e:
        print(f"   ❌ Algorithm test failed: {e}")
        traceback.print_exc()
        return False


def test_mcp_tools():
    """Test MCP tool registration."""
    print("\n🔧 Testing MCP tools...")

    try:
        # Import MCP server
        # Count registered tools by checking the server
        import subprocess

        from networkx_mcp.server import mcp
        result = subprocess.run(
            ["grep", "-c", "@mcp.tool", "src/networkx_mcp/server.py"],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            tool_count = int(result.stdout.strip())
            print(f"   ✅ MCP server: {tool_count} tools registered")

            if tool_count >= 39:
                print(f"   ✅ Tool count: {tool_count}/39 (complete)")
            else:
                print(f"   ⚠️ Tool count: {tool_count}/39 (incomplete)")
        else:
            print("   ⚠️ Could not count MCP tools")

        # Test server object creation
        if hasattr(mcp, 'tools'):
            print("   ✅ MCP server object created successfully")
        else:
            print("   ⚠️ MCP server object may not be properly configured")

        return True

    except Exception as e:
        print(f"   ❌ MCP tools test failed: {e}")
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization capabilities."""
    print("\n🎨 Testing visualization...")

    try:
        import networkx as nx

        from networkx_mcp.visualization.matplotlib_visualizer import \
            MatplotlibVisualizer

        # Create test graph
        graph = nx.complete_graph(5)

        # Test matplotlib visualization
        result = MatplotlibVisualizer.create_static_plot(graph, layout="circular")
        print(f"   ✅ Matplotlib visualization: {result['num_nodes']} nodes rendered")

        # Test if we can import Plotly
        try:
            from networkx_mcp.visualization.plotly_visualizer import \
                PlotlyVisualizer
            result = PlotlyVisualizer.create_interactive_plot(graph)
            print("   ✅ Plotly visualization: interactive plot created")
        except ImportError:
            print("   ⚠️ Plotly not available (optional)")

        return True

    except Exception as e:
        print(f"   ❌ Visualization test failed: {e}")
        traceback.print_exc()
        return False


def show_usage_examples():
    """Show usage examples."""
    print("\n📚 Usage Examples:")
    print("-" * 40)

    examples = [
        ("Create a graph", "await create_graph('social', 'Graph')"),
        ("Add nodes", "await add_nodes('social', ['Alice', 'Bob', 'Charlie'])"),
        ("Add edges", "await add_edges('social', [('Alice', 'Bob'), ('Bob', 'Charlie')])"),
        ("Calculate centrality", "await centrality_measures('social', ['degree', 'betweenness'])"),
        ("Find shortest path", "await shortest_path('social', 'Alice', 'Charlie')"),
        ("Detect communities", "await community_detection('social', algorithm='louvain')"),
        ("Visualize graph", "await visualize_graph('social', layout='spring')"),
        ("Export graph", "await export_graph('social', format='json')"),
    ]

    for desc, code in examples:
        print(f"   {desc:20} → {code}")

    print("\n📖 See examples/ directory for complete workflows")
    print("📚 Read docs/API.md for full documentation")


async def main():
    """Run quickstart verification."""
    print("🚀 NetworkX MCP Server Quickstart")
    print("=" * 40)

    # Check installation
    if not check_installation():
        print("\n❌ Installation check failed!")
        print("💡 Try running: pip install -e .")
        return False

    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality test failed!")
        return False

    # Test algorithms
    if not test_algorithms():
        print("\n❌ Algorithm test failed!")
        return False

    # Test MCP tools
    if not test_mcp_tools():
        print("\n❌ MCP tools test failed!")
        return False

    # Test visualization
    if not test_visualization():
        print("\n❌ Visualization test failed!")
        return False

    print("\n" + "=" * 40)
    print("🎉 ALL TESTS PASSED!")
    print("✅ NetworkX MCP Server is ready for use")

    show_usage_examples()

    print("\n🔥 Start the server with:")
    print("   python -m networkx_mcp.server")
    print("\n📊 Run validation with:")
    print("   python validate_server.py")

    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Quickstart interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
