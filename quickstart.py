#!/usr/bin/env python3
"""Quick start script to verify NetworkX MCP Server installation and demonstrate basic usage."""

import sys
import traceback


def check_installation():
    """Check if the NetworkX MCP Server is properly installed."""
    print("🔍 Checking NetworkX MCP Server installation...")

    try:
        # Test core imports
        import networkx_mcp
        from networkx_mcp.core.algorithms import GraphAlgorithms
        from networkx_mcp.core.graph_operations import GraphManager

        print("   ✅ Core modules imported successfully")

        # Test NetworkX
        import networkx as nx

        print(f"   ✅ NetworkX version: {nx.__version__}")

        # Check optional dependencies
        optional_status = []

        try:
            import matplotlib

            optional_status.append("matplotlib ✅")
        except ImportError:
            optional_status.append(
                "matplotlib ❌ (install with: pip install matplotlib)"
            )

        try:
            import plotly

            optional_status.append("plotly ✅")
        except ImportError:
            optional_status.append("plotly ❌ (install with: pip install plotly)")

        try:
            import redis

            optional_status.append("redis ✅")
        except ImportError:
            optional_status.append("redis ❌ (install with: pip install redis)")

        print("\n📦 Optional dependencies:")
        for status in optional_status:
            print(f"   {status}")

        return True

    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   💡 Try: pip install networkx-mcp-server")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        traceback.print_exc()
        return False


def demonstrate_basic_usage():
    """Demonstrate basic graph operations."""
    print("\n🎯 Demonstrating basic usage...")

    try:
        from networkx_mcp.core.algorithms import GraphAlgorithms
        from networkx_mcp.core.graph_operations import GraphManager
        from networkx_mcp.core.io import GraphIOHandler

        # Initialize components
        manager = GraphManager()
        algo = GraphAlgorithms()
        io_handler = GraphIOHandler()

        # Create a sample social network
        print("\n1️⃣ Creating a social network graph...")
        graph_id = "social_network"
        manager.create_graph(graph_id, "Graph")

        # Add people (nodes)
        people = ["Alice", "Bob", "Charlie", "David", "Eve"]
        manager.add_nodes_from(graph_id, people)
        print(f"   Added {len(people)} people to the network")

        # Add relationships (edges)
        relationships = [
            ("Alice", "Bob"),
            ("Bob", "Charlie"),
            ("Charlie", "David"),
            ("David", "Alice"),
            ("Bob", "Eve"),
            ("Eve", "Charlie"),
        ]
        manager.add_edges_from(graph_id, relationships)
        print(f"   Added {len(relationships)} relationships")

        # Get the graph
        graph = manager.get_graph(graph_id)

        # Analyze the network
        print("\n2️⃣ Analyzing the network...")

        # Basic info
        info = manager.get_graph_info(graph_id)
        print(
            f"   Network has {info['num_nodes']} people and {info['num_edges']} connections"
        )
        print(f"   Network density: {info['density']:.2f}")

        # Find shortest path
        path = algo.shortest_path(graph, "Alice", "Eve")
        print(f"   Shortest path from Alice to Eve: {' → '.join(path)}")

        # Calculate centrality (who's most connected?)
        centrality = algo.degree_centrality(graph)
        most_connected = max(centrality.items(), key=lambda x: x[1])
        print(
            f"   Most connected person: {most_connected[0]} (centrality: {most_connected[1]:.2f})"
        )

        # Find communities
        components = algo.connected_components(graph)
        print(f"   Number of friend groups: {len(components)}")

        # Export the graph
        print("\n3️⃣ Exporting the network...")
        json_data = io_handler.export_to_json(graph)
        print(f"   Exported to JSON format with {len(json_data['nodes'])} nodes")

        # Create a more complex example
        print("\n4️⃣ Creating a transportation network...")
        transport_id = "transport_network"
        manager.create_graph(transport_id, "DiGraph")  # Directed graph for routes

        # Add cities with attributes
        cities = [
            ("NYC", {"population": 8_000_000, "type": "megacity"}),
            ("Boston", {"population": 700_000, "type": "city"}),
            ("Philadelphia", {"population": 1_500_000, "type": "city"}),
            ("Washington", {"population": 700_000, "type": "capital"}),
        ]

        for city, attrs in cities:
            manager.add_node(transport_id, city, attrs)

        # Add routes with distances
        routes = [
            ("NYC", "Boston", {"distance": 215, "time": 4}),
            ("NYC", "Philadelphia", {"distance": 95, "time": 2}),
            ("Philadelphia", "Washington", {"distance": 140, "time": 3}),
            ("Boston", "Washington", {"distance": 440, "time": 8}),
        ]

        for source, target, attrs in routes:
            manager.add_edge(transport_id, source, target, attrs)

        transport_graph = manager.get_graph(transport_id)

        # Find optimal route
        optimal_path = algo.shortest_path(
            transport_graph, "Boston", "Washington", weight="time"
        )
        print(f"   Fastest route from Boston to Washington: {' → '.join(optimal_path)}")

        # Clean up
        manager.delete_graph(graph_id)
        manager.delete_graph(transport_id)

        print("\n✅ All demonstrations completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        traceback.print_exc()
        return False


def main():
    """Run the quickstart script."""
    print("🚀 NetworkX MCP Server Quick Start")
    print("=" * 50)

    # Check installation
    if not check_installation():
        print("\n⚠️  Please fix the installation issues before continuing.")
        sys.exit(1)

    # Demonstrate usage
    if not demonstrate_basic_usage():
        print("\n⚠️  Some demonstrations failed. Check the error messages above.")
        sys.exit(1)

    print("\n🎉 Quick start completed!")
    print("\nNext steps:")
    print("1. Start the MCP server: python -m networkx_mcp.server")
    print("2. Connect with an MCP client")
    print("3. Explore the 39+ available tools")
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main()
