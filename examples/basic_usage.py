"""Basic usage examples for NetworkX MCP server."""

import asyncio
import json


async def demonstrate_basic_operations():
    """Demonstrate basic graph operations."""
    print("=== NetworkX MCP Server Basic Usage ===\n")

    # Note: In actual usage, these would be MCP client calls
    # This example shows the expected inputs and outputs

    # 1. Create a graph
    print("1. Creating a new graph:")
    create_request = {
        "graph_id": "social_network",
        "graph_type": "Graph",
        "name": "Example Social Network",
    }
    print(f"Request: {json.dumps(create_request, indent=2)}")
    print("Expected Response: Graph created successfully\n")

    # 2. Add nodes
    print("2. Adding nodes to the graph:")
    nodes = [
        {"id": "Alice", "age": 30, "city": "New York"},
        {"id": "Bob", "age": 25, "city": "Boston"},
        {"id": "Charlie", "age": 35, "city": "Chicago"},
        {"id": "David", "age": 28, "city": "Denver"},
        {"id": "Eve", "age": 32, "city": "Seattle"},
    ]
    add_nodes_request = {"graph_id": "social_network", "nodes": nodes}
    print(f"Request: {json.dumps(add_nodes_request, indent=2)}")
    print("Expected Response: 5 nodes added\n")

    # 3. Add edges (friendships)
    print("3. Adding edges (friendships):")
    edges = [
        {"source": "Alice", "target": "Bob", "since": 2020},
        {"source": "Alice", "target": "Charlie", "since": 2018},
        {"source": "Bob", "target": "David", "since": 2021},
        {"source": "Charlie", "target": "David", "since": 2019},
        {"source": "David", "target": "Eve", "since": 2022},
        {"source": "Charlie", "target": "Eve", "since": 2017},
    ]
    add_edges_request = {"graph_id": "social_network", "edges": edges}
    print(f"Request: {json.dumps(add_edges_request, indent=2)}")
    print("Expected Response: 6 edges added\n")


async def demonstrate_algorithms():
    """Demonstrate graph algorithms."""
    print("\n=== Graph Algorithms ===\n")

    # 1. Shortest path
    print("1. Finding shortest path:")
    shortest_path_request = {
        "graph_id": "social_network",
        "source": "Alice",
        "target": "Eve",
    }
    print(f"Request: {json.dumps(shortest_path_request, indent=2)}")
    print("Expected Response: Path: Alice -> Charlie -> Eve (length: 2)\n")

    # 2. Centrality measures
    print("2. Calculating centrality measures:")
    centrality_request = {
        "graph_id": "social_network",
        "measures": ["degree", "betweenness", "closeness"],
        "top_k": 3,
    }
    print(f"Request: {json.dumps(centrality_request, indent=2)}")
    print("Expected Response: Top 3 nodes by centrality scores\n")

    # 3. Community detection
    print("3. Detecting communities:")
    community_request = {"graph_id": "social_network", "method": "louvain"}
    print(f"Request: {json.dumps(community_request, indent=2)}")
    print("Expected Response: Communities found with modularity score\n")


async def demonstrate_advanced_features():
    """Demonstrate advanced features."""
    print("\n=== Advanced Features ===\n")

    # 1. Create weighted directed graph
    print("1. Creating a weighted directed graph (transportation network):")
    create_digraph_request = {
        "graph_id": "transport_network",
        "graph_type": "DiGraph",
        "description": "City transportation network",
    }
    print(f"Request: {json.dumps(create_digraph_request, indent=2)}")

    # Add cities and routes
    cities = ["NYC", "Boston", "Philadelphia", "Washington DC", "Baltimore"]
    routes = [
        {"source": "NYC", "target": "Boston", "distance": 215, "time": 4.5},
        {"source": "NYC", "target": "Philadelphia", "distance": 95, "time": 2},
        {
            "source": "Philadelphia",
            "target": "Washington DC",
            "distance": 140,
            "time": 3,
        },
        {"source": "Philadelphia", "target": "Baltimore", "distance": 100, "time": 2},
        {"source": "Baltimore", "target": "Washington DC", "distance": 40, "time": 1},
        {"source": "Boston", "target": "NYC", "distance": 215, "time": 4.5},
    ]
    print(f"Cities: {cities}")
    print(f"Routes: {len(routes)} directed edges\n")

    # 2. Weighted shortest path
    print("2. Finding shortest path by distance:")
    weighted_path_request = {
        "graph_id": "transport_network",
        "source": "NYC",
        "target": "Washington DC",
        "weight": "distance",
        "method": "dijkstra",
    }
    print(f"Request: {json.dumps(weighted_path_request, indent=2)}")
    print("Expected: NYC -> Philadelphia -> Washington DC (235 miles)\n")

    # 3. Export/Import operations
    print("3. Export and import graph data:")
    export_request = {"graph_id": "social_network", "format": "json"}
    print(f"Export Request: {json.dumps(export_request, indent=2)}")
    print("Expected: JSON representation of the graph\n")


async def demonstrate_visualization():
    """Demonstrate visualization features."""
    print("\n=== Visualization ===\n")

    print("1. Generate visualization data:")
    viz_request = {
        "graph_id": "social_network",
        "layout": "spring",
        "output_format": "pyvis",
    }
    print(f"Request: {json.dumps(viz_request, indent=2)}")
    print("Expected: Node positions and visualization data\n")

    print("2. Layout algorithms available:")
    layouts = ["spring", "circular", "kamada_kawai", "spectral", "random"]
    for layout in layouts:
        print(f"  - {layout}: {get_layout_description(layout)}")


def get_layout_description(layout: str) -> str:
    """Get description for layout algorithm."""
    descriptions = {
        "spring": "Force-directed layout for natural-looking graphs",
        "circular": "Nodes arranged in a circle",
        "kamada_kawai": "Optimal distance-based layout",
        "spectral": "Based on graph eigenvalues",
        "random": "Random node positions",
    }
    return descriptions.get(layout, "Custom layout algorithm")


async def demonstrate_analysis_workflow():
    """Demonstrate a complete analysis workflow."""
    print("\n\n=== Complete Analysis Workflow ===\n")

    print("Analyzing a collaboration network:")
    print("1. Load graph from file")
    print("2. Calculate basic statistics")
    print("3. Find important nodes (centrality)")
    print("4. Detect research groups (communities)")
    print("5. Analyze connectivity patterns")
    print("6. Export results\n")

    # Example workflow steps
    workflow = [
        {
            "step": "Import graph",
            "tool": "import_graph",
            "params": {"format": "graphml", "path": "collaboration.graphml"},
        },
        {
            "step": "Get statistics",
            "tool": "graph_statistics",
            "expected": "Nodes: 150, Edges: 742, Density: 0.066",
        },
        {
            "step": "Find key researchers",
            "tool": "centrality_measures",
            "params": {"measures": ["degree", "betweenness"]},
        },
        {
            "step": "Detect research groups",
            "tool": "community_detection",
            "params": {"method": "louvain"},
        },
        {
            "step": "Export for visualization",
            "tool": "export_graph",
            "params": {"format": "gexf", "path": "results.gexf"},
        },
    ]

    for i, step in enumerate(workflow, 1):
        print(f"{i}. {step['step']}")
        print(f"   Tool: {step['tool']}")
        if "params" in step:
            print(f"   Parameters: {step['params']}")
        if "expected" in step:
            print(f"   Expected: {step['expected']}")
        print()


def main():
    """Run all demonstrations."""
    print("NetworkX MCP Server - Usage Examples")
    print("====================================\n")
    print("This file demonstrates how to use the NetworkX MCP server")
    print("for various graph operations and analyses.\n")
    print("Note: These are example requests and expected responses.")
    print("In actual usage, you would use an MCP client to connect")
    print("to the running server.\n")

    # Run demonstrations
    asyncio.run(demonstrate_basic_operations())
    asyncio.run(demonstrate_algorithms())
    asyncio.run(demonstrate_advanced_features())
    asyncio.run(demonstrate_visualization())
    asyncio.run(demonstrate_analysis_workflow())

    print("\n=== Starting the Server ===\n")
    print("To start the NetworkX MCP server, run:")
    print("  python -m networkx_mcp.server")
    print("\nOr use the command:")
    print("  networkx-mcp")
    print("\nThe server will start on http://localhost:8000")
    print("\n=== Connecting with MCP Client ===\n")
    print("Use any MCP-compatible client to connect and interact")
    print("with the server using the tools demonstrated above.")


if __name__ == "__main__":
    main()
