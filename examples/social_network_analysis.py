"""Example: Social Network Analysis using NetworkX MCP Server.

This example demonstrates how to analyze a social network using the
NetworkX MCP server, including centrality analysis, community detection,
and path analysis.
"""

import asyncio
import json
from typing import Any
from typing import Dict


# Simulated MCP client calls (in practice, use actual MCP client)
async def call_tool(tool_name: str, **params) -> Dict[str, Any]:
    """Simulate calling an MCP tool."""
    print(f"\n📍 Calling tool: {tool_name}")
    print(f"   Parameters: {json.dumps(params, indent=2)}")
    # In real usage, this would make an actual MCP call
    return {"success": True, "result": "simulated"}


async def create_social_network():
    """Create and analyze a social network."""
    print("=" * 60)
    print("Social Network Analysis Example")
    print("=" * 60)

    # 1. Create the graph
    print("\n1. Creating social network graph...")
    await call_tool(
        "create_graph",
        graph_id="social_network",
        graph_type="undirected",
        attributes={"name": "Friend Network", "created": "2024-01-01"},
    )

    # 2. Add people (nodes)
    print("\n2. Adding people to the network...")
    people = [
        {"id": "Alice", "age": 28, "city": "NYC", "profession": "Engineer"},
        {"id": "Bob", "age": 32, "city": "Boston", "profession": "Designer"},
        {"id": "Charlie", "age": 25, "city": "NYC", "profession": "Data Scientist"},
        {"id": "Diana", "age": 30, "city": "Chicago", "profession": "Manager"},
        {"id": "Eve", "age": 27, "city": "NYC", "profession": "Engineer"},
        {"id": "Frank", "age": 35, "city": "Boston", "profession": "CEO"},
        {"id": "Grace", "age": 29, "city": "Chicago", "profession": "Designer"},
        {"id": "Henry", "age": 31, "city": "NYC", "profession": "Data Scientist"},
        {"id": "Iris", "age": 26, "city": "Boston", "profession": "Engineer"},
        {"id": "Jack", "age": 33, "city": "Chicago", "profession": "Manager"},
    ]

    await call_tool("add_nodes", graph_id="social_network", nodes=people)

    # 3. Add friendships (edges)
    print("\n3. Adding friendship connections...")
    friendships = [
        # Core friend group
        {"source": "Alice", "target": "Bob", "since": 2018, "strength": 0.9},
        {"source": "Alice", "target": "Charlie", "since": 2020, "strength": 0.8},
        {"source": "Bob", "target": "Charlie", "since": 2019, "strength": 0.7},
        # Extended network
        {"source": "Alice", "target": "Diana", "since": 2021, "strength": 0.6},
        {"source": "Bob", "target": "Frank", "since": 2017, "strength": 0.95},
        {"source": "Charlie", "target": "Eve", "since": 2020, "strength": 0.85},
        {"source": "Diana", "target": "Grace", "since": 2019, "strength": 0.8},
        {"source": "Diana", "target": "Jack", "since": 2018, "strength": 0.7},
        {"source": "Eve", "target": "Henry", "since": 2021, "strength": 0.75},
        {"source": "Frank", "target": "Iris", "since": 2016, "strength": 0.9},
        {"source": "Grace", "target": "Jack", "since": 2020, "strength": 0.65},
        {"source": "Henry", "target": "Iris", "since": 2022, "strength": 0.6},
        # Bridge connections
        {"source": "Charlie", "target": "Henry", "since": 2021, "strength": 0.7},
        {"source": "Frank", "target": "Jack", "since": 2019, "strength": 0.5},
        {"source": "Eve", "target": "Iris", "since": 2022, "strength": 0.55},
    ]

    await call_tool("add_edges", graph_id="social_network", edges=friendships)

    # 4. Analyze network structure
    print("\n4. Analyzing network structure...")
    await call_tool("get_graph_info", graph_id="social_network")

    await call_tool(
        "graph_metrics", graph_id="social_network", include_distributions=True
    )


async def analyze_influence():
    """Analyze influence and importance in the network."""
    print("\n" + "=" * 60)
    print("Influence Analysis")
    print("=" * 60)

    # 1. Calculate centrality measures
    print("\n1. Calculating who are the most influential people...")
    await call_tool(
        "calculate_centrality",
        graph_id="social_network",
        centrality_type=["degree", "betweenness", "closeness", "eigenvector"],
        top_n=5,
        include_statistics=True,
    )

    # 2. Find key connectors (high betweenness)
    print("\n2. Identifying key connectors in the network...")
    await call_tool(
        "calculate_centrality",
        graph_id="social_network",
        centrality_type="betweenness",
        top_n=3,
    )


async def detect_communities():
    """Detect friend groups/communities."""
    print("\n" + "=" * 60)
    print("Community Detection")
    print("=" * 60)

    # 1. Find communities
    print("\n1. Detecting friend groups...")
    await call_tool("community_detection", graph_id="social_network", method="louvain")

    # 2. Analyze clustering
    print("\n2. Analyzing how tightly knit the groups are...")
    await call_tool(
        "clustering_analysis", graph_id="social_network", include_triangles=True
    )


async def analyze_connections():
    """Analyze paths and connections between people."""
    print("\n" + "=" * 60)
    print("Connection Analysis")
    print("=" * 60)

    # 1. Find shortest path between two people
    print("\n1. Finding how Alice and Jack are connected...")
    await call_tool(
        "shortest_path", graph_id="social_network", source="Alice", target="Jack"
    )

    # 2. Find all paths between two people
    print("\n2. Finding all possible connection paths...")
    await call_tool(
        "find_all_paths",
        graph_id="social_network",
        source="Alice",
        target="Jack",
        max_length=5,
        max_paths=10,
    )

    # 3. Analyze overall connectivity
    print("\n3. Analyzing network connectivity...")
    await call_tool("path_analysis", graph_id="social_network")


async def analyze_subgroups():
    """Analyze specific subgroups in the network."""
    print("\n" + "=" * 60)
    print("Subgroup Analysis")
    print("=" * 60)

    # 1. Extract NYC subnetwork
    print("\n1. Extracting NYC residents subnetwork...")
    await call_tool(
        "subgraph_extraction",
        graph_id="social_network",
        method="condition",
        condition="city = NYC",
        create_new=True,
        new_graph_id="nyc_network",
    )

    # 2. Analyze the NYC subnetwork
    print("\n2. Analyzing NYC subnetwork...")
    await call_tool("get_graph_info", graph_id="nyc_network")

    # 3. Extract 2-hop neighborhood around a person
    print("\n3. Extracting Bob's extended friend network (2 hops)...")
    await call_tool(
        "subgraph_extraction",
        graph_id="social_network",
        method="k_hop",
        center_node="Bob",
        k_hop=2,
        create_new=True,
        new_graph_id="bob_network",
    )


async def visualize_network():
    """Generate visualization data for the network."""
    print("\n" + "=" * 60)
    print("Network Visualization")
    print("=" * 60)

    print("\n1. Generating visualization layout...")
    await call_tool(
        "visualize_graph",
        graph_id="social_network",
        layout="spring",
        output_format="pyvis",
    )

    print("\n2. Generating community-based layout...")
    await call_tool(
        "visualize_graph",
        graph_id="social_network",
        layout="kamada_kawai",
        output_format="plotly",
    )


async def export_results():
    """Export the network for further analysis."""
    print("\n" + "=" * 60)
    print("Exporting Results")
    print("=" * 60)

    print("\n1. Exporting network to JSON...")
    await call_tool(
        "export_graph",
        graph_id="social_network",
        format="json",
        path="social_network.json",
    )

    print("\n2. Exporting edge list to CSV...")
    await call_tool(
        "export_graph",
        graph_id="social_network",
        format="csv",
        path="social_network_edges.csv",
    )

    print("\n3. Getting server statistics...")
    await call_tool("monitoring_stats")


async def main():
    """Run the complete social network analysis."""
    print("\n🌐 Social Network Analysis with NetworkX MCP Server")
    print("=" * 60)

    # Create and analyze the network
    await create_social_network()
    await analyze_influence()
    await detect_communities()
    await analyze_connections()
    await analyze_subgroups()
    await visualize_network()
    await export_results()

    print("\n" + "=" * 60)
    print("✅ Social network analysis complete!")
    print("\nKey insights that could be derived:")
    print("- Most influential people (high centrality)")
    print("- Key connectors between groups (high betweenness)")
    print("- Friend groups/communities")
    print("- How tightly knit the network is (clustering)")
    print("- Connection paths between people")
    print("- Geographic or professional subnetworks")


if __name__ == "__main__":
    asyncio.run(main())
