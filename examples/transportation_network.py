"""Example: Transportation Network Analysis using NetworkX MCP Server.

This example demonstrates analyzing a city transportation network,
including route optimization, flow analysis, and network resilience.
"""

import asyncio
import json
from typing import Any, Dict


# Simulated MCP client calls
async def call_tool(tool_name: str, **params) -> Dict[str, Any]:
    """Simulate calling an MCP tool."""
    print(f"\n🚊 Calling tool: {tool_name}")
    print(f"   Parameters: {json.dumps(params, indent=2)}")
    return {"success": True, "result": "simulated"}


async def create_transport_network():
    """Create a city transportation network."""
    print("=" * 60)
    print("Transportation Network Analysis")
    print("=" * 60)

    # 1. Create directed graph for transportation
    print("\n1. Creating transportation network...")
    await call_tool(
        "create_graph",
        graph_id="city_transport",
        graph_type="directed",
        attributes={
            "name": "Metro City Transport Network",
            "type": "multimodal",
            "last_updated": "2024-01-01"
        }
    )

    # 2. Add stations/intersections (nodes)
    print("\n2. Adding stations and intersections...")
    stations = [
        # Metro stations
        {"id": "Central", "type": "metro", "capacity": 10000, "lat": 40.7589, "lon": -73.9851},
        {"id": "North", "type": "metro", "capacity": 7000, "lat": 40.7614, "lon": -73.9776},
        {"id": "South", "type": "metro", "capacity": 7000, "lat": 40.7489, "lon": -73.9680},
        {"id": "East", "type": "metro", "capacity": 6000, "lat": 40.7549, "lon": -73.9740},
        {"id": "West", "type": "metro", "capacity": 6000, "lat": 40.7529, "lon": -73.9910},

        # Bus hubs
        {"id": "BusHub1", "type": "bus", "capacity": 3000, "lat": 40.7650, "lon": -73.9800},
        {"id": "BusHub2", "type": "bus", "capacity": 3000, "lat": 40.7450, "lon": -73.9750},

        # Major intersections
        {"id": "Junction1", "type": "road", "capacity": 5000, "lat": 40.7600, "lon": -73.9850},
        {"id": "Junction2", "type": "road", "capacity": 5000, "lat": 40.7500, "lon": -73.9800},
        {"id": "Junction3", "type": "road", "capacity": 4000, "lat": 40.7550, "lon": -73.9900},

        # Park & Ride
        {"id": "ParkRide1", "type": "parking", "capacity": 2000, "lat": 40.7700, "lon": -73.9950},
        {"id": "ParkRide2", "type": "parking", "capacity": 2000, "lat": 40.7400, "lon": -73.9650}
    ]

    await call_tool(
        "add_nodes",
        graph_id="city_transport",
        nodes=stations
    )

    # 3. Add routes (edges)
    print("\n3. Adding transportation routes...")
    routes = [
        # Metro lines
        {"source": "North", "target": "Central", "mode": "metro", "capacity": 5000, "time": 3, "distance": 2.1},
        {"source": "Central", "target": "South", "mode": "metro", "capacity": 5000, "time": 3, "distance": 2.3},
        {"source": "West", "target": "Central", "mode": "metro", "capacity": 4000, "time": 4, "distance": 2.5},
        {"source": "Central", "target": "East", "mode": "metro", "capacity": 4000, "time": 4, "distance": 2.4},
        {"source": "North", "target": "East", "mode": "metro", "capacity": 3000, "time": 5, "distance": 3.2},
        {"source": "West", "target": "South", "mode": "metro", "capacity": 3000, "time": 5, "distance": 3.5},

        # Bus routes
        {"source": "BusHub1", "target": "North", "mode": "bus", "capacity": 1500, "time": 10, "distance": 1.5},
        {"source": "BusHub1", "target": "Junction1", "mode": "bus", "capacity": 1500, "time": 8, "distance": 1.2},
        {"source": "Junction1", "target": "Central", "mode": "bus", "capacity": 1500, "time": 6, "distance": 0.8},
        {"source": "BusHub2", "target": "South", "mode": "bus", "capacity": 1500, "time": 10, "distance": 1.6},
        {"source": "BusHub2", "target": "Junction2", "mode": "bus", "capacity": 1500, "time": 8, "distance": 1.3},
        {"source": "Junction2", "target": "Central", "mode": "bus", "capacity": 1500, "time": 7, "distance": 1.0},

        # Road connections
        {"source": "Junction1", "target": "Junction3", "mode": "road", "capacity": 2000, "time": 15, "distance": 3.0},
        {"source": "Junction3", "target": "Junction2", "mode": "road", "capacity": 2000, "time": 12, "distance": 2.5},
        {"source": "Junction1", "target": "North", "mode": "walk", "capacity": 500, "time": 20, "distance": 0.5},
        {"source": "Junction2", "target": "South", "mode": "walk", "capacity": 500, "time": 20, "distance": 0.5},
        {"source": "Junction3", "target": "West", "mode": "walk", "capacity": 500, "time": 15, "distance": 0.4},

        # Park & Ride connections
        {"source": "ParkRide1", "target": "North", "mode": "shuttle", "capacity": 1000, "time": 5, "distance": 1.0},
        {"source": "ParkRide1", "target": "BusHub1", "mode": "shuttle", "capacity": 1000, "time": 7, "distance": 1.3},
        {"source": "ParkRide2", "target": "South", "mode": "shuttle", "capacity": 1000, "time": 5, "distance": 1.0},
        {"source": "ParkRide2", "target": "BusHub2", "mode": "shuttle", "capacity": 1000, "time": 7, "distance": 1.3},

        # Reverse routes for bidirectional travel
        {"source": "Central", "target": "North", "mode": "metro", "capacity": 5000, "time": 3, "distance": 2.1},
        {"source": "South", "target": "Central", "mode": "metro", "capacity": 5000, "time": 3, "distance": 2.3},
        {"source": "Central", "target": "West", "mode": "metro", "capacity": 4000, "time": 4, "distance": 2.5},
        {"source": "East", "target": "Central", "mode": "metro", "capacity": 4000, "time": 4, "distance": 2.4}
    ]

    await call_tool(
        "add_edges",
        graph_id="city_transport",
        edges=routes
    )


async def analyze_routes():
    """Analyze optimal routes in the network."""
    print("\n" + "=" * 60)
    print("Route Analysis")
    print("=" * 60)

    # 1. Find fastest route
    print("\n1. Finding fastest route from ParkRide1 to East station...")
    await call_tool(
        "shortest_path",
        graph_id="city_transport",
        source="ParkRide1",
        target="East",
        weight="time"
    )

    # 2. Find shortest distance route
    print("\n2. Finding shortest distance route...")
    await call_tool(
        "shortest_path",
        graph_id="city_transport",
        source="ParkRide1",
        target="East",
        weight="distance"
    )

    # 3. Find alternative routes
    print("\n3. Finding top 5 alternative routes...")
    await call_tool(
        "shortest_path",
        graph_id="city_transport",
        source="ParkRide1",
        target="East",
        weight="time",
        k_paths=5
    )

    # 4. Find all simple paths
    print("\n4. Finding all possible routes (max 6 stops)...")
    await call_tool(
        "find_all_paths",
        graph_id="city_transport",
        source="West",
        target="East",
        max_length=6,
        max_paths=20
    )


async def analyze_network_flow():
    """Analyze passenger flow capacity."""
    print("\n" + "=" * 60)
    print("Network Flow Analysis")
    print("=" * 60)

    # 1. Maximum flow analysis
    print("\n1. Analyzing maximum passenger flow from North to South...")
    await call_tool(
        "flow_paths",
        graph_id="city_transport",
        source="North",
        target="South",
        capacity="capacity",
        flow_type="maximum"
    )

    # 2. Find bottlenecks (minimum cut)
    print("\n2. Identifying network bottlenecks...")
    await call_tool(
        "flow_paths",
        graph_id="city_transport",
        source="ParkRide1",
        target="Central",
        capacity="capacity",
        flow_type="all"
    )

    # 3. Edge-disjoint paths (backup routes)
    print("\n3. Finding independent backup routes...")
    await call_tool(
        "flow_paths",
        graph_id="city_transport",
        source="West",
        target="East",
        flow_type="edge_disjoint"
    )


async def analyze_network_resilience():
    """Analyze network resilience and critical points."""
    print("\n" + "=" * 60)
    print("Network Resilience Analysis")
    print("=" * 60)

    # 1. Find critical stations
    print("\n1. Identifying critical stations (high betweenness)...")
    await call_tool(
        "calculate_centrality",
        graph_id="city_transport",
        centrality_type="betweenness",
        top_n=5
    )

    # 2. Analyze connectivity
    print("\n2. Analyzing network connectivity...")
    await call_tool(
        "connected_components",
        graph_id="city_transport",
        component_type="strongly"
    )

    # 3. Simulate station closure
    print("\n3. Simulating Central station closure...")
    # Extract network without Central station
    await call_tool(
        "subgraph_extraction",
        graph_id="city_transport",
        method="condition",
        condition="id != Central",
        create_new=True,
        new_graph_id="transport_no_central"
    )

    # Analyze impact
    await call_tool(
        "connected_components",
        graph_id="transport_no_central",
        component_type="strongly"
    )


async def analyze_by_mode():
    """Analyze network by transportation mode."""
    print("\n" + "=" * 60)
    print("Mode-Specific Analysis")
    print("=" * 60)

    # 1. Extract metro network only
    print("\n1. Extracting metro-only network...")
    await call_tool(
        "subgraph_extraction",
        graph_id="city_transport",
        method="condition",
        condition="mode = metro",
        create_new=True,
        new_graph_id="metro_network"
    )

    # 2. Analyze metro network
    print("\n2. Analyzing metro network efficiency...")
    await call_tool(
        "graph_metrics",
        graph_id="metro_network"
    )

    await call_tool(
        "path_analysis",
        graph_id="metro_network"
    )


async def optimize_network():
    """Suggest network optimizations."""
    print("\n" + "=" * 60)
    print("Network Optimization Suggestions")
    print("=" * 60)

    # 1. Find underutilized connections
    print("\n1. Analyzing edge utilization...")
    await call_tool(
        "calculate_centrality",
        graph_id="city_transport",
        centrality_type="betweenness",
        weight="capacity"
    )

    # 2. Identify potential new connections
    print("\n2. Finding stations that could benefit from direct connections...")
    await call_tool(
        "path_analysis",
        graph_id="city_transport",
        sample_size=500
    )

    # 3. Analyze clustering patterns
    print("\n3. Identifying areas with poor connectivity...")
    await call_tool(
        "clustering_analysis",
        graph_id="city_transport"
    )


async def generate_reports():
    """Generate analysis reports."""
    print("\n" + "=" * 60)
    print("Generating Reports")
    print("=" * 60)

    # 1. Overall network statistics
    print("\n1. Generating network statistics report...")
    await call_tool(
        "graph_metrics",
        graph_id="city_transport",
        include_distributions=True
    )

    # 2. Export for visualization
    print("\n2. Exporting network for visualization...")
    await call_tool(
        "visualize_graph",
        graph_id="city_transport",
        layout="kamada_kawai",
        output_format="pyvis"
    )

    # 3. Export data
    print("\n3. Exporting network data...")
    await call_tool(
        "export_graph",
        graph_id="city_transport",
        format="graphml",
        path="transport_network.graphml"
    )


async def main():
    """Run complete transportation network analysis."""
    print("\n🚇 Transportation Network Analysis with NetworkX MCP Server")
    print("=" * 60)

    # Build and analyze the network
    await create_transport_network()
    await analyze_routes()
    await analyze_network_flow()
    await analyze_network_resilience()
    await analyze_by_mode()
    await optimize_network()
    await generate_reports()

    print("\n" + "=" * 60)
    print("✅ Transportation network analysis complete!")
    print("\nKey insights derived:")
    print("- Optimal routes by time and distance")
    print("- Network capacity and bottlenecks")
    print("- Critical stations and resilience")
    print("- Mode-specific network properties")
    print("- Potential optimization opportunities")
    print("\nThis analysis can help:")
    print("- Improve route planning")
    print("- Identify infrastructure investments")
    print("- Plan for emergencies and closures")
    print("- Optimize passenger flow")


if __name__ == "__main__":
    asyncio.run(main())
