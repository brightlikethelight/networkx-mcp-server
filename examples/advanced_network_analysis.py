"""
Advanced Network Analysis Example

This example demonstrates the Phase 2 advanced analytics capabilities of the
NetworkX MCP Server, including community detection, network flow, ML integration,
and robustness analysis.
"""

import asyncio
import json
from typing import Dict, Any


class MCPClient:
    """Mock MCP client for demonstration purposes."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.graphs = {}
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate MCP tool calls."""
        print(f"\n🔧 Calling tool: {tool_name}")
        print(f"   Parameters: {json.dumps(params, indent=2)}")
        # In a real implementation, this would make an actual MCP call
        return {"success": True, "tool": tool_name, "params": params}


async def main():
    """Run advanced network analysis demonstration."""
    # Initialize MCP client
    client = MCPClient("http://localhost:8000")
    
    print("=" * 60)
    print("NetworkX MCP Server - Advanced Network Analysis Demo")
    print("=" * 60)
    
    # 1. Generate different types of networks
    print("\n1️⃣ GENERATING SYNTHETIC NETWORKS")
    print("-" * 40)
    
    # Generate a scale-free network (like social networks)
    result = await client.call_tool("generate_graph", {
        "graph_type": "scale_free",
        "n": 500,
        "m": 3,
        "graph_id": "social_network",
        "seed": 42
    })
    print("✅ Generated scale-free network (social network model)")
    
    # Generate a small-world network (like brain networks)
    result = await client.call_tool("generate_graph", {
        "graph_type": "small_world",
        "n": 300,
        "k": 6,
        "p": 0.3,
        "graph_id": "brain_network"
    })
    print("✅ Generated small-world network (brain network model)")
    
    # Generate a directed network for flow analysis
    result = await client.call_tool("create_graph", {
        "graph_id": "supply_chain",
        "graph_type": "directed"
    })
    
    # Add supply chain nodes and edges
    await client.call_tool("add_nodes", {
        "graph_id": "supply_chain",
        "nodes": [
            {"id": "supplier1", "type": "supplier", "capacity": 100},
            {"id": "supplier2", "type": "supplier", "capacity": 150},
            {"id": "factory1", "type": "factory"},
            {"id": "factory2", "type": "factory"},
            {"id": "warehouse1", "type": "warehouse"},
            {"id": "warehouse2", "type": "warehouse"},
            {"id": "retailer1", "type": "retailer"},
            {"id": "retailer2", "type": "retailer"},
            {"id": "retailer3", "type": "retailer"}
        ]
    })
    
    await client.call_tool("add_edges", {
        "graph_id": "supply_chain",
        "edges": [
            {"source": "supplier1", "target": "factory1", "capacity": 60},
            {"source": "supplier1", "target": "factory2", "capacity": 40},
            {"source": "supplier2", "target": "factory1", "capacity": 80},
            {"source": "supplier2", "target": "factory2", "capacity": 70},
            {"source": "factory1", "target": "warehouse1", "capacity": 100},
            {"source": "factory2", "target": "warehouse2", "capacity": 90},
            {"source": "warehouse1", "target": "retailer1", "capacity": 50},
            {"source": "warehouse1", "target": "retailer2", "capacity": 50},
            {"source": "warehouse2", "target": "retailer2", "capacity": 40},
            {"source": "warehouse2", "target": "retailer3", "capacity": 50}
        ]
    })
    print("✅ Created supply chain network")
    
    # 2. Advanced Community Detection
    print("\n2️⃣ ADVANCED COMMUNITY DETECTION")
    print("-" * 40)
    
    # Detect communities with auto-algorithm selection
    communities = await client.call_tool("advanced_community_detection", {
        "graph_id": "social_network",
        "algorithm": "auto",
        "resolution": 1.2
    })
    print("✅ Detected communities in social network")
    print("   - Algorithm auto-selected based on graph size")
    print("   - Found multiple communities with quality metrics")
    
    # Compare different community detection algorithms
    for algorithm in ["louvain", "label_propagation"]:
        result = await client.call_tool("advanced_community_detection", {
            "graph_id": "brain_network",
            "algorithm": algorithm
        })
        print(f"✅ {algorithm.title()} algorithm results analyzed")
    
    # 3. Network Flow Analysis
    print("\n3️⃣ NETWORK FLOW ANALYSIS")
    print("-" * 40)
    
    # Analyze maximum flow in supply chain
    flow_result = await client.call_tool("network_flow_analysis", {
        "graph_id": "supply_chain",
        "source": "supplier1",
        "sink": "retailer1",
        "capacity": "capacity",
        "algorithm": "auto",
        "flow_type": "max_flow"
    })
    print("✅ Analyzed maximum flow from supplier1 to retailer1")
    
    # Find minimum cut (bottlenecks)
    cut_result = await client.call_tool("network_flow_analysis", {
        "graph_id": "supply_chain",
        "source": "supplier2",
        "sink": "retailer3",
        "capacity": "capacity",
        "flow_type": "min_cut"
    })
    print("✅ Identified minimum cut (bottlenecks) in supply chain")
    
    # 4. Machine Learning Integration
    print("\n4️⃣ MACHINE LEARNING INTEGRATION")
    print("-" * 40)
    
    # Generate node embeddings for the social network
    embeddings = await client.call_tool("ml_graph_analysis", {
        "graph_id": "social_network",
        "analysis_type": "embeddings",
        "method": "node2vec",
        "dimensions": 64,
        "walk_length": 80,
        "num_walks": 10,
        "p": 1.0,
        "q": 0.5  # Bias towards BFS (structural equivalence)
    })
    print("✅ Generated Node2Vec embeddings")
    print("   - 64-dimensional embeddings for each node")
    print("   - Can be used for node classification, link prediction")
    
    # Extract graph features for ML
    features = await client.call_tool("ml_graph_analysis", {
        "graph_id": "brain_network",
        "analysis_type": "features",
        "feature_types": ["basic", "spectral", "centrality"]
    })
    print("✅ Extracted graph features for machine learning")
    
    # Detect anomalies in the network
    anomalies = await client.call_tool("ml_graph_analysis", {
        "graph_id": "social_network",
        "analysis_type": "anomaly",
        "method": "statistical",
        "contamination": 0.05  # Expect 5% anomalies
    })
    print("✅ Detected anomalous nodes")
    print("   - Found nodes with unusual connectivity patterns")
    
    # 5. Specialized Algorithms
    print("\n5️⃣ SPECIALIZED ALGORITHMS")
    print("-" * 40)
    
    # Find maximum clique
    clique = await client.call_tool("specialized_algorithms", {
        "graph_id": "brain_network",
        "algorithm": "max_clique",
        "method": "approximation"
    })
    print("✅ Found maximum clique (densely connected subgraph)")
    
    # Graph coloring for resource allocation
    coloring = await client.call_tool("specialized_algorithms", {
        "graph_id": "social_network",
        "algorithm": "coloring",
        "strategy": "dsatur"
    })
    print("✅ Colored graph (useful for scheduling, frequency assignment)")
    
    # Link prediction
    predictions = await client.call_tool("specialized_algorithms", {
        "graph_id": "social_network",
        "algorithm": "link_prediction",
        "method": "adamic_adar",
        "top_k": 10
    })
    print("✅ Predicted top 10 most likely future connections")
    
    # 6. Bipartite Analysis
    print("\n6️⃣ BIPARTITE GRAPH ANALYSIS")
    print("-" * 40)
    
    # Create a user-item bipartite graph
    await client.call_tool("create_graph", {
        "graph_id": "user_items",
        "graph_type": "undirected"
    })
    
    # Add bipartite structure
    await client.call_tool("add_edges", {
        "graph_id": "user_items",
        "edges": [
            {"source": "user1", "target": "item_a", "weight": 5},
            {"source": "user1", "target": "item_b", "weight": 3},
            {"source": "user2", "target": "item_b", "weight": 4},
            {"source": "user2", "target": "item_c", "weight": 5},
            {"source": "user3", "target": "item_a", "weight": 4},
            {"source": "user3", "target": "item_c", "weight": 2}
        ]
    })
    
    # Check if bipartite
    bipartite_check = await client.call_tool("bipartite_analysis", {
        "graph_id": "user_items",
        "analysis_type": "check"
    })
    print("✅ Verified graph is bipartite")
    
    # Find maximum matching
    matching = await client.call_tool("bipartite_analysis", {
        "graph_id": "user_items",
        "analysis_type": "matching",
        "weight": "weight"
    })
    print("✅ Found maximum weighted matching")
    
    # 7. Robustness Analysis
    print("\n7️⃣ NETWORK ROBUSTNESS ANALYSIS")
    print("-" * 40)
    
    # Simulate targeted attack
    attack_result = await client.call_tool("robustness_analysis", {
        "graph_id": "social_network",
        "analysis_type": "attack",
        "attack_type": "targeted_degree",
        "fraction": 0.1,  # Remove 10% highest degree nodes
        "measure": "largest_component"
    })
    print("✅ Simulated targeted attack on high-degree nodes")
    print("   - Measured impact on network connectivity")
    
    # Percolation analysis
    percolation = await client.call_tool("robustness_analysis", {
        "graph_id": "brain_network",
        "analysis_type": "percolation",
        "percolation_type": "site",
        "probability_range": [0.0, 1.0],
        "num_steps": 20,
        "num_trials": 10
    })
    print("✅ Found percolation threshold")
    print("   - Critical point for network breakdown")
    
    # Cascading failure simulation
    cascading = await client.call_tool("robustness_analysis", {
        "graph_id": "supply_chain",
        "analysis_type": "cascading",
        "initial_failures": ["factory1"],
        "failure_model": "threshold",
        "threshold": 0.5
    })
    print("✅ Simulated cascading failure")
    print("   - Analyzed failure propagation from factory1")
    
    # Overall resilience assessment
    resilience = await client.call_tool("robustness_analysis", {
        "graph_id": "social_network",
        "analysis_type": "resilience",
        "resilience_metrics": ["connectivity", "redundancy", "efficiency"]
    })
    print("✅ Computed comprehensive resilience metrics")
    
    # 8. Directed Graph Analysis
    print("\n8️⃣ DIRECTED GRAPH ANALYSIS")
    print("-" * 40)
    
    # Create a directed acyclic graph (DAG)
    await client.call_tool("create_graph", {
        "graph_id": "task_dag",
        "graph_type": "directed"
    })
    
    await client.call_tool("add_edges", {
        "graph_id": "task_dag",
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "A", "target": "C"},
            {"source": "B", "target": "D"},
            {"source": "C", "target": "D"},
            {"source": "D", "target": "E"}
        ]
    })
    
    # Check DAG properties
    dag_analysis = await client.call_tool("directed_graph_analysis", {
        "graph_id": "task_dag",
        "analysis_type": "dag_check"
    })
    print("✅ Verified DAG properties and found longest path")
    
    # Analyze hierarchy in supply chain
    hierarchy = await client.call_tool("directed_graph_analysis", {
        "graph_id": "supply_chain",
        "analysis_type": "hierarchy"
    })
    print("✅ Analyzed hierarchical structure of supply chain")
    
    # 9. Advanced Integration Example
    print("\n9️⃣ INTEGRATED ANALYSIS PIPELINE")
    print("-" * 40)
    
    print("Running comprehensive analysis on social network:")
    
    # Step 1: Community detection
    communities = await client.call_tool("advanced_community_detection", {
        "graph_id": "social_network",
        "algorithm": "auto"
    })
    print("  ✓ Communities detected")
    
    # Step 2: Extract subgraph of largest community
    await client.call_tool("subgraph_extraction", {
        "graph_id": "social_network",
        "method": "condition",
        "condition": "community = 0",  # Assuming community labels
        "create_new": True,
        "new_graph_id": "largest_community"
    })
    print("  ✓ Extracted largest community")
    
    # Step 3: Generate embeddings for the community
    embeddings = await client.call_tool("ml_graph_analysis", {
        "graph_id": "largest_community",
        "analysis_type": "embeddings",
        "method": "spectral",
        "dimensions": 16
    })
    print("  ✓ Generated embeddings for community members")
    
    # Step 4: Test robustness of the community
    robustness = await client.call_tool("robustness_analysis", {
        "graph_id": "largest_community",
        "analysis_type": "attack",
        "attack_type": "random",
        "fraction": 0.2
    })
    print("  ✓ Tested community robustness")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)
    print("\nThis demonstration showcased:")
    print("✅ Graph generation (scale-free, small-world, custom)")
    print("✅ Advanced community detection with auto-selection")
    print("✅ Network flow analysis (max flow, min cut)")
    print("✅ ML integration (embeddings, features, anomaly detection)")
    print("✅ Specialized algorithms (cliques, coloring, link prediction)")
    print("✅ Bipartite graph analysis")
    print("✅ Robustness analysis (attacks, percolation, cascading)")
    print("✅ Directed graph analysis (DAG, hierarchy)")
    print("✅ Integrated analysis pipelines")
    
    print("\nPhase 2 advanced analytics provide powerful tools for:")
    print("- Understanding complex network structures")
    print("- Predicting network behavior and evolution")
    print("- Identifying vulnerabilities and critical components")
    print("- Optimizing network flows and resource allocation")
    print("- Machine learning on graph data")


if __name__ == "__main__":
    asyncio.run(main())