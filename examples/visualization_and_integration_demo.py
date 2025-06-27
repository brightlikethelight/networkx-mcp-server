#!/usr/bin/env python3
"""
Phase 3 Visualization & Integration Demo for NetworkX MCP Server

This example demonstrates:
1. Data import from multiple sources (CSV, JSON, API)
2. Batch analysis of multiple graphs
3. Creating interactive visualizations
4. Building analysis workflows
5. Generating comprehensive reports
6. Setting up monitoring and alerts

Requirements:
- NetworkX MCP Server running on http://localhost:8000
- Sample data files in the examples/data/ directory
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

# In a real implementation, you would use an MCP client library
# For this demo, we'll simulate the client calls
class MCPClient:
    """Simulated MCP client for demonstration purposes."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        print(f"Connected to NetworkX MCP Server at {server_url}")
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate calling an MCP tool."""
        print(f"\n🔧 Calling tool: {tool_name}")
        print(f"   Parameters: {json.dumps(params, indent=2)}")
        # In reality, this would make an actual MCP call
        # For demo, we'll return simulated responses
        await asyncio.sleep(0.1)  # Simulate network delay
        return {"status": "success", "tool": tool_name, **params}


async def phase3_demo():
    """Demonstrate Phase 3 visualization and integration features."""
    client = MCPClient("http://localhost:8000")
    
    print("\n" + "="*80)
    print("NetworkX MCP Server - Phase 3: Visualization & Integration Demo")
    print("="*80)
    
    # Step 1: Import data from multiple sources
    print("\n📊 Step 1: Importing data from multiple sources...")
    
    # Import from CSV
    csv_result = await client.call_tool("import_from_source", {
        "source_type": "csv",
        "path": "examples/data/social_network.csv",
        "graph_id": "social_data",
        "type_inference": True,
        "edge_columns": ["user1", "user2"],
        "delimiter": ","
    })
    print(f"✅ Imported CSV: {csv_result.get('num_nodes', 'N/A')} nodes, {csv_result.get('num_edges', 'N/A')} edges")
    
    # Import from JSON
    json_result = await client.call_tool("import_from_source", {
        "source_type": "json",
        "path": "examples/data/network_structure.json",
        "graph_id": "json_network",
        "format_type": "node_link"
    })
    print(f"✅ Imported JSON: {json_result.get('format_detected', 'N/A')} format detected")
    
    # Import from database (simulated)
    db_result = await client.call_tool("import_from_source", {
        "source_type": "database",
        "path": "sqlite:///examples/data/network.db",
        "query": "SELECT source, target, weight FROM edges WHERE active = 1",
        "db_type": "sqlite",
        "graph_id": "db_network"
    })
    print(f"✅ Imported from database: Query executed successfully")
    
    # Step 2: Batch analysis of multiple graphs
    print("\n🔍 Step 2: Running batch analysis on imported graphs...")
    
    batch_result = await client.call_tool("batch_graph_analysis", {
        "graph_ids": ["social_data", "json_network", "db_network"],
        "operations": [
            {
                "name": "basic_metrics",
                "type": "metrics",
                "params": {"include_distributions": True}
            },
            {
                "name": "centrality_analysis",
                "type": "centrality",
                "params": {
                    "centrality_type": ["degree", "betweenness", "pagerank"],
                    "top_n": 5
                }
            },
            {
                "name": "community_detection",
                "type": "community",
                "params": {"algorithm": "louvain"}
            }
        ],
        "parallel": True,
        "batch_size": 10
    })
    print("✅ Batch analysis completed for all graphs")
    
    # Step 3: Create visualizations
    print("\n🎨 Step 3: Creating visualizations...")
    
    # Static visualization with custom styling
    static_viz = await client.call_tool("visualize_graph", {
        "graph_id": "social_data",
        "visualization_type": "static",
        "layout": "kamada_kawai",
        "node_size": "degree",  # Size nodes by degree
        "node_color": "community",  # Color by community
        "edge_width": 2,
        "edge_color": "gray",
        "show_labels": True,
        "title": "Social Network Structure",
        "format": "png",
        "figsize": [12, 8]
    })
    print(f"✅ Static visualization saved to: {static_viz.get('file_path', 'N/A')}")
    
    # Interactive physics-based visualization
    interactive_viz = await client.call_tool("visualize_graph", {
        "graph_id": "json_network",
        "visualization_type": "interactive",
        "layout": "force_atlas",
        "physics": True,
        "node_size": 300,
        "node_color": "lightblue",
        "edge_color": "gray",
        "show_labels": True
    })
    print(f"✅ Interactive visualization created: {interactive_viz.get('file_path', 'N/A')}")
    
    # 3D visualization
    viz_3d = await client.call_tool("visualize_3d", {
        "graph_id": "db_network",
        "layout": "spring_3d",
        "node_color": "category",
        "node_size": 8,
        "show_labels": False,
        "camera": {
            "eye": {"x": 1.5, "y": 1.5, "z": 1.5}
        }
    })
    print(f"✅ 3D visualization created: {viz_3d.get('file_path', 'N/A')}")
    
    # Specialized visualization - heatmap
    heatmap_viz = await client.call_tool("visualize_graph", {
        "graph_id": "social_data",
        "visualization_type": "specialized",
        "plot_type": "heatmap",
        "colormap": "viridis",
        "show_values": True
    })
    print("✅ Heatmap visualization created")
    
    # Step 4: Create analysis workflow
    print("\n⚙️ Step 4: Creating and executing analysis workflow...")
    
    workflow_result = await client.call_tool("create_analysis_workflow", {
        "graph_id": "social_data",
        "workflow": [
            {
                "name": "filter_low_degree",
                "operation": "filter_nodes",
                "params": {"min_degree": 3},
                "modifies_graph": True
            },
            {
                "name": "find_core",
                "operation": "k_core",
                "params": {"k": 2},
                "modifies_graph": True
            },
            {
                "name": "compute_centrality",
                "operation": "centrality",
                "params": {
                    "type": "betweenness",
                    "normalized": True,
                    "top_n": 10
                }
            },
            {
                "name": "detect_communities",
                "operation": "community",
                "params": {
                    "algorithm": "louvain",
                    "resolution": 1.2
                }
            },
            {
                "name": "calculate_metrics",
                "operation": "metrics",
                "params": {"include_distributions": True}
            }
        ],
        "cache_intermediate": True,
        "save_results": True
    })
    print(f"✅ Workflow completed: {workflow_result.get('steps_completed', 0)} steps executed")
    
    # Step 5: Generate comprehensive report
    print("\n📄 Step 5: Generating comprehensive analysis report...")
    
    report = await client.call_tool("generate_report", {
        "graph_id": "social_data",
        "format": "pdf",
        "template": "business",
        "sections": [
            "summary",
            "metrics",
            "visualizations",
            "centrality",
            "communities",
            "recommendations"
        ],
        "include_visualizations": True,
        "metadata": {
            "title": "Social Network Analysis Report",
            "author": "NetworkX MCP Analytics Team",
            "organization": "Demo Corporation",
            "description": "Comprehensive analysis of social network structure and dynamics"
        }
    })
    print(f"✅ Report generated: {report.get('file_path', 'N/A')}")
    print(f"   Format: {report.get('format', 'N/A')}, Pages: {report.get('pages', 'N/A')}")
    
    # Step 6: Create interactive dashboard
    print("\n📊 Step 6: Creating interactive dashboard...")
    
    dashboard = await client.call_tool("create_dashboard", {
        "graph_id": "social_data",
        "components": [
            {
                "type": "graph",
                "config": {
                    "visualization": "interactive",
                    "layout": "force_atlas"
                }
            },
            {
                "type": "metrics",
                "config": {
                    "metrics": ["density", "clustering", "diameter"]
                }
            },
            {
                "type": "distribution",
                "config": {
                    "plot_type": "degree_distribution",
                    "scale": "log"
                }
            },
            {
                "type": "timeline",
                "config": {
                    "metric": "num_edges",
                    "interval": "daily"
                }
            }
        ],
        "layout": "grid",
        "title": "Social Network Analytics Dashboard",
        "refresh_interval": 300  # 5 minutes
    })
    print(f"✅ Dashboard created: {dashboard.get('url', 'N/A')}")
    
    # Step 7: Setup monitoring and alerts
    print("\n🚨 Step 7: Setting up monitoring and alerts...")
    
    monitoring = await client.call_tool("setup_monitoring", {
        "graph_id": "social_data",
        "alert_rules": [
            {
                "name": "high_density_alert",
                "type": "threshold",
                "metric": "density",
                "threshold": 0.8,
                "operator": "gt",
                "severity": "medium"
            },
            {
                "name": "component_split_detection",
                "type": "pattern",
                "pattern": "component_split",
                "severity": "high"
            },
            {
                "name": "degree_anomaly",
                "type": "anomaly",
                "metric": "avg_degree",
                "sensitivity": 2.0,
                "severity": "medium"
            },
            {
                "name": "clustering_drop",
                "type": "threshold",
                "metric": "average_clustering",
                "threshold": 0.1,
                "operator": "lt",
                "severity": "low"
            }
        ],
        "check_interval": 300,  # Check every 5 minutes
        "notification_webhook": "https://alerts.example.com/networkx"
    })
    print(f"✅ Monitoring configured: {monitoring.get('rules_configured', 0)} rules active")
    print(f"   Monitoring ID: {monitoring.get('monitoring_id', 'N/A')}")
    
    # Step 8: Advanced use case - Multi-source integration with streaming
    print("\n🌊 Step 8: Demonstrating advanced integration patterns...")
    
    # Create a workflow that combines data from multiple sources
    integration_workflow = await client.call_tool("create_analysis_workflow", {
        "graph_id": "integrated_network",
        "workflow": [
            {
                "name": "merge_social_data",
                "operation": "merge_graph",
                "params": {"source_graph": "social_data"}
            },
            {
                "name": "merge_json_data",
                "operation": "merge_graph",
                "params": {"source_graph": "json_network"}
            },
            {
                "name": "merge_db_data",
                "operation": "merge_graph",
                "params": {"source_graph": "db_network"}
            },
            {
                "name": "deduplicate",
                "operation": "remove_duplicates",
                "modifies_graph": True
            },
            {
                "name": "enrich_with_ml",
                "operation": "ml_enrichment",
                "params": {
                    "embeddings": "node2vec",
                    "dimensions": 64
                }
            }
        ]
    })
    print("✅ Multi-source integration workflow completed")
    
    # Generate final summary
    print("\n" + "="*80)
    print("📈 Phase 3 Demo Summary")
    print("="*80)
    print("\nSuccessfully demonstrated:")
    print("✓ Data import from CSV, JSON, and database sources")
    print("✓ Batch analysis across multiple graphs")
    print("✓ Static, interactive, and 3D visualizations")
    print("✓ Complex analysis workflows with caching")
    print("✓ Comprehensive PDF report generation")
    print("✓ Interactive analytics dashboard creation")
    print("✓ Real-time monitoring with multiple alert types")
    print("✓ Advanced multi-source data integration")
    
    print("\n🎉 Phase 3 demonstration completed successfully!")
    print("\nKey capabilities showcased:")
    print("- Intelligent data pipelines with type inference")
    print("- Scalable batch processing with parallel execution")
    print("- Multiple visualization backends for different use cases")
    print("- Enterprise-grade reporting and monitoring")
    print("- Workflow orchestration with conditional logic")
    
    return {
        "graphs_created": 4,
        "visualizations": 4,
        "workflow_steps": 10,
        "monitoring_rules": 4,
        "report_pages": 12
    }


# Additional helper functions for advanced scenarios

async def streaming_demo(client: MCPClient):
    """Demonstrate real-time streaming data integration."""
    print("\n🌊 Streaming Data Integration Demo")
    
    # Simulate a stream of network updates
    async def generate_stream():
        """Simulate streaming network data."""
        for i in range(10):
            yield {
                "type": "edge",
                "source": f"node_{i}",
                "target": f"node_{i+1}",
                "timestamp": f"2024-01-15T12:00:{i:02d}",
                "weight": 0.5 + i * 0.1
            }
            await asyncio.sleep(0.5)
    
    # Process streaming data
    print("Processing streaming updates...")
    async for update in generate_stream():
        print(f"  → Received: {update['type']} from {update['source']} to {update['target']}")


async def enterprise_workflow_demo(client: MCPClient):
    """Demonstrate enterprise workflow with versioning and scheduling."""
    print("\n🏢 Enterprise Workflow Demo")
    
    # Create versioned analysis
    version_result = await client.call_tool("create_version", {
        "graph_id": "production_network",
        "version_name": "v1.2.0",
        "metadata": {
            "release": "stable",
            "analyzed_by": "automated_system",
            "changes": ["Added Q4 data", "Removed inactive nodes"]
        }
    })
    print(f"✅ Created version: {version_result.get('version_id', 'N/A')}")
    
    # Schedule periodic analysis
    schedule_result = await client.call_tool("schedule_analysis", {
        "job_name": "daily_network_health_check",
        "graph_id": "production_network",
        "schedule": {
            "type": "daily",
            "time": "02:00"
        },
        "operations": [
            {"type": "metrics", "params": {}},
            {"type": "anomaly_detection", "params": {"sensitivity": 2.5}},
            {"type": "report", "params": {"format": "html", "email": "team@example.com"}}
        ]
    })
    print(f"✅ Scheduled job: {schedule_result.get('job_id', 'N/A')}")


# Main execution
if __name__ == "__main__":
    print("Starting NetworkX MCP Server Phase 3 Demo...")
    
    # Create example data directory
    data_dir = Path("examples/data")
    data_dir.mkdir(exist_ok=True)
    
    # Run the main demo
    results = asyncio.run(phase3_demo())
    
    # Optional: Run additional demos
    # asyncio.run(streaming_demo(MCPClient("http://localhost:8000")))
    # asyncio.run(enterprise_workflow_demo(MCPClient("http://localhost:8000")))
    
    print("\n✨ Demo completed! Check the output files and dashboards.")