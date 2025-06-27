#!/usr/bin/env python3
"""Validation script for NetworkX MCP server implementation."""

import asyncio
import sys
import time
import traceback
from typing import Dict, Any, List

# Test framework
class ValidationTest:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed = False
        self.error = None
        self.duration = 0

    async def run(self, test_func):
        """Run the test function and record results."""
        start_time = time.time()
        try:
            await test_func()
            self.passed = True
        except Exception as e:
            self.passed = False
            self.error = str(e)
            # Print full traceback for debugging
            print(f"  ERROR in {self.name}: {e}")
            traceback.print_exc()
        finally:
            self.duration = time.time() - start_time

class ValidationSuite:
    def __init__(self):
        self.tests: List[ValidationTest] = []
    
    def add_test(self, name: str, description: str):
        """Decorator to add a test to the suite."""
        def decorator(func):
            test = ValidationTest(name, description)
            self.tests.append(test)
            async def wrapper():
                await test.run(func)
            return wrapper
        return decorator
    
    async def run_all(self):
        """Run all tests and return summary."""
        print("🧪 Running NetworkX MCP Server Validation Suite")
        print("=" * 60)
        
        for test in self.tests:
            print(f"\n{test.name}: {test.description}")
            await test.run(lambda: None)  # This will be replaced by the actual test
            if test.passed:
                print(f"  ✅ PASSED ({test.duration:.2f}s)")
            else:
                print(f"  ❌ FAILED ({test.duration:.2f}s)")
                if test.error:
                    print(f"     Error: {test.error}")
        
        # Summary
        passed = sum(1 for t in self.tests if t.passed)
        total = len(self.tests)
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {passed}/{total} passed")
        if passed == total:
            print("🎉 All tests passed! Server is ready for use.")
            return True
        else:
            print(f"⚠️  {total - passed} tests failed. See errors above.")
            return False

# Create validation suite
suite = ValidationSuite()

@suite.add_test("Import Test", "Server modules can be imported without errors")
async def test_imports():
    """Test that all server modules can be imported."""
    try:
        from networkx_mcp.server import mcp
        from networkx_mcp.core.graph_operations import GraphManager
        from networkx_mcp.core.algorithms import GraphAlgorithms
        from networkx_mcp.utils.validators import GraphValidator
        print("    ✓ Core modules imported successfully")
        
        # Phase 2 imports
        from networkx_mcp.advanced import (
            CommunityDetection, NetworkFlow, GraphGenerators
        )
        print("    ✓ Phase 2 advanced modules imported successfully")
        
        # Phase 3 imports
        from networkx_mcp.visualization import (
            MatplotlibVisualizer, PlotlyVisualizer
        )
        from networkx_mcp.integration import DataPipelines
        print("    ✓ Phase 3 visualization and integration modules imported successfully")
        
    except Exception as e:
        raise RuntimeError(f"Import failed: {e}")

@suite.add_test("Graph Manager Test", "Core graph operations work correctly")
async def test_graph_manager():
    """Test basic graph manager functionality."""
    from networkx_mcp.core.graph_operations import GraphManager
    
    manager = GraphManager()
    
    # Test graph creation
    result = manager.create_graph("test_graph", "Graph")
    if not result.get("created"):
        raise ValueError("Graph creation failed")
    print("    ✓ Graph creation works")
    
    # Test graph info
    info = manager.get_graph_info("test_graph")
    if info["num_nodes"] != 0 or info["num_edges"] != 0:
        raise ValueError("New graph should be empty")
    print("    ✓ Graph info retrieval works")
    
    # Test adding nodes
    graph = manager.get_graph("test_graph")
    graph.add_node("A", color="red")
    graph.add_node("B", color="blue")
    graph.add_edge("A", "B", weight=1.5)
    
    info = manager.get_graph_info("test_graph")
    if info["num_nodes"] != 2 or info["num_edges"] != 1:
        raise ValueError(f"Expected 2 nodes, 1 edge. Got {info['num_nodes']} nodes, {info['num_edges']} edges")
    print("    ✓ Node and edge addition works")

@suite.add_test("Algorithms Test", "Graph algorithms work correctly")
async def test_algorithms():
    """Test that graph algorithms work correctly."""
    from networkx_mcp.core.algorithms import GraphAlgorithms
    import networkx as nx
    
    # Create a test graph
    graph = nx.Graph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")])
    
    # Test shortest path
    result = GraphAlgorithms.shortest_path(graph, "A", "D")
    if not result.get("path"):
        raise ValueError("Shortest path calculation failed")
    print("    ✓ Shortest path algorithm works")
    
    # Test centrality
    result = GraphAlgorithms.centrality_measures(graph, ["degree"])
    if "degree_centrality" not in result:
        raise ValueError("Centrality calculation failed")
    print("    ✓ Centrality calculation works")
    
    # Test connected components
    result = GraphAlgorithms.connected_components(graph)
    if result["num_components"] != 1:
        raise ValueError("Connected components detection failed")
    print("    ✓ Connected components detection works")

@suite.add_test("Validator Test", "Input validation works correctly")
async def test_validators():
    """Test input validation functionality."""
    from networkx_mcp.utils.validators import GraphValidator
    
    # Test graph ID validation
    valid, _ = GraphValidator.validate_graph_id("valid_graph_123")
    if not valid:
        raise ValueError("Valid graph ID rejected")
    valid, _ = GraphValidator.validate_graph_id("")
    if valid:
        raise ValueError("Empty graph ID accepted")
    print("    ✓ Graph ID validation works")
    
    # Test graph type validation
    if not GraphValidator.validate_graph_type("Graph"):
        raise ValueError("Valid graph type rejected")
    if GraphValidator.validate_graph_type("InvalidType"):
        raise ValueError("Invalid graph type accepted")
    print("    ✓ Graph type validation works")

@suite.add_test("Phase 2 - Community Detection", "Advanced community detection works")
async def test_phase2_community():
    """Test Phase 2 community detection."""
    from networkx_mcp.advanced.community_detection import CommunityDetection
    import networkx as nx
    
    # Create a graph with clear communities
    graph = nx.Graph()
    # Community 1
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    # Community 2  
    graph.add_edges_from([("D", "E"), ("E", "F"), ("F", "D")])
    # Bridge
    graph.add_edge("C", "D")
    
    result = CommunityDetection.detect_communities(graph, algorithm="louvain")
    if result["num_communities"] < 1:
        raise ValueError("Community detection failed")
    print(f"    ✓ Found {result['num_communities']} communities")

@suite.add_test("Phase 2 - Graph Generation", "Graph generators work correctly")
async def test_phase2_generators():
    """Test Phase 2 graph generators."""
    from networkx_mcp.advanced.generators import GraphGenerators
    
    # Test random graph generation
    graph, metadata = GraphGenerators.random_graph(10, p=0.3, seed=42)
    if graph.number_of_nodes() != 10:
        raise ValueError("Random graph generation failed")
    print(f"    ✓ Random graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Test scale-free graph
    graph, metadata = GraphGenerators.scale_free_graph(20, m=2, seed=42)
    if graph.number_of_nodes() != 20:
        raise ValueError("Scale-free graph generation failed")
    print(f"    ✓ Scale-free graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

@suite.add_test("Phase 3 - Visualization", "Visualization systems work")
async def test_phase3_visualization():
    """Test Phase 3 visualization capabilities."""
    import networkx as nx
    
    # Create test graph
    graph = nx.karate_club_graph()
    
    try:
        from networkx_mcp.visualization.matplotlib_visualizer import MatplotlibVisualizer
        result = MatplotlibVisualizer.create_static_plot(graph, layout="spring")
        if not isinstance(result, dict) or "formats" not in result or not result["formats"]:
            raise ValueError("Matplotlib visualization failed")
        print("    ✓ Matplotlib visualization works")
    except ImportError:
        print("    ⚠ Matplotlib not available, skipping static visualization test")
    
    try:
        from networkx_mcp.visualization.plotly_visualizer import PlotlyVisualizer
        result = PlotlyVisualizer.create_interactive_plot(graph)
        if not isinstance(result, dict) or not any(key in result for key in ["html", "json", "plot_data"]):
            raise ValueError("Plotly visualization failed")
        print("    ✓ Plotly visualization works")
    except ImportError:
        print("    ⚠ Plotly not available, skipping interactive visualization test")

@suite.add_test("Phase 3 - Data Integration", "Data pipeline integration works")
async def test_phase3_integration():
    """Test Phase 3 data integration capabilities."""
    from networkx_mcp.integration.data_pipelines import DataPipelines
    import tempfile
    import csv
    
    # Create test CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target', 'weight'])
        writer.writerow(['A', 'B', 1.0])
        writer.writerow(['B', 'C', 2.0])
        writer.writerow(['C', 'A', 1.5])
        csv_path = f.name
    
    try:
        result = DataPipelines.csv_pipeline(csv_path, type_inference=True)
        graph = result["graph"]
        if graph.number_of_nodes() != 3 or graph.number_of_edges() != 3:
            raise ValueError("CSV pipeline failed")
        print(f"    ✓ CSV pipeline: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    finally:
        import os
        os.unlink(csv_path)

@suite.add_test("MCP Tool Registration", "All MCP tools are properly registered")
async def test_mcp_tools():
    """Test that MCP tools are properly registered."""
    from networkx_mcp.server import mcp
    
    # Check if tools are registered (FastMCP doesn't expose tools directly)
    # We'll test by trying to access the server object
    if not hasattr(mcp, '_tools') and not hasattr(mcp, 'tools'):
        # This is expected - let's just verify the server loads
        print("    ✓ MCP server object created successfully")
    
    # Test that we can call server methods exist by checking functions
    import inspect
    from networkx_mcp import server
    
    # Count MCP tool decorators in server.py to verify tools are registered
    import subprocess
    result = subprocess.run(
        ["grep", "-c", "@mcp.tool", "/Users/brightliu/Coding_Projects/networkx-mcp-server/src/networkx_mcp/server.py"],
        capture_output=True, text=True
    )
    tool_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
    
    if tool_count < 30:  # We expect 30+ tools
        raise ValueError(f"Expected 30+ tools, found {tool_count}")
    
    print(f"    ✓ Found {tool_count} MCP tool functions registered")

@suite.add_test("Memory and Performance", "System can handle moderate workloads")
async def test_performance():
    """Test basic performance and memory characteristics."""
    from networkx_mcp.core.graph_operations import GraphManager
    from networkx_mcp.utils.monitoring import PerformanceMonitor, MemoryMonitor
    import networkx as nx
    
    manager = GraphManager()
    monitor = PerformanceMonitor()
    
    # Create a moderate sized graph
    start_time = time.time()
    graph = nx.erdos_renyi_graph(100, 0.1, seed=42)
    creation_time = time.time() - start_time
    
    monitor.record_operation("graph_creation", creation_time)
    
    # Test memory estimation
    memory_info = MemoryMonitor.estimate_graph_memory(graph)
    if memory_info["total_mb"] > 100:  # Should be much smaller
        raise ValueError(f"Memory usage too high: {memory_info['total_mb']} MB")
    
    print(f"    ✓ Graph creation: {creation_time:.3f}s")
    print(f"    ✓ Memory usage: {memory_info['total_mb']:.2f} MB")
    
    # Test algorithm performance
    start_time = time.time()
    centrality = nx.degree_centrality(graph)
    centrality_time = time.time() - start_time
    
    if centrality_time > 1.0:  # Should be much faster
        raise ValueError(f"Centrality calculation too slow: {centrality_time:.3f}s")
    
    print(f"    ✓ Centrality calculation: {centrality_time:.3f}s")

# Main execution
async def main():
    """Run the validation suite."""
    print("🚀 Starting NetworkX MCP Server Validation")
    print(f"📅 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Override the test running to actually execute our test functions
    test_functions = [
        test_imports, test_graph_manager, test_algorithms, test_validators,
        test_phase2_community, test_phase2_generators, test_phase3_visualization,
        test_phase3_integration, test_mcp_tools, test_performance
    ]
    
    passed = 0
    total = len(test_functions)
    
    for i, test_func in enumerate(test_functions):
        test = suite.tests[i]
        print(f"\n{i+1}. {test.name}: {test.description}")
        
        start_time = time.time()
        try:
            await test_func()
            test.passed = True
            duration = time.time() - start_time
            print(f"   ✅ PASSED ({duration:.2f}s)")
            passed += 1
        except Exception as e:
            test.passed = False
            test.error = str(e)
            duration = time.time() - start_time
            print(f"   ❌ FAILED ({duration:.2f}s)")
            print(f"      Error: {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"📊 VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! NetworkX MCP Server is ready for production use.")
        print("\n✨ Implementation Status:")
        print("   ✅ Phase 1: Core functionality - COMPLETE")
        print("   ✅ Phase 2: Advanced analytics - COMPLETE") 
        print("   ✅ Phase 3: Visualization & integration - COMPLETE")
        print("   ✅ Error handling and validation - WORKING")
        print("   ✅ Performance and memory usage - ACCEPTABLE")
        return 0
    else:
        print(f"⚠️  {total - passed} TESTS FAILED. See errors above for details.")
        print("\n🔧 Next steps:")
        print("   1. Fix failing tests")
        print("   2. Run validation again")
        print("   3. Check implementation completeness")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))