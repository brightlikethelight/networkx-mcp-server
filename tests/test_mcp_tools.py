"""Comprehensive tests for all 39 MCP tools in the NetworkX MCP Server."""

import pytest
import asyncio
import json
import networkx as nx
import numpy as np
from typing import Dict, Any
from unittest.mock import patch

from networkx_mcp.server import mcp
from networkx_mcp.core.graph_operations import GraphManager


class TestMCPToolsCore:
    """Test core graph operation MCP tools."""
    
    @pytest.fixture(autouse=True)
    def setup(self, graph_manager):
        """Setup for each test."""
        self.manager = graph_manager
    
    @pytest.mark.asyncio
    async def test_create_graph_tool(self):
        """Test create_graph MCP tool."""
        from networkx_mcp.server import create_graph
        
        # Test basic graph creation
        result = await create_graph(
            graph_id="test_graph",
            graph_type="Graph",
            params={"name": "Test Graph", "description": "A test graph"}
        )
        
        assert result["graph_id"] == "test_graph"
        assert result["graph_type"] == "Graph"
        assert result["created"] is True
        assert "created_at" in result
        
        # Test directed graph
        result = await create_graph(
            graph_id="directed_test",
            graph_type="DiGraph"
        )
        assert result["graph_type"] == "DiGraph"
        
        # Test error case - duplicate ID
        with pytest.raises(ValueError):
            await create_graph(graph_id="test_graph", graph_type="Graph")
    
    @pytest.mark.asyncio
    async def test_add_nodes_tool(self):
        """Test add_nodes MCP tool."""
        from networkx_mcp.server import create_graph, add_nodes
        
        # Create graph first
        await create_graph(graph_id="test", graph_type="Graph")
        
        # Test adding nodes
        result = await add_nodes(
            graph_id="test",
            nodes=["A", "B", "C"],
            params={"color": "red"}
        )
        
        assert result["nodes_added"] == 3
        assert result["total_nodes"] == 3
        assert set(result["added_nodes"]) == {"A", "B", "C"}
        
        # Test adding nodes with individual attributes
        result = await add_nodes(
            graph_id="test",
            nodes=[("D", {"weight": 1}), ("E", {"weight": 2})]
        )
        assert result["nodes_added"] == 2
        assert result["total_nodes"] == 5
    
    @pytest.mark.asyncio
    async def test_add_edges_tool(self):
        """Test add_edges MCP tool."""
        from networkx_mcp.server import create_graph, add_nodes, add_edges
        
        # Setup graph with nodes
        await create_graph(graph_id="test", graph_type="Graph")
        await add_nodes(graph_id="test", nodes=["A", "B", "C", "D"])
        
        # Test adding edges
        result = await add_edges(
            graph_id="test",
            edges=[("A", "B"), ("B", "C"), ("C", "D")],
            params={"weight": 1.0}
        )
        
        assert result["edges_added"] == 3
        assert result["total_edges"] == 3
        
        # Test adding edges with individual attributes
        result = await add_edges(
            graph_id="test",
            edges=[("A", "C", {"weight": 2.5}), ("B", "D", {"weight": 1.8})]
        )
        assert result["edges_added"] == 2
        assert result["total_edges"] == 5
    
    @pytest.mark.asyncio
    async def test_graph_info_tool(self):
        """Test graph_info MCP tool."""
        from networkx_mcp.server import create_graph, add_nodes, add_edges, graph_info
        
        # Create graph with data
        await create_graph(graph_id="info_test", graph_type="DiGraph")
        await add_nodes(graph_id="info_test", nodes=["X", "Y", "Z"])
        await add_edges(graph_id="info_test", edges=[("X", "Y"), ("Y", "Z")])
        
        result = await graph_info(graph_id="info_test")
        
        assert result["graph_id"] == "info_test"
        assert result["graph_type"] == "DiGraph"
        assert result["num_nodes"] == 3
        assert result["num_edges"] == 2
        assert result["is_directed"] is True
        assert "density" in result
        assert "degree_stats" in result
    
    @pytest.mark.asyncio
    async def test_list_graphs_tool(self):
        """Test list_graphs MCP tool."""
        from networkx_mcp.server import create_graph, list_graphs
        
        # Initially empty
        result = await list_graphs()
        initial_count = len(result)
        
        # Create multiple graphs
        await create_graph(graph_id="graph1", graph_type="Graph")
        await create_graph(graph_id="graph2", graph_type="DiGraph")
        
        result = await list_graphs()
        assert len(result) == initial_count + 2
        
        graph_ids = [g["graph_id"] for g in result]
        assert "graph1" in graph_ids
        assert "graph2" in graph_ids
    
    @pytest.mark.asyncio
    async def test_delete_graph_tool(self):
        """Test delete_graph MCP tool."""
        from networkx_mcp.server import create_graph, delete_graph, list_graphs
        
        # Create and delete graph
        await create_graph(graph_id="temp_graph", graph_type="Graph")
        
        result = await delete_graph(graph_id="temp_graph")
        assert result["deleted"] is True
        assert result["graph_id"] == "temp_graph"
        
        # Verify deletion
        graphs = await list_graphs()
        graph_ids = [g["graph_id"] for g in graphs]
        assert "temp_graph" not in graph_ids


class TestMCPToolsAlgorithms:
    """Test algorithm MCP tools."""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup test graph for algorithms."""
        from networkx_mcp.server import create_graph, add_nodes, add_edges
        
        # Create weighted graph for testing
        await create_graph(graph_id="algo_test", graph_type="Graph")
        await add_nodes(graph_id="algo_test", nodes=["A", "B", "C", "D", "E"])
        await add_edges(
            graph_id="algo_test",
            edges=[
                ("A", "B", {"weight": 1.0}),
                ("B", "C", {"weight": 2.0}),
                ("C", "D", {"weight": 1.5}),
                ("D", "E", {"weight": 1.0}),
                ("E", "A", {"weight": 2.5}),
                ("A", "C", {"weight": 3.0})
            ]
        )
    
    @pytest.mark.asyncio
    async def test_shortest_path_tool(self):
        """Test shortest_path MCP tool."""
        from networkx_mcp.server import shortest_path
        
        # Test single path
        result = await shortest_path(
            graph_id="algo_test",
            source="A",
            target="D",
            weight="weight"
        )
        
        assert result["source"] == "A"
        assert result["target"] == "D"
        assert "path" in result
        assert "length" in result
        assert result["path"][0] == "A"
        assert result["path"][-1] == "D"
        
        # Test all paths from source
        result = await shortest_path(
            graph_id="algo_test",
            source="A"
        )
        
        assert "paths" in result
        assert "lengths" in result
        assert "A" in result["paths"]
        assert "D" in result["paths"]
    
    @pytest.mark.asyncio
    async def test_centrality_measures_tool(self):
        """Test centrality_measures MCP tool."""
        from networkx_mcp.server import centrality_measures
        
        result = await centrality_measures(
            graph_id="algo_test",
            measures=["degree", "betweenness", "closeness"],
            top_k=3
        )
        
        assert "degree_centrality" in result
        assert "betweenness_centrality" in result
        assert "closeness_centrality" in result
        
        # Check structure
        degree_cent = result["degree_centrality"]
        assert isinstance(degree_cent, dict)
        assert all(node in ["A", "B", "C", "D", "E"] for node in degree_cent.keys())
        assert all(0 <= val <= 1 for val in degree_cent.values())
    
    @pytest.mark.asyncio
    async def test_connected_components_tool(self):
        """Test connected_components MCP tool."""
        from networkx_mcp.server import connected_components
        
        result = await connected_components(graph_id="algo_test")
        
        assert "num_components" in result
        assert "is_connected" in result
        assert "connected_components" in result
        
        # Our test graph should be connected
        assert result["is_connected"] is True
        assert result["num_components"] == 1
        assert len(result["connected_components"]) == 1
    
    @pytest.mark.asyncio
    async def test_clustering_coefficient_tool(self):
        """Test clustering_coefficient MCP tool."""
        from networkx_mcp.server import clustering_coefficient
        
        result = await clustering_coefficient(graph_id="algo_test")
        
        assert "node_clustering" in result
        assert "average_clustering" in result
        assert "transitivity" in result
        
        # Check types and ranges
        assert isinstance(result["average_clustering"], float)
        assert 0 <= result["average_clustering"] <= 1
        assert isinstance(result["node_clustering"], dict)
    
    @pytest.mark.asyncio 
    async def test_minimum_spanning_tree_tool(self):
        """Test minimum_spanning_tree MCP tool."""
        from networkx_mcp.server import minimum_spanning_tree
        
        result = await minimum_spanning_tree(
            graph_id="algo_test",
            weight="weight",
            algorithm="kruskal"
        )
        
        assert "mst_edges" in result
        assert "total_weight" in result
        assert "num_edges" in result
        
        # MST of 5 nodes should have 4 edges
        assert result["num_edges"] == 4
        assert isinstance(result["total_weight"], (int, float))
        assert result["total_weight"] > 0


class TestMCPToolsAdvanced:
    """Test Phase 2 advanced analytics MCP tools."""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup test graph for advanced algorithms."""
        from networkx_mcp.server import create_graph, add_nodes, add_edges
        
        # Create community graph
        await create_graph(graph_id="community_test", graph_type="Graph")
        
        # Add two communities
        await add_nodes(graph_id="community_test", 
                       nodes=[f"A{i}" for i in range(5)] + [f"B{i}" for i in range(5)])
        
        # Dense connections within communities
        community_a_edges = [(f"A{i}", f"A{j}") for i in range(5) for j in range(i+1, 5)]
        community_b_edges = [(f"B{i}", f"B{j}") for i in range(5) for j in range(i+1, 5)]
        # Sparse connections between communities
        inter_edges = [("A0", "B0"), ("A2", "B3")]
        
        await add_edges(graph_id="community_test", 
                       edges=community_a_edges + community_b_edges + inter_edges)
    
    @pytest.mark.asyncio
    async def test_community_detection_tool(self):
        """Test advanced_community_detection MCP tool."""
        from networkx_mcp.server import advanced_community_detection
        
        result = await advanced_community_detection(
            graph_id="community_test",
            algorithm="louvain",
            params={"resolution": 1.0}
        )
        
        assert "communities" in result
        assert "num_communities" in result
        assert "modularity" in result
        assert "algorithm_used" in result
        
        # Should detect at least 2 communities
        assert result["num_communities"] >= 2
        assert result["modularity"] > 0
        
        # Check community structure
        communities = result["communities"]
        all_nodes = set()
        for comm in communities:
            all_nodes.update(comm)
        assert len(all_nodes) == 10  # All nodes should be assigned
    
    @pytest.mark.asyncio
    async def test_generate_graph_tool(self):
        """Test generate_graph MCP tool."""
        from networkx_mcp.server import generate_graph, graph_info
        
        # Test Erdos-Renyi generation
        result = await generate_graph(
            graph_id="generated_er",
            generator_type="erdos_renyi",
            params={"n": 20, "p": 0.1, "seed": 42}
        )
        
        assert result["graph_id"] == "generated_er"
        assert result["generator_type"] == "erdos_renyi"
        assert "num_nodes" in result
        assert "num_edges" in result
        
        # Verify graph was created
        info = await graph_info(graph_id="generated_er")
        assert info["num_nodes"] == 20
        
        # Test Barabasi-Albert generation
        result = await generate_graph(
            graph_id="generated_ba",
            generator_type="barabasi_albert",
            params={"n": 50, "m": 3, "seed": 42}
        )
        
        assert result["generator_type"] == "barabasi_albert"
        info = await graph_info(graph_id="generated_ba")
        assert info["num_nodes"] == 50
    
    @pytest.mark.asyncio
    async def test_network_flow_tool(self):
        """Test network_flow MCP tool."""
        from networkx_mcp.server import create_graph, add_nodes, add_edges, network_flow
        
        # Create flow network
        await create_graph(graph_id="flow_test", graph_type="DiGraph")
        await add_nodes(graph_id="flow_test", nodes=["s", "a", "b", "t"])
        await add_edges(
            graph_id="flow_test",
            edges=[
                ("s", "a", {"capacity": 10}),
                ("s", "b", {"capacity": 8}),
                ("a", "t", {"capacity": 5}),
                ("b", "t", {"capacity": 7}),
                ("a", "b", {"capacity": 3})
            ]
        )
        
        result = await network_flow(
            graph_id="flow_test",
            source="s",
            sink="t",
            capacity="capacity"
        )
        
        assert "flow_value" in result
        assert "flow_dict" in result
        assert "source" in result
        assert "sink" in result
        
        assert result["source"] == "s"
        assert result["sink"] == "t"
        assert result["flow_value"] > 0
    
    @pytest.mark.asyncio 
    async def test_bipartite_analysis_tool(self):
        """Test bipartite_analysis MCP tool."""
        from networkx_mcp.server import create_graph, add_nodes, add_edges, bipartite_analysis
        
        # Create bipartite graph
        await create_graph(graph_id="bipartite_test", graph_type="Graph")
        await add_nodes(graph_id="bipartite_test", 
                       nodes=[("A1", {"bipartite": 0}), ("A2", {"bipartite": 0}), 
                             ("B1", {"bipartite": 1}), ("B2", {"bipartite": 1})])
        await add_edges(graph_id="bipartite_test", 
                       edges=[("A1", "B1"), ("A1", "B2"), ("A2", "B1")])
        
        result = await bipartite_analysis(graph_id="bipartite_test")
        
        assert "is_bipartite" in result
        assert "node_sets" in result
        
        # Should be bipartite
        assert result["is_bipartite"] is True
        assert len(result["node_sets"]) == 2


class TestMCPToolsVisualization:
    """Test Phase 3 visualization MCP tools."""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup graph for visualization testing."""
        from networkx_mcp.server import create_graph, add_nodes, add_edges
        
        await create_graph(graph_id="viz_test", graph_type="Graph")
        await add_nodes(
            graph_id="viz_test",
            nodes=[("A", {"color": "red"}), ("B", {"color": "blue"}), 
                  ("C", {"color": "green"}), ("D", {"color": "yellow"})]
        )
        await add_edges(
            graph_id="viz_test",
            edges=[("A", "B", {"weight": 1}), ("B", "C", {"weight": 2}), 
                  ("C", "D", {"weight": 1}), ("D", "A", {"weight": 3})]
        )
    
    @pytest.mark.asyncio
    async def test_visualize_graph_tool(self):
        """Test visualize_graph MCP tool."""
        from networkx_mcp.server import visualize_graph
        
        result = await visualize_graph(
            graph_id="viz_test",
            layout="spring",
            params={
                "node_color_attr": "color",
                "edge_width_attr": "weight",
                "show_labels": True
            }
        )
        
        assert "visualization_type" in result
        assert "layout_used" in result
        assert "formats" in result
        
        # Should have PNG format at minimum
        formats = result["formats"]
        assert "png_base64" in formats or "image_data" in result
        
        assert result["layout_used"] == "spring"
    
    @pytest.mark.asyncio
    async def test_layout_calculation_tool(self):
        """Test layout_calculation MCP tool."""
        from networkx_mcp.server import layout_calculation
        
        result = await layout_calculation(
            graph_id="viz_test",
            algorithm="circular",
            params={}
        )
        
        assert "layout" in result
        assert "algorithm_used" in result
        assert "num_nodes" in result
        
        layout = result["layout"]
        assert isinstance(layout, dict)
        assert len(layout) == 4  # Should have positions for all 4 nodes
        
        # Check position format
        for node, pos in layout.items():
            assert len(pos) == 2  # x, y coordinates
            assert all(isinstance(coord, (int, float)) for coord in pos)


class TestMCPToolsIO:
    """Test import/export MCP tools."""
    
    @pytest.mark.asyncio
    async def test_export_graph_tool(self, temp_files):
        """Test export_graph MCP tool."""
        from networkx_mcp.server import create_graph, add_nodes, add_edges, export_graph
        
        # Create graph to export
        await create_graph(graph_id="export_test", graph_type="Graph")
        await add_nodes(graph_id="export_test", nodes=["X", "Y", "Z"])
        await add_edges(graph_id="export_test", edges=[("X", "Y"), ("Y", "Z")])
        
        # Test JSON export
        result = await export_graph(
            graph_id="export_test",
            format="json"
        )
        
        assert "data" in result
        assert "format" in result
        assert "num_nodes" in result
        assert "num_edges" in result
        
        # Verify JSON structure
        data = result["data"]
        if isinstance(data, str):
            data = json.loads(data)
        
        assert "nodes" in data or "links" in data or "edges" in data
    
    @pytest.mark.asyncio
    async def test_import_graph_tool(self, temp_files):
        """Test import_graph MCP tool."""
        from networkx_mcp.server import import_graph, graph_info
        
        # Test JSON import
        result = await import_graph(
            format="json",
            path=temp_files["json"],
            graph_id="imported_test"
        )
        
        assert "graph_id" in result
        assert "format" in result
        assert "num_nodes" in result
        assert "num_edges" in result
        
        # Verify graph was created
        info = await graph_info(graph_id="imported_test")
        assert info["num_nodes"] >= 2
    
    @pytest.mark.asyncio
    async def test_data_pipeline_tool(self, temp_files):
        """Test data_pipeline MCP tool.""" 
        from networkx_mcp.server import data_pipeline, graph_info
        
        result = await data_pipeline(
            source_type="csv",
            source_path=temp_files["csv"],
            graph_id="pipeline_test",
            params={"type_inference": True}
        )
        
        assert "graph_id" in result
        assert "source_type" in result
        assert "num_nodes" in result
        assert "num_edges" in result
        
        # Verify graph was created
        info = await graph_info(graph_id="pipeline_test")
        assert info["num_nodes"] >= 2


class TestMCPToolsErrorHandling:
    """Test error handling in MCP tools."""
    
    @pytest.mark.asyncio
    async def test_invalid_graph_id_errors(self):
        """Test error handling for invalid graph IDs."""
        from networkx_mcp.server import graph_info
        
        # Test non-existent graph
        with pytest.raises(Exception):  # Should raise appropriate error
            await graph_info(graph_id="nonexistent_graph")
    
    @pytest.mark.asyncio
    async def test_invalid_parameters_errors(self):
        """Test error handling for invalid parameters."""
        from networkx_mcp.server import create_graph, centrality_measures
        
        # Test invalid graph type
        with pytest.raises(Exception):
            await create_graph(graph_id="invalid_test", graph_type="InvalidType")
        
        # Create valid graph for further testing
        await create_graph(graph_id="error_test", graph_type="Graph")
        
        # Test invalid centrality measure
        with pytest.raises(Exception):
            await centrality_measures(
                graph_id="error_test", 
                measures=["invalid_measure"]
            )
    
    @pytest.mark.asyncio
    async def test_empty_graph_operations(self):
        """Test operations on empty graphs."""
        from networkx_mcp.server import create_graph, centrality_measures, connected_components
        
        # Create empty graph
        await create_graph(graph_id="empty_test", graph_type="Graph")
        
        # Operations should handle empty graphs gracefully
        result = await connected_components(graph_id="empty_test")
        assert result["num_components"] == 0
        assert result["is_connected"] is True  # Empty graph is vacuously connected
        
        # Centrality on empty graph
        result = await centrality_measures(graph_id="empty_test", measures=["degree"])
        assert "degree_centrality" in result
        assert result["degree_centrality"] == {}


class TestMCPToolsPerformance:
    """Test performance characteristics of MCP tools."""
    
    @pytest.mark.asyncio
    async def test_large_graph_performance(self, performance_thresholds):
        """Test performance with moderately large graphs."""
        from networkx_mcp.server import create_graph, add_nodes, add_edges, centrality_measures
        import time
        
        # Create large graph
        start_time = time.time()
        await create_graph(graph_id="perf_test", graph_type="Graph")
        creation_time = time.time() - start_time
        
        assert creation_time < performance_thresholds['create_graph']
        
        # Add many nodes
        nodes = list(range(500))
        start_time = time.time()
        await add_nodes(graph_id="perf_test", nodes=nodes)
        node_time = time.time() - start_time
        
        # Should be reasonably fast
        assert node_time < 1.0
        
        # Add edges
        edges = [(i, (i + 1) % 500) for i in range(500)]
        start_time = time.time()
        await add_edges(graph_id="perf_test", edges=edges)
        edge_time = time.time() - start_time
        
        assert edge_time < 1.0
        
        # Test algorithm performance
        start_time = time.time()
        await centrality_measures(graph_id="perf_test", measures=["degree"])
        centrality_time = time.time() - start_time
        
        # Degree centrality should be very fast
        assert centrality_time < 0.1
    
    @pytest.mark.asyncio
    async def test_memory_usage_estimation(self):
        """Test memory usage tracking."""
        from networkx_mcp.server import create_graph, add_nodes, graph_info
        
        await create_graph(graph_id="memory_test", graph_type="Graph")
        
        # Add nodes with attributes
        nodes_with_attrs = [(i, {"data": "x" * 100}) for i in range(100)]
        await add_nodes(graph_id="memory_test", nodes=nodes_with_attrs)
        
        info = await graph_info(graph_id="memory_test")
        
        # Should provide memory estimates
        assert "num_nodes" in info
        assert info["num_nodes"] == 100


@pytest.mark.asyncio
async def test_all_tools_accessible():
    """Test that all 39 MCP tools are accessible and don't crash."""
    from networkx_mcp import server
    import inspect
    
    # Get all async functions that look like MCP tools
    tool_functions = []
    for name, obj in inspect.getmembers(server):
        if inspect.iscoroutinefunction(obj) and not name.startswith('_'):
            tool_functions.append(name)
    
    # Should have at least 39 tools
    assert len(tool_functions) >= 39
    
    # Each tool should be callable
    for tool_name in tool_functions:
        tool_func = getattr(server, tool_name)
        assert callable(tool_func)
        assert inspect.iscoroutinefunction(tool_func)


@pytest.mark.asyncio
async def test_tool_parameter_validation():
    """Test that tools validate their parameters properly."""
    from networkx_mcp.server import create_graph
    
    # Test type validation
    with pytest.raises((ValueError, TypeError)):
        await create_graph(graph_id=123, graph_type="Graph")  # Invalid ID type
    
    with pytest.raises((ValueError, TypeError)):
        await create_graph(graph_id="test", graph_type="InvalidType")  # Invalid graph type


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])