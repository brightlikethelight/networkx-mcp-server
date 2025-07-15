"""
Tests for new graph operations added in Phase 4.

These tests verify the enhanced functionality works correctly.
"""

import pytest
import json
from networkx_mcp.server_minimal import (
    create_graph, add_nodes, add_edges, 
    degree_centrality, betweenness_centrality,
    connected_components, pagerank, community_detection,
    visualize_graph, import_csv, export_json,
    graphs
)


class TestCentralityOperations:
    """Test centrality algorithms."""
    
    def setup_method(self):
        """Clear graphs before each test."""
        graphs.clear()
    
    def test_degree_centrality(self):
        """Test degree centrality calculation."""
        create_graph("test_degree", directed=False)
        add_nodes("test_degree", ["A", "B", "C", "D", "E"])
        add_edges("test_degree", [["A", "B"], ["A", "C"], ["A", "D"], ["B", "C"], ["D", "E"]])
        
        result = degree_centrality("test_degree")
        
        assert "centrality" in result
        assert "most_central" in result
        # Node A has 3 connections out of 4 possible (n-1)
        assert result["most_central"][0] == "A"
        assert result["most_central"][1] == 0.75  # 3/4
    
    def test_betweenness_centrality(self):
        """Test betweenness centrality calculation."""
        create_graph("test_between", directed=False)
        # Create a graph where B is a bridge between two clusters
        add_nodes("test_between", ["A", "B", "C", "D", "E"])
        add_edges("test_between", [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"], ["E", "C"]])
        
        result = betweenness_centrality("test_between")
        
        assert "centrality" in result
        assert "most_central" in result
        # C should have high betweenness as it's on many shortest paths
        assert result["most_central"][0] == "C"
    
    def test_pagerank(self):
        """Test PageRank calculation."""
        create_graph("test_pagerank", directed=True)
        add_nodes("test_pagerank", ["A", "B", "C", "D"])
        # Create a directed graph where many point to A
        add_edges("test_pagerank", [["B", "A"], ["C", "A"], ["D", "A"], ["A", "B"]])
        
        result = pagerank("test_pagerank")
        
        assert "pagerank" in result
        assert "highest_rank" in result
        # A should have highest PageRank as it receives most links
        assert result["highest_rank"][0] == "A"


class TestGraphStructureOperations:
    """Test graph structure analysis."""
    
    def setup_method(self):
        """Clear graphs before each test."""
        graphs.clear()
    
    def test_connected_components_undirected(self):
        """Test connected components in undirected graph."""
        create_graph("test_components", directed=False)
        # Create two separate components
        add_nodes("test_components", [1, 2, 3, 4, 5, 6])
        add_edges("test_components", [[1, 2], [2, 3], [4, 5], [5, 6]])
        
        result = connected_components("test_components")
        
        assert result["num_components"] == 2
        assert result["component_sizes"] == [3, 3]
        assert len(result["largest_component"]) == 3
    
    def test_connected_components_directed(self):
        """Test weakly connected components in directed graph."""
        create_graph("test_directed_comp", directed=True)
        add_nodes("test_directed_comp", ["A", "B", "C", "D"])
        add_edges("test_directed_comp", [["A", "B"], ["C", "D"]])
        
        result = connected_components("test_directed_comp")
        
        assert result["num_components"] == 2
        assert result["component_sizes"] == [2, 2]
    
    def test_community_detection(self):
        """Test community detection."""
        create_graph("test_communities", directed=False)
        # Create two clear communities
        add_nodes("test_communities", list(range(1, 9)))
        # Community 1: 1-4 fully connected
        add_edges("test_communities", [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
        # Community 2: 5-8 fully connected
        add_edges("test_communities", [[5, 6], [5, 7], [5, 8], [6, 7], [6, 8], [7, 8]])
        # Weak link between communities
        add_edges("test_communities", [[4, 5]])
        
        result = community_detection("test_communities")
        
        assert result["num_communities"] >= 2
        assert "community_sizes" in result
        assert "largest_community" in result
        assert "node_community_map" in result


class TestVisualization:
    """Test graph visualization."""
    
    def setup_method(self):
        """Clear graphs before each test."""
        graphs.clear()
    
    def test_visualize_graph_spring_layout(self):
        """Test graph visualization with spring layout."""
        create_graph("test_viz", directed=False)
        add_nodes("test_viz", ["A", "B", "C", "D"])
        add_edges("test_viz", [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]])
        
        result = visualize_graph("test_viz", layout="spring")
        
        assert "image" in result
        assert result["image"].startswith("data:image/png;base64,")
        assert result["format"] == "png"
        assert result["layout"] == "spring"
    
    def test_visualize_graph_circular_layout(self):
        """Test graph visualization with circular layout."""
        create_graph("test_viz_circular", directed=True)
        add_nodes("test_viz_circular", [1, 2, 3, 4, 5])
        add_edges("test_viz_circular", [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1]])
        
        result = visualize_graph("test_viz_circular", layout="circular")
        
        assert "image" in result
        assert result["image"].startswith("data:image/png;base64,")
        assert result["layout"] == "circular"


class TestImportExport:
    """Test import/export functionality."""
    
    def setup_method(self):
        """Clear graphs before each test."""
        graphs.clear()
    
    def test_import_csv_undirected(self):
        """Test importing graph from CSV."""
        csv_data = """1,2
2,3
3,4
4,1"""
        
        result = import_csv("test_import", csv_data, directed=False)
        
        assert result["imported"] == "test_import"
        assert result["type"] == "undirected"
        assert result["nodes"] == 4
        assert result["edges"] == 4
    
    def test_import_csv_directed(self):
        """Test importing directed graph from CSV."""
        csv_data = """A,B
B,C
C,A"""
        
        result = import_csv("test_import_dir", csv_data, directed=True)
        
        assert result["imported"] == "test_import_dir"
        assert result["type"] == "directed"
        assert result["nodes"] == 3
        assert result["edges"] == 3
    
    def test_export_json(self):
        """Test exporting graph as JSON."""
        create_graph("test_export", directed=False)
        add_nodes("test_export", ["X", "Y", "Z"])
        add_edges("test_export", [["X", "Y"], ["Y", "Z"]])
        
        result = export_json("test_export")
        
        assert "graph_data" in result
        assert result["format"] == "node-link"
        assert result["nodes"] == 3
        assert result["edges"] == 2
        
        # Verify the structure
        data = result["graph_data"]
        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) == 3
        assert len(data["links"]) == 2


class TestErrorHandlingNew:
    """Test error handling for new operations."""
    
    def setup_method(self):
        """Clear graphs before each test."""
        graphs.clear()
    
    def test_centrality_nonexistent_graph(self):
        """Test centrality operations on non-existent graph."""
        with pytest.raises(ValueError, match="Graph 'nonexistent' not found"):
            degree_centrality("nonexistent")
        
        with pytest.raises(ValueError, match="Graph 'nonexistent' not found"):
            betweenness_centrality("nonexistent")
        
        with pytest.raises(ValueError, match="Graph 'nonexistent' not found"):
            pagerank("nonexistent")
    
    def test_empty_graph_operations(self):
        """Test operations on empty graph."""
        create_graph("empty", directed=False)
        
        # These should work but return empty results
        result = degree_centrality("empty")
        assert result["most_central"] is None
        
        result = connected_components("empty")
        assert result["num_components"] == 0
        
        result = community_detection("empty")
        assert result["num_communities"] == 0