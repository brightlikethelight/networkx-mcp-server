"""Comprehensive tests for graph operations."""

import pytest
import networkx as nx
import numpy as np
import time
from networkx_mcp.core.graph_operations import GraphManager
from networkx_mcp.utils.validators import GraphValidator
from networkx_mcp.utils.monitoring import PerformanceMonitor, OperationCounter


class TestGraphManager:
    """Test GraphManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = GraphManager()
    
    def test_create_graph(self):
        """Test graph creation."""
        # Test basic graph creation
        result = self.manager.create_graph("test1", "Graph")
        assert result["graph_id"] == "test1"
        assert result["graph_type"] == "Graph"
        assert result["created"] is True
        
        # Test directed graph
        result = self.manager.create_graph("test2", "DiGraph")
        assert result["graph_type"] == "DiGraph"
        assert isinstance(self.manager.graphs["test2"], nx.DiGraph)
        
        # Test with attributes
        result = self.manager.create_graph("test3", "Graph", name="TestGraph", weight=1.0)
        graph = self.manager.get_graph("test3")
        assert graph.graph["name"] == "TestGraph"
        assert graph.graph["weight"] == 1.0
        
        # Test duplicate graph ID
        with pytest.raises(ValueError):
            self.manager.create_graph("test1", "Graph")
    
    def test_get_graph(self):
        """Test graph retrieval."""
        self.manager.create_graph("test", "Graph")
        graph = self.manager.get_graph("test")
        assert isinstance(graph, nx.Graph)
        
        # Test non-existent graph
        with pytest.raises(KeyError):
            self.manager.get_graph("nonexistent")
    
    def test_delete_graph(self):
        """Test graph deletion."""
        self.manager.create_graph("test", "Graph")
        result = self.manager.delete_graph("test")
        assert result["deleted"] is True
        assert "test" not in self.manager.graphs
        
        # Test deleting non-existent graph
        with pytest.raises(KeyError):
            self.manager.delete_graph("nonexistent")
    
    def test_list_graphs(self):
        """Test listing graphs."""
        # Empty list
        assert self.manager.list_graphs() == []
        
        # Create some graphs
        self.manager.create_graph("g1", "Graph")
        self.manager.create_graph("g2", "DiGraph")
        
        graphs = self.manager.list_graphs()
        assert len(graphs) == 2
        assert any(g["graph_id"] == "g1" for g in graphs)
        assert any(g["graph_id"] == "g2" for g in graphs)
    
    def test_add_node(self):
        """Test adding nodes."""
        self.manager.create_graph("test", "Graph")
        
        # Add simple node
        result = self.manager.add_node("test", "node1")
        assert result["added"] is True
        assert result["node_id"] == "node1"
        
        # Add node with attributes
        result = self.manager.add_node("test", "node2", color="red", weight=5)
        graph = self.manager.get_graph("test")
        assert graph.nodes["node2"]["color"] == "red"
        assert graph.nodes["node2"]["weight"] == 5
    
    def test_add_nodes_from(self):
        """Test adding multiple nodes."""
        self.manager.create_graph("test", "Graph")
        
        # Add list of nodes
        nodes = ["a", "b", "c"]
        result = self.manager.add_nodes_from("test", nodes)
        assert result["nodes_added"] == 3
        assert result["total_nodes"] == 3
        
        # Add nodes with attributes
        nodes_with_attrs = [("d", {"color": "blue"}), ("e", {"weight": 10})]
        result = self.manager.add_nodes_from("test", nodes_with_attrs)
        assert result["total_nodes"] == 5
        
        graph = self.manager.get_graph("test")
        assert graph.nodes["d"]["color"] == "blue"
        assert graph.nodes["e"]["weight"] == 10
    
    def test_add_edge(self):
        """Test adding edges."""
        self.manager.create_graph("test", "Graph")
        self.manager.add_node("test", "A")
        self.manager.add_node("test", "B")
        
        # Add simple edge
        result = self.manager.add_edge("test", "A", "B")
        assert result["added"] is True
        assert result["edge"] == ("A", "B")
        
        # Add edge with attributes
        self.manager.add_node("test", "C")
        result = self.manager.add_edge("test", "B", "C", weight=2.5, color="green")
        
        graph = self.manager.get_graph("test")
        assert graph.edges["B", "C"]["weight"] == 2.5
        assert graph.edges["B", "C"]["color"] == "green"
    
    def test_add_edges_from(self):
        """Test adding multiple edges."""
        self.manager.create_graph("test", "Graph")
        nodes = ["A", "B", "C", "D"]
        self.manager.add_nodes_from("test", nodes)
        
        # Add list of edges
        edges = [("A", "B"), ("B", "C"), ("C", "D")]
        result = self.manager.add_edges_from("test", edges)
        assert result["edges_added"] == 3
        assert result["total_edges"] == 3
        
        # Add edges with attributes
        edges_with_attrs = [
            ("A", "C", {"weight": 1.5}),
            ("B", "D", {"weight": 2.0})
        ]
        result = self.manager.add_edges_from("test", edges_with_attrs)
        assert result["total_edges"] == 5
        
        graph = self.manager.get_graph("test")
        assert graph.edges["A", "C"]["weight"] == 1.5
    
    def test_remove_node(self):
        """Test removing nodes."""
        self.manager.create_graph("test", "Graph")
        self.manager.add_nodes_from("test", ["A", "B", "C"])
        self.manager.add_edges_from("test", [("A", "B"), ("B", "C")])
        
        result = self.manager.remove_node("test", "B")
        assert result["removed"] is True
        
        graph = self.manager.get_graph("test")
        assert "B" not in graph
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 0  # Edges with B are removed
        
        # Test removing non-existent node
        with pytest.raises(ValueError):
            self.manager.remove_node("test", "X")
    
    def test_remove_edge(self):
        """Test removing edges."""
        self.manager.create_graph("test", "Graph")
        self.manager.add_nodes_from("test", ["A", "B", "C"])
        self.manager.add_edges_from("test", [("A", "B"), ("B", "C")])
        
        result = self.manager.remove_edge("test", "A", "B")
        assert result["removed"] is True
        
        graph = self.manager.get_graph("test")
        assert not graph.has_edge("A", "B")
        assert graph.has_edge("B", "C")
        
        # Test removing non-existent edge
        with pytest.raises(ValueError):
            self.manager.remove_edge("test", "A", "C")
    
    def test_get_graph_info(self):
        """Test getting graph information."""
        self.manager.create_graph("test", "DiGraph")
        self.manager.add_nodes_from("test", ["A", "B", "C", "D"])
        self.manager.add_edges_from("test", [
            ("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")
        ])
        
        info = self.manager.get_graph_info("test")
        assert info["graph_id"] == "test"
        assert info["graph_type"] == "DiGraph"
        assert info["num_nodes"] == 4
        assert info["num_edges"] == 4
        assert info["is_directed"] is True
        assert info["is_multigraph"] is False
        assert "degree_stats" in info
    
    def test_get_neighbors(self):
        """Test getting node neighbors."""
        self.manager.create_graph("test", "Graph")
        self.manager.add_nodes_from("test", ["A", "B", "C", "D"])
        self.manager.add_edges_from("test", [
            ("A", "B"), ("A", "C"), ("B", "D")
        ])
        
        neighbors = self.manager.get_neighbors("test", "A")
        assert set(neighbors) == {"B", "C"}
        
        neighbors = self.manager.get_neighbors("test", "B")
        assert set(neighbors) == {"A", "D"}
        
        # Test non-existent node
        with pytest.raises(ValueError):
            self.manager.get_neighbors("test", "X")
    
    def test_node_attributes(self):
        """Test node attribute operations."""
        self.manager.create_graph("test", "Graph")
        self.manager.add_node("test", "A", color="red", weight=1)
        self.manager.add_node("test", "B", color="blue", weight=2)
        
        # Get single node attributes
        attrs = self.manager.get_node_attributes("test", "A")
        assert attrs["color"] == "red"
        assert attrs["weight"] == 1
        
        # Get all nodes with specific attribute
        colors = self.manager.get_node_attributes("test", attribute="color")
        assert colors == {"A": "red", "B": "blue"}
        
        # Set node attributes
        result = self.manager.set_node_attributes(
            "test",
            {"A": {"color": "green", "size": 10}, "B": {"size": 20}}
        )
        assert result["nodes_updated"] == 2
        
        graph = self.manager.get_graph("test")
        assert graph.nodes["A"]["color"] == "green"
        assert graph.nodes["A"]["size"] == 10
        assert graph.nodes["B"]["size"] == 20
    
    def test_edge_attributes(self):
        """Test edge attribute operations."""
        self.manager.create_graph("test", "Graph")
        self.manager.add_nodes_from("test", ["A", "B", "C"])
        self.manager.add_edge("test", "A", "B", weight=1.5, color="red")
        self.manager.add_edge("test", "B", "C", weight=2.5, color="blue")
        
        # Get single edge attributes
        attrs = self.manager.get_edge_attributes("test", ("A", "B"))
        assert attrs["weight"] == 1.5
        assert attrs["color"] == "red"
        
        # Get all edges with specific attribute
        weights = self.manager.get_edge_attributes("test", attribute="weight")
        assert weights[("A", "B")] == 1.5
        assert weights[("B", "C")] == 2.5
        
        # Set edge attributes
        result = self.manager.set_edge_attributes(
            "test",
            {("A", "B"): {"weight": 3.0}, ("B", "C"): {"color": "green"}}
        )
        assert result["edges_updated"] == 2
        
        graph = self.manager.get_graph("test")
        assert graph.edges["A", "B"]["weight"] == 3.0
        assert graph.edges["B", "C"]["color"] == "green"
    
    def test_subgraph(self):
        """Test subgraph creation."""
        self.manager.create_graph("test", "Graph")
        self.manager.add_nodes_from("test", ["A", "B", "C", "D", "E"])
        self.manager.add_edges_from("test", [
            ("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")
        ])
        
        # Create subgraph copy
        result = self.manager.subgraph("test", ["A", "B", "C"], create_copy=True)
        assert result["num_nodes"] == 3
        assert result["num_edges"] == 2
        assert set(result["nodes"]) == {"A", "B", "C"}
        
        # Original graph should be unchanged
        graph = self.manager.get_graph("test")
        assert graph.number_of_nodes() == 5
    
    def test_clear_graph(self):
        """Test clearing a graph."""
        self.manager.create_graph("test", "Graph")
        self.manager.add_nodes_from("test", ["A", "B", "C"])
        self.manager.add_edges_from("test", [("A", "B"), ("B", "C")])
        
        result = self.manager.clear_graph("test")
        assert result["cleared"] is True
        assert result["num_nodes"] == 0
        assert result["num_edges"] == 0
        
        graph = self.manager.get_graph("test")
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0


class TestPerformance:
    """Test performance with large graphs."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = GraphManager()
        self.monitor = PerformanceMonitor()
    
    def test_large_graph_creation(self):
        """Test creating and manipulating large graphs."""
        # Create large graph
        self.manager.create_graph("large", "Graph")
        graph = self.manager.get_graph("large")
        
        # Add many nodes
        start_time = time.time()
        nodes = list(range(1000))
        self.manager.add_nodes_from("large", nodes)
        node_time = time.time() - start_time
        
        assert graph.number_of_nodes() == 1000
        assert node_time < 1.0  # Should be fast
        
        # Add many edges
        start_time = time.time()
        edges = [(i, (i + 1) % 1000) for i in range(1000)]
        edges.extend([(i, (i + 2) % 1000) for i in range(0, 1000, 2)])
        self.manager.add_edges_from("large", edges)
        edge_time = time.time() - start_time
        
        assert graph.number_of_edges() == 1500
        assert edge_time < 1.0  # Should be fast
        
        # Record performance
        self.monitor.record_operation("large_graph_creation", node_time + edge_time)
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        self.manager.create_graph("memory_test", "Graph")
        
        # Add nodes with attributes
        nodes_with_attrs = [
            (i, {"data": "x" * 100, "value": i})
            for i in range(100)
        ]
        self.manager.add_nodes_from("memory_test", nodes_with_attrs)
        
        # Add edges with attributes
        edges_with_attrs = [
            (i, (i + 1) % 100, {"weight": i * 0.1})
            for i in range(100)
        ]
        self.manager.add_edges_from("memory_test", edges_with_attrs)
        
        info = self.manager.get_graph_info("memory_test")
        
        # Check memory estimate exists
        assert "metadata" in info
        # Memory should account for attributes
        assert info["num_nodes"] == 100
        assert info["num_edges"] == 100


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = GraphManager()
    
    def test_empty_graph_operations(self):
        """Test operations on empty graph."""
        self.manager.create_graph("empty", "Graph")
        
        # Operations on empty graph
        info = self.manager.get_graph_info("empty")
        assert info["num_nodes"] == 0
        assert info["num_edges"] == 0
        assert info["density"] == 0
        
        # Subgraph of empty graph
        result = self.manager.subgraph("empty", [])
        assert result["num_nodes"] == 0
    
    def test_self_loops(self):
        """Test self-loop handling."""
        self.manager.create_graph("loops", "DiGraph")
        
        # Add self-loops
        self.manager.add_nodes_from("loops", ["A", "B", "C"])
        self.manager.add_edges_from("loops", [
            ("A", "A"),  # Self-loop
            ("A", "B"),
            ("B", "B"),  # Self-loop
            ("B", "C")
        ])
        
        graph = self.manager.get_graph("loops")
        assert graph.number_of_edges() == 4
        assert graph.has_edge("A", "A")
        assert graph.has_edge("B", "B")
    
    def test_multigraph_operations(self):
        """Test MultiGraph specific operations."""
        self.manager.create_graph("multi", "MultiGraph")
        
        # Add multiple edges between same nodes
        self.manager.add_nodes_from("multi", ["X", "Y", "Z"])
        self.manager.add_edges_from("multi", [
            ("X", "Y", {"weight": 1}),
            ("X", "Y", {"weight": 2}),
            ("X", "Y", {"weight": 3}),
            ("Y", "Z", {"weight": 4})
        ])
        
        graph = self.manager.get_graph("multi")
        assert graph.number_of_edges() == 4
        assert graph.number_of_edges("X", "Y") == 3
    
    def test_node_attribute_operations(self):
        """Test complex node attribute operations."""
        self.manager.create_graph("attrs", "Graph")
        
        # Add nodes with various attribute types
        self.manager.add_node("attrs", "N1", 
            string_attr="test",
            int_attr=42,
            float_attr=3.14,
            list_attr=[1, 2, 3],
            dict_attr={"nested": "value"}
        )
        
        attrs = self.manager.get_node_attributes("attrs", "N1")
        assert attrs["string_attr"] == "test"
        assert attrs["int_attr"] == 42
        assert attrs["float_attr"] == 3.14
        assert attrs["list_attr"] == [1, 2, 3]
        assert attrs["dict_attr"] == {"nested": "value"}
        
        # Update attributes
        self.manager.set_node_attributes(
            "attrs",
            {"N1": {"new_attr": "new_value", "int_attr": 100}}
        )
        
        attrs = self.manager.get_node_attributes("attrs", "N1")
        assert attrs["new_attr"] == "new_value"
        assert attrs["int_attr"] == 100
        assert attrs["string_attr"] == "test"  # Original preserved


class TestValidation:
    """Test input validation."""
    
    def test_graph_id_validation(self):
        """Test graph ID validation."""
        # Valid IDs
        valid_ids = ["graph1", "my-graph", "test_123", "G"]
        for gid in valid_ids:
            valid, error = GraphValidator.validate_graph_id(gid)
            assert valid is True
            assert error is None
        
        # Invalid IDs
        invalid_ids = [
            "",  # Empty
            "graph with spaces",  # Spaces
            "graph@123",  # Special chars
            "a" * 256,  # Too long
            123,  # Not string
            None,  # None
        ]
        for gid in invalid_ids:
            valid, error = GraphValidator.validate_graph_id(gid)
            assert valid is False
            assert error is not None
    
    def test_file_format_validation(self):
        """Test file format validation."""
        # Create a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            f.write(b"test")
            temp_path = f.name
        
        try:
            # Valid file
            valid, error = GraphValidator.validate_file_format(
                temp_path, ["graphml"]
            )
            assert valid is True
            
            # Wrong format
            valid, error = GraphValidator.validate_file_format(
                temp_path, ["json"]
            )
            assert valid is False
            
        finally:
            import os
            os.unlink(temp_path)
    
    def test_graph_data_validation(self):
        """Test graph data structure validation."""
        # Valid data
        valid_data = {
            "nodes": [{"id": "A"}, {"id": "B"}],
            "edges": [{"source": "A", "target": "B"}]
        }
        valid, errors = GraphValidator.validate_graph_data(valid_data)
        assert valid is True
        assert len(errors) == 0
        
        # Invalid data - missing edge target
        invalid_data = {
            "edges": [{"source": "A"}]  # Missing target
        }
        valid, errors = GraphValidator.validate_graph_data(invalid_data)
        assert valid is False
        assert len(errors) > 0