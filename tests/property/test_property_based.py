"""Property-Based Testing for NetworkX MCP Server.

This module implements property-based testing using Hypothesis to ensure
mathematical correctness and robustness across all possible inputs.
"""

import pytest
import networkx as nx
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite

from networkx_mcp.core.graph_operations import GraphManager


# ======================================================================
# GRAPH GENERATION STRATEGIES
# ======================================================================

@composite
def simple_graphs(draw):
    """Generate simple graphs for testing."""
    n_nodes = draw(st.integers(min_value=0, max_value=20))
    
    if n_nodes == 0:
        return nx.Graph()
    
    # Generate node list
    nodes = list(range(n_nodes))
    
    # Generate edges
    max_edges = n_nodes * (n_nodes - 1) // 2
    n_edges = draw(st.integers(min_value=0, max_value=min(max_edges, 50)))
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    
    if n_edges > 0:
        # Generate valid edges
        possible_edges = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)]
        if possible_edges:
            edges = draw(st.lists(st.sampled_from(possible_edges), 
                                 min_size=0, max_size=min(n_edges, len(possible_edges))))
            G.add_edges_from(set(edges))  # Remove duplicates
    
    return G


# ======================================================================
# GRAPH PROPERTIES TESTING
# ======================================================================

class TestGraphProperties:
    """Test mathematical properties and invariants of graph operations."""
    
    @given(simple_graphs())
    @settings(max_examples=50, deadline=5000)
    def test_graph_basic_invariants(self, G):
        """Test basic graph invariants hold."""
        # Number of nodes and edges are non-negative
        assert G.number_of_nodes() >= 0
        assert G.number_of_edges() >= 0
        
        # For simple graphs, edges <= n*(n-1)/2
        n = G.number_of_nodes()
        if n > 1:
            max_edges = n * (n - 1) // 2
            assert G.number_of_edges() <= max_edges
        
        # Self-loops should not exist in simple graphs
        assert not any(u == v for u, v in G.edges())
        
        # Density should be between 0 and 1
        density = nx.density(G)
        assert 0 <= density <= 1
    
    @given(simple_graphs())
    @settings(max_examples=30, deadline=5000)
    def test_graph_manager_properties(self, G):
        """Test graph manager maintains invariants."""
        manager = GraphManager()
        
        # Creating graph should increase count
        initial_count = len(manager.list_graphs())
        manager.create_graph("test")
        
        # Now manually set the graph data
        manager.graphs["test"] = G
        
        assert len(manager.list_graphs()) == initial_count + 1
        
        # Retrieved graph should be identical
        retrieved = manager.get_graph("test")
        assert retrieved is not None
        assert retrieved.number_of_nodes() == G.number_of_nodes()
        assert retrieved.number_of_edges() == G.number_of_edges()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])