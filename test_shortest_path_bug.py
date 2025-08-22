"""Test to reproduce the shortest path bug."""

import networkx as nx

from networkx_mcp.core.algorithms import GraphAlgorithms

# Create a simple connected graph
graph = nx.Graph()
graph.add_edges_from([(1, 2), (2, 3), (3, 4)])

print("Graph edges:", graph.edges())
print("Is connected?", nx.is_connected(graph))

# Test shortest path
algorithms = GraphAlgorithms()

# Test 1: Direct path
try:
    result = algorithms.shortest_path(graph, 1, 3)
    print(f"Path from 1 to 3: {result}")
except Exception as e:
    print(f"ERROR finding path 1 to 3: {e}")

# Test 2: Longer path
try:
    result = algorithms.shortest_path(graph, 1, 4)
    print(f"Path from 1 to 4: {result}")
except Exception as e:
    print(f"ERROR finding path 1 to 4: {e}")

# Test with NetworkX directly
print("\nDirect NetworkX test:")
print(f"NX path 1 to 3: {nx.shortest_path(graph, 1, 3)}")
print(f"NX path 1 to 4: {nx.shortest_path(graph, 1, 4)}")
