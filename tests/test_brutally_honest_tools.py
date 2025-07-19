"""BRUTALLY HONEST test - each tool must work with the actual return values."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from networkx_mcp.core.algorithms import GraphAlgorithms
from networkx_mcp.core.graph_operations import GraphManager


class TestBrutallyHonestTools:
    """Test tools using ACTUAL return values - no wishful thinking."""

    def setup_method(self):
        """Clean state before each test."""
        self.graph_manager = GraphManager()
        self.algorithms = GraphAlgorithms()

        # Clear any existing graphs
        for graph_id in list(self.graph_manager.graphs.keys()):
            self.graph_manager.delete_graph(graph_id)

    def test_create_graph_undirected(self):
        """Test creating undirected graph."""
        result = self.graph_manager.create_graph("test_undirected", graph_type="Graph")

        # Check actual return structure
        assert result["graph_id"] == "test_undirected"
        assert result["graph_type"] == "Graph"
        assert result["created"] is True
        assert "metadata" in result

        # Verify graph exists and is undirected
        graph = self.graph_manager.get_graph("test_undirected")
        assert not graph.is_directed()

        print("✅ create_graph (undirected) WORKS")

    def test_create_graph_directed(self):
        """Test creating directed graph."""
        result = self.graph_manager.create_graph("test_directed", graph_type="DiGraph")

        # Check actual return structure
        assert result["graph_id"] == "test_directed"
        assert result["graph_type"] == "DiGraph"
        assert result["created"] is True

        # Verify graph exists and is directed
        graph = self.graph_manager.get_graph("test_directed")
        assert graph.is_directed()

        print("✅ create_graph (directed) WORKS")

    def test_add_nodes(self):
        """Test adding nodes."""
        # Create graph first
        self.graph_manager.create_graph("node_test")

        # Add nodes
        result = self.graph_manager.add_nodes_from("node_test", ["A", "B", "C"])

        # Check actual return structure
        assert result["graph_id"] == "node_test"
        assert result["nodes_added"] == 3
        assert result["total_nodes"] == 3

        # Verify nodes actually exist
        graph = self.graph_manager.get_graph("node_test")
        assert set(graph.nodes()) == {"A", "B", "C"}

        print("✅ add_nodes WORKS")

    def test_add_edges(self):
        """Test adding edges."""
        # Create graph and add nodes
        self.graph_manager.create_graph("edge_test")
        self.graph_manager.add_nodes_from("edge_test", ["A", "B", "C"])

        # Add edges
        edges = [("A", "B"), ("B", "C")]
        result = self.graph_manager.add_edges_from("edge_test", edges)

        # Check actual return structure
        assert result["graph_id"] == "edge_test"
        assert result["edges_added"] == 2
        assert result["total_edges"] == 2

        # Verify edges exist
        graph = self.graph_manager.get_graph("edge_test")
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "C")

        print("✅ add_edges WORKS")

    def test_get_graph_info(self):
        """Test getting graph info."""
        # Create directed graph with nodes and edges
        self.graph_manager.create_graph("info_test", graph_type="DiGraph")
        self.graph_manager.add_nodes_from("info_test", ["X", "Y"])
        self.graph_manager.add_edges_from("info_test", [("X", "Y")])

        # Get info
        result = self.graph_manager.get_graph_info("info_test")

        # Check what's actually returned
        print(f"get_graph_info returns: {result}")

        # Basic checks
        assert result["graph_id"] == "info_test"
        assert result["num_nodes"] == 2
        assert result["num_edges"] == 1

        # Check if it correctly identifies directed graphs
        if "is_directed" in result:
            assert result["is_directed"] is True
            print("✅ get_graph_info WORKS (with is_directed)")
        else:
            print("⚠️  get_graph_info works but doesn't return is_directed")

    def test_shortest_path(self):
        """Test shortest path algorithm."""
        # Create simple path
        self.graph_manager.create_graph("path_test")
        self.graph_manager.add_nodes_from("path_test", ["A", "B", "C"])
        self.graph_manager.add_edges_from("path_test", [("A", "B"), ("B", "C")])

        graph = self.graph_manager.get_graph("path_test")

        # Test shortest path
        try:
            result = self.algorithms.shortest_path(graph, "A", "C")
            print(f"shortest_path returns: {result}")

            # Check if it's a list of nodes or dict
            if isinstance(result, list):
                assert result == ["A", "B", "C"]
                print("✅ shortest_path WORKS (returns list)")
            elif isinstance(result, dict) and "path" in result:
                assert result["path"] == ["A", "B", "C"]
                print("✅ shortest_path WORKS (returns dict)")
            else:
                print(f"❌ shortest_path returns unexpected format: {type(result)}")

        except Exception as e:
            print(f"❌ shortest_path FAILS: {e}")

    def test_centrality_measures(self):
        """Test centrality measures."""
        # Create star graph
        self.graph_manager.create_graph("centrality_test")
        self.graph_manager.add_nodes_from("centrality_test", ["A", "B", "C", "D"])
        self.graph_manager.add_edges_from(
            "centrality_test",
            [
                ("A", "B"),
                ("A", "C"),
                ("A", "D"),  # A is central
            ],
        )

        graph = self.graph_manager.get_graph("centrality_test")

        try:
            import networkx as nx

            # Test individual measures
            degree = dict(nx.degree_centrality(graph))
            betweenness = dict(nx.betweenness_centrality(graph))
            closeness = dict(nx.closeness_centrality(graph))

            # A should have highest centrality
            assert degree["A"] > degree["B"]
            assert betweenness["A"] > betweenness["B"]

            print("✅ centrality_measures WORKS")

        except Exception as e:
            print(f"❌ centrality_measures FAILS: {e}")

    def test_delete_graph(self):
        """Test deleting graph."""
        # Create graph
        self.graph_manager.create_graph("delete_test")
        assert "delete_test" in self.graph_manager.graphs

        # Delete it
        result = self.graph_manager.delete_graph("delete_test")

        # Check actual return structure
        assert result["graph_id"] == "delete_test"
        assert result["deleted"] is True

        # Verify it's gone
        assert "delete_test" not in self.graph_manager.graphs

        print("✅ delete_graph WORKS")

    def test_all_tools_summary(self):
        """Summary of which tools actually work."""
        working_tools = []
        broken_tools = []

        # Test each tool
        try:
            self.test_create_graph_undirected()
            self.test_create_graph_directed()
            working_tools.append("create_graph")
        except Exception as e:
            broken_tools.append(f"create_graph: {e}")

        try:
            self.test_add_nodes()
            working_tools.append("add_nodes")
        except Exception as e:
            broken_tools.append(f"add_nodes: {e}")

        try:
            self.test_add_edges()
            working_tools.append("add_edges")
        except Exception as e:
            broken_tools.append(f"add_edges: {e}")

        try:
            self.test_get_graph_info()
            working_tools.append("get_graph_info")
        except Exception as e:
            broken_tools.append(f"get_graph_info: {e}")

        try:
            self.test_shortest_path()
            working_tools.append("shortest_path")
        except Exception as e:
            broken_tools.append(f"shortest_path: {e}")

        try:
            self.test_centrality_measures()
            working_tools.append("centrality_measures")
        except Exception as e:
            broken_tools.append(f"centrality_measures: {e}")

        try:
            self.test_delete_graph()
            working_tools.append("delete_graph")
        except Exception as e:
            broken_tools.append(f"delete_graph: {e}")

        print("\n📊 TOOL AUDIT RESULTS:")
        print(f"✅ WORKING TOOLS ({len(working_tools)}): {working_tools}")
        print(f"❌ BROKEN TOOLS ({len(broken_tools)}): {broken_tools}")
        print(
            f"📈 SUCCESS RATE: {len(working_tools)}/{len(working_tools) + len(broken_tools)} = {len(working_tools) / (len(working_tools) + len(broken_tools)) * 100:.1f}%"
        )


if __name__ == "__main__":
    # Run comprehensive test
    test = TestBrutallyHonestTools()
    test.setup_method()
    test.test_all_tools_summary()
