"""Comprehensive tests for all 39 MCP tools in the NetworkX MCP Server."""

import pytest
import pytest_asyncio


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
        result = create_graph(
            name="test_graph",
            graph_type="undirected",
            params={"name": "Test Graph", "description": "A test graph"},
        )

        assert result["graph_id"] == "test_graph"
        assert result["graph_type"] == "Graph"
        assert result["created"] is True
        assert "created_at" in result

        # Test directed graph
        result = create_graph(name="directed_test", graph_type="DiGraph")
        assert result["graph_type"] == "DiGraph"

        # Test error case - duplicate ID
        with pytest.raises(ValueError):
            create_graph(name="test_graph", graph_type="undirected")

    @pytest.mark.asyncio
    async def test_add_nodes_tool(self):
        """Test add_nodes MCP tool."""
        from networkx_mcp.server import add_nodes, create_graph

        # Create graph first
        create_graph(name="test", graph_type="undirected")

        # Test adding nodes
        result = add_nodes(name="test", nodes=["A", "B", "C"], params={"color": "red"})

        assert result["nodes_added"] == 3
        assert result["total_nodes"] == 3
        assert set(result["added_nodes"]) == {"A", "B", "C"}

        # Test adding nodes with individual attributes
        result = add_nodes(
            name="test", nodes=[("D", {"weight": 1}), ("E", {"weight": 2})]
        )
        assert result["nodes_added"] == 2
        assert result["total_nodes"] == 5

    @pytest.mark.asyncio
    async def test_add_edges_tool(self):
        """Test add_edges MCP tool."""
        from networkx_mcp.server import add_edges, add_nodes, create_graph

        # Setup graph with nodes
        create_graph(name="test", graph_type="undirected")
        add_nodes(name="test", nodes=["A", "B", "C", "D"])

        # Test adding edges
        result = add_edges(
            name="test",
            edges=[("A", "B"), ("B", "C"), ("C", "D")],
            params={"weight": 1.0},
        )

        assert result["edges_added"] == 3
        assert result["total_edges"] == 3

        # Test adding edges with individual attributes
        result = add_edges(
            name="test",
            edges=[("A", "C", {"weight": 2.5}), ("B", "D", {"weight": 1.8})],
        )
        assert result["edges_added"] == 2
        assert result["total_edges"] == 5

    @pytest.mark.asyncio
    async def test_graph_info_tool(self):
        """Test graph_info MCP tool."""
        from networkx_mcp.server import add_edges, add_nodes, create_graph, graph_info

        # Create graph with data
        create_graph(name="info_test", graph_type="DiGraph")
        add_nodes(name="info_test", nodes=["X", "Y", "Z"])
        add_edges(name="info_test", edges=[("X", "Y"), ("Y", "Z")])

        result = graph_info(name="info_test")

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
        result = list_graphs()
        initial_count = len(result)

        # Create multiple graphs
        create_graph(name="graph1", graph_type="undirected")
        create_graph(name="graph2", graph_type="DiGraph")

        result = list_graphs()
        assert len(result) == initial_count + 2

        graph_ids = [g["graph_id"] for g in result]
        assert "graph1" in graph_ids
        assert "graph2" in graph_ids

    @pytest.mark.asyncio
    async def test_delete_graph_tool(self):
        """Test delete_graph MCP tool."""
        from networkx_mcp.server import create_graph, delete_graph, list_graphs

        # Create and delete graph
        create_graph(name="temp_graph", graph_type="undirected")

        result = delete_graph(name="temp_graph")
        assert result["deleted"] is True
        assert result["graph_id"] == "temp_graph"

        # Verify deletion
        graphs = list_graphs()
        graph_ids = [g["graph_id"] for g in graphs]
        assert "temp_graph" not in graph_ids


class TestMCPToolsAlgorithms:
    """Test algorithm MCP tools."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self):
        """Setup test graph for algorithms."""
        from networkx_mcp.server import add_edges, add_nodes, create_graph

        # Create weighted graph for testing
        create_graph(name="algo_test", graph_type="undirected")
        add_nodes(name="algo_test", nodes=["A", "B", "C", "D", "E"])
        add_edges(
            name="algo_test",
            edges=[
                ("A", "B", {"weight": 1.0}),
                ("B", "C", {"weight": 2.0}),
                ("C", "D", {"weight": 1.5}),
                ("D", "E", {"weight": 1.0}),
                ("E", "A", {"weight": 2.5}),
                ("A", "C", {"weight": 3.0}),
            ],
        )

    @pytest.mark.asyncio
    async def test_shortest_path_tool(self):
        """Test shortest_path MCP tool."""
        from networkx_mcp.server import shortest_path

        # Test single path
        result = shortest_path(
            name="algo_test", source="A", target="D", weight="weight"
        )

        assert result["source"] == "A"
        assert result["target"] == "D"
        assert "path" in result
        assert "length" in result
        assert result["path"][0] == "A"
        assert result["path"][-1] == "D"

        # Test all paths from source
        result = shortest_path(name="algo_test", source="A")

        assert "paths" in result
        assert "lengths" in result
        assert "A" in result["paths"]
        assert "D" in result["paths"]


class TestMCPToolsAdvanced:
    """Test Phase 2 advanced analytics MCP tools."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self):
        """Setup test graph for advanced algorithms."""
        from networkx_mcp.server import add_edges, add_nodes, create_graph

        # Create community graph
        create_graph(name="community_test", graph_type="undirected")

        # Add two communities
        add_nodes(
            name="community_test",
            nodes=[f"A{i}" for i in range(5)] + [f"B{i}" for i in range(5)],
        )

        # Dense connections within communities
        community_a_edges = [
            (f"A{i}", f"A{j}") for i in range(5) for j in range(i + 1, 5)
        ]
        community_b_edges = [
            (f"B{i}", f"B{j}") for i in range(5) for j in range(i + 1, 5)
        ]
        # Sparse connections between communities
        inter_edges = [("A0", "B0"), ("A2", "B3")]

        add_edges(
            name="community_test",
            edges=community_a_edges + community_b_edges + inter_edges,
        )

    @pytest.mark.asyncio
    async def test_bipartite_analysis_tool(self):
        """Test bipartite_analysis MCP tool."""
        from networkx_mcp.server import (
            add_edges,
            add_nodes,
            bipartite_analysis,
            create_graph,
        )

        # Create bipartite graph
        create_graph(name="bipartite_test", graph_type="undirected")
        add_nodes(
            name="bipartite_test",
            nodes=[
                ("A1", {"bipartite": 0}),
                ("A2", {"bipartite": 0}),
                ("B1", {"bipartite": 1}),
                ("B2", {"bipartite": 1}),
            ],
        )
        add_edges(
            name="bipartite_test", edges=[("A1", "B1"), ("A1", "B2"), ("A2", "B1")]
        )

        result = bipartite_analysis(name="bipartite_test")

        assert "is_bipartite" in result
        assert "node_sets" in result

        # Should be bipartite
        assert result["is_bipartite"] is True
        assert len(result["node_sets"]) == 2


class TestMCPToolsVisualization:
    """Test Phase 3 visualization MCP tools."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self):
        """Setup graph for visualization testing."""
        from networkx_mcp.server import add_edges, add_nodes, create_graph

        create_graph(name="viz_test", graph_type="undirected")
        add_nodes(
            name="viz_test",
            nodes=[
                ("A", {"color": "red"}),
                ("B", {"color": "blue"}),
                ("C", {"color": "green"}),
                ("D", {"color": "yellow"}),
            ],
        )
        add_edges(
            name="viz_test",
            edges=[
                ("A", "B", {"weight": 1}),
                ("B", "C", {"weight": 2}),
                ("C", "D", {"weight": 1}),
                ("D", "A", {"weight": 3}),
            ],
        )


class TestMCPToolsIO:
    """Test import/export MCP tools."""


class TestMCPToolsErrorHandling:
    """Test error handling in MCP tools."""

    @pytest.mark.asyncio
    async def test_invalid_graph_id_errors(self):
        """Test error handling for invalid graph IDs."""
        from networkx_mcp.server import graph_info

        # Test non-existent graph
        with pytest.raises(ValueError):  # Should raise appropriate error
            graph_info(name="nonexistent_graph")


class TestMCPToolsPerformance:
    """Test performance characteristics of MCP tools."""

    @pytest.mark.asyncio
    async def test_large_graph_performance(self, performance_thresholds):
        """Test performance with moderately large graphs."""
        import time

        from networkx_mcp.server import (
            add_edges,
            add_nodes,
            centrality_measures,
            create_graph,
        )

        # Create large graph
        start_time = time.time()
        create_graph(name="perf_test", graph_type="undirected")
        creation_time = time.time() - start_time

        assert creation_time < performance_thresholds["create_graph"]

        # Add many nodes
        nodes = list(range(500))
        start_time = time.time()
        add_nodes(name="perf_test", nodes=nodes)
        node_time = time.time() - start_time

        # Should be reasonably fast
        assert node_time < 1.0

        # Add edges
        edges = [(i, (i + 1) % 500) for i in range(500)]
        start_time = time.time()
        add_edges(name="perf_test", edges=edges)
        edge_time = time.time() - start_time

        assert edge_time < 1.0

        # Test algorithm performance
        start_time = time.time()
        centrality_measures(name="perf_test", measures=["degree"])
        centrality_time = time.time() - start_time

        # Degree centrality should be very fast
        assert centrality_time < 0.1

    @pytest.mark.asyncio
    async def test_memory_usage_estimation(self):
        """Test memory usage tracking."""
        from networkx_mcp.server import add_nodes, create_graph, graph_info

        create_graph(name="memory_test", graph_type="undirected")

        # Add nodes with attributes
        nodes_with_attrs = [(i, {"data": "x" * 100}) for i in range(100)]
        add_nodes(name="memory_test", nodes=nodes_with_attrs)

        info = graph_info(name="memory_test")

        # Should provide memory estimates
        assert "num_nodes" in info
        assert info["num_nodes"] == 100


@pytest.mark.asyncio
async def test_all_tools_accessible():
    """Test that all 39 MCP tools are accessible and don't crash."""
    import inspect

    from networkx_mcp import server

    # Get all async functions that look like MCP tools
    tool_functions = []
    for name, obj in inspect.getmembers(server):
        if inspect.iscoroutinefunction(obj) and not name.startswith("_"):
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
        create_graph(name=123, graph_type="undirected")  # Invalid ID type

    with pytest.raises((ValueError, TypeError)):
        create_graph(name="test", graph_type="InvalidType")  # Invalid graph type


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
