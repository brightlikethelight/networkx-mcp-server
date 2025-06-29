"""Integration tests for the MCP server."""

import json

import pytest

from networkx_mcp.server import NetworkXMCPServer


@pytest.mark.asyncio
class TestMCPServerIntegration:
    """Test MCP server integration."""

    async def test_server_initialization(self):
        """Test server initializes correctly."""
        server = NetworkXMCPServer()

        # Server should have tools registered
        assert hasattr(server, 'tools')
        assert len(server.tools) > 30  # Should have 39+ tools

    async def test_create_and_analyze_graph(self, mcp_server):
        """Test creating a graph and running analysis."""
        # Create a graph
        create_result = await mcp_server.handle_tool_call(
            "create_graph",
            {"graph_type": "undirected"}
        )

        assert create_result["success"] is True
        assert "graph_id" in create_result

        graph_id = create_result["graph_id"]

        # Add nodes
        for i in range(5):
            await mcp_server.handle_tool_call(
                "add_node",
                {"graph_id": graph_id, "node_id": i}
            )

        # Add edges
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        for u, v in edges:
            await mcp_server.handle_tool_call(
                "add_edge",
                {"graph_id": graph_id, "source": u, "target": v}
            )

        # Get graph info
        info_result = await mcp_server.handle_tool_call(
            "get_graph_info",
            {"graph_id": graph_id}
        )

        assert info_result["nodes"] == 5
        assert info_result["edges"] == 5

        # Run analysis
        analysis_result = await mcp_server.handle_tool_call(
            "analyze_graph",
            {"graph_id": graph_id, "analyses": ["basic", "centrality"]}
        )

        assert "basic_stats" in analysis_result
        assert "centrality" in analysis_result
        assert analysis_result["basic_stats"]["is_cyclic"] is True

    async def test_import_export_cycle(self, mcp_server, temp_dir):
        """Test importing and exporting graphs."""
        # Create a graph
        create_result = await mcp_server.handle_tool_call(
            "create_graph",
            {"graph_type": "directed"}
        )

        graph_id = create_result["graph_id"]

        # Add some data
        await mcp_server.handle_tool_call(
            "add_edge",
            {"graph_id": graph_id, "source": "A", "target": "B", "weight": 1.5}
        )

        # Export to JSON
        export_result = await mcp_server.handle_tool_call(
            "export_graph",
            {"graph_id": graph_id, "format": "json"}
        )

        assert "data" in export_result
        json_data = json.loads(export_result["data"])

        # Delete the graph
        await mcp_server.handle_tool_call(
            "delete_graph",
            {"graph_id": graph_id}
        )

        # Import it back
        import_result = await mcp_server.handle_tool_call(
            "import_graph",
            {"format": "json", "data": export_result["data"]}
        )

        new_graph_id = import_result["graph_id"]

        # Verify it's the same
        info = await mcp_server.handle_tool_call(
            "get_graph_info",
            {"graph_id": new_graph_id}
        )

        assert info["directed"] is True
        assert info["edges"] == 1

    @pytest.mark.performance
    async def test_large_graph_performance(self, mcp_server, performance_config):
        """Test performance with larger graphs."""
        import time

        # Create a large graph
        create_result = await mcp_server.handle_tool_call(
            "create_graph",
            {"graph_type": "undirected"}
        )

        graph_id = create_result["graph_id"]

        # Generate a random graph
        start_time = time.time()

        generate_result = await mcp_server.handle_tool_call(
            "generate_graph",
            {
                "graph_id": graph_id,
                "generator": "random",
                "n": performance_config["medium_graph_size"],
                "p": 0.01
            }
        )

        generation_time = time.time() - start_time

        assert generation_time < performance_config["timeout"]

        # Run algorithms
        algo_start = time.time()

        analysis = await mcp_server.handle_tool_call(
            "analyze_graph",
            {"graph_id": graph_id, "analyses": ["basic"]}
        )

        algo_time = time.time() - algo_start

        assert algo_time < performance_config["timeout"]
        assert analysis["basic_stats"]["nodes"] == performance_config["medium_graph_size"]
