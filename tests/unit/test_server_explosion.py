"""SERVER MODULE EXPLOSION - Target: 0% → 80%+ coverage (721 lines).

This test suite targets the server.py module which is currently at 0% coverage.
This is our highest priority for reaching 90%+ total coverage.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import pytest

from networkx_mcp.server import (
    GraphManager,
    NetworkXMCPServer,
    add_edges,
    add_nodes,
    betweenness_centrality,
    community_detection,
    connected_components,
    create_graph,
    degree_centrality,
    delete_graph,
    export_json,
    get_graph_info,
    graphs,
    import_csv,
    main,
    mcp,
    pagerank,
    shortest_path,
    visualize_graph,
)


class TestGraphManager:
    """Test GraphManager class - lines 71-85."""

    def setup_method(self):
        """Setup for each test."""
        # Clear global graphs state
        graphs.clear()

    def test_graph_manager_creation(self):
        """Test GraphManager initialization."""
        manager = GraphManager()
        assert manager is not None
        assert manager.graphs is graphs
        assert isinstance(manager.graphs, dict)

    def test_get_graph_exists(self):
        """Test getting existing graph."""
        # Setup test graph
        test_graph = nx.Graph()
        test_graph.add_edge("A", "B")
        graphs["test_graph"] = test_graph

        manager = GraphManager()
        result = manager.get_graph("test_graph")
        
        assert result is test_graph
        assert result.number_of_nodes() == 2
        assert result.number_of_edges() == 1

    def test_get_graph_not_exists(self):
        """Test getting non-existent graph."""
        manager = GraphManager()
        result = manager.get_graph("nonexistent")
        assert result is None

    def test_delete_graph_exists(self):
        """Test deleting existing graph."""
        # Setup test graph
        graphs["to_delete"] = nx.Graph()
        
        manager = GraphManager()
        assert "to_delete" in graphs
        
        manager.delete_graph("to_delete")
        assert "to_delete" not in graphs

    def test_delete_graph_not_exists(self):
        """Test deleting non-existent graph (should not error)."""
        manager = GraphManager()
        # Should not raise exception
        manager.delete_graph("nonexistent")
        assert True  # Just verify no exception


class TestModuleLevelFunctions:
    """Test module-level wrapper functions - lines 108-166."""

    def setup_method(self):
        """Setup for each test."""
        graphs.clear()

    def test_create_graph_undirected(self):
        """Test creating undirected graph."""
        result = create_graph("test_undirected", directed=False)
        
        assert "test_undirected" in graphs
        assert not graphs["test_undirected"].is_directed()
        assert isinstance(result, dict)

    def test_create_graph_directed(self):
        """Test creating directed graph."""
        result = create_graph("test_directed", directed=True)
        
        assert "test_directed" in graphs
        assert graphs["test_directed"].is_directed()
        assert isinstance(result, dict)

    def test_add_nodes_to_graph(self):
        """Test adding nodes to existing graph."""
        create_graph("test_graph")
        result = add_nodes("test_graph", ["A", "B", "C"])
        
        assert isinstance(result, dict)
        assert graphs["test_graph"].number_of_nodes() == 3
        assert "A" in graphs["test_graph"].nodes()

    def test_add_edges_to_graph(self):
        """Test adding edges to existing graph."""
        create_graph("test_graph")
        add_nodes("test_graph", ["A", "B", "C"])
        result = add_edges("test_graph", [["A", "B"], ["B", "C"]])
        
        assert isinstance(result, dict)
        assert graphs["test_graph"].number_of_edges() == 2
        assert graphs["test_graph"].has_edge("A", "B")

    def test_get_graph_info_existing(self):
        """Test getting info for existing graph."""
        create_graph("test_graph")
        add_nodes("test_graph", ["A", "B"])
        add_edges("test_graph", [["A", "B"]])
        
        result = get_graph_info("test_graph")
        assert isinstance(result, dict)

    def test_shortest_path_existing_graph(self):
        """Test shortest path calculation."""
        create_graph("test_graph")
        add_nodes("test_graph", ["A", "B", "C"])
        add_edges("test_graph", [["A", "B"], ["B", "C"]])
        
        result = shortest_path("test_graph", "A", "C")
        assert isinstance(result, dict)

    def test_degree_centrality_calculation(self):
        """Test degree centrality calculation."""
        create_graph("test_graph")
        add_nodes("test_graph", ["A", "B", "C"])
        add_edges("test_graph", [["A", "B"], ["B", "C"]])
        
        result = degree_centrality("test_graph")
        assert isinstance(result, dict)

    def test_betweenness_centrality_calculation(self):
        """Test betweenness centrality calculation."""
        create_graph("test_graph")
        add_nodes("test_graph", ["A", "B", "C"])
        add_edges("test_graph", [["A", "B"], ["B", "C"]])
        
        result = betweenness_centrality("test_graph")
        assert isinstance(result, dict)

    def test_connected_components_calculation(self):
        """Test connected components calculation."""
        create_graph("test_graph")
        add_nodes("test_graph", ["A", "B", "C"])
        add_edges("test_graph", [["A", "B"]])
        
        result = connected_components("test_graph")
        assert isinstance(result, dict)

    def test_pagerank_calculation(self):
        """Test PageRank calculation."""
        create_graph("test_graph")
        add_nodes("test_graph", ["A", "B", "C"])
        add_edges("test_graph", [["A", "B"], ["B", "C"]])
        
        result = pagerank("test_graph")
        assert isinstance(result, dict)

    def test_community_detection_calculation(self):
        """Test community detection calculation."""
        create_graph("test_graph")
        add_nodes("test_graph", ["A", "B", "C"])
        add_edges("test_graph", [["A", "B"], ["B", "C"]])
        
        result = community_detection("test_graph")
        assert isinstance(result, dict)

    def test_visualize_graph_default_layout(self):
        """Test graph visualization with default layout."""
        create_graph("test_graph")
        add_nodes("test_graph", ["A", "B", "C"])
        add_edges("test_graph", [["A", "B"], ["B", "C"]])
        
        result = visualize_graph("test_graph")
        assert isinstance(result, dict)

    def test_visualize_graph_custom_layout(self):
        """Test graph visualization with custom layout."""
        create_graph("test_graph")
        add_nodes("test_graph", ["A", "B", "C"])
        add_edges("test_graph", [["A", "B"], ["B", "C"]])
        
        result = visualize_graph("test_graph", layout="circular")
        assert isinstance(result, dict)

    def test_import_csv_functionality(self):
        """Test CSV import functionality."""
        csv_data = "A,B\nB,C\nC,D"
        result = import_csv("csv_graph", csv_data, directed=False)
        
        assert isinstance(result, dict)
        assert "csv_graph" in graphs

    def test_export_json_functionality(self):
        """Test JSON export functionality."""
        create_graph("test_graph")
        add_nodes("test_graph", ["A", "B"])
        add_edges("test_graph", [["A", "B"]])
        
        result = export_json("test_graph")
        assert isinstance(result, dict)

    def test_delete_graph_existing(self):
        """Test deleting existing graph."""
        create_graph("to_delete")
        assert "to_delete" in graphs
        
        result = delete_graph("to_delete")
        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "to_delete" not in graphs

    def test_delete_graph_nonexistent(self):
        """Test deleting non-existent graph."""
        result = delete_graph("nonexistent")
        assert isinstance(result, dict)
        assert result.get("success") is False


class TestNetworkXMCPServerInitialization:
    """Test NetworkXMCPServer initialization - lines 172-194."""

    def test_server_basic_initialization(self):
        """Test basic server initialization."""
        server = NetworkXMCPServer()
        
        assert server.running is True
        assert server.mcp is server
        assert server.graphs is graphs
        assert server.auth_required is False
        assert server.auth is None
        assert server.monitoring_enabled is False
        assert server.monitor is None

    def test_server_with_auth_no_module(self):
        """Test server with auth enabled but no auth module."""
        # Auth should be disabled if HAS_AUTH is False
        server = NetworkXMCPServer(auth_required=True)
        assert server.auth_required is False  # Should fall back to False
        assert server.auth is None

    def test_server_with_monitoring_no_module(self):
        """Test server with monitoring enabled but no monitoring module."""
        # Monitoring should be disabled if HAS_MONITORING is False
        server = NetworkXMCPServer(enable_monitoring=True)
        assert server.monitoring_enabled is False  # Should fall back to False
        assert server.monitor is None

    @patch('networkx_mcp.server.HAS_AUTH', True)
    @patch('networkx_mcp.server.APIKeyManager')
    @patch('networkx_mcp.server.AuthMiddleware')
    def test_server_with_auth_enabled(self, mock_auth_middleware, mock_api_key_manager):
        """Test server with authentication enabled."""
        mock_key_manager = MagicMock()
        mock_api_key_manager.return_value = mock_key_manager
        mock_auth = MagicMock()
        mock_auth_middleware.return_value = mock_auth
        
        server = NetworkXMCPServer(auth_required=True)
        
        assert server.auth_required is True
        assert server.auth is mock_auth
        mock_api_key_manager.assert_called_once()
        mock_auth_middleware.assert_called_once_with(mock_key_manager, required=True)

    @patch('networkx_mcp.server.HAS_MONITORING', True)
    @patch('networkx_mcp.server.HealthMonitor')
    def test_server_with_monitoring_enabled(self, mock_health_monitor):
        """Test server with monitoring enabled."""
        mock_monitor = MagicMock()
        mock_health_monitor.return_value = mock_monitor
        
        server = NetworkXMCPServer(enable_monitoring=True)
        
        assert server.monitoring_enabled is True
        assert server.monitor is mock_monitor
        assert server.monitor.graphs is graphs

    def test_tool_decorator_mock(self):
        """Test tool decorator mock functionality."""
        server = NetworkXMCPServer()
        
        def dummy_func():
            return "test"
        
        decorated = server.tool(dummy_func)
        assert decorated is dummy_func  # Should return the function unchanged


class TestNetworkXMCPServerHandleRequest:
    """Test NetworkXMCPServer.handle_request method - lines 199-270."""

    def setup_method(self):
        """Setup for each test."""
        graphs.clear()
        self.server = NetworkXMCPServer()

    @pytest.mark.asyncio
    async def test_handle_initialize_request(self):
        """Test handling initialize request."""
        request = {
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0.0"}
            },
            "id": 1
        }
        
        response = await self.server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"
        assert "capabilities" in response["result"]
        assert "serverInfo" in response["result"]

    @pytest.mark.asyncio
    async def test_handle_initialized_notification(self):
        """Test handling initialized notification."""
        request = {
            "method": "initialized",
            "params": {}
        }
        
        response = await self.server.handle_request(request)
        assert response is None  # Notifications don't get responses

    @pytest.mark.asyncio 
    async def test_handle_initialized_with_id(self):
        """Test handling initialized with ID."""
        request = {
            "method": "initialized", 
            "params": {},
            "id": 2
        }
        
        response = await self.server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response

    @pytest.mark.asyncio
    async def test_handle_tools_list_request(self):
        """Test handling tools/list request."""
        request = {
            "method": "tools/list",
            "params": {},
            "id": 3
        }
        
        response = await self.server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "result" in response
        assert "tools" in response["result"]
        assert isinstance(response["result"]["tools"], list)

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self):
        """Test handling unknown method."""
        request = {
            "method": "unknown_method",
            "params": {},
            "id": 4
        }
        
        response = await self.server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 4
        assert "error" in response
        assert response["error"]["code"] == -32601
        assert "Unknown method" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_handle_tools_call_request(self):
        """Test handling tools/call request."""
        request = {
            "method": "tools/call",
            "params": {
                "name": "create_graph",
                "arguments": {"name": "test_graph", "directed": False}
            },
            "id": 5
        }
        
        response = await self.server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 5
        assert "result" in response


class TestNetworkXMCPServerGetTools:
    """Test NetworkXMCPServer._get_tools method - lines 272-518."""

    def test_get_tools_basic(self):
        """Test getting tools list."""
        server = NetworkXMCPServer()
        tools = server._get_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Check for expected tools
        tool_names = [tool["name"] for tool in tools]
        expected_tools = [
            "create_graph", "add_nodes", "add_edges", "shortest_path",
            "get_info", "degree_centrality", "betweenness_centrality",
            "connected_components", "pagerank", "community_detection",
            "visualize_graph", "import_csv", "export_json",
            "build_citation_network", "analyze_author_impact"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names

    def test_get_tools_structure(self):
        """Test tools have proper structure."""
        server = NetworkXMCPServer()
        tools = server._get_tools()
        
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert isinstance(tool["inputSchema"], dict)
            assert "type" in tool["inputSchema"]
            assert tool["inputSchema"]["type"] == "object"

    @patch('networkx_mcp.server.HAS_MONITORING', True)
    @patch('networkx_mcp.server.HealthMonitor')
    def test_get_tools_with_monitoring(self, mock_health_monitor):
        """Test getting tools with monitoring enabled."""
        mock_monitor = MagicMock()
        mock_health_monitor.return_value = mock_monitor
        
        server = NetworkXMCPServer(enable_monitoring=True)
        tools = server._get_tools()
        
        tool_names = [tool["name"] for tool in tools]
        assert "health_status" in tool_names

    def test_get_tools_without_monitoring(self):
        """Test getting tools without monitoring."""
        server = NetworkXMCPServer(enable_monitoring=False)
        tools = server._get_tools()
        
        tool_names = [tool["name"] for tool in tools]
        assert "health_status" not in tool_names


class TestNetworkXMCPServerCallTool:
    """Test NetworkXMCPServer._call_tool method - lines 520-665."""

    def setup_method(self):
        """Setup for each test."""
        graphs.clear()
        self.server = NetworkXMCPServer()

    @pytest.mark.asyncio
    async def test_call_tool_create_graph(self):
        """Test calling create_graph tool."""
        params = {
            "name": "create_graph",
            "arguments": {"name": "test_graph", "directed": False}
        }
        
        result = await self.server._call_tool(params)
        
        assert "content" in result
        assert isinstance(result["content"], list)
        assert "test_graph" in graphs
        assert not graphs["test_graph"].is_directed()

    @pytest.mark.asyncio
    async def test_call_tool_create_directed_graph(self):
        """Test calling create_graph tool for directed graph."""
        params = {
            "name": "create_graph", 
            "arguments": {"name": "directed_graph", "directed": True}
        }
        
        result = await self.server._call_tool(params)
        
        assert "content" in result
        assert "directed_graph" in graphs
        assert graphs["directed_graph"].is_directed()

    @pytest.mark.asyncio
    async def test_call_tool_add_nodes(self):
        """Test calling add_nodes tool."""
        # Create graph first
        graphs["test_graph"] = nx.Graph()
        
        params = {
            "name": "add_nodes",
            "arguments": {"graph": "test_graph", "nodes": ["A", "B", "C"]}
        }
        
        result = await self.server._call_tool(params)
        
        assert "content" in result
        assert graphs["test_graph"].number_of_nodes() == 3
        assert "A" in graphs["test_graph"].nodes()

    @pytest.mark.asyncio
    async def test_call_tool_add_edges(self):
        """Test calling add_edges tool."""
        # Setup graph with nodes
        graphs["test_graph"] = nx.Graph()
        graphs["test_graph"].add_nodes_from(["A", "B", "C"])
        
        params = {
            "name": "add_edges",
            "arguments": {"graph": "test_graph", "edges": [["A", "B"], ["B", "C"]]}
        }
        
        result = await self.server._call_tool(params)
        
        assert "content" in result
        assert graphs["test_graph"].number_of_edges() == 2
        assert graphs["test_graph"].has_edge("A", "B")

    @pytest.mark.asyncio
    async def test_call_tool_shortest_path(self):
        """Test calling shortest_path tool."""
        # Setup connected graph
        graphs["test_graph"] = nx.Graph()
        graphs["test_graph"].add_edge("A", "B")
        graphs["test_graph"].add_edge("B", "C")
        
        params = {
            "name": "shortest_path",
            "arguments": {"graph": "test_graph", "source": "A", "target": "C"}
        }
        
        result = await self.server._call_tool(params)
        
        assert "content" in result
        content_text = result["content"][0]["text"]
        parsed = json.loads(content_text)
        assert "path" in parsed
        assert parsed["path"] == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_call_tool_get_info(self):
        """Test calling get_info tool."""
        # Setup test graph
        graphs["test_graph"] = nx.Graph()
        graphs["test_graph"].add_edge("A", "B")
        
        params = {
            "name": "get_info",
            "arguments": {"graph": "test_graph"}
        }
        
        result = await self.server._call_tool(params)
        
        assert "content" in result
        content_text = result["content"][0]["text"]
        parsed = json.loads(content_text)
        assert "nodes" in parsed
        assert "edges" in parsed
        assert parsed["nodes"] == 2
        assert parsed["edges"] == 1

    @pytest.mark.asyncio
    async def test_call_tool_nonexistent_graph_error(self):
        """Test calling tool with non-existent graph."""
        params = {
            "name": "add_nodes",
            "arguments": {"graph": "nonexistent", "nodes": ["A"]}
        }
        
        result = await self.server._call_tool(params)
        
        assert "content" in result
        assert "isError" in result
        assert result["isError"] is True
        assert "Error:" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool_error(self):
        """Test calling unknown tool."""
        params = {
            "name": "unknown_tool",
            "arguments": {}
        }
        
        result = await self.server._call_tool(params)
        
        assert "content" in result
        assert "isError" in result
        assert result["isError"] is True
        assert "Unknown tool" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_call_tool_health_status_without_monitoring(self):
        """Test calling health_status without monitoring enabled."""
        params = {
            "name": "health_status",
            "arguments": {}
        }
        
        result = await self.server._call_tool(params)
        
        assert "content" in result
        content_text = result["content"][0]["text"]
        parsed = json.loads(content_text)
        assert parsed["status"] == "monitoring_disabled"


class TestNetworkXMCPServerRun:
    """Test NetworkXMCPServer.run method - lines 667-684."""

    @pytest.mark.asyncio
    async def test_run_method_structure(self):
        """Test run method exists and has proper structure."""
        server = NetworkXMCPServer()
        
        # Just verify the method exists and is callable
        assert hasattr(server, 'run')
        assert callable(server.run)
        
        # Don't actually run it as it would block indefinitely


class TestGlobalInstances:
    """Test global instances and main function - lines 686-721."""

    def test_global_mcp_instance(self):
        """Test global mcp instance exists."""
        assert mcp is not None
        assert isinstance(mcp, NetworkXMCPServer)

    @patch('networkx_mcp.server.NetworkXMCPServer')
    @patch('networkx_mcp.server.asyncio.run')
    @patch('os.environ.get')
    def test_main_function_default(self, mock_env_get, mock_asyncio_run, mock_server_class):
        """Test main function with default settings."""
        mock_env_get.return_value = "false"
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server
        
        main()
        
        mock_server_class.assert_called_once_with(
            auth_required=False, 
            enable_monitoring=False
        )
        mock_asyncio_run.assert_called_once_with(mock_server.run())

    @patch('networkx_mcp.server.NetworkXMCPServer')
    @patch('networkx_mcp.server.asyncio.run')
    @patch('os.environ.get')
    def test_main_function_with_auth(self, mock_env_get, mock_asyncio_run, mock_server_class):
        """Test main function with auth enabled."""
        def env_side_effect(key, default="false"):
            if key == "NETWORKX_MCP_AUTH":
                return "true"
            return default
        
        mock_env_get.side_effect = env_side_effect
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server
        
        with patch('logging.basicConfig') as mock_logging:
            main()
        
        mock_server_class.assert_called_once_with(
            auth_required=True, 
            enable_monitoring=False
        )
        mock_logging.assert_called_once()

    @patch('networkx_mcp.server.NetworkXMCPServer')
    @patch('networkx_mcp.server.asyncio.run')
    @patch('os.environ.get')
    def test_main_function_with_monitoring(self, mock_env_get, mock_asyncio_run, mock_server_class):
        """Test main function with monitoring enabled."""
        def env_side_effect(key, default="false"):
            if key == "NETWORKX_MCP_MONITORING":
                return "true"
            return default
        
        mock_env_get.side_effect = env_side_effect
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server
        
        with patch('logging.basicConfig') as mock_logging:
            main()
        
        mock_server_class.assert_called_once_with(
            auth_required=False, 
            enable_monitoring=True
        )
        mock_logging.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])