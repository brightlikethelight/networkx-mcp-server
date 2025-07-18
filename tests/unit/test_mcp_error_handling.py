"""Comprehensive tests for MCP protocol error handling.

Tests verify that all errors return proper JSON-RPC 2.0 compliant responses
with correct error codes and meaningful messages.
"""

import pytest

from networkx_mcp.errors import (
    AlgorithmError,
    EdgeNotFoundError,
    ErrorCodes,
    GraphAlreadyExistsError,
    GraphNotFoundError,
    InvalidEdgeError,
    InvalidGraphIdError,
    InvalidNodeIdError,
    MCPError,
    NodeNotFoundError,
    ValidationError,
    handle_error,
    validate_centrality_measures,
    validate_edge,
    validate_graph_id,
    validate_node_id,
    validate_required_params,
)
from networkx_mcp.server import NetworkXMCPServer


class TestMCPErrorClasses:
    """Test MCP error classes and their JSON-RPC compliance."""

    def test_mcp_error_base_class(self):
        """Test MCPError base class."""
        error = MCPError(ErrorCodes.INTERNAL_ERROR, "Test message", {"test": "data"})

        assert error.code == ErrorCodes.INTERNAL_ERROR
        assert error.message == "Test message"
        assert error.data == {"test": "data"}
        assert str(error) == "Test message"

        # Test to_dict for JSON-RPC compliance
        error_dict = error.to_dict()
        assert error_dict == {
            "code": ErrorCodes.INTERNAL_ERROR,
            "message": "Test message",
            "data": {"test": "data"},
        }

    def test_mcp_error_without_data(self):
        """Test MCPError without data field."""
        error = MCPError(ErrorCodes.INVALID_PARAMS, "No data")

        error_dict = error.to_dict()
        assert error_dict == {"code": ErrorCodes.INVALID_PARAMS, "message": "No data"}
        assert "data" not in error_dict

    def test_graph_not_found_error(self):
        """Test GraphNotFoundError."""
        error = GraphNotFoundError("test_graph")

        assert error.code == ErrorCodes.GRAPH_NOT_FOUND
        assert "test_graph" in error.message
        assert error.graph_id == "test_graph"
        assert error.data == {"graph_id": "test_graph"}

    def test_node_not_found_error(self):
        """Test NodeNotFoundError."""
        error = NodeNotFoundError("test_graph", "test_node")

        assert error.code == ErrorCodes.NODE_NOT_FOUND
        assert "test_node" in error.message
        assert "test_graph" in error.message
        assert error.graph_id == "test_graph"
        assert error.node_id == "test_node"

    def test_edge_not_found_error(self):
        """Test EdgeNotFoundError."""
        error = EdgeNotFoundError("test_graph", "source", "target")

        assert error.code == ErrorCodes.EDGE_NOT_FOUND
        assert "source" in error.message
        assert "target" in error.message
        assert error.graph_id == "test_graph"
        assert error.source == "source"
        assert error.target == "target"

    def test_graph_already_exists_error(self):
        """Test GraphAlreadyExistsError."""
        error = GraphAlreadyExistsError("test_graph")

        assert error.code == ErrorCodes.GRAPH_ALREADY_EXISTS
        assert "test_graph" in error.message
        assert error.graph_id == "test_graph"

    def test_invalid_graph_id_error(self):
        """Test InvalidGraphIdError."""
        error = InvalidGraphIdError("../invalid", "Path traversal")

        assert error.code == ErrorCodes.INVALID_GRAPH_ID
        assert "Invalid graph ID" in error.message
        assert "Path traversal" in error.message
        assert error.graph_id == "../invalid"
        # Ensure the malicious input is NOT in the error message for security
        assert "../invalid" not in error.message

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("param_name", "invalid_value", "Must be string")

        assert error.code == ErrorCodes.VALIDATION_ERROR
        assert "param_name" in error.message
        assert "Must be string" in error.message
        assert error.parameter == "param_name"
        assert error.value == "invalid_value"

    def test_algorithm_error(self):
        """Test AlgorithmError."""
        error = AlgorithmError("shortest_path", "test_graph", "No path exists")

        assert error.code == ErrorCodes.ALGORITHM_ERROR
        assert "shortest_path" in error.message
        assert "test_graph" in error.message
        assert "No path exists" in error.message


class TestErrorHandling:
    """Test error handling utility functions."""

    def test_handle_error_with_mcp_error(self):
        """Test handle_error with existing MCP error."""
        original_error = GraphNotFoundError("test_graph")
        result = handle_error(original_error, "test_operation")

        assert result == original_error.to_dict()
        assert result["code"] == ErrorCodes.GRAPH_NOT_FOUND

    def test_handle_error_with_networkx_error(self):
        """Test handle_error with NetworkX error."""
        # Mock a NetworkX error
        networkx_error = Exception("NetworkX error")
        networkx_error.__module__ = "networkx.algorithms.shortest_paths"

        result = handle_error(networkx_error, "test_operation")

        assert result["code"] == ErrorCodes.GRAPH_OPERATION_FAILED
        assert "Graph operation failed" in result["message"]
        assert "NetworkX error" in result["message"]

    def test_handle_error_with_generic_error(self):
        """Test handle_error with generic exception."""
        generic_error = ValueError("Generic error")
        result = handle_error(generic_error, "test_operation")

        assert result["code"] == ErrorCodes.INTERNAL_ERROR
        assert result["message"] == "Internal server error"
        assert result["data"]["operation"] == "test_operation"
        assert result["data"]["error_type"] == "ValueError"


class TestValidationFunctions:
    """Test input validation functions."""

    def test_validate_graph_id_valid(self):
        """Test validate_graph_id with valid inputs."""
        valid_ids = ["test", "test_graph", "test-graph", "graph123", "a" * 100]

        for valid_id in valid_ids:
            result = validate_graph_id(valid_id)
            assert result == valid_id.strip()

    def test_validate_graph_id_invalid(self):
        """Test validate_graph_id with invalid inputs."""
        invalid_cases = [
            (None, "Graph ID cannot be None"),
            (123, "Graph ID must be a string"),
            ("", "Graph ID cannot be empty"),
            ("   ", "Graph ID cannot be empty"),
            ("a" * 101, "Graph ID too long"),
            ("test@graph", "can only contain letters, numbers, underscore, and hyphen"),
            (
                "../etc/passwd",
                "can only contain letters, numbers, underscore, and hyphen",
            ),  # Caught by character check first
            (
                "test/graph",
                "can only contain letters, numbers, underscore, and hyphen",
            ),  # Caught by character check first
            (
                "test\\graph",
                "can only contain letters, numbers, underscore, and hyphen",
            ),  # Caught by character check first
            (
                "test..graph",
                "can only contain letters, numbers, underscore, and hyphen",
            ),  # Caught by character check first
        ]

        for invalid_id, expected_reason in invalid_cases:
            with pytest.raises(InvalidGraphIdError) as excinfo:
                validate_graph_id(invalid_id)
            assert expected_reason in str(excinfo.value)

    def test_validate_node_id_valid(self):
        """Test validate_node_id with valid inputs."""
        valid_ids = ["node1", "node_2", "node-3", 123, "a" * 100]

        for valid_id in valid_ids:
            result = validate_node_id(valid_id)
            assert result == str(valid_id).strip()

    def test_validate_node_id_invalid(self):
        """Test validate_node_id with invalid inputs."""
        invalid_cases = [
            (None, "Node ID cannot be None"),
            ([], "Node ID must be a string or integer"),
            ("", "Node ID cannot be empty"),
            ("   ", "Node ID cannot be empty"),
            ("a" * 101, "Node ID too long"),
        ]

        for invalid_id, expected_reason in invalid_cases:
            with pytest.raises(InvalidNodeIdError) as excinfo:
                validate_node_id(invalid_id)
            assert expected_reason in str(excinfo.value)

    def test_validate_edge_valid(self):
        """Test validate_edge with valid inputs."""
        valid_edges = [
            ["source", "target"],
            ("source", "target"),
            [1, 2],
            ("node1", "node2"),
        ]

        for valid_edge in valid_edges:
            source, target = validate_edge(valid_edge)
            assert source == str(valid_edge[0]).strip()
            assert target == str(valid_edge[1]).strip()

    def test_validate_edge_invalid(self):
        """Test validate_edge with invalid inputs."""
        invalid_cases = [
            ("not_a_list", "Edge must be a list or tuple"),
            (123, "Edge must be a list or tuple"),
            ([], "Edge must have exactly 2 elements"),
            (["single"], "Edge must have exactly 2 elements"),
            (["a", "b", "c"], "Edge must have exactly 2 elements"),
            ([None, "target"], "Invalid node in edge"),
            (["source", None], "Invalid node in edge"),
        ]

        for invalid_edge, expected_reason in invalid_cases:
            with pytest.raises(InvalidEdgeError) as excinfo:
                validate_edge(invalid_edge)
            assert expected_reason in str(excinfo.value)

    def test_validate_required_params_valid(self):
        """Test validate_required_params with valid inputs."""
        params = {"param1": "value1", "param2": "value2"}
        required = ["param1", "param2"]

        # Should not raise exception
        validate_required_params(params, required)

    def test_validate_required_params_invalid(self):
        """Test validate_required_params with invalid inputs."""
        # Missing parameter
        with pytest.raises(ValidationError) as excinfo:
            validate_required_params({}, ["required_param"])
        assert "Required parameter missing" in str(excinfo.value)

        # None parameter
        with pytest.raises(ValidationError) as excinfo:
            validate_required_params({"param": None}, ["param"])
        assert "Required parameter cannot be None" in str(excinfo.value)

    def test_validate_centrality_measures_valid(self):
        """Test validate_centrality_measures with valid inputs."""
        valid_measures = [
            ["degree"],
            ["degree", "betweenness"],
            ["degree", "betweenness", "closeness", "eigenvector"],
        ]

        for measures in valid_measures:
            result = validate_centrality_measures(measures)
            assert result == measures

    def test_validate_centrality_measures_invalid(self):
        """Test validate_centrality_measures with invalid inputs."""
        invalid_cases = [
            ("not_a_list", "Measures must be a list"),
            ([], "Measures list cannot be empty"),
            ([123], "Measure must be string"),
            (["invalid_measure"], "Invalid measure"),
            (["degree", "invalid"], "Invalid measure"),
        ]

        for invalid_measures, expected_reason in invalid_cases:
            with pytest.raises(ValidationError) as excinfo:
                validate_centrality_measures(invalid_measures)
            assert expected_reason in str(excinfo.value)


class TestServerErrorHandling:
    """Test server-level error handling with MCP protocol."""

    def setup_method(self):
        """Setup server for testing."""
        self.server = NetworkXMCPServer()
        self.server.initialized = True

    def test_create_error_response(self):
        """Test create_error_response method."""
        response = self.server.create_error_response(123, -32601, "Method not found")

        expected = {
            "jsonrpc": "2.0",
            "id": 123,
            "error": {"code": -32601, "message": "Method not found"},
        }

        assert response == expected

    def test_create_error_response_no_id(self):
        """Test create_error_response with no ID."""
        response = self.server.create_error_response(None, -32700, "Parse error")

        assert response["id"] is None
        assert response["error"]["code"] == -32700

    @pytest.mark.asyncio
    async def test_tool_create_graph_error_handling(self):
        """Test create_graph tool error handling."""
        # Test missing required parameter
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_create_graph({})
        assert excinfo.value.code == ErrorCodes.VALIDATION_ERROR

        # Test invalid graph_id
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_create_graph({"graph_id": "../invalid"})
        assert excinfo.value.code == ErrorCodes.INVALID_GRAPH_ID

        # Test invalid directed parameter
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_create_graph(
                {"graph_id": "test", "directed": "not_bool"}
            )
        assert excinfo.value.code == ErrorCodes.VALIDATION_ERROR

        # Test graph already exists
        await self.server.tool_create_graph({"graph_id": "test"})
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_create_graph({"graph_id": "test"})
        assert excinfo.value.code == ErrorCodes.GRAPH_ALREADY_EXISTS

    @pytest.mark.asyncio
    async def test_tool_add_nodes_error_handling(self):
        """Test add_nodes tool error handling."""
        # Test missing required parameters
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_add_nodes({})
        assert excinfo.value.code == ErrorCodes.VALIDATION_ERROR

        # Test invalid nodes parameter
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_add_nodes({"graph_id": "test", "nodes": "not_list"})
        assert excinfo.value.code == ErrorCodes.VALIDATION_ERROR

        # Test empty nodes list
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_add_nodes({"graph_id": "test", "nodes": []})
        assert excinfo.value.code == ErrorCodes.VALIDATION_ERROR

        # Test non-existent graph
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_add_nodes(
                {"graph_id": "nonexistent", "nodes": ["node1"]}
            )
        assert excinfo.value.code == ErrorCodes.GRAPH_NOT_FOUND

    @pytest.mark.asyncio
    async def test_tool_add_edges_error_handling(self):
        """Test add_edges tool error handling."""
        # Create a test graph first
        await self.server.tool_create_graph({"graph_id": "test"})

        # Test missing required parameters
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_add_edges({})
        assert excinfo.value.code == ErrorCodes.VALIDATION_ERROR

        # Test invalid edges parameter
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_add_edges({"graph_id": "test", "edges": "not_list"})
        assert excinfo.value.code == ErrorCodes.VALIDATION_ERROR

        # Test empty edges list
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_add_edges({"graph_id": "test", "edges": []})
        assert excinfo.value.code == ErrorCodes.VALIDATION_ERROR

        # Test invalid edge format
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_add_edges(
                {"graph_id": "test", "edges": [["single_element"]]}
            )
        assert excinfo.value.code == ErrorCodes.VALIDATION_ERROR

    @pytest.mark.asyncio
    async def test_tool_get_graph_info_error_handling(self):
        """Test get_graph_info tool error handling."""
        # Test missing required parameter
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_get_graph_info({})
        assert excinfo.value.code == ErrorCodes.VALIDATION_ERROR

        # Test non-existent graph
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_get_graph_info({"graph_id": "nonexistent"})
        assert excinfo.value.code == ErrorCodes.GRAPH_NOT_FOUND

    @pytest.mark.asyncio
    async def test_tool_shortest_path_error_handling(self):
        """Test shortest_path tool error handling."""
        # Test missing required parameters
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_shortest_path({})
        assert excinfo.value.code == ErrorCodes.VALIDATION_ERROR

        # Test non-existent graph
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_shortest_path(
                {"graph_id": "nonexistent", "source": "A", "target": "B"}
            )
        assert excinfo.value.code == ErrorCodes.GRAPH_NOT_FOUND

    @pytest.mark.asyncio
    async def test_tool_centrality_measures_error_handling(self):
        """Test centrality_measures tool error handling."""
        # Test missing required parameters
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_centrality_measures({})
        assert excinfo.value.code == ErrorCodes.VALIDATION_ERROR

        # Test invalid measures
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_centrality_measures(
                {"graph_id": "test", "measures": ["invalid_measure"]}
            )
        assert excinfo.value.code == ErrorCodes.VALIDATION_ERROR

        # Test non-existent graph
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_centrality_measures(
                {"graph_id": "nonexistent", "measures": ["degree"]}
            )
        assert excinfo.value.code == ErrorCodes.GRAPH_NOT_FOUND

    @pytest.mark.asyncio
    async def test_tool_delete_graph_error_handling(self):
        """Test delete_graph tool error handling."""
        # Test missing required parameter
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_delete_graph({})
        assert excinfo.value.code == ErrorCodes.VALIDATION_ERROR

        # Test non-existent graph
        with pytest.raises(MCPError) as excinfo:
            await self.server.tool_delete_graph({"graph_id": "nonexistent"})
        assert excinfo.value.code == ErrorCodes.GRAPH_NOT_FOUND


class TestJSONRPCProtocolErrors:
    """Test JSON-RPC protocol level error handling."""

    def setup_method(self):
        """Setup server for testing."""
        self.server = NetworkXMCPServer()

    @pytest.mark.asyncio
    async def test_invalid_json_rpc_version(self):
        """Test handling of invalid JSON-RPC version."""
        message = {"jsonrpc": "1.0", "id": 1, "method": "test"}
        response = await self.server.handle_message(message)

        assert response["error"]["code"] == -32600
        assert "Invalid Request" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_missing_method(self):
        """Test handling of missing method."""
        message = {"jsonrpc": "2.0", "id": 1}
        response = await self.server.handle_message(message)

        assert response["error"]["code"] == -32600
        assert "Missing method" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_unknown_method(self):
        """Test handling of unknown method."""
        message = {"jsonrpc": "2.0", "id": 1, "method": "unknown_method"}
        response = await self.server.handle_message(message)

        assert response["error"]["code"] == -32601
        assert "Method not found" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_server_not_initialized(self):
        """Test handling when server is not initialized."""
        message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        response = await self.server.handle_message(message)

        assert response["error"]["code"] == -32002
        assert "Server not initialized" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_tools_call_error_propagation(self):
        """Test error propagation in tools/call."""
        self.server.initialized = True

        # Test with invalid tool
        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "unknown_tool", "arguments": {}},
        }
        response = await self.server.handle_message(message)

        assert response["error"]["code"] == -32601
        assert "Unknown tool" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_tools_call_mcp_error_handling(self):
        """Test MCP error handling in tools/call."""
        self.server.initialized = True

        # Test with invalid graph_id
        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "create_graph", "arguments": {"graph_id": "../invalid"}},
        }
        response = await self.server.handle_message(message)

        assert response["error"]["code"] == ErrorCodes.INVALID_GRAPH_ID
        assert "Invalid graph ID" in response["error"]["message"]


class TestIntegrationErrorHandling:
    """Test end-to-end error handling scenarios."""

    def setup_method(self):
        """Setup server for testing."""
        self.server = NetworkXMCPServer()
        self.server.initialized = True

    @pytest.mark.asyncio
    async def test_complete_error_workflow(self):
        """Test complete error workflow through MCP protocol."""
        # 1. Try to add nodes to non-existent graph
        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "add_nodes",
                "arguments": {"graph_id": "nonexistent", "nodes": ["node1", "node2"]},
            },
        }
        response = await self.server.handle_message(message)

        # Should get graph not found error
        assert response["error"]["code"] == ErrorCodes.GRAPH_NOT_FOUND
        assert "Graph 'nonexistent' not found" in response["error"]["message"]

        # 2. Create the graph
        create_message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "create_graph", "arguments": {"graph_id": "test_graph"}},
        }
        response = await self.server.handle_message(create_message)
        assert "result" in response

        # 3. Try to create duplicate graph
        duplicate_message = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "create_graph", "arguments": {"graph_id": "test_graph"}},
        }
        response = await self.server.handle_message(duplicate_message)

        # Should get graph already exists error
        assert response["error"]["code"] == ErrorCodes.GRAPH_ALREADY_EXISTS
        assert "Graph 'test_graph' already exists" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_error_response_format_compliance(self):
        """Test that all error responses are JSON-RPC 2.0 compliant."""
        test_cases = [
            # Invalid JSON-RPC version
            {"jsonrpc": "1.0", "id": 1, "method": "test"},
            # Missing method
            {"jsonrpc": "2.0", "id": 2},
            # Unknown method
            {"jsonrpc": "2.0", "id": 3, "method": "unknown"},
        ]

        for message in test_cases:
            response = await self.server.handle_message(message)

            # Check JSON-RPC 2.0 compliance
            assert response["jsonrpc"] == "2.0"
            assert response["id"] == message.get("id")
            assert "error" in response
            assert "result" not in response

            # Check error object structure
            error = response["error"]
            assert "code" in error
            assert "message" in error
            assert isinstance(error["code"], int)
            assert isinstance(error["message"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
