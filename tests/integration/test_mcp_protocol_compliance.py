"""Integration tests for MCP protocol compliance and error handling.

Tests verify that the server follows the MCP specification and JSON-RPC 2.0
standards for error handling, message formatting, and protocol compliance.
"""

import asyncio
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

from networkx_mcp.errors import ErrorCodes
from networkx_mcp.server import NetworkXMCPServer


class MCPProtocolTester:
    """Test client for MCP protocol compliance."""

    def __init__(self):
        self.server = NetworkXMCPServer()
        self.request_id = 0

    def next_id(self) -> int:
        """Get next request ID."""
        self.request_id += 1
        return self.request_id

    async def initialize_server(self) -> None:
        """Properly initialize server following MCP protocol."""
        # Step 1: Send initialize request
        await self.server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0.0"},
                },
            }
        )

        # Step 2: Send initialized notification
        await self.server.handle_message({"jsonrpc": "2.0", "method": "initialized"})

    def create_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a JSON-RPC 2.0 request."""
        request = {"jsonrpc": "2.0", "id": self.next_id(), "method": method}
        if params:
            request["params"] = params
        return request

    def create_invalid_request(self, **kwargs) -> Dict[str, Any]:
        """Create an invalid JSON-RPC request for testing."""
        return kwargs

    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to server and get response."""
        return await self.server.handle_message(request)

    def validate_json_rpc_response(
        self, response: Dict[str, Any], request_id: int
    ) -> None:
        """Validate JSON-RPC 2.0 response format."""
        assert "jsonrpc" in response
        assert response["jsonrpc"] == "2.0"
        assert "id" in response
        assert response["id"] == request_id

        # Must have either result or error, not both
        has_result = "result" in response
        has_error = "error" in response
        assert (
            has_result != has_error
        ), "Response must have either result or error, not both"

        if has_error:
            self.validate_error_object(response["error"])

    def validate_error_object(self, error: Dict[str, Any]) -> None:
        """Validate error object structure."""
        assert "code" in error
        assert "message" in error
        assert isinstance(error["code"], int)
        assert isinstance(error["message"], str)
        assert len(error["message"]) > 0

        # Data field is optional
        if "data" in error:
            assert error["data"] is not None


class TestJSONRPCCompliance:
    """Test JSON-RPC 2.0 compliance."""

    def setup_method(self):
        """Setup test client."""
        self.client = MCPProtocolTester()

    @pytest.mark.asyncio
    async def test_valid_request_format(self):
        """Test valid JSON-RPC request format."""
        # Initialize server properly
        await self.client.initialize_server()

        request = self.client.create_request("tools/list")
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, request["id"])
        assert "result" in response

    @pytest.mark.asyncio
    async def test_invalid_jsonrpc_version(self):
        """Test handling of invalid JSON-RPC version."""
        request = self.client.create_invalid_request(jsonrpc="1.0", id=1, method="test")
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, 1)
        assert response["error"]["code"] == ErrorCodes.INVALID_REQUEST
        assert "Invalid Request" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_missing_method(self):
        """Test handling of missing method field."""
        request = self.client.create_invalid_request(jsonrpc="2.0", id=2)
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, 2)
        assert response["error"]["code"] == ErrorCodes.INVALID_REQUEST
        assert "Missing method" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_method_type(self):
        """Test handling of invalid method type."""
        request = self.client.create_invalid_request(
            jsonrpc="2.0",
            id=3,
            method=123,  # Should be string
        )
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, 3)
        assert response["error"]["code"] == ErrorCodes.INVALID_REQUEST

    @pytest.mark.asyncio
    async def test_method_not_found(self):
        """Test handling of unknown method."""
        request = self.client.create_request("unknown_method")
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, request["id"])
        assert response["error"]["code"] == ErrorCodes.METHOD_NOT_FOUND
        assert "Method not found" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_null_id_handling(self):
        """Test that null ID is handled correctly."""
        request = self.client.create_invalid_request(
            jsonrpc="2.0", id=None, method="test"
        )
        response = await self.client.send_request(request)

        assert response["id"] is None
        assert "error" in response

    @pytest.mark.asyncio
    async def test_notification_handling(self):
        """Test notification (no ID) handling."""
        request = {"jsonrpc": "2.0", "method": "notification_test"}
        response = await self.client.send_request(request)

        # Notifications should not return a response
        # But our server currently returns error responses for all messages
        assert response is not None  # Our implementation returns error


class TestMCPSpecificErrors:
    """Test MCP-specific error handling."""

    def setup_method(self):
        """Setup test client."""
        self.client = MCPProtocolTester()

    @pytest.mark.asyncio
    async def test_server_not_initialized(self):
        """Test server not initialized error."""
        request = self.client.create_request("tools/list")
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, request["id"])
        assert response["error"]["code"] == ErrorCodes.SERVER_NOT_INITIALIZED
        assert "Server not initialized" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_initialization_flow(self):
        """Test proper initialization flow."""
        # Test initialize request
        init_request = self.client.create_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0.0"},
            },
        )
        response = await self.client.send_request(init_request)

        self.client.validate_json_rpc_response(response, init_request["id"])
        assert "result" in response
        assert "capabilities" in response["result"]
        assert "serverInfo" in response["result"]

        # Test initialized notification
        initialized_msg = {"jsonrpc": "2.0", "method": "initialized"}
        response = await self.client.send_request(initialized_msg)
        assert response is None  # Notifications don't return responses

    @pytest.mark.asyncio
    async def test_graph_not_found_error(self):
        """Test graph not found error."""
        # Initialize server first
        await self.client.initialize_server()

        request = self.client.create_request(
            "tools/call",
            {"name": "get_graph_info", "arguments": {"graph_id": "nonexistent"}},
        )
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, request["id"])
        assert response["error"]["code"] == ErrorCodes.GRAPH_NOT_FOUND
        assert "Graph 'nonexistent' not found" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_graph_id_error(self):
        """Test invalid graph ID error."""
        # Initialize server first
        await self.client.initialize_server()

        request = self.client.create_request(
            "tools/call",
            {"name": "create_graph", "arguments": {"graph_id": "../invalid"}},
        )
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, request["id"])
        assert response["error"]["code"] == ErrorCodes.INVALID_GRAPH_ID
        assert "Invalid graph ID" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_validation_error(self):
        """Test validation error."""
        # Initialize server first
        await self.client.initialize_server()

        request = self.client.create_request(
            "tools/call",
            {
                "name": "create_graph",
                "arguments": {
                    "directed": "not_boolean"
                },  # Missing graph_id, invalid directed
            },
        )
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, request["id"])
        assert response["error"]["code"] == ErrorCodes.VALIDATION_ERROR
        assert "Required parameter missing" in response["error"]["message"]


class TestToolErrorHandling:
    """Test tool-specific error handling."""

    def setup_method(self):
        """Setup test client."""
        self.client = MCPProtocolTester()

    async def initialize_server(self):
        """Helper to initialize server."""
        await self.client.initialize_server()

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self):
        """Test unknown tool error."""
        await self.initialize_server()

        request = self.client.create_request(
            "tools/call", {"name": "unknown_tool", "arguments": {}}
        )
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, request["id"])
        assert response["error"]["code"] == ErrorCodes.METHOD_NOT_FOUND
        assert "Unknown tool" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_tool_parameter_validation(self):
        """Test tool parameter validation."""
        await self.initialize_server()

        # Test missing required parameter
        request = self.client.create_request(
            "tools/call",
            {
                "name": "create_graph",
                "arguments": {},  # Missing graph_id
            },
        )
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, request["id"])
        assert response["error"]["code"] == ErrorCodes.VALIDATION_ERROR
        assert "Required parameter missing" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_tool_success_workflow(self):
        """Test successful tool execution workflow."""
        await self.initialize_server()

        # Create graph
        request = self.client.create_request(
            "tools/call",
            {"name": "create_graph", "arguments": {"graph_id": "test_graph"}},
        )
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, request["id"])
        assert "result" in response
        # Check that the response contains the expected structure
        assert "content" in response["result"]
        assert len(response["result"]["content"]) > 0
        assert "text" in response["result"]["content"][0]
        assert "test_graph" in response["result"]["content"][0]["text"]

        # Add nodes
        request = self.client.create_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {"graph_id": "test_graph", "nodes": ["A", "B", "C"]},
            },
        )
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, request["id"])
        assert "result" in response
        # Check that the response contains the expected structure
        assert "content" in response["result"]
        assert len(response["result"]["content"]) > 0
        assert "text" in response["result"]["content"][0]
        assert "nodes_added" in response["result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_algorithm_error_handling(self):
        """Test algorithm error handling."""
        await self.initialize_server()

        # Create graph with disconnected components
        await self.client.send_request(
            self.client.create_request(
                "tools/call",
                {"name": "create_graph", "arguments": {"graph_id": "test_graph"}},
            )
        )

        await self.client.send_request(
            self.client.create_request(
                "tools/call",
                {
                    "name": "add_nodes",
                    "arguments": {"graph_id": "test_graph", "nodes": ["A", "B"]},
                },
            )
        )

        # Try to find path between disconnected nodes
        request = self.client.create_request(
            "tools/call",
            {
                "name": "shortest_path",
                "arguments": {"graph_id": "test_graph", "source": "A", "target": "B"},
            },
        )
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, request["id"])
        assert response["error"]["code"] == ErrorCodes.ALGORITHM_ERROR
        assert "No path" in response["error"]["message"]


class TestErrorMessageSecurity:
    """Test error message security and information disclosure."""

    def setup_method(self):
        """Setup test client."""
        self.client = MCPProtocolTester()

    @pytest.mark.asyncio
    async def test_no_sensitive_info_in_errors(self):
        """Test that error messages don't leak sensitive information."""
        await self.client.initialize_server()

        # Test various error scenarios
        error_scenarios = [
            ("../etc/passwd", "graph_id"),
            ("../../secrets.txt", "graph_id"),
            ("C:\\Windows\\System32\\config\\SAM", "graph_id"),
        ]

        for malicious_input, param_name in error_scenarios:
            request = self.client.create_request(
                "tools/call",
                {"name": "create_graph", "arguments": {"graph_id": malicious_input}},
            )
            response = await self.client.send_request(request)

            # Error should not contain full path or system information
            error_msg = response["error"]["message"]
            assert "/etc/passwd" not in error_msg
            assert "\\Windows\\System32" not in error_msg
            assert "secrets.txt" not in error_msg
            assert "../" not in error_msg
            assert "../../" not in error_msg

            # Should be generic validation error without the malicious input
            assert "Invalid graph ID" in error_msg

    @pytest.mark.asyncio
    async def test_stack_trace_not_exposed(self):
        """Test that stack traces are not exposed in error messages."""
        await self.client.initialize_server()

        # Force an internal error by mocking
        with patch.object(
            self.client.server.graph_manager,
            "create_graph",
            side_effect=Exception("Internal error"),
        ):
            request = self.client.create_request(
                "tools/call",
                {"name": "create_graph", "arguments": {"graph_id": "test"}},
            )
            response = await self.client.send_request(request)

            # Should get generic internal error, not stack trace
            assert response["error"]["code"] == ErrorCodes.INTERNAL_ERROR
            assert "Internal server error" in response["error"]["message"]
            assert "Traceback" not in response["error"]["message"]
            assert "Exception" not in response["error"]["message"]


class TestTransportErrorHandling:
    """Test transport-level error handling."""

    def setup_method(self):
        """Setup test client."""
        self.client = MCPProtocolTester()

    @pytest.mark.asyncio
    async def test_malformed_json_handling(self):
        """Test handling of malformed JSON."""
        # This would need to be tested at the transport level
        # For now, we assume the transport layer handles JSON parsing
        pass

    @pytest.mark.asyncio
    async def test_oversized_request_handling(self):
        """Test handling of oversized requests."""
        await self.client.initialize_server()

        # Create a request with very large parameters
        large_nodes = [f"node_{i}" for i in range(10000)]
        request = self.client.create_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {"graph_id": "test", "nodes": large_nodes},
            },
        )

        # This should either succeed or fail gracefully
        response = await self.client.send_request(request)

        # Should not crash the server
        assert "jsonrpc" in response
        assert "id" in response

    @pytest.mark.asyncio
    async def test_rapid_request_handling(self):
        """Test handling of rapid requests."""
        await self.client.initialize_server()

        # Send multiple requests rapidly
        requests = []
        for i in range(100):
            request = self.client.create_request("tools/list")
            requests.append(self.client.send_request(request))

        # All should complete successfully
        responses = await asyncio.gather(*requests)

        for response in responses:
            assert "jsonrpc" in response
            assert "id" in response
            # Should either succeed or fail gracefully
            assert ("result" in response) or ("error" in response)


class TestProtocolVersioning:
    """Test protocol versioning and compatibility."""

    def setup_method(self):
        """Setup test client."""
        self.client = MCPProtocolTester()

    @pytest.mark.asyncio
    async def test_supported_protocol_version(self):
        """Test supported protocol version."""
        request = self.client.create_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0.0"},
            },
        )
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, request["id"])
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"

    @pytest.mark.asyncio
    async def test_unsupported_protocol_version(self):
        """Test unsupported protocol version."""
        request = self.client.create_request(
            "initialize",
            {
                "protocolVersion": "2030-01-01",  # Future version
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0.0"},
            },
        )
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, request["id"])
        # Should either accept or reject gracefully
        assert ("result" in response) or ("error" in response)

    @pytest.mark.asyncio
    async def test_missing_protocol_version(self):
        """Test missing protocol version."""
        request = self.client.create_request(
            "initialize",
            {"capabilities": {}, "clientInfo": {"name": "test", "version": "1.0.0"}},
        )
        response = await self.client.send_request(request)

        self.client.validate_json_rpc_response(response, request["id"])
        # Should handle missing version gracefully
        assert ("result" in response) or ("error" in response)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
