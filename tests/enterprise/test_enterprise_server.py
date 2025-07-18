"""
Tests for the enterprise NetworkX MCP server integration.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip all tests if enterprise dependencies not available
pytest.importorskip("authlib")
pytest.importorskip("pydantic")

from networkx_mcp.enterprise.auth import Role, User
from networkx_mcp.enterprise.config import EnterpriseConfig
from networkx_mcp.enterprise.server import (
    EnterpriseNetworkXServer,
    RateLimitError,
    SecurityError,
    ValidationError,
)


@pytest.fixture
def mock_config():
    """Mock enterprise configuration."""
    config = EnterpriseConfig()
    config.enterprise_mode = True
    config.development_mode = True
    config.security.api_key_enabled = True
    config.security.api_keys = ["test_api_key"]
    config.security.rbac_enabled = True
    config.rate_limit.enabled = True
    config.rate_limit.requests_per_minute = 60
    config.rate_limit.burst_size = 10
    config.monitoring.metrics_enabled = True
    config.monitoring.audit_enabled = True
    return config


@pytest.fixture
def test_user():
    """Create a test user."""
    return User(id="test_user", username="Test User", roles={Role.USER})


@pytest.fixture
def admin_user():
    """Create an admin user."""
    return User(id="admin_user", username="Admin User", roles={Role.ADMIN})


@pytest.fixture
def readonly_user():
    """Create a readonly user."""
    return User(id="readonly_user", username="Readonly User", roles={Role.READONLY})


class TestEnterpriseServerInitialization:
    """Test enterprise server initialization and configuration."""

    def test_server_creation(self, mock_config):
        """Test creating enterprise server with valid config."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            assert server.config == mock_config
            assert server.auth_manager is not None
            assert server.rate_limiter is not None
            assert server.metrics is not None
            assert server.audit is not None
            assert server.running == True

    def test_server_creation_with_invalid_config(self):
        """Test server creation with invalid configuration."""
        invalid_config = EnterpriseConfig()
        invalid_config.security.api_key_enabled = True
        invalid_config.security.api_keys = []  # No API keys provided

        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=invalid_config
        ):
            with pytest.raises(ValueError) as exc_info:
                EnterpriseNetworkXServer()

            assert "configuration issues" in str(exc_info.value)
            assert "API key authentication enabled but no API keys configured" in str(
                exc_info.value
            )

    def test_error_code_mapping(self, mock_config):
        """Test error code mapping for different exception types."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            assert server._get_error_code(SecurityError("test")) == -32001
            assert server._get_error_code(RateLimitError("test")) == -32002
            assert server._get_error_code(ValidationError("test")) == -32602
            assert server._get_error_code(ValueError("test")) == -32603


class TestAuthentication:
    """Test enterprise server authentication."""

    @pytest.mark.asyncio
    async def test_valid_api_key_authentication(self, mock_config, test_user):
        """Test authentication with valid API key."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            # Mock authentication manager
            server.auth_manager.authenticate_request = MagicMock(return_value=test_user)

            request = {"headers": {"X-API-Key": "test_api_key"}, "params": {}}

            user = await server._authenticate_request(request)

            assert user == test_user
            server.auth_manager.authenticate_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_authentication_failure(self, mock_config):
        """Test authentication failure."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            # Mock authentication manager to return None
            server.auth_manager.authenticate_request = MagicMock(return_value=None)

            request = {"headers": {"X-API-Key": "invalid_key"}, "params": {}}

            with pytest.raises(SecurityError) as exc_info:
                await server._authenticate_request(request)

            assert "Authentication required" in str(exc_info.value)


class TestRequestHandling:
    """Test enterprise server request handling."""

    @pytest.mark.asyncio
    async def test_initialize_request(self, mock_config, test_user):
        """Test MCP initialize request."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()
            server._current_user = test_user

            # Mock audit context manager
            server.audit.operation_context = MagicMock()
            server.audit.operation_context.return_value.__enter__ = MagicMock()
            server.audit.operation_context.return_value.__exit__ = MagicMock()

            params = {"id": "test-id"}
            response = await server._handle_initialize(params)

            assert response["jsonrpc"] == "2.0"
            assert response["id"] == "test-id"
            assert "NetworkX MCP Server Enterprise" in response["result"]["name"]
            assert response["result"]["enterprise_mode"] == True

    @pytest.mark.asyncio
    async def test_tools_list_with_rbac(self, mock_config, readonly_user):
        """Test tools list filtering based on user permissions."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()
            server._current_user = readonly_user

            # Mock RBAC manager to return limited operations
            server.auth_manager.rbac_manager.get_allowed_operations = MagicMock(
                return_value=["get_info", "visualize_graph"]
            )

            # Mock audit context manager
            server.audit.operation_context = MagicMock()
            server.audit.operation_context.return_value.__enter__ = MagicMock()
            server.audit.operation_context.return_value.__exit__ = MagicMock()

            params = {"id": "test-id"}
            response = await server._handle_tools_list(params)

            # Should only include allowed tools
            tools = response["result"]["tools"]
            tool_names = [tool["name"] for tool in tools]

            # This is a simplified test - in reality we'd need to mock _get_all_tools
            server.auth_manager.rbac_manager.get_allowed_operations.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_call_authorization(self, mock_config, test_user):
        """Test tool call with authorization checking."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()
            server._current_user = test_user

            # Mock rate limiter
            rate_result = MagicMock()
            rate_result.allowed = True
            server.rate_limiter.check_rate_limit = MagicMock(return_value=rate_result)

            # Mock authorization
            server.auth_manager.authorize_operation = MagicMock(return_value=True)

            # Mock tool execution
            server._execute_tool = AsyncMock(return_value={"success": True})

            # Mock tracking context manager
            server.rate_limiter.operation_tracking = MagicMock()
            server.rate_limiter.operation_tracking.return_value.__enter__ = MagicMock()
            server.rate_limiter.operation_tracking.return_value.__exit__ = MagicMock()

            # Mock audit context manager
            server.audit.operation_context = MagicMock()
            server.audit.operation_context.return_value.__enter__ = MagicMock()
            server.audit.operation_context.return_value.__exit__ = MagicMock()

            params = {
                "id": "test-id",
                "name": "create_graph",
                "arguments": {"name": "test_graph"},
            }

            response = await server._handle_tool_call(params)

            # Verify authorization was checked
            server.auth_manager.authorize_operation.assert_called_once_with(
                test_user, "create_graph"
            )

            # Verify rate limiting was checked
            server.rate_limiter.check_rate_limit.assert_called_once_with(
                test_user.id, "create_graph"
            )

            assert response["jsonrpc"] == "2.0"
            assert response["id"] == "test-id"

    @pytest.mark.asyncio
    async def test_tool_call_rate_limited(self, mock_config, test_user):
        """Test tool call blocked by rate limiting."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()
            server._current_user = test_user

            # Mock rate limiter to reject request
            rate_result = MagicMock()
            rate_result.allowed = False
            rate_result.reason = "Rate limit exceeded"
            server.rate_limiter.check_rate_limit = MagicMock(return_value=rate_result)

            params = {"name": "create_graph", "arguments": {"name": "test_graph"}}

            with pytest.raises(RateLimitError) as exc_info:
                await server._handle_tool_call(params)

            assert "Rate limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tool_call_unauthorized(self, mock_config, test_user):
        """Test tool call blocked by authorization."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()
            server._current_user = test_user

            # Mock rate limiter to allow
            rate_result = MagicMock()
            rate_result.allowed = True
            server.rate_limiter.check_rate_limit = MagicMock(return_value=rate_result)

            # Mock authorization to deny
            server.auth_manager.authorize_operation = MagicMock(return_value=False)

            params = {"name": "admin_operation", "arguments": {}}

            with pytest.raises(SecurityError) as exc_info:
                await server._handle_tool_call(params)

            assert "Insufficient permissions" in str(exc_info.value)


class TestInputValidation:
    """Test input validation system."""

    def test_valid_graph_name_validation(self, mock_config):
        """Test validation of valid graph names."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            # Should not raise for valid inputs
            server._validate_tool_inputs("create_graph", {"name": "valid_graph"})
            server._validate_tool_inputs(
                "add_nodes", {"graph": "valid_graph", "nodes": ["A", "B"]}
            )

    def test_invalid_graph_name_validation(self, mock_config):
        """Test validation of invalid graph names."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            # Empty name
            with pytest.raises(ValidationError):
                server._validate_tool_inputs("create_graph", {"name": ""})

            # Too long name
            with pytest.raises(ValidationError):
                server._validate_tool_inputs("create_graph", {"name": "x" * 101})

            # Non-string name
            with pytest.raises(ValidationError):
                server._validate_tool_inputs("create_graph", {"name": 123})

    def test_missing_graph_parameter_validation(self, mock_config):
        """Test validation when required graph parameter is missing."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            with pytest.raises(ValidationError) as exc_info:
                server._validate_tool_inputs("add_nodes", {"nodes": ["A", "B"]})

            assert "Graph name is required" in str(exc_info.value)

    def test_graph_size_limits_validation(self, mock_config):
        """Test validation of graph size limits."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            # Large number of nodes
            large_nodes = list(range(mock_config.rate_limit.max_graph_size + 1))
            with pytest.raises(ValidationError) as exc_info:
                server._validate_tool_inputs(
                    "add_nodes", {"graph": "test", "nodes": large_nodes}
                )

            assert "Too many nodes" in str(exc_info.value)

            # Large number of edges
            large_edges = [
                (i, i + 1) for i in range(mock_config.rate_limit.max_graph_size + 1)
            ]
            with pytest.raises(ValidationError) as exc_info:
                server._validate_tool_inputs(
                    "add_edges", {"graph": "test", "edges": large_edges}
                )

            assert "Too many edges" in str(exc_info.value)

    def test_csv_size_limit_validation(self, mock_config):
        """Test validation of CSV import size limits."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            # Large CSV data (> 10MB)
            large_csv = "A,B\n" * (3 * 1024 * 1024)  # ~12MB
            with pytest.raises(ValidationError) as exc_info:
                server._validate_tool_inputs(
                    "import_csv", {"graph": "test", "csv_data": large_csv}
                )

            assert "CSV data too large" in str(exc_info.value)


class TestToolExecution:
    """Test tool execution functionality."""

    @pytest.mark.asyncio
    async def test_create_graph_execution(self, mock_config):
        """Test create_graph tool execution."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            result = await server._execute_tool(
                "create_graph", {"name": "test_graph", "directed": False}
            )

            assert result["created"] == "test_graph"
            assert result["type"] == "undirected"

    @pytest.mark.asyncio
    async def test_add_nodes_execution(self, mock_config):
        """Test add_nodes tool execution."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            # First create a graph
            await server._execute_tool("create_graph", {"name": "test_graph"})

            # Then add nodes
            result = await server._execute_tool(
                "add_nodes", {"graph": "test_graph", "nodes": ["A", "B", "C"]}
            )

            assert result["added"] == 3
            assert result["total"] == 3

    @pytest.mark.asyncio
    async def test_unknown_tool_execution(self, mock_config):
        """Test execution of unknown tool."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            with pytest.raises(ValueError) as exc_info:
                await server._execute_tool("unknown_tool", {})

            assert "Unknown tool" in str(exc_info.value)


class TestRequestFlow:
    """Test complete request flow integration."""

    @pytest.mark.asyncio
    async def test_successful_request_flow(self, mock_config, test_user):
        """Test complete successful request flow."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            # Mock authentication
            server.auth_manager.authenticate_request = MagicMock(return_value=test_user)

            # Mock rate limiting
            rate_result = MagicMock()
            rate_result.allowed = True
            server.rate_limiter.check_rate_limit = MagicMock(return_value=rate_result)

            # Mock authorization
            server.auth_manager.authorize_operation = MagicMock(return_value=True)

            # Mock tracking context managers
            server.rate_limiter.operation_tracking = MagicMock()
            server.rate_limiter.operation_tracking.return_value.__enter__ = MagicMock()
            server.rate_limiter.operation_tracking.return_value.__exit__ = MagicMock()

            server.audit.operation_context = MagicMock()
            server.audit.operation_context.return_value.__enter__ = MagicMock()
            server.audit.operation_context.return_value.__exit__ = MagicMock()

            # Mock metrics and audit
            server.metrics.record_auth_attempt = MagicMock()
            server.metrics.record_request = MagicMock()
            server.audit.set_request_context = MagicMock()

            request = {
                "jsonrpc": "2.0",
                "id": "test-request",
                "method": "tools/call",
                "params": {"name": "create_graph", "arguments": {"name": "test_graph"}},
                "headers": {"X-API-Key": "test_api_key"},
            }

            response = await server.handle_request(request)

            assert response["jsonrpc"] == "2.0"
            assert response["id"] == "test-request"
            assert "result" in response

            # Verify metrics were recorded
            server.metrics.record_auth_attempt.assert_called()
            server.metrics.record_request.assert_called()

    @pytest.mark.asyncio
    async def test_failed_request_flow(self, mock_config):
        """Test request flow with authentication failure."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            # Mock authentication failure
            server.auth_manager.authenticate_request = MagicMock(return_value=None)

            # Mock metrics and audit
            server.metrics.record_auth_attempt = MagicMock()
            server.metrics.record_request = MagicMock()
            server.metrics.record_error = MagicMock()
            server.audit.set_request_context = MagicMock()
            server.audit.log_security_event = MagicMock()

            request = {
                "jsonrpc": "2.0",
                "id": "test-request",
                "method": "tools/call",
                "params": {"name": "create_graph", "arguments": {"name": "test_graph"}},
                "headers": {"X-API-Key": "invalid_key"},
            }

            response = await server.handle_request(request)

            assert response["jsonrpc"] == "2.0"
            assert response["id"] == "test-request"
            assert "error" in response
            assert response["error"]["code"] == -32001  # Security error
            assert "Authentication required" in response["error"]["message"]

            # Verify error was recorded
            server.metrics.record_error.assert_called()
            server.audit.log_security_event.assert_called()

    @pytest.mark.asyncio
    async def test_rate_limited_request_flow(self, mock_config, test_user):
        """Test request flow with rate limiting."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            # Mock authentication success
            server.auth_manager.authenticate_request = MagicMock(return_value=test_user)

            # Mock rate limiting failure
            rate_result = MagicMock()
            rate_result.allowed = False
            rate_result.reason = "Rate limit exceeded"
            server.rate_limiter.check_rate_limit = MagicMock(return_value=rate_result)

            # Mock authorization success
            server.auth_manager.authorize_operation = MagicMock(return_value=True)

            # Mock metrics and audit
            server.metrics.record_auth_attempt = MagicMock()
            server.metrics.record_request = MagicMock()
            server.metrics.record_error = MagicMock()
            server.audit.set_request_context = MagicMock()
            server.audit.log_security_event = MagicMock()

            request = {
                "jsonrpc": "2.0",
                "id": "test-request",
                "method": "tools/call",
                "params": {"name": "create_graph", "arguments": {"name": "test_graph"}},
                "headers": {"X-API-Key": "test_api_key"},
            }

            response = await server.handle_request(request)

            assert response["jsonrpc"] == "2.0"
            assert response["id"] == "test-request"
            assert "error" in response
            assert response["error"]["code"] == -32002  # Rate limit error

            # Verify rate limiting was checked
            server.rate_limiter.check_rate_limit.assert_called()


@pytest.mark.integration
class TestEnterpriseServerIntegration:
    """Full integration tests for enterprise server."""

    @pytest.mark.asyncio
    async def test_complete_graph_workflow(self, mock_config, test_user):
        """Test complete graph creation and analysis workflow."""
        with patch(
            "networkx_mcp.enterprise.server.get_config", return_value=mock_config
        ):
            server = EnterpriseNetworkXServer()

            # Setup mocks for successful authentication and authorization
            server.auth_manager.authenticate_request = MagicMock(return_value=test_user)
            server.auth_manager.authorize_operation = MagicMock(return_value=True)

            # Setup rate limiting to allow requests
            rate_result = MagicMock()
            rate_result.allowed = True
            server.rate_limiter.check_rate_limit = MagicMock(return_value=rate_result)

            # Mock tracking context managers
            server.rate_limiter.operation_tracking = MagicMock()
            server.rate_limiter.operation_tracking.return_value.__enter__ = MagicMock()
            server.rate_limiter.operation_tracking.return_value.__exit__ = MagicMock()

            server.audit.operation_context = MagicMock()
            server.audit.operation_context.return_value.__enter__ = MagicMock()
            server.audit.operation_context.return_value.__exit__ = MagicMock()

            # Mock metrics
            server.metrics.record_auth_attempt = MagicMock()
            server.metrics.record_request = MagicMock()
            server.metrics.record_graph_metrics = MagicMock()
            server.audit.set_request_context = MagicMock()

            # 1. Create graph
            create_request = {
                "jsonrpc": "2.0",
                "id": "create-1",
                "method": "tools/call",
                "params": {
                    "name": "create_graph",
                    "arguments": {"name": "social_network", "directed": False},
                },
                "headers": {"X-API-Key": "test_api_key"},
            }

            response = await server.handle_request(create_request)
            assert "result" in response

            # 2. Add nodes
            add_nodes_request = {
                "jsonrpc": "2.0",
                "id": "add-nodes-1",
                "method": "tools/call",
                "params": {
                    "name": "add_nodes",
                    "arguments": {
                        "graph": "social_network",
                        "nodes": ["Alice", "Bob", "Charlie", "David"],
                    },
                },
                "headers": {"X-API-Key": "test_api_key"},
            }

            response = await server.handle_request(add_nodes_request)
            assert "result" in response

            # 3. Add edges
            add_edges_request = {
                "jsonrpc": "2.0",
                "id": "add-edges-1",
                "method": "tools/call",
                "params": {
                    "name": "add_edges",
                    "arguments": {
                        "graph": "social_network",
                        "edges": [
                            ["Alice", "Bob"],
                            ["Bob", "Charlie"],
                            ["Charlie", "David"],
                        ],
                    },
                },
                "headers": {"X-API-Key": "test_api_key"},
            }

            response = await server.handle_request(add_edges_request)
            assert "result" in response

            # 4. Analyze centrality
            centrality_request = {
                "jsonrpc": "2.0",
                "id": "centrality-1",
                "method": "tools/call",
                "params": {
                    "name": "degree_centrality",
                    "arguments": {"graph": "social_network"},
                },
                "headers": {"X-API-Key": "test_api_key"},
            }

            response = await server.handle_request(centrality_request)
            assert "result" in response

            # Verify all security checks were performed
            assert server.auth_manager.authenticate_request.call_count == 4
            assert server.auth_manager.authorize_operation.call_count == 4
            assert server.rate_limiter.check_rate_limit.call_count == 4

            # Verify metrics were recorded
            assert server.metrics.record_request.call_count == 4
            assert (
                server.metrics.record_graph_metrics.call_count >= 2
            )  # For add operations
