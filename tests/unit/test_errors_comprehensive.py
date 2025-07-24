"""Comprehensive tests for errors module - Target: 32% → 90%+ coverage (173 lines).

This test suite aims to boost coverage of the errors module from 32% to 90%+.
"""

import pytest

from networkx_mcp.errors import (
    AlgorithmError,
    EdgeNotFoundError,
    ErrorCodes,
    GraphAlreadyExistsError,
    GraphNotFoundError,
    GraphOperationError,
    InvalidEdgeError,
    InvalidGraphIdError,
    InvalidNodeIdError,
    MCPError,
    NodeNotFoundError,
    ResourceLimitExceededError,
    ServerNotInitializedError,
    ValidationError,
    handle_error,
    validate_centrality_measures,
    validate_edge,
    validate_graph_id,
    validate_node_id,
    validate_required_params,
)


class TestErrorCodes:
    """Test ErrorCodes class constants."""

    def test_json_rpc_standard_codes(self):
        """Test JSON-RPC 2.0 standard error codes."""
        assert ErrorCodes.PARSE_ERROR == -32700
        assert ErrorCodes.INVALID_REQUEST == -32600
        assert ErrorCodes.METHOD_NOT_FOUND == -32601
        assert ErrorCodes.INVALID_PARAMS == -32602
        assert ErrorCodes.INTERNAL_ERROR == -32603

    def test_mcp_specific_codes(self):
        """Test MCP-specific error codes."""
        assert ErrorCodes.GRAPH_NOT_FOUND == -32001
        assert ErrorCodes.NODE_NOT_FOUND == -32002
        assert ErrorCodes.EDGE_NOT_FOUND == -32003
        assert ErrorCodes.GRAPH_ALREADY_EXISTS == -32004
        assert ErrorCodes.INVALID_GRAPH_ID == -32005
        assert ErrorCodes.INVALID_NODE_ID == -32006
        assert ErrorCodes.INVALID_EDGE == -32007
        assert ErrorCodes.GRAPH_OPERATION_FAILED == -32008
        assert ErrorCodes.ALGORITHM_ERROR == -32009
        assert ErrorCodes.VALIDATION_ERROR == -32010
        assert ErrorCodes.RESOURCE_LIMIT_EXCEEDED == -32011
        assert ErrorCodes.SERVER_NOT_INITIALIZED == -32002  # Note: duplicate value


class TestMCPError:
    """Test base MCPError class."""

    def test_mcp_error_basic(self):
        """Test basic MCPError initialization."""
        error = MCPError(-32001, "Test error")
        assert error.code == -32001
        assert error.message == "Test error"
        assert error.data is None
        assert str(error) == "Test error"

    def test_mcp_error_with_data(self):
        """Test MCPError with additional data."""
        data = {"graph_id": "test", "details": "Not found"}
        error = MCPError(-32001, "Graph not found", data)
        assert error.code == -32001
        assert error.message == "Graph not found"
        assert error.data == data

    def test_mcp_error_to_dict(self):
        """Test MCPError to_dict method."""
        error = MCPError(-32001, "Test error", {"key": "value"})
        result = error.to_dict()
        assert result["code"] == -32001
        assert result["message"] == "Test error"
        assert result["data"] == {"key": "value"}

    def test_mcp_error_to_dict_no_data(self):
        """Test MCPError to_dict without data."""
        error = MCPError(-32001, "Test error")
        result = error.to_dict()
        assert result["code"] == -32001
        assert result["message"] == "Test error"
        assert "data" not in result


class TestSpecificErrors:
    """Test specific error classes."""

    def test_mcp_error_subclasses(self):
        """Test that all error classes inherit from MCPError."""
        error_classes = [
            GraphNotFoundError("test"),
            NodeNotFoundError("node", "graph"),
            EdgeNotFoundError("A", "B", "graph"),
            GraphAlreadyExistsError("test"),
            InvalidGraphIdError("123"),
            InvalidNodeIdError("node"),
            InvalidEdgeError("msg"),
            GraphOperationError("msg"),
            AlgorithmError("algo", "msg"),
            ValidationError(["error"]),
            ResourceLimitExceededError("resource", 100, 200),
            ServerNotInitializedError(),
        ]

        for error in error_classes:
            assert isinstance(error, MCPError)
            assert hasattr(error, "code")
            assert hasattr(error, "message")
            assert hasattr(error, "to_dict")

    def test_graph_not_found_error(self):
        """Test GraphNotFoundError."""
        error = GraphNotFoundError("test_graph")
        assert error.code == ErrorCodes.GRAPH_NOT_FOUND
        assert "test_graph" in error.message
        assert error.data == {"graph_id": "test_graph"}

    def test_node_not_found_error(self):
        """Test NodeNotFoundError."""
        error = NodeNotFoundError("node1", "graph1")
        assert error.code == ErrorCodes.NODE_NOT_FOUND
        assert "node1" in error.message
        assert "graph1" in error.message
        assert error.data == {"node_id": "node1", "graph_id": "graph1"}

    def test_edge_not_found_error(self):
        """Test EdgeNotFoundError."""
        error = EdgeNotFoundError("A", "B", "graph1")
        assert error.code == ErrorCodes.EDGE_NOT_FOUND
        assert "A" in error.message
        assert "B" in error.message
        assert error.data == {"source": "A", "target": "B", "graph_id": "graph1"}

    def test_graph_already_exists_error(self):
        """Test GraphAlreadyExistsError."""
        error = GraphAlreadyExistsError("existing_graph")
        assert error.code == ErrorCodes.GRAPH_ALREADY_EXISTS
        assert "existing_graph" in error.message
        assert error.data == {"graph_id": "existing_graph"}

    def test_invalid_graph_id_error(self):
        """Test InvalidGraphIdError."""
        error = InvalidGraphIdError("123-invalid")
        assert error.code == ErrorCodes.INVALID_GRAPH_ID
        assert "123-invalid" in error.message
        assert error.data == {"graph_id": "123-invalid"}

    def test_invalid_node_id_error(self):
        """Test InvalidNodeIdError."""
        error = InvalidNodeIdError(None)
        assert error.code == ErrorCodes.INVALID_NODE_ID
        assert "None" in error.message
        assert error.data == {"node_id": None}

    def test_invalid_edge_error(self):
        """Test InvalidEdgeError."""
        error = InvalidEdgeError(
            "Self-loops not allowed", {"source": "A", "target": "A"}
        )
        assert error.code == ErrorCodes.INVALID_EDGE
        assert "Self-loops not allowed" in error.message
        assert error.data == {"source": "A", "target": "A"}

    def test_graph_operation_error(self):
        """Test GraphOperationError."""
        error = GraphOperationError("Cannot merge graphs", {"operation": "merge"})
        assert error.code == ErrorCodes.GRAPH_OPERATION_FAILED
        assert "Cannot merge graphs" in error.message
        assert error.data == {"operation": "merge"}

    def test_algorithm_error(self):
        """Test AlgorithmError."""
        error = AlgorithmError(
            "shortest_path", "No path exists", {"source": "A", "target": "Z"}
        )
        assert error.code == ErrorCodes.ALGORITHM_ERROR
        assert "shortest_path" in error.message
        assert "No path exists" in error.message
        assert error.data == {
            "algorithm": "shortest_path",
            "details": {"source": "A", "target": "Z"},
        }

    def test_validation_error(self):
        """Test ValidationError."""
        errors = ["Invalid node", "Invalid edge"]
        error = ValidationError(errors)
        assert error.code == ErrorCodes.VALIDATION_ERROR
        assert "2 validation errors" in error.message
        assert error.data == {"errors": errors}

    def test_validation_error_single(self):
        """Test ValidationError with single error."""
        error = ValidationError(["Single error"])
        assert "1 validation error" in error.message

    def test_resource_limit_exceeded_error(self):
        """Test ResourceLimitExceededError."""
        error = ResourceLimitExceededError("nodes", 1000, 1500)
        assert error.code == ErrorCodes.RESOURCE_LIMIT_EXCEEDED
        assert "nodes" in error.message
        assert "1000" in error.message
        assert "1500" in error.message
        assert error.data == {"resource": "nodes", "limit": 1000, "requested": 1500}

    def test_server_not_initialized_error(self):
        """Test ServerNotInitializedError."""
        error = ServerNotInitializedError()
        assert error.code == ErrorCodes.SERVER_NOT_INITIALIZED
        assert "not initialized" in error.message.lower()


class TestValidationFunctions:
    """Test validation functions."""

    def test_validate_graph_id_valid(self):
        """Test validate_graph_id with valid IDs."""
        assert validate_graph_id("test_graph") == "test_graph"
        assert validate_graph_id("graph123") == "graph123"
        assert validate_graph_id("my-graph") == "my-graph"

    def test_validate_graph_id_invalid(self):
        """Test validate_graph_id with invalid IDs."""
        with pytest.raises(InvalidGraphIdError):
            validate_graph_id("")

        with pytest.raises(InvalidGraphIdError):
            validate_graph_id(None)

        with pytest.raises(InvalidGraphIdError):
            validate_graph_id(123)

        with pytest.raises(InvalidGraphIdError):
            validate_graph_id(["not", "a", "string"])

    def test_validate_node_id_valid(self):
        """Test validate_node_id with valid IDs."""
        assert validate_node_id("node1") == "node1"
        assert validate_node_id(123) == "123"
        assert validate_node_id(0) == "0"

    def test_validate_node_id_invalid(self):
        """Test validate_node_id with invalid IDs."""
        with pytest.raises(InvalidNodeIdError):
            validate_node_id("")

        with pytest.raises(InvalidNodeIdError):
            validate_node_id(None)

        with pytest.raises(InvalidNodeIdError):
            validate_node_id([])

    def test_validate_edge_valid(self):
        """Test validate_edge with valid edges."""
        assert validate_edge(("A", "B")) == ("A", "B")
        assert validate_edge([1, 2]) == ("1", "2")
        assert validate_edge(("node1", "node2")) == ("node1", "node2")

    def test_validate_edge_invalid(self):
        """Test validate_edge with invalid edges."""
        with pytest.raises(InvalidEdgeError):
            validate_edge("not a tuple")

        with pytest.raises(InvalidEdgeError):
            validate_edge(("A",))  # Only one element

        with pytest.raises(InvalidEdgeError):
            validate_edge(("A", "B", "C"))  # Too many elements

        with pytest.raises(InvalidEdgeError):
            validate_edge((None, "B"))

        with pytest.raises(InvalidEdgeError):
            validate_edge(("A", None))

    def test_validate_required_params_valid(self):
        """Test validate_required_params with all required params."""
        params = {"graph_id": "test", "node_id": "node1", "data": 123}
        validate_required_params(params, ["graph_id", "node_id"])  # Should not raise

    def test_validate_required_params_missing(self):
        """Test validate_required_params with missing params."""
        params = {"graph_id": "test"}

        with pytest.raises(ValidationError) as exc_info:
            validate_required_params(params, ["graph_id", "node_id", "data"])

        error = exc_info.value
        assert "2 validation errors" in error.message
        assert "node_id" in str(error.data)
        assert "data" in str(error.data)

    def test_validate_centrality_measures_valid(self):
        """Test validate_centrality_measures with valid measures."""
        result = validate_centrality_measures(["degree", "betweenness"])
        assert result == ["degree", "betweenness"]

        # Test single measure as string
        result = validate_centrality_measures("degree")
        assert result == ["degree"]

    def test_validate_centrality_measures_invalid(self):
        """Test validate_centrality_measures with invalid measures."""
        with pytest.raises(ValidationError):
            validate_centrality_measures(["degree", "invalid_measure"])

        with pytest.raises(ValidationError):
            validate_centrality_measures("not_a_valid_measure")

        with pytest.raises(ValidationError):
            validate_centrality_measures(123)  # Not a string or list


class TestHandleError:
    """Test handle_error function."""

    def test_handle_error_with_mcp_error(self):
        """Test handle_error with MCPError instance."""
        error = GraphNotFoundError("test_graph")
        result = handle_error(error, "get_graph")

        assert result["code"] == ErrorCodes.GRAPH_NOT_FOUND
        assert "test_graph" in result["message"]
        assert result["data"]["graph_id"] == "test_graph"

    def test_handle_error_with_standard_exception(self):
        """Test handle_error with standard exception."""
        error = ValueError("Invalid value")
        result = handle_error(error, "test_operation")

        assert result["code"] == ErrorCodes.INTERNAL_ERROR
        assert "Invalid value" in result["message"]
        assert result["data"]["operation"] == "test_operation"
        assert result["data"]["error_type"] == "ValueError"

    def test_handle_error_with_key_error(self):
        """Test handle_error with KeyError."""
        error = KeyError("missing_key")
        result = handle_error(error, "lookup")

        assert result["code"] == ErrorCodes.INVALID_PARAMS
        assert "missing_key" in result["message"]

    def test_handle_error_with_attribute_error(self):
        """Test handle_error with AttributeError."""
        error = AttributeError("'NoneType' object has no attribute 'graph'")
        result = handle_error(error, "access")

        assert result["code"] == ErrorCodes.INTERNAL_ERROR
        assert "attribute" in result["message"].lower()
