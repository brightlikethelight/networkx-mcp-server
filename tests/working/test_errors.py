"""Tests for error classes and validation functions in errors.py."""

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
    validate_graph_id,
)


# ---------------------------------------------------------------------------
# ErrorCodes — JSON-RPC 2.0 standard values
# ---------------------------------------------------------------------------


class TestErrorCodes:
    def test_parse_error(self):
        assert ErrorCodes.PARSE_ERROR == -32700

    def test_invalid_request(self):
        assert ErrorCodes.INVALID_REQUEST == -32600

    def test_method_not_found(self):
        assert ErrorCodes.METHOD_NOT_FOUND == -32601

    def test_invalid_params(self):
        assert ErrorCodes.INVALID_PARAMS == -32602

    def test_internal_error(self):
        assert ErrorCodes.INTERNAL_ERROR == -32603

    def test_mcp_specific_codes_in_server_range(self):
        """MCP-specific codes must fall in the -32000 to -32099 range."""
        mcp_codes = [
            ErrorCodes.GRAPH_NOT_FOUND,
            ErrorCodes.NODE_NOT_FOUND,
            ErrorCodes.EDGE_NOT_FOUND,
            ErrorCodes.GRAPH_ALREADY_EXISTS,
            ErrorCodes.INVALID_GRAPH_ID,
            ErrorCodes.INVALID_NODE_ID,
            ErrorCodes.INVALID_EDGE,
            ErrorCodes.GRAPH_OPERATION_FAILED,
            ErrorCodes.ALGORITHM_ERROR,
            ErrorCodes.VALIDATION_ERROR,
            ErrorCodes.RESOURCE_LIMIT_EXCEEDED,
            ErrorCodes.SERVER_NOT_INITIALIZED,
        ]
        for code in mcp_codes:
            assert -32099 <= code <= -32000

    def test_mcp_codes_are_unique(self):
        mcp_codes = [
            ErrorCodes.GRAPH_NOT_FOUND,
            ErrorCodes.NODE_NOT_FOUND,
            ErrorCodes.EDGE_NOT_FOUND,
            ErrorCodes.GRAPH_ALREADY_EXISTS,
            ErrorCodes.INVALID_GRAPH_ID,
            ErrorCodes.INVALID_NODE_ID,
            ErrorCodes.INVALID_EDGE,
            ErrorCodes.GRAPH_OPERATION_FAILED,
            ErrorCodes.ALGORITHM_ERROR,
            ErrorCodes.VALIDATION_ERROR,
            ErrorCodes.RESOURCE_LIMIT_EXCEEDED,
            ErrorCodes.SERVER_NOT_INITIALIZED,
        ]
        assert len(mcp_codes) == len(set(mcp_codes))


# ---------------------------------------------------------------------------
# MCPError base class
# ---------------------------------------------------------------------------


class TestMCPError:
    def test_construction_minimal(self):
        err = MCPError(code=-32600, message="bad request")
        assert err.code == -32600
        assert err.message == "bad request"
        assert err.data is None

    def test_construction_with_data(self):
        err = MCPError(code=-32603, message="boom", data={"detail": 42})
        assert err.data == {"detail": 42}

    def test_is_exception(self):
        err = MCPError(code=-32700, message="parse")
        assert isinstance(err, Exception)

    def test_str_is_message(self):
        err = MCPError(code=-32700, message="parse failure")
        assert str(err) == "parse failure"

    def test_to_dict_without_data(self):
        err = MCPError(code=-32600, message="invalid")
        d = err.to_dict()
        assert d == {"code": -32600, "message": "invalid"}
        assert "data" not in d

    def test_to_dict_with_data(self):
        err = MCPError(code=-32603, message="err", data={"k": "v"})
        d = err.to_dict()
        assert d == {"code": -32603, "message": "err", "data": {"k": "v"}}


# ---------------------------------------------------------------------------
# MCPError subclasses — hierarchy and attributes
# ---------------------------------------------------------------------------


class TestGraphNotFoundError:
    def test_inherits_mcp_error(self):
        assert issubclass(GraphNotFoundError, MCPError)

    def test_attributes(self):
        err = GraphNotFoundError("g1")
        assert err.code == ErrorCodes.GRAPH_NOT_FOUND
        assert err.graph_id == "g1"
        assert "g1" in err.message

    def test_to_dict_data(self):
        d = GraphNotFoundError("g1").to_dict()
        assert d["data"] == {"graph_id": "g1"}


class TestNodeNotFoundError:
    def test_inherits_mcp_error(self):
        assert issubclass(NodeNotFoundError, MCPError)

    def test_attributes(self):
        err = NodeNotFoundError("g1", "n1")
        assert err.code == ErrorCodes.NODE_NOT_FOUND
        assert err.graph_id == "g1"
        assert err.node_id == "n1"
        assert "n1" in err.message
        assert "g1" in err.message

    def test_to_dict_data(self):
        d = NodeNotFoundError("g1", "n1").to_dict()
        assert d["data"] == {"graph_id": "g1", "node_id": "n1"}


class TestEdgeNotFoundError:
    def test_inherits_mcp_error(self):
        assert issubclass(EdgeNotFoundError, MCPError)

    def test_attributes(self):
        err = EdgeNotFoundError("g1", "a", "b")
        assert err.code == ErrorCodes.EDGE_NOT_FOUND
        assert err.graph_id == "g1"
        assert err.source == "a"
        assert err.target == "b"
        assert "a" in err.message and "b" in err.message

    def test_to_dict_data(self):
        d = EdgeNotFoundError("g1", "a", "b").to_dict()
        assert d["data"] == {"graph_id": "g1", "source": "a", "target": "b"}


class TestGraphAlreadyExistsError:
    def test_inherits_mcp_error(self):
        assert issubclass(GraphAlreadyExistsError, MCPError)

    def test_attributes(self):
        err = GraphAlreadyExistsError("g1")
        assert err.code == ErrorCodes.GRAPH_ALREADY_EXISTS
        assert err.graph_id == "g1"
        assert "g1" in err.message


class TestInvalidGraphIdError:
    def test_inherits_mcp_error(self):
        assert issubclass(InvalidGraphIdError, MCPError)

    def test_default_reason(self):
        err = InvalidGraphIdError("bad")
        assert err.code == ErrorCodes.INVALID_GRAPH_ID
        assert err.graph_id == "bad"
        assert "Invalid format" in err.message

    def test_custom_reason(self):
        err = InvalidGraphIdError("x", reason="too short")
        assert "too short" in err.message
        assert err.to_dict()["data"] == {"reason": "too short"}

    def test_does_not_leak_raw_input(self):
        """Error message and data should NOT echo the raw graph_id (security)."""
        malicious = "../../../etc/passwd"
        err = InvalidGraphIdError(malicious, reason="nope")
        d = err.to_dict()
        assert malicious not in d["message"]
        assert "graph_id" not in d.get("data", {})


class TestInvalidNodeIdError:
    def test_inherits_mcp_error(self):
        assert issubclass(InvalidNodeIdError, MCPError)

    def test_attributes(self):
        err = InvalidNodeIdError("bad_node", reason="banned")
        assert err.code == ErrorCodes.INVALID_NODE_ID
        assert err.node_id == "bad_node"
        assert "banned" in err.message

    def test_does_not_leak_raw_input(self):
        err = InvalidNodeIdError("<script>alert(1)</script>", reason="xss")
        d = err.to_dict()
        assert "<script>" not in d["message"]
        assert "node_id" not in d.get("data", {})


class TestInvalidEdgeError:
    def test_inherits_mcp_error(self):
        assert issubclass(InvalidEdgeError, MCPError)

    def test_attributes(self):
        err = InvalidEdgeError([1, 2, 3], reason="too many elements")
        assert err.code == ErrorCodes.INVALID_EDGE
        assert err.edge_data == [1, 2, 3]
        assert "too many elements" in err.message


class TestGraphOperationError:
    def test_inherits_mcp_error(self):
        assert issubclass(GraphOperationError, MCPError)

    def test_attributes(self):
        err = GraphOperationError("add_node", "g1", "disk full")
        assert err.code == ErrorCodes.GRAPH_OPERATION_FAILED
        assert err.operation == "add_node"
        assert err.graph_id == "g1"
        assert "disk full" in err.message

    def test_to_dict_data(self):
        d = GraphOperationError("op", "g1", "reason").to_dict()
        assert d["data"] == {"operation": "op", "graph_id": "g1", "reason": "reason"}


class TestAlgorithmError:
    def test_inherits_mcp_error(self):
        assert issubclass(AlgorithmError, MCPError)

    def test_attributes(self):
        err = AlgorithmError("dijkstra", "g1", "negative weight")
        assert err.code == ErrorCodes.ALGORITHM_ERROR
        assert err.algorithm == "dijkstra"
        assert err.graph_id == "g1"
        assert "negative weight" in err.message


class TestValidationError:
    def test_inherits_mcp_error(self):
        assert issubclass(ValidationError, MCPError)

    def test_attributes(self):
        err = ValidationError("weight", -1, "must be positive")
        assert err.code == ErrorCodes.VALIDATION_ERROR
        assert err.parameter == "weight"
        assert err.value == -1
        assert "must be positive" in err.message

    def test_to_dict_data(self):
        d = ValidationError("p", "v", "r").to_dict()
        assert d["data"] == {"parameter": "p", "value": "v", "reason": "r"}


class TestResourceLimitExceededError:
    def test_inherits_mcp_error(self):
        assert issubclass(ResourceLimitExceededError, MCPError)

    def test_attributes(self):
        err = ResourceLimitExceededError("nodes", 1000, 1500)
        assert err.code == ErrorCodes.RESOURCE_LIMIT_EXCEEDED
        assert err.resource == "nodes"
        assert err.limit == 1000
        assert err.current == 1500
        assert "1500" in err.message and "1000" in err.message

    def test_to_dict_data(self):
        d = ResourceLimitExceededError("nodes", 10, 20).to_dict()
        assert d["data"] == {"resource": "nodes", "limit": 10, "current": 20}


class TestServerNotInitializedError:
    def test_inherits_mcp_error(self):
        assert issubclass(ServerNotInitializedError, MCPError)

    def test_attributes(self):
        err = ServerNotInitializedError("create_graph")
        assert err.code == ErrorCodes.SERVER_NOT_INITIALIZED
        assert err.operation == "create_graph"
        assert "create_graph" in err.message


# ---------------------------------------------------------------------------
# validate_graph_id
# ---------------------------------------------------------------------------


class TestValidateGraphId:
    def test_simple_valid(self):
        assert validate_graph_id("my_graph") == "my_graph"

    def test_alphanumeric(self):
        assert validate_graph_id("Graph123") == "Graph123"

    def test_hyphens_and_underscores(self):
        assert validate_graph_id("my-graph_1") == "my-graph_1"

    def test_strips_whitespace(self):
        assert validate_graph_id("  foo  ") == "foo"

    def test_none_rejected(self):
        with pytest.raises(InvalidGraphIdError, match="cannot be None"):
            validate_graph_id(None)

    def test_non_string_rejected(self):
        with pytest.raises(InvalidGraphIdError, match="must be a string"):
            validate_graph_id(42)

    def test_empty_string_rejected(self):
        with pytest.raises(InvalidGraphIdError, match="cannot be empty"):
            validate_graph_id("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(InvalidGraphIdError, match="cannot be empty"):
            validate_graph_id("   ")

    def test_too_long_rejected(self):
        with pytest.raises(InvalidGraphIdError, match="too long"):
            validate_graph_id("a" * 101)

    def test_exactly_100_chars_accepted(self):
        result = validate_graph_id("a" * 100)
        assert len(result) == 100

    def test_special_chars_rejected(self):
        with pytest.raises(InvalidGraphIdError, match="only contain"):
            validate_graph_id("graph!@#")

    def test_spaces_in_middle_rejected(self):
        with pytest.raises(InvalidGraphIdError, match="only contain"):
            validate_graph_id("my graph")

    def test_dot_rejected(self):
        with pytest.raises(InvalidGraphIdError, match="only contain"):
            validate_graph_id("my.graph")

    def test_path_traversal_dot_dot(self):
        # ".." contains ".", which gets caught by the alphanumeric check first
        with pytest.raises(InvalidGraphIdError):
            validate_graph_id("../etc")

    def test_path_traversal_slash(self):
        with pytest.raises(InvalidGraphIdError):
            validate_graph_id("foo/bar")

    def test_path_traversal_backslash(self):
        with pytest.raises(InvalidGraphIdError):
            validate_graph_id("foo\\bar")

    def test_returns_str(self):
        result = validate_graph_id("abc")
        assert isinstance(result, str)
