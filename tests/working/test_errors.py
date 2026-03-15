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
    handle_error,
    validate_centrality_measures,
    validate_edge,
    validate_graph_id,
    validate_node_id,
    validate_required_params,
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


# ---------------------------------------------------------------------------
# validate_node_id
# ---------------------------------------------------------------------------


class TestValidateNodeId:
    def test_valid_string(self):
        assert validate_node_id("node_1") == "node_1"

    def test_valid_integer(self):
        assert validate_node_id(42) == "42"

    def test_zero_is_valid(self):
        assert validate_node_id(0) == "0"

    def test_negative_int(self):
        assert validate_node_id(-1) == "-1"

    def test_strips_whitespace(self):
        assert validate_node_id("  n  ") == "n"

    def test_none_rejected(self):
        with pytest.raises(InvalidNodeIdError, match="cannot be None"):
            validate_node_id(None)

    def test_empty_string_rejected(self):
        with pytest.raises(InvalidNodeIdError, match="cannot be empty"):
            validate_node_id("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(InvalidNodeIdError, match="cannot be empty"):
            validate_node_id("   ")

    def test_too_long_rejected(self):
        with pytest.raises(InvalidNodeIdError, match="too long"):
            validate_node_id("n" * 101)

    def test_exactly_100_chars_accepted(self):
        result = validate_node_id("n" * 100)
        assert len(result) == 100

    def test_non_string_non_int_rejected(self):
        with pytest.raises(InvalidNodeIdError, match="must be a string or integer"):
            validate_node_id([1, 2])

    def test_float_rejected(self):
        with pytest.raises(InvalidNodeIdError, match="must be a string or integer"):
            validate_node_id(3.14)

    def test_dict_rejected(self):
        with pytest.raises(InvalidNodeIdError, match="must be a string or integer"):
            validate_node_id({"id": 1})

    def test_returns_str(self):
        result = validate_node_id("abc")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# validate_edge
# ---------------------------------------------------------------------------


class TestValidateEdge:
    def test_valid_list(self):
        assert validate_edge(["a", "b"]) == ("a", "b")

    def test_valid_tuple(self):
        assert validate_edge(("x", "y")) == ("x", "y")

    def test_integer_nodes(self):
        assert validate_edge([1, 2]) == ("1", "2")

    def test_mixed_types(self):
        assert validate_edge(["a", 1]) == ("a", "1")

    def test_string_rejected(self):
        with pytest.raises(InvalidEdgeError, match="must be a list or tuple"):
            validate_edge("a->b")

    def test_dict_rejected(self):
        with pytest.raises(InvalidEdgeError, match="must be a list or tuple"):
            validate_edge({"source": "a", "target": "b"})

    def test_none_rejected(self):
        with pytest.raises(InvalidEdgeError, match="must be a list or tuple"):
            validate_edge(None)

    def test_single_element_rejected(self):
        with pytest.raises(InvalidEdgeError, match="exactly 2 elements"):
            validate_edge(["a"])

    def test_three_elements_rejected(self):
        with pytest.raises(InvalidEdgeError, match="exactly 2 elements"):
            validate_edge(["a", "b", "c"])

    def test_empty_list_rejected(self):
        with pytest.raises(InvalidEdgeError, match="exactly 2 elements"):
            validate_edge([])

    def test_none_source_rejected(self):
        with pytest.raises(InvalidEdgeError, match="Invalid node in edge"):
            validate_edge([None, "b"])

    def test_none_target_rejected(self):
        with pytest.raises(InvalidEdgeError, match="Invalid node in edge"):
            validate_edge(["a", None])


# ---------------------------------------------------------------------------
# validate_required_params
# ---------------------------------------------------------------------------


class TestValidateRequiredParams:
    def test_all_present(self):
        # Should not raise
        validate_required_params({"a": 1, "b": 2}, ["a", "b"])

    def test_extra_params_ok(self):
        validate_required_params({"a": 1, "b": 2, "c": 3}, ["a"])

    def test_empty_required_list(self):
        validate_required_params({"a": 1}, [])

    def test_missing_param_raises(self):
        with pytest.raises(ValidationError, match="Required parameter missing"):
            validate_required_params({"a": 1}, ["a", "b"])

    def test_none_value_raises(self):
        with pytest.raises(ValidationError, match="cannot be None"):
            validate_required_params({"a": None}, ["a"])

    def test_error_names_parameter(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_required_params({}, ["graph_id"])
        assert exc_info.value.parameter == "graph_id"

    def test_false_value_accepted(self):
        """Falsy-but-not-None values should pass."""
        validate_required_params(
            {"flag": False, "count": 0, "name": ""}, ["flag", "count", "name"]
        )


# ---------------------------------------------------------------------------
# validate_centrality_measures
# ---------------------------------------------------------------------------


class TestValidateCentralityMeasures:
    def test_single_valid(self):
        assert validate_centrality_measures(["degree"]) == ["degree"]

    def test_all_valid(self):
        result = validate_centrality_measures(
            ["degree", "betweenness", "closeness", "eigenvector"]
        )
        assert len(result) == 4

    def test_returns_same_list(self):
        inp = ["degree", "closeness"]
        assert validate_centrality_measures(inp) is inp

    def test_not_a_list_rejected(self):
        with pytest.raises(ValidationError, match="must be a list"):
            validate_centrality_measures("degree")

    def test_tuple_rejected(self):
        with pytest.raises(ValidationError, match="must be a list"):
            validate_centrality_measures(("degree",))

    def test_empty_list_rejected(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_centrality_measures([])

    def test_invalid_measure_rejected(self):
        with pytest.raises(ValidationError, match="Invalid measure 'pagerank'"):
            validate_centrality_measures(["pagerank"])

    def test_non_string_element_rejected(self):
        with pytest.raises(ValidationError, match="must be string"):
            validate_centrality_measures([42])

    def test_mixed_valid_invalid_rejected(self):
        with pytest.raises(ValidationError, match="Invalid measure"):
            validate_centrality_measures(["degree", "fake"])

    def test_none_rejected(self):
        with pytest.raises(ValidationError, match="must be a list"):
            validate_centrality_measures(None)


# ---------------------------------------------------------------------------
# handle_error
# ---------------------------------------------------------------------------


class TestHandleError:
    # -- MCPError passthrough --
    def test_mcp_error_returns_to_dict(self):
        err = GraphNotFoundError("g1")
        result = handle_error(err, operation="get_graph")
        assert result == err.to_dict()
        assert result["code"] == ErrorCodes.GRAPH_NOT_FOUND
        assert result["data"] == {"graph_id": "g1"}

    def test_mcp_error_preserves_message(self):
        err = MCPError(code=-32600, message="custom msg", data={"x": 1})
        result = handle_error(err)
        assert result["message"] == "custom msg"
        assert result["data"] == {"x": 1}

    # -- NetworkX errors --
    def test_networkx_algorithm_error_no_path(self):
        """NetworkX errors containing 'no path' should map to ALGORITHM_ERROR."""
        nx_err = _make_networkx_error("No path between nodes 1 and 5")
        result = handle_error(nx_err, operation="shortest_path")
        assert result["code"] == ErrorCodes.ALGORITHM_ERROR
        assert "No path" in result["message"]
        assert result["data"]["operation"] == "shortest_path"

    def test_networkx_algorithm_error_not_connected(self):
        nx_err = _make_networkx_error("Graph is not connected")
        result = handle_error(nx_err, operation="components")
        assert result["code"] == ErrorCodes.ALGORITHM_ERROR

    def test_networkx_algorithm_error_unreachable(self):
        nx_err = _make_networkx_error("Node 5 unreachable from source")
        result = handle_error(nx_err, operation="bfs")
        assert result["code"] == ErrorCodes.ALGORITHM_ERROR

    def test_networkx_generic_error(self):
        """NetworkX errors without algorithm keywords map to GRAPH_OPERATION_FAILED."""
        nx_err = _make_networkx_error("Self-loops not allowed")
        result = handle_error(nx_err, operation="add_edge")
        assert result["code"] == ErrorCodes.GRAPH_OPERATION_FAILED
        assert "Self-loops" in result["message"]

    # -- Generic exceptions --
    def test_generic_exception(self):
        err = RuntimeError("disk failure")
        result = handle_error(err, operation="save")
        assert result["code"] == ErrorCodes.INTERNAL_ERROR
        assert result["message"] == "Internal server error"
        assert result["data"]["operation"] == "save"
        assert result["data"]["error_type"] == "RuntimeError"

    def test_generic_exception_default_operation(self):
        result = handle_error(ValueError("bad"))
        assert result["data"]["operation"] == "unknown"
        assert result["data"]["error_type"] == "ValueError"

    def test_key_error(self):
        result = handle_error(KeyError("missing_key"), operation="lookup")
        assert result["code"] == ErrorCodes.INTERNAL_ERROR
        assert result["data"]["error_type"] == "KeyError"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_networkx_error(message: str) -> Exception:
    """Create an exception that looks like it came from networkx.

    handle_error checks `error.__module__` for the string 'networkx',
    so we fabricate that attribute.
    """
    err = Exception(message)
    err.__module__ = "networkx.exception"
    return err
