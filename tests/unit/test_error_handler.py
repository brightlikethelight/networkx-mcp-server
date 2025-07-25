"""Tests for utils/error_handler.py - Target: 100% coverage (12 lines)."""

from unittest.mock import patch

import pytest


class TestErrorHandlerExceptions:
    """Test the exception classes in error_handler."""

    def test_mcp_error(self):
        """Test MCPError base exception."""
        from networkx_mcp.utils.error_handler import MCPError

        # Test creation and inheritance
        error = MCPError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

        # Test raising
        with pytest.raises(MCPError) as exc_info:
            raise MCPError("Custom MCP error")
        assert str(exc_info.value) == "Custom MCP error"

    def test_validation_error(self):
        """Test ValidationError exception."""
        from networkx_mcp.utils.error_handler import MCPError, ValidationError

        # Test inheritance
        error = ValidationError("Validation failed")
        assert isinstance(error, MCPError)
        assert isinstance(error, Exception)
        assert str(error) == "Validation failed"

        # Test raising
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Invalid input")
        assert str(exc_info.value) == "Invalid input"

    def test_graph_operation_error(self):
        """Test GraphOperationError exception."""
        from networkx_mcp.utils.error_handler import GraphOperationError, MCPError

        # Test inheritance
        error = GraphOperationError("Operation failed")
        assert isinstance(error, MCPError)
        assert isinstance(error, Exception)
        assert str(error) == "Operation failed"

        # Test raising
        with pytest.raises(GraphOperationError) as exc_info:
            raise GraphOperationError("Graph not found")
        assert str(exc_info.value) == "Graph not found"

    def test_resource_error(self):
        """Test ResourceError exception."""
        from networkx_mcp.utils.error_handler import MCPError, ResourceError

        # Test inheritance
        error = ResourceError("Resource limit exceeded")
        assert isinstance(error, MCPError)
        assert isinstance(error, Exception)
        assert str(error) == "Resource limit exceeded"

        # Test raising
        with pytest.raises(ResourceError) as exc_info:
            raise ResourceError("Memory limit reached")
        assert str(exc_info.value) == "Memory limit reached"


class TestErrorHandler:
    """Test the handle_error function."""

    @patch("networkx_mcp.utils.error_handler.logger")
    def test_handle_error_with_context(self, mock_logger):
        """Test handle_error with context provided."""
        from networkx_mcp.utils.error_handler import handle_error

        error = ValueError("Test error")
        handle_error(error, context="Processing graph")

        # Verify logger.error was called with formatted message
        mock_logger.error.assert_called_once_with("Processing graph: Test error")

    @patch("networkx_mcp.utils.error_handler.logger")
    def test_handle_error_without_context(self, mock_logger):
        """Test handle_error without context."""
        from networkx_mcp.utils.error_handler import handle_error

        error = RuntimeError("Another error")
        handle_error(error)

        # Verify logger.error was called with just the error string
        mock_logger.error.assert_called_once_with("Another error")

    @patch("networkx_mcp.utils.error_handler.logger")
    def test_handle_error_with_none_context(self, mock_logger):
        """Test handle_error with None context."""
        from networkx_mcp.utils.error_handler import handle_error

        error = Exception("Generic error")
        handle_error(error, context=None)

        # Should behave same as no context
        mock_logger.error.assert_called_once_with("Generic error")

    @patch("networkx_mcp.utils.error_handler.logger")
    def test_handle_error_with_custom_exceptions(self, mock_logger):
        """Test handle_error with custom exception types."""
        from networkx_mcp.utils.error_handler import (
            GraphOperationError,
            ValidationError,
            handle_error,
        )

        # Test with ValidationError
        val_error = ValidationError("Invalid graph ID")
        handle_error(val_error, context="Validation")
        mock_logger.error.assert_called_with("Validation: Invalid graph ID")

        # Test with GraphOperationError
        op_error = GraphOperationError("Failed to add node")
        handle_error(op_error, context="Graph operation")
        mock_logger.error.assert_called_with("Graph operation: Failed to add node")
