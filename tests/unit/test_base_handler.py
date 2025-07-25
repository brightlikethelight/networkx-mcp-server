"""Tests for core/io/base_handler.py - Target: 100% coverage."""


class TestBaseHandler:
    """Test the BaseHandlerHandler class."""

    def test_import_module(self):
        """Test that the module can be imported."""
        import networkx_mcp.core.io.base_handler as base_handler

        # Module should exist
        assert base_handler is not None

        # Should have logger
        assert hasattr(base_handler, "logger")
        assert base_handler.logger is not None

    def test_base_handler_handler_class(self):
        """Test BaseHandlerHandler class."""
        from networkx_mcp.core.io.base_handler import BaseHandlerHandler

        # Class should exist
        assert BaseHandlerHandler is not None

        # Should be a class
        assert isinstance(BaseHandlerHandler, type)

        # Create instance
        handler = BaseHandlerHandler()
        assert handler is not None

        # Check docstring
        assert BaseHandlerHandler.__doc__ is not None
        assert "Focused I/O handler" in BaseHandlerHandler.__doc__

    def test_logger_name(self):
        """Test that logger has correct name."""
        import networkx_mcp.core.io.base_handler as base_handler

        # Logger should have module name
        assert base_handler.logger.name == "networkx_mcp.core.io.base_handler"
