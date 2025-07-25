"""Tests for core/io/csv_handler.py - Target: 100% coverage."""


class TestCsvHandler:
    """Test the CsvHandlerHandler class."""

    def test_import_module(self):
        """Test that the module can be imported."""
        import networkx_mcp.core.io.csv_handler as csv_handler

        # Module should exist
        assert csv_handler is not None

        # Should have logger
        assert hasattr(csv_handler, "logger")
        assert csv_handler.logger is not None

    def test_csv_handler_handler_class(self):
        """Test CsvHandlerHandler class."""
        from networkx_mcp.core.io.csv_handler import CsvHandlerHandler

        # Class should exist
        assert CsvHandlerHandler is not None

        # Should be a class
        assert isinstance(CsvHandlerHandler, type)

        # Create instance
        handler = CsvHandlerHandler()
        assert handler is not None

        # Check docstring
        assert CsvHandlerHandler.__doc__ is not None
        assert "Focused I/O handler" in CsvHandlerHandler.__doc__

    def test_logger_name(self):
        """Test that logger has correct name."""
        import networkx_mcp.core.io.csv_handler as csv_handler

        # Logger should have module name
        assert csv_handler.logger.name == "networkx_mcp.core.io.csv_handler"
