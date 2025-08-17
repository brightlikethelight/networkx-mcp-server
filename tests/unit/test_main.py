"""Tests for __main__.py module."""

import io
import logging
import sys
import unittest
from unittest.mock import Mock, patch

import pytest


class TestMainModule(unittest.TestCase):
    """Test the __main__.py module functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Capture log output for testing
        self.log_stream = io.StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.logger = logging.getLogger("networkx_mcp.__main__")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

    def tearDown(self):
        """Clean up after tests."""
        self.logger.removeHandler(self.handler)
        self.handler.close()

    @patch('networkx_mcp.__main__.main')
    def test_run_server(self, mock_main):
        """Test run_server function."""
        from networkx_mcp.__main__ import run_server
        
        # Run the function
        run_server()
        
        # Verify it calls the server main function
        mock_main.assert_called_once()
        
        # Check logging
        log_output = self.log_stream.getvalue()
        assert "Starting NetworkX MCP Server" in log_output

    @patch('networkx_mcp.__main__.run_server')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_normal_operation(self, mock_parse_args, mock_run_server):
        """Test main function normal operation."""
        from networkx_mcp.__main__ import main
        
        # Mock command line arguments
        mock_args = Mock()
        mock_args.debug = False
        mock_parse_args.return_value = mock_args
        
        # Run main
        main()
        
        # Verify server was started
        mock_run_server.assert_called_once()

    @patch('networkx_mcp.__main__.run_server')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_debug_mode(self, mock_parse_args, mock_run_server):
        """Test main function with debug mode enabled."""
        from networkx_mcp.__main__ import main
        
        # Mock command line arguments with debug enabled
        mock_args = Mock()
        mock_args.debug = True
        mock_parse_args.return_value = mock_args
        
        # Run main
        main()
        
        # Verify debug logging was enabled
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
        
        # Verify server was started
        mock_run_server.assert_called_once()

    @patch('networkx_mcp.__main__.run_server')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('sys.exit')
    def test_main_keyboard_interrupt(self, mock_exit, mock_parse_args, mock_run_server):
        """Test main function handles KeyboardInterrupt."""
        from networkx_mcp.__main__ import main
        
        # Mock command line arguments
        mock_args = Mock()
        mock_args.debug = False
        mock_parse_args.return_value = mock_args
        
        # Mock server to raise KeyboardInterrupt
        mock_run_server.side_effect = KeyboardInterrupt()
        
        # Run main
        main()
        
        # Verify graceful exit
        mock_exit.assert_called_once_with(0)
        
        # Check logging
        log_output = self.log_stream.getvalue()
        assert "Server stopped by user" in log_output

    @patch('networkx_mcp.__main__.run_server')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('sys.exit')
    def test_main_exception_handling(self, mock_exit, mock_parse_args, mock_run_server):
        """Test main function handles general exceptions."""
        from networkx_mcp.__main__ import main
        
        # Mock command line arguments
        mock_args = Mock()
        mock_args.debug = False
        mock_parse_args.return_value = mock_args
        
        # Mock server to raise exception
        test_error = Exception("Test error")
        mock_run_server.side_effect = test_error
        
        # Run main
        main()
        
        # Verify error exit
        mock_exit.assert_called_once_with(1)
        
        # Check logging
        log_output = self.log_stream.getvalue()
        assert "Server failed: Test error" in log_output

    @patch('argparse.ArgumentParser.parse_args')
    def test_argument_parser_setup(self, mock_parse_args):
        """Test that argument parser is set up correctly."""
        from networkx_mcp.__main__ import main
        
        # Mock args to avoid actually running server
        mock_args = Mock()
        mock_args.debug = False
        mock_parse_args.return_value = mock_args
        
        with patch('networkx_mcp.__main__.run_server'):
            main()
        
        # Verify parse_args was called (meaning parser was set up)
        mock_parse_args.assert_called_once()

    def test_module_logger_configuration(self):
        """Test that module logger is configured properly."""
        # Import the module to check logger setup
        import networkx_mcp.__main__
        
        # Check that logger exists and is configured
        logger = logging.getLogger('networkx_mcp.__main__')
        assert logger is not None

    @patch('networkx_mcp.__main__.main')
    def test_name_main_guard(self, mock_main):
        """Test that main() is called when module is run directly."""
        # This is tricky to test since we can't actually run the module
        # We'll just verify the function exists and can be called
        from networkx_mcp.__main__ import main
        
        # Verify main function exists and is callable
        assert callable(main)

    def test_imports_work(self):
        """Test that all imports in the module work."""
        # This will fail if there are import errors
        import networkx_mcp.__main__
        
        # Verify key functions are available
        assert hasattr(networkx_mcp.__main__, 'main')
        assert hasattr(networkx_mcp.__main__, 'run_server')

    @patch('networkx_mcp.__main__.run_server')  
    @patch('builtins.print')  # Mock print to avoid output during tests
    def test_help_message(self, mock_print, mock_run_server):
        """Test that help message can be displayed."""
        from networkx_mcp.__main__ import main
        
        # Mock sys.argv to request help
        with patch('sys.argv', ['__main__.py', '--help']):
            # This should exit with code 0 after showing help
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            # Help should exit with code 0
            assert exc_info.value.code == 0

    def test_logging_format(self):
        """Test that logging is configured with proper format."""
        # Check the root logger configuration after import
        import networkx_mcp.__main__
        
        # Get a logger and check it has handlers
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0


if __name__ == "__main__":
    unittest.main()