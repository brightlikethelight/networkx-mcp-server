"""Tests for __main__.py entry point."""

from unittest.mock import patch

from networkx_mcp.__main__ import main, run_server


class TestMainEntry:
    def test_run_server_callable(self):
        """Test that run_server exists and is callable."""
        assert callable(run_server)

    def test_main_with_help(self):
        """Test main CLI parses --help without error."""
        with patch("sys.argv", ["networkx-mcp", "--help"]):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0

    def test_main_keyboard_interrupt(self):
        """Test main handles KeyboardInterrupt gracefully."""
        with (
            patch("sys.argv", ["networkx-mcp"]),
            patch(
                "networkx_mcp.__main__.run_server",
                side_effect=KeyboardInterrupt,
            ),
        ):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0

    def test_main_generic_exception(self):
        """Test main handles unexpected exceptions."""
        with (
            patch("sys.argv", ["networkx-mcp"]),
            patch(
                "networkx_mcp.__main__.run_server",
                side_effect=RuntimeError("test error"),
            ),
        ):
            try:
                main()
            except SystemExit as e:
                assert e.code == 1

    def test_main_debug_flag(self):
        """Test --debug flag sets logging level."""
        import logging

        with (
            patch("sys.argv", ["networkx-mcp", "--debug"]),
            patch(
                "networkx_mcp.__main__.run_server",
                side_effect=KeyboardInterrupt,
            ),
        ):
            try:
                main()
            except SystemExit:
                pass
            assert logging.getLogger().level == logging.DEBUG
