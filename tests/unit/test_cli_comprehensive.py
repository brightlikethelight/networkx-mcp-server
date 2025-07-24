"""Comprehensive tests for NetworkX CLI - Target: 100% coverage of cli.py (318 lines).

This test suite aims to provide complete coverage of the NetworkX CLI module,
which currently has 0% coverage.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from networkx_mcp.cli import NetworkXCLI, main


class TestNetworkXCLI:
    """Comprehensive test suite for NetworkXCLI class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = NetworkXCLI()

    def test_init(self):
        """Test CLI initialization."""
        cli = NetworkXCLI()
        assert cli.graph_manager is not None
        assert cli.performance_monitor is not None
        assert cli.operation_counter is not None
        assert cli.current_graph is None

    @patch("networkx_mcp.cli.console")
    def test_print_banner(self, mock_console):
        """Test banner printing."""
        self.cli.print_banner()
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "NetworkX MCP Server" in call_args
        assert "Interactive CLI" in call_args

    @patch("networkx_mcp.cli.console")
    def test_print_help(self, mock_console):
        """Test help printing."""
        self.cli.print_help()
        mock_console.print.assert_called()
        # Check that help text contains expected commands
        call_args = str(mock_console.print.call_args_list)
        assert "create" in call_args
        assert "list" in call_args
        assert "analyze" in call_args

    @patch("networkx_mcp.cli.console")
    def test_create_graph_basic(self, mock_console):
        """Test basic graph creation."""
        with patch.object(self.cli.graph_manager, "create_graph", return_value=True):
            self.cli.create_graph("test_graph")
            mock_console.print.assert_called()
            assert any(
                "Created" in str(call) for call in mock_console.print.call_args_list
            )

    @patch("networkx_mcp.cli.console")
    def test_create_graph_with_type(self, mock_console):
        """Test graph creation with type."""
        with patch.object(self.cli.graph_manager, "create_graph", return_value=True):
            self.cli.create_graph("test_graph", "DiGraph")
            mock_console.print.assert_called()

    @patch("networkx_mcp.cli.console")
    def test_create_graph_failure(self, mock_console):
        """Test graph creation failure."""
        with patch.object(
            self.cli.graph_manager, "create_graph", side_effect=Exception("Error")
        ):
            self.cli.create_graph("test_graph")
            assert any(
                "Failed" in str(call) for call in mock_console.print.call_args_list
            )

    @patch("networkx_mcp.cli.console")
    def test_list_graphs_empty(self, mock_console):
        """Test listing graphs when empty."""
        with patch.object(self.cli.graph_manager, "list_graphs", return_value=[]):
            self.cli.list_graphs()
            assert any(
                "No graphs" in str(call) for call in mock_console.print.call_args_list
            )

    @patch("networkx_mcp.cli.console")
    def test_list_graphs_with_data(self, mock_console):
        """Test listing graphs with data."""
        mock_graphs = [
            {"graph_id": "g1", "graph_type": "Graph", "node_count": 5, "edge_count": 3},
            {
                "graph_id": "g2",
                "graph_type": "DiGraph",
                "node_count": 10,
                "edge_count": 15,
            },
        ]
        with patch.object(
            self.cli.graph_manager, "list_graphs", return_value=mock_graphs
        ):
            self.cli.list_graphs()
            # Verify table is created
            assert any(
                "Table" in str(type(call[0][0]))
                for call in mock_console.print.call_args_list
            )

    @patch("networkx_mcp.cli.console")
    def test_show_graph_info_no_selection(self, mock_console):
        """Test showing graph info with no selection."""
        self.cli.current_graph = None
        self.cli.show_graph_info()
        assert any(
            "No graph selected" in str(call)
            for call in mock_console.print.call_args_list
        )

    @patch("networkx_mcp.cli.console")
    def test_show_graph_info_by_id(self, mock_console):
        """Test showing graph info by ID."""
        mock_info = {
            "graph_id": "test",
            "graph_type": "Graph",
            "node_count": 5,
            "edge_count": 3,
            "attributes": {"weighted": True},
        }
        with patch.object(
            self.cli.graph_manager, "get_graph_info", return_value=mock_info
        ):
            self.cli.show_graph_info("test")
            assert any(
                "Panel" in str(type(call[0][0]))
                for call in mock_console.print.call_args_list
            )

    @patch("networkx_mcp.cli.console")
    def test_show_graph_info_current(self, mock_console):
        """Test showing current graph info."""
        self.cli.current_graph = "current_test"
        mock_info = {"graph_id": "current_test", "node_count": 10}
        with patch.object(
            self.cli.graph_manager, "get_graph_info", return_value=mock_info
        ):
            self.cli.show_graph_info()
            mock_console.print.assert_called()

    @patch("networkx_mcp.cli.console")
    def test_select_graph_success(self, mock_console):
        """Test successful graph selection."""
        with patch.object(self.cli.graph_manager, "get_graph", return_value=Mock()):
            self.cli.select_graph("test_graph")
            assert self.cli.current_graph == "test_graph"
            assert any(
                "Selected" in str(call) for call in mock_console.print.call_args_list
            )

    @patch("networkx_mcp.cli.console")
    def test_select_graph_not_found(self, mock_console):
        """Test graph selection when not found."""
        with patch.object(self.cli.graph_manager, "get_graph", return_value=None):
            self.cli.select_graph("nonexistent")
            assert self.cli.current_graph is None
            assert any(
                "not found" in str(call) for call in mock_console.print.call_args_list
            )

    @patch("networkx_mcp.cli.console")
    def test_add_nodes_no_selection(self, mock_console):
        """Test adding nodes with no graph selected."""
        self.cli.current_graph = None
        self.cli.add_nodes("1,2,3")
        assert any(
            "No graph selected" in str(call)
            for call in mock_console.print.call_args_list
        )

    @patch("networkx_mcp.cli.console")
    def test_add_nodes_success(self, mock_console):
        """Test successful node addition."""
        self.cli.current_graph = "test"
        mock_graph = Mock()
        with patch.object(self.cli.graph_manager, "get_graph", return_value=mock_graph):
            self.cli.add_nodes("1, 2, 3")
            assert mock_graph.add_nodes_from.called
            assert any(
                "Added 3 nodes" in str(call)
                for call in mock_console.print.call_args_list
            )

    @patch("networkx_mcp.cli.console")
    def test_add_edges_no_selection(self, mock_console):
        """Test adding edges with no graph selected."""
        self.cli.current_graph = None
        self.cli.add_edges("(1,2),(2,3)")
        assert any(
            "No graph selected" in str(call)
            for call in mock_console.print.call_args_list
        )

    @patch("networkx_mcp.cli.console")
    def test_add_edges_success(self, mock_console):
        """Test successful edge addition."""
        self.cli.current_graph = "test"
        mock_graph = Mock()
        with patch.object(self.cli.graph_manager, "get_graph", return_value=mock_graph):
            self.cli.add_edges("(1,2), (2,3), (3,4)")
            assert mock_graph.add_edges_from.called
            assert any(
                "Added 3 edges" in str(call)
                for call in mock_console.print.call_args_list
            )

    @patch("networkx_mcp.cli.console")
    def test_analyze_graph_no_selection(self, mock_console):
        """Test analyzing graph with no selection."""
        self.cli.current_graph = None
        self.cli.analyze_graph("degree")
        assert any(
            "No graph selected" in str(call)
            for call in mock_console.print.call_args_list
        )

    @patch("networkx_mcp.cli.console")
    def test_analyze_graph_degree_centrality(self, mock_console):
        """Test degree centrality analysis."""
        self.cli.current_graph = "test"
        mock_graph = Mock()
        mock_algorithms = Mock()
        mock_algorithms.degree_centrality.return_value = {0: 0.5, 1: 0.7, 2: 0.3}

        with patch.object(self.cli.graph_manager, "get_graph", return_value=mock_graph):
            with patch(
                "networkx_mcp.cli.GraphAlgorithms", return_value=mock_algorithms
            ):
                self.cli.analyze_graph("degree")
                mock_algorithms.degree_centrality.assert_called_once()
                assert any(
                    "Degree Centrality" in str(call)
                    for call in mock_console.print.call_args_list
                )

    @patch("networkx_mcp.cli.console")
    def test_analyze_graph_shortest_path(self, mock_console):
        """Test shortest path analysis."""
        self.cli.current_graph = "test"
        mock_graph = Mock()
        mock_algorithms = Mock()
        mock_algorithms.shortest_path.return_value = {"path": [1, 2, 3], "length": 2}

        with patch.object(self.cli.graph_manager, "get_graph", return_value=mock_graph):
            with patch(
                "networkx_mcp.cli.GraphAlgorithms", return_value=mock_algorithms
            ):
                self.cli.analyze_graph("shortest_path", source=1, target=3)
                mock_algorithms.shortest_path.assert_called_once()

    @patch("networkx_mcp.cli.console")
    def test_analyze_graph_clustering(self, mock_console):
        """Test clustering coefficient analysis."""
        self.cli.current_graph = "test"
        mock_graph = Mock()
        mock_algorithms = Mock()
        mock_algorithms.clustering_coefficient.return_value = {
            "average": 0.5,
            "nodes": {0: 0.3},
        }

        with patch.object(self.cli.graph_manager, "get_graph", return_value=mock_graph):
            with patch(
                "networkx_mcp.cli.GraphAlgorithms", return_value=mock_algorithms
            ):
                self.cli.analyze_graph("clustering")
                mock_algorithms.clustering_coefficient.assert_called_once()

    @patch("networkx_mcp.cli.console")
    def test_analyze_graph_unknown_algorithm(self, mock_console):
        """Test unknown algorithm handling."""
        self.cli.current_graph = "test"
        mock_graph = Mock()

        with patch.object(self.cli.graph_manager, "get_graph", return_value=mock_graph):
            self.cli.analyze_graph("unknown_algo")
            assert any(
                "Unknown algorithm" in str(call)
                for call in mock_console.print.call_args_list
            )

    @patch("networkx_mcp.cli.console")
    def test_import_graph_file_not_found(self, mock_console):
        """Test importing non-existent file."""
        self.cli.import_graph("nonexistent.json", "test_graph")
        assert any(
            "File not found" in str(call) for call in mock_console.print.call_args_list
        )

    @patch("networkx_mcp.cli.console")
    @patch("pathlib.Path.exists")
    def test_import_graph_success(self, mock_exists, mock_console):
        """Test successful graph import."""
        mock_exists.return_value = True
        mock_io_handler = Mock()
        mock_io_handler.import_graph.return_value = Mock()

        with patch("networkx_mcp.cli.GraphIOHandler", return_value=mock_io_handler):
            with patch.object(self.cli.graph_manager, "add_graph"):
                self.cli.import_graph("test.json", "imported_graph")
                mock_io_handler.import_graph.assert_called_once()
                assert any(
                    "Successfully imported" in str(call)
                    for call in mock_console.print.call_args_list
                )

    @patch("networkx_mcp.cli.console")
    @patch("pathlib.Path.exists")
    def test_import_graph_failure(self, mock_exists, mock_console):
        """Test graph import failure."""
        mock_exists.return_value = True
        mock_io_handler = Mock()
        mock_io_handler.import_graph.side_effect = Exception("Import error")

        with patch("networkx_mcp.cli.GraphIOHandler", return_value=mock_io_handler):
            self.cli.import_graph("test.json", "imported_graph")
            assert any(
                "Failed to import" in str(call)
                for call in mock_console.print.call_args_list
            )

    @patch("networkx_mcp.cli.console")
    def test_export_graph_no_selection(self, mock_console):
        """Test exporting with no graph selected."""
        self.cli.current_graph = None
        self.cli.export_graph("output.json")
        assert any(
            "No graph selected" in str(call)
            for call in mock_console.print.call_args_list
        )

    @patch("networkx_mcp.cli.console")
    def test_export_graph_success(self, mock_console):
        """Test successful graph export."""
        self.cli.current_graph = "test"
        mock_graph = Mock()
        mock_io_handler = Mock()

        with patch.object(self.cli.graph_manager, "get_graph", return_value=mock_graph):
            with patch("networkx_mcp.cli.GraphIOHandler", return_value=mock_io_handler):
                self.cli.export_graph("output.json", format="json")
                mock_io_handler.export_graph.assert_called_once()
                assert any(
                    "Successfully exported" in str(call)
                    for call in mock_console.print.call_args_list
                )

    @patch("networkx_mcp.cli.console")
    def test_show_stats(self, mock_console):
        """Test showing statistics."""
        self.cli.operation_counter.operations = {"create": 5, "analyze": 10}
        self.cli.performance_monitor.metrics = {"avg_time": 0.5}

        self.cli.show_stats()
        # Verify both panels are created
        panel_count = sum(
            1
            for call in mock_console.print.call_args_list
            if "Panel" in str(type(call[0][0]))
        )
        assert panel_count >= 2

    @pytest.mark.asyncio
    async def test_run_interactive_exit(self):
        """Test interactive mode with exit command."""
        with patch("builtins.input", side_effect=["exit"]):
            with patch.object(self.cli, "print_banner"):
                with patch.object(self.cli, "print_help"):
                    await self.cli.run_interactive()

    @pytest.mark.asyncio
    async def test_run_interactive_help(self):
        """Test interactive mode with help command."""
        with patch("builtins.input", side_effect=["help", "exit"]):
            with patch.object(self.cli, "print_help") as mock_help:
                await self.cli.run_interactive()
                mock_help.assert_called()

    @pytest.mark.asyncio
    async def test_run_interactive_create(self):
        """Test interactive mode with create command."""
        with patch("builtins.input", side_effect=["create test_graph", "exit"]):
            with patch.object(self.cli, "create_graph") as mock_create:
                await self.cli.run_interactive()
                mock_create.assert_called_with("test_graph", None)

    @pytest.mark.asyncio
    async def test_run_interactive_list(self):
        """Test interactive mode with list command."""
        with patch("builtins.input", side_effect=["list", "exit"]):
            with patch.object(self.cli, "list_graphs") as mock_list:
                await self.cli.run_interactive()
                mock_list.assert_called()

    @pytest.mark.asyncio
    async def test_run_interactive_invalid_command(self):
        """Test interactive mode with invalid command."""
        with patch("builtins.input", side_effect=["invalid_cmd", "exit"]):
            with patch("networkx_mcp.cli.console") as mock_console:
                await self.cli.run_interactive()
                assert any(
                    "Unknown command" in str(call)
                    for call in mock_console.print.call_args_list
                )

    @pytest.mark.asyncio
    async def test_run_interactive_keyboard_interrupt(self):
        """Test interactive mode with keyboard interrupt."""
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with patch("networkx_mcp.cli.console"):
                await self.cli.run_interactive()

    def test_process_command_add_nodes(self):
        """Test process_command with add nodes."""
        with patch.object(self.cli, "add_nodes") as mock_add:
            self.cli.process_command("add nodes 1,2,3")
            mock_add.assert_called_with("1,2,3")

    def test_process_command_add_edges(self):
        """Test process_command with add edges."""
        with patch.object(self.cli, "add_edges") as mock_add:
            self.cli.process_command("add edges (1,2),(3,4)")
            mock_add.assert_called_with("(1,2),(3,4)")

    def test_process_command_analyze(self):
        """Test process_command with analyze."""
        with patch.object(self.cli, "analyze_graph") as mock_analyze:
            self.cli.process_command("analyze degree")
            mock_analyze.assert_called_with("degree")

    def test_process_command_import(self):
        """Test process_command with import."""
        with patch.object(self.cli, "import_graph") as mock_import:
            self.cli.process_command("import file.json as my_graph")
            mock_import.assert_called_with("file.json", "my_graph")

    def test_process_command_export(self):
        """Test process_command with export."""
        with patch.object(self.cli, "export_graph") as mock_export:
            self.cli.process_command("export output.json")
            mock_export.assert_called_with("output.json", format=None)

    def test_process_command_export_with_format(self):
        """Test process_command with export and format."""
        with patch.object(self.cli, "export_graph") as mock_export:
            self.cli.process_command("export output.csv as csv")
            mock_export.assert_called_with("output.csv", format="csv")


class TestMainFunction:
    """Test main function."""

    def test_main_interactive(self):
        """Test main function in interactive mode."""
        with patch("sys.argv", ["cli.py", "--interactive"]):
            with patch("networkx_mcp.cli.NetworkXCLI") as mock_cli_class:
                mock_cli = Mock()
                mock_cli.run_interactive = AsyncMock()
                mock_cli_class.return_value = mock_cli
                with patch("asyncio.run") as mock_run:
                    main()
                    # Verify asyncio.run was called with the coroutine
                    mock_run.assert_called_once()

    def test_main_with_command(self):
        """Test main function with command."""
        with patch("sys.argv", ["cli.py", "list"]):
            with patch("networkx_mcp.cli.NetworkXCLI") as mock_cli_class:
                mock_cli = Mock()
                mock_cli_class.return_value = mock_cli
                main()
                mock_cli.process_command.assert_called_with("list")

    def test_main_no_args(self):
        """Test main function with no arguments."""
        with patch("sys.argv", ["cli.py"]):
            with patch("networkx_mcp.cli.console") as mock_console:
                main()
                assert any(
                    "No command" in str(call)
                    for call in mock_console.print.call_args_list
                )
