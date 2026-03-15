"""Tests to ensure all live modules import successfully."""


class TestModuleImports:
    """Test that all live modules import without errors."""

    def test_version_module(self):
        from networkx_mcp.__version__ import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_main_init_module(self):
        import networkx_mcp

        assert hasattr(networkx_mcp, "__version__")

    def test_server_module(self):
        from networkx_mcp import server

        assert hasattr(server, "NetworkXMCPServer")
        assert hasattr(server, "main")
        assert hasattr(server, "graphs")

    def test_errors_module(self):
        from networkx_mcp.errors import (
            ErrorCodes,
            GraphNotFoundError,
            InvalidGraphIdError,
            InvalidNodeIdError,
            MCPError,
        )

        assert ErrorCodes.PARSE_ERROR == -32700
        assert issubclass(GraphNotFoundError, MCPError)
        assert issubclass(InvalidGraphIdError, MCPError)
        assert issubclass(InvalidNodeIdError, MCPError)

    def test_graph_cache_module(self):
        from networkx_mcp.graph_cache import GraphCache, GraphDict

        assert callable(GraphCache)
        assert callable(GraphDict)

    def test_auth_module(self):
        from networkx_mcp.auth import APIKeyManager, AuthMiddleware

        assert callable(APIKeyManager)
        assert callable(AuthMiddleware)

    def test_core_modules(self):
        from networkx_mcp.core.algorithms import GraphAlgorithms
        from networkx_mcp.core.basic_operations import create_graph
        from networkx_mcp.core.graph_operations import GraphManager

        assert callable(GraphAlgorithms)
        assert callable(GraphManager)
        assert callable(create_graph)

    def test_academic_modules(self):
        from networkx_mcp.academic.analytics import (
            calculate_h_index,
            detect_research_trends,
            find_collaboration_patterns,
        )
        from networkx_mcp.academic.citations import resolve_doi

        assert callable(calculate_h_index)
        assert callable(find_collaboration_patterns)
        assert callable(detect_research_trends)
        assert callable(resolve_doi)

    def test_monitoring_module(self):
        from networkx_mcp.monitoring.dora_metrics import get_dora_metrics

        assert callable(get_dora_metrics)

    def test_tools_module(self):
        from networkx_mcp.tools.cicd_control import CICDController

        assert callable(CICDController)
