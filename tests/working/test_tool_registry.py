"""Tests for tool_registry.py — ToolRegistry, ToolDef, and build_registry."""

from unittest.mock import MagicMock, patch

import pytest

from networkx_mcp.tool_registry import ToolDef, ToolRegistry, build_registry


# ═══════════════════════════════════════════════════════════════════════
# ToolDef
# ═══════════════════════════════════════════════════════════════════════


class TestToolDef:
    def test_defaults(self):
        td = ToolDef(
            name="t",
            description="d",
            input_schema={"type": "object"},
            handler=lambda a: a,
        )
        assert td.is_write is False
        assert td.graph_param is None

    def test_custom_fields(self):
        td = ToolDef(
            name="t",
            description="d",
            input_schema={"type": "object"},
            handler=lambda a: a,
            is_write=True,
            graph_param="graph",
        )
        assert td.is_write is True
        assert td.graph_param == "graph"


# ═══════════════════════════════════════════════════════════════════════
# ToolRegistry
# ═══════════════════════════════════════════════════════════════════════


class TestToolRegistry:
    @pytest.fixture()
    def registry(self):
        return ToolRegistry()

    @pytest.fixture()
    def sample_tool(self):
        return ToolDef(
            name="sample",
            description="A sample tool",
            input_schema={"type": "object", "properties": {}},
            handler=lambda a: {"ok": True},
        )

    def test_register_and_get(self, registry, sample_tool):
        registry.register(sample_tool)
        assert registry.get("sample") is sample_tool

    def test_get_missing_returns_none(self, registry):
        assert registry.get("nonexistent") is None

    def test_contains(self, registry, sample_tool):
        assert "sample" not in registry
        registry.register(sample_tool)
        assert "sample" in registry

    def test_len(self, registry, sample_tool):
        assert len(registry) == 0
        registry.register(sample_tool)
        assert len(registry) == 1

    def test_list_schemas(self, registry, sample_tool):
        registry.register(sample_tool)
        schemas = registry.list_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "sample"
        assert schemas[0]["description"] == "A sample tool"
        assert "inputSchema" in schemas[0]

    def test_write_tool_names_empty(self, registry, sample_tool):
        registry.register(sample_tool)  # is_write=False by default
        assert registry.write_tool_names() == set()

    def test_write_tool_names_with_write_tools(self, registry):
        read_tool = ToolDef(
            name="read_op",
            description="read",
            input_schema={"type": "object"},
            handler=lambda a: a,
        )
        write_tool = ToolDef(
            name="write_op",
            description="write",
            input_schema={"type": "object"},
            handler=lambda a: a,
            is_write=True,
        )
        registry.register(read_tool)
        registry.register(write_tool)
        assert registry.write_tool_names() == {"write_op"}

    def test_register_overwrites(self, registry):
        t1 = ToolDef(name="dup", description="v1", input_schema={}, handler=lambda a: 1)
        t2 = ToolDef(name="dup", description="v2", input_schema={}, handler=lambda a: 2)
        registry.register(t1)
        registry.register(t2)
        assert len(registry) == 1
        assert registry.get("dup").description == "v2"


# ═══════════════════════════════════════════════════════════════════════
# build_registry
# ═══════════════════════════════════════════════════════════════════════


class TestBuildRegistry:
    def test_default_returns_populated_registry(self):
        reg = build_registry()
        assert isinstance(reg, ToolRegistry)
        assert len(reg) > 0

    def test_core_tools_registered(self):
        reg = build_registry()
        core = [
            "create_graph",
            "add_nodes",
            "add_edges",
            "get_info",
            "delete_graph",
            "shortest_path",
            "degree_centrality",
            "betweenness_centrality",
            "connected_components",
            "pagerank",
            "community_detection",
            "visualize_graph",
            "import_csv",
            "export_json",
        ]
        for name in core:
            assert name in reg, f"{name} missing from registry"

    def test_academic_tools_registered(self):
        reg = build_registry()
        academic = [
            "build_citation_network",
            "analyze_author_impact",
            "find_collaboration_patterns",
            "detect_research_trends",
            "export_bibtex",
            "recommend_papers",
            "resolve_doi",
        ]
        for name in academic:
            assert name in reg, f"{name} missing from registry"

    def test_write_tools_marked_correctly(self):
        reg = build_registry()
        writes = reg.write_tool_names()
        assert "create_graph" in writes
        assert "add_nodes" in writes
        assert "add_edges" in writes
        assert "delete_graph" in writes
        assert "import_csv" in writes
        # Read-only tools should NOT be in writes
        assert "get_info" not in writes
        assert "shortest_path" not in writes
        assert "degree_centrality" not in writes

    def test_graph_param_set_correctly(self):
        reg = build_registry()
        # create_graph uses "name"
        assert reg.get("create_graph").graph_param == "name"
        # Most tools use "graph"
        assert reg.get("add_nodes").graph_param == "graph"
        assert reg.get("get_info").graph_param == "graph"
        # resolve_doi has no graph param
        assert reg.get("resolve_doi").graph_param is None

    def test_no_monitoring_by_default(self):
        reg = build_registry()
        assert "health_status" not in reg

    def test_monitoring_enabled(self):
        mock_monitor = MagicMock()
        mock_monitor.get_health_status.return_value = {"status": "ok"}
        reg = build_registry(monitoring_enabled=True, monitor=mock_monitor)
        assert "health_status" in reg

    def test_monitoring_enabled_without_monitor(self):
        reg = build_registry(monitoring_enabled=True, monitor=None)
        assert "health_status" not in reg

    def test_list_schemas_mcp_format(self):
        reg = build_registry()
        schemas = reg.list_schemas()
        assert len(schemas) > 0
        for s in schemas:
            assert "name" in s
            assert "description" in s
            assert "inputSchema" in s

    def test_cicd_tools_when_module_available(self):
        """CI/CD tools register when networkx_mcp.tools is importable."""
        reg = build_registry()
        # The module exists in this codebase, so CI/CD tools should be registered
        cicd_tools = [
            "trigger_workflow",
            "get_workflow_status",
            "cancel_workflow",
            "rerun_failed_jobs",
            "get_dora_metrics",
            "analyze_workflow_failures",
        ]
        for name in cicd_tools:
            assert name in reg, f"CI/CD tool {name} missing from registry"

    @patch("importlib.util.find_spec", return_value=None)
    def test_cicd_tools_absent_when_module_missing(self, mock_spec):
        reg = build_registry()
        assert "trigger_workflow" not in reg
        assert "get_workflow_status" not in reg
