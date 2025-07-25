"""Tests for schemas/__init__.py - Target: 100% coverage (22 lines)."""


class TestSchemasInit:
    """Test the schemas package imports."""

    def test_module_imports(self):
        """Test that all expected classes can be imported from schemas."""
        from networkx_mcp import schemas

        # Check that the module exists
        assert schemas is not None

        # Check __all__ exports
        assert hasattr(schemas, "__all__")
        assert isinstance(schemas.__all__, list)

        # Expected exports from __all__
        expected_exports = [
            "AddEdgeRequest",
            "AddEdgesRequest",
            "AddNodeRequest",
            "AddNodesRequest",
            "AlgorithmResponse",
            "CentralityRequest",
            "CommunityDetectionRequest",
            "CreateGraphRequest",
            "EdgeSchema",
            "ExportGraphRequest",
            "GraphAttributesRequest",
            "GraphInfoResponse",
            "GraphSchema",
            "ImportGraphRequest",
            "LayoutRequest",
            "NodeSchema",
            "ShortestPathRequest",
            "SubgraphRequest",
            "VisualizationData",
        ]

        # Verify all expected exports are in __all__
        assert len(schemas.__all__) == len(expected_exports)
        for export in expected_exports:
            assert export in schemas.__all__

    def test_direct_imports(self):
        """Test direct imports of all schema classes."""
        from networkx_mcp.schemas import (
            AddEdgeRequest,
            AddEdgesRequest,
            AddNodeRequest,
            AddNodesRequest,
            AlgorithmResponse,
            CentralityRequest,
            CommunityDetectionRequest,
            CreateGraphRequest,
            EdgeSchema,
            ExportGraphRequest,
            GraphAttributesRequest,
            GraphInfoResponse,
            GraphSchema,
            ImportGraphRequest,
            LayoutRequest,
            NodeSchema,
            ShortestPathRequest,
            SubgraphRequest,
            VisualizationData,
        )

        # Verify all imports are successful
        assert AddEdgeRequest is not None
        assert AddEdgesRequest is not None
        assert AddNodeRequest is not None
        assert AddNodesRequest is not None
        assert AlgorithmResponse is not None
        assert CentralityRequest is not None
        assert CommunityDetectionRequest is not None
        assert CreateGraphRequest is not None
        assert EdgeSchema is not None
        assert ExportGraphRequest is not None
        assert GraphAttributesRequest is not None
        assert GraphInfoResponse is not None
        assert GraphSchema is not None
        assert ImportGraphRequest is not None
        assert LayoutRequest is not None
        assert NodeSchema is not None
        assert ShortestPathRequest is not None
        assert SubgraphRequest is not None
        assert VisualizationData is not None

        # Verify they are all classes
        all_schemas = [
            AddEdgeRequest,
            AddEdgesRequest,
            AddNodeRequest,
            AddNodesRequest,
            AlgorithmResponse,
            CentralityRequest,
            CommunityDetectionRequest,
            CreateGraphRequest,
            EdgeSchema,
            ExportGraphRequest,
            GraphAttributesRequest,
            GraphInfoResponse,
            GraphSchema,
            ImportGraphRequest,
            LayoutRequest,
            NodeSchema,
            ShortestPathRequest,
            SubgraphRequest,
            VisualizationData,
        ]

        for schema in all_schemas:
            assert isinstance(schema, type), f"{schema.__name__} should be a class"

    def test_schema_basic_functionality(self):
        """Test basic functionality of imported schema classes."""
        from networkx_mcp.schemas import (
            CreateGraphRequest,
            EdgeSchema,
            GraphSchema,
            NodeSchema,
        )

        # All schema classes should be dataclasses or have similar structure
        # Just verify they can be imported and are classes
        assert hasattr(NodeSchema, "__init__")
        assert hasattr(EdgeSchema, "__init__")
        assert hasattr(GraphSchema, "__init__")
        assert hasattr(CreateGraphRequest, "__init__")

    def test_import_from_graph_schemas(self):
        """Test that imports are coming from graph_schemas module."""
        # This tests the actual import statement in __init__.py
        from networkx_mcp.schemas import NodeSchema, graph_schemas

        # Verify NodeSchema comes from graph_schemas
        assert hasattr(graph_schemas, "NodeSchema")
        assert graph_schemas.NodeSchema is NodeSchema
