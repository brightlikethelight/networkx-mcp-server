"""Tests for validators/__init__.py - Target: 100% coverage (4 lines)."""


class TestValidatorsInit:
    """Test the validators package imports."""

    def test_module_imports(self):
        """Test that all expected classes can be imported from validators."""
        from networkx_mcp import validators

        # Check that the module exists
        assert validators is not None

        # Check __all__ exports
        assert hasattr(validators, "__all__")
        assert isinstance(validators.__all__, list)

        # Expected exports from __all__
        expected_exports = [
            "GraphValidator",
            "GraphValidationRule",
            "AlgorithmValidator",
            "AlgorithmSpec",
        ]

        for export in expected_exports:
            assert export in validators.__all__

    def test_direct_imports(self):
        """Test direct imports of validator classes."""
        from networkx_mcp.validators import (
            AlgorithmSpec,
            AlgorithmValidator,
            GraphValidationRule,
            GraphValidator,
        )

        # Verify all classes are imported successfully
        assert GraphValidator is not None
        assert GraphValidationRule is not None
        assert AlgorithmValidator is not None
        assert AlgorithmSpec is not None

        # Verify they are classes (not instances)
        assert isinstance(GraphValidator, type)
        assert isinstance(GraphValidationRule, type)
        assert isinstance(AlgorithmValidator, type)
        assert isinstance(AlgorithmSpec, type)

    def test_validator_functionality(self):
        """Test basic functionality of imported validators."""
        from networkx_mcp.validators import AlgorithmValidator, GraphValidator

        # GraphValidator should have async validation methods
        assert hasattr(GraphValidator, "validate_graph_creation")
        assert hasattr(GraphValidator, "validate_graph_data")
        assert hasattr(GraphValidator, "validate_node_operation")
        assert hasattr(GraphValidator, "validate_edge_operation")

        # AlgorithmValidator should have validation methods
        assert hasattr(AlgorithmValidator, "validate_algorithm_request")
        assert hasattr(AlgorithmValidator, "validate_algorithm_compatibility")
