"""Simplest possible test to debug CI issues."""


def test_simple():
    """Test that always passes."""
    assert True


def test_import_networkx():
    """Test NetworkX import."""
    import networkx as nx

    assert nx is not None


def test_import_package():
    """Test package import."""
    import networkx_mcp

    assert networkx_mcp is not None
