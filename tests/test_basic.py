"""Basic tests to verify the package can be imported."""


def test_import():
    """Test that the package can be imported."""
    import networkx_mcp

    assert networkx_mcp.__version__


def test_networkx_available():
    """Test that NetworkX is available."""
    import networkx as nx

    assert nx.__version__


def test_basic_graph_creation():
    """Test basic graph creation."""
    import networkx as nx

    G = nx.Graph()
    G.add_edge(1, 2)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1
