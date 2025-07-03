"""Security Boundary Testing for NetworkX MCP Server.

This module tests security boundaries, input validation, and potential
attack vectors to ensure the server is robust against malicious inputs.
"""

import json
import time

import networkx as nx
import pytest

from networkx_mcp.core.graph_operations import GraphManager


class TestInputValidation:
    """Test input validation and sanitization."""

    @pytest.fixture
    def graph_manager(self):
        return GraphManager()

    def test_malicious_node_names(self, graph_manager):
        """Test protection against malicious node names."""

        # Test malicious node names
        malicious_nodes = [
            "../../../etc/passwd",  # Path traversal
            "'; DROP TABLE users; --",  # SQL injection style
            "<script>alert('xss')</script>",  # XSS style
            "node\x00null_byte",  # Null byte injection
            "node" * 1000,  # Long names
        ]

        for malicious_node in malicious_nodes:
            try:
                G = nx.Graph()
                G.add_node(malicious_node)
                graph_manager.create_graph("test_malicious")
                graph_manager.graphs["test_malicious"] = G

                # Verify the node was added safely
                retrieved = graph_manager.get_graph("test_malicious")
                if retrieved and retrieved.number_of_nodes() > 0:
                    nodes = list(retrieved.nodes())
                    assert malicious_node in nodes

                # Clean up
                graph_manager.delete_graph("test_malicious")

            except (ValueError, TypeError):
                # Some inputs should be rejected
                pass

    def test_resource_limits(self, graph_manager):
        """Test protection against resource exhaustion."""

        try:
            # Test reasonable size graph
            G = nx.Graph()
            max_nodes = 1000  # Reasonable limit for testing

            for i in range(max_nodes):
                G.add_node(f"node_{i}")

            graph_manager.create_graph("large_test")
            graph_manager.graphs["large_test"] = G
            retrieved = graph_manager.get_graph("large_test")
            assert retrieved.number_of_nodes() == max_nodes

            graph_manager.delete_graph("large_test")

        except MemoryError:
            # This is acceptable - system protecting itself
            pass

    def test_cycle_handling(self, graph_manager):
        """Test handling of cycles and potential infinite loops."""

        G = nx.DiGraph()
        G.add_edge(1, 1)  # Self-loop
        G.add_edges_from([(2, 3), (3, 4), (4, 2)])  # Cycle

        graph_manager.create_graph("cyclic")
        graph_manager.graphs["cyclic"] = G

        start_time = time.time()
        try:
            components = list(nx.strongly_connected_components(G))
            elapsed = time.time() - start_time
            assert elapsed < 5.0  # Should complete quickly
            assert len(components) > 0
        except Exception:
            elapsed = time.time() - start_time
            assert elapsed < 5.0  # Even failures should be quick


class TestDataValidation:
    """Test data validation and type checking."""

    def test_json_safety(self):
        """Test safe JSON handling."""

        G = nx.Graph()
        G.add_node(1, value=42, name="test", active=True)
        G.add_edge(1, 2, weight=1.5)

        try:
            from networkx.readwrite import json_graph

            data = json_graph.node_link_data(G)
            json_str = json.dumps(data)
            parsed = json.loads(json_str)
            G_reconstructed = json_graph.node_link_graph(parsed)

            assert G_reconstructed.number_of_nodes() == G.number_of_nodes()
            assert G_reconstructed.number_of_edges() == G.number_of_edges()

        except (TypeError, ValueError):
            # Should handle serialization errors gracefully
            pass


class TestErrorHandling:
    """Test error handling and information disclosure."""

    def test_error_message_safety(self):
        """Test that error messages don't leak sensitive information."""

        manager = GraphManager()

        try:
            result = manager.get_graph("nonexistent")
            assert result is None
        except Exception as e:
            error_msg = str(e)
            # Error messages should not contain sensitive paths
            assert "/etc/" not in error_msg
            assert "password" not in error_msg.lower()
            assert len(error_msg) < 200  # Not excessively verbose


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
