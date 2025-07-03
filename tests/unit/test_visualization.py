"""Comprehensive tests for Phase 3 visualization functionality."""

import base64
from unittest.mock import patch

import networkx as nx
import pytest

from networkx_mcp.visualization import (MatplotlibVisualizer, PlotlyVisualizer,
                                        PyvisVisualizer,
                                        SpecializedVisualizations)


class TestMatplotlibVisualizer:
    """Test matplotlib static visualization."""

    def test_create_static_plot_basic(self, sample_graphs):
        """Test basic static plot creation."""
        graph = sample_graphs["simple"]

        result = MatplotlibVisualizer.create_static_plot(
            graph, layout="spring", show_labels=True
        )

        assert isinstance(result, dict)
        assert "layout_used" in result
        assert "num_nodes" in result
        assert "num_edges" in result
        assert "formats" in result

        assert result["layout_used"] == "spring"
        assert result["num_nodes"] == graph.number_of_nodes()
        assert result["num_edges"] == graph.number_of_edges()

        # Check format outputs
        formats = result["formats"]
        assert "png_base64" in formats
        assert isinstance(formats["png_base64"], str)

        # Verify base64 encoding
        try:
            base64.b64decode(formats["png_base64"])
        except Exception:
            pytest.fail("PNG base64 data is invalid")

    def test_node_attribute_visualization(self, sample_graphs):
        """Test visualization with node attributes."""
        graph = sample_graphs["weighted"]

        # Set node colors based on attribute
        node_colors = {
            node: graph.nodes[node].get("color", "gray") for node in graph.nodes()
        }
        node_sizes = {node: (node + 1) * 100 for node in graph.nodes()}

        result = MatplotlibVisualizer.create_static_plot(
            graph,
            layout="circular",
            node_color=node_colors,
            node_size=node_sizes,
            show_labels=True,
        )

        assert result["layout_used"] == "circular"
        assert "formats" in result
        assert "png_base64" in result["formats"]

    def test_edge_attribute_visualization(self, sample_graphs):
        """Test visualization with edge attributes."""
        graph = sample_graphs["weighted"]

        # Set edge attributes for visualization
        edge_widths = {
            edge: graph.edges[edge].get("weight", 1.0) for edge in graph.edges()
        }
        edge_colors = {
            edge: "red" if graph.edges[edge].get("weight", 1) > 2 else "blue"
            for edge in graph.edges()
        }

        result = MatplotlibVisualizer.create_static_plot(
            graph,
            layout="spring",
            edge_width=edge_widths,
            edge_color=edge_colors,
            edge_style="solid",
        )

        assert "formats" in result
        assert result["num_edges"] == graph.number_of_edges()

    def test_directed_graph_visualization(self, sample_graphs):
        """Test directed graph visualization with arrows."""
        graph = sample_graphs["directed"]

        result = MatplotlibVisualizer.create_static_plot(
            graph, layout="kamada_kawai", show_labels=True, title="Directed Graph Test"
        )

        assert result["layout_used"] == "kamada_kawai"
        assert "formats" in result

    def test_different_layouts(self, sample_graphs):
        """Test different layout algorithms."""
        graph = sample_graphs["simple"]

        layouts = ["spring", "circular", "random", "shell", "spectral"]

        for layout in layouts:
            result = MatplotlibVisualizer.create_static_plot(graph, layout=layout)

            assert result["layout_used"] == layout
            assert "formats" in result

    def test_hierarchical_layout(self, sample_graphs):
        """Test hierarchical layout for tree-like structures."""
        graph = sample_graphs["tree"]

        result = MatplotlibVisualizer.create_static_plot(graph, layout="hierarchical")

        assert result["layout_used"] == "hierarchical"
        assert "formats" in result

    def test_large_graph_performance(self, sample_graphs):
        """Test visualization performance on larger graphs."""
        graph = sample_graphs["large"]

        import time

        start_time = time.time()

        result = MatplotlibVisualizer.create_static_plot(
            graph,
            layout="spring",
            show_labels=False,  # Disable labels for large graphs
        )

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (less than 5 seconds)
        assert elapsed_time < 5.0
        assert "formats" in result

    def test_custom_styling_options(self, sample_graphs):
        """Test custom styling and formatting options."""
        graph = sample_graphs["simple"]

        result = MatplotlibVisualizer.create_static_plot(
            graph,
            layout="spring",
            node_color="red",
            node_size=500,
            node_shape="square",
            edge_color="blue",
            edge_width=2.0,
            edge_style="dashed",
            title="Custom Styled Graph",
            figsize=(10, 8),
            dpi=150,
        )

        assert "formats" in result
        assert "png_base64" in result["formats"]

        # Test SVG inclusion
        if "svg" in result["formats"]:
            assert isinstance(result["formats"]["svg"], str)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Empty graph
        empty_graph = nx.Graph()

        # Should handle empty graph gracefully
        result = MatplotlibVisualizer.create_static_plot(empty_graph)
        assert "formats" in result
        assert result["num_nodes"] == 0
        assert result["num_edges"] == 0

        # Invalid layout
        graph = nx.complete_graph(3)
        result = MatplotlibVisualizer.create_static_plot(
            graph,
            layout="invalid_layout",  # Should fall back to spring
        )
        assert result["layout_used"] == "spring"


class TestPlotlyVisualizer:
    """Test Plotly interactive visualization."""

    def test_create_interactive_plot_basic(self, sample_graphs):
        """Test basic interactive plot creation."""
        graph = sample_graphs["simple"]

        result = PlotlyVisualizer.create_interactive_plot(graph, layout="spring")

        assert isinstance(result, dict)
        assert "html" in result or "json" in result or "plot_data" in result
        assert "layout_used" in result
        assert "num_nodes" in result
        assert "num_edges" in result

        assert result["layout_used"] == "spring"
        assert result["num_nodes"] == graph.number_of_nodes()

    def test_3d_visualization(self, sample_graphs):
        """Test 3D visualization."""
        graph = sample_graphs["simple"]

        result = PlotlyVisualizer.create_3d_plot(graph, layout="spring_3d")

        assert "plot_data" in result or "html" in result
        assert "layout_used" in result
        assert result["layout_used"] == "spring_3d"

    def test_interactive_features(self, sample_graphs):
        """Test interactive features like hover and selection."""
        graph = sample_graphs["weighted"]

        # Add node information for hover
        for node in graph.nodes():
            graph.nodes[node]["info"] = f"Node {node}"

        result = PlotlyVisualizer.create_interactive_plot(
            graph, layout="circular", show_hover=True, enable_selection=True
        )

        assert "plot_data" in result or "html" in result

    def test_network_analysis_dashboard(self, sample_graphs):
        """Test dashboard creation with multiple views."""
        graph = sample_graphs["karate"]

        result = PlotlyVisualizer.create_network_dashboard(
            graph, include_metrics=True, include_centrality=True
        )

        assert "dashboard_html" in result or "plot_data" in result
        assert "components" in result

    def test_animated_visualization(self, sample_graphs):
        """Test animated graph evolution."""
        graphs = [sample_graphs["simple"]]

        # Create evolution sequence
        for i in range(3):
            g = graphs[-1].copy()
            g.add_node(f"new_{i}")
            g.add_edge(next(iter(g.nodes())), f"new_{i}")
            graphs.append(g)

        result = PlotlyVisualizer.create_animated_plot(
            graphs, layout="spring", frame_duration=500
        )

        assert "animation_html" in result or "plot_data" in result
        assert "num_frames" in result
        assert result["num_frames"] == len(graphs)

    def test_large_graph_optimization(self, sample_graphs):
        """Test optimizations for large graphs."""
        graph = sample_graphs["large"]

        result = PlotlyVisualizer.create_interactive_plot(
            graph, layout="spring", optimize_for_size=True, max_nodes_full_render=500
        )

        assert "plot_data" in result or "html" in result
        # Should use optimization strategies for large graphs

    def test_custom_colorscales(self, sample_graphs):
        """Test custom color scales and palettes."""
        graph = sample_graphs["weighted"]

        # Set numeric node attributes
        for i, node in enumerate(graph.nodes()):
            graph.nodes[node]["value"] = i * 10

        result = PlotlyVisualizer.create_interactive_plot(
            graph, layout="spring", node_color_attr="value", colorscale="viridis"
        )

        assert "plot_data" in result or "html" in result

    @patch("networkx_mcp.visualization.plotly_visualizer.go")
    def test_plotly_dependency_error(self, mock_go, sample_graphs):
        """Test handling when Plotly is not available."""
        mock_go.side_effect = ImportError("Plotly not installed")
        graph = sample_graphs["simple"]

        with pytest.raises(ImportError):
            PlotlyVisualizer.create_interactive_plot(graph)


class TestPyvisVisualizer:
    """Test Pyvis network visualization."""

    def test_create_network_visualization(self, sample_graphs):
        """Test basic Pyvis network creation."""
        graph = sample_graphs["simple"]

        result = PyvisVisualizer.create_network(graph, height="400px", width="600px")

        assert "html" in result
        assert "num_nodes" in result
        assert "num_edges" in result
        assert isinstance(result["html"], str)
        assert "<html>" in result["html"].lower()

    def test_physics_simulation(self, sample_graphs):
        """Test physics simulation options."""
        graph = sample_graphs["weighted"]

        result = PyvisVisualizer.create_network(
            graph,
            physics=True,
            physics_config={"enabled": True, "stabilization": {"iterations": 100}},
        )

        assert "html" in result
        # Should include physics configuration in HTML

    def test_hierarchical_layout_pyvis(self, sample_graphs):
        """Test hierarchical layout in Pyvis."""
        graph = sample_graphs["tree"]

        result = PyvisVisualizer.create_hierarchical_network(
            graph,
            layout_direction="UD",  # Up-Down
        )

        assert "html" in result
        assert "layout_direction" in result
        assert result["layout_direction"] == "UD"

    def test_custom_node_styling(self, sample_graphs):
        """Test custom node styling options."""
        graph = sample_graphs["weighted"]

        # Add visual attributes
        for node in graph.nodes():
            graph.nodes[node]["color"] = "red" if node % 2 == 0 else "blue"
            graph.nodes[node]["size"] = (node + 1) * 5
            graph.nodes[node]["title"] = f"Node {node} information"

        result = PyvisVisualizer.create_network(
            graph,
            node_color_attr="color",
            node_size_attr="size",
            node_hover_attr="title",
        )

        assert "html" in result

    def test_edge_styling(self, sample_graphs):
        """Test edge styling and labels."""
        graph = sample_graphs["weighted"]

        # Add edge labels
        for edge in graph.edges():
            weight = graph.edges[edge].get("weight", 1.0)
            graph.edges[edge]["label"] = f"w={weight:.1f}"
            graph.edges[edge]["width"] = weight

        result = PyvisVisualizer.create_network(
            graph,
            edge_width_attr="width",
            edge_label_attr="label",
            show_edge_labels=True,
        )

        assert "html" in result

    def test_interactive_options(self, sample_graphs):
        """Test interactive options and controls."""
        graph = sample_graphs["simple"]

        result = PyvisVisualizer.create_network(
            graph, show_buttons=True, filter_menu=True, select_menu=True
        )

        assert "html" in result
        html_content = result["html"]

        # Should include interactive controls
        assert "configure" in html_content.lower() or "filter" in html_content.lower()


class TestSpecializedVisualizations:
    """Test specialized visualization functions."""

    def test_matrix_visualization(self, sample_graphs):
        """Test adjacency matrix visualization."""
        graph = sample_graphs["simple"]

        result = SpecializedVisualizations.adjacency_matrix_plot(
            graph, colormap="Blues", show_values=True
        )

        assert "matrix_data" in result
        assert "image_data" in result or "plot_data" in result
        assert result["matrix_data"]["shape"][0] == graph.number_of_nodes()

    def test_degree_distribution_plot(self, sample_graphs):
        """Test degree distribution visualization."""
        graph = sample_graphs["scale_free"]

        result = SpecializedVisualizations.degree_distribution_plot(
            graph, log_scale=True, fit_powerlaw=True
        )

        assert "degree_sequence" in result
        assert "distribution" in result
        assert "plot_data" in result or "image_data" in result

        if "powerlaw_fit" in result:
            assert "alpha" in result["powerlaw_fit"]
            assert "xmin" in result["powerlaw_fit"]

    def test_centrality_heatmap(self, sample_graphs):
        """Test centrality measures heatmap."""
        graph = sample_graphs["karate"]

        result = SpecializedVisualizations.centrality_heatmap(
            graph, measures=["degree", "betweenness", "closeness", "eigenvector"]
        )

        assert "centrality_data" in result
        assert "heatmap_data" in result or "plot_data" in result
        assert len(result["centrality_data"]) == 4  # Four measures

    def test_community_visualization(self, sample_graphs):
        """Test community structure visualization."""
        graph = sample_graphs["karate"]

        result = SpecializedVisualizations.community_plot(
            graph, algorithm="louvain", show_modularity=True
        )

        assert "communities" in result
        assert "modularity" in result
        assert "plot_data" in result or "image_data" in result
        assert result["modularity"] >= 0

    def test_subgraph_comparison(self, sample_graphs):
        """Test side-by-side subgraph comparison."""
        graph = sample_graphs["karate"]

        # Create two subgraphs
        nodes1 = list(graph.nodes())[:17]
        nodes2 = list(graph.nodes())[17:]

        result = SpecializedVisualizations.subgraph_comparison(
            graph, subgraph_nodes=[nodes1, nodes2], titles=["Group 1", "Group 2"]
        )

        assert "subgraphs" in result
        assert "comparison_plot" in result or "plot_data" in result
        assert len(result["subgraphs"]) == 2

    def test_temporal_network_plot(self):
        """Test temporal network evolution visualization."""
        # Create sequence of graphs
        graphs = []
        base = nx.path_graph(5)

        for i in range(5):
            g = base.copy()
            # Add time-specific edges
            for j in range(i):
                g.add_edge(0, j + 1)
            graphs.append(g)

        result = SpecializedVisualizations.temporal_network_plot(
            graphs, time_labels=[f"t={i}" for i in range(5)], layout="spring"
        )

        assert "temporal_data" in result
        assert "animation_data" in result or "plot_data" in result
        assert len(result["temporal_data"]) == 5

    def test_bipartite_visualization(self, sample_graphs):
        """Test bipartite graph specialized layout."""
        graph = sample_graphs["bipartite"]

        result = SpecializedVisualizations.bipartite_plot(
            graph, node_sets_attr="bipartite", vertical_layout=True
        )

        assert "node_sets" in result
        assert "bipartite_layout" in result
        assert "plot_data" in result or "image_data" in result
        assert len(result["node_sets"]) == 2

    def test_flow_network_visualization(self, sample_graphs):
        """Test flow network with capacity visualization."""
        graph = sample_graphs["directed"]

        result = SpecializedVisualizations.flow_network_plot(
            graph, source="start", sink="end", capacity_attr="capacity", show_flow=True
        )

        assert "flow_data" in result
        assert "max_flow_value" in result
        assert "plot_data" in result or "image_data" in result

    def test_error_handling_specialized(self):
        """Test error handling in specialized visualizations."""
        # Empty graph
        empty_graph = nx.Graph()

        # Should handle gracefully
        result = SpecializedVisualizations.degree_distribution_plot(empty_graph)
        assert "error" in result or "degree_sequence" in result

        # Graph with no edges
        isolated_graph = nx.Graph()
        isolated_graph.add_nodes_from([1, 2, 3])

        result = SpecializedVisualizations.centrality_heatmap(
            isolated_graph, measures=["degree"]
        )
        assert "centrality_data" in result


class TestVisualizationIntegration:
    """Test integration between different visualization backends."""

    def test_backend_comparison(self, sample_graphs):
        """Test creating same visualization with different backends."""
        graph = sample_graphs["simple"]

        # Matplotlib version
        matplotlib_result = MatplotlibVisualizer.create_static_plot(
            graph, layout="spring"
        )

        # Plotly version
        plotly_result = PlotlyVisualizer.create_interactive_plot(graph, layout="spring")

        # Both should succeed and have similar metadata
        assert matplotlib_result["num_nodes"] == plotly_result["num_nodes"]
        assert matplotlib_result["num_edges"] == plotly_result["num_edges"]
        assert matplotlib_result["layout_used"] == plotly_result["layout_used"]

    def test_format_conversion(self, sample_graphs):
        """Test converting between visualization formats."""
        graph = sample_graphs["simple"]

        # Create matplotlib plot
        static_result = MatplotlibVisualizer.create_static_plot(graph)

        # Create interactive version
        interactive_result = PlotlyVisualizer.create_interactive_plot(graph)

        # Both should represent the same graph
        assert static_result["num_nodes"] == interactive_result["num_nodes"]
        assert static_result["num_edges"] == interactive_result["num_edges"]

    def test_visualization_export_formats(self, sample_graphs):
        """Test exporting visualizations in different formats."""
        graph = sample_graphs["simple"]

        # Test matplotlib formats
        result = MatplotlibVisualizer.create_static_plot(
            graph, formats=["png", "svg", "pdf"]
        )

        formats = result.get("formats", {})
        assert "png_base64" in formats

        if "svg" in formats:
            assert isinstance(formats["svg"], str)
            assert "<svg" in formats["svg"]

    def test_large_graph_backend_selection(self, sample_graphs):
        """Test automatic backend selection for large graphs."""
        large_graph = sample_graphs["large"]

        # For large graphs, should prefer faster backends
        # or use optimizations

        import time

        start_time = time.time()
        result = MatplotlibVisualizer.create_static_plot(
            large_graph, layout="spring", show_labels=False
        )
        matplotlib_time = time.time() - start_time

        # Should complete in reasonable time
        assert matplotlib_time < 10.0
        assert "formats" in result


class TestVisualizationPerformance:
    """Test visualization performance and memory usage."""

    def test_memory_usage_large_graph(self, sample_graphs, performance_thresholds):
        """Test memory usage with large graphs."""
        graph = sample_graphs["large"]

        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create visualization
        result = MatplotlibVisualizer.create_static_plot(graph, layout="spring")

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Should not use excessive memory (less than 100MB increase)
        assert memory_increase < 100 * 1024 * 1024
        assert "formats" in result

    def test_rendering_time_scaling(self, benchmark_data, performance_thresholds):
        """Test how rendering time scales with graph size."""
        import time

        times = {}

        for name, graph in benchmark_data.items():
            if "erdos_renyi" in name:
                start_time = time.time()

                result = MatplotlibVisualizer.create_static_plot(
                    graph, layout="spring", show_labels=False
                )

                elapsed = time.time() - start_time
                times[name] = elapsed

                assert "formats" in result

        # Rendering time should scale reasonably
        # (may not be perfectly linear due to layout algorithms)
        assert all(t < 10.0 for t in times.values())

    def test_concurrent_visualization(self, sample_graphs):
        """Test creating multiple visualizations concurrently."""
        import concurrent.futures

        def create_viz(graph_name):
            graph = sample_graphs[graph_name]
            return MatplotlibVisualizer.create_static_plot(graph)

        graph_names = ["simple", "weighted", "complete", "tree"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(create_viz, name): name for name in graph_names}

            results = {}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    result = future.result(timeout=30)
                    results[name] = result
                except Exception as e:
                    pytest.fail(f"Visualization failed for {name}: {e}")

        # All visualizations should succeed
        assert len(results) == len(graph_names)
        for _name, result in results.items():
            assert "formats" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
