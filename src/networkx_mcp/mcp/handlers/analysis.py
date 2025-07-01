"""Analysis Handler for NetworkX MCP Server.

This module handles graph analysis tools including statistics,
community detection, bipartite analysis, and other analytical operations.
"""

from typing import Any, Optional
import networkx as nx
from networkx.algorithms import bipartite


try:
    from fastmcp import FastMCP
except ImportError:
    from networkx_mcp.mcp_mock import MockMCP as FastMCP


class AnalysisHandler:
    """Handler for graph analysis operations."""

    def __init__(self, mcp: FastMCP, graph_manager):
        """Initialize the handler with MCP server and graph manager."""
        self.mcp = mcp
        self.graph_manager = graph_manager
        self._register_tools()

    def _register_tools(self):
        """Register all analysis tools."""

        @self.mcp.tool()
        async def graph_statistics(graph_id: str) -> dict[str, Any]:
            """Calculate comprehensive graph statistics.

            Args:
                graph_id: ID of the graph

            Returns:
                Dict with various graph statistics
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                stats = {
                    "basic": {
                        "num_nodes": G.number_of_nodes(),
                        "num_edges": G.number_of_edges(),
                        "density": nx.density(G),
                        "is_directed": G.is_directed(),
                        "is_multigraph": G.is_multigraph(),
                    }
                }

                # Degree statistics
                degrees = [d for n, d in G.degree()]
                if degrees:
                    stats["degree"] = {
                        "average": sum(degrees) / len(degrees),
                        "max": max(degrees),
                        "min": min(degrees),
                        "distribution": dict(nx.degree_histogram(G)),
                    }

                # Connectivity
                if G.is_directed():
                    stats["connectivity"] = {
                        "is_weakly_connected": nx.is_weakly_connected(G),
                        "is_strongly_connected": nx.is_strongly_connected(G),
                        "num_weakly_connected_components": nx.number_weakly_connected_components(
                            G
                        ),
                        "num_strongly_connected_components": nx.number_strongly_connected_components(
                            G
                        ),
                    }
                else:
                    stats["connectivity"] = {
                        "is_connected": nx.is_connected(G),
                        "num_connected_components": nx.number_connected_components(G),
                    }

                # Additional metrics for undirected graphs
                if not G.is_directed() and G.number_of_nodes() > 0:
                    stats["clustering"] = {
                        "average_clustering": nx.average_clustering(G),
                        "transitivity": nx.transitivity(G),
                    }

                    if nx.is_connected(G):
                        stats["distance"] = {
                            "diameter": nx.diameter(G),
                            "radius": nx.radius(G),
                            "average_shortest_path_length": nx.average_shortest_path_length(
                                G
                            ),
                        }

                return stats

            except Exception as e:
                return {"error": f"Statistics calculation failed: {str(e)}"}

        @self.mcp.tool()
        async def community_detection(
            graph_id: str,
            method: str = "louvain",
            resolution: float = 1.0,
            k: Optional[int] = None,
        ) -> dict[str, Any]:
            """Detect communities in a graph.

            Args:
                graph_id: ID of the graph
                method: Detection method ('louvain', 'label_propagation', 'girvan_newman', 'spectral')
                resolution: Resolution parameter for Louvain method
                k: Number of communities for spectral clustering

            Returns:
                Dict with communities and modularity score
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                if method == "louvain":
                    # Import community detection library
                    try:
                        import community as community_louvain

                        partition = community_louvain.best_partition(
                            G, resolution=resolution
                        )
                        communities = {}
                        for node, comm_id in partition.items():
                            if comm_id not in communities:
                                communities[comm_id] = []
                            communities[comm_id].append(node)
                        communities = list(communities.values())

                        # Calculate modularity
                        modularity = community_louvain.modularity(partition, G)

                    except ImportError:
                        # Fallback to greedy modularity
                        from networkx.algorithms.community import (
                            greedy_modularity_communities,
                        )

                        communities = list(greedy_modularity_communities(G))
                        communities = [list(c) for c in communities]
                        modularity = nx.community.modularity(G, communities)

                elif method == "label_propagation":
                    from networkx.algorithms.community import (
                        label_propagation_communities,
                    )

                    communities = list(label_propagation_communities(G))
                    communities = [list(c) for c in communities]
                    modularity = nx.community.modularity(G, communities)

                elif method == "girvan_newman":
                    from networkx.algorithms.community import girvan_newman

                    # Get first k communities
                    comp = girvan_newman(G)
                    for _ in range(k or 2):
                        communities = next(comp)
                    communities = [list(c) for c in communities]
                    modularity = nx.community.modularity(G, communities)

                elif method == "spectral":
                    try:
                        from sklearn.cluster import SpectralClustering
                        import numpy as np

                        # Create adjacency matrix
                        adj_matrix = nx.adjacency_matrix(G)
                        n_clusters = k or min(5, G.number_of_nodes() // 10)

                        clustering = SpectralClustering(
                            n_clusters=n_clusters,
                            affinity="precomputed",
                            random_state=42,
                        )
                        labels = clustering.fit_predict(adj_matrix)

                        # Convert to communities
                        communities = {}
                        nodes = list(G.nodes())
                        for i, label in enumerate(labels):
                            if label not in communities:
                                communities[label] = []
                            communities[label].append(nodes[i])
                        communities = list(communities.values())

                        # Calculate modularity
                        modularity = nx.community.modularity(G, communities)

                    except ImportError:
                        return {"error": "Spectral clustering requires scikit-learn"}

                else:
                    return {"error": f"Unknown method: {method}"}

                return {
                    "method": method,
                    "num_communities": len(communities),
                    "modularity": float(modularity),
                    "communities": communities[:20],  # First 20 communities
                    "community_sizes": [len(c) for c in communities],
                }

            except Exception as e:
                return {"error": f"Community detection failed: {str(e)}"}

        @self.mcp.tool()
        async def bipartite_analysis(
            graph_id: str,
        ) -> dict[str, Any]:
            """Analyze bipartite properties of a graph.

            Args:
                graph_id: ID of the graph

            Returns:
                Dict with bipartite analysis results
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                # Check if bipartite
                is_bipartite_result = bipartite.is_bipartite(G)

                if not is_bipartite_result:
                    return {
                        "is_bipartite": False,
                        "reason": "Graph contains odd-length cycles",
                    }

                # Get node sets
                node_sets = bipartite.sets(G)
                set1, set2 = list(node_sets[0]), list(node_sets[1])

                # Calculate bipartite-specific metrics
                density = bipartite.density(G, set1)

                # Degree statistics
                degrees_set1 = bipartite.degrees(G, set1)[0]
                degrees_set2 = bipartite.degrees(G, set2)[1]

                result = {
                    "is_bipartite": True,
                    "set1_size": len(set1),
                    "set2_size": len(set2),
                    "set1_nodes": set1[:20],  # First 20 nodes
                    "set2_nodes": set2[:20],
                    "density": density,
                    "set1_avg_degree": sum(degrees_set1.values()) / len(set1)
                    if set1
                    else 0,
                    "set2_avg_degree": sum(degrees_set2.values()) / len(set2)
                    if set2
                    else 0,
                }

                # Projection analysis
                if len(set1) <= 1000 and len(set2) <= 1000:  # Limit for performance
                    proj1 = bipartite.projected_graph(G, set1)
                    proj2 = bipartite.projected_graph(G, set2)

                    result["projections"] = {
                        "set1_projection": {
                            "nodes": proj1.number_of_nodes(),
                            "edges": proj1.number_of_edges(),
                            "density": nx.density(proj1),
                        },
                        "set2_projection": {
                            "nodes": proj2.number_of_nodes(),
                            "edges": proj2.number_of_edges(),
                            "density": nx.density(proj2),
                        },
                    }

                return result

            except Exception as e:
                return {"error": f"Bipartite analysis failed: {str(e)}"}

        @self.mcp.tool()
        async def degree_distribution(
            graph_id: str, log_scale: bool = False
        ) -> dict[str, Any]:
            """Analyze degree distribution of a graph.

            Args:
                graph_id: ID of the graph
                log_scale: Whether to use log scale for bins

            Returns:
                Dict with degree distribution analysis
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                # Get degree sequence
                if G.is_directed():
                    in_degrees = [d for n, d in G.in_degree()]
                    out_degrees = [d for n, d in G.out_degree()]
                    degrees = in_degrees + out_degrees
                else:
                    degrees = [d for n, d in G.degree()]

                if not degrees:
                    return {"error": "Graph has no nodes"}

                # Calculate distribution
                degree_hist = nx.degree_histogram(G)

                # Statistics
                import numpy as np

                stats = {
                    "mean": np.mean(degrees),
                    "std": np.std(degrees),
                    "min": min(degrees),
                    "max": max(degrees),
                    "median": np.median(degrees),
                }

                # Check for power law
                if len(set(degrees)) > 1:
                    # Simple power law check
                    unique_degrees = sorted(set(degrees))
                    if len(unique_degrees) > 2:
                        # Rough estimate of power law exponent
                        try:
                            from scipy import stats as scipy_stats

                            x = np.array(unique_degrees[1:])  # Exclude 0
                            y = np.array([degrees.count(d) for d in unique_degrees[1:]])
                            if len(x) > 2 and np.all(y > 0):
                                (
                                    slope,
                                    intercept,
                                    r_value,
                                    p_value,
                                    std_err,
                                ) = scipy_stats.linregress(np.log(x), np.log(y))
                                power_law_exponent = -slope
                                r_squared = r_value**2
                            else:
                                power_law_exponent = None
                                r_squared = None
                        except:
                            power_law_exponent = None
                            r_squared = None
                    else:
                        power_law_exponent = None
                        r_squared = None
                else:
                    power_law_exponent = None
                    r_squared = None

                result = {
                    "histogram": degree_hist[:50],  # First 50 degrees
                    "statistics": stats,
                    "num_unique_degrees": len(set(degrees)),
                }

                if power_law_exponent is not None:
                    result["power_law"] = {
                        "estimated_exponent": power_law_exponent,
                        "r_squared": r_squared,
                        "is_scale_free": r_squared > 0.8 if r_squared else False,
                    }

                if G.is_directed():
                    result["directed"] = {
                        "in_degree_avg": np.mean(in_degrees),
                        "out_degree_avg": np.mean(out_degrees),
                        "in_degree_max": max(in_degrees),
                        "out_degree_max": max(out_degrees),
                    }

                return result

            except Exception as e:
                return {"error": f"Degree distribution analysis failed: {str(e)}"}

        @self.mcp.tool()
        async def node_classification_features(
            graph_id: str, feature_types: Optional[list[str]] = None
        ) -> dict[str, Any]:
            """Extract features for node classification tasks.

            Args:
                graph_id: ID of the graph
                feature_types: List of feature types to extract

            Returns:
                Dict with node features
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            if feature_types is None:
                feature_types = ["degree", "clustering", "centrality"]

            try:
                features = {}
                nodes = list(G.nodes())

                # Degree features
                if "degree" in feature_types:
                    degrees = dict(G.degree())
                    features["degree"] = {str(n): degrees[n] for n in nodes[:100]}

                # Clustering coefficient
                if "clustering" in feature_types and not G.is_directed():
                    clustering = nx.clustering(G)
                    features["clustering"] = {
                        str(n): clustering[n] for n in nodes[:100]
                    }

                # Centrality features
                if "centrality" in feature_types:
                    if G.number_of_nodes() <= 1000:  # Limit for performance
                        betweenness = nx.betweenness_centrality(G)
                        closeness = nx.closeness_centrality(G)
                        features["betweenness_centrality"] = {
                            str(n): betweenness[n] for n in nodes[:100]
                        }
                        features["closeness_centrality"] = {
                            str(n): closeness[n] for n in nodes[:100]
                        }

                # Node attributes
                if G.number_of_nodes() > 0:
                    sample_node = list(G.nodes())[0]
                    node_attrs = G.nodes[sample_node]
                    if node_attrs:
                        features["available_attributes"] = list(node_attrs.keys())

                return {
                    "num_nodes": G.number_of_nodes(),
                    "feature_types": list(features.keys()),
                    "features": features,
                    "sample_size": min(100, G.number_of_nodes()),
                }

            except Exception as e:
                return {"error": f"Feature extraction failed: {str(e)}"}

        @self.mcp.tool()
        async def assortativity_analysis(
            graph_id: str, attribute: Optional[str] = None
        ) -> dict[str, Any]:
            """Analyze assortativity patterns in the graph.

            Args:
                graph_id: ID of the graph
                attribute: Node attribute for attribute assortativity

            Returns:
                Dict with assortativity measures
            """
            G = self.graph_manager.get_graph(graph_id)
            if not G:
                return {"error": f"Graph '{graph_id}' not found"}

            try:
                result = {}

                # Degree assortativity
                result["degree_assortativity"] = nx.degree_assortativity_coefficient(G)

                # Attribute assortativity
                if attribute:
                    # Check if attribute exists
                    if G.number_of_nodes() > 0:
                        sample_node = list(G.nodes())[0]
                        if attribute in G.nodes[sample_node]:
                            # Check if numeric or categorical
                            attr_values = [G.nodes[n].get(attribute) for n in G.nodes()]
                            if all(
                                isinstance(v, (int, float))
                                for v in attr_values
                                if v is not None
                            ):
                                result[
                                    "numeric_assortativity"
                                ] = nx.numeric_assortativity_coefficient(G, attribute)
                            else:
                                result[
                                    "attribute_assortativity"
                                ] = nx.attribute_assortativity_coefficient(G, attribute)

                # Interpretation
                deg_assort = result.get("degree_assortativity", 0)
                if deg_assort > 0.3:
                    result[
                        "interpretation"
                    ] = "Assortative: High-degree nodes tend to connect to high-degree nodes"
                elif deg_assort < -0.3:
                    result[
                        "interpretation"
                    ] = "Disassortative: High-degree nodes tend to connect to low-degree nodes"
                else:
                    result[
                        "interpretation"
                    ] = "Neutral: No strong degree correlation pattern"

                return result

            except Exception as e:
                return {"error": f"Assortativity analysis failed: {str(e)}"}


__all__ = ["AnalysisHandler"]
