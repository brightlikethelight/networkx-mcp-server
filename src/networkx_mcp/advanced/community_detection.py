"""Advanced community detection algorithms and analysis."""

import logging
import time
import warnings

from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import networkx as nx
import numpy as np


# Suppress warnings for optional dependencies
warnings.filterwarnings("ignore", category=ImportWarning)

logger = logging.getLogger(__name__)


class CommunityDetection:
    """Advanced community detection algorithms and quality metrics."""

    @staticmethod
    def detect_communities(
        graph: nx.Graph,
        algorithm: str = "auto",
        **params
    ) -> Dict[str, Any]:
        """
        Detect communities using various algorithms.

        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        algorithm : str
            Algorithm to use: 'louvain', 'girvan_newman', 'label_propagation',
            'modularity', 'spectral', 'auto' (default: 'auto')
        **params : dict
            Algorithm-specific parameters

        Returns:
        --------
        Dict containing:
            - communities: List of sets of nodes
            - modularity: Modularity score
            - algorithm_used: Algorithm name
            - num_communities: Number of communities
            - execution_time_ms: Execution time
            - parameters: Parameters used
        """
        start_time = time.time()

        # Auto-select algorithm based on graph size
        if algorithm == "auto":
            num_nodes = graph.number_of_nodes()
            graph.number_of_edges()

            if num_nodes < 100:
                algorithm = "girvan_newman"  # Most accurate for small graphs
            elif num_nodes < 10000:
                algorithm = "louvain"  # Good balance
            else:
                algorithm = "label_propagation"  # Fastest for large graphs

            logger.info(f"Auto-selected {algorithm} for graph with {num_nodes} nodes")

        # Execute selected algorithm
        if algorithm == "louvain":
            communities = CommunityDetection._louvain_method(graph, **params)
        elif algorithm == "girvan_newman":
            communities = CommunityDetection._girvan_newman(graph, **params)
        elif algorithm == "label_propagation":
            communities = CommunityDetection._label_propagation(graph, **params)
        elif algorithm == "modularity":
            communities = CommunityDetection._modularity_optimization(graph, **params)
        elif algorithm == "spectral":
            communities = CommunityDetection._spectral_clustering(graph, **params)
        else:
            msg = f"Unknown algorithm: {algorithm}"
            raise ValueError(msg)

        # Calculate modularity
        modularity = nx.algorithms.community.modularity(graph, communities)

        execution_time = (time.time() - start_time) * 1000

        return {
            "communities": [list(community) for community in communities],
            "modularity": modularity,
            "algorithm_used": algorithm,
            "num_communities": len(communities),
            "execution_time_ms": execution_time,
            "parameters": params,
            "community_sizes": sorted([len(c) for c in communities], reverse=True)
        }

    @staticmethod
    def _louvain_method(graph: nx.Graph, resolution: float = 1.0, **kwargs) -> List[Set]:
        """Louvain community detection with resolution parameter."""
        try:
            # Try to use python-louvain if available
            import community as community_louvain
            partition = community_louvain.best_partition(
                graph,
                resolution=resolution,
                randomize=kwargs.get("randomize", True),
                random_state=kwargs.get("seed", None)
            )
            # Convert partition dict to list of sets
            communities = defaultdict(set)
            for node, comm_id in partition.items():
                communities[comm_id].add(node)
            return list(communities.values())
        except ImportError:
            # Fallback to NetworkX greedy modularity
            logger.warning("python-louvain not available, using greedy modularity")
            return list(nx.algorithms.community.greedy_modularity_communities(
                graph,
                resolution=resolution
            ))

    @staticmethod
    def _girvan_newman(graph: nx.Graph, num_communities: Optional[int] = None, **kwargs) -> List[Set]:
        """Girvan-Newman edge betweenness community detection."""
        # Create a copy to avoid modifying original
        G = graph.copy()

        if num_communities is None:
            # Use modularity to determine optimal number
            best_modularity = -1
            best_communities = []

            comp = nx.algorithms.community.girvan_newman(G)
            for communities in comp:
                mod = nx.algorithms.community.modularity(graph, communities)
                if mod > best_modularity:
                    best_modularity = mod
                    best_communities = list(communities)
                elif mod < best_modularity * 0.95:  # Stop when modularity drops
                    break

                if len(communities) >= graph.number_of_nodes() / 2:
                    break  # Avoid too many communities

            return best_communities
        else:
            # Get specific number of communities
            comp = nx.algorithms.community.girvan_newman(G)
            for communities in comp:
                if len(communities) >= num_communities:
                    return list(communities)
            return list(communities)  # Return last if not enough splits

    @staticmethod
    def _label_propagation(graph: nx.Graph, max_iterations: int = 100, **kwargs) -> List[Set]:
        """Asynchronous label propagation algorithm."""
        communities = nx.algorithms.community.asyn_lpa_communities(
            graph,
            weight=kwargs.get("weight", None),
            seed=kwargs.get("seed", None)
        )
        return list(communities)

    @staticmethod
    def _modularity_optimization(graph: nx.Graph, **kwargs) -> List[Set]:
        """Modularity-based optimization using greedy approach."""
        return list(nx.algorithms.community.greedy_modularity_communities(
            graph,
            weight=kwargs.get("weight", None),
            resolution=kwargs.get("resolution", 1.0),
            n_communities=kwargs.get("n_communities", None)
        ))

    @staticmethod
    def _spectral_clustering(graph: nx.Graph, num_communities: Optional[int] = None, **kwargs) -> List[Set]:
        """Spectral clustering for community detection."""
        try:
            from scipy.sparse import csr_matrix
            from sklearn.cluster import SpectralClustering

            # Convert to adjacency matrix
            adj_matrix = nx.adjacency_matrix(graph)

            # Estimate number of communities if not provided
            if num_communities is None:
                # Use eigengap heuristic
                eigenvalues = np.linalg.eigvalsh(adj_matrix.todense())
                gaps = np.diff(sorted(eigenvalues))
                num_communities = np.argmax(gaps) + 2  # +2 because of diff and 0-indexing
                num_communities = min(num_communities, graph.number_of_nodes() // 10)
                num_communities = max(2, num_communities)

            # Perform spectral clustering
            clustering = SpectralClustering(
                n_clusters=num_communities,
                affinity="precomputed",
                random_state=kwargs.get("seed", None)
            )
            labels = clustering.fit_predict(adj_matrix)

            # Convert labels to communities
            communities = defaultdict(set)
            nodes = list(graph.nodes())
            for i, label in enumerate(labels):
                communities[label].add(nodes[i])

            return list(communities.values())

        except ImportError:
            logger.warning("scikit-learn not available for spectral clustering")
            # Fallback to modularity optimization
            return CommunityDetection._modularity_optimization(graph, **kwargs)

    @staticmethod
    def community_quality(
        graph: nx.Graph,
        communities: List[List[Any]]
    ) -> Dict[str, Any]:
        """
        Calculate various quality metrics for communities.

        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        communities : List[List[Any]]
            List of communities (each community is a list of nodes)

        Returns:
        --------
        Dict containing various quality metrics
        """
        # Convert to sets for efficiency
        comm_sets = [set(c) for c in communities]

        # Basic metrics
        modularity = nx.algorithms.community.modularity(graph, comm_sets)

        # Coverage: fraction of edges within communities
        intra_edges = 0
        total_edges = graph.number_of_edges()

        for community in comm_sets:
            subgraph = graph.subgraph(community)
            intra_edges += subgraph.number_of_edges()

        coverage = intra_edges / total_edges if total_edges > 0 else 0

        # Performance: fraction of correctly classified pairs
        n = graph.number_of_nodes()
        correct_pairs = 0

        # Intra-community pairs that are connected
        for community in comm_sets:
            subgraph = graph.subgraph(community)
            correct_pairs += subgraph.number_of_edges()

        # Inter-community pairs that are not connected
        for i, comm1 in enumerate(comm_sets):
            for _j, comm2 in enumerate(comm_sets[i+1:], i+1):
                for node1 in comm1:
                    for node2 in comm2:
                        if not graph.has_edge(node1, node2):
                            correct_pairs += 1

        total_pairs = n * (n - 1) // 2
        performance = correct_pairs / total_pairs if total_pairs > 0 else 0

        # Conductance for each community
        conductances = []
        for community in comm_sets:
            if len(community) == 0 or len(community) == n:
                continue

            # Edges leaving the community
            cut_size = 0
            # Volume of community
            volume = 0

            for node in community:
                for neighbor in graph[node]:
                    if neighbor in community:
                        volume += 1
                    else:
                        cut_size += 1
                        volume += 1

            # Volume of complement
            comp_volume = 2 * total_edges - volume

            if volume > 0 and comp_volume > 0:
                conductance = cut_size / min(volume, comp_volume)
                conductances.append(conductance)

        avg_conductance = np.mean(conductances) if conductances else 1.0

        # Inter/intra community edge ratio
        inter_edges = 0
        for node in graph:
            node_comm = None
            for i, comm in enumerate(comm_sets):
                if node in comm:
                    node_comm = i
                    break

            if node_comm is not None:
                for neighbor in graph[node]:
                    neighbor_comm = None
                    for i, comm in enumerate(comm_sets):
                        if neighbor in comm:
                            neighbor_comm = i
                            break

                    if neighbor_comm is not None and node_comm != neighbor_comm:
                        inter_edges += 1

        inter_edges //= 2  # Each edge counted twice
        intra_edges = total_edges - inter_edges
        edge_ratio = inter_edges / intra_edges if intra_edges > 0 else float("inf")

        # Community sizes statistics
        sizes = [len(c) for c in comm_sets]

        return {
            "modularity": modularity,
            "coverage": coverage,
            "performance": performance,
            "average_conductance": avg_conductance,
            "inter_intra_edge_ratio": edge_ratio,
            "num_communities": len(comm_sets),
            "community_sizes": {
                "min": min(sizes) if sizes else 0,
                "max": max(sizes) if sizes else 0,
                "mean": np.mean(sizes) if sizes else 0,
                "std": np.std(sizes) if sizes else 0
            },
            "singleton_communities": sum(1 for s in sizes if s == 1),
            "largest_community_fraction": max(sizes) / n if sizes and n > 0 else 0
        }

    @staticmethod
    def hierarchical_communities(
        graph: nx.Graph,
        method: str = "louvain",
        max_levels: int = 5,
        resolution_range: Tuple[float, float] = (0.1, 2.0),
        **params
    ) -> Dict[str, Any]:
        """
        Generate hierarchical community structure.

        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        method : str
            Method to use for hierarchy ('louvain' or 'divisive')
        max_levels : int
            Maximum hierarchy levels
        resolution_range : Tuple[float, float]
            Range of resolution parameters for Louvain

        Returns:
        --------
        Dict containing:
            - hierarchy: Nested structure of communities
            - dendrogram: Dendrogram data
            - levels: Community assignments at each level
        """
        if method == "louvain":
            return CommunityDetection._louvain_hierarchy(
                graph, max_levels, resolution_range, **params
            )
        elif method == "divisive":
            return CommunityDetection._divisive_hierarchy(
                graph, max_levels, **params
            )
        else:
            msg = f"Unknown hierarchical method: {method}"
            raise ValueError(msg)

    @staticmethod
    def _louvain_hierarchy(
        graph: nx.Graph,
        max_levels: int,
        resolution_range: Tuple[float, float],
        **params
    ) -> Dict[str, Any]:
        """Generate hierarchy by varying Louvain resolution."""
        levels = []
        resolutions = np.linspace(
            resolution_range[0],
            resolution_range[1],
            max_levels
        )

        hierarchy = {"id": "root", "children": []}
        dendrogram_data = []

        for level, resolution in enumerate(resolutions):
            communities = CommunityDetection._louvain_method(
                graph, resolution=resolution, **params
            )

            level_data = {
                "level": level,
                "resolution": resolution,
                "communities": [list(c) for c in communities],
                "num_communities": len(communities),
                "modularity": nx.algorithms.community.modularity(graph, communities)
            }
            levels.append(level_data)

            # Build hierarchy tree
            if level == 0:
                # First level: direct children of root
                for i, comm in enumerate(communities):
                    child = {
                        "id": f"L{level}_C{i}",
                        "level": level,
                        "nodes": list(comm),
                        "size": len(comm),
                        "children": []
                    }
                    hierarchy["children"].append(child)
            else:
                # Subsequent levels: map to previous level
                levels[level-1]["communities"]

                # Create mapping from node to community at this level
                node_to_comm = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        node_to_comm[node] = i

                # Update hierarchy
                for prev_child in hierarchy["children"]:
                    _update_hierarchy_recursive(
                        prev_child, node_to_comm, level, communities
                    )

            # Add dendrogram data
            dendrogram_data.append({
                "level": level,
                "height": resolution,
                "num_clusters": len(communities),
                "merges": []  # Would need more complex tracking for full dendrogram
            })

        return {
            "hierarchy": hierarchy,
            "dendrogram": dendrogram_data,
            "levels": levels,
            "method": "louvain_multiresolution"
        }

    @staticmethod
    def _divisive_hierarchy(
        graph: nx.Graph,
        max_levels: int,
        **params
    ) -> Dict[str, Any]:
        """Generate hierarchy using divisive approach (Girvan-Newman)."""
        levels = []
        hierarchy = {"id": "root", "children": []}
        dendrogram_data = []

        # Use Girvan-Newman generator
        gn_generator = nx.algorithms.community.girvan_newman(graph.copy())

        for level in range(max_levels):
            try:
                communities = next(gn_generator)

                level_data = {
                    "level": level,
                    "communities": [list(c) for c in communities],
                    "num_communities": len(communities),
                    "modularity": nx.algorithms.community.modularity(graph, communities)
                }
                levels.append(level_data)

                # Stop if we have one community per node
                if len(communities) >= graph.number_of_nodes() / 2:
                    break

            except StopIteration:
                break

        # Build hierarchy from levels
        # This is simplified - full implementation would track actual splits
        for level_data in levels:
            level = level_data["level"]
            if level == 0:
                for i, comm in enumerate(level_data["communities"]):
                    child = {
                        "id": f"L{level}_C{i}",
                        "level": level,
                        "nodes": comm,
                        "size": len(comm),
                        "children": []
                    }
                    hierarchy["children"].append(child)

        return {
            "hierarchy": hierarchy,
            "dendrogram": dendrogram_data,
            "levels": levels,
            "method": "girvan_newman_divisive"
        }

    @staticmethod
    def community_comparison(
        graph: nx.Graph,
        algorithms: Optional[List[str]] = None,
        **params
    ) -> Dict[str, Any]:
        """
        Compare different community detection algorithms.

        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        algorithms : List[str]
            Algorithms to compare (default: all available)

        Returns:
        --------
        Dict containing comparison metrics and results
        """
        if algorithms is None:
            algorithms = ["louvain", "label_propagation", "modularity"]
            if graph.number_of_nodes() < 500:
                algorithms.append("girvan_newman")
            if graph.number_of_nodes() < 1000:
                algorithms.append("spectral")

        results = {}
        all_communities = {}

        # Run each algorithm
        for algo in algorithms:
            try:
                result = CommunityDetection.detect_communities(
                    graph, algorithm=algo, **params.get(algo, {})
                )
                results[algo] = result
                all_communities[algo] = [set(c) for c in result["communities"]]
            except Exception as e:
                logger.warning(f"Algorithm {algo} failed: {e}")
                continue

        # Compare all pairs of algorithms
        comparisons = {}
        algo_list = list(all_communities.keys())

        for i, algo1 in enumerate(algo_list):
            for algo2 in algo_list[i+1:]:
                pair_key = f"{algo1}_vs_{algo2}"

                # Calculate similarity metrics
                ari = CommunityDetection._adjusted_rand_index(
                    all_communities[algo1],
                    all_communities[algo2],
                    list(graph.nodes())
                )

                nmi = CommunityDetection._normalized_mutual_info(
                    all_communities[algo1],
                    all_communities[algo2],
                    list(graph.nodes())
                )

                comparisons[pair_key] = {
                    "adjusted_rand_index": ari,
                    "normalized_mutual_info": nmi,
                    "modularity_diff": abs(
                        results[algo1]["modularity"] - results[algo2]["modularity"]
                    ),
                    "num_communities_diff": abs(
                        results[algo1]["num_communities"] - results[algo2]["num_communities"]
                    )
                }

        # Stability analysis - run each algorithm multiple times
        stability = {}
        num_runs = params.get("stability_runs", 5)

        for algo in ["louvain", "label_propagation"]:  # Only non-deterministic algorithms
            if algo not in algorithms:
                continue

            modularities = []
            community_counts = []

            for run in range(num_runs):
                try:
                    result = CommunityDetection.detect_communities(
                        graph,
                        algorithm=algo,
                        seed=run,
                        **params.get(algo, {})
                    )
                    modularities.append(result["modularity"])
                    community_counts.append(result["num_communities"])
                except:
                    continue

            if modularities:
                stability[algo] = {
                    "modularity_mean": np.mean(modularities),
                    "modularity_std": np.std(modularities),
                    "num_communities_mean": np.mean(community_counts),
                    "num_communities_std": np.std(community_counts)
                }

        # Summary statistics
        summary = {
            "best_modularity": max(
                results.values(),
                key=lambda x: x["modularity"]
            )["algorithm_used"],
            "fastest_algorithm": min(
                results.values(),
                key=lambda x: x["execution_time_ms"]
            )["algorithm_used"],
            "consensus_communities": CommunityDetection._find_consensus_communities(
                all_communities
            )
        }

        return {
            "algorithm_results": results,
            "pairwise_comparisons": comparisons,
            "stability_analysis": stability,
            "summary": summary,
            "algorithms_compared": list(results.keys())
        }

    @staticmethod
    def _adjusted_rand_index(
        communities1: List[Set],
        communities2: List[Set],
        nodes: List[Any]
    ) -> float:
        """Calculate Adjusted Rand Index between two partitions."""
        # Create label arrays
        labels1 = np.zeros(len(nodes))
        labels2 = np.zeros(len(nodes))
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        for i, comm in enumerate(communities1):
            for node in comm:
                if node in node_to_idx:
                    labels1[node_to_idx[node]] = i

        for i, comm in enumerate(communities2):
            for node in comm:
                if node in node_to_idx:
                    labels2[node_to_idx[node]] = i

        # Calculate contingency matrix
        n = len(nodes)
        n_classes1 = len(communities1)
        n_classes2 = len(communities2)
        contingency = np.zeros((n_classes1, n_classes2))

        for i in range(n):
            contingency[int(labels1[i]), int(labels2[i])] += 1

        # Calculate ARI
        sum_squares_total = np.sum(contingency ** 2)
        sum_squares_row = np.sum(np.sum(contingency, axis=1) ** 2)
        sum_squares_col = np.sum(np.sum(contingency, axis=0) ** 2)

        n_pairs = n * (n - 1) / 2
        term1 = sum_squares_total - n
        term2 = (sum_squares_row - n) * (sum_squares_col - n) / (2 * n_pairs)
        term3 = 0.5 * (sum_squares_row + sum_squares_col) - n

        if term3 == term2:
            return 1.0
        else:
            return (term1 - term2) / (term3 - term2)

    @staticmethod
    def _normalized_mutual_info(
        communities1: List[Set],
        communities2: List[Set],
        nodes: List[Any]
    ) -> float:
        """Calculate Normalized Mutual Information between two partitions."""
        # Create label arrays
        labels1 = np.zeros(len(nodes))
        labels2 = np.zeros(len(nodes))
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        for i, comm in enumerate(communities1):
            for node in comm:
                if node in node_to_idx:
                    labels1[node_to_idx[node]] = i

        for i, comm in enumerate(communities2):
            for node in comm:
                if node in node_to_idx:
                    labels2[node_to_idx[node]] = i

        # Calculate entropy and mutual information
        n = len(nodes)

        # Entropy of partition 1
        h1 = 0
        for comm in communities1:
            if len(comm) > 0:
                p = len(comm) / n
                h1 -= p * np.log2(p)

        # Entropy of partition 2
        h2 = 0
        for comm in communities2:
            if len(comm) > 0:
                p = len(comm) / n
                h2 -= p * np.log2(p)

        # Mutual information
        mi = 0
        for i, comm1 in enumerate(communities1):
            for _j, comm2 in enumerate(communities2):
                intersection = len(comm1 & comm2)
                if intersection > 0:
                    p_ij = intersection / n
                    p_i = len(comm1) / n
                    p_j = len(comm2) / n
                    mi += p_ij * np.log2(p_ij / (p_i * p_j))

        # Normalized MI
        if h1 == 0 and h2 == 0:
            return 1.0
        elif h1 == 0 or h2 == 0:
            return 0.0
        else:
            return 2 * mi / (h1 + h2)

    @staticmethod
    def _find_consensus_communities(
        all_communities: Dict[str, List[Set]]
    ) -> List[List[Any]]:
        """Find consensus communities across multiple algorithms."""
        if not all_communities:
            return []

        # Get all nodes
        all_nodes = set()
        for communities in all_communities.values():
            for comm in communities:
                all_nodes.update(comm)

        # Create co-occurrence matrix
        nodes_list = list(all_nodes)
        n = len(nodes_list)
        node_to_idx = {node: i for i, node in enumerate(nodes_list)}

        co_occurrence = np.zeros((n, n))

        for communities in all_communities.values():
            for comm in communities:
                comm_list = list(comm)
                for i, node1 in enumerate(comm_list):
                    for node2 in comm_list[i:]:
                        idx1 = node_to_idx[node1]
                        idx2 = node_to_idx[node2]
                        co_occurrence[idx1, idx2] += 1
                        co_occurrence[idx2, idx1] += 1

        # Normalize by number of algorithms
        co_occurrence /= len(all_communities)

        # Find consensus communities (nodes that are together > 50% of the time)
        consensus_communities = []
        visited = set()

        for i, node in enumerate(nodes_list):
            if node in visited:
                continue

            community = {node}
            visited.add(node)

            for j, other_node in enumerate(nodes_list):
                if i != j and other_node not in visited:
                    if co_occurrence[i, j] > 0.5:
                        community.add(other_node)
                        visited.add(other_node)

            consensus_communities.append(list(community))

        return consensus_communities


def _update_hierarchy_recursive(
    node: Dict,
    node_to_comm: Dict,
    level: int,
    communities: List[Set]
) -> None:
    """Helper function to update hierarchy tree recursively."""
    if node.get("nodes"):
        # Group nodes by their new community
        new_groups = defaultdict(list)
        for n in node["nodes"]:
            if n in node_to_comm:
                new_groups[node_to_comm[n]].append(n)

        # Create new children if this node splits
        if len(new_groups) > 1:
            node["children"] = []
            for comm_id, comm_nodes in new_groups.items():
                child = {
                    "id": f"L{level}_C{comm_id}_{node['id']}",
                    "level": level,
                    "nodes": comm_nodes,
                    "size": len(comm_nodes),
                    "children": []
                }
                node["children"].append(child)

    # Recurse on children
    for child in node.get("children", []):
        _update_hierarchy_recursive(child, node_to_comm, level, communities)
