"""Machine learning integrations for graph analysis."""

import logging
import random  # Using for non-cryptographic ML training purposes only
import time

from collections import defaultdict
from itertools import combinations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import networkx as nx
import numpy as np

from sklearn.preprocessing import StandardScaler


# Performance thresholds and constants
MAX_NODES_FOR_EXPENSIVE_COMPUTATION = 1000
MAX_NODES_FOR_SPECTRAL_FEATURES = 5000
MILLISECONDS_PER_SECOND = 1000
DEFAULT_EMBEDDING_DIMENSIONS = 2
DEFAULT_WALK_LENGTH = 10
DEFAULT_NUM_WALKS = 80
DEFAULT_WINDOW_SIZE = 5

# Optional imports - not all may be available
try:
    from sklearn.decomposition import PCA
    HAS_PCA = True
except ImportError:
    HAS_PCA = False
    PCA = None

try:
    from scipy.sparse.linalg import eigs
    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = False
    eigs = None


logger = logging.getLogger(__name__)


class MLIntegration:
    """Machine learning integrations for graph analysis."""

    @staticmethod
    def node_embeddings(
        graph: Union[nx.Graph, nx.DiGraph],
        method: str = "node2vec",
        dimensions: int = 128,
        **params
    ) -> Dict[str, Any]:
        """
        Generate node embeddings for machine learning.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            Input graph
        method : str
            Method: 'node2vec', 'deepwalk', 'spectral', 'structural'
        dimensions : int
            Embedding dimensions

        Returns:
        --------
        Dict containing embeddings and metadata
        """
        start_time = time.time()

        if method == "node2vec":
            embeddings = MLIntegration._node2vec_embeddings(
                graph, dimensions, **params
            )

        elif method == "deepwalk":
            embeddings = MLIntegration._deepwalk_embeddings(
                graph, dimensions, **params
            )

        elif method == "spectral":
            embeddings = MLIntegration._spectral_embeddings(
                graph, dimensions, **params
            )

        elif method == "structural":
            embeddings = MLIntegration._structural_embeddings(
                graph, dimensions, **params
            )

        else:
            msg = f"Unknown embedding method: {method}"
            raise ValueError(msg)

        # Calculate embedding statistics
        embedding_matrix = np.array(list(embeddings.values()))

        results = {
            "embeddings": embeddings,
            "method": method,
            "dimensions": dimensions,
            "num_nodes": len(embeddings),
            "embedding_stats": {
                "mean": float(np.mean(embedding_matrix)),
                "std": float(np.std(embedding_matrix)),
                "min": float(np.min(embedding_matrix)),
                "max": float(np.max(embedding_matrix)),
                "sparsity": float(np.sum(embedding_matrix == 0) / embedding_matrix.size)
            },
            "parameters": params,
            "execution_time_ms": (time.time() - start_time) * MILLISECONDS_PER_SECOND
        }

        # Sample embeddings for inspection
        sample_nodes = list(embeddings.keys())[:5]
        results["sample_embeddings"] = {
            node: embeddings[node].tolist()[:DEFAULT_WALK_LENGTH]  # First dimensions
            for node in sample_nodes
        }

        return results

    @staticmethod
    def _node2vec_embeddings(
        graph: Union[nx.Graph, nx.DiGraph],
        dimensions: int,
        walk_length: int = DEFAULT_NUM_WALKS,
        num_walks: int = DEFAULT_WALK_LENGTH,
        p: float = 1.0,
        q: float = 1.0,
        **params
    ) -> Dict[Any, np.ndarray]:
        """Generate Node2Vec embeddings (simplified version)."""
        # Generate biased random walks
        walks = MLIntegration._generate_node2vec_walks(
            graph, walk_length, num_walks, p, q
        )

        # Create co-occurrence matrix from walks
        vocab = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(vocab)}

        # Skip-gram style context
        window_size = params.get("window_size", DEFAULT_WALK_LENGTH)
        co_occurrence = defaultdict(lambda: defaultdict(int))

        for walk in walks:
            for i, node in enumerate(walk):
                for j in range(max(0, i - window_size), min(len(walk), i + window_size + 1)):
                    if i != j:
                        co_occurrence[node][walk[j]] += 1

        # Convert to matrix
        n = len(vocab)
        matrix = np.zeros((n, n))

        for node, contexts in co_occurrence.items():
            i = node_to_idx[node]
            for context, count in contexts.items():
                j = node_to_idx[context]
                matrix[i, j] = count

        # Apply SVD for dimensionality reduction
        try:
            U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
            # Keep top dimensions
            embeddings = U[:, :dimensions] * np.sqrt(s[:dimensions])
        except Exception as e:
            # Fallback to random embeddings
            logger.debug(f"SVD failed for co-occurrence matrix: {e}")
            embeddings = np.random.randn(n, dimensions) * 0.1

        return {vocab[i]: embeddings[i] for i in range(n)}

    @staticmethod
    def _generate_node2vec_walks(
        graph: Union[nx.Graph, nx.DiGraph],
        walk_length: int,
        num_walks: int,
        p: float,
        q: float
    ) -> List[List]:
        """Generate biased random walks for Node2Vec."""
        walks = []

        for _ in range(num_walks):
            for start_node in graph.nodes():
                walk = [start_node]

                while len(walk) < walk_length:
                    current = walk[-1]
                    neighbors = list(graph.neighbors(current))

                    if not neighbors:
                        break

                    if len(walk) == 1:
                        # First step: uniform random
                        next_node = random.choice(neighbors)  # noqa: S311
                    else:
                        # Biased walk
                        prev = walk[-2]

                        # Calculate transition probabilities
                        probs = []
                        for neighbor in neighbors:
                            if neighbor == prev:
                                # Return parameter
                                probs.append(1.0 / p)
                            elif graph.has_edge(neighbor, prev):
                                # BFS-like parameter
                                probs.append(1.0)
                            else:
                                # DFS-like parameter
                                probs.append(1.0 / q)

                        # Normalize
                        probs = np.array(probs)
                        probs = probs / probs.sum()

                        # Sample next node
                        next_node = np.random.choice(neighbors, p=probs)

                    walk.append(next_node)

                walks.append(walk)

        return walks

    @staticmethod
    def _deepwalk_embeddings(
        graph: Union[nx.Graph, nx.DiGraph],
        dimensions: int,
        walk_length: int = DEFAULT_NUM_WALKS,
        num_walks: int = DEFAULT_WALK_LENGTH,
        **params
    ) -> Dict[Any, np.ndarray]:
        """Generate DeepWalk embeddings (uniform random walks)."""
        # DeepWalk is Node2Vec with p=1, q=1
        return MLIntegration._node2vec_embeddings(
            graph, dimensions, walk_length, num_walks, p=1.0, q=1.0, **params
        )

    @staticmethod
    def _spectral_embeddings(
        graph: Union[nx.Graph, nx.DiGraph],
        dimensions: int,
        **_params
    ) -> Dict[Any, np.ndarray]:
        """Generate spectral embeddings using graph Laplacian."""
        # Get adjacency matrix
        adj_matrix = nx.adjacency_matrix(graph)

        # Compute normalized Laplacian
        degree_matrix = np.diag(np.array(adj_matrix.sum(axis=1)).flatten())
        laplacian = degree_matrix - adj_matrix.toarray()

        # Normalize
        degree_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(degree_matrix.diagonal(), 1e-10)))
        norm_laplacian = degree_sqrt_inv @ laplacian @ degree_sqrt_inv

        # Compute eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(norm_laplacian)
            # Use smallest non-zero eigenvectors
            embeddings = eigenvectors[:, 1:dimensions+1]
        except Exception as e:
            # Fallback to random
            logger.debug(f"Eigenvalue decomposition failed: {e}")
            embeddings = np.random.randn(graph.number_of_nodes(), dimensions) * 0.1

        nodes = list(graph.nodes())
        return {nodes[i]: embeddings[i] for i in range(len(nodes))}

    @staticmethod
    def _structural_embeddings(
        graph: Union[nx.Graph, nx.DiGraph],
        dimensions: int,
        **_params
    ) -> Dict[Any, np.ndarray]:
        """Generate structural feature embeddings."""
        features = []
        nodes = list(graph.nodes())

        for node in nodes:
            node_features = []

            # Degree features
            if graph.is_directed():
                node_features.extend([
                    graph.in_degree(node),
                    graph.out_degree(node),
                    graph.in_degree(node) + graph.out_degree(node)
                ])
            else:
                node_features.append(graph.degree(node))

            # Local clustering
            try:
                node_features.append(nx.clustering(graph, node))
            except Exception as e:
                logger.debug(f"Failed to compute clustering coefficient for node: {e}")
                node_features.append(0)

            # Neighbor degrees
            neighbor_degrees = [graph.degree(n) for n in graph.neighbors(node)]
            if neighbor_degrees:
                node_features.extend([
                    np.mean(neighbor_degrees),
                    np.std(neighbor_degrees),
                    np.min(neighbor_degrees),
                    np.max(neighbor_degrees)
                ])
            else:
                node_features.extend([0, 0, 0, 0])

            # Centrality features (if graph is small)
            if graph.number_of_nodes() < MAX_NODES_FOR_EXPENSIVE_COMPUTATION:
                try:
                    # Betweenness
                    betweenness = nx.betweenness_centrality(graph, normalized=True)
                    node_features.append(betweenness.get(node, 0))

                    # Closeness
                    if nx.is_connected(graph):
                        closeness = nx.closeness_centrality(graph, normalized=True)
                        node_features.append(closeness.get(node, 0))
                    else:
                        node_features.append(0)
                except Exception as e:
                    logger.debug(f"Failed to compute centrality measures: {e}")
                    node_features.extend([0, 0])

            features.append(node_features)

        # Convert to numpy array
        features = np.array(features)

        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Reduce dimensions if needed
        if features.shape[1] > dimensions:
            # Use PCA
            if HAS_PCA:
                pca = PCA(n_components=dimensions)
                features = pca.fit_transform(features)
            else:
                # Simple truncation
                features = features[:, :dimensions]
        elif features.shape[1] < dimensions:
            # Pad with zeros
            padding = np.zeros((features.shape[0], dimensions - features.shape[1]))
            features = np.hstack([features, padding])

        return {nodes[i]: features[i] for i in range(len(nodes))}

    @staticmethod
    def graph_features(
        graph: Union[nx.Graph, nx.DiGraph],
        feature_types: Optional[List[str]] = None,
        **_params
    ) -> Dict[str, Any]:
        """
        Extract features for machine learning from graph.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            Input graph
        feature_types : List[str]
            Types of features to extract

        Returns:
        --------
        Dict containing features
        """
        if feature_types is None:
            feature_types = ["basic", "spectral", "graphlet"]

        features = {}

        if "basic" in feature_types:
            # Basic graph statistics
            features["basic"] = {
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "density": nx.density(graph),
                "is_directed": graph.is_directed(),
                "is_connected": nx.is_weakly_connected(graph) if graph.is_directed() else nx.is_connected(graph)
            }

            # Degree statistics
            degrees = [d for n, d in graph.degree()]
            if degrees:
                features["basic"].update({
                    "degree_mean": np.mean(degrees),
                    "degree_std": np.std(degrees),
                    "degree_min": min(degrees),
                    "degree_max": max(degrees),
                    "degree_gini": MLIntegration._gini_coefficient(degrees)
                })

            # Clustering
            if not graph.is_directed():
                features["basic"]["average_clustering"] = nx.average_clustering(graph)
                features["basic"]["transitivity"] = nx.transitivity(graph)

            # Components
            if graph.is_directed():
                features["basic"]["num_sccs"] = nx.number_strongly_connected_components(graph)
                features["basic"]["num_wccs"] = nx.number_weakly_connected_components(graph)
            else:
                features["basic"]["num_components"] = nx.number_connected_components(graph)

        if "spectral" in feature_types and graph.number_of_nodes() < MAX_NODES_FOR_SPECTRAL_FEATURES:
            # Spectral features
            adj_matrix = nx.adjacency_matrix(graph)

            try:
                # Compute eigenvalues of adjacency matrix
                if graph.number_of_nodes() < MAX_NODES_FOR_EXPENSIVE_COMPUTATION:
                    eigenvalues = np.linalg.eigvalsh(adj_matrix.toarray())
                    eigenvalues = sorted(eigenvalues, reverse=True)

                    features["spectral"] = {
                        "largest_eigenvalue": eigenvalues[0],
                        "second_largest_eigenvalue": eigenvalues[1] if len(eigenvalues) > 1 else 0,
                        "spectral_gap": eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0,
                        "spectral_radius": max(abs(eigenvalues)),
                        "algebraic_connectivity": eigenvalues[-2] if len(eigenvalues) > 1 else 0
                    }
                elif HAS_SCIPY_SPARSE:
                    # Use sparse methods for larger graphs
                    eigenvalues, _ = eigs(adj_matrix.asfptype(), k=min(6, graph.number_of_nodes()-1))
                    eigenvalues = sorted(eigenvalues.real, reverse=True)

                    features["spectral"] = {
                            "largest_eigenvalue": eigenvalues[0],
                            "second_largest_eigenvalue": eigenvalues[1] if len(eigenvalues) > 1 else 0,
                            "spectral_gap": eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0
                        }
                    else:
                        features["spectral"] = {"error": "scipy.sparse required for large graphs"}
            except Exception as e:
                logger.debug(f"Failed to compute spectral features: {e}")
                features["spectral"] = {"error": f"Could not compute spectral features: {e!s}"}

        if "graphlet" in feature_types and graph.number_of_nodes() < MAX_NODES_FOR_EXPENSIVE_COMPUTATION:
            # Graphlet features (small subgraph counts)
            features["graphlet"] = MLIntegration._graphlet_features(graph)

        if "centrality" in feature_types and graph.number_of_nodes() < MAX_NODES_FOR_EXPENSIVE_COMPUTATION:
            # Centrality statistics
            centrality_stats = {}

            # Betweenness
            betweenness = nx.betweenness_centrality(graph)
            betweenness_values = list(betweenness.values())
            centrality_stats["betweenness"] = {
                "mean": np.mean(betweenness_values),
                "std": np.std(betweenness_values),
                "max": max(betweenness_values),
                "gini": MLIntegration._gini_coefficient(betweenness_values)
            }

            # PageRank (if directed)
            if graph.is_directed():
                try:
                    pagerank = nx.pagerank(graph)
                    pr_values = list(pagerank.values())
                    centrality_stats["pagerank"] = {
                        "mean": np.mean(pr_values),
                        "std": np.std(pr_values),
                        "max": max(pr_values),
                        "gini": MLIntegration._gini_coefficient(pr_values)
                    }
                except Exception as e:
                    logger.debug(f"Failed to compute PageRank: {e}")
                    pass

            features["centrality"] = centrality_stats

        # Create feature vector
        feature_vector = []
        feature_names = []

        for category, cat_features in features.items():
            if isinstance(cat_features, dict):
                for name, value in cat_features.items():
                    if isinstance(value, (int, float)):
                        feature_vector.append(value)
                        feature_names.append(f"{category}_{name}")

        return {
            "features": features,
            "feature_vector": feature_vector,
            "feature_names": feature_names,
            "vector_dimension": len(feature_vector)
        }

    @staticmethod
    def _gini_coefficient(values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if not values or len(values) == 0:
            return 0.0

        # Sort values
        sorted_values = sorted(values)
        n = len(sorted_values)

        # Calculate Gini
        cumsum = 0
        for i, value in enumerate(sorted_values):
            cumsum += (2 * (i + 1) - n - 1) * value

        return cumsum / (n * sum(sorted_values)) if sum(sorted_values) > 0 else 0

    @staticmethod
    def _graphlet_features(graph: nx.Graph) -> Dict[str, int]:
        """Count small subgraph patterns (graphlets)."""
        features = {
            "triangles": 0,
            "4_cliques": 0,
            "4_cycles": 0,
            "4_paths": 0,
            "4_stars": 0
        }

        # Count triangles
        features["triangles"] = sum(nx.triangles(graph).values()) // 3

        # Count 4-node patterns (simplified)
        nodes = list(graph.nodes())

        for nodes_subset in combinations(nodes, 4):
            subgraph = graph.subgraph(nodes_subset)
            edges = subgraph.number_of_edges()

            FOUR_CLIQUE_EDGES = 6  # noqa: PLR2004
            if edges == FOUR_CLIQUE_EDGES:
                features["4_cliques"] += 1
            elif edges == 4:  # noqa: PLR2004
                # Could be 4-cycle or other pattern
                degrees = [subgraph.degree(n) for n in nodes_subset]
                if all(d == 2 for d in degrees):
                    features["4_cycles"] += 1
            elif edges == 3:
                # Could be 4-path or 4-star
                degrees = sorted([subgraph.degree(n) for n in nodes_subset])
                if degrees == [1, 1, 2, 2]:
                    features["4_paths"] += 1
                elif degrees == [1, 1, 1, 3]:
                    features["4_stars"] += 1

        return features

    @staticmethod
    def similarity_metrics(
        graph1: nx.Graph,
        graph2: nx.Graph,
        metrics: Optional[List[str]] = None,
        **_params
    ) -> Dict[str, Any]:
        """
        Calculate similarity between two graphs.

        Parameters:
        -----------
        graph1, graph2 : nx.Graph
            Graphs to compare
        metrics : List[str]
            Similarity metrics to compute

        Returns:
        --------
        Dict containing similarity scores
        """
        if metrics is None:
            metrics = ["structural", "spectral", "feature"]

        results = {}

        if "structural" in metrics:
            # Basic structural similarity
            structural_sim = {
                "node_ratio": min(graph1.number_of_nodes(), graph2.number_of_nodes()) /
                              max(graph1.number_of_nodes(), graph2.number_of_nodes()),
                "edge_ratio": min(graph1.number_of_edges(), graph2.number_of_edges()) /
                              max(graph1.number_of_edges(), graph2.number_of_edges()) if max(graph1.number_of_edges(), graph2.number_of_edges()) > 0 else 0,
                "density_diff": abs(nx.density(graph1) - nx.density(graph2))
            }

            # Degree distribution similarity
            deg1 = sorted([d for n, d in graph1.degree()], reverse=True)
            deg2 = sorted([d for n, d in graph2.degree()], reverse=True)

            # Truncate to same length
            min_len = min(len(deg1), len(deg2))
            deg1 = deg1[:min_len]
            deg2 = deg2[:min_len]

            if deg1 and deg2:
                # Cosine similarity of degree sequences
                dot_product = sum(d1 * d2 for d1, d2 in zip(deg1, deg2))
                norm1 = np.sqrt(sum(d * d for d in deg1))
                norm2 = np.sqrt(sum(d * d for d in deg2))

                structural_sim["degree_cosine_similarity"] = (
                    dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
                )

            results["structural"] = structural_sim

        if "spectral" in metrics and max(graph1.number_of_nodes(), graph2.number_of_nodes()) < MAX_NODES_FOR_EXPENSIVE_COMPUTATION:
            # Spectral similarity
            try:
                # Compare eigenvalue distributions
                eig1 = sorted(np.linalg.eigvalsh(nx.adjacency_matrix(graph1).toarray()), reverse=True)
                eig2 = sorted(np.linalg.eigvalsh(nx.adjacency_matrix(graph2).toarray()), reverse=True)

                # Truncate to same length
                min_len = min(len(eig1), len(eig2))
                eig1 = eig1[:min_len]
                eig2 = eig2[:min_len]

                # Spectral distance
                spectral_distance = np.sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(eig1, eig2)))

                results["spectral"] = {
                    "spectral_distance": spectral_distance,
                    "normalized_distance": spectral_distance / np.sqrt(min_len) if min_len > 0 else 0
                }
            except Exception as e:
                logger.debug(f"Failed to compute spectral similarity: {e}")
                results["spectral"] = {"error": f"Could not compute spectral similarity: {str(e)}"}

        if "feature" in metrics:
            # Feature-based similarity
            feat1 = MLIntegration.graph_features(graph1)["feature_vector"]
            feat2 = MLIntegration.graph_features(graph2)["feature_vector"]

            # Ensure same length
            max_len = max(len(feat1), len(feat2))
            feat1 = feat1 + [0] * (max_len - len(feat1))
            feat2 = feat2 + [0] * (max_len - len(feat2))

            # Normalize
            norm1 = np.linalg.norm(feat1)
            norm2 = np.linalg.norm(feat2)

            if norm1 > 0:
                feat1 = feat1 / norm1
            if norm2 > 0:
                feat2 = feat2 / norm2

            # Cosine similarity
            feature_similarity = np.dot(feat1, feat2)

            results["feature"] = {
                "cosine_similarity": feature_similarity,
                "euclidean_distance": np.linalg.norm(np.array(feat1) - np.array(feat2))
            }

        return results

    @staticmethod
    def anomaly_detection(
        graph: Union[nx.Graph, nx.DiGraph],
        method: str = "statistical",
        contamination: float = 0.1,
        **_params
    ) -> Dict[str, Any]:
        """
        Detect anomalous nodes or edges in the graph.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            Input graph
        method : str
            Method: 'statistical', 'structural', 'spectral'
        contamination : float
            Expected fraction of anomalies

        Returns:
        --------
        Dict containing anomaly scores and detected anomalies
        """
        start_time = time.time()

        if method == "statistical":
            # Statistical anomaly detection based on node features
            node_scores = {}

            # Calculate features for each node
            for node in graph.nodes():
                features = []

                # Degree
                if graph.is_directed():
                    features.extend([graph.in_degree(node), graph.out_degree(node)])
                else:
                    features.append(graph.degree(node))

                # Local clustering
                try:
                    features.append(nx.clustering(graph, node))
                except Exception as e:
                    logger.debug(f"Failed to compute clustering coefficient: {e}")
                    features.append(0)

                # Neighbor statistics
                neighbor_degrees = [graph.degree(n) for n in graph.neighbors(node)]
                if neighbor_degrees:
                    features.extend([
                        np.mean(neighbor_degrees),
                        np.std(neighbor_degrees)
                    ])
                else:
                    features.extend([0, 0])

                node_scores[node] = features

            # Convert to matrix
            nodes = list(node_scores.keys())
            feature_matrix = np.array(list(node_scores.values()))

            # Normalize
            if feature_matrix.shape[0] > 1:
                scaler = StandardScaler()
                feature_matrix = scaler.fit_transform(feature_matrix)

            # Calculate anomaly scores using Mahalanobis distance
            mean = np.mean(feature_matrix, axis=0)

            # Covariance with regularization
            cov = np.cov(feature_matrix.T)
            cov += np.eye(cov.shape[0]) * 1e-6  # Regularization

            try:
                inv_cov = np.linalg.inv(cov)

                anomaly_scores = {}
                for i, node in enumerate(nodes):
                    diff = feature_matrix[i] - mean
                    score = np.sqrt(diff @ inv_cov @ diff)
                    anomaly_scores[node] = score

            except Exception as e:
                # Fallback to Euclidean distance
                logger.debug(f"Failed to compute Mahalanobis distance, using Euclidean: {e}")
                anomaly_scores = {}
                for i, node in enumerate(nodes):
                    score = np.linalg.norm(feature_matrix[i] - mean)
                    anomaly_scores[node] = score

        elif method == "structural":
            # Structural anomaly detection
            anomaly_scores = {}

            # Use ego-graph similarity
            for node in graph.nodes():
                # Get ego graph
                ego = nx.ego_graph(graph, node, radius=1)

                # Compare to neighbor ego graphs
                neighbor_similarities = []

                for neighbor in graph.neighbors(node):
                    neighbor_ego = nx.ego_graph(graph, neighbor, radius=1)

                    # Simple similarity: Jaccard of nodes
                    ego_nodes = set(ego.nodes())
                    neighbor_nodes = set(neighbor_ego.nodes())

                    jaccard = len(ego_nodes & neighbor_nodes) / len(ego_nodes | neighbor_nodes)
                    neighbor_similarities.append(jaccard)

                # Anomaly score: inverse of average similarity
                if neighbor_similarities:
                    anomaly_scores[node] = 1 - np.mean(neighbor_similarities)
                else:
                    anomaly_scores[node] = 1.0

        elif method == "spectral":
            # Spectral anomaly detection
            if graph.number_of_nodes() > MAX_NODES_FOR_SPECTRAL_FEATURES:
                return {"error": "Graph too large for spectral method"}

            # Use spectral embedding
            embeddings = MLIntegration._spectral_embeddings(graph, dimensions=DEFAULT_WALK_LENGTH)

            # Calculate distances from center
            embedding_matrix = np.array(list(embeddings.values()))
            center = np.mean(embedding_matrix, axis=0)

            anomaly_scores = {}
            nodes = list(embeddings.keys())

            for i, node in enumerate(nodes):
                score = np.linalg.norm(embedding_matrix[i] - center)
                anomaly_scores[node] = score

        else:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

        # Identify anomalies based on contamination rate
        threshold = np.percentile(
            list(anomaly_scores.values()),
            (1 - contamination) * 100
        )

        anomalous_nodes = [
            node for node, score in anomaly_scores.items()
            if score > threshold
        ]

        # Edge anomalies (for small graphs)
        edge_anomalies = []
        if graph.number_of_edges() < 10000:
            edge_scores = {}

            for u, v in graph.edges():
                # Edge is anomalous if it connects nodes with very different properties
                if u in anomaly_scores and v in anomaly_scores:
                    # High score if one node is anomalous and other isn't
                    score = abs(anomaly_scores[u] - anomaly_scores[v])
                    edge_scores[(u, v)] = score

            if edge_scores:
                edge_threshold = np.percentile(
                    list(edge_scores.values()),
                    (1 - contamination) * 100
                )

                edge_anomalies = [
                    edge for edge, score in edge_scores.items()
                    if score > edge_threshold
                ]

        results = {
            "method": method,
            "node_anomaly_scores": anomaly_scores,
            "anomalous_nodes": anomalous_nodes,
            "num_anomalous_nodes": len(anomalous_nodes),
            "threshold": threshold,
            "contamination_rate": contamination,
            "edge_anomalies": edge_anomalies,
            "num_anomalous_edges": len(edge_anomalies),
            "execution_time_ms": (time.time() - start_time) * MILLISECONDS_PER_SECOND
        }

        # Top anomalies
        sorted_nodes = sorted(
            anomaly_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        results["top_anomalies"] = sorted_nodes[:10]

        return results
