"""Graph generators for creating various types of synthetic networks."""

import logging
import random  # Using for non-cryptographic graph generation purposes only
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Analysis thresholds
MIN_DEGREE_SAMPLES_FOR_ANALYSIS = 10
MAX_NODES_FOR_DETAILED_ANALYSIS = 1000
DEFAULT_EDGE_PROBABILITY = 0.5
MIN_EDGES_FACTOR = 2
DEFAULT_REWIRING_PROBABILITY = 0.1
DEFAULT_NEIGHBORS = 4


class GraphGenerators:
    """Various graph generators for creating synthetic networks."""

    @staticmethod
    def random_graph(
        n: int,
        graph_type: str = "gnp",
        p: float | None = None,
        m: int | None = None,
        seed: int | None = None,
        directed: bool = False,
        **_params,
    ) -> tuple[nx.Graph, dict[str, Any]]:
        """
        Generate random graphs using various models.

        Parameters:
        -----------
        n : int
            Number of nodes
        graph_type : str
            Type of random graph: 'gnp' (G(n,p)), 'gnm' (G(n,m))
        p : float
            Probability for G(n,p) model
        m : int
            Number of edges for G(n,m) model
        seed : int
            Random seed
        directed : bool
            Whether to create directed graph

        Returns:
        --------
        Tuple of (graph, metadata)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if graph_type == "gnp":
            # Erdős-Rényi G(n,p) model
            if p is None:
                # Default to average degree of log(n)
                p = np.log(n) / n if n > 1 else DEFAULT_EDGE_PROBABILITY

            if directed:
                G = nx.erdos_renyi_graph(n, p, directed=True, seed=seed)
            else:
                G = nx.erdos_renyi_graph(n, p, seed=seed)

            metadata = {
                "generator": "erdos_renyi_gnp",
                "n": n,
                "p": p,
                "expected_edges": (
                    n * (n - 1) * p / MIN_EDGES_FACTOR
                    if not directed
                    else n * (n - 1) * p
                ),
                "directed": directed,
            }

        elif graph_type == "gnm":
            # Erdős-Rényi G(n,m) model
            if m is None:
                # Default to n * log(n) edges
                m = int(n * np.log(n))

            max_edges = n * (n - 1) // MIN_EDGES_FACTOR if not directed else n * (n - 1)
            m = min(m, max_edges)

            if directed:
                G = nx.gnm_random_graph(n, m, directed=True, seed=seed)
            else:
                G = nx.gnm_random_graph(n, m, seed=seed)

            metadata = {
                "generator": "erdos_renyi_gnm",
                "n": n,
                "m": m,
                "max_possible_edges": max_edges,
                "density": m / max_edges if max_edges > 0 else 0,
                "directed": directed,
            }

        else:
            msg = f"Unknown random graph type: {graph_type}"
            raise ValueError(msg)

        # Add node attributes
        for i, node in enumerate(G.nodes()):
            G.nodes[node]["id"] = i
            G.nodes[node]["created_by"] = "random_generator"

        return G, metadata

    @staticmethod
    def scale_free_graph(
        n: int,
        m: int = MIN_EDGES_FACTOR,
        m0: int | None = None,
        alpha: float = 0.41,
        beta: float = 0.54,
        gamma: float = 0.05,
        delta_in: float = 0.2,
        delta_out: float = 0.2,
        seed: int | None = None,
        model: str = "barabasi_albert",
        **params,
    ) -> tuple[nx.Graph, dict[str, Any]]:
        """
        Generate scale-free graphs using various models.

        Parameters:
        -----------
        n : int
            Number of nodes
        m : int
            Number of edges to attach from new node (BA model)
        m0 : int
            Initial number of nodes (default: m)
        alpha, beta, gamma : float
            Parameters for extended BA model
        model : str
            Model type: 'barabasi_albert', 'extended_ba', 'powerlaw_cluster'

        Returns:
        --------
        Tuple of (graph, metadata)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if model == "barabasi_albert":
            # Classic Barabási-Albert preferential attachment
            if m0 is None:
                m0 = m

            G = nx.barabasi_albert_graph(n, m, seed=seed)

            metadata = {
                "generator": "barabasi_albert",
                "n": n,
                "m": m,
                "expected_edges": m0 * (m0 - 1) / MIN_EDGES_FACTOR + m * (n - m0),
                "model": "preferential_attachment",
            }

        elif model == "extended_ba":
            # Extended Barabási-Albert with additional mechanisms
            G = nx.extended_barabasi_albert_graph(n, m, p=alpha, q=beta, seed=seed)

            metadata = {
                "generator": "extended_barabasi_albert",
                "n": n,
                "m": m,
                "p": alpha,
                "q": beta,
                "model": "extended_preferential_attachment",
            }

        elif model == "powerlaw_cluster":
            # Powerlaw cluster graph (Holme-Kim)
            p = params.get("triangle_prob", 0.3)
            G = nx.powerlaw_cluster_graph(n, m, p, seed=seed)

            metadata = {
                "generator": "powerlaw_cluster",
                "n": n,
                "m": m,
                "triangle_probability": p,
                "model": "powerlaw_with_clustering",
            }

        elif model == "directed_scale_free":
            # Directed scale-free graph
            G = nx.scale_free_graph(
                n,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta_in=delta_in,
                delta_out=delta_out,
                seed=seed,
            )

            metadata = {
                "generator": "directed_scale_free",
                "n": n,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "delta_in": delta_in,
                "delta_out": delta_out,
                "model": "directed_preferential_attachment",
            }

        else:
            msg = f"Unknown scale-free model: {model}"
            raise ValueError(msg)

        # Calculate power-law exponent estimate
        degrees = [d for n, d in G.degree()]
        if len(degrees) > 0 and max(degrees) > 1:
            # Simple estimate using log-log regression
            log_degrees = np.log(sorted(degrees, reverse=True)[1:])  # Exclude max
            log_ranks = np.log(np.arange(1, len(log_degrees) + 1))

            if len(log_degrees) > MIN_DEGREE_SAMPLES_FOR_ANALYSIS:
                # Linear regression in log-log space
                slope, _ = np.polyfit(
                    log_ranks[: len(log_ranks) // MIN_EDGES_FACTOR],
                    log_degrees[: len(log_degrees) // MIN_EDGES_FACTOR],
                    1,
                )
                metadata["estimated_exponent"] = -slope

        return G, metadata

    @staticmethod
    def small_world_graph(
        n: int,
        k: int = DEFAULT_NEIGHBORS,
        p: float = DEFAULT_REWIRING_PROBABILITY,
        tries: int = 100,
        seed: int | None = None,
        model: str = "watts_strogatz",
        **_params,
    ) -> tuple[nx.Graph, dict[str, Any]]:
        """
        Generate small-world graphs.

        Parameters:
        -----------
        n : int
            Number of nodes
        k : int
            Each node connected to k nearest neighbors
        p : float
            Probability of rewiring each edge
        model : str
            Model type: 'watts_strogatz', 'newman_watts_strogatz', 'connected_watts_strogatz'

        Returns:
        --------
        Tuple of (graph, metadata)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if model == "watts_strogatz":
            # Classic Watts-Strogatz model
            G = nx.watts_strogatz_graph(n, k, p, seed=seed)

            metadata = {
                "generator": "watts_strogatz",
                "n": n,
                "k": k,
                "p": p,
                "expected_edges": n * k // MIN_EDGES_FACTOR,
                "model": "ring_rewiring",
            }

        elif model == "newman_watts_strogatz":
            # Newman-Watts-Strogatz (adds edges instead of rewiring)
            G = nx.newman_watts_strogatz_graph(n, k, p, seed=seed)

            metadata = {
                "generator": "newman_watts_strogatz",
                "n": n,
                "k": k,
                "p": p,
                "expected_edges": n * k // MIN_EDGES_FACTOR
                + n * k * p // MIN_EDGES_FACTOR,
                "model": "ring_addition",
            }

        elif model == "connected_watts_strogatz":
            # Ensures graph remains connected
            G = nx.connected_watts_strogatz_graph(n, k, p, tries=tries, seed=seed)

            metadata = {
                "generator": "connected_watts_strogatz",
                "n": n,
                "k": k,
                "p": p,
                "tries": tries,
                "model": "connected_ring_rewiring",
            }

        else:
            msg = f"Unknown small-world model: {model}"
            raise ValueError(msg)

        # Calculate small-world metrics
        if n < MAX_NODES_FOR_DETAILED_ANALYSIS:  # Only for smaller graphs
            try:
                # Average shortest path length
                if nx.is_connected(G):
                    avg_path_length = nx.average_shortest_path_length(G)
                    metadata["average_shortest_path_length"] = avg_path_length

                    # For reference: random graph would have ~log(n)/log(k)
                    random_path_length = np.log(n) / np.log(k) if k > 1 else n
                    metadata["random_graph_path_length"] = random_path_length

                # Clustering coefficient
                avg_clustering = nx.average_clustering(G)
                metadata["average_clustering"] = avg_clustering

                # For reference: random graph would have ~k/n
                random_clustering = k / n if n > 0 else 0
                metadata["random_graph_clustering"] = random_clustering

            except Exception as e:
                logger.debug(f"Failed to compute clustering coefficient: {e}")

        return G, metadata

    @staticmethod
    def regular_graph(
        n: int,
        d: int,
        seed: int | None = None,
        graph_type: str = "random_regular",
        **params,
    ) -> tuple[nx.Graph, dict[str, Any]]:
        """
        Generate regular graphs where every node has the same degree.

        Parameters:
        -----------
        n : int
            Number of nodes
        d : int
            Degree of each node
        graph_type : str
            Type: 'random_regular', 'circulant', 'cycle', 'complete'

        Returns:
        --------
        Tuple of (graph, metadata)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if graph_type == "random_regular":
            # Random d-regular graph
            if n * d % MIN_EDGES_FACTOR != 0:
                msg = "n * d must be even for regular graphs"
                raise ValueError(msg)

            G = nx.random_regular_graph(d, n, seed=seed)

            metadata = {
                "generator": "random_regular",
                "n": n,
                "d": d,
                "edges": n * d // MIN_EDGES_FACTOR,
                "is_regular": True,
            }

        elif graph_type == "circulant":
            # Circulant graph
            if "offsets" in params:
                offsets = params["offsets"]
            else:
                # Default offsets for d-regular circulant
                offsets = list(range(1, d // MIN_EDGES_FACTOR + 1))
                if d % 2 == 0:
                    offsets[-1] = n // 2  # For even d, connect to opposite node

            G = nx.circulant_graph(n, offsets)

            metadata = {
                "generator": "circulant",
                "n": n,
                "offsets": offsets,
                "is_regular": True,
                "is_vertex_transitive": True,
            }

        elif graph_type == "cycle":
            # Cycle graph (2-regular)
            G = nx.cycle_graph(n)

            metadata = {
                "generator": "cycle",
                "n": n,
                "d": 2,
                "edges": n,
                "is_regular": True,
                "diameter": n // 2 if n % 2 == 0 else (n - 1) // 2,
            }

        elif graph_type == "complete":
            # Complete graph (n-1 regular)
            G = nx.complete_graph(n)

            metadata = {
                "generator": "complete",
                "n": n,
                "d": n - 1,
                "edges": n * (n - 1) // 2,
                "is_regular": True,
                "diameter": 1,
            }

        else:
            msg = f"Unknown regular graph type: {graph_type}"
            raise ValueError(msg)

        return G, metadata

    @staticmethod
    def tree_graph(
        n: int,
        tree_type: str = "random",
        branching: int = 2,
        seed: int | None = None,
        **params,
    ) -> tuple[nx.Graph, dict[str, Any]]:
        """
        Generate tree graphs.

        Parameters:
        -----------
        n : int
            Number of nodes
        tree_type : str
            Type: 'random', 'balanced', 'star', 'path', 'caterpillar'
        branching : int
            Branching factor for balanced trees

        Returns:
        --------
        Tuple of (graph, metadata)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if tree_type == "random":
            # Random tree using Prüfer sequence
            if n <= 0:
                G = nx.empty_graph(0)
            else:
                sequence = [
                    random.randint(0, n - 1) for _ in range(n - 2)
                ]  # noqa: S311
                G = nx.from_prufer_sequence(sequence)

            metadata = {
                "generator": "random_tree",
                "n": n,
                "edges": n - 1 if n > 0 else 0,
                "is_tree": True,
            }

        elif tree_type == "balanced":
            # Balanced r-ary tree
            r = branching
            height = int(np.log(n * (r - 1) + 1) / np.log(r))
            G = nx.balanced_tree(r, height)

            # Trim to exactly n nodes if needed
            if G.number_of_nodes() > n:
                nodes_to_remove = list(G.nodes())[n:]
                G.remove_nodes_from(nodes_to_remove)

            metadata = {
                "generator": "balanced_tree",
                "n": G.number_of_nodes(),
                "branching": r,
                "height": height,
                "is_tree": True,
            }

        elif tree_type == "star":
            # Star graph (one central node connected to all others)
            G = nx.star_graph(n - 1)

            metadata = {
                "generator": "star",
                "n": n,
                "edges": n - 1,
                "diameter": 2 if n > 2 else 1,  # noqa: PLR2004
                "is_tree": True,
            }

        elif tree_type == "path":
            # Path graph (linear tree)
            G = nx.path_graph(n)

            metadata = {
                "generator": "path",
                "n": n,
                "edges": n - 1 if n > 0 else 0,
                "diameter": n - 1 if n > 0 else 0,
                "is_tree": True,
            }

        elif tree_type == "caterpillar":
            # Caterpillar tree (path with leaves)
            backbone_size = params.get("backbone_size", n // 2)
            G = nx.path_graph(backbone_size)

            # Add leaves
            node_id = backbone_size
            for i in range(min(backbone_size, n - backbone_size)):
                G.add_edge(i, node_id)
                node_id += 1
                if node_id >= n:
                    break

            metadata = {
                "generator": "caterpillar",
                "n": G.number_of_nodes(),
                "backbone_size": backbone_size,
                "leaves": G.number_of_nodes() - backbone_size,
                "is_tree": True,
            }

        else:
            msg = f"Unknown tree type: {tree_type}"
            raise ValueError(msg)

        # Add tree-specific metrics
        if n > 0 and nx.is_tree(G):
            leaves = [n for n in G.nodes() if G.degree(n) == 1]
            metadata["num_leaves"] = len(leaves)
            metadata["leaf_fraction"] = len(leaves) / n

            # Tree center
            center = nx.center(G)
            metadata["center_nodes"] = center
            metadata["radius"] = nx.radius(G)

        return G, metadata

    @staticmethod
    def geometric_graph(
        n: int,
        radius: float | None = None,
        dim: int = 2,
        pos: dict | None = None,
        p: float = 2,
        seed: int | None = None,
        graph_type: str = "random_geometric",
        **params,
    ) -> tuple[nx.Graph, dict[str, Any]]:
        """
        Generate geometric graphs.

        Parameters:
        -----------
        n : int
            Number of nodes
        radius : float
            Connection radius
        dim : int
            Dimension of space
        p : float
            Minkowski distance metric parameter
        graph_type : str
            Type: 'random_geometric', 'soft_random_geometric', 'geographical_threshold'

        Returns:
        --------
        Tuple of (graph, metadata)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Default radius to achieve expected average degree ~log(n)
        if radius is None:
            if dim == 2:  # noqa: PLR2004
                # Area of circle = πr², expected degree ≈ n * πr²
                DEFAULT_DEGREE = 2
                expected_degree = np.log(n) if n > 1 else DEFAULT_DEGREE
                radius = np.sqrt(expected_degree / (n * np.pi))
            else:
                # Rough approximation for higher dimensions
                radius = (expected_degree / n) ** (1 / dim)

        if graph_type == "random_geometric":
            # Standard random geometric graph
            G = nx.random_geometric_graph(n, radius, dim=dim, pos=pos, p=p, seed=seed)

            metadata = {
                "generator": "random_geometric",
                "n": n,
                "radius": radius,
                "dimension": dim,
                "metric": f"L{p}" if p != 2 else "Euclidean",  # noqa: PLR2004
                "expected_degree": (
                    n * (np.pi * radius**2) if dim == 2 else "varies"
                ),  # noqa: PLR2004
            }

        elif graph_type == "soft_random_geometric":
            # Soft random geometric graph (probabilistic connections)
            if pos is None:
                pos = {i: np.random.rand(dim) for i in range(n)}

            G = nx.soft_random_geometric_graph(
                n,
                radius,
                dim=dim,
                pos=pos,
                p_dist=params.get("p_dist", None),
                seed=seed,
            )

            metadata = {
                "generator": "soft_random_geometric",
                "n": n,
                "radius": radius,
                "dimension": dim,
                "connection_function": "probabilistic",
            }

        elif graph_type == "geographical_threshold":
            # Geographical threshold graph
            theta = params.get("theta", 0.1)
            G = nx.geographical_threshold_graph(
                n, theta, dim=dim, pos=pos, weight=params.get("weight", None), seed=seed
            )

            metadata = {
                "generator": "geographical_threshold",
                "n": n,
                "theta": theta,
                "dimension": dim,
                "model": "weight_distance_threshold",
            }

        elif graph_type == "knn":
            # K-nearest neighbors graph
            k = params.get("k", int(np.log(n)))

            if pos is None:
                pos = {i: np.random.rand(dim) for i in range(n)}

            # Build KNN graph
            G = nx.Graph()
            G.add_nodes_from(range(n))

            # For each node, connect to k nearest neighbors
            for i in range(n):
                distances = []
                for j in range(n):
                    if i != j:
                        dist = np.linalg.norm(
                            np.array(pos[i]) - np.array(pos[j]), ord=p
                        )
                        distances.append((dist, j))

                distances.sort()
                for _, j in distances[:k]:
                    G.add_edge(i, j)

            # Set positions
            nx.set_node_attributes(G, pos, "pos")

            metadata = {
                "generator": "k_nearest_neighbors",
                "n": n,
                "k": k,
                "dimension": dim,
                "directed": False,
            }

        else:
            msg = f"Unknown geometric graph type: {graph_type}"
            raise ValueError(msg)

        # Store positions if not already stored
        if "pos" not in G.nodes[next(iter(G.nodes()))]:
            if pos is None:
                pos = {i: np.random.rand(dim) for i in G.nodes()}
            nx.set_node_attributes(G, pos, "pos")

        # Calculate spatial statistics
        positions = nx.get_node_attributes(G, "pos")
        if positions:
            # Average edge length
            edge_lengths = []
            for u, v in G.edges():
                if u in positions and v in positions:
                    length = np.linalg.norm(
                        np.array(positions[u]) - np.array(positions[v]), ord=p
                    )
                    edge_lengths.append(length)

            if edge_lengths:
                metadata["average_edge_length"] = np.mean(edge_lengths)
                metadata["edge_length_std"] = np.std(edge_lengths)

        return G, metadata

    @staticmethod
    def social_network_graph(
        n: int,
        model: str = "stochastic_block",
        communities: int | None = None,
        p_in: float = 0.3,
        p_out: float = 0.01,
        tau1: float = 3.0,
        tau2: float = 1.5,
        mu: float = 0.1,
        seed: int | None = None,
        **params,
    ) -> tuple[nx.Graph, dict[str, Any]]:
        """
        Generate social network models.

        Parameters:
        -----------
        n : int
            Number of nodes
        model : str
            Model type: 'stochastic_block', 'lfr_benchmark', 'caveman', 'relaxed_caveman'
        communities : int
            Number of communities
        p_in : float
            Intra-community connection probability
        p_out : float
            Inter-community connection probability
        tau1, tau2 : float
            Power-law exponents for LFR model
        mu : float
            Mixing parameter for LFR model

        Returns:
        --------
        Tuple of (graph, metadata)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if communities is None:
            communities = max(2, int(np.sqrt(n)))

        if model == "stochastic_block":
            # Stochastic block model
            sizes = [n // communities] * communities
            # Distribute remaining nodes
            for i in range(n % communities):
                sizes[i] += 1

            # Create probability matrix
            probs = []
            for i in range(communities):
                row = []
                for j in range(communities):
                    if i == j:
                        row.append(p_in)
                    else:
                        row.append(p_out)
                probs.append(row)

            G = nx.stochastic_block_model(sizes, probs, seed=seed)

            # Add community labels
            node_id = 0
            for comm_id, size in enumerate(sizes):
                for _ in range(size):
                    G.nodes[node_id]["community"] = comm_id
                    node_id += 1

            metadata = {
                "generator": "stochastic_block_model",
                "n": n,
                "communities": communities,
                "community_sizes": sizes,
                "p_in": p_in,
                "p_out": p_out,
                "expected_modularity": (p_in - p_out) * (1 - 1 / communities),
            }

        elif model == "lfr_benchmark":
            # LFR benchmark graph
            try:
                G = nx.generators.community.LFR_benchmark_graph(
                    n,
                    tau1=tau1,
                    tau2=tau2,
                    mu=mu,
                    average_degree=params.get("average_degree", 10),
                    min_degree=params.get("min_degree", None),
                    max_degree=params.get("max_degree", None),
                    min_community=params.get("min_community", 10),
                    max_community=params.get("max_community", n // 5),
                    seed=seed,
                )

                # Extract communities
                communities_dict = {frozenset(G.nodes[v]["community"]) for v in G}
                num_communities = len(communities_dict)

                metadata = {
                    "generator": "lfr_benchmark",
                    "n": n,
                    "tau1": tau1,
                    "tau2": tau2,
                    "mu": mu,
                    "communities_found": num_communities,
                    "model": "powerlaw_degree_community_size",
                }

            except Exception as e:
                # Fallback to stochastic block model
                logger.warning(f"LFR generation failed, falling back to SBM: {e}")
                return GraphGenerators.social_network_graph(
                    n,
                    model="stochastic_block",
                    communities=communities,
                    p_in=p_in,
                    p_out=p_out,
                    seed=seed,
                )

        elif model == "caveman":
            # Connected caveman graph (cliques connected in a ring)
            clique_size = n // communities
            G = nx.connected_caveman_graph(communities, clique_size)

            # Add community labels
            for i, node in enumerate(G.nodes()):
                G.nodes[node]["community"] = i // clique_size

            metadata = {
                "generator": "connected_caveman",
                "n": G.number_of_nodes(),
                "communities": communities,
                "clique_size": clique_size,
                "model": "ring_of_cliques",
            }

        elif model == "relaxed_caveman":
            # Relaxed caveman graph
            clique_size = n // communities
            p_relaxation = params.get("p_relaxation", 0.1)
            G = nx.relaxed_caveman_graph(
                communities, clique_size, p_relaxation, seed=seed
            )

            # Add community labels
            for i, node in enumerate(G.nodes()):
                G.nodes[node]["community"] = i // clique_size

            metadata = {
                "generator": "relaxed_caveman",
                "n": G.number_of_nodes(),
                "communities": communities,
                "clique_size": clique_size,
                "p_relaxation": p_relaxation,
                "model": "relaxed_ring_of_cliques",
            }

        else:
            msg = f"Unknown social network model: {model}"
            raise ValueError(msg)

        # Calculate community statistics
        if "community" in G.nodes[next(iter(G.nodes()))]:
            comm_nodes = {}
            for node in G.nodes():
                comm = G.nodes[node].get("community")
                if comm is not None:
                    if comm not in comm_nodes:
                        comm_nodes[comm] = []
                    comm_nodes[comm].append(node)

            # Calculate modularity
            if comm_nodes:
                communities_list = list(comm_nodes.values())
                modularity = nx.algorithms.community.modularity(G, communities_list)
                metadata["actual_modularity"] = modularity

                # Calculate mixing parameter (fraction of inter-community edges)
                inter_edges = 0
                total_edges = G.number_of_edges()

                for u, v in G.edges():
                    if G.nodes[u].get("community") != G.nodes[v].get("community"):
                        inter_edges += 1

                metadata["actual_mixing_parameter"] = (
                    inter_edges / total_edges if total_edges > 0 else 0
                )

        return G, metadata

    @staticmethod
    def graph_from_degree_sequence(
        degree_sequence: list[int],
        method: str = "configuration",
        create_using: nx.Graph | None = None,
        seed: int | None = None,
        **_params,
    ) -> tuple[nx.Graph, dict[str, Any]]:
        """
        Generate graph from a given degree sequence.

        Parameters:
        -----------
        degree_sequence : List[int]
            Desired degree for each node
        method : str
            Method: 'configuration', 'expected_degree', 'havel_hakimi'
        create_using : nx.Graph
            Graph type to create

        Returns:
        --------
        Tuple of (graph, metadata)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Validate degree sequence
        if sum(degree_sequence) % 2 != 0:
            msg = "Sum of degree sequence must be even"
            raise ValueError(msg)

        n = len(degree_sequence)

        # Check if degree sequence is graphical
        is_graphical = nx.is_graphical(degree_sequence)

        if not is_graphical and method == "havel_hakimi":
            msg = "Degree sequence is not graphical"
            raise ValueError(msg)

        if method == "configuration":
            # Configuration model (may have multi-edges and self-loops)
            if create_using is None:
                G = nx.configuration_model(degree_sequence, seed=seed)
                # Remove parallel edges and self-loops
                G = nx.Graph(G)
                G.remove_edges_from(nx.selfloop_edges(G))
            else:
                G = nx.configuration_model(
                    degree_sequence, create_using=create_using, seed=seed
                )

            metadata = {
                "generator": "configuration_model",
                "method": "stub_matching",
                "graphical": is_graphical,
            }

        elif method == "expected_degree":
            # Expected degree sequence (probabilistic)
            # Normalize to probabilities
            max_degree = max(degree_sequence)
            if max_degree > 0:
                normalized = [d / max_degree for d in degree_sequence]
            else:
                normalized = degree_sequence

            G = nx.expected_degree_graph(normalized, seed=seed)

            metadata = {
                "generator": "expected_degree_graph",
                "method": "probabilistic",
                "graphical": is_graphical,
            }

        elif method == "havel_hakimi":
            # Havel-Hakimi algorithm (deterministic)
            G = nx.havel_hakimi_graph(degree_sequence)

            metadata = {
                "generator": "havel_hakimi",
                "method": "deterministic",
                "graphical": True,
            }

        else:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

        # Add degree sequence statistics
        metadata.update(
            {
                "n": n,
                "degree_sequence_sum": sum(degree_sequence),
                "degree_sequence_min": min(degree_sequence),
                "degree_sequence_max": max(degree_sequence),
                "degree_sequence_mean": np.mean(degree_sequence),
                "degree_sequence_std": np.std(degree_sequence),
            }
        )

        # Compare actual degrees with requested
        actual_degrees = [G.degree(i) for i in range(n)]
        degree_errors = [abs(actual_degrees[i] - degree_sequence[i]) for i in range(n)]

        metadata.update(
            {
                "degree_error_mean": np.mean(degree_errors),
                "degree_error_max": max(degree_errors),
                "exact_degree_match": all(e == 0 for e in degree_errors),
            }
        )

        return G, metadata
