"""Network robustness and resilience analysis."""

import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class RobustnessAnalysis:
    """Network robustness and resilience analysis tools."""

    @staticmethod
    def attack_simulation(
        graph: Union[nx.Graph, nx.DiGraph],
        attack_type: str = "random",
        fraction: float = 0.5,
        measure: str = "connectivity",
        **params
    ) -> Dict[str, Any]:
        """
        Simulate node/edge removal attacks on the network.
        
        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            Input graph
        attack_type : str
            Type: 'random', 'targeted_degree', 'targeted_betweenness', 'targeted_eigenvector'
        fraction : float
            Fraction of nodes/edges to remove
        measure : str
            Robustness measure: 'connectivity', 'largest_component', 'efficiency'
            
        Returns:
        --------
        Dict containing attack simulation results
        """
        start_time = time.time()

        # Create a copy to work with
        G = graph.copy()

        # Determine number of nodes to remove
        num_nodes = int(graph.number_of_nodes() * fraction)

        # Get removal order based on attack type
        if attack_type == "random":
            nodes_to_remove = random.sample(list(G.nodes()), num_nodes)

        elif attack_type == "targeted_degree":
            # Remove highest degree nodes first
            node_degrees = dict(G.degree())
            nodes_to_remove = sorted(
                node_degrees.keys(),
                key=lambda x: node_degrees[x],
                reverse=True
            )[:num_nodes]

        elif attack_type == "targeted_betweenness":
            # Remove highest betweenness nodes first
            if G.number_of_nodes() < 1000:
                betweenness = nx.betweenness_centrality(G)
                nodes_to_remove = sorted(
                    betweenness.keys(),
                    key=lambda x: betweenness[x],
                    reverse=True
                )[:num_nodes]
            else:
                # Approximate for large graphs
                betweenness = nx.betweenness_centrality(
                    G, k=min(100, G.number_of_nodes()//10)
                )
                nodes_to_remove = sorted(
                    betweenness.keys(),
                    key=lambda x: betweenness[x],
                    reverse=True
                )[:num_nodes]

        elif attack_type == "targeted_eigenvector":
            # Remove highest eigenvector centrality nodes
            try:
                eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
                nodes_to_remove = sorted(
                    eigenvector.keys(),
                    key=lambda x: eigenvector[x],
                    reverse=True
                )[:num_nodes]
            except:
                # Fallback to degree-based
                node_degrees = dict(G.degree())
                nodes_to_remove = sorted(
                    node_degrees.keys(),
                    key=lambda x: node_degrees[x],
                    reverse=True
                )[:num_nodes]

        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Track metrics during attack
        removal_sequence = []

        # Initial metrics
        initial_metrics = RobustnessAnalysis._calculate_robustness_metrics(G, measure)

        # Simulate progressive removal
        for i, node in enumerate(nodes_to_remove):
            G.remove_node(node)

            # Calculate metrics after removal
            current_metrics = RobustnessAnalysis._calculate_robustness_metrics(G, measure)

            removal_sequence.append({
                "step": i + 1,
                "removed_node": node,
                "fraction_removed": (i + 1) / graph.number_of_nodes(),
                "metrics": current_metrics
            })

            # Early stopping if network is destroyed
            if measure == "connectivity" and current_metrics.get("is_connected", True) == False:
                if i < len(nodes_to_remove) - 1:
                    logger.info(f"Network disconnected after removing {i+1} nodes")

        # Calculate robustness index (area under curve)
        robustness_index = RobustnessAnalysis._calculate_robustness_index(
            removal_sequence, measure
        )

        execution_time = (time.time() - start_time) * 1000

        return {
            "attack_type": attack_type,
            "fraction_removed": fraction,
            "num_nodes_removed": len(nodes_to_remove),
            "measure": measure,
            "initial_metrics": initial_metrics,
            "final_metrics": removal_sequence[-1]["metrics"] if removal_sequence else initial_metrics,
            "robustness_index": robustness_index,
            "removal_sequence": removal_sequence[:100],  # Limit to first 100 steps
            "critical_fraction": RobustnessAnalysis._find_critical_fraction(
                removal_sequence, measure
            ),
            "execution_time_ms": execution_time
        }

    @staticmethod
    def _calculate_robustness_metrics(
        graph: Union[nx.Graph, nx.DiGraph],
        measure: str
    ) -> Dict[str, Any]:
        """Calculate robustness metrics for current graph state."""
        metrics = {}

        if graph.number_of_nodes() == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "is_connected": False,
                "largest_component_size": 0,
                "efficiency": 0
            }

        if measure == "connectivity" or measure == "all":
            # Connectivity metrics
            if graph.is_directed():
                metrics["is_strongly_connected"] = nx.is_strongly_connected(graph)
                metrics["is_weakly_connected"] = nx.is_weakly_connected(graph)
                # Use weak connectivity for general "is_connected"
                metrics["is_connected"] = metrics["is_weakly_connected"]
            else:
                metrics["is_connected"] = nx.is_connected(graph)

        if measure == "largest_component" or measure == "all":
            # Largest component size
            if graph.is_directed():
                sccs = list(nx.strongly_connected_components(graph))
                wccs = list(nx.weakly_connected_components(graph))
                metrics["largest_scc_size"] = max(len(c) for c in sccs) if sccs else 0
                metrics["largest_wcc_size"] = max(len(c) for c in wccs) if wccs else 0
                metrics["largest_component_size"] = metrics["largest_wcc_size"]
                metrics["num_components"] = len(wccs)
            else:
                ccs = list(nx.connected_components(graph))
                metrics["largest_component_size"] = max(len(c) for c in ccs) if ccs else 0
                metrics["num_components"] = len(ccs)

            # Relative size
            metrics["largest_component_fraction"] = (
                metrics["largest_component_size"] / graph.number_of_nodes()
                if graph.number_of_nodes() > 0 else 0
            )

        if measure == "efficiency" or measure == "all":
            # Global efficiency
            if graph.number_of_nodes() < 1000:
                metrics["global_efficiency"] = nx.global_efficiency(graph)
            else:
                # Approximate for large graphs
                metrics["global_efficiency"] = RobustnessAnalysis._approximate_efficiency(graph)

            # Local efficiency (clustering-based approximation)
            if not graph.is_directed():
                metrics["average_clustering"] = nx.average_clustering(graph)

        # Always include basic stats
        metrics["num_nodes"] = graph.number_of_nodes()
        metrics["num_edges"] = graph.number_of_edges()

        return metrics

    @staticmethod
    def _approximate_efficiency(
        graph: Union[nx.Graph, nx.DiGraph],
        sample_size: int = 1000
    ) -> float:
        """Approximate global efficiency for large graphs."""
        nodes = list(graph.nodes())

        if len(nodes) <= sample_size:
            return nx.global_efficiency(graph)

        # Sample node pairs
        sampled_pairs = []
        for _ in range(sample_size):
            u = random.choice(nodes)
            v = random.choice(nodes)
            if u != v:
                sampled_pairs.append((u, v))

        # Calculate average inverse distance
        total_inv_dist = 0
        valid_pairs = 0

        for u, v in sampled_pairs:
            try:
                dist = nx.shortest_path_length(graph, u, v)
                if dist > 0:
                    total_inv_dist += 1.0 / dist
                    valid_pairs += 1
            except nx.NetworkXNoPath:
                pass

        # Scale to full graph
        n = graph.number_of_nodes()
        if valid_pairs > 0:
            avg_inv_dist = total_inv_dist / valid_pairs
            # Approximate total efficiency
            return avg_inv_dist * valid_pairs / sample_size
        else:
            return 0.0

    @staticmethod
    def _calculate_robustness_index(
        removal_sequence: List[Dict],
        measure: str
    ) -> float:
        """Calculate robustness index (area under curve)."""
        if not removal_sequence:
            return 1.0

        # Determine which metric to use
        if measure == "connectivity":
            metric_key = "is_connected"
            is_binary = True
        elif measure == "largest_component":
            metric_key = "largest_component_fraction"
            is_binary = False
        elif measure == "efficiency":
            metric_key = "global_efficiency"
            is_binary = False
        else:
            return 0.0

        # Calculate area under curve
        area = 0.0
        prev_x = 0.0
        prev_y = 1.0  # Start with full functionality

        for step in removal_sequence:
            x = step["fraction_removed"]

            if metric_key in step["metrics"]:
                y = step["metrics"][metric_key]
                if is_binary:
                    y = 1.0 if y else 0.0
            else:
                y = 0.0

            # Trapezoidal rule
            area += (x - prev_x) * (y + prev_y) / 2

            prev_x = x
            prev_y = y

        # Add final segment to x=1 if needed
        if prev_x < 1.0:
            area += (1.0 - prev_x) * prev_y / 2

        return area

    @staticmethod
    def _find_critical_fraction(
        removal_sequence: List[Dict],
        measure: str
    ) -> Optional[float]:
        """Find critical fraction where network fails."""
        if not removal_sequence:
            return None

        if measure == "connectivity":
            # Find when network becomes disconnected
            for step in removal_sequence:
                if not step["metrics"].get("is_connected", True):
                    return step["fraction_removed"]

        elif measure == "largest_component":
            # Find when largest component drops below 50%
            for step in removal_sequence:
                if step["metrics"].get("largest_component_fraction", 1.0) < 0.5:
                    return step["fraction_removed"]

        elif measure == "efficiency":
            # Find when efficiency drops below 50%
            initial_efficiency = removal_sequence[0]["metrics"].get("global_efficiency", 1.0)
            for step in removal_sequence:
                current_efficiency = step["metrics"].get("global_efficiency", 0.0)
                if current_efficiency < 0.5 * initial_efficiency:
                    return step["fraction_removed"]

        return None

    @staticmethod
    def percolation_analysis(
        graph: Union[nx.Graph, nx.DiGraph],
        percolation_type: str = "site",
        probability_range: Tuple[float, float] = (0.0, 1.0),
        num_steps: int = 20,
        num_trials: int = 10,
        **params
    ) -> Dict[str, Any]:
        """
        Analyze percolation threshold of the network.
        
        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            Input graph
        percolation_type : str
            Type: 'site' (node) or 'bond' (edge) percolation
        probability_range : Tuple[float, float]
            Range of occupation probabilities to test
        num_steps : int
            Number of probability values to test
        num_trials : int
            Number of trials per probability
            
        Returns:
        --------
        Dict containing percolation analysis
        """
        start_time = time.time()

        probabilities = np.linspace(
            probability_range[0],
            probability_range[1],
            num_steps
        )

        results = []

        for p in probabilities:
            trial_results = []

            for trial in range(num_trials):
                G = graph.copy()

                if percolation_type == "site":
                    # Site percolation: randomly remove nodes
                    nodes_to_keep = [
                        node for node in G.nodes()
                        if random.random() < p
                    ]
                    nodes_to_remove = set(G.nodes()) - set(nodes_to_keep)
                    G.remove_nodes_from(nodes_to_remove)

                elif percolation_type == "bond":
                    # Bond percolation: randomly remove edges
                    edges_to_remove = [
                        edge for edge in G.edges()
                        if random.random() > p
                    ]
                    G.remove_edges_from(edges_to_remove)

                else:
                    raise ValueError(f"Unknown percolation type: {percolation_type}")

                # Calculate metrics
                if G.number_of_nodes() > 0:
                    if G.is_directed():
                        components = list(nx.weakly_connected_components(G))
                    else:
                        components = list(nx.connected_components(G))

                    if components:
                        largest_component_size = max(len(c) for c in components)
                        giant_component_fraction = (
                            largest_component_size / graph.number_of_nodes()
                        )
                    else:
                        giant_component_fraction = 0.0

                    num_components = len(components)

                    # Average component size (excluding giant)
                    if len(components) > 1:
                        non_giant_sizes = sorted([len(c) for c in components])[:-1]
                        avg_component_size = (
                            np.mean(non_giant_sizes) if non_giant_sizes else 0
                        )
                    else:
                        avg_component_size = 0

                else:
                    giant_component_fraction = 0.0
                    num_components = 0
                    avg_component_size = 0

                trial_results.append({
                    "giant_component_fraction": giant_component_fraction,
                    "num_components": num_components,
                    "avg_component_size": avg_component_size
                })

            # Average over trials
            avg_results = {
                "probability": p,
                "giant_component_fraction": np.mean([
                    r["giant_component_fraction"] for r in trial_results
                ]),
                "giant_component_std": np.std([
                    r["giant_component_fraction"] for r in trial_results
                ]),
                "num_components": np.mean([
                    r["num_components"] for r in trial_results
                ]),
                "avg_component_size": np.mean([
                    r["avg_component_size"] for r in trial_results
                ])
            }

            results.append(avg_results)

        # Find percolation threshold
        threshold = RobustnessAnalysis._find_percolation_threshold(results)

        # Theoretical threshold for some graph types
        theoretical_threshold = None

        if not graph.is_directed() and percolation_type == "site":
            avg_degree = np.mean([d for n, d in graph.degree()])
            if avg_degree > 0:
                # Mean-field approximation
                theoretical_threshold = 1.0 / avg_degree

        execution_time = (time.time() - start_time) * 1000

        return {
            "percolation_type": percolation_type,
            "results": results,
            "percolation_threshold": threshold,
            "theoretical_threshold": theoretical_threshold,
            "num_steps": num_steps,
            "num_trials": num_trials,
            "execution_time_ms": execution_time
        }

    @staticmethod
    def _find_percolation_threshold(results: List[Dict]) -> Optional[float]:
        """Find percolation threshold from results."""
        # Look for sharp transition in giant component size
        max_derivative = 0
        threshold = None

        for i in range(1, len(results)):
            p1 = results[i-1]["probability"]
            p2 = results[i]["probability"]
            gc1 = results[i-1]["giant_component_fraction"]
            gc2 = results[i]["giant_component_fraction"]

            if p2 > p1:
                derivative = (gc2 - gc1) / (p2 - p1)

                if derivative > max_derivative:
                    max_derivative = derivative
                    # Threshold is midpoint of steepest increase
                    threshold = (p1 + p2) / 2

        # Alternative: find where giant component reaches 50% of its max
        max_gc = max(r["giant_component_fraction"] for r in results)

        if threshold is None and max_gc > 0:
            for i, r in enumerate(results):
                if r["giant_component_fraction"] >= 0.5 * max_gc:
                    threshold = r["probability"]
                    break

        return threshold

    @staticmethod
    def cascading_failure(
        graph: Union[nx.Graph, nx.DiGraph],
        initial_failures: List[Any],
        failure_model: str = "threshold",
        **params
    ) -> Dict[str, Any]:
        """
        Simulate cascading failures in the network.
        
        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            Input graph with node capacities/loads
        initial_failures : List[Any]
            Initial failed nodes
        failure_model : str
            Model: 'threshold', 'load_redistribution', 'epidemic'
            
        Returns:
        --------
        Dict containing cascade simulation results
        """
        start_time = time.time()

        G = graph.copy()

        # Initialize node states
        failed_nodes = set(initial_failures)
        cascade_sequence = [{
            "step": 0,
            "new_failures": initial_failures,
            "total_failed": len(failed_nodes),
            "fraction_failed": len(failed_nodes) / G.number_of_nodes()
        }]

        if failure_model == "threshold":
            # Threshold model: node fails if too many neighbors have failed
            threshold = params.get("threshold", 0.5)

            step = 0
            while step < 100:  # Prevent infinite loops
                step += 1
                new_failures = []

                for node in G.nodes():
                    if node not in failed_nodes:
                        # Check neighbor failure rate
                        neighbors = list(G.neighbors(node))
                        if neighbors:
                            failed_neighbors = sum(
                                1 for n in neighbors if n in failed_nodes
                            )
                            failure_rate = failed_neighbors / len(neighbors)

                            if failure_rate >= threshold:
                                new_failures.append(node)

                if not new_failures:
                    break

                failed_nodes.update(new_failures)

                cascade_sequence.append({
                    "step": step,
                    "new_failures": new_failures,
                    "total_failed": len(failed_nodes),
                    "fraction_failed": len(failed_nodes) / G.number_of_nodes()
                })

        elif failure_model == "load_redistribution":
            # Load redistribution model
            capacity_factor = params.get("capacity_factor", 1.2)

            # Initialize loads and capacities
            loads = {}
            capacities = {}

            for node in G.nodes():
                # Initial load proportional to degree
                loads[node] = G.degree(node)
                # Capacity is load times factor
                capacities[node] = loads[node] * capacity_factor

            # Remove initial failures
            for node in initial_failures:
                G.remove_node(node)
                del loads[node]
                del capacities[node]

            step = 0
            while step < 100:
                step += 1

                # Redistribute loads (simplified: equally among neighbors)
                new_loads = loads.copy()

                # Check for overloaded nodes
                new_failures = []

                for node in G.nodes():
                    if new_loads[node] > capacities[node]:
                        new_failures.append(node)

                if not new_failures:
                    break

                # Remove failed nodes and redistribute their load
                for node in new_failures:
                    neighbors = list(G.neighbors(node))
                    if neighbors:
                        load_per_neighbor = loads[node] / len(neighbors)
                        for neighbor in neighbors:
                            if neighbor in new_loads:
                                new_loads[neighbor] += load_per_neighbor

                    G.remove_node(node)
                    del loads[node]
                    del capacities[node]
                    failed_nodes.add(node)

                loads = new_loads

                cascade_sequence.append({
                    "step": step,
                    "new_failures": new_failures,
                    "total_failed": len(failed_nodes),
                    "fraction_failed": len(failed_nodes) / graph.number_of_nodes()
                })

        elif failure_model == "epidemic":
            # Epidemic-like spread
            infection_prob = params.get("infection_probability", 0.1)

            step = 0
            infected = set(initial_failures)

            while step < 100:
                step += 1
                new_infections = []

                for node in infected:
                    if node in G:
                        for neighbor in G.neighbors(node):
                            if neighbor not in failed_nodes and random.random() < infection_prob:
                                new_infections.append(neighbor)

                if not new_infections:
                    break

                failed_nodes.update(new_infections)
                infected = set(new_infections)

                cascade_sequence.append({
                    "step": step,
                    "new_failures": new_infections,
                    "total_failed": len(failed_nodes),
                    "fraction_failed": len(failed_nodes) / graph.number_of_nodes()
                })

        else:
            raise ValueError(f"Unknown failure model: {failure_model}")

        # Calculate cascade size statistics
        cascade_size = len(failed_nodes) - len(initial_failures)

        execution_time = (time.time() - start_time) * 1000

        return {
            "failure_model": failure_model,
            "initial_failures": initial_failures,
            "num_initial_failures": len(initial_failures),
            "final_failed_nodes": list(failed_nodes),
            "total_failed": len(failed_nodes),
            "cascade_size": cascade_size,
            "cascade_ratio": cascade_size / len(initial_failures) if initial_failures else 0,
            "fraction_failed": len(failed_nodes) / graph.number_of_nodes(),
            "cascade_sequence": cascade_sequence,
            "num_steps": len(cascade_sequence) - 1,
            "parameters": params,
            "execution_time_ms": execution_time
        }

    @staticmethod
    def network_resilience(
        graph: Union[nx.Graph, nx.DiGraph],
        resilience_metrics: List[str] = None,
        **params
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive network resilience metrics.
        
        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            Input graph
        resilience_metrics : List[str]
            Metrics to calculate
            
        Returns:
        --------
        Dict containing resilience metrics
        """
        if resilience_metrics is None:
            resilience_metrics = [
                "connectivity", "redundancy", "clustering",
                "efficiency", "robustness"
            ]

        results = {}

        if "connectivity" in resilience_metrics:
            # Connectivity resilience
            conn_metrics = {}

            # Algebraic connectivity (Fiedler value)
            if not graph.is_directed() and nx.is_connected(graph) and graph.number_of_nodes() < 1000:
                try:
                    conn_metrics["algebraic_connectivity"] = nx.algebraic_connectivity(graph)
                except:
                    conn_metrics["algebraic_connectivity"] = None

            # Node and edge connectivity
            if graph.number_of_nodes() < 1000:
                if graph.is_directed():
                    conn_metrics["node_connectivity"] = nx.node_connectivity(graph)
                    conn_metrics["edge_connectivity"] = nx.edge_connectivity(graph)
                else:
                    conn_metrics["node_connectivity"] = nx.node_connectivity(graph)
                    conn_metrics["edge_connectivity"] = nx.edge_connectivity(graph)

            results["connectivity"] = conn_metrics

        if "redundancy" in resilience_metrics:
            # Path redundancy
            redundancy_metrics = {}

            # Average number of node-disjoint paths (sample)
            if graph.number_of_nodes() < 100:
                path_counts = []
                nodes = list(graph.nodes())

                for i in range(min(50, len(nodes))):
                    for j in range(i+1, min(i+10, len(nodes))):
                        try:
                            if graph.is_directed():
                                num_paths = len(list(nx.node_disjoint_paths(
                                    graph, nodes[i], nodes[j]
                                )))
                            else:
                                num_paths = nx.node_connectivity(
                                    graph, nodes[i], nodes[j]
                                )
                            path_counts.append(num_paths)
                        except:
                            path_counts.append(0)

                redundancy_metrics["avg_node_disjoint_paths"] = (
                    np.mean(path_counts) if path_counts else 0
                )

            # Cycle basis
            if not graph.is_directed() and graph.number_of_edges() < 1000:
                cycle_basis = nx.cycle_basis(graph)
                redundancy_metrics["num_cycles"] = len(cycle_basis)
                redundancy_metrics["cycle_density"] = (
                    len(cycle_basis) / graph.number_of_edges()
                    if graph.number_of_edges() > 0 else 0
                )

            results["redundancy"] = redundancy_metrics

        if "clustering" in resilience_metrics:
            # Clustering resilience
            cluster_metrics = {}

            if not graph.is_directed():
                cluster_metrics["average_clustering"] = nx.average_clustering(graph)
                cluster_metrics["transitivity"] = nx.transitivity(graph)

                # Clustering distribution
                clustering_values = list(nx.clustering(graph).values())
                if clustering_values:
                    cluster_metrics["clustering_std"] = np.std(clustering_values)
                    cluster_metrics["high_clustering_nodes"] = sum(
                        1 for c in clustering_values if c > 0.5
                    )

            results["clustering"] = cluster_metrics

        if "efficiency" in resilience_metrics:
            # Efficiency metrics
            efficiency_metrics = {}

            if graph.number_of_nodes() < 1000:
                efficiency_metrics["global_efficiency"] = nx.global_efficiency(graph)

                if not graph.is_directed():
                    efficiency_metrics["local_efficiency"] = nx.local_efficiency(graph)
            else:
                # Approximate for large graphs
                efficiency_metrics["global_efficiency"] = (
                    RobustnessAnalysis._approximate_efficiency(graph)
                )

            results["efficiency"] = efficiency_metrics

        if "robustness" in resilience_metrics:
            # Quick robustness assessment
            robustness_metrics = {}

            # Degree distribution characteristics
            degrees = [d for n, d in graph.degree()]
            if degrees:
                degree_variance = np.var(degrees)
                degree_mean = np.mean(degrees)

                # Heterogeneity (high = vulnerable to targeted attacks)
                robustness_metrics["degree_heterogeneity"] = (
                    degree_variance / (degree_mean ** 2) if degree_mean > 0 else 0
                )

                # Degree assortativity (positive = robust)
                robustness_metrics["degree_assortativity"] = (
                    nx.degree_assortativity_coefficient(graph)
                )

            # Core number distribution
            if not graph.is_directed():
                core_numbers = nx.core_number(graph)
                max_core = max(core_numbers.values()) if core_numbers else 0

                robustness_metrics["max_core_number"] = max_core
                robustness_metrics["core_size"] = sum(
                    1 for v, k in core_numbers.items() if k == max_core
                )

            results["robustness"] = robustness_metrics

        # Overall resilience score (normalized combination)
        resilience_score = RobustnessAnalysis._calculate_resilience_score(results)
        results["overall_resilience_score"] = resilience_score

        return results

    @staticmethod
    def _calculate_resilience_score(metrics: Dict[str, Any]) -> float:
        """Calculate overall resilience score from individual metrics."""
        scores = []

        # Connectivity score
        if "connectivity" in metrics:
            conn = metrics["connectivity"]
            if "algebraic_connectivity" in conn and conn["algebraic_connectivity"]:
                # Normalize to [0, 1]
                scores.append(min(1.0, conn["algebraic_connectivity"] / 2.0))

            if "node_connectivity" in conn:
                # Normalize by assuming 5+ is very good
                scores.append(min(1.0, conn["node_connectivity"] / 5.0))

        # Efficiency score
        if "efficiency" in metrics:
            eff = metrics["efficiency"]
            if "global_efficiency" in eff:
                scores.append(eff["global_efficiency"])

        # Clustering score
        if "clustering" in metrics:
            clust = metrics["clustering"]
            if "average_clustering" in clust:
                scores.append(clust["average_clustering"])

        # Robustness score
        if "robustness" in metrics:
            rob = metrics["robustness"]

            # Lower heterogeneity is better
            if "degree_heterogeneity" in rob:
                het_score = 1.0 / (1.0 + rob["degree_heterogeneity"])
                scores.append(het_score)

            # Positive assortativity is better
            if "degree_assortativity" in rob:
                assort_score = (rob["degree_assortativity"] + 1.0) / 2.0
                scores.append(assort_score)

        # Average all scores
        if scores:
            return np.mean(scores)
        else:
            return 0.0
