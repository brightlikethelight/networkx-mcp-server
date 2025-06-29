"""Advanced network flow algorithms and analysis."""

import logging
import time

from collections import defaultdict
from collections import deque
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import networkx as nx


logger = logging.getLogger(__name__)


class NetworkFlow:
    """Advanced network flow algorithms and analysis."""

    @staticmethod
    def max_flow_analysis(
        graph: Union[nx.Graph, nx.DiGraph],
        source: Any,
        sink: Any,
        capacity: str = "capacity",
        algorithm: str = "auto",
        **_params
    ) -> Dict[str, Any]:
        """
        Analyze maximum flow using various algorithms.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            Input graph (will be converted to directed if needed)
        source : node
            Source node
        sink : node
            Sink node (target)
        capacity : str
            Edge attribute for capacity (default: "capacity")
        algorithm : str
            Algorithm to use: 'ford_fulkerson', 'edmonds_karp', 'dinic', 'auto'

        Returns:
        --------
        Dict containing:
            - max_flow_value: Maximum flow value
            - flow_dict: Flow on each edge
            - min_cut: Minimum cut sets
            - algorithm_used: Algorithm name
            - execution_time_ms: Execution time
        """
        start_time = time.time()

        # Convert to directed if needed
        if not graph.is_directed():
            G = graph.to_directed()
        else:
            G = graph.copy()

        # Set default capacities if not present
        for u, v, data in G.edges(data=True):
            if capacity not in data:
                data[capacity] = 1

        # Auto-select algorithm
        if algorithm == "auto":
            num_edges = G.number_of_edges()
            if num_edges < 1000:
                algorithm = "edmonds_karp"  # Good for small graphs
            elif num_edges < 10000:
                algorithm = "dinic"  # Better for medium graphs
            else:
                algorithm = "preflow_push"  # Best for large graphs

        # Run selected algorithm
        if algorithm == "ford_fulkerson":
            flow_value, flow_dict = NetworkFlow._ford_fulkerson(G, source, sink, capacity)
        elif algorithm == "edmonds_karp":
            flow_value, flow_dict = NetworkFlow._edmonds_karp(G, source, sink, capacity)
        elif algorithm == "dinic":
            flow_value, flow_dict = NetworkFlow._dinic(G, source, sink, capacity)
        elif algorithm == "preflow_push":
            # Use NetworkX implementation
            flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity=capacity)
        else:
            # Fallback to NetworkX
            flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity=capacity)

        # Find minimum cut
        min_cut = NetworkFlow._find_min_cut(G, source, sink, flow_dict, capacity)

        execution_time = (time.time() - start_time) * 1000

        # Calculate flow statistics
        flow_edges = []
        total_flow_edges = 0
        for u in flow_dict:
            for v, flow in flow_dict[u].items():
                if flow > 0:
                    flow_edges.append({
                        "source": u,
                        "target": v,
                        "flow": flow,
                        "capacity": G[u][v].get(capacity, 1),
                        "utilization": flow / G[u][v].get(capacity, 1)
                    })
                    total_flow_edges += 1

        return {
            "max_flow_value": flow_value,
            "flow_dict": flow_dict,
            "min_cut": {
                "source_partition": list(min_cut[0]),
                "sink_partition": list(min_cut[1]),
                "cut_edges": NetworkFlow._get_cut_edges(G, min_cut[0], min_cut[1]),
                "cut_capacity": sum(
                    G[u][v].get(capacity, 1)
                    for u in min_cut[0]
                    for v in G[u]
                    if v in min_cut[1]
                )
            },
            "flow_edges": flow_edges,
            "num_flow_edges": total_flow_edges,
            "algorithm_used": algorithm,
            "execution_time_ms": execution_time
        }

    @staticmethod
    def _ford_fulkerson(
        graph: nx.DiGraph,
        source: Any,
        sink: Any,
        capacity: str
    ) -> Tuple[float, Dict]:
        """Ford-Fulkerson algorithm implementation."""
        # Create residual graph
        residual = graph.copy()

        # Initialize flow
        flow_dict = defaultdict(lambda: defaultdict(float))

        def find_augmenting_path_dfs(residual, source, sink, path, visited):
            """DFS to find augmenting path."""
            if source == sink:
                return path

            visited.add(source)

            for neighbor in residual[source]:
                if neighbor not in visited:
                    edge_capacity = residual[source][neighbor].get(capacity, 0)
                    if edge_capacity > 0:
                        result = find_augmenting_path_dfs(
                            residual, neighbor, sink,
                            [*path, (source, neighbor, edge_capacity)],
                            visited
                        )
                        if result:
                            return result

            return None

        max_flow = 0

        while True:
            # Find augmenting path
            path = find_augmenting_path_dfs(residual, source, sink, [], set())

            if not path:
                break

            # Find minimum capacity along path
            path_flow = min(edge[2] for edge in path)

            # Update residual capacities
            for u, v, _ in path:
                # Forward edge
                residual[u][v][capacity] -= path_flow
                if residual[u][v][capacity] == 0:
                    residual.remove_edge(u, v)

                # Backward edge
                if residual.has_edge(v, u):
                    residual[v][u][capacity] += path_flow
                else:
                    residual.add_edge(v, u, **{capacity: path_flow})

                # Update flow
                flow_dict[u][v] += path_flow
                flow_dict[v][u] -= path_flow

            max_flow += path_flow

        # Clean up flow dict
        clean_flow = {}
        for u in flow_dict:
            clean_flow[u] = {}
            for v in flow_dict[u]:
                if flow_dict[u][v] > 0:
                    clean_flow[u][v] = flow_dict[u][v]

        return max_flow, clean_flow

    @staticmethod
    def _edmonds_karp(
        graph: nx.DiGraph,
        source: Any,
        sink: Any,
        capacity: str
    ) -> Tuple[float, Dict]:
        """Edmonds-Karp algorithm (BFS-based Ford-Fulkerson)."""
        # Create residual graph
        residual = graph.copy()

        # Add reverse edges with 0 capacity
        for u, v in list(graph.edges()):
            if not residual.has_edge(v, u):
                residual.add_edge(v, u, **{capacity: 0})

        # Initialize flow
        flow_dict = defaultdict(lambda: defaultdict(float))

        def bfs_augmenting_path(residual, source, sink):
            """BFS to find shortest augmenting path."""
            parent = {source: None}
            visited = {source}
            queue = deque([source])

            while queue:
                u = queue.popleft()

                for v in residual[u]:
                    if v not in visited and residual[u][v].get(capacity, 0) > 0:
                        visited.add(v)
                        parent[v] = u
                        queue.append(v)

                        if v == sink:
                            # Reconstruct path
                            path = []
                            current = sink
                            while parent[current] is not None:
                                prev = parent[current]
                                path.append((prev, current))
                                current = prev
                            return list(reversed(path))

            return None

        max_flow = 0

        while True:
            # Find augmenting path using BFS
            path = bfs_augmenting_path(residual, source, sink)

            if not path:
                break

            # Find minimum capacity along path
            path_flow = float("inf")
            for u, v in path:
                path_flow = min(path_flow, residual[u][v].get(capacity, 0))

            # Update residual capacities and flow
            for u, v in path:
                residual[u][v][capacity] -= path_flow
                residual[v][u][capacity] += path_flow
                flow_dict[u][v] += path_flow
                flow_dict[v][u] -= path_flow

            max_flow += path_flow

        # Clean up flow dict
        clean_flow = {}
        for u in flow_dict:
            clean_flow[u] = {}
            for v in flow_dict[u]:
                if flow_dict[u][v] > 0:
                    clean_flow[u][v] = flow_dict[u][v]

        return max_flow, clean_flow

    @staticmethod
    def _dinic(
        graph: nx.DiGraph,
        source: Any,
        sink: Any,
        capacity: str
    ) -> Tuple[float, Dict]:
        """Dinic's algorithm with level graph optimization."""
        # Create residual graph
        residual = graph.copy()

        # Add reverse edges
        for u, v in list(graph.edges()):
            if not residual.has_edge(v, u):
                residual.add_edge(v, u, **{capacity: 0})

        flow_dict = defaultdict(lambda: defaultdict(float))

        def bfs_level_graph(residual, source, sink):
            """BFS to create level graph."""
            level = {source: 0}
            queue = deque([source])

            while queue:
                u = queue.popleft()

                for v in residual[u]:
                    if v not in level and residual[u][v].get(capacity, 0) > 0:
                        level[v] = level[u] + 1
                        queue.append(v)

            return level if sink in level else None

        def dfs_blocking_flow(residual, u, sink, flow, level):
            """DFS to find blocking flow."""
            if u == sink:
                return flow

            for v in list(residual[u]):
                if (v in level and
                    level[v] == level[u] + 1 and
                    residual[u][v].get(capacity, 0) > 0):

                    min_flow = min(flow, residual[u][v].get(capacity, 0))
                    result = dfs_blocking_flow(residual, v, sink, min_flow, level)

                    if result > 0:
                        residual[u][v][capacity] -= result
                        residual[v][u][capacity] += result
                        flow_dict[u][v] += result
                        flow_dict[v][u] -= result
                        return result

            return 0

        max_flow = 0

        while True:
            # Build level graph
            level = bfs_level_graph(residual, source, sink)

            if not level:
                break

            # Find blocking flows
            while True:
                flow = dfs_blocking_flow(residual, source, sink, float("inf"), level)
                if flow == 0:
                    break
                max_flow += flow

        # Clean up flow dict
        clean_flow = {}
        for u in flow_dict:
            clean_flow[u] = {}
            for v in flow_dict[u]:
                if flow_dict[u][v] > 0:
                    clean_flow[u][v] = flow_dict[u][v]

        return max_flow, clean_flow

    @staticmethod
    def _find_min_cut(
        graph: nx.DiGraph,
        source: Any,
        _sink: Any,
        flow_dict: Dict,
        capacity: str
    ) -> Tuple[Set, Set]:
        """Find minimum cut given maximum flow."""
        # Build residual graph from flow
        residual = nx.DiGraph()

        for u, v, data in graph.edges(data=True):
            cap = data.get(capacity, 1)
            flow = flow_dict.get(u, {}).get(v, 0)

            if cap - flow > 0:
                residual.add_edge(u, v, **{capacity: cap - flow})

            # Backward edge
            if flow > 0:
                residual.add_edge(v, u, **{capacity: flow})

        # Find reachable nodes from source in residual graph
        reachable = set()
        queue = deque([source])
        reachable.add(source)

        while queue:
            u = queue.popleft()

            for v in residual.neighbors(u):
                if v not in reachable:
                    reachable.add(v)
                    queue.append(v)

        # Min cut is (reachable, non-reachable)
        non_reachable = set(graph.nodes()) - reachable

        return reachable, non_reachable

    @staticmethod
    def _get_cut_edges(
        graph: nx.DiGraph,
        source_set: Set,
        sink_set: Set
    ) -> List[Tuple]:
        """Get edges in the cut."""
        cut_edges = []

        for u in source_set:
            for v in graph.neighbors(u):
                if v in sink_set:
                    cut_edges.append((u, v))

        return cut_edges

    @staticmethod
    def min_cut_analysis(
        graph: Union[nx.Graph, nx.DiGraph],
        source: Optional[Any] = None,
        sink: Optional[Any] = None,
        capacity: str = "capacity",
        **params
    ) -> Dict[str, Any]:
        """
        Find minimum cut sets in the graph.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            Input graph
        source, sink : nodes, optional
            If provided, find s-t cut. Otherwise, find global minimum cut

        Returns:
        --------
        Dict containing cut analysis
        """
        start_time = time.time()

        if source and sink:
            # s-t cut
            result = NetworkFlow.max_flow_analysis(
                graph, source, sink, capacity=capacity, **params
            )

            return {
                "cut_type": "s-t_cut",
                "source": source,
                "sink": sink,
                "min_cut_value": result["max_flow_value"],
                "source_partition": result["min_cut"]["source_partition"],
                "sink_partition": result["min_cut"]["sink_partition"],
                "cut_edges": result["min_cut"]["cut_edges"],
                "execution_time_ms": result["execution_time_ms"]
            }
        else:
            # Global minimum cut using Stoer-Wagner
            if graph.is_directed():
                G = graph.to_undirected()
            else:
                G = graph.copy()

            try:
                cut_value, partition = nx.stoer_wagner(G, weight=capacity)

                execution_time = (time.time() - start_time) * 1000

                # Get cut edges
                cut_edges = []
                for u in partition[0]:
                    for v in G[u]:
                        if v in partition[1]:
                            cut_edges.append((u, v))

                return {
                    "cut_type": "global_min_cut",
                    "min_cut_value": cut_value,
                    "partition_1": list(partition[0]),
                    "partition_2": list(partition[1]),
                    "cut_edges": cut_edges,
                    "execution_time_ms": execution_time
                }

            except nx.NetworkXError:
                # Fallback: find minimum s-t cut over all pairs
                nodes = list(graph.nodes())
                min_cut_value = float("inf")
                best_cut = None

                for i, s in enumerate(nodes[:-1]):
                    for t in nodes[i+1:]:
                        try:
                            flow_value = nx.maximum_flow_value(
                                graph, s, t, capacity=capacity
                            )
                            if flow_value < min_cut_value:
                                min_cut_value = flow_value
                                best_cut = (s, t)
                        except (nx.NetworkXError, ValueError) as e:
                            logger.debug(f"Failed to compute flow between {s} and {t}: {e}")
                            continue

                if best_cut:
                    return NetworkFlow.min_cut_analysis(
                        graph, best_cut[0], best_cut[1], capacity=capacity
                    )
                else:
                    return {
                        "error": "Could not find minimum cut",
                        "execution_time_ms": (time.time() - start_time) * 1000
                    }

    @staticmethod
    def multi_commodity_flow(
        graph: Union[nx.Graph, nx.DiGraph],
        demands: List[Tuple[Any, Any, float]],
        capacity: str = "capacity",
        **_params
    ) -> Dict[str, Any]:
        """
        Solve multi-commodity flow problem.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            Input graph
        demands : List[Tuple[source, sink, demand]]
            List of (source, sink, demand) tuples
        capacity : str
            Edge capacity attribute

        Returns:
        --------
        Dict containing flow solution or infeasibility proof
        """
        start_time = time.time()

        # Convert to directed if needed
        if not graph.is_directed():
            G = graph.to_directed()
        else:
            G = graph.copy()

        # Check feasibility using linear programming formulation
        # Simplified: Use sequential maximum flows (not optimal but practical)

        flows = {}
        total_flow = 0
        remaining_capacity = {}

        # Initialize remaining capacities
        for u, v, data in G.edges(data=True):
            remaining_capacity[(u, v)] = data.get(capacity, 1)

        # Process each commodity sequentially (greedy approach)
        for i, (source, sink, demand) in enumerate(demands):
            # Create temporary graph with remaining capacities
            temp_graph = nx.DiGraph()
            for (u, v), cap in remaining_capacity.items():
                if cap > 0:
                    temp_graph.add_edge(u, v, **{capacity: cap})

            # Find maximum flow for this commodity
            try:
                flow_value, flow_dict = nx.maximum_flow(
                    temp_graph, source, sink, capacity=capacity
                )

                # Scale flow if it exceeds demand
                if flow_value > demand:
                    scale_factor = demand / flow_value
                    for u in flow_dict:
                        for v in flow_dict[u]:
                            flow_dict[u][v] *= scale_factor
                    flow_value = demand

                # Update remaining capacities
                for u in flow_dict:
                    for v, flow in flow_dict[u].items():
                        if flow > 0:
                            remaining_capacity[(u, v)] -= flow

                flows[f"commodity_{i}"] = {
                    "source": source,
                    "sink": sink,
                    "demand": demand,
                    "flow_value": flow_value,
                    "flow_dict": flow_dict,
                    "satisfied": flow_value >= demand * 0.99  # Allow small numerical error
                }

                total_flow += flow_value

            except nx.NetworkXError:
                flows[f"commodity_{i}"] = {
                    "source": source,
                    "sink": sink,
                    "demand": demand,
                    "flow_value": 0,
                    "flow_dict": {},
                    "satisfied": False
                }

        execution_time = (time.time() - start_time) * 1000

        # Calculate statistics
        num_satisfied = sum(1 for f in flows.values() if f["satisfied"])
        total_demand = sum(d[2] for d in demands)

        return {
            "feasible": num_satisfied == len(demands),
            "flows": flows,
            "total_flow": total_flow,
            "total_demand": total_demand,
            "satisfaction_rate": total_flow / total_demand if total_demand > 0 else 0,
            "num_commodities": len(demands),
            "num_satisfied": num_satisfied,
            "execution_time_ms": execution_time,
            "method": "sequential_greedy"
        }

    @staticmethod
    def flow_decomposition(
        _graph: Union[nx.Graph, nx.DiGraph],
        flow_dict: Dict[Any, Dict[Any, float]],
        source: Any,
        sink: Any
    ) -> Dict[str, Any]:
        """
        Decompose a flow into path flows.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            Input graph
        flow_dict : Dict
            Flow dictionary from max flow computation
        source, sink : nodes
            Source and sink nodes

        Returns:
        --------
        Dict containing path decomposition
        """
        # Create residual flow graph
        flow_graph = nx.DiGraph()

        for u in flow_dict:
            for v, flow in flow_dict[u].items():
                if flow > 0:
                    flow_graph.add_edge(u, v, flow=flow)

        paths = []
        total_path_flow = 0

        # Find paths from source to sink
        while flow_graph.has_node(source) and flow_graph.has_node(sink):
            # Find a path using DFS
            path = NetworkFlow._find_flow_path(flow_graph, source, sink)

            if not path:
                break

            # Find minimum flow along path
            min_flow = float("inf")
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if flow_graph.has_edge(u, v):
                    min_flow = min(min_flow, flow_graph[u][v]["flow"])

            if min_flow == float("inf") or min_flow <= 0:
                break

            # Record path
            paths.append({
                "path": path,
                "flow": min_flow,
                "length": len(path) - 1
            })
            total_path_flow += min_flow

            # Update residual flows
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                flow_graph[u][v]["flow"] -= min_flow
                if flow_graph[u][v]["flow"] <= 0:
                    flow_graph.remove_edge(u, v)

        # Find any remaining cycles
        cycles = []
        while flow_graph.number_of_edges() > 0:
            # Find a cycle
            cycle = NetworkFlow._find_flow_cycle(flow_graph)

            if not cycle:
                break

            # Find minimum flow in cycle
            min_flow = float("inf")
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if flow_graph.has_edge(u, v):
                    min_flow = min(min_flow, flow_graph[u][v]["flow"])

            if min_flow == float("inf") or min_flow <= 0:
                break

            cycles.append({
                "cycle": cycle,
                "flow": min_flow,
                "length": len(cycle)
            })

            # Update residual flows
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                flow_graph[u][v]["flow"] -= min_flow
                if flow_graph[u][v]["flow"] <= 0:
                    flow_graph.remove_edge(u, v)

        return {
            "paths": paths,
            "num_paths": len(paths),
            "total_path_flow": total_path_flow,
            "cycles": cycles,
            "num_cycles": len(cycles),
            "path_length_distribution": dict(NetworkFlow._count_by_key(paths, "length").items())
        }

    @staticmethod
    def _find_flow_path(flow_graph: nx.DiGraph, source: Any, sink: Any) -> Optional[List]:
        """Find a path from source to sink in flow graph using DFS."""
        visited = set()
        path = []

        def dfs(node):
            if node == sink:
                path.append(node)
                return True

            visited.add(node)
            path.append(node)

            for neighbor in flow_graph.neighbors(node):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True

            path.pop()
            return False

        if dfs(source):
            return path
        return None

    @staticmethod
    def _find_flow_cycle(flow_graph: nx.DiGraph) -> Optional[List]:
        """Find a cycle in flow graph."""
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in flow_graph.neighbors(node):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:]

            path.pop()
            rec_stack.remove(node)
            return False

        for node in flow_graph.nodes():
            if node not in visited:
                result = dfs(node)
                if result and isinstance(result, list):
                    return result

        return None

    @staticmethod
    def _count_by_key(items: List[Dict], key: str) -> Dict:
        """Count items by a specific key."""
        counts = defaultdict(int)
        for item in items:
            counts[item[key]] += 1
        return dict(counts)

    @staticmethod
    def circulation_analysis(
        graph: Union[nx.Graph, nx.DiGraph],
        demands: Dict[Any, float],
        capacity: str = "capacity",
        **_params
    ) -> Dict[str, Any]:
        """
        Check feasibility of circulation with demands.

        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            Input graph
        demands : Dict[node, demand]
            Node demands (negative for supply, positive for demand)
        capacity : str
            Edge capacity attribute

        Returns:
        --------
        Dict containing feasibility and circulation if it exists
        """
        start_time = time.time()

        # Convert to directed if needed
        if not graph.is_directed():
            G = graph.to_directed()
        else:
            G = graph.copy()

        # Check if demands sum to zero (necessary condition)
        total_demand = sum(demands.values())
        if abs(total_demand) > 1e-10:
            return {
                "feasible": False,
                "reason": "Demands do not sum to zero",
                "total_demand": total_demand,
                "execution_time_ms": (time.time() - start_time) * 1000
            }

        # Create auxiliary graph with super source and sink
        aux_graph = G.copy()
        super_source = "SUPER_SOURCE"
        super_sink = "SUPER_SINK"

        aux_graph.add_node(super_source)
        aux_graph.add_node(super_sink)

        # Add edges from super source to supply nodes
        # Add edges from demand nodes to super sink
        for node, demand in demands.items():
            if demand < 0:  # Supply node
                aux_graph.add_edge(super_source, node, **{capacity: -demand})
            elif demand > 0:  # Demand node
                aux_graph.add_edge(node, super_sink, **{capacity: demand})

        # Find maximum flow from super source to super sink
        try:
            max_flow_value, flow_dict = nx.maximum_flow(
                aux_graph, super_source, super_sink, capacity=capacity
            )

            # Check if all demands are satisfied
            total_supply = sum(-d for d in demands.values() if d < 0)

            feasible = abs(max_flow_value - total_supply) < 1e-10

            if feasible:
                # Extract circulation from flow
                circulation = {}
                for u in flow_dict:
                    if u not in [super_source, super_sink]:
                        circulation[u] = {}
                        for v, flow in flow_dict[u].items():
                            if v not in [super_source, super_sink] and flow > 0:
                                circulation[u][v] = flow

                # Verify circulation
                verification = NetworkFlow._verify_circulation(
                    G, circulation, demands, capacity
                )

                execution_time = (time.time() - start_time) * 1000

                return {
                    "feasible": True,
                    "circulation": circulation,
                    "total_flow": sum(
                        sum(flows.values()) for flows in circulation.values()
                    ),
                    "verification": verification,
                    "execution_time_ms": execution_time
                }
            else:
                execution_time = (time.time() - start_time) * 1000

                return {
                    "feasible": False,
                    "reason": "Cannot satisfy all demands",
                    "max_flow_achieved": max_flow_value,
                    "required_flow": total_supply,
                    "execution_time_ms": execution_time
                }

        except nx.NetworkXError as e:
            return {
                "feasible": False,
                "reason": f"Flow computation failed: {e!s}",
                "execution_time_ms": (time.time() - start_time) * 1000
            }

    @staticmethod
    def _verify_circulation(
        graph: nx.DiGraph,
        circulation: Dict,
        demands: Dict,
        capacity: str
    ) -> Dict[str, Any]:
        """Verify that a circulation satisfies all constraints."""
        violations = []

        # Check flow conservation at each node
        for node in graph.nodes():
            inflow = sum(
                circulation.get(u, {}).get(node, 0)
                for u in graph.predecessors(node)
            )
            outflow = sum(
                circulation.get(node, {}).get(v, 0)
                for v in graph.successors(node)
            )

            net_flow = inflow - outflow
            required_demand = demands.get(node, 0)

            if abs(net_flow - required_demand) > 1e-10:
                violations.append({
                    "node": node,
                    "type": "flow_conservation",
                    "net_flow": net_flow,
                    "required_demand": required_demand,
                    "difference": net_flow - required_demand
                })

        # Check capacity constraints
        for u, v, data in graph.edges(data=True):
            flow = circulation.get(u, {}).get(v, 0)
            cap = data.get(capacity, float("inf"))

            if flow > cap + 1e-10:
                violations.append({
                    "edge": (u, v),
                    "type": "capacity_violation",
                    "flow": flow,
                    "capacity": cap,
                    "excess": flow - cap
                })

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "num_violations": len(violations)
        }
