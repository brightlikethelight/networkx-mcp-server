"""Test thread safety of GraphManager."""

import concurrent.futures
import random
import threading
import time

from networkx_mcp.core.graph_operations import GraphManager


class TestGraphManagerThreadSafety:
    """Test thread safety of GraphManager operations."""

    def test_concurrent_graph_creation(self):
        """Test creating multiple graphs concurrently."""
        manager = GraphManager()
        num_threads = 10
        graphs_per_thread = 5

        def create_graphs(thread_id: int) -> list[str]:
            """Create multiple graphs in a thread."""
            created = []
            for i in range(graphs_per_thread):
                graph_id = f"thread_{thread_id}_graph_{i}"
                try:
                    manager.create_graph(graph_id)
                    created.append(graph_id)
                except Exception:
                    pass  # Graph may already exist from another thread
            return created

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(create_graphs, i) for i in range(num_threads)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Verify total graphs created
        total_created = sum(len(r) for r in results)
        assert total_created == len(manager.graphs)
        assert total_created <= num_threads * graphs_per_thread

    def test_concurrent_node_operations(self):
        """Test concurrent node additions and deletions."""
        manager = GraphManager()
        graph_id = "concurrent_test"
        manager.create_graph(graph_id)

        num_threads = 20
        nodes_per_thread = 100

        def add_nodes(thread_id: int) -> None:
            """Add nodes in a thread."""
            for i in range(nodes_per_thread):
                node_id = f"thread_{thread_id}_node_{i}"
                manager.add_node(graph_id, node_id, thread=thread_id)

        # Add nodes concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(add_nodes, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        # Verify all nodes were added
        graph = manager.get_graph(graph_id)
        assert graph.number_of_nodes() == num_threads * nodes_per_thread

    def test_concurrent_edge_operations(self):
        """Test concurrent edge additions."""
        manager = GraphManager()
        graph_id = "edge_test"
        manager.create_graph(graph_id)

        # Pre-populate with nodes
        num_nodes = 100
        for i in range(num_nodes):
            manager.add_node(graph_id, f"node_{i}")

        num_threads = 10
        edges_per_thread = 50

        def add_edges(thread_id: int) -> None:
            """Add random edges in a thread."""
            random.seed(thread_id)
            for _ in range(edges_per_thread):
                source = f"node_{random.randint(0, num_nodes-1)}"
                target = f"node_{random.randint(0, num_nodes-1)}"
                if source != target:
                    try:
                        manager.add_edge(graph_id, source, target, thread=thread_id)
                    except Exception:
                        pass  # Edge may already exist

        # Add edges concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(add_edges, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        # Verify edges were added (some may be duplicates)
        graph = manager.get_graph(graph_id)
        assert graph.number_of_edges() > 0
        assert graph.number_of_edges() <= num_threads * edges_per_thread

    def test_concurrent_read_write_operations(self):
        """Test concurrent reads and writes don't corrupt data."""
        manager = GraphManager()
        graph_id = "read_write_test"
        manager.create_graph(graph_id)

        # Add initial nodes
        for i in range(10):
            manager.add_node(graph_id, f"node_{i}")

        errors = []
        operations_completed = threading.Event()

        def reader_thread() -> None:
            """Continuously read graph info."""
            while not operations_completed.is_set():
                try:
                    info = manager.get_graph_info(graph_id)
                    nodes = manager.list_graphs()
                    # Verify data consistency
                    assert isinstance(info, dict)
                    assert isinstance(nodes, list)
                except Exception as e:
                    errors.append(f"Reader error: {e}")
                time.sleep(0.001)

        def writer_thread(thread_id: int) -> None:
            """Continuously modify the graph."""
            for i in range(50):
                try:
                    # Add node
                    node_id = f"writer_{thread_id}_node_{i}"
                    manager.add_node(graph_id, node_id)

                    # Add edge
                    if i > 0:
                        prev_node = f"writer_{thread_id}_node_{i-1}"
                        manager.add_edge(graph_id, prev_node, node_id)

                    # Get info (mixed read)
                    info = manager.get_graph_info(graph_id)
                    assert info["graph_id"] == graph_id

                except Exception as e:
                    errors.append(f"Writer {thread_id} error: {e}")
                time.sleep(0.001)

        # Start reader threads
        readers = []
        for _ in range(3):
            t = threading.Thread(target=reader_thread)
            t.start()
            readers.append(t)

        # Start writer threads
        writers = []
        for i in range(5):
            t = threading.Thread(target=writer_thread, args=(i,))
            t.start()
            writers.append(t)

        # Wait for writers to complete
        for t in writers:
            t.join()

        # Signal readers to stop
        operations_completed.set()
        for t in readers:
            t.join()

        # Check for errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"

        # Verify final graph state
        graph = manager.get_graph(graph_id)
        assert graph.number_of_nodes() > 10  # Initial + added nodes

    def test_concurrent_graph_deletion(self):
        """Test concurrent graph deletion is safe."""
        manager = GraphManager()

        # Create graphs
        for i in range(20):
            manager.create_graph(f"delete_test_{i}")

        deleted_count = threading.Lock()
        deleted = 0

        def delete_graphs(start: int, end: int) -> None:
            """Delete a range of graphs."""
            nonlocal deleted
            for i in range(start, end):
                graph_id = f"delete_test_{i}"
                try:
                    manager.delete_graph(graph_id)
                    with deleted_count:
                        deleted += 1
                except Exception:
                    pass  # Graph may already be deleted

        # Delete graphs concurrently from multiple threads
        threads = []
        for i in range(4):
            # Overlapping ranges to test race conditions
            start = i * 5
            end = min(start + 10, 20)
            t = threading.Thread(target=delete_graphs, args=(start, end))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # All graphs should be deleted
        remaining = [gid for gid in manager.graphs if gid.startswith("delete_test_")]
        assert len(remaining) == 0
        assert deleted == 20

    def test_stress_test_mixed_operations(self):
        """Stress test with many concurrent mixed operations."""
        manager = GraphManager()
        num_threads = 50
        operations_per_thread = 100

        def stress_operations(thread_id: int) -> dict[str, int]:
            """Perform random operations."""
            counts: dict[str, int] = {
                "creates": 0,
                "adds": 0,
                "reads": 0,
                "deletes": 0,
            }

            random.seed(thread_id)
            graph_id = f"stress_{thread_id}"

            for _ in range(operations_per_thread):
                op = random.choice(["create", "add", "read", "delete"])

                try:
                    if op == "create":
                        manager.create_graph(graph_id)
                        counts["creates"] += 1
                    elif op == "add":
                        if graph_id in manager.graphs:
                            node = f"node_{random.randint(0, 1000)}"
                            manager.add_node(graph_id, node)
                            counts["adds"] += 1
                    elif op == "read":
                        if graph_id in manager.graphs:
                            info = manager.get_graph_info(graph_id)
                            assert isinstance(info, dict)
                            counts["reads"] += 1
                    elif op == "delete":
                        if graph_id in manager.graphs:
                            manager.delete_graph(graph_id)
                            counts["deletes"] += 1
                except Exception:
                    pass  # Expected in concurrent environment

            return counts

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(stress_operations, i) for i in range(num_threads)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Verify operations completed without deadlock
        total_ops = sum(sum(r.values()) for r in results)
        assert total_ops > 0, "No operations completed - possible deadlock"

        # Verify manager is in valid state
        assert isinstance(manager.graphs, dict)
        assert isinstance(manager.metadata, dict)
        assert len(manager.graphs) == len(manager.metadata)
