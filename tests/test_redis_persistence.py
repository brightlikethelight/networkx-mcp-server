#!/usr/bin/env python3
"""Test Redis persistence and data survival across restarts."""

import asyncio
import sys
import time
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRedisPersistence:
    """Test Redis-based persistence functionality."""

    def __init__(self):
        self.server_process = None

    async def test_data_survives_restart(self):
        """Verify graphs persist across server restarts."""

        try:
            # Import server components
            import add_persistence
            import security_patches
            from networkx_mcp.server import graph_manager

            # Apply security and persistence
            security_patches.apply_critical_patches()
            storage = add_persistence.patch_graph_manager_with_persistence()

            if not storage:
                msg = "Failed to initialize persistence layer"
                raise Exception(msg)

            # Phase 1: Create test data
            graphs_created = {}

            for i in range(5):
                graph_id = f"persist_test_{i}"

                # Create graph
                result = graph_manager.create_graph(graph_id, "DiGraph")
                if not result.get("created"):
                    msg = f"Failed to create graph {graph_id}"
                    raise Exception(msg)

                # Add nodes and edges
                nodes = [f"N{j}" for j in range(10)]
                graph_manager.add_nodes_from(graph_id, nodes)

                # Add some edges
                edges = [(f"N{j}", f"N{j + 1}") for j in range(9)]
                graph_manager.add_edges_from(graph_id, edges)

                # Store expected data
                info = graph_manager.get_graph_info(graph_id)
                graphs_created[graph_id] = {
                    "nodes": info["num_nodes"],
                    "edges": info["num_edges"],
                    "type": info["graph_type"],
                }

            # Phase 2: Simulate restart by clearing memory

            # Clear the in-memory graphs
            graph_manager.graphs.copy()
            graph_manager.metadata.copy()

            graph_manager.graphs.clear()
            graph_manager.metadata.clear()

            # Phase 3: Verify data recovery

            # Re-apply persistence layer
            storage = add_persistence.patch_graph_manager_with_persistence()

            # Check if we can load graphs from persistent storage
            recovered_count = 0
            for graph_id, expected_data in graphs_created.items():
                # Try to load from persistent storage
                if hasattr(graph_manager, "_storage"):
                    stored_graph = graph_manager._storage.load_graph(
                        graph_manager._default_user, graph_id
                    )

                    if stored_graph is not None:
                        # Restore to memory
                        graph_manager.graphs[graph_id] = stored_graph
                        graph_manager.metadata[graph_id] = {
                            "created_at": "recovered",
                            "graph_type": type(stored_graph).__name__,
                            "attributes": {},
                        }

                        # Verify data integrity
                        info = graph_manager.get_graph_info(graph_id)

                        if (
                            info["num_nodes"] == expected_data["nodes"]
                            and info["num_edges"] == expected_data["edges"]
                        ):
                            recovered_count += 1
                        else:
                            pass
                    else:
                        pass
                else:
                    pass

            # Final validation

            if recovered_count == len(graphs_created):
                return True
            else:
                return False

        except Exception:
            return False

    def check_redis_availability(self):
        """Check if Redis is available and configured properly."""

        try:
            import redis

            # Try to connect to Redis
            r = redis.Redis(host="localhost", port=6379, decode_responses=True)
            r.ping()

            # Test basic operations
            r.set("test_key", "test_value")
            value = r.get("test_key")
            r.delete("test_key")

            if value == "test_value":
                return True
            else:
                return False

        except ImportError:
            return False
        except Exception:
            return False

    async def test_concurrent_access_safety(self):
        """Test that concurrent access to persistence is safe."""

        try:
            import add_persistence
            import security_patches
            from networkx_mcp.server import graph_manager

            # Apply security and persistence
            security_patches.apply_critical_patches()
            add_persistence.patch_graph_manager_with_persistence()

            async def worker(worker_id: int, operations: int):
                """Worker function that creates/modifies graphs."""
                for op in range(operations):
                    graph_id = f"worker_{worker_id}_graph_{op}"

                    # Create graph
                    graph_manager.create_graph(graph_id, "Graph")

                    # Add some nodes
                    nodes = [f"W{worker_id}_N{i}" for i in range(10)]
                    graph_manager.add_nodes_from(graph_id, nodes)

                    # Delete graph
                    graph_manager.delete_graph(graph_id)

            # Run multiple workers concurrently
            workers = [worker(i, 10) for i in range(5)]

            start_time = time.time()
            await asyncio.gather(*workers)
            time.time() - start_time

            return True

        except Exception:
            return False


async def main():
    """Run all Redis persistence tests."""

    test_suite = TestRedisPersistence()

    # Check 1: Redis availability
    redis_available = test_suite.check_redis_availability()

    # Check 2: Data persistence
    persistence_works = await test_suite.test_data_survives_restart()

    # Check 3: Concurrent safety
    concurrent_safe = await test_suite.test_concurrent_access_safety()

    # Summary

    checks = [
        ("Redis available", redis_available),
        ("Data survives restart", persistence_works),
        ("Concurrent access safe", concurrent_safe),
    ]

    passed = 0
    for _check_name, result in checks:
        if result:
            passed += 1

    if passed == len(checks):
        pass
    else:
        pass

    return passed == len(checks)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
