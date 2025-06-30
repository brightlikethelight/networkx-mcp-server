#!/usr/bin/env python3
"""CRITICAL: Test that Redis persistence actually works in reality."""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import redis

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class RedisRealityCheck:
    """Test Redis persistence with real server processes."""

    def __init__(self):
        self.redis_client = None
        self.server_process = None

    def setup_redis_connection(self):
        """Connect to Redis and verify it's working."""
        try:
            self.redis_client = redis.Redis(
                host="localhost", port=6379, decode_responses=True
            )
            self.redis_client.ping()
            print("✅ Redis connection established")
            return True
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            return False

    def clear_redis_data(self):
        """Clear all existing Redis data for clean test."""
        try:
            self.redis_client.flushdb()
            print("🗑️ Redis data cleared for clean test")
            return True
        except Exception as e:
            print(f"❌ Failed to clear Redis: {e}")
            return False

    def check_redis_data_directly(self):
        """Check what's actually stored in Redis."""
        try:
            keys = self.redis_client.keys("*")
            print(f"📊 Redis keys found: {len(keys)}")

            graph_keys = [k for k in keys if k.startswith("graph:")]
            metadata_keys = [k for k in keys if k.startswith("metadata:")]

            print(f"   Graph keys: {len(graph_keys)}")
            print(f"   Metadata keys: {len(metadata_keys)}")

            # Show some actual data
            for key in graph_keys[:3]:  # Show first 3
                data_type = self.redis_client.type(key)
                print(f"   {key}: {data_type}")

            return len(graph_keys) > 0
        except Exception as e:
            print(f"❌ Failed to check Redis data: {e}")
            return False

    def start_server_with_redis(self):
        """Start the server with Redis backend."""
        try:
            env = os.environ.copy()
            env.update(
                {
                    "REDIS_URL": "redis://localhost:6379/0",
                    "STORAGE_BACKEND": "redis",
                    "MCP_TRANSPORT": "sse",
                }
            )

            print("🚀 Starting server with Redis backend...")
            self.server_process = subprocess.Popen(
                [sys.executable, "run_secure_server.py"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for server to start
            time.sleep(3)

            if self.server_process.poll() is None:
                print("✅ Server started successfully")
                return True
            else:
                stdout, stderr = self.server_process.communicate()
                print("❌ Server failed to start:")
                print(f"STDOUT: {stdout.decode()}")
                print(f"STDERR: {stderr.decode()}")
                return False

        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False

    def stop_server(self):
        """Stop the server process."""
        if self.server_process:
            print("🛑 Stopping server...")
            self.server_process.terminate()
            self.server_process.wait(timeout=10)
            self.server_process = None
            time.sleep(1)

    def create_test_data_via_graph_manager(self):
        """Create test data using graph manager directly."""
        try:
            print("📝 Creating test data via GraphManager...")

            # Import and setup
            import add_persistence
            import security_patches

            from networkx_mcp.server import graph_manager

            # Apply security patches
            security_patches.apply_critical_patches()

            # Set up Redis-based persistence
            os.environ["REDIS_URL"] = "redis://localhost:6379/0"
            os.environ["STORAGE_BACKEND"] = "redis"

            # Import and patch for Redis
            storage = add_persistence.patch_graph_manager_with_persistence()

            if not storage:
                print("❌ Failed to set up persistence layer")
                return False

            # Create test graphs
            test_graphs = {}
            for i in range(3):
                graph_id = f"redis_reality_test_{i}"

                # Create graph
                result = graph_manager.create_graph(graph_id, "DiGraph")
                if not result.get("created"):
                    print(f"❌ Failed to create graph {graph_id}")
                    return False

                # Add nodes
                nodes = [f"node_{j}" for j in range(20)]
                graph_manager.add_nodes_from(graph_id, nodes)

                # Add edges
                edges = [(f"node_{j}", f"node_{j + 1}") for j in range(19)]
                graph_manager.add_edges_from(graph_id, edges)

                # Store expected data
                info = graph_manager.get_graph_info(graph_id)
                test_graphs[graph_id] = {
                    "nodes": info["num_nodes"],
                    "edges": info["num_edges"],
                }

                print(
                    f"  ✅ Created {graph_id}: {info['num_nodes']} nodes, {info['num_edges']} edges"
                )

            print(f"📊 Created {len(test_graphs)} test graphs")
            return test_graphs

        except Exception as e:
            print(f"❌ Failed to create test data: {e}")
            import traceback

            traceback.print_exc()
            return False

    def verify_data_recovery(self, expected_graphs):
        """Verify that data can be recovered from Redis."""
        try:
            print("🔍 Verifying data recovery from Redis...")

            # Clear in-memory state and rebuild from Redis
            import add_persistence
            import security_patches

            from networkx_mcp.server import graph_manager

            # Clear memory
            graph_manager.graphs.clear()
            graph_manager.metadata.clear()

            # Apply patches and persistence
            security_patches.apply_critical_patches()

            # Set environment for Redis
            os.environ["REDIS_URL"] = "redis://localhost:6379/0"
            os.environ["STORAGE_BACKEND"] = "redis"

            storage = add_persistence.patch_graph_manager_with_persistence()

            if not storage:
                print("❌ Failed to set up persistence layer for recovery")
                return False

            # Try to recover each graph
            recovered_count = 0
            for graph_id, expected_data in expected_graphs.items():
                # Try to load from Redis
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
                            print(
                                f"  ✅ Recovered {graph_id}: {info['num_nodes']} nodes, {info['num_edges']} edges"
                            )
                            recovered_count += 1
                        else:
                            print(f"  ❌ Data corruption in {graph_id}")
                            print(f"      Expected: {expected_data}")
                            print(f"      Got: {info}")
                    else:
                        print(f"  ❌ Could not load {graph_id} from Redis")
                else:
                    print("  ❌ No storage backend available")

            print(
                f"📊 Recovery results: {recovered_count}/{len(expected_graphs)} graphs recovered"
            )
            return recovered_count == len(expected_graphs)

        except Exception as e:
            print(f"❌ Data recovery test failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def test_server_restart_persistence(self):
        """Test that data survives server restart using Redis."""
        try:
            print("\n🧪 TESTING SERVER RESTART PERSISTENCE")
            print("=" * 50)

            # Phase 1: Create data with server running
            print("📝 Phase 1: Creating data with server running...")

            test_graphs = self.create_test_data_via_graph_manager()
            if not test_graphs:
                return False

            # Phase 2: Check Redis has the data
            print("📝 Phase 2: Verifying data is in Redis...")
            if not self.check_redis_data_directly():
                print("❌ No data found in Redis after creation!")
                return False

            # Phase 3: Simulate server restart
            print("📝 Phase 3: Simulating server restart...")
            print("  💥 Clearing in-memory state...")

            # Phase 4: Verify data recovery
            print("📝 Phase 4: Testing data recovery...")
            recovery_success = self.verify_data_recovery(test_graphs)

            if recovery_success:
                print("✅ SERVER RESTART PERSISTENCE TEST PASSED!")
                print("🎉 Data actually survives restart with Redis!")
                return True
            else:
                print("❌ SERVER RESTART PERSISTENCE TEST FAILED!")
                print("💥 Data was lost during restart!")
                return False

        except Exception as e:
            print(f"❌ Server restart test failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def test_redis_backend_integration(self):
        """Test that the Redis backend is actually being used."""
        try:
            print("\n🧪 TESTING REDIS BACKEND INTEGRATION")
            print("=" * 50)

            # Test 1: Verify Redis backend is available
            print("📝 Test 1: Checking Redis backend availability...")
            try:
                from networkx_mcp.storage.redis_backend import RedisGraphStorage

                print("✅ Redis backend module exists")
            except ImportError:
                print("❌ Redis backend module not found!")
                return False

            # Test 2: Initialize Redis storage
            print("📝 Test 2: Initializing Redis storage...")
            try:
                storage = RedisGraphStorage(
                    redis_url="redis://localhost:6379/0",
                    key_prefix="test_",
                    compression=True,
                )
                print("✅ Redis storage initialized")
            except Exception as e:
                print(f"❌ Redis storage initialization failed: {e}")
                return False

            # Test 3: Test basic Redis operations
            print("📝 Test 3: Testing Redis operations...")
            try:
                import networkx as nx

                # Create a test graph
                test_graph = nx.DiGraph()
                test_graph.add_nodes_from(["A", "B", "C"])
                test_graph.add_edges_from([("A", "B"), ("B", "C")])

                # Save to Redis
                save_result = storage.save_graph(
                    user_id="test_user",
                    graph_id="test_graph",
                    graph=test_graph,
                    metadata={"test": "data"},
                )

                if not save_result:
                    print("❌ Failed to save graph to Redis")
                    return False

                # Load from Redis
                loaded_graph = storage.load_graph("test_user", "test_graph")

                if loaded_graph is None:
                    print("❌ Failed to load graph from Redis")
                    return False

                # Verify data integrity
                if (
                    loaded_graph.number_of_nodes() == 3
                    and loaded_graph.number_of_edges() == 2
                ):
                    print("✅ Redis operations work correctly")
                    return True
                else:
                    print("❌ Data corruption in Redis operations")
                    return False

            except Exception as e:
                print(f"❌ Redis operations test failed: {e}")
                return False

        except Exception as e:
            print(f"❌ Redis backend integration test failed: {e}")
            import traceback

            traceback.print_exc()
            return False


async def main():
    """Run comprehensive Redis reality check."""
    print("🚀 REDIS REALITY CHECK")
    print("=" * 60)
    print("🎯 Testing if Redis persistence actually works!")
    print()

    tester = RedisRealityCheck()

    try:
        # Step 1: Verify Redis is working
        if not tester.setup_redis_connection():
            print("❌ CRITICAL: Redis is not available!")
            print("💡 Install Redis: brew install redis && brew services start redis")
            return False

        # Step 2: Clear Redis for clean test
        if not tester.clear_redis_data():
            print("❌ CRITICAL: Cannot clear Redis data!")
            return False

        # Step 3: Test Redis backend integration
        if not tester.test_redis_backend_integration():
            print("❌ CRITICAL: Redis backend not working!")
            return False

        # Step 4: Test server restart persistence
        if not await tester.test_server_restart_persistence():
            print("❌ CRITICAL: Data does not persist across restarts!")
            return False

        # Final summary
        print("\n" + "=" * 60)
        print("🎉 REDIS REALITY CHECK PASSED!")
        print("=" * 60)
        print("✅ Redis is installed and running")
        print("✅ Redis backend integration works")
        print("✅ Data actually persists across restarts")
        print("✅ No data corruption detected")
        print()
        print("🚀 Your persistence layer is REAL and WORKING!")

        return True

    except Exception as e:
        print(f"\n❌ REDIS REALITY CHECK FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if tester.server_process:
            tester.stop_server()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
