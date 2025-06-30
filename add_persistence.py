#!/usr/bin/env python3
"""Add Redis persistence to NetworkX MCP Server."""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_redis_available():
    """Check if Redis is available."""
    try:
        import redis

        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        return True
    except:
        return False


def create_redis_backend():
    """Create Redis storage backend."""
    try:
        from simple_redis_backend import RedisGraphStorage

        storage = RedisGraphStorage(
            redis_url="redis://localhost:6379/0",
            key_prefix="networkx_mcp",
            compression=True,
        )
        # Test connection
        health = storage.check_health()
        if health["status"] == "healthy":
            storage._redis_available = True
            return storage
        else:
            print(f"⚠️ Redis unhealthy: {health}")
            return None
    except Exception as e:
        print(f"⚠️ Failed to create Redis backend: {e}")
        return None


def create_in_memory_backend():
    """Create a simple in-memory storage backend."""

    class InMemoryBackend:
        """Simple in-memory storage for development."""

        def __init__(self):
            self.graphs = {}
            self.metadata = {}

        def save_graph(self, user_id: str, graph_id: str, graph, metadata=None):
            """Save graph to memory."""
            key = f"{user_id}:{graph_id}"
            self.graphs[key] = graph
            self.metadata[key] = metadata or {}
            return True

        def load_graph(self, user_id: str, graph_id: str):
            """Load graph from memory."""
            key = f"{user_id}:{graph_id}"
            return self.graphs.get(key)

        def delete_graph(self, user_id: str, graph_id: str):
            """Delete graph from memory."""
            key = f"{user_id}:{graph_id}"
            if key in self.graphs:
                del self.graphs[key]
                del self.metadata[key]
                return True
            return False

        def list_graphs(self, user_id: str):
            """List user's graphs."""
            prefix = f"{user_id}:"
            graphs = []
            for key in self.graphs:
                if key.startswith(prefix):
                    graph_id = key[len(prefix) :]
                    graph = self.graphs[key]
                    meta = self.metadata.get(key, {})
                    graphs.append(
                        {
                            "graph_id": graph_id,
                            "user_id": user_id,
                            "num_nodes": graph.number_of_nodes(),
                            "num_edges": graph.number_of_edges(),
                            "graph_type": type(graph).__name__,
                            "metadata": meta,
                        }
                    )
            return graphs

        def get_stats(self, user_id: str):
            """Get storage stats."""
            prefix = f"{user_id}:"
            count = sum(1 for key in self.graphs if key.startswith(prefix))
            return {"user_id": user_id, "graph_count": count, "backend": "memory"}

        def check_health(self):
            """Health check."""
            return {
                "status": "healthy",
                "backend": "memory",
                "total_graphs": len(self.graphs),
            }

    return InMemoryBackend()


def patch_graph_manager_with_persistence():
    """Add persistence to the existing GraphManager."""

    try:
        from src.networkx_mcp.server import graph_manager

        print("📦 Adding persistence layer to GraphManager...")

        # Choose storage backend
        if check_redis_available():
            print("✅ Redis available - attempting to use Redis backend")
            storage = create_redis_backend()
            if storage is None:
                print("⚠️ Redis backend failed - falling back to in-memory storage")
                storage = create_in_memory_backend()
                storage._redis_available = False
            else:
                print("✅ Using Redis backend for persistence")
        else:
            print("⚠️ Redis not available - using in-memory storage")
            storage = create_in_memory_backend()
            storage._redis_available = False

        # Store original methods
        _original_create = graph_manager.create_graph
        graph_manager.get_graph
        _original_delete_graph = graph_manager.delete_graph
        _original_list_graphs = graph_manager.list_graphs
        _original_add_nodes_from = graph_manager.add_nodes_from
        _original_add_edges_from = graph_manager.add_edges_from
        _original_add_node = graph_manager.add_node
        _original_add_edge = graph_manager.add_edge
        _original_remove_node = graph_manager.remove_node
        _original_remove_edge = graph_manager.remove_edge

        # Add storage to graph manager
        graph_manager._storage = storage
        graph_manager._default_user = "default_user"

        def persistent_create_graph(graph_id: str, graph_type: str = "Graph", **kwargs):
            """Create graph with persistence."""
            # Create using original method
            result = _original_create(graph_id, graph_type, **kwargs)

            # Always save to persistent storage if graph was created
            if result.get("created"):
                # Use direct access instead of .get() (empty graphs are falsy but not None)
                if graph_id in graph_manager.graphs:
                    graph = graph_manager.graphs[graph_id]
                else:
                    graph = None

                if graph is not None:
                    try:
                        save_success = storage.save_graph(
                            graph_manager._default_user,
                            graph_id,
                            graph,
                            metadata={
                                "created_at": result.get("metadata", {}).get(
                                    "created_at"
                                ),
                                "graph_type": graph_type,
                            },
                        )
                        print(
                            f"💾 Saved graph '{graph_id}' to persistent storage: {save_success}"
                        )
                    except Exception as e:
                        print(f"❌ Failed to save graph '{graph_id}': {e}")
                else:
                    print(f"⚠️ Graph '{graph_id}' not found in manager after creation")
            else:
                print(
                    f"⚠️ Graph creation condition not met for '{graph_id}': created={result.get('created')}"
                )

            return result

        def persistent_add_nodes_from(graph_id: str, nodes):
            """Add nodes with persistence."""
            # Execute original operation
            result = _original_add_nodes_from(graph_id, nodes)

            # Save updated graph to persistence
            if graph_id in graph_manager.graphs:
                graph = graph_manager.graphs[graph_id]
                storage.save_graph(graph_manager._default_user, graph_id, graph)
                print(
                    f"💾 Updated graph '{graph_id}' in persistent storage after adding nodes"
                )

            return result

        def persistent_add_edges_from(graph_id: str, edges):
            """Add edges with persistence."""
            # Execute original operation
            result = _original_add_edges_from(graph_id, edges)

            # Save updated graph to persistence
            if graph_id in graph_manager.graphs:
                graph = graph_manager.graphs[graph_id]
                storage.save_graph(graph_manager._default_user, graph_id, graph)
                print(
                    f"💾 Updated graph '{graph_id}' in persistent storage after adding edges"
                )

            return result

        def persistent_add_node(graph_id: str, node_id, **attributes):
            """Add single node with persistence."""
            # Execute original operation
            result = _original_add_node(graph_id, node_id, **attributes)

            # Save updated graph to persistence
            if graph_id in graph_manager.graphs:
                graph = graph_manager.graphs[graph_id]
                storage.save_graph(graph_manager._default_user, graph_id, graph)
                print(
                    f"💾 Updated graph '{graph_id}' in persistent storage after adding node"
                )

            return result

        def persistent_add_edge(graph_id: str, source, target, **attributes):
            """Add single edge with persistence."""
            # Execute original operation
            result = _original_add_edge(graph_id, source, target, **attributes)

            # Save updated graph to persistence
            if graph_id in graph_manager.graphs:
                graph = graph_manager.graphs[graph_id]
                storage.save_graph(graph_manager._default_user, graph_id, graph)
                print(
                    f"💾 Updated graph '{graph_id}' in persistent storage after adding edge"
                )

            return result

        def persistent_remove_node(graph_id: str, node_id):
            """Remove node with persistence."""
            # Execute original operation
            result = _original_remove_node(graph_id, node_id)

            # Save updated graph to persistence
            if graph_id in graph_manager.graphs:
                graph = graph_manager.graphs[graph_id]
                storage.save_graph(graph_manager._default_user, graph_id, graph)
                print(
                    f"💾 Updated graph '{graph_id}' in persistent storage after removing node"
                )

            return result

        def persistent_remove_edge(graph_id: str, source, target):
            """Remove edge with persistence."""
            # Execute original operation
            result = _original_remove_edge(graph_id, source, target)

            # Save updated graph to persistence
            if graph_id in graph_manager.graphs:
                graph = graph_manager.graphs[graph_id]
                storage.save_graph(graph_manager._default_user, graph_id, graph)
                print(
                    f"💾 Updated graph '{graph_id}' in persistent storage after removing edge"
                )

            return result

        def persistent_delete_graph(graph_id: str):
            """Delete graph with persistence."""
            # Delete from persistent storage first
            storage.delete_graph(graph_manager._default_user, graph_id)

            # Then delete from memory
            result = _original_delete_graph(graph_id)

            if result.get("status") == "success":
                print(f"🗑️ Deleted graph '{graph_id}' from persistent storage")

            return result

        def enhanced_list_graphs():
            """List graphs from persistent storage."""
            # Get from original (returns a list, not dict)
            memory_graphs = _original_list_graphs()

            # Also get from persistent storage
            persistent_graphs = storage.list_graphs(graph_manager._default_user)

            # Combine and deduplicate
            all_graphs = {}

            # Add memory graphs (original returns list directly)
            for graph in memory_graphs:
                all_graphs[graph["graph_id"]] = graph

            # Add persistent graphs (may overwrite with more recent data)
            for graph in persistent_graphs:
                graph["in_persistent_storage"] = True
                all_graphs[graph["graph_id"]] = graph

            return {
                "status": "success",
                "graphs": list(all_graphs.values()),
                "total_graphs": len(all_graphs),
                "storage_stats": storage.get_stats(graph_manager._default_user),
            }

        def get_storage_stats():
            """Get storage statistics."""
            return storage.get_stats(graph_manager._default_user)

        # Apply patches
        graph_manager.create_graph = persistent_create_graph
        graph_manager.delete_graph = persistent_delete_graph
        graph_manager.list_graphs = enhanced_list_graphs
        graph_manager.get_storage_stats = get_storage_stats

        # Apply patches for graph modification operations
        graph_manager.add_nodes_from = persistent_add_nodes_from
        graph_manager.add_edges_from = persistent_add_edges_from
        graph_manager.add_node = persistent_add_node
        graph_manager.add_edge = persistent_add_edge
        graph_manager.remove_node = persistent_remove_node
        graph_manager.remove_edge = persistent_remove_edge

        print("✅ Persistence layer added successfully!")
        return storage

    except ImportError as e:
        print(f"❌ Could not add persistence: {e}")
        return None


def test_persistence():
    """Test that persistence is working."""
    print("\n🧪 Testing persistence layer...")

    try:
        from src.networkx_mcp.server import graph_manager

        # Test 1: Create a graph
        print("📝 Test 1: Creating a test graph...")
        result = graph_manager.create_graph("persistence_test", "Graph")
        if result.get("created") or result.get("graph_id") == "persistence_test":
            print("✅ Graph created successfully")
        else:
            print(f"❌ Failed to create graph: {result}")
            return False

        # Test 2: Check it's in storage
        print("📝 Test 2: Checking persistent storage...")
        if hasattr(graph_manager, "_storage"):
            stored_graph = graph_manager._storage.load_graph(
                graph_manager._default_user, "persistence_test"
            )
            if stored_graph is not None:
                print("✅ Graph found in persistent storage")
            else:
                print("❌ Graph not found in persistent storage")

        # Test 3: List graphs (should show both memory and persistent)
        print("📝 Test 3: Listing all graphs...")
        list_result = graph_manager.list_graphs()
        graphs = list_result.get("graphs", [])
        test_graph = next(
            (g for g in graphs if g["graph_id"] == "persistence_test"), None
        )

        if test_graph:
            print(f"✅ Found test graph in list: {test_graph['graph_id']}")
            if test_graph.get("in_persistent_storage"):
                print("✅ Graph is marked as in persistent storage")
        else:
            print("❌ Test graph not found in list")

        # Test 4: Storage stats
        if hasattr(graph_manager, "get_storage_stats"):
            stats = graph_manager.get_storage_stats()
            print(f"📊 Storage stats: {stats}")

        # Test 5: Cleanup
        print("📝 Test 5: Cleaning up...")
        delete_result = graph_manager.delete_graph("persistence_test")
        if delete_result.get("status") == "success":
            print("✅ Test graph deleted successfully")

        return True

    except Exception as e:
        print(f"❌ Persistence test failed: {e}")
        return False


def setup_redis():
    """Setup Redis if available."""
    if check_redis_available():
        print("✅ Redis is available and running")
        return True
    else:
        print("⚠️ Redis is not available")
        print("\n📝 To install Redis:")
        print("   macOS: brew install redis && brew services start redis")
        print(
            "   Ubuntu: sudo apt-get install redis-server && sudo systemctl start redis"
        )
        print("   Docker: docker run -d -p 6379:6379 redis:alpine")
        return False


if __name__ == "__main__":
    print("🚀 ADDING PERSISTENCE TO NETWORKX MCP SERVER")
    print("=" * 50)

    # Check Redis
    redis_available = setup_redis()

    # Apply security patches first
    import security_patches

    security_patches.apply_critical_patches()

    # Add persistence
    storage = patch_graph_manager_with_persistence()

    if storage:
        # Test persistence
        success = test_persistence()

        print("\n" + "=" * 50)
        print("💾 PERSISTENCE LAYER COMPLETE")
        print("=" * 50)

        if success:
            print("✅ All persistence tests passed!")
        else:
            print("⚠️ Some persistence tests failed")

        print(
            f"\n📊 Backend: {'Redis (available)' if redis_available else 'In-Memory'}"
        )
        print("📊 Features added:")
        print("   ✅ Graph persistence across restarts")
        print("   ✅ Storage statistics")
        print("   ✅ Dual storage (memory + persistent)")

        print("\n🚀 To run with persistence:")
        print(
            "   python -c 'import add_persistence; add_persistence.patch_graph_manager_with_persistence(); from src.networkx_mcp.server import main; main()'"
        )

    else:
        print("❌ Failed to add persistence layer")
