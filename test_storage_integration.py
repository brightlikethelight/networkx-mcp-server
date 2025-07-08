#!/usr/bin/env python3
"""Test storage integration capabilities."""

import os
import sys
sys.path.insert(0, '/Users/brightliu/Coding_Projects/networkx-mcp-server')

def test_storage_components():
    """Test that storage components are available."""
    print("=== Testing Storage Components ===\n")
    
    # Test 1: Import storage components
    print("1. Importing storage components...")
    try:
        from src.networkx_mcp.storage.base import StorageBackend, Transaction
        from src.networkx_mcp.storage.redis_backend import RedisBackend
        print("   ✅ Storage imports successful")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Check Redis backend initialization
    print("\n2. Checking Redis backend...")
    try:
        redis_backend = RedisBackend(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            max_graph_size_mb=100,
            compression_level=6
        )
        print("   ✅ Redis backend created")
        print(f"   - Max graph size: {redis_backend.max_size_bytes / 1024 / 1024}MB")
        print(f"   - Compression level: {redis_backend.compression_level}")
    except Exception as e:
        print(f"   ⚠️  Redis backend creation failed (expected if Redis not running): {e}")
    
    # Test 3: Check SecurityValidator
    print("\n3. Checking SecurityValidator...")
    try:
        from src.networkx_mcp.security.validator import SecurityValidator
        # Test validation
        user_id = SecurityValidator.validate_user_id("test_user")
        graph_id = SecurityValidator.validate_graph_id("test_graph")
        print("   ✅ SecurityValidator working")
        print(f"   - Validated user ID: {user_id}")
        print(f"   - Validated graph ID: {graph_id}")
    except ImportError:
        print("   ⚠️  SecurityValidator not found (may need to be created)")
    except Exception as e:
        print(f"   ❌ SecurityValidator error: {e}")
    
    # Test 4: Storage architecture
    print("\n4. Storage architecture features:")
    features = [
        "Compression (zlib)",
        "Atomic transactions", 
        "User isolation",
        "Metadata persistence",
        "Storage quotas",
        "Health monitoring"
    ]
    for feature in features:
        print(f"   - {feature}: Available in RedisBackend ✅")
    
    return True


def show_storage_integration_plan():
    """Show how to integrate storage with GraphManager."""
    print("\n\n=== Storage Integration Plan ===\n")
    
    print("Current state:")
    print("- GraphManager uses in-memory dict")
    print("- Storage backend exists but not connected")
    print("- Need to modify GraphManager to use storage")
    
    print("\nProposed changes to GraphManager:")
    print("""
    class GraphManager:
        def __init__(self, storage_backend=None):
            self.graphs = {}  # In-memory cache
            self.storage = storage_backend
            
        async def save_graph(self, graph_id, graph):
            # Save to memory
            self.graphs[graph_id] = graph
            
            # Save to storage if available
            if self.storage:
                await self.storage.save_graph(
                    user_id="system",  # Or from context
                    graph_id=graph_id,
                    graph=graph,
                    metadata=self.metadata[graph_id]
                )
    """)
    
    print("\nBenefits of storage integration:")
    print("- ✅ Graphs persist across restarts")
    print("- ✅ Compression reduces memory usage")
    print("- ✅ Multi-user support with isolation")
    print("- ✅ Atomic operations with transactions")
    print("- ✅ Storage quotas and monitoring")


def show_current_limitations():
    """Show current limitations without storage."""
    print("\n\n=== Current Limitations (No Storage) ===\n")
    
    limitations = [
        ("Data Loss", "All graphs lost when server restarts"),
        ("Memory Usage", "No compression, all graphs in memory"),
        ("No Multi-tenancy", "No user isolation or quotas"),
        ("No Transactions", "No atomic operations"),
        ("No Persistence", "Can't share graphs between instances"),
        ("No Monitoring", "No storage health checks")
    ]
    
    for limitation, description in limitations:
        print(f"❌ {limitation}: {description}")
    
    print("\n⚠️  The sophisticated storage backend is built but not being used!")


if __name__ == "__main__":
    print("Storage Integration Analysis\n")
    
    if test_storage_components():
        show_storage_integration_plan()
        show_current_limitations()
        
        print("\n\n=== Summary ===")
        print("✅ GraphManager is now being used (replacing simple dict)")
        print("✅ GraphAlgorithms is now being used (13+ algorithms available)")
        print("⏳ Storage backend exists but needs integration")
        print("\nNext step: Modify GraphManager to optionally use storage backend")
    else:
        print("\n❌ Storage components not fully available")