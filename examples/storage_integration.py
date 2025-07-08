#!/usr/bin/env python3
"""Example of integrating the storage-enabled NetworkX MCP Server."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.networkx_mcp.server_with_storage import NetworkXMCPServer
from src.networkx_mcp.server import NetworkXMCPServer as LegacyServer


def compare_servers():
    """Compare legacy vs storage-enabled servers."""
    print("=== NetworkX MCP Server Comparison ===\n")
    
    print("Legacy Server (server.py):")
    print("- ✓ GraphManager integration")
    print("- ✓ GraphAlgorithms integration")
    print("- ✓ Security features")
    print("- ✗ No persistence")
    print("- ✗ Data lost on restart")
    
    print("\nStorage-Enabled Server (server_with_storage.py):")
    print("- ✓ All legacy features")
    print("- ✓ Configurable storage backend")
    print("- ✓ Redis support (if REDIS_URL set)")
    print("- ✓ Automatic fallback to in-memory")
    print("- ✓ Background sync")
    print("- ✓ Storage health monitoring")
    
    print("\n=== Configuration Options ===\n")
    
    print("Environment Variables:")
    print("- REDIS_URL: Redis connection URL (e.g., redis://localhost:6379)")
    print("- MAX_GRAPH_SIZE_MB: Maximum graph size in MB (default: 100)")
    print("- COMPRESSION_LEVEL: Compression level for Redis (0-9, default: 6)")
    print("- STORAGE_BACKEND: Force backend type ('redis' or 'memory')")
    
    print("\n=== Usage Examples ===\n")
    
    print("1. Run with in-memory storage (default):")
    print("   python -m src.networkx_mcp.server_with_storage")
    
    print("\n2. Run with Redis storage:")
    print("   REDIS_URL=redis://localhost:6379 python -m src.networkx_mcp.server_with_storage")
    
    print("\n3. Run with Docker:")
    print("   docker-compose up")
    
    print("\n4. Import in your code:")
    print("""
from src.networkx_mcp.server_with_storage import NetworkXMCPServer
import asyncio

async def main():
    server = NetworkXMCPServer()
    await server.run()

asyncio.run(main())
""")


def show_storage_api():
    """Show storage-specific API additions."""
    print("\n=== Storage API Additions ===\n")
    
    print("New Tools:")
    print("- storage_status(): Get storage backend status and statistics")
    
    print("\nGraph Operations (enhanced):")
    print("- create_graph(): Now includes 'persisted' field in response")
    print("- add_nodes/edges(): Automatically syncs to storage")
    print("- delete_graph(): Removes from both memory and storage")
    
    print("\nStorage Manager Methods:")
    print("""
# Access storage manager directly
storage_manager = server.storage_manager

# Save specific graph
await storage_manager.save_graph("my_graph")

# Load specific graph
await storage_manager.load_graph("my_graph")

# Sync all graphs
await storage_manager.sync_all_graphs()

# Get storage statistics
stats = await storage_manager.get_storage_stats()
""")


def show_migration_guide():
    """Show how to migrate from legacy to storage-enabled server."""
    print("\n=== Migration Guide ===\n")
    
    print("To migrate from server.py to server_with_storage.py:\n")
    
    print("1. No code changes required! The API is fully backward compatible.")
    print("2. Simply import from server_with_storage instead of server")
    print("3. Set REDIS_URL if you want persistence")
    
    print("\nBefore:")
    print("  from src.networkx_mcp.server import create_graph, add_nodes")
    
    print("\nAfter:")
    print("  from src.networkx_mcp.server_with_storage import create_graph, add_nodes")
    
    print("\nThat's it! Your existing code will work with added persistence.")


if __name__ == "__main__":
    compare_servers()
    show_storage_api()
    show_migration_guide()
    
    print("\n=== Current Configuration ===\n")
    print(f"REDIS_URL: {os.getenv('REDIS_URL', 'Not set (using in-memory)')}")
    print(f"MAX_GRAPH_SIZE_MB: {os.getenv('MAX_GRAPH_SIZE_MB', '100')}")
    print(f"COMPRESSION_LEVEL: {os.getenv('COMPRESSION_LEVEL', '6')}")
    
    print("\n✅ Storage integration is ready to use!")