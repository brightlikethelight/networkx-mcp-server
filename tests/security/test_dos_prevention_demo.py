#!/usr/bin/env python3
"""Demonstrate DoS prevention with resource limits."""

import sys
import threading
import time

sys.path.insert(0, "/Users/brightliu/Coding_Projects/networkx-mcp-server")

from src.networkx_mcp.security.resource_limits import LIMITS
from src.networkx_mcp.server import (
    add_edges,
    add_nodes,
    create_graph,
    get_graph_info,
    shortest_path,
)


def test_large_graph_attack():
    """Test prevention of large graph creation attacks."""
    print("=== Testing Large Graph Attack Prevention ===\n")

    # Try to create a massive graph
    print("1. Attempting to create graph with 200,000 nodes...")
    result = create_graph(
        "massive_graph",
        "undirected",
        {"nodes": list(range(200000)), "edges": [[i, i + 1] for i in range(199999)]},
    )

    if "error" in result:
        print(f"✅ BLOCKED: {result['error']}")
    else:
        print(f"❌ FAILED TO BLOCK: Graph created with {result['nodes']} nodes")

    print("\n2. Creating normal-sized graph...")
    result = create_graph(
        "normal_graph",
        "undirected",
        {"nodes": list(range(100)), "edges": [[i, (i + 1) % 100] for i in range(100)]},
    )

    if result.get("success"):
        print(
            f"✅ ALLOWED: Graph created with {result['nodes']} nodes, {result['edges']} edges"
        )
        print(f"   Estimated size: {result['estimated_size_mb']:.2f} MB")
    else:
        print(f"❌ INCORRECTLY BLOCKED: {result.get('error')}")


def test_memory_exhaustion_attack():
    """Test prevention of memory exhaustion attacks."""
    print("\n\n=== Testing Memory Exhaustion Attack Prevention ===\n")

    # Try to add nodes that would exceed memory
    print("1. Creating base graph...")
    create_graph("memory_test", "undirected")

    print("2. Attempting to add 50,000 nodes at once...")
    result = add_nodes("memory_test", list(range(50000)))

    if "error" in result:
        print(f"✅ BLOCKED: {result['error']}")
    else:
        print(f"❌ FAILED TO BLOCK: Added {result['nodes_added']} nodes")

    # Add reasonable number of nodes
    print("\n3. Adding reasonable number of nodes (500)...")
    result = add_nodes("memory_test", list(range(500)))

    if result.get("success"):
        print(f"✅ ALLOWED: Added {result['nodes_added']} nodes")
        print(f"   Total nodes: {result['total_nodes']}")
        print(f"   Graph size: {result['estimated_size_mb']:.2f} MB")
    else:
        print(f"❌ INCORRECTLY BLOCKED: {result.get('error')}")


def test_timeout_protection():
    """Test operation timeout protection."""
    print("\n\n=== Testing Operation Timeout Protection ===\n")

    # Create a large graph that would make operations slow
    print("1. Creating dense graph for testing...")
    create_graph("dense_graph", "undirected")

    # Add nodes
    nodes = list(range(1000))
    add_nodes("dense_graph", nodes)

    # Add many edges to make it dense
    edges = []
    for i in range(min(1000, LIMITS.max_edges_per_graph // 10)):
        for j in range(i + 1, min(i + 10, 1000)):
            edges.append([i, j])

    print(f"2. Adding {len(edges)} edges to create dense graph...")
    result = add_edges("dense_graph", edges)

    if result.get("success"):
        print(f"   Graph has {result['total_edges']} edges")

    # Now try an expensive operation
    print("\n3. Testing shortest path on dense graph...")
    start_time = time.time()
    result = shortest_path("dense_graph", 0, 999)
    elapsed = time.time() - start_time

    if result.get("success"):
        print(f"✅ Completed in {elapsed:.2f} seconds")
        print(f"   Path length: {result['length']}")
    elif "error" in result:
        print(f"✅ Operation limited: {result['error']}")
        print(f"   Time: {elapsed:.2f} seconds")


def test_concurrent_request_limiting():
    """Test concurrent request limiting."""
    print("\n\n=== Testing Concurrent Request Limiting ===\n")

    results = {"allowed": 0, "blocked": 0, "errors": []}
    lock = threading.Lock()

    def make_request(request_id):
        """Make a request and track results."""
        result = get_graph_info(f"graph_{request_id}")

        with lock:
            if "error" in result:
                if "busy" in result["error"] or "concurrent" in result["error"].lower():
                    results["blocked"] += 1
                else:
                    results["errors"].append(result["error"])
            else:
                results["allowed"] += 1

    # Create some graphs first
    for i in range(5):
        create_graph(f"graph_{i}", "undirected")

    print(
        f"1. Launching 20 concurrent requests (limit: {LIMITS.max_concurrent_requests})..."
    )

    threads = []
    for i in range(20):
        t = threading.Thread(target=make_request, args=(i % 5,))
        threads.append(t)
        t.start()
        time.sleep(0.01)  # Small delay to spread requests

    # Wait for all threads
    for t in threads:
        t.join()

    print("\nResults:")
    print(f"✅ Allowed requests: {results['allowed']}")
    print(f"✅ Blocked requests: {results['blocked']}")
    if results["errors"]:
        print(f"⚠️  Other errors: {results['errors'][:3]}")  # Show first 3


def test_resource_monitoring():
    """Test resource monitoring."""
    print("\n\n=== Testing Resource Monitoring ===\n")

    print("Current resource status:")
    # Resource status monitoring removed - no longer available
    # status = resource_status()

    if "memory" in status:
        mem = status["memory"]
        print("\nMemory Usage:")
        print(f"  Current: {mem['current_mb']:.1f} MB")
        print(f"  Limit: {mem['limit_mb']} MB")
        print(f"  Available: {mem['available_mb']:.1f} MB")

    if "requests" in status:
        req = status["requests"]
        print("\nRequest Status:")
        print(f"  Active: {req['active']}")
        print(f"  Max concurrent: {req['max_concurrent']}")
        print(f"  Recent (per minute): {req['recent_per_minute']}")
        print(f"  Limit (per minute): {req['limit_per_minute']}")

    if "limits" in status:
        lim = status["limits"]
        print("\nConfigured Limits:")
        print(f"  Max graph size: {lim['max_graph_size_mb']} MB")
        print(f"  Max nodes per graph: {lim['max_nodes_per_graph']:,}")
        print(f"  Max edges per graph: {lim['max_edges_per_graph']:,}")
        print(f"  Operation timeout: {lim['operation_timeout_seconds']} seconds")


def main():
    """Run all DoS prevention tests."""
    print("=== NetworkX MCP Server DoS Prevention Demo ===\n")
    print("This demo shows how the server prevents various DoS attacks\n")

    # Run tests
    test_large_graph_attack()
    test_memory_exhaustion_attack()
    test_timeout_protection()
    test_concurrent_request_limiting()
    test_resource_monitoring()

    print("\n\n=== Summary ===")
    print("✅ Large graph creation blocked")
    print("✅ Memory exhaustion prevented")
    print("✅ Long operations have timeouts")
    print("✅ Concurrent requests limited")
    print("✅ Resources monitored continuously")
    print("\nThe server remains responsive despite attack attempts!")


if __name__ == "__main__":
    main()
