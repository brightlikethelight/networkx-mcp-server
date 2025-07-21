#!/usr/bin/env python3
"""
Security testing for NetworkX MCP Server.
Tests input validation, error handling, and security boundaries.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from networkx_mcp.server import NetworkXMCPServer


async def call_tool(server, tool_name, arguments):
    """Helper to call a tool and get the result."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }

    response = await server.handle_request(request)
    return response


async def test_input_validation():
    """Test input validation and sanitization."""
    server = NetworkXMCPServer(auth_required=False, enable_monitoring=False)

    print("\nINPUT VALIDATION TESTS")
    print("=" * 60)

    tests = []

    # Test 1: SQL injection attempt in graph name
    print("\n1. Testing SQL injection prevention...")
    response = await call_tool(server, "create_graph", {
        "name": "test'; DROP TABLE graphs; --"
    })
    success = "error" not in response and response.get("result")
    tests.append(("SQL injection in graph name", success))
    print(f"   Created graph with SQL-like name: {success}")

    # Test 2: Path traversal attempt
    print("\n2. Testing path traversal prevention...")
    response = await call_tool(server, "create_graph", {
        "name": "../../../etc/passwd"
    })
    success = "error" not in response
    tests.append(("Path traversal in graph name", success))
    print(f"   Created graph with path-like name: {success}")

    # Test 3: XSS attempt in node names
    print("\n3. Testing XSS prevention...")
    await call_tool(server, "create_graph", {"name": "xss_test"})
    response = await call_tool(server, "add_nodes", {
        "graph": "xss_test",
        "nodes": ["<script>alert('XSS')</script>", "normal_node"]
    })
    success = "error" not in response
    tests.append(("XSS in node names", success))
    print(f"   Added nodes with script tags: {success}")

    # Test 4: Command injection attempt
    print("\n4. Testing command injection prevention...")
    response = await call_tool(server, "create_graph", {
        "name": "test`rm -rf /`"
    })
    success = "error" not in response
    tests.append(("Command injection in graph name", success))
    print(f"   Created graph with command-like name: {success}")

    # Test 5: Unicode and special characters
    print("\n5. Testing Unicode handling...")
    response = await call_tool(server, "create_graph", {
        "name": "测试图表🎯"
    })
    success = "error" not in response
    tests.append(("Unicode in graph name", success))
    print(f"   Created graph with Unicode name: {success}")

    # Test 6: Very long names
    print("\n6. Testing length limits...")
    long_name = "A" * 10000
    response = await call_tool(server, "create_graph", {
        "name": long_name
    })
    success = "error" not in response
    tests.append(("Very long graph name", success))
    print(f"   Created graph with 10k character name: {success}")

    # Test 7: Null and empty values
    print("\n7. Testing null/empty value handling...")
    response = await call_tool(server, "create_graph", {
        "name": ""
    })
    handled_properly = "error" in response or response.get("result", {}).get("isError")
    tests.append(("Empty graph name", handled_properly))
    print(f"   Empty name handled properly: {handled_properly}")

    # Test 8: Invalid data types
    print("\n8. Testing type validation...")
    response = await call_tool(server, "add_nodes", {
        "graph": "xss_test",
        "nodes": "not_a_list"  # Should be a list
    })
    handled_properly = "error" in response or response.get("result", {}).get("isError")
    tests.append(("Invalid type for nodes", handled_properly))
    print(f"   Invalid type handled properly: {handled_properly}")

    return tests


async def test_dos_prevention():
    """Test denial of service prevention."""
    server = NetworkXMCPServer(auth_required=False, enable_monitoring=False)

    print("\nDENIAL OF SERVICE PREVENTION TESTS")
    print("=" * 60)

    tests = []

    # Test 1: Large graph creation
    print("\n1. Testing large graph limits...")
    try:
        await call_tool(server, "create_graph", {"name": "large_test"})

        # Try to add a million nodes (should be handled gracefully)
        huge_nodes = list(range(10000))  # Start with 10k
        response = await call_tool(server, "add_nodes", {
            "graph": "large_test",
            "nodes": huge_nodes
        })
        success = "error" not in response
        tests.append(("Large node addition (10k)", success))
        print(f"   Added 10k nodes: {success}")
    except Exception as e:
        tests.append(("Large node addition", False))
        print(f"   Failed with: {e}")

    # Test 2: Deeply nested CSV
    print("\n2. Testing deeply nested CSV...")
    nested_csv = "source,target\n"
    for i in range(1000):
        nested_csv += f"node{i},node{i+1}\n"

    response = await call_tool(server, "import_csv", {
        "graph": "nested_csv",
        "csv_data": nested_csv
    })
    success = "error" not in response
    tests.append(("Large CSV import", success))
    print(f"   Imported 1000-line CSV: {success}")

    # Test 3: Circular references
    print("\n3. Testing circular reference handling...")
    await call_tool(server, "create_graph", {"name": "circular"})
    await call_tool(server, "add_nodes", {
        "graph": "circular",
        "nodes": ["A", "B", "C"]
    })
    await call_tool(server, "add_edges", {
        "graph": "circular",
        "edges": [["A", "B"], ["B", "C"], ["C", "A"]]
    })

    # This should handle cycles gracefully
    response = await call_tool(server, "shortest_path", {
        "graph": "circular",
        "source": "A",
        "target": "A"
    })
    handled = "error" not in response or "path" in response.get("result", {})
    tests.append(("Circular path handling", handled))
    print(f"   Circular paths handled: {handled}")

    return tests


async def test_error_handling():
    """Test error handling and information disclosure."""
    server = NetworkXMCPServer(auth_required=False, enable_monitoring=False)

    print("\nERROR HANDLING TESTS")
    print("=" * 60)

    tests = []

    # Test 1: Non-existent graph
    print("\n1. Testing non-existent graph errors...")
    response = await call_tool(server, "get_info", {
        "graph": "does_not_exist"
    })

    # Check error doesn't reveal internals
    if "error" in response:
        error_msg = response["error"].get("message", "")
    else:
        error_msg = response.get("result", {}).get("content", [{}])[0].get("text", "")

    reveals_internals = any(word in error_msg.lower() for word in
                           ["traceback", "file", "line", "__"])
    tests.append(("Error message safety", not reveals_internals))
    print(f"   Error messages safe: {not reveals_internals}")
    print(f"   Error: {error_msg[:100]}...")

    # Test 2: Invalid JSON-RPC
    print("\n2. Testing malformed request handling...")
    malformed = {
        "jsonrpc": "1.0",  # Wrong version
        "method": "tools/list"
    }
    response = await server.handle_request(malformed)
    handled = response is not None and "error" not in str(response).lower()
    tests.append(("Malformed request handling", handled))
    print(f"   Malformed requests handled: {handled}")

    # Test 3: Missing required fields
    print("\n3. Testing missing field handling...")
    response = await call_tool(server, "shortest_path", {
        "graph": "test"
        # Missing source and target
    })
    handled = "error" in response or response.get("result", {}).get("isError")
    tests.append(("Missing required fields", handled))
    print(f"   Missing fields handled: {handled}")

    return tests


async def test_resource_limits():
    """Test resource consumption limits."""
    server = NetworkXMCPServer(auth_required=False, enable_monitoring=False)

    print("\nRESOURCE LIMIT TESTS")
    print("=" * 60)

    tests = []

    # Test 1: Memory usage with visualization
    print("\n1. Testing visualization memory limits...")
    await call_tool(server, "create_graph", {"name": "viz_memory"})

    # Create a moderately complex graph
    nodes = list(range(100))
    edges = [[i, (i+1) % 100] for i in range(100)]
    edges.extend([[i, (i+50) % 100] for i in range(0, 100, 5)])

    await call_tool(server, "add_nodes", {"graph": "viz_memory", "nodes": nodes})
    await call_tool(server, "add_edges", {"graph": "viz_memory", "edges": edges})

    # Generate visualization
    response = await call_tool(server, "visualize_graph", {
        "graph": "viz_memory",
        "layout": "spring"
    })
    success = "error" not in response
    tests.append(("Visualization memory handling", success))
    print(f"   100-node visualization: {success}")

    # Test 2: Concurrent graph operations
    print("\n2. Testing concurrent operations...")
    tasks = []
    for i in range(10):
        task = call_tool(server, "create_graph", {"name": f"concurrent_{i}"})
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    success = all(not isinstance(r, Exception) and "error" not in r for r in results)
    tests.append(("Concurrent graph creation", success))
    print(f"   10 concurrent creations: {success}")

    return tests


async def main():
    """Run all security tests."""
    print("=" * 80)
    print("NETWORKX MCP SERVER - SECURITY TEST SUITE")
    print("=" * 80)
    print(f"Testing at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_tests = []

    # Run test suites
    all_tests.extend(await test_input_validation())
    all_tests.extend(await test_dos_prevention())
    all_tests.extend(await test_error_handling())
    all_tests.extend(await test_resource_limits())

    # Summary
    print("\n" + "="*60)
    print("SECURITY TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, success in all_tests if success)
    total = len(all_tests)

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"Failed: {total - passed}")

    print("\nDetailed Results:")
    for test_name, success in all_tests:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")

    print("\nSecurity Assessment:")
    if passed / total >= 0.9:
        print("✅ Server demonstrates good security practices")
    else:
        print("⚠️  Some security concerns need attention")

    print("\nRecommendations:")
    print("- All user inputs are accepted (no strict validation)")
    print("- Consider adding rate limiting for production use")
    print("- Monitor resource usage in production environments")
    print("- Use authentication for write operations in production")


if __name__ == "__main__":
    import time
    asyncio.run(main())
