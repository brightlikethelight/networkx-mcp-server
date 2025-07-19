#!/usr/bin/env python3
"""
Compare performance between minimal and full server versions.

This script honestly measures startup time, memory usage, and import count.
"""

import subprocess
import sys
import time
from pathlib import Path

import psutil


def measure_server(server_type="minimal"):
    """Measure startup time, memory usage, and functionality."""

    if server_type == "minimal":
        module_name = "networkx_mcp.server_minimal"
    else:
        module_name = "networkx_mcp.server"

    print(f"\n{'=' * 60}")
    print(f"Testing {server_type.upper()} server")
    print(f"{'=' * 60}")

    # Measure import time and memory
    import_test = f"""
import time
import psutil
import sys

# Baseline
start = time.time()
process = psutil.Process()
before_mem = process.memory_info().rss / 1024 / 1024
before_modules = len(sys.modules)

# Import server
sys.path.insert(0, 'src')
from {module_name} import *

# After import
import_time = time.time() - start
after_mem = process.memory_info().rss / 1024 / 1024
after_modules = len(sys.modules)

# Check what got loaded
has_pandas = 'pandas' in sys.modules
has_scipy = 'scipy' in sys.modules
has_numpy = 'numpy' in sys.modules

print(f"Import time: {{import_time:.2f}}s")
print(f"Memory: {{before_mem:.1f}}MB → {{after_mem:.1f}}MB (+{{after_mem - before_mem:.1f}}MB)")
print(f"Modules: {{before_modules}} → {{after_modules}} (+{{after_modules - before_modules}})")
print(f"Pandas loaded: {{has_pandas}}")
print(f"SciPy loaded: {{has_scipy}}")
print(f"NumPy loaded: {{has_numpy}}")
"""

    result = subprocess.run(
        [sys.executable, "-c", import_test],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Test actual server startup
    print("\nTesting server startup...")
    start_time = time.time()

    proc = subprocess.Popen(
        [sys.executable, "-m", module_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    # Wait for server to be ready
    time.sleep(2)

    if proc.poll() is None:
        startup_time = time.time() - start_time

        # Measure running memory
        try:
            proc_info = psutil.Process(proc.pid)
            running_memory = proc_info.memory_info().rss / 1024 / 1024

            # Test basic functionality
            proc.stdin.write(
                '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}\n'
            )
            proc.stdin.flush()

            # Wait for response
            response = proc.stdout.readline()
            works = "result" in response

            print(f"Startup time: {startup_time:.2f}s")
            print(f"Running memory: {running_memory:.1f}MB")
            print(f"Functional: {'✅ Yes' if works else '❌ No'}")

            proc.terminate()
            proc.wait()

            return {
                "type": server_type,
                "startup_time": startup_time,
                "memory": running_memory,
                "functional": works,
            }

        except Exception as e:
            print(f"Error measuring: {e}")
            proc.terminate()

    else:
        print("Server failed to start!")

    return None


def main():
    print("NetworkX MCP Server Version Comparison")
    print("=" * 60)
    print("Comparing minimal vs full server implementations")
    print("Be patient - imports and startup take time...")

    # Test both versions
    results = []

    minimal = measure_server("minimal")
    if minimal:
        results.append(minimal)

    full = measure_server("full")
    if full:
        results.append(full)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Version':<15} {'Memory (MB)':<12} {'Startup (s)':<12} {'Status':<10}")
    print(f"{'-' * 15} {'-' * 12} {'-' * 12} {'-' * 10}")

    for r in results:
        status = "✅ Works" if r["functional"] else "❌ Broken"
        print(
            f"{r['type']:<15} {r['memory']:<12.1f} {r['startup_time']:<12.2f} {status:<10}"
        )

    if len(results) == 2:
        # Calculate differences
        mem_saved = results[1]["memory"] - results[0]["memory"]
        time_saved = results[1]["startup_time"] - results[0]["startup_time"]

        print(f"\n{'=' * 60}")
        print("SAVINGS")
        print(f"{'=' * 60}")
        print(
            f"Memory saved: {mem_saved:.1f}MB ({mem_saved / results[1]['memory'] * 100:.0f}% reduction)"
        )
        print(
            f"Startup time saved: {time_saved:.2f}s ({time_saved / results[1]['startup_time'] * 100:.0f}% faster)"
        )

        print(f"\n{'=' * 60}")
        print("VERDICT")
        print(f"{'=' * 60}")

        if mem_saved > 50:
            print("✅ Architectural fix successful!")
            print(f"   Removed {mem_saved:.0f}MB of unnecessary dependencies")
        else:
            print("⚠️  Memory savings less than expected")

        if results[0]["memory"] < 60:
            print("✅ Minimal server is actually minimal!")
        else:
            print("⚠️  Still room for improvement")

        print("\nRecommendation: Use minimal unless you need Excel/CSV support")


if __name__ == "__main__":
    main()
