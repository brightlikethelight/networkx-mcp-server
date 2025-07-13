#!/usr/bin/env python3
"""
Honest Performance Benchmarks - No Fabrication

Measures actual performance of both implementations.
If numbers look suspicious, we investigate rather than publish them.
"""

import time
import psutil
import subprocess
import json
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def measure_import_time(use_minimal: bool):
    """Measure how long it takes to import the server."""
    env = {"USE_MINIMAL_SERVER": str(use_minimal).lower()}
    
    start = time.perf_counter()
    proc = subprocess.run([
        sys.executable, "-c", 
        "import networkx_mcp.server; print('imported')"
    ], env={**os.environ, **env}, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    
    return {
        "import_time_seconds": round(elapsed, 3),
        "success": proc.returncode == 0,
        "output": proc.stdout.strip(),
        "error": proc.stderr.strip() if proc.stderr else None
    }


def measure_memory_usage(use_minimal: bool):
    """Measure actual memory usage of running server."""
    env = {"USE_MINIMAL_SERVER": str(use_minimal).lower()}
    
    # Start server process
    proc = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, **env}
    )
    
    try:
        # Wait for startup
        time.sleep(1)
        
        # Measure memory
        process = psutil.Process(proc.pid)
        memory_info = process.memory_info()
        
        # Send basic initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }
        
        proc.stdin.write((json.dumps(init_request) + "\n").encode())
        proc.stdin.flush()
        
        # Read response (with timeout)
        import select
        ready, _, _ = select.select([proc.stdout], [], [], 2.0)
        response = None
        if ready:
            response = proc.stdout.readline().decode().strip()
        
        return {
            "memory_mb": round(memory_info.rss / 1024 / 1024, 1),
            "memory_peak_mb": round(memory_info.peak_wss / 1024 / 1024, 1) if hasattr(memory_info, 'peak_wss') else None,
            "response_received": response is not None,
            "response": response
        }
        
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def measure_module_count(use_minimal: bool):
    """Count how many modules are loaded."""
    env = {"USE_MINIMAL_SERVER": str(use_minimal).lower()}
    
    proc = subprocess.run([
        sys.executable, "-c", 
        f"""
import os
os.environ['USE_MINIMAL_SERVER'] = '{str(use_minimal).lower()}'
import sys
initial_modules = len(sys.modules)
import networkx_mcp.server
final_modules = len(sys.modules)
print(json.dumps({{
    'initial_modules': initial_modules,
    'final_modules': final_modules,
    'imported_modules': final_modules - initial_modules
}}))
"""
    ], env={**os.environ, **env}, capture_output=True, text=True)
    
    if proc.returncode == 0:
        return json.loads(proc.stdout.strip())
    else:
        return {"error": proc.stderr.strip()}


def run_honest_benchmarks():
    """Run comprehensive, honest benchmarks."""
    print("🔍 Honest Performance Benchmarks")
    print("=" * 50)
    print("No fabrication. No negative memory. Just truth.")
    print()
    
    implementations = [
        ("Minimal", True),
        ("Legacy", False)
    ]
    
    results = {}
    
    for name, use_minimal in implementations:
        print(f"📊 Testing {name} Implementation...")
        
        # Import time
        import_result = measure_import_time(use_minimal)
        print(f"   Import time: {import_result['import_time_seconds']}s")
        
        # Memory usage
        memory_result = measure_memory_usage(use_minimal)
        print(f"   Memory usage: {memory_result['memory_mb']}MB")
        
        # Module count
        module_result = measure_module_count(use_minimal)
        if 'imported_modules' in module_result:
            print(f"   Modules imported: {module_result['imported_modules']}")
        
        results[name.lower()] = {
            "import": import_result,
            "memory": memory_result,
            "modules": module_result
        }
        
        print()
    
    # Compare results
    print("📈 Comparison:")
    print("-" * 30)
    
    if 'minimal' in results and 'legacy' in results:
        minimal_mem = results['minimal']['memory']['memory_mb']
        legacy_mem = results['legacy']['memory']['memory_mb']
        
        if minimal_mem > 0 and legacy_mem > 0:
            reduction = (1 - minimal_mem / legacy_mem) * 100
            print(f"Memory reduction: {reduction:.1f}%")
        
        minimal_import = results['minimal']['import']['import_time_seconds']
        legacy_import = results['legacy']['import']['import_time_seconds']
        
        if minimal_import > 0 and legacy_import > 0:
            speedup = legacy_import / minimal_import
            print(f"Import speedup: {speedup:.1f}x")
    
    print()
    print("📝 Raw Results:")
    print(json.dumps(results, indent=2))
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"honest_benchmark_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "note": "These are actual measurements, not fabricated numbers",
            "methodology": "Subprocess measurement with psutil",
            "results": results
        }, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    try:
        run_honest_benchmarks()
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)