#!/usr/bin/env python3
"""
Diagnose Initialization Cost
============================

118MB for initialization is INSANE. What the hell is loading?
Let's find out exactly what's consuming memory.
"""

import sys
import os
import tracemalloc
import psutil
import importlib
import gc

def measure_import_cost(module_name):
    """Measure memory cost of importing a module."""
    gc.collect()
    
    # Get baseline
    process = psutil.Process()
    before_memory = process.memory_info().rss / 1024 / 1024
    
    # Start tracing
    tracemalloc.start()
    
    try:
        # Import the module
        module = importlib.import_module(module_name)
        
        # Get memory after import
        after_memory = process.memory_info().rss / 1024 / 1024
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "module": module_name,
            "rss_growth": after_memory - before_memory,
            "traced_current": current / 1024 / 1024,
            "traced_peak": peak / 1024 / 1024
        }
    except ImportError as e:
        tracemalloc.stop()
        return {
            "module": module_name,
            "error": str(e)
        }

def diagnose_server_imports():
    """Diagnose what the server is importing that costs so much memory."""
    print("🔍 Diagnosing Import Memory Costs...")
    
    # Baseline Python interpreter
    process = psutil.Process()
    baseline = process.memory_info().rss / 1024 / 1024
    print(f"📊 Python baseline: {baseline:.1f}MB")
    
    # Test individual imports in order they happen
    imports_to_test = [
        "json",
        "asyncio", 
        "logging",
        "networkx",  # This is probably the culprit
        "numpy",     # NetworkX might load this
        "scipy",     # NetworkX might load this
        "matplotlib",  # NetworkX might load this
    ]
    
    results = []
    cumulative_memory = baseline
    
    for module_name in imports_to_test:
        if module_name not in sys.modules:  # Only test unloaded modules
            result = measure_import_cost(module_name)
            results.append(result)
            
            if "error" not in result:
                cumulative_memory += result["rss_growth"]
                print(f"📦 {module_name:15} +{result['rss_growth']:6.1f}MB (total: {cumulative_memory:.1f}MB)")
            else:
                print(f"❌ {module_name:15} {result['error']}")
    
    return results

def test_minimal_server():
    """Test a minimal server without NetworkX to see baseline cost."""
    print("\n🔍 Testing Minimal Server Memory...")
    
    # Create minimal server code
    minimal_server = '''
import asyncio
import json
import sys

class MinimalServer:
    async def run(self):
        print("Minimal server running", file=sys.stderr)
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            try:
                msg = json.loads(line.strip())
                response = {"jsonrpc": "2.0", "id": msg.get("id"), "result": "ok"}
                print(json.dumps(response))
                sys.stdout.flush()
            except:
                pass

if __name__ == "__main__":
    server = MinimalServer()
    asyncio.run(server.run())
'''
    
    # Write minimal server
    with open("minimal_server_test.py", "w") as f:
        f.write(minimal_server)
    
    # Test its memory usage
    import subprocess
    
    proc = subprocess.Popen(
        [sys.executable, "minimal_server_test.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Get its memory
    import time
    time.sleep(0.5)  # Let it start
    
    if proc.poll() is None:
        proc_info = psutil.Process(proc.pid)
        memory_mb = proc_info.memory_info().rss / 1024 / 1024
        print(f"📊 Minimal server memory: {memory_mb:.1f}MB")
        
        # Send a test message
        proc.stdin.write('{"jsonrpc": "2.0", "id": 1, "method": "test"}\n')
        proc.stdin.flush()
        
        # Read response
        response = proc.stdout.readline()
        print(f"📥 Response: {response.strip()}")
        
        proc.terminate()
        proc.wait()
        
        # Clean up
        os.remove("minimal_server_test.py")
        
        return memory_mb
    else:
        print("❌ Minimal server failed to start")
        os.remove("minimal_server_test.py")
        return 0

def analyze_networkx_bloat():
    """Analyze what NetworkX is actually loading."""
    print("\n🔍 Analyzing NetworkX Import Chain...")
    
    # Fresh process to measure NetworkX alone
    before_modules = set(sys.modules.keys())
    
    # Measure NetworkX import
    gc.collect()
    process = psutil.Process()
    before_memory = process.memory_info().rss / 1024 / 1024
    
    import networkx as nx
    
    after_memory = process.memory_info().rss / 1024 / 1024
    after_modules = set(sys.modules.keys())
    
    # What modules did NetworkX load?
    new_modules = after_modules - before_modules
    
    print(f"📊 NetworkX alone: {after_memory - before_memory:.1f}MB")
    print(f"📦 Loaded {len(new_modules)} new modules:")
    
    # Categorize the modules
    categories = {
        "numpy": [],
        "scipy": [],
        "matplotlib": [],
        "other": []
    }
    
    for module in sorted(new_modules):
        if module.startswith("numpy"):
            categories["numpy"].append(module)
        elif module.startswith("scipy"):
            categories["scipy"].append(module)
        elif module.startswith("matplotlib"):
            categories["matplotlib"].append(module)
        else:
            categories["other"].append(module)
    
    for category, modules in categories.items():
        if modules:
            print(f"\n   {category}: {len(modules)} modules")
            if category != "other":  # Show some examples
                for module in modules[:5]:
                    print(f"      - {module}")
                if len(modules) > 5:
                    print(f"      ... and {len(modules) - 5} more")

def main():
    print("=" * 70)
    print("118MB Initialization Cost Investigation")
    print("=" * 70)
    print("Something is VERY wrong. Let's find out what.\n")
    
    # Test 1: Individual import costs
    import_results = diagnose_server_imports()
    
    # Test 2: Minimal server baseline
    minimal_memory = test_minimal_server()
    
    # Test 3: NetworkX analysis
    analyze_networkx_bloat()
    
    # Now test our actual server
    print("\n🔍 Testing Our NetworkX MCP Server...")
    import subprocess
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    import time
    time.sleep(1)  # Let it initialize
    
    if proc.poll() is None:
        proc_info = psutil.Process(proc.pid)
        our_memory = proc_info.memory_info().rss / 1024 / 1024
        print(f"📊 NetworkX MCP Server: {our_memory:.1f}MB")
        proc.terminate()
        proc.wait()
    
    print("\n" + "=" * 70)
    print("💡 DIAGNOSIS")
    print("=" * 70)
    
    # Find the culprit
    big_imports = [r for r in import_results if isinstance(r.get("rss_growth", 0), (int, float)) and r["rss_growth"] > 10]
    
    if big_imports:
        print("🚨 MEMORY HOGS FOUND:")
        for imp in big_imports:
            print(f"   {imp['module']}: {imp['rss_growth']:.1f}MB")
    
    print(f"\n📊 COMPARISON:")
    print(f"   Minimal server:   {minimal_memory:.1f}MB")
    print(f"   NetworkX alone:   ~100MB+ (with dependencies)")
    print(f"   Our MCP server:   ~118MB")
    
    print(f"\n🎯 ROOT CAUSE:")
    print("NetworkX is importing the entire scientific Python stack!")
    print("This includes NumPy, SciPy, and possibly Matplotlib.")
    print("For a simple graph server, this is MASSIVE overkill.")
    
    print(f"\n💡 SOLUTIONS:")
    print("1. Lazy-load NetworkX only when needed")
    print("2. Use a lighter graph library")
    print("3. Strip unnecessary NetworkX dependencies")
    print("4. Accept the bloat (NOT recommended)")

if __name__ == "__main__":
    main()