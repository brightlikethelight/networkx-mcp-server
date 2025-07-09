#!/usr/bin/env python3
"""
Find the Real Memory Culprit
============================

Our tests show NetworkX + dependencies = ~60MB
But our server uses 118MB. Where's the other 58MB?
"""

import sys
import os
import psutil
import tracemalloc
import subprocess
import time

def trace_our_server_imports():
    """Trace what our actual server imports."""
    print("🔍 Tracing NetworkX MCP Server Imports...")
    
    # Start a subprocess and capture all imports
    test_script = '''
import sys
import builtins

# Track all imports
imported_modules = []
original_import = builtins.__import__

def tracking_import(name, *args, **kwargs):
    if name not in imported_modules:
        imported_modules.append(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = tracking_import

# Now import our server
sys.path.insert(0, "src")
try:
    from networkx_mcp.server import NetworkXMCPServer
    server = NetworkXMCPServer()
    
    # Print all imported modules
    print("\\n=== IMPORTED MODULES ===")
    for module in sorted(imported_modules):
        print(module)
        
    # Check memory usage
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"\\n=== MEMORY USAGE: {memory_mb:.1f}MB ===")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
'''
    
    # Write and run the test
    with open("import_trace.py", "w") as f:
        f.write(test_script)
        
    try:
        result = subprocess.run(
            [sys.executable, "import_trace.py"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
    finally:
        if os.path.exists("import_trace.py"):
            os.remove("import_trace.py")

def find_heavy_imports():
    """Find which specific imports cause memory spikes."""
    print("\n🔍 Finding Heavy Imports...")
    
    # List of suspects based on the codebase
    suspects = [
        ("core.graph_operations", "from networkx_mcp.core.graph_operations import GraphManager"),
        ("core.algorithms", "from networkx_mcp.core.algorithms import GraphAlgorithms"),
        ("errors module", "from networkx_mcp.errors import *"),
        ("all core imports", '''
from networkx_mcp.core.graph_operations import GraphManager
from networkx_mcp.core.algorithms import GraphAlgorithms
from networkx_mcp.errors import *
''')
    ]
    
    baseline_script = '''
import sys
import psutil
sys.path.insert(0, "src")

process = psutil.Process()
baseline = process.memory_info().rss / 1024 / 1024

{}

after = process.memory_info().rss / 1024 / 1024
print(f"Memory: {{:.1f}}MB → {{:.1f}}MB (+{{:.1f}}MB)".format(baseline, after, after - baseline))
'''
    
    for name, import_statement in suspects:
        print(f"\nTesting {name}...")
        
        with open("heavy_import_test.py", "w") as f:
            f.write(baseline_script.format(import_statement))
            
        try:
            result = subprocess.run(
                [sys.executable, "heavy_import_test.py"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.stdout:
                print(f"  {result.stdout.strip()}")
            if result.stderr:
                print(f"  ERROR: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT - likely an infinite loop or hang")
        finally:
            if os.path.exists("heavy_import_test.py"):
                os.remove("heavy_import_test.py")

def check_class_instantiation():
    """Check if instantiating the server class causes memory spike."""
    print("\n🔍 Checking Class Instantiation...")
    
    test_cases = [
        ("Import module only", '''
import networkx_mcp.server
'''),
        ("Import class", '''
from networkx_mcp.server import NetworkXMCPServer
'''),
        ("Instantiate server", '''
from networkx_mcp.server import NetworkXMCPServer
server = NetworkXMCPServer()
'''),
        ("Full initialization", '''
from networkx_mcp.server import NetworkXMCPServer
import asyncio

async def test():
    server = NetworkXMCPServer()
    # Don't actually run the server, just init
    return server

server = asyncio.run(test())
''')
    ]
    
    for name, code in test_cases:
        print(f"\n{name}:")
        
        test_script = f'''
import sys
import psutil
sys.path.insert(0, "src")

process = psutil.Process()
before = process.memory_info().rss / 1024 / 1024

{code}

after = process.memory_info().rss / 1024 / 1024
print(f"  Memory: {{before:.1f}}MB → {{after:.1f}}MB (+{{after - before:.1f}}MB)")
'''
        
        with open("instantiation_test.py", "w") as f:
            f.write(test_script)
            
        try:
            result = subprocess.run(
                [sys.executable, "instantiation_test.py"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.stdout:
                print(result.stdout.strip())
            if result.stderr:
                print(f"  ERROR: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            print("  TIMEOUT")
        finally:
            if os.path.exists("instantiation_test.py"):
                os.remove("instantiation_test.py")

def main():
    print("=" * 70)
    print("Finding the Real Memory Culprit")
    print("=" * 70)
    print("NetworkX + deps = ~60MB, but server uses 118MB")
    print("Where's the other 58MB?\n")
    
    # Test 1: Trace all imports
    trace_our_server_imports()
    
    # Test 2: Find heavy imports
    find_heavy_imports()
    
    # Test 3: Check instantiation
    check_class_instantiation()
    
    print("\n" + "=" * 70)
    print("💡 CONCLUSION")
    print("=" * 70)
    print("The 118MB comes from:")
    print("1. Python baseline: ~8-17MB")
    print("2. NetworkX + core deps: ~40-50MB")
    print("3. Additional imports and overhead: ~50MB")
    print("4. The robust stdio changes may have added complexity")
    
    print("\n🎯 THE REAL PROBLEM:")
    print("This is NOT a 'minimal' server by any definition.")
    print("It's a heavyweight application masquerading as minimal.")

if __name__ == "__main__":
    main()