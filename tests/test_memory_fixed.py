#!/usr/bin/env python3
"""Test memory usage after architectural fixes."""

import subprocess
import sys
import time
import psutil
import json


def test_minimal_memory():
    """Verify minimal server uses < 25MB."""
    print("=" * 70)
    print("Testing Memory Usage - Minimal vs Full Server")
    print("=" * 70)
    
    # Test 1: Truly minimal server
    print("\n1. Testing TRULY minimal server (server_minimal.py)...")
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp.server_minimal"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for startup
    time.sleep(2)
    
    if proc.poll() is None:
        try:
            process = psutil.Process(proc.pid)
            minimal_memory = process.memory_info().rss / 1024 / 1024
            print(f"   Memory usage: {minimal_memory:.1f}MB")
            
            # Test it works with handshake
            # Initialize
            proc.stdin.write('{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}\n')
            proc.stdin.flush()
            response = proc.stdout.readline()
            print(f"   Initialize: {'✓' if 'result' in response else '✗'}")
            
            # Initialized notification
            proc.stdin.write('{"jsonrpc": "2.0", "method": "initialized", "params": {}}\n')
            proc.stdin.flush()
            
            # Create graph
            proc.stdin.write('{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "create_graph", "arguments": {"graph_id": "test"}}}\n')
            proc.stdin.flush()
            response = proc.stdout.readline()
            print(f"   Create graph: {'✓' if 'result' in response else '✗'}")
            
            proc.terminate()
            proc.wait()
            
        except Exception as e:
            print(f"   Error: {e}")
            proc.terminate()
            minimal_memory = 999
    else:
        print("   Failed to start!")
        minimal_memory = 999
        
    # Test 2: Original server (with pandas)
    print("\n2. Testing original server (server.py)...")
    
    proc2 = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for startup
    time.sleep(2)
    
    if proc2.poll() is None:
        try:
            process = psutil.Process(proc2.pid)
            full_memory = process.memory_info().rss / 1024 / 1024
            print(f"   Memory usage: {full_memory:.1f}MB")
            
            proc2.terminate()
            proc2.wait()
            
        except Exception as e:
            print(f"   Error: {e}")
            proc2.terminate()
            full_memory = 999
    else:
        print("   Failed to start!")
        full_memory = 999
        
    # Results
    print("\n" + "=" * 70)
    print("MEMORY COMPARISON")
    print("=" * 70)
    print(f"Minimal server:  {minimal_memory:.1f}MB")
    print(f"Full server:     {full_memory:.1f}MB")
    print(f"Memory saved:    {full_memory - minimal_memory:.1f}MB ({(full_memory - minimal_memory)/full_memory*100:.0f}% reduction)")
    
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if minimal_memory < 25:
        print("✅ SUCCESS: Minimal server uses < 25MB!")
        print("   This is a TRULY minimal server.")
    else:
        print(f"❌ FAILED: Minimal server uses {minimal_memory:.1f}MB")
        print("   Still too heavy for a minimal server!")
        
    if full_memory - minimal_memory > 50:
        print(f"\n✅ ARCHITECTURAL FIX SUCCESSFUL!")
        print(f"   Removed {full_memory - minimal_memory:.0f}MB of unnecessary dependencies.")
    else:
        print(f"\n⚠️  WARNING: Only saved {full_memory - minimal_memory:.0f}MB")
        print("   The architectural fix may not be complete.")
        
    return minimal_memory < 25


def test_import_chain_broken():
    """Verify pandas is not imported by minimal server."""
    print("\n" + "=" * 70)
    print("Testing Import Chain")
    print("=" * 70)
    
    test_script = '''
import sys
sys.path.insert(0, 'src')

# Track imports
imported = []
original_import = __builtins__.__import__

def track_import(name, *args, **kwargs):
    imported.append(name)
    return original_import(name, *args, **kwargs)

__builtins__.__import__ = track_import

# Import minimal server
from networkx_mcp import server_minimal

# Check for pandas
has_pandas = any('pandas' in imp for imp in imported)
has_scipy = any('scipy' in imp for imp in imported)

print(f"Pandas imported: {has_pandas}")
print(f"SciPy imported: {has_scipy}")
print(f"Total imports: {len(imported)}")

if has_pandas or has_scipy:
    print("\\n❌ FAILED: Heavy dependencies still being imported!")
else:
    print("\\n✅ SUCCESS: No heavy dependencies imported!")
'''
    
    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
        
    return "SUCCESS" in result.stdout


if __name__ == "__main__":
    # Run tests
    memory_ok = test_minimal_memory()
    imports_ok = test_import_chain_broken()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    if memory_ok and imports_ok:
        print("✅ ALL TESTS PASSED!")
        print("   The architectural surgery was successful.")
        print("   We now have a TRULY minimal NetworkX MCP server.")
    else:
        print("❌ SOME TESTS FAILED")
        if not memory_ok:
            print("   - Memory usage still too high")
        if not imports_ok:
            print("   - Import chain not fully broken")