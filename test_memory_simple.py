#!/usr/bin/env python3
"""Simple memory test for minimal server."""

import subprocess
import sys
import time
import psutil
import os

print("=" * 70)
print("Memory Test: Minimal Server")
print("=" * 70)

# Test script that starts server and exits
test_script = '''
import sys
import psutil
import time

# Check baseline
process = psutil.Process()
before = process.memory_info().rss / 1024 / 1024
print(f"Python baseline: {before:.1f}MB")

# Add src to path
sys.path.insert(0, 'src')

# Import minimal server
from networkx_mcp.server_minimal import TrulyMinimalServer

# Create instance
server = TrulyMinimalServer()

# Check memory after import
after = process.memory_info().rss / 1024 / 1024
print(f"After import: {after:.1f}MB")
print(f"Overhead: {after - before:.1f}MB")

# Test if pandas was imported
import sys
has_pandas = 'pandas' in sys.modules
has_scipy = 'scipy' in sys.modules

print(f"\\nPandas loaded: {has_pandas}")
print(f"SciPy loaded: {has_scipy}")
print(f"Total modules: {len(sys.modules)}")

if after < 30:
    print("\\n✅ SUCCESS: Truly minimal memory usage!")
else:
    print(f"\\n❌ FAILED: Using {after:.1f}MB is not minimal!")
'''

# Run test
result = subprocess.run(
    [sys.executable, "-c", test_script],
    capture_output=True,
    text=True,
    cwd=os.getcwd()
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Now test original server
print("\n" + "=" * 70)
print("Memory Test: Original Server")
print("=" * 70)

test_script2 = '''
import sys
import psutil

# Check baseline
process = psutil.Process()
before = process.memory_info().rss / 1024 / 1024
print(f"Python baseline: {before:.1f}MB")

# Add src to path
sys.path.insert(0, 'src')

try:
    # Import original server
    from networkx_mcp.server import NetworkXMCPServer
    
    # Create instance
    server = NetworkXMCPServer()
    
    # Check memory after import
    after = process.memory_info().rss / 1024 / 1024
    print(f"After import: {after:.1f}MB")
    print(f"Overhead: {after - before:.1f}MB")
    
    # Test if pandas was imported
    import sys
    has_pandas = 'pandas' in sys.modules
    has_scipy = 'scipy' in sys.modules
    
    print(f"\\nPandas loaded: {has_pandas}")
    print(f"SciPy loaded: {has_scipy}")
    print(f"Total modules: {len(sys.modules)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
'''

result2 = subprocess.run(
    [sys.executable, "-c", test_script2],
    capture_output=True,
    text=True,
    cwd=os.getcwd()
)

print(result2.stdout)
if result2.stderr:
    print("STDERR:", result2.stderr)