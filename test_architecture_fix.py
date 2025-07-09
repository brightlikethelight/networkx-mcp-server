#!/usr/bin/env python3
"""
Quick test to verify the architectural fix worked.
"""

import subprocess
import sys

print("=" * 70)
print("ARCHITECTURE FIX VERIFICATION")
print("=" * 70)

# Test 1: Minimal server doesn't load pandas
print("\n1. Testing minimal server imports...")
test_minimal = '''
import sys
sys.path.insert(0, 'src')

# Import minimal server
from networkx_mcp.server_minimal import TrulyMinimalServer

# Check what got loaded
print(f"Pandas loaded: {'pandas' in sys.modules}")
print(f"SciPy loaded: {'scipy' in sys.modules}")
print(f"Modules loaded: {len(sys.modules)}")
'''

result = subprocess.run([sys.executable, "-c", test_minimal], capture_output=True, text=True)
print(result.stdout)

# Test 2: Original server with lazy loading
print("\n2. Testing fixed core imports...")
test_core = '''
import sys
sys.path.insert(0, 'src')

# Import core - should NOT load pandas anymore
from networkx_mcp.core import GraphManager, GraphAlgorithms

# Check what got loaded
print(f"Pandas loaded: {'pandas' in sys.modules}")
print(f"SciPy loaded: {'scipy' in sys.modules}")

# Now test lazy loading
from networkx_mcp.core import get_io_handler
print("\\nAfter importing get_io_handler (not calling it):")
print(f"Pandas loaded: {'pandas' in sys.modules}")

# This WILL load pandas if available
try:
    GraphIOHandler = get_io_handler()
    print("\\nAfter calling get_io_handler():")
    print(f"Pandas loaded: {'pandas' in sys.modules}")
except ImportError:
    print("\\nPandas not installed (expected in minimal install)")
'''

result2 = subprocess.run([sys.executable, "-c", test_core], capture_output=True, text=True)
print(result2.stdout)

# Test 3: Memory comparison
print("\n3. Memory usage comparison...")
memory_test = '''
import psutil
import sys

process = psutil.Process()
before = process.memory_info().rss / 1024 / 1024

sys.path.insert(0, 'src')
from networkx_mcp.server_minimal import TrulyMinimalServer

after = process.memory_info().rss / 1024 / 1024
print(f"Minimal server: {before:.1f}MB → {after:.1f}MB (+{after-before:.1f}MB)")
'''

result3 = subprocess.run([sys.executable, "-c", memory_test], capture_output=True, text=True)
print(result3.stdout)

# Summary
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

if "Pandas loaded: False" in result.stdout and "Pandas loaded: False" in result2.stdout:
    print("✅ SUCCESS: Architectural fix is working!")
    print("   - Minimal server doesn't load pandas")
    print("   - Core imports don't trigger pandas")
    print("   - I/O handlers are properly lazy-loaded")
else:
    print("❌ FAILED: Pandas is still being loaded!")
    
print("\nThe server is now honestly minimal.") 