#!/usr/bin/env python3
"""Test NetworkX memory usage alone."""

import psutil
import sys

print("=" * 70)
print("NetworkX Memory Usage Analysis")
print("=" * 70)

# Baseline
process = psutil.Process()
baseline = process.memory_info().rss / 1024 / 1024
print(f"1. Python baseline: {baseline:.1f}MB")

# Import NetworkX only
import networkx as nx
after_nx = process.memory_info().rss / 1024 / 1024
print(f"2. After importing NetworkX: {after_nx:.1f}MB (+{after_nx - baseline:.1f}MB)")

# Check what NetworkX imported
numpy_loaded = 'numpy' in sys.modules
scipy_loaded = 'scipy' in sys.modules
pandas_loaded = 'pandas' in sys.modules
matplotlib_loaded = 'matplotlib' in sys.modules

print(f"\nDependencies loaded by NetworkX:")
print(f"  NumPy: {numpy_loaded}")
print(f"  SciPy: {scipy_loaded}")
print(f"  Pandas: {pandas_loaded}")
print(f"  Matplotlib: {matplotlib_loaded}")

# Create a graph
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 4)])
after_graph = process.memory_info().rss / 1024 / 1024
print(f"\n3. After creating small graph: {after_graph:.1f}MB (+{after_graph - after_nx:.1f}MB)")

# Import our minimal server
sys.path.insert(0, 'src')
from networkx_mcp.server_minimal import TrulyMinimalServer
after_server = process.memory_info().rss / 1024 / 1024
print(f"4. After importing minimal server: {after_server:.1f}MB (+{after_server - after_graph:.1f}MB)")

# Create server instance
server = TrulyMinimalServer()
after_instance = process.memory_info().rss / 1024 / 1024
print(f"5. After creating server instance: {after_instance:.1f}MB (+{after_instance - after_server:.1f}MB)")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)
print(f"NetworkX alone adds: {after_nx - baseline:.1f}MB")
print(f"Our minimal server adds: {after_instance - after_nx:.1f}MB")
print(f"Total memory usage: {after_instance:.1f}MB")

print(f"\n{'✅' if not pandas_loaded else '❌'} Pandas NOT loaded (saved ~35MB)")
print(f"{'✅' if not scipy_loaded else '❌'} SciPy NOT loaded (saved ~15MB)")
print(f"{'✅' if after_instance < 60 else '❌'} Total usage under 60MB")

print("\nCONCLUSION:")
if after_instance < 60 and not pandas_loaded and not scipy_loaded:
    print("✅ ARCHITECTURAL SURGERY SUCCESSFUL!")
    print(f"   Reduced memory from 118MB to {after_instance:.1f}MB")
    print(f"   Savings: {118 - after_instance:.0f}MB ({(118 - after_instance)/118*100:.0f}% reduction)")
    print("   NetworkX itself uses ~38MB, which is unavoidable.")
else:
    print("❌ Still using too much memory or loading heavy dependencies")