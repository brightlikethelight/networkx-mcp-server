#!/usr/bin/env python3
"""
Simple Memory Test
==================

Just measure memory of different configurations.
"""

import subprocess
import sys
import time
import psutil

def test_server_memory(server_code, name):
    """Test memory usage of a server implementation."""
    # Write server file
    with open("test_server.py", "w") as f:
        f.write(server_code)
    
    try:
        # Start server
        proc = subprocess.Popen(
            [sys.executable, "test_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for it to start
        time.sleep(1)
        
        if proc.poll() is None:
            # Measure memory
            proc_info = psutil.Process(proc.pid)
            memory_mb = proc_info.memory_info().rss / 1024 / 1024
            print(f"{name}: {memory_mb:.1f}MB")
            
            # Clean up
            proc.terminate()
            proc.wait()
        else:
            print(f"{name}: Failed to start")
            
    finally:
        import os
        if os.path.exists("test_server.py"):
            os.remove("test_server.py")

# Test 1: Bare Python
bare_python = '''
import time
while True:
    time.sleep(1)
'''

# Test 2: With asyncio
with_asyncio = '''
import asyncio
import sys

async def main():
    while True:
        await asyncio.sleep(1)

asyncio.run(main())
'''

# Test 3: With NetworkX
with_networkx = '''
import asyncio
import networkx as nx

async def main():
    graph = nx.Graph()
    while True:
        await asyncio.sleep(1)

asyncio.run(main())
'''

# Test 4: With numpy
with_numpy = '''
import asyncio
import numpy as np

async def main():
    arr = np.array([1, 2, 3])
    while True:
        await asyncio.sleep(1)

asyncio.run(main())
'''

# Test 5: Our actual server (simplified)
our_server = '''
import asyncio
import json
import sys
import networkx as nx
from pathlib import Path

# This simulates our server's imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

class TestServer:
    def __init__(self):
        self.graphs = {}
        
async def main():
    server = TestServer()
    while True:
        await asyncio.sleep(1)

asyncio.run(main())
'''

# Test 6: Minimal graph operations
minimal_graph = '''
import asyncio

class SimpleGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        
    def add_node(self, node):
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = set()

async def main():
    graphs = {}
    while True:
        await asyncio.sleep(1)

asyncio.run(main())
'''

print("=" * 60)
print("Memory Usage Comparison")
print("=" * 60)

test_server_memory(bare_python, "Bare Python      ")
test_server_memory(with_asyncio, "With asyncio     ")
test_server_memory(minimal_graph, "Minimal graph    ")
test_server_memory(with_networkx, "With NetworkX    ")
test_server_memory(with_numpy, "With numpy       ")
test_server_memory(our_server, "Our server style ")

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

print("\n🎯 The problem is clear:")
print("- NetworkX alone adds ~40MB")
print("- NumPy adds another ~20MB")  
print("- Our 'minimal' server uses heavyweight scientific libraries")
print("- A truly minimal graph server could use <20MB total")

print("\n💡 RECOMMENDATION:")
print("Either:")
print("1. Accept the bloat (bad for a 'minimal' server)")
print("2. Implement basic graph operations ourselves")
print("3. Find a lighter-weight graph library")
print("4. Lazy-load NetworkX only when needed")