#!/usr/bin/env python3
"""Trace imports to find heavyweight dependencies."""

import sys
import importlib.util

class ImportTracer:
    def __init__(self):
        self.imports = []
        self.original_import = __builtins__.__import__
        
    def trace_import(self, name, *args, **kwargs):
        """Track all imports"""
        self.imports.append(name)
        return self.original_import(name, *args, **kwargs)
        
    def start_tracing(self):
        __builtins__.__import__ = self.trace_import
        
    def stop_tracing(self):
        __builtins__.__import__ = self.original_import
        
    def analyze_heavyweight(self):
        """Find unnecessary imports"""
        heavyweight = {
            'pandas': 0,
            'scipy': 0,
            'matplotlib': 0,
            'sklearn': 0,
            'torch': 0,
            'tensorflow': 0
        }
        
        for imp in self.imports:
            for heavy in heavyweight:
                if imp.startswith(heavy):
                    heavyweight[heavy] += 1
                    
        return heavyweight

# Trace server imports
print("🔍 Tracing imports for NetworkX MCP Server...")
print("=" * 60)

tracer = ImportTracer()
tracer.start_tracing()

# Add src to path
sys.path.insert(0, 'src')

try:
    import networkx_mcp.server
    tracer.stop_tracing()
    
    print(f"Total imports: {len(tracer.imports)}")
    print(f"Unique imports: {len(set(tracer.imports))}")
    
    heavy = tracer.analyze_heavyweight()
    print(f"\nHeavyweight imports found:")
    for lib, count in heavy.items():
        if count > 0:
            print(f"  {lib}: {count} modules")
    
    # Find the culprit
    if any(imp.startswith('pandas') for imp in tracer.imports):
        print("\n🚨 PANDAS IMPORTED! Finding source...")
        
        # Find pandas in import order
        for i, imp in enumerate(tracer.imports):
            if imp.startswith('pandas'):
                print(f"\nFirst pandas import at position {i}: {imp}")
                # Show context
                print("Previous imports:")
                for j in range(max(0, i-5), i):
                    print(f"  {j}: {tracer.imports[j]}")
                break
                
    # Show import chain for io_handlers
    print("\n📊 Import chain analysis:")
    io_handler_found = False
    for i, imp in enumerate(tracer.imports):
        if 'io_handlers' in imp:
            io_handler_found = True
            print(f"\nio_handlers imported at position {i}")
        if io_handler_found and imp.startswith('pandas'):
            print(f"→ pandas imported at position {i} (after io_handlers)")
            break
            
except Exception as e:
    tracer.stop_tracing()
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()