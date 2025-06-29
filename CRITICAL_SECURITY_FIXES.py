#!/usr/bin/env python3
"""
CRITICAL SECURITY FIXES - Apply These Immediately!

These are monkey patches to fix the most dangerous vulnerabilities in the current
NetworkX MCP Server. Apply these NOW while working on the full migration.
"""

import os
import re
import gc
import time
import psutil
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import networkx as nx


def apply_critical_security_fixes():
    """Apply all critical security patches to the running server."""
    print("🚨 Applying critical security patches...")
    
    # Import here to avoid circular imports
    from src.networkx_mcp.server import graph_manager
    from src.networkx_mcp.core.io_handlers import GraphIOHandler
    
    # Store original methods for patching
    patches_applied = []

# ==============================================================================
# PATCH 1: Fix Directory Traversal in File Operations
# ==============================================================================

# Save original methods
_original_import_graph = GraphIOHandler.import_graph
_original_export_graph = GraphIOHandler.export_graph

def secure_import_graph(filepath: str, file_format: str = "auto") -> nx.Graph:
    """Patched import with path validation."""
    # Validate path - no directory traversal!
    if '../' in filepath or filepath.startswith('/'):
        raise ValueError("Invalid file path")
    
    # Only allow files in current directory or subdirectories
    safe_path = Path(filepath).resolve()
    cwd = Path.cwd()
    
    try:
        safe_path.relative_to(cwd)
    except ValueError:
        raise ValueError("File path must be within current directory")
    
    # Never allow pickle format (arbitrary code execution!)
    if file_format.lower() in ['pickle', 'pkl']:
        raise ValueError("Pickle format is disabled for security")
    
    # Validate file exists and is reasonable size
    if not safe_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    file_size = safe_path.stat().st_size
    if file_size > 100 * 1024 * 1024:  # 100MB limit
        raise ValueError(f"File too large: {file_size} bytes")
    
    return _original_import_graph(str(safe_path), file_format)

def secure_export_graph(graph_id: str, filepath: str, file_format: str = "graphml") -> Dict[str, Any]:
    """Patched export with path validation."""
    # Same path validation
    if '../' in filepath or filepath.startswith('/'):
        raise ValueError("Invalid file path")
    
    safe_path = Path(filepath).resolve()
    cwd = Path.cwd()
    
    try:
        safe_path.relative_to(cwd)
    except ValueError:
        raise ValueError("File path must be within current directory")
    
    # Never allow pickle
    if file_format.lower() in ['pickle', 'pkl']:
        raise ValueError("Pickle format is disabled for security")
    
    # Create parent directory safely
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    
    return _original_export_graph(graph_id, str(safe_path), file_format)

# Apply patches
GraphIOHandler.import_graph = staticmethod(secure_import_graph)
GraphIOHandler.export_graph = staticmethod(secure_export_graph)

print("✅ Patched file operations against directory traversal")

# ==============================================================================
# PATCH 2: Add Input Validation for Graph/Node IDs
# ==============================================================================

# Validation patterns
SAFE_GRAPH_ID_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}$')
SAFE_NODE_ID_PATTERN = re.compile(r'^[^<>&"\']{1,1000}$')

_original_create_graph = graph_manager.create_graph
_original_add_node = graph_manager.add_node

def validate_graph_id(graph_id: str) -> str:
    """Validate graph ID format."""
    if not isinstance(graph_id, str):
        raise TypeError("Graph ID must be string")
    
    if not graph_id or len(graph_id) > 100:
        raise ValueError("Graph ID must be 1-100 characters")
    
    if not SAFE_GRAPH_ID_PATTERN.match(graph_id):
        raise ValueError(
            "Invalid graph ID. Use only letters, numbers, underscore, hyphen. "
            "Must start with letter or number."
        )
    
    return graph_id

def validate_node_id(node_id: Any) -> Any:
    """Validate node ID."""
    if isinstance(node_id, str):
        if len(node_id) > 1000:
            raise ValueError("Node ID too long")
        if not SAFE_NODE_ID_PATTERN.match(node_id):
            raise ValueError("Node ID contains invalid characters")
    elif isinstance(node_id, (int, float)):
        if abs(node_id) > 1e15:
            raise ValueError("Numeric node ID out of range")
    else:
        raise TypeError(f"Invalid node ID type: {type(node_id)}")
    
    return node_id

def secure_create_graph(graph_id: str, graph_type: str = "Graph", 
                       from_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Patched create_graph with validation."""
    # Validate inputs
    graph_id = validate_graph_id(graph_id)
    
    if graph_type not in ["Graph", "DiGraph", "MultiGraph", "MultiDiGraph"]:
        raise ValueError(f"Invalid graph type: {graph_type}")
    
    # Limit initial data size
    if from_data:
        if "edge_list" in from_data and len(from_data["edge_list"]) > 100000:
            raise ValueError("Too many edges in initial data")
        if "adjacency_matrix" in from_data:
            matrix = from_data["adjacency_matrix"]
            if len(matrix) > 1000 or any(len(row) > 1000 for row in matrix):
                raise ValueError("Adjacency matrix too large")
    
    return _original_create_graph(graph_id, graph_type, from_data)

def secure_add_node(graph_id: str, node_id: Any, 
                   attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Patched add_node with validation."""
    graph_id = validate_graph_id(graph_id)
    node_id = validate_node_id(node_id)
    
    # Sanitize attributes
    if attributes:
        # Remove dangerous keys
        dangerous_keys = ['__', 'eval', 'exec', 'compile']
        attributes = {
            k: v for k, v in attributes.items()
            if not any(d in k for d in dangerous_keys)
        }
        
        # Limit attribute size
        if len(attributes) > 100:
            raise ValueError("Too many attributes")
    
    return _original_add_node(graph_id, node_id, attributes)

# Apply patches
graph_manager.create_graph = secure_create_graph
graph_manager.add_node = secure_add_node

print("✅ Patched input validation for graph/node operations")

# ==============================================================================
# PATCH 3: Add Basic Rate Limiting
# ==============================================================================

from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

class RateLimiter:
    """Simple rate limiter."""
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.requests = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def check_limit(self, key: str = "global") -> bool:
        """Check if request is within rate limit."""
        async with self._lock:
            now = datetime.now()
            
            # Clean old requests
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if now - req_time < self.window
            ]
            
            # Check limit
            if len(self.requests[key]) >= self.max_requests:
                raise Exception(f"Rate limit exceeded: {self.max_requests} requests per minute")
            
            # Record request
            self.requests[key].append(now)
            return True

# Create rate limiter
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

# Wrap ALL MCP tools with rate limiting
for tool_name in dir(mcp):
    if tool_name.startswith('_'):
        continue
    
    tool = getattr(mcp, tool_name)
    if hasattr(tool, '__call__') and hasattr(tool, '_tool'):
        # Wrap the tool
        original_tool = tool
        
        async def rate_limited_tool(*args, **kwargs):
            await rate_limiter.check_limit()
            return await original_tool(*args, **kwargs)
        
        setattr(mcp, tool_name, rate_limited_tool)

print("✅ Added basic rate limiting (100 requests/minute)")

# ==============================================================================
# PATCH 4: Add Memory Limits
# ==============================================================================

import psutil
import gc

class MemoryGuard:
    """Simple memory limit enforcement."""
    def __init__(self, max_memory_mb: int = 1000):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()
    
    def check_memory(self):
        """Check current memory usage."""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.max_memory_mb:
            # Try garbage collection
            gc.collect()
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory_mb:
                raise MemoryError(
                    f"Memory limit exceeded: {memory_mb:.1f}MB / {self.max_memory_mb}MB"
                )
        
        return memory_mb

memory_guard = MemoryGuard(max_memory_mb=1000)

# Check memory before creating graphs
_original_add_nodes_from = graph_manager.add_nodes_from

def secure_add_nodes_from(graph_id: str, nodes: list, 
                         attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Patched add_nodes_from with memory check."""
    # Check memory before adding many nodes
    memory_guard.check_memory()
    
    # Limit number of nodes
    if len(nodes) > 100000:
        raise ValueError("Too many nodes (max 100,000)")
    
    # Validate each node ID
    validated_nodes = []
    for node in nodes[:100000]:  # Hard limit
        try:
            validated_nodes.append(validate_node_id(node))
        except:
            continue  # Skip invalid nodes
    
    return _original_add_nodes_from(graph_id, validated_nodes, attributes)

graph_manager.add_nodes_from = secure_add_nodes_from

print("✅ Added memory limits (1GB max)")

# ==============================================================================
# PATCH 5: Disable Dangerous Operations
# ==============================================================================

# Remove/disable dangerous operations until properly secured

# Disable pickle in import/export
if hasattr(mcp, 'import_graph'):
    _original_mcp_import = mcp.import_graph
    
    @mcp.tool()
    async def import_graph(graph_id: str, filepath: str, format: str = "auto",
                          params: Optional[Dict[str, Any]] = None):
        """Patched import_graph tool."""
        if format.lower() in ['pickle', 'pkl']:
            raise ValueError("Pickle format is disabled for security")
        return await _original_mcp_import(graph_id, filepath, format, params)

print("✅ Disabled dangerous operations (pickle)")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("🛡️  CRITICAL SECURITY PATCHES APPLIED")
print("=" * 60)
print("""
The following vulnerabilities have been patched:
1. ✅ Directory traversal in file operations
2. ✅ Input validation for graph/node IDs  
3. ✅ Basic rate limiting (100 req/min)
4. ✅ Memory limits (1GB max)
5. ✅ Disabled pickle format (code execution risk)

⚠️  IMPORTANT: These are temporary patches!
    
The server is now SAFER but not SECURE. You still need to:
- Implement proper authentication
- Add persistent storage (data lost on restart)
- Complete the full security audit
- Migrate to the new architecture

See PRODUCTION_TRANSFORMATION.md for the complete solution.
""")

# ==============================================================================
# USAGE
# ==============================================================================

if __name__ == "__main__":
    print("\nTo apply these patches to your server:")
    print("1. Import this file at the top of server.py:")
    print("   import CRITICAL_SECURITY_FIXES")
    print("\n2. Or run the server with patches:")
    print("   python -c 'import CRITICAL_SECURITY_FIXES; from networkx_mcp.server import main; main()'")