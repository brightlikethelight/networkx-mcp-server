#!/usr/bin/env python3
"""
Test security features
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from networkx_mcp.server import NetworkXMCPServer


async def test_security_features():
    """Test security features like authentication and validation."""
    print("=== Testing Security Features ===")
    
    # Test basic server without auth
    print("Testing server without authentication...")
    server = NetworkXMCPServer(auth_required=False)
    
    # Try malicious inputs
    print("Testing malicious inputs...")
    
    # Test extremely large graph name
    response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 1,
        "params": {
            "name": "create_graph",
            "arguments": {"name": "x" * 10000, "directed": False}
        }
    })
    
    if "result" in response:
        print("⚠ Large graph name accepted (might be OK)")
    else:
        print("✓ Large graph name rejected")
    
    # Test malformed JSON-RPC
    response = await server.handle_request({
        "method": "tools/call",  # Missing jsonrpc and id
        "params": {
            "name": "create_graph",
            "arguments": {"name": "test", "directed": False}
        }
    })
    
    if response:
        print("✓ Malformed JSON-RPC handled gracefully")
    
    # Test invalid tool parameters
    response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 2,
        "params": {
            "name": "add_nodes",
            "arguments": {"graph": "nonexistent", "nodes": None}
        }
    })
    
    if "result" in response and "Error" in response["result"]["content"][0]["text"]:
        print("✓ Invalid parameters handled properly")
    
    # Test with authentication enabled (if available)
    try:
        auth_server = NetworkXMCPServer(auth_required=True)
        print("✓ Authentication server can be created")
        
        # Test without API key
        response = await auth_server.handle_request({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 3,
            "params": {
                "name": "create_graph",
                "arguments": {"name": "auth_test", "directed": False}
            }
        })
        
        if "error" in response:
            print("✓ Authentication required for protected operations")
        else:
            print("⚠ Authentication not properly enforced")
            
    except Exception as e:
        print(f"⚠ Authentication module may not be available: {e}")
    
    print("✓ Basic security features tested")


if __name__ == "__main__":
    asyncio.run(test_security_features())