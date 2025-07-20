#!/usr/bin/env python3
"""
Test academic features in detail
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from networkx_mcp.server import NetworkXMCPServer


async def test_academic_features():
    """Test academic features comprehensively."""
    print("=== Testing Academic Features in Detail ===")
    
    server = NetworkXMCPServer()
    
    # Test DOI resolution
    print("Testing DOI resolution...")
    response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 1,
        "params": {
            "name": "resolve_doi",
            "arguments": {"doi": "10.1038/nature12373"}
        }
    })
    
    print(f"DOI resolution response: {response}")
    
    # Test building citation network (will likely fail without proper DOIs)
    print("\nTesting citation network building...")
    response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 2,
        "params": {
            "name": "build_citation_network",
            "arguments": {
                "graph": "citation_test",
                "seed_dois": ["10.1038/nature12373"],
                "max_depth": 1
            }
        }
    })
    
    print(f"Citation network response: {response}")
    
    # Test BibTeX export
    print("\nTesting BibTeX export...")
    response = await server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 3,
        "params": {
            "name": "export_bibtex",
            "arguments": {"graph": "citation_test"}
        }
    })
    
    print(f"BibTeX export response: {response}")


if __name__ == "__main__":
    asyncio.run(test_academic_features())