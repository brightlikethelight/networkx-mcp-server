#!/usr/bin/env python3
"""
Test academic features with proper directed graph setup.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from networkx_mcp.server import NetworkXMCPServer


async def call_tool(server, tool_name, arguments):
    """Helper to call a tool and get the result."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    
    response = await server.handle_request(request)
    if "error" in response:
        raise Exception(response["error"]["message"])
    
    # Check if it's an error response
    if response["result"].get("isError"):
        error_text = response["result"]["content"][0]["text"]
        raise Exception(error_text)
    
    result_text = response["result"]["content"][0]["text"]
    
    # Handle empty results
    if not result_text or result_text == "null":
        return {}
        
    return json.loads(result_text)


async def test_academic_features():
    """Test academic features with proper setup."""
    server = NetworkXMCPServer(auth_required=False, enable_monitoring=False)
    
    print("Testing Academic Features with Directed Graph")
    print("=" * 60)
    
    # Create a directed citation network (papers cite other papers)
    print("\n1. Creating directed citation network...")
    await call_tool(server, "create_graph", {
        "name": "citations", 
        "directed": True  # Important: must be directed for citations
    })
    
    # Add papers as nodes
    papers = [
        "10.1234/paper1",  # Seminal paper
        "10.1234/paper2",  # Cites paper1
        "10.1234/paper3",  # Cites paper1
        "10.1234/paper4",  # Cites paper2 and paper3
        "10.1234/paper5",  # Cites paper4
    ]
    
    print("\n2. Adding papers...")
    result = await call_tool(server, "add_nodes", {
        "graph": "citations",
        "nodes": papers
    })
    print(f"Added {result['added']} papers")
    
    # Add citation relationships (edge from citing to cited)
    citations = [
        ["10.1234/paper2", "10.1234/paper1"],  # paper2 cites paper1
        ["10.1234/paper3", "10.1234/paper1"],  # paper3 cites paper1
        ["10.1234/paper4", "10.1234/paper2"],  # paper4 cites paper2
        ["10.1234/paper4", "10.1234/paper3"],  # paper4 cites paper3
        ["10.1234/paper5", "10.1234/paper4"],  # paper5 cites paper4
    ]
    
    print("\n3. Adding citation relationships...")
    result = await call_tool(server, "add_edges", {
        "graph": "citations",
        "edges": citations
    })
    print(f"Added {result['added']} citations")
    
    # Test paper recommendations
    print("\n4. Testing paper recommendations...")
    try:
        result = await call_tool(server, "recommend_papers", {
            "graph": "citations",
            "seed_doi": "10.1234/paper2",
            "max_recommendations": 5
        })
        print(f"Recommendations based on paper2:")
        print(f"  - Total found: {result.get('total_found', 0)}")
        print(f"  - Based on {result.get('based_on', {})}")
        if result.get('recommendations'):
            print(f"  - Recommended papers: {result['recommendations']}")
        else:
            print("  - No recommendations (expected with mock data)")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test collaboration patterns
    print("\n5. Testing collaboration patterns...")
    try:
        result = await call_tool(server, "find_collaboration_patterns", {
            "graph": "citations"
        })
        print(f"Collaboration patterns: {result}")
    except Exception as e:
        print(f"Error (expected with citation-only data): {e}")
    
    # Test research trends
    print("\n6. Testing research trends...")
    try:
        result = await call_tool(server, "detect_research_trends", {
            "graph": "citations",
            "time_window": 3
        })
        print(f"Research trends: {result}")
    except Exception as e:
        print(f"Error (expected without temporal data): {e}")
    
    # Test BibTeX export
    print("\n7. Testing BibTeX export...")
    try:
        result = await call_tool(server, "export_bibtex", {
            "graph": "citations"
        })
        print(f"BibTeX export successful: {result.get('entries', 0)} entries")
        if 'bibtex_data' in result:
            print("Sample BibTeX data:")
            print(result['bibtex_data'][:200] + "...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with real DOI resolution
    print("\n8. Testing with real DOI...")
    try:
        # Test DOI resolution first
        result = await call_tool(server, "resolve_doi", {
            "doi": "10.1038/nature12373"
        })
        print(f"Resolved DOI: {result.get('title', 'Unknown')}")
        
        # Build small citation network
        print("\n9. Building real citation network...")
        result = await call_tool(server, "build_citation_network", {
            "graph": "real_citations",
            "seed_dois": ["10.1038/nature12373"],
            "max_depth": 1
        })
        print(f"Built network: {result}")
    except Exception as e:
        print(f"Error (may be API rate limit): {e}")
    
    print("\n" + "="*60)
    print("Academic Features Test Complete")


if __name__ == "__main__":
    asyncio.run(test_academic_features())