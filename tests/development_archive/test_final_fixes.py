#!/usr/bin/env python3
"""Test the final fixes for all tools."""

import json
import subprocess
import sys


def test_final_fixes():
    """Test the final fixes for the remaining broken tools."""
    print("🔧 TESTING FINAL FIXES 🔧")

    # Start server
    process = subprocess.Popen(
        [sys.executable, "-m", "networkx_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def send_request(method, params=None):
        request = {"jsonrpc": "2.0", "method": method, "id": 1}
        if params:
            request["params"] = params

        try:
            request_str = json.dumps(request) + "\n"
            process.stdin.write(request_str)
            process.stdin.flush()

            response_line = process.stdout.readline()
            return json.loads(response_line.strip()) if response_line else None
        except Exception as e:
            return {"error": str(e)}

    try:
        # Initialize
        send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "final-test", "version": "1.0.0"},
            },
        )

        print("\n=== TEST 1: CSV IMPORT (CLEAN) ===")

        # Test CSV import with actual CSV data (not file path)
        csv_content = """source,target
X,Y
Y,Z
Z,X"""

        # Use a unique graph name to avoid conflicts
        response = send_request(
            "tools/call",
            {
                "name": "import_csv",
                "arguments": {
                    "graph": "csv_test_unique",
                    "csv_data": csv_content,
                    "directed": True,
                },
            },
        )

        if "result" in response and not response["result"].get("isError"):
            result = json.loads(response["result"]["content"][0]["text"])
            print(f"✅ CSV Import successful: {result}")
            assert result["nodes"] == 3, f"Expected 3 nodes, got {result['nodes']}"
            assert result["edges"] == 3, f"Expected 3 edges, got {result['edges']}"
        else:
            print(f"❌ CSV Import failed: {response}")

        print("\n=== TEST 2: COMMUNITY DETECTION ===")

        # Create a graph with clear communities
        send_request(
            "tools/call",
            {
                "name": "create_graph",
                "arguments": {"name": "community_test", "directed": False},
            },
        )

        # Add nodes and edges that form communities
        nodes = ["A", "B", "C", "D", "E", "F"]
        send_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {"graph": "community_test", "nodes": nodes},
            },
        )

        # Two clear communities: ABC and DEF with one bridge
        edges = [
            ["A", "B"],
            ["B", "C"],
            ["C", "A"],  # Community 1
            ["D", "E"],
            ["E", "F"],
            ["F", "D"],  # Community 2
            ["C", "D"],  # Bridge
        ]
        send_request(
            "tools/call",
            {
                "name": "add_edges",
                "arguments": {"graph": "community_test", "edges": edges},
            },
        )

        # Test community detection
        response = send_request(
            "tools/call",
            {"name": "community_detection", "arguments": {"graph": "community_test"}},
        )

        if "result" in response and not response["result"].get("isError"):
            result = json.loads(response["result"]["content"][0]["text"])
            print("✅ Community detection successful!")
            print(f"   Found {result.get('num_communities', 0)} communities")
            print(f"   Method: {result.get('method', 'N/A')}")
            print(f"   Communities: {result.get('communities', [])[:3]}")
        else:
            print(f"❌ Community detection failed: {response}")

        print("\n=== TEST 3: COLLABORATION PATTERNS ===")

        # Create a graph for collaboration testing
        send_request(
            "tools/call",
            {
                "name": "create_graph",
                "arguments": {"name": "collab_test", "directed": False},
            },
        )

        send_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {
                    "graph": "collab_test",
                    "nodes": ["Group1", "Group2", "Group3"],
                },
            },
        )

        send_request(
            "tools/call",
            {
                "name": "add_edges",
                "arguments": {
                    "graph": "collab_test",
                    "edges": [["Group1", "Group2"], ["Group2", "Group3"]],
                },
            },
        )

        response = send_request(
            "tools/call",
            {
                "name": "find_collaboration_patterns",
                "arguments": {"graph": "collab_test"},
            },
        )

        if "result" in response and not response["result"].get("isError"):
            result = json.loads(response["result"]["content"][0]["text"])
            print("✅ Collaboration patterns found!")
            if "collaboration_clusters" in result:
                print(f"   Clusters: {len(result['collaboration_clusters'])}")
            else:
                print(f"   Result keys: {list(result.keys())}")
        else:
            print(f"❌ Collaboration patterns failed: {response}")

        print("\n=== TEST 4: RESEARCH TRENDS ===")

        response = send_request(
            "tools/call",
            {"name": "detect_research_trends", "arguments": {"graph": "collab_test"}},
        )

        if "result" in response and not response["result"].get("isError"):
            result = json.loads(response["result"]["content"][0]["text"])
            print("✅ Research trends analyzed!")
            print(f"   Trend: {result.get('trend', 'N/A')}")
            if "trends_by_year" in result:
                print(f"   Time periods: {len(result['trends_by_year'])}")
            if "node_count" in result:
                print(
                    f"   Graph stats: {result['node_count']} nodes, {result['edge_count']} edges"
                )
        else:
            print(f"❌ Research trends failed: {response}")

        print("\n=== TEST 5: RECOMMEND PAPERS ===")

        # Create a citation network
        send_request(
            "tools/call",
            {
                "name": "create_graph",
                "arguments": {"name": "citation_test", "directed": True},
            },
        )

        papers = ["Paper1", "Paper2", "Paper3", "Paper4", "Paper5"]
        send_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {"graph": "citation_test", "nodes": papers},
            },
        )

        # Paper1 cites Paper2 and Paper3
        # Paper4 and Paper5 also cite Paper2
        edges = [
            ["Paper1", "Paper2"],
            ["Paper1", "Paper3"],
            ["Paper4", "Paper2"],
            ["Paper5", "Paper2"],
        ]
        send_request(
            "tools/call",
            {
                "name": "add_edges",
                "arguments": {"graph": "citation_test", "edges": edges},
            },
        )

        # Test recommendations with correct parameter name
        response = send_request(
            "tools/call",
            {
                "name": "recommend_papers",
                "arguments": {
                    "graph": "citation_test",
                    "seed_doi": "Paper1",  # Correct parameter name
                    "max_recommendations": 3,
                },
            },
        )

        if "result" in response and not response["result"].get("isError"):
            result = json.loads(response["result"]["content"][0]["text"])
            print("✅ Paper recommendations successful!")
            print(f"   Found {len(result.get('recommendations', []))} recommendations")
            for rec in result.get("recommendations", [])[:3]:
                if isinstance(rec, dict):
                    print(
                        f"   - {rec.get('paper', rec.get('doi', 'Unknown'))} (score: {rec.get('score', 'N/A')})"
                    )
        else:
            print(f"❌ Paper recommendations failed: {response}")

    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()

    finally:
        process.terminate()
        process.wait()


if __name__ == "__main__":
    test_final_fixes()
