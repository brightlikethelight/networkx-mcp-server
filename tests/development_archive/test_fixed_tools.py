#!/usr/bin/env python3
"""Test that the previously broken tools are now fixed."""

import json
import subprocess
import sys


def test_fixed_tools():
    """Test CSV import and visualization tools."""
    print("🔧 TESTING FIXED TOOLS 🔧")

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
                "clientInfo": {"name": "fix-test", "version": "1.0.0"},
            },
        )

        print("\n=== TEST 1: CSV IMPORT ===")

        # Test CSV import with actual CSV data (not file path)
        csv_content = """source,target
A,B
B,C
C,D
D,A
E,F"""

        response = send_request(
            "tools/call",
            {
                "name": "import_csv",
                "arguments": {
                    "graph": "csv_test",
                    "csv_data": csv_content,
                    "directed": False,
                },
            },
        )

        if "result" in response and not response["result"].get("isError"):
            result = json.loads(response["result"]["content"][0]["text"])
            print(f"✅ CSV Import successful: {result}")

            # Verify the graph was created correctly
            info_response = send_request(
                "tools/call", {"name": "get_info", "arguments": {"graph": "csv_test"}}
            )
            if "result" in info_response:
                info = json.loads(info_response["result"]["content"][0]["text"])
                print(f"✅ Graph created: {info['nodes']} nodes, {info['edges']} edges")
        else:
            print(f"❌ CSV Import failed: {response}")

        print("\n=== TEST 2: VISUALIZATION ===")

        # Create a small test graph
        send_request(
            "tools/call",
            {
                "name": "create_graph",
                "arguments": {"name": "viz_test", "directed": False},
            },
        )

        send_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {"graph": "viz_test", "nodes": ["A", "B", "C", "D"]},
            },
        )

        send_request(
            "tools/call",
            {
                "name": "add_edges",
                "arguments": {
                    "graph": "viz_test",
                    "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]],
                },
            },
        )

        # Test visualization
        response = send_request(
            "tools/call",
            {
                "name": "visualize_graph",
                "arguments": {"graph": "viz_test", "layout": "spring"},
            },
        )

        if "result" in response and not response["result"].get("isError"):
            result = json.loads(response["result"]["content"][0]["text"])
            if "visualization" in result:
                print("✅ Visualization successful!")
                print(f"   Format: {result['format']}")
                print(f"   Layout: {result['layout']}")
                print(f"   Image size: {len(result['visualization'])} chars")
                # Check if it's a valid base64 image
                if result["visualization"].startswith("data:image/png;base64,"):
                    print("✅ Valid base64 PNG image generated")
                else:
                    print("❌ Invalid image format")
            else:
                print(f"❌ No visualization data in result: {result}")
        else:
            print(f"❌ Visualization failed: {response}")

        print("\n=== TEST 3: EXPORT JSON ===")

        # Test JSON export
        response = send_request(
            "tools/call", {"name": "export_json", "arguments": {"graph": "viz_test"}}
        )

        if "result" in response and not response["result"].get("isError"):
            result = json.loads(response["result"]["content"][0]["text"])
            print("✅ JSON Export successful!")
            # Parse the exported JSON to verify it's valid
            try:
                exported_data = json.loads(result)
                print(f"   Exported {len(exported_data.get('nodes', []))} nodes")
                print(f"   Exported {len(exported_data.get('links', []))} links")
            except:
                print(
                    f"   Result format: {list(result.keys()) if isinstance(result, dict) else type(result)}"
                )
        else:
            print(f"❌ JSON Export failed: {response}")

        print("\n=== TEST 4: COMMUNITY DETECTION ===")

        # Create a graph with clear communities
        send_request(
            "tools/call",
            {
                "name": "create_graph",
                "arguments": {"name": "community_test", "directed": False},
            },
        )

        # Add nodes and edges that form communities
        nodes = ["A", "B", "C", "D", "E", "F", "G", "H"]
        send_request(
            "tools/call",
            {
                "name": "add_nodes",
                "arguments": {"graph": "community_test", "nodes": nodes},
            },
        )

        # Community 1: A, B, C, D (fully connected)
        # Community 2: E, F, G, H (fully connected)
        # Bridge: D-E
        edges = [
            ["A", "B"],
            ["A", "C"],
            ["A", "D"],
            ["B", "C"],
            ["B", "D"],
            ["C", "D"],  # Community 1
            ["E", "F"],
            ["E", "G"],
            ["E", "H"],
            ["F", "G"],
            ["F", "H"],
            ["G", "H"],  # Community 2
            ["D", "E"],  # Bridge
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
            print(f"   Found {result['count']} communities")
            print(f"   Method: {result['method']}")
            for i, community in enumerate(result["communities"]):
                print(f"   Community {i + 1}: {community}")
        else:
            print(f"❌ Community detection failed: {response}")

    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()

    finally:
        process.terminate()
        process.wait()


if __name__ == "__main__":
    test_fixed_tools()
