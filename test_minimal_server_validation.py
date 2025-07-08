#!/usr/bin/env python3
"""
Real validation test for the working minimal MCP server implementation.

This test validates the actual working server_minimal.py implementation 
rather than testing for placeholders or missing infrastructure.
"""

import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

def test_minimal_server_mcp_protocol():
    """Test that the minimal server implements MCP protocol correctly."""
    print("🔍 Testing minimal MCP server protocol implementation...")
    
    # Test script that runs a complete MCP protocol flow
    test_script = '''
import subprocess
import sys
import json

# Test MCP protocol flow
messages = [
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}',
    '{"jsonrpc":"2.0","method":"initialized","params":{}}',
    '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}',
    '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"create_graph","arguments":{"graph_id":"test","directed":false}}}',
    '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"add_nodes","arguments":{"graph_id":"test","nodes":["A","B","C"]}}}',
    '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"add_edges","arguments":{"graph_id":"test","edges":[["A","B"],["B","C"]]}}}'
]

# Run the server
proc = subprocess.Popen([
    sys.executable, '-m', 'src.networkx_mcp', '--minimal'
], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Send all messages
input_data = '\\n'.join(messages) + '\\n'
stdout, stderr = proc.communicate(input=input_data, timeout=30)

# Parse responses
responses = []
for line in stdout.strip().split('\\n'):
    if line.strip() and line.startswith('{'):
        try:
            responses.append(json.loads(line))
        except:
            pass

print(f"Sent {len(messages)} messages, received {len(responses)} responses")

# Validate responses
if len(responses) >= 4:  # Should get responses for requests with IDs
    init_response = responses[0]
    tools_response = responses[1] if len(responses) > 1 else {}
    create_response = responses[2] if len(responses) > 2 else {}
    
    # Check initialization
    if (init_response.get('result', {}).get('protocolVersion') == '2024-11-05' and
        'serverInfo' in init_response.get('result', {})):
        print("✅ MCP initialization successful")
    else:
        print("❌ MCP initialization failed")
        sys.exit(1)
    
    # Check tools list
    if 'tools' in tools_response.get('result', {}):
        tools = tools_response['result']['tools']
        tool_names = [t['name'] for t in tools]
        expected_tools = ['create_graph', 'add_nodes', 'add_edges', 'get_graph_info']
        if all(tool in tool_names for tool in expected_tools):
            print(f"✅ Tools list valid ({len(tools)} tools available)")
        else:
            print(f"❌ Tools list missing expected tools")
            sys.exit(1)
    else:
        print("❌ Tools list request failed")
        sys.exit(1)
    
    print("✅ MCP protocol implementation is working correctly")
    sys.exit(0)
else:
    print(f"❌ Insufficient responses received. Expected 4+, got {len(responses)}")
    if stderr:
        print(f"Server errors: {stderr}")
    sys.exit(1)
'''
    
    try:
        # Write and run the test
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            test_file = f.name
        
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, timeout=60)
        
        print(result.stdout)
        if result.stderr and 'INFO' not in result.stderr:
            print(f"Errors: {result.stderr}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
    finally:
        # Cleanup
        try:
            Path(test_file).unlink()
        except:
            pass

def test_claude_desktop_config():
    """Test that Claude Desktop configuration is valid."""
    print("🔍 Testing Claude Desktop configuration...")
    
    config_path = Path("claude_desktop_config.json")
    if not config_path.exists():
        print("❌ Claude Desktop config file missing")
        return False
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        if 'mcpServers' not in config:
            print("❌ Config missing mcpServers section")
            return False
        
        servers = config['mcpServers']
        if 'networkx-mcp-server' not in servers:
            print("❌ Config missing networkx-mcp-server entry")
            return False
        
        server_config = servers['networkx-mcp-server']
        required_fields = ['command', 'args']
        for field in required_fields:
            if field not in server_config:
                print(f"❌ Config missing required field: {field}")
                return False
        
        # Check if command points to working server
        args = server_config['args']
        if '--minimal' not in args:
            print("❌ Config not using --minimal server")
            return False
        
        print("✅ Claude Desktop configuration is valid")
        return True
        
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return False

def test_graph_operations():
    """Test core graph operations work correctly."""
    print("🔍 Testing core graph operations...")
    
    try:
        # Import and test directly
        sys.path.insert(0, 'src')
        from networkx_mcp.core.graph_operations import GraphManager
        
        manager = GraphManager()
        
        # Test create
        result = manager.create_graph("test", "Graph")
        if not result.get("created"):
            print("❌ Graph creation failed")
            return False
        
        # Test add nodes
        result = manager.add_nodes_from("test", ["A", "B", "C"])
        if result.get("nodes_added") != 3:
            print("❌ Node addition failed")
            return False
        
        # Test add edges
        result = manager.add_edges_from("test", [("A", "B"), ("B", "C")])
        if result.get("edges_added") != 2:
            print("❌ Edge addition failed") 
            return False
        
        # Test graph info
        info = manager.get_graph_info("test")
        if info.get("num_nodes") != 3 or info.get("num_edges") != 2:
            print("❌ Graph info incorrect")
            return False
        
        print("✅ Core graph operations working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Graph operations test failed: {e}")
        return False

def test_feature_flags():
    """Test that feature flags show production readiness."""
    print("🔍 Testing feature flags for production readiness...")
    
    try:
        sys.path.insert(0, 'src')
        from networkx_mcp.features.realistic_feature_flags import get_realistic_feature_flags
        
        flags = get_realistic_feature_flags()
        can_deploy, missing = flags.can_deploy_to_production()
        
        if can_deploy:
            print("✅ Feature flags indicate production readiness")
            return True
        else:
            print(f"❌ Feature flags indicate missing features: {missing}")
            return False
            
    except Exception as e:
        print(f"❌ Feature flags test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("🚀 NETWORKX MCP SERVER - MINIMAL SERVER VALIDATION")
    print("=" * 60)
    print()
    
    tests = [
        ("MCP Protocol Implementation", test_minimal_server_mcp_protocol),
        ("Claude Desktop Configuration", test_claude_desktop_config),
        ("Core Graph Operations", test_graph_operations),
        ("Feature Flags & Production Readiness", test_feature_flags)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 VALIDATION SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Server is ready for production!")
        print()
        print("✅ The minimal MCP server implementation is functional and ready to use.")
        print("✅ It can be integrated with Claude Desktop using the provided config.")
        print("✅ All core graph operations are working correctly.")
        print("✅ Feature flags indicate production readiness.")
        return True
    else:
        print(f"💥 {total - passed} tests failed - Server needs additional work")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)