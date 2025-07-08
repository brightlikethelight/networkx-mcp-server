#!/usr/bin/env python3
"""Test enhanced stdio transport directly."""

import asyncio
import json
from networkx_mcp.protocol.mcp_handler import MCPProtocolHandler
from networkx_mcp.transport.stdio_transport import StdioTransport
import io
import sys


class MockStdio:
    """Mock stdio for testing."""
    
    def __init__(self):
        self.input_buffer = io.BytesIO()
        self.output_buffer = io.BytesIO()
        self.messages_written = []
        
    def write_input(self, data):
        """Write data to input buffer."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        self.input_buffer.write(data)
        self.input_buffer.seek(0)
        
    def get_output(self):
        """Get written output."""
        return self.output_buffer.getvalue()
        
        
async def test_enhanced_transport():
    """Test the enhanced transport directly."""
    print("🧪 Testing Enhanced Stdio Transport\n")
    
    # Create handler and transport
    handler = MCPProtocolHandler()
    transport = StdioTransport(handler)
    
    # Test edge cases
    test_cases = [
        # Valid JSON
        ('{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}', "Valid JSON"),
        
        # Malformed JSON
        ('{"invalid": json}', "Malformed JSON"),
        
        # Unicode
        ('{"jsonrpc":"2.0","id":"unicode","method":"tools/call","params":{"name":"create_graph","arguments":{"name":"测试_🚀","graph_type":"directed"}}}', "Unicode"),
        
        # Empty lines
        ('\n\n\n', "Empty lines"),
        
        # Whitespace
        ('   \t  \n', "Whitespace"),
        
        # Binary data (will be replaced during decode)
        (b'\x00\x01\x02{"jsonrpc":"2.0","id":"binary","method":"test"}\n', "Binary prefix"),
        
        # Large message
        (json.dumps({
            "jsonrpc": "2.0",
            "id": "large",
            "method": "tools/call",
            "params": {
                "name": "add_nodes",
                "arguments": {
                    "graph_name": "test",
                    "nodes": [f"node_{i}" for i in range(1000)]
                }
            }
        }), "Large message"),
        
        # Special characters
        ('{"jsonrpc":"2.0","id":"special","method":"test","params":{"data":"\\n\\r\\t\\"\\\\"}}', "Special chars"),
    ]
    
    # Process each test case
    results = []
    for test_input, test_name in test_cases:
        print(f"Testing: {test_name}")
        
        try:
            # Convert to async iterator
            async def single_message():
                if isinstance(test_input, bytes):
                    yield test_input.decode('utf-8', errors='replace').strip()
                else:
                    yield test_input.strip()
                    
            # Process message
            response = None
            async for msg in single_message():
                if msg:  # Only process non-empty messages
                    try:
                        response = await handler.handle_message(msg)
                        if response:
                            response_dict = json.loads(response) if isinstance(response, str) else response
                            if "error" in response_dict:
                                print(f"  ❌ Error: {response_dict['error']['message']}")
                            else:
                                print(f"  ✅ Success")
                        else:
                            print(f"  ⚠️  No response (notification?)")
                    except Exception as e:
                        print(f"  ❌ Exception: {e}")
                else:
                    print(f"  ⚠️  Empty message ignored")
                    
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            
    # Test concurrent handling
    print("\n⚡ Testing Concurrent Message Handling")
    
    # Initialize first
    await handler.handle_message('{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}')
    
    # Create multiple concurrent requests
    tasks = []
    for i in range(10):
        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": f"concurrent_{i}",
            "method": "tools/call",
            "params": {
                "name": "create_graph",
                "arguments": {
                    "name": f"graph_{i}",
                    "graph_type": "undirected"
                }
            }
        })
        tasks.append(handler.handle_message(msg))
        
    # Wait for all
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    success_count = sum(1 for r in responses if r and "error" not in r)
    print(f"✅ {success_count}/10 concurrent requests succeeded")
    
    print("\n✨ Enhanced transport test complete!")


if __name__ == "__main__":
    asyncio.run(test_enhanced_transport())