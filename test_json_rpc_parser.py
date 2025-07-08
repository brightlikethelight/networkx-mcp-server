"""Test JSON-RPC parser with various message formats."""

import asyncio
import json
import sys
from networkx_mcp.protocol import JsonRpcHandler, JsonRpcErrorCode


async def test_parser():
    """Test JSON-RPC parser with various inputs."""
    handler = JsonRpcHandler()
    
    # Test cases
    test_messages = [
        # Valid single request
        '{"jsonrpc":"2.0","id":1,"method":"test","params":{"foo":"bar"}}',
        
        # Valid notification (no id)
        '{"jsonrpc":"2.0","method":"notify","params":{"data":"test"}}',
        
        # Missing method (invalid)
        '{"jsonrpc":"2.0","id":2}',
        
        # Missing jsonrpc version (invalid)
        '{"id":3,"method":"test"}',
        
        # Invalid JSON
        '{"invalid json}',
        
        # Empty batch
        '[]',
        
        # Valid batch with mixed requests
        '[{"jsonrpc":"2.0","id":4,"method":"test1"},{"jsonrpc":"2.0","method":"notify"},{"invalid"}]',
        
        # Non-object request
        '"string instead of object"',
        
        # Null request
        'null',
        
        # Number instead of object
        '42',
    ]
    
    print("🧪 Testing JSON-RPC Parser\n")
    
    for i, message in enumerate(test_messages, 1):
        print(f"Test {i}: {message[:50]}{'...' if len(message) > 50 else ''}")
        
        try:
            parsed = await handler.parse_message(message)
            
            if hasattr(parsed, 'code'):  # JsonRpcError
                print(f"  ❌ Parse Error: Code {parsed.code} - {parsed.message}")
                if parsed.data:
                    print(f"     Data: {parsed.data}")
            elif isinstance(parsed, list):  # Batch
                print(f"  📦 Batch with {len(parsed)} requests:")
                for j, req in enumerate(parsed):
                    if hasattr(req, 'code'):  # Error in batch
                        print(f"     [{j}] ❌ Error: {req.message}")
                    else:
                        print(f"     [{j}] ✅ Method: {req.method}, ID: {req.id}")
            else:  # Single request
                print(f"  ✅ Valid Request: Method={parsed.method}, ID={parsed.id}")
                
        except Exception as e:
            print(f"  💥 Exception: {type(e).__name__}: {e}")
        
        print()
    
    # Test error handling
    print("\n🔧 Testing Error Responses\n")
    
    # Register a test method
    async def test_method(params):
        if params and params.get("error"):
            raise ValueError("Test error")
        return {"success": True, "params": params}
    
    handler.register_method("test", test_method)
    
    # Test various error scenarios
    error_tests = [
        ('{"jsonrpc":"2.0","id":10,"method":"unknown"}', "Method not found"),
        ('{"jsonrpc":"2.0","id":11,"method":"test","params":{"error":true}}', "Internal error"),
        ('{"jsonrpc":"2.0","id":12,"method":"test","params":"wrong type"}', "Valid response"),
    ]
    
    for message, expected in error_tests:
        print(f"Testing: {message}")
        response = await handler.handle_message(message)
        if response:
            resp_data = json.loads(response)
            if "error" in resp_data:
                print(f"  ❌ Error Response: {resp_data['error']['message']}")
            else:
                print(f"  ✅ Success Response: {resp_data.get('result')}")
        else:
            print("  🔕 No response (notification)")
        print()


async def test_concurrent_requests():
    """Test concurrent request handling."""
    handler = JsonRpcHandler()
    
    # Register test methods
    async def slow_method(params):
        await asyncio.sleep(0.1)
        return {"processed": params}
    
    async def fast_method(params):
        return {"immediate": True}
    
    handler.register_method("slow", slow_method)
    handler.register_method("fast", fast_method)
    
    print("\n⚡ Testing Concurrent Requests\n")
    
    # Create multiple requests
    requests = [
        '{"jsonrpc":"2.0","id":"req1","method":"slow","params":{"data":1}}',
        '{"jsonrpc":"2.0","id":"req2","method":"fast","params":{"data":2}}',
        '{"jsonrpc":"2.0","id":"req3","method":"slow","params":{"data":3}}',
        '{"jsonrpc":"2.0","id":"req4","method":"fast","params":{"data":4}}',
    ]
    
    # Handle concurrently
    tasks = [handler.handle_message(req) for req in requests]
    start_time = asyncio.get_event_loop().time()
    responses = await asyncio.gather(*tasks)
    end_time = asyncio.get_event_loop().time()
    
    print(f"Processed {len(requests)} requests in {(end_time - start_time)*1000:.1f}ms")
    
    for i, response in enumerate(responses):
        if response:
            resp_data = json.loads(response)
            print(f"  Response {resp_data['id']}: {resp_data.get('result', resp_data.get('error'))}")


async def test_stdin_input():
    """Test parsing from stdin if provided."""
    if not sys.stdin.isatty():
        handler = JsonRpcHandler()
        
        print("\n📥 Testing stdin input\n")
        
        # Read from stdin
        input_data = sys.stdin.read().strip()
        
        if input_data:
            print(f"Input: {input_data}")
            
            try:
                response = await handler.handle_message(input_data)
                if response:
                    print(f"Response: {response}")
                else:
                    print("No response (notification)")
            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}")


async def main():
    """Run all tests."""
    await test_parser()
    await test_concurrent_requests()
    await test_stdin_input()
    
    print("\n✅ JSON-RPC Parser Tests Complete!")


if __name__ == "__main__":
    asyncio.run(main())