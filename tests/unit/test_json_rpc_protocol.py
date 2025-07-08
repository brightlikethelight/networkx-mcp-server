"""Unit tests for JSON-RPC 2.0 protocol implementation."""

import json
import pytest
import asyncio
from typing import Any, Dict

from networkx_mcp.protocol import (
    JsonRpcHandler,
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    JsonRpcErrorCode,
    create_error_response
)


class TestJsonRpcParsing:
    """Test JSON-RPC message parsing."""
    
    @pytest.mark.asyncio
    async def test_parse_valid_request(self):
        """Test parsing valid JSON-RPC request."""
        handler = JsonRpcHandler()
        message = '{"jsonrpc":"2.0","id":1,"method":"test","params":{"foo":"bar"}}'
        
        request = await handler.parse_message(message)
        
        assert isinstance(request, JsonRpcRequest)
        assert request.jsonrpc == "2.0"
        assert request.id == 1
        assert request.method == "test"
        assert request.params == {"foo": "bar"}
    
    @pytest.mark.asyncio
    async def test_parse_notification(self):
        """Test parsing notification (no id)."""
        handler = JsonRpcHandler()
        message = '{"jsonrpc":"2.0","method":"notify","params":{"data":"test"}}'
        
        request = await handler.parse_message(message)
        
        assert isinstance(request, JsonRpcRequest)
        assert request.is_notification()
        assert request.id is None
    
    @pytest.mark.asyncio
    async def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        handler = JsonRpcHandler()
        message = '{"invalid json}'
        
        result = await handler.parse_message(message)
        
        assert isinstance(result, JsonRpcError)
        assert result.code == JsonRpcErrorCode.PARSE_ERROR
        assert "Parse error" in result.message
    
    @pytest.mark.asyncio
    async def test_parse_missing_jsonrpc(self):
        """Test parsing request missing jsonrpc field."""
        handler = JsonRpcHandler()
        message = '{"id":1,"method":"test"}'
        
        result = await handler.parse_message(message)
        
        assert isinstance(result, JsonRpcError)
        assert result.code == JsonRpcErrorCode.INVALID_REQUEST
        assert "Missing jsonrpc field" in result.data
    
    @pytest.mark.asyncio
    async def test_parse_missing_method(self):
        """Test parsing request missing method field."""
        handler = JsonRpcHandler()
        message = '{"jsonrpc":"2.0","id":1}'
        
        result = await handler.parse_message(message)
        
        assert isinstance(result, JsonRpcError)
        assert result.code == JsonRpcErrorCode.INVALID_REQUEST
        assert "Missing method field" in result.data
    
    @pytest.mark.asyncio
    async def test_parse_batch_request(self):
        """Test parsing batch request."""
        handler = JsonRpcHandler()
        message = '[{"jsonrpc":"2.0","id":1,"method":"test1"},{"jsonrpc":"2.0","id":2,"method":"test2"}]'
        
        requests = await handler.parse_message(message)
        
        assert isinstance(requests, list)
        assert len(requests) == 2
        assert all(isinstance(r, JsonRpcRequest) for r in requests)
        assert requests[0].method == "test1"
        assert requests[1].method == "test2"
    
    @pytest.mark.asyncio
    async def test_parse_empty_batch(self):
        """Test parsing empty batch request."""
        handler = JsonRpcHandler()
        message = '[]'
        
        result = await handler.parse_message(message)
        
        assert isinstance(result, JsonRpcError)
        assert result.code == JsonRpcErrorCode.INVALID_REQUEST
        assert "Batch request cannot be empty" in result.data
    
    @pytest.mark.asyncio 
    async def test_parse_non_object_request(self):
        """Test parsing non-object request."""
        handler = JsonRpcHandler()
        
        for message in ['"string"', '42', 'null', 'true']:
            result = await handler.parse_message(message)
            
            assert isinstance(result, JsonRpcError)
            assert result.code == JsonRpcErrorCode.INVALID_REQUEST
            assert "Request must be an object" in result.data


class TestJsonRpcHandling:
    """Test JSON-RPC request handling."""
    
    @pytest.mark.asyncio
    async def test_handle_valid_request(self):
        """Test handling valid request."""
        handler = JsonRpcHandler()
        
        # Register test method
        async def test_method(params):
            return {"echo": params}
        
        handler.register_method("test", test_method)
        
        request = JsonRpcRequest(id=1, method="test", params={"foo": "bar"})
        response = await handler.handle_request(request)
        
        assert isinstance(response, JsonRpcResponse)
        assert response.id == 1
        assert response.result == {"echo": {"foo": "bar"}}
        assert response.error is None
    
    @pytest.mark.asyncio
    async def test_handle_method_not_found(self):
        """Test handling request for non-existent method."""
        handler = JsonRpcHandler()
        
        request = JsonRpcRequest(id=1, method="unknown")
        response = await handler.handle_request(request)
        
        assert isinstance(response, JsonRpcResponse)
        assert response.id == 1
        assert response.result is None
        assert response.error is not None
        assert response.error["code"] == JsonRpcErrorCode.METHOD_NOT_FOUND
    
    @pytest.mark.asyncio
    async def test_handle_notification(self):
        """Test handling notification (no response)."""
        handler = JsonRpcHandler()
        
        called = False
        
        def notify_method(params):
            nonlocal called
            called = True
        
        handler.register_method("notify", notify_method)
        
        request = JsonRpcRequest(method="notify", params={})
        response = await handler.handle_request(request)
        
        assert response is None  # No response for notifications
        assert called  # But method was called
    
    @pytest.mark.asyncio
    async def test_handle_method_error(self):
        """Test handling method that raises error."""
        handler = JsonRpcHandler()
        
        def error_method(params):
            raise ValueError("Test error")
        
        handler.register_method("error", error_method)
        
        request = JsonRpcRequest(id=1, method="error")
        response = await handler.handle_request(request)
        
        assert isinstance(response, JsonRpcResponse)
        assert response.id == 1
        assert response.result is None
        assert response.error is not None
        assert response.error["code"] == JsonRpcErrorCode.INTERNAL_ERROR
        assert "Test error" in response.error["data"]
    
    @pytest.mark.asyncio
    async def test_handle_invalid_params(self):
        """Test handling request with invalid parameters."""
        handler = JsonRpcHandler()
        
        def strict_method(params: Dict[str, Any]):
            # This expects a dict
            return params["required_field"]
        
        handler.register_method("strict", strict_method)
        
        # Send string instead of dict
        request = JsonRpcRequest(id=1, method="strict", params="not a dict")
        response = await handler.handle_request(request)
        
        assert isinstance(response, JsonRpcResponse)
        assert response.error is not None
        # Could be INVALID_PARAMS or INTERNAL_ERROR depending on how the error is caught
        assert response.error["code"] in [JsonRpcErrorCode.INVALID_PARAMS, JsonRpcErrorCode.INTERNAL_ERROR]


class TestJsonRpcMessages:
    """Test JSON-RPC message handling."""
    
    @pytest.mark.asyncio
    async def test_handle_single_message(self):
        """Test handling single JSON-RPC message."""
        handler = JsonRpcHandler()
        
        handler.register_method("echo", lambda params: params)
        
        message = '{"jsonrpc":"2.0","id":1,"method":"echo","params":{"test":true}}'
        response = await handler.handle_message(message)
        
        assert response is not None
        response_data = json.loads(response)
        
        assert response_data["jsonrpc"] == "2.0"
        assert response_data["id"] == 1
        assert response_data["result"] == {"test": True}
    
    @pytest.mark.asyncio
    async def test_handle_batch_message(self):
        """Test handling batch JSON-RPC message."""
        handler = JsonRpcHandler()
        
        handler.register_method("echo", lambda params: params)
        handler.register_method("reverse", lambda params: params[::-1] if isinstance(params, str) else None)
        
        message = '''[
            {"jsonrpc":"2.0","id":1,"method":"echo","params":{"test":1}},
            {"jsonrpc":"2.0","id":2,"method":"reverse","params":"hello"},
            {"jsonrpc":"2.0","method":"echo","params":"notification"}
        ]'''
        
        response = await handler.handle_message(message)
        
        assert response is not None
        response_data = json.loads(response)
        
        assert isinstance(response_data, list)
        assert len(response_data) == 2  # Notification not included
        
        # Check first response
        assert response_data[0]["id"] == 1
        assert response_data[0]["result"] == {"test": 1}
        
        # Check second response
        assert response_data[1]["id"] == 2
        assert response_data[1]["result"] == "olleh"
    
    @pytest.mark.asyncio
    async def test_handle_parse_error_message(self):
        """Test handling message with parse error."""
        handler = JsonRpcHandler()
        
        message = '{"invalid": json}'
        response = await handler.handle_message(message)
        
        assert response is not None
        response_data = json.loads(response)
        
        assert "error" in response_data
        assert response_data["error"]["code"] == JsonRpcErrorCode.PARSE_ERROR


class TestJsonRpcConcurrency:
    """Test JSON-RPC concurrent request handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        handler = JsonRpcHandler()
        
        # Register methods with different delays
        async def slow_method(params):
            await asyncio.sleep(0.1)
            return {"slow": True, "params": params}
        
        async def fast_method(params):
            return {"fast": True, "params": params}
        
        handler.register_method("slow", slow_method)
        handler.register_method("fast", fast_method)
        
        # Create multiple requests
        requests = [
            JsonRpcRequest(id=f"req{i}", method="slow" if i % 2 else "fast", params={"index": i})
            for i in range(10)
        ]
        
        # Handle concurrently
        tasks = [handler.handle_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        
        # Verify all responses
        assert len(responses) == 10
        assert all(isinstance(r, JsonRpcResponse) for r in responses)
        assert all(r.error is None for r in responses)
        
        # Check that fast methods completed even with slow methods running
        fast_responses = [r for r in responses if r.result.get("fast")]
        slow_responses = [r for r in responses if r.result.get("slow")]
        
        assert len(fast_responses) == 5
        assert len(slow_responses) == 5


class TestJsonRpcErrorHandling:
    """Test JSON-RPC error scenarios."""
    
    @pytest.mark.asyncio
    async def test_create_error_response(self):
        """Test creating error response."""
        response = create_error_response(
            JsonRpcErrorCode.METHOD_NOT_FOUND,
            "Method not found",
            data="Unknown method: test",
            request_id=123
        )
        
        assert isinstance(response, JsonRpcResponse)
        assert response.id == 123
        assert response.error is not None
        assert response.error["code"] == JsonRpcErrorCode.METHOD_NOT_FOUND
        assert response.error["message"] == "Method not found"
        assert response.error["data"] == "Unknown method: test"
    
    def test_response_validation(self):
        """Test response object validation."""
        # Valid response with result
        response = JsonRpcResponse(id=1, result={"success": True})
        assert response.to_dict() == {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"success": True}
        }
        
        # Valid response with error
        response = JsonRpcResponse(id=2, error={"code": -32600, "message": "Invalid Request"})
        assert response.to_dict() == {
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"code": -32600, "message": "Invalid Request"}
        }
        
        # Invalid: both result and error
        with pytest.raises(ValueError, match="cannot have both result and error"):
            JsonRpcResponse(id=3, result="data", error={"code": -1, "message": "error"})
        
        # Invalid: neither result nor error
        with pytest.raises(ValueError, match="must have either result or error"):
            JsonRpcResponse(id=4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])