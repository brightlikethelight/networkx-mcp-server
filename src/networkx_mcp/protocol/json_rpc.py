"""JSON-RPC 2.0 protocol implementation for MCP server.

This module provides a complete JSON-RPC 2.0 implementation following the
specification at https://www.jsonrpc.org/specification.
"""

import json
import asyncio
import logging
from typing import Any, Dict, Optional, Union, List, Callable
from dataclasses import dataclass, asdict, field
from enum import IntEnum

logger = logging.getLogger(__name__)


class JsonRpcErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Server implementation defined errors
    SERVER_ERROR_START = -32099
    SERVER_ERROR_END = -32000


@dataclass
class JsonRpcRequest:
    """JSON-RPC 2.0 request message."""
    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    method: str = ""
    params: Optional[Union[Dict[str, Any], List[Any]]] = None
    
    def __post_init__(self):
        """Validate request structure."""
        if self.jsonrpc != "2.0":
            raise ValueError("JSON-RPC version must be 2.0")
        if not self.method:
            raise ValueError("Method is required")
    
    def is_notification(self) -> bool:
        """Check if this is a notification (no id)."""
        return self.id is None


@dataclass
class JsonRpcResponse:
    """JSON-RPC 2.0 response message."""
    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate response structure."""
        if self.jsonrpc != "2.0":
            raise ValueError("JSON-RPC version must be 2.0")
        if self.result is not None and self.error is not None:
            raise ValueError("Response cannot have both result and error")
        if self.result is None and self.error is None:
            raise ValueError("Response must have either result or error")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            data["id"] = self.id
        if self.result is not None:
            data["result"] = self.result
        if self.error is not None:
            data["error"] = self.error
        return data


@dataclass
class JsonRpcError:
    """JSON-RPC 2.0 error object."""
    code: int
    message: str
    data: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        error_dict = {"code": self.code, "message": self.message}
        if self.data is not None:
            error_dict["data"] = self.data
        return error_dict


class JsonRpcHandler:
    """Handler for JSON-RPC 2.0 protocol messages."""
    
    def __init__(self):
        """Initialize the JSON-RPC handler."""
        self.pending_requests: Dict[Union[str, int], asyncio.Future] = {}
        self.method_handlers: Dict[str, Callable] = {}
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Set up default method handlers."""
        # These will be overridden by MCP-specific handlers
        self.register_method("rpc.discover", self._handle_discover)
    
    def register_method(self, method: str, handler: Callable):
        """Register a method handler."""
        self.method_handlers[method] = handler
        logger.debug(f"Registered handler for method: {method}")
    
    async def parse_message(self, message: str) -> Union[JsonRpcRequest, List[JsonRpcRequest], JsonRpcError]:
        """Parse incoming JSON-RPC message (single or batch)."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return JsonRpcError(
                code=JsonRpcErrorCode.PARSE_ERROR,
                message="Parse error",
                data=str(e)
            )
        
        # Handle batch requests
        if isinstance(data, list):
            if not data:  # Empty array
                return JsonRpcError(
                    code=JsonRpcErrorCode.INVALID_REQUEST,
                    message="Invalid Request",
                    data="Batch request cannot be empty"
                )
            
            requests = []
            for item in data:
                try:
                    requests.append(self._parse_single_request(item))
                except Exception as e:
                    # In batch, invalid requests get error responses
                    requests.append(JsonRpcError(
                        code=JsonRpcErrorCode.INVALID_REQUEST,
                        message="Invalid Request",
                        data=str(e)
                    ))
            return requests
        
        # Handle single request
        try:
            return self._parse_single_request(data)
        except Exception as e:
            return JsonRpcError(
                code=JsonRpcErrorCode.INVALID_REQUEST,
                message="Invalid Request",
                data=str(e)
            )
    
    def _parse_single_request(self, data: Dict[str, Any]) -> JsonRpcRequest:
        """Parse a single JSON-RPC request."""
        # Validate required fields
        if not isinstance(data, dict):
            raise ValueError("Request must be an object")
        
        if "jsonrpc" not in data:
            raise ValueError("Missing jsonrpc field")
        
        if "method" not in data:
            raise ValueError("Missing method field")
        
        # Create request object
        return JsonRpcRequest(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params")
        )
    
    async def handle_message(self, message: str) -> Optional[str]:
        """Handle incoming JSON-RPC message and return response."""
        parsed = await self.parse_message(message)
        
        # Handle parse errors
        if isinstance(parsed, JsonRpcError):
            response = JsonRpcResponse(
                id=None,
                error=parsed.to_dict()
            )
            return json.dumps(response.to_dict())
        
        # Handle batch requests
        if isinstance(parsed, list):
            responses = []
            for item in parsed:
                if isinstance(item, JsonRpcError):
                    responses.append(JsonRpcResponse(
                        id=None,
                        error=item.to_dict()
                    ))
                else:
                    response = await self.handle_request(item)
                    if response is not None:  # Don't include notification responses
                        responses.append(response)
            
            if responses:
                return json.dumps([r.to_dict() for r in responses])
            return None  # All notifications
        
        # Handle single request
        response = await self.handle_request(parsed)
        if response is not None:
            return json.dumps(response.to_dict())
        return None  # Notification
    
    async def handle_request(self, request: JsonRpcRequest) -> Optional[JsonRpcResponse]:
        """Route request to appropriate handler."""
        # Notifications don't get responses
        request_id = request.id
        
        try:
            # Check if method exists
            if request.method not in self.method_handlers:
                if request_id is not None:
                    return JsonRpcResponse(
                        id=request_id,
                        error=JsonRpcError(
                            code=JsonRpcErrorCode.METHOD_NOT_FOUND,
                            message="Method not found",
                            data=f"Unknown method: {request.method}"
                        ).to_dict()
                    )
                return None  # No response for notification
            
            # Call handler
            handler = self.method_handlers[request.method]
            
            # Validate params match handler signature
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(request.params)
                else:
                    result = handler(request.params)
            except TypeError as e:
                if request_id is not None:
                    return JsonRpcResponse(
                        id=request_id,
                        error=JsonRpcError(
                            code=JsonRpcErrorCode.INVALID_PARAMS,
                            message="Invalid params",
                            data=str(e)
                        ).to_dict()
                    )
                return None
            
            # Return result
            if request_id is not None:
                return JsonRpcResponse(
                    id=request_id,
                    result=result
                )
            return None  # No response for notification
            
        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            if request_id is not None:
                return JsonRpcResponse(
                    id=request_id,
                    error=JsonRpcError(
                        code=JsonRpcErrorCode.INTERNAL_ERROR,
                        message="Internal error",
                        data=str(e)
                    ).to_dict()
                )
            return None
    
    async def _handle_discover(self, params: Any) -> Dict[str, List[str]]:
        """Handle rpc.discover method."""
        return {"methods": list(self.method_handlers.keys())}
    
    async def send_request(self, method: str, params: Optional[Any] = None, 
                          notification: bool = False) -> Optional[Any]:
        """Send a JSON-RPC request (client functionality)."""
        request_id = None if notification else self._generate_id()
        
        request = JsonRpcRequest(
            id=request_id,
            method=method,
            params=params
        )
        
        if not notification:
            # Create future for response
            future = asyncio.Future()
            self.pending_requests[request_id] = future
            
        # This would be sent over transport
        request_json = json.dumps(asdict(request))
        logger.debug(f"Sending request: {request_json}")
        
        if notification:
            return None
            
        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            del self.pending_requests[request_id]
            raise
    
    def handle_response(self, response: JsonRpcResponse):
        """Handle incoming response (client functionality)."""
        if response.id in self.pending_requests:
            future = self.pending_requests.pop(response.id)
            if response.error:
                future.set_exception(Exception(response.error))
            else:
                future.set_result(response.result)
    
    def _generate_id(self) -> Union[str, int]:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())


class JsonRpcBatchHandler:
    """Handler for JSON-RPC 2.0 batch requests."""
    
    def __init__(self, handler: JsonRpcHandler):
        """Initialize batch handler."""
        self.handler = handler
    
    async def handle_batch(self, requests: List[JsonRpcRequest]) -> List[JsonRpcResponse]:
        """Handle a batch of requests."""
        tasks = []
        for request in requests:
            if isinstance(request, JsonRpcError):
                # Invalid request in batch
                tasks.append(asyncio.create_task(
                    asyncio.coroutine(lambda: JsonRpcResponse(
                        id=None,
                        error=request.to_dict()
                    ))()
                ))
            else:
                tasks.append(asyncio.create_task(
                    self.handler.handle_request(request)
                ))
        
        responses = await asyncio.gather(*tasks)
        # Filter out None (notifications)
        return [r for r in responses if r is not None]


def create_error_response(error_code: JsonRpcErrorCode, message: str, 
                         data: Optional[Any] = None, 
                         request_id: Optional[Union[str, int]] = None) -> JsonRpcResponse:
    """Create a standard error response."""
    return JsonRpcResponse(
        id=request_id,
        error=JsonRpcError(
            code=error_code,
            message=message,
            data=data
        ).to_dict()
    )