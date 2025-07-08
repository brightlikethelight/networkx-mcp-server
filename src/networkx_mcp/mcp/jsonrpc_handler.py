"""JSON-RPC 2.0 message handler for MCP protocol."""

import json
import logging
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# JSON-RPC Error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# MCP-specific error codes
TOOL_NOT_FOUND = -32001
RESOURCE_NOT_FOUND = -32002
PROMPT_NOT_FOUND = -32003
VALIDATION_ERROR = -32004


@dataclass
class JsonRpcError:
    """JSON-RPC error object."""
    code: int
    message: str
    data: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {"code": self.code, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result


@dataclass
class JsonRpcRequest:
    """JSON-RPC request object."""
    jsonrpc: str
    method: str
    params: Optional[Union[Dict[str, Any], List[Any]]] = None
    id: Optional[Union[str, int]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JsonRpcRequest":
        """Create from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc"),  # Don't default - it's required
            method=data.get("method", ""),
            params=data.get("params"),
            id=data.get("id")
        )
    
    def validate(self) -> Optional[JsonRpcError]:
        """Validate request format."""
        if self.jsonrpc is None:
            return JsonRpcError(INVALID_REQUEST, "Missing jsonrpc field")
            
        if self.jsonrpc != "2.0":
            return JsonRpcError(INVALID_REQUEST, "Invalid JSON-RPC version")
        
        if not self.method:
            return JsonRpcError(INVALID_REQUEST, "Missing method")
        
        if self.params is not None and not isinstance(self.params, (dict, list)):
            return JsonRpcError(INVALID_REQUEST, "Invalid params type")
        
        return None


@dataclass
class JsonRpcResponse:
    """JSON-RPC response object."""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[JsonRpcError] = None
    id: Optional[Union[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        response = {"jsonrpc": self.jsonrpc}
        
        if self.error is not None:
            response["error"] = self.error.to_dict()
        else:
            response["result"] = self.result
            
        if self.id is not None:
            response["id"] = self.id
            
        return response
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class JsonRpcHandler:
    """Handles JSON-RPC message processing for MCP."""
    
    def __init__(self, server):
        """Initialize with MCP server instance."""
        self.server = server
        
    def parse_message(self, message: str) -> Union[JsonRpcRequest, JsonRpcResponse]:
        """Parse incoming JSON-RPC message."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            return JsonRpcResponse(
                error=JsonRpcError(PARSE_ERROR, f"Parse error: {str(e)}")
            )
        
        if not isinstance(data, dict):
            return JsonRpcResponse(
                error=JsonRpcError(INVALID_REQUEST, "Request must be an object")
            )
        
        # Handle batch requests (array of requests)
        if isinstance(data, list):
            # MCP doesn't currently support batch requests
            return JsonRpcResponse(
                error=JsonRpcError(INVALID_REQUEST, "Batch requests not supported")
            )
        
        request = JsonRpcRequest.from_dict(data)
        error = request.validate()
        
        if error:
            return JsonRpcResponse(error=error, id=request.id)
            
        return request
    
    def handle_request(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Route request to appropriate handler."""
        try:
            # MCP protocol methods
            if request.method == "initialize":
                result = self._handle_initialize(request.params or {})
            elif request.method == "initialized":
                result = self._handle_initialized(request.params or {})
            elif request.method == "tools/list":
                result = self._handle_tools_list(request.params or {})
            elif request.method == "tools/call":
                result = self._handle_tool_call(request.params or {})
            elif request.method == "resources/list":
                result = self._handle_resources_list(request.params or {})
            elif request.method == "resources/read":
                result = self._handle_resource_read(request.params or {})
            elif request.method == "prompts/list":
                result = self._handle_prompts_list(request.params or {})
            elif request.method == "prompts/get":
                result = self._handle_prompt_get(request.params or {})
            else:
                return JsonRpcResponse(
                    error=JsonRpcError(METHOD_NOT_FOUND, f"Unknown method: {request.method}"),
                    id=request.id
                )
                
            return JsonRpcResponse(result=result, id=request.id)
            
        except ValueError as e:
            # ValueError typically indicates invalid parameters or arguments
            error_msg = str(e)
            if ("Invalid arguments:" in error_msg or 
                "Missing required parameter:" in error_msg or
                "Missing tool name" in error_msg):
                logger.error(f"Invalid parameters: {e}")
                return JsonRpcResponse(
                    error=JsonRpcError(INVALID_PARAMS, error_msg),
                    id=request.id
                )
            else:
                logger.error(f"Value error: {e}")
                return JsonRpcResponse(
                    error=JsonRpcError(INTERNAL_ERROR, error_msg),
                    id=request.id
                )
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return JsonRpcResponse(
                error=JsonRpcError(INTERNAL_ERROR, str(e)),
                id=request.id
            )
    
    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        return {
            "protocolVersion": "1.0",
            "capabilities": {
                "tools": {
                    "listChanged": True
                },
                "resources": {
                    "subscribe": False,
                    "listChanged": True
                },
                "prompts": {
                    "listChanged": True
                }
            },
            "serverInfo": {
                "name": self.server.name,
                "version": self.server.version
            }
        }
    
    def _handle_initialized(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialized notification."""
        logger.info("MCP client initialized")
        return {}
    
    def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = []
        
        for tool_name, tool in self.server.tools.items():
            tools.append({
                "name": tool_name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                "outputSchema": tool.output_schema
            })
            
        return {"tools": tools}
    
    def _handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not tool_name:
            raise ValueError("Missing tool name")
            
        if tool_name not in self.server.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
            
        tool = self.server.tools[tool_name]
        
        # Validate arguments
        try:
            validated_args = tool.validate_params(arguments)
        except Exception as e:
            raise ValueError(f"Invalid arguments: {str(e)}")
        
        # Execute tool
        result = tool.func(**validated_args)
        
        # Ensure result is JSON-serializable
        if not isinstance(result, dict):
            result = {"result": result}
            
        return result
    
    def _handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request."""
        resources = []
        
        for uri, resource in self.server.resources.items():
            resources.append({
                "uri": uri,
                "name": resource.get("name", uri),
                "description": resource.get("description", ""),
                "mimeType": resource.get("mimeType", "application/json")
            })
            
        return {"resources": resources}
    
    def _handle_resource_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")
        
        if not uri:
            raise ValueError("Missing resource URI")
            
        if uri not in self.server.resources:
            raise ValueError(f"Unknown resource: {uri}")
            
        # Execute resource handler
        resource_func = self.server.resources[uri]["handler"]
        content = resource_func()
        
        return {
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(content)
            }]
        }
    
    def _handle_prompts_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/list request."""
        prompts = []
        
        for prompt_name, prompt in self.server.prompts.items():
            prompts.append({
                "name": prompt_name,
                "description": prompt.get("description", ""),
                "arguments": prompt.get("arguments", [])
            })
            
        return {"prompts": prompts}
    
    def _handle_prompt_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/get request."""
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not prompt_name:
            raise ValueError("Missing prompt name")
            
        if prompt_name not in self.server.prompts:
            raise ValueError(f"Unknown prompt: {prompt_name}")
            
        # Execute prompt handler
        prompt_func = self.server.prompts[prompt_name]["handler"]
        messages = prompt_func(**arguments)
        
        return {"messages": messages}


def format_notification(method: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Format a JSON-RPC notification (no id)."""
    notification = {
        "jsonrpc": "2.0",
        "method": method
    }
    
    if params:
        notification["params"] = params
        
    return json.dumps(notification)