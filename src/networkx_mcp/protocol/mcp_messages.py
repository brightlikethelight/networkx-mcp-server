"""MCP protocol message definitions and handlers.

This module defines MCP-specific messages and protocol handling as per
the Model Context Protocol specification.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# MCP Protocol version
PROTOCOL_VERSION = "2024-11-05"  # Latest MCP version


@dataclass
class ServerInfo:
    """MCP server information."""
    name: str
    version: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ClientInfo:
    """MCP client information."""
    name: str
    version: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ClientInfo":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "0.0.0")
        )


@dataclass 
class Implementation:
    """MCP implementation details."""
    name: str
    version: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ToolInfo:
    """MCP tool information."""
    name: str
    description: str
    inputSchema: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ResourceInfo:
    """MCP resource information."""
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {"uri": self.uri, "name": self.name}
        if self.description:
            data["description"] = self.description
        if self.mimeType:
            data["mimeType"] = self.mimeType
        return data


@dataclass
class PromptInfo:
    """MCP prompt information."""
    name: str
    description: Optional[str] = None
    arguments: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {"name": self.name}
        if self.description:
            data["description"] = self.description
        if self.arguments:
            data["arguments"] = self.arguments
        return data


class MCPMessages:
    """MCP protocol message builders."""
    
    @staticmethod
    def initialize_response(
        server_info: ServerInfo,
        capabilities: Dict[str, Any],
        instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create initialize response."""
        response = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": capabilities,
            "serverInfo": server_info.to_dict()
        }
        
        if instructions:
            response["instructions"] = instructions
            
        return response
    
    @staticmethod
    def tools_list_response(tools: List[ToolInfo]) -> Dict[str, Any]:
        """Create tools/list response."""
        return {
            "tools": [tool.to_dict() for tool in tools]
        }
    
    @staticmethod
    def resources_list_response(resources: List[ResourceInfo]) -> Dict[str, Any]:
        """Create resources/list response."""
        return {
            "resources": [resource.to_dict() for resource in resources]
        }
    
    @staticmethod
    def prompts_list_response(prompts: List[PromptInfo]) -> Dict[str, Any]:
        """Create prompts/list response."""
        return {
            "prompts": [prompt.to_dict() for prompt in prompts]
        }
    
    @staticmethod
    def tool_call_response(
        content: List[Dict[str, Any]],
        isError: bool = False
    ) -> Dict[str, Any]:
        """Create tool call response."""
        return {
            "content": content,
            "isError": isError
        }
    
    @staticmethod
    def create_text_content(text: str) -> Dict[str, str]:
        """Create text content for responses."""
        return {
            "type": "text",
            "text": text
        }
    
    @staticmethod
    def create_image_content(data: str, mimeType: str) -> Dict[str, str]:
        """Create image content for responses."""
        return {
            "type": "image",
            "data": data,
            "mimeType": mimeType
        }
    
    @staticmethod
    def create_resource_content(uri: str, text: Optional[str] = None) -> Dict[str, str]:
        """Create resource content for responses."""
        content = {
            "type": "resource",
            "resource": {"uri": uri}
        }
        if text:
            content["resource"]["text"] = text
        return content
    
    @staticmethod
    def capabilities() -> Dict[str, Any]:
        """Get server capabilities."""
        return {
            "tools": {
                "listChanged": True  # We support dynamic tool updates
            },
            "resources": {
                "subscribe": False,  # No subscription support yet
                "listChanged": False
            },
            "prompts": {
                "listChanged": False
            },
            "logging": {}  # Basic logging support
        }
    
    @staticmethod
    def parse_initialize_request(params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse initialize request parameters."""
        return {
            "protocolVersion": params.get("protocolVersion", "unknown"),
            "capabilities": params.get("capabilities", {}),
            "clientInfo": ClientInfo.from_dict(params.get("clientInfo", {}))
        }
    
    @staticmethod
    def parse_tool_call_request(params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse tool call request parameters."""
        return {
            "name": params.get("name"),
            "arguments": params.get("arguments", {})
        }
    
    @staticmethod
    def parse_resource_read_request(params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse resource read request parameters."""
        return {
            "uri": params.get("uri")
        }
    
    @staticmethod
    def parse_prompt_get_request(params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse prompt get request parameters."""
        return {
            "name": params.get("name"),
            "arguments": params.get("arguments", {})
        }
    
    @staticmethod
    def create_error_content(error_message: str) -> List[Dict[str, str]]:
        """Create error content for tool responses."""
        return [MCPMessages.create_text_content(f"Error: {error_message}")]
    
    @staticmethod
    def create_success_content(result: Any) -> List[Dict[str, str]]:
        """Create success content for tool responses."""
        import json
        
        # Convert result to JSON string for consistent format
        if isinstance(result, str):
            text = result
        else:
            text = json.dumps(result, indent=2)
            
        return [MCPMessages.create_text_content(text)]


class MCPErrorMessages:
    """Standard MCP error messages."""
    
    INVALID_PROTOCOL_VERSION = "Unsupported protocol version"
    TOOL_NOT_FOUND = "Tool not found"
    RESOURCE_NOT_FOUND = "Resource not found"
    PROMPT_NOT_FOUND = "Prompt not found"
    INVALID_ARGUMENTS = "Invalid arguments"
    INTERNAL_ERROR = "Internal server error"
    UNAUTHORIZED = "Unauthorized"
    RESOURCE_UNAVAILABLE = "Resource temporarily unavailable"
    RATE_LIMITED = "Rate limit exceeded"