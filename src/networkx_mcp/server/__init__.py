"""NetworkX MCP Server - Modular architecture."""

from networkx_mcp.server.handlers import *
from networkx_mcp.server.prompts import *
from networkx_mcp.server.resources import *

__all__ = ["handlers", "resources", "prompts", "middleware"]