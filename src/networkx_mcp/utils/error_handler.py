"""Error handling utilities."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class MCPError(Exception):
    """Base exception for MCP errors."""
    pass

class ValidationError(MCPError):
    """Validation error."""
    pass

class GraphOperationError(MCPError):
    """Graph operation error."""
    pass

class ResourceError(MCPError):
    """Resource limit error."""
    pass

def handle_error(error: Exception, context: Optional[str] = None) -> None:
    """Log error with context."""
    if context:
        logger.error(f"{context}: {error}")
    else:
        logger.error(str(error))
