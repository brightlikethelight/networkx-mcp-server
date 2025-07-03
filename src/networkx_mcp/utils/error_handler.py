"""Error handling utilities."""

import logging

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base exception for MCP errors."""


class ValidationError(MCPError):
    """Validation error."""


class GraphOperationError(MCPError):
    """Graph operation error."""


class ResourceError(MCPError):
    """Resource limit error."""


def handle_error(error: Exception, context: str | None = None) -> None:
    """Log error with context."""
    if context:
        logger.error(f"{context}: {error}")
    else:
        logger.error(str(error))
