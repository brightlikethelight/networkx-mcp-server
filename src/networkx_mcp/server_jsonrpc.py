"""NetworkX MCP Server with JSON-RPC 2.0 protocol.

This is the main entry point for the MCP server with full protocol support.
"""

import argparse
import logging
import sys
from typing import Optional

from .protocol.stdio_transport import run_stdio_server

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure handlers
    handlers = []
    
    # Always log to stderr for stdio transport
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    handlers.append(stderr_handler)
    
    # Optionally log to file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers
    )


def main():
    """Main entry point for MCP server when called from __main__.py."""
    # When called from __main__.py, arguments are already parsed
    # Just set up logging with defaults and run the server
    setup_logging("INFO", None)
    
    logger.info("Starting NetworkX MCP Server with JSON-RPC protocol")
    
    # Run the server
    try:
        run_stdio_server()
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()