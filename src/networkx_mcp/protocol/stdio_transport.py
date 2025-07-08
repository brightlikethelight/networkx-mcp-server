"""Stdio transport for MCP server communication.

This module implements the stdio (standard input/output) transport layer
for MCP server communication, enabling clients to communicate via stdin/stdout.
"""

import asyncio
import json
import logging
import sys
from typing import Optional

from .mcp_handler import MCPProtocolHandler
from ..transport import StdioTransport as EnhancedStdioTransport, configure_stdio_mode

logger = logging.getLogger(__name__)


class StdioTransport:
    """Transport layer for stdio communication."""
    
    def __init__(self, handler: Optional[MCPProtocolHandler] = None):
        """Initialize stdio transport."""
        self.handler = handler or MCPProtocolHandler()
        self.reader = None
        self.writer = None
        self.running = False
    
    async def start(self):
        """Start the stdio transport server."""
        logger.info("Starting stdio transport server")
        
        # Set up async stdio
        loop = asyncio.get_event_loop()
        
        # Create reader for stdin
        self.reader = asyncio.StreamReader()
        reader_protocol = asyncio.StreamReaderProtocol(self.reader)
        
        await loop.connect_read_pipe(lambda: reader_protocol, sys.stdin)
        
        # Use stdout directly for writing
        self.writer = sys.stdout
        
        self.running = True
        
        # Start reading messages
        await self._read_loop()
    
    async def _read_loop(self):
        """Read messages from stdin."""
        while self.running:
            try:
                # Read line from stdin
                line = await self.reader.readline()
                
                if not line:
                    # EOF reached
                    logger.info("EOF reached, stopping server")
                    break
                
                # Decode and process message
                message = line.decode('utf-8').strip()
                
                if not message:
                    continue
                
                logger.debug(f"Received message: {message[:100]}...")
                
                # Handle the message
                response = await self.handler.handle_message(message)
                
                if response:
                    # Send response
                    await self._send_response(response)
                    
            except asyncio.CancelledError:
                logger.info("Read loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in read loop: {e}", exc_info=True)
                
                # Send error response
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": "Internal error",
                        "data": str(e)
                    }
                }
                await self._send_response(json.dumps(error_response))
    
    async def _send_response(self, response: str):
        """Send response to stdout."""
        try:
            # Write response followed by newline
            self.writer.write(response + '\n')
            self.writer.flush()
            
            logger.debug(f"Sent response: {response[:100]}...")
            
        except Exception as e:
            logger.error(f"Error sending response: {e}", exc_info=True)
    
    async def stop(self):
        """Stop the transport server."""
        logger.info("Stopping stdio transport server")
        self.running = False


class StdioMCPServer:
    """MCP server with stdio transport."""
    
    def __init__(self):
        """Initialize MCP server."""
        self.transport = StdioTransport()
        
    async def run(self):
        """Run the MCP server."""
        logger.info("NetworkX MCP Server starting...")
        
        try:
            await self.transport.start()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            await self.transport.stop()
            
        logger.info("NetworkX MCP Server stopped")


def run_stdio_server():
    """Run the stdio MCP server with enhanced transport."""
    # Configure stdio mode for proper binary handling
    configure_stdio_mode()
    
    # Create MCP protocol handler
    handler = MCPProtocolHandler()
    
    # Use enhanced transport for better edge case handling
    transport = EnhancedStdioTransport(handler)
    
    async def run_server():
        """Run the transport server."""
        logger.info("NetworkX MCP Server starting with enhanced stdio transport...")
        try:
            await transport.run()
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise
        finally:
            logger.info("NetworkX MCP Server stopped")
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_stdio_server()