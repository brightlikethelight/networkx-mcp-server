"""Enhanced stdio transport for MCP server communication.

This module implements a robust stdio transport layer with proper edge case handling,
binary mode support, and concurrent message safety.
"""

import sys
import asyncio
import json
import logging
from typing import AsyncIterator, Optional, Dict, Any
from asyncio import Lock

logger = logging.getLogger(__name__)


class StdioTransport:
    """Enhanced stdio transport with robust error handling."""
    
    def __init__(self, json_handler):
        """Initialize the transport with a JSON-RPC handler."""
        self.json_handler = json_handler
        self.reader = None
        self.writer = None
        self._running = False
        self._write_lock = Lock()  # Prevent output interleaving
        self._read_buffer = bytearray()  # Buffer for partial reads
        
    async def start(self):
        """Start reading from stdin and writing to stdout."""
        self._running = True
        
        # Critical: Set stdout to binary mode to prevent encoding issues
        self.writer = sys.stdout.buffer
        
        # Create stdin reader with proper binary handling
        loop = asyncio.get_event_loop()
        self.reader = asyncio.StreamReader()
        
        # Connect stdin to async reader
        await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(self.reader),
            sys.stdin.buffer  # Use binary mode
        )
        
        logger.info("Stdio transport started in binary mode")
        
    async def read_messages(self) -> AsyncIterator[str]:
        """Read newline-delimited JSON from stdin with robust handling."""
        while self._running:
            try:
                # Read until newline or EOF
                line = await self.reader.readline()
                
                if not line:
                    # EOF reached
                    if self._read_buffer:
                        # Process any remaining data
                        yield self._read_buffer.decode('utf-8', errors='replace').strip()
                        self._read_buffer.clear()
                    break
                
                # Add to buffer
                self._read_buffer.extend(line)
                
                # Check if we have a complete line
                if line.endswith(b'\n'):
                    # Extract complete message
                    message = self._read_buffer[:-1].decode('utf-8', errors='replace').strip()
                    self._read_buffer.clear()
                    
                    if message:
                        yield message
                        
            except asyncio.CancelledError:
                logger.info("Read cancelled")
                break
            except UnicodeDecodeError as e:
                # Log to stderr, never stdout
                logger.error(f"Unicode decode error: {e}", exc_info=True)
                self._read_buffer.clear()  # Clear corrupted buffer
            except Exception as e:
                logger.error(f"Read error: {e}", exc_info=True)
                
    async def write_message(self, message: dict):
        """Write JSON message to stdout with newline, using lock for safety."""
        async with self._write_lock:
            try:
                # Compact JSON with no extra spaces
                json_str = json.dumps(message, separators=(',', ':'), ensure_ascii=True)
                
                # Write as bytes with newline
                data = f"{json_str}\n".encode('utf-8')
                self.writer.write(data)
                self.writer.flush()
                
                logger.debug(f"Sent message: {json_str[:100]}...")
                
            except BrokenPipeError:
                logger.error("Broken pipe - client disconnected")
                self._running = False
            except Exception as e:
                logger.error(f"Write error: {e}", exc_info=True)
                
    async def run(self):
        """Main transport loop handling messages."""
        await self.start()
        
        try:
            async for message in self.read_messages():
                asyncio.create_task(self._handle_message(message))
                
        except Exception as e:
            logger.error(f"Transport loop error: {e}", exc_info=True)
        finally:
            await self.stop()
            
    async def _handle_message(self, message: str):
        """Handle a single message asynchronously."""
        try:
            # Process with JSON-RPC handler
            response = await self.json_handler.handle_message(message)
            
            if response:
                # Parse response to dict if it's a string
                if isinstance(response, str):
                    response_dict = json.loads(response)
                else:
                    response_dict = response
                    
                await self.write_message(response_dict)
                
        except json.JSONDecodeError as e:
            # Send parse error
            await self.write_message({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": "Parse error",
                    "data": str(e)
                }
            })
        except Exception as e:
            logger.error(f"Message handling error: {e}", exc_info=True)
            # Send internal error
            await self.write_message({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            })
            
    async def stop(self):
        """Stop the transport cleanly."""
        logger.info("Stopping stdio transport")
        self._running = False
        
        # Flush any pending output
        if self.writer:
            try:
                self.writer.flush()
            except Exception:
                pass


class EnhancedStdioServer:
    """MCP server with enhanced stdio transport."""
    
    def __init__(self, json_handler):
        """Initialize with JSON-RPC handler."""
        self.transport = StdioTransport(json_handler)
        
    async def run(self):
        """Run the server."""
        logger.info("Starting enhanced stdio MCP server")
        
        try:
            await self.transport.run()
        except KeyboardInterrupt:
            logger.info("Server interrupted")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise
            
        logger.info("Server stopped")


def configure_stdio_mode():
    """Configure stdin/stdout for proper binary operation."""
    # Disable buffering for real-time communication
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
            
    if hasattr(sys.stdin, 'reconfigure'):
        try:
            sys.stdin.reconfigure(line_buffering=True)
        except Exception:
            pass
            
    # Ensure stderr is used for all logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr,
        force=True
    )
    
    # Prevent any accidental stdout writes from libraries
    import warnings
    warnings.filterwarnings("ignore")
    
    logger.info("Stdio mode configured for MCP transport")


def run_stdio_server(json_handler):
    """Run the stdio MCP server with the given handler."""
    configure_stdio_mode()
    
    server = EnhancedStdioServer(json_handler)
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        sys.exit(1)