#!/usr/bin/env python3
"""HTTP transport for remote MCP server access.

Implements streamable HTTP transport alongside stdio for remote MCP clients.
Includes session management, SSE support, and security features.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Set, Any
from aiohttp import web, ClientSession, hdrs
import aiohttp

from ..protocol.json_rpc import JsonRpcRequest, JsonRpcResponse
from ..logging import get_logger
from ..config.production import production_config

logger = get_logger(__name__)


@dataclass
class TransportSession:
    """Represents an active MCP session over HTTP."""
    session_id: str
    created_at: float
    last_activity: float
    client_info: Optional[Dict[str, Any]] = None
    initialized: bool = False
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    @property
    def age_seconds(self) -> float:
        """Get session age in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        """Get seconds since last activity."""
        return time.time() - self.last_activity


class StreamableHTTPTransport:
    """HTTP transport for MCP with SSE support."""
    
    def __init__(self, json_handler, port: int = 3000, host: str = "0.0.0.0"):
        self.json_handler = json_handler
        self.port = port
        self.host = host
        self.app = web.Application(middlewares=[self.security_middleware])
        self.sessions: Dict[str, TransportSession] = {}
        self.sse_connections: Dict[str, web.StreamResponse] = {}
        
        # Configuration
        self.session_timeout = 3600  # 1 hour
        self.heartbeat_interval = 30  # 30 seconds
        self.max_sessions = production_config.MAX_CONCURRENT_CONNECTIONS
        
        # Security settings
        self.allowed_origins: Set[str] = {
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://localhost:3000",
            "https://127.0.0.1:3000"
        }
        
        self.setup_routes()
        logger.info(f"HTTP transport configured on {host}:{port}")
    
    def setup_routes(self):
        """Setup HTTP routes for MCP transport."""
        # Main MCP endpoints
        self.app.router.add_post('/mcp', self.handle_post)
        self.app.router.add_get('/mcp', self.handle_sse)
        
        # Session management
        self.app.router.add_post('/mcp/session', self.create_session)
        self.app.router.add_delete('/mcp/session/{session_id}', self.delete_session)
        
        # Health and info endpoints
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/info', self.server_info)
        
        # CORS preflight
        self.app.router.add_options('/mcp', self.handle_options)
        self.app.router.add_options('/mcp/session', self.handle_options)
        
        logger.info("HTTP routes configured")
    
    @web.middleware
    async def security_middleware(self, request, handler):
        """Security middleware for CORS and DNS rebinding protection."""
        # Handle CORS preflight
        if request.method == 'OPTIONS':
            return await self.handle_options(request)
        
        # Check Origin header for DNS rebinding protection
        origin = request.headers.get('Origin')
        if origin and not self.is_allowed_origin(origin):
            logger.warning(f"Blocked request from disallowed origin: {origin}")
            return web.Response(
                text="Forbidden: Origin not allowed",
                status=403
            )
        
        # Process request
        try:
            response = await handler(request)
        except Exception as e:
            logger.error(f"Request handler error: {e}", exc_info=True)
            return web.json_response(
                {"error": "Internal server error"},
                status=500
            )
        
        # Add CORS headers
        self.add_cors_headers(response, origin)
        
        return response
    
    def is_allowed_origin(self, origin: str) -> bool:
        """Check if origin is allowed."""
        # In development, allow any localhost/127.0.0.1
        if not production_config.is_production:
            if any(host in origin for host in ['localhost', '127.0.0.1']):
                return True
        
        return origin in self.allowed_origins
    
    def add_cors_headers(self, response: web.Response, origin: Optional[str]):
        """Add CORS headers to response."""
        if origin and self.is_allowed_origin(origin):
            response.headers['Access-Control-Allow-Origin'] = origin
        else:
            response.headers['Access-Control-Allow-Origin'] = 'null'
            
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, DELETE'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Session-ID'
        response.headers['Access-Control-Max-Age'] = '86400'  # 24 hours
    
    async def handle_options(self, request):
        """Handle CORS preflight requests."""
        response = web.Response()
        self.add_cors_headers(response, request.headers.get('Origin'))
        return response
    
    async def create_session(self, request):
        """Create a new MCP session."""
        try:
            # Check session limit
            if len(self.sessions) >= self.max_sessions:
                return web.json_response(
                    {"error": "Maximum sessions exceeded"},
                    status=503
                )
            
            # Create new session
            session_id = str(uuid.uuid4())
            session = TransportSession(
                session_id=session_id,
                created_at=time.time(),
                last_activity=time.time()
            )
            
            self.sessions[session_id] = session
            
            logger.info(f"Created session {session_id}")
            
            return web.json_response({
                "session_id": session_id,
                "expires_in": self.session_timeout,
                "endpoints": {
                    "jsonrpc": "/mcp",
                    "sse": "/mcp"
                }
            })
            
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            return web.json_response(
                {"error": "Failed to create session"},
                status=500
            )
    
    async def delete_session(self, request):
        """Delete an MCP session."""
        session_id = request.match_info['session_id']
        
        if session_id in self.sessions:
            # Close SSE connection if exists
            if session_id in self.sse_connections:
                try:
                    await self.sse_connections[session_id].write_eof()
                except:
                    pass
                del self.sse_connections[session_id]
            
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            
            return web.json_response({"status": "deleted"})
        else:
            return web.json_response(
                {"error": "Session not found"},
                status=404
            )
    
    async def handle_post(self, request):
        """Handle JSON-RPC requests over HTTP."""
        session_id = request.headers.get('X-Session-ID')
        
        if not session_id or session_id not in self.sessions:
            return web.json_response(
                {"error": "Invalid or missing session ID"},
                status=400
            )
        
        session = self.sessions[session_id]
        session.update_activity()
        
        try:
            data = await request.json()
            
            # Handle single or batch requests
            if isinstance(data, list):
                responses = []
                for req_data in data:
                    try:
                        req = JsonRpcRequest(**req_data)
                        resp = await self.json_handler.handle_request(req)
                        if resp:  # Don't include responses for notifications
                            responses.append(asdict(resp))
                    except Exception as e:
                        logger.error(f"Batch request error: {e}")
                        responses.append({
                            "jsonrpc": "2.0",
                            "id": req_data.get("id"),
                            "error": {
                                "code": -32603,
                                "message": "Internal error",
                                "data": str(e)
                            }
                        })
                
                return web.json_response(responses)
            else:
                req = JsonRpcRequest(**data)
                resp = await self.json_handler.handle_request(req)
                
                if resp:
                    return web.json_response(asdict(resp))
                else:
                    # Notification - no response
                    return web.Response(status=204)
                    
        except json.JSONDecodeError:
            return web.json_response({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": "Parse error"
                }
            }, status=400)
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            return web.json_response({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            }, status=500)
    
    async def handle_sse(self, request):
        """Handle Server-Sent Events connection."""
        session_id = request.headers.get('X-Session-ID')
        
        if not session_id or session_id not in self.sessions:
            return web.Response(
                text="Invalid or missing session ID",
                status=400
            )
        
        session = self.sessions[session_id]
        session.update_activity()
        
        # Setup SSE response
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        response.headers['X-Accel-Buffering'] = 'no'  # Disable Nginx buffering
        
        self.add_cors_headers(response, request.headers.get('Origin'))
        
        await response.prepare(request)
        
        # Store connection
        self.sse_connections[session_id] = response
        
        logger.info(f"SSE connection established for session {session_id}")
        
        try:
            # Send initial connection event
            await self.send_sse_event(response, 'connected', {
                'session_id': session_id,
                'timestamp': time.time()
            })
            
            # Heartbeat loop
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Check if session still exists
                if session_id not in self.sessions:
                    break
                
                # Send heartbeat
                await self.send_sse_event(response, 'heartbeat', {
                    'timestamp': time.time()
                })
                
        except Exception as e:
            logger.info(f"SSE connection closed for session {session_id}: {e}")
        finally:
            # Cleanup
            if session_id in self.sse_connections:
                del self.sse_connections[session_id]
            
            try:
                await response.write_eof()
            except:
                pass
        
        return response
    
    async def send_sse_event(self, response: web.StreamResponse, event_type: str, data: Any):
        """Send an SSE event."""
        try:
            event_data = json.dumps(data)
            await response.write(f"event: {event_type}\n".encode())
            await response.write(f"data: {event_data}\n\n".encode())
            await response.drain()
        except Exception as e:
            logger.warning(f"Failed to send SSE event: {e}")
            raise
    
    async def health_check(self, request):
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "transport": "http",
            "sessions": len(self.sessions),
            "sse_connections": len(self.sse_connections),
            "timestamp": time.time()
        })
    
    async def server_info(self, request):
        """Server information endpoint."""
        return web.json_response({
            "name": "NetworkX MCP Server",
            "version": production_config.SERVER_VERSION,
            "transport": "http",
            "protocol_version": production_config.PROTOCOL_VERSION,
            "capabilities": {
                "tools": True,
                "resources": False,
                "prompts": False
            },
            "limits": {
                "max_sessions": self.max_sessions,
                "session_timeout": self.session_timeout,
                "max_graph_nodes": production_config.MAX_GRAPH_SIZE_NODES
            }
        })
    
    async def cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions."""
        while True:
            try:
                current_time = time.time()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if current_time - session.last_activity > self.session_timeout:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    logger.info(f"Cleaning up expired session {session_id}")
                    
                    # Close SSE connection
                    if session_id in self.sse_connections:
                        try:
                            await self.sse_connections[session_id].write_eof()
                        except:
                            pass
                        del self.sse_connections[session_id]
                    
                    del self.sessions[session_id]
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def start(self):
        """Start the HTTP transport."""
        # Start cleanup task
        asyncio.create_task(self.cleanup_expired_sessions())
        
        # Setup and start HTTP server
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"HTTP MCP transport started on {self.host}:{self.port}")
        return runner
    
    async def stop(self, runner):
        """Stop the HTTP transport."""
        await runner.cleanup()
        logger.info("HTTP MCP transport stopped")