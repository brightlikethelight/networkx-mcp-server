"""NetworkX MCP Server entry point with dual-mode transport support."""

import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_minimal_server():
    """Run the minimal server (guaranteed to work)."""
    logger.info("Starting NetworkX MCP Server (Minimal Version)")
    from .server_minimal import main

    main()


def run_full_server():
    """Run the full server with all features."""
    logger.info("Starting NetworkX MCP Server (Full Version)")
    from .server import main

    main()


def run_jsonrpc_server():
    """Run the JSON-RPC protocol server for MCP clients."""
    logger.info("Starting NetworkX MCP Server (JSON-RPC Protocol)")
    from .server_jsonrpc import main
    
    main()


async def run_http_server(port: int = 3000, host: str = "0.0.0.0", enable_auth: bool = True):
    """Run the HTTP MCP server for remote access."""
    logger.info(f"Starting NetworkX MCP Server (HTTP Mode) on {host}:{port}")
    
    # Import components
    from .server_jsonrpc import NetworkXMCPHandler
    from .transport.http_transport import StreamableHTTPTransport
    from .auth.oauth import create_oauth_handler
    from .core.graceful_shutdown import initialize_shutdown_handler
    
    # Create JSON-RPC handler
    json_handler = NetworkXMCPHandler()
    
    # Create HTTP transport
    http_transport = StreamableHTTPTransport(json_handler, port=port, host=host)
    
    # Setup authentication if enabled
    if enable_auth:
        oauth_handler = create_oauth_handler()
        # Add auth middleware to transport
        http_transport.oauth_handler = oauth_handler
        logger.info("OAuth authentication enabled")
    
    # Initialize graceful shutdown
    shutdown_handler = initialize_shutdown_handler()
    
    try:
        # Start HTTP server
        runner = await http_transport.start()
        
        logger.info(f"HTTP MCP server running on http://{host}:{port}")
        logger.info("Available endpoints:")
        logger.info(f"  POST http://{host}:{port}/mcp/session - Create session")
        logger.info(f"  POST http://{host}:{port}/mcp - JSON-RPC requests")
        logger.info(f"  GET  http://{host}:{port}/mcp - SSE connection")
        logger.info(f"  GET  http://{host}:{port}/health - Health check")
        logger.info(f"  GET  http://{host}:{port}/info - Server info")
        
        # Wait for shutdown signal
        shutdown_event = asyncio.Event()
        
        def signal_handler():
            logger.info("Shutdown signal received")
            shutdown_event.set()
        
        # Setup signal handlers
        import signal
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, lambda s, f: signal_handler())
        
        await shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"HTTP server error: {e}")
        raise
    finally:
        # Cleanup
        try:
            await http_transport.stop(runner)
            await shutdown_handler.shutdown()
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


def main():
    """Main entry point with dual-mode transport selection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NetworkX MCP Server - Graph operations via Model Context Protocol"
    )
    
    # Server mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--minimal",
        action="store_true",
        help="Run minimal server (basic functionality, guaranteed to work)",
    )
    mode_group.add_argument(
        "--jsonrpc",
        action="store_true", 
        help="Run JSON-RPC protocol server via stdio (for local MCP clients)",
    )
    mode_group.add_argument(
        "--http",
        action="store_true",
        help="Run HTTP server for remote MCP access (with authentication)",
    )
    
    # Transport configuration
    parser.add_argument(
        "--port", 
        type=int, 
        default=3000,
        help="Port for HTTP server (default: 3000)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable authentication for HTTP mode (not recommended for production)"
    )
    
    # Logging
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.minimal:
            logger.info("Running minimal server as requested")
            run_minimal_server()
            
        elif args.http:
            logger.info("Running HTTP server for remote access")
            if args.no_auth:
                logger.warning("⚠️  Authentication disabled - not recommended for production!")
            
            # Run HTTP server
            asyncio.run(run_http_server(
                port=args.port,
                host=args.host,
                enable_auth=not args.no_auth
            ))
            
        elif args.jsonrpc:
            logger.info("Running JSON-RPC protocol server via stdio")
            run_jsonrpc_server()
            
        else:
            # Default to stdio JSON-RPC for MCP compatibility
            logger.info("Running JSON-RPC protocol server via stdio (default)")
            logger.info("Use --http for remote access or --help for options")
            run_jsonrpc_server()
            
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()