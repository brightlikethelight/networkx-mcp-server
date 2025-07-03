"""NetworkX MCP Server v2 with Modular Architecture.

This is the next-generation server implementation that uses the new
service-oriented architecture with dependency injection, event system,
and comprehensive validation.
"""

import asyncio
import logging
import signal
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.logging import get_logger

from .core.config import Environment, load_config
from .core.service_config import ServiceManager
from .mcp.handlers.algorithms_v2 import AlgorithmHandler
from .mcp.handlers.graph_ops_v2 import GraphOpsHandler

# Setup logging
logger = get_logger(__name__)


class NetworkXMCPServerV2:
    """NetworkX MCP Server with modern service architecture."""

    def __init__(
        self, config_file: str | None = None, environment: Environment | None = None
    ):
        """Initialize the server.

        Args:
            config_file: Optional path to configuration file
            environment: Optional environment override
        """
        # Load configuration
        self.config = load_config(config_file, environment=environment)

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.mcp = FastMCP("NetworkX MCP Server v2")
        self.service_manager = ServiceManager()
        self.container = None

        # Handlers
        self.graph_ops_handler: GraphOpsHandler | None = None
        self.algorithm_handler: AlgorithmHandler | None = None

        # Server state
        self._running = False
        self._shutdown_event = asyncio.Event()

        logger.info(
            f"NetworkX MCP Server v2 initialized for {self.config.environment.value} environment"
        )

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Configure file logging if specified
        if self.config.logging.file:
            file_handler = logging.FileHandler(self.config.logging.file)
            file_handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)

            # Add to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)

        logger.info("Logging configured")

    async def initialize(self) -> None:
        """Initialize the server and all services."""
        try:
            logger.info("Initializing NetworkX MCP Server v2...")

            # Start service manager
            self.container = await self.service_manager.start()

            # Initialize and register handlers
            await self._setup_handlers()

            # Register server-level tools
            self._register_server_tools()

            logger.info("Server initialization completed successfully")
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            await self.cleanup()
            raise

    async def _setup_handlers(self) -> None:
        """Setup and initialize MCP handlers."""
        try:
            # Create handlers
            self.graph_ops_handler = GraphOpsHandler(self.mcp, self.container)
            self.algorithm_handler = AlgorithmHandler(self.mcp, self.container)

            # Initialize handlers
            await self.graph_ops_handler.initialize()
            await self.algorithm_handler.initialize()

            logger.info("MCP handlers initialized successfully")
        except Exception as e:
            logger.error(f"Failed to setup handlers: {e}")
            raise

    def _register_server_tools(self) -> None:
        """Register server-level management tools."""

        @self.mcp.tool()
        async def server_health() -> dict:
            """Check server health status.

            Returns:
                Dict with health status of all components
            """
            try:
                health = await self.service_manager.health_check()

                # Add handler health if available
                if self.graph_ops_handler:
                    health["handlers"] = {
                        "graph_ops": await self.graph_ops_handler.health_check(),
                        "algorithms": await self.algorithm_handler.health_check(),
                    }

                health["server_running"] = self._running
                health["configuration"] = {
                    "environment": self.config.environment.value,
                    "debug": self.config.debug,
                    "version": self.config.version,
                }

                return health
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return {"healthy": False, "error": str(e)}

        @self.mcp.tool()
        async def server_info() -> dict:
            """Get server information.

            Returns:
                Dict with server details
            """
            return {
                "name": self.config.name,
                "version": self.config.version,
                "environment": self.config.environment.value,
                "features": {
                    "machine_learning": self.config.features.machine_learning,
                    "visualization": self.config.features.visualization,
                    "gpu_acceleration": self.config.features.gpu_acceleration,
                    "enterprise_features": self.config.features.enterprise_features,
                    "monitoring": self.config.features.monitoring,
                },
                "performance": {
                    "max_nodes": self.config.performance.max_nodes,
                    "max_edges": self.config.performance.max_edges,
                    "memory_limit_mb": self.config.performance.memory_limit_mb,
                    "enable_caching": self.config.performance.enable_caching,
                    "parallel_processing": self.config.performance.parallel_processing,
                },
            }

        @self.mcp.tool()
        async def server_stats() -> dict:
            """Get server runtime statistics.

            Returns:
                Dict with runtime statistics
            """
            try:
                from ..events.graph_events import MetricsEventListener

                # Get metrics from event listener
                metrics_listener = await self.container.resolve(MetricsEventListener)
                metrics = metrics_listener.get_metrics()

                return {
                    "uptime_status": "running" if self._running else "stopped",
                    "metrics": metrics,
                }
            except Exception as e:
                logger.error(f"Failed to get server stats: {e}")
                return {"error": str(e)}

        logger.debug("Server-level tools registered")

    async def run(self) -> None:
        """Run the server."""
        try:
            self._running = True
            logger.info("Starting NetworkX MCP Server v2...")

            # Setup signal handlers
            self._setup_signal_handlers()

            # Run the MCP server
            await self.mcp.run()

        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            await self.cleanup()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def cleanup(self) -> None:
        """Cleanup server resources."""
        try:
            logger.info("Cleaning up server resources...")

            self._running = False

            # Stop service manager
            if self.service_manager:
                await self.service_manager.stop()

            logger.info("Server cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


@asynccontextmanager
async def create_server(
    config_file: str | None = None, environment: Environment | None = None
):
    """Create and manage server lifecycle.

    Args:
        config_file: Optional path to configuration file
        environment: Optional environment override

    Yields:
        NetworkXMCPServerV2: Initialized server instance
    """
    server = NetworkXMCPServerV2(config_file, environment)
    try:
        await server.initialize()
        yield server
    finally:
        await server.cleanup()


async def main():
    """Main entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="NetworkX MCP Server v2")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--environment",
        type=str,
        choices=["development", "testing", "staging", "production"],
        help="Environment to run in",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Convert environment string to enum
    environment = None
    if args.environment:
        environment = Environment(args.environment)

    # Create and run server
    async with create_server(args.config, environment) as server:
        if args.debug:
            server.config.debug = True
            logging.getLogger().setLevel(logging.DEBUG)

        await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        raise
