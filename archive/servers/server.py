"""NetworkX MCP Server v2 - Modular implementation with Resources and Prompts."""

import logging

# Use compatibility layer instead of direct FastMCP import
from networkx_mcp.compat.fastmcp_compat import FastMCPCompat as FastMCP
from networkx_mcp.core.graph_operations import GraphManager
from networkx_mcp.mcp.handlers import (AlgorithmHandler, AnalysisHandler,
                                       GraphOpsHandler, VisualizationHandler)
from networkx_mcp.mcp.prompts import GraphPrompts
from networkx_mcp.mcp.resources import GraphResources

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NetworkXMCPServer:
    """Enhanced NetworkX MCP Server with Resources and Prompts."""

    def __init__(self, name: str = "NetworkX Graph Analysis Server v2"):
        """Initialize the enhanced MCP server."""
        self.mcp = FastMCP(name)
        self.graph_manager = GraphManager()

        # Initialize resources and prompts
        self.resources = GraphResources(self.mcp, self.graph_manager)
        self.prompts = GraphPrompts(self.mcp)

        # Initialize handlers
        self.graph_ops_handler = GraphOpsHandler(self.mcp, self.graph_manager)
        self.algorithm_handler = AlgorithmHandler(self.mcp, self.graph_manager)
        self.analysis_handler = AnalysisHandler(self.mcp, self.graph_manager)
        self.visualization_handler = VisualizationHandler(self.mcp, self.graph_manager)

        # Register remaining core tools
        self._register_core_tools()

        logger.info(f"Initialized {name}")
        logger.info("MCP Features: Tools ✓, Resources ✓, Prompts ✓")

    def _register_core_tools(self):
        """Register remaining core tools not handled by handlers."""
        # All tools have been migrated to handlers
        pass

    async def run(self):
        """Run the MCP server."""
        logger.info("Starting NetworkX MCP Server v2...")
        logger.info("Features: Tools, Resources, Prompts")
        logger.info(f"Tools: {len(self.mcp._tools)}")
        logger.info(f"Resources: {len(self.mcp._resources)}")
        logger.info(f"Prompts: {len(self.mcp._prompts)}")

        await self.mcp.run()


async def main():
    """Main entry point."""
    server = NetworkXMCPServer()
    await server.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
