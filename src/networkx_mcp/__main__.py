"""NetworkX MCP Server entry point - stdio transport only."""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_server():
    """Run the NetworkX MCP server."""
    logger.info("Starting NetworkX MCP Server")
    from .server import main

    main()


def main():
    """Main entry point for NetworkX MCP Server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NetworkX MCP Server - Graph operations via Model Context Protocol"
    )

    # Logging
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        logger.info("Starting NetworkX MCP Server via stdio transport")
        run_server()

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
