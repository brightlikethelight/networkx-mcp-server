"""NetworkX MCP Server entry point with minimal/full server options."""

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


def main():
    """Main entry point with server selection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NetworkX MCP Server - Graph operations via Model Context Protocol"
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Run minimal server (basic functionality, guaranteed to work)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.minimal:
            logger.info("Running minimal server as requested")
            run_minimal_server()
        else:
            # Try full server first, fall back to minimal if it fails
            try:
                logger.info("Attempting to run full server...")
                run_full_server()
            except Exception as e:
                logger.warning(f"Full server failed to start: {e}")
                logger.info("Falling back to minimal server...")
                run_minimal_server()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
