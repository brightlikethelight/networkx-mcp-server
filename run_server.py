#!/usr/bin/env python3
"""Run NetworkX MCP Server without FastMCP/Pydantic v2 dependencies."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Use the CLI for now until Pydantic dependencies are resolved
from networkx_mcp.cli import main

if __name__ == "__main__":
    print("Starting NetworkX MCP Server CLI...")
    print("Note: Full MCP server requires Pydantic v2. Using CLI mode.")
    main()