#!/usr/bin/env python3
"""Run NetworkX MCP Server with security patches applied."""

import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🚀 Starting Secure NetworkX MCP Server...")

# Apply security patches FIRST
import security_patches
security_patches.apply_critical_patches()

print("✅ Security patches applied successfully!")

# Now start the server
from src.networkx_mcp.server import main

if __name__ == "__main__":
    try:
        print("🌐 Starting MCP server...")
        main()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n💥 Server error: {e}")
        sys.exit(1)