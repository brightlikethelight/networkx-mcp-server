"""Secure version of NetworkX MCP Server with integrated security fixes."""

import sys

from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Apply security fixes BEFORE importing the server
from apply_security_fixes import apply_security_fixes


security_patcher = apply_security_fixes()

# Now import and run the server
from networkx_mcp.server import main


if __name__ == "__main__":
    main()
