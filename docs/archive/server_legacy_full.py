#!/usr/bin/env python3
"""
NetworkX MCP Server - Router Implementation

This file implements the Strangler Fig pattern for migrating from the
legacy complex implementation to the minimal working implementation.

Environment Variables:
- USE_MINIMAL_SERVER=true: Use the new minimal server (recommended)
- USE_MINIMAL_SERVER=false: Use the legacy server (deprecated)

Default: Uses minimal server (as of v0.1.3)
"""

import os
import sys
import warnings

# Determine which implementation to use
USE_MINIMAL = os.getenv("USE_MINIMAL_SERVER", "true").lower() in ("true", "1", "yes")

if USE_MINIMAL:
    # Use the minimal implementation (recommended)
    try:
        from .server_minimal import *

        print(
            "✅ Using minimal NetworkX MCP Server (150 lines, actually works)",
            file=sys.stderr,
        )
    except ImportError as e:
        print(f"❌ Failed to import minimal server: {e}", file=sys.stderr)
        print("🔄 Falling back to legacy server", file=sys.stderr)
        USE_MINIMAL = False

if not USE_MINIMAL:
    # Use the legacy implementation (deprecated)
    warnings.warn(
        "⚠️  Using legacy server implementation. "
        "This is deprecated and will be removed in v0.3.0. "
        "Set USE_MINIMAL_SERVER=true to use the working implementation.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from .server_legacy import *

        print(
            "⚠️  Using legacy NetworkX MCP Server (900+ lines, may not work)",
            file=sys.stderr,
        )
    except ImportError as e:
        print(
            "💥 Both servers failed to import. This is a configuration problem.",
            file=sys.stderr,
        )
        print("Minimal server error: see above", file=sys.stderr)
        print(f"Legacy server error: {e}", file=sys.stderr)
        sys.exit(1)

# Export version info
__version__ = "0.1.3-strangler-fig"
__implementation__ = "minimal" if USE_MINIMAL else "legacy"

if __name__ == "__main__":
    print(
        f"NetworkX MCP Server v{__version__} ({__implementation__} implementation)",
        file=sys.stderr,
    )
    if hasattr(sys.modules[__name__], "main"):
        main()
    else:
        print("❌ No main() function found in selected implementation", file=sys.stderr)
        sys.exit(1)
