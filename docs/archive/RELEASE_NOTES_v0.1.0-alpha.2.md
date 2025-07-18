# Release Notes: v0.1.0-alpha.2

## The "Actually Minimal" Release

We discovered our "minimal" server was secretly loading the entire scientific Python stack. This release fixes that architectural dishonesty.

### What Changed

#### Memory Usage (THE BIG FIX)

- **Before**: 118MB (loaded pandas, scipy, matplotlib for everyone)
- **After**: 54.6MB minimal, 118MB full (user choice)
- **Reduction**: 54% for basic operations (not 83% - let's be honest)

#### Import Hygiene

- **Before**: 900+ modules loaded at startup
- **After**: ~600 modules (removed scientific stack)
- **Improvement**: No pandas/scipy unless explicitly needed

#### Architecture

- Removed eager loading of I/O handlers from core
- Made pandas/scipy truly optional via lazy loading
- Created separate minimal and full installation options
- Added honest documentation about memory usage

### Breaking Changes

⚠️ **If you were using I/O handlers (Excel import, etc):**

```bash
# You now need to explicitly install extras:
pip install networkx-mcp[excel]  # Adds pandas support
```

**Import changes:**

```python
# Old (no longer works):
from networkx_mcp.core import GraphIOHandler

# New (lazy loading):
from networkx_mcp.core import get_io_handler
GraphIOHandler = get_io_handler()  # Only loads pandas when called
```

### How to Choose Your Version

```bash
# Just need basic graph operations? (90% of users)
pip install networkx-mcp  # 54.6MB

# Need Excel/CSV import?
pip install networkx-mcp[excel]  # ~89MB

# Need everything?
pip install networkx-mcp[full]  # ~118MB
```

### Honesty Statement

The previous version falsely claimed to be "minimal" while loading 900+ modules and using 118MB for basic operations. **We apologize for this architectural dishonesty.**

This version is actually minimal by default, with heavy dependencies as optional extras. The 54.6MB includes:

- Python interpreter: ~16MB
- NetworkX library: ~20MB
- Server + asyncio: ~18MB

This is the realistic minimum for a Python-based NetworkX server.

### Still Alpha Quality

This remains alpha software. The architecture is now honest, but:

- ✅ Single-user workflows fixed (stdin handling robust)
- ❌ No HTTP transport
- ❌ Limited production testing
- ❌ Some CI/CD tests still failing

**Use for local development and prototyping only.**

### Installation & Usage

```bash
# Basic installation (most users)
pip install networkx-mcp

# Add to Claude Desktop config:
{
  "mcpServers": {
    "networkx": {
      "command": "networkx-mcp",
      "cwd": "/your/project/path"
    }
  }
}
```

### What's Next

- Fix remaining CI/CD issues
- Add HTTP transport option
- Explore further memory optimizations
- Production hardening

---

**Memory Budget Reality**: 54.6MB isn't tiny, but it's honest for Python + NetworkX. We choose architectural honesty over false claims.
