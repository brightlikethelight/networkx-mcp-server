# ADR-001: Remove Heavyweight Dependencies from Core

## Status
Accepted (2025-01-08)

## Context

During a critical investigation of "memory leaks", we discovered our "minimal" MCP server was anything but minimal:

- **Claimed**: "Minimal NetworkX MCP server"
- **Reality**: 118MB memory usage, loading 900+ modules
- **Root cause**: `core/__init__.py` eagerly imported `io_handlers.py` which imported:
  - pandas (+35MB) - for Excel/CSV that 90% of users never touch
  - scipy (+15MB) - for sparse matrices rarely used
  - Related dependencies totaling 50MB+

This was architectural malpractice. We were forcing every user to load the entire scientific Python stack for basic graph operations that need 10% of that.

### The Fatal Import Chain
```python
server.py 
  → from .core.graph_operations import GraphManager
  → triggers core/__init__.py loading
  → from .io import GraphIOHandler  # Line 5 - THE KILLER
  → loads io_handlers.py
  → import pandas as pd  # Line 12 - BOOM! +35MB
```

### Evidence of the Problem
- Import trace showed 900+ modules loaded at startup
- Memory profiling: 118MB for a "minimal" server
- Startup time: 2+ seconds to load unused libraries
- Users complained about bloat but we didn't investigate

## Decision

1. **Break the import chain**: Remove `GraphIOHandler` from `core/__init__.py`
2. **Lazy loading**: Add `get_io_handler()` function for on-demand loading
3. **Separate implementations**:
   - `server_minimal.py`: NetworkX only (~54MB)
   - `server.py`: Full features when needed
4. **Optional dependencies**: Move pandas/scipy to `extras_require`

## Consequences

### Positive
- Memory reduced from 118MB to 54MB (54% reduction)
- Module count reduced from 900+ to ~600
- Startup time improved (though still not instant due to NetworkX)
- **Honest architecture**: Users only pay for what they use
- Clear separation between core and optional features

### Negative
- Breaking change: `GraphIOHandler` no longer auto-imported
- Users must explicitly install extras for Excel/CSV
- Two server implementations to maintain
- Some users may be surprised when Excel import fails

### Realistic Performance

| Component | Memory | Justification |
|-----------|--------|---------------|
| Python baseline | 16MB | Interpreter overhead |
| NetworkX | 20MB | Core graph library |
| Our minimal code | 18MB | Server, asyncio, JSON handling |
| **Total Minimal** | **54MB** | **Realistic minimum** |
| + pandas | +35MB | DataFrames, Excel support |
| + scipy | +15MB | Sparse matrices |
| **Total Full** | **104MB** | **All features** |

## Lessons Learned

1. **"Minimal" is a promise**: Don't claim minimal while loading pandas
2. **Measure before claiming**: We assumed, never verified
3. **Import chains matter**: One innocent import can cascade
4. **Optional should be optional**: Core features only in core

## Implementation Details

### Before (Broken)
```python
# core/__init__.py
from .io import GraphIOHandler  # Always loads pandas!
```

### After (Fixed)
```python
# core/__init__.py
def get_io_handler():
    """Lazy load I/O handler only when needed."""
    from .io import GraphIOHandler
    return GraphIOHandler
```

### Installation Options
```bash
# Minimal - what most users need
pip install networkx-mcp          # 54MB

# With data I/O support
pip install networkx-mcp[excel]   # 89MB

# Everything
pip install networkx-mcp[full]    # 118MB
```

## Verification

Test results confirm the fix:
```
✅ Pandas NOT loaded in minimal mode
✅ SciPy NOT loaded in minimal mode
✅ Memory: 118MB → 54MB (54% reduction)
✅ Modules: 900+ → 600 (33% reduction)
```

## Future Considerations

1. Consider lazy-loading NetworkX itself for even smaller footprint
2. Investigate if we can reduce below 54MB
3. Add memory budget tests to CI to prevent regression
4. Document memory requirements prominently

## Conclusion

This was not a feature - it was a bug in our architecture. We've fixed it by being honest about dependencies and making heavyweight libraries truly optional. The server is now closer to actually being minimal, though at 54MB it's still not tiny. At least it's honest.